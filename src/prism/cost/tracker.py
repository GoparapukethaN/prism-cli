"""Cost tracking — records and queries API call costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import structlog

from prism.cost.pricing import calculate_cost, get_provider_for_model

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CostEntry:
    """A single cost tracking record."""

    id: str
    created_at: str  # ISO 8601
    session_id: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    complexity_tier: str  # simple, medium, complex


@dataclass
class CostSummary:
    """Aggregated cost data for a time period."""

    period: str  # session, day, month
    total_cost: float
    total_requests: int
    model_breakdown: list[ModelCostBreakdown] = field(default_factory=list)
    budget_limit: float | None = None
    budget_remaining: float | None = None


@dataclass
class ModelCostBreakdown:
    """Cost breakdown for a single model within a period."""

    model_id: str
    display_name: str
    request_count: int
    total_cost: float
    percentage: float  # Of total cost (0-100)


class CostTracker:
    """Tracks API call costs and provides budget enforcement.

    Records every API call's cost to the database and provides
    aggregation queries for dashboards and budget checks.
    """

    def __init__(self, db: object, settings: object) -> None:
        """Initialize the cost tracker.

        Args:
            db: Database instance for persistence.
            settings: Settings instance for budget configuration.
        """
        self._db = db
        self._settings = settings

    def track(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        session_id: str,
        complexity_tier: str,
        cached_tokens: int = 0,
    ) -> CostEntry:
        """Record an API call's cost.

        Args:
            model_id: LiteLLM model identifier.
            input_tokens: Total input tokens.
            output_tokens: Output tokens generated.
            session_id: Current session ID.
            complexity_tier: Task complexity tier.
            cached_tokens: Tokens served from cache.

        Returns:
            The created CostEntry record.
        """
        try:
            cost = calculate_cost(model_id, input_tokens, output_tokens, cached_tokens)
        except ValueError:
            # Unknown model — log as zero cost rather than failing
            logger.warning("unknown_model_for_cost", model_id=model_id)
            cost = 0.0

        provider = get_provider_for_model(model_id)
        now = datetime.now(UTC).isoformat()

        entry = CostEntry(
            id=str(uuid4()),
            created_at=now,
            session_id=session_id,
            model_id=model_id,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost,
            complexity_tier=complexity_tier,
        )

        # Persist to database
        try:
            from prism.db.queries import save_cost_entry, update_session

            save_cost_entry(self._db, entry)
            update_session(self._db, session_id, cost_delta=cost, request_delta=1)
        except Exception:
            logger.exception("cost_tracking_db_error", model_id=model_id)

        logger.info(
            "cost_tracked",
            model=model_id,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost=f"${cost:.6f}",
            tier=complexity_tier,
        )

        return entry

    def get_session_cost(self, session_id: str) -> float:
        """Get total cost for a session.

        Args:
            session_id: The session ID to query.

        Returns:
            Total cost in USD.
        """
        try:
            from prism.db.queries import get_session_cost

            return get_session_cost(self._db, session_id)
        except Exception:
            logger.exception("session_cost_query_error")
            return 0.0

    def get_daily_cost(self) -> float:
        """Get total cost for today.

        Returns:
            Total cost in USD.
        """
        try:
            from prism.db.queries import get_daily_cost

            return get_daily_cost(self._db)
        except Exception:
            logger.exception("daily_cost_query_error")
            return 0.0

    def get_monthly_cost(self) -> float:
        """Get total cost for the current month.

        Returns:
            Total cost in USD.
        """
        try:
            from prism.db.queries import get_monthly_cost

            return get_monthly_cost(self._db)
        except Exception:
            logger.exception("monthly_cost_query_error")
            return 0.0

    def get_budget_remaining(self) -> float | None:
        """Calculate remaining budget.

        Checks both daily and monthly limits and returns the more restrictive.

        Returns:
            Remaining budget in USD, or None if no limits set.
        """
        daily_limit = self._settings.get("budget.daily_limit")
        monthly_limit = self._settings.get("budget.monthly_limit")

        if daily_limit is None and monthly_limit is None:
            return None

        remaining_values: list[float] = []

        if daily_limit is not None:
            daily_cost = self.get_daily_cost()
            remaining_values.append(max(0.0, daily_limit - daily_cost))

        if monthly_limit is not None:
            monthly_cost = self.get_monthly_cost()
            remaining_values.append(max(0.0, monthly_limit - monthly_cost))

        return min(remaining_values) if remaining_values else None

    def check_budget(self, estimated_cost: float) -> BudgetAction:
        """Check if a request fits within budget.

        Args:
            estimated_cost: Estimated cost of the pending request.

        Returns:
            BudgetAction indicating whether to proceed, warn, or block.
        """
        remaining = self.get_budget_remaining()

        if remaining is None:
            return BudgetAction.PROCEED

        if estimated_cost <= remaining:
            # Check if we should warn
            warn_percent = self._settings.get("budget.warn_at_percent", 80.0)
            daily_limit = self._settings.get("budget.daily_limit")
            monthly_limit = self._settings.get("budget.monthly_limit")

            should_warn = False
            if daily_limit is not None:
                daily_cost = self.get_daily_cost()
                if (daily_cost / daily_limit * 100) >= warn_percent:
                    should_warn = True
            if monthly_limit is not None:
                monthly_cost = self.get_monthly_cost()
                if (monthly_cost / monthly_limit * 100) >= warn_percent:
                    should_warn = True

            return BudgetAction.WARN if should_warn else BudgetAction.PROCEED

        if estimated_cost <= remaining * 1.5:
            return BudgetAction.WARN

        return BudgetAction.BLOCK

    def get_cost_summary(self, period: str, session_id: str = "") -> CostSummary:
        """Get a cost summary for a time period.

        Args:
            period: One of 'session', 'day', 'month'.
            session_id: Required if period is 'session'.

        Returns:
            CostSummary with breakdown by model.
        """
        try:
            from prism.db.queries import get_cost_breakdown

            breakdown_data = get_cost_breakdown(self._db, period, session_id)
        except Exception:
            logger.exception("cost_summary_query_error")
            breakdown_data = []

        total_cost = sum(row["total_cost"] for row in breakdown_data)
        total_requests = sum(row["request_count"] for row in breakdown_data)

        breakdown = [
            ModelCostBreakdown(
                model_id=row["model_id"],
                display_name=row.get("display_name", row["model_id"]),
                request_count=row["request_count"],
                total_cost=row["total_cost"],
                percentage=(row["total_cost"] / total_cost * 100) if total_cost > 0 else 0.0,
            )
            for row in breakdown_data
        ]

        budget_limit = None
        budget_remaining = None
        if period == "day":
            budget_limit = self._settings.get("budget.daily_limit")
        elif period == "month":
            budget_limit = self._settings.get("budget.monthly_limit")
        if budget_limit is not None:
            budget_remaining = max(0.0, budget_limit - total_cost)

        return CostSummary(
            period=period,
            total_cost=total_cost,
            total_requests=total_requests,
            model_breakdown=breakdown,
            budget_limit=budget_limit,
            budget_remaining=budget_remaining,
        )

    def calculate_savings(self) -> tuple[float, float, float]:
        """Calculate how much the user saved vs. using only premium models.

        Compares actual spend against hypothetical cost if all requests
        had been routed to Claude Sonnet 4.

        Returns:
            Tuple of (hypothetical_cost, actual_cost, savings).
        """
        try:
            from prism.cost.pricing import MODEL_PRICING
            from prism.db.queries import get_all_cost_entries_for_month

            entries = get_all_cost_entries_for_month(self._db)
        except Exception:
            logger.exception("savings_calc_error")
            return 0.0, 0.0, 0.0

        premium_pricing = MODEL_PRICING.get("claude-sonnet-4-20250514")
        if premium_pricing is None:
            return 0.0, 0.0, 0.0

        hypothetical_cost = sum(
            (e["input_tokens"] / 1_000_000) * premium_pricing.input_cost_per_1m
            + (e["output_tokens"] / 1_000_000) * premium_pricing.output_cost_per_1m
            for e in entries
        )

        actual_cost = sum(e["cost_usd"] for e in entries)
        savings = hypothetical_cost - actual_cost

        return hypothetical_cost, actual_cost, max(0.0, savings)


class BudgetAction:
    """Budget check result actions."""

    PROCEED = "proceed"
    WARN = "warn"
    BLOCK = "block"

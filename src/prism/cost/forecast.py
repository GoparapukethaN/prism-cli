"""Cost forecasting — spending velocity tracking, projections, and alternative suggestions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SpendingVelocity:
    """Spending rate for a time window."""

    cost_per_hour: float
    tokens_per_hour: int
    requests_per_hour: float
    window_hours: float
    total_cost: float
    total_tokens: int
    total_requests: int


@dataclass
class ModelCostDriver:
    """Cost breakdown for a single model."""

    model_id: str
    display_name: str
    total_cost: float
    percentage: float
    request_count: int
    avg_cost_per_request: float
    cheapest_alternative: str | None = None
    potential_savings: float = 0.0


@dataclass
class CostForecast:
    """Monthly cost projection."""

    projected_monthly_cost: float
    current_monthly_cost: float
    daily_average: float
    days_remaining: int
    budget_limit: float | None
    budget_used_percent: float
    alert_level: str  # "ok", "warning", "critical"
    velocity: SpendingVelocity
    model_drivers: list[ModelCostDriver] = field(default_factory=list)


@dataclass
class FeatureCost:
    """Cost tracking for a specific feature/task."""

    name: str
    started_at: str
    completed_at: str | None = None
    total_cost: float = 0.0
    request_count: int = 0
    models_used: list[str] = field(default_factory=list)


# Known cheaper alternatives for popular models.
ALTERNATIVE_MAP: dict[str, str] = {
    "claude-sonnet-4-20250514": "deepseek/deepseek-chat",
    "gpt-4o": "deepseek/deepseek-chat",
    "claude-opus-4-20250514": "claude-sonnet-4-20250514",
    "gpt-4-turbo": "gpt-4o-mini",
    "gemini/gemini-2.5-pro": "gemini/gemini-2.0-flash",
    "gemini/gemini-1.5-pro": "gemini/gemini-1.5-flash",
    "o3": "o4-mini",
}

# Rough cost ratio of alternative relative to the original model.
# E.g. 0.05 means the alternative costs ~5% of the original.
COST_RATIOS: dict[str, float] = {
    "deepseek/deepseek-chat": 0.05,
    "gpt-4o-mini": 0.15,
    "gemini/gemini-2.0-flash": 0.10,
    "gemini/gemini-1.5-flash": 0.10,
    "claude-sonnet-4-20250514": 0.50,
    "groq/llama-3.3-70b-versatile": 0.03,
    "o4-mini": 0.11,
}

# Default working hours assumed per day for projection blending.
_WORKING_HOURS_PER_DAY = 8

# Minimum elapsed session hours before velocity-based blending kicks in.
_MIN_VELOCITY_WINDOW_HOURS = 0.1

# Minimum elapsed seconds mapped to fractional hours to prevent division by zero.
_MIN_ELAPSED_HOURS = 0.001


class CostForecaster:
    """Forecasts monthly spend, tracks feature costs, and suggests alternatives.

    Combines real-time session velocity with historical monthly cost data to
    produce a blended monthly forecast.  Tracks per-feature costs and generates
    weekly Markdown reports saved to ``~/.prism/reports/``.
    """

    def __init__(
        self,
        cost_tracker: object,
        settings: object,
        reports_dir: Path | None = None,
    ) -> None:
        """Initialize the cost forecaster.

        Args:
            cost_tracker: A CostTracker (or compatible) instance.  Must expose
                ``get_monthly_cost()`` if historical cost is needed.
            settings: A Settings (or compatible) instance.  Must expose
                ``get(key, default=None)`` for budget configuration.
            reports_dir: Directory for weekly cost reports.  Defaults to
                ``~/.prism/reports/``.
        """
        self._tracker = cost_tracker
        self._settings = settings
        self._reports_dir = reports_dir or Path.home() / ".prism" / "reports"
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._feature_costs: dict[str, FeatureCost] = {}
        self._session_start = datetime.now(UTC)
        self._session_costs: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Session request tracking
    # ------------------------------------------------------------------

    def track_request(
        self,
        model_id: str,
        cost_usd: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Track a request for velocity calculation.

        Args:
            model_id: LiteLLM model identifier.
            cost_usd: Cost for this request in USD.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        if not model_id:
            return

        self._session_costs.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "model_id": model_id,
            "cost_usd": cost_usd,
            "tokens": input_tokens + output_tokens,
        })

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------

    def get_velocity(self) -> SpendingVelocity:
        """Calculate current spending velocity from session data.

        Returns:
            SpendingVelocity with per-hour rates.  If no requests have been
            tracked, all values are zero.
        """
        if not self._session_costs:
            return SpendingVelocity(
                cost_per_hour=0.0,
                tokens_per_hour=0,
                requests_per_hour=0.0,
                window_hours=0.0,
                total_cost=0.0,
                total_tokens=0,
                total_requests=0,
            )

        elapsed = max(
            (datetime.now(UTC) - self._session_start).total_seconds() / 3600,
            _MIN_ELAPSED_HOURS,
        )

        total_cost = sum(float(c["cost_usd"]) for c in self._session_costs)
        total_tokens = sum(int(c["tokens"]) for c in self._session_costs)
        total_requests = len(self._session_costs)

        return SpendingVelocity(
            cost_per_hour=total_cost / elapsed,
            tokens_per_hour=int(total_tokens / elapsed),
            requests_per_hour=total_requests / elapsed,
            window_hours=elapsed,
            total_cost=total_cost,
            total_tokens=total_tokens,
            total_requests=total_requests,
        )

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def forecast(self) -> CostForecast:
        """Generate a monthly cost forecast based on current velocity.

        The projection blends historical daily average cost with session
        velocity when the session has lasted at least
        ``_MIN_VELOCITY_WINDOW_HOURS`` hours.

        Returns:
            CostForecast with projection, budget status, and model drivers.
        """
        velocity = self.get_velocity()

        now = datetime.now(UTC)
        days_in_month = 30  # simplified constant
        day_of_month = now.day
        days_remaining = max(1, days_in_month - day_of_month)

        # Historical monthly cost from the tracker
        monthly_cost = 0.0
        if hasattr(self._tracker, "get_monthly_cost"):
            monthly_cost = self._tracker.get_monthly_cost()

        daily_average = monthly_cost / max(day_of_month, 1)
        projected = monthly_cost + (daily_average * days_remaining)

        # Blend session velocity when it's meaningful
        if velocity.cost_per_hour > 0 and velocity.window_hours > _MIN_VELOCITY_WINDOW_HOURS:
            session_daily_rate = velocity.cost_per_hour * _WORKING_HOURS_PER_DAY
            if daily_average > 0:
                blended_daily = session_daily_rate * 0.6 + daily_average * 0.4
                projected = monthly_cost + (blended_daily * days_remaining)
            else:
                projected = session_daily_rate * days_remaining

        # Budget check
        budget_limit = self._resolve_budget_limit()

        budget_used_pct = 0.0
        alert_level = "ok"
        if budget_limit is not None and budget_limit > 0:
            budget_used_pct = (monthly_cost / budget_limit) * 100
            if budget_used_pct >= 100:
                alert_level = "critical"
            elif budget_used_pct >= 70:
                alert_level = "warning"

        model_drivers = self._get_model_drivers()

        return CostForecast(
            projected_monthly_cost=projected,
            current_monthly_cost=monthly_cost,
            daily_average=daily_average,
            days_remaining=days_remaining,
            budget_limit=budget_limit,
            budget_used_percent=budget_used_pct,
            alert_level=alert_level,
            velocity=velocity,
            model_drivers=model_drivers,
        )

    # ------------------------------------------------------------------
    # Model driver analysis
    # ------------------------------------------------------------------

    def _get_model_drivers(self) -> list[ModelCostDriver]:
        """Analyze which models are driving the most cost.

        Returns:
            List of ModelCostDriver sorted by cost descending.
        """
        model_costs: dict[str, dict[str, float | int]] = {}
        for entry in self._session_costs:
            mid = str(entry["model_id"])
            if mid not in model_costs:
                model_costs[mid] = {"cost": 0.0, "count": 0}
            model_costs[mid]["cost"] = float(model_costs[mid]["cost"]) + float(entry["cost_usd"])
            model_costs[mid]["count"] = int(model_costs[mid]["count"]) + 1

        total = sum(float(m["cost"]) for m in model_costs.values())

        drivers: list[ModelCostDriver] = []
        for model_id, data in sorted(
            model_costs.items(), key=lambda x: float(x[1]["cost"]), reverse=True
        ):
            cost_val = float(data["cost"])
            count_val = int(data["count"])
            alt, savings = self._find_cheapest_alternative(model_id, cost_val, count_val)
            display = model_id.split("/")[-1] if "/" in model_id else model_id
            drivers.append(
                ModelCostDriver(
                    model_id=model_id,
                    display_name=display,
                    total_cost=cost_val,
                    percentage=(cost_val / total * 100) if total > 0 else 0.0,
                    request_count=count_val,
                    avg_cost_per_request=cost_val / max(count_val, 1),
                    cheapest_alternative=alt,
                    potential_savings=savings,
                )
            )

        return drivers

    @staticmethod
    def _find_cheapest_alternative(
        model_id: str,
        current_cost: float,
        request_count: int,
    ) -> tuple[str | None, float]:
        """Find a cheaper model that could handle similar tasks.

        Args:
            model_id: Current model identifier.
            current_cost: Total cost incurred by this model.
            request_count: Number of requests (reserved for future use).

        Returns:
            Tuple of (alternative model id or None, potential savings USD).
        """
        alt = ALTERNATIVE_MAP.get(model_id)
        if alt is not None and alt in COST_RATIOS:
            alt_cost = current_cost * COST_RATIOS[alt]
            savings = current_cost - alt_cost
            if savings > 0:
                return alt, savings

        return None, 0.0

    # ------------------------------------------------------------------
    # Feature cost tracking
    # ------------------------------------------------------------------

    def start_feature(self, name: str) -> FeatureCost:
        """Start tracking cost for a feature/task.

        Args:
            name: Human-readable feature name.

        Returns:
            The newly created FeatureCost.
        """
        if not name:
            msg = "Feature name must not be empty."
            raise ValueError(msg)

        feature = FeatureCost(
            name=name,
            started_at=datetime.now(UTC).isoformat(),
        )
        self._feature_costs[name] = feature
        return feature

    def end_feature(self, name: str) -> FeatureCost | None:
        """End tracking for a feature and return its cost.

        Args:
            name: Feature name previously passed to ``start_feature``.

        Returns:
            The FeatureCost with ``completed_at`` set, or ``None`` if
            the feature was not found.
        """
        feature = self._feature_costs.get(name)
        if feature is not None:
            feature.completed_at = datetime.now(UTC).isoformat()
        return feature

    def track_feature_cost(self, name: str, cost: float, model: str) -> None:
        """Add cost to an active feature.

        Args:
            name: Feature name.
            cost: Incremental cost in USD.
            model: Model ID that generated the cost.
        """
        feature = self._feature_costs.get(name)
        if feature is not None:
            feature.total_cost += cost
            feature.request_count += 1
            if model not in feature.models_used:
                feature.models_used.append(model)

    def get_feature_costs(self) -> list[FeatureCost]:
        """Get all tracked feature costs.

        Returns:
            List of FeatureCost objects.
        """
        return list(self._feature_costs.values())

    # ------------------------------------------------------------------
    # Weekly report generation
    # ------------------------------------------------------------------

    def generate_weekly_report(self) -> Path:
        """Generate a weekly cost report and save to disk.

        The report is a Markdown file named ``YYYY-WWW.md`` (ISO week) and
        saved under the ``reports_dir``.

        Returns:
            Path to the generated report file.
        """
        now = datetime.now(UTC)
        week_num = now.isocalendar()[1]
        year = now.year
        filename = f"{year}-W{week_num:02d}.md"

        fc = self.forecast()
        velocity = self.get_velocity()

        lines = [
            f"# Prism Weekly Cost Report — {year} Week {week_num}",
            "",
            f"Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Summary",
            f"- Monthly spend so far: ${fc.current_monthly_cost:.4f}",
            f"- Projected monthly total: ${fc.projected_monthly_cost:.4f}",
            f"- Daily average: ${fc.daily_average:.4f}",
            f"- Days remaining: {fc.days_remaining}",
            "",
            "## Velocity",
            f"- Cost/hour: ${velocity.cost_per_hour:.4f}",
            f"- Tokens/hour: {velocity.tokens_per_hour:,}",
            f"- Requests/hour: {velocity.requests_per_hour:.1f}",
            "",
        ]

        if fc.budget_limit is not None:
            lines.extend([
                "## Budget",
                f"- Limit: ${fc.budget_limit:.2f}",
                f"- Used: {fc.budget_used_percent:.1f}%",
                f"- Alert: {fc.alert_level.upper()}",
                "",
            ])

        if fc.model_drivers:
            lines.extend([
                "## Model Cost Drivers",
                "| Model | Cost | % | Requests | Alternative | Savings |",
                "|-------|------|---|----------|-------------|---------|",
            ])
            for d in fc.model_drivers:
                alt = d.cheapest_alternative or "—"
                sav = f"${d.potential_savings:.4f}" if d.potential_savings > 0 else "—"
                lines.append(
                    f"| {d.display_name} | ${d.total_cost:.4f} | {d.percentage:.1f}% | "
                    f"{d.request_count} | {alt} | {sav} |"
                )
            lines.append("")

        if self._feature_costs:
            lines.extend([
                "## Feature Costs",
                "| Feature | Cost | Requests | Models |",
                "|---------|------|----------|--------|",
            ])
            for ftr in self._feature_costs.values():
                models = ", ".join(ftr.models_used) if ftr.models_used else "—"
                lines.append(
                    f"| {ftr.name} | ${ftr.total_cost:.4f} | {ftr.request_count} | {models} |"
                )
            lines.append("")

        report_path = self._reports_dir / filename
        report_path.write_text("\n".join(lines))
        logger.info("weekly_report_generated", path=str(report_path))
        return report_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_budget_limit(self) -> float | None:
        """Resolve the monthly budget limit from settings.

        Returns:
            Monthly budget limit in USD, or ``None`` if not configured.
        """
        if hasattr(self._settings, "get"):
            val = self._settings.get("budget.monthly_limit")
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
        return None

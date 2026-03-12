"""Smart cost optimizer tool — analyzes usage and recommends cheaper alternatives.

No single-model CLI can do this: Prism routes across *all* providers, so it
can compare real success rates and costs across models.  This tool reads the
user's actual usage history, cross-references it with the adaptive learner's
success-rate data, and produces actionable recommendations like:

    "You're using GPT-4 for simple questions.  Switch to Groq Llama (free)
     — same quality for this tier."

Actions:
    ``analyze``   — group historical costs by model, tier, and provider.
    ``recommend`` — suggest cheaper models with comparable success rates.
    ``report``    — generate a period-over-period cost report with savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from prism.cost.pricing import MODEL_PRICING, ModelPricing
from prism.tools.base import PermissionLevel, Tool, ToolResult

if TYPE_CHECKING:
    from prism.cost.tracker import CostTracker
    from prism.router.learning import AdaptiveLearner

logger = structlog.get_logger(__name__)

# Minimum success-rate a cheaper model must have to be recommended.
_MIN_ACCEPTABLE_SUCCESS_RATE: float = 0.70

# Maximum cost ratio (cheaper / current) to count as a worthwhile recommendation.
_MAX_COST_RATIO: float = 0.60


@dataclass(frozen=True)
class ModelRecommendation:
    """A single recommendation to switch from one model to another.

    Attributes:
        current_model:        The model the user is currently paying for.
        recommended_model:    A cheaper model with acceptable quality.
        current_cost_per_req: Average cost per request with the current model.
        recommended_cost:     Average cost per request with the recommended model.
        savings_percent:      Estimated savings as a percentage (0-100).
        success_rate_current: EWMA success rate of the current model.
        success_rate_rec:     EWMA success rate of the recommended model.
        tier:                 Complexity tier this recommendation applies to.
        reason:               Human-readable explanation.
    """

    current_model: str
    recommended_model: str
    current_cost_per_req: float
    recommended_cost: float
    savings_percent: float
    success_rate_current: float
    success_rate_rec: float
    tier: str
    reason: str


class CostOptimizerTool(Tool):
    """Analyze API usage patterns and recommend cheaper model alternatives.

    This tool is unique to multi-provider routers.  A single-model CLI has
    no cost data to compare because it talks to only one provider.  Prism
    tracks every call across every provider, so this tool can:

    * **analyze** — break down costs by model, tier, and time period.
    * **recommend** — find cheaper models with comparable success rates.
    * **report** — generate a savings report vs. using only premium models.

    Uses in-memory data from :class:`CostTracker` and
    :class:`AdaptiveLearner`; never makes API calls.
    """

    def __init__(
        self,
        cost_tracker: CostTracker,
        adaptive_learner: AdaptiveLearner,
    ) -> None:
        """Initialise the cost optimiser tool.

        Args:
            cost_tracker:     A :class:`CostTracker` with usage history.
            adaptive_learner: An :class:`AdaptiveLearner` with success data.
        """
        self._cost_tracker = cost_tracker
        self._learner = adaptive_learner

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "cost_optimizer"

    @property
    def description(self) -> str:
        return (
            "Analyze API usage patterns and recommend cheaper model "
            "alternatives. Actions: 'analyze' (cost breakdown by model "
            "and tier), 'recommend' (suggest cheaper models with equal "
            "quality), 'report' (savings report vs. premium-only usage)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "One of 'analyze', 'recommend', or 'report'."
                    ),
                },
                "period": {
                    "type": "string",
                    "description": (
                        "Time period: 'session', 'day', or 'month'. "
                        "Default 'month'."
                    ),
                    "default": "month",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID (required when period='session').",
                    "default": "",
                },
            },
            "required": ["action"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.AUTO

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the cost optimizer action.

        Args:
            arguments: Must contain ``action`` (str).  Optional ``period``
                (str) and ``session_id`` (str).

        Returns:
            A :class:`ToolResult` with structured analysis output.
        """
        validated = self.validate_arguments(arguments)
        action: str = validated["action"]
        period: str = validated.get("period", "month")
        session_id: str = validated.get("session_id", "")

        if action not in ("analyze", "recommend", "report"):
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Unknown action '{action}'. "
                    "Must be 'analyze', 'recommend', or 'report'."
                ),
            )

        if period not in ("session", "day", "month"):
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Unknown period '{period}'. "
                    "Must be 'session', 'day', or 'month'."
                ),
            )

        try:
            if action == "analyze":
                return self._action_analyze(period, session_id)
            if action == "recommend":
                return self._action_recommend()
            return self._action_report()
        except Exception as exc:
            logger.exception("cost_optimizer_error", action=action)
            return ToolResult(
                success=False,
                output="",
                error=f"Cost optimizer failed: {exc}",
                metadata={"action": action},
            )

    # ------------------------------------------------------------------
    # analyze
    # ------------------------------------------------------------------

    def _action_analyze(self, period: str, session_id: str) -> ToolResult:
        """Break down costs by model and tier for a time period.

        Args:
            period: Time period (``"session"``, ``"day"``, or ``"month"``).
            session_id: Session ID (used when *period* is ``"session"``).

        Returns:
            A :class:`ToolResult` with the cost breakdown.
        """
        summary = self._cost_tracker.get_cost_summary(period, session_id)

        breakdown_lines: list[str] = []
        breakdown_lines.append(f"Cost Analysis ({period})")
        breakdown_lines.append("=" * 50)
        breakdown_lines.append(
            f"Total cost: ${summary.total_cost:.4f}"
        )
        breakdown_lines.append(
            f"Total requests: {summary.total_requests}"
        )

        if summary.budget_limit is not None:
            breakdown_lines.append(
                f"Budget limit: ${summary.budget_limit:.2f}"
            )
        if summary.budget_remaining is not None:
            breakdown_lines.append(
                f"Budget remaining: ${summary.budget_remaining:.2f}"
            )

        if summary.model_breakdown:
            breakdown_lines.append("")
            breakdown_lines.append("Model Breakdown:")
            breakdown_lines.append("-" * 50)
            for entry in summary.model_breakdown:
                avg_cost = (
                    entry.total_cost / entry.request_count
                    if entry.request_count > 0
                    else 0.0
                )
                breakdown_lines.append(
                    f"  {entry.display_name}: "
                    f"${entry.total_cost:.4f} "
                    f"({entry.request_count} reqs, "
                    f"${avg_cost:.6f}/req, "
                    f"{entry.percentage:.1f}%)"
                )

        output_text = "\n".join(breakdown_lines)

        metadata: dict[str, Any] = {
            "action": "analyze",
            "period": period,
            "total_cost": summary.total_cost,
            "total_requests": summary.total_requests,
            "budget_limit": summary.budget_limit,
            "budget_remaining": summary.budget_remaining,
            "models": [
                {
                    "model_id": m.model_id,
                    "display_name": m.display_name,
                    "request_count": m.request_count,
                    "total_cost": m.total_cost,
                    "percentage": m.percentage,
                }
                for m in summary.model_breakdown
            ],
        }

        return ToolResult(
            success=True,
            output=output_text,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def _action_recommend(self) -> ToolResult:
        """Generate model switch recommendations based on usage and quality.

        Compares each model the user is paying for against cheaper
        alternatives and recommends switches where the cheaper model has
        an acceptable success rate.

        Returns:
            A :class:`ToolResult` with recommendations.
        """
        recommendations: list[ModelRecommendation] = []

        # Get all models the user has data for
        models_used: set[str] = set()
        for model_id in MODEL_PRICING:
            stats = self._learner.get_model_stats(model_id)
            if stats["interactions"] and int(str(stats["interactions"])) > 0:
                models_used.add(model_id)

        if not models_used:
            return ToolResult(
                success=True,
                output=(
                    "Not enough usage data to generate recommendations. "
                    "Use Prism for a while to build up interaction history."
                ),
                metadata={"action": "recommend", "recommendations": []},
            )

        # For each used model, find cheaper alternatives
        for model_id in sorted(models_used):
            pricing = MODEL_PRICING.get(model_id)
            if pricing is None:
                continue

            current_stats = self._learner.get_model_stats(model_id)
            current_rate = float(str(current_stats["success_rate"]))
            current_cost = self._avg_cost_per_token(pricing)

            # Find cheaper alternatives with acceptable success
            for alt_id, alt_pricing in MODEL_PRICING.items():
                if alt_id == model_id:
                    continue

                alt_cost = self._avg_cost_per_token(alt_pricing)

                # Must be meaningfully cheaper
                if current_cost <= 0 or alt_cost / current_cost > _MAX_COST_RATIO:
                    continue

                alt_rate = self._learner.get_success_rate(alt_id)

                # Must have acceptable quality
                if alt_rate < _MIN_ACCEPTABLE_SUCCESS_RATE:
                    continue

                # Must not be much worse than current model
                if alt_rate < current_rate - 0.15:
                    continue

                savings_pct = (
                    (1.0 - alt_cost / current_cost) * 100
                    if current_cost > 0
                    else 0.0
                )

                reason = self._build_recommendation_reason(
                    model_id, alt_id, savings_pct, alt_rate
                )

                recommendations.append(
                    ModelRecommendation(
                        current_model=model_id,
                        recommended_model=alt_id,
                        current_cost_per_req=current_cost,
                        recommended_cost=alt_cost,
                        savings_percent=round(savings_pct, 1),
                        success_rate_current=round(current_rate, 4),
                        success_rate_rec=round(alt_rate, 4),
                        tier="all",
                        reason=reason,
                    )
                )

        # Sort by savings potential (highest first)
        recommendations.sort(key=lambda r: r.savings_percent, reverse=True)

        # Build output
        if not recommendations:
            output_text = (
                "No cost optimizations found. Your current model "
                "selection is already cost-effective."
            )
        else:
            lines: list[str] = []
            lines.append("Cost Optimization Recommendations")
            lines.append("=" * 50)
            for i, rec in enumerate(recommendations[:10], 1):
                lines.append(f"\n{i}. {rec.reason}")
                lines.append(
                    f"   Current: {rec.current_model} "
                    f"(${rec.current_cost_per_req:.6f}/req, "
                    f"{rec.success_rate_current:.0%} success)"
                )
                lines.append(
                    f"   Switch to: {rec.recommended_model} "
                    f"(${rec.recommended_cost:.6f}/req, "
                    f"{rec.success_rate_rec:.0%} success)"
                )
                lines.append(
                    f"   Savings: ~{rec.savings_percent:.0f}%"
                )
            output_text = "\n".join(lines)

        metadata: dict[str, Any] = {
            "action": "recommend",
            "recommendation_count": len(recommendations),
            "recommendations": [
                {
                    "current_model": r.current_model,
                    "recommended_model": r.recommended_model,
                    "current_cost": r.current_cost_per_req,
                    "recommended_cost": r.recommended_cost,
                    "savings_percent": r.savings_percent,
                    "success_rate_current": r.success_rate_current,
                    "success_rate_rec": r.success_rate_rec,
                    "reason": r.reason,
                }
                for r in recommendations[:10]
            ],
        }

        return ToolResult(
            success=True,
            output=output_text,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------

    def _action_report(self) -> ToolResult:
        """Generate a savings report comparing actual spend vs. premium-only.

        Returns:
            A :class:`ToolResult` with the savings report.
        """
        hypothetical, actual, savings = (
            self._cost_tracker.calculate_savings()
        )

        lines: list[str] = []
        lines.append("Cost Savings Report (This Month)")
        lines.append("=" * 50)
        lines.append(
            f"If you had used only Claude Sonnet 4: ${hypothetical:.4f}"
        )
        lines.append(f"Actual spend with Prism routing: ${actual:.4f}")
        lines.append(f"You saved: ${savings:.4f}")

        if hypothetical > 0:
            pct = (savings / hypothetical) * 100
            lines.append(f"Savings percentage: {pct:.1f}%")
        else:
            pct = 0.0

        # Add daily and monthly cost context
        daily_cost = self._cost_tracker.get_daily_cost()
        monthly_cost = self._cost_tracker.get_monthly_cost()

        lines.append("")
        lines.append("Current Period Costs:")
        lines.append(f"  Today: ${daily_cost:.4f}")
        lines.append(f"  This month: ${monthly_cost:.4f}")

        remaining = self._cost_tracker.get_budget_remaining()
        if remaining is not None:
            lines.append(f"  Budget remaining: ${remaining:.2f}")

        output_text = "\n".join(lines)

        metadata: dict[str, Any] = {
            "action": "report",
            "hypothetical_cost": hypothetical,
            "actual_cost": actual,
            "savings": savings,
            "savings_percent": round(pct, 1),
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "budget_remaining": remaining,
        }

        return ToolResult(
            success=True,
            output=output_text,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_cost_per_token(pricing: ModelPricing) -> float:
        """Compute a blended cost-per-1K-token figure for comparison.

        Uses a 3:1 input:output ratio typical of conversational use.

        Args:
            pricing: Model pricing data.

        Returns:
            Approximate cost per 1K tokens (blended).
        """
        # Weighted average assuming 75% input, 25% output
        blended = (
            0.75 * pricing.input_cost_per_1m + 0.25 * pricing.output_cost_per_1m
        )
        return blended / 1_000_000 * 1000  # per 1K tokens

    @staticmethod
    def _build_recommendation_reason(
        current: str, alternative: str, savings_pct: float, alt_rate: float
    ) -> str:
        """Build a human-readable recommendation reason.

        Args:
            current: Current model ID.
            alternative: Recommended model ID.
            savings_pct: Savings percentage.
            alt_rate: Success rate of the alternative model.

        Returns:
            Human-readable recommendation string.
        """
        alt_display = alternative.rsplit("/", maxsplit=1)[-1] if "/" in alternative else alternative
        cur_display = current.rsplit("/", maxsplit=1)[-1] if "/" in current else current

        if savings_pct >= 90:
            return (
                f"Switch from {cur_display} to {alt_display} — "
                f"nearly free with {alt_rate:.0%} success rate."
            )
        if savings_pct >= 50:
            return (
                f"Switch from {cur_display} to {alt_display} — "
                f"save ~{savings_pct:.0f}% with comparable quality "
                f"({alt_rate:.0%} success)."
            )
        return (
            f"Consider {alt_display} instead of {cur_display} — "
            f"save ~{savings_pct:.0f}% ({alt_rate:.0%} success)."
        )

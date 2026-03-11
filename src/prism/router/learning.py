"""Adaptive learning — tracks model outcomes and refines selection over time.

Phase 1 (0-100 interactions): rule-based routing only; this module
collects data but does not influence decisions.

Phase 2 (100+ interactions): uses a simple weighted moving average of
success rates to recommend models per tier.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from prism.db.database import Database
    from prism.providers.base import ComplexityTier

logger = structlog.get_logger(__name__)

# Minimum number of recorded interactions before learning activates.
MIN_INTERACTIONS: int = 100

# How many recent outcomes to keep per model (rolling window).
_MAX_HISTORY_PER_MODEL: int = 500

# Decay factor for the exponential weighted moving average.
_EWMA_ALPHA: float = 0.05


@dataclass
class _Outcome:
    """A single recorded interaction outcome."""

    model: str
    tier: ComplexityTier
    outcome: str  # "accepted", "rejected", "corrected"
    cost: float
    timestamp: float = field(default_factory=time.time)


class AdaptiveLearner:
    """Collects interaction outcomes and recommends models per tier.

    Uses an **exponential weighted moving average** (EWMA) of binary
    success values (accepted = 1, else 0) to track each model's running
    success rate.

    Recommendations are only made after at least
    :data:`MIN_INTERACTIONS` total interactions have been recorded.
    """

    # Outcome → numeric value for EWMA
    _OUTCOME_VALUES: ClassVar[dict[str, float]] = {
        "accepted": 1.0,
        "corrected": 0.5,
        "rejected": 0.0,
    }

    def __init__(self, *, min_interactions: int = MIN_INTERACTIONS) -> None:
        """Initialise the learner.

        Args:
            min_interactions: Threshold before learned recommendations
                are used.
        """
        self._min_interactions = min_interactions
        self._total_interactions: int = 0

        # model_id → list of _Outcome (bounded)
        self._history: dict[str, list[_Outcome]] = defaultdict(list)

        # model_id → running EWMA success rate
        self._ewma: dict[str, float] = {}

        # model_id → total cost
        self._total_cost: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        model: str,
        tier: ComplexityTier,
        outcome: str,
        cost: float,
    ) -> None:
        """Record the result of an interaction.

        Args:
            model: LiteLLM model identifier.
            tier: The task's complexity tier.
            outcome: One of ``"accepted"``, ``"rejected"``, ``"corrected"``.
            cost: Actual cost in USD for this interaction.
        """
        record = _Outcome(model=model, tier=tier, outcome=outcome, cost=cost)

        history = self._history[model]
        history.append(record)
        if len(history) > _MAX_HISTORY_PER_MODEL:
            history.pop(0)

        self._total_interactions += 1
        self._total_cost[model] += cost

        # Update EWMA
        value = self._OUTCOME_VALUES.get(outcome, 0.0)
        prev = self._ewma.get(model, 0.5)  # default neutral
        self._ewma[model] = prev * (1 - _EWMA_ALPHA) + value * _EWMA_ALPHA

    def get_success_rate(self, model: str) -> float:
        """Return the EWMA success rate for *model*.

        Args:
            model: LiteLLM model identifier.

        Returns:
            Success rate between 0.0 and 1.0.  Returns ``0.5`` (neutral)
            if the model has no recorded outcomes.
        """
        return self._ewma.get(model, 0.5)

    def get_model_recommendation(self, tier: ComplexityTier) -> str | None:
        """Recommend the best model for a complexity tier.

        Returns ``None`` when there are fewer than *min_interactions*
        total recorded outcomes, or when no model has been recorded for
        the given tier.

        Args:
            tier: Complexity tier to get a recommendation for.

        Returns:
            The model ID with the highest EWMA success rate for *tier*,
            or ``None``.
        """
        if self._total_interactions < self._min_interactions:
            return None

        # Find models that have been used on this tier
        tier_models: dict[str, float] = {}
        for model_id, history in self._history.items():
            tier_outcomes = [o for o in history if o.tier == tier]
            if tier_outcomes:
                tier_models[model_id] = self._ewma.get(model_id, 0.5)

        if not tier_models:
            return None

        return max(tier_models, key=tier_models.get)  # type: ignore[arg-type]

    @property
    def total_interactions(self) -> int:
        """Total number of recorded interactions."""
        return self._total_interactions

    @property
    def is_active(self) -> bool:
        """Whether the learner has enough data to make recommendations."""
        return self._total_interactions >= self._min_interactions

    def get_total_cost(self, model: str) -> float:
        """Return cumulative cost for a model.

        Args:
            model: LiteLLM model identifier.

        Returns:
            Total USD spent through this model.
        """
        return self._total_cost.get(model, 0.0)

    def get_model_stats(self, model: str) -> dict[str, object]:
        """Return summary statistics for a model.

        Args:
            model: LiteLLM model identifier.

        Returns:
            Dict with ``interactions``, ``success_rate``, and ``total_cost``.
        """
        return {
            "interactions": len(self._history.get(model, [])),
            "success_rate": self.get_success_rate(model),
            "total_cost": self.get_total_cost(model),
        }


# ======================================================================
# Routing data exporter
# ======================================================================


class RoutingDataExporter:
    """Export routing decision data for analysis."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def export_csv(self, path: Path, days: int = 30) -> int:
        """Export recent routing decisions to CSV. Returns row count.

        Args:
            path: Destination file path.
            days: Number of days of history to export.

        Returns:
            Number of rows written.
        """
        rows = self.db.fetchall(
            """
            SELECT id, created_at, session_id, complexity_tier,
                   complexity_score, model_selected, model_actual,
                   estimated_cost, actual_cost, input_tokens,
                   output_tokens, cached_tokens, latency_ms, outcome
            FROM routing_decisions
            WHERE created_at >= datetime('now', ?)
            ORDER BY created_at DESC
            """,
            (f"-{days} days",),
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer: Any = None
            for row in rows:
                row_dict = dict(row)
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(row_dict.keys()))
                    writer.writeheader()
                writer.writerow(row_dict)
                count += 1

        logger.info("routing_data_exported", path=str(path), rows=count)
        return count

    def export_summary(self) -> dict[str, Any]:
        """Return summary statistics of routing decisions.

        Returns:
            Dict with total_decisions, tier breakdown, top models, avg cost.
        """
        total_row = self.db.fetchone(
            "SELECT COUNT(*) AS total FROM routing_decisions",
        )
        total = int(total_row["total"]) if total_row else 0

        tier_rows = self.db.fetchall(
            """
            SELECT complexity_tier, COUNT(*) AS cnt
            FROM routing_decisions
            GROUP BY complexity_tier
            """,
        )
        tier_breakdown: dict[str, int] = {
            row["complexity_tier"]: int(row["cnt"]) for row in tier_rows
        }

        model_rows = self.db.fetchall(
            """
            SELECT model_selected, COUNT(*) AS cnt
            FROM routing_decisions
            GROUP BY model_selected
            ORDER BY cnt DESC
            LIMIT 10
            """,
        )
        top_models: list[dict[str, Any]] = [
            {"model": row["model_selected"], "count": int(row["cnt"])}
            for row in model_rows
        ]

        avg_row = self.db.fetchone(
            "SELECT AVG(actual_cost) AS avg_cost FROM routing_decisions WHERE actual_cost IS NOT NULL",
        )
        avg_cost = float(avg_row["avg_cost"]) if avg_row and avg_row["avg_cost"] is not None else 0.0

        return {
            "total_decisions": total,
            "tier_breakdown": tier_breakdown,
            "top_models": top_models,
            "avg_cost": avg_cost,
        }

    def get_model_performance(self) -> list[dict[str, Any]]:
        """Get performance metrics per model: success_rate, avg_cost, usage_count.

        Returns:
            List of dicts with model_id, usage_count, success_rate, avg_cost.
        """
        rows = self.db.fetchall(
            """
            SELECT
                model_selected,
                COUNT(*) AS usage_count,
                SUM(CASE WHEN outcome = 'accepted' THEN 1 ELSE 0 END) AS accepted,
                SUM(CASE WHEN outcome != 'unknown' THEN 1 ELSE 0 END) AS known,
                AVG(actual_cost) AS avg_cost
            FROM routing_decisions
            GROUP BY model_selected
            ORDER BY usage_count DESC
            """,
        )
        results: list[dict[str, Any]] = []
        for row in rows:
            known = int(row["known"])
            success_rate = float(row["accepted"]) / known if known > 0 else 0.5
            results.append({
                "model_id": row["model_selected"],
                "usage_count": int(row["usage_count"]),
                "success_rate": round(success_rate, 4),
                "avg_cost": round(float(row["avg_cost"] or 0.0), 6),
            })
        return results


# ======================================================================
# Feedback tracker
# ======================================================================


class FeedbackTracker:
    """Track explicit user feedback (thumbs up/down) on responses.

    Stores feedback in the ``user_feedback`` table (created by migration 3).
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def record_feedback(
        self,
        session_id: str,
        model: str,
        tier: str,
        feedback: str,
        routing_decision_id: str | None = None,
    ) -> None:
        """Record user feedback.

        Args:
            session_id: Current session identifier.
            model: Model that generated the response.
            tier: Complexity tier for the request.
            feedback: ``"up"`` or ``"down"``.
            routing_decision_id: Optional link to a routing_decisions row.

        Raises:
            ValueError: If *feedback* is not ``"up"`` or ``"down"``.
        """
        if feedback not in ("up", "down"):
            raise ValueError(f"feedback must be 'up' or 'down', got {feedback!r}")

        now = datetime.now(UTC).isoformat()
        feedback_id = str(uuid4())

        self.db.execute(
            """
            INSERT INTO user_feedback (
                id, created_at, session_id, model, tier, feedback, routing_decision_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (feedback_id, now, session_id, model, tier, feedback, routing_decision_id),
        )
        self.db.commit()

        logger.info(
            "user_feedback_recorded",
            feedback=feedback,
            model=model,
            tier=tier,
        )

    def get_feedback_stats(self) -> dict[str, Any]:
        """Get feedback statistics by model and tier.

        Returns:
            Dict with ``total``, ``up``, ``down``, ``by_model``, ``by_tier``.
        """
        total_row = self.db.fetchone(
            "SELECT COUNT(*) AS total FROM user_feedback",
        )
        total = int(total_row["total"]) if total_row else 0

        up_row = self.db.fetchone(
            "SELECT COUNT(*) AS cnt FROM user_feedback WHERE feedback = 'up'",
        )
        up_count = int(up_row["cnt"]) if up_row else 0

        down_count = total - up_count

        model_rows = self.db.fetchall(
            """
            SELECT model,
                   SUM(CASE WHEN feedback = 'up' THEN 1 ELSE 0 END) AS up,
                   SUM(CASE WHEN feedback = 'down' THEN 1 ELSE 0 END) AS down,
                   COUNT(*) AS total
            FROM user_feedback
            GROUP BY model
            ORDER BY total DESC
            """,
        )
        by_model: list[dict[str, Any]] = [
            {
                "model": row["model"],
                "up": int(row["up"]),
                "down": int(row["down"]),
                "total": int(row["total"]),
            }
            for row in model_rows
        ]

        tier_rows = self.db.fetchall(
            """
            SELECT tier,
                   SUM(CASE WHEN feedback = 'up' THEN 1 ELSE 0 END) AS up,
                   SUM(CASE WHEN feedback = 'down' THEN 1 ELSE 0 END) AS down,
                   COUNT(*) AS total
            FROM user_feedback
            GROUP BY tier
            ORDER BY total DESC
            """,
        )
        by_tier: list[dict[str, Any]] = [
            {
                "tier": row["tier"],
                "up": int(row["up"]),
                "down": int(row["down"]),
                "total": int(row["total"]),
            }
            for row in tier_rows
        ]

        return {
            "total": total,
            "up": up_count,
            "down": down_count,
            "by_model": by_model,
            "by_tier": by_tier,
        }

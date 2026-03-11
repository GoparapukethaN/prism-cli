"""Integration tests for the adaptive learning feedback loop.

Tests RoutingDataExporter, FeedbackTracker, and AdaptiveLearner
working together with a real database. All tests run offline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from prism.db import models as dbm
from prism.db.queries import create_session, save_routing_decision
from prism.providers.base import ComplexityTier
from prism.router.learning import (
    AdaptiveLearner,
    FeedbackTracker,
    RoutingDataExporter,
)

if TYPE_CHECKING:
    from pathlib import Path

    from prism.db.database import Database

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _insert_routing_decision(
    db: Database,
    model: str = "gpt-4o-mini",
    tier: str = "medium",
    outcome: str = "accepted",
    cost: float = 0.001,
    session_id: str = "test-session",
) -> str:
    """Insert a routing decision and return its ID."""
    decision_id = str(uuid4())
    decision = dbm.RoutingDecision(
        id=decision_id,
        created_at=datetime.now(UTC).isoformat(),
        session_id=session_id,
        prompt_hash="hash-" + decision_id[:8],
        complexity_tier=dbm.ComplexityTier(tier),
        complexity_score=0.5,
        model_selected=model,
        fallback_chain="[]",
        estimated_cost=cost,
        actual_cost=cost,
        outcome=dbm.Outcome(outcome),
        features="{}",
    )
    save_routing_decision(db, decision)
    return decision_id


# ------------------------------------------------------------------
# RoutingDataExporter tests
# ------------------------------------------------------------------


class TestRoutingDataExporter:
    """Tests for CSV export and summary statistics."""

    def test_export_csv_creates_file(
        self, integration_db: Database, tmp_path: Path
    ) -> None:
        create_session(integration_db, "export-sess", "/tmp")
        for _ in range(5):
            _insert_routing_decision(
                integration_db, session_id="export-sess"
            )

        exporter = RoutingDataExporter(integration_db)
        csv_path = tmp_path / "export.csv"
        count = exporter.export_csv(csv_path, days=30)

        assert csv_path.exists()
        assert count == 5

        # Verify CSV structure
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 6  # header + 5 rows
        assert "model_selected" in lines[0]

    def test_export_csv_empty(
        self, integration_db: Database, tmp_path: Path
    ) -> None:
        exporter = RoutingDataExporter(integration_db)
        csv_path = tmp_path / "empty.csv"
        count = exporter.export_csv(csv_path, days=30)
        assert count == 0

    def test_export_summary(self, integration_db: Database) -> None:
        create_session(integration_db, "summary-sess", "/tmp")

        # Insert decisions with different tiers and models
        for _ in range(3):
            _insert_routing_decision(
                integration_db,
                model="gpt-4o-mini",
                tier="medium",
                session_id="summary-sess",
            )
        for _ in range(2):
            _insert_routing_decision(
                integration_db,
                model="gpt-4o",
                tier="complex",
                session_id="summary-sess",
            )

        exporter = RoutingDataExporter(integration_db)
        summary = exporter.export_summary()

        assert summary["total_decisions"] == 5
        assert "medium" in summary["tier_breakdown"]
        assert "complex" in summary["tier_breakdown"]
        assert summary["tier_breakdown"]["medium"] == 3
        assert summary["tier_breakdown"]["complex"] == 2
        assert len(summary["top_models"]) >= 2

    def test_get_model_performance(self, integration_db: Database) -> None:
        create_session(integration_db, "perf-sess", "/tmp")

        for _ in range(5):
            _insert_routing_decision(
                integration_db,
                model="gpt-4o-mini",
                outcome="accepted",
                session_id="perf-sess",
            )
        for _ in range(2):
            _insert_routing_decision(
                integration_db,
                model="gpt-4o-mini",
                outcome="rejected",
                session_id="perf-sess",
            )

        exporter = RoutingDataExporter(integration_db)
        perf = exporter.get_model_performance()

        assert len(perf) >= 1
        mini_perf = next(p for p in perf if p["model_id"] == "gpt-4o-mini")
        assert mini_perf["usage_count"] == 7
        # 5 accepted out of 7 known outcomes
        assert abs(mini_perf["success_rate"] - 5 / 7) < 0.01


# ------------------------------------------------------------------
# FeedbackTracker tests
# ------------------------------------------------------------------


class TestFeedbackTracker:
    """Tests for explicit user feedback tracking."""

    def test_record_feedback_up(self, integration_db: Database) -> None:
        tracker = FeedbackTracker(integration_db)
        tracker.record_feedback(
            session_id="fb-sess",
            model="gpt-4o-mini",
            tier="medium",
            feedback="up",
        )

        row = integration_db.fetchone(
            "SELECT * FROM user_feedback WHERE session_id = 'fb-sess'"
        )
        assert row is not None
        assert row["feedback"] == "up"
        assert row["model"] == "gpt-4o-mini"
        assert row["tier"] == "medium"

    def test_record_feedback_down(self, integration_db: Database) -> None:
        tracker = FeedbackTracker(integration_db)
        tracker.record_feedback(
            session_id="fb-sess-2",
            model="gpt-4o",
            tier="complex",
            feedback="down",
        )

        row = integration_db.fetchone(
            "SELECT * FROM user_feedback WHERE session_id = 'fb-sess-2'"
        )
        assert row is not None
        assert row["feedback"] == "down"

    def test_record_feedback_invalid(self, integration_db: Database) -> None:
        tracker = FeedbackTracker(integration_db)
        with pytest.raises(ValueError, match="up.*or.*down"):
            tracker.record_feedback(
                session_id="fb-bad",
                model="gpt-4o-mini",
                tier="medium",
                feedback="maybe",
            )

    def test_record_feedback_with_routing_decision(
        self, integration_db: Database
    ) -> None:
        create_session(integration_db, "fb-rd-sess", "/tmp")
        decision_id = _insert_routing_decision(
            integration_db, session_id="fb-rd-sess"
        )
        tracker = FeedbackTracker(integration_db)
        tracker.record_feedback(
            session_id="fb-rd-sess",
            model="gpt-4o-mini",
            tier="medium",
            feedback="up",
            routing_decision_id=decision_id,
        )

        row = integration_db.fetchone(
            "SELECT * FROM user_feedback WHERE routing_decision_id = ?",
            (decision_id,),
        )
        assert row is not None
        assert row["routing_decision_id"] == decision_id

    def test_get_feedback_stats(self, integration_db: Database) -> None:
        tracker = FeedbackTracker(integration_db)

        # Record mixed feedback
        for _ in range(4):
            tracker.record_feedback("s1", "gpt-4o-mini", "medium", "up")
        for _ in range(2):
            tracker.record_feedback("s1", "gpt-4o-mini", "medium", "down")
        for _ in range(3):
            tracker.record_feedback("s1", "gpt-4o", "complex", "up")
        tracker.record_feedback("s1", "gpt-4o", "complex", "down")

        stats = tracker.get_feedback_stats()
        assert stats["total"] == 10
        assert stats["up"] == 7
        assert stats["down"] == 3

        # by_model checks
        assert len(stats["by_model"]) == 2
        mini = next(m for m in stats["by_model"] if m["model"] == "gpt-4o-mini")
        assert mini["up"] == 4
        assert mini["down"] == 2

        # by_tier checks
        assert len(stats["by_tier"]) == 2


# ------------------------------------------------------------------
# AdaptiveLearner tests
# ------------------------------------------------------------------


class TestAdaptiveLearnerIntegration:
    """AdaptiveLearner with enough data to make recommendations."""

    def test_learner_activates_after_threshold(self) -> None:
        learner = AdaptiveLearner(min_interactions=5)
        assert not learner.is_active

        for _i in range(5):
            learner.record_outcome(
                model="gpt-4o-mini",
                tier=ComplexityTier.MEDIUM,
                outcome="accepted",
                cost=0.001,
            )

        assert learner.is_active
        rec = learner.get_model_recommendation(ComplexityTier.MEDIUM)
        assert rec == "gpt-4o-mini"

    def test_learner_recommends_best_model(self) -> None:
        learner = AdaptiveLearner(min_interactions=10)

        # Record 7 accepted for model A
        for _ in range(7):
            learner.record_outcome("model-a", ComplexityTier.SIMPLE, "accepted", 0.0)

        # Record 3 rejected for model B
        for _ in range(3):
            learner.record_outcome("model-b", ComplexityTier.SIMPLE, "rejected", 0.0)

        assert learner.is_active
        rec = learner.get_model_recommendation(ComplexityTier.SIMPLE)
        assert rec == "model-a"

    def test_learner_tracks_cost(self) -> None:
        learner = AdaptiveLearner(min_interactions=1)

        learner.record_outcome("model-x", ComplexityTier.MEDIUM, "accepted", 0.05)
        learner.record_outcome("model-x", ComplexityTier.MEDIUM, "accepted", 0.03)

        assert abs(learner.get_total_cost("model-x") - 0.08) < 1e-9

    def test_learner_ewma_tracks_trend(self) -> None:
        learner = AdaptiveLearner(min_interactions=1)

        # Start with accepted outcomes
        for _ in range(20):
            learner.record_outcome("model-t", ComplexityTier.COMPLEX, "accepted", 0.0)

        high_rate = learner.get_success_rate("model-t")

        # Switch to rejected outcomes
        for _ in range(20):
            learner.record_outcome("model-t", ComplexityTier.COMPLEX, "rejected", 0.0)

        low_rate = learner.get_success_rate("model-t")
        assert low_rate < high_rate

    def test_learner_model_stats(self) -> None:
        learner = AdaptiveLearner(min_interactions=1)

        learner.record_outcome("model-s", ComplexityTier.MEDIUM, "accepted", 0.01)
        learner.record_outcome("model-s", ComplexityTier.MEDIUM, "rejected", 0.02)

        stats = learner.get_model_stats("model-s")
        assert stats["interactions"] == 2
        assert abs(stats["total_cost"] - 0.03) < 1e-9  # type: ignore[arg-type]
        assert isinstance(stats["success_rate"], float)

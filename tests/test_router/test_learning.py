"""Tests for prism.router.learning — AdaptiveLearner."""

from __future__ import annotations

import pytest

from prism.providers.base import ComplexityTier
from prism.router.learning import AdaptiveLearner

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def learner() -> AdaptiveLearner:
    """A fresh learner with default settings."""
    return AdaptiveLearner()


@pytest.fixture
def learner_low_threshold() -> AdaptiveLearner:
    """A learner with a low min_interactions threshold for easy testing."""
    return AdaptiveLearner(min_interactions=5)


def _fill_interactions(
    learner: AdaptiveLearner,
    model: str,
    tier: ComplexityTier,
    outcome: str,
    count: int,
    cost: float = 0.01,
) -> None:
    """Record *count* identical interactions."""
    for _ in range(count):
        learner.record_outcome(model, tier, outcome, cost)


# ------------------------------------------------------------------
# record_outcome
# ------------------------------------------------------------------


class TestRecordOutcome:
    """Tests for recording interaction outcomes."""

    def test_increments_total_interactions(self, learner: AdaptiveLearner) -> None:
        assert learner.total_interactions == 0
        learner.record_outcome("model-a", ComplexityTier.SIMPLE, "accepted", 0.001)
        assert learner.total_interactions == 1

    def test_tracks_cost(self, learner: AdaptiveLearner) -> None:
        learner.record_outcome("model-a", ComplexityTier.SIMPLE, "accepted", 0.05)
        learner.record_outcome("model-a", ComplexityTier.SIMPLE, "accepted", 0.03)
        assert abs(learner.get_total_cost("model-a") - 0.08) < 1e-9

    def test_multiple_models(self, learner: AdaptiveLearner) -> None:
        learner.record_outcome("model-a", ComplexityTier.SIMPLE, "accepted", 0.01)
        learner.record_outcome("model-b", ComplexityTier.MEDIUM, "rejected", 0.02)
        assert learner.total_interactions == 2
        assert learner.get_total_cost("model-a") == pytest.approx(0.01)
        assert learner.get_total_cost("model-b") == pytest.approx(0.02)


# ------------------------------------------------------------------
# get_success_rate
# ------------------------------------------------------------------


class TestGetSuccessRate:
    """Tests for the EWMA success rate."""

    def test_neutral_for_unknown_model(self, learner: AdaptiveLearner) -> None:
        assert learner.get_success_rate("never-seen") == 0.5

    def test_increases_on_accepted(self, learner: AdaptiveLearner) -> None:
        initial = learner.get_success_rate("model-a")
        learner.record_outcome("model-a", ComplexityTier.SIMPLE, "accepted", 0.01)
        assert learner.get_success_rate("model-a") > initial

    def test_decreases_on_rejected(self, learner: AdaptiveLearner) -> None:
        initial = learner.get_success_rate("model-a")
        learner.record_outcome("model-a", ComplexityTier.SIMPLE, "rejected", 0.01)
        assert learner.get_success_rate("model-a") < initial

    def test_corrected_is_middle_ground(self, learner: AdaptiveLearner) -> None:
        # Start neutral, record corrected — should stay near 0.5
        learner.record_outcome("model-a", ComplexityTier.SIMPLE, "corrected", 0.01)
        rate = learner.get_success_rate("model-a")
        assert 0.48 <= rate <= 0.52

    def test_many_accepts_push_rate_high(self, learner: AdaptiveLearner) -> None:
        _fill_interactions(learner, "model-a", ComplexityTier.SIMPLE, "accepted", 100)
        assert learner.get_success_rate("model-a") > 0.9

    def test_many_rejects_push_rate_low(self, learner: AdaptiveLearner) -> None:
        _fill_interactions(learner, "model-a", ComplexityTier.SIMPLE, "rejected", 100)
        assert learner.get_success_rate("model-a") < 0.1


# ------------------------------------------------------------------
# get_model_recommendation
# ------------------------------------------------------------------


class TestGetModelRecommendation:
    """Tests for tier-based model recommendations."""

    def test_none_before_threshold(self, learner: AdaptiveLearner) -> None:
        learner.record_outcome("model-a", ComplexityTier.SIMPLE, "accepted", 0.01)
        assert learner.get_model_recommendation(ComplexityTier.SIMPLE) is None

    def test_returns_best_after_threshold(
        self, learner_low_threshold: AdaptiveLearner
    ) -> None:
        lrn = learner_low_threshold
        # model-a: all accepted (good)
        _fill_interactions(lrn, "model-a", ComplexityTier.SIMPLE, "accepted", 4)
        # model-b: all rejected (bad)
        _fill_interactions(lrn, "model-b", ComplexityTier.SIMPLE, "rejected", 4)
        assert lrn.total_interactions >= 5
        rec = lrn.get_model_recommendation(ComplexityTier.SIMPLE)
        assert rec == "model-a"

    def test_none_for_unseen_tier(
        self, learner_low_threshold: AdaptiveLearner
    ) -> None:
        lrn = learner_low_threshold
        _fill_interactions(lrn, "model-a", ComplexityTier.SIMPLE, "accepted", 6)
        # No COMPLEX interactions recorded
        assert lrn.get_model_recommendation(ComplexityTier.COMPLEX) is None

    def test_recommends_per_tier(
        self, learner_low_threshold: AdaptiveLearner
    ) -> None:
        lrn = learner_low_threshold
        _fill_interactions(lrn, "model-a", ComplexityTier.SIMPLE, "accepted", 3)
        _fill_interactions(lrn, "model-b", ComplexityTier.MEDIUM, "accepted", 3)
        assert lrn.get_model_recommendation(ComplexityTier.SIMPLE) == "model-a"
        assert lrn.get_model_recommendation(ComplexityTier.MEDIUM) == "model-b"


# ------------------------------------------------------------------
# is_active
# ------------------------------------------------------------------


class TestIsActive:
    """Tests for the activation threshold."""

    def test_not_active_initially(self, learner: AdaptiveLearner) -> None:
        assert learner.is_active is False

    def test_active_after_threshold(self) -> None:
        lrn = AdaptiveLearner(min_interactions=3)
        _fill_interactions(lrn, "m", ComplexityTier.SIMPLE, "accepted", 3)
        assert lrn.is_active is True


# ------------------------------------------------------------------
# get_model_stats
# ------------------------------------------------------------------


class TestGetModelStats:
    """Tests for the per-model statistics summary."""

    def test_stats_unknown_model(self, learner: AdaptiveLearner) -> None:
        stats = learner.get_model_stats("unknown")
        assert stats["interactions"] == 0
        assert stats["success_rate"] == 0.5
        assert stats["total_cost"] == 0.0

    def test_stats_after_interactions(self, learner: AdaptiveLearner) -> None:
        _fill_interactions(learner, "model-x", ComplexityTier.MEDIUM, "accepted", 10, cost=0.05)
        stats = learner.get_model_stats("model-x")
        assert stats["interactions"] == 10
        assert stats["total_cost"] == pytest.approx(0.50)
        assert stats["success_rate"] > 0.5

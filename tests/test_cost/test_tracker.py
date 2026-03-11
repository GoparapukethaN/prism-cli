"""Tests for the CostTracker module.

Covers: track(), get_session_cost(), get_daily_cost(), get_monthly_cost(),
get_budget_remaining(), check_budget(), get_cost_summary(), calculate_savings(),
and BudgetAction.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from prism.cost.tracker import (
    BudgetAction,
    CostEntry,
    CostSummary,
    CostTracker,
    ModelCostBreakdown,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_settings(
    daily_limit: float | None = None,
    monthly_limit: float | None = None,
    warn_at_percent: float = 80.0,
) -> MagicMock:
    """Build a mock Settings that responds to .get() for budget keys."""
    mapping: dict[str, Any] = {
        "budget.daily_limit": daily_limit,
        "budget.monthly_limit": monthly_limit,
        "budget.warn_at_percent": warn_at_percent,
    }
    settings = MagicMock()
    settings.get = MagicMock(side_effect=lambda key, default=None: mapping.get(key, default))
    return settings


def _make_tracker(
    *,
    daily_limit: float | None = None,
    monthly_limit: float | None = None,
    warn_at_percent: float = 80.0,
) -> CostTracker:
    """Create a CostTracker with a mock db and settings."""
    db = MagicMock()
    settings = _make_settings(daily_limit, monthly_limit, warn_at_percent)
    return CostTracker(db=db, settings=settings)


# ------------------------------------------------------------------
# CostEntry dataclass
# ------------------------------------------------------------------


class TestCostEntry:
    """Tests for the CostEntry frozen dataclass."""

    def test_create_entry(self) -> None:
        entry = CostEntry(
            id="abc-123",
            created_at="2026-01-15T10:00:00+00:00",
            session_id="sess-1",
            model_id="gpt-4o",
            provider="openai",
            input_tokens=500,
            output_tokens=200,
            cached_tokens=100,
            cost_usd=0.005,
            complexity_tier="medium",
        )
        assert entry.id == "abc-123"
        assert entry.model_id == "gpt-4o"
        assert entry.provider == "openai"
        assert entry.input_tokens == 500
        assert entry.output_tokens == 200
        assert entry.cached_tokens == 100
        assert entry.cost_usd == 0.005
        assert entry.complexity_tier == "medium"

    def test_entry_is_frozen(self) -> None:
        entry = CostEntry(
            id="x",
            created_at="2026-01-01T00:00:00",
            session_id="s",
            model_id="m",
            provider="p",
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            cost_usd=0.0,
            complexity_tier="simple",
        )
        with pytest.raises(AttributeError):
            entry.cost_usd = 99.0  # type: ignore[misc]


# ------------------------------------------------------------------
# CostSummary / ModelCostBreakdown
# ------------------------------------------------------------------


class TestCostSummary:
    def test_defaults(self) -> None:
        summary = CostSummary(period="day", total_cost=1.23, total_requests=5)
        assert summary.model_breakdown == []
        assert summary.budget_limit is None
        assert summary.budget_remaining is None

    def test_with_budget(self) -> None:
        summary = CostSummary(
            period="month",
            total_cost=4.50,
            total_requests=12,
            budget_limit=10.0,
            budget_remaining=5.50,
        )
        assert summary.budget_limit == 10.0
        assert summary.budget_remaining == 5.50


class TestModelCostBreakdown:
    def test_fields(self) -> None:
        b = ModelCostBreakdown(
            model_id="gpt-4o",
            display_name="GPT-4o",
            request_count=10,
            total_cost=1.25,
            percentage=55.5,
        )
        assert b.model_id == "gpt-4o"
        assert b.display_name == "GPT-4o"
        assert b.request_count == 10
        assert b.total_cost == 1.25
        assert b.percentage == 55.5


# ------------------------------------------------------------------
# BudgetAction
# ------------------------------------------------------------------


class TestBudgetAction:
    def test_action_values(self) -> None:
        assert BudgetAction.PROCEED == "proceed"
        assert BudgetAction.WARN == "warn"
        assert BudgetAction.BLOCK == "block"


# ------------------------------------------------------------------
# CostTracker.track()
# ------------------------------------------------------------------


class TestTrack:
    """Tests for CostTracker.track() — lines 94-137."""

    @patch("prism.cost.tracker.get_provider_for_model", return_value="openai")
    @patch("prism.cost.tracker.calculate_cost", return_value=0.0125)
    def test_track_returns_cost_entry(
        self,
        mock_calc: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        tracker = _make_tracker()

        with patch("prism.cost.tracker.uuid4", return_value="fixed-uuid"):
            entry = tracker.track(
                model_id="gpt-4o",
                input_tokens=1000,
                output_tokens=500,
                session_id="sess-1",
                complexity_tier="medium",
                cached_tokens=200,
            )

        assert isinstance(entry, CostEntry)
        assert entry.model_id == "gpt-4o"
        assert entry.provider == "openai"
        assert entry.input_tokens == 1000
        assert entry.output_tokens == 500
        assert entry.cached_tokens == 200
        assert entry.cost_usd == 0.0125
        assert entry.complexity_tier == "medium"
        assert entry.session_id == "sess-1"

    @patch("prism.cost.tracker.get_provider_for_model", return_value="openai")
    @patch(
        "prism.cost.tracker.calculate_cost",
        side_effect=ValueError("Unknown model"),
    )
    def test_track_unknown_model_logs_zero_cost(
        self,
        mock_calc: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        """When calculate_cost raises ValueError, cost defaults to 0.0."""
        tracker = _make_tracker()

        entry = tracker.track(
            model_id="unknown-model",
            input_tokens=100,
            output_tokens=50,
            session_id="sess-2",
            complexity_tier="simple",
        )

        assert entry.cost_usd == 0.0

    @patch("prism.cost.tracker.get_provider_for_model", return_value="anthropic")
    @patch("prism.cost.tracker.calculate_cost", return_value=0.003)
    def test_track_persists_to_database(
        self,
        mock_calc: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        """track() should call save_cost_entry and update_session."""
        tracker = _make_tracker()

        with (
            patch("prism.db.queries.save_cost_entry") as mock_save,
            patch("prism.db.queries.update_session") as mock_update,
        ):
            tracker.track(
                model_id="claude-sonnet-4-20250514",
                input_tokens=500,
                output_tokens=200,
                session_id="sess-3",
                complexity_tier="simple",
            )

            mock_save.assert_called_once()
            saved_entry = mock_save.call_args[0][1]
            assert saved_entry.model_id == "claude-sonnet-4-20250514"

            mock_update.assert_called_once_with(
                tracker._db, "sess-3", cost_delta=0.003, request_delta=1
            )

    @patch("prism.cost.tracker.get_provider_for_model", return_value="openai")
    @patch("prism.cost.tracker.calculate_cost", return_value=0.01)
    def test_track_db_error_does_not_raise(
        self,
        mock_calc: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        """Database errors during persistence should be caught, not propagated."""
        tracker = _make_tracker()

        with patch(
            "prism.db.queries.save_cost_entry",
            side_effect=RuntimeError("DB failure"),
        ):
            # Should not raise
            entry = tracker.track(
                model_id="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                session_id="sess-4",
                complexity_tier="medium",
            )

        assert entry.cost_usd == 0.01

    @patch("prism.cost.tracker.get_provider_for_model", return_value="openai")
    @patch("prism.cost.tracker.calculate_cost", return_value=0.005)
    def test_track_with_zero_cached_tokens(
        self,
        mock_calc: MagicMock,
        mock_provider: MagicMock,
    ) -> None:
        """Default cached_tokens=0 is applied correctly."""
        tracker = _make_tracker()

        with (
            patch("prism.db.queries.save_cost_entry"),
            patch("prism.db.queries.update_session"),
        ):
            entry = tracker.track(
                model_id="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                session_id="sess-5",
                complexity_tier="simple",
            )

        assert entry.cached_tokens == 0


# ------------------------------------------------------------------
# CostTracker.get_session_cost()
# ------------------------------------------------------------------


class TestGetSessionCost:
    """Tests for lines 152-154 (exception path)."""

    def test_returns_cost_on_success(self) -> None:
        tracker = _make_tracker()

        with patch(
            "prism.db.queries.get_session_cost",
            return_value=1.23,
        ):
            result = tracker.get_session_cost("sess-1")

        assert result == 1.23

    def test_returns_zero_on_exception(self) -> None:
        tracker = _make_tracker()

        with patch(
            "prism.db.queries.get_session_cost",
            side_effect=RuntimeError("DB error"),
        ):
            result = tracker.get_session_cost("sess-fail")

        assert result == 0.0


# ------------------------------------------------------------------
# CostTracker.get_daily_cost()
# ------------------------------------------------------------------


class TestGetDailyCost:
    """Tests for lines 166-168 (exception path)."""

    def test_returns_cost_on_success(self) -> None:
        tracker = _make_tracker()

        with patch("prism.db.queries.get_daily_cost", return_value=0.42):
            result = tracker.get_daily_cost()

        assert result == 0.42

    def test_returns_zero_on_exception(self) -> None:
        tracker = _make_tracker()

        with patch(
            "prism.db.queries.get_daily_cost",
            side_effect=RuntimeError("DB error"),
        ):
            result = tracker.get_daily_cost()

        assert result == 0.0


# ------------------------------------------------------------------
# CostTracker.get_monthly_cost()
# ------------------------------------------------------------------


class TestGetMonthlyCost:
    """Tests for lines 180-182 (exception path)."""

    def test_returns_cost_on_success(self) -> None:
        tracker = _make_tracker()

        with patch("prism.db.queries.get_monthly_cost", return_value=5.67):
            result = tracker.get_monthly_cost()

        assert result == 5.67

    def test_returns_zero_on_exception(self) -> None:
        tracker = _make_tracker()

        with patch(
            "prism.db.queries.get_monthly_cost",
            side_effect=RuntimeError("DB error"),
        ):
            result = tracker.get_monthly_cost()

        assert result == 0.0


# ------------------------------------------------------------------
# CostTracker.get_budget_remaining()
# ------------------------------------------------------------------


class TestGetBudgetRemaining:
    def test_no_limits_returns_none(self) -> None:
        tracker = _make_tracker(daily_limit=None, monthly_limit=None)
        assert tracker.get_budget_remaining() is None

    def test_daily_limit_only(self) -> None:
        tracker = _make_tracker(daily_limit=5.0)

        with patch.object(tracker, "get_daily_cost", return_value=3.0):
            remaining = tracker.get_budget_remaining()

        assert remaining == pytest.approx(2.0)

    def test_monthly_limit_only(self) -> None:
        tracker = _make_tracker(monthly_limit=50.0)

        with patch.object(tracker, "get_monthly_cost", return_value=30.0):
            remaining = tracker.get_budget_remaining()

        assert remaining == pytest.approx(20.0)

    def test_both_limits_returns_more_restrictive(self) -> None:
        """When both limits are set, return the smaller remaining value."""
        tracker = _make_tracker(daily_limit=5.0, monthly_limit=50.0)

        with (
            patch.object(tracker, "get_daily_cost", return_value=4.5),
            patch.object(tracker, "get_monthly_cost", return_value=10.0),
        ):
            remaining = tracker.get_budget_remaining()

        # daily remaining = 0.5, monthly remaining = 40.0 -> min is 0.5
        assert remaining == pytest.approx(0.5)

    def test_budget_remaining_floors_at_zero(self) -> None:
        """If cost exceeds limit, remaining should be 0.0, not negative."""
        tracker = _make_tracker(daily_limit=1.0)

        with patch.object(tracker, "get_daily_cost", return_value=2.0):
            remaining = tracker.get_budget_remaining()

        assert remaining == 0.0


# ------------------------------------------------------------------
# CostTracker.check_budget() — lines 231-243
# ------------------------------------------------------------------


class TestCheckBudget:
    def test_no_limits_proceed(self) -> None:
        tracker = _make_tracker()
        with patch.object(tracker, "get_budget_remaining", return_value=None):
            assert tracker.check_budget(1.0) == BudgetAction.PROCEED

    def test_within_budget_proceed(self) -> None:
        tracker = _make_tracker(daily_limit=10.0)

        with (
            patch.object(tracker, "get_budget_remaining", return_value=5.0),
            patch.object(tracker, "get_daily_cost", return_value=1.0),
        ):
            assert tracker.check_budget(0.01) == BudgetAction.PROCEED

    def test_within_budget_but_warn_daily(self) -> None:
        """When daily cost >= warn_at_percent of daily limit, should WARN."""
        tracker = _make_tracker(daily_limit=10.0, warn_at_percent=80.0)

        with (
            patch.object(tracker, "get_budget_remaining", return_value=1.5),
            patch.object(tracker, "get_daily_cost", return_value=8.5),
        ):
            action = tracker.check_budget(0.01)

        assert action == BudgetAction.WARN

    def test_within_budget_but_warn_monthly(self) -> None:
        """When monthly cost >= warn_at_percent of monthly limit, should WARN."""
        tracker = _make_tracker(monthly_limit=100.0, warn_at_percent=80.0)

        with (
            patch.object(tracker, "get_budget_remaining", return_value=15.0),
            patch.object(tracker, "get_monthly_cost", return_value=85.0),
        ):
            action = tracker.check_budget(0.01)

        assert action == BudgetAction.WARN

    def test_slightly_over_budget_warn(self) -> None:
        """estimated_cost > remaining but <= remaining * 1.5 -> WARN (line 242-243)."""
        tracker = _make_tracker(daily_limit=10.0)

        with patch.object(tracker, "get_budget_remaining", return_value=1.0):
            # 1.4 > 1.0 but <= 1.5 -> WARN
            action = tracker.check_budget(1.4)

        assert action == BudgetAction.WARN

    def test_far_over_budget_block(self) -> None:
        """estimated_cost > remaining * 1.5 -> BLOCK."""
        tracker = _make_tracker(daily_limit=10.0)

        with patch.object(tracker, "get_budget_remaining", return_value=1.0):
            # 2.0 > 1.5 -> BLOCK
            action = tracker.check_budget(2.0)

        assert action == BudgetAction.BLOCK

    def test_exactly_at_budget_proceed(self) -> None:
        """estimated_cost == remaining and not at warn threshold -> PROCEED."""
        tracker = _make_tracker(daily_limit=10.0, warn_at_percent=80.0)

        with (
            patch.object(tracker, "get_budget_remaining", return_value=5.0),
            patch.object(tracker, "get_daily_cost", return_value=5.0),
        ):
            # cost fits, daily usage is 50% < 80% warn threshold
            action = tracker.check_budget(5.0)

        assert action == BudgetAction.PROCEED

    def test_zero_remaining_blocks(self) -> None:
        """When remaining is 0.0, any positive cost > 0 * 1.5 = 0 -> BLOCK."""
        tracker = _make_tracker(daily_limit=10.0)

        with patch.object(tracker, "get_budget_remaining", return_value=0.0):
            action = tracker.check_budget(0.01)

        assert action == BudgetAction.BLOCK

    def test_warn_both_daily_and_monthly(self) -> None:
        """When both daily and monthly are set and both exceed warn threshold."""
        tracker = _make_tracker(
            daily_limit=10.0, monthly_limit=100.0, warn_at_percent=80.0
        )

        with (
            patch.object(tracker, "get_budget_remaining", return_value=5.0),
            patch.object(tracker, "get_daily_cost", return_value=8.5),
            patch.object(tracker, "get_monthly_cost", return_value=85.0),
        ):
            action = tracker.check_budget(0.01)

        assert action == BudgetAction.WARN


# ------------------------------------------------------------------
# CostTracker.get_cost_summary() — lines 257-288
# ------------------------------------------------------------------


class TestGetCostSummary:
    def test_session_summary(self) -> None:
        tracker = _make_tracker()
        breakdown_data = [
            {"model_id": "gpt-4o", "request_count": 5, "total_cost": 0.5},
            {"model_id": "gpt-4o-mini", "request_count": 10, "total_cost": 0.1},
        ]

        with patch("prism.db.queries.get_cost_breakdown", return_value=breakdown_data):
            summary = tracker.get_cost_summary("session", session_id="sess-1")

        assert summary.period == "session"
        assert summary.total_cost == pytest.approx(0.6)
        assert summary.total_requests == 15
        assert len(summary.model_breakdown) == 2
        assert summary.budget_limit is None
        assert summary.budget_remaining is None

    def test_day_summary_with_budget(self) -> None:
        tracker = _make_tracker(daily_limit=5.0)
        breakdown_data = [
            {"model_id": "gpt-4o", "request_count": 3, "total_cost": 2.0},
        ]

        with patch("prism.db.queries.get_cost_breakdown", return_value=breakdown_data):
            summary = tracker.get_cost_summary("day")

        assert summary.period == "day"
        assert summary.budget_limit == 5.0
        assert summary.budget_remaining == pytest.approx(3.0)

    def test_month_summary_with_budget(self) -> None:
        tracker = _make_tracker(monthly_limit=50.0)
        breakdown_data = [
            {"model_id": "gpt-4o", "request_count": 20, "total_cost": 30.0},
        ]

        with patch("prism.db.queries.get_cost_breakdown", return_value=breakdown_data):
            summary = tracker.get_cost_summary("month")

        assert summary.period == "month"
        assert summary.budget_limit == 50.0
        assert summary.budget_remaining == pytest.approx(20.0)

    def test_empty_breakdown(self) -> None:
        tracker = _make_tracker()

        with patch("prism.db.queries.get_cost_breakdown", return_value=[]):
            summary = tracker.get_cost_summary("day")

        assert summary.total_cost == 0.0
        assert summary.total_requests == 0
        assert summary.model_breakdown == []

    def test_db_error_returns_empty_summary(self) -> None:
        """Line 261-263: exception in get_cost_breakdown yields empty data."""
        tracker = _make_tracker()

        with patch(
            "prism.db.queries.get_cost_breakdown",
            side_effect=RuntimeError("DB error"),
        ):
            summary = tracker.get_cost_summary("session", session_id="x")

        assert summary.total_cost == 0.0
        assert summary.total_requests == 0
        assert summary.model_breakdown == []

    def test_percentage_calculation(self) -> None:
        """Each model's percentage should be (model_cost / total_cost * 100)."""
        tracker = _make_tracker()
        breakdown_data = [
            {"model_id": "gpt-4o", "request_count": 5, "total_cost": 3.0},
            {"model_id": "gpt-4o-mini", "request_count": 10, "total_cost": 1.0},
        ]

        with patch("prism.db.queries.get_cost_breakdown", return_value=breakdown_data):
            summary = tracker.get_cost_summary("month")

        gpt4o = next(m for m in summary.model_breakdown if m.model_id == "gpt-4o")
        gpt4o_mini = next(m for m in summary.model_breakdown if m.model_id == "gpt-4o-mini")

        assert gpt4o.percentage == pytest.approx(75.0)
        assert gpt4o_mini.percentage == pytest.approx(25.0)

    def test_percentage_zero_when_total_zero(self) -> None:
        """If total_cost is 0, percentage should be 0 (avoid div-by-zero)."""
        tracker = _make_tracker()
        breakdown_data = [
            {"model_id": "ollama/llama3.2:3b", "request_count": 5, "total_cost": 0.0},
        ]

        with patch("prism.db.queries.get_cost_breakdown", return_value=breakdown_data):
            summary = tracker.get_cost_summary("session", session_id="s")

        assert summary.model_breakdown[0].percentage == 0.0

    def test_display_name_fallback(self) -> None:
        """If display_name is not in row, falls back to model_id."""
        tracker = _make_tracker()
        breakdown_data = [
            {"model_id": "gpt-4o", "request_count": 1, "total_cost": 0.5},
        ]

        with patch("prism.db.queries.get_cost_breakdown", return_value=breakdown_data):
            summary = tracker.get_cost_summary("session", session_id="s")

        assert summary.model_breakdown[0].display_name == "gpt-4o"

    def test_display_name_from_data(self) -> None:
        """If display_name is in row, it should be used."""
        tracker = _make_tracker()
        breakdown_data = [
            {
                "model_id": "gpt-4o",
                "display_name": "GPT-4o",
                "request_count": 1,
                "total_cost": 0.5,
            },
        ]

        with patch("prism.db.queries.get_cost_breakdown", return_value=breakdown_data):
            summary = tracker.get_cost_summary("session", session_id="s")

        assert summary.model_breakdown[0].display_name == "GPT-4o"

    def test_budget_remaining_floors_at_zero(self) -> None:
        """budget_remaining should not go negative."""
        tracker = _make_tracker(daily_limit=1.0)
        breakdown_data = [
            {"model_id": "gpt-4o", "request_count": 100, "total_cost": 5.0},
        ]

        with patch("prism.db.queries.get_cost_breakdown", return_value=breakdown_data):
            summary = tracker.get_cost_summary("day")

        assert summary.budget_remaining == 0.0


# ------------------------------------------------------------------
# CostTracker.calculate_savings() — lines 306-328
# ------------------------------------------------------------------


class TestCalculateSavings:
    def test_savings_calculation(self) -> None:
        tracker = _make_tracker()

        entries = [
            {"input_tokens": 1000, "output_tokens": 500, "cost_usd": 0.001},
            {"input_tokens": 2000, "output_tokens": 1000, "cost_usd": 0.002},
        ]

        with (
            patch(
                "prism.db.queries.get_all_cost_entries_for_month",
                return_value=entries,
                create=True,
            ),
            patch(
                "prism.cost.pricing.MODEL_PRICING",
                {
                    "claude-sonnet-4-20250514": MagicMock(
                        input_cost_per_1m=3.0, output_cost_per_1m=15.0
                    ),
                },
            ),
        ):
            hypo, actual, savings = tracker.calculate_savings()

        # hypothetical: (1000+2000)/1M * 3.0 + (500+1000)/1M * 15.0
        expected_hypo = (3000 / 1_000_000) * 3.0 + (1500 / 1_000_000) * 15.0
        assert hypo == pytest.approx(expected_hypo, abs=1e-8)
        assert actual == pytest.approx(0.003, abs=1e-8)
        assert savings == pytest.approx(max(0.0, expected_hypo - 0.003), abs=1e-8)

    def test_savings_db_error_returns_zeros(self) -> None:
        """Line 311-313: DB error returns (0, 0, 0)."""
        tracker = _make_tracker()

        with patch(
            "prism.db.queries.get_all_cost_entries_for_month",
            side_effect=RuntimeError("DB error"),
            create=True,
        ):
            hypo, actual, savings = tracker.calculate_savings()

        assert hypo == 0.0
        assert actual == 0.0
        assert savings == 0.0

    def test_savings_no_premium_pricing_returns_zeros(self) -> None:
        """Line 316-317: if claude-sonnet-4-20250514 not in MODEL_PRICING."""
        tracker = _make_tracker()

        entries = [
            {"input_tokens": 1000, "output_tokens": 500, "cost_usd": 0.001},
        ]

        with (
            patch(
                "prism.db.queries.get_all_cost_entries_for_month",
                return_value=entries,
                create=True,
            ),
            patch("prism.cost.pricing.MODEL_PRICING", {}),
        ):
            hypo, actual, savings = tracker.calculate_savings()

        assert hypo == 0.0
        assert actual == 0.0
        assert savings == 0.0

    def test_savings_no_entries(self) -> None:
        """Empty entries list should yield zero savings."""
        tracker = _make_tracker()

        with (
            patch(
                "prism.db.queries.get_all_cost_entries_for_month",
                return_value=[],
                create=True,
            ),
            patch(
                "prism.cost.pricing.MODEL_PRICING",
                {
                    "claude-sonnet-4-20250514": MagicMock(
                        input_cost_per_1m=3.0, output_cost_per_1m=15.0
                    ),
                },
            ),
        ):
            hypo, actual, savings = tracker.calculate_savings()

        assert hypo == 0.0
        assert actual == 0.0
        assert savings == 0.0

    def test_savings_never_negative(self) -> None:
        """If actual > hypothetical, savings should floor at 0."""
        tracker = _make_tracker()

        # Actual cost is higher than hypothetical premium cost
        entries = [
            {"input_tokens": 100, "output_tokens": 50, "cost_usd": 100.0},
        ]

        with (
            patch(
                "prism.db.queries.get_all_cost_entries_for_month",
                return_value=entries,
                create=True,
            ),
            patch(
                "prism.cost.pricing.MODEL_PRICING",
                {
                    "claude-sonnet-4-20250514": MagicMock(
                        input_cost_per_1m=3.0, output_cost_per_1m=15.0
                    ),
                },
            ),
        ):
            _hypo, _actual, savings = tracker.calculate_savings()

        assert savings == 0.0

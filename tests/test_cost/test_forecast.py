"""Tests for the cost forecasting module.

Covers: SpendingVelocity, ModelCostDriver, CostForecast, FeatureCost,
CostForecaster (init, track_request, get_velocity, forecast with/without
budget/velocity, model drivers, cheapest alternative, feature tracking,
weekly report generation), and edge cases.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from prism.cost.forecast import (
    ALTERNATIVE_MAP,
    COST_RATIOS,
    CostForecast,
    CostForecaster,
    FeatureCost,
    ModelCostDriver,
    SpendingVelocity,
)

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_settings(
    monthly_limit: float | None = None,
) -> MagicMock:
    """Build a mock Settings that responds to .get() for budget keys."""
    mapping: dict[str, Any] = {
        "budget.monthly_limit": monthly_limit,
    }
    settings = MagicMock()
    settings.get = MagicMock(side_effect=lambda key, default=None: mapping.get(key, default))
    return settings


def _make_tracker(monthly_cost: float = 0.0) -> MagicMock:
    """Build a mock CostTracker with configurable monthly cost."""
    tracker = MagicMock()
    tracker.get_monthly_cost = MagicMock(return_value=monthly_cost)
    return tracker


def _make_forecaster(
    *,
    monthly_cost: float = 0.0,
    monthly_limit: float | None = None,
    reports_dir: Path | None = None,
) -> CostForecaster:
    """Create a CostForecaster with mocked dependencies."""
    tracker = _make_tracker(monthly_cost)
    settings = _make_settings(monthly_limit)
    return CostForecaster(
        cost_tracker=tracker,
        settings=settings,
        reports_dir=reports_dir,
    )


# ==================================================================
# SpendingVelocity dataclass
# ==================================================================


class TestSpendingVelocity:
    """Tests for the SpendingVelocity dataclass."""

    def test_default_fields(self) -> None:
        vel = SpendingVelocity(
            cost_per_hour=1.5,
            tokens_per_hour=10000,
            requests_per_hour=5.0,
            window_hours=2.0,
            total_cost=3.0,
            total_tokens=20000,
            total_requests=10,
        )
        assert vel.cost_per_hour == 1.5
        assert vel.tokens_per_hour == 10000
        assert vel.requests_per_hour == 5.0
        assert vel.window_hours == 2.0
        assert vel.total_cost == 3.0
        assert vel.total_tokens == 20000
        assert vel.total_requests == 10

    def test_zero_velocity(self) -> None:
        vel = SpendingVelocity(
            cost_per_hour=0.0,
            tokens_per_hour=0,
            requests_per_hour=0.0,
            window_hours=0.0,
            total_cost=0.0,
            total_tokens=0,
            total_requests=0,
        )
        assert vel.total_cost == 0.0
        assert vel.total_requests == 0


# ==================================================================
# ModelCostDriver dataclass
# ==================================================================


class TestModelCostDriver:
    """Tests for the ModelCostDriver dataclass."""

    def test_fields(self) -> None:
        driver = ModelCostDriver(
            model_id="gpt-4o",
            display_name="gpt-4o",
            total_cost=0.50,
            percentage=75.0,
            request_count=10,
            avg_cost_per_request=0.05,
        )
        assert driver.model_id == "gpt-4o"
        assert driver.cheapest_alternative is None
        assert driver.potential_savings == 0.0

    def test_with_alternative(self) -> None:
        driver = ModelCostDriver(
            model_id="gpt-4o",
            display_name="gpt-4o",
            total_cost=1.00,
            percentage=100.0,
            request_count=5,
            avg_cost_per_request=0.20,
            cheapest_alternative="deepseek/deepseek-chat",
            potential_savings=0.95,
        )
        assert driver.cheapest_alternative == "deepseek/deepseek-chat"
        assert driver.potential_savings == 0.95


# ==================================================================
# CostForecast dataclass
# ==================================================================


class TestCostForecast:
    """Tests for the CostForecast dataclass."""

    def test_fields(self) -> None:
        vel = SpendingVelocity(
            cost_per_hour=0.0, tokens_per_hour=0, requests_per_hour=0.0,
            window_hours=0.0, total_cost=0.0, total_tokens=0, total_requests=0,
        )
        fc = CostForecast(
            projected_monthly_cost=10.0,
            current_monthly_cost=5.0,
            daily_average=0.5,
            days_remaining=20,
            budget_limit=50.0,
            budget_used_percent=10.0,
            alert_level="ok",
            velocity=vel,
        )
        assert fc.projected_monthly_cost == 10.0
        assert fc.alert_level == "ok"
        assert fc.model_drivers == []

    def test_alert_levels(self) -> None:
        vel = SpendingVelocity(
            cost_per_hour=0.0, tokens_per_hour=0, requests_per_hour=0.0,
            window_hours=0.0, total_cost=0.0, total_tokens=0, total_requests=0,
        )
        for level in ("ok", "warning", "critical"):
            fc = CostForecast(
                projected_monthly_cost=0.0,
                current_monthly_cost=0.0,
                daily_average=0.0,
                days_remaining=1,
                budget_limit=None,
                budget_used_percent=0.0,
                alert_level=level,
                velocity=vel,
            )
            assert fc.alert_level == level


# ==================================================================
# FeatureCost dataclass
# ==================================================================


class TestFeatureCost:
    """Tests for the FeatureCost dataclass."""

    def test_fields(self) -> None:
        fc = FeatureCost(name="refactor-auth", started_at="2026-03-01T00:00:00+00:00")
        assert fc.name == "refactor-auth"
        assert fc.completed_at is None
        assert fc.total_cost == 0.0
        assert fc.request_count == 0
        assert fc.models_used == []

    def test_completed_feature(self) -> None:
        fc = FeatureCost(
            name="fix-bug",
            started_at="2026-03-01T00:00:00+00:00",
            completed_at="2026-03-01T01:00:00+00:00",
            total_cost=0.12,
            request_count=3,
            models_used=["gpt-4o", "deepseek/deepseek-chat"],
        )
        assert fc.completed_at is not None
        assert fc.total_cost == 0.12
        assert len(fc.models_used) == 2


# ==================================================================
# CostForecaster
# ==================================================================


class TestCostForecasterInit:
    """Tests for CostForecaster initialization."""

    def test_default_reports_dir(self, tmp_path: Path) -> None:
        """Reports directory is created on init."""
        reports = tmp_path / "reports"
        fc = _make_forecaster(reports_dir=reports)
        assert reports.exists()
        assert fc._session_costs == []
        assert fc._feature_costs == {}

    def test_custom_reports_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom_reports"
        _make_forecaster(reports_dir=custom)
        assert custom.is_dir()

    def test_nested_reports_dir_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "reports"
        _make_forecaster(reports_dir=nested)
        assert nested.is_dir()


class TestTrackRequest:
    """Tests for CostForecaster.track_request."""

    def test_track_single_request(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 0.01, 500, 200)
        assert len(fc._session_costs) == 1
        entry = fc._session_costs[0]
        assert entry["model_id"] == "gpt-4o"
        assert entry["cost_usd"] == 0.01
        assert entry["tokens"] == 700

    def test_track_multiple_requests(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 0.01, 500, 200)
        fc.track_request("deepseek/deepseek-chat", 0.001, 300, 100)
        fc.track_request("gpt-4o", 0.02, 1000, 400)
        assert len(fc._session_costs) == 3

    def test_track_request_empty_model_ignored(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("", 0.01, 500, 200)
        assert len(fc._session_costs) == 0

    def test_track_request_timestamp_present(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 0.01, 100, 50)
        assert "timestamp" in fc._session_costs[0]


class TestGetVelocity:
    """Tests for CostForecaster.get_velocity."""

    def test_empty_session(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        vel = fc.get_velocity()
        assert vel.cost_per_hour == 0.0
        assert vel.tokens_per_hour == 0
        assert vel.requests_per_hour == 0.0
        assert vel.window_hours == 0.0
        assert vel.total_cost == 0.0
        assert vel.total_tokens == 0
        assert vel.total_requests == 0

    def test_with_data(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 0.05, 1000, 500)
        fc.track_request("gpt-4o", 0.03, 600, 300)
        vel = fc.get_velocity()
        assert vel.total_cost == pytest.approx(0.08, abs=1e-9)
        assert vel.total_tokens == 2400
        assert vel.total_requests == 2
        assert vel.cost_per_hour > 0
        assert vel.tokens_per_hour > 0
        assert vel.requests_per_hour > 0
        assert vel.window_hours > 0

    def test_velocity_nonzero_elapsed(self, tmp_path: Path) -> None:
        """Even immediately after init, velocity calculation doesn't divide by zero."""
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc._session_start = datetime.now(UTC)  # Just started
        fc.track_request("gpt-4o", 1.0, 5000, 2000)
        vel = fc.get_velocity()
        assert vel.cost_per_hour > 0  # Not infinity or NaN

    def test_velocity_after_elapsed_time(self, tmp_path: Path) -> None:
        """Velocity reflects elapsed time accurately."""
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        # Simulate session started 1 hour ago
        fc._session_start = datetime.now(UTC) - timedelta(hours=1)
        fc.track_request("gpt-4o", 2.0, 10000, 5000)
        vel = fc.get_velocity()
        # Cost/hour should be approximately $2.00
        assert vel.cost_per_hour == pytest.approx(2.0, abs=0.1)
        assert vel.window_hours == pytest.approx(1.0, abs=0.05)


class TestForecast:
    """Tests for CostForecaster.forecast."""

    def test_forecast_no_budget(self, tmp_path: Path) -> None:
        fc = _make_forecaster(monthly_cost=5.0, reports_dir=tmp_path / "r")
        result = fc.forecast()
        assert result.current_monthly_cost == 5.0
        assert result.budget_limit is None
        assert result.budget_used_percent == 0.0
        assert result.alert_level == "ok"
        assert result.days_remaining >= 1
        assert result.projected_monthly_cost >= 0

    def test_forecast_with_budget_ok(self, tmp_path: Path) -> None:
        fc = _make_forecaster(
            monthly_cost=5.0, monthly_limit=100.0, reports_dir=tmp_path / "r",
        )
        result = fc.forecast()
        assert result.budget_limit == 100.0
        assert result.budget_used_percent == pytest.approx(5.0, abs=0.1)
        assert result.alert_level == "ok"

    def test_forecast_with_budget_warning(self, tmp_path: Path) -> None:
        fc = _make_forecaster(
            monthly_cost=75.0, monthly_limit=100.0, reports_dir=tmp_path / "r",
        )
        result = fc.forecast()
        assert result.budget_used_percent == pytest.approx(75.0, abs=0.1)
        assert result.alert_level == "warning"

    def test_forecast_with_budget_critical(self, tmp_path: Path) -> None:
        fc = _make_forecaster(
            monthly_cost=105.0, monthly_limit=100.0, reports_dir=tmp_path / "r",
        )
        result = fc.forecast()
        assert result.budget_used_percent == pytest.approx(105.0, abs=0.1)
        assert result.alert_level == "critical"

    def test_forecast_at_70_percent_boundary(self, tmp_path: Path) -> None:
        fc = _make_forecaster(
            monthly_cost=70.0, monthly_limit=100.0, reports_dir=tmp_path / "r",
        )
        result = fc.forecast()
        assert result.alert_level == "warning"

    def test_forecast_at_100_percent_boundary(self, tmp_path: Path) -> None:
        fc = _make_forecaster(
            monthly_cost=100.0, monthly_limit=100.0, reports_dir=tmp_path / "r",
        )
        result = fc.forecast()
        assert result.alert_level == "critical"

    def test_forecast_below_70_percent(self, tmp_path: Path) -> None:
        fc = _make_forecaster(
            monthly_cost=69.9, monthly_limit=100.0, reports_dir=tmp_path / "r",
        )
        result = fc.forecast()
        assert result.alert_level == "ok"

    def test_forecast_velocity_blending(self, tmp_path: Path) -> None:
        """When session velocity is present, projection should blend it in."""
        fc = _make_forecaster(monthly_cost=10.0, reports_dir=tmp_path / "r")
        # Simulate 1 hour of session, $5/hr spending
        fc._session_start = datetime.now(UTC) - timedelta(hours=1)
        fc.track_request("gpt-4o", 5.0, 50000, 25000)
        result = fc.forecast()
        # With velocity blending, projected should be higher than pure historical
        day_of_month = datetime.now(UTC).day
        pure_historical = 10.0 + (10.0 / max(day_of_month, 1)) * result.days_remaining
        assert result.projected_monthly_cost > pure_historical

    def test_forecast_velocity_blending_no_historical(self, tmp_path: Path) -> None:
        """When no historical cost, use only session velocity."""
        fc = _make_forecaster(monthly_cost=0.0, reports_dir=tmp_path / "r")
        fc._session_start = datetime.now(UTC) - timedelta(hours=1)
        fc.track_request("gpt-4o", 2.0, 20000, 10000)
        result = fc.forecast()
        # Should project based on session rate only: $2/hr * 8hrs/day * days_remaining
        assert result.projected_monthly_cost > 0

    def test_forecast_contains_velocity(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 0.01, 100, 50)
        result = fc.forecast()
        assert isinstance(result.velocity, SpendingVelocity)
        assert result.velocity.total_requests == 1

    def test_forecast_model_drivers_populated(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 0.05, 1000, 500)
        fc.track_request("deepseek/deepseek-chat", 0.001, 200, 100)
        result = fc.forecast()
        assert len(result.model_drivers) == 2
        # Sorted by cost descending
        assert result.model_drivers[0].model_id == "gpt-4o"
        assert result.model_drivers[1].model_id == "deepseek/deepseek-chat"


class TestModelDrivers:
    """Tests for _get_model_drivers analysis."""

    def test_empty_session(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        drivers = fc._get_model_drivers()
        assert drivers == []

    def test_single_model(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 0.10, 2000, 1000)
        fc.track_request("gpt-4o", 0.05, 1000, 500)
        drivers = fc._get_model_drivers()
        assert len(drivers) == 1
        assert drivers[0].model_id == "gpt-4o"
        assert drivers[0].total_cost == pytest.approx(0.15)
        assert drivers[0].percentage == pytest.approx(100.0)
        assert drivers[0].request_count == 2
        assert drivers[0].avg_cost_per_request == pytest.approx(0.075)

    def test_multiple_models_sorted_by_cost(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("deepseek/deepseek-chat", 0.01, 500, 200)
        fc.track_request("gpt-4o", 0.50, 10000, 5000)
        fc.track_request("gpt-4o-mini", 0.02, 800, 300)
        drivers = fc._get_model_drivers()
        assert len(drivers) == 3
        assert drivers[0].model_id == "gpt-4o"
        assert drivers[1].model_id == "gpt-4o-mini"
        assert drivers[2].model_id == "deepseek/deepseek-chat"

    def test_display_name_with_prefix(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("deepseek/deepseek-chat", 0.01, 500, 200)
        drivers = fc._get_model_drivers()
        assert drivers[0].display_name == "deepseek-chat"

    def test_display_name_without_prefix(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 0.01, 500, 200)
        drivers = fc._get_model_drivers()
        assert drivers[0].display_name == "gpt-4o"

    def test_percentage_sums_to_100(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 0.30, 5000, 2000)
        fc.track_request("deepseek/deepseek-chat", 0.10, 3000, 1000)
        fc.track_request("gpt-4o-mini", 0.10, 2000, 800)
        drivers = fc._get_model_drivers()
        total_pct = sum(d.percentage for d in drivers)
        assert total_pct == pytest.approx(100.0, abs=0.1)


class TestCheapestAlternative:
    """Tests for _find_cheapest_alternative."""

    def test_known_alternative_gpt4o(self) -> None:
        alt, savings = CostForecaster._find_cheapest_alternative("gpt-4o", 1.0, 10)
        assert alt == "deepseek/deepseek-chat"
        assert savings == pytest.approx(0.95)

    def test_known_alternative_claude_sonnet(self) -> None:
        alt, savings = CostForecaster._find_cheapest_alternative(
            "claude-sonnet-4-20250514", 1.0, 5,
        )
        assert alt == "deepseek/deepseek-chat"
        assert savings == pytest.approx(0.95)

    def test_known_alternative_claude_opus(self) -> None:
        alt, savings = CostForecaster._find_cheapest_alternative(
            "claude-opus-4-20250514", 1.0, 5,
        )
        assert alt == "claude-sonnet-4-20250514"
        assert savings == pytest.approx(0.50)

    def test_known_alternative_gemini_pro(self) -> None:
        alt, savings = CostForecaster._find_cheapest_alternative(
            "gemini/gemini-2.5-pro", 1.0, 5,
        )
        assert alt == "gemini/gemini-2.0-flash"
        assert savings == pytest.approx(0.90)

    def test_known_alternative_o3(self) -> None:
        alt, savings = CostForecaster._find_cheapest_alternative("o3", 1.0, 5)
        assert alt == "o4-mini"
        assert savings == pytest.approx(0.89)

    def test_no_alternative_for_cheap_model(self) -> None:
        alt, savings = CostForecaster._find_cheapest_alternative(
            "deepseek/deepseek-chat", 0.50, 10,
        )
        assert alt is None
        assert savings == 0.0

    def test_no_alternative_for_unknown_model(self) -> None:
        alt, savings = CostForecaster._find_cheapest_alternative(
            "some-unknown-model", 1.0, 5,
        )
        assert alt is None
        assert savings == 0.0

    def test_zero_cost_no_savings(self) -> None:
        _alt, savings = CostForecaster._find_cheapest_alternative("gpt-4o", 0.0, 10)
        # With zero cost, savings would be zero so no suggestion
        assert savings == 0.0

    def test_alternative_map_and_ratios_consistent(self) -> None:
        """Every target in ALTERNATIVE_MAP should have a COST_RATIOS entry."""
        for _source, target in ALTERNATIVE_MAP.items():
            assert target in COST_RATIOS, f"{target} missing from COST_RATIOS"

    def test_drivers_include_alternatives(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 1.0, 20000, 10000)
        drivers = fc._get_model_drivers()
        assert len(drivers) == 1
        assert drivers[0].cheapest_alternative == "deepseek/deepseek-chat"
        assert drivers[0].potential_savings > 0


class TestFeatureTracking:
    """Tests for feature cost tracking."""

    def test_start_feature(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        feature = fc.start_feature("auth-refactor")
        assert feature.name == "auth-refactor"
        assert feature.started_at is not None
        assert feature.completed_at is None
        assert feature.total_cost == 0.0
        assert feature.request_count == 0
        assert feature.models_used == []

    def test_start_feature_empty_name_raises(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        with pytest.raises(ValueError, match="empty"):
            fc.start_feature("")

    def test_end_feature(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.start_feature("fix-login")
        result = fc.end_feature("fix-login")
        assert result is not None
        assert result.completed_at is not None

    def test_end_feature_not_found(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        result = fc.end_feature("nonexistent")
        assert result is None

    def test_track_feature_cost(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.start_feature("router-update")
        fc.track_feature_cost("router-update", 0.05, "gpt-4o")
        fc.track_feature_cost("router-update", 0.01, "deepseek/deepseek-chat")
        fc.track_feature_cost("router-update", 0.03, "gpt-4o")

        feature = fc._feature_costs["router-update"]
        assert feature.total_cost == pytest.approx(0.09)
        assert feature.request_count == 3
        assert feature.models_used == ["gpt-4o", "deepseek/deepseek-chat"]

    def test_track_feature_cost_unknown_feature(self, tmp_path: Path) -> None:
        """Tracking cost for a non-existent feature is a no-op."""
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_feature_cost("ghost-feature", 0.05, "gpt-4o")
        assert "ghost-feature" not in fc._feature_costs

    def test_get_feature_costs(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.start_feature("feat-a")
        fc.start_feature("feat-b")
        costs = fc.get_feature_costs()
        assert len(costs) == 2
        names = {c.name for c in costs}
        assert names == {"feat-a", "feat-b"}

    def test_get_feature_costs_empty(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        assert fc.get_feature_costs() == []

    def test_feature_models_deduplication(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.start_feature("dedup-test")
        fc.track_feature_cost("dedup-test", 0.01, "gpt-4o")
        fc.track_feature_cost("dedup-test", 0.02, "gpt-4o")
        fc.track_feature_cost("dedup-test", 0.03, "gpt-4o")
        feature = fc._feature_costs["dedup-test"]
        assert feature.models_used == ["gpt-4o"]

    def test_overwrite_feature(self, tmp_path: Path) -> None:
        """Starting a feature with the same name overwrites the old one."""
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.start_feature("dup")
        fc.track_feature_cost("dup", 0.50, "gpt-4o")
        fc.start_feature("dup")  # Overwrite
        feature = fc._feature_costs["dup"]
        assert feature.total_cost == 0.0


class TestWeeklyReport:
    """Tests for generate_weekly_report."""

    def test_report_file_created(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports"
        fc = _make_forecaster(monthly_cost=5.0, reports_dir=reports_dir)
        fc.track_request("gpt-4o", 0.10, 2000, 1000)
        path = fc.generate_weekly_report()
        assert path.exists()
        assert path.suffix == ".md"
        assert path.parent == reports_dir

    def test_report_filename_format(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "reports")
        path = fc.generate_weekly_report()
        now = datetime.now(UTC)
        year = now.year
        week = now.isocalendar()[1]
        expected_name = f"{year}-W{week:02d}.md"
        assert path.name == expected_name

    def test_report_contains_summary(self, tmp_path: Path) -> None:
        fc = _make_forecaster(monthly_cost=12.50, reports_dir=tmp_path / "reports")
        path = fc.generate_weekly_report()
        content = path.read_text()
        assert "# Prism Weekly Cost Report" in content
        assert "## Summary" in content
        assert "$12.5000" in content
        assert "## Velocity" in content

    def test_report_contains_budget_section(self, tmp_path: Path) -> None:
        fc = _make_forecaster(
            monthly_cost=5.0, monthly_limit=50.0, reports_dir=tmp_path / "reports",
        )
        path = fc.generate_weekly_report()
        content = path.read_text()
        assert "## Budget" in content
        assert "$50.00" in content

    def test_report_no_budget_section_when_unset(self, tmp_path: Path) -> None:
        fc = _make_forecaster(monthly_cost=5.0, reports_dir=tmp_path / "reports")
        path = fc.generate_weekly_report()
        content = path.read_text()
        assert "## Budget" not in content

    def test_report_contains_model_drivers(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "reports")
        fc.track_request("gpt-4o", 0.50, 10000, 5000)
        fc.track_request("deepseek/deepseek-chat", 0.01, 500, 200)
        path = fc.generate_weekly_report()
        content = path.read_text()
        assert "## Model Cost Drivers" in content
        assert "gpt-4o" in content
        assert "deepseek-chat" in content

    def test_report_contains_feature_costs(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "reports")
        fc.start_feature("auth-module")
        fc.track_feature_cost("auth-module", 0.25, "gpt-4o")
        path = fc.generate_weekly_report()
        content = path.read_text()
        assert "## Feature Costs" in content
        assert "auth-module" in content
        assert "$0.2500" in content

    def test_report_no_feature_section_when_empty(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "reports")
        path = fc.generate_weekly_report()
        content = path.read_text()
        assert "## Feature Costs" not in content

    def test_report_overwrite_same_week(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports"
        fc = _make_forecaster(monthly_cost=1.0, reports_dir=reports_dir)
        path1 = fc.generate_weekly_report()
        content1 = path1.read_text()

        fc.track_request("gpt-4o", 0.50, 10000, 5000)
        path2 = fc.generate_weekly_report()
        content2 = path2.read_text()

        assert path1 == path2  # Same filename
        assert content2 != content1  # Updated content

    def test_report_generation_date(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "reports")
        path = fc.generate_weekly_report()
        content = path.read_text()
        assert "Generated:" in content
        # Should contain a date pattern
        assert re.search(r"\d{4}-\d{2}-\d{2}", content)


# ==================================================================
# Edge Cases
# ==================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_zero_elapsed_time(self, tmp_path: Path) -> None:
        """Velocity calculation handles near-zero elapsed time."""
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc._session_start = datetime.now(UTC)
        fc.track_request("gpt-4o", 1.0, 5000, 2000)
        vel = fc.get_velocity()
        # Should not be infinity or NaN
        assert vel.cost_per_hour >= 0
        assert vel.cost_per_hour < float("inf")

    def test_no_tracker_methods(self, tmp_path: Path) -> None:
        """Forecast works even if tracker lacks get_monthly_cost."""
        settings = _make_settings()
        tracker = MagicMock(spec=[])  # No attributes at all
        fc = CostForecaster(
            cost_tracker=tracker, settings=settings, reports_dir=tmp_path / "r",
        )
        result = fc.forecast()
        assert result.current_monthly_cost == 0.0
        assert result.projected_monthly_cost >= 0

    def test_no_settings_get(self, tmp_path: Path) -> None:
        """Forecast works if settings lacks .get method."""
        tracker = _make_tracker(monthly_cost=5.0)
        settings = MagicMock(spec=[])  # No .get
        fc = CostForecaster(
            cost_tracker=tracker, settings=settings, reports_dir=tmp_path / "r",
        )
        result = fc.forecast()
        assert result.budget_limit is None
        assert result.alert_level == "ok"

    def test_budget_limit_zero(self, tmp_path: Path) -> None:
        """Zero budget limit means no budget enforcement."""
        fc = _make_forecaster(monthly_cost=5.0, monthly_limit=0.0, reports_dir=tmp_path / "r")
        result = fc.forecast()
        # budget_limit=0.0 -> skip budget checks
        assert result.alert_level == "ok"

    def test_very_large_cost(self, tmp_path: Path) -> None:
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 999999.99, 100_000_000, 50_000_000)
        vel = fc.get_velocity()
        assert vel.total_cost == pytest.approx(999999.99)

    def test_forecast_daily_average_first_day(self, tmp_path: Path) -> None:
        """On day 1 of month, daily_average = monthly_cost."""
        fc = _make_forecaster(monthly_cost=3.0, reports_dir=tmp_path / "r")
        result = fc.forecast()
        day = datetime.now(UTC).day
        expected_avg = 3.0 / max(day, 1)
        assert result.daily_average == pytest.approx(expected_avg)

    def test_multiple_features_independent(self, tmp_path: Path) -> None:
        """Feature costs don't leak between features."""
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.start_feature("feat-a")
        fc.start_feature("feat-b")
        fc.track_feature_cost("feat-a", 1.00, "gpt-4o")
        fc.track_feature_cost("feat-b", 0.10, "deepseek/deepseek-chat")
        assert fc._feature_costs["feat-a"].total_cost == pytest.approx(1.00)
        assert fc._feature_costs["feat-b"].total_cost == pytest.approx(0.10)

    def test_budget_limit_non_numeric_in_settings(self, tmp_path: Path) -> None:
        """Non-numeric budget limit gracefully returns None."""
        tracker = _make_tracker(5.0)
        settings = MagicMock()
        settings.get = MagicMock(return_value="not-a-number")
        fc = CostForecaster(
            cost_tracker=tracker, settings=settings, reports_dir=tmp_path / "r",
        )
        result = fc.forecast()
        assert result.budget_limit is None
        assert result.alert_level == "ok"

    def test_negative_cost_handled(self, tmp_path: Path) -> None:
        """Negative cost values don't crash."""
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", -0.01, 100, 50)
        vel = fc.get_velocity()
        assert vel.total_cost == pytest.approx(-0.01)

    def test_session_costs_are_separate_instances(self, tmp_path: Path) -> None:
        """Each tracked request is independent."""
        fc = _make_forecaster(reports_dir=tmp_path / "r")
        fc.track_request("gpt-4o", 0.01, 100, 50)
        fc.track_request("gpt-4o", 0.02, 200, 100)
        assert fc._session_costs[0]["cost_usd"] != fc._session_costs[1]["cost_usd"]

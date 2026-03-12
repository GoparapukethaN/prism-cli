"""Budget enforcement end-to-end tests.

Tests that budget limits are properly enforced across the full pipeline
with database-backed cost tracking. All tests run offline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from prism.config.schema import BudgetConfig, PrismConfig, RoutingConfig
from prism.config.settings import Settings
from prism.cost.tracker import BudgetAction, CostTracker
from prism.db.database import Database
from prism.db.queries import create_session

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_settings(
    tmp_path: Path,
    daily_limit: float | None = None,
    monthly_limit: float | None = None,
) -> Settings:
    """Create a Settings instance with specific budget limits."""
    config = PrismConfig(
        routing=RoutingConfig(),
        budget=BudgetConfig(
            daily_limit=daily_limit,
            monthly_limit=monthly_limit,
            warn_at_percent=80.0,
        ),
        prism_home=tmp_path / ".prism",
    )
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


def _insert_cost(
    db: Database,
    model: str,
    cost: float,
    session_id: str,
    created_at: str | None = None,
) -> None:
    """Insert a cost entry directly into the DB.

    Uses the pydantic CostEntry from db.models which matches the
    save_cost_entry query exactly.
    """
    from prism.cost.pricing import get_provider_for_model
    from prism.db import models as dbm
    from prism.db.queries import save_cost_entry, update_session

    entry = dbm.CostEntry(
        id=str(uuid4()),
        created_at=created_at or datetime.now(UTC).isoformat(),
        session_id=session_id,
        model_id=model,
        provider=get_provider_for_model(model),
        input_tokens=500,
        output_tokens=200,
        cached_tokens=0,
        cost_usd=cost,
        complexity_tier=dbm.ComplexityTier.MEDIUM,
    )
    save_cost_entry(db, entry)
    update_session(db, session_id, cost_delta=cost, request_delta=1)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestDailyLimit:
    """Daily budget enforcement."""

    def test_daily_limit_enforced(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path, daily_limit=0.01)
        db = Database(":memory:")
        db.initialize()
        tracker = CostTracker(db=db, settings=settings)

        session_id = "daily-test"
        create_session(db, session_id, "/tmp")

        # Insert costs that exceed the daily limit
        _insert_cost(db, "gpt-4o-mini", 0.005, session_id)
        _insert_cost(db, "gpt-4o-mini", 0.006, session_id)

        action = tracker.check_budget(0.01)
        # Should be WARN or BLOCK once we exceed the limit
        assert action in (BudgetAction.WARN, BudgetAction.BLOCK)


class TestMonthlyLimit:
    """Monthly budget enforcement."""

    def test_monthly_limit_enforced(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path, monthly_limit=0.01)
        db = Database(":memory:")
        db.initialize()
        tracker = CostTracker(db=db, settings=settings)

        session_id = "monthly-test"
        create_session(db, session_id, "/tmp")

        _insert_cost(db, "gpt-4o-mini", 0.005, session_id)
        _insert_cost(db, "gpt-4o-mini", 0.006, session_id)

        remaining = tracker.get_budget_remaining()
        assert remaining is not None
        assert remaining <= 0.001


class TestNoLimit:
    """No limit should allow all requests."""

    def test_no_limit_allows_all(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path, daily_limit=None, monthly_limit=None)
        db = Database(":memory:")
        db.initialize()
        tracker = CostTracker(db=db, settings=settings)

        session_id = "no-limit"
        create_session(db, session_id, "/tmp")

        _insert_cost(db, "gpt-4o", 0.05, session_id)
        _insert_cost(db, "gpt-4o", 0.05, session_id)

        remaining = tracker.get_budget_remaining()
        assert remaining is None  # No limits set

        action = tracker.check_budget(100.0)
        assert action == BudgetAction.PROCEED


class TestBudgetWarning:
    """Warnings should appear near the limit."""

    def test_budget_warning_near_limit(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path, daily_limit=0.01)
        db = Database(":memory:")
        db.initialize()
        tracker = CostTracker(db=db, settings=settings)

        session_id = "warn-test"
        create_session(db, session_id, "/tmp")

        # Insert a cost that puts us over 80% of the limit
        _insert_cost(db, "gpt-4o-mini", 0.009, session_id)

        action = tracker.check_budget(0.0001)
        # At 90% of budget, should warn
        assert action in (BudgetAction.WARN, BudgetAction.BLOCK)


class TestCostAccumulation:
    """Costs should accumulate across requests."""

    def test_cost_accumulates_across_requests(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path, daily_limit=10.0)
        db = Database(":memory:")
        db.initialize()
        tracker = CostTracker(db=db, settings=settings)

        session_id = "accumulate"
        create_session(db, session_id, "/tmp")

        _insert_cost(db, "gpt-4o-mini", 0.003, session_id)
        _insert_cost(db, "gpt-4o-mini", 0.004, session_id)

        session_total = tracker.get_session_cost(session_id)
        assert abs(session_total - 0.007) < 1e-9


class TestBudgetResets:
    """Budget tracking resets on schedule."""

    def test_budget_resets_daily(self, tmp_path: Path) -> None:
        """get_daily_cost should only count today's costs."""
        settings = _make_settings(tmp_path, daily_limit=10.0)
        db = Database(":memory:")
        db.initialize()
        tracker = CostTracker(db=db, settings=settings)

        session_id = "reset-test"
        create_session(db, session_id, "/tmp")

        # Insert a cost entry with a past date
        _insert_cost(
            db, "gpt-4o-mini", 5.0, session_id,
            created_at="2020-01-01T00:00:00Z",
        )

        # Today's cost should not include the past entry
        daily_cost = tracker.get_daily_cost()
        assert daily_cost < 5.0  # Past entries are on a different date


class TestBudgetRemaining:
    """Budget remaining should be accurate."""

    def test_budget_remaining_accurate(self, tmp_path: Path) -> None:
        daily_limit = 1.0
        settings = _make_settings(tmp_path, daily_limit=daily_limit)
        db = Database(":memory:")
        db.initialize()
        tracker = CostTracker(db=db, settings=settings)

        session_id = "remaining"
        create_session(db, session_id, "/tmp")

        cost = 0.05
        _insert_cost(db, "gpt-4o-mini", cost, session_id)

        remaining = tracker.get_budget_remaining()
        assert remaining is not None
        assert abs(remaining - (daily_limit - cost)) < 1e-6


class TestCheapestModelSuggestion:
    """When over budget, the selector should prefer free models or raise."""

    def test_over_budget_selects_free_or_raises(
        self,
        tmp_path: Path,
    ) -> None:
        from unittest.mock import MagicMock

        from prism.exceptions import BudgetExceededError
        from prism.providers.base import ComplexityTier
        from prism.providers.registry import ProviderRegistry
        from prism.router.selector import ModelSelector

        settings = _make_settings(tmp_path, daily_limit=0.0001)
        db = Database(":memory:")
        db.initialize()

        session_id = "over-budget"
        create_session(db, session_id, "/tmp")

        # Exhaust the budget
        _insert_cost(db, "gpt-4o", 0.01, session_id)

        tracker = CostTracker(db=db, settings=settings)

        auth = MagicMock()
        auth.get_key = MagicMock(return_value="fake-key")
        registry = ProviderRegistry(settings=settings, auth_manager=auth)
        selector = ModelSelector(
            settings=settings, registry=registry, cost_tracker=tracker
        )

        try:
            result = selector.select(
                tier=ComplexityTier.COMPLEX,
                prompt="design a system",
            )
            # If it selects a model, it must be a free one ($0.00 estimated cost)
            assert result.estimated_cost == 0.0
        except BudgetExceededError as exc:
            assert exc.budget_remaining >= 0.0

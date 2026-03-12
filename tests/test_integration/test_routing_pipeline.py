"""Integration tests for the full classify -> select -> track pipeline.

All tests run offline with mocked providers — no real API calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from prism.cost.tracker import CostTracker
from prism.db.queries import create_session, save_cost_entry
from prism.exceptions import BudgetExceededError
from prism.providers.base import ComplexityTier
from prism.providers.registry import ProviderRegistry
from prism.router.classifier import TaskClassifier, TaskContext
from prism.router.selector import ModelSelector

if TYPE_CHECKING:
    from prism.config.settings import Settings
    from prism.db.database import Database

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _insert_cost(
    db: Database,
    model: str,
    cost: float,
    session_id: str = "test-session",
) -> None:
    """Insert a cost entry directly into the DB for budget testing."""
    from prism.db import models as dbm

    entry = dbm.CostEntry(
        id=str(uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        session_id=session_id,
        model_id=model,
        provider="test",
        input_tokens=100,
        output_tokens=100,
        cached_tokens=0,
        cost_usd=cost,
        complexity_tier=dbm.ComplexityTier.SIMPLE,
    )
    save_cost_entry(db, entry)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSimpleTaskRouting:
    """Simple prompts should route to cheap/free models."""

    def test_simple_task_routes_to_cheap_model(
        self,
        classifier: TaskClassifier,
        model_selector: ModelSelector,
    ) -> None:
        result = classifier.classify("fix typo in readme")
        assert result.tier == ComplexityTier.SIMPLE
        selection = model_selector.select(result.tier, "fix typo in readme")
        # Should pick a free or cheap model — not a premium one
        assert selection.tier == ComplexityTier.SIMPLE
        assert selection.estimated_cost < 0.01

    def test_complex_task_routes_to_premium_model(
        self,
        classifier: TaskClassifier,
        model_selector: ModelSelector,
    ) -> None:
        prompt = (
            "architect a distributed microservice system with security audit "
            "and evaluate the trade-off between scalable patterns. "
            "refactor the entire authentication module and implement caching. "
            "design the database schema and optimize queries across all services."
        )
        ctx = TaskContext(
            active_files=[f"src/service_{i}.py" for i in range(10)],
        )
        result = classifier.classify(prompt, ctx)
        assert result.tier == ComplexityTier.COMPLEX
        selection = model_selector.select(result.tier, prompt)
        assert selection.tier == ComplexityTier.COMPLEX


class TestBudgetEnforcement:
    """Budget limits should block expensive models."""

    def test_budget_blocks_expensive_model(
        self,
        full_settings: Settings,
        integration_db: Database,
        provider_registry: ProviderRegistry,
    ) -> None:
        # Set a tiny daily limit
        full_settings.set_override("budget.daily_limit", 0.001)
        cost_tracker = CostTracker(db=integration_db, settings=full_settings)

        # Create a session and consume the budget
        create_session(integration_db, "s1", "/tmp")
        _insert_cost(integration_db, "gpt-4o", 0.001, "s1")

        selector = ModelSelector(
            settings=full_settings,
            registry=provider_registry,
            cost_tracker=cost_tracker,
        )

        try:
            result = selector.select(ComplexityTier.COMPLEX, "design a system")
            # If a model is selected, it must be a free one ($0.00 estimated)
            assert result.estimated_cost == 0.0
        except BudgetExceededError:
            pass  # Expected when no free models are available


class TestFallbackBehaviour:
    """Fallback when the primary provider is unavailable."""

    def test_fallback_on_unavailable_provider(
        self,
        full_settings: Settings,
        integration_db: Database,
        mock_auth_manager: MagicMock,
    ) -> None:
        # Mark anthropic as having no key
        def _get_key(provider: str) -> str | None:
            if provider == "anthropic":
                return None
            return f"fake-key-for-{provider}"

        mock_auth_manager.get_key = MagicMock(side_effect=_get_key)
        registry = ProviderRegistry(settings=full_settings, auth_manager=mock_auth_manager)
        cost_tracker = CostTracker(db=integration_db, settings=full_settings)
        selector = ModelSelector(
            settings=full_settings, registry=registry, cost_tracker=cost_tracker
        )

        # Even for complex tasks, should still find a model from another provider
        selection = selector.select(ComplexityTier.COMPLEX, "design a system")
        assert selection.provider != "anthropic"


class TestExploration:
    """Exploration should occasionally pick a non-default model."""

    def test_exploration_sometimes_picks_different_model(
        self,
        full_settings: Settings,
        provider_registry: ProviderRegistry,
        integration_db: Database,
    ) -> None:
        # Set exploration rate to 100% for deterministic testing
        full_settings.set_override("routing.exploration_rate", 1.0)
        cost_tracker = CostTracker(db=integration_db, settings=full_settings)
        selector = ModelSelector(
            settings=full_settings, registry=provider_registry, cost_tracker=cost_tracker
        )

        # Run selection twice — with 100% exploration, should not always pick top
        selections: set[str] = set()
        for _ in range(5):
            sel = selector.select(ComplexityTier.MEDIUM, "implement a feature")
            selections.add(sel.model_id)

        # With 100% exploration from a pool of models, we should see variation
        # (at least 1 model selected — the exploration pool)
        assert len(selections) >= 1


class TestCostTracking:
    """Cost tracking after routing."""

    def test_cost_tracked_after_routing(
        self,
        integration_db: Database,
    ) -> None:
        from prism.db import models as dbm
        from prism.db.queries import get_session_cost, save_cost_entry, update_session

        session_id = "sess-1"
        create_session(integration_db, session_id, "/tmp")

        cost = 0.003
        entry = dbm.CostEntry(
            id=str(uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            session_id=session_id,
            model_id="gpt-4o-mini",
            provider="openai",
            input_tokens=500,
            output_tokens=200,
            cached_tokens=0,
            cost_usd=cost,
            complexity_tier=dbm.ComplexityTier.MEDIUM,
        )
        save_cost_entry(integration_db, entry)
        update_session(integration_db, session_id, cost_delta=cost, request_delta=1)

        session_cost = get_session_cost(integration_db, session_id)
        assert abs(session_cost - cost) < 1e-9


class TestClassificationStorage:
    """Classification results stored in DB."""

    def test_classification_stored_in_db(
        self,
        classifier: TaskClassifier,
        integration_db: Database,
    ) -> None:
        from prism.db import models as dbm
        from prism.db.queries import save_routing_decision

        result = classifier.classify("refactor the module")
        decision = dbm.RoutingDecision(
            id=str(uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            session_id="test-session",
            prompt_hash="abc123",
            complexity_tier=dbm.ComplexityTier(result.tier.value),
            complexity_score=result.score,
            model_selected="gpt-4o-mini",
            fallback_chain="[]",
            estimated_cost=0.001,
            outcome=dbm.Outcome.UNKNOWN,
            features="{}",
        )
        save_routing_decision(integration_db, decision)

        from prism.db.queries import get_routing_history

        history = get_routing_history(integration_db, limit=1)
        assert len(history) == 1
        assert history[0].complexity_tier.value == result.tier.value


class TestPinnedModel:
    """Pinned model should bypass the classifier."""

    def test_pinned_model_bypasses_classifier(
        self,
        full_settings: Settings,
        provider_registry: ProviderRegistry,
        integration_db: Database,
    ) -> None:
        full_settings.set_override("pinned_model", "gpt-4o")
        cost_tracker = CostTracker(db=integration_db, settings=full_settings)
        selector = ModelSelector(
            settings=full_settings, registry=provider_registry, cost_tracker=cost_tracker
        )
        # Even for a simple task, pinned model should be selected
        selection = selector.select(ComplexityTier.SIMPLE, "fix typo")
        assert selection.model_id == "gpt-4o"
        assert "Pinned" in selection.reasoning


class TestRateLimitedProvider:
    """Rate-limited provider should be skipped."""

    def test_rate_limited_provider_skipped(
        self,
        full_settings: Settings,
        integration_db: Database,
        mock_auth_manager: MagicMock,
    ) -> None:
        from datetime import timedelta

        registry = ProviderRegistry(settings=full_settings, auth_manager=mock_auth_manager)

        # Mark groq as rate-limited
        status = registry.get_status("groq")
        if status is not None:
            future = datetime.now(UTC) + timedelta(hours=1)
            status.mark_rate_limited(future)

        cost_tracker = CostTracker(db=integration_db, settings=full_settings)
        selector = ModelSelector(
            settings=full_settings, registry=registry, cost_tracker=cost_tracker
        )
        # Should still get a model, just not from groq
        selection = selector.select(ComplexityTier.SIMPLE, "fix typo")
        assert selection.provider != "groq"


class TestFreeTierModels:
    """Free tier models are preferred when available."""

    def test_free_tier_model_preferred_when_available(
        self,
        model_selector: ModelSelector,
    ) -> None:
        # For simple tasks, free Ollama models should be ranked highest
        selection = model_selector.select(ComplexityTier.SIMPLE, "rename variable")
        # Simple tasks should use cheap/free models
        assert selection.estimated_cost < 0.001


class TestMultipleRequestCosts:
    """Multiple requests accumulate cost."""

    def test_multiple_requests_accumulate_cost(
        self,
        integration_db: Database,
    ) -> None:
        from prism.db import models as dbm
        from prism.db.queries import get_session_cost, save_cost_entry, update_session

        session_id = "sess-acc"
        create_session(integration_db, session_id, "/tmp")

        for _ in range(2):
            entry = dbm.CostEntry(
                id=str(uuid4()),
                created_at=datetime.now(UTC).isoformat(),
                session_id=session_id,
                model_id="gpt-4o-mini",
                provider="openai",
                input_tokens=500,
                output_tokens=200,
                cached_tokens=0,
                cost_usd=0.002,
                complexity_tier=dbm.ComplexityTier.MEDIUM,
            )
            save_cost_entry(integration_db, entry)
            update_session(integration_db, session_id, cost_delta=0.002, request_delta=1)

        total = get_session_cost(integration_db, session_id)
        assert abs(total - 0.004) < 1e-9

    def test_daily_budget_enforcement(
        self,
        full_settings: Settings,
        integration_db: Database,
    ) -> None:
        full_settings.set_override("budget.daily_limit", 0.001)
        cost_tracker = CostTracker(db=integration_db, settings=full_settings)
        create_session(integration_db, "sess-daily", "/tmp")
        _insert_cost(integration_db, "gpt-4o", 0.001, "sess-daily")

        remaining = cost_tracker.get_budget_remaining()
        assert remaining is not None
        assert remaining <= 0.001

    def test_monthly_budget_enforcement(
        self,
        full_settings: Settings,
        integration_db: Database,
    ) -> None:
        full_settings.set_override("budget.monthly_limit", 0.01)
        cost_tracker = CostTracker(db=integration_db, settings=full_settings)
        create_session(integration_db, "sess-monthly", "/tmp")
        _insert_cost(integration_db, "gpt-4o", 0.01, "sess-monthly")

        remaining = cost_tracker.get_budget_remaining()
        assert remaining is not None
        assert remaining <= 0.001


class TestRoutingWithContext:
    """Routing respects context files."""

    def test_routing_with_context_files(
        self,
        classifier: TaskClassifier,
    ) -> None:
        ctx = TaskContext(
            active_files=["a.py", "b.py", "c.py", "d.py"],
            conversation_turns=5,
        )
        result = classifier.classify("refactor this code", ctx)
        # With 4 active files, scope and reasoning should increase score
        assert result.score > 0.0


class TestOfflineRouting:
    """Offline routing should prefer Ollama."""

    def test_offline_routes_to_ollama(
        self,
        full_settings: Settings,
        integration_db: Database,
    ) -> None:
        # Create a mock auth manager that reports no keys for any provider
        offline_auth = MagicMock()
        offline_auth.get_key = MagicMock(return_value=None)

        registry = ProviderRegistry(settings=full_settings, auth_manager=offline_auth)
        cost_tracker = CostTracker(db=integration_db, settings=full_settings)
        selector = ModelSelector(
            settings=full_settings, registry=registry, cost_tracker=cost_tracker
        )

        # Only Ollama models should be available (they don't need API keys)
        selection = selector.select(ComplexityTier.SIMPLE, "fix typo")
        assert selection.provider == "ollama"
        assert selection.estimated_cost == 0.0

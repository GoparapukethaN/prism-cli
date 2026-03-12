"""Shared fixtures for orchestrator tests.

All fixtures use MockLiteLLM — absolutely no real API calls.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from prism.config.schema import BudgetConfig, PrismConfig
from prism.config.settings import Settings
from prism.llm.completion import CompletionEngine
from prism.llm.mock import MockLiteLLM, MockResponse
from prism.orchestrator.swarm import (
    CrossReviewer,
    ModelPool,
    PlanReviewer,
    ResearchSwarm,
    SwarmConfig,
    SwarmOrchestrator,
    TaskDecomposer,
)
from prism.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Settings / config
# ------------------------------------------------------------------


@pytest.fixture()
def orch_settings(tmp_path: Path) -> Settings:
    """Settings with a reasonable budget for orchestrator testing."""
    config = PrismConfig(
        prism_home=tmp_path / ".prism",
        budget=BudgetConfig(daily_limit=50.0, monthly_limit=500.0),
    )
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


# ------------------------------------------------------------------
# Auth manager (mock)
# ------------------------------------------------------------------


@pytest.fixture()
def mock_auth() -> MagicMock:
    """Mock AuthManager that returns test keys for all providers."""
    mgr = MagicMock()
    mgr.get_key.return_value = "test-key-1234"
    return mgr


# ------------------------------------------------------------------
# Cost tracker (mock)
# ------------------------------------------------------------------


@pytest.fixture()
def mock_cost_tracker() -> MagicMock:
    """Mock CostTracker that always allows requests."""
    tracker = MagicMock()
    tracker.check_budget.return_value = "proceed"
    tracker.get_budget_remaining.return_value = None
    tracker.track.return_value = MagicMock(cost_usd=0.001)
    return tracker


# ------------------------------------------------------------------
# Provider registry (real, with mock auth)
# ------------------------------------------------------------------


@pytest.fixture()
def mock_registry(mock_auth: MagicMock, orch_settings: Settings) -> ProviderRegistry:
    """Real ProviderRegistry using mock auth."""
    return ProviderRegistry(settings=orch_settings, auth_manager=mock_auth)


# ------------------------------------------------------------------
# MockLiteLLM with common response presets
# ------------------------------------------------------------------


def _make_decompose_response() -> str:
    """Return a valid JSON decomposition response."""
    tasks = [
        {
            "description": "Read and analyse the existing codebase",
            "complexity": "simple",
            "dependencies": [],
            "files_changed": [],
        },
        {
            "description": "Implement the new authentication module",
            "complexity": "complex",
            "dependencies": [0],
            "files_changed": ["src/auth.py"],
        },
        {
            "description": "Write unit tests for the auth module",
            "complexity": "medium",
            "dependencies": [1],
            "files_changed": ["tests/test_auth.py"],
        },
    ]
    return json.dumps(tasks)


def _make_review_response(*, approved: bool = True) -> str:
    """Return a valid JSON cross-review response."""
    return json.dumps({
        "severity": "info" if approved else "error",
        "approved": approved,
        "comments": "Looks good overall." if approved else "Missing error handling.",
    })


@pytest.fixture()
def mock_litellm() -> MockLiteLLM:
    """MockLiteLLM pre-loaded with orchestrator-friendly responses."""
    mock = MockLiteLLM()

    # Default response works for most phases
    mock.set_default_response(
        MockResponse(
            content="Mock execution output for the task.",
            input_tokens=100,
            output_tokens=50,
        ),
    )
    return mock


@pytest.fixture()
def mock_litellm_decompose() -> MockLiteLLM:
    """MockLiteLLM that returns a decomposition JSON for planning models."""
    mock = MockLiteLLM()
    decompose_content = _make_decompose_response()
    # Set for all known planning models
    for model in (
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "gpt-4o",
        "o3",
        "gemini/gemini-2.5-pro",
    ):
        mock.set_response(
            model,
            MockResponse(content=decompose_content, input_tokens=200, output_tokens=300),
        )
    # Default for anything else
    mock.set_default_response(
        MockResponse(content="Mock response.", input_tokens=50, output_tokens=30),
    )
    return mock


@pytest.fixture()
def mock_litellm_review() -> MockLiteLLM:
    """MockLiteLLM that returns review JSON for review models."""
    mock = MockLiteLLM()
    review_content = _make_review_response(approved=True)
    mock.set_default_response(
        MockResponse(content=review_content, input_tokens=100, output_tokens=80),
    )
    return mock


@pytest.fixture()
def mock_litellm_review_error() -> MockLiteLLM:
    """MockLiteLLM that returns error-level review JSON."""
    mock = MockLiteLLM()
    review_content = _make_review_response(approved=False)
    mock.set_default_response(
        MockResponse(content=review_content, input_tokens=100, output_tokens=80),
    )
    return mock


# ------------------------------------------------------------------
# CompletionEngine
# ------------------------------------------------------------------


@pytest.fixture()
def completion_engine(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm: MockLiteLLM,
) -> CompletionEngine:
    """CompletionEngine wired to mocks — no real API calls."""
    return CompletionEngine(
        settings=orch_settings,
        cost_tracker=mock_cost_tracker,
        auth_manager=mock_auth,
        provider_registry=mock_registry,
        litellm_backend=mock_litellm,
    )


def _make_engine(
    settings: Settings,
    cost_tracker: MagicMock,
    auth: MagicMock,
    registry: ProviderRegistry,
    litellm: MockLiteLLM,
) -> CompletionEngine:
    """Helper to create a CompletionEngine from components."""
    return CompletionEngine(
        settings=settings,
        cost_tracker=cost_tracker,
        auth_manager=auth,
        provider_registry=registry,
        litellm_backend=litellm,
    )


# ------------------------------------------------------------------
# Orchestrator sub-components
# ------------------------------------------------------------------


@pytest.fixture()
def model_pool(mock_registry: ProviderRegistry) -> ModelPool:
    """A real ModelPool using the mock registry."""
    return ModelPool(mock_registry)


@pytest.fixture()
def task_decomposer(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_decompose: MockLiteLLM,
) -> TaskDecomposer:
    """TaskDecomposer with decomposition-ready mock."""
    engine = _make_engine(
        orch_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_decompose,
    )
    pool = ModelPool(mock_registry)
    return TaskDecomposer(engine, pool)


@pytest.fixture()
def research_swarm(
    completion_engine: CompletionEngine,
    model_pool: ModelPool,
) -> ResearchSwarm:
    """ResearchSwarm wired to mocks."""
    return ResearchSwarm(completion_engine, model_pool)


@pytest.fixture()
def plan_reviewer(
    completion_engine: CompletionEngine,
    model_pool: ModelPool,
) -> PlanReviewer:
    """PlanReviewer wired to mocks."""
    return PlanReviewer(completion_engine, model_pool)


@pytest.fixture()
def cross_reviewer_approved(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_review: MockLiteLLM,
) -> CrossReviewer:
    """CrossReviewer that approves everything."""
    engine = _make_engine(
        orch_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_review,
    )
    pool = ModelPool(mock_registry)
    return CrossReviewer(engine, pool)


@pytest.fixture()
def cross_reviewer_error(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_review_error: MockLiteLLM,
) -> CrossReviewer:
    """CrossReviewer that returns errors."""
    engine = _make_engine(
        orch_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_review_error,
    )
    pool = ModelPool(mock_registry)
    return CrossReviewer(engine, pool)


@pytest.fixture()
def orchestrator(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_decompose: MockLiteLLM,
) -> SwarmOrchestrator:
    """Full SwarmOrchestrator wired to mocks (advanced features disabled)."""
    engine = _make_engine(
        orch_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_decompose,
    )
    # Disable advanced features for backwards-compatible tests
    config = SwarmConfig(use_debate=False, use_moa=False, use_cascade=False, use_tools=False)
    return SwarmOrchestrator(engine, mock_registry, config=config)


@pytest.fixture()
def orchestrator_with_debate(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_decompose: MockLiteLLM,
) -> SwarmOrchestrator:
    """SwarmOrchestrator with debate enabled."""
    engine = _make_engine(
        orch_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_decompose,
    )
    config = SwarmConfig(use_debate=True, use_moa=False, use_cascade=False, use_tools=False)
    return SwarmOrchestrator(engine, mock_registry, config=config)


@pytest.fixture()
def orchestrator_with_cascade(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_decompose: MockLiteLLM,
) -> SwarmOrchestrator:
    """SwarmOrchestrator with cascade enabled."""
    engine = _make_engine(
        orch_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_decompose,
    )
    config = SwarmConfig(use_debate=False, use_moa=False, use_cascade=True, use_tools=False)
    return SwarmOrchestrator(engine, mock_registry, config=config)


@pytest.fixture()
def orchestrator_with_moa(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_decompose: MockLiteLLM,
) -> SwarmOrchestrator:
    """SwarmOrchestrator with MoA enabled for complex tasks."""
    engine = _make_engine(
        orch_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_decompose,
    )
    config = SwarmConfig(
        use_debate=False, use_moa=True, use_cascade=False, use_tools=False,
        moa_complexity_threshold="complex",
    )
    return SwarmOrchestrator(engine, mock_registry, config=config)


@pytest.fixture()
def orchestrator_full(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_decompose: MockLiteLLM,
) -> SwarmOrchestrator:
    """SwarmOrchestrator with ALL advanced features enabled."""
    engine = _make_engine(
        orch_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_decompose,
    )
    config = SwarmConfig(use_debate=True, use_moa=True, use_cascade=True, use_tools=False)
    return SwarmOrchestrator(engine, mock_registry, config=config)


@pytest.fixture()
def orchestrator_with_aei(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_decompose: MockLiteLLM,
    tmp_path: Path,
) -> SwarmOrchestrator:
    """SwarmOrchestrator with AEI error intelligence wired in."""
    from prism.intelligence.aei import AdaptiveExecutionIntelligence

    engine = _make_engine(
        orch_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_decompose,
    )
    config = SwarmConfig(use_debate=False, use_moa=False, use_cascade=False, use_tools=False)
    aei = AdaptiveExecutionIntelligence(db_path=tmp_path / "aei_test.db", repo="test-repo")
    return SwarmOrchestrator(
        engine, mock_registry, config=config, error_intelligence=aei,
    )


@pytest.fixture()
def orchestrator_with_context_manager(
    orch_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm_decompose: MockLiteLLM,
    tmp_path: Path,
) -> SwarmOrchestrator:
    """SwarmOrchestrator with SmartContextBudgetManager wired in."""
    from prism.intelligence.context_budget import SmartContextBudgetManager

    engine = _make_engine(
        orch_settings, mock_cost_tracker, mock_auth, mock_registry, mock_litellm_decompose,
    )
    config = SwarmConfig(use_debate=False, use_moa=False, use_cascade=False, use_tools=False)
    ctx_mgr = SmartContextBudgetManager(project_root=tmp_path)
    return SwarmOrchestrator(
        engine, mock_registry, config=config, context_manager=ctx_mgr,
    )

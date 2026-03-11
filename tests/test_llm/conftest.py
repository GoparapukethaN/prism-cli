"""Shared fixtures for LLM tests.

All fixtures use MockLiteLLM — absolutely no real API calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from prism.config.schema import BudgetConfig, PrismConfig
from prism.config.settings import Settings
from prism.llm.completion import CompletionEngine
from prism.llm.mock import MockLiteLLM, MockResponse
from prism.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Settings / config
# ------------------------------------------------------------------


@pytest.fixture
def llm_settings(tmp_path: Path) -> Settings:
    """Settings with a reasonable daily budget for testing."""
    config = PrismConfig(
        prism_home=tmp_path / ".prism",
        budget=BudgetConfig(daily_limit=10.0, monthly_limit=100.0),
    )
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


@pytest.fixture
def llm_settings_no_budget(tmp_path: Path) -> Settings:
    """Settings with no budget limits."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


# ------------------------------------------------------------------
# Auth manager (mock)
# ------------------------------------------------------------------


@pytest.fixture
def mock_auth() -> MagicMock:
    """Mock AuthManager that returns test keys for all providers."""
    mgr = MagicMock()
    mgr.get_key.return_value = "test-key-1234"
    return mgr


# ------------------------------------------------------------------
# Cost tracker (mock)
# ------------------------------------------------------------------


@pytest.fixture
def mock_cost_tracker() -> MagicMock:
    """Mock CostTracker that always allows requests."""
    tracker = MagicMock()
    tracker.check_budget.return_value = "proceed"
    tracker.get_budget_remaining.return_value = None
    tracker.track.return_value = MagicMock(cost_usd=0.001)
    return tracker


@pytest.fixture
def mock_cost_tracker_blocked() -> MagicMock:
    """Mock CostTracker that blocks all requests."""
    tracker = MagicMock()
    tracker.check_budget.return_value = "block"
    tracker.get_budget_remaining.return_value = 0.001
    tracker.track.return_value = MagicMock(cost_usd=0.0)
    return tracker


# ------------------------------------------------------------------
# Provider registry (mock)
# ------------------------------------------------------------------


@pytest.fixture
def mock_registry(mock_auth: MagicMock, llm_settings: Settings) -> ProviderRegistry:
    """Real ProviderRegistry using mock auth."""
    return ProviderRegistry(settings=llm_settings, auth_manager=mock_auth)


# ------------------------------------------------------------------
# MockLiteLLM
# ------------------------------------------------------------------


@pytest.fixture
def mock_litellm() -> MockLiteLLM:
    """Fresh MockLiteLLM instance."""
    return MockLiteLLM()


@pytest.fixture
def mock_litellm_with_response() -> MockLiteLLM:
    """MockLiteLLM pre-loaded with a GPT-4o response."""
    mock = MockLiteLLM()
    mock.set_response(
        "gpt-4o",
        MockResponse(
            content="Hello from the mock!",
            model="gpt-4o",
            input_tokens=50,
            output_tokens=20,
        ),
    )
    return mock


# ------------------------------------------------------------------
# CompletionEngine
# ------------------------------------------------------------------


@pytest.fixture
def completion_engine(
    llm_settings: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_registry: ProviderRegistry,
    mock_litellm: MockLiteLLM,
) -> CompletionEngine:
    """CompletionEngine wired to all mocks — no real API calls."""
    return CompletionEngine(
        settings=llm_settings,
        cost_tracker=mock_cost_tracker,
        auth_manager=mock_auth,
        provider_registry=mock_registry,
        litellm_backend=mock_litellm,
    )


@pytest.fixture
def completion_engine_no_budget(
    llm_settings_no_budget: Settings,
    mock_cost_tracker: MagicMock,
    mock_auth: MagicMock,
    mock_litellm: MockLiteLLM,
) -> CompletionEngine:
    """CompletionEngine with no budget limits."""
    registry = ProviderRegistry(
        settings=llm_settings_no_budget, auth_manager=mock_auth,
    )
    return CompletionEngine(
        settings=llm_settings_no_budget,
        cost_tracker=mock_cost_tracker,
        auth_manager=mock_auth,
        provider_registry=registry,
        litellm_backend=mock_litellm,
    )


# ------------------------------------------------------------------
# Simple message helpers
# ------------------------------------------------------------------


@pytest.fixture
def simple_messages() -> list[dict[str, str]]:
    """Minimal chat messages."""
    return [{"role": "user", "content": "Say hello."}]


@pytest.fixture
def long_messages() -> list[dict[str, str]]:
    """Messages that exceed a typical context window for testing trimming.

    Each message is ~4000 chars -> ~1600 tokens (char estimate).
    50 messages -> ~80,000 tokens, which exceeds mixtral's 32k and
    will trigger trimming for most models.
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        *[
            {"role": "user", "content": f"Message number {i}. " * 1000}
            for i in range(50)
        ],
    ]

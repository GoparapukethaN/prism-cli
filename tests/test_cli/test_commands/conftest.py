"""Fixtures for slash-command tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, PropertyMock

import pytest

from prism.cli.commands.slash_commands import SlashCommandHandler
from prism.config.schema import PrismConfig
from prism.config.settings import Settings
from prism.context.manager import ContextManager
from prism.context.memory import ProjectMemory
from prism.context.session import SessionManager

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Create test settings."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    s = Settings(config=config, project_root=tmp_path)
    s.ensure_directories()
    return s


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.fixture
def context_manager() -> ContextManager:
    """Create a context manager for tests."""
    return ContextManager(max_tokens=16_000)


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------


@pytest.fixture
def session_manager(tmp_path: Path) -> SessionManager:
    """Create a session manager backed by a temp directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    return SessionManager(sessions_dir)


# ---------------------------------------------------------------------------
# Project memory
# ---------------------------------------------------------------------------


@pytest.fixture
def project_memory(tmp_path: Path) -> ProjectMemory:
    """Create a project memory backed by a temp directory."""
    return ProjectMemory(tmp_path)


# ---------------------------------------------------------------------------
# Mock cost tracker
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cost_tracker() -> MagicMock:
    """Create a mock CostTracker."""
    tracker = MagicMock()
    tracker.get_budget_remaining.return_value = 8.50
    tracker.get_daily_cost.return_value = 1.50
    tracker.get_monthly_cost.return_value = 12.00
    tracker.get_session_cost.return_value = 0.25
    tracker.calculate_savings.return_value = (20.0, 12.0, 8.0)
    tracker.get_cost_summary.return_value = MagicMock(
        total_cost=1.50,
        total_requests=5,
        model_breakdown=[],
        budget_limit=10.0,
        budget_remaining=8.50,
    )
    return tracker


# ---------------------------------------------------------------------------
# Mock provider registry
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider_registry() -> MagicMock:
    """Create a mock ProviderRegistry."""
    registry = MagicMock()

    # all_models property
    type(registry).all_models = PropertyMock(
        return_value={
            "claude-sonnet-4-20250514": MagicMock(display_name="Claude Sonnet"),
            "gpt-4o-mini": MagicMock(display_name="GPT-4o Mini"),
        }
    )

    registry.get_model_info.side_effect = lambda m: (
        MagicMock(display_name=m) if m in ("claude-sonnet-4-20250514", "gpt-4o-mini") else None
    )

    registry.list_providers.return_value = [
        {
            "name": "anthropic",
            "display_name": "Anthropic",
            "configured": True,
            "available": True,
            "model_count": 2,
            "models": ["Claude Sonnet", "Claude Haiku"],
        },
        {
            "name": "openai",
            "display_name": "OpenAI",
            "configured": True,
            "available": True,
            "model_count": 3,
            "models": ["GPT-4o", "GPT-4o Mini", "o1-mini"],
        },
    ]

    return registry


# ---------------------------------------------------------------------------
# Mock git repo
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_git_repo() -> MagicMock:
    """Create a mock GitRepo."""
    repo = MagicMock()
    repo._run.return_value = ""
    repo.status.return_value = MagicMock(is_clean=True)
    repo.current_branch.return_value = "main"
    return repo


# ---------------------------------------------------------------------------
# Full handler
# ---------------------------------------------------------------------------


@pytest.fixture
def handler(
    settings: Settings,
    mock_cost_tracker: MagicMock,
    context_manager: ContextManager,
    session_manager: SessionManager,
    project_memory: ProjectMemory,
    mock_provider_registry: MagicMock,
    mock_git_repo: MagicMock,
) -> SlashCommandHandler:
    """Create a fully-wired SlashCommandHandler for testing."""
    h = SlashCommandHandler(
        settings=settings,
        cost_tracker=mock_cost_tracker,
        context_manager=context_manager,
        session_manager=session_manager,
        project_memory=project_memory,
        provider_registry=mock_provider_registry,
        git_repo=mock_git_repo,
    )
    h.session_id = "test-session-1234"
    return h


@pytest.fixture
def handler_minimal(settings: Settings) -> SlashCommandHandler:
    """Create a handler with only settings (no optional deps)."""
    return SlashCommandHandler(settings=settings)

"""Integration test fixtures — fully wired components with mock externals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from prism.config.schema import BudgetConfig, PrismConfig, RoutingConfig, ToolsConfig
from prism.config.settings import Settings
from prism.cost.tracker import CostTracker
from prism.db.database import Database
from prism.providers.registry import ProviderRegistry
from prism.router.classifier import TaskClassifier
from prism.security.audit import AuditLogger
from prism.security.path_guard import PathGuard

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------


@pytest.fixture()
def full_settings(tmp_path: Path) -> Settings:
    """Fully configured Settings for integration tests."""
    config = PrismConfig(
        routing=RoutingConfig(
            simple_threshold=0.3,
            medium_threshold=0.7,
            exploration_rate=0.0,  # disable exploration for deterministic tests
            quality_weight=0.7,
        ),
        budget=BudgetConfig(
            daily_limit=5.0,
            monthly_limit=50.0,
            warn_at_percent=80.0,
        ),
        tools=ToolsConfig(
            auto_approve=False,
            command_timeout=10,
        ),
        prism_home=tmp_path / ".prism",
    )
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


@pytest.fixture()
def unlimited_settings(tmp_path: Path) -> Settings:
    """Settings without any budget limits."""
    config = PrismConfig(
        routing=RoutingConfig(
            simple_threshold=0.3,
            medium_threshold=0.7,
            exploration_rate=0.0,
            quality_weight=0.7,
        ),
        budget=BudgetConfig(
            daily_limit=None,
            monthly_limit=None,
        ),
        prism_home=tmp_path / ".prism",
    )
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


# ------------------------------------------------------------------
# Database
# ------------------------------------------------------------------


@pytest.fixture()
def integration_db(tmp_path: Path) -> Database:
    """Initialized in-memory database for integration tests."""
    db = Database(":memory:")
    db.initialize()
    return db


# ------------------------------------------------------------------
# Cost tracking
# ------------------------------------------------------------------


@pytest.fixture()
def cost_tracker(integration_db: Database, full_settings: Settings) -> CostTracker:
    """CostTracker wired to integration DB."""
    return CostTracker(db=integration_db, settings=full_settings)


# ------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------


@pytest.fixture()
def classifier(full_settings: Settings) -> TaskClassifier:
    """TaskClassifier with default settings."""
    return TaskClassifier(full_settings)


# ------------------------------------------------------------------
# Auth manager mock
# ------------------------------------------------------------------


@pytest.fixture()
def mock_auth_manager() -> MagicMock:
    """A mock AuthManager that reports all providers as having keys."""
    auth = MagicMock()

    def _get_key(provider: str) -> str | None:
        # Return a fake key for every known provider (never a real key)
        return f"fake-key-for-{provider}"

    auth.get_key = MagicMock(side_effect=_get_key)
    return auth


# ------------------------------------------------------------------
# Provider registry
# ------------------------------------------------------------------


@pytest.fixture()
def provider_registry(
    full_settings: Settings,
    mock_auth_manager: MagicMock,
) -> ProviderRegistry:
    """ProviderRegistry with all providers available."""
    return ProviderRegistry(settings=full_settings, auth_manager=mock_auth_manager)


# ------------------------------------------------------------------
# Model selector (imports here to keep fixture fast)
# ------------------------------------------------------------------


@pytest.fixture()
def model_selector(
    full_settings: Settings,
    provider_registry: ProviderRegistry,
    cost_tracker: CostTracker,
    integration_db: Database,
) -> Any:
    """ModelSelector with all dependencies wired."""
    from prism.router.selector import ModelSelector

    return ModelSelector(
        settings=full_settings,
        registry=provider_registry,
        cost_tracker=cost_tracker,
    )


# ------------------------------------------------------------------
# Path guard
# ------------------------------------------------------------------


@pytest.fixture()
def path_guard(tmp_path: Path) -> PathGuard:
    """PathGuard rooted at tmp_path."""
    return PathGuard(project_root=tmp_path, excluded_patterns=[".env", "*.key"])


# ------------------------------------------------------------------
# Audit logger
# ------------------------------------------------------------------


@pytest.fixture()
def audit_logger(tmp_path: Path) -> AuditLogger:
    """AuditLogger writing to a temporary directory."""
    return AuditLogger(log_path=tmp_path / "audit.log")

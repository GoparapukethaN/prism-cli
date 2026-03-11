"""Fixtures for network tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prism.network.connectivity import ConnectivityChecker, OfflineRouter
from prism.providers.base import (
    ComplexityTier,
    ModelInfo,
    ProviderConfig,
)


@pytest.fixture
def checker() -> ConnectivityChecker:
    """A ConnectivityChecker with a short interval for testing."""
    return ConnectivityChecker(check_interval=0.1)


@pytest.fixture
def mock_registry() -> MagicMock:
    """Mock ProviderRegistry with an Ollama provider configured."""
    registry = MagicMock()
    ollama_config = ProviderConfig(
        name="ollama",
        display_name="Ollama (Local)",
        api_key_env="",
        models=[
            ModelInfo(
                id="ollama/qwen2.5-coder:7b",
                display_name="Qwen 2.5 Coder 7B",
                provider="ollama",
                tier=ComplexityTier.SIMPLE,
                input_cost_per_1m=0.0,
                output_cost_per_1m=0.0,
                context_window=32_768,
            ),
        ],
    )
    registry.get_provider.side_effect = lambda name: (
        ollama_config if name == "ollama" else None
    )
    return registry


@pytest.fixture
def offline_router(
    mock_registry: MagicMock,
    checker: ConnectivityChecker,
) -> OfflineRouter:
    """An OfflineRouter with mocked dependencies."""
    return OfflineRouter(
        provider_registry=mock_registry,
        connectivity_checker=checker,
    )

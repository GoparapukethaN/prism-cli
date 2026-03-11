"""Tests for the provider registry."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from prism.config.schema import PrismConfig
from prism.config.settings import Settings
from prism.providers.base import ComplexityTier
from prism.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    config = PrismConfig(prism_home=tmp_path / ".prism")
    return Settings(config=config, project_root=tmp_path)


@pytest.fixture
def auth_with_all_keys() -> MagicMock:
    """AuthManager mock with all provider keys configured."""
    manager = MagicMock()
    manager.get_key.side_effect = {
        "anthropic": "sk-ant-test-1234",
        "openai": "sk-test-1234",
        "google": "test-google-1234",
        "deepseek": "sk-deepseek-1234",
        "groq": "gsk_test_1234",
        "mistral": "test-mistral-1234",
    }.get
    return manager


@pytest.fixture
def auth_openai_only() -> MagicMock:
    """AuthManager mock with only OpenAI key configured."""
    manager = MagicMock()
    manager.get_key.side_effect = lambda provider: (
        "sk-test-1234" if provider == "openai" else None
    )
    return manager


@pytest.fixture
def registry(settings: Settings, auth_with_all_keys: MagicMock) -> ProviderRegistry:
    return ProviderRegistry(settings, auth_with_all_keys)


class TestProviderRegistration:
    def test_all_builtin_providers_registered(self, registry: ProviderRegistry) -> None:
        providers = registry.all_providers
        assert "anthropic" in providers
        assert "openai" in providers
        assert "google" in providers
        assert "deepseek" in providers
        assert "groq" in providers
        assert "mistral" in providers
        assert "ollama" in providers

    def test_all_models_registered(self, registry: ProviderRegistry) -> None:
        models = registry.all_models
        assert len(models) > 10  # Should have many models

    def test_get_model_info(self, registry: ProviderRegistry) -> None:
        model = registry.get_model_info("gpt-4o")
        assert model is not None
        assert model.provider == "openai"
        assert model.display_name == "GPT-4o"
        assert model.tier == ComplexityTier.COMPLEX

    def test_get_unknown_model(self, registry: ProviderRegistry) -> None:
        assert registry.get_model_info("nonexistent-model") is None


class TestProviderAvailability:
    def test_available_with_key(
        self, settings: Settings, auth_with_all_keys: MagicMock
    ) -> None:
        registry = ProviderRegistry(settings, auth_with_all_keys)
        assert registry.is_provider_available("anthropic") is True
        assert registry.is_provider_available("openai") is True

    def test_unavailable_without_key(
        self, settings: Settings, auth_openai_only: MagicMock
    ) -> None:
        registry = ProviderRegistry(settings, auth_openai_only)
        assert registry.is_provider_available("openai") is True
        assert registry.is_provider_available("anthropic") is False

    def test_ollama_available_without_key(
        self, settings: Settings, auth_openai_only: MagicMock
    ) -> None:
        registry = ProviderRegistry(settings, auth_openai_only)
        assert registry.is_provider_available("ollama") is True

    def test_nonexistent_provider_unavailable(self, registry: ProviderRegistry) -> None:
        assert registry.is_provider_available("nonexistent") is False


class TestModelRetrieval:
    def test_get_available_models_all_tiers(self, registry: ProviderRegistry) -> None:
        models = registry.get_available_models()
        assert len(models) > 0
        # Should be sorted by cost (cheapest first)
        costs = [m.input_cost_per_1m + m.output_cost_per_1m for m in models]
        assert costs == sorted(costs)

    def test_get_simple_tier_models(self, registry: ProviderRegistry) -> None:
        models = registry.get_available_models(ComplexityTier.SIMPLE)
        for model in models:
            assert model.tier == ComplexityTier.SIMPLE

    def test_get_complex_tier_models(self, registry: ProviderRegistry) -> None:
        models = registry.get_available_models(ComplexityTier.COMPLEX)
        for model in models:
            assert model.tier == ComplexityTier.COMPLEX

    def test_get_models_for_tier_includes_fallbacks(self, registry: ProviderRegistry) -> None:
        # MEDIUM tier should include MEDIUM models + SIMPLE fallbacks
        models = registry.get_models_for_tier(ComplexityTier.MEDIUM)
        tiers = {m.tier for m in models}
        assert ComplexityTier.MEDIUM in tiers
        # SIMPLE models should be available as fallbacks
        assert ComplexityTier.SIMPLE in tiers

    def test_simple_tier_no_upward_fallback(self, registry: ProviderRegistry) -> None:
        models = registry.get_models_for_tier(ComplexityTier.SIMPLE)
        for model in models:
            assert model.tier == ComplexityTier.SIMPLE

    def test_only_available_models_returned(
        self, settings: Settings, auth_openai_only: MagicMock
    ) -> None:
        registry = ProviderRegistry(settings, auth_openai_only)
        models = registry.get_available_models()
        providers = {m.provider for m in models}
        # Only OpenAI and Ollama should have models (others have no key)
        assert providers.issubset({"openai", "ollama"})


class TestProviderStatus:
    def test_rate_limited_provider_excluded(self, registry: ProviderRegistry) -> None:
        from datetime import datetime, timedelta

        status = registry.get_status("groq")
        assert status is not None
        future = datetime.now(UTC) + timedelta(seconds=60)
        status.mark_rate_limited(future)

        assert not registry.is_provider_available("groq")

    def test_recovered_provider_available(self, registry: ProviderRegistry) -> None:
        from datetime import datetime, timedelta

        status = registry.get_status("groq")
        assert status is not None
        past = datetime.now(UTC) - timedelta(seconds=60)
        status.mark_rate_limited(past)

        # Rate limit expired
        assert registry.is_provider_available("groq")


class TestFreeTier:
    def test_google_has_free_tier(self, registry: ProviderRegistry) -> None:
        remaining = registry.get_free_tier_remaining("google")
        assert remaining is not None
        assert remaining == 1500

    def test_ollama_no_free_tier(self, registry: ProviderRegistry) -> None:
        remaining = registry.get_free_tier_remaining("ollama")
        assert remaining is None

    def test_free_tier_decrements(self, registry: ProviderRegistry) -> None:
        status = registry.get_status("google")
        assert status is not None
        status.increment_free_tier_usage()
        status.increment_free_tier_usage()
        remaining = registry.get_free_tier_remaining("google")
        assert remaining == 1498


class TestExcludedProviders:
    def test_excluded_provider_not_registered(self, tmp_path: Path) -> None:
        config = PrismConfig(
            prism_home=tmp_path / ".prism",
            excluded_providers=["deepseek", "groq"],
        )
        settings = Settings(config=config, project_root=tmp_path)
        auth = MagicMock()
        auth.get_key.return_value = "test-key"

        registry = ProviderRegistry(settings, auth)
        assert "deepseek" not in registry.all_providers
        assert "groq" not in registry.all_providers
        assert "openai" in registry.all_providers


class TestListProviders:
    def test_list_returns_all_providers(self, registry: ProviderRegistry) -> None:
        providers = registry.list_providers()
        names = [p["name"] for p in providers]
        assert "anthropic" in names
        assert "openai" in names
        assert "ollama" in names

    def test_list_includes_model_count(self, registry: ProviderRegistry) -> None:
        providers = registry.list_providers()
        for p in providers:
            assert "model_count" in p
            assert p["model_count"] > 0

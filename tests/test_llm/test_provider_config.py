"""Tests for extended provider configs — 8+ tests, fully offline."""

from __future__ import annotations

import pytest

from prism.cost.pricing import MODEL_PRICING
from prism.llm.provider_config import (
    COHERE_CONFIG,
    EXTENDED_PRICING,
    EXTENDED_PROVIDERS,
    KIMI_CONFIG,
    PERPLEXITY_CONFIG,
    QWEN_CONFIG,
    register_extended_providers,
)
from prism.providers.base import BUILTIN_PROVIDERS, ComplexityTier


@pytest.fixture(autouse=True)
def _register() -> None:
    """Ensure extended providers are registered for all tests in this module."""
    register_extended_providers()


class TestNewProvidersRegistered:
    """All new providers appear in BUILTIN_PROVIDERS after registration."""

    def test_kimi_registered(self) -> None:
        names = {p.name for p in BUILTIN_PROVIDERS}
        assert "kimi" in names

    def test_perplexity_registered(self) -> None:
        names = {p.name for p in BUILTIN_PROVIDERS}
        assert "perplexity" in names

    def test_qwen_registered(self) -> None:
        names = {p.name for p in BUILTIN_PROVIDERS}
        assert "qwen" in names

    def test_cohere_registered(self) -> None:
        names = {p.name for p in BUILTIN_PROVIDERS}
        assert "cohere" in names

    def test_together_ai_registered(self) -> None:
        names = {p.name for p in BUILTIN_PROVIDERS}
        assert "together_ai" in names

    def test_fireworks_ai_registered(self) -> None:
        names = {p.name for p in BUILTIN_PROVIDERS}
        assert "fireworks_ai" in names


class TestModelIDs:
    """Model identifiers follow LiteLLM naming convention."""

    def test_kimi_model_ids(self) -> None:
        ids = [m.id for m in KIMI_CONFIG.models]
        assert "moonshot/moonshot-v1-8k" in ids
        assert "moonshot/moonshot-v1-32k" in ids
        assert "moonshot/moonshot-v1-128k" in ids

    def test_perplexity_model_ids(self) -> None:
        ids = [m.id for m in PERPLEXITY_CONFIG.models]
        assert "perplexity/llama-3.1-sonar-large-128k-online" in ids
        assert "perplexity/llama-3.1-sonar-small-128k-online" in ids

    def test_qwen_model_ids(self) -> None:
        ids = [m.id for m in QWEN_CONFIG.models]
        assert "qwen/qwen-max" in ids
        assert "qwen/qwen-turbo" in ids
        assert "qwen/qwen-plus" in ids

    def test_cohere_model_ids(self) -> None:
        ids = [m.id for m in COHERE_CONFIG.models]
        assert "cohere/command-r-plus" in ids
        assert "cohere/command-r" in ids


class TestContextWindows:
    """Every model has a positive context window."""

    def test_all_models_have_context_window(self) -> None:
        for provider in EXTENDED_PROVIDERS:
            for model in provider.models:
                assert model.context_window > 0, (
                    f"{model.id} has invalid context_window: {model.context_window}"
                )


class TestPricing:
    """Pricing data is populated for all new models."""

    def test_all_extended_models_have_pricing(self) -> None:
        for provider in EXTENDED_PROVIDERS:
            for model in provider.models:
                assert model.id in MODEL_PRICING, (
                    f"{model.id} missing from MODEL_PRICING"
                )

    def test_pricing_costs_non_negative(self) -> None:
        for model_id, pricing in EXTENDED_PRICING.items():
            assert pricing.input_cost_per_1m >= 0, f"{model_id} negative input cost"
            assert pricing.output_cost_per_1m >= 0, f"{model_id} negative output cost"

    def test_pricing_matches_model_info(self) -> None:
        """Model info costs should match the pricing table."""
        for provider in EXTENDED_PROVIDERS:
            for model in provider.models:
                pricing = MODEL_PRICING.get(model.id)
                if pricing:
                    assert pricing.input_cost_per_1m == model.input_cost_per_1m
                    assert pricing.output_cost_per_1m == model.output_cost_per_1m


class TestTierAssignments:
    """Tiers are assigned logically."""

    def test_qwen_turbo_is_simple(self) -> None:
        model = next(m for m in QWEN_CONFIG.models if m.id == "qwen/qwen-turbo")
        assert model.tier == ComplexityTier.SIMPLE

    def test_qwen_max_is_complex(self) -> None:
        model = next(m for m in QWEN_CONFIG.models if m.id == "qwen/qwen-max")
        assert model.tier == ComplexityTier.COMPLEX

    def test_cohere_command_r_is_medium(self) -> None:
        model = next(m for m in COHERE_CONFIG.models if m.id == "cohere/command-r")
        assert model.tier == ComplexityTier.MEDIUM

    def test_cohere_command_r_plus_is_complex(self) -> None:
        model = next(m for m in COHERE_CONFIG.models if m.id == "cohere/command-r-plus")
        assert model.tier == ComplexityTier.COMPLEX


class TestExistingProviderExtensions:
    """Extra models added to existing providers (e.g., Gemini 2.5 Flash)."""

    def test_gemini_2_5_flash_in_google(self) -> None:
        google = next(p for p in BUILTIN_PROVIDERS if p.name == "google")
        ids = [m.id for m in google.models]
        assert "gemini/gemini-2.5-flash" in ids

    def test_gemini_2_5_flash_has_pricing(self) -> None:
        assert "gemini/gemini-2.5-flash" in MODEL_PRICING


class TestIdempotentRegistration:
    """Calling register_extended_providers() multiple times is safe."""

    def test_no_duplicate_providers(self) -> None:
        register_extended_providers()
        register_extended_providers()
        names = [p.name for p in BUILTIN_PROVIDERS]
        # Each new provider should appear exactly once.
        for provider in EXTENDED_PROVIDERS:
            assert names.count(provider.name) == 1


class TestAPIKeyEnv:
    """Each provider has an api_key_env set."""

    def test_all_providers_have_api_key_env(self) -> None:
        for provider in EXTENDED_PROVIDERS:
            assert provider.api_key_env, (
                f"{provider.name} missing api_key_env"
            )

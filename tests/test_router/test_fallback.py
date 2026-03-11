"""Tests for prism.router.fallback — FallbackChain."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prism.providers.base import ComplexityTier, ModelInfo
from prism.router.fallback import FallbackChain

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _model(model_id: str, tier: ComplexityTier, cost: float = 1.0) -> ModelInfo:
    """Shortcut to build a ModelInfo for testing."""
    return ModelInfo(
        id=model_id,
        display_name=model_id,
        provider=model_id.split("/", maxsplit=1)[0] if "/" in model_id else "test",
        tier=tier,
        input_cost_per_1m=cost,
        output_cost_per_1m=cost,
        context_window=128_000,
    )


def _make_registry(
    models_by_tier: dict[ComplexityTier, list[ModelInfo]] | None = None,
    all_models: dict[str, ModelInfo] | None = None,
) -> MagicMock:
    """Build a mock ProviderRegistry with canned model lists."""
    registry = MagicMock()
    models_by_tier = models_by_tier or {}
    all_models = all_models or {}

    def _get_available(tier: ComplexityTier) -> list[ModelInfo]:
        return models_by_tier.get(tier, [])

    def _get_info(model_id: str) -> ModelInfo | None:
        return all_models.get(model_id)

    registry.get_available_models.side_effect = _get_available
    registry.get_model_info.side_effect = _get_info
    return registry


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def complex_model() -> ModelInfo:
    return _model("anthropic/claude-sonnet", ComplexityTier.COMPLEX, 3.0)


@pytest.fixture
def medium_model() -> ModelInfo:
    return _model("deepseek/deepseek-chat", ComplexityTier.MEDIUM, 0.27)


@pytest.fixture
def simple_model() -> ModelInfo:
    return _model("groq/mixtral-8x7b", ComplexityTier.SIMPLE, 0.24)


@pytest.fixture
def ollama_model() -> ModelInfo:
    return _model("ollama/qwen2.5-coder:7b", ComplexityTier.SIMPLE, 0.0)


# ------------------------------------------------------------------
# build_chain()
# ------------------------------------------------------------------


class TestBuildChain:
    """Tests for FallbackChain.build_chain()."""

    def test_primary_is_first(
        self, complex_model: ModelInfo, medium_model: ModelInfo
    ) -> None:
        registry = _make_registry(
            {ComplexityTier.COMPLEX: [complex_model], ComplexityTier.MEDIUM: [medium_model]},
        )
        chain = FallbackChain(registry)
        result = chain.build_chain(ComplexityTier.COMPLEX, complex_model.id)
        assert result[0] == complex_model.id

    def test_chain_includes_same_tier(self) -> None:
        m1 = _model("a/m1", ComplexityTier.MEDIUM, 0.5)
        m2 = _model("b/m2", ComplexityTier.MEDIUM, 1.0)
        registry = _make_registry(
            {ComplexityTier.MEDIUM: [m1, m2], ComplexityTier.SIMPLE: []},
        )
        chain = FallbackChain(registry)
        result = chain.build_chain(ComplexityTier.MEDIUM, m1.id)
        assert m2.id in result

    def test_chain_includes_cheaper_tier(
        self, complex_model: ModelInfo, medium_model: ModelInfo
    ) -> None:
        registry = _make_registry(
            {ComplexityTier.COMPLEX: [complex_model], ComplexityTier.MEDIUM: [medium_model]},
        )
        chain = FallbackChain(registry)
        result = chain.build_chain(ComplexityTier.COMPLEX, complex_model.id)
        assert medium_model.id in result

    def test_chain_includes_ollama_fallback(
        self, complex_model: ModelInfo, ollama_model: ModelInfo
    ) -> None:
        registry = _make_registry(
            {ComplexityTier.COMPLEX: [complex_model], ComplexityTier.MEDIUM: []},
            {ollama_model.id: ollama_model},
        )
        chain = FallbackChain(registry)
        result = chain.build_chain(ComplexityTier.COMPLEX, complex_model.id)
        assert ollama_model.id in result

    def test_no_duplicates(self) -> None:
        m = _model("ollama/qwen2.5-coder:7b", ComplexityTier.SIMPLE, 0.0)
        registry = _make_registry(
            {ComplexityTier.SIMPLE: [m]},
            {m.id: m},
        )
        chain = FallbackChain(registry)
        result = chain.build_chain(ComplexityTier.SIMPLE, m.id)
        assert len(result) == len(set(result))

    def test_simple_tier_has_no_cheaper_tier(self, simple_model: ModelInfo) -> None:
        registry = _make_registry({ComplexityTier.SIMPLE: [simple_model]})
        chain = FallbackChain(registry)
        result = chain.build_chain(ComplexityTier.SIMPLE, simple_model.id)
        # Should still work — just won't add cheaper-tier models
        assert result[0] == simple_model.id

    def test_empty_registry(self) -> None:
        registry = _make_registry()
        chain = FallbackChain(registry)
        result = chain.build_chain(ComplexityTier.MEDIUM, "nonexistent/model")
        # Primary is always added
        assert result == ["nonexistent/model"]


# ------------------------------------------------------------------
# next_model()
# ------------------------------------------------------------------


class TestNextModel:
    """Tests for FallbackChain.next_model()."""

    def test_returns_first_untried(self) -> None:
        chain = ["a", "b", "c"]
        assert FallbackChain.next_model(chain, failed=set()) == "a"

    def test_skips_failed(self) -> None:
        chain = ["a", "b", "c"]
        assert FallbackChain.next_model(chain, failed={"a"}) == "b"

    def test_returns_none_when_all_failed(self) -> None:
        chain = ["a", "b"]
        assert FallbackChain.next_model(chain, failed={"a", "b"}) is None

    def test_skips_multiple_failed(self) -> None:
        chain = ["a", "b", "c", "d"]
        assert FallbackChain.next_model(chain, failed={"a", "b", "c"}) == "d"

    def test_empty_chain(self) -> None:
        assert FallbackChain.next_model([], failed=set()) is None

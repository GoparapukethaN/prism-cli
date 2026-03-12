"""Tests for prism.router.selector — model selection and quality floor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from prism.config.schema import PrismConfig, RoutingConfig
from prism.config.settings import Settings
from prism.exceptions import NoModelsAvailableError
from prism.providers.base import ComplexityTier, ModelInfo
from prism.router.selector import ModelSelection, ModelSelector

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SIMPLE_MODEL = ModelInfo(
    id="groq/mixtral-8x7b-32768",
    display_name="Mixtral 8x7B",
    provider="groq",
    tier=ComplexityTier.SIMPLE,
    input_cost_per_1m=0.24,
    output_cost_per_1m=0.24,
    context_window=32_768,
    supports_tools=True,
)

_SIMPLE_NO_TOOLS = ModelInfo(
    id="ollama/llama3.2:3b",
    display_name="Llama 3.2 3B",
    provider="ollama",
    tier=ComplexityTier.SIMPLE,
    input_cost_per_1m=0.00,
    output_cost_per_1m=0.00,
    context_window=32_768,
    supports_tools=False,
)

_MEDIUM_MODEL = ModelInfo(
    id="gpt-4o-mini",
    display_name="GPT-4o Mini",
    provider="openai",
    tier=ComplexityTier.MEDIUM,
    input_cost_per_1m=0.15,
    output_cost_per_1m=0.60,
    context_window=128_000,
    supports_tools=True,
)

_COMPLEX_MODEL = ModelInfo(
    id="gpt-4o",
    display_name="GPT-4o",
    provider="openai",
    tier=ComplexityTier.COMPLEX,
    input_cost_per_1m=2.50,
    output_cost_per_1m=10.00,
    context_window=128_000,
    supports_tools=True,
)


def _make_selector(
    tmp_path: Path,
    models_for_tier: dict[ComplexityTier, list[ModelInfo]] | None = None,
    routing_config: RoutingConfig | None = None,
    budget_remaining: float | None = None,
) -> ModelSelector:
    """Build a ModelSelector with a mocked registry and cost tracker."""
    routing = routing_config or RoutingConfig()
    config = PrismConfig(prism_home=tmp_path / ".prism", routing=routing)
    settings = Settings(config=config, project_root=tmp_path)

    # Mock registry
    registry = MagicMock()

    def _get_models_for_tier(tier: ComplexityTier) -> list[ModelInfo]:
        if models_for_tier is not None:
            return list(models_for_tier.get(tier, []))
        return []

    registry.get_models_for_tier.side_effect = _get_models_for_tier

    # Mock cost tracker
    cost_tracker = MagicMock()
    cost_tracker.get_budget_remaining.return_value = budget_remaining

    return ModelSelector(
        settings=settings,
        registry=registry,
        cost_tracker=cost_tracker,
    )


# ---------------------------------------------------------------------------
# Tests: basic selection (sanity)
# ---------------------------------------------------------------------------


class TestBasicSelection:
    """Sanity checks for model selection without tools."""

    def test_select_returns_model_selection(self, tmp_path: Path) -> None:
        selector = _make_selector(
            tmp_path,
            models_for_tier={ComplexityTier.MEDIUM: [_MEDIUM_MODEL]},
        )
        result = selector.select(
            tier=ComplexityTier.MEDIUM,
            prompt="refactor this function",
        )
        assert isinstance(result, ModelSelection)
        assert result.model_id == _MEDIUM_MODEL.id

    def test_no_models_raises(self, tmp_path: Path) -> None:
        selector = _make_selector(tmp_path, models_for_tier={})
        with pytest.raises(NoModelsAvailableError):
            selector.select(
                tier=ComplexityTier.MEDIUM,
                prompt="hello",
            )


# ---------------------------------------------------------------------------
# Tests: Gap 3 — Quality floor for tool-use tasks
# ---------------------------------------------------------------------------


class TestToolUseQualityFloor:
    """When tools_enabled=True, SIMPLE tasks should be bumped to MEDIUM."""

    def test_simple_tier_bumped_to_medium_when_tools_enabled(
        self, tmp_path: Path,
    ) -> None:
        """SIMPLE + tools_enabled → effective tier is MEDIUM."""
        selector = _make_selector(
            tmp_path,
            models_for_tier={
                ComplexityTier.SIMPLE: [_SIMPLE_MODEL],
                ComplexityTier.MEDIUM: [_MEDIUM_MODEL],
            },
        )
        result = selector.select(
            tier=ComplexityTier.SIMPLE,
            prompt="fix typo",
            tools_enabled=True,
        )
        # The selection should be from MEDIUM tier, not SIMPLE
        assert result.tier == ComplexityTier.MEDIUM
        assert result.model_id == _MEDIUM_MODEL.id

    def test_simple_tier_not_bumped_when_tools_disabled(
        self, tmp_path: Path,
    ) -> None:
        """Without tools, SIMPLE stays SIMPLE."""
        selector = _make_selector(
            tmp_path,
            models_for_tier={
                ComplexityTier.SIMPLE: [_SIMPLE_MODEL],
                ComplexityTier.MEDIUM: [_MEDIUM_MODEL],
            },
        )
        result = selector.select(
            tier=ComplexityTier.SIMPLE,
            prompt="fix typo",
            tools_enabled=False,
        )
        assert result.tier == ComplexityTier.SIMPLE
        assert result.model_id == _SIMPLE_MODEL.id

    def test_medium_tier_not_changed_when_tools_enabled(
        self, tmp_path: Path,
    ) -> None:
        """MEDIUM is already >= minimum, should stay MEDIUM."""
        selector = _make_selector(
            tmp_path,
            models_for_tier={
                ComplexityTier.MEDIUM: [_MEDIUM_MODEL],
            },
        )
        result = selector.select(
            tier=ComplexityTier.MEDIUM,
            prompt="refactor code",
            tools_enabled=True,
        )
        assert result.tier == ComplexityTier.MEDIUM

    def test_complex_tier_not_changed_when_tools_enabled(
        self, tmp_path: Path,
    ) -> None:
        """COMPLEX is above minimum, should stay COMPLEX."""
        selector = _make_selector(
            tmp_path,
            models_for_tier={
                ComplexityTier.COMPLEX: [_COMPLEX_MODEL],
            },
        )
        result = selector.select(
            tier=ComplexityTier.COMPLEX,
            prompt="redesign the architecture",
            tools_enabled=True,
        )
        assert result.tier == ComplexityTier.COMPLEX

    def test_custom_minimum_tier_complex(self, tmp_path: Path) -> None:
        """tool_use_minimum_tier='complex' should bump MEDIUM to COMPLEX."""
        routing = RoutingConfig(tool_use_minimum_tier="complex")
        selector = _make_selector(
            tmp_path,
            models_for_tier={
                ComplexityTier.MEDIUM: [_MEDIUM_MODEL],
                ComplexityTier.COMPLEX: [_COMPLEX_MODEL],
            },
            routing_config=routing,
        )
        result = selector.select(
            tier=ComplexityTier.MEDIUM,
            prompt="edit file",
            tools_enabled=True,
        )
        assert result.tier == ComplexityTier.COMPLEX

    def test_custom_minimum_tier_simple_no_bump(self, tmp_path: Path) -> None:
        """tool_use_minimum_tier='simple' means no bump for any tier."""
        routing = RoutingConfig(tool_use_minimum_tier="simple")
        selector = _make_selector(
            tmp_path,
            models_for_tier={
                ComplexityTier.SIMPLE: [_SIMPLE_MODEL],
            },
            routing_config=routing,
        )
        result = selector.select(
            tier=ComplexityTier.SIMPLE,
            prompt="fix typo",
            tools_enabled=True,
        )
        assert result.tier == ComplexityTier.SIMPLE

    def test_tools_enabled_filters_out_no_tools_models(
        self, tmp_path: Path,
    ) -> None:
        """Models with supports_tools=False are excluded when tools_enabled."""
        selector = _make_selector(
            tmp_path,
            models_for_tier={
                ComplexityTier.MEDIUM: [_MEDIUM_MODEL, _SIMPLE_NO_TOOLS],
            },
        )
        result = selector.select(
            tier=ComplexityTier.MEDIUM,
            prompt="write a file",
            tools_enabled=True,
        )
        # _SIMPLE_NO_TOOLS should have been filtered out
        assert result.model_id == _MEDIUM_MODEL.id

    def test_all_models_no_tools_raises(self, tmp_path: Path) -> None:
        """If all models lack tool support and tools_enabled, raise."""
        selector = _make_selector(
            tmp_path,
            models_for_tier={
                ComplexityTier.MEDIUM: [_SIMPLE_NO_TOOLS],
            },
        )
        with pytest.raises(NoModelsAvailableError):
            selector.select(
                tier=ComplexityTier.MEDIUM,
                prompt="write code",
                tools_enabled=True,
            )

    def test_tools_disabled_keeps_no_tool_models(
        self, tmp_path: Path,
    ) -> None:
        """When tools_enabled=False, supports_tools=False models stay."""
        selector = _make_selector(
            tmp_path,
            models_for_tier={
                ComplexityTier.SIMPLE: [_SIMPLE_NO_TOOLS],
            },
        )
        result = selector.select(
            tier=ComplexityTier.SIMPLE,
            prompt="explain code",
            tools_enabled=False,
        )
        assert result.model_id == _SIMPLE_NO_TOOLS.id


# ---------------------------------------------------------------------------
# Tests: ModelSelection dataclass
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Tests for the ModelSelection dataclass."""

    def test_fields(self) -> None:
        sel = ModelSelection(
            model_id="gpt-4o",
            provider="openai",
            tier=ComplexityTier.COMPLEX,
            estimated_cost=0.01,
            fallback_chain=["gpt-4o-mini"],
            reasoning="test",
        )
        assert sel.model_id == "gpt-4o"
        assert sel.provider == "openai"
        assert sel.tier == ComplexityTier.COMPLEX
        assert sel.fallback_chain == ["gpt-4o-mini"]


# ---------------------------------------------------------------------------
# Tests: _TIER_ORDER constant
# ---------------------------------------------------------------------------


class TestTierOrder:
    """Tests for tier ordering used by quality floor."""

    def test_tier_order_values(self) -> None:
        order = ModelSelector._TIER_ORDER
        assert order["simple"] < order["medium"] < order["complex"]

    def test_tier_order_completeness(self) -> None:
        order = ModelSelector._TIER_ORDER
        for tier in ComplexityTier:
            assert tier.value in order

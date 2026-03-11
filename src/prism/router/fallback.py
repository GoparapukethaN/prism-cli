"""Fallback chain management for model selection.

Builds an ordered list of candidate models so that when the primary
model fails, Prism automatically tries the next best option until it
exhausts the chain or succeeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prism.providers.base import ComplexityTier

if TYPE_CHECKING:
    from prism.providers.registry import ProviderRegistry


# Tier ordering from cheapest to most expensive
_TIER_ORDER: list[ComplexityTier] = [
    ComplexityTier.SIMPLE,
    ComplexityTier.MEDIUM,
    ComplexityTier.COMPLEX,
]

# Ollama model IDs used as ultimate local fallbacks
_OLLAMA_FALLBACKS: list[str] = [
    "ollama/qwen2.5-coder:7b",
    "ollama/llama3.2:3b",
    "ollama/deepseek-coder-v2:16b",
]


class FallbackChain:
    """Builds and manages ordered fallback chains for model routing.

    Given a primary model and complexity tier, the chain is built as:

    1. The *primary* model.
    2. Same-tier alternatives (sorted by cost, cheapest first).
    3. Models from the next-cheaper tier.
    4. Ollama local models (always-available fallback).
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        """Initialise with a provider registry.

        Args:
            registry: Provider registry for model/availability lookups.
        """
        self._registry = registry

    def build_chain(self, tier: ComplexityTier, primary: str) -> list[str]:
        """Build a complete fallback chain.

        Args:
            tier: The task's complexity tier.
            primary: LiteLLM model ID of the primary (first-choice) model.

        Returns:
            Ordered list of model IDs starting with *primary*.
        """
        seen: set[str] = set()
        chain: list[str] = []

        def _add(model_id: str) -> None:
            if model_id not in seen:
                seen.add(model_id)
                chain.append(model_id)

        # 1. Primary
        _add(primary)

        # 2. Same-tier alternatives (cheapest first — registry already sorts)
        for model in self._registry.get_available_models(tier):
            _add(model.id)

        # 3. Next-cheaper tier
        cheaper_tier = self._cheaper_tier(tier)
        if cheaper_tier is not None:
            for model in self._registry.get_available_models(cheaper_tier):
                _add(model.id)

        # 4. Ollama local fallbacks
        for fallback_id in _OLLAMA_FALLBACKS:
            model_info = self._registry.get_model_info(fallback_id)
            if model_info is not None:
                _add(fallback_id)

        return chain

    @staticmethod
    def next_model(chain: list[str], failed: set[str]) -> str | None:
        """Return the next untried model from the chain.

        Args:
            chain: The ordered fallback chain.
            failed: Set of model IDs that have already been tried and failed.

        Returns:
            The next model ID to try, or ``None`` if all have been tried.
        """
        for model_id in chain:
            if model_id not in failed:
                return model_id
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cheaper_tier(tier: ComplexityTier) -> ComplexityTier | None:
        """Return the next-cheaper tier, or ``None`` if already cheapest."""
        idx = _TIER_ORDER.index(tier)
        if idx > 0:
            return _TIER_ORDER[idx - 1]
        return None

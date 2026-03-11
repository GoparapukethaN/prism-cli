"""Model selection — picks the optimal model for a classified task."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from prism.cost.pricing import calculate_cost, estimate_input_tokens, estimate_output_tokens
from prism.exceptions import BudgetExceededError, NoModelsAvailableError
from prism.providers.base import ComplexityTier, ModelInfo

if TYPE_CHECKING:
    from prism.config.settings import Settings
    from prism.cost.tracker import CostTracker
    from prism.providers.registry import ProviderRegistry

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ModelSelection:
    """Result of model selection."""

    model_id: str  # LiteLLM model identifier
    provider: str  # Provider name
    tier: ComplexityTier  # Why this model was chosen
    estimated_cost: float  # Estimated cost in USD
    fallback_chain: list[str]  # Ordered fallback model IDs
    reasoning: str  # Human-readable explanation


@dataclass
class ModelCandidate:
    """A candidate model during selection."""

    model: ModelInfo
    estimated_cost: float
    success_rate: float  # Historical success rate (0-1)
    rank_score: float = 0.0  # Combined quality/cost score
    available: bool = True
    unavailable_reason: str = ""


class ModelSelector:
    """Selects the optimal model for a classified task.

    Evaluates candidate models by estimated cost, historical success rate,
    and availability, then returns the best option with a fallback chain.
    """

    def __init__(
        self,
        settings: Settings,
        registry: ProviderRegistry,
        cost_tracker: CostTracker,
    ) -> None:
        """Initialize the model selector.

        Args:
            settings: Application settings.
            registry: Provider registry for model availability.
            cost_tracker: Cost tracker for budget checks.
        """
        self._settings = settings
        self._registry = registry
        self._cost_tracker = cost_tracker
        self._quality_weight = settings.get("routing.quality_weight", 0.7)
        self._exploration_rate = settings.get("routing.exploration_rate", 0.1)

    def select(
        self,
        tier: ComplexityTier,
        prompt: str,
        context_tokens: int = 0,
    ) -> ModelSelection:
        """Select the optimal model for a task.

        Args:
            tier: Classified complexity tier.
            prompt: User's prompt (for token estimation).
            context_tokens: Additional context tokens (repo map, files, history).

        Returns:
            ModelSelection with chosen model and fallback chain.

        Raises:
            NoModelsAvailableError: No models available for the tier.
            BudgetExceededError: No model fits within remaining budget.
        """
        # Check for pinned model override
        pinned = self._settings.config.pinned_model
        if pinned:
            return self._select_pinned(pinned, tier, prompt, context_tokens)

        # Get candidates
        candidates = self._build_candidates(tier, prompt, context_tokens)
        if not candidates:
            raise NoModelsAvailableError(tier.value)

        # Budget check
        budget_remaining = self._cost_tracker.get_budget_remaining()
        if budget_remaining is not None:
            candidates = self._filter_by_budget(candidates, budget_remaining)
            if not candidates:
                cheapest = min(
                    self._build_candidates(tier, prompt, context_tokens),
                    key=lambda c: c.estimated_cost,
                    default=None,
                )
                cheapest_cost = cheapest.estimated_cost if cheapest else 0.0
                raise BudgetExceededError(budget_remaining, cheapest_cost)

        # Rank candidates
        ranked = self._rank_candidates(candidates)

        # Exploration: occasionally try non-default model
        selected = self._maybe_explore(ranked)

        # Build fallback chain from remaining candidates
        fallback_chain = [
            c.model.id for c in ranked if c.model.id != selected.model.id
        ][:4]  # Keep top 4 fallbacks

        reasoning = self._explain_selection(selected, ranked, tier)

        logger.info(
            "model_selected",
            model=selected.model.id,
            provider=selected.model.provider,
            tier=tier.value,
            cost_est=f"${selected.estimated_cost:.6f}",
            rank_score=round(selected.rank_score, 3),
            success_rate=round(selected.success_rate, 3),
            fallback_count=len(fallback_chain),
        )

        return ModelSelection(
            model_id=selected.model.id,
            provider=selected.model.provider,
            tier=tier,
            estimated_cost=selected.estimated_cost,
            fallback_chain=fallback_chain,
            reasoning=reasoning,
        )

    def _select_pinned(
        self,
        model_id: str,
        tier: ComplexityTier,
        prompt: str,
        context_tokens: int,
    ) -> ModelSelection:
        """Select a pinned model (user override)."""
        model = self._registry.get_model_info(model_id)
        if model is None:
            logger.warning("pinned_model_not_found", model=model_id)
            raise NoModelsAvailableError(f"Pinned model not found: {model_id}")

        input_tokens = estimate_input_tokens(prompt) + context_tokens
        output_tokens = estimate_output_tokens(prompt)

        try:
            cost = calculate_cost(model_id, input_tokens, output_tokens)
        except ValueError:
            cost = 0.0

        return ModelSelection(
            model_id=model_id,
            provider=model.provider,
            tier=tier,
            estimated_cost=cost,
            fallback_chain=[],
            reasoning=f"Pinned model: {model.display_name} (user override)",
        )

    def _build_candidates(
        self,
        tier: ComplexityTier,
        prompt: str,
        context_tokens: int,
    ) -> list[ModelCandidate]:
        """Build the list of candidate models for a tier.

        Args:
            tier: Target complexity tier.
            prompt: User's prompt.
            context_tokens: Additional context tokens.

        Returns:
            List of ModelCandidate with estimated costs and success rates.
        """
        models = self._registry.get_models_for_tier(tier)
        input_tokens = estimate_input_tokens(prompt) + context_tokens
        output_tokens = estimate_output_tokens(prompt)

        candidates: list[ModelCandidate] = []
        for model in models:
            # Skip models that can't fit the context
            if input_tokens + output_tokens > model.context_window * 0.9:
                logger.debug(
                    "model_context_too_small",
                    model=model.id,
                    needed=input_tokens + output_tokens,
                    available=model.context_window,
                )
                continue

            try:
                cost = calculate_cost(model.id, input_tokens, output_tokens)
            except ValueError:
                cost = 0.0

            # Default success rate: higher-tier models get higher default
            success_rate = self._get_success_rate(model, tier)

            candidates.append(
                ModelCandidate(
                    model=model,
                    estimated_cost=cost,
                    success_rate=success_rate,
                )
            )

        return candidates

    def _get_success_rate(self, model: ModelInfo, tier: ComplexityTier) -> float:
        """Get historical success rate for a model on a tier.

        Falls back to default rates based on tier alignment.

        Args:
            model: Model info.
            tier: Task complexity tier.

        Returns:
            Success rate between 0.0 and 1.0.
        """
        # TODO: Query DB for actual historical success rate when available
        # For now, use defaults based on model tier vs task tier alignment

        if model.tier == tier:
            # Model is designed for this tier
            return 0.85
        if model.tier == ComplexityTier.COMPLEX and tier == ComplexityTier.MEDIUM:
            # Overqualified — works but expensive
            return 0.95
        if model.tier == ComplexityTier.MEDIUM and tier == ComplexityTier.SIMPLE:
            return 0.90
        if model.tier == ComplexityTier.SIMPLE and tier == ComplexityTier.MEDIUM:
            # Underqualified — might struggle
            return 0.60
        if model.tier == ComplexityTier.MEDIUM and tier == ComplexityTier.COMPLEX:
            return 0.50
        if model.tier == ComplexityTier.SIMPLE and tier == ComplexityTier.COMPLEX:
            return 0.30
        return 0.70

    def _filter_by_budget(
        self,
        candidates: list[ModelCandidate],
        budget_remaining: float,
    ) -> list[ModelCandidate]:
        """Filter candidates that exceed the remaining budget."""
        return [c for c in candidates if c.estimated_cost <= budget_remaining]

    def _rank_candidates(self, candidates: list[ModelCandidate]) -> list[ModelCandidate]:
        """Rank candidates by quality/cost ratio.

        Free models are ranked by success rate alone.
        Paid models are ranked by a quality/cost blend.

        Args:
            candidates: List of candidates to rank.

        Returns:
            Sorted list (best first).
        """
        for candidate in candidates:
            if candidate.estimated_cost == 0:
                # Free models: rank by success rate alone (huge bonus)
                candidate.rank_score = candidate.success_rate * 100
            else:
                # Paid models: quality per dollar
                cost_weight = 1.0 - self._quality_weight
                quality_score = candidate.success_rate * self._quality_weight
                cost_score = (1.0 / (candidate.estimated_cost + 0.0001)) * cost_weight
                candidate.rank_score = quality_score * cost_score

        candidates.sort(key=lambda c: c.rank_score, reverse=True)
        return candidates

    def _maybe_explore(self, ranked: list[ModelCandidate]) -> ModelCandidate:
        """Occasionally select a non-default model for exploration.

        10% of the time (configurable), picks a random non-top candidate
        to gather comparison data for adaptive learning.

        Args:
            ranked: Ranked list of candidates (best first).

        Returns:
            The selected candidate.
        """
        if len(ranked) <= 1:
            return ranked[0]

        if random.random() < self._exploration_rate:  # noqa: S311
            # Explore: pick a random non-top candidate
            exploration_pool = ranked[1:min(4, len(ranked))]
            selected = random.choice(exploration_pool)  # noqa: S311
            logger.info(
                "exploration_round",
                default=ranked[0].model.id,
                exploring=selected.model.id,
            )
            return selected

        return ranked[0]

    @staticmethod
    def _explain_selection(
        selected: ModelCandidate,
        ranked: list[ModelCandidate],
        tier: ComplexityTier,
    ) -> str:
        """Generate a human-readable explanation of the selection.

        Args:
            selected: The selected candidate.
            ranked: All ranked candidates.
            tier: Target tier.

        Returns:
            Explanation string.
        """
        parts: list[str] = [
            f"Selected {selected.model.display_name} for {tier.value} task"
        ]

        if selected.estimated_cost == 0:
            parts.append("(free)")
        else:
            parts.append(f"(est. ${selected.estimated_cost:.4f})")

        parts.append(f"success rate: {selected.success_rate:.0%}")

        if len(ranked) > 1:
            alternatives = len(ranked) - 1
            parts.append(f"{alternatives} fallback(s) available")

        return ". ".join(parts) + "."

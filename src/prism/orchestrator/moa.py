"""Mixture-of-Agents (MoA) parallel generation with LLM-Blender pairwise ranking.

Academic basis:
    - Wang et al. (Together AI, 2024): MoA ensembles of open-source models
      achieved 65.1% on AlpacaEval 2.0, beating GPT-4o's 57.5%.  Architecture:
      proposer models generate diverse responses, aggregator synthesizes.
    - Jiang et al. (ACL 2023): LLM-Blender pairwise ranking + generative fusion
      outperforms every individual model.

Architecture:
    Layer 1: N proposer models generate independently in parallel.
    Layer 2+: Each model sees ALL outputs from the previous layer, refines.
    Ranking: Pairwise comparison to find best individual output (LLM-Blender).
    Fusion: Aggregator model synthesizes best elements from all outputs.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from prism.llm.completion import CompletionEngine
    from prism.orchestrator.swarm import ModelPool

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass
class MoALayer:
    """A single layer of MoA parallel generation.

    Attributes:
        layer_number: Zero-indexed layer position in the pipeline.
        models: LiteLLM model identifiers used in this layer.
        outputs: Mapping of model_id to its generated output text.
        cost: Total cost in USD for all generations in this layer.
    """

    layer_number: int
    models: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    cost: float = 0.0


@dataclass
class MoAConfig:
    """Configuration for the Mixture-of-Agents pipeline.

    Attributes:
        num_proposers: How many models generate in parallel per layer.
        num_layers: Total MoA layers (each layer sees previous layer outputs).
        use_ranking: Enable LLM-Blender pairwise ranking before fusion.
        fusion_model: Override model for the final synthesis step.  If ``None``,
            the planning model from the model pool is used.
    """

    num_proposers: int = 3
    num_layers: int = 2
    use_ranking: bool = True
    fusion_model: str | None = None


@dataclass
class PairwiseRanking:
    """Result of a single pairwise comparison between two model outputs.

    Attributes:
        candidate_a: LiteLLM model ID of the first candidate.
        candidate_b: LiteLLM model ID of the second candidate.
        winner: Model ID of the comparison winner.
        reason: Human-readable explanation of why the winner is better.
        confidence: Confidence score between 0 and 1.
    """

    candidate_a: str
    candidate_b: str
    winner: str
    reason: str
    confidence: float


@dataclass
class MoAResult:
    """Full result from a Mixture-of-Agents pipeline execution.

    Attributes:
        layers: Ordered list of MoA layers executed.
        rankings: Pairwise ranking results (empty if ranking disabled).
        final_output: The synthesized/fused result text.
        best_individual: Model ID with the highest win rate, or ``None``.
        total_cost: Total cost in USD across all layers, ranking, and fusion.
        participating_models: All model IDs that contributed to the result.
    """

    layers: list[MoALayer] = field(default_factory=list)
    rankings: list[PairwiseRanking] = field(default_factory=list)
    final_output: str = ""
    best_individual: str | None = None
    total_cost: float = 0.0
    participating_models: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_PROPOSER_SYSTEM = """\
You are a helpful AI assistant.  Provide a thorough, accurate, and well-structured \
response to the user's request.  Focus on quality and correctness."""

_REFINE_SYSTEM = """\
You are a helpful AI assistant participating in a multi-model collaboration.  \
Below are responses from other AI models to the same prompt.  Your job is to:

1. Review all previous responses carefully
2. Identify the strengths and weaknesses of each
3. Produce an IMPROVED response that combines the best elements

Previous model outputs:
{previous_outputs}

Now provide your own improved response to the original prompt."""

_JUDGE_SYSTEM = """\
You are an impartial judge comparing two AI responses to the same prompt.  \
Evaluate both responses on:
- Accuracy and correctness
- Completeness and thoroughness
- Clarity and structure
- Relevance to the original prompt

Respond with ONLY a JSON object (no other text):
{{
  "winner": "A" or "B",
  "reason": "brief explanation of why the winner is better",
  "confidence": 0.0 to 1.0
}}"""

_FUSION_SYSTEM = """\
You are a synthesis engine.  Given multiple AI responses to the same prompt \
and quality rankings, produce a single superior response that:

1. Incorporates the best elements from all responses
2. Corrects any errors found in individual responses
3. Fills gaps that individual responses missed
4. Maintains a coherent, unified structure

{ranking_context}

Model outputs:
{model_outputs}

Now synthesize the best possible response to the original prompt."""


# ------------------------------------------------------------------
# OutputRanker — pairwise ranking via judge model (LLM-Blender style)
# ------------------------------------------------------------------


class OutputRanker:
    """Pairwise ranking of outputs using a judge model (LLM-Blender style).

    Compares all pairs of outputs, builds a win-rate matrix, and returns
    models sorted by quality.  The judge model must be different from any
    proposer to avoid self-evaluation bias.

    Args:
        engine: Completion engine for LLM calls (must have a mock backend
            injected for testing).
        judge_model: LiteLLM model identifier for the judge.
    """

    def __init__(self, engine: CompletionEngine, judge_model: str) -> None:
        """Initialise the output ranker.

        Args:
            engine: Completion engine (mock-safe).
            judge_model: Model used as the impartial judge.
        """
        self._engine = engine
        self._judge_model = judge_model

    async def rank(
        self,
        outputs: dict[str, str],
        prompt: str,
    ) -> list[PairwiseRanking]:
        """Rank all outputs via pairwise comparison.

        For *N* outputs this produces *N*(N-1)/2 comparisons run in parallel.

        Args:
            outputs: Mapping of model_id to output text.
            prompt: The original user prompt for context.

        Returns:
            List of ``PairwiseRanking`` results for every pair.

        Raises:
            ValueError: If fewer than 2 outputs are provided.
        """
        model_ids = list(outputs.keys())
        if len(model_ids) < 2:
            raise ValueError("At least 2 outputs are required for pairwise ranking")

        pairs = list(itertools.combinations(model_ids, 2))
        coros = [
            self._compare_pair(
                model_a=a,
                output_a=outputs[a],
                model_b=b,
                output_b=outputs[b],
                prompt=prompt,
            )
            for a, b in pairs
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        rankings: list[PairwiseRanking] = []
        for pair, result in zip(pairs, results, strict=False):
            if isinstance(result, Exception):
                logger.warning(
                    "pairwise_comparison_failed",
                    pair=pair,
                    error=str(result),
                )
                # Fallback: tie goes to the first candidate
                rankings.append(
                    PairwiseRanking(
                        candidate_a=pair[0],
                        candidate_b=pair[1],
                        winner=pair[0],
                        reason=f"Comparison failed ({result}); defaulting to first candidate.",
                        confidence=0.0,
                    ),
                )
            else:
                rankings.append(result)

        logger.info(
            "pairwise_ranking_complete",
            comparisons=len(rankings),
            judge=self._judge_model,
        )
        return rankings

    async def _compare_pair(
        self,
        model_a: str,
        output_a: str,
        model_b: str,
        output_b: str,
        prompt: str,
    ) -> PairwiseRanking:
        """Compare a single pair of outputs using the judge model.

        The comparison prompt anonymises the candidates as "Response A" and
        "Response B" to avoid name-based bias.

        Args:
            model_a: Model ID of the first candidate.
            output_a: Text output from model A.
            model_b: Model ID of the second candidate.
            output_b: Text output from model B.
            prompt: Original user prompt.

        Returns:
            A ``PairwiseRanking`` with the winner, reason, and confidence.
        """
        user_content = (
            f"Original prompt: {prompt}\n\n"
            f"--- Response A ---\n{output_a}\n\n"
            f"--- Response B ---\n{output_b}"
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        result = await self._engine.complete(
            messages=messages,
            model=self._judge_model,
            temperature=0.1,
            max_tokens=512,
        )

        return self._parse_judge_response(result.content, model_a, model_b)

    @staticmethod
    def _parse_judge_response(
        raw: str,
        model_a: str,
        model_b: str,
    ) -> PairwiseRanking:
        """Parse the judge model's JSON response into a PairwiseRanking.

        Falls back to the first candidate if parsing fails.

        Args:
            raw: Raw judge output (expected JSON).
            model_a: Model ID of candidate A.
            model_b: Model ID of candidate B.

        Returns:
            Parsed ``PairwiseRanking``.
        """
        stripped = raw.strip()
        json_start = stripped.find("{")
        json_end = stripped.rfind("}")

        if json_start >= 0 and json_end > json_start:
            json_str = stripped[json_start : json_end + 1]
            try:
                data = json.loads(json_str)
                winner_label = str(data.get("winner", "A")).upper()
                winner = model_b if winner_label == "B" else model_a
                reason = str(data.get("reason", "No reason provided."))
                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                return PairwiseRanking(
                    candidate_a=model_a,
                    candidate_b=model_b,
                    winner=winner,
                    reason=reason,
                    confidence=confidence,
                )
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                pass

        # Fallback: first candidate wins with zero confidence
        return PairwiseRanking(
            candidate_a=model_a,
            candidate_b=model_b,
            winner=model_a,
            reason="Judge response could not be parsed; defaulting to first candidate.",
            confidence=0.0,
        )

    @staticmethod
    def get_win_rates(rankings: list[PairwiseRanking]) -> dict[str, float]:
        """Compute win rates from pairwise rankings.

        Args:
            rankings: List of pairwise comparison results.

        Returns:
            Dict mapping model_id to win rate (0.0 to 1.0).
        """
        if not rankings:
            return {}

        wins: dict[str, int] = {}
        appearances: dict[str, int] = {}

        for r in rankings:
            for candidate in (r.candidate_a, r.candidate_b):
                wins.setdefault(candidate, 0)
                appearances.setdefault(candidate, 0)
                appearances[candidate] += 1
            wins[r.winner] = wins.get(r.winner, 0) + 1

        return {
            model: wins[model] / appearances[model]
            if appearances[model] > 0
            else 0.0
            for model in appearances
        }

    @staticmethod
    def get_best(rankings: list[PairwiseRanking]) -> str:
        """Return the model_id with the highest win rate.

        Args:
            rankings: List of pairwise comparison results.

        Returns:
            Model ID with the highest win rate.

        Raises:
            ValueError: If rankings is empty.
        """
        if not rankings:
            raise ValueError("Cannot determine best from empty rankings")

        win_rates = OutputRanker.get_win_rates(rankings)
        return max(win_rates, key=lambda m: win_rates[m])


# ------------------------------------------------------------------
# MixtureOfAgents — main MoA pipeline
# ------------------------------------------------------------------


class MixtureOfAgents:
    """Multi-layer parallel generation with pairwise ranking and fusion.

    Layer 1: N proposer models generate independently in parallel.
    Layer 2+: Each model sees ALL outputs from the previous layer, refines.
    Ranking: Pairwise comparison to find best individual output.
    Fusion: Aggregator model synthesizes best elements from all outputs.

    This implements the Mixture-of-Agents architecture (Wang et al., 2024)
    combined with LLM-Blender-style pairwise ranking (Jiang et al., 2023).

    Args:
        engine: Completion engine for all LLM calls (must have a mock backend
            injected for testing).
        model_pool: Model pool for selecting proposers and judge/fusion models.
        config: MoA pipeline configuration.  If ``None``, defaults are used.
    """

    def __init__(
        self,
        engine: CompletionEngine,
        model_pool: ModelPool,
        config: MoAConfig | None = None,
    ) -> None:
        """Initialise the Mixture-of-Agents pipeline.

        Args:
            engine: Completion engine (mock-safe).
            model_pool: For selecting proposer, judge, and fusion models.
            config: Pipeline configuration.
        """
        self._engine = engine
        self._model_pool = model_pool
        self._config = config or MoAConfig()

    async def generate(self, prompt: str, context: str = "") -> MoAResult:
        """Execute the full MoA pipeline.

        Steps:
            1. Select diverse proposer models from different providers.
            2. Run ``num_layers`` generation layers (layer 1 is independent,
               subsequent layers refine from previous outputs).
            3. Optionally rank final-layer outputs via pairwise comparison.
            4. Fuse all outputs into a single superior response.

        Args:
            prompt: The user's original prompt/query.
            context: Optional additional context to include.

        Returns:
            ``MoAResult`` with all layers, rankings, and the fused output.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If no proposer models are available.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty")

        proposers = self._select_proposers()
        if not proposers:
            msg = "No proposer models available"
            raise RuntimeError(msg)

        all_models: set[str] = set(proposers)
        layers: list[MoALayer] = []
        total_cost = 0.0
        previous_outputs: dict[str, str] | None = None

        # Run MoA layers
        for layer_num in range(self._config.num_layers):
            layer = await self._run_layer(
                layer_num=layer_num,
                prompt=prompt,
                previous_outputs=previous_outputs,
                context=context,
            )
            layers.append(layer)
            total_cost += layer.cost
            previous_outputs = layer.outputs
            all_models.update(layer.models)

            logger.info(
                "moa_layer_complete",
                layer=layer_num,
                models=len(layer.models),
                outputs=len(layer.outputs),
                cost=layer.cost,
            )

        # Get final layer outputs for ranking and fusion
        final_outputs = layers[-1].outputs if layers else {}

        # Pairwise ranking (LLM-Blender)
        rankings: list[PairwiseRanking] = []
        best_individual: str | None = None
        if self._config.use_ranking and len(final_outputs) >= 2:
            judge_model = self._model_pool.get_review_model()
            all_models.add(judge_model)
            ranker = OutputRanker(self._engine, judge_model)
            rankings = await ranker.rank(final_outputs, prompt)
            best_individual = ranker.get_best(rankings)

            # Estimate ranking cost (judge calls)
            ranking_cost = len(rankings) * 0.001  # Approximate
            total_cost += ranking_cost

            logger.info(
                "moa_ranking_complete",
                comparisons=len(rankings),
                best=best_individual,
                judge=judge_model,
            )

        # Fusion
        fused_output = await self._fuse(prompt, final_outputs, rankings)
        fusion_model = self._config.fusion_model or self._model_pool.get_planning_model()
        all_models.add(fusion_model)
        total_cost += 0.002  # Approximate fusion cost

        logger.info(
            "moa_pipeline_complete",
            layers=len(layers),
            total_cost=total_cost,
            participating_models=len(all_models),
        )

        return MoAResult(
            layers=layers,
            rankings=rankings,
            final_output=fused_output,
            best_individual=best_individual,
            total_cost=total_cost,
            participating_models=sorted(all_models),
        )

    async def _run_layer(
        self,
        layer_num: int,
        prompt: str,
        previous_outputs: dict[str, str] | None,
        context: str,
    ) -> MoALayer:
        """Execute a single MoA layer with parallel generation.

        Layer 0 generates independently.  Subsequent layers include all
        outputs from the previous layer so each model can see and improve
        upon others' work.

        Args:
            layer_num: Zero-indexed layer number.
            prompt: Original user prompt.
            previous_outputs: Outputs from the previous layer, or ``None``
                for the first layer.
            context: Additional context string.

        Returns:
            Completed ``MoALayer`` with outputs from all proposers.
        """
        proposers = self._select_proposers()
        coros = [
            self._generate_one(model, prompt, previous_outputs, context)
            for model in proposers
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        outputs: dict[str, str] = {}
        layer_cost = 0.0

        for model, result in zip(proposers, results, strict=False):
            if isinstance(result, Exception):
                logger.warning(
                    "moa_generation_failed",
                    layer=layer_num,
                    model=model,
                    error=str(result),
                )
                outputs[model] = f"[Generation failed: {result}]"
            else:
                content, cost = result
                outputs[model] = content
                layer_cost += cost

        return MoALayer(
            layer_number=layer_num,
            models=list(proposers),
            outputs=outputs,
            cost=layer_cost,
        )

    async def _generate_one(
        self,
        model: str,
        prompt: str,
        previous_outputs: dict[str, str] | None,
        context: str,
    ) -> tuple[str, float]:
        """Generate a single response from one model.

        For the first layer (no previous outputs), the model generates
        independently.  For subsequent layers, the model sees all outputs
        from the previous layer to inform its refinement.

        Args:
            model: LiteLLM model identifier.
            prompt: Original user prompt.
            previous_outputs: Outputs from the previous layer, or ``None``.
            context: Additional context string.

        Returns:
            Tuple of (output_text, cost_usd).
        """
        if previous_outputs:
            # Subsequent layer: include previous outputs for refinement
            formatted_outputs = "\n\n".join(
                f"### Model: {mid}\n{output}"
                for mid, output in previous_outputs.items()
            )
            system = _REFINE_SYSTEM.format(previous_outputs=formatted_outputs)
        else:
            system = _PROPOSER_SYSTEM

        user_content = prompt
        if context:
            user_content = f"{prompt}\n\nAdditional context:\n{context}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        result = await self._engine.complete(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=2048,
        )

        return result.content, result.cost_usd

    async def _rank_outputs(
        self,
        outputs: dict[str, str],
        prompt: str,
    ) -> list[PairwiseRanking]:
        """Rank outputs via pairwise comparison using a judge model.

        This is a convenience wrapper around ``OutputRanker.rank()``.

        Args:
            outputs: Mapping of model_id to output text.
            prompt: Original user prompt.

        Returns:
            List of pairwise ranking results.
        """
        if len(outputs) < 2:
            return []
        judge_model = self._model_pool.get_review_model()
        ranker = OutputRanker(self._engine, judge_model)
        return await ranker.rank(outputs, prompt)

    async def _fuse(
        self,
        prompt: str,
        outputs: dict[str, str],
        rankings: list[PairwiseRanking],
    ) -> str:
        """Synthesize a single superior response from all outputs.

        The fusion prompt includes ranking information (if available) so
        the synthesizer knows which outputs were judged highest quality,
        and all individual outputs for reference.

        Args:
            prompt: Original user prompt.
            outputs: All model outputs from the final layer.
            rankings: Pairwise ranking results (may be empty).

        Returns:
            The fused/synthesized response text.
        """
        if not outputs:
            return ""

        # If only one output, return it directly (no fusion needed)
        if len(outputs) == 1:
            return next(iter(outputs.values()))

        # Build ranking context
        ranking_context = ""
        if rankings:
            win_rates = OutputRanker.get_win_rates(rankings)
            sorted_models = sorted(win_rates, key=lambda m: win_rates[m], reverse=True)
            ranking_lines = [
                f"  {i + 1}. {mid} (win rate: {win_rates[mid]:.0%})"
                for i, mid in enumerate(sorted_models)
            ]
            ranking_context = (
                "Quality ranking (from pairwise comparisons):\n"
                + "\n".join(ranking_lines)
            )

        # Build model outputs section
        output_lines = "\n\n".join(
            f"### {mid}\n{text}" for mid, text in outputs.items()
        )

        system = _FUSION_SYSTEM.format(
            ranking_context=ranking_context or "No ranking information available.",
            model_outputs=output_lines,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        fusion_model = self._config.fusion_model or self._model_pool.get_planning_model()

        result = await self._engine.complete(
            messages=messages,
            model=fusion_model,
            temperature=0.3,
            max_tokens=4096,
        )

        logger.info(
            "moa_fusion_complete",
            fusion_model=fusion_model,
            output_length=len(result.content),
        )
        return result.content

    def _select_proposers(self) -> list[str]:
        """Select diverse proposer models from different providers.

        Aims to pick models from as many different providers as possible
        for maximum output diversity.  Falls back to duplicate providers
        if fewer unique providers are available than ``num_proposers``.

        Returns:
            List of LiteLLM model identifiers for proposer generation.

        Raises:
            RuntimeError: If no models are available at all.
        """
        num = self._config.num_proposers

        # Gather all available models from the registry
        all_models = self._model_pool._registry.get_available_models()
        if not all_models:
            msg = "No models available for MoA proposer selection"
            raise RuntimeError(msg)

        # Group by provider for diversity
        by_provider: dict[str, list[Any]] = {}
        for model in all_models:
            by_provider.setdefault(model.provider, []).append(model)

        # Round-robin across providers to maximise diversity
        selected: list[str] = []
        providers = list(by_provider.keys())
        provider_indices: dict[str, int] = {p: 0 for p in providers}
        provider_cycle = itertools.cycle(providers)

        while len(selected) < num:
            provider = next(provider_cycle)
            idx = provider_indices[provider]
            models = by_provider[provider]
            if idx < len(models):
                model_id = models[idx].id
                if model_id not in selected:
                    selected.append(model_id)
                provider_indices[provider] = idx + 1
            # This provider is exhausted; check if all are
            elif all(
                provider_indices[p] >= len(by_provider[p])
                for p in providers
            ):
                # All providers exhausted; allow duplicates from the start
                if selected:
                    break
                msg = "No models available for MoA proposer selection"
                raise RuntimeError(msg)

        logger.info(
            "moa_proposers_selected",
            count=len(selected),
            models=selected,
        )
        return selected

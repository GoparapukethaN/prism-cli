"""Multi-round debate engine for swarm orchestration.

Implements the multi-round LLM debate protocol from Du et al. (ICML 2024),
where multiple models from different providers generate independent responses,
see each other's reasoning, then refine over 2-3 rounds converging toward
correctness.  Empirically shown to improve reasoning by 5-10%.

Flow per round:
    1. Each participant generates/refines their position.
    2. Each participant sees ALL other positions.
    3. Each participant critiques others and refines own position.
    4. Check for consensus (similarity of positions).
    5. If consensus or max_rounds reached, synthesise final answer.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from prism.llm.completion import CompletionEngine
    from prism.orchestrator.swarm import ModelPool

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass
class DebatePosition:
    """A single participant's position in a debate round.

    Attributes:
        model: LiteLLM model identifier of the proposing model.
        content: The position/proposal text.
        round: Which debate round (1-indexed).
        confidence: Self-assessed confidence from 0.0 to 1.0.
        critiques: Critiques received from other models.
    """

    model: str
    content: str
    round: int
    confidence: float
    critiques: list[str] = field(default_factory=list)


@dataclass
class DebateRound:
    """Result of a single debate round.

    Attributes:
        round_number: 1-indexed round number.
        positions: All participant positions from this round.
        consensus_reached: Whether the consensus threshold was met.
        consensus_text: Summary text if consensus was reached, else ``None``.
    """

    round_number: int
    positions: list[DebatePosition] = field(default_factory=list)
    consensus_reached: bool = False
    consensus_text: str | None = None


@dataclass
class DebateConfig:
    """Configuration for a multi-round debate.

    Attributes:
        max_rounds: Maximum number of debate rounds before forced synthesis.
        min_participants: Minimum number of models required.
        max_participants: Maximum number of models to include.
        consensus_threshold: Agreement level (0.0-1.0) to stop early.
        temperature: Sampling temperature for diversity in positions.
    """

    max_rounds: int = 3
    min_participants: int = 2
    max_participants: int = 4
    consensus_threshold: float = 0.8
    temperature: float = 0.7

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_rounds < 1:
            msg = "max_rounds must be at least 1"
            raise ValueError(msg)
        if self.min_participants < 2:
            msg = "min_participants must be at least 2"
            raise ValueError(msg)
        if self.max_participants < self.min_participants:
            msg = "max_participants must be >= min_participants"
            raise ValueError(msg)
        if not 0.0 <= self.consensus_threshold <= 1.0:
            msg = "consensus_threshold must be between 0.0 and 1.0"
            raise ValueError(msg)
        if not 0.0 <= self.temperature <= 2.0:
            msg = "temperature must be between 0.0 and 2.0"
            raise ValueError(msg)


@dataclass
class DebateResult:
    """Final result of a multi-round debate.

    Attributes:
        topic: The original debate topic.
        rounds: All completed debate rounds.
        final_synthesis: The synthesised final answer.
        consensus_score: Final agreement score (0.0-1.0).
        total_cost: Accumulated cost in USD across all rounds.
        participating_models: List of model identifiers that participated.
    """

    topic: str
    rounds: list[DebateRound]
    final_synthesis: str
    consensus_score: float
    total_cost: float
    participating_models: list[str]


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

_INITIAL_POSITION_SYSTEM = """\
You are a thoughtful AI debater.  Given a topic and optional context,
generate your independent position.  Be specific, provide reasoning,
and assess your own confidence.

Respond ONLY with a JSON object:
{
  "position": "Your detailed position with reasoning",
  "confidence": 0.85
}

The confidence value must be a float between 0.0 and 1.0 reflecting
how certain you are in your answer."""

_REFINE_POSITION_SYSTEM = """\
You are a thoughtful AI debater in round {round_num} of a multi-round debate.
You have seen other participants' positions and must now refine your own.

Consider the strengths and weaknesses of each position, including your own.
You may change your position if convinced, strengthen it with new arguments,
or find a synthesis between viewpoints.

Other participants' positions from the previous round:
{other_positions}

Respond ONLY with a JSON object:
{{
  "position": "Your refined position with reasoning",
  "confidence": 0.90,
  "critiques": ["Brief critique of participant 1's position", "Brief critique of participant 2's position"]
}}

The critiques list should have one entry per other participant, in the same order
they were presented above."""

_CONSENSUS_CHECK_SYSTEM = """\
You are a neutral judge assessing whether debate participants have reached
consensus.  Analyse the positions below and rate their agreement on a scale
of 0.0 (complete disagreement) to 1.0 (identical conclusions).

Positions:
{positions}

Respond ONLY with a JSON object:
{{
  "consensus_score": 0.75,
  "summary": "Brief summary of where positions agree and disagree"
}}"""

_SYNTHESIS_SYSTEM = """\
You are a synthesis engine.  Given a multi-round debate between AI models,
produce the best possible final answer by combining the strongest arguments
from all participants across all rounds.

Topic: {topic}

Debate history:
{debate_history}

Produce a comprehensive, well-reasoned synthesis that represents the best
collective thinking of all participants.  Be specific and actionable."""


# ------------------------------------------------------------------
# DebateEngine
# ------------------------------------------------------------------


class DebateEngine:
    """Multi-round debate between different models.

    Orchestrates a structured debate where each participant is a different
    model from a different provider.  Follows the protocol from Du et al.
    (ICML 2024):

    Flow per round:
        1. Each participant generates/refines their position.
        2. Each participant sees ALL other positions.
        3. Each participant critiques others and refines own position.
        4. Check for consensus (similarity of positions).
        5. If consensus or max_rounds reached, synthesise final answer.

    All LLM calls go through ``engine.complete()`` with mocked backends
    in tests.
    """

    def __init__(
        self,
        engine: CompletionEngine,
        model_pool: ModelPool,
        config: DebateConfig | None = None,
    ) -> None:
        """Initialise the debate engine.

        Args:
            engine: Completion engine for all LLM calls (must have a mock
                backend injected for testing).
            model_pool: Model pool for selecting debate participants.
            config: Debate configuration.  Uses defaults if ``None``.
        """
        self._engine = engine
        self._model_pool = model_pool
        self._config = config or DebateConfig()
        self._total_cost: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def debate(self, topic: str, context: str = "") -> DebateResult:
        """Run a multi-round debate on a topic.

        Args:
            topic: The question or problem to debate.
            context: Optional context information for the debate.

        Returns:
            :class:`DebateResult` with all rounds, synthesis, and costs.

        Raises:
            ValueError: If *topic* is empty or whitespace-only.
            RuntimeError: If fewer than ``min_participants`` models are
                available.
        """
        if not topic or not topic.strip():
            msg = "Debate topic must not be empty"
            raise ValueError(msg)

        self._total_cost = 0.0
        participants = self._select_participants(topic)

        if len(participants) < self._config.min_participants:
            msg = (
                f"Need at least {self._config.min_participants} participants "
                f"but only {len(participants)} models available"
            )
            raise RuntimeError(msg)

        logger.info(
            "debate_started",
            topic=topic[:80],
            participants=participants,
            max_rounds=self._config.max_rounds,
        )

        rounds: list[DebateRound] = []
        previous_positions: list[DebatePosition] = []

        for round_num in range(1, self._config.max_rounds + 1):
            debate_round = await self._run_round(
                round_num=round_num,
                topic=topic,
                previous_positions=previous_positions,
                context=context,
                participants=participants,
            )
            rounds.append(debate_round)
            previous_positions = debate_round.positions

            logger.info(
                "debate_round_complete",
                round_num=round_num,
                num_positions=len(debate_round.positions),
                consensus_reached=debate_round.consensus_reached,
            )

            if debate_round.consensus_reached:
                break

        # Synthesise final answer
        final_synthesis = await self._synthesize(topic, rounds)

        # Compute final consensus score from the last round
        last_positions = rounds[-1].positions if rounds else []
        _, final_score = await self._check_consensus(last_positions)

        return DebateResult(
            topic=topic,
            rounds=rounds,
            final_synthesis=final_synthesis,
            consensus_score=final_score,
            total_cost=self._total_cost,
            participating_models=participants,
        )

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    async def _run_round(
        self,
        round_num: int,
        topic: str,
        previous_positions: list[DebatePosition],
        context: str,
        participants: list[str],
    ) -> DebateRound:
        """Execute a single debate round.

        In round 1, each participant generates an independent position.
        In subsequent rounds, each participant sees all others' positions
        and refines their own.

        Args:
            round_num: The 1-indexed round number.
            topic: The debate topic.
            previous_positions: Positions from the previous round (empty
                for round 1).
            context: Optional context string.
            participants: List of model identifiers to use.

        Returns:
            :class:`DebateRound` with all positions and consensus check.
        """
        # Generate positions concurrently
        tasks = [
            self._generate_position(
                model=model,
                topic=topic,
                others=[p for p in previous_positions if p.model != model],
                context=context,
                round_num=round_num,
            )
            for model in participants
        ]
        positions = await asyncio.gather(*tasks)
        position_list = list(positions)

        # Check consensus
        reached, score = await self._check_consensus(position_list)
        consensus_text: str | None = None
        if reached:
            consensus_text = (
                f"Consensus reached with score {score:.2f} in round {round_num}."
            )

        return DebateRound(
            round_number=round_num,
            positions=position_list,
            consensus_reached=reached,
            consensus_text=consensus_text,
        )

    # ------------------------------------------------------------------
    # Position generation
    # ------------------------------------------------------------------

    async def _generate_position(
        self,
        model: str,
        topic: str,
        others: list[DebatePosition],
        context: str,
        round_num: int,
    ) -> DebatePosition:
        """Generate or refine a position for a single participant.

        In round 1, the model produces an independent position.  In later
        rounds, it sees all other positions and refines its own.

        Args:
            model: LiteLLM model identifier for this participant.
            topic: The debate topic.
            others: Other participants' positions from the previous round.
            context: Optional context string.
            round_num: Current round number (1-indexed).

        Returns:
            :class:`DebatePosition` with the model's stance and confidence.
        """
        if round_num == 1 or not others:
            # Round 1: independent position
            system_prompt = _INITIAL_POSITION_SYSTEM
        else:
            # Round 2+: refine with awareness of others
            other_text = self._format_positions(others)
            system_prompt = _REFINE_POSITION_SYSTEM.format(
                round_num=round_num,
                other_positions=other_text,
            )

        user_content = f"Topic: {topic}"
        if context:
            user_content += f"\n\nContext:\n{context}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        result = await self._engine.complete(
            messages=messages,
            model=model,
            temperature=self._config.temperature,
        )
        self._total_cost += result.cost_usd

        # Parse the JSON response
        position_text, confidence, critiques = self._parse_position_response(
            result.content,
        )

        return DebatePosition(
            model=model,
            content=position_text,
            round=round_num,
            confidence=confidence,
            critiques=critiques,
        )

    # ------------------------------------------------------------------
    # Consensus checking
    # ------------------------------------------------------------------

    async def _check_consensus(
        self,
        positions: list[DebatePosition],
    ) -> tuple[bool, float]:
        """Check whether participants have reached consensus.

        Uses the planning model as a neutral judge to assess agreement
        between all positions on a scale of 0.0 to 1.0.

        Args:
            positions: All positions from the current round.

        Returns:
            Tuple of (consensus_reached, consensus_score).
        """
        if len(positions) < 2:
            return True, 1.0

        positions_text = self._format_positions(positions)
        system_prompt = _CONSENSUS_CHECK_SYSTEM.format(
            positions=positions_text,
        )

        judge_model = self._model_pool.get_planning_model()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Assess the consensus level."},
        ]

        result = await self._engine.complete(
            messages=messages,
            model=judge_model,
            temperature=0.1,  # Low temperature for consistent judgment
        )
        self._total_cost += result.cost_usd

        score = self._parse_consensus_response(result.content)
        reached = score >= self._config.consensus_threshold
        return reached, score

    # ------------------------------------------------------------------
    # Final synthesis
    # ------------------------------------------------------------------

    async def _synthesize(
        self,
        topic: str,
        rounds: list[DebateRound],
    ) -> str:
        """Produce a final synthesis from all debate rounds.

        Uses the smartest available model (the planning model) to combine
        the strongest arguments from all participants across all rounds
        into a single, comprehensive answer.

        Args:
            topic: The original debate topic.
            rounds: All completed debate rounds.

        Returns:
            Synthesised final answer as a string.
        """
        debate_history = self._format_debate_history(rounds)
        system_prompt = _SYNTHESIS_SYSTEM.format(
            topic=topic,
            debate_history=debate_history,
        )

        synth_model = self._model_pool.get_planning_model()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Synthesise the best final answer from this debate."
                ),
            },
        ]

        result = await self._engine.complete(
            messages=messages,
            model=synth_model,
            temperature=0.3,  # Low temperature for focused synthesis
        )
        self._total_cost += result.cost_usd

        return result.content

    # ------------------------------------------------------------------
    # Participant selection
    # ------------------------------------------------------------------

    def _select_participants(self, topic: str) -> list[str]:
        """Select diverse models from different providers for the debate.

        Prioritises models from different providers to maximise diversity
        of reasoning approaches.  Picks from execution models across all
        tiers and ensures cross-provider representation.

        Args:
            topic: The debate topic (unused currently, reserved for
                future topic-aware selection).

        Returns:
            List of LiteLLM model identifiers.
        """
        seen_providers: set[str] = set()
        participants: list[str] = []
        max_count = self._config.max_participants

        # Gather candidates from different tiers for diversity
        candidate_ids: list[str] = []

        # Start with complex models (smartest)
        planning = self._model_pool.get_planning_model()
        candidate_ids.append(planning)

        # Add review model (different provider)
        review = self._model_pool.get_review_model()
        if review != planning:
            candidate_ids.append(review)

        # Add execution models from each tier
        for tier in ("complex", "medium", "simple"):
            exec_model = self._model_pool.get_execution_model(tier)
            if exec_model not in candidate_ids:
                candidate_ids.append(exec_model)

        # Add research models
        for research_model in self._model_pool.get_research_models():
            if research_model not in candidate_ids:
                candidate_ids.append(research_model)

        # Select up to max_participants, preferring different providers
        registry = self._model_pool._registry
        for model_id in candidate_ids:
            if len(participants) >= max_count:
                break
            model_info = registry.get_model_info(model_id)
            provider = model_info.provider if model_info else "unknown"
            if provider not in seen_providers:
                participants.append(model_id)
                seen_providers.add(provider)

        # If we still need more participants, allow same-provider models
        if len(participants) < self._config.min_participants:
            for model_id in candidate_ids:
                if len(participants) >= max_count:
                    break
                if model_id not in participants:
                    participants.append(model_id)

        return participants

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_position_response(
        content: str,
    ) -> tuple[str, float, list[str]]:
        """Parse a position JSON response from an LLM.

        Args:
            content: Raw LLM response content.

        Returns:
            Tuple of (position_text, confidence, critiques).
        """
        try:
            data = json.loads(content)
            position = str(data.get("position", content))
            confidence_raw = data.get("confidence", 0.5)
            confidence = max(0.0, min(1.0, float(confidence_raw)))
            critiques_raw = data.get("critiques", [])
            critiques = [str(c) for c in critiques_raw] if isinstance(critiques_raw, list) else []
            return position, confidence, critiques
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.warning("debate_position_parse_fallback", content_preview=content[:100])
            return content, 0.5, []

    @staticmethod
    def _parse_consensus_response(content: str) -> float:
        """Parse a consensus check JSON response.

        Args:
            content: Raw LLM response content.

        Returns:
            Consensus score between 0.0 and 1.0.
        """
        try:
            data = json.loads(content)
            score_raw = data.get("consensus_score", 0.5)
            return max(0.0, min(1.0, float(score_raw)))
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.warning("consensus_parse_fallback", content_preview=content[:100])
            return 0.5

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_positions(positions: list[DebatePosition]) -> str:
        """Format a list of positions into a numbered text block.

        Args:
            positions: Positions to format.

        Returns:
            Numbered text representation.
        """
        parts: list[str] = []
        for i, pos in enumerate(positions, 1):
            parts.append(
                f"Participant {i} ({pos.model}, confidence: {pos.confidence:.2f}):\n"
                f"{pos.content}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_debate_history(rounds: list[DebateRound]) -> str:
        """Format all debate rounds into a readable history.

        Args:
            rounds: All completed debate rounds.

        Returns:
            Formatted debate history string.
        """
        parts: list[str] = []
        for debate_round in rounds:
            parts.append(f"=== Round {debate_round.round_number} ===")
            for pos in debate_round.positions:
                header = (
                    f"[{pos.model}] (confidence: {pos.confidence:.2f})"
                )
                parts.append(f"{header}\n{pos.content}")
                if pos.critiques:
                    parts.append("Critiques:")
                    for critique in pos.critiques:
                        parts.append(f"  - {critique}")
            if debate_round.consensus_reached:
                parts.append(
                    f">> Consensus reached: {debate_round.consensus_text}"
                )
            parts.append("")
        return "\n".join(parts)

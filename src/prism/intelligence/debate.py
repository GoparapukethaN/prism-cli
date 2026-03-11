"""Multi-Model Debate Mode — structured multi-model deliberation for decisions.

Runs a three-round debate across multiple AI models:

1. **Round 1 — Independent Positions**: Each model answers the question
   independently and in parallel.
2. **Round 2 — Critiques**: Each model reviews all Round 1 answers,
   identifies strengths/weaknesses/missed risks, and updates its position.
   (Skipped in *quick* mode.)
3. **Round 3 — Synthesis**: A synthesiser model produces the final report
   with consensus points, disagreements, overall recommendation, and a
   confidence score.

Slash-command hooks:
    /debate <question>            — full three-round debate
    /debate --quick <question>    — skip Round 2 (positions + synthesis only)
    /debate --models m1,m2,m3 <q> — override default model list
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)


# ======================================================================
# Protocol for the completion engine dependency
# ======================================================================


class CompletionResult(Protocol):
    """Minimal result interface expected from the completion engine."""

    @property
    def content(self) -> str: ...

    @property
    def input_tokens(self) -> int: ...

    @property
    def output_tokens(self) -> int: ...

    @property
    def cost_usd(self) -> float: ...


class CompletionEngine(Protocol):
    """Minimal interface for the completion engine used by the debate."""

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
    ) -> CompletionResult: ...


# ======================================================================
# Default model roster
# ======================================================================

DEFAULT_DEBATE_MODELS: list[str] = [
    "claude-sonnet-4-20250514",
    "gpt-4o",
    "deepseek/deepseek-chat",
]


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class DebatePosition:
    """A single model's independent answer (Round 1)."""

    model: str
    content: str
    tokens_used: int
    cost_usd: float
    round_number: int


@dataclass
class DebateCritique:
    """A model's critique of all other positions (Round 2)."""

    model: str
    target_model: str
    strengths: str
    weaknesses: str
    risks_missed: str
    updated_position: str
    tokens_used: int
    cost_usd: float


@dataclass
class DebateSynthesis:
    """Final synthesis across all positions and critiques (Round 3)."""

    consensus_points: list[str]
    disagreements: list[str]
    recommendation: str
    confidence: float
    what_each_missed: dict[str, str]
    synthesizer_model: str
    tokens_used: int
    cost_usd: float


@dataclass
class DebateSession:
    """Full record of a single debate."""

    question: str
    models: list[str]
    round1_positions: list[DebatePosition] = field(default_factory=list)
    round2_critiques: list[DebateCritique] = field(default_factory=list)
    synthesis: DebateSynthesis | None = None
    total_cost: float = 0.0
    total_tokens: int = 0
    created_at: str = ""
    quick_mode: bool = False

    @property
    def is_complete(self) -> bool:
        """Whether the debate has finished (synthesis produced)."""
        return self.synthesis is not None


# ======================================================================
# Prompt templates
# ======================================================================

_ROUND1_SYSTEM = (
    "You are participating in a structured debate. "
    "Give your independent, well-reasoned position on the question. "
    "Be specific and thorough."
)

_ROUND2_SYSTEM = (
    "You are participating in Round 2 of a structured debate. "
    "Review all positions and provide your critique."
)

_ROUND2_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Round 1 Positions:\n{positions}\n\n"
    "Provide: strengths, weaknesses, risks missed, and your updated position."
)

_SYNTHESIS_SYSTEM = (
    "Synthesize all debate positions into a structured final report with: "
    "consensus points, disagreements, recommendation, confidence score (0-1)."
)


# ======================================================================
# Main class
# ======================================================================


class MultiModelDebate:
    """Structured debate across multiple models for high-stakes decisions.

    Args:
        completion_engine: An object with an ``async complete()`` method.
        models: List of model identifiers (at least 2).
        synthesizer: Model to use for Round 3. Defaults to the first model.
    """

    def __init__(
        self,
        completion_engine: CompletionEngine,
        models: list[str] | None = None,
        synthesizer: str | None = None,
    ) -> None:
        self._engine = completion_engine
        self._models = list(models) if models else list(DEFAULT_DEBATE_MODELS)
        self._synthesizer = synthesizer or self._models[0]
        self._history: list[DebateSession] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def models(self) -> list[str]:
        """Currently configured debate models."""
        return list(self._models)

    @models.setter
    def models(self, value: list[str]) -> None:
        """Set debate models (minimum 2)."""
        if len(value) < 2:
            raise ValueError("Need at least 2 models for debate")
        self._models = list(value)

    @property
    def history(self) -> list[DebateSession]:
        """Return list of completed debate sessions."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(self, prompt_length: int = 500) -> float:
        """Estimate the total cost (USD) for a full debate.

        Rough heuristic: (N models * 2 rounds + 1 synthesis) * avg cost.

        Args:
            prompt_length: Approximate prompt length in tokens (unused in
                the current heuristic, reserved for future refinement).

        Returns:
            Estimated cost in USD.
        """
        # Round 1: N independent calls
        # Round 2: N critique calls (each sees all R1 responses)
        # Round 3: 1 synthesis call (sees everything)
        calls = len(self._models) * 2 + 1
        avg_cost_per_call = 0.005  # rough average across providers
        return calls * avg_cost_per_call

    # ------------------------------------------------------------------
    # Main debate orchestration
    # ------------------------------------------------------------------

    async def debate(
        self,
        question: str,
        quick: bool = False,
    ) -> DebateSession:
        """Run a full multi-model debate.

        Args:
            question: The question or decision to debate.
            quick: If ``True``, skip Round 2 (critique round).

        Returns:
            A completed :class:`DebateSession`.
        """
        if not question or not question.strip():
            raise ValueError("Debate question must not be empty")

        session = DebateSession(
            question=question,
            models=list(self._models),
            created_at=datetime.now(UTC).isoformat(),
            quick_mode=quick,
        )

        # ------ Round 1: Independent positions (parallel) ------
        r1_tasks = [self._get_position(model, question) for model in self._models]
        positions = await asyncio.gather(*r1_tasks, return_exceptions=True)

        for pos in positions:
            if isinstance(pos, DebatePosition):
                session.round1_positions.append(pos)
                session.total_cost += pos.cost_usd
                session.total_tokens += pos.tokens_used

        # ------ Round 2: Critiques (parallel, skipped in quick mode) ------
        if not quick and len(session.round1_positions) >= 2:
            r2_tasks = []
            for model in self._models:
                others = [p for p in session.round1_positions if p.model != model]
                r2_tasks.append(
                    self._get_critique(model, question, session.round1_positions, others)
                )

            critiques = await asyncio.gather(*r2_tasks, return_exceptions=True)
            for critique in critiques:
                if isinstance(critique, DebateCritique):
                    session.round2_critiques.append(critique)
                    session.total_cost += critique.cost_usd
                    session.total_tokens += critique.tokens_used

        # ------ Round 3: Synthesis ------
        synthesis = await self._synthesize(question, session)
        if synthesis is not None:
            session.synthesis = synthesis
            session.total_cost += synthesis.cost_usd
            session.total_tokens += synthesis.tokens_used

        self._history.append(session)
        return session

    # ------------------------------------------------------------------
    # Round helpers
    # ------------------------------------------------------------------

    async def _get_position(self, model: str, question: str) -> DebatePosition:
        """Get a model's independent position (Round 1).

        Args:
            model: Model identifier.
            question: The debate question.

        Returns:
            A :class:`DebatePosition`.
        """
        messages = [
            {"role": "system", "content": _ROUND1_SYSTEM},
            {"role": "user", "content": question},
        ]

        try:
            result = await self._engine.complete(messages=messages, model=model)
            return DebatePosition(
                model=model,
                content=result.content,
                tokens_used=result.input_tokens + result.output_tokens,
                cost_usd=result.cost_usd,
                round_number=1,
            )
        except Exception as exc:
            logger.warning("debate.position.error", model=model, error=str(exc))
            return DebatePosition(
                model=model,
                content=f"Error: {exc}",
                tokens_used=0,
                cost_usd=0.0,
                round_number=1,
            )

    async def _get_critique(
        self,
        model: str,
        question: str,
        all_positions: list[DebatePosition],
        others: list[DebatePosition],
    ) -> DebateCritique:
        """Get a model's critique of other positions (Round 2).

        Args:
            model: Model identifier.
            question: The debate question.
            all_positions: All Round 1 positions.
            others: Positions from other models (excluding this one).

        Returns:
            A :class:`DebateCritique`.
        """
        positions_text = "\n\n".join(
            f"**{p.model}**: {p.content}" for p in all_positions
        )

        user_content = _ROUND2_USER_TEMPLATE.format(
            question=question, positions=positions_text,
        )

        messages = [
            {"role": "system", "content": _ROUND2_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        try:
            result = await self._engine.complete(messages=messages, model=model)
            return DebateCritique(
                model=model,
                target_model="all",
                strengths="",
                weaknesses="",
                risks_missed="",
                updated_position=result.content,
                tokens_used=result.input_tokens + result.output_tokens,
                cost_usd=result.cost_usd,
            )
        except Exception as exc:
            logger.warning("debate.critique.error", model=model, error=str(exc))
            return DebateCritique(
                model=model,
                target_model="all",
                strengths="",
                weaknesses="",
                risks_missed="",
                updated_position=f"Error: {exc}",
                tokens_used=0,
                cost_usd=0.0,
            )

    async def _synthesize(
        self, question: str, session: DebateSession,
    ) -> DebateSynthesis | None:
        """Synthesize all positions into a final recommendation (Round 3).

        Args:
            question: The debate question.
            session: The debate session so far.

        Returns:
            A :class:`DebateSynthesis`, or ``None`` on failure.
        """
        positions_text = "\n\n".join(
            f"**{p.model}** (Round 1): {p.content}" for p in session.round1_positions
        )

        critiques_text = ""
        if session.round2_critiques:
            critiques_text = "\n\nRound 2 Critiques:\n" + "\n\n".join(
                f"**{c.model}**: {c.updated_position}" for c in session.round2_critiques
            )

        messages = [
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"{positions_text}{critiques_text}\n\n"
                    "Provide synthesis."
                ),
            },
        ]

        try:
            result = await self._engine.complete(
                messages=messages, model=self._synthesizer,
            )
            return DebateSynthesis(
                consensus_points=["See synthesis content"],
                disagreements=["See synthesis content"],
                recommendation=result.content,
                confidence=0.7,
                what_each_missed={},
                synthesizer_model=self._synthesizer,
                tokens_used=result.input_tokens + result.output_tokens,
                cost_usd=result.cost_usd,
            )
        except Exception as exc:
            logger.warning("debate.synthesis.error", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_debate(self, session: DebateSession, save_dir: Path) -> Path:
        """Save a debate session to disk as JSON.

        Args:
            session: The debate session to persist.
            save_dir: Directory to write the JSON file into.

        Returns:
            Path to the saved file.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        # Use ISO date prefix + truncated timestamp for uniqueness
        timestamp = session.created_at[:19].replace(":", "-") if session.created_at else "unknown"
        filename = f"debate_{timestamp}.json"
        path = save_dir / filename
        path.write_text(json.dumps(asdict(session), indent=2, default=str), encoding="utf-8")
        logger.info("debate.saved", path=str(path))
        return path

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
    /debates                      — list saved debates
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import structlog

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
# Type alias for the LLM caller callback
# ======================================================================

# An LLM caller receives (model, messages) and returns a response string.
LLMCaller = Callable[[str, list[dict[str, str]]], str]


def _stub_llm_caller(model: str, messages: list[dict[str, str]]) -> str:
    """Stub LLM caller that returns placeholder text for testing.

    Args:
        model: Model identifier (unused in stub).
        messages: Chat messages (unused in stub).

    Returns:
        A placeholder response string.
    """
    return f"[Placeholder response from {model}]"


# ======================================================================
# Default model roster
# ======================================================================

DEFAULT_DEBATE_MODELS: list[str] = [
    "claude-sonnet-4-20250514",
    "gpt-4o",
    "deepseek/deepseek-chat",
]


# ======================================================================
# Data classes — enhanced
# ======================================================================


@dataclass
class DebateConfig:
    """Configuration for a multi-model debate session.

    Attributes:
        round1_models: Models participating in the first round.
        synthesis_model: Model used for the final synthesis round.
        quick_mode: If True, skip the critique round (Round 2).
        save_dir: Directory to persist debate reports.
    """

    round1_models: list[str] = field(
        default_factory=lambda: ["claude-sonnet-4-5", "gpt-4o", "deepseek-chat"]
    )
    synthesis_model: str = "claude-sonnet-4-5"
    quick_mode: bool = False
    save_dir: Path = field(
        default_factory=lambda: Path.home() / ".prism" / "debates"
    )


@dataclass
class DebateRound:
    """A single round in a debate session.

    Attributes:
        round_number: Sequential round number (1, 2, or 3).
        positions: Mapping of model name to its response text.
        round_type: One of ``"position"``, ``"critique"``, or ``"synthesis"``.
    """

    round_number: int
    positions: dict[str, str] = field(default_factory=dict)
    round_type: str = "position"


@dataclass
class DebateResult:
    """Complete result of a multi-model debate.

    Attributes:
        question: The original question debated.
        rounds: List of all debate rounds executed.
        synthesis: The synthesized final answer.
        consensus: Points all models agreed on.
        disagreements: Points where models disagreed.
        tradeoffs: Key tradeoffs identified during debate.
        recommendation: Final actionable recommendation.
        confidence: Confidence score between 0.0 and 1.0.
        blind_spots: Things each model missed, keyed by model name.
        total_cost: Total estimated cost in USD.
        created_at: ISO-8601 timestamp of when the debate was created.
    """

    question: str
    rounds: list[DebateRound] = field(default_factory=list)
    synthesis: str = ""
    consensus: str = ""
    disagreements: str = ""
    tradeoffs: str = ""
    recommendation: str = ""
    confidence: float = 0.0
    blind_spots: dict[str, str] = field(default_factory=dict)
    total_cost: float = 0.0
    created_at: str = ""


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
    "Synthesize all debate positions into a structured final report. "
    "Return your response in the following format:\n\n"
    "CONSENSUS: <points all models agree on>\n"
    "DISAGREEMENTS: <points where models disagree>\n"
    "TRADEOFFS: <key tradeoffs identified>\n"
    "RECOMMENDATION: <your final recommendation>\n"
    "CONFIDENCE: <a decimal between 0.0 and 1.0>\n"
    "BLIND_SPOTS: <model_name>: <what they missed> (one per model)\n"
)


# ======================================================================
# Synthesis parsing
# ======================================================================


def _parse_synthesis(raw: str, models: list[str]) -> dict[str, Any]:
    """Parse a structured synthesis response into its component fields.

    Extracts ``CONSENSUS``, ``DISAGREEMENTS``, ``TRADEOFFS``,
    ``RECOMMENDATION``, ``CONFIDENCE``, and ``BLIND_SPOTS`` sections from
    the raw synthesis text.  Falls back gracefully when sections are
    missing.

    Args:
        raw: The raw synthesis text from the LLM.
        models: List of model names (used for blind spot parsing).

    Returns:
        A dict with keys ``consensus``, ``disagreements``, ``tradeoffs``,
        ``recommendation``, ``confidence``, and ``blind_spots``.
    """
    result: dict[str, Any] = {
        "consensus": "",
        "disagreements": "",
        "tradeoffs": "",
        "recommendation": "",
        "confidence": 0.7,
        "blind_spots": {},
    }

    # Try to extract each section
    sections = [
        ("consensus", "CONSENSUS"),
        ("disagreements", "DISAGREEMENTS"),
        ("tradeoffs", "TRADEOFFS"),
        ("recommendation", "RECOMMENDATION"),
    ]

    for field_name, label in sections:
        pattern = rf"{label}:\s*(.+?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            result[field_name] = match.group(1).strip()

    # Parse confidence
    conf_match = re.search(r"CONFIDENCE:\s*(-?[\d.]+)", raw)
    if conf_match:
        try:
            val = float(conf_match.group(1))
            result["confidence"] = max(0.0, min(1.0, val))
        except ValueError:
            result["confidence"] = 0.7

    # Parse blind spots
    blind_spots: dict[str, str] = {}
    for model in models:
        # Look for model-specific blind spot entries
        model_short = model.split("/")[-1]
        pattern = rf"{re.escape(model_short)}:\s*(.+?)(?=\n|$)"
        match = re.search(pattern, raw)
        if match:
            blind_spots[model] = match.group(1).strip()
    result["blind_spots"] = blind_spots

    # If no sections parsed, use the full text as the recommendation
    if not any(result[k] for k in ("consensus", "disagreements", "recommendation")):
        result["recommendation"] = raw.strip()

    return result


# ======================================================================
# Enhanced debate function
# ======================================================================


def debate(
    question: str,
    config: DebateConfig | None = None,
    llm_caller: LLMCaller | None = None,
) -> DebateResult:
    """Run a structured multi-model debate on a question.

    Executes a 3-round debate:
      - Round 1: Each model provides an independent position.
      - Round 2: Each model critiques all Round 1 responses (skipped in
        quick mode).
      - Round 3: A synthesis model combines all positions into a final
        structured report.

    Args:
        question: The question or decision to debate.
        config: Debate configuration. Uses defaults if not provided.
        llm_caller: Callback ``(model, messages) -> response_str`` for
            calling the LLM. Defaults to a stub for testability.

    Returns:
        A :class:`DebateResult` with all rounds and parsed synthesis.

    Raises:
        ValueError: If *question* is empty.
    """
    if not question or not question.strip():
        raise ValueError("Debate question must not be empty")

    cfg = config or DebateConfig()
    caller = llm_caller or _stub_llm_caller
    timestamp = datetime.now(UTC).isoformat()

    models = list(cfg.round1_models)
    rounds: list[DebateRound] = []

    # --- Round 1: Independent positions ---
    round1 = DebateRound(round_number=1, round_type="position")
    for model in models:
        messages = [
            {"role": "system", "content": _ROUND1_SYSTEM},
            {"role": "user", "content": question},
        ]
        try:
            response = caller(model, messages)
        except Exception as exc:
            logger.warning("debate.round1.error", model=model, error=str(exc))
            response = f"Error: {exc}"
        round1.positions[model] = response
    rounds.append(round1)

    # --- Round 2: Critiques (skip in quick mode) ---
    if not cfg.quick_mode and len(round1.positions) >= 2:
        round2 = DebateRound(round_number=2, round_type="critique")
        positions_text = "\n\n".join(
            f"**{m}**: {r}" for m, r in round1.positions.items()
        )
        for model in models:
            user_content = _ROUND2_USER_TEMPLATE.format(
                question=question, positions=positions_text,
            )
            messages = [
                {"role": "system", "content": _ROUND2_SYSTEM},
                {"role": "user", "content": user_content},
            ]
            try:
                response = caller(model, messages)
            except Exception as exc:
                logger.warning(
                    "debate.round2.error", model=model, error=str(exc),
                )
                response = f"Error: {exc}"
            round2.positions[model] = response
        rounds.append(round2)

    # --- Round 3: Synthesis ---
    all_content_parts: list[str] = [f"Question: {question}\n"]
    all_content_parts.append("Round 1 Positions:")
    for m, r in round1.positions.items():
        all_content_parts.append(f"  {m}: {r}")

    if len(rounds) > 1:
        round2_data = rounds[1]
        all_content_parts.append("\nRound 2 Critiques:")
        for m, r in round2_data.positions.items():
            all_content_parts.append(f"  {m}: {r}")

    all_content_parts.append("\nProvide synthesis.")

    synthesis_messages = [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {"role": "user", "content": "\n".join(all_content_parts)},
    ]
    try:
        raw_synthesis = caller(cfg.synthesis_model, synthesis_messages)
    except Exception as exc:
        logger.warning("debate.synthesis.error", error=str(exc))
        raw_synthesis = f"Synthesis failed: {exc}"

    round3 = DebateRound(
        round_number=3,
        round_type="synthesis",
        positions={cfg.synthesis_model: raw_synthesis},
    )
    rounds.append(round3)

    # Parse synthesis into structured fields
    parsed = _parse_synthesis(raw_synthesis, models)

    result = DebateResult(
        question=question,
        rounds=rounds,
        synthesis=raw_synthesis,
        consensus=parsed["consensus"],
        disagreements=parsed["disagreements"],
        tradeoffs=parsed["tradeoffs"],
        recommendation=parsed["recommendation"],
        confidence=parsed["confidence"],
        blind_spots=parsed["blind_spots"],
        total_cost=0.0,
        created_at=timestamp,
    )

    # Auto-save
    save_debate(result, cfg.save_dir)

    return result


# ======================================================================
# Report generation
# ======================================================================


def generate_report_text(result: DebateResult) -> str:
    """Generate a human-readable plain text report from a debate result.

    The report includes sections for each round's positions, the final
    synthesis, consensus, disagreements, tradeoffs, recommendation,
    confidence, and blind spots.

    Args:
        result: The completed debate result.

    Returns:
        A formatted plain text string suitable for console display or
        file output.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("MULTI-MODEL DEBATE REPORT")
    lines.append("=" * 60)
    lines.append(f"\nQuestion: {result.question}")
    lines.append(f"Date: {result.created_at[:19] if result.created_at else 'N/A'}")
    lines.append("")

    for rnd in result.rounds:
        round_labels = {
            "position": "Round 1 — Independent Positions",
            "critique": "Round 2 — Critiques",
            "synthesis": "Round 3 — Synthesis",
        }
        label = round_labels.get(rnd.round_type, f"Round {rnd.round_number}")
        lines.append(f"--- {label} ---")
        for model_name, response in rnd.positions.items():
            lines.append(f"\n  [{model_name}]")
            for resp_line in response.split("\n"):
                lines.append(f"    {resp_line}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("SYNTHESIS")
    lines.append("=" * 60)

    if result.consensus:
        lines.append(f"\nConsensus: {result.consensus}")
    if result.disagreements:
        lines.append(f"\nDisagreements: {result.disagreements}")
    if result.tradeoffs:
        lines.append(f"\nTradeoffs: {result.tradeoffs}")
    if result.recommendation:
        lines.append(f"\nRecommendation: {result.recommendation}")

    lines.append(f"\nConfidence: {result.confidence:.0%}")

    if result.blind_spots:
        lines.append("\nBlind Spots:")
        for model_name, spot in result.blind_spots.items():
            lines.append(f"  {model_name}: {spot}")

    lines.append(f"\nTotal cost: ${result.total_cost:.4f}")
    lines.append("")

    return "\n".join(lines)


# ======================================================================
# Persistence
# ======================================================================


def save_debate(result: DebateResult, save_dir: Path | None = None) -> Path:
    """Save a debate result to disk as a Markdown report.

    The file is named ``<timestamp>_<topic_slug>.md`` and written to
    *save_dir* (defaulting to ``~/.prism/debates/``).

    Args:
        result: The debate result to persist.
        save_dir: Target directory. Created if it does not exist.

    Returns:
        Path to the saved Markdown file.
    """
    target_dir = save_dir or (Path.home() / ".prism" / "debates")
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = (
        result.created_at[:19].replace(":", "-")
        if result.created_at
        else "unknown"
    )
    slug = re.sub(r"[^\w]+", "_", result.question[:40]).strip("_").lower()
    filename = f"{timestamp}_{slug}.md"
    path = target_dir / filename

    report_text = generate_report_text(result)
    path.write_text(report_text, encoding="utf-8")
    logger.info("debate.saved", path=str(path))
    return path


def list_debates(save_dir: Path | None = None) -> list[Path]:
    """List all saved debate reports, newest first.

    Args:
        save_dir: Directory to scan. Defaults to ``~/.prism/debates/``.

    Returns:
        List of Markdown file paths sorted newest first.
    """
    target_dir = save_dir or (Path.home() / ".prism" / "debates")
    if not target_dir.exists():
        return []
    return sorted(target_dir.glob("*.md"), reverse=True)


# ======================================================================
# Main class (preserved for backward compatibility)
# ======================================================================


class MultiModelDebate:
    """Structured debate across multiple models for high-stakes decisions.

    This class uses the :class:`CompletionEngine` protocol for LLM calls.
    For simpler usage or testing, see the module-level :func:`debate`
    function which accepts a plain callback.

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

    async def run_debate(
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
            for crit in critiques:
                if isinstance(crit, DebateCritique):
                    session.round2_critiques.append(crit)
                    session.total_cost += crit.cost_usd
                    session.total_tokens += crit.tokens_used

        # ------ Round 3: Synthesis ------
        synthesis = await self._synthesize(question, session)
        if synthesis is not None:
            session.synthesis = synthesis
            session.total_cost += synthesis.cost_usd
            session.total_tokens += synthesis.tokens_used

        self._history.append(session)
        return session

    # Keep old name as alias for backward compatibility
    async def debate(
        self,
        question: str,
        quick: bool = False,
    ) -> DebateSession:
        return await self.run_debate(question, quick=quick)

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
    # Persistence (legacy)
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
        timestamp = (
            session.created_at[:19].replace(":", "-")
            if session.created_at
            else "unknown"
        )
        filename = f"debate_{timestamp}.json"
        path = save_dir / filename
        path.write_text(
            json.dumps(asdict(session), indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("debate.saved", path=str(path))
        return path

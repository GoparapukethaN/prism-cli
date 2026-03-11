"""Adaptive Execution Intelligence — learning fix strategies from past attempts.

Tracks every fix attempt in an SQLite table (attempt_log), fingerprints errors
to detect recurring patterns, and applies 3-strike escalation rules to
automatically switch strategies and models when cheap approaches keep failing.

Supports cross-repo learning so that a solution found in one project can be
suggested in another.

Slash-command hooks:
    /aei stats  — show attempt statistics
    /aei reset  — clear attempt history
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)


# ======================================================================
# Strategy enum and ordering
# ======================================================================


class FixStrategy(Enum):
    """Available fix strategies, ordered from cheapest to most expensive."""

    REGEX_PATCH = "regex_patch"
    AST_DIFF = "ast_diff"
    FULL_REWRITE = "full_rewrite"
    CONTEXT_EXPAND = "context_expand"
    ADD_DEFENSIVE_CODE = "add_defensive_code"
    MODEL_ESCALATE = "model_escalate"
    DECOMPOSE_SUBTASKS = "decompose_subtasks"
    REVERT_AND_REDESIGN = "revert_and_redesign"
    MULTI_MODEL_DEBATE = "multi_model_debate"


STRATEGY_ORDER: list[FixStrategy] = [
    FixStrategy.REGEX_PATCH,
    FixStrategy.AST_DIFF,
    FixStrategy.FULL_REWRITE,
    FixStrategy.CONTEXT_EXPAND,
    FixStrategy.ADD_DEFENSIVE_CODE,
    FixStrategy.MODEL_ESCALATE,
    FixStrategy.DECOMPOSE_SUBTASKS,
    FixStrategy.REVERT_AND_REDESIGN,
    FixStrategy.MULTI_MODEL_DEBATE,
]

ESCALATION_THRESHOLD: int = 3

MODEL_TIERS: dict[str, list[str]] = {
    "cheap": [
        "deepseek/deepseek-chat",
        "groq/llama-3.1-70b-versatile",
        "gpt-4o-mini",
    ],
    "mid": [
        "claude-sonnet-4-20250514",
        "gpt-4o",
    ],
    "premium": [
        "claude-3-opus-20240229",
        "gpt-4-turbo",
    ],
}


# ======================================================================
# Data classes
# ======================================================================


@dataclass(frozen=True)
class ErrorFingerprint:
    """Deterministic fingerprint for an error occurrence.

    The fingerprint hash is derived from (error_type, stack_pattern,
    file_path, function_name) so that the same logical error produces
    the same hash regardless of transient details like line numbers or
    memory addresses.

    Attributes:
        error_type: The exception class name (e.g. ``"TypeError"``).
        stack_pattern: Normalised stack trace pattern.
        file_path: Source file where the error originated.
        function_name: Function or method where the error originated.
        fingerprint_hash: Auto-computed SHA-256 prefix (16 hex chars).
    """

    error_type: str
    stack_pattern: str
    file_path: str
    function_name: str
    fingerprint_hash: str = ""

    def __post_init__(self) -> None:
        """Compute the fingerprint hash if not supplied."""
        if not self.fingerprint_hash:
            raw = (
                f"{self.error_type}:{self.stack_pattern}"
                f":{self.file_path}:{self.function_name}"
            )
            computed = hashlib.sha256(raw.encode()).hexdigest()[:16]
            object.__setattr__(self, "fingerprint_hash", computed)


@dataclass
class AttemptRecord:
    """A single fix attempt persisted in the ``attempt_log`` table.

    Attributes:
        id: Auto-incremented row ID.
        repo: Repository identifier (directory name or URL).
        fingerprint: The :class:`ErrorFingerprint` hash.
        strategy: The :class:`FixStrategy` value used.
        model: The LiteLLM model identifier used.
        context_size: Token count of the context window supplied.
        outcome: ``"success"`` or ``"failure"``.
        timestamp: ISO-8601 timestamp of the attempt.
        error_type: Original exception class name for grouping.
        reasoning: Human-readable rationale for the strategy choice.
    """

    id: int
    repo: str
    fingerprint: str
    strategy: str
    model: str
    context_size: int
    outcome: str
    timestamp: str
    error_type: str = ""
    reasoning: str = ""


@dataclass
class StrategyRecommendation:
    """The AEI engine's recommended approach for a fix attempt.

    Attributes:
        strategy: Recommended :class:`FixStrategy`.
        model_tier: ``"cheap"``, ``"mid"``, or ``"premium"``.
        context_multiplier: Multiply the default context window by this.
        reasoning: Human-readable explanation of why this was chosen.
        past_attempts: Number of past attempts with this fingerprint.
        past_successes: Number of past successes with this fingerprint.
        confidence: Estimated probability of success (0.0 - 1.0).
    """

    strategy: FixStrategy
    model_tier: str
    context_multiplier: float
    reasoning: str
    past_attempts: int
    past_successes: int
    confidence: float


@dataclass
class AEIStats:
    """Aggregate statistics from the attempt log.

    Attributes:
        total_attempts: Total number of recorded attempts.
        total_successes: Count of successful attempts.
        total_failures: Count of failed attempts.
        success_rate: Overall success rate (0.0 - 1.0).
        strategies_used: Mapping of strategy name to usage count.
        escalation_count: Number of escalation-level strategy uses.
        top_error_types: Most frequent error types as (type, count) pairs.
    """

    total_attempts: int
    total_successes: int
    total_failures: int
    success_rate: float
    strategies_used: dict[str, int]
    escalation_count: int
    top_error_types: list[tuple[str, int]]


# ======================================================================
# Main engine
# ======================================================================


class AdaptiveExecutionIntelligence:
    """Learning engine that tracks fix attempts and recommends strategies.

    Uses an SQLite database (thread-safe, WAL mode) to persist every
    fix attempt.  On each new attempt the engine queries past results
    for the same error fingerprint and applies 3-strike escalation rules:

    - 3 failures with a cheap model   -> escalate to mid/premium tier
    - 3 failures with regex patches   -> switch to AST diff
    - 3 failures with small context   -> expand context 2x
    - 3 failures with single strategy -> try next in STRATEGY_ORDER
    - All strategies exhausted         -> Multi-Model Debate

    Args:
        db_path: Path to the SQLite database file.
        repo: Repository identifier for scoping queries.
    """

    def __init__(self, db_path: Path, repo: str = "") -> None:
        self._db_path = db_path
        self._repo = repo
        self._local = threading.local()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    # ------------------------------------------------------------------
    # Connection management (thread-local)
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection.

        Returns:
            An open :class:`sqlite3.Connection` with WAL journaling and
            ``Row`` factory enabled.
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return self._local.conn

    def _initialize_db(self) -> None:
        """Create the ``attempt_log`` table and indexes if they don't exist."""
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS attempt_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                strategy TEXT NOT NULL,
                model TEXT NOT NULL,
                context_size INTEGER DEFAULT 0,
                outcome TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                error_type TEXT DEFAULT '',
                reasoning TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_attempt_fingerprint
                ON attempt_log(fingerprint);
            CREATE INDEX IF NOT EXISTS idx_attempt_repo
                ON attempt_log(repo);
            CREATE INDEX IF NOT EXISTS idx_attempt_outcome
                ON attempt_log(outcome);
            """
        )

    # ------------------------------------------------------------------
    # Error fingerprinting
    # ------------------------------------------------------------------

    @staticmethod
    def fingerprint_error(
        error_type: str,
        stack_trace: str,
        file_path: str,
        function_name: str,
    ) -> ErrorFingerprint:
        """Create a normalised :class:`ErrorFingerprint` for an error.

        Normalisation removes volatile details (line numbers, memory
        addresses, date stamps) so that the same logical bug produces
        the same fingerprint across runs.

        Args:
            error_type: Exception class name (e.g. ``"KeyError"``).
            stack_trace: Raw stack trace string.
            file_path: Source file path.
            function_name: Enclosing function or method name.

        Returns:
            An :class:`ErrorFingerprint` with a deterministic hash.
        """
        pattern = re.sub(r"line \d+", "line N", stack_trace)
        pattern = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", pattern)
        pattern = re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", pattern)
        return ErrorFingerprint(
            error_type=error_type,
            stack_pattern=pattern[:500],
            file_path=file_path,
            function_name=function_name,
        )

    # ------------------------------------------------------------------
    # Querying past attempts
    # ------------------------------------------------------------------

    def get_past_attempts(self, fingerprint: ErrorFingerprint) -> list[AttemptRecord]:
        """Retrieve all past attempts for a given error fingerprint.

        Args:
            fingerprint: The error fingerprint to look up.

        Returns:
            List of :class:`AttemptRecord` instances, newest first.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM attempt_log WHERE fingerprint = ? "
            "ORDER BY timestamp DESC",
            (fingerprint.fingerprint_hash,),
        ).fetchall()
        return [AttemptRecord(**dict(r)) for r in rows]

    def _get_cross_repo_attempts(self, error_type: str) -> list[AttemptRecord]:
        """Find attempts from *other* repos with the same error type.

        Args:
            error_type: Exception class name to search for.

        Returns:
            Up to 20 recent :class:`AttemptRecord` instances from other repos.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM attempt_log "
            "WHERE error_type = ? AND repo != ? "
            "ORDER BY timestamp DESC LIMIT 20",
            (error_type, self._repo),
        ).fetchall()
        return [AttemptRecord(**dict(r)) for r in rows]

    # ------------------------------------------------------------------
    # Strategy recommendation
    # ------------------------------------------------------------------

    def recommend_strategy(
        self, fingerprint: ErrorFingerprint
    ) -> StrategyRecommendation:
        """Recommend a fix strategy based on past attempts.

        Applies the following 3-strike escalation rules:

        1. If a previous attempt *succeeded*, re-use that strategy.
        2. If no history exists, check cross-repo data for the same error type.
        3. Otherwise, escalate through strategies and model tiers based on
           failure counts.

        Args:
            fingerprint: The error fingerprint to get a recommendation for.

        Returns:
            A :class:`StrategyRecommendation` with strategy, tier, and
            reasoning.
        """
        attempts = self.get_past_attempts(fingerprint)

        if not attempts:
            return self._recommend_from_cross_repo_or_default(fingerprint)

        failures = [a for a in attempts if a.outcome == "failure"]
        successes = [a for a in attempts if a.outcome == "success"]

        if successes:
            best = successes[0]
            return StrategyRecommendation(
                strategy=FixStrategy(best.strategy),
                model_tier=self._get_model_tier(best.model),
                context_multiplier=1.0,
                reasoning=f"Previously successful with {best.strategy}",
                past_attempts=len(attempts),
                past_successes=len(successes),
                confidence=0.8,
            )

        return self._escalate(failures, attempts)

    def _recommend_from_cross_repo_or_default(
        self, fingerprint: ErrorFingerprint
    ) -> StrategyRecommendation:
        """Recommend based on cross-repo data or fall back to the cheapest.

        Args:
            fingerprint: The error fingerprint.

        Returns:
            A :class:`StrategyRecommendation`.
        """
        cross_repo = self._get_cross_repo_attempts(fingerprint.error_type)
        if cross_repo:
            successful = [a for a in cross_repo if a.outcome == "success"]
            if successful:
                best = successful[0]
                return StrategyRecommendation(
                    strategy=FixStrategy(best.strategy),
                    model_tier="cheap",
                    context_multiplier=1.0,
                    reasoning=(
                        f"Cross-repo success with {best.strategy} "
                        f"for {fingerprint.error_type}"
                    ),
                    past_attempts=len(cross_repo),
                    past_successes=len(successful),
                    confidence=0.6,
                )

        return StrategyRecommendation(
            strategy=FixStrategy.REGEX_PATCH,
            model_tier="cheap",
            context_multiplier=1.0,
            reasoning="First attempt -- starting with cheapest strategy",
            past_attempts=0,
            past_successes=0,
            confidence=0.5,
        )

    def _escalate(
        self,
        failures: list[AttemptRecord],
        all_attempts: list[AttemptRecord],
    ) -> StrategyRecommendation:
        """Apply 3-strike escalation rules over past failures.

        Args:
            failures: All failed attempts for this fingerprint.
            all_attempts: All attempts (successes + failures).

        Returns:
            A :class:`StrategyRecommendation` after escalation.
        """
        strategy_failures: dict[str, int] = {}
        model_tier_failures: dict[str, int] = {}
        for f in failures:
            strategy_failures[f.strategy] = (
                strategy_failures.get(f.strategy, 0) + 1
            )
            tier = self._get_model_tier(f.model)
            model_tier_failures[tier] = model_tier_failures.get(tier, 0) + 1

        reasoning_parts: list[str] = []

        # --- Model tier escalation ---
        model_tier = "cheap"
        if model_tier_failures.get("cheap", 0) >= ESCALATION_THRESHOLD:
            model_tier = "mid"
            reasoning_parts.append(
                f"{ESCALATION_THRESHOLD} failures with cheap models "
                "-> escalating to mid-tier"
            )
        if model_tier_failures.get("mid", 0) >= ESCALATION_THRESHOLD:
            model_tier = "premium"
            reasoning_parts.append(
                f"{ESCALATION_THRESHOLD} failures with mid-tier "
                "-> escalating to premium"
            )

        # --- Strategy escalation ---
        strategy = FixStrategy.REGEX_PATCH
        found_viable = False
        for s in STRATEGY_ORDER:
            count = strategy_failures.get(s.value, 0)
            if count < ESCALATION_THRESHOLD:
                strategy = s
                found_viable = True
                break

        if not found_viable:
            strategy = FixStrategy.MULTI_MODEL_DEBATE
            reasoning_parts.append(
                "All strategies exhausted -> Multi-Model Debate"
            )

        # --- Context and strategy-specific escalation ---
        context_mult = 1.0

        # Specific regex -> AST escalation
        if (
            strategy_failures.get(FixStrategy.REGEX_PATCH.value, 0)
            >= ESCALATION_THRESHOLD
            and strategy == FixStrategy.REGEX_PATCH
        ):
            strategy = FixStrategy.AST_DIFF
            reasoning_parts.append(
                f"{ESCALATION_THRESHOLD} regex failures -> switching to AST diff"
            )

        # Full-rewrite exhaustion -> add_defensive_code (with context expansion)
        if (
            strategy_failures.get(FixStrategy.FULL_REWRITE.value, 0)
            >= ESCALATION_THRESHOLD
            and strategy_failures.get(FixStrategy.ADD_DEFENSIVE_CODE.value, 0)
            < ESCALATION_THRESHOLD
        ):
            strategy = FixStrategy.ADD_DEFENSIVE_CODE
            context_mult = max(context_mult, 2.0)
            reasoning_parts.append(
                f"{ESCALATION_THRESHOLD} full_rewrite failures "
                "-> context_expand + add_defensive_code"
            )

        # Decompose exhaustion -> revert_and_redesign before debate
        if (
            strategy_failures.get(FixStrategy.DECOMPOSE_SUBTASKS.value, 0)
            >= ESCALATION_THRESHOLD
            and strategy_failures.get(FixStrategy.REVERT_AND_REDESIGN.value, 0)
            < ESCALATION_THRESHOLD
        ):
            strategy = FixStrategy.REVERT_AND_REDESIGN
            reasoning_parts.append(
                f"{ESCALATION_THRESHOLD} decompose failures "
                "-> revert_and_redesign before multi_model_debate"
            )

        # Small-context expansion
        context_failures = sum(
            1
            for f in failures
            if f.context_size > 0 and f.context_size < 4000
        )
        if context_failures >= ESCALATION_THRESHOLD:
            context_mult = 2.0
            reasoning_parts.append(
                f"{ESCALATION_THRESHOLD} small-context failures -> expanding 2x"
            )

        if not reasoning_parts:
            reasoning_parts.append(
                f"Trying {strategy.value} after {len(failures)} prior failures"
            )

        confidence = max(0.1, 0.5 - (len(failures) * 0.05))

        return StrategyRecommendation(
            strategy=strategy,
            model_tier=model_tier,
            context_multiplier=context_mult,
            reasoning="; ".join(reasoning_parts),
            past_attempts=len(all_attempts),
            past_successes=0,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Recording attempts
    # ------------------------------------------------------------------

    def record_attempt(
        self,
        fingerprint: ErrorFingerprint,
        strategy: FixStrategy,
        model: str,
        context_size: int,
        outcome: str,
        reasoning: str = "",
    ) -> AttemptRecord:
        """Persist a fix attempt to the database.

        Args:
            fingerprint: The error fingerprint for this attempt.
            strategy: The fix strategy used.
            model: LiteLLM model identifier.
            context_size: Token count of the context supplied.
            outcome: ``"success"`` or ``"failure"``.
            reasoning: Optional human-readable rationale.

        Returns:
            The persisted :class:`AttemptRecord` with its new ``id``.

        Raises:
            ValueError: If *outcome* is not ``"success"`` or ``"failure"``.
        """
        if outcome not in ("success", "failure"):
            raise ValueError(
                f"outcome must be 'success' or 'failure', got {outcome!r}"
            )

        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        cursor = conn.execute(
            """
            INSERT INTO attempt_log
                (repo, fingerprint, strategy, model, context_size,
                 outcome, timestamp, error_type, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._repo,
                fingerprint.fingerprint_hash,
                strategy.value,
                model,
                context_size,
                outcome,
                now,
                fingerprint.error_type,
                reasoning,
            ),
        )
        conn.commit()

        logger.info(
            "aei_attempt_recorded",
            fingerprint=fingerprint.fingerprint_hash,
            strategy=strategy.value,
            model=model,
            outcome=outcome,
        )

        return AttemptRecord(
            id=cursor.lastrowid or 0,
            repo=self._repo,
            fingerprint=fingerprint.fingerprint_hash,
            strategy=strategy.value,
            model=model,
            context_size=context_size,
            outcome=outcome,
            timestamp=now,
            error_type=fingerprint.error_type,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, repo: str | None = None) -> AEIStats:
        """Return aggregate statistics from the attempt log.

        Args:
            repo: Optional repo filter.  ``None`` means all repos.

        Returns:
            An :class:`AEIStats` dataclass with totals, breakdowns, and
            top error types.
        """
        conn = self._get_conn()
        where = "WHERE repo = ?" if repo else ""
        params: tuple[Any, ...] = (repo,) if repo else ()

        rows = conn.execute(
            f"SELECT * FROM attempt_log {where}", params  # noqa: S608
        ).fetchall()
        total = len(rows)
        successes = sum(1 for r in rows if r["outcome"] == "success")
        failures = total - successes

        strategies: dict[str, int] = {}
        error_types: dict[str, int] = {}
        for r in rows:
            strategies[r["strategy"]] = strategies.get(r["strategy"], 0) + 1
            if r["error_type"]:
                error_types[r["error_type"]] = (
                    error_types.get(r["error_type"], 0) + 1
                )

        escalations = sum(
            1
            for r in rows
            if r["strategy"]
            in (
                FixStrategy.MODEL_ESCALATE.value,
                FixStrategy.MULTI_MODEL_DEBATE.value,
            )
        )

        top_errors = sorted(
            error_types.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return AEIStats(
            total_attempts=total,
            total_successes=successes,
            total_failures=failures,
            success_rate=successes / total if total > 0 else 0.0,
            strategies_used=strategies,
            escalation_count=escalations,
            top_error_types=top_errors,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, repo: str | None = None) -> int:
        """Delete attempt records.

        Args:
            repo: If specified, only delete records for this repo.
                ``None`` deletes everything.

        Returns:
            Number of rows deleted.
        """
        conn = self._get_conn()
        if repo:
            cursor = conn.execute(
                "DELETE FROM attempt_log WHERE repo = ?", (repo,)
            )
        else:
            cursor = conn.execute("DELETE FROM attempt_log")
        conn.commit()

        deleted = cursor.rowcount
        logger.info("aei_reset", repo=repo, deleted=deleted)
        return deleted

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain(self, fingerprint: ErrorFingerprint) -> str:
        """Return a human-readable explanation of the AEI's reasoning.

        Builds a multi-line narrative covering past attempt history,
        success/failure breakdown, the sequence of strategies tried,
        the current recommendation, and the confidence level.

        Args:
            fingerprint: The error fingerprint to explain.

        Returns:
            A multi-line string suitable for display in a Rich panel
            or plain terminal.
        """
        attempts = self.get_past_attempts(fingerprint)
        recommendation = self.recommend_strategy(fingerprint)

        successes = [a for a in attempts if a.outcome == "success"]
        failures = [a for a in attempts if a.outcome == "failure"]

        strategy_history: list[str] = []
        seen: set[str] = set()
        for a in reversed(attempts):
            label = f"{a.strategy} ({a.outcome})"
            if label not in seen:
                strategy_history.append(label)
                seen.add(label)

        lines: list[str] = [
            f"Error fingerprint: {fingerprint.fingerprint_hash}",
            f"Error type: {fingerprint.error_type}",
            f"File: {fingerprint.file_path}",
            f"Function: {fingerprint.function_name}",
            "",
            f"Past attempts: {len(attempts)}",
            f"  Successes: {len(successes)}",
            f"  Failures: {len(failures)}",
        ]

        if strategy_history:
            lines.append("")
            lines.append("Strategy history:")
            for entry in strategy_history:
                lines.append(f"  - {entry}")

        lines.append("")
        lines.append(
            f"Current recommendation: {recommendation.strategy.value}"
        )
        lines.append(f"  Model tier: {recommendation.model_tier}")
        lines.append(
            f"  Context multiplier: {recommendation.context_multiplier}x"
        )
        lines.append(f"  Reasoning: {recommendation.reasoning}")

        confidence_pct = f"{recommendation.confidence:.0%}"
        if recommendation.confidence >= 0.7:
            level = "high"
        elif recommendation.confidence >= 0.4:
            level = "moderate"
        else:
            level = "low"
        lines.append(
            f"  Confidence: {confidence_pct} ({level})"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model_tier(model: str) -> str:
        """Classify a model into its pricing tier.

        Args:
            model: LiteLLM model identifier.

        Returns:
            ``"cheap"``, ``"mid"``, or ``"premium"``.  Defaults to ``"cheap"``
            for unknown models.
        """
        for tier, models in MODEL_TIERS.items():
            if model in models:
                return tier
        return "cheap"

    def close(self) -> None:
        """Close the thread-local database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

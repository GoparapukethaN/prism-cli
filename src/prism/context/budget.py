"""Smart Context Budget Manager — intelligent context selection and token optimization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Context window sizes per model family (tokens)
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-sonnet-4-20250514": 200000,
    "claude-3-opus-20240229": 200000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "deepseek/deepseek-chat": 128000,
    "gemini/gemini-1.5-pro": 1000000,
    "gemini/gemini-1.5-flash": 1000000,
    "groq/llama-3.1-70b-versatile": 131072,
    "ollama/llama3.1:8b": 8192,
    "ollama/qwen2.5-coder:7b": 32768,
}

DEFAULT_CONTEXT_WINDOW = 128000
RESERVED_OUTPUT_TOKENS = 4096
TOKEN_CHARS_RATIO = 4  # rough chars per token


@dataclass
class ContextChunk:
    """A chunk of context with relevance scoring.

    Attributes:
        source: File path or identifier for this chunk.
        content: The raw text content of the chunk.
        score: Relevance score from 0.0 (irrelevant) to 1.0 (critical).
        token_estimate: Estimated token count for this chunk.
        chunk_type: Category — ``"file"``, ``"function"``, ``"test"``,
            ``"git_diff"``, ``"docstring"``, or ``"error"``.
        reason: Human-readable explanation of why this chunk was included.
    """

    source: str
    content: str
    score: float
    token_estimate: int
    chunk_type: str
    reason: str

    @property
    def is_included(self) -> bool:
        """Whether this chunk should be included (score > 0.0)."""
        return self.score > 0.0


@dataclass
class SmartContextBudget:
    """Token budget state returned by :meth:`ContextBudgetManager.select_context`.

    Attributes:
        model: The model identifier the budget was computed for.
        total_budget: Total token window for the model.
        used_tokens: Tokens consumed by included chunks.
        remaining_tokens: Tokens still available after included chunks.
        included_chunks: Chunks selected for inclusion.
        excluded_chunks: Chunks that did not fit or scored too low.
        efficiency: Ratio of tokens saved vs. naive full inclusion.
    """

    model: str
    total_budget: int
    used_tokens: int
    remaining_tokens: int
    included_chunks: list[ContextChunk]
    excluded_chunks: list[ContextChunk]
    efficiency: float

    @property
    def usage_percent(self) -> float:
        """Percentage of total budget consumed by included chunks."""
        return (self.used_tokens / max(self.total_budget, 1)) * 100


@dataclass
class ContextStats:
    """Aggregate statistics over multiple context-selection requests.

    Attributes:
        total_requests: Number of context selections performed.
        avg_tokens_used: Mean tokens included per request.
        avg_tokens_saved: Mean tokens saved per request.
        avg_efficiency: Mean efficiency ratio per request.
        avg_chunks_included: Mean number of included chunks.
        avg_chunks_excluded: Mean number of excluded chunks.
        total_tokens_saved: Cumulative tokens saved across all requests.
    """

    total_requests: int
    avg_tokens_used: float
    avg_tokens_saved: float
    avg_efficiency: float
    avg_chunks_included: float
    avg_chunks_excluded: float
    total_tokens_saved: int


@dataclass
class _HistoryEntry:
    """Internal record for one context-selection request."""

    timestamp: str
    model: str
    included: int
    excluded: int
    tokens_used: int
    tokens_total: int
    efficiency: float


class ContextBudgetManager:
    """Manages context selection and token budget for LLM requests.

    Given a task description and a list of available files, this manager
    scores each file for relevance, then greedily fills the model's token
    budget with the highest-scoring chunks.  Manual include/exclude
    overrides and historical statistics are also supported.

    Args:
        project_root: Absolute path to the project root directory.
    """

    def __init__(self, project_root: Path) -> None:
        self._root: Path = project_root.resolve()
        self._manual_includes: set[str] = set()
        self._manual_excludes: set[str] = set()
        self._history: list[_HistoryEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_context(
        self,
        task_description: str,
        available_files: list[str],
        model: str = "claude-sonnet-4-20250514",
        conversation_messages: list[dict[str, str]] | None = None,
        error_context: str | None = None,
    ) -> SmartContextBudget:
        """Select optimal context for *task_description* within token budget.

        Args:
            task_description: Natural-language description of the task.
            available_files: List of file paths (relative or absolute).
            model: Model identifier used to look up the context window.
            conversation_messages: Optional prior conversation messages, each
                a dict with at least a ``"content"`` key.
            error_context: Optional error / stack-trace text to inject at
                highest priority.

        Returns:
            A :class:`SmartContextBudget` describing what was included and
            excluded.
        """
        budget = self._get_token_budget(model)

        # Score all available files
        chunks: list[ContextChunk] = []
        for file_path in available_files:
            # Honour manual excludes
            if file_path in self._manual_excludes:
                continue

            score = self._score_relevance(
                file_path, task_description, conversation_messages,
            )

            # Force-include manual adds
            if file_path in self._manual_includes:
                score = max(score, 0.9)

            content = self._read_file_safe(file_path)
            if not content:
                continue

            token_est = len(content) // TOKEN_CHARS_RATIO
            chunk_type = self._classify_chunk(file_path)
            reason = self._get_inclusion_reason(file_path, task_description, score)

            chunks.append(ContextChunk(
                source=file_path,
                content=content,
                score=score,
                token_estimate=token_est,
                chunk_type=chunk_type,
                reason=reason,
            ))

        # Add error context with highest priority
        if error_context:
            token_est = len(error_context) // TOKEN_CHARS_RATIO
            chunks.append(ContextChunk(
                source="error_context",
                content=error_context,
                score=1.0,
                token_estimate=token_est,
                chunk_type="error",
                reason="Error/stack trace — always included",
            ))

        # Sort by score descending (highest first)
        chunks.sort(key=lambda c: c.score, reverse=True)

        # Reserve tokens for conversation history
        conversation_tokens = 0
        if conversation_messages:
            for msg in conversation_messages:
                conversation_tokens += len(str(msg.get("content", ""))) // TOKEN_CHARS_RATIO

        available_budget = budget - conversation_tokens - RESERVED_OUTPUT_TOKENS

        # Greedily fill budget
        included: list[ContextChunk] = []
        excluded: list[ContextChunk] = []
        used = 0

        for chunk in chunks:
            if chunk.score <= 0.0:
                excluded.append(chunk)
                continue

            if used + chunk.token_estimate <= available_budget:
                included.append(chunk)
                used += chunk.token_estimate
            else:
                excluded.append(chunk)

        # Calculate efficiency (token reduction vs naive)
        total_naive = sum(c.token_estimate for c in chunks)
        efficiency = 1.0 - (used / max(total_naive, 1))

        # Track for learning
        self._history.append(_HistoryEntry(
            timestamp=datetime.now(UTC).isoformat(),
            model=model,
            included=len(included),
            excluded=len(excluded),
            tokens_used=used,
            tokens_total=total_naive,
            efficiency=efficiency,
        ))

        logger.debug(
            "context_selected",
            model=model,
            included=len(included),
            excluded=len(excluded),
            tokens_used=used,
            budget=budget,
            efficiency=f"{efficiency:.1%}",
        )

        return SmartContextBudget(
            model=model,
            total_budget=budget,
            used_tokens=used,
            remaining_tokens=available_budget - used,
            included_chunks=included,
            excluded_chunks=excluded,
            efficiency=efficiency,
        )

    def add_file(self, file_path: str) -> None:
        """Manually force-include a file in context.

        Args:
            file_path: Path of the file to force-include.
        """
        self._manual_includes.add(file_path)
        self._manual_excludes.discard(file_path)

    def drop_file(self, file_path: str) -> None:
        """Manually exclude a file from context.

        Args:
            file_path: Path of the file to exclude.
        """
        self._manual_excludes.add(file_path)
        self._manual_includes.discard(file_path)

    def get_stats(self) -> ContextStats:
        """Get aggregate context-selection statistics.

        Returns:
            A :class:`ContextStats` summarising all prior selections.
        """
        if not self._history:
            return ContextStats(
                total_requests=0,
                avg_tokens_used=0.0,
                avg_tokens_saved=0.0,
                avg_efficiency=0.0,
                avg_chunks_included=0.0,
                avg_chunks_excluded=0.0,
                total_tokens_saved=0,
            )

        n = len(self._history)
        total_used = sum(h.tokens_used for h in self._history)
        total_all = sum(h.tokens_total for h in self._history)

        return ContextStats(
            total_requests=n,
            avg_tokens_used=total_used / n,
            avg_tokens_saved=(total_all - total_used) / n,
            avg_efficiency=sum(h.efficiency for h in self._history) / n,
            avg_chunks_included=sum(h.included for h in self._history) / n,
            avg_chunks_excluded=sum(h.excluded for h in self._history) / n,
            total_tokens_saved=total_all - total_used,
        )

    def show_context(self, budget: SmartContextBudget) -> str:
        """Format context budget for display.

        Args:
            budget: The :class:`SmartContextBudget` to render.

        Returns:
            A multi-line human-readable string.
        """
        lines = [
            f"Context Budget: {budget.used_tokens:,} / {budget.total_budget:,} tokens"
            f" ({budget.usage_percent:.1f}%)",
            f"Efficiency: {budget.efficiency:.1%} token reduction",
            "",
            "Included:",
        ]

        for chunk in budget.included_chunks:
            lines.append(
                f"  [{chunk.score:.2f}] {chunk.source}"
                f" ({chunk.token_estimate:,} tokens) — {chunk.reason}"
            )

        if budget.excluded_chunks:
            lines.append("")
            lines.append("Excluded:")
            for chunk in budget.excluded_chunks[:10]:
                lines.append(
                    f"  [{chunk.score:.2f}] {chunk.source}"
                    f" ({chunk.token_estimate:,} tokens)"
                )

        return "\n".join(lines)

    def reset_overrides(self) -> None:
        """Clear all manual includes and excludes."""
        self._manual_includes.clear()
        self._manual_excludes.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_token_budget(self, model: str) -> int:
        """Get the token budget for *model*."""
        return MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)

    def _score_relevance(
        self,
        file_path: str,
        task_description: str,
        messages: list[dict[str, str]] | None = None,
    ) -> float:
        """Score relevance of *file_path* to *task_description* (0.0-1.0)."""
        score = 0.0
        path_lower = file_path.lower()
        task_lower = task_description.lower()

        # Direct mention in task description
        filename = Path(file_path).stem.lower()
        if filename in task_lower or file_path in task_description:
            return 1.0

        # Keyword matching
        task_words = set(re.findall(r"\b\w{3,}\b", task_lower))
        path_words = set(
            re.findall(r"\b\w{3,}\b", path_lower.replace("/", " ").replace("_", " "))
        )

        overlap = task_words & path_words
        if overlap:
            word_score = min(len(overlap) * 0.2, 0.8)
            score = max(score, word_score)

        # Check conversation messages for mentions
        if messages:
            for msg in messages:
                content = str(msg.get("content", "")).lower()
                if filename in content or file_path in content:
                    score = max(score, 0.7)
                    break

        # Test files for mentioned source files
        if "test_" in path_lower:
            tested_name = filename.replace("test_", "")
            if tested_name in task_lower:
                score = max(score, 0.6)

        # Same directory bonus
        for word in task_words:
            if word in path_lower:
                score = max(score, 0.3)
                break

        return min(score, 1.0)

    def _classify_chunk(self, file_path: str) -> str:
        """Classify a context chunk by type."""
        if "test" in file_path.lower():
            return "test"
        if file_path.endswith(".md"):
            return "docstring"
        return "file"

    def _get_inclusion_reason(self, file_path: str, task: str, score: float) -> str:
        """Generate a human-readable reason for inclusion."""
        if score >= 0.9:
            return "Directly mentioned in task"
        if score >= 0.7:
            return "Referenced in conversation"
        if score >= 0.5:
            return "Keyword match with task"
        if score >= 0.3:
            return "Related by directory/module"
        return "Low relevance"

    def _read_file_safe(self, file_path: str) -> str:
        """Read a file safely, returning empty string on error.

        Files larger than 500 KB are silently skipped.
        """
        try:
            full_path = (
                self._root / file_path
                if not Path(file_path).is_absolute()
                else Path(file_path)
            )
            if full_path.is_file() and full_path.stat().st_size < 500_000:
                return full_path.read_text(errors="replace")
        except OSError:
            pass
        return ""

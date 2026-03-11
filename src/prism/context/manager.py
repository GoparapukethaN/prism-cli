"""Context manager — assembles context for LLM calls.

Maintains conversation history, manages token budgets, injects system
prompts, and tracks active files so that each request sends the most
relevant information within the model's context window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from prism.exceptions import ContextError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# Rough heuristic: 1 word ≈ 1.3 tokens.
_TOKENS_PER_WORD: float = 1.3


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text* using word-count heuristic.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count (always >= 0).
    """
    if not text:
        return 0
    word_count = len(text.split())
    return max(1, int(word_count * _TOKENS_PER_WORD))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContextBudget:
    """Token budget allocation for each section of the context window."""

    system_prompt: int = 0
    conversation: int = 0
    repo_context: int = 0
    active_files: int = 0

    @property
    def total(self) -> int:
        return self.system_prompt + self.conversation + self.repo_context + self.active_files


@dataclass
class Message:
    """A single message in the conversation history."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = estimate_tokens(self.content)


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are Prism, an intelligent coding assistant. "
    "You help developers with code generation, debugging, refactoring, "
    "and answering questions about their codebase. "
    "Be concise, accurate, and provide working code."
)


class ContextManager:
    """Assembles context for LLM calls, managing token budgets.

    Token budget allocation (as fractions of *max_tokens*):
    - 40% conversation history
    - 30% repository context (repo map, etc.)
    - 20% active files
    - 10% system prompt

    Args:
        system_prompt: Custom system prompt (uses a default if *None*).
        max_tokens: Maximum total tokens for assembled context.
    """

    # Budget fractions
    CONVERSATION_FRACTION: float = 0.40
    REPO_CONTEXT_FRACTION: float = 0.30
    ACTIVE_FILES_FRACTION: float = 0.20
    SYSTEM_PROMPT_FRACTION: float = 0.10

    def __init__(
        self,
        system_prompt: str | None = None,
        max_tokens: int = 128_000,
    ) -> None:
        self._system_prompt: str = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._max_tokens: int = max_tokens
        self._messages: list[Message] = []
        self._active_files: dict[str, str] = {}  # path -> content
        self._repo_context: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        if value <= 0:
            raise ContextError("max_tokens must be positive")
        self._max_tokens = value

    @property
    def messages(self) -> list[Message]:
        """Return a copy of the internal message list."""
        return list(self._messages)

    @property
    def active_files(self) -> dict[str, str]:
        """Return a copy of the active files mapping."""
        return dict(self._active_files)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    # ------------------------------------------------------------------
    # Budget calculation
    # ------------------------------------------------------------------

    def allocate_budget(self, max_tokens: int | None = None) -> ContextBudget:
        """Calculate token budget allocation.

        Args:
            max_tokens: Override for the instance-level max_tokens.

        Returns:
            A :class:`ContextBudget` with per-section limits.
        """
        total = max_tokens or self._max_tokens
        return ContextBudget(
            system_prompt=int(total * self.SYSTEM_PROMPT_FRACTION),
            conversation=int(total * self.CONVERSATION_FRACTION),
            repo_context=int(total * self.REPO_CONTEXT_FRACTION),
            active_files=int(total * self.ACTIVE_FILES_FRACTION),
        )

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> Message:
        """Append a message to the conversation history.

        Args:
            role: Message role (``"user"``, ``"assistant"``, ``"system"``).
            content: Message text content.

        Returns:
            The newly created :class:`Message`.

        Raises:
            ContextError: If *role* or *content* is empty.
        """
        if not role:
            raise ContextError("Message role must not be empty")
        if not content:
            raise ContextError("Message content must not be empty")

        msg = Message(role=role, content=content)
        self._messages.append(msg)
        logger.debug(
            "message_added",
            role=role,
            tokens=msg.token_estimate,
            total_messages=len(self._messages),
        )
        return msg

    def clear_messages(self) -> None:
        """Remove all messages from the conversation history."""
        self._messages.clear()

    # ------------------------------------------------------------------
    # Active file tracking
    # ------------------------------------------------------------------

    def add_active_file(self, path: str, content: str) -> None:
        """Add a file to the active-files set.

        Args:
            path: File path (used as a key).
            content: File contents.
        """
        self._active_files[path] = content
        logger.debug("active_file_added", path=path, tokens=estimate_tokens(content))

    def remove_active_file(self, path: str) -> bool:
        """Remove a file from the active-files set.

        Returns:
            *True* if the file was present and removed.
        """
        if path in self._active_files:
            del self._active_files[path]
            logger.debug("active_file_removed", path=path)
            return True
        return False

    def clear_active_files(self) -> None:
        """Remove all active files."""
        self._active_files.clear()

    # ------------------------------------------------------------------
    # Repository context
    # ------------------------------------------------------------------

    def set_repo_context(self, context: str) -> None:
        """Set the repository-level context (repo map, etc.)."""
        self._repo_context = context

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def get_context(self, max_tokens: int | None = None) -> list[dict[str, str]]:
        """Assemble the full context for an LLM call.

        Messages are trimmed from the oldest end to fit the token budget.
        The system prompt is always the first message.  Active files and
        repo context are injected as system messages after the prompt.

        Args:
            max_tokens: Override for the instance-level max_tokens.

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts ready for
            an LLM API call.
        """
        budget = self.allocate_budget(max_tokens)
        result: list[dict[str, str]] = []
        tokens_used = 0

        # 1. System prompt (always included, trimmed if needed)
        system_text = self._system_prompt
        system_tokens = estimate_tokens(system_text)
        if system_tokens > budget.system_prompt and budget.system_prompt > 0:
            # Truncate system prompt to budget
            words = system_text.split()
            max_words = int(budget.system_prompt / _TOKENS_PER_WORD)
            system_text = " ".join(words[:max(1, max_words)])
            system_tokens = estimate_tokens(system_text)
        result.append({"role": "system", "content": system_text})
        tokens_used += system_tokens

        # 2. Repository context
        if self._repo_context:
            repo_tokens = estimate_tokens(self._repo_context)
            if repo_tokens <= budget.repo_context:
                result.append({
                    "role": "system",
                    "content": f"[Repository Map]\n{self._repo_context}",
                })
                tokens_used += repo_tokens

        # 3. Active files
        if self._active_files:
            files_text = self._format_active_files(budget.active_files)
            if files_text:
                files_tokens = estimate_tokens(files_text)
                result.append({
                    "role": "system",
                    "content": f"[Active Files]\n{files_text}",
                })
                tokens_used += files_tokens

        # 4. Conversation history (trimmed to fit remaining budget)
        remaining = (max_tokens or self._max_tokens) - tokens_used
        conversation_budget = min(budget.conversation, remaining)
        trimmed = self._trim_messages(conversation_budget)
        result.extend(trimmed)

        return result

    def _format_active_files(self, max_tokens: int) -> str:
        """Format active files for inclusion in context, respecting token budget."""
        if not self._active_files:
            return ""

        parts: list[str] = []
        tokens_remaining = max_tokens

        for path, content in self._active_files.items():
            header = f"--- {path} ---"
            entry = f"{header}\n{content}"
            entry_tokens = estimate_tokens(entry)

            if entry_tokens <= tokens_remaining:
                parts.append(entry)
                tokens_remaining -= entry_tokens
            elif tokens_remaining > estimate_tokens(header) + 10:
                # Truncate file content
                words = content.split()
                available_words = int(tokens_remaining / _TOKENS_PER_WORD) - 5
                truncated = " ".join(words[:max(1, available_words)])
                parts.append(f"{header}\n{truncated}\n... [truncated]")
                break
            else:
                break

        return "\n\n".join(parts)

    def _trim_messages(self, max_tokens: int) -> list[dict[str, str]]:
        """Return conversation messages trimmed to fit *max_tokens*.

        Keeps the most recent messages, dropping oldest first.

        Args:
            max_tokens: Maximum tokens for the conversation portion.

        Returns:
            List of message dicts in chronological order.
        """
        if not self._messages:
            return []

        if max_tokens <= 0:
            return []

        # Walk backwards, accumulating messages until budget is exhausted
        selected: list[dict[str, str]] = []
        tokens_used = 0

        for msg in reversed(self._messages):
            if tokens_used + msg.token_estimate > max_tokens:
                break
            selected.append({"role": msg.role, "content": msg.content})
            tokens_used += msg.token_estimate

        # Reverse so messages are in chronological order
        selected.reverse()
        return selected

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def total_tokens(self) -> int:
        """Estimate the total tokens across all stored content."""
        total = estimate_tokens(self._system_prompt)
        total += sum(m.token_estimate for m in self._messages)
        total += estimate_tokens(self._repo_context)
        for content in self._active_files.values():
            total += estimate_tokens(content)
        return total

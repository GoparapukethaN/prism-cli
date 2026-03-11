"""Conversation summarizer — compresses old conversation history.

Uses a simple extractive approach (no LLM call):
- Always keeps system messages
- Always keeps the first user message
- Keeps the most recent *N* messages
- Replaces older messages with a ``[Earlier conversation summarized]`` block
"""

from __future__ import annotations

import structlog

from prism.context.manager import estimate_tokens

logger = structlog.get_logger(__name__)

# Number of recent messages to always preserve (at the tail)
_DEFAULT_KEEP_RECENT: int = 6


def summarize(
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    keep_recent: int = _DEFAULT_KEEP_RECENT,
) -> str:
    """Summarize a list of conversation messages to fit within *max_tokens*.

    The strategy:
    1. Always include system messages (they carry context).
    2. Always include the first user message (establishes intent).
    3. Keep the last *keep_recent* messages.
    4. Replace everything in between with
       ``[Earlier conversation summarized]``.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        max_tokens: Target maximum token count for the summary.
        keep_recent: Number of recent messages to preserve verbatim.

    Returns:
        A single string containing the summarised conversation.
    """
    if not messages:
        return ""

    # ---- Identify preserved messages ----
    preserved_head: list[dict[str, str]] = []
    preserved_tail: list[dict[str, str]] = []
    middle_messages: list[dict[str, str]] = []

    # System messages and first user message
    first_user_found = False
    head_end_idx = 0

    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            preserved_head.append(msg)
            head_end_idx = i + 1
        elif not first_user_found and msg.get("role") == "user":
            preserved_head.append(msg)
            first_user_found = True
            head_end_idx = i + 1
        else:
            break

    # Recent messages (tail)
    total = len(messages)
    tail_start = max(head_end_idx, total - keep_recent)
    preserved_tail = list(messages[tail_start:])

    # Middle (will be summarised)
    middle_messages = list(messages[head_end_idx:tail_start])

    # ---- Build the summary ----
    parts: list[str] = []

    # Head
    for msg in preserved_head:
        parts.append(_format_message(msg))

    # Summarised middle
    if middle_messages:
        summary_block = _summarize_middle(middle_messages)
        parts.append(summary_block)

    # Tail
    for msg in preserved_tail:
        parts.append(_format_message(msg))

    result = "\n\n".join(parts)

    # Trim if still over budget
    result = _trim_to_budget(result, max_tokens)

    logger.debug(
        "conversation_summarized",
        original_messages=len(messages),
        summarized_middle=len(middle_messages),
        kept_head=len(preserved_head),
        kept_tail=len(preserved_tail),
        result_tokens=estimate_tokens(result),
    )

    return result


def _format_message(msg: dict[str, str]) -> str:
    """Format a single message for display."""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    return f"[{role}]: {content}"


def _summarize_middle(messages: list[dict[str, str]]) -> str:
    """Create a summary block for the middle messages.

    This is a simple extractive summary: we note the roles and count,
    and extract key sentences from user messages.
    """
    if not messages:
        return ""

    user_count = sum(1 for m in messages if m.get("role") == "user")
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")

    # Extract first sentence of each user message for key topics
    topics: list[str] = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "").strip()
            if content:
                # Take first sentence (up to ~100 chars)
                first_sentence = content.split(".")[0].strip()
                if len(first_sentence) > 100:
                    first_sentence = first_sentence[:97] + "..."
                if first_sentence:
                    topics.append(first_sentence)

    summary_lines = [
        "[Earlier conversation summarized]",
        f"({user_count} user message(s), {assistant_count} assistant message(s))",
    ]

    if topics:
        summary_lines.append("Topics discussed:")
        for topic in topics[:5]:  # Limit to 5 topics
            summary_lines.append(f"  - {topic}")

    return "\n".join(summary_lines)


def _trim_to_budget(text: str, max_tokens: int) -> str:
    """Trim *text* to fit within *max_tokens*."""
    current = estimate_tokens(text)
    if current <= max_tokens:
        return text

    # Trim from the middle
    words = text.split()
    from prism.context.manager import _TOKENS_PER_WORD
    target_words = int(max_tokens / _TOKENS_PER_WORD)

    if target_words >= len(words):
        return text

    # Keep first half and last portion
    half = target_words // 2
    trimmed_words = [*words[:half], "...", "[trimmed]", "...", *words[-half:]]
    return " ".join(trimmed_words)

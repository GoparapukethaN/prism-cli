"""Tests for prism.context.summarizer — conversation summarization."""

from __future__ import annotations

from prism.context.manager import estimate_tokens
from prism.context.summarizer import (
    _format_message,
    _summarize_middle,
    summarize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _make_conversation(n: int) -> list[dict[str, str]]:
    """Create a conversation with *n* user+assistant pairs plus a system msg."""
    msgs = [_msg("system", "You are a helpful assistant.")]
    for i in range(n):
        msgs.append(_msg("user", f"Question {i}: What about topic {i}?"))
        msgs.append(_msg("assistant", f"Answer {i}: Here is info about topic {i}."))
    return msgs


# ---------------------------------------------------------------------------
# _format_message
# ---------------------------------------------------------------------------


class TestFormatMessage:
    def test_user_message(self) -> None:
        result = _format_message(_msg("user", "Hello"))
        assert result == "[user]: Hello"

    def test_assistant_message(self) -> None:
        result = _format_message(_msg("assistant", "Hi there"))
        assert result == "[assistant]: Hi there"

    def test_system_message(self) -> None:
        result = _format_message(_msg("system", "Be helpful"))
        assert result == "[system]: Be helpful"


# ---------------------------------------------------------------------------
# _summarize_middle
# ---------------------------------------------------------------------------


class TestSummarizeMiddle:
    def test_empty_list(self) -> None:
        assert _summarize_middle([]) == ""

    def test_includes_marker(self) -> None:
        msgs = [_msg("user", "Hello"), _msg("assistant", "Hi")]
        result = _summarize_middle(msgs)
        assert "[Earlier conversation summarized]" in result

    def test_includes_counts(self) -> None:
        msgs = [
            _msg("user", "Q1"),
            _msg("assistant", "A1"),
            _msg("user", "Q2"),
            _msg("assistant", "A2"),
        ]
        result = _summarize_middle(msgs)
        assert "2 user message(s)" in result
        assert "2 assistant message(s)" in result

    def test_includes_topics(self) -> None:
        msgs = [
            _msg("user", "Fix the authentication bug. It fails on login."),
            _msg("assistant", "Sure, let me look at that."),
        ]
        result = _summarize_middle(msgs)
        assert "Fix the authentication bug" in result


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_empty_messages(self) -> None:
        result = summarize([], max_tokens=1000)
        assert result == ""

    def test_preserves_system_messages(self) -> None:
        msgs = _make_conversation(5)
        result = summarize(msgs, max_tokens=5000, keep_recent=2)
        assert "[system]:" in result

    def test_preserves_first_user_message(self) -> None:
        msgs = _make_conversation(5)
        result = summarize(msgs, max_tokens=5000, keep_recent=2)
        assert "Question 0" in result

    def test_preserves_recent_messages(self) -> None:
        msgs = _make_conversation(10)
        result = summarize(msgs, max_tokens=50000, keep_recent=4)
        # The last 4 messages should be there
        # Last pair: Question 9, Answer 9
        assert "Question 9" in result
        assert "Answer 9" in result

    def test_includes_summary_marker(self) -> None:
        msgs = _make_conversation(10)
        result = summarize(msgs, max_tokens=50000, keep_recent=4)
        assert "[Earlier conversation summarized]" in result

    def test_short_conversation_no_summary(self) -> None:
        msgs = [
            _msg("system", "System prompt"),
            _msg("user", "Hello"),
            _msg("assistant", "Hi!"),
        ]
        result = summarize(msgs, max_tokens=50000, keep_recent=6)
        # All messages fit, no summary marker should be needed
        # The head captures system + first user, tail captures from head_end_idx to end
        assert "[system]: System prompt" in result
        assert "[user]: Hello" in result
        assert "[assistant]: Hi!" in result

    def test_respects_max_tokens(self) -> None:
        msgs = _make_conversation(50)
        result = summarize(msgs, max_tokens=100)
        tokens = estimate_tokens(result)
        assert tokens <= 110  # Allow small overshoot due to estimation

    def test_single_message(self) -> None:
        msgs = [_msg("user", "Just one message")]
        result = summarize(msgs, max_tokens=5000)
        assert "Just one message" in result

    def test_keep_recent_zero(self) -> None:
        msgs = _make_conversation(5)
        result = summarize(msgs, max_tokens=50000, keep_recent=0)
        # Should still have system and first user
        assert "[system]:" in result
        assert "Question 0" in result
        # Middle should be summarized
        assert "[Earlier conversation summarized]" in result

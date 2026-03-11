"""Tests for prism.context.manager — ContextManager."""

from __future__ import annotations

import pytest

from prism.context.manager import (
    ContextManager,
    Message,
    estimate_tokens,
)
from prism.exceptions import ContextError

# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_single_word(self) -> None:
        result = estimate_tokens("hello")
        assert result >= 1

    def test_multiple_words(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        result = estimate_tokens(text)
        # 9 words × 1.3 ≈ 11.7 → 11
        assert result == int(9 * 1.3)

    def test_proportional_to_length(self) -> None:
        short = estimate_tokens("hello world")
        long = estimate_tokens("hello world " * 100)
        assert long > short


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_auto_token_count(self) -> None:
        msg = Message(role="user", content="hello world")
        assert msg.token_estimate > 0

    def test_explicit_token_count(self) -> None:
        msg = Message(role="user", content="hello world", token_estimate=42)
        assert msg.token_estimate == 42

    def test_timestamp_set(self) -> None:
        msg = Message(role="user", content="test")
        assert msg.timestamp > 0


# ---------------------------------------------------------------------------
# ContextManager — basics
# ---------------------------------------------------------------------------


class TestContextManagerBasics:
    def test_default_system_prompt(self) -> None:
        cm = ContextManager()
        assert "Prism" in cm.system_prompt

    def test_custom_system_prompt(self) -> None:
        cm = ContextManager(system_prompt="Custom prompt")
        assert cm.system_prompt == "Custom prompt"

    def test_set_system_prompt(self) -> None:
        cm = ContextManager()
        cm.system_prompt = "New prompt"
        assert cm.system_prompt == "New prompt"

    def test_max_tokens_property(self) -> None:
        cm = ContextManager(max_tokens=50_000)
        assert cm.max_tokens == 50_000

    def test_set_max_tokens_invalid(self) -> None:
        cm = ContextManager()
        with pytest.raises(ContextError, match="positive"):
            cm.max_tokens = 0

    def test_message_count_starts_zero(self) -> None:
        cm = ContextManager()
        assert cm.message_count == 0


# ---------------------------------------------------------------------------
# ContextManager — add_message
# ---------------------------------------------------------------------------


class TestAddMessage:
    def test_add_user_message(self, ctx: ContextManager) -> None:
        msg = ctx.add_message("user", "Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert ctx.message_count == 1

    def test_add_multiple_messages(self, ctx: ContextManager) -> None:
        ctx.add_message("user", "Hi")
        ctx.add_message("assistant", "Hello!")
        ctx.add_message("user", "Help me")
        assert ctx.message_count == 3

    def test_add_message_empty_role_raises(self, ctx: ContextManager) -> None:
        with pytest.raises(ContextError, match="role"):
            ctx.add_message("", "Hello")

    def test_add_message_empty_content_raises(self, ctx: ContextManager) -> None:
        with pytest.raises(ContextError, match="content"):
            ctx.add_message("user", "")

    def test_clear_messages(self, ctx: ContextManager) -> None:
        ctx.add_message("user", "test")
        ctx.add_message("assistant", "reply")
        ctx.clear_messages()
        assert ctx.message_count == 0

    def test_messages_returns_copy(self, ctx: ContextManager) -> None:
        ctx.add_message("user", "test")
        msgs = ctx.messages
        msgs.clear()
        assert ctx.message_count == 1


# ---------------------------------------------------------------------------
# ContextManager — active files
# ---------------------------------------------------------------------------


class TestActiveFiles:
    def test_add_active_file(self, ctx: ContextManager) -> None:
        ctx.add_active_file("src/main.py", "print('hello')")
        assert "src/main.py" in ctx.active_files

    def test_remove_active_file(self, ctx: ContextManager) -> None:
        ctx.add_active_file("src/main.py", "code here")
        removed = ctx.remove_active_file("src/main.py")
        assert removed is True
        assert "src/main.py" not in ctx.active_files

    def test_remove_nonexistent_file(self, ctx: ContextManager) -> None:
        removed = ctx.remove_active_file("does_not_exist.py")
        assert removed is False

    def test_clear_active_files(self, ctx: ContextManager) -> None:
        ctx.add_active_file("a.py", "code a")
        ctx.add_active_file("b.py", "code b")
        ctx.clear_active_files()
        assert len(ctx.active_files) == 0

    def test_active_files_returns_copy(self, ctx: ContextManager) -> None:
        ctx.add_active_file("x.py", "x")
        files = ctx.active_files
        files.clear()
        assert len(ctx.active_files) == 1


# ---------------------------------------------------------------------------
# ContextManager — budget allocation
# ---------------------------------------------------------------------------


class TestBudgetAllocation:
    def test_budget_fractions_sum(self) -> None:
        total = (
            ContextManager.CONVERSATION_FRACTION
            + ContextManager.REPO_CONTEXT_FRACTION
            + ContextManager.ACTIVE_FILES_FRACTION
            + ContextManager.SYSTEM_PROMPT_FRACTION
        )
        assert total == pytest.approx(1.0)

    def test_allocate_budget(self) -> None:
        cm = ContextManager(max_tokens=10_000)
        budget = cm.allocate_budget()
        assert budget.system_prompt == 1_000  # 10%
        assert budget.conversation == 4_000   # 40%
        assert budget.repo_context == 3_000   # 30%
        assert budget.active_files == 2_000   # 20%

    def test_allocate_budget_override(self) -> None:
        cm = ContextManager(max_tokens=10_000)
        budget = cm.allocate_budget(max_tokens=20_000)
        assert budget.conversation == 8_000

    def test_budget_total(self) -> None:
        cm = ContextManager(max_tokens=10_000)
        budget = cm.allocate_budget()
        assert budget.total == 10_000


# ---------------------------------------------------------------------------
# ContextManager — get_context
# ---------------------------------------------------------------------------


class TestGetContext:
    def test_system_prompt_always_first(self, ctx: ContextManager) -> None:
        ctx.add_message("user", "Hello")
        result = ctx.get_context()
        assert result[0]["role"] == "system"
        assert "test assistant" in result[0]["content"]

    def test_includes_conversation(self, ctx: ContextManager) -> None:
        ctx.add_message("user", "Tell me about Python")
        ctx.add_message("assistant", "Python is great!")
        result = ctx.get_context()
        roles = [m["role"] for m in result]
        assert "user" in roles
        assert "assistant" in roles

    def test_includes_repo_context(self, ctx: ContextManager) -> None:
        ctx.set_repo_context("src/main.py:\n  def main() -> None")
        result = ctx.get_context()
        contents = " ".join(m["content"] for m in result)
        assert "Repository Map" in contents

    def test_includes_active_files(self, ctx: ContextManager) -> None:
        ctx.add_active_file("test.py", "print('hello')")
        result = ctx.get_context()
        contents = " ".join(m["content"] for m in result)
        assert "Active Files" in contents

    def test_trims_old_messages(self) -> None:
        # Very small budget — should drop old messages
        cm = ContextManager(system_prompt="Hi", max_tokens=100)
        for i in range(50):
            cm.add_message("user", f"Message number {i} with some padding words to consume tokens")
        result = cm.get_context()
        # Should have fewer messages than we added
        user_messages = [m for m in result if m["role"] == "user"]
        assert len(user_messages) < 50

    def test_empty_conversation(self, ctx: ContextManager) -> None:
        result = ctx.get_context()
        # Should still have system prompt
        assert len(result) >= 1
        assert result[0]["role"] == "system"

    def test_max_tokens_override(self, ctx: ContextManager) -> None:
        for i in range(20):
            ctx.add_message("user", f"Message {i} with extra words padding")
        small = ctx.get_context(max_tokens=200)
        large = ctx.get_context(max_tokens=200_000)
        assert len(small) <= len(large)


# ---------------------------------------------------------------------------
# ContextManager — total_tokens
# ---------------------------------------------------------------------------


class TestTotalTokens:
    def test_total_tokens_empty(self) -> None:
        cm = ContextManager(system_prompt="short")
        tokens = cm.total_tokens()
        assert tokens > 0  # At least the system prompt

    def test_total_tokens_increases(self, ctx: ContextManager) -> None:
        before = ctx.total_tokens()
        ctx.add_message("user", "A fairly long message with many words")
        after = ctx.total_tokens()
        assert after > before

    def test_total_tokens_includes_files(self, ctx: ContextManager) -> None:
        before = ctx.total_tokens()
        ctx.add_active_file("big.py", "code " * 500)
        after = ctx.total_tokens()
        assert after > before

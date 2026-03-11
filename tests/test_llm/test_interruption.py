"""Tests for streaming interruption — 30+ tests, fully offline.

Covers every class and function in ``prism.llm.interruption``:
- InterruptAction enum values
- PartialResponse dataclass fields and ``is_empty``
- InterruptionState defaults and mutation
- StreamInterruptHandler install/uninstall, interrupt detection,
  tool-execution deferral, partial save, action recording,
  resume messages, file-write safety, reset
- prompt_interrupt_action with mocked ``input()``
"""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import pytest

from prism.llm.interruption import (
    InterruptAction,
    InterruptionState,
    PartialResponse,
    StreamInterruptHandler,
    prompt_interrupt_action,
)

# ======================================================================
# InterruptAction
# ======================================================================


class TestInterruptAction:
    """Verify enum members and their string values."""

    def test_keep_value(self) -> None:
        assert InterruptAction.KEEP.value == "keep"

    def test_discard_value(self) -> None:
        assert InterruptAction.DISCARD.value == "discard"

    def test_retry_value(self) -> None:
        assert InterruptAction.RETRY.value == "retry"

    def test_all_members(self) -> None:
        members = {m.name for m in InterruptAction}
        assert members == {"KEEP", "DISCARD", "RETRY"}

    def test_from_value(self) -> None:
        assert InterruptAction("keep") is InterruptAction.KEEP
        assert InterruptAction("discard") is InterruptAction.DISCARD
        assert InterruptAction("retry") is InterruptAction.RETRY

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            InterruptAction("invalid")


# ======================================================================
# PartialResponse
# ======================================================================


class TestPartialResponse:
    """PartialResponse dataclass fields and computed properties."""

    def test_basic_fields(self) -> None:
        pr = PartialResponse(
            content="Hello wor",
            model="gpt-4o",
            provider="openai",
            tokens_generated=5,
            was_interrupted=True,
        )
        assert pr.content == "Hello wor"
        assert pr.model == "gpt-4o"
        assert pr.provider == "openai"
        assert pr.tokens_generated == 5
        assert pr.was_interrupted is True
        assert pr.original_messages == []
        assert pr.tool_calls_in_progress is False

    def test_is_empty_with_content(self) -> None:
        pr = PartialResponse(
            content="Some text",
            model="gpt-4o",
            provider="openai",
            tokens_generated=3,
            was_interrupted=True,
        )
        assert pr.is_empty is False

    def test_is_empty_with_empty_string(self) -> None:
        pr = PartialResponse(
            content="",
            model="gpt-4o",
            provider="openai",
            tokens_generated=0,
            was_interrupted=True,
        )
        assert pr.is_empty is True

    def test_is_empty_with_whitespace_only(self) -> None:
        pr = PartialResponse(
            content="   \n\t  ",
            model="gpt-4o",
            provider="openai",
            tokens_generated=0,
            was_interrupted=True,
        )
        assert pr.is_empty is True

    def test_original_messages_preserved(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        pr = PartialResponse(
            content="Hi",
            model="gpt-4o",
            provider="openai",
            tokens_generated=1,
            was_interrupted=True,
            original_messages=msgs,
        )
        assert pr.original_messages == msgs

    def test_tool_calls_in_progress_flag(self) -> None:
        pr = PartialResponse(
            content="Calling func",
            model="gpt-4o",
            provider="openai",
            tokens_generated=2,
            was_interrupted=True,
            tool_calls_in_progress=True,
        )
        assert pr.tool_calls_in_progress is True


# ======================================================================
# InterruptionState
# ======================================================================


class TestInterruptionState:
    """InterruptionState defaults and attribute access."""

    def test_default_state(self) -> None:
        state = InterruptionState()
        assert state.interrupted is False
        assert state.partial is None
        assert state.action_taken is None
        assert state.resume_messages is None

    def test_all_fields_set(self) -> None:
        pr = PartialResponse(
            content="partial",
            model="gpt-4o",
            provider="openai",
            tokens_generated=2,
            was_interrupted=True,
        )
        resume = [{"role": "user", "content": "continue"}]
        state = InterruptionState(
            interrupted=True,
            partial=pr,
            action_taken=InterruptAction.KEEP,
            resume_messages=resume,
        )
        assert state.interrupted is True
        assert state.partial is pr
        assert state.action_taken == InterruptAction.KEEP
        assert state.resume_messages == resume

    def test_mutable_update(self) -> None:
        state = InterruptionState()
        state.interrupted = True
        state.action_taken = InterruptAction.DISCARD
        assert state.interrupted is True
        assert state.action_taken == InterruptAction.DISCARD


# ======================================================================
# StreamInterruptHandler
# ======================================================================


class TestStreamInterruptHandlerInit:
    """Initial state of a freshly created handler."""

    def test_not_interrupted_initially(self) -> None:
        handler = StreamInterruptHandler()
        assert handler.is_interrupted is False

    def test_state_is_clean(self) -> None:
        handler = StreamInterruptHandler()
        assert handler.state.interrupted is False
        assert handler.state.partial is None
        assert handler.state.action_taken is None

    def test_not_in_tool_execution_initially(self) -> None:
        handler = StreamInterruptHandler()
        assert handler.in_tool_execution is False


class TestStreamInterruptHandlerInstallUninstall:
    """Install and uninstall save/restore the SIGINT handler."""

    def test_install_saves_original_handler(self) -> None:
        handler = StreamInterruptHandler()
        original = signal.getsignal(signal.SIGINT)
        handler.install()
        try:
            # After install, the handler should be our custom one
            current = signal.getsignal(signal.SIGINT)
            assert current is not original
        finally:
            handler.uninstall()

    def test_uninstall_restores_original_handler(self) -> None:
        handler = StreamInterruptHandler()
        original = signal.getsignal(signal.SIGINT)
        handler.install()
        handler.uninstall()
        restored = signal.getsignal(signal.SIGINT)
        assert restored is original

    def test_uninstall_idempotent(self) -> None:
        handler = StreamInterruptHandler()
        original = signal.getsignal(signal.SIGINT)
        handler.install()
        handler.uninstall()
        handler.uninstall()  # second call should not error
        restored = signal.getsignal(signal.SIGINT)
        assert restored is original

    def test_install_clears_previous_state(self) -> None:
        handler = StreamInterruptHandler()
        # Simulate a previous interruption
        handler._interrupted.set()
        handler._state.interrupted = True
        handler.install()
        try:
            assert handler.is_interrupted is False
            assert handler.state.interrupted is False
        finally:
            handler.uninstall()


class TestStreamInterruptHandlerInterrupt:
    """Interrupt detection and the _handle_interrupt callback."""

    def test_check_interrupted_before_signal(self) -> None:
        handler = StreamInterruptHandler()
        assert handler.check_interrupted() is False

    def test_check_interrupted_after_signal(self) -> None:
        handler = StreamInterruptHandler()
        handler.install()
        try:
            # Simulate Ctrl+C by calling the handler directly
            handler._handle_interrupt(signal.SIGINT, None)
            assert handler.check_interrupted() is True
            assert handler.is_interrupted is True
        finally:
            handler.uninstall()

    def test_handle_interrupt_sets_flag(self) -> None:
        handler = StreamInterruptHandler()
        handler._handle_interrupt(signal.SIGINT, None)
        assert handler._interrupted.is_set()

    def test_deferred_interrupt_during_tool_execution(self) -> None:
        handler = StreamInterruptHandler()
        handler.enter_tool_execution()
        handler._handle_interrupt(signal.SIGINT, None)
        # Flag is still set even during tool execution
        assert handler.is_interrupted is True
        # But we're still in tool execution — caller should defer break
        assert handler.in_tool_execution is True


class TestStreamInterruptHandlerToolExecution:
    """Tool-execution enter/exit flag management."""

    def test_enter_sets_flag(self) -> None:
        handler = StreamInterruptHandler()
        handler.enter_tool_execution()
        assert handler.in_tool_execution is True

    def test_exit_clears_flag(self) -> None:
        handler = StreamInterruptHandler()
        handler.enter_tool_execution()
        handler.exit_tool_execution()
        assert handler.in_tool_execution is False

    def test_exit_without_enter_is_safe(self) -> None:
        handler = StreamInterruptHandler()
        handler.exit_tool_execution()
        assert handler.in_tool_execution is False

    def test_multiple_enters_single_exit(self) -> None:
        handler = StreamInterruptHandler()
        handler.enter_tool_execution()
        handler.enter_tool_execution()  # Event.set() is idempotent
        handler.exit_tool_execution()
        assert handler.in_tool_execution is False


class TestStreamInterruptHandlerSavePartial:
    """Saving partial responses after interruption."""

    def test_save_partial_creates_response(self) -> None:
        handler = StreamInterruptHandler()
        partial = handler.save_partial(
            content="Hello wor",
            model="gpt-4o",
            provider="openai",
            tokens_generated=5,
        )
        assert isinstance(partial, PartialResponse)
        assert partial.content == "Hello wor"
        assert partial.model == "gpt-4o"
        assert partial.provider == "openai"
        assert partial.tokens_generated == 5
        assert partial.was_interrupted is True

    def test_save_partial_updates_state(self) -> None:
        handler = StreamInterruptHandler()
        handler.save_partial(
            content="Some text",
            model="claude-3-opus",
            provider="anthropic",
            tokens_generated=3,
        )
        assert handler.state.interrupted is True
        assert handler.state.partial is not None
        assert handler.state.partial.content == "Some text"

    def test_save_partial_with_messages(self) -> None:
        handler = StreamInterruptHandler()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Write code"},
        ]
        partial = handler.save_partial(
            content="def hello():",
            model="gpt-4o",
            provider="openai",
            tokens_generated=4,
            messages=msgs,
        )
        assert partial.original_messages == msgs
        # Verify it's a copy, not the same list object
        assert partial.original_messages is not msgs

    def test_save_partial_with_tool_in_progress(self) -> None:
        handler = StreamInterruptHandler()
        partial = handler.save_partial(
            content="Calling search...",
            model="gpt-4o",
            provider="openai",
            tokens_generated=3,
            tool_in_progress=True,
        )
        assert partial.tool_calls_in_progress is True

    def test_save_partial_with_none_messages(self) -> None:
        handler = StreamInterruptHandler()
        partial = handler.save_partial(
            content="text",
            model="gpt-4o",
            provider="openai",
            tokens_generated=1,
            messages=None,
        )
        assert partial.original_messages == []

    def test_save_partial_empty_content(self) -> None:
        handler = StreamInterruptHandler()
        partial = handler.save_partial(
            content="",
            model="gpt-4o",
            provider="openai",
            tokens_generated=0,
        )
        assert partial.is_empty is True
        assert handler.state.interrupted is True


class TestStreamInterruptHandlerRecordAction:
    """Recording the user's post-interruption action."""

    def _setup_handler_with_partial(self) -> StreamInterruptHandler:
        """Helper: create a handler with a saved partial response."""
        handler = StreamInterruptHandler()
        handler.save_partial(
            content="Hello world, this is a partial",
            model="gpt-4o",
            provider="openai",
            tokens_generated=6,
            messages=[{"role": "user", "content": "say hello"}],
        )
        return handler

    def test_record_action_keep(self) -> None:
        handler = self._setup_handler_with_partial()
        handler.record_action(InterruptAction.KEEP)
        assert handler.state.action_taken == InterruptAction.KEEP

    def test_record_action_discard(self) -> None:
        handler = self._setup_handler_with_partial()
        handler.record_action(InterruptAction.DISCARD)
        assert handler.state.action_taken == InterruptAction.DISCARD

    def test_record_action_retry(self) -> None:
        handler = self._setup_handler_with_partial()
        handler.record_action(InterruptAction.RETRY)
        assert handler.state.action_taken == InterruptAction.RETRY

    def test_keep_builds_resume_messages(self) -> None:
        handler = self._setup_handler_with_partial()
        handler.record_action(InterruptAction.KEEP)
        resume = handler.state.resume_messages
        assert resume is not None
        assert len(resume) == 3  # original user + assistant partial + continue

    def test_discard_no_resume_messages(self) -> None:
        handler = self._setup_handler_with_partial()
        handler.record_action(InterruptAction.DISCARD)
        assert handler.state.resume_messages is None

    def test_retry_no_resume_messages(self) -> None:
        handler = self._setup_handler_with_partial()
        handler.record_action(InterruptAction.RETRY)
        assert handler.state.resume_messages is None


class TestStreamInterruptHandlerResumeMessages:
    """Verify the structure of resume messages after KEEP."""

    def test_resume_messages_structure(self) -> None:
        handler = StreamInterruptHandler()
        original_msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Write a poem"},
        ]
        handler.save_partial(
            content="Roses are red,\nViolets are blue,",
            model="gpt-4o",
            provider="openai",
            tokens_generated=8,
            messages=original_msgs,
        )
        handler.record_action(InterruptAction.KEEP)

        resume = handler.get_resume_messages()
        assert resume is not None
        assert len(resume) == 4

        # First two are the original messages
        assert resume[0] == {"role": "system", "content": "You are helpful."}
        assert resume[1] == {"role": "user", "content": "Write a poem"}

        # Third is the assistant's partial response
        assert resume[2]["role"] == "assistant"
        assert resume[2]["content"] == "Roses are red,\nViolets are blue,"

        # Fourth is the continuation prompt
        assert resume[3]["role"] == "user"
        assert "continue" in resume[3]["content"].lower()

    def test_get_resume_messages_returns_none_without_keep(self) -> None:
        handler = StreamInterruptHandler()
        assert handler.get_resume_messages() is None

    def test_get_resume_messages_returns_none_after_discard(self) -> None:
        handler = StreamInterruptHandler()
        handler.save_partial(
            content="partial",
            model="gpt-4o",
            provider="openai",
            tokens_generated=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        handler.record_action(InterruptAction.DISCARD)
        assert handler.get_resume_messages() is None

    def test_resume_messages_with_empty_original(self) -> None:
        handler = StreamInterruptHandler()
        handler.save_partial(
            content="partial text",
            model="gpt-4o",
            provider="openai",
            tokens_generated=2,
            messages=[],
        )
        handler.record_action(InterruptAction.KEEP)

        resume = handler.get_resume_messages()
        assert resume is not None
        assert len(resume) == 2  # assistant partial + continue prompt
        assert resume[0]["role"] == "assistant"
        assert resume[1]["role"] == "user"


class TestStreamInterruptHandlerShouldWriteFiles:
    """File-write safety gating."""

    def test_not_interrupted_allows_write(self) -> None:
        handler = StreamInterruptHandler()
        assert handler.should_write_files() is True

    def test_interrupted_keep_allows_write(self) -> None:
        handler = StreamInterruptHandler()
        handler.save_partial(
            content="code",
            model="gpt-4o",
            provider="openai",
            tokens_generated=1,
        )
        handler.record_action(InterruptAction.KEEP)
        assert handler.should_write_files() is True

    def test_interrupted_discard_blocks_write(self) -> None:
        handler = StreamInterruptHandler()
        handler.save_partial(
            content="code",
            model="gpt-4o",
            provider="openai",
            tokens_generated=1,
        )
        handler.record_action(InterruptAction.DISCARD)
        assert handler.should_write_files() is False

    def test_interrupted_retry_blocks_write(self) -> None:
        handler = StreamInterruptHandler()
        handler.save_partial(
            content="code",
            model="gpt-4o",
            provider="openai",
            tokens_generated=1,
        )
        handler.record_action(InterruptAction.RETRY)
        assert handler.should_write_files() is False

    def test_interrupted_no_action_blocks_write(self) -> None:
        handler = StreamInterruptHandler()
        handler.save_partial(
            content="code",
            model="gpt-4o",
            provider="openai",
            tokens_generated=1,
        )
        # No action recorded yet — state.action_taken is None, not KEEP
        assert handler.should_write_files() is False


class TestStreamInterruptHandlerReset:
    """Reset clears all state."""

    def test_reset_clears_interrupted(self) -> None:
        handler = StreamInterruptHandler()
        handler._interrupted.set()
        handler.reset()
        assert handler.is_interrupted is False

    def test_reset_clears_tool_execution(self) -> None:
        handler = StreamInterruptHandler()
        handler.enter_tool_execution()
        handler.reset()
        assert handler.in_tool_execution is False

    def test_reset_clears_state(self) -> None:
        handler = StreamInterruptHandler()
        handler.save_partial(
            content="text",
            model="gpt-4o",
            provider="openai",
            tokens_generated=1,
        )
        handler.record_action(InterruptAction.KEEP)

        handler.reset()
        assert handler.state.interrupted is False
        assert handler.state.partial is None
        assert handler.state.action_taken is None
        assert handler.state.resume_messages is None

    def test_multiple_interruptions_with_reset(self) -> None:
        handler = StreamInterruptHandler()

        # First interruption
        handler._handle_interrupt(signal.SIGINT, None)
        handler.save_partial(
            content="first partial",
            model="gpt-4o",
            provider="openai",
            tokens_generated=2,
        )
        handler.record_action(InterruptAction.DISCARD)
        assert handler.state.action_taken == InterruptAction.DISCARD

        # Reset for second operation
        handler.reset()
        assert handler.is_interrupted is False
        assert handler.state.partial is None

        # Second interruption
        handler._handle_interrupt(signal.SIGINT, None)
        handler.save_partial(
            content="second partial",
            model="claude-3-opus",
            provider="anthropic",
            tokens_generated=3,
        )
        handler.record_action(InterruptAction.KEEP)
        assert handler.state.partial is not None
        assert handler.state.partial.content == "second partial"
        assert handler.state.partial.provider == "anthropic"
        assert handler.state.action_taken == InterruptAction.KEEP


class TestStreamInterruptHandlerBuildResumeNoPartial:
    """Edge case: _build_resume_messages with no partial."""

    def test_build_resume_no_partial_returns_empty(self) -> None:
        handler = StreamInterruptHandler()
        result = handler._build_resume_messages()
        assert result == []


# ======================================================================
# prompt_interrupt_action
# ======================================================================


class TestPromptInterruptAction:
    """Test the interactive prompt (all I/O mocked)."""

    @patch("builtins.input", return_value="k")
    def test_keep_shorthand(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.KEEP

    @patch("builtins.input", return_value="keep")
    def test_keep_full_word(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.KEEP

    @patch("builtins.input", return_value="d")
    def test_discard_shorthand(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.DISCARD

    @patch("builtins.input", return_value="discard")
    def test_discard_full_word(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.DISCARD

    @patch("builtins.input", return_value="r")
    def test_retry_shorthand(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.RETRY

    @patch("builtins.input", return_value="retry")
    def test_retry_full_word(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.RETRY

    @patch("builtins.input", return_value="K")
    def test_case_insensitive(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.KEEP

    @patch("builtins.input", return_value="  r  ")
    def test_whitespace_stripped(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.RETRY

    @patch("builtins.input", side_effect=["x", "invalid", "d"])
    def test_invalid_input_retries(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.DISCARD
        assert mock_input.call_count == 3

    @patch("builtins.input", side_effect=EOFError)
    def test_eof_returns_discard(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.DISCARD

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_returns_discard(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.DISCARD

    @patch("builtins.input", side_effect=["", "k"])
    def test_empty_input_retries(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.KEEP
        assert mock_input.call_count == 2

    @patch("builtins.input", return_value="KEEP")
    def test_full_word_case_insensitive(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.KEEP

    @patch("builtins.input", return_value="DISCARD")
    def test_discard_uppercase(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.DISCARD

    @patch("builtins.input", return_value="RETRY")
    def test_retry_uppercase(self, mock_input: MagicMock) -> None:
        action = prompt_interrupt_action()
        assert action == InterruptAction.RETRY

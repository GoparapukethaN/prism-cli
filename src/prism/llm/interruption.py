"""Response streaming interruption — clean Ctrl+C handling with partial response preservation.

Provides:
- ``StreamInterruptHandler``: installs a SIGINT handler that sets an
  interruption flag checked on every streaming chunk, defers interruption
  if an atomic tool execution is in progress, and saves the partial
  response for the user to keep, discard, or retry.
- ``PartialResponse``: lightweight container for the text and metadata
  accumulated before the stream was interrupted.
- ``InterruptAction``: enum representing the user's post-interruption
  choice (keep / discard / retry).
- ``prompt_interrupt_action``: blocking prompt that asks the user what
  to do with the partial response.
"""

from __future__ import annotations

import signal
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class InterruptAction(Enum):
    """User-chosen action after a streaming interruption."""

    KEEP = "keep"
    DISCARD = "discard"
    RETRY = "retry"


@dataclass
class PartialResponse:
    """A partially completed LLM response captured after Ctrl+C.

    Attributes:
        content: The text accumulated before interruption.
        model: LiteLLM model identifier that was streaming.
        provider: Provider name (e.g. ``"openai"``).
        tokens_generated: Approximate output tokens produced so far.
        was_interrupted: Always ``True`` for captured partials.
        original_messages: The messages list that initiated the request,
            preserved so the request can be resumed or retried.
        tool_calls_in_progress: ``True`` if a tool call was being
            assembled when the interruption occurred.
    """

    content: str
    model: str
    provider: str
    tokens_generated: int
    was_interrupted: bool
    original_messages: list[dict] = field(default_factory=list)
    tool_calls_in_progress: bool = False

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if the partial response contains no meaningful text."""
        return not self.content.strip()


@dataclass
class InterruptionState:
    """Tracks the full lifecycle of a single stream interruption.

    Attributes:
        interrupted: ``True`` once Ctrl+C has been handled.
        partial: The captured partial response (if any).
        action_taken: The user's chosen action after the prompt.
        resume_messages: Messages list for continuing generation after
            the user chose *keep*.
    """

    interrupted: bool = False
    partial: PartialResponse | None = None
    action_taken: InterruptAction | None = None
    resume_messages: list[dict] | None = None


class StreamInterruptHandler:
    """Handles Ctrl+C during streaming with clean interruption and user choice.

    Typical lifecycle::

        handler = StreamInterruptHandler()
        handler.install()
        try:
            async for chunk in stream:
                if handler.check_interrupted():
                    handler.save_partial(...)
                    break
                # ... process chunk ...
        finally:
            handler.uninstall()

        if handler.is_interrupted:
            action = prompt_interrupt_action()
            handler.record_action(action)
    """

    def __init__(self) -> None:
        self._interrupted = threading.Event()
        self._in_tool_execution = threading.Event()
        self._original_handler: Any = None
        self._state = InterruptionState()
        self._lock = threading.Lock()
        self._active = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_interrupted(self) -> bool:
        """``True`` if Ctrl+C was received while this handler was active."""
        return self._interrupted.is_set()

    @property
    def state(self) -> InterruptionState:
        """Current interruption state (read-only snapshot)."""
        return self._state

    @property
    def in_tool_execution(self) -> bool:
        """``True`` if an atomic tool execution is in progress."""
        return self._in_tool_execution.is_set()

    # ------------------------------------------------------------------
    # Install / uninstall
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Install the custom SIGINT handler for stream interruption.

        Saves the previous handler so it can be restored by
        :meth:`uninstall`.
        """
        self._interrupted.clear()
        self._state = InterruptionState()
        self._active = True
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)
        logger.debug("interrupt_handler_installed")

    def uninstall(self) -> None:
        """Restore the previous SIGINT handler.

        Safe to call multiple times — only restores once.
        """
        self._active = False
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
            self._original_handler = None
        logger.debug("interrupt_handler_uninstalled")

    # ------------------------------------------------------------------
    # Signal handler
    # ------------------------------------------------------------------

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """SIGINT handler — sets the interrupted flag.

        If an atomic tool execution is in progress the flag is still
        set, but the streaming loop should defer the actual break
        until :meth:`exit_tool_execution` is called.

        Args:
            signum: Signal number (always ``signal.SIGINT``).
            frame: Current stack frame (unused).
        """
        if self._in_tool_execution.is_set():
            logger.info("interrupt_deferred_tool_execution")

        self._interrupted.set()
        logger.info("stream_interrupted")

    # ------------------------------------------------------------------
    # Interruption check
    # ------------------------------------------------------------------

    def check_interrupted(self) -> bool:
        """Check if interruption was requested.

        Designed to be called on every iteration of the streaming loop.

        Returns:
            ``True`` if Ctrl+C was received.
        """
        return self._interrupted.is_set()

    # ------------------------------------------------------------------
    # Tool-execution guards
    # ------------------------------------------------------------------

    def enter_tool_execution(self) -> None:
        """Mark the start of an atomic tool execution.

        While this flag is set the streaming loop should defer
        breaking even if :meth:`check_interrupted` returns ``True``.
        """
        self._in_tool_execution.set()

    def exit_tool_execution(self) -> None:
        """Mark the end of an atomic tool execution."""
        self._in_tool_execution.clear()

    # ------------------------------------------------------------------
    # Partial response management
    # ------------------------------------------------------------------

    def save_partial(
        self,
        content: str,
        model: str,
        provider: str,
        tokens_generated: int,
        messages: list[dict] | None = None,
        tool_in_progress: bool = False,
    ) -> PartialResponse:
        """Save the partial response from an interrupted stream.

        Args:
            content: Text accumulated so far.
            model: LiteLLM model identifier.
            provider: Provider name.
            tokens_generated: Approximate output tokens generated.
            messages: Original messages list for retry/resume.
            tool_in_progress: ``True`` if a tool call was being built.

        Returns:
            The :class:`PartialResponse` that was saved.
        """
        partial = PartialResponse(
            content=content,
            model=model,
            provider=provider,
            tokens_generated=tokens_generated,
            was_interrupted=True,
            original_messages=list(messages or []),
            tool_calls_in_progress=tool_in_progress,
        )

        with self._lock:
            self._state.interrupted = True
            self._state.partial = partial

        logger.info(
            "partial_response_saved",
            tokens=tokens_generated,
            content_length=len(content),
        )
        return partial

    # ------------------------------------------------------------------
    # Action recording
    # ------------------------------------------------------------------

    def record_action(self, action: InterruptAction) -> None:
        """Record the user's chosen action for the partial response.

        If the action is :attr:`InterruptAction.KEEP`, resume messages
        are built automatically so the caller can continue the
        generation later.

        Args:
            action: The action chosen by the user.
        """
        with self._lock:
            self._state.action_taken = action

        if action == InterruptAction.KEEP and self._state.partial:
            self._state.resume_messages = self._build_resume_messages()

        logger.info("interrupt_action", action=action.value)

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------

    def _build_resume_messages(self) -> list[dict]:
        """Build a messages list for resuming after a kept partial response.

        The list contains the original messages, the partial assistant
        response, and a continuation prompt.

        Returns:
            A new messages list suitable for passing to the completion
            engine, or an empty list if no partial exists.
        """
        if not self._state.partial:
            return []

        partial = self._state.partial
        messages: list[dict] = list(partial.original_messages)

        # Add the partial response as an assistant message
        messages.append({
            "role": "assistant",
            "content": partial.content,
        })

        # Add a continuation prompt
        messages.append({
            "role": "user",
            "content": "Please continue from where you left off.",
        })

        return messages

    def get_resume_messages(self) -> list[dict] | None:
        """Get messages for resuming after a kept partial response.

        Returns:
            Messages list if available, otherwise ``None``.
        """
        return self._state.resume_messages

    # ------------------------------------------------------------------
    # File-write safety
    # ------------------------------------------------------------------

    def should_write_files(self) -> bool:
        """Check if it is safe to write files.

        Files should **never** be written from an interrupted stream
        unless the user explicitly chose to keep the partial response.

        Returns:
            ``True`` if file writes are safe.
        """
        if not self._state.interrupted:
            return True
        return self._state.action_taken == InterruptAction.KEEP

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the handler state for a new streaming operation.

        Clears all flags and the saved partial response.  Does **not**
        change the installed signal handler.
        """
        self._interrupted.clear()
        self._in_tool_execution.clear()
        self._state = InterruptionState()


def prompt_interrupt_action() -> InterruptAction:
    """Prompt the user for an action after a streaming interruption.

    Displays a menu and blocks until the user chooses one of:

    * **k** — keep the partial response
    * **d** — discard the response
    * **r** — retry the request from scratch

    Returns:
        The :class:`InterruptAction` corresponding to the user's choice.
        Returns :attr:`InterruptAction.DISCARD` on ``EOF`` or a second
        ``KeyboardInterrupt``.
    """
    sys.stdout.write("\n[Stream interrupted]\n")
    sys.stdout.write("  [k] Keep partial response\n")
    sys.stdout.write("  [d] Discard response\n")
    sys.stdout.write("  [r] Retry from beginning\n")
    sys.stdout.flush()

    while True:
        try:
            choice = input("\nChoice (k/d/r): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return InterruptAction.DISCARD

        if choice in ("k", "keep"):
            return InterruptAction.KEEP
        if choice in ("d", "discard"):
            return InterruptAction.DISCARD
        if choice in ("r", "retry"):
            return InterruptAction.RETRY

        sys.stdout.write("Invalid choice. Enter k, d, or r.\n")
        sys.stdout.flush()

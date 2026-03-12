"""Keyboard shortcuts for the Prism REPL.

Provides a factory function that creates prompt_toolkit key bindings
with all the standard shortcuts expected in an interactive CLI:

- ``Ctrl+C``: Cancel / clear current input
- ``Ctrl+L``: Clear screen
- ``Ctrl+U``: Clear line before cursor
- ``Ctrl+K``: Clear line after cursor (kill to end)
- ``Ctrl+A``: Move cursor to start of line
- ``Ctrl+E``: Move cursor to end of line
- ``Ctrl+R``: Reverse history search (built-in via prompt_toolkit)
- ``Ctrl+D``: Exit REPL (handled by prompt_toolkit EOFError)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from prompt_toolkit.key_binding import KeyBindings

if TYPE_CHECKING:
    from rich.console import Console

logger = structlog.get_logger(__name__)


def create_keybindings(state: Any, console: Console) -> KeyBindings:
    """Create all key bindings for the Prism REPL.

    Args:
        state:   The REPL session state (``_SessionState``).  Currently
                 unused but available for future bindings that need to
                 inspect or mutate session state.
        console: The Rich console used for output.  Used by ``Ctrl+L``
                 to clear the terminal.

    Returns:
        A :class:`KeyBindings` instance ready to pass to
        :class:`PromptSession`.
    """
    bindings = KeyBindings()

    @bindings.add("c-c")
    def _handle_ctrl_c(event: Any) -> None:
        """Cancel current input — clear the buffer without exiting."""
        event.app.current_buffer.reset()

    @bindings.add("c-l")
    def _handle_ctrl_l(event: Any) -> None:
        """Clear the terminal screen."""
        console.clear()
        event.app.renderer.clear()

    @bindings.add("c-u")
    def _handle_ctrl_u(event: Any) -> None:
        """Clear the line from the cursor position to the start."""
        buf = event.app.current_buffer
        buf.delete_before_cursor(count=len(buf.document.text_before_cursor))

    @bindings.add("c-k")
    def _handle_ctrl_k(event: Any) -> None:
        """Clear the line from the cursor position to the end."""
        buf = event.app.current_buffer
        text_after = buf.document.text_after_cursor
        if text_after:
            buf.delete(count=len(text_after))

    @bindings.add("c-a")
    def _handle_ctrl_a(event: Any) -> None:
        """Move cursor to the start of the line."""
        buf = event.app.current_buffer
        buf.cursor_position = 0

    @bindings.add("c-e")
    def _handle_ctrl_e(event: Any) -> None:
        """Move cursor to the end of the line."""
        buf = event.app.current_buffer
        buf.cursor_position = len(buf.text)

    logger.debug("keybindings_created", count=len(bindings.bindings))

    return bindings

"""Streaming token display handler for the REPL.

Uses Rich's Live display to progressively render Markdown as tokens arrive.
Shows a spinner while waiting for the first token, then switches to
streaming Markdown with proper word wrapping, syntax highlighting, and
colored formatting. Response lines are prefixed with a bullet indicator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from rich.console import Console

logger = structlog.get_logger(__name__)


def _render_response(text: str) -> object:
    """Render AI response text with a ● prefix like Claude CLI.

    Returns a Rich renderable that shows a cyan ● bullet on the first
    line, with the response rendered as Markdown and indented to align.
    """
    from rich.console import Group
    from rich.markdown import Markdown
    from rich.text import Text

    if not text:
        return Text("● ", style="bold bright_cyan")

    # Split into first line and rest for the bullet alignment
    bullet = Text("● ", style="bold bright_cyan")
    md = Markdown(text)
    return Group(bullet, md)


class StreamHandler:
    """Handles streaming token display with Rich Markdown rendering.

    Shows a "Thinking..." spinner until the first token arrives, then
    progressively renders Markdown in a Rich Live display with
    per-token refresh for a visible typing effect.

    Args:
        console: The Rich console for rendering.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self.buffer: str = ""
        self._token_count: int = 0
        self._thinking: bool = False
        self._streaming: bool = False
        self._live: object | None = None

    def show_thinking(self) -> None:
        """Show a spinner while waiting for the first token."""
        from rich.live import Live
        from rich.spinner import Spinner

        self._live = Live(
            Spinner("dots", text="[dim] Thinking...[/dim]"),
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()
        self._thinking = True

    def on_token(self, token: str) -> None:
        """Called for each streaming token delta.

        On first token: replaces spinner with Live Markdown display.
        On subsequent tokens: updates the Markdown render and forces
        an immediate refresh for a visible typing effect.
        """
        if not token:
            return

        # First token — switch from spinner to markdown streaming
        if not self._streaming:
            if self._live is not None:
                self._live.stop()
                self._live = None

            from rich.live import Live

            self._live = Live(
                _render_response(""),
                console=self.console,
                refresh_per_second=30,
                vertical_overflow="visible",
            )
            self._live.start()
            self._streaming = True
            self._thinking = False

        self.buffer += token
        self._token_count += 1

        # Update live display and force immediate refresh for typing effect
        if self._live is not None:
            self._live.update(_render_response(self.buffer))
            self._live.refresh()

    def finalize(self) -> str:
        """Stop the Live display and return the full content."""
        if self._live is not None:
            if self._streaming:
                self._live.update(_render_response(self.buffer))
            self._live.stop()
            self._live = None

        return self.buffer

    @property
    def token_count(self) -> int:
        """Number of tokens received so far."""
        return self._token_count

    @property
    def has_content(self) -> bool:
        """Whether any content has been received."""
        return len(self.buffer) > 0

    def reset(self) -> None:
        """Reset the handler for reuse."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        self.buffer = ""
        self._token_count = 0
        self._thinking = False
        self._streaming = False

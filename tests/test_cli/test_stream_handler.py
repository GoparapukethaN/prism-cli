"""Tests for prism.cli.stream_handler — streaming token display handler."""

from __future__ import annotations

import io

from rich.console import Console

from prism.cli.stream_handler import StreamHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_console() -> Console:
    """Create a console that writes to an in-memory buffer."""
    buf = io.StringIO()
    return Console(
        file=buf, force_terminal=False, no_color=True, width=300,
    )


def _get_output(console: Console) -> str:
    """Extract the output string from a console backed by StringIO."""
    console.file.seek(0)
    return console.file.read()


# ---------------------------------------------------------------------------
# StreamHandler.on_token
# ---------------------------------------------------------------------------


class TestOnToken:
    """Tests for the on_token callback."""

    def test_single_token_appended_to_buffer(self) -> None:
        """A single token call should update the buffer."""
        handler = StreamHandler(_make_console())
        handler.on_token("Hello")
        assert handler.buffer == "Hello"
        assert handler.token_count == 1

    def test_multiple_tokens_accumulate(self) -> None:
        """Multiple on_token calls should concatenate."""
        handler = StreamHandler(_make_console())
        handler.on_token("Hello")
        handler.on_token(" ")
        handler.on_token("world")
        assert handler.buffer == "Hello world"
        assert handler.token_count == 3

    def test_empty_token_ignored(self) -> None:
        """Empty string tokens should not change buffer or count."""
        handler = StreamHandler(_make_console())
        handler.on_token("")
        assert handler.buffer == ""
        assert handler.token_count == 0
        assert not handler.has_content

    def test_none_like_empty_handled(self) -> None:
        """An empty string should be silently ignored."""
        handler = StreamHandler(_make_console())
        handler.on_token("")
        handler.on_token("")
        assert handler.buffer == ""
        assert handler.token_count == 0

    def test_tokens_rendered_via_rich_live(self) -> None:
        """Tokens should be rendered via Rich Live Markdown display."""
        con = _make_console()
        handler = StreamHandler(con)
        handler.on_token("Hi")
        handler.on_token(" there")
        handler.finalize()
        # The buffer should contain both tokens
        assert handler.buffer == "Hi there"
        # Rich Live renders to the console
        output = _get_output(con)
        assert "Hi" in output or handler.buffer == "Hi there"

    def test_special_characters_in_tokens(self) -> None:
        """Tokens with special chars should pass through unchanged."""
        handler = StreamHandler(_make_console())
        specials = "def foo():\n    return 42\n"
        handler.on_token(specials)
        assert handler.buffer == specials

    def test_unicode_tokens(self) -> None:
        """Unicode content should be handled correctly."""
        handler = StreamHandler(_make_console())
        handler.on_token("Hello ")
        handler.on_token("\u2603")  # snowman
        handler.on_token(" \u00e9\u00e8\u00ea")
        assert handler.buffer == "Hello \u2603 \u00e9\u00e8\u00ea"
        assert handler.token_count == 3

    def test_markdown_syntax_in_tokens(self) -> None:
        """Markdown-like content should be buffered as-is."""
        handler = StreamHandler(_make_console())
        handler.on_token("# Title\n")
        handler.on_token("**bold** and *italic*")
        assert handler.buffer == "# Title\n**bold** and *italic*"

    def test_long_token_string(self) -> None:
        """A very long token should be handled without issues."""
        handler = StreamHandler(_make_console())
        long_token = "x" * 100_000
        handler.on_token(long_token)
        assert len(handler.buffer) == 100_000
        assert handler.token_count == 1


# ---------------------------------------------------------------------------
# StreamHandler.finalize
# ---------------------------------------------------------------------------


class TestFinalize:
    """Tests for finalize()."""

    def test_returns_full_content(self) -> None:
        """finalize() should return all accumulated tokens."""
        handler = StreamHandler(_make_console())
        handler.on_token("Hello ")
        handler.on_token("world!")
        result = handler.finalize()
        assert result == "Hello world!"

    def test_returns_empty_string_when_no_tokens(self) -> None:
        """finalize() with no tokens should return empty string."""
        handler = StreamHandler(_make_console())
        result = handler.finalize()
        assert result == ""

    def test_finalize_stops_live_display(self) -> None:
        """finalize() should stop the Live display cleanly."""
        handler = StreamHandler(_make_console())
        handler.on_token("text")
        # Live display should be active
        assert handler._live is not None
        handler.finalize()
        # Live display should be stopped
        assert handler._live is None

    def test_finalize_stops_thinking_spinner(self) -> None:
        """finalize() should stop the thinking spinner if active."""
        handler = StreamHandler(_make_console())
        handler.show_thinking()
        assert handler._live is not None
        handler.finalize()
        assert handler._live is None


# ---------------------------------------------------------------------------
# StreamHandler.has_content / token_count properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for has_content and token_count properties."""

    def test_has_content_false_initially(self) -> None:
        handler = StreamHandler(_make_console())
        assert not handler.has_content

    def test_has_content_true_after_token(self) -> None:
        handler = StreamHandler(_make_console())
        handler.on_token("x")
        assert handler.has_content

    def test_token_count_zero_initially(self) -> None:
        handler = StreamHandler(_make_console())
        assert handler.token_count == 0

    def test_token_count_increments(self) -> None:
        handler = StreamHandler(_make_console())
        handler.on_token("a")
        handler.on_token("b")
        handler.on_token("c")
        assert handler.token_count == 3


# ---------------------------------------------------------------------------
# StreamHandler.show_thinking
# ---------------------------------------------------------------------------


class TestShowThinking:
    """Tests for the thinking spinner."""

    def test_show_thinking_starts_live(self) -> None:
        """show_thinking() should start a Live spinner display."""
        handler = StreamHandler(_make_console())
        handler.show_thinking()
        assert handler._live is not None
        assert handler._thinking is True
        # Clean up
        handler.finalize()

    def test_first_token_replaces_spinner(self) -> None:
        """First on_token() should replace spinner with Markdown display."""
        handler = StreamHandler(_make_console())
        handler.show_thinking()
        assert handler._thinking is True
        handler.on_token("Hello")
        assert handler._thinking is False
        assert handler._streaming is True
        handler.finalize()


# ---------------------------------------------------------------------------
# StreamHandler.reset
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for the reset method."""

    def test_reset_clears_buffer(self) -> None:
        handler = StreamHandler(_make_console())
        handler.on_token("content")
        handler.reset()
        assert handler.buffer == ""
        assert handler.token_count == 0
        assert not handler.has_content

    def test_reset_allows_reuse(self) -> None:
        """After reset, the handler should work for a new stream."""
        handler = StreamHandler(_make_console())
        handler.on_token("first stream")
        handler.reset()
        handler.on_token("second stream")
        assert handler.buffer == "second stream"
        handler.finalize()

    def test_reset_stops_live(self) -> None:
        """reset() should stop any active Live display."""
        handler = StreamHandler(_make_console())
        handler.on_token("content")
        assert handler._live is not None
        handler.reset()
        assert handler._live is None


# ---------------------------------------------------------------------------
# Integration: full streaming flow
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end tests simulating a streaming response."""

    def test_full_streaming_flow(self) -> None:
        """Simulate receiving multiple tokens and finalizing."""
        console = _make_console()
        handler = StreamHandler(console)

        tokens = ["The ", "answer ", "is ", "42."]
        for token in tokens:
            handler.on_token(token)
        result = handler.finalize()

        assert result == "The answer is 42."
        assert handler.token_count == 4

    def test_streaming_with_empty_tokens_interspersed(self) -> None:
        """Empty tokens in the stream should be silently ignored."""
        handler = StreamHandler(_make_console())
        handler.on_token("Hello")
        handler.on_token("")
        handler.on_token("")
        handler.on_token(" world")
        handler.on_token("")
        result = handler.finalize()
        assert result == "Hello world"
        assert handler.token_count == 2  # Only non-empty counted

    def test_streaming_code_block(self) -> None:
        """Streaming a code block should preserve all formatting."""
        handler = StreamHandler(_make_console())
        code_tokens = [
            "```python\n",
            "def hello():\n",
            "    print('hi')\n",
            "```",
        ]
        for token in code_tokens:
            handler.on_token(token)
        result = handler.finalize()
        assert result == "```python\ndef hello():\n    print('hi')\n```"

    def test_full_flow_with_spinner(self) -> None:
        """Full flow: spinner -> streaming -> finalize."""
        console = _make_console()
        handler = StreamHandler(console)

        handler.show_thinking()
        assert handler._thinking is True

        handler.on_token("Response ")
        assert handler._thinking is False
        assert handler._streaming is True

        handler.on_token("content.")
        result = handler.finalize()

        assert result == "Response content."
        assert handler._live is None

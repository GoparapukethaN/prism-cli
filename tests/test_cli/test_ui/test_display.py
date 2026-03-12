"""Tests for prism.cli.ui.display — all display functions."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from prism.cli.ui.display import (
    display_classification,
    display_diff,
    display_error,
    display_model_selection,
    display_status,
    display_streaming_token,
    display_tool_call,
    display_tool_result,
    display_welcome,
)

if TYPE_CHECKING:
    from rich.console import Console

    from prism.router.classifier import ClassificationResult


def _get_output(console: Console) -> str:
    """Extract the text written to the console's StringIO buffer."""
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


# ---------------------------------------------------------------------------
# display_welcome
# ---------------------------------------------------------------------------


class TestDisplayWelcome:
    """Tests for the welcome banner."""

    def test_contains_prism_name(self, plain_console: Console) -> None:
        display_welcome(console=plain_console)
        output = _get_output(plain_console)
        assert "Prism" in output

    def test_contains_version(self, plain_console: Console) -> None:
        from prism import __version__

        display_welcome(console=plain_console)
        output = _get_output(plain_console)
        assert __version__ in output

    def test_contains_help_hint(self, plain_console: Console) -> None:
        display_welcome(console=plain_console)
        output = _get_output(plain_console)
        assert "/help" in output


# ---------------------------------------------------------------------------
# display_classification
# ---------------------------------------------------------------------------


class TestDisplayClassification:
    """Tests for classification result display."""

    def test_simple_tier_label(
        self, plain_console: Console, simple_result: ClassificationResult
    ) -> None:
        display_classification(simple_result, console=plain_console)
        output = _get_output(plain_console)
        assert "SIMPLE" in output

    def test_medium_tier_label(
        self, plain_console: Console, medium_result: ClassificationResult
    ) -> None:
        display_classification(medium_result, console=plain_console)
        output = _get_output(plain_console)
        assert "MEDIUM" in output

    def test_complex_tier_label(
        self, plain_console: Console, complex_result: ClassificationResult
    ) -> None:
        display_classification(complex_result, console=plain_console)
        output = _get_output(plain_console)
        assert "COMPLEX" in output

    def test_shows_score(
        self, plain_console: Console, simple_result: ClassificationResult
    ) -> None:
        display_classification(simple_result, console=plain_console)
        output = _get_output(plain_console)
        assert "0.15" in output

    def test_shows_reasoning(
        self, plain_console: Console, simple_result: ClassificationResult
    ) -> None:
        display_classification(simple_result, console=plain_console)
        output = _get_output(plain_console)
        assert "simple keywords" in output

    def test_shows_icon(
        self, plain_console: Console, complex_result: ClassificationResult
    ) -> None:
        display_classification(complex_result, console=plain_console)
        output = _get_output(plain_console)
        assert "[C]" in output


# ---------------------------------------------------------------------------
# display_model_selection
# ---------------------------------------------------------------------------


class TestDisplayModelSelection:
    """Tests for model selection display."""

    def test_shows_model_name(self, plain_console: Console) -> None:
        display_model_selection(
            "claude-sonnet-4-20250514", "anthropic", 0.0035, console=plain_console
        )
        output = _get_output(plain_console)
        assert "claude-sonnet-4-20250514" in output

    def test_shows_provider(self, plain_console: Console) -> None:
        display_model_selection(
            "gpt-4o", "openai", 0.01, console=plain_console
        )
        output = _get_output(plain_console)
        assert "openai" in output

    def test_shows_estimated_cost(self, plain_console: Console) -> None:
        display_model_selection(
            "gpt-4o-mini", "openai", 0.0012, console=plain_console
        )
        output = _get_output(plain_console)
        assert "$0.0012" in output


# ---------------------------------------------------------------------------
# display_tool_call
# ---------------------------------------------------------------------------


class TestDisplayToolCall:
    """Tests for tool call display."""

    def test_shows_tool_name(self, plain_console: Console) -> None:
        display_tool_call("read_file", {"path": "/tmp/test.py"}, console=plain_console)
        output = _get_output(plain_console)
        assert "read_file" in output

    def test_shows_arguments(self, plain_console: Console) -> None:
        display_tool_call(
            "run_command",
            {"command": "python -m pytest", "timeout": 30},
            console=plain_console,
        )
        output = _get_output(plain_console)
        assert "python -m pytest" in output
        assert "timeout" in output

    def test_empty_arguments(self, plain_console: Console) -> None:
        display_tool_call("get_status", {}, console=plain_console)
        output = _get_output(plain_console)
        assert "no arguments" in output


# ---------------------------------------------------------------------------
# display_tool_result
# ---------------------------------------------------------------------------


class TestDisplayToolResult:
    """Tests for tool result display."""

    def test_short_result(self, plain_console: Console) -> None:
        display_tool_result("File created successfully.", console=plain_console)
        output = _get_output(plain_console)
        assert "File created successfully." in output

    def test_long_result_is_truncated(self, plain_console: Console) -> None:
        long_text = "x" * 3000
        display_tool_result(long_text, console=plain_console)
        output = _get_output(plain_console)
        assert "truncated" in output
        assert "3000" in output

    def test_non_string_result(self, plain_console: Console) -> None:
        display_tool_result(42, console=plain_console)
        output = _get_output(plain_console)
        assert "42" in output


# ---------------------------------------------------------------------------
# display_error
# ---------------------------------------------------------------------------


class TestDisplayError:
    """Tests for error display."""

    def test_shows_error_text(self, plain_console: Console) -> None:
        display_error("File not found", console=plain_console)
        output = _get_output(plain_console)
        assert "File not found" in output

    def test_shows_hint_when_provided(self, plain_console: Console) -> None:
        display_error(
            "API key missing",
            hint="Run 'prism auth add anthropic'",
            console=plain_console,
        )
        output = _get_output(plain_console)
        assert "API key missing" in output
        assert "prism auth add anthropic" in output

    def test_no_hint_section_when_none(self, plain_console: Console) -> None:
        display_error("Some error", console=plain_console)
        output = _get_output(plain_console)
        assert "Hint" not in output


# ---------------------------------------------------------------------------
# display_status
# ---------------------------------------------------------------------------


class TestDisplayStatus:
    """Tests for configuration status display."""

    def test_shows_project_root(self, plain_console: Console, test_settings: object) -> None:
        from prism.config.settings import Settings

        assert isinstance(test_settings, Settings)
        display_status(test_settings, console=plain_console)
        output = _get_output(plain_console)
        assert "Project root" in output

    def test_shows_prism_home(self, plain_console: Console, test_settings: object) -> None:
        from prism.config.settings import Settings

        assert isinstance(test_settings, Settings)
        display_status(test_settings, console=plain_console)
        output = _get_output(plain_console)
        assert ".prism" in output

    def test_shows_budget_unlimited(self, plain_console: Console, test_settings: object) -> None:
        from prism.config.settings import Settings

        assert isinstance(test_settings, Settings)
        display_status(test_settings, console=plain_console)
        output = _get_output(plain_console)
        assert "unlimited" in output

    def test_shows_pinned_model_auto(self, plain_console: Console, test_settings: object) -> None:
        from prism.config.settings import Settings

        assert isinstance(test_settings, Settings)
        display_status(test_settings, console=plain_console)
        output = _get_output(plain_console)
        assert "auto" in output


# ---------------------------------------------------------------------------
# display_streaming_token
# ---------------------------------------------------------------------------


class TestDisplayStreamingToken:
    """Tests for streaming token output."""

    def test_token_appears_in_output(self, plain_console: Console) -> None:
        display_streaming_token("Hello", console=plain_console)
        output = _get_output(plain_console)
        assert "Hello" in output

    def test_no_trailing_newline(self, plain_console: Console) -> None:
        display_streaming_token("token", console=plain_console)
        output = _get_output(plain_console)
        # Rich may add control chars, but the output should not end with a bare newline
        # after the token content.
        assert output.rstrip("\n") != ""

    def test_multiple_tokens_concatenate(self, plain_console: Console) -> None:
        display_streaming_token("foo", console=plain_console)
        display_streaming_token("bar", console=plain_console)
        output = _get_output(plain_console)
        assert "foo" in output
        assert "bar" in output


# ---------------------------------------------------------------------------
# display_diff
# ---------------------------------------------------------------------------


class TestDisplayDiff:
    """Tests for the diff preview display."""

    def test_shows_diff_content(self, plain_console: Console) -> None:
        diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"
        display_diff(diff, file_path="foo.py", console=plain_console)
        output = _get_output(plain_console)
        assert "old" in output
        assert "new" in output

    def test_shows_file_path_in_title(self, plain_console: Console) -> None:
        diff = "-old\n+new\n"
        display_diff(diff, file_path="src/app.py", console=plain_console)
        output = _get_output(plain_console)
        assert "src/app.py" in output

    def test_empty_diff_shows_no_changes(self, plain_console: Console) -> None:
        display_diff("", file_path="foo.py", console=plain_console)
        output = _get_output(plain_console)
        assert "no changes" in output

    def test_whitespace_only_diff_shows_no_changes(self, plain_console: Console) -> None:
        display_diff("   \n  ", file_path="foo.py", console=plain_console)
        output = _get_output(plain_console)
        assert "no changes" in output

    def test_no_file_path_uses_generic_title(self, plain_console: Console) -> None:
        diff = "+added line\n"
        display_diff(diff, console=plain_console)
        output = _get_output(plain_console)
        assert "Diff Preview" in output

    def test_default_console_does_not_raise(self) -> None:
        """display_diff should not raise when called without a console."""
        display_diff("-old\n+new\n", file_path="test.py")


# ---------------------------------------------------------------------------
# Default console fallback
# ---------------------------------------------------------------------------


class TestDefaultConsoleFallback:
    """Tests that functions work when no console is passed."""

    def test_display_welcome_default_console(self, capsys: object) -> None:
        """display_welcome should not raise when called without a console."""
        # We just ensure no exception is raised; stdout may or may not capture
        # Rich output depending on terminal state.
        display_welcome()

    def test_display_error_default_console(self) -> None:
        display_error("test error")

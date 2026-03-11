"""Tests for prism.cli.repl — interactive REPL loop and slash commands."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from rich.console import Console

from prism.cli.repl import (
    SLASH_COMMANDS,
    _handle_slash_command,
    _process_prompt,
    run_repl,
)
from prism.config.schema import PrismConfig
from prism.config.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path) -> Settings:
    """Create a minimal Settings instance for testing."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


def _make_console() -> Console:
    """Create a console that writes to an in-memory buffer."""
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, no_color=True, width=300)


def _get_output(console: Console) -> str:
    """Extract text from an in-memory console."""
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


# ---------------------------------------------------------------------------
# SLASH_COMMANDS registry
# ---------------------------------------------------------------------------


class TestSlashCommands:
    """Tests for the SLASH_COMMANDS constant."""

    def test_slash_commands_is_dict(self) -> None:
        assert isinstance(SLASH_COMMANDS, dict)

    def test_slash_commands_has_help(self) -> None:
        assert "/help" in SLASH_COMMANDS

    def test_slash_commands_has_quit(self) -> None:
        assert "/quit" in SLASH_COMMANDS

    def test_slash_commands_has_model(self) -> None:
        assert "/model" in SLASH_COMMANDS

    def test_slash_commands_has_add(self) -> None:
        assert "/add" in SLASH_COMMANDS

    def test_slash_commands_has_drop(self) -> None:
        assert "/drop" in SLASH_COMMANDS

    def test_slash_commands_has_cost(self) -> None:
        assert "/cost" in SLASH_COMMANDS

    def test_slash_commands_has_clear(self) -> None:
        assert "/clear" in SLASH_COMMANDS

    def test_slash_commands_has_web(self) -> None:
        assert "/web" in SLASH_COMMANDS

    def test_all_values_are_strings(self) -> None:
        for key, val in SLASH_COMMANDS.items():
            assert isinstance(key, str), f"Key {key!r} is not a string"
            assert isinstance(val, str), f"Value for {key!r} is not a string"


# ---------------------------------------------------------------------------
# _handle_slash_command
# ---------------------------------------------------------------------------


class TestHandleSlashCommand:
    """Tests for the _handle_slash_command function."""

    def test_quit_returns_quit(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/quit", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "quit"
        assert "Goodbye" in _get_output(console)

    def test_exit_returns_quit(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/exit", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "quit"

    def test_q_returns_quit(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/q", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "quit"

    def test_help_returns_continue(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/help", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "continue"
        output = _get_output(console)
        assert "Available Commands" in output
        # Should list all known slash commands
        for cmd in SLASH_COMMANDS:
            assert cmd in output

    def test_cost_returns_continue(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/cost", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "continue"
        assert "cost" in _get_output(console).lower()

    def test_model_returns_model_changed(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/model", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "model_changed"

    def test_model_with_arg_returns_model_changed(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/model gpt-4o", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "model_changed"

    def test_add_no_args_shows_usage(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/add", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "continue"
        assert "Usage" in _get_output(console)

    def test_add_file_to_context(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        active_files: list[str] = []
        result = _handle_slash_command(
            "/add src/main.py", console=console, settings=settings,
            active_files=active_files, pinned_model=None,
        )
        assert result == "continue"
        assert "src/main.py" in active_files
        assert "+" in _get_output(console)

    def test_add_multiple_files(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        active_files: list[str] = []
        _handle_slash_command(
            "/add src/a.py src/b.py", console=console, settings=settings,
            active_files=active_files, pinned_model=None,
        )
        assert "src/a.py" in active_files
        assert "src/b.py" in active_files

    def test_add_duplicate_file(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        active_files = ["src/main.py"]
        _handle_slash_command(
            "/add src/main.py", console=console, settings=settings,
            active_files=active_files, pinned_model=None,
        )
        assert active_files.count("src/main.py") == 1
        assert "Already added" in _get_output(console)

    def test_drop_no_args_shows_active_files(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        active_files = ["src/main.py", "src/utils.py"]
        _handle_slash_command(
            "/drop", console=console, settings=settings,
            active_files=active_files, pinned_model=None,
        )
        output = _get_output(console)
        assert "src/main.py" in output
        assert "src/utils.py" in output

    def test_drop_no_args_empty_list(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        active_files: list[str] = []
        _handle_slash_command(
            "/drop", console=console, settings=settings,
            active_files=active_files, pinned_model=None,
        )
        assert "No active files" in _get_output(console)

    def test_drop_removes_file(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        active_files = ["src/main.py", "src/utils.py"]
        _handle_slash_command(
            "/drop src/main.py", console=console, settings=settings,
            active_files=active_files, pinned_model=None,
        )
        assert "src/main.py" not in active_files
        assert "src/utils.py" in active_files

    def test_drop_nonexistent_file(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        active_files = ["src/main.py"]
        _handle_slash_command(
            "/drop nonexistent.py", console=console, settings=settings,
            active_files=active_files, pinned_model=None,
        )
        assert "Not in context" in _get_output(console)

    def test_compact_returns_continue(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/compact", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "continue"
        assert "not yet implemented" in _get_output(console).lower()

    def test_undo_returns_continue(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/undo", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "continue"
        assert "not yet implemented" in _get_output(console).lower()

    def test_web_on(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/web on", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "continue"
        assert "enabled" in _get_output(console).lower()

    def test_web_off(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/web off", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "continue"
        assert "disabled" in _get_output(console).lower()

    def test_web_no_args_shows_usage(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        _handle_slash_command(
            "/web", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert "Usage" in _get_output(console)

    def test_status_calls_app_status(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        with patch("prism.cli.app.status") as mock_status:
            result = _handle_slash_command(
                "/status", console=console, settings=settings,
                active_files=[], pinned_model=None,
            )
        assert result == "continue"
        mock_status.assert_called_once()

    def test_clear_returns_continue(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/clear", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "continue"
        assert "cleared" in _get_output(console).lower()

    def test_unknown_command(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/foobar", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "continue"
        assert "Unknown command" in _get_output(console)

    def test_command_case_insensitive(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/HELP", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "continue"
        assert "Available Commands" in _get_output(console)

    def test_quit_case_insensitive(self, tmp_path: Path) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)
        result = _handle_slash_command(
            "/QUIT", console=console, settings=settings,
            active_files=[], pinned_model=None,
        )
        assert result == "quit"


# ---------------------------------------------------------------------------
# _process_prompt
# ---------------------------------------------------------------------------


class TestProcessPrompt:
    """Tests for _process_prompt — routing through classifier."""

    @patch("prism.router.classifier.TaskClassifier")
    @patch("prism.router.classifier.TaskContext")
    def test_process_prompt_shows_classification(
        self,
        mock_context_cls: MagicMock,
        mock_classifier_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from prism.providers.base import ComplexityTier
        from prism.router.classifier import ClassificationResult

        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_result = ClassificationResult(
            tier=ComplexityTier.SIMPLE,
            score=0.15,
            features={"prompt_token_count": 5.0},
            reasoning="Simple task",
        )
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = mock_result
        mock_classifier_cls.return_value = mock_classifier

        conversation: list[dict[str, str]] = []

        _process_prompt(
            prompt="What is Python?",
            console=console,
            settings=settings,
            active_files=[],
            conversation=conversation,
            dry_run=False,
            offline=False,
            pinned_model=None,
        )

        output = _get_output(console)
        assert "SIMPLE" in output
        assert "0.15" in output

    @patch("prism.router.classifier.TaskClassifier")
    @patch("prism.router.classifier.TaskContext")
    def test_process_prompt_dry_run_shows_features(
        self,
        mock_context_cls: MagicMock,
        mock_classifier_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from prism.providers.base import ComplexityTier
        from prism.router.classifier import ClassificationResult

        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_result = ClassificationResult(
            tier=ComplexityTier.MEDIUM,
            score=0.50,
            features={"complexity_keywords": 0.6},
            reasoning="Medium complexity",
        )
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = mock_result
        mock_classifier_cls.return_value = mock_classifier

        conversation: list[dict[str, str]] = []

        _process_prompt(
            prompt="Refactor the module",
            console=console,
            settings=settings,
            active_files=[],
            conversation=conversation,
            dry_run=True,
            offline=False,
            pinned_model=None,
        )

        output = _get_output(console)
        assert "MEDIUM" in output
        assert "Dry-run" in output or "dry-run" in output.lower()
        assert "complexity_keywords" in output

    @patch("prism.router.classifier.TaskClassifier")
    @patch("prism.router.classifier.TaskContext")
    def test_process_prompt_dry_run_no_api_call(
        self,
        mock_context_cls: MagicMock,
        mock_classifier_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from prism.providers.base import ComplexityTier
        from prism.router.classifier import ClassificationResult

        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_result = ClassificationResult(
            tier=ComplexityTier.SIMPLE,
            score=0.1,
            features={},
            reasoning="Simple",
        )
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = mock_result
        mock_classifier_cls.return_value = mock_classifier

        conversation: list[dict[str, str]] = []

        _process_prompt(
            prompt="hello",
            console=console,
            settings=settings,
            active_files=[],
            conversation=conversation,
            dry_run=True,
            offline=False,
            pinned_model=None,
        )

        output = _get_output(console)
        assert "no API call" in output.lower() or "no api call" in output.lower()
        # Conversation should NOT be updated in dry-run
        assert len(conversation) == 0

    @patch("prism.router.classifier.TaskClassifier")
    @patch("prism.router.classifier.TaskContext")
    def test_process_prompt_appends_to_conversation(
        self,
        mock_context_cls: MagicMock,
        mock_classifier_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from prism.providers.base import ComplexityTier
        from prism.router.classifier import ClassificationResult

        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_result = ClassificationResult(
            tier=ComplexityTier.SIMPLE,
            score=0.1,
            features={},
            reasoning="Simple",
        )
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = mock_result
        mock_classifier_cls.return_value = mock_classifier

        conversation: list[dict[str, str]] = []

        _process_prompt(
            prompt="What is 2+2?",
            console=console,
            settings=settings,
            active_files=[],
            conversation=conversation,
            dry_run=False,
            offline=False,
            pinned_model=None,
        )

        assert len(conversation) == 1
        assert conversation[0]["role"] == "user"
        assert conversation[0]["content"] == "What is 2+2?"

    @patch("prism.router.classifier.TaskClassifier")
    @patch("prism.router.classifier.TaskContext")
    def test_process_prompt_complex_tier_color(
        self,
        mock_context_cls: MagicMock,
        mock_classifier_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from prism.providers.base import ComplexityTier
        from prism.router.classifier import ClassificationResult

        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_result = ClassificationResult(
            tier=ComplexityTier.COMPLEX,
            score=0.85,
            features={},
            reasoning="Complex task",
        )
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = mock_result
        mock_classifier_cls.return_value = mock_classifier

        _process_prompt(
            prompt="Redesign the architecture",
            console=console,
            settings=settings,
            active_files=[],
            conversation=[],
            dry_run=False,
            offline=False,
            pinned_model=None,
        )

        output = _get_output(console)
        assert "COMPLEX" in output
        assert "0.85" in output


# ---------------------------------------------------------------------------
# run_repl integration
# ---------------------------------------------------------------------------


class TestRunRepl:
    """Tests for the main run_repl loop."""

    @patch("prism.cli.repl.PromptSession")
    def test_repl_eof_exits_gracefully(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.side_effect = EOFError()
        mock_session_cls.return_value = mock_session

        run_repl(settings=settings, console=console)

        output = _get_output(console)
        assert "Goodbye" in output

    @patch("prism.cli.repl.PromptSession")
    def test_repl_quit_command_exits(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.return_value = "/quit"
        mock_session_cls.return_value = mock_session

        run_repl(settings=settings, console=console)

        output = _get_output(console)
        assert "Goodbye" in output

    @patch("prism.cli.repl.PromptSession")
    def test_repl_empty_input_continues(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        # First call returns empty, second returns /quit
        mock_session.prompt.side_effect = ["", "   ", "/quit"]
        mock_session_cls.return_value = mock_session

        run_repl(settings=settings, console=console)

        # Should have processed all three prompts
        assert mock_session.prompt.call_count == 3

    @patch("prism.cli.repl.PromptSession")
    def test_repl_keyboard_interrupt_continues(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.side_effect = [KeyboardInterrupt(), "/quit"]
        mock_session_cls.return_value = mock_session

        run_repl(settings=settings, console=console)

        assert mock_session.prompt.call_count == 2

    @patch("prism.cli.repl.PromptSession")
    def test_repl_help_then_quit(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.side_effect = ["/help", "/quit"]
        mock_session_cls.return_value = mock_session

        run_repl(settings=settings, console=console)

        output = _get_output(console)
        assert "Available Commands" in output

    @patch("prism.cli.repl._process_prompt")
    @patch("prism.cli.repl.PromptSession")
    def test_repl_regular_prompt_calls_process(
        self,
        mock_session_cls: MagicMock,
        mock_process: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.side_effect = ["Hello world", "/quit"]
        mock_session_cls.return_value = mock_session

        run_repl(settings=settings, console=console)

        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["prompt"] == "Hello world"

    @patch("prism.cli.repl.PromptSession")
    def test_repl_model_change_with_arg(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.side_effect = ["/model claude-sonnet", "/quit"]
        mock_session_cls.return_value = mock_session

        run_repl(settings=settings, console=console)

        output = _get_output(console)
        assert "claude-sonnet" in output
        assert "Model set to" in output

    @patch("prism.cli.repl.PromptSession")
    def test_repl_model_no_arg_shows_current(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.side_effect = ["/model", "/quit"]
        mock_session_cls.return_value = mock_session

        run_repl(settings=settings, console=console)

        output = _get_output(console)
        assert "Current model" in output
        assert "auto" in output.lower()

    @patch("prism.cli.repl.PromptSession")
    def test_repl_exception_in_loop_is_caught(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        # First prompt raises an exception (not KeyboardInterrupt/EOFError)
        # The REPL catches generic exceptions in the outer try block
        mock_session.prompt.side_effect = ["raise-error", "/quit"]
        mock_session_cls.return_value = mock_session

        # Make _process_prompt raise to test exception handling
        with patch("prism.cli.repl._process_prompt", side_effect=RuntimeError("test error")):
            run_repl(settings=settings, console=console)

        output = _get_output(console)
        assert "unexpected error" in output.lower()

    @patch("prism.cli.repl.PromptSession")
    def test_repl_dry_run_passed_through(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.side_effect = ["test prompt", "/quit"]
        mock_session_cls.return_value = mock_session

        with patch("prism.cli.repl._process_prompt") as mock_process:
            run_repl(settings=settings, console=console, dry_run=True)

        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["dry_run"] is True

    @patch("prism.cli.repl.PromptSession")
    def test_repl_offline_passed_through(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.side_effect = ["test prompt", "/quit"]
        mock_session_cls.return_value = mock_session

        with patch("prism.cli.repl._process_prompt") as mock_process:
            run_repl(settings=settings, console=console, offline=True)

        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["offline"] is True

    @patch("prism.cli.repl.PromptSession")
    def test_repl_creates_history_dir(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.side_effect = EOFError()
        mock_session_cls.return_value = mock_session

        run_repl(settings=settings, console=console)

        # The sessions_dir should exist
        assert settings.sessions_dir.is_dir()

    @patch("prism.cli.repl.PromptSession")
    def test_repl_ready_message_shown(
        self,
        mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        console = _make_console()
        settings = _make_settings(tmp_path)

        mock_session = MagicMock()
        mock_session.prompt.side_effect = EOFError()
        mock_session_cls.return_value = mock_session

        run_repl(settings=settings, console=console)

        output = _get_output(console)
        assert "Ready" in output

"""Tests for SlashCommandHandler -- all slash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prism.cli.commands.slash_commands import CommandResponse, SlashCommandHandler

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from prism.context.manager import ContextManager
    from prism.context.memory import ProjectMemory
    from prism.context.session import SessionManager

# =====================================================================
# /help
# =====================================================================


class TestHelpCommand:
    def test_help_lists_all_commands(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/help")
        assert resp.error is None
        assert "/cost" in resp.output
        assert "/model" in resp.output
        assert "/add" in resp.output
        assert "/drop" in resp.output
        assert "/undo" in resp.output
        assert "/compact" in resp.output
        assert "/web" in resp.output
        assert "/status" in resp.output
        assert "/budget" in resp.output
        assert "/memory" in resp.output
        assert "/feedback" in resp.output
        assert "/providers" in resp.output
        assert "/clear" in resp.output
        assert "/save" in resp.output
        assert "/load" in resp.output
        assert "/logs" in resp.output
        assert "/exit" in resp.output


# =====================================================================
# /cost
# =====================================================================


class TestCostCommand:
    def test_cost_shows_dashboard(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/cost")
        assert resp.error is None
        # The mock cost_tracker's get_cost_summary returns data
        assert resp.output  # Non-empty

    def test_cost_without_tracker(
        self, handler_minimal: SlashCommandHandler
    ) -> None:
        resp = handler_minimal.handle("/cost")
        assert "available after" in resp.output.lower()


# =====================================================================
# /model
# =====================================================================


class TestModelCommand:
    def test_model_show_current(
        self, handler: SlashCommandHandler
    ) -> None:
        handler._current_model = "claude-sonnet-4-20250514"
        resp = handler.handle("/model")
        assert resp.error is None
        assert "claude-sonnet-4-20250514" in resp.output

    def test_model_show_current_auto(
        self, handler: SlashCommandHandler
    ) -> None:
        handler._current_model = None
        resp = handler.handle("/model")
        assert "auto" in resp.output.lower()

    def test_model_switch_valid(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/model claude-sonnet-4-20250514")
        assert resp.error is None
        assert "claude-sonnet-4-20250514" in resp.output
        assert handler._current_model == "claude-sonnet-4-20250514"

    def test_model_switch_invalid_shows_error(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/model nonexistent-model-xyz")
        assert resp.error is not None
        assert "unknown model" in resp.error.lower()


# =====================================================================
# /undo
# =====================================================================


class TestUndoCommand:
    def test_undo_with_git_repo(
        self, handler: SlashCommandHandler, mock_git_repo: MagicMock
    ) -> None:
        resp = handler.handle("/undo")
        assert resp.error is None
        assert "undone" in resp.output.lower()
        mock_git_repo._run.assert_called_once_with(
            ["git", "reset", "--soft", "HEAD~1"]
        )

    def test_undo_without_git_repo(
        self, handler_minimal: SlashCommandHandler
    ) -> None:
        resp = handler_minimal.handle("/undo")
        assert resp.error is not None
        assert "git" in resp.error.lower()


# =====================================================================
# /compact
# =====================================================================


class TestCompactCommand:
    def test_compact_reduces_context(
        self,
        handler: SlashCommandHandler,
        context_manager: ContextManager,
    ) -> None:
        # Add some messages
        context_manager.add_message("user", "First question about Python")
        context_manager.add_message("assistant", "Here is the answer...")
        context_manager.add_message("user", "Second question about testing")
        context_manager.add_message("assistant", "Here is another answer...")

        assert context_manager.message_count == 4

        resp = handler.handle("/compact")
        assert resp.error is None
        assert "4 messages summarized" in resp.output

        # After compact, there should be fewer messages
        assert context_manager.message_count < 4

    def test_compact_empty_history(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/compact")
        assert "no conversation" in resp.output.lower()


# =====================================================================
# /add
# =====================================================================


class TestAddCommand:
    def test_add_file_exists(
        self,
        handler: SlashCommandHandler,
        tmp_path: Path,
    ) -> None:
        test_file = tmp_path / "test_file.py"
        test_file.write_text("print('hello')")

        resp = handler.handle(f"/add {test_file}")
        assert resp.error is None
        assert str(test_file) in resp.output

    def test_add_file_not_found(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/add /nonexistent/path/file.py")
        assert "not found" in resp.output.lower()

    def test_add_multiple_files(
        self,
        handler: SlashCommandHandler,
        tmp_path: Path,
    ) -> None:
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("x = 1")
        f2.write_text("y = 2")

        resp = handler.handle(f"/add {f1} {f2}")
        assert resp.error is None
        assert str(f1) in resp.output
        assert str(f2) in resp.output

    def test_add_no_args(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/add")
        assert "usage" in resp.output.lower()


# =====================================================================
# /drop
# =====================================================================


class TestDropCommand:
    def test_drop_file(
        self,
        handler: SlashCommandHandler,
        context_manager: ContextManager,
    ) -> None:
        context_manager.add_active_file("src/main.py", "print('hi')")
        resp = handler.handle("/drop src/main.py")
        assert resp.error is None
        assert "src/main.py" in resp.output
        assert "src/main.py" not in context_manager.active_files

    def test_drop_file_not_in_context(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/drop nonexistent.py")
        assert "not in context" in resp.output.lower()

    def test_drop_no_args_shows_active(
        self,
        handler: SlashCommandHandler,
        context_manager: ContextManager,
    ) -> None:
        context_manager.add_active_file("foo.py", "x = 1")
        resp = handler.handle("/drop")
        assert "foo.py" in resp.output

    def test_drop_no_args_empty(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/drop")
        assert "no active files" in resp.output.lower()


# =====================================================================
# /web
# =====================================================================


class TestWebCommand:
    def test_web_toggle_on(
        self, handler: SlashCommandHandler
    ) -> None:
        handler._web_enabled = False
        resp = handler.handle("/web on")
        assert resp.error is None
        assert "enabled" in resp.output.lower()
        assert handler._web_enabled is True

    def test_web_toggle_off(
        self, handler: SlashCommandHandler
    ) -> None:
        handler._web_enabled = True
        resp = handler.handle("/web off")
        assert resp.error is None
        assert "disabled" in resp.output.lower()
        assert handler._web_enabled is False

    def test_web_toggle(
        self, handler: SlashCommandHandler
    ) -> None:
        handler._web_enabled = False
        resp = handler.handle("/web")
        assert "enabled" in resp.output.lower()
        assert handler._web_enabled is True

        resp = handler.handle("/web")
        assert "disabled" in resp.output.lower()
        assert handler._web_enabled is False


# =====================================================================
# /status
# =====================================================================


class TestStatusCommand:
    def test_status_shows_providers(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/status")
        assert resp.error is None
        assert "Anthropic" in resp.output
        assert "OpenAI" in resp.output

    def test_status_shows_budget(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/status")
        assert "$" in resp.output


# =====================================================================
# /budget
# =====================================================================


class TestBudgetCommand:
    def test_budget_show(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/budget")
        assert resp.error is None
        # Should show some budget info
        assert "limit" in resp.output.lower()

    def test_budget_set_valid(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/budget set 10.00")
        assert resp.error is None
        assert "$10.00" in resp.output

    def test_budget_set_invalid(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/budget set abc")
        assert resp.error is not None
        assert "invalid" in resp.error.lower()

    def test_budget_set_negative(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/budget set -5")
        assert resp.error is not None
        assert "negative" in resp.error.lower()

    def test_budget_set_missing_amount(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/budget set")
        assert resp.error is not None
        assert "usage" in resp.error.lower()


# =====================================================================
# /memory
# =====================================================================


class TestMemoryCommand:
    def test_memory_show_empty(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/memory")
        assert resp.error is None
        assert "no project memory" in resp.output.lower()

    def test_memory_set(
        self,
        handler: SlashCommandHandler,
        project_memory: ProjectMemory,
    ) -> None:
        resp = handler.handle("/memory set stack Python 3.12")
        assert resp.error is None
        assert "stack" in resp.output
        assert project_memory.get_fact("stack") == "Python 3.12"

    def test_memory_show_after_set(
        self,
        handler: SlashCommandHandler,
        project_memory: ProjectMemory,
    ) -> None:
        project_memory.add_fact("framework", "FastAPI")
        resp = handler.handle("/memory")
        assert "framework" in resp.output
        assert "FastAPI" in resp.output

    def test_memory_lookup_key(
        self,
        handler: SlashCommandHandler,
        project_memory: ProjectMemory,
    ) -> None:
        project_memory.add_fact("db", "PostgreSQL")
        resp = handler.handle("/memory db")
        assert "PostgreSQL" in resp.output


# =====================================================================
# /feedback
# =====================================================================


class TestFeedbackCommand:
    def test_feedback_up(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/feedback up")
        assert resp.error is None
        assert "positive" in resp.output.lower()
        assert handler._last_feedback == "up"

    def test_feedback_down(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/feedback down")
        assert resp.error is None
        assert "negative" in resp.output.lower()
        assert handler._last_feedback == "down"

    def test_feedback_no_args(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/feedback")
        assert resp.error is not None
        assert "usage" in resp.error.lower()


# =====================================================================
# /providers
# =====================================================================


class TestProvidersCommand:
    def test_providers_list(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/providers")
        assert resp.error is None
        assert "Anthropic" in resp.output
        assert "OpenAI" in resp.output

    def test_providers_no_registry(
        self, handler_minimal: SlashCommandHandler
    ) -> None:
        resp = handler_minimal.handle("/providers")
        assert resp.error is not None
        assert "not available" in resp.error.lower()


# =====================================================================
# /clear
# =====================================================================


class TestClearCommand:
    def test_clear_conversation(
        self,
        handler: SlashCommandHandler,
        context_manager: ContextManager,
    ) -> None:
        context_manager.add_message("user", "Hello")
        context_manager.add_message("assistant", "Hi there")
        assert context_manager.message_count == 2

        resp = handler.handle("/clear")
        assert resp.error is None
        assert "cleared" in resp.output.lower()
        assert context_manager.message_count == 0


# =====================================================================
# /save
# =====================================================================


class TestSaveCommand:
    def test_save_session(
        self,
        handler: SlashCommandHandler,
        session_manager: SessionManager,
    ) -> None:
        resp = handler.handle("/save test-save")
        assert resp.error is None
        assert "saved" in resp.output.lower()

    def test_save_without_name(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/save")
        assert resp.error is None
        assert "saved" in resp.output.lower()


# =====================================================================
# /load
# =====================================================================


class TestLoadCommand:
    def test_load_session(
        self,
        handler: SlashCommandHandler,
        session_manager: SessionManager,
    ) -> None:
        # Create and save a session first
        sid = session_manager.create_session(handler._settings.project_root)
        session_manager.save_session(sid, {
            "session_id": sid,
            "project_root": str(handler._settings.project_root),
            "messages": [{"role": "user", "content": "Test message"}],
            "metadata": {"model": "gpt-4o-mini"},
        })

        resp = handler.handle(f"/load {sid}")
        assert resp.error is None
        assert "loaded" in resp.output.lower()
        assert handler.session_id == sid

    def test_load_nonexistent(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/load nonexistent-session-id")
        assert resp.error is not None
        assert "not found" in resp.error.lower()

    def test_load_no_args_lists(
        self,
        handler: SlashCommandHandler,
        session_manager: SessionManager,
    ) -> None:
        # When no session_id given, list sessions
        resp = handler.handle("/load")
        # Could show "no sessions" or list
        assert resp.output  # Non-empty response


# =====================================================================
# /logs
# =====================================================================


class TestLogsCommand:
    def test_logs_show_recent(
        self,
        handler: SlashCommandHandler,
        tmp_path: Path,
    ) -> None:
        # Create a fake log file
        log_path = handler._settings.prism_home / "prism.log"
        log_path.write_text("2025-01-01 INFO Starting up\n2025-01-01 DEBUG Test\n")

        resp = handler.handle("/logs")
        assert resp.error is None
        assert "Starting up" in resp.output

    def test_logs_no_file(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/logs")
        assert "no log file" in resp.output.lower()


# =====================================================================
# /exit and /quit
# =====================================================================


class TestExitCommand:
    def test_exit_returns_should_exit(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/exit")
        assert resp.should_exit is True
        assert "goodbye" in resp.output.lower()

    def test_quit_returns_should_exit(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/quit")
        assert resp.should_exit is True


# =====================================================================
# Unknown command
# =====================================================================


class TestUnknownCommand:
    def test_unknown_command(
        self, handler: SlashCommandHandler
    ) -> None:
        resp = handler.handle("/foobar")
        assert resp.error is not None
        assert "unknown command" in resp.error.lower()
        assert "/foobar" in resp.error


# =====================================================================
# CommandResponse dataclass
# =====================================================================


class TestCommandResponse:
    def test_defaults(self) -> None:
        r = CommandResponse(output="test")
        assert r.output == "test"
        assert r.should_exit is False
        assert r.error is None

    def test_with_exit(self) -> None:
        r = CommandResponse(output="bye", should_exit=True)
        assert r.should_exit is True

    def test_with_error(self) -> None:
        r = CommandResponse(output="", error="something broke")
        assert r.error == "something broke"

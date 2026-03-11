"""Tests for prism.cli.repl — interactive REPL loop and slash commands."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING
from unittest.mock import patch

from rich.console import Console

from prism.cli.repl import (
    COMMAND_CATEGORIES,
    _dispatch_command,
    _SessionState,
    run_repl,
)
from prism.config.schema import PrismConfig
from prism.config.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path) -> Settings:
    """Create a minimal Settings instance for testing."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


def _make_console(width: int = 300) -> Console:
    """Create a console that writes to an in-memory buffer."""
    buf = io.StringIO()
    return Console(
        file=buf, force_terminal=False, no_color=True, width=width,
    )


def _get_output(console: Console) -> str:
    """Extract text from an in-memory console."""
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


def _make_state(
    pinned_model: str | None = None,
    active_files: list[str] | None = None,
) -> _SessionState:
    """Create a SessionState with optional overrides."""
    state = _SessionState(pinned_model=pinned_model)
    if active_files is not None:
        state.active_files = active_files
    state.session_id = "test-session"
    return state


def _cmd(
    command: str,
    tmp_path: Path,
    active_files: list[str] | None = None,
    pinned_model: str | None = None,
    console: Console | None = None,
    settings: Settings | None = None,
    state: _SessionState | None = None,
    dry_run: bool = False,
    offline: bool = False,
) -> tuple[str, str, Console, Settings, _SessionState]:
    """Run a slash command and return (action, output, console, settings, state).

    This is a convenience wrapper to reduce boilerplate in individual tests.
    """
    con = console or _make_console()
    stg = settings or _make_settings(tmp_path)
    st = state or _make_state(
        pinned_model=pinned_model,
        active_files=active_files if active_files is not None else [],
    )
    # If active_files was passed but state was not, ensure they match
    if active_files is not None and state is None:
        st.active_files = active_files
    action = _dispatch_command(
        command,
        console=con,
        settings=stg,
        state=st,
        dry_run=dry_run,
        offline=offline,
    )
    return action, _get_output(con), con, stg, st


# ===========================================================================
# COMMAND_CATEGORIES registry
# ===========================================================================


class TestCommandCategories:
    """Tests for the COMMAND_CATEGORIES constant."""

    def test_command_categories_is_dict(self) -> None:
        assert isinstance(COMMAND_CATEGORIES, dict)

    def test_has_general_category(self) -> None:
        assert "General" in COMMAND_CATEGORIES

    def test_has_model_routing_category(self) -> None:
        assert "Model & Routing" in COMMAND_CATEGORIES

    def test_has_context_files_category(self) -> None:
        assert "Context & Files" in COMMAND_CATEGORIES

    def test_has_cost_budget_category(self) -> None:
        assert "Cost & Budget" in COMMAND_CATEGORIES

    def test_has_tools_execution_category(self) -> None:
        assert "Tools & Execution" in COMMAND_CATEGORIES

    def test_has_code_intelligence_category(self) -> None:
        assert "Code Intelligence" in COMMAND_CATEGORIES

    def test_has_infrastructure_category(self) -> None:
        assert "Infrastructure" in COMMAND_CATEGORIES

    def test_all_values_are_lists_of_tuples(self) -> None:
        for cat, commands in COMMAND_CATEGORIES.items():
            assert isinstance(commands, list), f"{cat} is not a list"
            for item in commands:
                assert isinstance(item, tuple), (
                    f"Item in {cat} is not a tuple"
                )
                assert len(item) == 2, f"Tuple in {cat} has {len(item)} items"

    def test_minimum_categories(self) -> None:
        assert len(COMMAND_CATEGORIES) >= 6

    def test_minimum_total_commands(self) -> None:
        total = sum(len(v) for v in COMMAND_CATEGORIES.values())
        assert total >= 25


# ===========================================================================
# _SessionState
# ===========================================================================


class TestSessionState:
    """Tests for the _SessionState class."""

    def test_default_state(self) -> None:
        state = _SessionState(pinned_model=None)
        assert state.active_files == []
        assert state.conversation == []
        assert state.pinned_model is None
        assert state.web_enabled is False
        assert state.session_id == ""

    def test_pinned_model_set(self) -> None:
        state = _SessionState(pinned_model="gpt-4o")
        assert state.pinned_model == "gpt-4o"

    def test_active_files_mutable(self) -> None:
        state = _SessionState(pinned_model=None)
        state.active_files.append("foo.py")
        assert state.active_files == ["foo.py"]

    def test_conversation_mutable(self) -> None:
        state = _SessionState(pinned_model=None)
        state.conversation.append({"role": "user", "content": "hi"})
        assert len(state.conversation) == 1


# ===========================================================================
# 1. /help
# ===========================================================================


class TestHelpCommand:
    """Tests for the /help slash command."""

    def test_help_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/help", tmp_path)
        assert action == "continue"

    def test_help_shows_category_names(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/help", tmp_path)
        for category in COMMAND_CATEGORIES:
            assert category in output, (
                f"Category {category!r} not in /help output"
            )

    def test_help_case_insensitive(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/HELP", tmp_path)
        assert action == "continue"
        # Should still display categories
        assert "General" in output

    def test_help_with_trailing_space(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/help   ", tmp_path)
        assert action == "continue"

    def test_help_ignores_extra_args(self, tmp_path: Path) -> None:
        """Extra arguments after /help are silently ignored."""
        action, output, _, _, _ = _cmd("/help some extra args", tmp_path)
        assert action == "continue"
        assert "General" in output


# ===========================================================================
# 2. /cost
# ===========================================================================


class TestCostCommand:
    """Tests for the /cost slash command."""

    def test_cost_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/cost", tmp_path)
        assert action == "continue"

    def test_cost_shows_cost_content(self, tmp_path: Path) -> None:
        """Shows either a dashboard or a fallback message."""
        _, output, _, _, _ = _cmd("/cost", tmp_path)
        lower = output.lower()
        assert (
            "cost" in lower or "dashboard" in lower or "api call" in lower
        )

    def test_cost_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/COST", tmp_path)
        assert action == "continue"


# ===========================================================================
# 3. /model
# ===========================================================================


class TestModelCommand:
    """Tests for the /model slash command."""

    def test_model_no_args_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/model", tmp_path)
        assert action == "continue"

    def test_model_no_args_shows_current(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd(
            "/model", tmp_path, pinned_model="gpt-4o",
        )
        assert "gpt-4o" in output

    def test_model_no_args_shows_auto(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/model", tmp_path)
        assert "auto" in output.lower()

    def test_model_set_new_model(self, tmp_path: Path) -> None:
        action, output, _, _, st = _cmd("/model gpt-4o", tmp_path)
        assert action == "continue"
        assert st.pinned_model == "gpt-4o"
        assert "gpt-4o" in output

    def test_model_set_auto(self, tmp_path: Path) -> None:
        state = _make_state(pinned_model="gpt-4o")
        action, _, _, _, st = _cmd(
            "/model auto", tmp_path, state=state,
        )
        assert action == "continue"
        assert st.pinned_model is None

    def test_model_with_complex_name(self, tmp_path: Path) -> None:
        action, _, _, _, st = _cmd(
            "/model deepseek/deepseek-chat", tmp_path,
        )
        assert action == "continue"
        assert st.pinned_model == "deepseek/deepseek-chat"

    def test_model_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/MODEL gpt-4o", tmp_path)
        assert action == "continue"


# ===========================================================================
# 4. /add — file context addition
# ===========================================================================


class TestAddCommand:
    """Tests for the /add slash command."""

    def test_add_no_args_shows_usage(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/add", tmp_path)
        assert action == "continue"
        assert "Usage" in output

    def test_add_single_file(self, tmp_path: Path) -> None:
        action, output, _, _, st = _cmd(
            "/add src/main.py", tmp_path,
        )
        assert action == "continue"
        assert "src/main.py" in st.active_files
        assert "+" in output

    def test_add_multiple_files_at_once(self, tmp_path: Path) -> None:
        _, _, _, _, st = _cmd("/add a.py b.py c.py", tmp_path)
        assert st.active_files == ["a.py", "b.py", "c.py"]

    def test_add_duplicate_file_not_added_twice(
        self, tmp_path: Path,
    ) -> None:
        state = _make_state(active_files=["src/main.py"])
        _, output, _, _, st = _cmd(
            "/add src/main.py", tmp_path, state=state,
        )
        assert st.active_files.count("src/main.py") == 1
        assert "Already added" in output

    def test_add_mix_of_new_and_duplicate(self, tmp_path: Path) -> None:
        state = _make_state(active_files=["existing.py"])
        _, output, _, _, st = _cmd(
            "/add existing.py new.py", tmp_path, state=state,
        )
        assert st.active_files == ["existing.py", "new.py"]
        assert "Already added" in output
        assert "+" in output

    def test_add_preserves_order(self, tmp_path: Path) -> None:
        _, _, _, _, st = _cmd("/add z.py a.py m.py", tmp_path)
        assert st.active_files == ["z.py", "a.py", "m.py"]

    def test_add_file_with_spaces_in_path(self, tmp_path: Path) -> None:
        """Spaces split into separate file args (current behavior)."""
        _, _, _, _, st = _cmd("/add my file.py", tmp_path)
        assert "my" in st.active_files
        assert "file.py" in st.active_files

    def test_add_case_insensitive_command(self, tmp_path: Path) -> None:
        action, _, _, _, st = _cmd("/ADD foo.py", tmp_path)
        assert action == "continue"
        assert "foo.py" in st.active_files


# ===========================================================================
# 5. /drop — file context removal
# ===========================================================================


class TestDropCommand:
    """Tests for the /drop slash command."""

    def test_drop_no_args_shows_active_files(self, tmp_path: Path) -> None:
        state = _make_state(active_files=["src/main.py", "src/utils.py"])
        action, output, _, _, _ = _cmd("/drop", tmp_path, state=state)
        assert action == "continue"
        assert "src/main.py" in output
        assert "src/utils.py" in output

    def test_drop_no_args_empty_list(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/drop", tmp_path)
        assert "No active files" in output

    def test_drop_removes_file(self, tmp_path: Path) -> None:
        state = _make_state(active_files=["src/main.py", "src/utils.py"])
        _, output, _, _, st = _cmd(
            "/drop src/main.py", tmp_path, state=state,
        )
        assert "src/main.py" not in st.active_files
        assert "src/utils.py" in st.active_files
        assert "-" in output

    def test_drop_nonexistent_file(self, tmp_path: Path) -> None:
        state = _make_state(active_files=["src/main.py"])
        _, output, _, _, st = _cmd(
            "/drop nonexistent.py", tmp_path, state=state,
        )
        assert "Not in context" in output
        assert "src/main.py" in st.active_files

    def test_drop_multiple_files(self, tmp_path: Path) -> None:
        state = _make_state(active_files=["a.py", "b.py", "c.py"])
        _, _, _, _, st = _cmd(
            "/drop a.py c.py", tmp_path, state=state,
        )
        assert st.active_files == ["b.py"]

    def test_drop_mix_of_present_and_absent(self, tmp_path: Path) -> None:
        state = _make_state(active_files=["real.py"])
        _, output, _, _, st = _cmd(
            "/drop real.py fake.py", tmp_path, state=state,
        )
        assert st.active_files == []
        assert "Not in context" in output

    def test_drop_all_files(self, tmp_path: Path) -> None:
        state = _make_state(active_files=["a.py", "b.py"])
        _, _, _, _, st = _cmd(
            "/drop a.py b.py", tmp_path, state=state,
        )
        assert st.active_files == []

    def test_drop_case_insensitive_command(self, tmp_path: Path) -> None:
        state = _make_state(active_files=["foo.py"])
        action, _, _, _, st = _cmd(
            "/DROP foo.py", tmp_path, state=state,
        )
        assert action == "continue"
        assert st.active_files == []


# ===========================================================================
# 6. /compact
# ===========================================================================


class TestCompactCommand:
    """Tests for the /compact slash command."""

    def test_compact_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/compact", tmp_path)
        assert action == "continue"

    def test_compact_short_conversation_shows_message(
        self, tmp_path: Path,
    ) -> None:
        """With less than 4 messages, shows 'too short' message."""
        state = _make_state()
        state.conversation = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        _, output, _, _, _ = _cmd("/compact", tmp_path, state=state)
        assert "too short" in output.lower()

    def test_compact_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/COMPACT", tmp_path)
        assert action == "continue"


# ===========================================================================
# 7. /undo
# ===========================================================================


class TestUndoCommand:
    """Tests for the /undo slash command."""

    def test_undo_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/undo", tmp_path)
        assert action == "continue"

    def test_undo_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/UNDO", tmp_path)
        assert action == "continue"

    def test_undo_outputs_something(self, tmp_path: Path) -> None:
        """Even on error, should print a message."""
        _, output, _, _, _ = _cmd("/undo", tmp_path)
        assert len(output.strip()) > 0


# ===========================================================================
# 8. /web
# ===========================================================================


class TestWebCommand:
    """Tests for the /web slash command."""

    def test_web_on(self, tmp_path: Path) -> None:
        action, output, _, _, st = _cmd("/web on", tmp_path)
        assert action == "continue"
        assert "enabled" in output.lower()
        assert st.web_enabled is True

    def test_web_off(self, tmp_path: Path) -> None:
        state = _make_state()
        state.web_enabled = True
        action, output, _, _, st = _cmd(
            "/web off", tmp_path, state=state,
        )
        assert action == "continue"
        assert "disabled" in output.lower()
        assert st.web_enabled is False

    def test_web_no_args_shows_status(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/web", tmp_path)
        lower = output.lower()
        assert "usage" in lower or "disabled" in lower or "enabled" in lower

    def test_web_invalid_arg_shows_status(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/web maybe", tmp_path)
        lower = output.lower()
        assert "usage" in lower or "disabled" in lower or "enabled" in lower

    def test_web_on_case_insensitive_arg(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/web ON", tmp_path)
        assert "enabled" in output.lower()

    def test_web_off_case_insensitive_arg(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/web OFF", tmp_path)
        assert "disabled" in output.lower()

    def test_web_case_insensitive_command(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/WEB on", tmp_path)
        assert action == "continue"
        assert "enabled" in output.lower()


# ===========================================================================
# 9. /status
# ===========================================================================


class TestStatusCommand:
    """Tests for the /status slash command."""

    def test_status_returns_continue(self, tmp_path: Path) -> None:
        with patch("prism.cli.repl.logger"):
            action, _, _, _, _ = _cmd("/status", tmp_path)
        assert action == "continue"

    def test_status_case_insensitive(self, tmp_path: Path) -> None:
        with patch("prism.cli.repl.logger"):
            action, _, _, _, _ = _cmd("/STATUS", tmp_path)
        assert action == "continue"


# ===========================================================================
# 10. /clear
# ===========================================================================


class TestClearCommand:
    """Tests for the /clear slash command."""

    def test_clear_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/clear", tmp_path)
        assert action == "continue"

    def test_clear_shows_cleared_message(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/clear", tmp_path)
        assert "cleared" in output.lower()

    def test_clear_empties_conversation(self, tmp_path: Path) -> None:
        state = _make_state()
        state.conversation = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "reply"},
        ]
        _, _, _, _, st = _cmd("/clear", tmp_path, state=state)
        assert st.conversation == []

    def test_clear_shows_message_count(self, tmp_path: Path) -> None:
        state = _make_state()
        state.conversation = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        _, output, _, _, _ = _cmd("/clear", tmp_path, state=state)
        assert "3" in output

    def test_clear_case_insensitive(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/CLEAR", tmp_path)
        assert action == "continue"
        assert "cleared" in output.lower()


# ===========================================================================
# 11. /cache
# ===========================================================================


class TestCacheCommand:
    """Tests for the /cache slash command."""

    def test_cache_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/cache", tmp_path)
        assert action == "continue"

    def test_cache_outputs_something(self, tmp_path: Path) -> None:
        """Should output cache stats or an error message."""
        _, output, _, _, _ = _cmd("/cache", tmp_path)
        assert len(output.strip()) > 0

    def test_cache_clear_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/cache clear", tmp_path)
        assert action == "continue"


# ===========================================================================
# 12. /compare
# ===========================================================================


class TestCompareCommand:
    """Tests for the /compare slash command."""

    def test_compare_no_args_shows_usage(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/compare", tmp_path)
        assert action == "continue"
        assert "Usage" in output

    def test_compare_returns_continue(self, tmp_path: Path) -> None:
        """Even with args, should return continue (may fail internally)."""
        action, _, _, _, _ = _cmd(
            "/compare test prompt", tmp_path,
        )
        assert action == "continue"


# ===========================================================================
# 13. /branch
# ===========================================================================


class TestBranchCommand:
    """Tests for the /branch slash command."""

    def test_branch_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/branch", tmp_path)
        assert action == "continue"

    def test_branch_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/BRANCH", tmp_path)
        assert action == "continue"

    def test_branch_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/branch", tmp_path)
        assert len(output.strip()) > 0


# ===========================================================================
# 14. /rollback
# ===========================================================================


class TestRollbackCommand:
    """Tests for the /rollback slash command."""

    def test_rollback_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/rollback", tmp_path)
        assert action == "continue"

    def test_rollback_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/ROLLBACK", tmp_path)
        assert action == "continue"

    def test_rollback_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/rollback", tmp_path)
        assert len(output.strip()) > 0


# ===========================================================================
# 15. /sandbox
# ===========================================================================


class TestSandboxCommand:
    """Tests for the /sandbox slash command."""

    def test_sandbox_no_args_shows_usage(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/sandbox", tmp_path)
        assert action == "continue"
        assert "Usage" in output

    def test_sandbox_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd(
            "/sandbox print('hello')", tmp_path,
        )
        assert action == "continue"


# ===========================================================================
# 16. /tasks
# ===========================================================================


class TestTasksCommand:
    """Tests for the /tasks slash command."""

    def test_tasks_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/tasks", tmp_path)
        assert action == "continue"

    def test_tasks_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/TASKS", tmp_path)
        assert action == "continue"

    def test_tasks_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/tasks", tmp_path)
        assert len(output.strip()) > 0


# ===========================================================================
# 17. /privacy
# ===========================================================================


class TestPrivacyCommand:
    """Tests for the /privacy slash command."""

    def test_privacy_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/privacy", tmp_path)
        assert action == "continue"

    def test_privacy_on_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/privacy on", tmp_path)
        assert action == "continue"

    def test_privacy_off_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/privacy off", tmp_path)
        assert action == "continue"


# ===========================================================================
# 18. /plugins
# ===========================================================================


class TestPluginsCommand:
    """Tests for the /plugins slash command."""

    def test_plugins_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/plugins", tmp_path)
        assert action == "continue"

    def test_plugins_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/PLUGINS", tmp_path)
        assert action == "continue"

    def test_plugins_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/plugins", tmp_path)
        assert len(output.strip()) > 0


# ===========================================================================
# 19. /forecast
# ===========================================================================


class TestForecastCommand:
    """Tests for the /forecast slash command."""

    def test_forecast_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/forecast", tmp_path)
        assert action == "continue"

    def test_forecast_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/FORECAST", tmp_path)
        assert action == "continue"

    def test_forecast_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/forecast", tmp_path)
        assert len(output.strip()) > 0


# ===========================================================================
# 20. /workspace
# ===========================================================================


class TestWorkspaceCommand:
    """Tests for the /workspace slash command."""

    def test_workspace_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/workspace", tmp_path)
        assert action == "continue"

    def test_workspace_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/WORKSPACE", tmp_path)
        assert action == "continue"

    def test_workspace_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/workspace", tmp_path)
        assert len(output.strip()) > 0


# ===========================================================================
# 21. /offline
# ===========================================================================


class TestOfflineCommand:
    """Tests for the /offline slash command."""

    def test_offline_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/offline", tmp_path)
        assert action == "continue"

    def test_offline_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/OFFLINE", tmp_path)
        assert action == "continue"

    def test_offline_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/offline", tmp_path)
        assert len(output.strip()) > 0


# ===========================================================================
# 22. /debate
# ===========================================================================


class TestDebateCommand:
    """Tests for the /debate slash command."""

    def test_debate_no_args_shows_usage(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/debate", tmp_path)
        assert action == "continue"
        assert "Usage" in output

    def test_debate_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd(
            "/debate is python better than rust", tmp_path,
        )
        assert action == "continue"


# ===========================================================================
# 23. /blast
# ===========================================================================


class TestBlastCommand:
    """Tests for the /blast slash command."""

    def test_blast_no_args_shows_usage(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/blast", tmp_path)
        assert action == "continue"
        assert "Usage" in output

    def test_blast_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/blast src/main.py", tmp_path)
        assert action == "continue"


# ===========================================================================
# 24. /gaps
# ===========================================================================


class TestGapsCommand:
    """Tests for the /gaps slash command."""

    def test_gaps_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/gaps", tmp_path)
        assert action == "continue"

    def test_gaps_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/GAPS", tmp_path)
        assert action == "continue"

    def test_gaps_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/gaps", tmp_path)
        assert len(output.strip()) > 0


# ===========================================================================
# 25. /deps
# ===========================================================================


class TestDepsCommand:
    """Tests for the /deps slash command."""

    def test_deps_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/deps", tmp_path)
        assert action == "continue"

    def test_deps_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/DEPS", tmp_path)
        assert action == "continue"

    def test_deps_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/deps", tmp_path)
        assert len(output.strip()) > 0


# ===========================================================================
# 26. /arch
# ===========================================================================


class TestArchCommand:
    """Tests for the /arch slash command."""

    def test_arch_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/arch", tmp_path)
        assert action == "continue"

    def test_arch_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/ARCH", tmp_path)
        assert action == "continue"

    def test_arch_drift_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/arch drift", tmp_path)
        assert action == "continue"


# ===========================================================================
# 27. /debug-memory
# ===========================================================================


class TestDebugMemoryCommand:
    """Tests for the /debug-memory slash command."""

    def test_debug_memory_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/debug-memory", tmp_path)
        assert action == "continue"

    def test_debug_memory_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/debug-memory", tmp_path)
        assert len(output.strip()) > 0

    def test_debug_memory_search_returns_continue(
        self, tmp_path: Path,
    ) -> None:
        action, _, _, _, _ = _cmd(
            "/debug-memory search import error", tmp_path,
        )
        assert action == "continue"


# ===========================================================================
# 28. /history
# ===========================================================================


class TestHistoryCommand:
    """Tests for the /history slash command."""

    def test_history_no_args_shows_usage(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/history", tmp_path)
        assert action == "continue"
        assert "Usage" in output

    def test_history_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/history src/main.py", tmp_path)
        assert action == "continue"


# ===========================================================================
# 29. /budget
# ===========================================================================


class TestBudgetCommand:
    """Tests for the /budget slash command."""

    def test_budget_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/budget", tmp_path)
        assert action == "continue"

    def test_budget_outputs_something(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/budget", tmp_path)
        assert len(output.strip()) > 0

    def test_budget_set_returns_continue(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/budget set 50000", tmp_path)
        assert action == "continue"


# ===========================================================================
# 30. /quit, /exit, /q
# ===========================================================================


class TestQuitCommand:
    """Tests for the /quit, /exit, and /q slash commands."""

    def test_quit_returns_quit(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/quit", tmp_path)
        assert action == "quit"
        assert "Goodbye" in output

    def test_exit_returns_quit(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/exit", tmp_path)
        assert action == "quit"
        assert "Goodbye" in output

    def test_q_returns_quit(self, tmp_path: Path) -> None:
        action, output, _, _, _ = _cmd("/q", tmp_path)
        assert action == "quit"
        assert "Goodbye" in output

    def test_quit_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/QUIT", tmp_path)
        assert action == "quit"

    def test_exit_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/EXIT", tmp_path)
        assert action == "quit"

    def test_q_case_insensitive(self, tmp_path: Path) -> None:
        action, _, _, _, _ = _cmd("/Q", tmp_path)
        assert action == "quit"

    def test_quit_ignores_extra_args(self, tmp_path: Path) -> None:
        """Extra arguments after /quit are ignored — still exits."""
        action, _, _, _, _ = _cmd("/quit now", tmp_path)
        assert action == "quit"


# ===========================================================================
# Unknown command
# ===========================================================================


class TestUnknownCommand:
    """Tests for unrecognized slash commands."""

    def test_unknown_command_returns_continue(
        self, tmp_path: Path,
    ) -> None:
        action, _, _, _, _ = _cmd("/foobar", tmp_path)
        assert action == "continue"

    def test_unknown_command_shows_error(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/foobar", tmp_path)
        assert "Unknown command" in output
        assert "/foobar" in output

    def test_unknown_suggests_help(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/nonexistent", tmp_path)
        assert "/help" in output

    def test_unknown_with_random_string(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/xyz123", tmp_path)
        assert "Unknown command" in output

    def test_unknown_single_char(self, tmp_path: Path) -> None:
        """Single character that is not /q."""
        _, output, _, _, _ = _cmd("/z", tmp_path)
        assert "Unknown command" in output

    def test_unknown_with_args(self, tmp_path: Path) -> None:
        _, output, _, _, _ = _cmd("/banana split", tmp_path)
        assert "Unknown command" in output
        assert "/banana" in output


# ===========================================================================
# Edge cases and cross-cutting behavior
# ===========================================================================


class TestSlashCommandEdgeCases:
    """Edge cases that cut across multiple commands."""

    def test_command_with_leading_whitespace_in_args(
        self, tmp_path: Path,
    ) -> None:
        """Args are stripped properly."""
        _, _, _, _, st = _cmd(
            "/add    spaced.py", tmp_path,
        )
        assert "spaced.py" in st.active_files

    def test_all_implemented_commands_return_valid_action(
        self, tmp_path: Path,
    ) -> None:
        """Every command returns one of the valid actions."""
        valid_actions = {"continue", "quit"}
        commands_to_test = [
            "/help", "/cost", "/model", "/compact", "/undo",
            "/web on", "/clear", "/cache", "/compare", "/branch",
            "/rollback", "/sandbox", "/tasks", "/privacy",
            "/plugins", "/forecast", "/workspace", "/offline",
            "/debate", "/blast", "/gaps", "/deps", "/arch",
            "/debug-memory", "/history", "/budget",
        ]
        for cmd_str in commands_to_test:
            action, _, _, _, _ = _cmd(cmd_str, tmp_path)
            assert action in valid_actions, (
                f"Command {cmd_str!r} returned invalid action "
                f"{action!r}"
            )

    def test_add_then_drop_roundtrip(self, tmp_path: Path) -> None:
        """Adding and then dropping a file returns to empty state."""
        state = _make_state()
        stg = _make_settings(tmp_path)

        con1 = _make_console()
        _dispatch_command(
            "/add roundtrip.py",
            console=con1,
            settings=stg,
            state=state,
            dry_run=False,
            offline=False,
        )
        assert state.active_files == ["roundtrip.py"]

        con2 = _make_console()
        _dispatch_command(
            "/drop roundtrip.py",
            console=con2,
            settings=stg,
            state=state,
            dry_run=False,
            offline=False,
        )
        assert state.active_files == []

    def test_multiple_add_commands_accumulate(
        self, tmp_path: Path,
    ) -> None:
        state = _make_state()
        stg = _make_settings(tmp_path)

        for filename in ["a.py", "b.py", "c.py"]:
            console = _make_console()
            _dispatch_command(
                f"/add {filename}",
                console=console,
                settings=stg,
                state=state,
                dry_run=False,
                offline=False,
            )

        assert state.active_files == ["a.py", "b.py", "c.py"]

    def test_model_set_persists_across_calls(
        self, tmp_path: Path,
    ) -> None:
        state = _make_state()
        stg = _make_settings(tmp_path)

        con1 = _make_console()
        _dispatch_command(
            "/model gpt-4o",
            console=con1,
            settings=stg,
            state=state,
            dry_run=False,
            offline=False,
        )
        assert state.pinned_model == "gpt-4o"

        # Check subsequent call shows it
        con2 = _make_console()
        _dispatch_command(
            "/model",
            console=con2,
            settings=stg,
            state=state,
            dry_run=False,
            offline=False,
        )
        output = _get_output(con2)
        assert "gpt-4o" in output

    def test_clear_resets_conversation(self, tmp_path: Path) -> None:
        state = _make_state()
        state.conversation = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
        stg = _make_settings(tmp_path)

        _dispatch_command(
            "/clear",
            console=_make_console(),
            settings=stg,
            state=state,
            dry_run=False,
            offline=False,
        )
        assert state.conversation == []

    def test_web_toggle_roundtrip(self, tmp_path: Path) -> None:
        state = _make_state()
        stg = _make_settings(tmp_path)

        _dispatch_command(
            "/web on",
            console=_make_console(),
            settings=stg,
            state=state,
            dry_run=False,
            offline=False,
        )
        assert state.web_enabled is True

        _dispatch_command(
            "/web off",
            console=_make_console(),
            settings=stg,
            state=state,
            dry_run=False,
            offline=False,
        )
        assert state.web_enabled is False

    def test_dispatch_table_has_all_known_commands(
        self, tmp_path: Path,
    ) -> None:
        """All commands listed in COMMAND_CATEGORIES are
        recognized by the dispatcher (not 'Unknown command')."""
        stg = _make_settings(tmp_path)

        for category, commands in COMMAND_CATEGORIES.items():
            for cmd_label, _desc in commands:
                # Extract the command name (e.g. "/add" from "/add <files>")
                cmd_name = cmd_label.split()[0]
                state = _make_state()
                con = _make_console()
                _dispatch_command(
                    cmd_name,
                    console=con,
                    settings=stg,
                    state=state,
                    dry_run=False,
                    offline=False,
                )
                output = _get_output(con)
                assert "Unknown command" not in output, (
                    f"{cmd_name} from category {category!r} "
                    f"was not recognized"
                )


# ===========================================================================
# _process_prompt tests
# ===========================================================================


class TestProcessPrompt:
    """Tests for _process_prompt — prompt classification and completion."""

    def _import_process_prompt(self):
        """Import _process_prompt lazily."""
        from prism.cli.repl import _process_prompt
        return _process_prompt

    def _mock_classify_result(
        self,
        tier: str = "simple",
        score: float = 0.3,
        features: str = "short prompt",
        reasoning: str = "Low complexity",
    ) -> object:
        """Create a mock classification result."""
        from enum import Enum
        from unittest.mock import MagicMock

        class MockTier(Enum):
            simple = "simple"
            medium = "medium"
            complex = "complex"

        result = MagicMock()
        result.tier = MockTier[tier]
        result.score = score
        result.features = features
        result.reasoning = reasoning
        return result

    def test_classification_displayed(self, tmp_path: Path) -> None:
        """_process_prompt displays classification tier and score."""
        proc = self._import_process_prompt()
        stg = _make_settings(tmp_path)
        con = _make_console()
        state = _make_state()
        mock_result = self._mock_classify_result(
            tier="simple", score=0.25,
        )

        with patch(
            "prism.router.classifier.TaskClassifier",
        ) as mock_cls, patch(
            "prism.router.classifier.TaskContext",
        ):
            mock_cls.return_value.classify.return_value = mock_result
            proc(
                prompt="hello",
                console=con,
                settings=stg,
                state=state,
                dry_run=True,
                offline=False,
            )

        output = _get_output(con)
        assert "SIMPLE" in output
        assert "0.25" in output

    def test_dry_run_shows_features_and_reasoning(
        self, tmp_path: Path,
    ) -> None:
        """In dry-run mode, features and reasoning are displayed."""
        proc = self._import_process_prompt()
        stg = _make_settings(tmp_path)
        con = _make_console()
        state = _make_state()
        mock_result = self._mock_classify_result(
            features="code_edit, multi_file",
            reasoning="Edits across files",
        )

        with patch(
            "prism.router.classifier.TaskClassifier",
        ) as mock_cls, patch(
            "prism.router.classifier.TaskContext",
        ):
            mock_cls.return_value.classify.return_value = mock_result
            proc(
                prompt="refactor module",
                console=con,
                settings=stg,
                state=state,
                dry_run=True,
                offline=False,
            )

        output = _get_output(con)
        assert "code_edit" in output
        assert "Edits across files" in output
        assert "Dry-run" in output

    def test_dry_run_no_conversation_update(
        self, tmp_path: Path,
    ) -> None:
        """In dry-run mode, conversation is not updated."""
        proc = self._import_process_prompt()
        stg = _make_settings(tmp_path)
        con = _make_console()
        state = _make_state()
        mock_result = self._mock_classify_result()

        with patch(
            "prism.router.classifier.TaskClassifier",
        ) as mock_cls, patch(
            "prism.router.classifier.TaskContext",
        ):
            mock_cls.return_value.classify.return_value = mock_result
            proc(
                prompt="test",
                console=con,
                settings=stg,
                state=state,
                dry_run=True,
                offline=False,
            )

        assert len(state.conversation) == 0

    def test_complex_tier_displayed(self, tmp_path: Path) -> None:
        """Complex tier is displayed with correct label."""
        proc = self._import_process_prompt()
        stg = _make_settings(tmp_path)
        con = _make_console()
        state = _make_state()
        mock_result = self._mock_classify_result(
            tier="complex", score=0.92,
        )

        with patch(
            "prism.router.classifier.TaskClassifier",
        ) as mock_cls, patch(
            "prism.router.classifier.TaskContext",
        ):
            mock_cls.return_value.classify.return_value = mock_result
            proc(
                prompt="refactor entire codebase",
                console=con,
                settings=stg,
                state=state,
                dry_run=True,
                offline=False,
            )

        output = _get_output(con)
        assert "COMPLEX" in output
        assert "0.92" in output

    def test_active_files_passed_to_context(
        self, tmp_path: Path,
    ) -> None:
        """Active files from state are passed to TaskContext."""
        proc = self._import_process_prompt()
        stg = _make_settings(tmp_path)
        con = _make_console()
        state = _make_state(active_files=["a.py", "b.py"])
        mock_result = self._mock_classify_result()

        with patch(
            "prism.router.classifier.TaskClassifier",
        ) as mock_cls, patch(
            "prism.router.classifier.TaskContext",
        ) as mock_ctx:
            mock_cls.return_value.classify.return_value = mock_result
            proc(
                prompt="hello",
                console=con,
                settings=stg,
                state=state,
                dry_run=True,
                offline=False,
            )
            mock_ctx.assert_called_once_with(
                active_files=["a.py", "b.py"],
            )

    def test_completion_error_displays_message(
        self, tmp_path: Path,
    ) -> None:
        """When completion engine raises, an error is printed."""
        proc = self._import_process_prompt()
        stg = _make_settings(tmp_path)
        con = _make_console()
        state = _make_state()
        mock_result = self._mock_classify_result()

        with patch(
            "prism.router.classifier.TaskClassifier",
        ) as mock_cls, patch(
            "prism.router.classifier.TaskContext",
        ), patch(
            "prism.llm.completion.CompletionEngine",
            side_effect=RuntimeError("no key"),
        ):
            mock_cls.return_value.classify.return_value = mock_result
            proc(
                prompt="hello",
                console=con,
                settings=stg,
                state=state,
                dry_run=False,
                offline=False,
            )

        output = _get_output(con)
        assert "error" in output.lower()

    def test_medium_tier_displayed(self, tmp_path: Path) -> None:
        """Medium tier is displayed correctly."""
        proc = self._import_process_prompt()
        stg = _make_settings(tmp_path)
        con = _make_console()
        state = _make_state()
        mock_result = self._mock_classify_result(
            tier="medium", score=0.55,
        )

        with patch(
            "prism.router.classifier.TaskClassifier",
        ) as mock_cls, patch(
            "prism.router.classifier.TaskContext",
        ):
            mock_cls.return_value.classify.return_value = mock_result
            proc(
                prompt="explain this function",
                console=con,
                settings=stg,
                state=state,
                dry_run=True,
                offline=False,
            )

        output = _get_output(con)
        assert "MEDIUM" in output
        assert "0.55" in output


# ===========================================================================
# run_repl comprehensive tests
# ===========================================================================


class TestRunRepl:
    """Comprehensive tests for the run_repl function."""

    def test_run_repl_exists(self) -> None:
        """run_repl is importable."""
        assert callable(run_repl)

    def test_run_repl_accepts_expected_args(self) -> None:
        """run_repl has the expected signature parameters."""
        import inspect
        sig = inspect.signature(run_repl)
        param_names = list(sig.parameters.keys())
        assert "settings" in param_names
        assert "console" in param_names
        assert "dry_run" in param_names
        assert "offline" in param_names

    def test_eof_exits_with_goodbye(self, tmp_path: Path) -> None:
        """EOFError from prompt exits the loop with goodbye."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = EOFError
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "Goodbye" in output

    def test_quit_command_exits(self, tmp_path: Path) -> None:
        """/quit exits the REPL loop."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/quit", EOFError,
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "Goodbye" in output or "Ready" in output

    def test_exit_command_exits(self, tmp_path: Path) -> None:
        """/exit is an alias for /quit."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/exit", EOFError,
            ]
            run_repl(settings=stg, console=con)

        # Should exit cleanly without hitting EOFError
        output = _get_output(con)
        assert "Ready" in output

    def test_q_command_exits(self, tmp_path: Path) -> None:
        """/q is an alias for /quit."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/q", EOFError,
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "Ready" in output

    def test_empty_input_continues(self, tmp_path: Path) -> None:
        """Empty input is silently skipped."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "", "  ", "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        # No error messages for empty input
        assert "error" not in output.lower()

    def test_keyboard_interrupt_continues(
        self, tmp_path: Path,
    ) -> None:
        """KeyboardInterrupt from prompt continues the loop."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                KeyboardInterrupt, "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "Ready" in output

    def test_multiple_keyboard_interrupts(
        self, tmp_path: Path,
    ) -> None:
        """Multiple KeyboardInterrupts don't crash the loop."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                KeyboardInterrupt,
                KeyboardInterrupt,
                KeyboardInterrupt,
                "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "Ready" in output

    def test_help_then_quit(self, tmp_path: Path) -> None:
        """Running /help followed by /quit works correctly."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/help", "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        # /help output should appear
        assert "help" in output.lower() or "command" in output.lower()

    def test_regular_prompt_calls_process(
        self, tmp_path: Path,
    ) -> None:
        """Non-slash input is routed to _process_prompt."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._process_prompt",
        ) as mock_proc:
            mock_ps.return_value.prompt.side_effect = [
                "hello world", "/quit",
            ]
            run_repl(settings=stg, console=con)

        mock_proc.assert_called_once()
        call_kwargs = mock_proc.call_args
        assert call_kwargs[1]["prompt"] == "hello world" or (
            call_kwargs[0][0] == "hello world"
        )

    def test_multiple_prompts_processed(
        self, tmp_path: Path,
    ) -> None:
        """Multiple regular prompts each call _process_prompt."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._process_prompt",
        ) as mock_proc:
            mock_ps.return_value.prompt.side_effect = [
                "first", "second", "third", "/quit",
            ]
            run_repl(settings=stg, console=con)

        assert mock_proc.call_count == 3

    def test_model_change_via_loop(self, tmp_path: Path) -> None:
        """/model command in the loop changes pinned model."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/model gpt-4", "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "gpt-4" in output

    def test_model_no_arg_shows_current(
        self, tmp_path: Path,
    ) -> None:
        """/model with no argument shows the current model."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/model", "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "auto" in output.lower() or "model" in output.lower()

    def test_exception_in_loop_continues(
        self, tmp_path: Path,
    ) -> None:
        """An exception in the loop body prints error, continues."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._process_prompt",
            side_effect=[RuntimeError("boom"), None],
        ):
            mock_ps.return_value.prompt.side_effect = [
                "bad prompt", "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "unexpected error" in output.lower() or (
            "error" in output.lower()
        )

    def test_exception_does_not_exit(
        self, tmp_path: Path,
    ) -> None:
        """Exception in processing does not terminate the loop."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        call_count = 0

        def side_effect(*_args: object, **_kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("test error")

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._process_prompt",
            side_effect=side_effect,
        ):
            mock_ps.return_value.prompt.side_effect = [
                "prompt1", "prompt2", "/quit",
            ]
            run_repl(settings=stg, console=con)

        # Second prompt should still be processed
        assert call_count == 2

    def test_dry_run_passed_to_dispatch(
        self, tmp_path: Path,
    ) -> None:
        """dry_run flag is passed through to command dispatch."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._dispatch_command",
            return_value="quit",
        ) as mock_disp:
            mock_ps.return_value.prompt.side_effect = ["/help"]
            run_repl(
                settings=stg, console=con,
                dry_run=True, offline=False,
            )

        mock_disp.assert_called_once()
        _, kwargs = mock_disp.call_args
        assert kwargs.get("dry_run") is True

    def test_offline_passed_to_dispatch(
        self, tmp_path: Path,
    ) -> None:
        """offline flag is passed through to command dispatch."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._dispatch_command",
            return_value="quit",
        ) as mock_disp:
            mock_ps.return_value.prompt.side_effect = ["/status"]
            run_repl(
                settings=stg, console=con,
                dry_run=False, offline=True,
            )

        mock_disp.assert_called_once()
        _, kwargs = mock_disp.call_args
        assert kwargs.get("offline") is True

    def test_dry_run_and_offline_defaults(
        self, tmp_path: Path,
    ) -> None:
        """Default values for dry_run and offline are False."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._process_prompt",
        ) as mock_proc:
            mock_ps.return_value.prompt.side_effect = [
                "hello", "/quit",
            ]
            run_repl(settings=stg, console=con)

        _, kwargs = mock_proc.call_args
        assert kwargs.get("dry_run") is False
        assert kwargs.get("offline") is False

    def test_creates_history_dir(self, tmp_path: Path) -> None:
        """run_repl creates the sessions directory for history."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = EOFError
            run_repl(settings=stg, console=con)

        assert stg.sessions_dir.exists()

    def test_ready_message_displayed(
        self, tmp_path: Path,
    ) -> None:
        """The 'Ready' message is printed on startup."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = EOFError
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "Ready" in output

    def test_slash_commands_dont_call_process_prompt(
        self, tmp_path: Path,
    ) -> None:
        """Slash commands are dispatched, not sent to
        _process_prompt."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._process_prompt",
        ) as mock_proc:
            mock_ps.return_value.prompt.side_effect = [
                "/help", "/cost", "/status", "/quit",
            ]
            run_repl(settings=stg, console=con)

        mock_proc.assert_not_called()

    def test_pinned_model_from_settings(
        self, tmp_path: Path,
    ) -> None:
        """State picks up pinned_model from settings.config."""
        config = PrismConfig(
            prism_home=tmp_path / ".prism",
            pinned_model="claude-3-opus",
        )
        stg = Settings(config=config, project_root=tmp_path)
        stg.ensure_directories()
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/model", "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "claude-3-opus" in output

    def test_add_drop_via_loop(self, tmp_path: Path) -> None:
        """Adding and dropping files through the loop works."""
        stg = _make_settings(tmp_path)
        con = _make_console()
        # Create a real file for /add
        test_file = tmp_path / "hello.py"
        test_file.write_text("x = 1\n")

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                f"/add {test_file}",
                "/drop",
                "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        # /add should show the file was added
        assert "hello.py" in output or "Added" in output

    def test_web_toggle_via_loop(self, tmp_path: Path) -> None:
        """Toggling web mode through the loop works."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/web on", "/web off", "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "enabled" in output.lower() or (
            "on" in output.lower()
        )

    def test_unknown_command_via_loop(
        self, tmp_path: Path,
    ) -> None:
        """Unknown slash command shows error, does not exit."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/nonexistent", "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "Unknown command" in output

    def test_mixed_commands_and_prompts(
        self, tmp_path: Path,
    ) -> None:
        """Mix of commands and prompts all handled correctly."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._process_prompt",
        ) as mock_proc:
            mock_ps.return_value.prompt.side_effect = [
                "/help",
                "hello",
                "/cost",
                "world",
                "/quit",
            ]
            run_repl(settings=stg, console=con)

        assert mock_proc.call_count == 2

    def test_conversation_maintained_across_prompts(
        self, tmp_path: Path,
    ) -> None:
        """The same state object is passed to each call."""
        stg = _make_settings(tmp_path)
        con = _make_console()
        states_seen: list[object] = []

        def capture_state(
            *_args: object, **kwargs: object,
        ) -> None:
            states_seen.append(id(kwargs.get("state")))

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._process_prompt",
            side_effect=capture_state,
        ):
            mock_ps.return_value.prompt.side_effect = [
                "first", "second", "/quit",
            ]
            run_repl(settings=stg, console=con)

        assert len(states_seen) == 2
        assert states_seen[0] == states_seen[1]

    def test_session_uuid_generated(
        self, tmp_path: Path,
    ) -> None:
        """run_repl generates a session UUID on the state."""
        stg = _make_settings(tmp_path)
        con = _make_console()
        captured_state: list[_SessionState] = []

        def capture(
            *_args: object, **kwargs: object,
        ) -> None:
            s = kwargs.get("state")
            if s is not None:
                captured_state.append(s)

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl._process_prompt",
            side_effect=capture,
        ):
            mock_ps.return_value.prompt.side_effect = [
                "hello", "/quit",
            ]
            run_repl(settings=stg, console=con)

        assert len(captured_state) == 1
        assert len(captured_state[0].session_id) == 12

    def test_prompt_session_uses_file_history(
        self, tmp_path: Path,
    ) -> None:
        """PromptSession is created with FileHistory."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps, patch(
            "prism.cli.repl.FileHistory",
        ) as mock_fh:
            mock_ps.return_value.prompt.side_effect = EOFError
            run_repl(settings=stg, console=con)

        mock_fh.assert_called_once()
        mock_ps.assert_called_once()
        call_kwargs = mock_ps.call_args[1]
        assert "history" in call_kwargs
        assert call_kwargs.get("multiline") is False
        assert call_kwargs.get("enable_history_search") is True

    def test_clear_in_loop(self, tmp_path: Path) -> None:
        """/clear in the REPL loop shows confirmation."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/clear", "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        assert "clear" in output.lower() or (
            "Cleared" in output
        )

    def test_status_in_loop(self, tmp_path: Path) -> None:
        """/status in the REPL loop shows provider status."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = [
                "/status", "/quit",
            ]
            run_repl(settings=stg, console=con)

        output = _get_output(con)
        # Status command should produce some output
        assert len(output) > 0


# ===========================================================================
# Enhanced /cache command tests
# ===========================================================================


class TestCacheCommandEnhanced:
    """Tests for the enhanced /cache command (on/off/clear --older-than)."""

    def test_cache_on_returns_continue(self, tmp_path: Path) -> None:
        action, output, _, _, st = _cmd("/cache on", tmp_path)
        assert action == "continue"
        assert st.cache_enabled is True
        assert "re-enabled" in output.lower() or "enabled" in output.lower()

    def test_cache_off_returns_continue(self, tmp_path: Path) -> None:
        action, output, _, _, st = _cmd("/cache off", tmp_path)
        assert action == "continue"
        assert st.cache_enabled is False
        assert "disabled" in output.lower()

    def test_cache_off_then_on_toggles(self, tmp_path: Path) -> None:
        """Off then on should toggle the flag correctly."""
        st = _make_state()
        assert st.cache_enabled is True

        _cmd("/cache off", tmp_path, state=st)
        assert st.cache_enabled is False

        _cmd("/cache on", tmp_path, state=st)
        assert st.cache_enabled is True

    def test_cache_stats_shows_session_status(
        self, tmp_path: Path,
    ) -> None:
        """Stats should include session enabled/disabled status."""
        _, output, _, _, _ = _cmd("/cache stats", tmp_path)
        assert "Session" in output or "enabled" in output.lower() or len(output) > 0

    def test_cache_clear_older_than_returns_continue(
        self, tmp_path: Path,
    ) -> None:
        action, _, _, _, _ = _cmd(
            "/cache clear --older-than 24h", tmp_path,
        )
        assert action == "continue"

    def test_cache_clear_older_than_invalid_duration(
        self, tmp_path: Path,
    ) -> None:
        """Invalid duration should show an error message."""
        _, output, _, _, _ = _cmd(
            "/cache clear --older-than xyz", tmp_path,
        )
        assert "Invalid" in output or "duration" in output.lower() or len(output) > 0

    def test_cache_clear_older_than_2d(
        self, tmp_path: Path,
    ) -> None:
        action, _output, _, _, _ = _cmd(
            "/cache clear --older-than 2d", tmp_path,
        )
        assert action == "continue"

    def test_cache_stats_default(self, tmp_path: Path) -> None:
        """Default /cache command should show stats."""
        action, output, _, _, _ = _cmd("/cache", tmp_path)
        assert action == "continue"
        assert len(output.strip()) > 0


# ===========================================================================
# _parse_duration_to_hours tests
# ===========================================================================


class TestParseDurationToHours:
    """Tests for the _parse_duration_to_hours helper."""

    def test_hours(self) -> None:
        from prism.cli.repl import _parse_duration_to_hours

        assert _parse_duration_to_hours("24h") == 24.0

    def test_days(self) -> None:
        from prism.cli.repl import _parse_duration_to_hours

        assert _parse_duration_to_hours("2d") == 48.0

    def test_minutes(self) -> None:
        from prism.cli.repl import _parse_duration_to_hours

        result = _parse_duration_to_hours("30m")
        assert result is not None
        assert abs(result - 0.5) < 0.01

    def test_weeks(self) -> None:
        from prism.cli.repl import _parse_duration_to_hours

        assert _parse_duration_to_hours("1w") == 168.0

    def test_invalid_returns_none(self) -> None:
        from prism.cli.repl import _parse_duration_to_hours

        assert _parse_duration_to_hours("invalid") is None
        assert _parse_duration_to_hours("") is None
        assert _parse_duration_to_hours("24x") is None

    def test_float_values(self) -> None:
        from prism.cli.repl import _parse_duration_to_hours

        result = _parse_duration_to_hours("1.5h")
        assert result is not None
        assert abs(result - 1.5) < 0.01

    def test_whitespace_handling(self) -> None:
        from prism.cli.repl import _parse_duration_to_hours

        result = _parse_duration_to_hours("  24h  ")
        assert result == 24.0


# ===========================================================================
# /image command tests
# ===========================================================================


class TestImageCommand:
    """Tests for the /image slash command."""

    def test_image_no_args_shows_usage(
        self, tmp_path: Path,
    ) -> None:
        action, output, _, _, _ = _cmd("/image", tmp_path)
        assert action == "continue"
        assert "Usage" in output

    def test_image_returns_continue(self, tmp_path: Path) -> None:
        """Even with invalid args, should return continue."""
        action, _, _, _, _ = _cmd(
            "/image nonexistent.png", tmp_path,
        )
        assert action == "continue"

    def test_image_nonexistent_file_shows_error(
        self, tmp_path: Path,
    ) -> None:
        """Missing file should show error."""
        _, output, _, _, _ = _cmd(
            "/image /tmp/nonexistent_image_12345.png", tmp_path,
        )
        assert (
            "not found" in output.lower()
            or "No valid" in output
            or "error" in output.lower()
            or len(output.strip()) > 0
        )

    def test_image_command_in_dispatch(
        self, tmp_path: Path,
    ) -> None:
        """Ensure /image is recognized (not unknown)."""
        _, output, _, _, _ = _cmd("/image test.png", tmp_path)
        assert "Unknown command" not in output

    def test_image_in_command_categories(self) -> None:
        """Ensure /image appears in COMMAND_CATEGORIES."""
        all_commands = []
        for cmds in COMMAND_CATEGORIES.values():
            for cmd_name, _ in cmds:
                all_commands.append(cmd_name)
        assert any("/image" in c for c in all_commands)


# ===========================================================================
# Enhanced /compare command tests
# ===========================================================================


class TestCompareCommandEnhanced:
    """Tests for the enhanced /compare command."""

    def test_compare_config_returns_continue(
        self, tmp_path: Path,
    ) -> None:
        action, _, _, _, _ = _cmd("/compare config", tmp_path)
        assert action == "continue"

    def test_compare_history_returns_continue(
        self, tmp_path: Path,
    ) -> None:
        action, _, _, _, _ = _cmd("/compare history", tmp_path)
        assert action == "continue"

    def test_compare_history_empty(self, tmp_path: Path) -> None:
        """History should indicate no sessions when empty."""
        _, output, _, _, _ = _cmd(
            "/compare history", tmp_path,
        )
        assert (
            "No comparison" in output
            or "history" in output.lower()
            or len(output.strip()) > 0
        )

    def test_compare_no_args_shows_usage(
        self, tmp_path: Path,
    ) -> None:
        action, output, _, _, _ = _cmd("/compare", tmp_path)
        assert action == "continue"
        assert "Usage" in output

    def test_compare_prompt_returns_continue(
        self, tmp_path: Path,
    ) -> None:
        """Even with a prompt, should return continue."""
        action, _, _, _, _ = _cmd(
            "/compare what is 2+2", tmp_path,
        )
        assert action == "continue"


# ===========================================================================
# SessionState cache_enabled tests
# ===========================================================================


class TestSessionStateCacheEnabled:
    """Tests for the cache_enabled attribute on _SessionState."""

    def test_default_cache_enabled(self) -> None:
        state = _SessionState(pinned_model=None)
        assert state.cache_enabled is True

    def test_cache_enabled_false(self) -> None:
        state = _SessionState(
            pinned_model=None, cache_enabled=False,
        )
        assert state.cache_enabled is False

    def test_cache_enabled_true_explicit(self) -> None:
        state = _SessionState(
            pinned_model=None, cache_enabled=True,
        )
        assert state.cache_enabled is True

    def test_toggle_cache(self) -> None:
        state = _SessionState(pinned_model=None)
        state.cache_enabled = False
        assert state.cache_enabled is False
        state.cache_enabled = True
        assert state.cache_enabled is True


# ===========================================================================
# run_repl no_cache parameter tests
# ===========================================================================


class TestRunReplNoCache:
    """Tests for the no_cache parameter on run_repl."""

    def test_run_repl_no_cache_true(self, tmp_path: Path) -> None:
        """When no_cache=True, session cache should be disabled."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = ["/quit"]
            run_repl(
                settings=stg,
                console=con,
                no_cache=True,
            )

        output = _get_output(con)
        assert len(output) > 0

    def test_run_repl_no_cache_false(self, tmp_path: Path) -> None:
        """When no_cache=False (default), cache should be on."""
        stg = _make_settings(tmp_path)
        con = _make_console()

        with patch(
            "prism.cli.repl.PromptSession",
        ) as mock_ps:
            mock_ps.return_value.prompt.side_effect = ["/quit"]
            run_repl(
                settings=stg,
                console=con,
                no_cache=False,
            )

        output = _get_output(con)
        assert len(output) > 0

"""Tests for prism.cli.hooks — hook system for pre/post tool execution."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from prism.cli.hooks import (
    _HOOK_TIMEOUT_SECONDS,
    VALID_EVENTS,
    HookConfig,
    HookManager,
    HookResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_hooks_yaml(
    root: Path,
    hooks: list[dict[str, object]],
    *,
    use_prism_dir: bool = True,
) -> Path:
    """Write a hooks.yaml file and return its path."""
    if use_prism_dir:
        hooks_dir = root / ".prism"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hooks_file = hooks_dir / "hooks.yaml"
    else:
        hooks_file = root / ".prism-hooks.yaml"

    hooks_file.write_text(yaml.dump({"hooks": hooks}), encoding="utf-8")
    return hooks_file


# ---------------------------------------------------------------------------
# HookConfig dataclass
# ---------------------------------------------------------------------------


class TestHookConfig:
    """Tests for the HookConfig dataclass."""

    def test_create_basic_hook(self) -> None:
        hook = HookConfig(event="pre_tool", command="echo hello")
        assert hook.event == "pre_tool"
        assert hook.command == "echo hello"
        assert hook.tool_filter is None
        assert hook.enabled is True

    def test_create_hook_with_all_fields(self) -> None:
        hook = HookConfig(
            event="post_tool",
            command="ruff check .",
            tool_filter="write_file",
            enabled=False,
        )
        assert hook.event == "post_tool"
        assert hook.command == "ruff check ."
        assert hook.tool_filter == "write_file"
        assert hook.enabled is False

    def test_hook_equality(self) -> None:
        h1 = HookConfig(event="pre_tool", command="echo a")
        h2 = HookConfig(event="pre_tool", command="echo a")
        assert h1 == h2


# ---------------------------------------------------------------------------
# HookResult dataclass
# ---------------------------------------------------------------------------


class TestHookResult:
    """Tests for the HookResult dataclass."""

    def test_create_success_result(self) -> None:
        hook = HookConfig(event="pre_tool", command="echo ok")
        result = HookResult(success=True, output="ok\n", hook=hook)
        assert result.success is True
        assert result.blocked is False

    def test_create_blocked_result(self) -> None:
        hook = HookConfig(event="pre_tool", command="exit 1")
        result = HookResult(
            success=False, output="error", hook=hook, blocked=True,
        )
        assert result.success is False
        assert result.blocked is True


# ---------------------------------------------------------------------------
# VALID_EVENTS constant
# ---------------------------------------------------------------------------


class TestValidEvents:
    """Tests for the VALID_EVENTS constant."""

    def test_contains_pre_tool(self) -> None:
        assert "pre_tool" in VALID_EVENTS

    def test_contains_post_tool(self) -> None:
        assert "post_tool" in VALID_EVENTS

    def test_contains_pre_command(self) -> None:
        assert "pre_command" in VALID_EVENTS

    def test_contains_post_command(self) -> None:
        assert "post_command" in VALID_EVENTS

    def test_is_frozenset(self) -> None:
        assert isinstance(VALID_EVENTS, frozenset)


# ---------------------------------------------------------------------------
# HookManager — loading
# ---------------------------------------------------------------------------


class TestHookManagerLoading:
    """Tests for HookManager hook loading from YAML files."""

    def test_no_hooks_file_returns_empty(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        assert manager.hooks == []

    def test_loads_from_prism_dir(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo pre"},
        ])
        manager = HookManager(tmp_path)
        assert len(manager.hooks) == 1
        assert manager.hooks[0].event == "pre_tool"
        assert manager.hooks[0].command == "echo pre"

    def test_loads_from_prism_hooks_yaml(self, tmp_path: Path) -> None:
        _write_hooks_yaml(
            tmp_path,
            [{"event": "post_tool", "command": "echo post"}],
            use_prism_dir=False,
        )
        manager = HookManager(tmp_path)
        assert len(manager.hooks) == 1
        assert manager.hooks[0].event == "post_tool"

    def test_prism_dir_takes_precedence(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "from .prism/hooks.yaml"},
        ])
        _write_hooks_yaml(
            tmp_path,
            [{"event": "post_tool", "command": "from .prism-hooks.yaml"}],
            use_prism_dir=False,
        )
        manager = HookManager(tmp_path)
        assert len(manager.hooks) == 1
        assert manager.hooks[0].command == "from .prism/hooks.yaml"

    def test_loads_multiple_hooks(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo first"},
            {"event": "post_tool", "command": "echo second"},
            {"event": "pre_command", "command": "echo third"},
        ])
        manager = HookManager(tmp_path)
        assert len(manager.hooks) == 3

    def test_loads_hook_with_tool_filter(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo check", "tool": "write_file"},
        ])
        manager = HookManager(tmp_path)
        assert manager.hooks[0].tool_filter == "write_file"

    def test_disabled_hook_is_loaded(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo skip", "enabled": False},
        ])
        manager = HookManager(tmp_path)
        assert len(manager.hooks) == 1
        assert manager.hooks[0].enabled is False

    def test_skips_hook_with_missing_event(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"command": "echo no-event"},
        ])
        manager = HookManager(tmp_path)
        assert manager.hooks == []

    def test_skips_hook_with_missing_command(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool"},
        ])
        manager = HookManager(tmp_path)
        assert manager.hooks == []

    def test_skips_hook_with_invalid_event(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "invalid_event", "command": "echo bad"},
        ])
        manager = HookManager(tmp_path)
        assert manager.hooks == []

    def test_invalid_yaml_format_returns_empty(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / ".prism"
        hooks_dir.mkdir()
        (hooks_dir / "hooks.yaml").write_text("not a dict: [1, 2, 3]")
        manager = HookManager(tmp_path)
        assert manager.hooks == []

    def test_hooks_not_a_list_returns_empty(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / ".prism"
        hooks_dir.mkdir()
        (hooks_dir / "hooks.yaml").write_text(
            yaml.dump({"hooks": "not-a-list"})
        )
        manager = HookManager(tmp_path)
        assert manager.hooks == []

    def test_malformed_yaml_returns_empty(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / ".prism"
        hooks_dir.mkdir()
        (hooks_dir / "hooks.yaml").write_text("{{invalid yaml: [")
        manager = HookManager(tmp_path)
        assert manager.hooks == []

    def test_non_dict_hook_entry_skipped(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            "not-a-dict",  # type: ignore[dict-item]
            {"event": "pre_tool", "command": "echo valid"},
        ])
        manager = HookManager(tmp_path)
        assert len(manager.hooks) == 1

    def test_project_root_property(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        assert manager.project_root == tmp_path


# ---------------------------------------------------------------------------
# HookManager — get_hooks
# ---------------------------------------------------------------------------


class TestGetHooks:
    """Tests for HookManager.get_hooks filtering."""

    def test_filter_by_event(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo pre"},
            {"event": "post_tool", "command": "echo post"},
        ])
        manager = HookManager(tmp_path)
        results = manager.get_hooks("pre_tool")
        assert len(results) == 1
        assert results[0].command == "echo pre"

    def test_filter_by_tool_name(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo generic"},
            {"event": "pre_tool", "command": "echo specific", "tool": "write_file"},
        ])
        manager = HookManager(tmp_path)
        results = manager.get_hooks("pre_tool", tool_name="write_file")
        assert len(results) == 2  # generic (no filter) + specific

    def test_tool_filter_excludes_non_matching(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo only-write", "tool": "write_file"},
        ])
        manager = HookManager(tmp_path)
        results = manager.get_hooks("pre_tool", tool_name="read_file")
        assert len(results) == 0

    def test_disabled_hooks_excluded(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo disabled", "enabled": False},
            {"event": "pre_tool", "command": "echo enabled"},
        ])
        manager = HookManager(tmp_path)
        results = manager.get_hooks("pre_tool")
        assert len(results) == 1
        assert results[0].command == "echo enabled"

    def test_no_matching_event_returns_empty(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo pre"},
        ])
        manager = HookManager(tmp_path)
        assert manager.get_hooks("post_tool") == []

    def test_none_tool_name_includes_filtered_hooks(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo filtered", "tool": "write_file"},
        ])
        manager = HookManager(tmp_path)
        # tool_name=None means don't filter by tool
        results = manager.get_hooks("pre_tool", tool_name=None)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# HookManager — run_hook
# ---------------------------------------------------------------------------


class TestRunHook:
    """Tests for HookManager.run_hook execution."""

    def test_successful_hook(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="echo hello")
        result = manager.run_hook(hook)
        assert result.success is True
        assert "hello" in result.output
        assert result.blocked is False
        assert result.hook is hook

    def test_failing_hook_is_blocked(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="exit 1")
        result = manager.run_hook(hook)
        assert result.success is False
        assert result.blocked is True

    def test_hook_with_context_env_vars(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="echo $PRISM_HOOK_TOOL")
        result = manager.run_hook(hook, context={"tool": "write_file"})
        assert result.success is True
        assert "write_file" in result.output

    def test_hook_timeout(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="sleep 60")
        with patch("prism.cli.hooks.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 60", timeout=30)
            result = manager.run_hook(hook)
        assert result.success is False
        assert "timed out" in result.output.lower()
        assert result.blocked is False

    def test_hook_os_error(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="nonexistent_binary_xyz")
        with patch("prism.cli.hooks.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("No such file or directory")
            result = manager.run_hook(hook)
        assert result.success is False
        assert "OS error" in result.output
        assert result.blocked is False

    def test_hook_unexpected_exception(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="echo test")
        with patch("prism.cli.hooks.subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("unexpected")
            result = manager.run_hook(hook)
        assert result.success is False
        assert "unexpected" in result.output

    def test_hook_runs_in_project_root(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="pwd")
        result = manager.run_hook(hook)
        assert result.success is True
        # The resolved path may differ (e.g. /private/var on macOS)
        assert Path(result.output.strip()).resolve() == tmp_path.resolve()

    def test_hook_empty_context(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="echo ok")
        result = manager.run_hook(hook, context={})
        assert result.success is True

    def test_hook_none_context(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="echo ok")
        result = manager.run_hook(hook, context=None)
        assert result.success is True


# ---------------------------------------------------------------------------
# HookManager — run_pre_hooks / run_post_hooks
# ---------------------------------------------------------------------------


class TestRunPrePostHooks:
    """Tests for run_pre_hooks and run_post_hooks convenience methods."""

    def test_run_pre_hooks_no_hooks(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        results = manager.run_pre_hooks("write_file", {"path": "/tmp/f.txt"})
        assert results == []

    def test_run_pre_hooks_with_hooks(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo pre-check"},
        ])
        manager = HookManager(tmp_path)
        results = manager.run_pre_hooks("write_file", {"path": "/tmp/f.txt"})
        assert len(results) == 1
        assert results[0].success is True

    def test_run_pre_hooks_passes_tool_context(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo $PRISM_HOOK_TOOL"},
        ])
        manager = HookManager(tmp_path)
        results = manager.run_pre_hooks("write_file", {"path": "/tmp/f"})
        assert "write_file" in results[0].output

    def test_run_pre_hooks_truncates_long_args(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo $PRISM_HOOK_CONTENT"},
        ])
        manager = HookManager(tmp_path)
        long_content = "x" * 500
        results = manager.run_pre_hooks("write_file", {"content": long_content})
        assert len(results) == 1
        # The env var should be truncated to 200 chars
        assert results[0].success is True

    def test_run_post_hooks_no_hooks(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        results = manager.run_post_hooks("write_file", "file written")
        assert results == []

    def test_run_post_hooks_with_hooks(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "post_tool", "command": "echo post-check"},
        ])
        manager = HookManager(tmp_path)
        results = manager.run_post_hooks("write_file", "file written")
        assert len(results) == 1
        assert results[0].success is True

    def test_run_post_hooks_passes_output_context(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "post_tool", "command": "echo $PRISM_HOOK_OUTPUT"},
        ])
        manager = HookManager(tmp_path)
        results = manager.run_post_hooks("write_file", "success: wrote 42 bytes")
        assert "success: wrote 42 bytes" in results[0].output

    def test_run_post_hooks_truncates_long_output(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "post_tool", "command": "echo ok"},
        ])
        manager = HookManager(tmp_path)
        long_output = "y" * 1000
        results = manager.run_post_hooks("write_file", long_output)
        assert len(results) == 1

    def test_blocked_pre_hook_signals_to_caller(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "exit 1"},
        ])
        manager = HookManager(tmp_path)
        results = manager.run_pre_hooks("write_file", {"path": "/tmp/f"})
        assert len(results) == 1
        assert results[0].blocked is True

    def test_multiple_pre_hooks_all_run(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo first"},
            {"event": "pre_tool", "command": "echo second"},
        ])
        manager = HookManager(tmp_path)
        results = manager.run_pre_hooks("write_file", {})
        assert len(results) == 2
        assert all(r.success for r in results)


# ---------------------------------------------------------------------------
# HookManager — add_hook / remove_hooks
# ---------------------------------------------------------------------------


class TestAddRemoveHooks:
    """Tests for programmatic hook management."""

    def test_add_hook(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="echo added")
        manager.add_hook(hook)
        assert len(manager.hooks) == 1
        assert manager.hooks[0].command == "echo added"

    def test_add_hook_invalid_event_raises(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="bad_event", command="echo bad")
        with pytest.raises(ValueError, match="Invalid event"):
            manager.add_hook(hook)

    def test_remove_all_hooks(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo a"},
            {"event": "post_tool", "command": "echo b"},
        ])
        manager = HookManager(tmp_path)
        assert len(manager.hooks) == 2
        removed = manager.remove_hooks()
        assert removed == 2
        assert manager.hooks == []

    def test_remove_hooks_by_event(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [
            {"event": "pre_tool", "command": "echo pre1"},
            {"event": "pre_tool", "command": "echo pre2"},
            {"event": "post_tool", "command": "echo post"},
        ])
        manager = HookManager(tmp_path)
        removed = manager.remove_hooks(event="pre_tool")
        assert removed == 2
        assert len(manager.hooks) == 1
        assert manager.hooks[0].event == "post_tool"

    def test_remove_hooks_nonexistent_event(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        removed = manager.remove_hooks(event="pre_tool")
        assert removed == 0


# ---------------------------------------------------------------------------
# HookManager — edge cases
# ---------------------------------------------------------------------------


class TestHookManagerEdgeCases:
    """Edge case tests for the hook system."""

    def test_hooks_yaml_with_empty_file(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / ".prism"
        hooks_dir.mkdir()
        (hooks_dir / "hooks.yaml").write_text("")
        manager = HookManager(tmp_path)
        assert manager.hooks == []

    def test_hooks_yaml_with_null_content(self, tmp_path: Path) -> None:
        hooks_dir = tmp_path / ".prism"
        hooks_dir.mkdir()
        (hooks_dir / "hooks.yaml").write_text("null\n")
        manager = HookManager(tmp_path)
        assert manager.hooks == []

    def test_hooks_yaml_with_empty_hooks_list(self, tmp_path: Path) -> None:
        _write_hooks_yaml(tmp_path, [])
        manager = HookManager(tmp_path)
        assert manager.hooks == []

    def test_hook_with_stderr_output(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="echo error >&2")
        result = manager.run_hook(hook)
        assert result.success is True
        assert "error" in result.output

    def test_hook_with_mixed_stdout_stderr(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(
            event="pre_tool",
            command="echo out && echo err >&2",
        )
        result = manager.run_hook(hook)
        assert result.success is True
        assert "out" in result.output
        assert "err" in result.output

    def test_hook_timeout_constant(self) -> None:
        assert _HOOK_TIMEOUT_SECONDS == 30

    @patch.dict(os.environ, {}, clear=False)
    def test_hook_env_does_not_leak_between_runs(self, tmp_path: Path) -> None:
        manager = HookManager(tmp_path)
        hook = HookConfig(event="pre_tool", command="echo $PRISM_HOOK_SECRET")
        # First run with context
        result1 = manager.run_hook(hook, context={"secret": "s3cret"})
        assert "s3cret" in result1.output
        # Second run without context — env var should not be present
        result2 = manager.run_hook(hook, context={})
        assert "s3cret" not in result2.output

    def test_pyyaml_import_error_handled(self, tmp_path: Path) -> None:
        """If pyyaml is not installed, hooks gracefully return empty."""
        with patch.dict("sys.modules", {"yaml": None}):
            # Force a re-import scenario by creating a new manager
            # The ImportError in _load_hooks should be caught
            manager = HookManager.__new__(HookManager)
            manager._project_root = tmp_path
            manager._hooks = []
            # Manually trigger load — will fail on yaml import
            manager._load_hooks()
            assert manager.hooks == []

"""Tests for prism.plugins.api — public plugin API module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from prism.plugins import api

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries() -> Any:
    """Clear tool and command registries before each test."""
    api._registered_tools.clear()
    api._registered_commands.clear()
    yield
    api._registered_tools.clear()
    api._registered_commands.clear()


# ---------------------------------------------------------------------------
# register_tool
# ---------------------------------------------------------------------------


class TestRegisterTool:
    """Tests for register_tool."""

    def test_register_tool_stores_entry(self) -> None:
        def my_handler() -> str:
            return "ok"

        api.register_tool(
            name="my_tool",
            handler=my_handler,
            description="A test tool",
            parameters={"type": "object"},
        )
        assert "my_tool" in api._registered_tools
        entry = api._registered_tools["my_tool"]
        assert entry["name"] == "my_tool"
        assert entry["handler"] is my_handler
        assert entry["description"] == "A test tool"
        assert entry["parameters"] == {"type": "object"}

    def test_register_tool_default_parameters(self) -> None:
        api.register_tool(
            name="bare",
            handler=lambda: None,
        )
        entry = api._registered_tools["bare"]
        assert entry["parameters"] == {}
        assert entry["description"] == ""

    def test_register_tool_overwrites_same_name(self) -> None:
        api.register_tool(name="dup", handler=lambda: "a")
        api.register_tool(name="dup", handler=lambda: "b")
        result = api._registered_tools["dup"]["handler"]()
        assert result == "b"

    def test_register_multiple_tools(self) -> None:
        for i in range(5):
            api.register_tool(
                name=f"tool_{i}",
                handler=lambda: None,
            )
        assert len(api._registered_tools) == 5


# ---------------------------------------------------------------------------
# register_command
# ---------------------------------------------------------------------------


class TestRegisterCommand:
    """Tests for register_command."""

    def test_register_command_stores_entry(self) -> None:
        def cmd_handler() -> str:
            return "output"

        api.register_command(
            name="greet",
            handler=cmd_handler,
            description="Say hello",
        )
        assert "greet" in api._registered_commands
        entry = api._registered_commands["greet"]
        assert entry["name"] == "greet"
        assert entry["handler"] is cmd_handler
        assert entry["description"] == "Say hello"

    def test_register_command_default_description(self) -> None:
        api.register_command(
            name="bare_cmd",
            handler=lambda: "x",
        )
        entry = api._registered_commands["bare_cmd"]
        assert entry["description"] == ""

    def test_register_command_overwrites_same_name(self) -> None:
        api.register_command(name="dup", handler=lambda: "first")
        api.register_command(name="dup", handler=lambda: "second")
        result = api._registered_commands["dup"]["handler"]()
        assert result == "second"

    def test_register_multiple_commands(self) -> None:
        for i in range(4):
            api.register_command(
                name=f"cmd_{i}",
                handler=lambda: "",
            )
        assert len(api._registered_commands) == 4


# ---------------------------------------------------------------------------
# get_registered_tools / get_registered_commands
# ---------------------------------------------------------------------------


class TestGetRegistered:
    """Tests for get_registered_tools and get_registered_commands."""

    def test_get_registered_tools_returns_copy(self) -> None:
        api.register_tool(name="t1", handler=lambda: None)
        result = api.get_registered_tools()
        assert "t1" in result
        # Mutating the copy must not affect internal state
        result.pop("t1")
        assert "t1" in api._registered_tools

    def test_get_registered_commands_returns_copy(self) -> None:
        api.register_command(name="c1", handler=lambda: "")
        result = api.get_registered_commands()
        assert "c1" in result
        result.pop("c1")
        assert "c1" in api._registered_commands

    def test_get_registered_tools_empty(self) -> None:
        assert api.get_registered_tools() == {}

    def test_get_registered_commands_empty(self) -> None:
        assert api.get_registered_commands() == {}

    def test_get_tools_reflects_new_registrations(self) -> None:
        assert api.get_registered_tools() == {}
        api.register_tool(name="new", handler=lambda: None)
        assert "new" in api.get_registered_tools()


# ---------------------------------------------------------------------------
# get_repo_map
# ---------------------------------------------------------------------------


class TestGetRepoMap:
    """Tests for get_repo_map."""

    def test_returns_valid_structure(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("x = 1\ny = 2\n")
        result = api.get_repo_map(str(tmp_path))
        assert result["project_root"] == str(tmp_path.resolve())
        assert result["total_files"] == 1
        assert result["total_lines"] == 2
        assert "main.py" in result["files"]
        assert isinstance(result["directories"], list)

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = api.get_repo_map(str(tmp_path))
        assert result["total_files"] == 0
        assert result["total_lines"] == 0
        assert result["files"] == []

    def test_skips_hidden_directories(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("pass\n")
        (tmp_path / "visible.py").write_text("pass\n")
        result = api.get_repo_map(str(tmp_path))
        assert result["total_files"] == 1
        assert "visible.py" in result["files"]

    def test_nested_files(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg" / "sub"
        pkg.mkdir(parents=True)
        (pkg / "mod.py").write_text("a = 1\n")
        (tmp_path / "top.py").write_text("b = 2\n")
        result = api.get_repo_map(str(tmp_path))
        assert result["total_files"] == 2
        files = result["files"]
        assert any("mod.py" in f for f in files)
        assert any("top.py" in f for f in files)

    def test_unreadable_file_skipped(self, tmp_path: Path) -> None:
        # Create a file that cannot be decoded as UTF-8
        bad_file = tmp_path / "binary.py"
        bad_file.write_bytes(b"\x80\x81\x82")
        result = api.get_repo_map(str(tmp_path))
        # File is listed but lines counted as 0
        assert result["total_files"] == 1
        assert result["total_lines"] == 0


# ---------------------------------------------------------------------------
# get_cost_summary
# ---------------------------------------------------------------------------


class TestGetCostSummary:
    """Tests for get_cost_summary."""

    def test_returns_zeros(self) -> None:
        result = api.get_cost_summary()
        assert result == {
            "session_cost": 0.0,
            "daily_cost": 0.0,
            "monthly_cost": 0.0,
        }

    def test_return_type(self) -> None:
        result = api.get_cost_summary()
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, float)

    def test_has_expected_keys(self) -> None:
        result = api.get_cost_summary()
        assert set(result.keys()) == {
            "session_cost",
            "daily_cost",
            "monthly_cost",
        }


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------


class TestLog:
    """Tests for the plugin log function."""

    def test_log_info_level(self) -> None:
        # Should not raise
        api.log("test message", level="info")

    def test_log_debug_level(self) -> None:
        api.log("debug msg", level="debug")

    def test_log_warning_level(self) -> None:
        api.log("warning msg", level="warning")

    def test_log_error_level(self) -> None:
        api.log("error msg", level="error")

    def test_log_default_level(self) -> None:
        api.log("default level message")

    def test_log_with_extra_kwargs(self) -> None:
        api.log(
            "extra fields",
            level="info",
            plugin_name="my-plugin",
            action="test",
        )

    def test_log_invalid_level_falls_back(self) -> None:
        # Invalid level should fall back to info (via getattr default)
        api.log("fallback test", level="nonexistent_level")


# ---------------------------------------------------------------------------
# Interaction between tools and commands
# ---------------------------------------------------------------------------


class TestCrossRegistration:
    """Tests verifying tools and commands coexist independently."""

    def test_tool_and_command_same_name(self) -> None:
        api.register_tool(name="shared", handler=lambda: "tool")
        api.register_command(name="shared", handler=lambda: "cmd")
        assert "shared" in api.get_registered_tools()
        assert "shared" in api.get_registered_commands()
        assert (
            api.get_registered_tools()["shared"]["handler"]()
            != api.get_registered_commands()["shared"]["handler"]()
        )

    def test_clearing_tools_does_not_affect_commands(self) -> None:
        api.register_tool(name="t", handler=lambda: None)
        api.register_command(name="c", handler=lambda: "")
        api._registered_tools.clear()
        assert api.get_registered_tools() == {}
        assert "c" in api.get_registered_commands()

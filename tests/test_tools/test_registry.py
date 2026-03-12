"""Tests for the tool registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.exceptions import ToolNotFoundError
from prism.tools.base import PermissionLevel, Tool, ToolResult
from prism.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from prism.security.path_guard import PathGuard
    from prism.security.sandbox import CommandSandbox


class _DummyTool(Tool):
    """A minimal tool for registry tests."""

    def __init__(self, tool_name: str = "dummy") -> None:
        self._name = tool_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A dummy tool for testing."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string", "description": "A test value"},
            },
            "required": ["value"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.AUTO

    def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(success=True, output=f"got {arguments.get('value')}")


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get_tool(self) -> None:
        """Can register a tool and retrieve it by name."""
        reg = ToolRegistry()
        tool = _DummyTool("test_tool")
        reg.register(tool)
        assert reg.get_tool("test_tool") is tool

    def test_get_unknown_tool_raises(self) -> None:
        """ToolNotFoundError for unknown tool names."""
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            reg.get_tool("nonexistent")

    def test_duplicate_registration_raises(self) -> None:
        """Registering two tools with the same name raises ValueError."""
        reg = ToolRegistry()
        reg.register(_DummyTool("dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_DummyTool("dup"))

    def test_list_tools(self) -> None:
        """list_tools returns all registered tools sorted by name."""
        reg = ToolRegistry()
        reg.register(_DummyTool("beta"))
        reg.register(_DummyTool("alpha"))
        tools = reg.list_tools()
        assert len(tools) == 2
        assert tools[0].name == "alpha"
        assert tools[1].name == "beta"

    def test_get_schema(self) -> None:
        """get_schema returns the JSON schema for a tool."""
        reg = ToolRegistry()
        reg.register(_DummyTool("schema_test"))
        schema = reg.get_schema("schema_test")
        assert schema["name"] == "schema_test"
        assert "parameters" in schema
        assert "description" in schema

    def test_get_schema_unknown_raises(self) -> None:
        """get_schema raises ToolNotFoundError for unknown tool."""
        reg = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            reg.get_schema("nonexistent")

    def test_all_schemas(self) -> None:
        """all_schemas returns schemas for every registered tool."""
        reg = ToolRegistry()
        reg.register(_DummyTool("one"))
        reg.register(_DummyTool("two"))
        schemas = reg.all_schemas()
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"one", "two"}

    def test_tool_names(self) -> None:
        """tool_names returns sorted list of names."""
        reg = ToolRegistry()
        reg.register(_DummyTool("zeta"))
        reg.register(_DummyTool("alpha"))
        assert reg.tool_names == ["alpha", "zeta"]

    def test_contains(self) -> None:
        """Registry supports 'in' operator."""
        reg = ToolRegistry()
        reg.register(_DummyTool("present"))
        assert "present" in reg
        assert "absent" not in reg

    def test_len(self) -> None:
        """Registry supports len()."""
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(_DummyTool("one"))
        assert len(reg) == 1

    def test_create_default(
        self, path_guard: PathGuard, sandbox: CommandSandbox
    ) -> None:
        """create_default registers all built-in tools."""
        reg = ToolRegistry.create_default(path_guard, sandbox)

        expected = {
            "read_file",
            "write_file",
            "edit_file",
            "list_directory",
            "search_codebase",
            "execute_command",
            "git",
            "analyze_image",
            "auto_test",
            "quality_gate",
        }
        assert set(reg.tool_names) == expected

    def test_create_default_tools_are_functional(
        self, path_guard: PathGuard, sandbox: CommandSandbox, project_dir: Path
    ) -> None:
        """Tools created by create_default actually work."""
        reg = ToolRegistry.create_default(path_guard, sandbox)
        read_tool = reg.get_tool("read_file")
        result = read_tool.execute({"path": "README.md"})
        assert result.success is True
        assert "Test Project" in result.output

"""Tool registry — registers all built-in tools and provides lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from prism.exceptions import ToolNotFoundError

if TYPE_CHECKING:
    from prism.security.path_guard import PathGuard
    from prism.security.sandbox import CommandSandbox
    from prism.tools.base import Tool

logger = structlog.get_logger(__name__)


class ToolRegistry:
    """Central registry for all available tools.

    Provides methods to register, look up, and list tools.  Includes a
    factory method :meth:`create_default` that builds a registry with all
    built-in tools wired up to the given security objects.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register a *tool* instance.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug("tool_registered", name=tool.name)

    def get_tool(self, name: str) -> Tool:
        """Return the tool with the given *name*.

        Raises:
            ToolNotFoundError: If no tool with *name* is registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def list_tools(self) -> list[Tool]:
        """Return all registered tools sorted by name."""
        return sorted(self._tools.values(), key=lambda t: t.name)

    def get_schema(self, name: str) -> dict[str, Any]:
        """Return the JSON schema for the tool with the given *name*.

        Raises:
            ToolNotFoundError: If no tool with *name* is registered.
        """
        tool = self.get_tool(name)
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        }

    def all_schemas(self) -> list[dict[str, Any]]:
        """Return JSON schemas for all registered tools."""
        return [self.get_schema(t.name) for t in self.list_tools()]

    @property
    def tool_names(self) -> list[str]:
        """Return sorted list of all registered tool names."""
        return sorted(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create_default(
        cls,
        path_guard: PathGuard,
        sandbox: CommandSandbox,
        excluded_patterns: list[str] | None = None,
        *,
        web_enabled: bool = False,
    ) -> ToolRegistry:
        """Build a registry pre-loaded with all built-in tools.

        Args:
            path_guard:        A :class:`PathGuard` for file-system tools.
            sandbox:           A :class:`CommandSandbox` for the terminal tool.
            excluded_patterns: Optional extra exclusion patterns for directory
                               and search tools.
            web_enabled:       If ``True``, register the ``browse_web`` and
                               ``screenshot`` tools.

        Returns:
            A fully populated :class:`ToolRegistry`.
        """
        from prism.tools.directory import ListDirectoryTool
        from prism.tools.file_edit import EditFileTool
        from prism.tools.file_read import ReadFileTool
        from prism.tools.file_write import WriteFileTool
        from prism.tools.search import SearchCodebaseTool
        from prism.tools.terminal import ExecuteCommandTool

        registry = cls()

        registry.register(ReadFileTool(path_guard))
        registry.register(WriteFileTool(path_guard))
        registry.register(EditFileTool(path_guard))
        registry.register(ListDirectoryTool(path_guard, excluded_patterns=excluded_patterns))
        registry.register(SearchCodebaseTool(path_guard))
        registry.register(ExecuteCommandTool(sandbox))

        if web_enabled:
            from prism.tools.browser import BrowseWebTool
            from prism.tools.screenshot import ScreenshotTool

            registry.register(BrowseWebTool())
            registry.register(ScreenshotTool())

        return registry

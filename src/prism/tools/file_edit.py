"""Edit file tool — search/replace editing."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

if TYPE_CHECKING:
    from prism.security.path_guard import PathGuard

logger = structlog.get_logger(__name__)


class EditFileTool(Tool):
    """Edit a file using exact search and replace.

    The search string must appear exactly once in the file (unless
    ``replace_all`` is set).  Returns a unified diff of the change.

    Security:
    - Path validated by :class:`PathGuard`.
    - Permission level is CONFIRM.
    """

    def __init__(self, path_guard: PathGuard) -> None:
        self._path_guard = path_guard

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Edit a file using search and replace. The search string must match exactly."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to project root",
                },
                "search": {
                    "type": "string",
                    "description": (
                        "The exact text to find in the file "
                        "(must match exactly including whitespace)"
                    ),
                },
                "replace": {
                    "type": "string",
                    "description": "The text to replace it with",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default false)",
                    "default": False,
                },
            },
            "required": ["path", "search", "replace"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Perform search/replace on a file and return a diff."""
        validated = self.validate_arguments(arguments)
        file_path_str: str = validated["path"]
        search: str = validated["search"]
        replace: str = validated["replace"]
        replace_all: bool = validated.get("replace_all", False)

        # Validate the path through PathGuard
        try:
            resolved = self._path_guard.validate(file_path_str)
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

        # File must exist
        if not resolved.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {file_path_str}",
            )

        if not resolved.is_file():
            return ToolResult(
                success=False,
                output="",
                error=f"Not a file: {file_path_str}",
            )

        # Read current content
        try:
            original = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                original = resolved.read_text(encoding="latin-1")
            except OSError as exc:
                return ToolResult(success=False, output="", error=str(exc))
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        # Check how many times the search string appears
        count = original.count(search)

        if count == 0:
            return ToolResult(
                success=False,
                output="",
                error=f"Search string not found in {file_path_str}",
                metadata={"occurrences": 0},
            )

        if count > 1 and not replace_all:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Search string found {count} times in {file_path_str}. "
                    f"Use replace_all=true to replace all occurrences, "
                    f"or provide a more specific search string."
                ),
                metadata={"occurrences": count},
            )

        # Perform the replacement
        if replace_all:
            new_content = original.replace(search, replace)
            replacements = count
        else:
            new_content = original.replace(search, replace, 1)
            replacements = 1

        # Generate diff
        original_lines = original.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{file_path_str}",
            tofile=f"b/{file_path_str}",
        )
        diff_text = "".join(diff)

        # Write the new content
        try:
            resolved.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write file: {exc}",
            )

        logger.info(
            "file_edited",
            path=file_path_str,
            replacements=replacements,
        )

        return ToolResult(
            success=True,
            output=diff_text or "(no visible diff — replacement was identical)",
            metadata={
                "replacements": replacements,
                "path": str(resolved),
            },
        )

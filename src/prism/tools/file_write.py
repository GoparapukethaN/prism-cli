"""Write file tool — creates or overwrites files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

if TYPE_CHECKING:
    from prism.security.path_guard import PathGuard

logger = structlog.get_logger(__name__)


class WriteFileTool(Tool):
    """Create a new file or overwrite an existing file.

    Security:
    - Path validated by :class:`PathGuard`.
    - Permission level is CONFIRM (requires user approval).
    - Parent directories are created automatically when needed.
    """

    def __init__(self, path_guard: PathGuard) -> None:
        self._path_guard = path_guard

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Create a new file or overwrite an existing file."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to project root",
                },
                "content": {
                    "type": "string",
                    "description": "The complete file content to write",
                },
            },
            "required": ["path", "content"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Write content to a file, creating parent directories if needed."""
        validated = self.validate_arguments(arguments)
        file_path_str: str = validated["path"]
        content: str = validated["content"]

        # Validate the path through PathGuard
        try:
            resolved = self._path_guard.validate(file_path_str)
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

        # Determine if this is a new file or an overwrite
        is_new = not resolved.exists()

        # Create parent directories if they don't exist
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to create parent directories: {exc}",
            )

        # Write the file
        try:
            encoded = content.encode("utf-8")
            resolved.write_bytes(encoded)
            bytes_written = len(encoded)
        except OSError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to write file: {exc}",
            )

        action = "Created" if is_new else "Overwritten"
        output = f"{action} {file_path_str} ({bytes_written} bytes)"

        logger.info(
            "file_written",
            path=file_path_str,
            bytes_written=bytes_written,
            new_file=is_new,
        )

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "bytes_written": bytes_written,
                "new_file": is_new,
                "path": str(resolved),
            },
        )

"""Write file tool — creates or overwrites files."""

from __future__ import annotations

import difflib
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

    # ------------------------------------------------------------------
    # Diff preview (read-only)
    # ------------------------------------------------------------------

    def generate_preview_diff(
        self, arguments: dict[str, Any],
    ) -> tuple[str, bool]:
        """Generate a unified diff without writing anything.

        For new files every line is shown as an addition.  For existing
        files a standard unified diff with three lines of context is
        produced.

        Args:
            arguments: The same dict that would be passed to
                :meth:`execute` (must contain ``path`` and ``content``).

        Returns:
            A ``(diff_text, is_new_file)`` tuple.  *diff_text* may be
            empty when the new content is identical to the existing file.

        Raises:
            Exception: If path validation fails (propagated from
                :class:`PathGuard`).
        """
        validated = self.validate_arguments(arguments)
        file_path_str: str = validated["path"]
        content: str = validated["content"]

        resolved = self._path_guard.validate(file_path_str)
        is_new = not resolved.exists()

        if is_new:
            # Show all lines as additions
            new_lines = content.splitlines(keepends=True)
            # Ensure the last line ends with a newline for clean diff
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            diff = difflib.unified_diff(
                [],
                new_lines,
                fromfile="/dev/null",
                tofile=f"b/{file_path_str}",
            )
            return "".join(diff), True

        # Existing file — produce a proper unified diff
        try:
            original = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            original = resolved.read_text(encoding="latin-1")

        original_lines = original.splitlines(keepends=True)
        new_lines = content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{file_path_str}",
            tofile=f"b/{file_path_str}",
            n=3,
        )
        return "".join(diff), False

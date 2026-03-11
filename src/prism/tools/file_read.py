"""Read file tool — reads file contents with line numbers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from prism.config.defaults import MAX_FILE_READ_BYTES
from prism.tools.base import PermissionLevel, Tool, ToolResult

if TYPE_CHECKING:
    from prism.security.path_guard import PathGuard

logger = structlog.get_logger(__name__)

# Number of bytes to check for null bytes (binary detection).
_BINARY_CHECK_BYTES = 8192


class ReadFileTool(Tool):
    """Read a file's contents and return them with line numbers.

    Security:
    - Path validated by :class:`PathGuard` (no traversal, no excluded files).
    - Binary files are detected and rejected with a descriptive message.
    - Files larger than ``MAX_FILE_READ_BYTES`` are truncated with a warning.
    """

    def __init__(self, path_guard: PathGuard) -> None:
        self._path_guard = path_guard

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Returns the full file content with line numbers."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file, relative to project root",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional: start reading from this line number (1-indexed)",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional: stop reading at this line number (inclusive)",
                },
            },
            "required": ["path"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.AUTO

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Read file contents and return them with line numbers."""
        validated = self.validate_arguments(arguments)
        file_path_str: str = validated["path"]
        start_line: int | None = validated.get("start_line")
        end_line: int | None = validated.get("end_line")

        # Validate the path through PathGuard
        try:
            resolved = self._path_guard.validate(file_path_str)
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

        # Check file exists
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

        # Check file size
        try:
            file_size = resolved.stat().st_size
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        truncated = False
        if file_size > MAX_FILE_READ_BYTES:
            truncated = True

        # Binary detection: read the first 8KB and look for null bytes
        try:
            with resolved.open("rb") as fh:
                head = fh.read(_BINARY_CHECK_BYTES)
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        if b"\x00" in head:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Binary file detected: {file_path_str} "
                    f"({file_size} bytes). Cannot display binary content."
                ),
                metadata={"binary": True, "size": file_size},
            )

        # Read the file content
        try:
            raw = resolved.read_bytes()
            if truncated:
                raw = raw[:MAX_FILE_READ_BYTES]
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        # Decode
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")

        # Split into lines and apply offset/limit
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        start_idx = max(0, start_line - 1) if start_line is not None else 0

        end_idx = min(total_lines, end_line) if end_line is not None else total_lines

        selected = lines[start_idx:end_idx]

        # Format with line numbers
        numbered: list[str] = []
        for i, line in enumerate(selected, start=start_idx + 1):
            # Strip trailing newline for display, then re-add
            numbered.append(f"{i}: {line.rstrip(chr(10)).rstrip(chr(13))}")

        content = "\n".join(numbered)

        if truncated:
            content += (
                f"\n\n... [truncated — file is {file_size} bytes, "
                f"showing first {MAX_FILE_READ_BYTES} bytes]"
            )

        metadata: dict[str, Any] = {
            "size": file_size,
            "lines": total_lines,
            "start_line": start_idx + 1,
            "end_line": end_idx,
        }
        if truncated:
            metadata["truncated"] = True

        return ToolResult(success=True, output=content, metadata=metadata)

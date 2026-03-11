"""List directory tool — lists files with metadata."""

from __future__ import annotations

import fnmatch
import time
from typing import TYPE_CHECKING, Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

if TYPE_CHECKING:
    from pathlib import Path

    from prism.security.path_guard import PathGuard

logger = structlog.get_logger(__name__)

# Maximum number of entries to return before truncating.
_MAX_ENTRIES = 500

# Patterns that are always excluded from listing.
_DEFAULT_EXCLUDED: list[str] = [
    ".git",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "*.pyc",
    "*.pyo",
]


def _format_size(size: int) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024  # type: ignore[assignment]
    return f"{size:.1f}TB"


def _format_time(ts: float) -> str:
    """Format a Unix timestamp as a short date-time string."""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


class ListDirectoryTool(Tool):
    """List directory contents with file metadata.

    Supports optional glob pattern filtering and respects excluded patterns.

    Security:
    - Path validated by :class:`PathGuard`.
    - Permission level is AUTO (read-only operation).
    """

    def __init__(
        self,
        path_guard: PathGuard,
        excluded_patterns: list[str] | None = None,
    ) -> None:
        self._path_guard = path_guard
        self._excluded: list[str] = list(excluded_patterns or _DEFAULT_EXCLUDED)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List the contents of a directory. Shows files and subdirectories with sizes."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the directory, relative to project root. "
                        "Use '.' for project root."
                    ),
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter entries (e.g., '*.py')",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, list recursively. Default false.",
                    "default": False,
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth for recursive listing. Default 3.",
                    "default": 3,
                },
            },
            "required": ["path"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.AUTO

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """List directory contents."""
        validated = self.validate_arguments(arguments)
        dir_path_str: str = validated["path"]
        pattern: str | None = validated.get("pattern")
        recursive: bool = validated.get("recursive", False)
        max_depth: int = validated.get("max_depth", 3)

        # Validate the path through PathGuard
        try:
            resolved = self._path_guard.validate(dir_path_str)
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

        if not resolved.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Directory not found: {dir_path_str}",
            )

        if not resolved.is_dir():
            return ToolResult(
                success=False,
                output="",
                error=f"Not a directory: {dir_path_str}",
            )

        # Collect entries
        entries: list[str] = []
        total_count = 0

        if recursive:
            self._list_recursive(
                resolved, resolved, pattern, max_depth, 0, entries
            )
            total_count = len(entries)
        else:
            try:
                children = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except PermissionError as exc:
                return ToolResult(success=False, output="", error=str(exc))

            for child in children:
                if self._is_excluded(child.name):
                    continue
                if pattern and not fnmatch.fnmatch(child.name, pattern):
                    continue

                total_count += 1
                if len(entries) < _MAX_ENTRIES:
                    entries.append(self._format_entry(child))

        # Build output
        output_lines = entries[:_MAX_ENTRIES]
        truncated = total_count > _MAX_ENTRIES
        output = "\n".join(output_lines)

        if truncated:
            remaining = total_count - _MAX_ENTRIES
            output += f"\n\n... [{remaining} more entries not shown]"

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "entries": min(total_count, _MAX_ENTRIES),
                "total": total_count,
                "truncated": truncated,
                "path": str(resolved),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_excluded(self, name: str) -> bool:
        """Check if *name* matches any excluded pattern."""
        return any(fnmatch.fnmatch(name, pat) for pat in self._excluded)

    def _format_entry(self, path: Path, prefix: str = "") -> str:
        """Format a single directory entry."""
        try:
            stat = path.stat()
            kind = "dir" if path.is_dir() else "file"
            size_str = _format_size(stat.st_size) if kind == "file" else "-"
            mtime = _format_time(stat.st_mtime)
            return f"{prefix}{kind:4s}  {size_str:>8s}  {mtime}  {path.name}"
        except OSError:
            return f"{prefix}????  {'?':>8s}  {'?':16s}  {path.name}"

    def _list_recursive(
        self,
        base: Path,
        current: Path,
        pattern: str | None,
        max_depth: int,
        depth: int,
        entries: list[str],
    ) -> None:
        """Recursively list a directory tree up to *max_depth*."""
        if depth > max_depth:
            return
        if len(entries) >= _MAX_ENTRIES:
            return

        try:
            children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError:
            return

        indent = "  " * depth
        for child in children:
            if self._is_excluded(child.name):
                continue
            if pattern and not child.is_dir() and not fnmatch.fnmatch(child.name, pattern):
                continue
            if len(entries) >= _MAX_ENTRIES:
                return

            entries.append(self._format_entry(child, prefix=indent))

            if child.is_dir():
                self._list_recursive(
                    base, child, pattern, max_depth, depth + 1, entries
                )

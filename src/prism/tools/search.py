"""Search codebase tool — regex search across project files."""

from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING, Any

import structlog

from prism.config.defaults import MAX_SEARCH_RESULTS
from prism.tools.base import PermissionLevel, Tool, ToolResult

if TYPE_CHECKING:
    from pathlib import Path

    from prism.security.path_guard import PathGuard

logger = structlog.get_logger(__name__)

# Binary check: read this many bytes to detect binary files.
_BINARY_CHECK_BYTES = 8192

# Default context lines before and after a match.
_CONTEXT_LINES = 2

# Patterns that are always excluded from search.
_DEFAULT_EXCLUDED_DIRS: list[str] = [
    ".git",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    ".eggs",
]

_DEFAULT_EXCLUDED_FILES: list[str] = [
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.o",
    "*.a",
    "*.jar",
    "*.class",
    "*.whl",
    "*.tar.gz",
    "*.zip",
]


class SearchCodebaseTool(Tool):
    """Full-text regex search across the project codebase.

    Features:
    - Regex pattern matching.
    - Optional glob pattern to filter files by extension.
    - Context lines around each match.
    - Respects excluded patterns (binary files, build dirs).
    - Max result count to prevent overwhelming output.

    Security:
    - Path validated by :class:`PathGuard`.
    - Permission level is AUTO (read-only operation).
    """

    def __init__(
        self,
        path_guard: PathGuard,
        excluded_dirs: list[str] | None = None,
        excluded_files: list[str] | None = None,
    ) -> None:
        self._path_guard = path_guard
        self._excluded_dirs = list(excluded_dirs or _DEFAULT_EXCLUDED_DIRS)
        self._excluded_files = list(excluded_files or _DEFAULT_EXCLUDED_FILES)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "search_codebase"

    @property
    def description(self) -> str:
        return "Search for text patterns across the entire codebase using regex."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for",
                },
                "file_pattern": {
                    "type": "string",
                    "description": (
                        "Optional glob pattern to filter files "
                        "(e.g., '*.py', '*.ts')"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results. Default 50.",
                    "default": MAX_SEARCH_RESULTS,
                },
            },
            "required": ["pattern"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.AUTO

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Search the codebase for a regex pattern."""
        validated = self.validate_arguments(arguments)
        pattern_str: str = validated["pattern"]
        file_pattern: str | None = validated.get("file_pattern")
        max_results: int = validated.get("max_results", MAX_SEARCH_RESULTS)

        # Compile the regex
        try:
            regex = re.compile(pattern_str)
        except re.error as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid regex pattern: {exc}",
            )

        project_root = self._path_guard.project_root
        results: list[str] = []
        files_searched = 0
        total_matches = 0

        for file_path in self._walk_files(project_root, file_pattern):
            # Skip binary files
            if self._is_binary(file_path):
                continue

            files_searched += 1

            # Read and search
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    text = file_path.read_text(encoding="latin-1")
                except OSError:
                    continue
            except OSError:
                continue

            lines = text.splitlines()
            for line_num, line in enumerate(lines, start=1):
                if regex.search(line):
                    total_matches += 1
                    if len(results) < max_results:
                        rel = file_path.relative_to(project_root)
                        # Gather context
                        context_before = lines[
                            max(0, line_num - 1 - _CONTEXT_LINES) : line_num - 1
                        ]
                        context_after = lines[
                            line_num : line_num + _CONTEXT_LINES
                        ]

                        entry_lines: list[str] = [f"{rel}:{line_num}: {line.rstrip()}"]
                        if context_before:
                            for i, ctx in enumerate(
                                context_before,
                                start=max(1, line_num - _CONTEXT_LINES),
                            ):
                                entry_lines.insert(
                                    len(entry_lines) - 1,
                                    f"  {i}: {ctx.rstrip()}",
                                )
                        if context_after:
                            for i, ctx in enumerate(
                                context_after, start=line_num + 1
                            ):
                                entry_lines.append(f"  {i}: {ctx.rstrip()}")

                        results.append("\n".join(entry_lines))

            if total_matches >= max_results:
                break

        if not results:
            return ToolResult(
                success=True,
                output=f"No matches found for pattern: {pattern_str}",
                metadata={
                    "matches": 0,
                    "files_searched": files_searched,
                },
            )

        output = "\n\n".join(results)
        if total_matches > max_results:
            output += (
                f"\n\n... [{total_matches - max_results} more matches not shown]"
            )

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "matches": total_matches,
                "files_searched": files_searched,
                "truncated": total_matches > max_results,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _walk_files(
        self, root: Path, file_pattern: str | None
    ) -> list[Path]:
        """Walk the project tree, respecting exclusions."""
        collected: list[Path] = []

        def _recurse(directory: Path) -> None:
            try:
                children = sorted(directory.iterdir())
            except PermissionError:
                return

            for child in children:
                if child.is_dir():
                    if child.name in self._excluded_dirs:
                        continue
                    _recurse(child)
                elif child.is_file():
                    if any(
                        fnmatch.fnmatch(child.name, pat)
                        for pat in self._excluded_files
                    ):
                        continue
                    if file_pattern and not fnmatch.fnmatch(child.name, file_pattern):
                        continue
                    collected.append(child)

        _recurse(root)
        return collected

    @staticmethod
    def _is_binary(path: Path) -> bool:
        """Return True if *path* appears to be a binary file."""
        try:
            with path.open("rb") as fh:
                chunk = fh.read(_BINARY_CHECK_BYTES)
            return b"\x00" in chunk
        except OSError:
            return True

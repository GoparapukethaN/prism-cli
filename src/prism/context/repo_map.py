"""Repository map generator — compressed view of the codebase structure.

Generates a map of classes, functions, and their signatures from Python
files using simple regex-based parsing.  The map is cached and invalidated
when file modification times change.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for Python source parsing
# ---------------------------------------------------------------------------

_CLASS_RE = re.compile(
    r"^class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:", re.MULTILINE
)
_FUNCTION_RE = re.compile(
    r"^([ \t]*)def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^\n:]+))?\s*:", re.MULTILINE
)

# Default patterns that should be excluded (similar to .gitignore defaults)
_DEFAULT_IGNORE_PATTERNS: list[str] = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "*.egg-info",
    ".eggs",
]


# ---------------------------------------------------------------------------
# .gitignore parsing
# ---------------------------------------------------------------------------


def _load_gitignore_patterns(project_root: Path) -> list[str]:
    """Load patterns from the project's ``.gitignore`` file.

    Returns the default ignore patterns if no ``.gitignore`` exists.
    """
    patterns = list(_DEFAULT_IGNORE_PATTERNS)
    gitignore = project_root / ".gitignore"
    if gitignore.is_file():
        try:
            for line in gitignore.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    patterns.append(stripped)
        except OSError:
            pass
    return patterns


def _should_ignore(path: Path, project_root: Path, patterns: list[str]) -> bool:
    """Return *True* if *path* matches any of the ignore *patterns*."""
    try:
        rel = path.relative_to(project_root)
    except ValueError:
        return True

    rel_str = str(rel)
    name = path.name

    for pattern in patterns:
        # Strip trailing slashes (directory markers)
        clean = pattern.rstrip("/")

        # Match against name (for simple patterns like *.pyc) and
        # against relative path (for path-like patterns)
        if fnmatch.fnmatch(name, clean):
            return True
        if fnmatch.fnmatch(rel_str, clean):
            return True
        # Also check if any path component matches (e.g. __pycache__)
        for part in rel.parts:
            if fnmatch.fnmatch(part, clean):
                return True

    return False


# ---------------------------------------------------------------------------
# Python file parsing
# ---------------------------------------------------------------------------


def _parse_python_file(path: Path) -> list[str]:
    """Extract class/function/method signatures from a Python file.

    Returns a list of formatted signature lines.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines: list[str] = []
    current_class: str | None = None

    # Build a list of (line_number, type, info) entries so we can
    # interleave classes and functions in source order.
    entries: list[tuple[int, str, dict[str, Any]]] = []

    for m in _CLASS_RE.finditer(source):
        entries.append((
            m.start(),
            "class",
            {"name": m.group(1), "bases": m.group(2) or ""},
        ))

    for m in _FUNCTION_RE.finditer(source):
        indent_str = m.group(1)
        indent_level = len(indent_str.replace("\t", "    "))
        return_type = m.group(4).strip() if m.group(4) else None
        params = m.group(3).strip()

        entries.append((
            m.start(),
            "function",
            {
                "name": m.group(2),
                "params": params,
                "return_type": return_type,
                "indent": indent_level,
            },
        ))

    # Sort by position in the file
    entries.sort(key=lambda e: e[0])

    for _, kind, info in entries:
        if kind == "class":
            current_class = info["name"]
            bases_part = f"({info['bases']})" if info["bases"] else ""
            lines.append(f"  class {current_class}{bases_part}:")
        elif kind == "function":
            name = info["name"]
            params = info["params"]
            return_type = info["return_type"]
            indent = info["indent"]

            ret_str = f" -> {return_type}" if return_type else ""

            if indent > 0 and current_class is not None:
                # Method inside a class
                lines.append(f"    def {name}({params}){ret_str}")
            else:
                # Top-level function — reset class context
                current_class = None
                lines.append(f"  def {name}({params}){ret_str}")

    return lines


# ---------------------------------------------------------------------------
# Repo map cache
# ---------------------------------------------------------------------------


class _RepoMapCache:
    """Simple cache keyed on ``{path: mtime}`` snapshot."""

    def __init__(self) -> None:
        self._file_states: dict[str, float] = {}
        self._cached_map: str = ""

    def is_valid(self, current_states: dict[str, float]) -> bool:
        return current_states == self._file_states

    def store(self, file_states: dict[str, float], repo_map: str) -> None:
        self._file_states = dict(file_states)
        self._cached_map = repo_map

    @property
    def cached_map(self) -> str:
        return self._cached_map


# Module-level cache instance (one per process)
_cache = _RepoMapCache()


def invalidate_cache() -> None:
    """Force cache invalidation (useful for tests)."""
    global _cache  # noqa: PLW0603
    _cache = _RepoMapCache()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_repo_map(
    project_root: Path,
    max_tokens: int = 5000,
    *,
    use_cache: bool = True,
) -> str:
    """Generate a compressed repository map.

    Walks *project_root*, parses Python files for class/function signatures,
    and returns a formatted string that fits within *max_tokens*.

    Args:
        project_root: Root directory of the project.
        max_tokens: Maximum estimated tokens for the output.
        use_cache: If *True*, return cached map when files haven't changed.

    Returns:
        A formatted repo map string.
    """
    global _cache  # noqa: PLW0602

    project_root = project_root.resolve()
    ignore_patterns = _load_gitignore_patterns(project_root)

    # Collect Python files and their mtimes
    file_states: dict[str, float] = {}
    python_files: list[Path] = []

    for root_dir, dirs, files in os.walk(project_root):
        root_path = Path(root_dir)

        # Filter directories in-place to skip ignored dirs
        dirs[:] = [
            d for d in dirs
            if not _should_ignore(root_path / d, project_root, ignore_patterns)
        ]

        for fname in sorted(files):
            fpath = root_path / fname
            if _should_ignore(fpath, project_root, ignore_patterns):
                continue
            if fpath.suffix == ".py":
                try:
                    file_states[str(fpath)] = fpath.stat().st_mtime
                except OSError:
                    continue
                python_files.append(fpath)

    # Check cache
    if use_cache and _cache.is_valid(file_states):
        logger.debug("repo_map_cache_hit")
        return _cache.cached_map

    # Sort by modification time (most recently modified first)
    python_files.sort(key=lambda p: file_states.get(str(p), 0), reverse=True)

    # Build the map
    sections: list[str] = []
    from prism.context.manager import estimate_tokens

    tokens_used = 0

    for fpath in python_files:
        try:
            rel = fpath.relative_to(project_root)
        except ValueError:
            continue

        sigs = _parse_python_file(fpath)
        entry = f"{rel}:" if not sigs else f"{rel}:\n" + "\n".join(sigs)

        entry_tokens = estimate_tokens(entry)
        if tokens_used + entry_tokens > max_tokens:
            # If we haven't added anything yet, add a truncated entry
            if not sections:
                sections.append(entry)
            break

        sections.append(entry)
        tokens_used += entry_tokens

    repo_map = "\n\n".join(sections)

    # Cache the result
    _cache.store(file_states, repo_map)
    logger.debug("repo_map_generated", files=len(python_files), tokens=tokens_used)

    return repo_map

"""Prismignore — .gitignore-compatible file exclusion for Prism operations.

Provides a ``PrismIgnore`` class that loads patterns from a ``.prismignore``
file (syntax identical to ``.gitignore``) and exposes helpers to check whether
a given path should be excluded from Prism file reads, context building, and
tool operations.

Default patterns cover environment files, cryptographic keys, cloud
credentials, dependency caches, logs, build artefacts, and IDE directories.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# -------------------------------------------------------------------------
# Default patterns — written into .prismignore on ``prism init``
# -------------------------------------------------------------------------

DEFAULT_PATTERNS: list[str] = [
    "# Environment and secrets",
    ".env",
    ".env.*",
    "*.env",
    "secrets/",
    "credentials/",
    "private/",
    "",
    "# Cryptographic keys",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_ed25519",
    "*.pub",
    "",
    "# Cloud credentials",
    ".aws/",
    ".ssh/",
    ".gcloud/",
    "service-account*.json",
    "credentials.json",
    "",
    "# Dependencies and caches",
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "venv/",
    ".tox/",
    ".nox/",
    "",
    "# Logs",
    "*.log",
    "*.log.*",
    "",
    "# Build artifacts",
    "dist/",
    "build/",
    "*.egg-info/",
    "",
    "# IDE",
    ".idea/",
    ".vscode/",
    "*.swp",
    "*.swo",
]


class PrismIgnore:
    """Manages a ``.prismignore`` file with ``.gitignore``-compatible pattern matching.

    Patterns follow the same semantics as ``.gitignore``:

    * Blank lines and lines starting with ``#`` are ignored.
    * A trailing ``/`` matches directories only (any path component).
    * A pattern without ``/`` is matched against the **basename** of every
      path component.
    * A pattern *with* ``/`` (other than a trailing one) is matched against
      the full relative path.
    * A leading ``!`` negates the pattern (re-includes a previously
      excluded file).

    Attributes:
        file_path: Absolute path to the ``.prismignore`` file.
        patterns: List of active (non-comment, non-blank) patterns.
    """

    def __init__(self, project_root: Path) -> None:
        """Initialise PrismIgnore.

        Args:
            project_root: The root directory of the project. The
                ``.prismignore`` file is expected at ``project_root / ".prismignore"``.
        """
        self._root = project_root.resolve()
        self._file_path = self._root / ".prismignore"
        self._raw_lines: list[str] = []
        self._compiled: list[tuple[str, bool]] = []  # (pattern, is_negated)
        self._load()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def file_path(self) -> Path:
        """Return the absolute path to the ``.prismignore`` file."""
        return self._file_path

    @property
    def patterns(self) -> list[str]:
        """Return all active patterns (excluding comments and blank lines)."""
        return [line for line in self._raw_lines if line and not line.startswith("#")]

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def is_ignored(self, path: str | Path) -> bool:
        """Check whether *path* matches any ignore pattern.

        Last-match-wins semantics are used, so a later negation (``!``)
        pattern can re-include a previously excluded path.

        Args:
            path: Absolute or relative path to check.

        Returns:
            ``True`` if the path should be ignored, ``False`` otherwise.
        """
        rel = self._get_relative(path)
        if rel is None:
            return False

        ignored = False
        for pattern, negated in self._compiled:
            if self._matches(rel, pattern):
                ignored = not negated

        return ignored

    def filter_paths(self, paths: list[str | Path]) -> list[Path]:
        """Return *paths* with ignored entries removed.

        Args:
            paths: A list of paths (absolute or relative) to filter.

        Returns:
            A new list containing only those paths that are **not** ignored.
        """
        result: list[Path] = []
        for p in paths:
            if not self.is_ignored(p):
                result.append(Path(p))
        return result

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_pattern(self, pattern: str) -> None:
        """Append *pattern* to the ignore list and persist to disk.

        If the pattern already exists, the call is a no-op.

        Args:
            pattern: A ``.gitignore``-compatible glob pattern.
        """
        pattern = pattern.strip()
        if not pattern:
            return

        if pattern in self._raw_lines:
            logger.debug("prismignore_pattern_duplicate", pattern=pattern)
            return

        self._raw_lines.append(pattern)
        self._compile()
        self._save()
        logger.info("prismignore_pattern_added", pattern=pattern)

    def remove_pattern(self, pattern: str) -> bool:
        """Remove *pattern* from the ignore list and persist.

        Args:
            pattern: The exact pattern string to remove.

        Returns:
            ``True`` if the pattern was found and removed, ``False``
            otherwise.
        """
        pattern = pattern.strip()
        if pattern in self._raw_lines:
            self._raw_lines.remove(pattern)
            self._compile()
            self._save()
            logger.info("prismignore_pattern_removed", pattern=pattern)
            return True
        return False

    def create_default(self) -> Path:
        """Create (or overwrite) ``.prismignore`` with the default pattern set.

        Returns:
            The path to the created file.
        """
        self._raw_lines = list(DEFAULT_PATTERNS)
        self._compile()
        self._save()
        logger.info("prismignore_default_created", path=str(self._file_path))
        return self._file_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load patterns from ``.prismignore`` on disk, falling back to defaults."""
        if self._file_path.is_file():
            raw = self._file_path.read_text(encoding="utf-8").splitlines()
            self._raw_lines = [line.rstrip() for line in raw]
        else:
            self._raw_lines = list(DEFAULT_PATTERNS)
        self._compile()

    def _compile(self) -> None:
        """Parse raw lines into a list of ``(clean_pattern, is_negated)`` tuples."""
        self._compiled = []
        for line in self._raw_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            negated = stripped.startswith("!")
            if negated:
                stripped = stripped[1:]
            self._compiled.append((stripped, negated))

    def _save(self) -> None:
        """Write current patterns to ``.prismignore``."""
        self._file_path.write_text("\n".join(self._raw_lines) + "\n", encoding="utf-8")

    def _get_relative(self, path: str | Path) -> str | None:
        """Resolve *path* relative to the project root.

        Args:
            path: The path to resolve.

        Returns:
            A POSIX-style relative path string, or ``None`` if the path
            is empty.
        """
        path_str = str(path).strip()
        if not path_str:
            return None

        try:
            p = Path(path_str)
            resolved = p.resolve() if p.is_absolute() else (self._root / p).resolve()
            return str(resolved.relative_to(self._root)).replace("\\", "/")
        except ValueError:
            # Path is not under project root — match against the raw string
            return str(path).replace("\\", "/")

    @staticmethod
    def _matches(rel_path: str, pattern: str) -> bool:
        """Check whether *rel_path* matches a single ``.gitignore``-style pattern.

        Args:
            rel_path: Forward-slash-separated path relative to the project root.
            pattern: A compiled pattern string (already stripped of ``!`` prefix).

        Returns:
            ``True`` if *rel_path* matches *pattern*.
        """
        # Normalise to forward slashes for consistency
        rel_path = rel_path.replace("\\", "/")

        # Directory patterns (trailing /)
        if pattern.endswith("/"):
            dir_pattern = pattern.rstrip("/")
            parts = rel_path.split("/")
            return any(fnmatch.fnmatch(part, dir_pattern) for part in parts)

        # Pattern contains a slash (other than trailing) → match full path
        if "/" in pattern:
            return fnmatch.fnmatch(rel_path, pattern)

        # Pure basename pattern — check each path component *and* basename
        basename = rel_path.rsplit("/", 1)[-1] if "/" in rel_path else rel_path
        if fnmatch.fnmatch(basename, pattern):
            return True

        # Also check each directory component (e.g., pattern "secrets"
        # should match "secrets/foo.txt" via the first component)
        parts = rel_path.split("/")
        return any(fnmatch.fnmatch(part, pattern) for part in parts)

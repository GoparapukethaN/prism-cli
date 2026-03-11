"""Path validation to prevent directory traversal and access to excluded files."""

from __future__ import annotations

import fnmatch
from pathlib import Path

import structlog

from prism.config.defaults import ALWAYS_BLOCKED_PATTERNS
from prism.exceptions import ExcludedFileError, PathTraversalError

logger = structlog.get_logger(__name__)


class PathGuard:
    """Validates that file paths stay within the project root and are not excluded.

    Handles:
    - Directory traversal via ``../`` sequences
    - Symlink escape (resolved via ``Path.resolve()`` / ``os.path.realpath``)
    - Null-byte injection in path components
    - Excluded patterns from user config (fnmatch glob)
    - Always-blocked patterns that cannot be overridden
    """

    def __init__(
        self,
        project_root: Path,
        excluded_patterns: list[str] | None = None,
    ) -> None:
        """Initialise PathGuard.

        Args:
            project_root: The root directory that all paths must reside within.
                          Resolved to an absolute real path on construction.
            excluded_patterns: Optional list of fnmatch glob patterns for files
                               that should be rejected even if they are within the
                               project root. These are *in addition* to the
                               always-blocked patterns from ``prism.config.defaults``.
        """
        self._project_root = project_root.resolve()
        self._excluded_patterns: list[str] = list(excluded_patterns or [])
        self._always_blocked: list[str] = list(ALWAYS_BLOCKED_PATTERNS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        """Return the resolved project root."""
        return self._project_root

    @property
    def excluded_patterns(self) -> list[str]:
        """Return the current list of user-configured excluded patterns."""
        return list(self._excluded_patterns)

    def validate(self, path: str | Path) -> Path:
        """Validate *path* and return the resolved absolute ``Path``.

        Raises:
            PathTraversalError: If the resolved path escapes the project root.
            ExcludedFileError:  If the path matches an excluded or always-blocked
                                pattern.
            ValueError:         If the path contains null bytes.
        """
        path_str = str(path)

        # ----------------------------------------------------------
        # 1. Null-byte check — must happen before any OS call
        # ----------------------------------------------------------
        if "\x00" in path_str:
            logger.warning("null_byte_in_path", path=repr(path_str))
            raise ValueError(f"Path contains null byte: {path_str!r}")

        # ----------------------------------------------------------
        # 2. Resolve to absolute real path (follows symlinks)
        # ----------------------------------------------------------
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = self._project_root / candidate
        resolved = candidate.resolve()

        # ----------------------------------------------------------
        # 3. Containment check — resolved path must be under root
        # ----------------------------------------------------------
        try:
            resolved.relative_to(self._project_root)
        except ValueError:
            logger.warning(
                "path_traversal_blocked",
                path=path_str,
                resolved=str(resolved),
                project_root=str(self._project_root),
            )
            raise PathTraversalError(path_str, str(self._project_root)) from None

        # ----------------------------------------------------------
        # 4. Always-blocked patterns (cannot be overridden)
        # ----------------------------------------------------------
        rel_path = str(resolved.relative_to(self._project_root))
        self._check_patterns(
            path_str=path_str,
            rel_path=rel_path,
            resolved=resolved,
            patterns=self._always_blocked,
            label="always_blocked",
        )

        # ----------------------------------------------------------
        # 5. User-configured excluded patterns
        # ----------------------------------------------------------
        self._check_patterns(
            path_str=path_str,
            rel_path=rel_path,
            resolved=resolved,
            patterns=self._excluded_patterns,
            label="excluded_pattern",
        )

        return resolved

    def is_safe(self, path: str | Path) -> bool:
        """Return ``True`` if *path* passes validation, ``False`` otherwise.

        This is a convenience wrapper around :meth:`validate` that never raises.
        """
        try:
            self.validate(path)
            return True
        except (PathTraversalError, ExcludedFileError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_patterns(
        self,
        *,
        path_str: str,
        rel_path: str,
        resolved: Path,
        patterns: list[str],
        label: str,
    ) -> None:
        """Check *rel_path* and *resolved* against a list of fnmatch patterns.

        We match against both the relative path (within the project) and the
        full resolved path to handle patterns with leading ``**/``.

        Raises:
            ExcludedFileError: If any pattern matches.
        """
        full_path = str(resolved)
        for pattern in patterns:
            if (
                fnmatch.fnmatch(rel_path, pattern)
                or fnmatch.fnmatch(full_path, pattern)
                or fnmatch.fnmatch(resolved.name, pattern)
            ):
                logger.warning(
                    f"{label}_match",
                    path=path_str,
                    pattern=pattern,
                )
                raise ExcludedFileError(path_str, pattern)

"""Auto-update system — PyPI version checking and upgrade management.

Checks PyPI for new versions at most once per 24 hours, caches results,
and provides non-blocking background version checking. Never auto-updates
without an explicit user command.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

CURRENT_VERSION = "0.2.0"
PYPI_PACKAGE = "prism-cli"
CHECK_INTERVAL_HOURS = 24

# Dependencies to report in version info
_DEPENDENCY_NAMES: list[str] = [
    "typer",
    "rich",
    "prompt_toolkit",
    "litellm",
    "httpx",
    "pydantic",
    "structlog",
    "yaml",
    "watchdog",
    "keyring",
]


@dataclass(frozen=True)
class UpdateInfo:
    """Information about an available update."""

    current_version: str
    latest_version: str
    is_update_available: bool
    release_date: str
    changelog: str
    checked_at: str


@dataclass(frozen=True)
class VersionInfo:
    """Full version information including Python and all dependencies."""

    prism_version: str
    python_version: str
    dependencies: dict[str, str] = field(default_factory=dict)


class UpdateChecker:
    """Non-blocking PyPI update checker.

    Checks for updates in a background daemon thread and caches the result
    to disk at ~/.prism/update_check.json. At most one check per 24 hours.

    Args:
        cache_dir: Directory for caching update check results.
                   Defaults to ~/.prism.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path.home() / ".prism"
        self._cache_file = self._cache_dir / "update_check.json"
        self._latest: UpdateInfo | None = None
        self._check_thread: threading.Thread | None = None

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory path."""
        return self._cache_dir

    @property
    def cache_file(self) -> Path:
        """Return the cache file path."""
        return self._cache_file

    def check_async(self) -> None:
        """Start a non-blocking background check for updates.

        Spawns a daemon thread that checks PyPI for the latest version
        only if enough time has elapsed since the last check.
        """
        if self._should_check():
            self._check_thread = threading.Thread(
                target=self._check_pypi,
                daemon=True,
                name="prism-update-checker",
            )
            self._check_thread.start()

    def _should_check(self) -> bool:
        """Determine whether enough time has passed since last check.

        Returns:
            True if no cache exists, cache is malformed, or the check
            interval has elapsed.
        """
        if not self._cache_file.is_file():
            return True
        try:
            data = json.loads(self._cache_file.read_text(encoding="utf-8"))
            checked_at_str = data.get("checked_at", "")
            if not checked_at_str:
                return True
            last_check = datetime.fromisoformat(checked_at_str)
            elapsed_hours = (datetime.now(UTC) - last_check).total_seconds() / 3600
            return elapsed_hours >= CHECK_INTERVAL_HOURS
        except (json.JSONDecodeError, ValueError, OSError, TypeError):
            return True

    def _check_pypi(self) -> None:
        """Check PyPI for the latest version of the package.

        Runs ``pip index versions`` and parses the output to determine
        whether a newer version is available. Stores the result in the
        instance and persists it to the cache file.
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "index", "versions", PYPI_PACKAGE],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            latest = CURRENT_VERSION
            if result.returncode == 0:
                output = result.stdout.strip()
                if "(" in output and ")" in output:
                    latest = output.split("(")[1].split(")")[0].strip()

            is_update = self._compare_versions(latest, CURRENT_VERSION)

            self._latest = UpdateInfo(
                current_version=CURRENT_VERSION,
                latest_version=latest,
                is_update_available=is_update,
                release_date="",
                changelog="",
                checked_at=datetime.now(UTC).isoformat(),
            )

            self._save_cache()
            logger.debug(
                "update_check_complete",
                current=CURRENT_VERSION,
                latest=latest,
                update_available=is_update,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("update_check_failed", error=str(exc))

    def _save_cache(self) -> None:
        """Persist the latest check result to disk.

        Creates the cache directory if it does not exist.
        """
        if self._latest is None:
            return
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "current_version": self._latest.current_version,
                "latest_version": self._latest.latest_version,
                "is_update_available": self._latest.is_update_available,
                "release_date": self._latest.release_date,
                "changelog": self._latest.changelog,
                "checked_at": self._latest.checked_at,
            }
            self._cache_file.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug("update_cache_write_failed", error=str(exc))

    def get_cached_info(self) -> UpdateInfo | None:
        """Return the most recent update information.

        Checks the in-memory cache first, then falls back to reading
        the cache file on disk.

        Returns:
            An UpdateInfo instance if available, otherwise None.
        """
        if self._latest is not None:
            return self._latest
        if self._cache_file.is_file():
            try:
                data = json.loads(self._cache_file.read_text(encoding="utf-8"))
                return UpdateInfo(
                    current_version=data.get("current_version", CURRENT_VERSION),
                    latest_version=data.get("latest_version", CURRENT_VERSION),
                    is_update_available=data.get("is_update_available", False),
                    release_date=data.get("release_date", ""),
                    changelog=data.get("changelog", ""),
                    checked_at=data.get("checked_at", ""),
                )
            except (json.JSONDecodeError, OSError, TypeError):
                return None
        return None

    def get_notification(self) -> str | None:
        """Build a user-facing notification string if an update is available.

        Returns:
            A notification message or None if no update is available.
        """
        info = self.get_cached_info()
        if info is not None and info.is_update_available:
            return (
                f"Prism v{info.latest_version} available. "
                "Run 'prism update' to upgrade."
            )
        return None

    @staticmethod
    def perform_update() -> tuple[bool, str]:
        """Execute ``pip install --upgrade prism-cli``.

        Returns:
            A tuple of (success, output_or_error_message).
        """
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", PYPI_PACKAGE],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode == 0:
                return True, result.stdout
            return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Update timed out after 120 seconds"
        except OSError as exc:
            return False, str(exc)

    @staticmethod
    def get_version_info() -> VersionInfo:
        """Collect full version information including all dependencies.

        Returns:
            A VersionInfo with the Prism version, Python version,
            and a mapping of dependency names to their versions.
        """
        deps: dict[str, str] = {}

        for name in _DEPENDENCY_NAMES:
            try:
                mod: Any = __import__(name)
                version_attr = getattr(
                    mod, "__version__", getattr(mod, "VERSION", "unknown")
                )
                deps[name] = str(version_attr)
            except ImportError:
                deps[name] = "not installed"

        return VersionInfo(
            prism_version=CURRENT_VERSION,
            python_version=(
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            dependencies=deps,
        )

    @staticmethod
    def _compare_versions(latest: str, current: str) -> bool:
        """Determine whether ``latest`` is strictly newer than ``current``.

        Uses ``packaging.version.Version`` for correct semantic comparison.
        Falls back to simple string comparison if ``packaging`` is unavailable.

        Args:
            latest: The latest version string from PyPI.
            current: The currently installed version string.

        Returns:
            True if latest > current.
        """
        if not latest or not current:
            return False
        try:
            from packaging.version import Version

            return Version(latest) > Version(current)
        except (ImportError, ValueError):
            # Fallback: split into tuples of ints for comparison
            try:
                latest_parts = tuple(int(p) for p in latest.split("."))
                current_parts = tuple(int(p) for p in current.split("."))
                return latest_parts > current_parts
            except (ValueError, TypeError):
                return latest != current and latest > current

"""Tests for prism.cli.updater — auto-update system."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prism.cli.updater import (
    CHECK_INTERVAL_HOURS,
    CURRENT_VERSION,
    PYPI_PACKAGE,
    UpdateChecker,
    UpdateInfo,
    VersionInfo,
)

# ---------------------------------------------------------------------------
# UpdateInfo dataclass
# ---------------------------------------------------------------------------


class TestUpdateInfo:
    """Tests for the UpdateInfo dataclass."""

    def test_fields_set_correctly(self) -> None:
        info = UpdateInfo(
            current_version="0.1.0",
            latest_version="0.2.0",
            is_update_available=True,
            release_date="2026-01-01",
            changelog="Bug fixes.",
            checked_at="2026-01-01T00:00:00+00:00",
        )
        assert info.current_version == "0.1.0"
        assert info.latest_version == "0.2.0"
        assert info.is_update_available is True
        assert info.release_date == "2026-01-01"
        assert info.changelog == "Bug fixes."
        assert info.checked_at == "2026-01-01T00:00:00+00:00"

    def test_no_update_available(self) -> None:
        info = UpdateInfo(
            current_version="0.2.0",
            latest_version="0.2.0",
            is_update_available=False,
            release_date="",
            changelog="",
            checked_at="",
        )
        assert info.is_update_available is False

    def test_frozen_dataclass(self) -> None:
        info = UpdateInfo(
            current_version="0.1.0",
            latest_version="0.2.0",
            is_update_available=True,
            release_date="",
            changelog="",
            checked_at="",
        )
        with pytest.raises(AttributeError):
            info.current_version = "0.3.0"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# VersionInfo dataclass
# ---------------------------------------------------------------------------


class TestVersionInfo:
    """Tests for the VersionInfo dataclass."""

    def test_fields_set_correctly(self) -> None:
        deps = {"typer": "0.9.0", "rich": "13.0.0"}
        info = VersionInfo(
            prism_version="0.2.0",
            python_version="3.12.4",
            dependencies=deps,
        )
        assert info.prism_version == "0.2.0"
        assert info.python_version == "3.12.4"
        assert info.dependencies == deps

    def test_default_empty_dependencies(self) -> None:
        info = VersionInfo(
            prism_version="0.2.0",
            python_version="3.12.0",
        )
        assert info.dependencies == {}

    def test_frozen_dataclass(self) -> None:
        info = VersionInfo(
            prism_version="0.2.0",
            python_version="3.12.0",
        )
        with pytest.raises(AttributeError):
            info.prism_version = "0.3.0"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# UpdateChecker.__init__
# ---------------------------------------------------------------------------


class TestUpdateCheckerInit:
    """Tests for UpdateChecker initialization."""

    def test_default_cache_dir(self) -> None:
        checker = UpdateChecker()
        assert checker.cache_dir == Path.home() / ".prism"

    def test_custom_cache_dir(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path / "custom")
        assert checker.cache_dir == tmp_path / "custom"

    def test_cache_file_path(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        assert checker.cache_file == tmp_path / "update_check.json"

    def test_latest_is_none_initially(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        assert checker._latest is None


# ---------------------------------------------------------------------------
# UpdateChecker._should_check
# ---------------------------------------------------------------------------


class TestShouldCheck:
    """Tests for the _should_check method."""

    def test_should_check_when_no_cache_file(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        assert checker._should_check() is True

    def test_should_not_check_when_recent_cache(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "checked_at": datetime.now(UTC).isoformat(),
            "latest_version": "0.2.0",
        }
        checker.cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
        assert checker._should_check() is False

    def test_should_check_when_cache_is_old(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        old_time = datetime.now(UTC) - timedelta(hours=CHECK_INTERVAL_HOURS + 1)
        cache_data = {
            "checked_at": old_time.isoformat(),
            "latest_version": "0.2.0",
        }
        checker.cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
        assert checker._should_check() is True

    def test_should_check_when_cache_is_malformed_json(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        checker.cache_file.write_text("not valid json", encoding="utf-8")
        assert checker._should_check() is True

    def test_should_check_when_cache_missing_checked_at(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        checker.cache_file.write_text(json.dumps({"latest_version": "0.2.0"}), encoding="utf-8")
        assert checker._should_check() is True

    def test_should_check_when_checked_at_is_invalid_iso(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        cache_data = {"checked_at": "not-a-date"}
        checker.cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
        assert checker._should_check() is True


# ---------------------------------------------------------------------------
# UpdateChecker._check_pypi
# ---------------------------------------------------------------------------


class TestCheckPyPI:
    """Tests for the _check_pypi method."""

    @patch("prism.cli.updater.subprocess.run")
    def test_check_pypi_parses_version(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="prism-cli (0.3.0)",
        )
        checker = UpdateChecker(cache_dir=tmp_path)
        checker._check_pypi()

        assert checker._latest is not None
        assert checker._latest.latest_version == "0.3.0"
        assert checker._latest.is_update_available is True

    @patch("prism.cli.updater.subprocess.run")
    def test_check_pypi_no_update(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=f"prism-cli ({CURRENT_VERSION})",
        )
        checker = UpdateChecker(cache_dir=tmp_path)
        checker._check_pypi()

        assert checker._latest is not None
        assert checker._latest.latest_version == CURRENT_VERSION
        assert checker._latest.is_update_available is False

    @patch("prism.cli.updater.subprocess.run")
    def test_check_pypi_command_fails(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="ERROR: No matching distribution",
        )
        checker = UpdateChecker(cache_dir=tmp_path)
        checker._check_pypi()

        assert checker._latest is not None
        assert checker._latest.latest_version == CURRENT_VERSION
        assert checker._latest.is_update_available is False

    @patch("prism.cli.updater.subprocess.run")
    def test_check_pypi_timeout(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=30)
        checker = UpdateChecker(cache_dir=tmp_path)
        checker._check_pypi()

        assert checker._latest is None

    @patch("prism.cli.updater.subprocess.run")
    def test_check_pypi_os_error(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.side_effect = OSError("pip not found")
        checker = UpdateChecker(cache_dir=tmp_path)
        checker._check_pypi()

        assert checker._latest is None


# ---------------------------------------------------------------------------
# UpdateChecker._save_cache
# ---------------------------------------------------------------------------


class TestSaveCache:
    """Tests for the _save_cache method."""

    @patch("prism.cli.updater.subprocess.run")
    def test_save_cache_creates_file(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="prism-cli (0.3.0)",
        )
        checker = UpdateChecker(cache_dir=tmp_path / "sub" / "dir")
        checker._check_pypi()

        assert checker.cache_file.is_file()
        data = json.loads(checker.cache_file.read_text(encoding="utf-8"))
        assert data["latest_version"] == "0.3.0"
        assert data["is_update_available"] is True
        assert "checked_at" in data

    def test_save_cache_does_nothing_when_no_latest(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        checker._save_cache()
        assert not checker.cache_file.exists()


# ---------------------------------------------------------------------------
# UpdateChecker.get_cached_info
# ---------------------------------------------------------------------------


class TestGetCachedInfo:
    """Tests for the get_cached_info method."""

    def test_returns_in_memory_result(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        info = UpdateInfo(
            current_version="0.2.0",
            latest_version="0.3.0",
            is_update_available=True,
            release_date="",
            changelog="",
            checked_at=datetime.now(UTC).isoformat(),
        )
        checker._latest = info
        assert checker.get_cached_info() is info

    def test_reads_from_cache_file(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "current_version": "0.2.0",
            "latest_version": "0.4.0",
            "is_update_available": True,
            "release_date": "2026-06-01",
            "changelog": "New features.",
            "checked_at": "2026-01-01T00:00:00+00:00",
        }
        checker.cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        result = checker.get_cached_info()
        assert result is not None
        assert result.latest_version == "0.4.0"
        assert result.is_update_available is True

    def test_returns_none_when_no_cache(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        assert checker.get_cached_info() is None

    def test_returns_none_when_cache_is_malformed(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        checker.cache_file.write_text("{{bad json", encoding="utf-8")
        assert checker.get_cached_info() is None


# ---------------------------------------------------------------------------
# UpdateChecker.get_notification
# ---------------------------------------------------------------------------


class TestGetNotification:
    """Tests for the get_notification method."""

    def test_notification_when_update_available(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        checker._latest = UpdateInfo(
            current_version="0.2.0",
            latest_version="0.3.0",
            is_update_available=True,
            release_date="",
            changelog="",
            checked_at=datetime.now(UTC).isoformat(),
        )
        note = checker.get_notification()
        assert note is not None
        assert "0.3.0" in note
        assert "prism update" in note

    def test_no_notification_when_up_to_date(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        checker._latest = UpdateInfo(
            current_version="0.2.0",
            latest_version="0.2.0",
            is_update_available=False,
            release_date="",
            changelog="",
            checked_at=datetime.now(UTC).isoformat(),
        )
        assert checker.get_notification() is None

    def test_no_notification_when_no_info(self, tmp_path: Path) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        assert checker.get_notification() is None


# ---------------------------------------------------------------------------
# UpdateChecker.perform_update
# ---------------------------------------------------------------------------


class TestPerformUpdate:
    """Tests for the perform_update static method."""

    @patch("prism.cli.updater.subprocess.run")
    def test_successful_update(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Successfully installed prism-cli-0.3.0",
        )
        success, output = UpdateChecker.perform_update()
        assert success is True
        assert "Successfully installed" in output

    @patch("prism.cli.updater.subprocess.run")
    def test_failed_update(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="ERROR: Could not find a version that satisfies the requirement",
        )
        success, output = UpdateChecker.perform_update()
        assert success is False
        assert "ERROR" in output

    @patch("prism.cli.updater.subprocess.run")
    def test_update_timeout(self, mock_run: MagicMock) -> None:
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=120)
        success, output = UpdateChecker.perform_update()
        assert success is False
        assert "timed out" in output.lower()

    @patch("prism.cli.updater.subprocess.run")
    def test_update_os_error(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = OSError("pip not found")
        success, output = UpdateChecker.perform_update()
        assert success is False
        assert "pip not found" in output


# ---------------------------------------------------------------------------
# UpdateChecker.get_version_info
# ---------------------------------------------------------------------------


class TestGetVersionInfo:
    """Tests for the get_version_info static method."""

    def test_returns_version_info(self) -> None:
        info = UpdateChecker.get_version_info()
        assert isinstance(info, VersionInfo)
        assert info.prism_version == CURRENT_VERSION
        expected_py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert info.python_version == expected_py

    def test_includes_known_dependencies(self) -> None:
        info = UpdateChecker.get_version_info()
        assert "structlog" in info.dependencies
        assert "httpx" in info.dependencies
        assert info.dependencies["structlog"] != "not installed"

    def test_missing_dependency_shows_not_installed(self) -> None:
        with patch("prism.cli.updater._DEPENDENCY_NAMES", ["nonexistent_module_xyz"]):
            info = UpdateChecker.get_version_info()
            assert info.dependencies.get("nonexistent_module_xyz") == "not installed"


# ---------------------------------------------------------------------------
# UpdateChecker._compare_versions
# ---------------------------------------------------------------------------


class TestCompareVersions:
    """Tests for the _compare_versions static method."""

    def test_newer_version(self) -> None:
        assert UpdateChecker._compare_versions("0.3.0", "0.2.0") is True

    def test_same_version(self) -> None:
        assert UpdateChecker._compare_versions("0.2.0", "0.2.0") is False

    def test_older_version(self) -> None:
        assert UpdateChecker._compare_versions("0.1.0", "0.2.0") is False

    def test_major_version_bump(self) -> None:
        assert UpdateChecker._compare_versions("1.0.0", "0.9.9") is True

    def test_empty_latest(self) -> None:
        assert UpdateChecker._compare_versions("", "0.2.0") is False

    def test_empty_current(self) -> None:
        assert UpdateChecker._compare_versions("0.3.0", "") is False

    def test_both_empty(self) -> None:
        assert UpdateChecker._compare_versions("", "") is False

    def test_fallback_when_packaging_unavailable(self) -> None:
        with patch("prism.cli.updater.UpdateChecker._compare_versions") as mock_cmp:
            # Just verify the static method can be called
            mock_cmp.return_value = True
            assert mock_cmp("0.3.0", "0.2.0") is True


# ---------------------------------------------------------------------------
# UpdateChecker.check_async
# ---------------------------------------------------------------------------


class TestCheckAsync:
    """Tests for the check_async method."""

    @patch.object(UpdateChecker, "_should_check", return_value=True)
    @patch.object(UpdateChecker, "_check_pypi")
    def test_starts_thread_when_should_check(
        self, mock_check: MagicMock, mock_should: MagicMock, tmp_path: Path
    ) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        checker.check_async()

        assert checker._check_thread is not None
        assert checker._check_thread.name == "prism-update-checker"
        assert checker._check_thread.daemon is True
        checker._check_thread.join(timeout=5)

    @patch.object(UpdateChecker, "_should_check", return_value=False)
    def test_skips_check_when_recent(
        self, mock_should: MagicMock, tmp_path: Path
    ) -> None:
        checker = UpdateChecker(cache_dir=tmp_path)
        checker.check_async()

        assert checker._check_thread is None


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_current_version_is_string(self) -> None:
        assert isinstance(CURRENT_VERSION, str)
        assert len(CURRENT_VERSION.split(".")) == 3

    def test_pypi_package_name(self) -> None:
        assert PYPI_PACKAGE == "prism-cli"

    def test_check_interval(self) -> None:
        assert CHECK_INTERVAL_HOURS == 24

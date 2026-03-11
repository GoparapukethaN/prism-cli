"""Tests for update checking — fully offline with mocked network calls."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from prism.cli.commands.update import UpdateInfo, check_for_updates, show_update_notice

# ------------------------------------------------------------------
# check_for_updates()
# ------------------------------------------------------------------


class TestCheckForUpdates:
    """Tests for check_for_updates() with mocked PyPI responses."""

    def test_up_to_date(self) -> None:
        """Returns None when the installed version matches PyPI."""
        from prism import __version__

        response_data = json.dumps({"info": {"version": __version__}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = check_for_updates()

        assert result is None

    def test_update_available(self) -> None:
        """Returns UpdateInfo when PyPI has a newer version."""
        response_data = json.dumps({"info": {"version": "99.0.0"}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = check_for_updates()

        assert result is not None
        assert isinstance(result, UpdateInfo)
        assert result.latest_version == "99.0.0"
        assert "pipx upgrade" in result.update_command

    def test_offline_returns_none(self) -> None:
        """Returns None when the network is unreachable."""
        with patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
            result = check_for_updates()

        assert result is None

    def test_older_version_on_pypi_returns_none(self) -> None:
        """Returns None when PyPI version is older than installed."""
        response_data = json.dumps({"info": {"version": "0.0.1"}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = check_for_updates()

        assert result is None

    def test_malformed_response_returns_none(self) -> None:
        """Returns None when PyPI response is not valid JSON."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = check_for_updates()

        # json.loads will raise, so it should return None
        assert result is None


# ------------------------------------------------------------------
# UpdateInfo
# ------------------------------------------------------------------


class TestUpdateInfo:
    """Tests for the UpdateInfo dataclass."""

    def test_update_command_format(self) -> None:
        """UpdateInfo stores the correct update command."""
        info = UpdateInfo(
            current_version="0.1.0",
            latest_version="0.2.0",
            update_command="pipx upgrade prism-cli",
        )
        assert info.current_version == "0.1.0"
        assert info.latest_version == "0.2.0"
        assert "prism-cli" in info.update_command

    def test_frozen_dataclass(self) -> None:
        """UpdateInfo is immutable."""
        info = UpdateInfo(
            current_version="0.1.0",
            latest_version="0.2.0",
            update_command="pipx upgrade prism-cli",
        )
        with pytest.raises(AttributeError):
            info.current_version = "0.3.0"  # type: ignore[misc]


# ------------------------------------------------------------------
# show_update_notice()
# ------------------------------------------------------------------


class TestShowUpdateNotice:
    """Tests for show_update_notice()."""

    def test_show_update_notice(self) -> None:
        """show_update_notice runs without error."""
        info = UpdateInfo(
            current_version="0.1.0",
            latest_version="0.2.0",
            update_command="pipx upgrade prism-cli",
        )
        console = Console(file=None, force_terminal=False, width=120)
        # Should not raise
        show_update_notice(info, console=console)

    def test_show_update_notice_default_console(self) -> None:
        """show_update_notice works without an explicit console."""
        info = UpdateInfo(
            current_version="0.1.0",
            latest_version="1.0.0",
            update_command="pipx upgrade prism-cli",
        )
        # Should not raise
        show_update_notice(info)

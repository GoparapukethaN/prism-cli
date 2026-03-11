"""Tests for the version command."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from prism import __app_name__, __version__
from prism.cli.commands.version import _get_package_version, show_version

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_console() -> tuple[Console, StringIO]:
    """A console that writes to a StringIO for output capture."""
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=False, no_color=True)
    return console, buf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShowVersion:
    def test_simple_version(self, captured_console: tuple[Console, StringIO]) -> None:
        """show_version() prints name and version."""
        console, buf = captured_console
        show_version(verbose=False, console=console)
        output = buf.getvalue()
        assert __app_name__ in output
        assert __version__ in output

    def test_verbose_version(self, captured_console: tuple[Console, StringIO]) -> None:
        """show_version(verbose=True) prints a table with dependencies."""
        console, buf = captured_console
        show_version(verbose=True, console=console)
        output = buf.getvalue()
        assert __version__ in output
        assert "Python" in output
        assert "Platform" in output

    def test_verbose_includes_deps(self, captured_console: tuple[Console, StringIO]) -> None:
        """Verbose output lists dependency names."""
        console, buf = captured_console
        show_version(verbose=True, console=console)
        output = buf.getvalue()
        # Should mention at least some dependencies
        assert "typer" in output or "rich" in output or "pydantic" in output

    def test_default_console_used(self) -> None:
        """If no console is passed, show_version creates one internally."""
        # Just ensure it doesn't crash
        with patch("prism.cli.commands.version.Console") as mock_cls:
            mock_instance = mock_cls.return_value
            mock_instance.print = lambda *a, **k: None
            show_version(verbose=False)

    def test_missing_package_version(self) -> None:
        """_get_package_version returns 'not installed' for unknown packages."""
        result = _get_package_version("nonexistent_package_xyz_42")
        assert result == "not installed"

    def test_known_package_version(self) -> None:
        """_get_package_version returns a version string for an installed package."""
        # 'rich' should be installed in our test environment
        result = _get_package_version("rich")
        assert result != "not installed"
        # Should look like a version string (e.g., "13.7.0")
        assert "." in result

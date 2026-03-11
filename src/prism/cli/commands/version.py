"""Version information display for Prism CLI."""

from __future__ import annotations

import platform
import sys

from rich.console import Console
from rich.table import Table

from prism import __app_name__, __version__


def show_version(verbose: bool = False, console: Console | None = None) -> None:
    """Show version info.  If verbose, include all dependency versions.

    Args:
        verbose: When ``True``, display Python version and installed
            dependency versions.
        console: Optional Rich console for output.
    """
    con = console or Console()

    if not verbose:
        con.print(f"[bold]{__app_name__}[/bold] version {__version__}")
        return

    table = Table(title=f"{__app_name__} v{__version__}", show_header=True)
    table.add_column("Component", style="bold")
    table.add_column("Version")

    # Python
    table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    table.add_row("Platform", platform.platform())

    # Core dependencies
    deps = [
        "typer",
        "rich",
        "pydantic",
        "structlog",
        "yaml",
        "litellm",
        "click",
    ]

    for dep in deps:
        version = _get_package_version(dep)
        table.add_row(dep, version)

    con.print(table)


def _get_package_version(package_name: str) -> str:
    """Get the installed version of a package.

    Args:
        package_name: The package name to look up.

    Returns:
        Version string, or ``"not installed"`` if the package is missing.
    """
    # Map module names to distribution names where they differ
    dist_names: dict[str, str] = {
        "yaml": "pyyaml",
    }
    dist_name = dist_names.get(package_name, package_name)

    try:
        from importlib.metadata import version

        return version(dist_name)
    except Exception:
        return "not installed"

"""Update checking and notification for Prism CLI.

Checks PyPI for newer versions and displays an update notice when
available.  All network calls are isolated to :func:`check_for_updates`
so they can be easily mocked in tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel

from prism import __version__


@dataclass(frozen=True)
class UpdateInfo:
    """Information about an available update."""

    current_version: str
    latest_version: str
    update_command: str


def check_for_updates() -> UpdateInfo | None:
    """Check PyPI for a newer version of ``prism-cli``.

    Returns:
        An :class:`UpdateInfo` if a newer version is available,
        or ``None`` if the CLI is up to date **or** the network
        is unreachable.
    """
    try:
        import urllib.request

        url = "https://pypi.org/pypi/prism-cli/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            import json

            data = json.loads(resp.read().decode())
            latest = data.get("info", {}).get("version", __version__)
    except Exception:
        return None

    if latest == __version__:
        return None

    # Simple semver comparison: split on '.' and compare tuples
    try:
        current_parts = tuple(int(x) for x in __version__.split("."))
        latest_parts = tuple(int(x) for x in latest.split("."))
    except ValueError:
        # Non-numeric version components — fall back to string comparison
        if latest <= __version__:
            return None
        return UpdateInfo(
            current_version=__version__,
            latest_version=latest,
            update_command="pipx upgrade prism-cli",
        )

    if latest_parts <= current_parts:
        return None

    return UpdateInfo(
        current_version=__version__,
        latest_version=latest,
        update_command="pipx upgrade prism-cli",
    )


def show_update_notice(info: UpdateInfo, console: Console | None = None) -> None:
    """Display an update notice to the user.

    Args:
        info: The :class:`UpdateInfo` describing the available update.
        console: Optional Rich console for output.
    """
    cons = console or Console()
    cons.print(
        Panel(
            f"[bold yellow]Update available:[/] "
            f"{info.current_version} -> {info.latest_version}\n"
            f"Run: [cyan]{info.update_command}[/]",
            title="Prism Update",
            border_style="yellow",
        )
    )

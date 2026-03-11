"""CLI commands for multi-project workspace management.

Provides ``prism projects list|new|switch|remove`` subcommands for
registering, switching between, and managing multiple projects.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — required at runtime by Typer

import typer
from rich.console import Console
from rich.table import Table

from prism.workspace.manager import WorkspaceManager

console = Console()

projects_app = typer.Typer(
    help="Manage multi-project workspace.",
    no_args_is_help=True,
)


def _get_workspace_manager() -> WorkspaceManager:
    """Create a WorkspaceManager using the resolved Prism home directory.

    Returns:
        A ``WorkspaceManager`` instance.
    """
    from prism.config.settings import load_settings

    settings = load_settings()
    settings.ensure_directories()
    return WorkspaceManager(settings.prism_home)


@projects_app.command("list")
def projects_list() -> None:
    """Show all known projects."""
    manager = _get_workspace_manager()
    projects = manager.list_projects()

    if not projects:
        console.print("[yellow]No projects registered.[/]")
        console.print("Use [cyan]prism projects new <name> <path>[/] to register one.")
        return

    table = Table(title="Registered Projects", show_lines=False)
    table.add_column("", width=3)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Path")
    table.add_column("Description", style="dim")
    table.add_column("Last Accessed", style="dim")

    for project in projects:
        marker = "[green]*[/]" if project.active else " "
        # Truncate timestamp for display
        accessed = project.last_accessed[:19].replace("T", " ") if project.last_accessed else ""
        table.add_row(
            marker,
            project.name,
            project.path,
            project.description or "-",
            accessed,
        )

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]* = active project[/]")
    console.print()


@projects_app.command("new")
def projects_new(
    name: str = typer.Argument(help="Unique project name (alphanumeric, hyphens, underscores)."),
    path: Path = typer.Argument(
        help="Path to the project root directory.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    description: str = typer.Option(
        "",
        "--description",
        "-d",
        help="Optional short description of the project.",
    ),
) -> None:
    """Register a new project in the workspace."""
    manager = _get_workspace_manager()

    try:
        project = manager.register_project(
            name=name,
            path=path,
            description=description,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Registered[/] project '{project.name}' at {project.path}")
    if project.active:
        console.print(f"[cyan]Active project:[/] {project.name}")


@projects_app.command("switch")
def projects_switch(
    name: str = typer.Argument(help="Name of the project to switch to."),
) -> None:
    """Switch the active project."""
    manager = _get_workspace_manager()

    try:
        project = manager.switch_project(name)
    except ValueError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Switched[/] to project '{project.name}' at {project.path}")


@projects_app.command("remove")
def projects_remove(
    name: str = typer.Argument(help="Name of the project to unregister."),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Unregister a project (does not delete files)."""
    manager = _get_workspace_manager()

    try:
        project = manager.get_project(name)
    except ValueError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    if not force:
        confirm = typer.confirm(
            f"Remove project '{name}' ({project.path})? Files will NOT be deleted."
        )
        if not confirm:
            console.print("[yellow]Cancelled.[/]")
            raise typer.Exit()

    try:
        manager.remove_project(name)
    except ValueError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Removed[/] project '{name}' from workspace.")

    # Show new active project if any
    active = manager.get_active_project()
    if active:
        console.print(f"[cyan]Active project:[/] {active.name}")

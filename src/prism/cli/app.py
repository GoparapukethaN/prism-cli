"""Prism CLI application — Typer command definitions and entry point."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from prism import __app_name__, __version__

app = typer.Typer(
    name=__app_name__,
    help="Prism — Multi-API Intelligent Router CLI. "
    "Claude Code-level capabilities with cost-optimizing routing.",
    no_args_is_help=False,
    add_completion=True,
)

console = Console()

# --- Sub-command groups ---

auth_app = typer.Typer(help="Manage API keys and provider authentication.")
app.add_typer(auth_app, name="auth")

config_app = typer.Typer(help="View and modify Prism configuration.")
app.add_typer(config_app, name="config")

db_app = typer.Typer(help="Database maintenance commands.")
app.add_typer(db_app, name="db")

plugins_app = typer.Typer(help="Manage Prism plugins.")
app.add_typer(plugins_app, name="plugins")

from prism.cli.commands.projects import projects_app  # noqa: E402

app.add_typer(projects_app, name="projects")


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold]{__app_name__}[/bold] version {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        "-r",
        help="Project root directory (default: current directory).",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Force a specific model for all requests.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output (INFO level logging).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug output (DEBUG level logging).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Auto-approve file writes and command execution.",
    ),
    web: bool = typer.Option(
        False,
        "--web",
        help="Enable web browsing and screenshot tools.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show routing decisions without executing.",
    ),
    new_session: bool = typer.Option(
        False,
        "--new-session",
        help="Force a new session instead of resuming.",
    ),
    budget: float | None = typer.Option(
        None,
        "--budget",
        help="Set daily budget limit in USD.",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="Disable all cloud providers (Ollama only).",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable response caching.",
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Switch to a registered project by name before starting.",
    ),
    no_verify_ssl: bool = typer.Option(
        False,
        "--no-verify-ssl",
        help="Skip SSL certificate verification (dangerous).",
    ),
) -> None:
    """Start Prism interactive REPL or run with subcommands."""
    if ctx.invoked_subcommand is not None:
        # A subcommand was invoked, let it handle things
        return

    # No subcommand — start interactive REPL
    _start_repl(
        root=root,
        model=model,
        verbose=verbose,
        debug=debug,
        yes=yes,
        web=web,
        dry_run=dry_run,
        new_session=new_session,
        budget=budget,
        offline=offline,
        no_cache=no_cache,
        project=project,
        no_verify_ssl=no_verify_ssl,
    )


def _start_repl(
    root: Path | None,
    model: str | None,
    verbose: bool,
    debug: bool,
    yes: bool,
    web: bool,
    dry_run: bool,
    new_session: bool,
    budget: float | None,
    offline: bool,
    no_cache: bool = False,
    project: str | None = None,
    no_verify_ssl: bool = False,
) -> None:
    """Initialize and start the interactive REPL."""
    import os

    from prism.config.settings import load_settings

    # Handle --no-verify-ssl
    if no_verify_ssl:
        os.environ["PRISM_SSL_VERIFY"] = "false"
        console.print(
            "[yellow]Warning:[/] SSL verification "
            "disabled. This is insecure."
        )

    # Build config overrides from CLI flags
    overrides: dict[str, object] = {}
    if model:
        overrides["pinned_model"] = model
    if yes:
        overrides["tools.auto_approve"] = True
    if web:
        overrides["tools.web_enabled"] = True
    if budget is not None:
        overrides["budget.daily_limit"] = budget

    # Configure logging
    log_level = "WARNING"
    if verbose:
        log_level = "INFO"
    if debug:
        log_level = "DEBUG"

    _configure_logging(log_level)

    # Handle --project flag: switch to the named project and use its root
    effective_root = root
    if project:
        settings_tmp = load_settings()
        settings_tmp.ensure_directories()
        from prism.workspace.manager import WorkspaceManager

        workspace = WorkspaceManager(settings_tmp.prism_home)
        try:
            proj = workspace.switch_project(project)
            effective_root = Path(proj.path)
            console.print(f"[cyan]Project:[/] {proj.name} ({proj.path})")
        except ValueError as exc:
            console.print(f"[red]Error:[/] {exc}")
            raise typer.Exit(1) from exc

    # Load settings
    settings = load_settings(
        project_root=effective_root,
        config_overrides=overrides,
    )
    settings.ensure_directories()

    if dry_run:
        console.print("[yellow]Dry-run mode:[/] routing decisions shown but not executed.")

    if offline:
        console.print("[yellow]Offline mode:[/] cloud providers disabled, Ollama only.")

    # Show recent projects on startup
    _show_recent_projects(settings)

    # Print welcome banner
    _print_banner(settings)

    # Start REPL
    from prism.cli.repl import run_repl

    run_repl(
        settings=settings,
        console=console,
        dry_run=dry_run,
        offline=offline,
        no_cache=no_cache,
    )


def _configure_logging(level: str) -> None:
    """Configure structlog for the application."""
    import logging

    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.WARNING)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _show_recent_projects(settings: object) -> None:
    """Show recent projects on startup if any are registered."""
    from prism.config.settings import Settings

    if not isinstance(settings, Settings):
        return

    try:
        from prism.workspace.manager import WorkspaceManager

        workspace = WorkspaceManager(settings.prism_home)
        recent = workspace.get_recent_projects(limit=3)
        if recent:
            active = workspace.get_active_project()
            active_name = active.name if active else None
            names = []
            for p in recent:
                if p.name == active_name:
                    names.append(f"[bold cyan]{p.name}[/bold cyan]")
                else:
                    names.append(f"[dim]{p.name}[/dim]")
            console.print(f"[dim]Projects:[/dim] {' | '.join(names)}")
    except Exception:  # noqa: S110
        pass  # Non-critical — don't block startup


def _print_banner(settings: object) -> None:
    """Print the welcome banner."""
    from rich.panel import Panel

    console.print()
    console.print(
        Panel(
            f"[bold cyan]Prism[/bold cyan] v{__version__} — "
            "Multi-API Intelligent Router\n"
            "[dim]Type your request, or use /help for commands.[/dim]",
            border_style="cyan",
        )
    )
    console.print()


# --- Auth Subcommands ---


@auth_app.command("add")
def auth_add(
    provider: str = typer.Argument(
        help="Provider name (anthropic, openai, google, deepseek, groq, mistral).",
    ),
) -> None:
    """Add an API key for a provider."""
    from prism.config.settings import load_settings

    settings = load_settings()
    settings.ensure_directories()

    from prism.auth.manager import AuthManager

    auth = AuthManager(settings)

    console.print(f"Adding API key for [bold]{provider}[/bold]...")
    key = typer.prompt("API key", hide_input=True)

    if not key.strip():
        console.print("[red]Error:[/] API key cannot be empty.")
        raise typer.Exit(1)

    from prism.auth.validator import KeyValidator

    validator = KeyValidator()
    if not validator.validate_key(provider, key.strip()):
        console.print(
            f"[yellow]Warning:[/] Key format doesn't match expected pattern for {provider}. "
            "Storing anyway."
        )

    auth.store_key(provider, key.strip())
    masked = "..." + key.strip()[-4:] if len(key.strip()) > 4 else "****"
    console.print(f"[green]Stored[/green] API key for {provider} ({masked})")


@auth_app.command("status")
def auth_status() -> None:
    """Show status of all configured providers."""
    from prism.config.settings import load_settings

    settings = load_settings()

    from prism.auth.manager import AuthManager

    auth = AuthManager(settings)
    statuses = auth.list_configured()

    console.print()
    for status in statuses:
        if status["configured"]:
            models = ", ".join(status.get("models", []))
            console.print(
                f"  [green]✓[/green] [bold]{status['display_name']:15s}[/bold] ({models})"
            )
        else:
            console.print(
                f"  [red]✗[/red] [bold]{status['display_name']:15s}[/bold] (not configured)"
            )
    console.print()


@auth_app.command("remove")
def auth_remove(
    provider: str = typer.Argument(help="Provider name to remove."),
) -> None:
    """Remove a stored API key."""
    from prism.config.settings import load_settings

    settings = load_settings()

    from prism.auth.manager import AuthManager

    auth = AuthManager(settings)
    auth.remove_key(provider)
    console.print(f"[green]Removed[/green] API key for {provider}")


# --- Config Subcommands ---


@config_app.command("get")
def config_get(
    key: str = typer.Argument(help="Configuration key (e.g., 'routing.simple_threshold')."),
) -> None:
    """Get a configuration value."""
    from prism.config.settings import load_settings

    settings = load_settings()
    value = settings.get(key)
    if value is None:
        console.print(f"[yellow]Not set:[/] {key}")
    else:
        console.print(f"{key} = {value}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(help="Configuration key."),
    value: str = typer.Argument(help="Value to set."),
) -> None:
    """Set a configuration value."""
    import yaml

    from prism.config.settings import load_config_file, load_settings

    settings = load_settings()
    config_path = settings.config_file_path

    # Load existing config
    data = load_config_file(config_path)

    # Set the value (handle dot-separated keys)
    parts = key.split(".")
    target = data
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]

    # Type coercion
    parsed_value: object
    if value.lower() in ("true", "false"):
        parsed_value = value.lower() == "true"
    elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
        parsed_value = float(value) if "." in value else int(value)
    elif value.lower() == "null" or value.lower() == "none":
        parsed_value = None
    else:
        parsed_value = value

    target[parts[-1]] = parsed_value

    # Write back
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

    console.print(f"[green]Set[/green] {key} = {parsed_value}")


# --- DB Subcommands ---


@db_app.command("stats")
def db_stats() -> None:
    """Show database statistics."""
    from prism.config.settings import load_settings

    settings = load_settings()
    db_path = settings.db_path

    if not db_path.exists():
        console.print("[yellow]No database found.[/] Run Prism first to create it.")
        return

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    tables = ["routing_decisions", "cost_entries", "sessions", "tool_executions"]
    console.print(f"\n[bold]Database:[/] {db_path} ({db_path.stat().st_size / 1024:.1f} KB)\n")

    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            count = cursor.fetchone()[0]
            console.print(f"  {table:25s} {count:>8d} rows")
        except sqlite3.OperationalError:
            console.print(f"  {table:25s} [dim]not found[/dim]")

    conn.close()
    console.print()


@db_app.command("vacuum")
def db_vacuum() -> None:
    """Reclaim unused database space."""
    import sqlite3

    from prism.config.settings import load_settings

    settings = load_settings()
    db_path = settings.db_path

    if not db_path.exists():
        console.print("[yellow]No database found.[/]")
        return

    size_before = db_path.stat().st_size
    conn = sqlite3.connect(str(db_path))
    conn.execute("VACUUM")
    conn.close()
    size_after = db_path.stat().st_size

    saved = size_before - size_after
    console.print(
        f"[green]Vacuum complete.[/] Reclaimed {saved / 1024:.1f} KB "
        f"({size_after / 1024:.1f} KB remaining)"
    )


# --- Plugins Subcommands ---


@plugins_app.command("list")
def plugins_list() -> None:
    """List installed and available plugins."""
    from rich.table import Table

    from prism.config.settings import load_settings
    from prism.plugins.manager import PluginManager

    settings = load_settings()
    settings.ensure_directories()
    manager = PluginManager(
        plugins_dir=settings.prism_home / "plugins",
    )

    installed = manager.list_installed()
    available = manager.list_available()

    # Installed plugins table
    console.print("\n[bold]Installed Plugins[/bold]\n")
    if installed:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green")
        table.add_column("Version")
        table.add_column("Enabled")
        table.add_column("Description")
        for info in installed:
            m = info.manifest
            status = "[green]yes[/green]" if info.enabled else "[red]no[/red]"
            table.add_row(m.name, m.version, status, m.description)
        console.print(table)
    else:
        console.print("  [dim]No plugins installed.[/dim]")

    # Available plugins table
    console.print("\n[bold]Available Plugins[/bold]\n")
    installed_names = {i.manifest.name for i in installed}
    not_installed = [
        a for a in available if a.name not in installed_names
    ]
    if not_installed:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="yellow")
        table.add_column("Version")
        table.add_column("Description")
        for m in not_installed:
            table.add_row(m.name, m.version, m.description)
        console.print(table)
    else:
        console.print("  [dim]All available plugins are installed.[/dim]")
    console.print()


@plugins_app.command("install")
def plugins_install(
    source: str = typer.Argument(
        help="Plugin name, GitHub URL, or local path.",
    ),
) -> None:
    """Install a plugin."""
    from prism.config.settings import load_settings
    from prism.plugins.manager import PluginError, PluginManager

    settings = load_settings()
    settings.ensure_directories()
    manager = PluginManager(
        plugins_dir=settings.prism_home / "plugins",
    )

    console.print(f"[dim]Installing plugin from:[/dim] {source}")
    try:
        info = manager.install(source)
        console.print(
            f"[green]Installed[/green] "
            f"{info.manifest.name} v{info.manifest.version}"
        )
    except PluginError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc


@plugins_app.command("remove")
def plugins_remove(
    name: str = typer.Argument(help="Plugin name to remove."),
) -> None:
    """Remove an installed plugin."""
    from prism.config.settings import load_settings
    from prism.plugins.manager import PluginError, PluginManager

    settings = load_settings()
    settings.ensure_directories()
    manager = PluginManager(
        plugins_dir=settings.prism_home / "plugins",
    )

    try:
        removed = manager.remove(name)
        if removed:
            console.print(
                f"[green]Removed[/green] plugin: {name}"
            )
        else:
            console.print(
                f"[yellow]Plugin not found:[/] {name}"
            )
    except PluginError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc


@plugins_app.command("info")
def plugins_info(
    name: str = typer.Argument(help="Plugin name."),
) -> None:
    """Show plugin details."""
    from prism.config.settings import load_settings
    from prism.plugins.manager import PluginManager

    settings = load_settings()
    settings.ensure_directories()
    manager = PluginManager(
        plugins_dir=settings.prism_home / "plugins",
    )

    info = manager.get_plugin(name)
    if info is None:
        console.print(f"[yellow]Plugin not found:[/] {name}")
        raise typer.Exit(1)

    m = info.manifest
    console.print(f"\n[bold]{m.name}[/bold] v{m.version}")
    console.print(f"  Author:      {m.author or 'Unknown'}")
    console.print(f"  Description: {m.description or 'N/A'}")
    console.print(f"  License:     {m.license or 'N/A'}")
    console.print(f"  Homepage:    {m.homepage or 'N/A'}")

    enabled = "[green]yes[/green]" if info.enabled else "[red]no[/red]"
    console.print(f"  Enabled:     {enabled}")
    console.print(f"  Path:        {info.install_path}")

    if info.installed_at:
        console.print(f"  Installed:   {info.installed_at}")
    if info.source:
        console.print(f"  Source:      {info.source}")

    if m.tools:
        names = ", ".join(t.name for t in m.tools)
        console.print(f"  Tools:       {names}")
    if m.commands:
        names = ", ".join(c.name for c in m.commands)
        console.print(f"  Commands:    {names}")
    if m.dependencies:
        deps = ", ".join(m.dependencies)
        console.print(f"  Deps:        {deps}")
    console.print()


# --- Single-shot Commands ---


@app.command("ask")
def ask(
    prompt: str = typer.Argument(help="Question or task to perform."),
    model: str | None = typer.Option(None, "--model", "-m"),
    root: Path | None = typer.Option(None, "--root", "-r"),
) -> None:
    """Ask a one-shot question (no interactive REPL)."""
    console.print(f"[dim]Processing:[/] {prompt[:80]}...")
    console.print("[yellow]Single-shot mode not yet implemented. Use interactive REPL.[/]")


@app.command("architect")
def architect_command(
    goal: str = typer.Argument(
        default="",
        help="Goal description for the architect to plan and execute.",
    ),
    resume: str | None = typer.Option(
        None, "--resume",
        help="Resume a paused plan by ID.",
    ),
    list_plans: bool = typer.Option(
        False, "--list",
        help="List all plans.",
    ),
    status: str | None = typer.Option(
        None, "--status",
        help="Show plan status by ID.",
    ),
    rollback: str | None = typer.Option(
        None, "--rollback",
        help="Rollback a plan by ID.",
    ),
    root: Path | None = typer.Option(
        None, "--root", "-r",
        help="Project root directory.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Plan and execute complex multi-step tasks with architect mode."""
    import asyncio

    from prism.config.settings import load_settings

    settings = load_settings(project_root=root)
    settings.ensure_directories()

    from prism.architect.display import (
        display_execution_summary,
        display_plan_list,
        display_plan_review,
        display_rollback_result,
    )
    from prism.architect.executor import ArchitectExecutor
    from prism.architect.planner import ArchitectPlanner
    from prism.architect.storage import PlanStorage
    from prism.cost.tracker import CostTracker
    from prism.db.database import Database

    try:
        db = Database(settings.db_path)
        tracker = CostTracker(db=db, settings=settings)

        # --- list ---
        if list_plans:
            storage = PlanStorage(db)
            plans = storage.list_plans()
            display_plan_list(plans, console)
            return

        # --- status ---
        if status is not None:
            storage = PlanStorage(db)
            plan = storage.load_plan(status)
            if plan is None:
                console.print(
                    f"[yellow]Plan not found:[/] {status}"
                )
                raise typer.Exit(1)
            display_plan_review(plan, console)
            return

        # --- resume ---
        if resume is not None:
            storage = PlanStorage(db)
            plan = storage.load_plan(resume)
            if plan is None:
                console.print(
                    f"[yellow]Plan not found:[/] {resume}"
                )
                raise typer.Exit(1)
            executor = ArchitectExecutor(
                settings=settings,
                cost_tracker=tracker,
            )
            summary = asyncio.get_event_loop(
            ).run_until_complete(
                executor.resume(plan, storage=storage)
            )
            display_execution_summary(summary, console)
            storage.save_plan(plan)
            return

        # --- rollback ---
        if rollback is not None:
            storage = PlanStorage(db)
            plan = storage.load_plan(rollback)
            if plan is None:
                console.print(
                    f"[yellow]Plan not found:[/] {rollback}"
                )
                raise typer.Exit(1)
            executor = ArchitectExecutor(
                settings=settings,
                cost_tracker=tracker,
            )
            success, description = executor.rollback(plan)
            display_rollback_result(
                success, console, description=description,
            )
            storage.save_plan(plan)
            return

        # --- create + execute (default) ---
        if not goal.strip():
            console.print(
                "[yellow]Usage:[/] prism architect <goal>\n"
                "[dim]Options: --list, --status <id>, "
                "--resume <id>, --rollback <id>[/dim]"
            )
            raise typer.Exit(1)

        planner = ArchitectPlanner(
            settings=settings,
            cost_tracker=tracker,
        )
        plan = planner.create_plan(goal)
        display_plan_review(plan, console)

        # Auto-approve and execute
        plan.status = "approved"
        console.print(
            "[green]Plan auto-approved. Executing...[/green]"
        )

        executor = ArchitectExecutor(
            settings=settings,
            cost_tracker=tracker,
        )
        summary = asyncio.get_event_loop(
        ).run_until_complete(
            executor.execute_plan(plan)
        )
        display_execution_summary(summary, console)

        # Persist plan
        storage = PlanStorage(db)
        storage.save_plan(plan)
        console.print(
            f"[dim]Plan {plan.id[:12]}... saved.[/dim]"
        )

    except typer.Exit:
        raise
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Architect error:[/] {exc}")
        raise typer.Exit(1) from exc


@app.command("init")
def init_project(
    root: Path | None = typer.Option(None, "--root", "-r"),
) -> None:
    """Initialize Prism for a project (setup wizard)."""
    from prism.config.settings import load_settings

    project_root = root or Path.cwd()
    settings = load_settings(project_root=project_root)
    settings.ensure_directories()

    prism_md = project_root / ".prism.md"
    if prism_md.exists():
        console.print(f"[yellow]Project already initialized:[/] {prism_md}")
        return

    # Create default .prism.md
    project_name = project_root.name
    content = f"""# Project: {project_name}

## Stack
<!-- Describe your tech stack here -->

## Conventions
<!-- Describe your coding conventions here -->

## Architecture
<!-- Describe your project structure here -->

## Notes for AI
<!-- Add any special instructions for Prism here -->
"""
    prism_md.write_text(content)
    console.print(f"[green]Created[/] {prism_md}")
    console.print("Edit this file to help Prism understand your project.")

    # Ensure ~/.prism/config.yaml exists
    config_path = settings.config_file_path
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        import yaml

        default_config = {
            "routing": {"simple_threshold": 0.3, "medium_threshold": 0.7},
            "budget": {"daily_limit": None, "monthly_limit": None},
            "tools": {"web_enabled": False, "allowed_commands": ["python -m pytest"]},
        }
        with config_path.open("w") as f:
            yaml.safe_dump(default_config, f, default_flow_style=False)
        console.print(f"[green]Created[/] {config_path}")

    console.print("\n[bold]Next steps:[/]")
    console.print("  1. Add API keys: [cyan]prism auth add anthropic[/]")
    console.print("  2. Start coding:  [cyan]prism[/]")


@app.command("status")
def status() -> None:
    """Check status of all providers and system health."""
    from prism.config.settings import load_settings

    settings = load_settings()
    console.print("\n[bold]Prism Status[/bold]\n")

    # Database
    db_path = settings.db_path
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        console.print(f"  Database: [green]OK[/] ({size_kb:.1f} KB)")
    else:
        console.print("  Database: [yellow]Not created yet[/]")

    # Config
    config_path = settings.config_file_path
    if config_path.exists():
        console.print(f"  Config:   [green]OK[/] ({config_path})")
    else:
        console.print("  Config:   [yellow]Not found[/] (using defaults)")

    # Project
    project_root = settings.project_root
    prism_md = project_root / ".prism.md"
    if prism_md.exists():
        console.print(f"  Project:  [green]OK[/] ({project_root})")
    else:
        console.print(f"  Project:  [yellow]No .prism.md[/] ({project_root})")

    # Providers
    console.print("\n[bold]Providers:[/bold]\n")
    auth_status()


@app.command("blame")
def blame(
    description: str = typer.Argument(
        default="", help="Bug description",
    ),
    test_command: str | None = typer.Option(
        None, "--test", help="Test command for bisect",
    ),
    good_commit: str | None = typer.Option(
        None, "--good", help="Known good commit",
    ),
    list_reports: bool = typer.Option(
        False, "--list", help="List saved reports",
    ),
    root: Path | None = typer.Option(
        None, "--root", "-r",
        help="Project root directory.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Trace the root cause of a bug using git history analysis."""
    from rich.panel import Panel
    from rich.table import Table

    from prism.intelligence.blame import CausalBlameTracer

    project_root = root or Path.cwd()

    try:
        tracer = CausalBlameTracer(project_root=project_root)
    except Exception as exc:
        console.print(f"[red]Error initializing blame tracer:[/] {exc}")
        raise typer.Exit(1) from exc

    # --- list ---
    if list_reports:
        reports = tracer.list_reports()
        if not reports:
            console.print("[yellow]No blame reports found.[/]")
            return

        table = Table(title="Saved Blame Reports")
        table.add_column("File", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Modified")

        for rp in reports:
            size_kb = rp.stat().st_size / 1024
            mtime = datetime.fromtimestamp(
                rp.stat().st_mtime, tz=UTC,
            ).strftime("%Y-%m-%d %H:%M")
            table.add_row(rp.name, f"{size_kb:.1f} KB", mtime)

        console.print(table)
        return

    # --- trace ---
    if not description.strip():
        console.print(
            "[yellow]Usage:[/] prism blame <description>\n"
            "[dim]Options: --list, --test <cmd>, --good <commit>[/dim]"
        )
        raise typer.Exit(1)

    console.print(
        f"[dim]Tracing blame for:[/] {description[:80]}..."
    )
    if test_command:
        console.print(f"[dim]Test command:[/] {test_command}")
    if good_commit:
        console.print(f"[dim]Good commit:[/] {good_commit}")

    try:
        report = tracer.trace(
            bug_description=description,
            test_command=test_command,
            good_commit=good_commit,
        )
    except Exception as exc:
        console.print(f"[red]Blame trace failed:[/] {exc}")
        raise typer.Exit(1) from exc

    # Display results
    confidence_pct = int(report.confidence * 100)
    if confidence_pct >= 70:
        conf_style = "green"
    elif confidence_pct >= 40:
        conf_style = "yellow"
    else:
        conf_style = "red"

    panel_content = (
        f"[bold]Breaking Commit:[/] {report.breaking_commit[:12]}\n"
        f"[bold]Author:[/] {report.breaking_author}\n"
        f"[bold]Date:[/] {report.breaking_date}\n"
        f"[bold]Message:[/] {report.breaking_message}\n"
        f"[bold]Confidence:[/] [{conf_style}]{confidence_pct}%[/{conf_style}]\n"
        f"[bold]Bisect Steps:[/] {report.bisect_steps}"
    )
    console.print(Panel(panel_content, title="Blame Report", border_style="cyan"))

    if report.affected_files:
        console.print(f"\n[bold]Affected Files ({len(report.affected_files)}):[/]")
        for f in report.affected_files:
            console.print(f"  [cyan]{f}[/]")

    if report.related_tests:
        console.print(f"\n[bold]Related Tests ({len(report.related_tests)}):[/]")
        for t in report.related_tests:
            console.print(f"  [green]{t}[/]")

    if report.causal_narrative:
        console.print(
            Panel(
                report.causal_narrative,
                title="Causal Narrative",
                border_style="dim",
            )
        )


@app.command("impact")
def impact(
    description: str = typer.Argument(
        default="", help="Description of planned change",
    ),
    files: list[str] | None = typer.Option(
        None, "--file", "-f", help="Target files",
    ),
    list_reports: bool = typer.Option(
        False, "--list", help="List saved reports",
    ),
    root: Path | None = typer.Option(
        None, "--root", "-r",
        help="Project root directory.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Analyze blast radius and impact of a planned code change."""
    from rich.panel import Panel
    from rich.table import Table

    from prism.intelligence.blast_radius import (
        BlastRadiusAnalyzer,
        RiskLevel,
    )

    project_root = root or Path.cwd()

    try:
        analyzer = BlastRadiusAnalyzer(project_root=project_root)
    except Exception as exc:
        console.print(f"[red]Error initializing analyzer:[/] {exc}")
        raise typer.Exit(1) from exc

    # --- list ---
    if list_reports:
        reports = analyzer.list_reports()
        if not reports:
            console.print("[yellow]No impact reports found.[/]")
            return

        table = Table(title="Saved Impact Reports")
        table.add_column("File", style="cyan")
        table.add_column("Size", justify="right")
        table.add_column("Modified")

        for rp in reports:
            size_kb = rp.stat().st_size / 1024
            mtime = datetime.fromtimestamp(
                rp.stat().st_mtime, tz=UTC,
            ).strftime("%Y-%m-%d %H:%M")
            table.add_row(rp.name, f"{size_kb:.1f} KB", mtime)

        console.print(table)
        return

    # --- analyze ---
    if not description.strip():
        console.print(
            "[yellow]Usage:[/] prism impact <description>\n"
            "[dim]Options: --list, --file <path>[/dim]"
        )
        raise typer.Exit(1)

    console.print(
        f"[dim]Analyzing impact:[/] {description[:80]}..."
    )
    if files:
        console.print(
            f"[dim]Target files:[/] {', '.join(files)}"
        )

    try:
        report = analyzer.analyze(
            description=description,
            target_files=files,
        )
    except Exception as exc:
        console.print(f"[red]Impact analysis failed:[/] {exc}")
        raise typer.Exit(1) from exc

    # Display results — enhanced format matching /blast REPL command
    _display_impact_report(console, report, analyzer, RiskLevel, Panel, Table)


def _display_impact_report(
    con: Console,
    report: Any,
    analyzer: Any,
    risk_level_cls: Any,
    panel_cls: type,
    table_cls: type,
) -> None:
    """Render a detailed blast-radius report to the console.

    This is the shared display logic for the ``prism impact`` CLI command.
    The format mirrors ``_display_blast_report`` in the REPL module.

    Args:
        con: Rich console instance for output.
        report: The :class:`ImpactReport` to display.
        analyzer: The :class:`BlastRadiusAnalyzer` that produced the report.
        risk_level_cls: The :class:`RiskLevel` constants class.
        panel_cls: Rich Panel class (lazy-imported by caller).
        table_cls: Rich Table class (lazy-imported by caller).
    """
    complexity_effort: dict[str, str] = {
        "trivial": "<1 hour",
        "simple": "1-2 hours",
        "moderate": "2-4 hours",
        "complex": "4-8 hours",
    }
    approach_map: dict[str, str] = {
        "complex": "incremental (test-first on critical areas)",
        "moderate": "incremental (test-first on critical areas)",
        "simple": "direct",
        "trivial": "direct",
    }

    # --- Header panel ---
    if report.risk_score >= 70:
        score_style = "red bold"
        border_style = "red"
    elif report.risk_score >= 40:
        score_style = "yellow"
        border_style = "yellow"
    else:
        score_style = "green"
        border_style = "green"

    header_lines = [
        f"[bold]Change:[/bold] {report.description}",
        (
            f"[bold]Risk Score:[/bold] "
            f"[{score_style}]{report.risk_score}/100[/{score_style}]"
        ),
        f"[bold]Complexity:[/bold] {report.estimated_complexity}",
        f"[bold]Files Affected:[/bold] {report.file_count}",
    ]
    con.print(panel_cls(
        "\n".join(header_lines),
        title="[bold]Blast Radius Report[/bold]",
        border_style=border_style,
    ))

    # Group files by risk level
    high_files = [
        af for af in report.affected_files
        if af.risk_level == risk_level_cls.HIGH
    ]
    medium_files = [
        af for af in report.affected_files
        if af.risk_level == risk_level_cls.MEDIUM
    ]
    low_files = [
        af for af in report.affected_files
        if af.risk_level == risk_level_cls.LOW
    ]

    # --- Critical (HIGH) areas ---
    if high_files:
        table = table_cls(
            title="[red bold]Critical Areas (HIGH risk)[/]",
            border_style="red",
        )
        table.add_column("File", style="cyan")
        table.add_column("Tests", justify="center")
        table.add_column("Callers", justify="center")
        table.add_column("Functions")
        table.add_column("Reason")
        for af in high_files:
            tested = "[green]yes[/]" if af.has_tests else "[red]no[/]"
            funcs = ", ".join(af.affected_functions[:5]) or "-"
            if len(af.affected_functions) > 5:
                funcs += f" (+{len(af.affected_functions) - 5})"
            table.add_row(
                af.path,
                tested,
                str(len(af.affected_functions)),
                funcs,
                af.reason[:50],
            )
        con.print(table)

    # --- Medium areas ---
    if medium_files:
        table = table_cls(
            title="[yellow]Medium Risk Areas[/]",
            border_style="yellow",
        )
        table.add_column("File", style="cyan")
        table.add_column("Tests", justify="center")
        table.add_column("Depth", justify="center")
        table.add_column("Reason")
        for af in medium_files:
            tested = "[green]yes[/]" if af.has_tests else "[red]no[/]"
            table.add_row(
                af.path, tested, str(af.depth), af.reason[:50],
            )
        con.print(table)

    # --- Low areas ---
    if low_files:
        table = table_cls(
            title="[green]Low Risk Areas[/]",
            border_style="green",
        )
        table.add_column("File", style="cyan")
        table.add_column("Tests", justify="center")
        table.add_column("Depth", justify="center")
        table.add_column("Reason")
        for af in low_files:
            tested = "[green]yes[/]" if af.has_tests else "[red]no[/]"
            table.add_row(
                af.path, tested, str(af.depth), af.reason[:50],
            )
        con.print(table)

    # --- Test recommendations ---
    if report.recommended_test_order:
        test_paths = " ".join(report.recommended_test_order)
        con.print(panel_cls(
            f"[bold]Run before changes:[/bold]\n  pytest {test_paths}",
            title="[bold]Test Recommendations[/bold]",
            border_style="blue",
        ))

    # --- Missing tests ---
    if report.missing_tests:
        critical_missing = {
            af.path for af in high_files if not af.has_tests
        }
        missing_lines: list[str] = []
        for mt in report.missing_tests:
            if mt in critical_missing:
                missing_lines.append(
                    f"  [red bold]HIGH PRIORITY[/] {mt}"
                )
            else:
                missing_lines.append(f"  [dim]normal[/dim]       {mt}")
        con.print(panel_cls(
            "\n".join(missing_lines),
            title=(
                f"[bold yellow]Missing Tests "
                f"({len(report.missing_tests)} file(s))[/]"
            ),
            border_style="yellow",
        ))

    # --- Execution order ---
    if report.execution_order:
        order_lines = [
            f"  {idx}. {path}"
            for idx, path in enumerate(report.execution_order, 1)
        ]
        con.print(panel_cls(
            "\n".join(order_lines),
            title="[bold]Recommended Execution Order[/bold]",
            border_style="dim",
        ))

    # --- Effort estimate ---
    effort = complexity_effort.get(
        report.estimated_complexity, "unknown",
    )
    approach = approach_map.get(
        report.estimated_complexity, "direct",
    )
    con.print(panel_cls(
        (
            f"[bold]Complexity:[/bold] {report.estimated_complexity}\n"
            f"[bold]Estimated effort:[/bold] {effort}\n"
            f"[bold]Recommended approach:[/bold] {approach}"
        ),
        title="[bold]Effort Estimate[/bold]",
        border_style="dim",
    ))

    # --- Report path ---
    report_path = analyzer.last_report_path
    if report_path:
        con.print(
            f"\n[dim]Report saved to {report_path}[/dim]"
        )


@app.command("test-gaps")
def test_gaps(
    critical: bool = typer.Option(
        False, "--critical",
        help="Show only critical gaps.",
    ),
    fix: bool = typer.Option(
        False, "--fix",
        help="Auto-generate top 5 test files.",
    ),
    ci: bool = typer.Option(
        False, "--ci",
        help="Exit non-zero if critical gaps found.",
    ),
    module: str | None = typer.Option(
        None, "--module", "-m",
        help="Analyze specific module only.",
    ),
    root: Path | None = typer.Option(
        None, "--root", "-r",
        help="Project root directory.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Analyze test coverage gaps with risk-based prioritization."""
    from rich.panel import Panel
    from rich.table import Table

    from prism.intelligence.test_gaps import GapRisk, TestGapHunter

    project_root = root or Path.cwd()

    try:
        hunter = TestGapHunter(project_root=project_root)
    except Exception as exc:
        console.print(f"[red]Error initializing test gap hunter:[/] {exc}")
        raise typer.Exit(1) from exc

    try:
        report = hunter.analyze_module(module) if module else hunter.analyze()
    except Exception as exc:
        console.print(f"[red]Test gap analysis failed:[/] {exc}")
        raise typer.Exit(1) from exc

    # Summary panel
    summary_lines = [
        f"Total functions: {report.total_functions}",
        f"Tested: {report.tested_functions}",
        f"Untested: {report.untested_functions}",
        f"Coverage: {report.coverage_percent:.1f}%",
        f"Critical gaps: {report.critical_count}",
        f"High gaps: {report.high_count}",
    ]
    if module:
        summary_lines.insert(0, f"Module: {module}")
    console.print(Panel(
        "\n".join(summary_lines),
        title="[bold]Test Gap Analysis[/bold]",
        border_style="blue",
    ))

    # Filter gaps
    gaps = report.gaps
    if critical:
        gaps = [g for g in gaps if g.risk_level == GapRisk.CRITICAL]
        console.print("[dim]Filtered to critical risk only.[/dim]")

    # Display gaps table
    if gaps:
        table = Table(title="Test Gaps")
        table.add_column("Function", style="cyan")
        table.add_column("File")
        table.add_column("Line", justify="right")
        table.add_column("Risk", justify="center")
        table.add_column("Scenarios", justify="right")
        table.add_column("Effort")

        risk_colors = {
            "critical": "red bold",
            "high": "red",
            "medium": "yellow",
            "low": "green",
        }

        for gap in gaps[:30]:
            color = risk_colors.get(gap.risk_level, "white")
            table.add_row(
                gap.function_name,
                gap.file_path,
                str(gap.line_number),
                f"[{color}]{gap.risk_level.upper()}[/{color}]",
                str(len(gap.scenarios)),
                gap.estimated_effort,
            )
        console.print(table)

        # Show scenarios indented under each gap
        for gap in gaps[:30]:
            if gap.scenarios:
                console.print(
                    f"  [dim]{gap.function_name}:[/dim]"
                )
                for scenario in gap.scenarios:
                    console.print(f"    [dim]- {scenario}[/dim]")

        if len(gaps) > 30:
            console.print(
                f"[dim]... and {len(gaps) - 30} more. "
                "Use --critical to filter.[/dim]"
            )
    else:
        console.print(
            "[green]No test gaps found at this risk level.[/]"
        )

    # Auto-generate test files
    if fix:
        # Prefer critical/high gaps for generation
        gen_gaps = [
            g for g in report.gaps
            if g.risk_level in (GapRisk.CRITICAL, GapRisk.HIGH)
        ]
        if not gen_gaps:
            gen_gaps = report.gaps
        gen_gaps = gen_gaps[:5]

        if gen_gaps:
            generated = hunter.generate_tests(gen_gaps, count=5)
            for test_path, content in generated.items():
                out = Path(test_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(content)
                console.print(
                    f"[green]Generated:[/] {test_path}"
                )
        else:
            console.print("[dim]No gaps to generate tests for.[/dim]")

    # CI exit code
    if ci and report.has_critical_gaps:
        console.print(
            "[red]CI check failed:[/] critical test gaps found."
        )
        raise typer.Exit(1)


@app.command("deps")
def deps(
    action: str = typer.Argument(
        "status",
        help="Action to perform: status, audit, or unused.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        "-r",
        help="Project root directory.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Dependency health monitoring — status, audit, or unused detection."""
    from rich.panel import Panel
    from rich.table import Table

    from prism.intelligence.deps import DependencyMonitor

    project_root = root or Path.cwd()

    try:
        monitor = DependencyMonitor(project_root=project_root)
    except ValueError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    action_lower = action.lower().strip()

    if action_lower == "status":
        _deps_status(monitor, console, Panel, Table)
    elif action_lower == "audit":
        _deps_audit(monitor, console, Table)
    elif action_lower == "unused":
        _deps_unused(monitor, console)
    else:
        console.print(
            f"[yellow]Unknown deps action:[/] {action}\n"
            "[dim]Valid actions: status, audit, unused[/dim]"
        )
        raise typer.Exit(1)


def _deps_status(
    monitor: Any,
    console: Console,
    panel_cls: type,
    table_cls: type,
) -> None:
    """Show dependency status table."""
    report = monitor.get_status()

    console.print(panel_cls(
        f"Total: {report.total_deps} | "
        f"Outdated: {report.outdated} | "
        f"Vulnerable: {report.vulnerable} | "
        f"Unused: {report.unused}",
        title="[bold]Dependency Health[/bold]",
        border_style="blue",
    ))

    if report.dependencies:
        table = table_cls(title="Dependencies")
        table.add_column("Package", style="cyan")
        table.add_column("Current")
        table.add_column("Latest")
        table.add_column("Ecosystem")
        table.add_column("Security")
        table.add_column("Risk")

        # Build a set of vulnerable package names
        vuln_packages = {v.package for v in report.vulnerabilities}

        for dep in report.dependencies[:50]:
            mig_color = {
                "trivial": "green",
                "simple": "yellow",
                "moderate": "yellow bold",
                "complex": "red",
            }.get(dep.migration_complexity.value, "white")

            security = (
                "[red]VULNERABLE[/red]"
                if dep.name in vuln_packages
                else "[green]OK[/green]"
            )

            table.add_row(
                dep.name,
                dep.current_version,
                dep.latest_version or "-",
                dep.ecosystem,
                security,
                f"[{mig_color}]"
                f"{dep.migration_complexity.value}"
                f"[/{mig_color}]",
            )
        console.print(table)

    if report.vulnerabilities:
        vuln_table = table_cls(
            title="Vulnerabilities",
            border_style="red",
        )
        vuln_table.add_column("Package", style="red bold")
        vuln_table.add_column("CVE")
        vuln_table.add_column("Severity")
        vuln_table.add_column("Fix Version")

        for v in report.vulnerabilities:
            vuln_table.add_row(
                v.package,
                v.cve_id,
                v.severity.value.upper(),
                v.fixed_version or "-",
            )
        console.print(vuln_table)

    if report.unused_deps:
        console.print(
            f"\n[yellow]Potentially unused "
            f"({len(report.unused_deps)}):[/]"
        )
        for name in report.unused_deps:
            console.print(f"  {name}")


def _deps_audit(
    monitor: Any,
    console: Console,
    table_cls: type,
) -> None:
    """Run security-only scan."""
    report = monitor.get_status()

    if not report.vulnerabilities:
        console.print("[green]No vulnerabilities found.[/green]")
        return

    vuln_table = table_cls(
        title="Security Audit Results",
        border_style="red",
    )
    vuln_table.add_column("Package", style="red bold")
    vuln_table.add_column("CVE")
    vuln_table.add_column("Severity")
    vuln_table.add_column("Current")
    vuln_table.add_column("Fix Version")

    from prism.intelligence.deps import VulnerabilitySeverity

    has_critical_or_high = False
    for v in report.vulnerabilities:
        vuln_table.add_row(
            v.package,
            v.cve_id,
            v.severity.value.upper(),
            v.current_version,
            v.fixed_version or "-",
        )
        if v.severity in (
            VulnerabilitySeverity.CRITICAL,
            VulnerabilitySeverity.HIGH,
        ):
            has_critical_or_high = True

    console.print(vuln_table)
    console.print(
        f"\n[bold]Total vulnerabilities:[/] {len(report.vulnerabilities)}"
    )

    if has_critical_or_high:
        console.print(
            "[red]CRITICAL/HIGH vulnerabilities found. "
            "Immediate action recommended.[/red]"
        )
        raise typer.Exit(1)


def _deps_unused(
    monitor: Any,
    console: Console,
) -> None:
    """Find unused dependencies."""
    report = monitor.get_status()

    if not report.unused_deps:
        console.print("[green]No unused dependencies detected.[/green]")
        return

    console.print(
        f"\n[yellow]Potentially unused dependencies "
        f"({len(report.unused_deps)}):[/yellow]\n"
    )
    for name in report.unused_deps:
        console.print(f"  [dim]-[/dim] {name}")

    console.print(
        "\n[dim]Note: These packages have zero import references "
        "in source files. Build tools and test runners are "
        "excluded from this check.[/dim]"
    )


@app.command("context")
def context_command(
    action: str = typer.Argument(
        "show",
        help="Action to perform: show or stats.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        "-r",
        help="Project root directory.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Display smart context budget allocation or efficiency statistics."""
    from rich.panel import Panel
    from rich.table import Table

    from prism.config.settings import load_settings
    from prism.intelligence.context_budget import SmartContextBudgetManager

    project_root = root or Path.cwd()

    try:
        settings = load_settings(project_root=project_root)
        settings.ensure_directories()
    except Exception as exc:
        console.print(f"[red]Settings error:[/] {exc}")
        raise typer.Exit(1) from exc

    if action == "stats":
        try:
            from prism.db.database import Database

            db = Database(settings.db_path)
            manager = SmartContextBudgetManager(
                project_root=settings.project_root,
                db=db,
            )
            stats = manager.get_efficiency_stats()

            table = Table(
                show_header=False, box=None,
                padding=(0, 2),
            )
            table.add_column("Key", style="bold")
            table.add_column("Value", justify="right")

            table.add_row(
                "Total records",
                str(stats.total_records),
            )
            table.add_row(
                "Avg tokens used",
                f"{stats.avg_tokens_used:,.0f}",
            )
            table.add_row(
                "Avg efficiency",
                f"{stats.avg_efficiency_pct:.1f}%",
            )
            table.add_row(
                "Success rate",
                f"{stats.success_rate:.0%}",
            )
            table.add_row(
                "Est. tokens saved",
                f"{stats.total_tokens_saved:,}",
            )

            console.print(Panel(
                table,
                title="[bold]Context Efficiency Stats[/bold]",
                border_style="blue",
            ))
        except Exception as exc:
            console.print(f"[red]Stats error:[/] {exc}")
            raise typer.Exit(1) from exc

    else:
        # show (default)
        manager = SmartContextBudgetManager(
            project_root=settings.project_root,
        )

        # Collect Python files relative to project root
        try:
            py_files = [
                str(p.relative_to(settings.project_root))
                for p in settings.project_root.rglob("*.py")
                if not any(
                    part.startswith(".")
                    for part in p.relative_to(settings.project_root).parts
                )
            ][:50]  # Limit to 50 for display
        except OSError:
            py_files = []

        allocation = manager.allocate(
            task_description="project overview",
            available_files=py_files,
        )

        display = SmartContextBudgetManager.generate_context_display(
            allocation,
        )
        console.print(Panel(
            display,
            title="[bold]Smart Context Budget[/bold]",
            border_style="blue",
        ))


@app.command("debate")
def debate_command(
    question: str = typer.Argument(
        ...,
        help="The question or decision to debate across multiple models.",
    ),
    quick: bool = typer.Option(
        False, "--quick",
        help="Skip the critique round (Round 2) for faster results.",
    ),
    models: str | None = typer.Option(
        None, "--models",
        help="Comma-separated list of models to use for the debate.",
    ),
) -> None:
    """Run a structured multi-model debate on a question."""
    from rich.markdown import Markdown
    from rich.panel import Panel

    from prism.intelligence.debate import (
        DebateConfig,
        debate,
    )

    try:
        cfg = DebateConfig(quick_mode=quick)

        if models:
            model_list = [m.strip() for m in models.split(",") if m.strip()]
            if model_list:
                cfg.round1_models = model_list

        console.print("[dim]Starting multi-model debate...[/dim]")
        if quick:
            console.print("[dim]Quick mode: skipping critique round.[/dim]")

        result = debate(question=question, config=cfg)

        # Display each round
        for rnd in result.rounds:
            round_labels = {
                "position": ("Round 1 — Independent Positions", "blue"),
                "critique": ("Round 2 — Critiques", "yellow"),
                "synthesis": ("Round 3 — Synthesis", "green"),
            }
            label, color = round_labels.get(
                rnd.round_type,
                (f"Round {rnd.round_number}", "white"),
            )
            console.print(f"\n[bold]{label}[/bold]")
            for model_name, response in rnd.positions.items():
                console.print(Panel(
                    Markdown(response) if response else "[dim](no response)[/dim]",
                    title=model_name,
                    border_style=color,
                ))

        # Synthesis summary
        console.print("\n[bold]Synthesis Summary[/bold]")
        summary_parts: list[str] = []
        if result.consensus:
            summary_parts.append(f"**Consensus**: {result.consensus}")
        if result.disagreements:
            summary_parts.append(f"**Disagreements**: {result.disagreements}")
        if result.tradeoffs:
            summary_parts.append(f"**Tradeoffs**: {result.tradeoffs}")
        if result.recommendation:
            summary_parts.append(f"**Recommendation**: {result.recommendation}")
        summary_parts.append(f"**Confidence**: {result.confidence:.0%}")

        if result.blind_spots:
            summary_parts.append("**Blind Spots**:")
            for m, spot in result.blind_spots.items():
                summary_parts.append(f"  - {m}: {spot}")

        console.print(Panel(
            Markdown("\n\n".join(summary_parts)),
            border_style="green",
        ))

        console.print(
            f"\n[dim]Total cost: ${result.total_cost:.4f}[/dim]"
        )

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        console.print(f"[red]Debate error:[/] {exc}")


@app.command("why")
def why_command(
    target: str = typer.Argument(
        ...,
        help="Target to investigate: file:line, function_name, or class_name.",
    ),
    module: str | None = typer.Option(
        None, "--module", "-m",
        help="Restrict search to a specific module.",
    ),
    root: Path | None = typer.Option(
        None, "--root", "-r",
        help="Project root directory.",
        exists=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Investigate the history and evolution of code (temporal archaeology)."""
    from rich.panel import Panel
    from rich.table import Table

    from prism.intelligence.archaeologist import (
        investigate,
    )

    project_root = root or Path.cwd()

    # If module is specified, prefix the target
    effective_target = target
    if module and ":" not in target and "/" not in target and not target.endswith(".py"):
        effective_target = f"src/prism/{module}/{target}"

    try:
        report = investigate(
            target=effective_target,
            project_root=project_root,
        )

        # Summary panel
        stability_color = (
            "green" if report.stability_score >= 0.7
            else "yellow" if report.stability_score >= 0.4
            else "red"
        )

        console.print(Panel(
            f"Target: {report.target}\n"
            f"Primary Author: {report.primary_author}\n"
            f"Total Commits: {len(report.timeline)}\n"
            f"Stability: [{stability_color}]"
            f"{report.stability_score:.0%}"
            f"[/{stability_color}]",
            title="[bold]Code Archaeology[/bold]",
            border_style="blue",
        ))

        # Timeline table
        if report.timeline:
            table = Table(title="Timeline")
            table.add_column("Hash", style="cyan", width=9)
            table.add_column("Date")
            table.add_column("Author")
            table.add_column("Subject")

            for commit in report.timeline[:20]:
                table.add_row(
                    commit.hash[:7],
                    commit.date[:10],
                    commit.author,
                    commit.subject[:50],
                )
            if len(report.timeline) > 20:
                console.print(
                    f"[dim]... and {len(report.timeline) - 20} more[/dim]"
                )
            console.print(table)

        # Author distribution
        if report.author_distribution:
            auth_table = Table(title="Contributors")
            auth_table.add_column("Author", style="cyan")
            auth_table.add_column("Commits", justify="right")

            for author, count in sorted(
                report.author_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                auth_table.add_row(author, str(count))
            console.print(auth_table)

        # Co-evolution
        if report.co_evolution:
            co_table = Table(title="Co-evolving Files")
            co_table.add_column("File", style="cyan")
            co_table.add_column("Co-change Rate", justify="right")

            for co_file, pct in report.co_evolution[:10]:
                co_table.add_row(co_file, f"{pct:.0%}")
            console.print(co_table)

        # Risks
        if report.risks:
            risk_text = "\n".join(f"- {r}" for r in report.risks)
            console.print(Panel(
                risk_text,
                title="Risks",
                border_style="red" if report.stability_score < 0.5 else "yellow",
            ))

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        console.print(f"[red]Archaeology error:[/] {exc}")


def main() -> None:
    """Entry point for the Prism CLI."""
    app()

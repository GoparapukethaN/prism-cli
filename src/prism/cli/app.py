"""Prism CLI application — Typer command definitions and entry point."""

from __future__ import annotations

from pathlib import Path

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
) -> None:
    """Initialize and start the interactive REPL."""
    from prism.config.settings import load_settings

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

    # Load settings
    settings = load_settings(
        project_root=root,
        config_overrides=overrides,
    )
    settings.ensure_directories()

    if dry_run:
        console.print("[yellow]Dry-run mode:[/] routing decisions shown but not executed.")

    if offline:
        console.print("[yellow]Offline mode:[/] cloud providers disabled, Ollama only.")

    # Print welcome banner
    _print_banner(settings)

    # Start REPL
    from prism.cli.repl import run_repl

    run_repl(settings=settings, console=console, dry_run=dry_run, offline=offline)


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


def main() -> None:
    """Entry point for the Prism CLI."""
    app()

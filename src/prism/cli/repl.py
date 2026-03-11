"""Interactive REPL for Prism — the main user interface.

Provides 30 fully-functional slash commands covering cost management,
model comparison, conversation branching, code sandbox execution,
background tasks, privacy mode, plugins, dependency health, architecture
mapping, debug memory, code archaeology, and more.

All heavy modules are lazy-imported inside command handlers to keep
startup time fast.
"""

from __future__ import annotations

import asyncio
from datetime import UTC
from typing import TYPE_CHECKING, Any

import structlog
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

    from prism.config.settings import Settings

logger = structlog.get_logger(__name__)


# ======================================================================
# Slash command metadata — used by /help
# ======================================================================

COMMAND_CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "General": [
        ("/help", "Show all commands organized by category"),
        ("/quit", "Exit Prism"),
        ("/clear", "Clear conversation history"),
        ("/status", "Check provider status"),
    ],
    "Model & Routing": [
        ("/model [name]", "Show or set the current model"),
        ("/compare <prompt|config|history>",
         "Compare models side-by-side"),
        ("/debate <prompt>", "Multi-model debate on a question"),
    ],
    "Context & Files": [
        ("/add <files>", "Add files to context"),
        ("/drop [files]",
         "Remove files from context / list active"),
        ("/image <path> [prompt]",
         "Attach image for vision analysis"),
        ("/compact", "Compress conversation history"),
        ("/branch <name>",
         "Create / list / switch / merge / delete / save"),
        ("/budget [status|set] [tokens]",
         "Context budget management"),
    ],
    "Cost & Budget": [
        ("/cost", "Show cost dashboard"),
        ("/forecast", "Monthly cost forecast and model drivers"),
        ("/cache [stats|clear|on|off]", "Cache management"),
    ],
    "Tools & Execution": [
        ("/sandbox <code|on|off|status>",
         "Sandbox execution with language detection"),
        ("/tasks [list|cancel] [id]",
         "Background task management"),
        ("/web on|off", "Toggle web browsing"),
    ],
    "Code Intelligence": [
        ("/blast <file>", "Blast radius analysis for a file"),
        ("/gaps [file|dir]", "Test gap analysis"),
        ("/deps", "Dependency health report"),
        ("/arch [map|drift]", "Architecture map / drift detection"),
        ("/debug-memory [search] [query]", "Debug memory search"),
        ("/history <file>", "Code archaeology — file evolution"),
    ],
    "Planning": [
        ("/architect <goal>",
         "Plan and execute complex multi-step tasks"),
        ("/architect list", "List all plans"),
        ("/architect resume [id]", "Resume a paused plan"),
        ("/architect status [id]", "Show plan status"),
        ("/architect rollback [id]", "Rollback a plan"),
    ],
    "Infrastructure": [
        ("/undo [N|all]",
         "Undo last N changes with diff"),
        ("/rollback [list|diff|restore] [N|hash]",
         "Session timeline and rollback"),
        ("/privacy [on|off|status]", "Privacy mode (local-only)"),
        ("/plugins [list|install|remove] [name]",
         "Plugin management"),
        ("/workspace [list|switch|add|remove] [name]",
         "Multi-project workspace"),
        ("/offline [status|queue]", "Offline mode info"),
    ],
}


# ======================================================================
# Session state — mutable container shared across handlers
# ======================================================================

class _SessionState:
    """Mutable session state passed to all command handlers.

    Attributes:
        active_files: Files currently in context.
        conversation: Conversation history messages.
        pinned_model: User-pinned model override, or ``None``.
        web_enabled: Whether web browsing is enabled.
        session_id: Unique session identifier.
        cache_enabled: Whether response caching is active.
        sandbox_enabled: Whether sandbox execution is active.
        sandbox_type: Forced sandbox type (``None`` = auto).
    """

    def __init__(
        self,
        pinned_model: str | None,
        cache_enabled: bool = True,
    ) -> None:
        self.active_files: list[str] = []
        self.conversation: list[dict[str, str]] = []
        self.pinned_model: str | None = pinned_model
        self.web_enabled: bool = False
        self.session_id: str = ""
        self.cache_enabled: bool = cache_enabled
        self.sandbox_enabled: bool = True
        self.sandbox_type: str | None = None


# ======================================================================
# Public entry point
# ======================================================================

def run_repl(
    settings: Settings,
    console: Console,
    dry_run: bool = False,
    offline: bool = False,
    no_cache: bool = False,
) -> None:
    """Run the interactive REPL loop.

    Args:
        settings: Application settings.
        console: Rich console for output.
        dry_run: If True, show routing decisions without executing.
        offline: If True, only use local models.
        no_cache: If True, disable response caching.
    """
    history_path = settings.sessions_dir / "repl_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    bindings = KeyBindings()

    @bindings.add("c-c")
    def _handle_ctrl_c(event: object) -> None:
        """Handle Ctrl+C — cancel current input."""
        event.app.current_buffer.reset()  # type: ignore[union-attr]

    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        key_bindings=bindings,
        multiline=False,
        enable_history_search=True,
    )

    state = _SessionState(
        pinned_model=settings.config.pinned_model,
        cache_enabled=not no_cache,
    )

    # Generate a stable session id
    from uuid import uuid4

    state.session_id = str(uuid4())[:12]

    console.print(
        "[dim]Ready. Type your request or /help for commands.[/dim]\n"
    )

    while True:
        try:
            try:
                user_input = session.prompt("prism> ").strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            # Slash command dispatch
            if user_input.startswith("/"):
                result = _dispatch_command(
                    user_input,
                    console=console,
                    settings=settings,
                    state=state,
                    dry_run=dry_run,
                    offline=offline,
                )
                if result == "quit":
                    break
                continue

            # Regular prompt — process through router
            _process_prompt(
                prompt=user_input,
                console=console,
                settings=settings,
                state=state,
                dry_run=dry_run,
                offline=offline,
            )

        except Exception:
            logger.exception("repl_error")
            console.print(
                "[red]An unexpected error occurred. "
                "Check logs for details.[/]"
            )


# ======================================================================
# Command dispatcher
# ======================================================================

def _dispatch_command(
    command: str,
    console: Console,
    settings: Settings,
    state: _SessionState,
    dry_run: bool,
    offline: bool,
) -> str:
    """Parse and dispatch a slash command to its handler.

    Args:
        command: Full user input string starting with ``/``.
        console: Rich console.
        settings: App settings.
        state: Mutable session state.
        dry_run: Dry-run mode flag.
        offline: Offline mode flag.

    Returns:
        ``"quit"`` to exit the REPL, ``"continue"`` otherwise.
    """
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    dispatch: dict[str, Any] = {
        "/help": _cmd_help,
        "/quit": _cmd_quit,
        "/exit": _cmd_quit,
        "/q": _cmd_quit,
        "/cost": _cmd_cost,
        "/model": _cmd_model,
        "/add": _cmd_add,
        "/drop": _cmd_drop,
        "/compact": _cmd_compact,
        "/undo": _cmd_undo,
        "/web": _cmd_web,
        "/status": _cmd_status,
        "/clear": _cmd_clear,
        "/cache": _cmd_cache,
        "/compare": _cmd_compare,
        "/image": _cmd_image,
        "/branch": _cmd_branch,
        "/rollback": _cmd_rollback,
        "/sandbox": _cmd_sandbox,
        "/tasks": _cmd_tasks,
        "/privacy": _cmd_privacy,
        "/plugins": _cmd_plugins,
        "/forecast": _cmd_forecast,
        "/workspace": _cmd_workspace,
        "/offline": _cmd_offline,
        "/debate": _cmd_debate,
        "/blast": _cmd_blast,
        "/gaps": _cmd_gaps,
        "/deps": _cmd_deps,
        "/arch": _cmd_arch,
        "/debug-memory": _cmd_debug_memory,
        "/history": _cmd_history,
        "/budget": _cmd_budget,
        "/architect": _cmd_architect,
    }

    handler = dispatch.get(cmd)
    if handler is None:
        console.print(
            f"[yellow]Unknown command:[/] {cmd}. "
            "Type /help for available commands."
        )
        return "continue"

    return handler(
        args=args,
        console=console,
        settings=settings,
        state=state,
        dry_run=dry_run,
        offline=offline,
    )


# ======================================================================
# Individual command handlers
# ======================================================================

def _cmd_help(
    args: str,
    console: Console,
    **_: Any,
) -> str:
    """Display all commands organized by category."""
    console.print()
    for category, commands in COMMAND_CATEGORIES.items():
        table = Table(
            title=category,
            show_header=False,
            box=None,
            padding=(0, 2),
            title_style="bold cyan",
        )
        table.add_column("Command", style="cyan", min_width=40)
        table.add_column("Description")
        for cmd_name, desc in commands:
            table.add_row(cmd_name, desc)
        console.print(table)
        console.print()
    return "continue"


def _cmd_quit(
    args: str,
    console: Console,
    **_: Any,
) -> str:
    """Exit the REPL."""
    console.print("[dim]Goodbye![/dim]")
    return "quit"


def _cmd_cost(
    args: str,
    console: Console,
    settings: Settings,
    state: _SessionState,
    **_: Any,
) -> str:
    """Show the cost dashboard."""
    try:
        from prism.cost.dashboard import render_cost_dashboard
        from prism.cost.tracker import CostTracker
        from prism.db.database import Database

        db = Database(settings.db_path)
        tracker = CostTracker(db=db, settings=settings)
        render_cost_dashboard(
            tracker,
            session_id=state.session_id,
            console=console,
        )
    except Exception as exc:
        logger.debug("cost_dashboard_error", error=str(exc))
        console.print(
            Panel(
                "[dim]No cost data yet. Costs will appear "
                "after your first API call.[/dim]",
                title="[bold]Cost Dashboard[/bold]",
                border_style="blue",
            )
        )
    return "continue"


def _cmd_model(
    args: str,
    console: Console,
    state: _SessionState,
    **_: Any,
) -> str:
    """Show or set the current model."""
    if not args:
        current = state.pinned_model or "auto (routing)"
        console.print(f"[bold]Current model:[/] {current}")
        return "continue"

    if args.lower() == "auto":
        state.pinned_model = None
        console.print("[green]Model reset to auto-routing.[/]")
    else:
        state.pinned_model = args.strip()
        console.print(f"[green]Model set to:[/] {state.pinned_model}")
    return "continue"


def _cmd_add(
    args: str,
    console: Console,
    state: _SessionState,
    **_: Any,
) -> str:
    """Add files to the context."""
    if not args:
        console.print("[yellow]Usage:[/] /add <file1> [file2] ...")
        return "continue"

    new_files = args.split()
    for f in new_files:
        if f not in state.active_files:
            state.active_files.append(f)
            console.print(f"  [green]+[/] {f}")
        else:
            console.print(f"  [dim]Already added:[/dim] {f}")
    return "continue"


def _cmd_drop(
    args: str,
    console: Console,
    state: _SessionState,
    **_: Any,
) -> str:
    """Remove files from context or list active files."""
    if not args:
        if state.active_files:
            console.print("[bold]Active files:[/bold]")
            for f in state.active_files:
                console.print(f"  {f}")
        else:
            console.print("[dim]No active files.[/dim]")
        return "continue"

    drop_files = args.split()
    for f in drop_files:
        if f in state.active_files:
            state.active_files.remove(f)
            console.print(f"  [red]-[/] {f}")
        else:
            console.print(f"  [dim]Not in context:[/dim] {f}")
    return "continue"


def _cmd_compact(
    args: str,
    console: Console,
    state: _SessionState,
    **_: Any,
) -> str:
    """Compress conversation history via summarizer."""
    try:
        from prism.context.summarizer import summarize

        if len(state.conversation) < 4:
            console.print(
                "[dim]Conversation too short to compact.[/dim]"
            )
            return "continue"

        summary = summarize(
            state.conversation,
            max_tokens=4000,
        )

        old_count = len(state.conversation)
        state.conversation = [
            {"role": "system", "content": summary},
        ]

        console.print(
            f"[green]Compacted {old_count} messages into summary "
            f"({len(summary)} chars).[/]"
        )
    except Exception as exc:
        logger.debug("compact_error", error=str(exc))
        console.print(
            f"[red]Compact failed:[/] {exc}"
        )
    return "continue"


def _cmd_undo(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Undo file changes via RollbackManager.

    Supports:
        /undo       -- undo the last change, show diff
        /undo N     -- undo the last N changes with summary
        /undo all   -- undo every change this session
    """
    try:
        from prism.git.history import RollbackManager

        manager = RollbackManager(settings.project_root)
        manager.start_session()

        arg = args.strip().lower()

        # Determine undo count
        if arg == "all":
            count = manager.change_count
            if count == 0:
                console.print(
                    "[dim]No changes to undo.[/dim]"
                )
                return "continue"
        elif arg and arg.isdigit():
            count = int(arg)
            if count < 1:
                console.print(
                    "[yellow]Count must be at least 1.[/]"
                )
                return "continue"
        elif arg:
            console.print(
                "[yellow]Usage:[/] "
                "/undo [N | all]"
            )
            return "continue"
        else:
            count = 1

        # Capture change records before undoing for display
        timeline = manager.get_timeline()
        changes_to_undo = timeline.changes[-count:]

        reverted = manager.undo(count=count)

        if not reverted:
            console.print("[dim]Nothing to undo.[/dim]")
            return "continue"

        # Show summary table for multi-undo
        if len(reverted) > 1:
            table = Table(
                title=f"Undone {len(reverted)} Changes",
            )
            table.add_column(
                "Hash", style="cyan", width=10,
            )
            table.add_column("Files Affected")
            table.add_column("+/-", justify="right")

            for change in reversed(changes_to_undo):
                files_str = ", ".join(
                    change.files_changed[:3],
                )
                if len(change.files_changed) > 3:
                    extra = len(change.files_changed) - 3
                    files_str += f" (+{extra} more)"
                table.add_row(
                    change.short_hash,
                    files_str or "(unknown)",
                    f"+{change.insertions}"
                    f"/-{change.deletions}",
                )
            console.print(table)
        else:
            # Single undo -- show affected files
            console.print(
                f"[green]Undone 1 change "
                f"(commit {reverted[0][:8]}).[/]"
            )
            if changes_to_undo:
                ch = changes_to_undo[0]
                for fname in ch.files_changed:
                    console.print(f"  [dim]{fname}[/dim]")

        # Show diff of last undone change
        try:
            if changes_to_undo:
                last = changes_to_undo[-1]
                diff_text = manager.get_diff(
                    last.index,
                )
                if diff_text.strip():
                    console.print(
                        Panel(
                            diff_text,
                            title="[bold]Diff (last undone)[/bold]",
                            border_style="yellow",
                        )
                    )
        except (ValueError, RuntimeError):
            pass  # Diff display is best-effort

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except RuntimeError as exc:
        console.print(f"[red]Undo failed:[/] {exc}")
    except Exception as exc:
        logger.debug("undo_error", error=str(exc))
        console.print(f"[red]Undo failed:[/] {exc}")
    return "continue"


def _cmd_web(
    args: str,
    console: Console,
    state: _SessionState,
    **_: Any,
) -> str:
    """Toggle web browsing on/off."""
    lower = args.lower().strip()
    if lower == "on":
        state.web_enabled = True
        console.print("[green]Web browsing enabled.[/]")
    elif lower == "off":
        state.web_enabled = False
        console.print("[yellow]Web browsing disabled.[/]")
    else:
        status = "enabled" if state.web_enabled else "disabled"
        console.print(
            f"[dim]Web browsing is currently {status}.[/]\n"
            "[dim]Usage:[/] /web on|off"
        )
    return "continue"


def _cmd_status(
    args: str,
    console: Console,
    **_: Any,
) -> str:
    """Check provider status."""
    try:
        from prism.cli.app import status
        status()
    except Exception as exc:
        logger.debug("status_error", error=str(exc))
        console.print(f"[red]Status check failed:[/] {exc}")
    return "continue"


def _cmd_clear(
    args: str,
    console: Console,
    state: _SessionState,
    **_: Any,
) -> str:
    """Clear conversation history."""
    count = len(state.conversation)
    state.conversation.clear()
    console.print(
        f"[dim]Conversation cleared ({count} messages removed).[/dim]"
    )
    return "continue"
def _parse_duration_to_hours(duration: str) -> float | None:
    """Parse a human-readable duration into hours.

    Supports ``"30m"``, ``"24h"``, ``"2d"``, ``"1w"``.

    Args:
        duration: Duration string with a numeric value
            followed by a unit character.

    Returns:
        Hours as a float, or ``None`` if parsing fails.
    """
    import re

    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*([mhdw])", duration.strip(),
    )
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    multipliers: dict[str, float] = {
        "m": 1.0 / 60,
        "h": 1.0,
        "d": 24.0,
        "w": 168.0,
    }
    return value * multipliers[unit]


def _cmd_cache(
    args: str,
    console: Console,
    settings: Settings,
    state: _SessionState,
    **_: Any,
) -> str:
    """Cache management -- stats, clear, on/off.

    Subcommands:
        ``/cache`` or ``/cache stats`` -- show statistics.
        ``/cache clear`` -- remove all entries.
        ``/cache clear --older-than <dur>`` -- clear entries
        older than duration (``24h``, ``2d``, ``30m``).
        ``/cache off`` -- disable caching for this session.
        ``/cache on`` -- re-enable caching.
    """
    try:
        from prism.cache.response_cache import ResponseCache

        sub = args.strip()
        sub_lower = sub.lower()

        # --- on / off toggling ---
        if sub_lower == "off":
            state.cache_enabled = False
            console.print(
                "[yellow]Cache disabled for this session.[/]"
            )
            return "continue"

        if sub_lower == "on":
            state.cache_enabled = True
            console.print(
                "[green]Cache re-enabled for this session.[/]"
            )
            return "continue"

        cache = ResponseCache(
            cache_dir=settings.cache_dir,
            enabled=True,
        )

        # --- clear with optional --older-than ---
        if sub_lower.startswith("clear"):
            remainder = sub[5:].strip()
            if remainder.lower().startswith("--older-than"):
                dur_str = remainder[
                    len("--older-than"):
                ].strip()
                hours = _parse_duration_to_hours(dur_str)
                if hours is None:
                    console.print(
                        "[yellow]Invalid duration. "
                        "Use e.g. 30m, 24h, 2d, 1w[/]"
                    )
                    cache.close()
                    return "continue"
                max_h = max(int(hours), 1) if hours >= 1 else 1
                deleted = cache.clear(max_age_hours=max_h)
                console.print(
                    f"[green]Cleared {deleted} entries "
                    f"older than {dur_str}.[/]"
                )
            else:
                deleted = cache.clear()
                console.print(
                    f"[green]Cache cleared: "
                    f"{deleted} entries removed.[/]"
                )
            cache.close()
            return "continue"

        # --- stats (default) ---
        stats = cache.get_stats()

        table = Table(
            title="Response Cache",
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Entries", str(stats.total_entries))
        table.add_row("Hits", str(stats.total_hits))
        table.add_row("Misses", str(stats.total_misses))
        table.add_row(
            "Hit rate", f"{stats.hit_rate:.1%}",
        )
        table.add_row(
            "Tokens saved", f"{stats.tokens_saved:,}",
        )
        table.add_row(
            "Cost saved", f"${stats.cost_saved:.4f}",
        )

        size_kb = stats.cache_size_bytes / 1024
        if size_kb >= 1024:
            table.add_row(
                "DB size", f"{size_kb / 1024:.1f} MB",
            )
        else:
            table.add_row("DB size", f"{size_kb:.1f} KB")

        if stats.oldest_entry:
            table.add_row(
                "Oldest", stats.oldest_entry[:19],
            )
        if stats.newest_entry:
            table.add_row(
                "Newest", stats.newest_entry[:19],
            )

        enabled_label = (
            "[green]enabled[/]" if state.cache_enabled
            else "[yellow]disabled[/]"
        )
        table.add_row("Session", enabled_label)

        console.print(Panel(table, border_style="blue"))
        cache.close()

    except Exception as exc:
        logger.debug("cache_error", error=str(exc))
        console.print(f"[red]Cache error:[/] {exc}")
    return "continue"


def _cmd_compare(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Compare models side-by-side.

    Subcommands:
        ``/compare <prompt>`` -- run comparison (shows
        estimated cost first).
        ``/compare config`` -- show / change comparison models.
        ``/compare history`` -- show past sessions with winners.
    """
    if not args:
        console.print(
            "[yellow]Usage:[/] "
            "/compare <prompt|config|history>"
        )
        return "continue"

    sub_lower = args.strip().lower()

    try:
        from prism.cli.compare import (
            MODEL_DISPLAY_NAMES,
            ModelComparator,
        )
        from prism.llm.completion import CompletionEngine

        engine = CompletionEngine(settings=settings)
        comparator = ModelComparator(
            completion_engine=engine,
            console=console,
        )

        # --- /compare config ---
        if sub_lower == "config":
            comparator.display_config()
            console.print(
                "[dim]Change models with:[/] "
                "/model <name> then /compare <prompt>"
            )
            return "continue"

        # --- /compare history ---
        if sub_lower == "history":
            history = comparator.history
            if not history:
                console.print(
                    "[dim]No comparison history yet.[/dim]"
                )
                return "continue"

            table = Table(title="Comparison History")
            table.add_column("#", style="dim", width=4)
            table.add_column("Prompt", min_width=30)
            table.add_column(
                "Models", justify="right",
            )
            table.add_column(
                "Winner", style="green",
            )
            table.add_column(
                "Cost", justify="right",
            )
            table.add_column("Date")

            for i, sess in enumerate(history, 1):
                winner_name = (
                    sess.winner.display_name
                    if sess.winner else "-"
                )
                table.add_row(
                    str(i),
                    sess.prompt[:40],
                    str(len(sess.results)),
                    winner_name,
                    f"${sess.total_cost:.4f}",
                    sess.created_at[:19],
                )
            console.print(table)
            return "continue"

        # --- /compare <prompt> (default) ---
        # Show estimated cost before running
        from prism.cost.pricing import get_model_pricing

        models = comparator.models
        est_lines: list[str] = []
        total_est = 0.0
        for mdl in models:
            display = MODEL_DISPLAY_NAMES.get(mdl, mdl)
            try:
                pricing = get_model_pricing(mdl)
                # Rough estimate: ~500 input + ~500 output tokens
                est = (
                    500 * pricing.input_cost_per_token
                    + 500 * pricing.output_cost_per_token
                )
                total_est += est
                est_lines.append(
                    f"  {display}: ~${est:.4f}"
                )
            except (ValueError, KeyError):
                est_lines.append(
                    f"  {display}: (pricing unavailable)"
                )

        console.print(
            f"[dim]Estimated cost: ~${total_est:.4f}[/]"
        )
        for line in est_lines:
            console.print(f"[dim]{line}[/]")

        console.print(
            "[dim]Running comparison across models...[/]"
        )
        session = asyncio.get_event_loop().run_until_complete(
            comparator.compare(args)
        )

        comparator.display_results(session)
        comparator.display_cost_table(session)
        comparator.display_prompt_hint(session)

    except Exception as exc:
        logger.debug("compare_error", error=str(exc))
        console.print(f"[red]Comparison failed:[/] {exc}")
    return "continue"




def _cmd_image(
    args: str,
    console: Console,
    settings: Settings,
    state: _SessionState,
    **_: Any,
) -> str:
    """Attach image(s) for vision model analysis.

    Usage:
        ``/image <path> [prompt]`` -- single image with optional
        prompt text.
        ``/image img1.png img2.png "Compare these"`` -- multiple
        images with a quoted prompt at the end.

    Auto-routes to a vision-capable model if the current model
    does not support images.
    """
    if not args:
        console.print(
            "[yellow]Usage:[/] /image <path> [prompt]\n"
            "[dim]Example: /image screenshot.png "
            "What does this show?[/dim]"
        )
        return "continue"

    try:
        import shlex
        from pathlib import Path

        from prism.tools.vision import (
            ALL_VISION_MODELS,
            build_multimodal_messages,
            detect_terminal_image_support,
            display_image_preview,
            is_vision_model,
            process_image,
        )

        # Parse arguments: split on spaces but respect quotes
        try:
            tokens = shlex.split(args)
        except ValueError:
            tokens = args.split()

        # Separate image paths from prompt text
        image_paths: list[Path] = []
        prompt_parts: list[str] = []

        for token in tokens:
            p = Path(token).expanduser()
            if p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".gif",
                ".webp", ".bmp", ".tiff", ".tif",
            } or p.is_file():
                image_paths.append(p)
            else:
                prompt_parts.append(token)

        if not image_paths:
            console.print(
                "[yellow]No valid image paths found.[/]\n"
                "[dim]Supported: jpg, png, gif, webp, bmp, "
                "tiff[/dim]"
            )
            return "continue"

        # Validate all paths exist
        for img_path in image_paths:
            if not img_path.is_file():
                console.print(
                    f"[red]File not found:[/] {img_path}"
                )
                return "continue"

        prompt_text = (
            " ".join(prompt_parts)
            if prompt_parts
            else "Describe this image in detail."
        )

        # Process images
        attachments = []
        for img_path in image_paths:
            att = process_image(img_path)
            attachments.append(att)
            size_str = (
                f"{att.size_bytes / 1024:.1f} KB"
                if att.size_bytes < 1_048_576
                else f"{att.size_bytes / 1_048_576:.1f} MB"
            )
            dim_str = (
                f"{att.width}x{att.height}"
                if att.width > 0 else "unknown"
            )
            compressed = (
                " [yellow](compressed)[/]"
                if att.was_compressed else ""
            )
            console.print(
                f"  [green]+[/] {img_path.name} "
                f"({dim_str}, {size_str}){compressed}"
            )

        # Show inline preview if terminal supports it
        protocol = detect_terminal_image_support()
        if protocol:
            for img_path in image_paths:
                display_image_preview(img_path, protocol)

        # Determine vision model
        current_model = state.pinned_model or ""
        if current_model and is_vision_model(current_model):
            vision_model = current_model
        else:
            # Auto-select cheapest vision model
            vision_model = "gemini/gemini-2.0-flash"
            # Check if any vision model is available
            for candidate in ALL_VISION_MODELS:
                if "flash" in candidate.lower():
                    vision_model = candidate
                    break
            if current_model:
                console.print(
                    f"[yellow]Model {current_model} does not "
                    f"support vision. "
                    f"Using {vision_model}.[/]"
                )

        console.print(
            f"[dim]Vision model: {vision_model}[/]"
        )

        # Build multimodal message content
        provider = "openai"  # default format
        if "claude" in vision_model.lower():
            provider = "anthropic"
        elif "gemini" in vision_model.lower():
            provider = "google"

        content = build_multimodal_messages(
            text_prompt=prompt_text,
            images=attachments,
            provider=provider,
        )

        # Send through completion engine
        from prism.llm.completion import CompletionEngine

        engine = CompletionEngine(settings=settings)

        messages = list(state.conversation)
        messages.append({
            "role": "user",
            "content": content,
        })

        from rich.markdown import Markdown

        result = asyncio.get_event_loop().run_until_complete(
            engine.complete(
                messages=messages,
                model=vision_model,
            )
        )

        resp_content = getattr(result, "content", "")
        if resp_content:
            state.conversation.append({
                "role": "user",
                "content": f"[image: {', '.join(str(p) for p in image_paths)}] {prompt_text}",
            })
            state.conversation.append({
                "role": "assistant",
                "content": resp_content,
            })
            console.print(
                Panel(
                    Markdown(resp_content),
                    border_style="blue",
                )
            )

        cost = getattr(result, "cost_usd", 0.0)
        tokens_in = getattr(result, "input_tokens", 0)
        tokens_out = getattr(result, "output_tokens", 0)
        console.print(
            f"[dim]{vision_model} | "
            f"{tokens_in}+{tokens_out} tokens | "
            f"${cost:.4f}[/dim]\n"
        )

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("image_error", error=str(exc))
        console.print(f"[red]Image error:[/] {exc}")
    return "continue"


def _cmd_branch(
    args: str,
    console: Console,
    settings: Settings,
    state: _SessionState,
    **_: Any,
) -> str:
    """Conversation branching with full lifecycle.

    Supports:
        /branch <name>         -- create a new branch
        /branch list           -- list all branches
        /branch switch <name>  -- switch to a branch
        /branch merge <name>   -- merge branch into current
        /branch delete <name>  -- delete a branch
        /branch save           -- mark current branch persistent

    Max 20 branches per session.
    """
    max_branches = 20

    try:
        from prism.context.branching import BranchManager

        branches_dir = settings.prism_home / "branches"
        manager = BranchManager(branches_dir)

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        name = parts[1].strip() if len(parts) > 1 else ""

        if sub == "list":
            _branch_list(manager, console)

        elif sub == "switch":
            if not name:
                console.print(
                    "[yellow]Usage:[/] "
                    "/branch switch <name>"
                )
                return "continue"
            msgs = manager.switch_branch(
                name=name,
                current_messages=state.conversation,
            )
            state.conversation = list(msgs)

            # Restore active files from branch
            branch = manager.get_branch(name)
            state.active_files = list(
                branch.file_edits,
            )

            console.print(
                f"[green]Switched to branch '{name}' "
                f"({len(msgs)} messages).[/]"
            )

        elif sub == "merge":
            if not name:
                console.print(
                    "[yellow]Usage:[/] "
                    "/branch merge <name>"
                )
                return "continue"
            source = manager.get_branch(name)
            fork_idx = source.metadata.fork_point_index
            new_msg_count = max(
                0, len(source.messages) - fork_idx,
            )

            merged = manager.merge_branch(
                source_name=name,
                current_messages=state.conversation,
            )
            state.conversation = list(merged)

            console.print(
                f"[green]Merged '{name}' into "
                f"'{manager.active_branch}' "
                f"({new_msg_count} new messages).[/]"
            )

        elif sub == "delete":
            if not name:
                console.print(
                    "[yellow]Usage:[/] "
                    "/branch delete <name>"
                )
                return "continue"
            manager.delete_branch(name)
            console.print(
                f"[green]Branch '{name}' deleted.[/]"
            )

        elif sub == "save":
            # Mark the current branch as persistent
            active_name = manager.active_branch
            if active_name in (
                br.name for br in manager.list_branches()
            ):
                branch = manager.get_branch(active_name)
                branch.metadata.description = (
                    branch.metadata.description
                    or "(persistent)"
                )
                manager._save_branch(active_name)
                console.print(
                    f"[green]Branch '{active_name}' "
                    f"marked as persistent.[/]"
                )
            else:
                console.print(
                    "[yellow]No active branch to save.[/]"
                )

        elif sub in ("create", ""):
            # Any non-keyword argument is treated as
            # a branch name (create)
            branch_name = name if sub == "create" else sub
            if not branch_name or branch_name == "list":
                _branch_list(manager, console)
                return "continue"

            if manager.branch_count >= max_branches:
                console.print(
                    f"[red]Branch limit reached "
                    f"({max_branches}). "
                    f"Delete a branch first.[/]"
                )
                return "continue"

            meta = manager.create_branch(
                name=branch_name,
                current_messages=state.conversation,
            )
            console.print(
                f"[green]Branch '{meta.name}' created "
                f"(forked from '{meta.parent_branch}' "
                f"at message "
                f"{meta.fork_point_index}).[/]"
            )

        else:
            # Bare name treated as branch creation
            if manager.branch_count >= max_branches:
                console.print(
                    f"[red]Branch limit reached "
                    f"({max_branches}). "
                    f"Delete a branch first.[/]"
                )
                return "continue"

            meta = manager.create_branch(
                name=sub,
                current_messages=state.conversation,
            )
            console.print(
                f"[green]Branch '{meta.name}' created "
                f"(forked from '{meta.parent_branch}' "
                f"at message "
                f"{meta.fork_point_index}).[/]"
            )

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("branch_error", error=str(exc))
        console.print(f"[red]Branch error:[/] {exc}")
    return "continue"


def _branch_list(
    manager: Any,
    console: Console,
) -> None:
    """Display all conversation branches as a Rich table.

    Args:
        manager: A :class:`BranchManager` instance.
        console: Rich console for output.
    """
    branches = manager.list_branches()
    if not branches:
        console.print(
            "[dim]No branches. "
            "Use /branch <name> to create one.[/dim]"
        )
        return

    table = Table(title="Conversation Branches")
    table.add_column("Name", style="cyan")
    table.add_column("Active", justify="center")
    table.add_column("Messages", justify="right")
    table.add_column("Parent")
    table.add_column("Description")
    table.add_column("Created")

    active = manager.active_branch
    for b in branches:
        marker = (
            "[green]>[/]" if b.name == active else ""
        )
        table.add_row(
            b.name,
            marker,
            str(b.message_count),
            b.parent_branch,
            b.description[:30] if b.description else "",
            b.created_at[:19],
        )
    console.print(table)


def _cmd_rollback(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Rollback history management.

    Supports:
        /rollback           -- show session timeline
        /rollback list      -- show session timeline
        /rollback diff <N>  -- show coloured diff of Nth change
        /rollback restore <hash> -- restore to a commit hash
    """
    try:
        from prism.git.history import RollbackManager

        manager = RollbackManager(settings.project_root)
        manager.start_session()

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        extra = parts[1].strip() if len(parts) > 1 else ""

        if sub in ("list", ""):
            timeline = manager.get_timeline()
            if not timeline.changes:
                console.print(
                    "[dim]No changes recorded "
                    "this session.[/dim]"
                )
                return "continue"

            table = Table(title="Session Timeline")
            table.add_column(
                "#", style="dim", width=4,
            )
            table.add_column("Hash", style="cyan")
            table.add_column("Message")
            table.add_column(
                "Files", justify="right",
            )
            table.add_column(
                "+/-", justify="right",
            )
            table.add_column("File Names")
            table.add_column("Timestamp")

            for ch in timeline.changes:
                files_display = ", ".join(
                    ch.files_changed[:3],
                )
                if len(ch.files_changed) > 3:
                    remain = len(ch.files_changed) - 3
                    files_display += f" (+{remain})"
                table.add_row(
                    str(ch.index),
                    ch.short_hash,
                    ch.message[:50],
                    str(len(ch.files_changed)),
                    f"+{ch.insertions}"
                    f"/-{ch.deletions}",
                    files_display,
                    ch.timestamp[:19],
                )
            console.print(table)

        elif sub == "diff":
            if not extra or not extra.isdigit():
                console.print(
                    "[yellow]Usage:[/] "
                    "/rollback diff <change-number>"
                )
                return "continue"

            change_idx = int(extra)
            change = manager.get_change(change_idx)
            diff_text = manager.get_diff(change_idx)

            # Show header with change info
            console.print(
                f"\n[bold]Change #{change.index}[/bold] "
                f"[cyan]{change.short_hash}[/cyan] "
                f"-- {change.message}"
            )
            for fname in change.files_changed:
                console.print(f"  [dim]{fname}[/dim]")

            if diff_text.strip():
                console.print(
                    Panel(
                        diff_text,
                        title=(
                            f"[bold]Diff #{change_idx}"
                            f"[/bold]"
                        ),
                        border_style="yellow",
                    )
                )
            else:
                console.print(
                    "[dim]No diff available.[/dim]"
                )

        elif sub == "restore":
            if not extra:
                console.print(
                    "[yellow]Usage:[/] "
                    "/rollback restore <commit-hash>"
                )
                return "continue"

            # Show preview first
            preview = manager.get_restore_preview(extra)
            if preview.strip():
                console.print(
                    Panel(
                        preview,
                        title=(
                            "[bold]Restore Preview"
                            "[/bold]"
                        ),
                        border_style="cyan",
                    )
                )

            new_hash = manager.restore(extra)
            console.print(
                f"[green]Restored to {extra[:8]}. "
                f"New commit: {new_hash[:8]}.[/]"
            )

        else:
            console.print(
                "[yellow]Usage:[/] "
                "/rollback [list|diff|restore] "
                "[N|hash]"
            )

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except RuntimeError as exc:
        console.print(f"[red]Rollback error:[/] {exc}")
    except Exception as exc:
        logger.debug("rollback_error", error=str(exc))
        console.print(f"[red]Rollback error:[/] {exc}")
    return "continue"


def _cmd_sandbox(
    args: str,
    console: Console,
    state: _SessionState,
    **_: Any,
) -> str:
    """Code sandbox execution with language auto-detection.

    Supports:
        /sandbox <code>       -- execute code in sandbox
        /sandbox on           -- enable sandbox execution
        /sandbox off          -- disable sandbox execution
        /sandbox docker       -- force Docker sandbox type
        /sandbox subprocess   -- force subprocess sandbox type
        /sandbox status       -- show sandbox configuration
    """
    if not args:
        console.print(
            "[yellow]Usage:[/] "
            "/sandbox <code> | on | off | "
            "docker | subprocess | status\n"
            "[dim]Example: "
            "/sandbox print('hello')[/dim]"
        )
        return "continue"

    arg_lower = args.strip().lower()

    # --- Toggle commands ---
    if arg_lower == "on":
        state.sandbox_enabled = True
        console.print(
            "[green]Sandbox execution enabled.[/]"
        )
        return "continue"

    if arg_lower == "off":
        state.sandbox_enabled = False
        console.print(
            "[yellow]Sandbox execution disabled.[/]"
        )
        return "continue"

    if arg_lower == "docker":
        state.sandbox_type = "docker"
        console.print(
            "[green]Sandbox type forced to Docker.[/]"
        )
        return "continue"

    if arg_lower == "subprocess":
        state.sandbox_type = "subprocess"
        console.print(
            "[green]Sandbox type forced to "
            "subprocess.[/]"
        )
        return "continue"

    if arg_lower == "status":
        _sandbox_status(state, console)
        return "continue"

    # --- Execute code ---
    if not state.sandbox_enabled:
        console.print(
            "[yellow]Sandbox is disabled. "
            "Use /sandbox on to enable.[/]"
        )
        return "continue"

    try:
        from prism.tools.code_sandbox import CodeSandbox

        language = _detect_language(args)

        sandbox = CodeSandbox(
            timeout=30, enabled=True,
        )

        # Force sandbox type if user requested it
        if state.sandbox_type == "docker":
            sandbox._docker_available = True
        elif state.sandbox_type == "subprocess":
            sandbox._docker_available = False

        result = sandbox.execute(
            args, language=language,
        )

        style = (
            "green" if result.exit_code == 0 else "red"
        )
        header = (
            f"[{style}]Exit: {result.exit_code}"
            f"[/{style}] | "
            f"{result.execution_time_ms:.0f}ms | "
            f"{result.sandbox_type} | "
            f"{language}"
        )

        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(
                f"[stderr]\n{result.stderr}"
            )
        if result.timed_out:
            output_parts.append("[Timed out]")
        if result.memory_exceeded:
            output_parts.append(
                "[Memory limit exceeded]"
            )

        body = (
            "\n".join(output_parts)
            if output_parts
            else "(no output)"
        )
        console.print(
            Panel(body, title=header, border_style=style)
        )

    except RuntimeError as exc:
        console.print(f"[yellow]{exc}[/]")
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("sandbox_error", error=str(exc))
        console.print(f"[red]Sandbox error:[/] {exc}")
    return "continue"


def _detect_language(code: str) -> str:
    """Auto-detect programming language from code.

    Inspects the first non-blank line for common patterns.

    Args:
        code: Source code string.

    Returns:
        A language identifier (e.g. ``"python"``).
    """
    first_line = ""
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break

    lowered = first_line.lower()

    # Python indicators
    python_starts = (
        "import ", "from ", "def ", "class ",
        "for ", "while ", "if ", "print(",
        "async ", "with ", "try:", "raise ",
        "return ", "yield ", "lambda ",
    )
    if any(lowered.startswith(p) for p in python_starts):
        return "python"

    # JavaScript / TypeScript
    js_starts = (
        "const ", "let ", "var ", "function ",
        "console.", "export ", "import {",
        "require(", "async function",
    )
    if any(lowered.startswith(p) for p in js_starts):
        return "javascript"

    # Bash / shell
    bash_starts = (
        "#!/bin/bash", "#!/bin/sh", "echo ",
        "export ", "cd ", "ls ", "mkdir ",
        "rm ", "cp ", "mv ",
    )
    if any(lowered.startswith(p) for p in bash_starts):
        return "bash"

    # Ruby
    ruby_starts = (
        "puts ", "require ", "def ",
        "module ", "class ",
    )
    if (
        any(lowered.startswith(p) for p in ruby_starts)
        and "require " in lowered
        and "'" in lowered
    ):
        return "ruby"

    # Default to Python
    return "python"


def _sandbox_status(
    state: _SessionState,
    console: Console,
) -> None:
    """Display current sandbox configuration.

    Args:
        state: Session state with sandbox settings.
        console: Rich console for output.
    """
    from prism.tools.code_sandbox import (
        DEFAULT_MEMORY_MB,
        DEFAULT_TIMEOUT,
    )

    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
    )
    table.add_column("Key", style="bold")
    table.add_column("Value")

    enabled_str = (
        "[green]yes[/]"
        if state.sandbox_enabled
        else "[red]no[/]"
    )
    table.add_row("Enabled", enabled_str)

    sb_type = state.sandbox_type or "auto"
    table.add_row("Type", sb_type)
    table.add_row(
        "Timeout", f"{DEFAULT_TIMEOUT}s",
    )
    table.add_row(
        "Memory limit", f"{DEFAULT_MEMORY_MB} MB",
    )

    console.print(
        Panel(
            table,
            title="[bold]Sandbox Status[/bold]",
            border_style="blue",
        )
    )


def _cmd_tasks(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Background task management — list, cancel, results, clear.

    Subcommands:
        ``/tasks`` or ``/tasks list`` -- table of all tasks
            with ID, description, coloured status, progress %,
            elapsed time, and ETA.
        ``/tasks cancel <id>`` -- cancel a running task.
        ``/tasks results <id>`` -- view completed output in a
            Rich Panel.
        ``/tasks clear`` -- remove completed/failed tasks.

    Status group counts are printed above the table so users
    can see at a glance how many tasks are in each state.
    """
    import time as _time
    from datetime import datetime as _dt

    try:
        from prism.tools.task_queue import TaskQueue, TaskStatus

        tasks_dir = settings.prism_home / "tasks"
        queue = TaskQueue(tasks_dir=tasks_dir)

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        extra = parts[1].strip() if len(parts) > 1 else ""

        # --- /tasks cancel <id> ---
        if sub == "cancel":
            if not extra:
                console.print(
                    "[yellow]Usage:[/] "
                    "/tasks cancel <task-id>"
                )
                return "continue"
            cancelled = queue.cancel(extra)
            if cancelled:
                console.print(
                    f"[green]Task {extra} cancelled.[/]"
                )
            else:
                console.print(
                    f"[yellow]Task {extra} already "
                    f"completed.[/]"
                )
            return "continue"

        # --- /tasks results <id> ---
        if sub == "results":
            if not extra:
                console.print(
                    "[yellow]Usage:[/] "
                    "/tasks results <task-id>"
                )
                return "continue"
            result = queue.get_results(extra)
            if result is None:
                task_obj = queue.get_task(extra)
                console.print(
                    f"[yellow]Task {extra} "
                    f"({task_obj.status.value}) "
                    f"has no results yet.[/]"
                )
                return "continue"
            style = (
                "green" if result.exit_code == 0
                else "red"
            )
            header = (
                f"[{style}]Exit: {result.exit_code}"
                f"[/{style}] | "
                f"{result.duration_ms:.0f}ms"
            )
            body_parts: list[str] = []
            if result.output:
                body_parts.append(result.output)
            if result.error:
                body_parts.append(
                    f"[red]{result.error}[/red]"
                )
            body = (
                "\n".join(body_parts)
                if body_parts else "(no output)"
            )
            console.print(
                Panel(
                    body,
                    title=header,
                    border_style=style,
                )
            )
            return "continue"

        # --- /tasks clear ---
        if sub == "clear":
            cleared = queue.cleanup_completed(
                max_age_hours=0,
            )
            console.print(
                f"[green]Cleared {cleared} "
                f"completed/failed task(s).[/]"
            )
            return "continue"

        # --- /tasks list (default) ---
        tasks = queue.list_tasks()
        if not tasks:
            console.print(
                "[dim]No background tasks.[/dim]"
            )
            return "continue"

        # Status group counts
        status_colors: dict[str, str] = {
            "queued": "dim",
            "running": "cyan",
            "completed": "green",
            "failed": "red",
            "cancelled": "yellow",
        }

        counts: dict[str, int] = {}
        for task in tasks:
            sv = task.status.value
            counts[sv] = counts.get(sv, 0) + 1

        summary_parts: list[str] = []
        for sv, count in counts.items():
            color = status_colors.get(sv, "white")
            summary_parts.append(
                f"[{color}]{count} {sv}[/{color}]"
            )
        console.print(
            "[bold]Tasks:[/bold] "
            + ", ".join(summary_parts)
        )

        # Build the task table
        table = Table(title="Background Tasks")
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Description")
        table.add_column("Status", justify="center")
        table.add_column(
            "Progress", justify="right",
        )
        table.add_column(
            "Elapsed", justify="right",
        )
        table.add_column("ETA", justify="right")

        now_ts = _time.time()

        for task in tasks:
            color = status_colors.get(
                task.status.value, "white",
            )

            # Progress display with bar for running
            if task.status == TaskStatus.RUNNING:
                pct = f"{task.progress:.0%}"
                bar_len = 10
                filled = int(task.progress * bar_len)
                bar = (
                    "[cyan]"
                    + "#" * filled
                    + "-" * (bar_len - filled)
                    + "[/cyan]"
                )
                progress_str = f"{bar} {pct}"
            elif task.progress >= 1.0:
                progress_str = "[green]Done[/green]"
            else:
                progress_str = f"{task.progress:.0%}"

            # Elapsed time calculation
            elapsed_s = 0.0
            try:
                start_str = (
                    task.started_at or task.created_at
                )
                start_dt = _dt.fromisoformat(start_str)
                if task.completed_at:
                    end_dt = _dt.fromisoformat(
                        task.completed_at,
                    )
                    elapsed_s = (
                        end_dt - start_dt
                    ).total_seconds()
                else:
                    elapsed_s = (
                        now_ts
                        - start_dt.replace(
                            tzinfo=UTC,
                        ).timestamp()
                    )
                elapsed_s = max(elapsed_s, 0)
                if elapsed_s >= 3600:
                    elapsed_str = (
                        f"{elapsed_s / 3600:.1f}h"
                    )
                elif elapsed_s >= 60:
                    elapsed_str = (
                        f"{elapsed_s / 60:.1f}m"
                    )
                else:
                    elapsed_str = f"{elapsed_s:.0f}s"
            except (ValueError, TypeError):
                elapsed_str = "-"

            # ETA for running tasks with progress
            eta_str = "-"
            if (
                task.status == TaskStatus.RUNNING
                and 0 < task.progress < 1.0
                and elapsed_s > 0
            ):
                remaining_s = (
                    elapsed_s / task.progress
                    * (1.0 - task.progress)
                )
                if remaining_s >= 3600:
                    eta_str = (
                        f"~{remaining_s / 3600:.1f}h"
                    )
                elif remaining_s >= 60:
                    eta_str = (
                        f"~{remaining_s / 60:.1f}m"
                    )
                else:
                    eta_str = f"~{remaining_s:.0f}s"

            table.add_row(
                task.id,
                task.description,
                (
                    f"[{color}]{task.status.value}"
                    f"[/{color}]"
                ),
                progress_str,
                elapsed_str,
                eta_str,
            )
        console.print(table)

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("tasks_error", error=str(exc))
        console.print(f"[red]Tasks error:[/] {exc}")
    return "continue"


def _cmd_privacy(
    args: str,
    console: Console,
    **_: Any,
) -> str:
    """Privacy mode — Ollama-only routing."""
    try:
        from prism.network.privacy import PrivacyManager

        pm = PrivacyManager()
        sub = args.lower().strip()

        if sub == "on":
            status = pm.enable_private_mode()
            console.print("[green]Privacy mode ENABLED.[/]")
            console.print(
                f"  Ollama running: "
                f"{'yes' if status.ollama_running else 'no'}"
            )
            console.print(
                f"  Local models: {len(status.available_models)}"
            )

        elif sub == "off":
            pm.disable_private_mode()
            console.print(
                "[yellow]Privacy mode disabled. "
                "Cloud providers re-enabled.[/]"
            )

        else:
            # status (default)
            status = pm.get_status()
            level_color = (
                "green" if status.level.value == "normal" else "red"
            )
            console.print(
                f"[bold]Privacy:[/] "
                f"[{level_color}]{status.level.value.upper()}"
                f"[/{level_color}]"
            )
            console.print(
                f"  Ollama: "
                f"{'running' if status.ollama_running else 'not running'}"
            )
            if status.available_models:
                console.print("  Local models:")
                for m in status.available_models:
                    size_gb = m.size_bytes / 1_073_741_824
                    console.print(
                        f"    {m.name} ({size_gb:.1f} GB)"
                    )
            else:
                console.print("  [dim]No local models found.[/dim]")

    except Exception as exc:
        logger.debug("privacy_error", error=str(exc))
        console.print(f"[red]Privacy error:[/] {exc}")
    return "continue"


def _cmd_plugins(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Plugin management — list, install, or remove plugins."""
    try:
        from prism.plugins.manager import PluginManager

        plugins_dir = settings.prism_home / "plugins"
        pm = PluginManager(plugins_dir)

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        name = parts[1].strip() if len(parts) > 1 else ""

        if sub == "install":
            if not name:
                console.print(
                    "[yellow]Usage:[/] "
                    "/plugins install <github-url|path>"
                )
                return "continue"
            console.print(f"[dim]Installing plugin from {name}...[/]")
            info = pm.install(name)
            console.print(
                f"[green]Installed:[/] {info.manifest.name} "
                f"v{info.manifest.version} — "
                f"{info.manifest.description}"
            )

        elif sub == "remove":
            if not name:
                console.print(
                    "[yellow]Usage:[/] /plugins remove <name>"
                )
                return "continue"
            pm.remove(name)
            console.print(
                f"[green]Plugin '{name}' removed.[/]"
            )

        else:
            # list
            installed = pm.list_installed()
            available = pm.list_available()

            if installed:
                table = Table(title="Installed Plugins")
                table.add_column("Name", style="cyan")
                table.add_column("Version")
                table.add_column("Description")
                table.add_column("Enabled", justify="center")

                for p in installed:
                    enabled = (
                        "[green]yes[/]" if p.enabled
                        else "[red]no[/]"
                    )
                    table.add_row(
                        p.manifest.name,
                        p.manifest.version,
                        p.manifest.description[:50],
                        enabled,
                    )
                console.print(table)
            else:
                console.print("[dim]No plugins installed.[/dim]")

            if available:
                console.print()
                table = Table(title="Available Plugins")
                table.add_column("Name", style="cyan")
                table.add_column("Version")
                table.add_column("Description")

                for m in available:
                    table.add_row(
                        m.name,
                        m.version,
                        m.description[:50],
                    )
                console.print(table)

    except Exception as exc:
        logger.debug("plugins_error", error=str(exc))
        console.print(f"[red]Plugin error:[/] {exc}")
    return "continue"


def _cmd_forecast(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Cost forecast — monthly projection and model drivers."""
    try:
        from prism.cost.forecast import CostForecaster
        from prism.cost.tracker import CostTracker
        from prism.db.database import Database

        db = Database(settings.db_path)
        tracker = CostTracker(db=db, settings=settings)
        forecaster = CostForecaster(
            cost_tracker=tracker,
            settings=settings,
        )
        fc = forecaster.forecast()

        # Alert styling
        alert_colors = {
            "ok": "green",
            "warning": "yellow",
            "critical": "red",
        }
        alert_color = alert_colors.get(fc.alert_level, "white")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value", justify="right")

        table.add_row(
            "Month so far",
            f"${fc.current_monthly_cost:.4f}",
        )
        table.add_row(
            "Projected total",
            f"${fc.projected_monthly_cost:.4f}",
        )
        table.add_row("Daily average", f"${fc.daily_average:.4f}")
        table.add_row("Days remaining", str(fc.days_remaining))

        if fc.budget_limit is not None:
            table.add_row(
                "Budget",
                f"${fc.budget_limit:.2f} "
                f"({fc.budget_used_percent:.1f}% used)",
            )

        table.add_row(
            "Alert",
            f"[{alert_color}]{fc.alert_level.upper()}[/{alert_color}]",
        )

        # Velocity
        v = fc.velocity
        table.add_row(
            "Cost/hour", f"${v.cost_per_hour:.4f}"
        )
        table.add_row(
            "Tokens/hour", f"{v.tokens_per_hour:,}"
        )

        console.print(
            Panel(table, title="[bold]Cost Forecast[/bold]",
                  border_style="blue")
        )

        # Model drivers
        if fc.model_drivers:
            drivers_table = Table(title="Model Cost Drivers")
            drivers_table.add_column("Model", style="cyan")
            drivers_table.add_column("Cost", justify="right")
            drivers_table.add_column("%", justify="right")
            drivers_table.add_column("Requests", justify="right")
            drivers_table.add_column("Alternative")
            drivers_table.add_column(
                "Potential Savings", justify="right"
            )

            for d in fc.model_drivers:
                alt = d.cheapest_alternative or "-"
                sav = (
                    f"${d.potential_savings:.4f}"
                    if d.potential_savings > 0 else "-"
                )
                drivers_table.add_row(
                    d.display_name,
                    f"${d.total_cost:.4f}",
                    f"{d.percentage:.1f}%",
                    str(d.request_count),
                    alt,
                    sav,
                )
            console.print(drivers_table)

    except Exception as exc:
        logger.debug("forecast_error", error=str(exc))
        console.print(f"[red]Forecast error:[/] {exc}")
    return "continue"


def _cmd_workspace(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Multi-project workspace management."""
    try:
        from prism.workspace.manager import WorkspaceManager

        wm = WorkspaceManager(prism_home=settings.prism_home)

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        name = parts[1].strip() if len(parts) > 1 else ""

        if sub == "switch":
            if not name:
                console.print(
                    "[yellow]Usage:[/] /workspace switch <name>"
                )
                return "continue"
            project = wm.switch_project(name)
            console.print(
                f"[green]Switched to project '{project.name}' "
                f"({project.path}).[/]"
            )

        elif sub == "add":
            if not name:
                console.print(
                    "[yellow]Usage:[/] "
                    "/workspace add <name> [path]\n"
                    "[dim]Path defaults to current directory.[/dim]"
                )
                return "continue"
            name_parts = name.split(maxsplit=1)
            proj_name = name_parts[0]
            proj_path = (
                name_parts[1]
                if len(name_parts) > 1
                else str(settings.project_root)
            )
            info = wm.register_project(
                name=proj_name,
                path=proj_path,
            )
            console.print(
                f"[green]Registered project '{info.name}' "
                f"at {info.path}.[/]"
            )

        elif sub == "remove":
            if not name:
                console.print(
                    "[yellow]Usage:[/] /workspace remove <name>"
                )
                return "continue"
            wm.remove_project(name)
            console.print(
                f"[green]Project '{name}' unregistered.[/]"
            )

        else:
            # list
            projects = wm.list_projects()
            if not projects:
                console.print(
                    "[dim]No projects registered. "
                    "Use /workspace add <name>.[/dim]"
                )
                return "continue"

            table = Table(title="Workspace Projects")
            table.add_column("Name", style="cyan")
            table.add_column("Active", justify="center")
            table.add_column("Path")
            table.add_column("Last Accessed")

            active = wm.get_active_project()
            active_name = active.name if active else ""

            for p in projects:
                marker = (
                    "[green]>[/]" if p.name == active_name else ""
                )
                table.add_row(
                    p.name,
                    marker,
                    p.path,
                    p.last_accessed[:19],
                )
            console.print(table)

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("workspace_error", error=str(exc))
        console.print(f"[red]Workspace error:[/] {exc}")
    return "continue"


def _cmd_offline(
    args: str,
    console: Console,
    **_: Any,
) -> str:
    """Offline mode — show status and queue."""
    try:
        from prism.network.offline import OfflineModeManager

        om = OfflineModeManager()

        sub = args.lower().strip()

        if sub == "queue":
            queued = om.get_queued_requests()
            if not queued:
                console.print(
                    "[dim]No queued requests.[/dim]"
                )
                return "continue"

            table = Table(title="Queued Requests")
            table.add_column("ID", style="cyan")
            table.add_column("Model")
            table.add_column("Provider")
            table.add_column("Retries", justify="right")
            table.add_column("Queued At")

            for req in queued:
                table.add_row(
                    req.id,
                    req.model,
                    req.provider,
                    str(req.retry_count),
                    req.created_at[:19],
                )
            console.print(table)

        else:
            # status (default)
            details = om.get_status_details()
            caps = om.get_capabilities()

            state_color = (
                "green" if details["is_online"] else "red"
            )

            table = Table(
                show_header=False, box=None, padding=(0, 2),
            )
            table.add_column("Key", style="bold")
            table.add_column("Value")

            table.add_row(
                "State",
                f"[{state_color}]{details['state'].upper()}"
                f"[/{state_color}]",
            )
            table.add_row(
                "Manual offline",
                "yes" if details["manual_offline"] else "no",
            )
            table.add_row(
                "Queued requests",
                str(details["queued_requests"]),
            )
            table.add_row(
                "Monitoring",
                "active" if details["monitoring_active"] else "off",
            )

            console.print(
                Panel(table, title="[bold]Offline Mode[/bold]",
                      border_style="blue")
            )

            # Show capabilities
            cap_table = Table(title="Capabilities")
            cap_table.add_column("Feature")
            cap_table.add_column("Available", justify="center")

            for field_name in [
                "file_operations", "terminal_execution",
                "git_operations", "local_inference",
                "cache_hits", "cloud_inference",
                "web_browsing", "plugin_install",
            ]:
                available = getattr(caps, field_name, False)
                icon = (
                    "[green]yes[/]" if available
                    else "[red]no[/]"
                )
                label = field_name.replace("_", " ").title()
                cap_table.add_row(label, icon)
            console.print(cap_table)

    except Exception as exc:
        logger.debug("offline_error", error=str(exc))
        console.print(f"[red]Offline error:[/] {exc}")
    return "continue"


def _cmd_debate(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Multi-model debate on a question."""
    if not args:
        console.print(
            "[yellow]Usage:[/] /debate <question or decision>\n"
            "[dim]Runs a structured 3-round debate across "
            "multiple AI models.[/dim]"
        )
        return "continue"

    try:
        from prism.intelligence.debate import MultiModelDebate
        from prism.llm.completion import CompletionEngine

        engine = CompletionEngine(settings=settings)
        debate = MultiModelDebate(completion_engine=engine)

        console.print(
            "[dim]Starting multi-model debate...[/]"
        )

        session = asyncio.get_event_loop().run_until_complete(
            debate.debate(args)
        )

        # Display Round 1 positions
        console.print("\n[bold]Round 1 — Independent Positions[/bold]")
        for pos in session.round1_positions:
            console.print(Panel(
                Markdown(pos.content) if pos.content else Text(
                    "(no response)", style="dim"
                ),
                title=f"{pos.model}",
                subtitle=(
                    f"Tokens: {pos.tokens_used} | "
                    f"Cost: ${pos.cost_usd:.4f}"
                ),
                border_style="blue",
            ))

        # Display Round 2 critiques if present
        if session.round2_critiques:
            console.print(
                "\n[bold]Round 2 — Critiques[/bold]"
            )
            for crit in session.round2_critiques:
                console.print(Panel(
                    Markdown(crit.updated_position)
                    if crit.updated_position
                    else Text("(no critique)", style="dim"),
                    title=f"{crit.model}",
                    border_style="yellow",
                ))

        # Display synthesis
        if session.synthesis:
            syn = session.synthesis
            console.print(
                "\n[bold]Round 3 — Synthesis[/bold]"
            )
            console.print(Panel(
                Markdown(syn.recommendation)
                if syn.recommendation
                else Text("(no synthesis)", style="dim"),
                title=(
                    f"Synthesizer: {syn.synthesizer_model} | "
                    f"Confidence: {syn.confidence:.0%}"
                ),
                border_style="green",
            ))

        # Cost summary
        console.print(
            f"\n[dim]Total cost: ${session.total_cost:.4f} | "
            f"Total tokens: {session.total_tokens:,}[/dim]"
        )

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("debate_error", error=str(exc))
        console.print(f"[red]Debate error:[/] {exc}")
    return "continue"


def _cmd_blast(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Blast radius analysis for a planned change."""
    if not args:
        console.print(
            "[yellow]Usage:[/] /blast <file or description>"
        )
        return "continue"

    try:
        from prism.intelligence.blast_radius import (
            BlastRadiusAnalyzer,
        )

        analyzer = BlastRadiusAnalyzer(
            project_root=settings.project_root,
        )
        report = analyzer.analyze(description=args)

        # Risk score color
        if report.risk_score >= 70:
            score_style = "red bold"
        elif report.risk_score >= 40:
            score_style = "yellow"
        else:
            score_style = "green"

        console.print(
            f"\n[bold]Blast Radius Analysis[/bold] — "
            f"[{score_style}]Risk: {report.risk_score}/100"
            f"[/{score_style}] | "
            f"Complexity: {report.estimated_complexity}"
        )

        if report.affected_files:
            table = Table(title="Affected Files")
            table.add_column("File", style="cyan")
            table.add_column("Risk", justify="center")
            table.add_column("Depth", justify="center")
            table.add_column("Reason")
            table.add_column("Tests", justify="center")

            risk_colors = {
                "high": "red",
                "medium": "yellow",
                "low": "green",
            }
            for af in report.affected_files:
                color = risk_colors.get(af.risk_level, "white")
                tested = (
                    "[green]yes[/]" if af.has_tests
                    else "[red]no[/]"
                )
                table.add_row(
                    af.path,
                    f"[{color}]{af.risk_level.upper()}"
                    f"[/{color}]",
                    str(af.depth),
                    af.reason[:40],
                    tested,
                )
            console.print(table)

        if report.missing_tests:
            console.print(
                f"\n[yellow]Missing tests for "
                f"{len(report.missing_tests)} file(s):[/]"
            )
            for f in report.missing_tests[:10]:
                console.print(f"  {f}")

    except Exception as exc:
        logger.debug("blast_error", error=str(exc))
        console.print(f"[red]Blast radius error:[/] {exc}")
    return "continue"


def _cmd_gaps(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Test gap analysis — find untested code."""
    try:
        from prism.intelligence.test_gaps import TestGapHunter

        hunter = TestGapHunter(
            project_root=settings.project_root,
        )
        report = hunter.analyze()

        # Summary panel
        summary_lines = [
            f"Total functions: {report.total_functions}",
            f"Tested: {report.tested_functions}",
            f"Untested: {report.untested_functions}",
            f"Coverage: {report.coverage_percent:.1f}%",
            f"Critical gaps: {report.critical_count}",
            f"High gaps: {report.high_count}",
        ]
        console.print(Panel(
            "\n".join(summary_lines),
            title="[bold]Test Gap Analysis[/bold]",
            border_style="blue",
        ))

        # Filter by risk if requested
        gaps = report.gaps
        filter_level = args.lower().strip()
        if filter_level in ("critical", "high", "medium", "low"):
            gaps = [g for g in gaps if g.risk_level == filter_level]
            console.print(
                f"[dim]Filtered to {filter_level} risk only.[/dim]"
            )

        if gaps:
            table = Table(title="Test Gaps")
            table.add_column("Function", style="cyan")
            table.add_column("File")
            table.add_column("Line", justify="right")
            table.add_column("Risk", justify="center")
            table.add_column("Effort")
            table.add_column("Reason")

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
                    f"[{color}]{gap.risk_level.upper()}"
                    f"[/{color}]",
                    gap.estimated_effort,
                    gap.reason[:40],
                )
            console.print(table)

            if len(report.gaps) > 30:
                console.print(
                    f"[dim]... and {len(report.gaps) - 30} more. "
                    "Use /gaps critical to filter.[/dim]"
                )
        else:
            console.print(
                "[green]No test gaps found at this risk level.[/]"
            )

    except Exception as exc:
        logger.debug("gaps_error", error=str(exc))
        console.print(f"[red]Test gap error:[/] {exc}")
    return "continue"


def _cmd_deps(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Dependency health report."""
    try:
        from prism.intelligence.deps import DependencyMonitor

        monitor = DependencyMonitor(
            project_root=settings.project_root,
        )
        report = monitor.get_status()

        # Summary
        console.print(Panel(
            f"Total: {report.total_deps} | "
            f"Outdated: {report.outdated} | "
            f"Vulnerable: {report.vulnerable} | "
            f"Unused: {report.unused}",
            title="[bold]Dependency Health[/bold]",
            border_style="blue",
        ))

        # Dependencies table
        if report.dependencies:
            table = Table(title="Dependencies")
            table.add_column("Package", style="cyan")
            table.add_column("Current")
            table.add_column("Ecosystem")
            table.add_column("Usages", justify="right")
            table.add_column("Migration")

            for dep in report.dependencies[:30]:
                mig_color = {
                    "trivial": "green",
                    "simple": "yellow",
                    "moderate": "yellow bold",
                    "complex": "red",
                }.get(dep.migration_complexity.value, "white")
                table.add_row(
                    dep.name,
                    dep.current_version,
                    dep.ecosystem,
                    str(dep.usages),
                    f"[{mig_color}]"
                    f"{dep.migration_complexity.value}"
                    f"[/{mig_color}]",
                )
            console.print(table)

        # Vulnerabilities
        if report.vulnerabilities:
            vuln_table = Table(
                title="Vulnerabilities",
                border_style="red",
            )
            vuln_table.add_column(
                "Package", style="red bold"
            )
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

        # Unused
        if report.unused_deps:
            console.print(
                f"\n[yellow]Potentially unused "
                f"({len(report.unused_deps)}):[/]"
            )
            for name in report.unused_deps:
                console.print(f"  {name}")

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("deps_error", error=str(exc))
        console.print(f"[red]Deps error:[/] {exc}")
    return "continue"


def _cmd_arch(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Architecture map and drift detection."""
    try:
        from prism.intelligence.architecture import ArchitectureMapper

        mapper = ArchitectureMapper(
            project_root=settings.project_root,
        )

        sub = args.lower().strip()

        if sub == "drift":
            current = mapper.generate()
            violations = mapper.detect_drift(current)

            if not violations:
                console.print(
                    "[green]No architecture drift detected.[/]"
                )
                return "continue"

            table = Table(title="Architecture Drift")
            table.add_column("Type")
            table.add_column("Source", style="cyan")
            table.add_column("Target")
            table.add_column("Severity", justify="center")
            table.add_column("Description")

            sev_colors = {
                "high": "red",
                "medium": "yellow",
                "low": "green",
            }

            for v in violations:
                color = sev_colors.get(v.severity, "white")
                table.add_row(
                    v.violation_type,
                    v.source,
                    v.target or "-",
                    f"[{color}]{v.severity.upper()}"
                    f"[/{color}]",
                    v.description[:50],
                )
            console.print(table)

        else:
            # map (default)
            state = mapper.generate()
            saved_path = mapper.save(state)

            console.print(Panel(
                f"Modules: {state.total_modules}\n"
                f"Lines: {state.total_lines:,}\n"
                f"Dependencies: {len(state.dependencies)}",
                title="[bold]Architecture Map[/bold]",
                border_style="blue",
            ))

            # Show top-level modules
            if state.modules:
                table = Table(title="Modules")
                table.add_column("Module", style="cyan")
                table.add_column("Lines", justify="right")
                table.add_column("Public API", justify="right")
                table.add_column("Description")

                for m in sorted(
                    state.modules,
                    key=lambda x: x.line_count,
                    reverse=True,
                )[:20]:
                    table.add_row(
                        m.name,
                        str(m.line_count),
                        str(len(m.public_api)),
                        m.description[:50],
                    )
                console.print(table)

            console.print(
                f"\n[dim]ARCHITECTURE.md saved to "
                f"{saved_path}[/dim]"
            )

    except Exception as exc:
        logger.debug("arch_error", error=str(exc))
        console.print(f"[red]Architecture error:[/] {exc}")
    return "continue"


def _cmd_debug_memory(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Debug memory — search past bug fixes."""
    try:
        from prism.intelligence.debug_memory import DebugMemory

        db_path = settings.prism_home / "debug_memory.db"
        memory = DebugMemory(db_path=db_path)

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "stats"
        query = parts[1].strip() if len(parts) > 1 else ""

        if sub == "search" and query:
            records = memory.search_by_description(query)
            if not records:
                console.print(
                    f"[dim]No fixes found matching '{query}'.[/dim]"
                )
                memory.close()
                return "continue"

            table = Table(title=f"Debug Memory: '{query}'")
            table.add_column("ID", style="dim", width=5)
            table.add_column("Error Type", style="red")
            table.add_column("Fix Pattern")
            table.add_column("Project")
            table.add_column("Confidence", justify="right")

            for r in records[:15]:
                table.add_row(
                    str(r.id),
                    r.error_type,
                    r.fix_pattern[:40],
                    r.project,
                    f"{r.confidence:.0%}",
                )
            console.print(table)

        else:
            # stats (default)
            stats = memory.get_stats()

            table = Table(
                show_header=False, box=None, padding=(0, 2),
            )
            table.add_column("Key", style="bold")
            table.add_column("Value", justify="right")

            table.add_row(
                "Total fixes", str(stats["total_fixes"])
            )
            table.add_row("Projects", str(stats["projects"]))
            table.add_row(
                "Error types", str(stats["error_types"])
            )
            table.add_row(
                "Avg confidence",
                f"{stats['avg_confidence']:.0%}",
            )

            console.print(Panel(
                table,
                title="[bold]Debug Memory[/bold]",
                border_style="blue",
            ))

        memory.close()

    except Exception as exc:
        logger.debug("debug_memory_error", error=str(exc))
        console.print(f"[red]Debug memory error:[/] {exc}")
    return "continue"


def _cmd_history(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Code archaeology — trace file evolution."""
    if not args:
        console.print(
            "[yellow]Usage:[/] /history <file_path[:line]>"
        )
        return "continue"

    try:
        from prism.intelligence.archaeologist import CodeArchaeologist

        archaeologist = CodeArchaeologist(
            project_root=settings.project_root,
        )
        evolution = archaeologist.investigate(args)

        # Summary panel
        stability_color = (
            "green" if evolution.stability_score >= 0.7
            else "yellow" if evolution.stability_score >= 0.4
            else "red"
        )

        console.print(Panel(
            f"File: {evolution.file_path}\n"
            f"Commits: {evolution.total_commits}\n"
            f"Age: {evolution.age_days} days\n"
            f"Stability: [{stability_color}]"
            f"{evolution.stability_score:.0%}"
            f"[/{stability_color}]\n"
            f"Risk: {evolution.risk_assessment}",
            title=f"[bold]History: {args}[/bold]",
            border_style="blue",
        ))

        # Timeline
        if evolution.timeline:
            table = Table(title="Timeline")
            table.add_column("Hash", style="cyan", width=9)
            table.add_column("Type")
            table.add_column("Author")
            table.add_column("Date")
            table.add_column("Message")

            type_colors = {
                "created": "green",
                "bugfix": "red",
                "refactored": "yellow",
                "feature": "blue",
                "modified": "dim",
            }

            for event in evolution.timeline[:20]:
                color = type_colors.get(
                    event.event_type, "white"
                )
                table.add_row(
                    event.short_hash,
                    f"[{color}]{event.event_type}[/{color}]",
                    event.author,
                    event.date[:10],
                    event.message[:50],
                )
            console.print(table)

        # Authors
        if evolution.authors:
            auth_table = Table(title="Contributors")
            auth_table.add_column("Author", style="cyan")
            auth_table.add_column(
                "Commits", justify="right"
            )
            auth_table.add_column(
                "Lines +/-", justify="right"
            )
            auth_table.add_column(
                "Expertise", justify="right"
            )

            for author in evolution.authors:
                auth_table.add_row(
                    author.name,
                    str(author.commits),
                    f"+{author.lines_added}/"
                    f"-{author.lines_removed}",
                    f"{author.expertise_score:.0%}",
                )
            console.print(auth_table)

        # Narrative
        if evolution.narrative:
            console.print(
                Panel(
                    Markdown(evolution.narrative),
                    title="Narrative",
                    border_style="dim",
                )
            )

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("history_error", error=str(exc))
        console.print(f"[red]History error:[/] {exc}")
    return "continue"


def _cmd_budget(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Context budget management — status and configuration."""
    try:
        from prism.context.budget import ContextBudgetManager

        manager = ContextBudgetManager(
            project_root=settings.project_root,
        )

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "status"
        extra = parts[1].strip() if len(parts) > 1 else ""

        if sub == "set" and extra:
            try:
                token_val = int(extra)
                settings.set_override(
                    "context.max_tokens", token_val
                )
                console.print(
                    f"[green]Context budget set to "
                    f"{token_val:,} tokens.[/]"
                )
            except ValueError:
                console.print(
                    "[yellow]Usage:[/] /budget set <token-count>"
                )

        else:
            # status (default)
            stats = manager.get_stats()

            table = Table(
                show_header=False, box=None, padding=(0, 2),
            )
            table.add_column("Key", style="bold")
            table.add_column("Value", justify="right")

            table.add_row(
                "Selections",
                str(stats.total_requests),
            )
            table.add_row(
                "Avg tokens used",
                f"{stats.avg_tokens_used:,.0f}",
            )
            table.add_row(
                "Avg tokens saved",
                f"{stats.avg_tokens_saved:,.0f}",
            )
            table.add_row(
                "Avg efficiency",
                f"{stats.avg_efficiency:.1%}",
            )
            table.add_row(
                "Total tokens saved",
                f"{stats.total_tokens_saved:,}",
            )

            console.print(Panel(
                table,
                title="[bold]Context Budget[/bold]",
                border_style="blue",
            ))

    except Exception as exc:
        logger.debug("budget_error", error=str(exc))
        console.print(f"[red]Budget error:[/] {exc}")
    return "continue"


def _cmd_architect(
    args: str,
    console: Console,
    settings: Settings,
    state: _SessionState,
    **_: Any,
) -> str:
    """Plan and execute complex multi-step tasks."""
    try:
        from prism.architect.display import (
            display_execution_summary,
            display_plan_list,
            display_plan_review,
            display_rollback_result,
        )
        from prism.architect.executor import (
            ArchitectExecutor,
        )
        from prism.architect.planner import ArchitectPlanner
        from prism.architect.storage import PlanStorage
        from prism.cost.tracker import CostTracker
        from prism.db.database import Database

        db = Database(settings.db_path)
        tracker = CostTracker(db=db, settings=settings)

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        extra = parts[1].strip() if len(parts) > 1 else ""

        # --- /architect list ---
        if sub == "list":
            storage = PlanStorage(db)
            plans = storage.list_plans()
            display_plan_list(plans, console)
            return "continue"

        # --- /architect status [plan_id] ---
        if sub == "status":
            if not extra:
                console.print(
                    "[yellow]Usage:[/] "
                    "/architect status <plan-id>"
                )
                return "continue"
            storage = PlanStorage(db)
            plan = storage.load_plan(extra)
            if plan is None:
                console.print(
                    f"[yellow]Plan not found:[/] {extra}"
                )
                return "continue"
            display_plan_review(plan, console)
            return "continue"

        # --- /architect resume [plan_id] ---
        if sub == "resume":
            if not extra:
                console.print(
                    "[yellow]Usage:[/] "
                    "/architect resume <plan-id>"
                )
                return "continue"
            storage = PlanStorage(db)
            plan = storage.load_plan(extra)
            if plan is None:
                console.print(
                    f"[yellow]Plan not found:[/] {extra}"
                )
                return "continue"

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
            return "continue"

        # --- /architect rollback [plan_id] ---
        if sub == "rollback":
            if not extra:
                console.print(
                    "[yellow]Usage:[/] "
                    "/architect rollback <plan-id>"
                )
                return "continue"
            storage = PlanStorage(db)
            plan = storage.load_plan(extra)
            if plan is None:
                console.print(
                    f"[yellow]Plan not found:[/] {extra}"
                )
                return "continue"

            executor = ArchitectExecutor(
                settings=settings,
                cost_tracker=tracker,
            )
            success, description = executor.rollback(plan)
            display_rollback_result(
                success, console, description=description,
            )
            storage.save_plan(plan)
            return "continue"

        # --- /architect <goal> (default: create + execute) ---
        if not args.strip():
            console.print(
                "[yellow]Usage:[/] /architect <goal>\n"
                "[dim]Subcommands: list, status, "
                "resume, rollback[/dim]"
            )
            return "continue"

        goal = args.strip()
        planner = ArchitectPlanner(
            settings=settings,
            cost_tracker=tracker,
        )
        plan = planner.create_plan(goal)
        display_plan_review(plan, console)

        # Auto-approve and execute
        plan.status = "approved"
        console.print(
            "[green]Plan auto-approved. "
            "Executing...[/green]"
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

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("architect_error", error=str(exc))
        console.print(f"[red]Architect error:[/] {exc}")
    return "continue"


# ======================================================================
# Prompt processing (non-slash input)
# ======================================================================

def _process_prompt(
    prompt: str,
    console: Console,
    settings: Settings,
    state: _SessionState,
    dry_run: bool,
    offline: bool,
) -> None:
    """Process a user prompt through the routing engine.

    Args:
        prompt: User's input text.
        console: Rich console for output.
        settings: App settings.
        state: Mutable session state.
        dry_run: Show routing without executing.
        offline: Only use local models.
    """
    from prism.router.classifier import TaskClassifier, TaskContext

    classifier = TaskClassifier(settings)
    context = TaskContext(active_files=state.active_files)
    result = classifier.classify(prompt, context)

    tier_colors = {
        "simple": "green",
        "medium": "yellow",
        "complex": "red",
    }
    color = tier_colors.get(result.tier.value, "white")
    console.print(
        f"\n[dim]Tier:[/] [{color}]{result.tier.value.upper()}"
        f"[/{color}] "
        f"[dim](score: {result.score:.2f})[/dim]"
    )

    if dry_run:
        console.print(f"[dim]Features:[/] {result.features}")
        console.print(f"[dim]Reasoning:[/] {result.reasoning}")
        console.print(
            "[yellow]Dry-run mode — no API call made.[/]\n"
        )
        return

    state.conversation.append(
        {"role": "user", "content": prompt}
    )

    # Attempt completion through the engine
    try:
        from prism.llm.completion import CompletionEngine

        engine = CompletionEngine(settings=settings)
        messages = list(state.conversation)

        completion = asyncio.get_event_loop().run_until_complete(
            engine.complete(
                messages=messages,
                model=state.pinned_model or "",
            )
        )

        content = getattr(completion, "content", "")
        if content:
            state.conversation.append(
                {"role": "assistant", "content": content}
            )
            console.print(Panel(
                Markdown(content),
                border_style="blue",
            ))

        cost = getattr(completion, "cost_usd", 0.0)
        tokens_in = getattr(completion, "input_tokens", 0)
        tokens_out = getattr(completion, "output_tokens", 0)
        model_used = getattr(completion, "model", "unknown")

        console.print(
            f"[dim]{model_used} | "
            f"{tokens_in}+{tokens_out} tokens | "
            f"${cost:.4f}[/dim]\n"
        )
    except Exception as exc:
        logger.debug("completion_error", error=str(exc))
        console.print(
            f"[red]Completion error:[/] {exc}\n"
        )

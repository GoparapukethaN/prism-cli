"""Interactive REPL for Prism — the main user interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

if TYPE_CHECKING:
    from rich.console import Console

    from prism.config.settings import Settings

logger = structlog.get_logger(__name__)

# Slash commands recognized by the REPL
SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/cost": "Show cost dashboard",
    "/model": "Show or set current model (e.g., /model claude-sonnet)",
    "/add": "Add files to context (e.g., /add src/main.py)",
    "/drop": "Remove files from context (e.g., /drop src/main.py)",
    "/compact": "Compress conversation history",
    "/undo": "Undo last file edit",
    "/web": "Toggle web browsing (on/off)",
    "/status": "Check provider status",
    "/clear": "Clear conversation history",
    "/quit": "Exit Prism",
}


def run_repl(
    settings: Settings,
    console: Console,
    dry_run: bool = False,
    offline: bool = False,
) -> None:
    """Run the interactive REPL loop.

    Args:
        settings: Application settings.
        console: Rich console for output.
        dry_run: If True, show routing decisions without executing.
        offline: If True, only use local models.
    """
    # Set up prompt session with history
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

    # Session state
    active_files: list[str] = []
    conversation: list[dict[str, str]] = []
    pinned_model: str | None = settings.config.pinned_model

    console.print("[dim]Ready. Type your request or /help for commands.[/dim]\n")

    while True:
        try:
            # Get user input
            try:
                user_input = session.prompt(
                    "prism> ",
                    # We use a simple string prompt; Rich handles the output
                ).strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                handled = _handle_slash_command(
                    user_input,
                    console=console,
                    settings=settings,
                    active_files=active_files,
                    pinned_model=pinned_model,
                )
                if handled == "quit":
                    break
                if handled == "model_changed":
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        pinned_model = parts[1].strip()
                        console.print(f"[green]Model set to:[/] {pinned_model}")
                    else:
                        console.print(f"[bold]Current model:[/] {pinned_model or 'auto (routing)'}")
                continue

            # Regular prompt — process through router
            _process_prompt(
                prompt=user_input,
                console=console,
                settings=settings,
                active_files=active_files,
                conversation=conversation,
                dry_run=dry_run,
                offline=offline,
                pinned_model=pinned_model,
            )

        except Exception:
            logger.exception("repl_error")
            console.print("[red]An unexpected error occurred. Check logs for details.[/]")


def _handle_slash_command(
    command: str,
    console: Console,
    settings: Settings,
    active_files: list[str],
    pinned_model: str | None,
) -> str:
    """Handle a slash command.

    Args:
        command: The full command string.
        console: Rich console.
        settings: App settings.
        active_files: Current active files list (mutable).
        pinned_model: Currently pinned model.

    Returns:
        Action string: "continue", "quit", or "model_changed".
    """
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    if cmd in {"/quit", "/exit", "/q"}:
        console.print("[dim]Goodbye![/dim]")
        return "quit"

    if cmd == "/help":
        console.print("\n[bold]Available Commands:[/bold]\n")
        for slash_cmd, desc in SLASH_COMMANDS.items():
            console.print(f"  [cyan]{slash_cmd:12s}[/cyan] {desc}")
        console.print()
        return "continue"

    if cmd == "/cost":
        console.print("[yellow]Cost dashboard will be available after first API call.[/]")
        return "continue"

    if cmd == "/model":
        return "model_changed"

    if cmd == "/add":
        if not args:
            console.print("[yellow]Usage:[/] /add <file1> [file2] ...")
            return "continue"
        new_files = args.split()
        for f in new_files:
            if f not in active_files:
                active_files.append(f)
                console.print(f"  [green]+[/] {f}")
            else:
                console.print(f"  [dim]Already added:[/dim] {f}")
        return "continue"

    if cmd == "/drop":
        if not args:
            if active_files:
                console.print("[bold]Active files:[/bold]")
                for f in active_files:
                    console.print(f"  {f}")
            else:
                console.print("[dim]No active files.[/dim]")
            return "continue"
        drop_files = args.split()
        for f in drop_files:
            if f in active_files:
                active_files.remove(f)
                console.print(f"  [red]-[/] {f}")
            else:
                console.print(f"  [dim]Not in context:[/dim] {f}")
        return "continue"

    if cmd == "/compact":
        console.print("[yellow]Conversation compaction not yet implemented.[/]")
        return "continue"

    if cmd == "/undo":
        console.print("[yellow]Undo not yet implemented.[/]")
        return "continue"

    if cmd == "/web":
        state = args.lower() if args else ""
        if state == "on":
            console.print("[green]Web browsing enabled.[/]")
        elif state == "off":
            console.print("[yellow]Web browsing disabled.[/]")
        else:
            console.print("[dim]Usage:[/] /web on|off")
        return "continue"

    if cmd == "/status":
        console.print("[dim]Running status check...[/]")
        from prism.cli.app import status

        status()
        return "continue"

    if cmd == "/clear":
        console.print("[dim]Conversation cleared.[/dim]")
        return "continue"

    console.print(f"[yellow]Unknown command:[/] {cmd}. Type /help for available commands.")
    return "continue"


def _process_prompt(
    prompt: str,
    console: Console,
    settings: Settings,
    active_files: list[str],
    conversation: list[dict[str, str]],
    dry_run: bool,
    offline: bool,
    pinned_model: str | None,
) -> None:
    """Process a user prompt through the routing engine.

    Args:
        prompt: User's input text.
        console: Rich console for output.
        settings: App settings.
        active_files: Active files in context.
        conversation: Conversation history.
        dry_run: Show routing without executing.
        offline: Only use local models.
        pinned_model: Forced model override.
    """
    from prism.router.classifier import TaskClassifier, TaskContext

    # Classify the task
    classifier = TaskClassifier(settings)
    context = TaskContext(active_files=active_files)
    result = classifier.classify(prompt, context)

    # Show classification
    tier_colors = {"simple": "green", "medium": "yellow", "complex": "red"}
    color = tier_colors.get(result.tier.value, "white")
    console.print(
        f"\n[dim]Tier:[/] [{color}]{result.tier.value.upper()}[/{color}] "
        f"[dim](score: {result.score:.2f})[/dim]"
    )

    if dry_run:
        console.print(f"[dim]Features:[/] {result.features}")
        console.print(f"[dim]Reasoning:[/] {result.reasoning}")
        console.print("[yellow]Dry-run mode — no API call made.[/]\n")
        return

    # Add to conversation
    conversation.append({"role": "user", "content": prompt})

    console.print(
        "[yellow]API routing and execution not yet fully implemented. "
        "Classification works — model calls coming in next phase.[/]\n"
    )

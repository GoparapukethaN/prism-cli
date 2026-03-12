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
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from prism.cli.keybindings import create_keybindings

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
        ("/context [show|add|drop|stats] [file]",
         "Smart context budget allocation"),
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
        ("/blast <file>",
         "Blast radius / impact analysis (alias: /impact)"),
        ("/gaps [critical|high|generate]", "Test gap analysis"),
        ("/deps [status|audit|unused]",
         "Dependency health, security audit, unused scan"),
        ("/arch [map|drift|mermaid|check|diff]",
         "Architecture map / drift / diagram"),
        ("/debug-memory [stats|search|bugs|forget|"
         "export|import]",
         "Debug memory management"),
        ("/history <file>", "Code archaeology — file evolution"),
        ("/aei [stats|reset|explain]",
         "AEI statistics, reset, or explain fingerprint"),
        ("/blame <description>",
         "Causal blame trace with optional bisect"),
        ("/why <file:line|func>",
         "Investigate code history and evolution"),
        ("/debates", "List saved debate reports"),
    ],
    "Planning": [
        ("/architect <goal>",
         "Plan and execute complex multi-step tasks"),
        ("/architect list", "List all plans"),
        ("/architect resume [id]", "Resume a paused plan"),
        ("/architect status [id]", "Show plan status"),
        ("/architect rollback [id]", "Rollback a plan"),
    ],
    "Smart Tools (unique to Prism)": [
        ("/test [files]", "Auto-discover and run relevant tests"),
        ("/quality [files]", "Run lint + security + type checks (alias: /lint)"),
        ("/optimize [report|recommend]", "Cost optimization recommendations"),
        ("/memory [show|add|remove] [key] [value]",
         "Project memory (PRISM_MEMORY.md)"),
    ],
    "Multi-Agent": [
        ("/swarm <goal>", "Multi-model collaborative task execution"),
        ("/swarm status", "Show current swarm execution status"),
        ("/swarm plan <goal>", "Plan only — show decomposition without executing"),
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
        ("/ignore [add|list|check|create] [pattern|file]",
         "Manage .prismignore patterns"),
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
        permission_cache: Tool names approved for the session ("always").
        skip_all_permissions: When True, skip all permission prompts.
        settings: Reference to the Settings object.
    """

    def __init__(
        self,
        pinned_model: str | None,
        cache_enabled: bool = True,
    ) -> None:
        self.active_files: list[str] = []
        self.conversation: list[dict[str, Any]] = []
        self.pinned_model: str | None = pinned_model
        self.web_enabled: bool = False
        self.session_id: str = ""
        self.cache_enabled: bool = cache_enabled
        self.sandbox_enabled: bool = True
        self.sandbox_type: str | None = None

        # Permission memory — tools approved for the session
        self.permission_cache: set[str] = set()
        self.skip_all_permissions: bool = False
        self.settings: Any | None = None

        # Full module stack — populated in run_repl()
        self.db: Any | None = None
        self.auth: Any | None = None
        self.registry: Any | None = None
        self.cost_tracker: Any | None = None
        self.classifier: Any | None = None
        self.selector: Any | None = None
        self.fallback_builder: Any | None = None
        self.learner: Any | None = None
        self.completion_engine: Any | None = None
        self.tool_registry: Any | None = None
        self.session_manager: Any | None = None
        self.hook_manager: Any | None = None
        self.prompt_enhancer: Any | None = None
        self.mcp_client: Any | None = None
        self.rate_limiter: Any | None = None
        self.audit_logger: Any | None = None
        self.auto_committer: Any | None = None
        self.privacy_manager: Any | None = None
        self.offline_manager: Any | None = None


# ======================================================================
# Completer — slash commands + file paths
# ======================================================================

def _build_completer() -> Any:
    """Build a completer that handles slash commands and file paths."""
    from prompt_toolkit.completion import (
        Completer,
        Completion,
        PathCompleter,
        merge_completers,
    )

    slash_commands = sorted({
        "/help", "/quit", "/exit", "/cost", "/model", "/add", "/drop",
        "/compact", "/undo", "/web", "/status", "/clear", "/cache",
        "/compare", "/image", "/branch", "/rollback", "/sandbox",
        "/tasks", "/privacy", "/plugins", "/forecast", "/workspace",
        "/offline", "/debate", "/blast", "/impact", "/gaps", "/deps",
        "/arch", "/debug-memory", "/history", "/budget", "/architect",
        "/ignore", "/aei", "/blame", "/context", "/why", "/debates",
        "/test", "/autotest", "/quality", "/lint", "/optimize", "/memory",
    })

    class SlashCommandCompleter(Completer):
        """Complete slash commands."""

        def get_completions(self, document: Any, complete_event: Any) -> Any:
            text = document.text_before_cursor
            if text.startswith("/"):
                for cmd in slash_commands:
                    if cmd.startswith(text):
                        yield Completion(cmd, start_position=-len(text))

    return merge_completers([
        SlashCommandCompleter(),
        PathCompleter(expanduser=True),
    ])


# ======================================================================
# UI helpers — Claude CLI-style display
# ======================================================================


def _display_user_message(console: Console, text: str) -> None:
    """Display the user's message with a highlighted background."""
    from rich.text import Text as RichText

    chevron = "\u276f"  # heavy right-pointing angle
    msg = RichText()
    msg.append(f"{chevron} ", style="bold bright_cyan")
    msg.append(text)
    msg.stylize("on grey11")  # subtle dark background highlight
    # Pad to full terminal width for the background bar effect
    padding = max(0, console.width - len(f"{chevron} {text}"))
    msg.append(" " * padding, style="on grey11")
    console.print(msg)


def _display_ai_response(console: Console, content: str) -> None:
    """Display an AI response with a ● prefix like Claude CLI.

    Shows the response with a cyan bullet indicator and rendered
    Markdown formatting.
    """
    from rich.console import Group
    from rich.markdown import Markdown as RichMarkdown
    from rich.text import Text as RichText

    bullet = RichText("● ", style="bold bright_cyan")
    console.print()
    console.print(Group(bullet, RichMarkdown(content)))
    console.print()


# ======================================================================
# Public entry point
# ======================================================================

def run_repl(
    settings: Settings,
    console: Console,
    dry_run: bool = False,
    offline: bool = False,
    no_cache: bool = False,
    skip_permissions: bool = False,
) -> None:
    """Run the interactive REPL loop.

    Args:
        settings: Application settings.
        console: Rich console for output.
        dry_run: If True, show routing decisions without executing.
        offline: If True, only use local models.
        no_cache: If True, disable response caching.
        skip_permissions: If True, skip all permission prompts (dangerous).
    """
    # Suppress all library noise — users should only see responses
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*Task was destroyed.*")
    warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited.*")
    warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")

    import litellm
    litellm.suppress_debug_info = True
    import logging as _logging
    _logging.getLogger("LiteLLM").setLevel(_logging.CRITICAL)
    _logging.getLogger("litellm").setLevel(_logging.CRITICAL)
    _logging.getLogger("litellm.utils").setLevel(_logging.CRITICAL)
    _logging.getLogger("litellm.cost_calculator").setLevel(_logging.CRITICAL)
    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)
    _logging.getLogger("huggingface_hub").setLevel(_logging.CRITICAL)

    history_path = settings.sessions_dir / "repl_history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    state = _SessionState(
        pinned_model=settings.config.pinned_model,
        cache_enabled=not no_cache,
    )
    state.settings = settings
    state.skip_all_permissions = skip_permissions

    if skip_permissions:
        console.print(
            "[bold red]All permission prompts disabled. "
            "Tools will execute without confirmation.[/bold red]"
        )

    bindings = create_keybindings(state=state, console=console)

    completer = _build_completer()

    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_path)),
        key_bindings=bindings,
        completer=completer,
        multiline=False,
        enable_history_search=True,
        wrap_lines=True,
    )

    # Generate a stable session id
    from uuid import uuid4

    state.session_id = str(uuid4())[:12]

    # --- Initialize full module stack ---
    try:
        from pathlib import Path as _Path

        from prism.auth.manager import AuthManager
        from prism.cost.tracker import CostTracker
        from prism.db.database import Database
        from prism.llm.completion import CompletionEngine
        from prism.llm.retry import RetryPolicy
        from prism.providers.registry import ProviderRegistry
        from prism.router.classifier import TaskClassifier
        from prism.router.fallback import FallbackChain
        from prism.router.learning import AdaptiveLearner
        from prism.router.rate_limiter import RateLimiter
        from prism.router.selector import ModelSelector
        from prism.security.audit import AuditLogger
        from prism.security.path_guard import PathGuard
        from prism.security.sandbox import CommandSandbox
        from prism.security.secret_filter import SecretFilter
        from prism.tools.registry import ToolRegistry

        db = Database(settings.db_path)
        db.initialize()
        state.db = db

        auth = AuthManager()
        state.auth = auth

        registry = ProviderRegistry(settings, auth)
        state.registry = registry

        cost_tracker = CostTracker(db, settings)
        state.cost_tracker = cost_tracker

        state.classifier = TaskClassifier(settings)
        state.selector = ModelSelector(settings, registry, cost_tracker)
        state.fallback_builder = FallbackChain(registry)
        state.learner = AdaptiveLearner()

        state.completion_engine = CompletionEngine(
            settings=settings,
            cost_tracker=cost_tracker,
            auth_manager=auth,
            provider_registry=registry,
            retry_policy=RetryPolicy(),
        )

        project_root = _Path.cwd().resolve()
        path_guard = PathGuard(project_root=project_root)
        secret_filter = SecretFilter()
        sandbox = CommandSandbox(
            project_root=project_root,
            secret_filter=secret_filter,
        )
        state.rate_limiter = RateLimiter()
        state.audit_logger = AuditLogger(settings.audit_log_path)
        state.tool_registry = ToolRegistry.create_default(
            path_guard, sandbox,
            web_enabled=state.web_enabled,
            cost_tracker=cost_tracker,
            adaptive_learner=state.learner,
        )

        from prism.context.session import SessionManager
        state.session_manager = SessionManager(settings.sessions_dir)

        # Hook manager — pre/post tool execution hooks
        from prism.cli.hooks import HookManager
        state.hook_manager = HookManager(project_root)

        # Prompt enhancer — injects project context into prompts
        from prism.cli.prompt_enhancer import PromptEnhancer
        state.prompt_enhancer = PromptEnhancer(project_root)

        # Auto-committer — commit after file writes/edits
        try:
            from prism.git.auto_commit import AutoCommitter
            from prism.git.operations import GitRepo

            state.auto_committer = AutoCommitter(GitRepo(project_root))
        except Exception:
            logger.debug("auto_committer_init_skipped")

        # Privacy manager — Ollama-only routing in private mode
        try:
            from prism.network.privacy import PrivacyManager

            state.privacy_manager = PrivacyManager()
        except Exception:
            logger.debug("privacy_manager_init_skipped")

        # Offline mode manager — connectivity monitoring
        try:
            from prism.network.offline import OfflineModeManager

            state.offline_manager = OfflineModeManager()
            if offline:
                state.offline_manager.enable_manual_offline()
        except Exception:
            logger.debug("offline_manager_init_skipped")

        # MCP client — connect to external tool servers
        from prism.mcp.client import MCPClient
        mcp = MCPClient(project_root)
        mcp.load_config()
        if mcp.servers:
            mcp.connect_all()
            console.print(
                f"[green]MCP:[/green] {len(mcp.servers)} server(s)"
            )
        state.mcp_client = mcp

        # Show available providers
        available = registry.get_available_models()
        if available:
            providers_found = {m.provider for m in available}
            console.print(
                f"[green]Providers:[/green] {', '.join(sorted(providers_found))}  "
                f"[dim]({len(available)} models)[/dim]"
            )
        else:
            console.print(
                "[yellow]No API keys found.[/yellow] "
                "Run [bold]prism auth add <provider>[/bold] or "
                "set environment variables."
            )

    except Exception as exc:
        logger.debug("module_init_failed", error=str(exc))

    while True:
        try:
            try:
                from prompt_toolkit.formatted_text import HTML
                from rich.rule import Rule

                # Input border — thin line above the prompt (like Claude CLI)
                console.print(Rule(style="dim"))

                user_input = session.prompt(
                    HTML("<ansibrightcyan><b>\u276f </b></ansibrightcyan>")
                ).strip()

                # Line below the prompt input area
                console.print(Rule(style="dim"))
            except KeyboardInterrupt:
                continue
            except EOFError:
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            # Re-display user input with highlighted background
            _display_user_message(console, user_input)

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

    # Cleanup MCP connections on exit
    if state.mcp_client:
        try:
            state.mcp_client.disconnect_all()
        except Exception:
            logger.debug("mcp_disconnect_failed")


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
        "/impact": _cmd_blast,
        "/gaps": _cmd_gaps,
        "/deps": _cmd_deps,
        "/arch": _cmd_arch,
        "/debug-memory": _cmd_debug_memory,
        "/history": _cmd_history,
        "/budget": _cmd_budget,
        "/architect": _cmd_architect,
        "/ignore": _cmd_ignore,
        "/aei": _cmd_aei,
        "/blame": _cmd_blame,
        "/context": _cmd_context,
        "/why": _cmd_why,
        "/debates": _cmd_debates,
        "/test": _cmd_autotest,
        "/autotest": _cmd_autotest,
        "/quality": _cmd_quality,
        "/lint": _cmd_quality,
        "/optimize": _cmd_optimize,
        "/memory": _cmd_memory,
        "/swarm": _cmd_swarm,
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
    settings: Settings,
    **_: Any,
) -> str:
    """Add files to the context.

    Warns when a file matches a ``.prismignore`` pattern (likely
    contains secrets) but still adds it per user intent.
    """
    if not args:
        console.print("[yellow]Usage:[/] /add <file1> [file2] ...")
        return "continue"

    # Load .prismignore for warning checks (best-effort)
    try:
        from prism.security.prismignore import PrismIgnore

        prismignore: PrismIgnore | None = PrismIgnore(
            settings.project_root,
        )
    except Exception:
        prismignore = None

    new_files = args.split()
    for f in new_files:
        if f not in state.active_files:
            # Warn if the file is in .prismignore
            if prismignore is not None and prismignore.is_ignored(f):
                console.print(
                    f"  [yellow]Warning:[/] {f} is in "
                    ".prismignore (likely contains secrets)"
                )
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
    state: _SessionState | None = None,
    **_: Any,
) -> str:
    """Privacy mode — Ollama-only routing."""
    try:
        from prism.network.privacy import (
            RECOMMENDED_MODELS,
            PrivacyManager,
        )

        # Use the shared privacy manager from session state when available
        if state is not None and state.privacy_manager is not None:
            pm = state.privacy_manager
        else:
            pm = PrivacyManager()
            if state is not None:
                state.privacy_manager = pm
        sub = args.lower().strip()

        if sub == "on":
            status = pm.enable_private_mode()
            console.print(
                "[bold green]Privacy mode ENABLED"
                "[/bold green] "
                "[green](all traffic stays local)"
                "[/green]"
            )
            ollama_ind = (
                "[green]running[/green]"
                if status.ollama_running
                else "[red]not running[/red]"
            )
            console.print(f"  Ollama: {ollama_ind}")
            if status.available_models:
                console.print(
                    "  [bold]Installed models"
                    f" ({len(status.available_models)}"
                    "):[/bold]"
                )
                for m in status.available_models:
                    size_gb = (
                        m.size_bytes / 1_073_741_824
                    )
                    console.print(
                        f"    [cyan]{m.name}[/cyan]"
                        f" ({size_gb:.1f} GB)"
                    )
            else:
                console.print(
                    "  [yellow]No local models "
                    "found.[/yellow]"
                )
                console.print(
                    "  [bold]Recommended "
                    "models:[/bold]"
                )
                for name, desc in (
                    RECOMMENDED_MODELS.items()
                ):
                    console.print(
                        f"    [cyan]{name}[/cyan]"
                        f" — {desc}"
                    )
                console.print(
                    "\n  [dim]Try:[/dim] "
                    "[bold]ollama pull "
                    "qwen2.5-coder:7b[/bold]"
                )

        elif sub == "off":
            pm.disable_private_mode()
            console.print(
                "[yellow]Privacy mode disabled."
                "[/yellow] "
                "Cloud providers re-enabled."
            )

        else:
            # status (default)
            status = pm.get_status()
            if status.level.value == "private":
                indicator = (
                    "[bold green]PRIVATE"
                    "[/bold green] "
                    "[green](local only)[/green]"
                )
            else:
                indicator = (
                    "[bold]NORMAL[/bold]"
                    " (cloud + local)"
                )
            console.print(
                "[bold]Privacy mode:[/bold]"
                f" {indicator}"
            )
            ollama_ind = (
                "[green]running[/green]"
                if status.ollama_running
                else "[red]not running[/red]"
            )
            console.print(f"  Ollama: {ollama_ind}")
            if status.available_models:
                console.print(
                    "  [bold]Installed models"
                    f" ({len(status.available_models)}"
                    "):[/bold]"
                )
                for m in status.available_models:
                    size_gb = (
                        m.size_bytes / 1_073_741_824
                    )
                    console.print(
                        f"    [cyan]{m.name}[/cyan]"
                        f" ({size_gb:.1f} GB)"
                    )
            else:
                console.print(
                    "  [dim]No local models "
                    "installed.[/dim]"
                )
            console.print(
                "  [bold]Recommended models:"
                "[/bold]"
            )
            for name, desc in (
                RECOMMENDED_MODELS.items()
            ):
                console.print(
                    f"    [cyan]{name}[/cyan]"
                    f" — {desc}"
                )

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
    state: _SessionState | None = None,
    **_: Any,
) -> str:
    """Offline mode — show status and queue."""
    try:
        from prism.network.offline import OfflineModeManager

        # Use the shared offline manager from session state when available
        if state is not None and state.offline_manager is not None:
            om = state.offline_manager
        else:
            om = OfflineModeManager()
            if state is not None:
                state.offline_manager = om

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
    """Multi-model debate on a question.

    Supports:
    - ``/debate <question>`` — full 3-round debate
    - ``/debate quick <question>`` — skip critique round
    """
    if not args:
        console.print(
            "[yellow]Usage:[/] /debate <question or decision>\n"
            "[dim]  /debate quick <question> — skip critique round\n"
            "  /debates — list saved debates[/dim]"
        )
        return "continue"

    # Check for "quick" prefix
    quick_mode = False
    question = args
    if args.lower().startswith("quick "):
        quick_mode = True
        question = args[6:].strip()

    if not question:
        console.print("[yellow]Please provide a question to debate.[/]")
        return "continue"

    try:
        from prism.intelligence.debate import (
            DebateConfig,
            debate,
        )

        cfg = DebateConfig(quick_mode=quick_mode)

        console.print("[dim]Starting multi-model debate...[/]")
        if quick_mode:
            console.print("[dim]Quick mode: skipping critique round.[/]")

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
                    Markdown(response) if response else Text(
                        "(no response)", style="dim"
                    ),
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
            Markdown("\n\n".join(summary_parts)) if summary_parts else Text(
                "(no synthesis)", style="dim"
            ),
            border_style="green",
        ))

        # Cost summary
        console.print(
            f"\n[dim]Total cost: ${result.total_cost:.4f}[/dim]"
        )

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("debate_error", error=str(exc))
        console.print(f"[red]Debate error:[/] {exc}")
    return "continue"


def _display_blast_report(
    console: Console,
    report: Any,
    analyzer: Any,
    risk_level_cls: Any,
) -> None:
    """Render a detailed blast-radius report to the console.

    Displays a rich-formatted report with header, files grouped by risk
    level, test recommendations, missing tests with priority labels,
    execution order, and effort estimate.

    Args:
        console: Rich console instance for output.
        report: The :class:`ImpactReport` to display.
        analyzer: The :class:`BlastRadiusAnalyzer` that produced the report.
        risk_level_cls: The :class:`RiskLevel` constants class.
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
    console.print(Panel(
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
        table = Table(
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
        console.print(table)

    # --- Medium areas ---
    if medium_files:
        table = Table(
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
        console.print(table)

    # --- Low areas ---
    if low_files:
        table = Table(
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
        console.print(table)

    # --- Test recommendations ---
    if report.recommended_test_order:
        test_paths = " ".join(report.recommended_test_order)
        console.print(Panel(
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
        console.print(Panel(
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
        console.print(Panel(
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
    console.print(Panel(
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
        console.print(
            f"\n[dim]Report saved to {report_path}[/dim]"
        )


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
            RiskLevel,
        )

        analyzer = BlastRadiusAnalyzer(
            project_root=settings.project_root,
        )
        report = analyzer.analyze(description=args)

        _display_blast_report(console, report, analyzer, RiskLevel)

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
    """Test gap analysis — find untested code.

    Subcommands:
        /gaps                — full analysis
        /gaps critical       — show only critical gaps
        /gaps high           — show only high gaps
        /gaps generate       — generate top 5 test stubs
    """
    try:
        from prism.intelligence.test_gaps import GapRisk, TestGapHunter

        hunter = TestGapHunter(
            project_root=settings.project_root,
        )

        sub = args.lower().strip()

        # Handle "generate" subcommand
        if sub == "generate":
            report = hunter.analyze()
            gen_gaps = [
                g for g in report.gaps
                if g.risk_level in (GapRisk.CRITICAL, GapRisk.HIGH)
            ]
            if not gen_gaps:
                gen_gaps = report.gaps
            gen_gaps = gen_gaps[:5]

            if not gen_gaps:
                console.print(
                    "[green]No test gaps to generate tests for.[/]"
                )
                return "continue"

            from pathlib import Path

            generated = hunter.generate_tests(gen_gaps, count=5)
            for test_path, content in generated.items():
                out = Path(test_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(content)
                console.print(
                    f"[green]Generated:[/] {test_path}"
                )
            return "continue"

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
        filter_level = sub
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
            table.add_column("Scenarios", justify="right")
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
                    str(len(gap.scenarios)),
                    gap.estimated_effort,
                    gap.reason[:40],
                )
            console.print(table)

            # Show scenarios for each gap
            for gap in gaps[:30]:
                if gap.scenarios:
                    console.print(
                        f"  [dim]{gap.function_name}:[/dim]"
                    )
                    for scenario in gap.scenarios:
                        console.print(
                            f"    [dim]- {scenario}[/dim]"
                        )

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
    """Dependency health report with sub-actions: status, audit, unused."""
    try:
        from prism.intelligence.deps import DependencyMonitor

        monitor = DependencyMonitor(
            project_root=settings.project_root,
        )

        sub = args.lower().strip() if args else "status"

        if sub in ("", "status"):
            _repl_deps_status(monitor, console)
        elif sub == "audit":
            _repl_deps_audit(monitor, console)
        elif sub == "unused":
            _repl_deps_unused(monitor, console)
        else:
            console.print(
                f"[yellow]Unknown deps sub-command:[/] {sub}\n"
                "[dim]Usage: /deps [status|audit|unused][/dim]"
            )

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("deps_error", error=str(exc))
        console.print(f"[red]Deps error:[/] {exc}")
    return "continue"


def _repl_deps_status(
    monitor: Any,
    console: Console,
) -> None:
    """Display full dependency status table in the REPL."""
    report = monitor.get_status()

    console.print(Panel(
        f"Total: {report.total_deps} | "
        f"Outdated: {report.outdated} | "
        f"Vulnerable: {report.vulnerable} | "
        f"Unused: {report.unused}",
        title="[bold]Dependency Health[/bold]",
        border_style="blue",
    ))

    if report.dependencies:
        table = Table(title="Dependencies")
        table.add_column("Package", style="cyan")
        table.add_column("Current")
        table.add_column("Latest")
        table.add_column("Ecosystem")
        table.add_column("Security")
        table.add_column("Risk")

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

    if report.unused_deps:
        console.print(
            f"\n[yellow]Potentially unused "
            f"({len(report.unused_deps)}):[/]"
        )
        for name in report.unused_deps:
            console.print(f"  {name}")


def _repl_deps_audit(
    monitor: Any,
    console: Console,
) -> None:
    """Display security-only audit in the REPL."""
    report = monitor.get_status()

    if not report.vulnerabilities:
        console.print("[green]No vulnerabilities found.[/green]")
        return

    vuln_table = Table(
        title="Security Audit Results",
        border_style="red",
    )
    vuln_table.add_column("Package", style="red bold")
    vuln_table.add_column("CVE")
    vuln_table.add_column("Severity")
    vuln_table.add_column("Current")
    vuln_table.add_column("Fix Version")

    for v in report.vulnerabilities:
        vuln_table.add_row(
            v.package,
            v.cve_id,
            v.severity.value.upper(),
            v.current_version,
            v.fixed_version or "-",
        )

    console.print(vuln_table)
    console.print(
        f"\n[bold]Total vulnerabilities:[/] "
        f"{len(report.vulnerabilities)}"
    )


def _repl_deps_unused(
    monitor: Any,
    console: Console,
) -> None:
    """Display unused dependencies in the REPL."""
    report = monitor.get_status()

    if not report.unused_deps:
        console.print(
            "[green]No unused dependencies detected.[/green]"
        )
        return

    console.print(
        f"\n[yellow]Potentially unused dependencies "
        f"({len(report.unused_deps)}):[/yellow]\n"
    )
    for name in report.unused_deps:
        console.print(f"  [dim]-[/dim] {name}")

    console.print(
        "\n[dim]Note: Build tools and test runners "
        "are excluded from this check.[/dim]"
    )


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

        elif sub == "mermaid":
            state = mapper.generate()
            mermaid = mapper.generate_mermaid(state)
            console.print(Panel(
                mermaid,
                title="[bold]Mermaid Dependency Diagram[/bold]",
                border_style="cyan",
            ))

        elif sub == "check":
            current = mapper.generate()
            violations = mapper.detect_drift(current)
            boundary_violations = [
                v for v in violations
                if v.violation_type == "boundary_crossing"
            ]

            if not boundary_violations:
                console.print(
                    "[green]No boundary violations detected.[/]"
                )
                return "continue"

            table = Table(
                title="Boundary Violations",
            )
            table.add_column("Source", style="cyan")
            table.add_column("Target", style="red")
            table.add_column("Severity", justify="center")
            table.add_column("Description")

            for v in boundary_violations:
                table.add_row(
                    v.source,
                    v.target or "-",
                    f"[red]{v.severity.upper()}[/red]",
                    v.description[:60],
                )
            console.print(table)

        elif sub == "diff":
            diff_text = mapper.get_diff()
            console.print(Panel(
                diff_text,
                title="[bold]Architecture Diff[/bold]",
                border_style="yellow",
            ))

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
        import json as _json
        from pathlib import Path as _Path

        from prism.intelligence.debug_memory import (
            BugFingerprint,
            DebugMemory,
        )

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

        elif sub in ("bugs", "list"):
            records = memory.browse_fixes(limit=100)
            if not records:
                console.print(
                    "[dim]No fixes stored yet.[/dim]"
                )
                memory.close()
                return "continue"

            table = Table(title="All Stored Fixes")
            table.add_column("ID", style="dim", width=5)
            table.add_column("Error Type", style="red")
            table.add_column("Fix Pattern")
            table.add_column("Project")
            table.add_column("Date", width=10)

            for r in records:
                table.add_row(
                    str(r.id),
                    r.error_type,
                    r.fix_pattern[:40],
                    r.project,
                    r.timestamp[:10],
                )
            console.print(table)

        elif sub == "forget" and query:
            fix_id = query.strip()
            try:
                deleted = memory.forget(int(fix_id))
            except ValueError:
                console.print(
                    f"[red]Invalid fix ID:[/] {fix_id}"
                )
                memory.close()
                return "continue"

            if deleted:
                console.print(
                    f"[green]Fix #{fix_id} deleted.[/]"
                )
            else:
                console.print(
                    f"[yellow]Fix #{fix_id} not found.[/]"
                )

        elif sub == "export":
            records = memory.browse_fixes(limit=10000)
            export_path = (
                settings.prism_home
                / "debug_memory_export.json"
            )
            export_data = [
                {
                    "id": r.id,
                    "fingerprint": r.fingerprint,
                    "error_type": r.error_type,
                    "stack_pattern": r.stack_pattern,
                    "fix_pattern": r.fix_pattern,
                    "fix_diff": r.fix_diff,
                    "confidence": r.confidence,
                    "project": r.project,
                    "model_used": r.model_used,
                    "timestamp": r.timestamp,
                    "language": r.language,
                    "framework": r.framework,
                    "affected_files_json": (
                        r.affected_files_json
                    ),
                    "affected_functions_json": (
                        r.affected_functions_json
                    ),
                }
                for r in records
            ]
            export_path.parent.mkdir(
                parents=True, exist_ok=True,
            )
            export_path.write_text(
                _json.dumps(export_data, indent=2),
                encoding="utf-8",
            )
            console.print(
                f"[green]Exported {len(records)} fixes to "
                f"{export_path}[/]"
            )

        elif sub == "import" and query:
            import_path = _Path(query).expanduser().resolve()
            if not import_path.is_file():
                console.print(
                    f"[red]File not found:[/] {import_path}"
                )
                memory.close()
                return "continue"

            try:
                raw = import_path.read_text(encoding="utf-8")
                data = _json.loads(raw)
            except (
                _json.JSONDecodeError, OSError,
            ) as err:
                console.print(
                    f"[red]Import error:[/] {err}"
                )
                memory.close()
                return "continue"

            if not isinstance(data, list):
                console.print(
                    "[red]Invalid format: expected a JSON "
                    "array of fix records.[/]"
                )
                memory.close()
                return "continue"

            imported = 0
            for entry in data:
                try:
                    affected_files: list[str] = (
                        _json.loads(
                            entry.get(
                                "affected_files_json",
                                "[]",
                            ),
                        )
                        if isinstance(
                            entry.get(
                                "affected_files_json"
                            ),
                            str,
                        )
                        else entry.get(
                            "affected_files_json", []
                        )
                    )
                    affected_fns: list[str] = (
                        _json.loads(
                            entry.get(
                                "affected_functions_json",
                                "[]",
                            ),
                        )
                        if isinstance(
                            entry.get(
                                "affected_functions_json",
                            ),
                            str,
                        )
                        else entry.get(
                            "affected_functions_json", []
                        )
                    )
                    fp = BugFingerprint(
                        error_type=entry.get(
                            "error_type", "Unknown"
                        ),
                        stack_pattern=entry.get(
                            "stack_pattern", ""
                        ),
                        affected_files=affected_files,
                        affected_functions=affected_fns,
                        language=entry.get(
                            "language", ""
                        ),
                        framework=entry.get(
                            "framework", ""
                        ),
                        fingerprint_hash=entry.get(
                            "fingerprint", ""
                        ),
                    )
                    memory.store_fix(
                        fingerprint=fp,
                        fix_pattern=entry.get(
                            "fix_pattern", ""
                        ),
                        fix_diff=entry.get(
                            "fix_diff", ""
                        ),
                        project=entry.get("project", ""),
                        model_used=entry.get(
                            "model_used", ""
                        ),
                        confidence=float(
                            entry.get("confidence", 0.5)
                        ),
                    )
                    imported += 1
                except (
                    KeyError, TypeError, ValueError,
                ) as err:
                    logger.debug(
                        "debug_memory.import_skip",
                        error=str(err),
                    )
                    continue

            console.print(
                f"[green]Imported {imported} fixes from "
                f"{import_path}[/]"
            )

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


def _cmd_ignore(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Manage ``.prismignore`` patterns.

    Subcommands:
        ``/ignore`` or ``/ignore list`` -- show all current patterns.
        ``/ignore add <pattern>``       -- add a pattern to .prismignore.
        ``/ignore check <file>``        -- check if a file is ignored.
        ``/ignore create``              -- create default .prismignore.
    """
    from prism.security.prismignore import PrismIgnore

    try:
        prismignore = PrismIgnore(settings.project_root)
    except Exception as exc:
        logger.debug("prismignore_load_error", error=str(exc))
        console.print(f"[red]Failed to load .prismignore:[/] {exc}")
        return "continue"

    parts = args.strip().split(maxsplit=1)
    sub = parts[0].lower() if parts and parts[0] else "list"
    sub_args = parts[1].strip() if len(parts) > 1 else ""

    # --- /ignore list (default) ---
    if sub == "list":
        patterns = prismignore.patterns
        if not patterns:
            console.print(
                "[dim]No active patterns in .prismignore.[/dim]"
            )
            return "continue"

        table = Table(
            title=".prismignore patterns",
            show_header=False,
            box=None,
            padding=(0, 2),
            title_style="bold cyan",
        )
        table.add_column(
            "#", style="dim", justify="right", width=4,
        )
        table.add_column("Pattern", style="cyan")
        for i, pattern in enumerate(patterns, 1):
            table.add_row(str(i), pattern)
        console.print(table)
        console.print(
            f"\n[dim]{len(patterns)} active pattern(s) "
            f"from {prismignore.file_path}[/dim]"
        )
        return "continue"

    # --- /ignore add <pattern> ---
    if sub == "add":
        if not sub_args:
            console.print(
                "[yellow]Usage:[/] /ignore add <pattern>\n"
                "[dim]Example: /ignore add *.secret[/dim]"
            )
            return "continue"
        pattern = sub_args.strip()
        if pattern in prismignore.patterns:
            console.print(
                f"[dim]Pattern already exists:[/dim] {pattern}"
            )
            return "continue"
        prismignore.add_pattern(pattern)
        console.print(
            f"[green]Added pattern:[/] {pattern}"
        )
        return "continue"

    # --- /ignore check <file> ---
    if sub == "check":
        if not sub_args:
            console.print(
                "[yellow]Usage:[/] /ignore check <file>\n"
                "[dim]Example: /ignore check .env.local[/dim]"
            )
            return "continue"
        filepath = sub_args.strip()
        ignored = prismignore.is_ignored(filepath)
        if ignored:
            # Find the matching pattern by iterating compiled
            rel = prismignore._get_relative(filepath)
            matching = ""
            if rel is not None:
                for pat, negated in prismignore._compiled:
                    if prismignore._matches(rel, pat):
                        matching = pat if not negated else ""
            if matching:
                console.print(
                    f"[red]IGNORED[/] {filepath}"
                    f" — matched pattern: [cyan]{matching}[/]"
                )
            else:
                console.print(
                    f"[red]IGNORED[/] {filepath}"
                )
        else:
            console.print(
                f"[green]NOT IGNORED[/] {filepath}"
            )
        return "continue"

    # --- /ignore create ---
    if sub == "create":
        if prismignore.file_path.is_file():
            console.print(
                "[yellow].prismignore already exists at:[/] "
                f"{prismignore.file_path}"
            )
            return "continue"
        created = prismignore.create_default()
        console.print(
            f"[green]Created default .prismignore:[/] {created}"
        )
        return "continue"

    # Unknown subcommand
    console.print(
        "[yellow]Usage:[/] /ignore [add|list|check|create] "
        "[pattern|file]\n"
        "[dim]Subcommands: list, add <pattern>, "
        "check <file>, create[/dim]"
    )
    return "continue"


def _cmd_aei(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """AEI statistics, reset, or explain a specific fingerprint.

    Subcommands:
        ``/aei`` or ``/aei stats`` -- show aggregate statistics.
        ``/aei reset`` -- clear all attempt history.
        ``/aei explain <hash>`` -- explain reasoning for a fingerprint.
    """
    try:
        from prism.intelligence.aei import (
            AdaptiveExecutionIntelligence,
            ErrorFingerprint,
        )

        db_path = settings.prism_home / "aei" / "attempts.db"
        aei = AdaptiveExecutionIntelligence(
            db_path=db_path,
            repo=str(settings.project_root),
        )

        sub = args.strip()
        sub_lower = sub.lower()

        # --- /aei reset ---
        if sub_lower == "reset":
            deleted = aei.reset()
            aei.close()
            console.print(
                f"[green]AEI history cleared: "
                f"{deleted} attempt(s) removed.[/]"
            )
            return "continue"

        # --- /aei explain <hash> ---
        if sub_lower.startswith("explain"):
            fp_hash = sub[len("explain"):].strip()
            if not fp_hash:
                aei.close()
                console.print(
                    "[yellow]Usage:[/] /aei explain <fingerprint_hash>"
                )
                return "continue"

            fp = ErrorFingerprint(
                error_type="",
                stack_pattern="",
                file_path="",
                function_name="",
                fingerprint_hash=fp_hash,
            )
            explanation = aei.explain(fp)
            aei.close()
            console.print(Panel(
                explanation,
                title="[bold]AEI Explanation[/bold]",
                border_style="blue",
            ))
            return "continue"

        # --- /aei stats (default) ---
        stats = aei.get_stats()
        aei.close()

        table = Table(
            title="AEI Statistics",
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row(
            "Total attempts", str(stats.total_attempts),
        )
        table.add_row(
            "Successes", str(stats.total_successes),
        )
        table.add_row(
            "Failures", str(stats.total_failures),
        )
        table.add_row(
            "Success rate",
            f"{stats.success_rate:.1%}",
        )
        table.add_row(
            "Escalation count",
            str(stats.escalation_count),
        )

        if stats.strategies_used:
            strat_parts: list[str] = []
            for name, count in sorted(
                stats.strategies_used.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                strat_parts.append(f"{name}: {count}")
            table.add_row(
                "Strategies used",
                ", ".join(strat_parts),
            )

        if stats.top_error_types:
            err_parts: list[str] = []
            for etype, count in stats.top_error_types[:5]:
                err_parts.append(f"{etype}: {count}")
            table.add_row(
                "Top error types",
                ", ".join(err_parts),
            )

        console.print(Panel(table, border_style="blue"))

    except Exception as exc:
        logger.debug("aei_command_error", error=str(exc))
        console.print(f"[red]AEI error:[/] {exc}")
    return "continue"


def _cmd_blame(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Causal blame trace with optional git bisect.

    Subcommands:
        ``/blame <description>`` -- run a blame trace.
        ``/blame --test "cmd" --good abc123 <description>``
            -- run with bisect.
        ``/blame list`` -- list saved blame reports.
    """
    if not args:
        console.print(
            "[yellow]Usage:[/] /blame <description> | "
            "/blame --test \"cmd\" --good abc123 <desc> | "
            "/blame list"
        )
        return "continue"

    try:
        from prism.intelligence.blame import CausalBlameTracer

        tracer = CausalBlameTracer(
            project_root=settings.project_root,
        )

        sub_lower = args.strip().lower()

        # --- /blame list ---
        if sub_lower == "list":
            reports = tracer.list_reports()
            if not reports:
                console.print(
                    "[dim]No blame reports found.[/dim]"
                )
                return "continue"

            table = Table(title="Blame Reports")
            table.add_column("#", style="dim", width=4)
            table.add_column("Commit", style="cyan", width=10)
            table.add_column("Date")
            table.add_column("Bug Description")
            table.add_column("Confidence", justify="right")

            for i, report_path in enumerate(reports[:20], 1):
                report = tracer.load_report(report_path)
                table.add_row(
                    str(i),
                    report.breaking_commit[:8],
                    report.created_at[:10],
                    report.bug_description[:50],
                    f"{report.confidence:.0%}",
                )

            console.print(table)
            return "continue"

        # --- Parse --test and --good flags ---
        test_command: str | None = None
        good_commit: str | None = None
        remaining = args.strip()

        if "--test" in remaining:
            import shlex

            parts = shlex.split(remaining)
            cleaned_parts: list[str] = []
            i = 0
            while i < len(parts):
                if parts[i] == "--test" and i + 1 < len(parts):
                    test_command = parts[i + 1]
                    i += 2
                elif parts[i] == "--good" and i + 1 < len(parts):
                    good_commit = parts[i + 1]
                    i += 2
                else:
                    cleaned_parts.append(parts[i])
                    i += 1
            remaining = " ".join(cleaned_parts)

        if not remaining:
            console.print(
                "[yellow]Please provide a bug description.[/]"
            )
            return "continue"

        console.print(
            "[dim]Running blame trace...[/dim]"
        )
        report = tracer.trace(
            bug_description=remaining,
            test_command=test_command,
            good_commit=good_commit,
        )

        # Display the report
        confidence_color = (
            "green" if report.confidence >= 0.7
            else "yellow" if report.confidence >= 0.4
            else "red"
        )

        summary_lines = [
            f"Breaking commit: {report.breaking_commit[:12]}",
            f"Author: {report.breaking_author}",
            f"Date: {report.breaking_date}",
            f"Message: {report.breaking_message}",
        ]

        if report.bisect_steps > 0:
            summary_lines.append(
                f"Bisect steps: {report.bisect_steps}"
            )

        console.print(Panel(
            "\n".join(summary_lines),
            title="[bold]Blame Report[/bold]",
            border_style="blue",
        ))

        if report.affected_files:
            file_table = Table(title="Affected Files")
            file_table.add_column("File", style="cyan")
            for af in report.affected_files[:20]:
                file_table.add_row(af)
            console.print(file_table)

        if report.causal_narrative:
            console.print(Panel(
                report.causal_narrative,
                title="[bold]Causal Narrative[/bold]",
                border_style="yellow",
            ))

        console.print(
            f"[{confidence_color}]Confidence: "
            f"{report.confidence:.0%}[/{confidence_color}]"
        )

        if report.related_tests:
            console.print(
                f"\n[dim]Related tests: "
                f"{', '.join(report.related_tests[:5])}[/dim]"
            )

    except Exception as exc:
        logger.debug("blame_error", error=str(exc))
        console.print(f"[red]Blame error:[/] {exc}")
    return "continue"


def _cmd_context(
    args: str,
    console: Console,
    settings: Settings,
    state: _SessionState,
    **_: Any,
) -> str:
    """Smart context budget allocation.

    Subcommands:
        ``/context``          -- show current context allocation.
        ``/context show``     -- show current context allocation.
        ``/context add <f>``  -- force-include a file.
        ``/context drop <f>`` -- force-exclude a file.
        ``/context stats``    -- show efficiency metrics.
    """
    try:
        from prism.intelligence.context_budget import (
            SmartContextBudgetManager,
        )

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "show"
        extra = parts[1].strip() if len(parts) > 1 else ""

        manager = SmartContextBudgetManager(
            project_root=settings.project_root,
        )

        if sub == "add":
            if not extra:
                console.print(
                    "[yellow]Usage:[/] /context add <file>"
                )
                return "continue"
            manager.add_file(extra)
            if extra not in state.active_files:
                state.active_files.append(extra)
            console.print(
                f"[green]Force-included:[/] {extra} "
                "(score set to 1.0)"
            )
            return "continue"

        if sub == "drop":
            if not extra:
                console.print(
                    "[yellow]Usage:[/] /context drop <file>"
                )
                return "continue"
            manager.drop_file(extra)
            if extra in state.active_files:
                state.active_files.remove(extra)
            console.print(
                f"[red]Force-excluded:[/] {extra} "
                "(score set to 0.0)"
            )
            return "continue"

        if sub == "stats":
            try:
                from prism.db.database import Database

                db = Database(settings.db_path)
                stats_manager = SmartContextBudgetManager(
                    project_root=settings.project_root,
                    db=db,
                )
                stats = stats_manager.get_efficiency_stats()

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
                logger.debug(
                    "context_stats_error", error=str(exc),
                )
                console.print(
                    "[dim]No efficiency data yet.[/dim]"
                )
            return "continue"

        # show (default)
        allocation = manager.allocate(
            task_description="current session context",
            available_files=list(state.active_files),
        )
        display = SmartContextBudgetManager.generate_context_display(
            allocation,
        )
        console.print(Panel(
            display,
            title="[bold]Smart Context Budget[/bold]",
            border_style="blue",
        ))

    except Exception as exc:
        logger.debug("context_error", error=str(exc))
        console.print(f"[red]Context error:[/] {exc}")
    return "continue"


# ======================================================================
# /why — Code archaeology investigation
# ======================================================================


def _cmd_why(
    args: str,
    console: Console,
    settings: Settings,
    **_: Any,
) -> str:
    """Investigate code history and evolution — temporal archaeology."""
    if not args:
        console.print(
            "[yellow]Usage:[/] /why <file_path[:line]> or "
            "/why <function_name>\n"
            "[dim]Investigate why code looks the way it does.[/dim]"
        )
        return "continue"

    try:
        from prism.intelligence.archaeologist import (
            investigate,
        )

        console.print("[dim]Investigating code history...[/dim]")

        report = investigate(
            target=args,
            project_root=settings.project_root,
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
        logger.debug("why_error", error=str(exc))
        console.print(f"[red]Archaeology error:[/] {exc}")
    return "continue"


# ======================================================================
# /debates — List saved debate reports
# ======================================================================


def _cmd_debates(
    args: str,
    console: Console,
    **_: Any,
) -> str:
    """List all saved debate reports."""
    try:
        from prism.intelligence.debate import list_debates

        debates = list_debates()

        if not debates:
            console.print("[dim]No saved debates found.[/dim]")
            return "continue"

        table = Table(title="Saved Debates")
        table.add_column("File", style="cyan")
        table.add_column("Size", justify="right")

        for debate_path in debates[:20]:
            size = debate_path.stat().st_size
            size_str = (
                f"{size / 1024:.1f} KB"
                if size > 1024
                else f"{size} B"
            )
            table.add_row(debate_path.name, size_str)

        console.print(table)

        if len(debates) > 20:
            console.print(
                f"[dim]... and {len(debates) - 20} more[/dim]"
            )

    except Exception as exc:
        logger.debug("debates_list_error", error=str(exc))
        console.print(f"[red]Error listing debates:[/] {exc}")
    return "continue"


def _cmd_autotest(
    args: str,
    console: Console,
    state: _SessionState,
    **_: Any,
) -> str:
    """Run auto-test: discover and execute relevant tests for changed files."""
    try:
        if not state.tool_registry:
            console.print("[yellow]Tool registry not initialized.[/]")
            return "continue"

        tool = state.tool_registry.get_tool("auto_test")
        changed_files = [f.strip() for f in args.split() if f.strip()] if args else []

        if not changed_files:
            # Auto-detect from git
            import subprocess
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            changed_files = [
                f for f in result.stdout.strip().splitlines() if f.strip()
            ]
            if not changed_files:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "--cached"],
                    capture_output=True, text=True, timeout=10,
                )
                changed_files = [
                    f for f in result.stdout.strip().splitlines() if f.strip()
                ]

        if not changed_files:
            console.print("[dim]No changed files detected. Pass files explicitly: /test file1.py file2.py[/dim]")
            return "continue"

        console.print(f"[dim]Running tests for {len(changed_files)} changed file(s)...[/dim]")
        result = tool.execute({"changed_files": changed_files})

        if result.success:
            console.print(Panel(result.output, title="Test Results", border_style="green"))
        else:
            console.print(Panel(result.error or result.output, title="Test Results", border_style="red"))

    except Exception as exc:
        logger.debug("autotest_error", error=str(exc))
        console.print(f"[red]Error:[/] {exc}")
    return "continue"


def _cmd_quality(
    args: str,
    console: Console,
    state: _SessionState,
    **_: Any,
) -> str:
    """Run quality gate: lint + security + type checks on files."""
    try:
        if not state.tool_registry:
            console.print("[yellow]Tool registry not initialized.[/]")
            return "continue"

        tool = state.tool_registry.get_tool("quality_gate")
        changed_files = [f.strip() for f in args.split() if f.strip()] if args else []

        if not changed_files:
            # Auto-detect from git
            import subprocess
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            changed_files = [
                f for f in result.stdout.strip().splitlines() if f.strip()
            ]

        if not changed_files:
            console.print("[dim]No changed files. Pass files explicitly: /quality file1.py[/dim]")
            return "continue"

        console.print(f"[dim]Running quality checks on {len(changed_files)} file(s)...[/dim]")
        result = tool.execute({"changed_files": changed_files, "action": "check"})

        if result.success:
            console.print(Panel(result.output, title="Quality Gate", border_style="green"))
        else:
            console.print(Panel(result.error or result.output, title="Quality Gate", border_style="yellow"))

    except Exception as exc:
        logger.debug("quality_gate_error", error=str(exc))
        console.print(f"[red]Error:[/] {exc}")
    return "continue"


def _cmd_optimize(
    args: str,
    console: Console,
    state: _SessionState,
    **_: Any,
) -> str:
    """Show cost optimization recommendations."""
    try:
        if not state.tool_registry:
            console.print("[yellow]Tool registry not initialized.[/]")
            return "continue"

        try:
            tool = state.tool_registry.get_tool("cost_optimizer")
        except Exception:
            console.print(
                "[yellow]Cost optimizer requires usage history. "
                "Make some API calls first.[/]"
            )
            return "continue"

        action = args.strip() if args else "recommend"
        result = tool.execute({"action": action})

        if result.success:
            console.print(Panel(
                result.output,
                title="Cost Optimization",
                border_style="cyan",
            ))
        else:
            console.print(f"[yellow]{result.error}[/]")

    except Exception as exc:
        logger.debug("optimize_error", error=str(exc))
        console.print(f"[red]Error:[/] {exc}")
    return "continue"


def _cmd_memory(
    args: str,
    console: Console,
    **_: Any,
) -> str:
    """Manage project memory (PRISM_MEMORY.md)."""
    from pathlib import Path as _Path

    try:
        from prism.context.memory import ProjectMemory

        memory = ProjectMemory(_Path.cwd())
        parts = args.split(maxsplit=2) if args else []
        sub = parts[0] if parts else "show"

        if sub == "show":
            facts = memory.get_facts()
            if not facts:
                console.print("[dim]No project memory facts stored. Use /memory add <key> <value>[/dim]")
                return "continue"

            table = Table(title="Project Memory")
            table.add_column("Key", style="cyan")
            table.add_column("Value")

            for key, value in sorted(facts.items()):
                table.add_row(key, value)

            console.print(table)

        elif sub == "add":
            if len(parts) < 3:
                console.print("[yellow]Usage: /memory add <key> <value>[/]")
                return "continue"
            key = parts[1]
            value = parts[2]
            memory.add_fact(key, value)
            console.print(f"[green]Saved:[/] {key} = {value}")

        elif sub == "remove":
            if len(parts) < 2:
                console.print("[yellow]Usage: /memory remove <key>[/]")
                return "continue"
            key = parts[1]
            if memory.remove_fact(key):
                console.print(f"[green]Removed:[/] {key}")
            else:
                console.print(f"[yellow]Key not found:[/] {key}")

        elif sub == "clear":
            memory.clear()
            console.print("[green]Project memory cleared.[/]")

        else:
            console.print(
                "[yellow]Usage: /memory [show|add|remove|clear] [key] [value][/]"
            )

    except Exception as exc:
        logger.debug("memory_cmd_error", error=str(exc))
        console.print(f"[red]Error:[/] {exc}")
    return "continue"


def _cmd_swarm(
    args: str,
    console: Console,
    state: _SessionState,
    settings: Settings,
    **_: Any,
) -> str:
    """Multi-model collaborative task execution via swarm orchestration.

    Subcommands:
        ``plan <goal>``  -- Decompose + research + plan only (no execution).
        ``status``       -- Show current swarm execution status.
        ``<goal>``       -- Full seven-phase orchestration pipeline.
    """
    try:
        from prism.orchestrator.swarm import (
            SwarmConfig,
            SwarmOrchestrator,
            SwarmPhase,
            SwarmPlan,
            TaskStatus,
        )

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else ""
        extra = parts[1].strip() if len(parts) > 1 else ""

        # --- /swarm status ---
        if sub == "status":
            console.print(
                "[dim]No swarm execution in progress. "
                "Use [cyan]/swarm <goal>[/cyan] to start one.[/dim]"
            )
            return "continue"

        # --- /swarm plan <goal> ---
        if sub == "plan":
            if not extra:
                console.print(
                    "[yellow]Usage:[/] /swarm plan <goal>"
                )
                return "continue"

            if state.completion_engine is None or state.registry is None:
                console.print(
                    "[red]Swarm requires initialized providers. "
                    "Configure API keys first.[/red]"
                )
                return "continue"

            swarm_config = SwarmConfig(
                use_debate=True,
                use_moa=True,
                use_cascade=True,
                use_tools=state.tool_registry is not None,
            )
            orchestrator = SwarmOrchestrator(
                engine=state.completion_engine,
                registry=state.registry,
                config=swarm_config,
                tool_registry=state.tool_registry,
            )

            console.print(
                Panel(
                    f"[bold cyan]Swarm Plan[/bold cyan]\n"
                    f"[dim]{extra}[/dim]",
                    title="Swarm Orchestrator",
                    border_style="cyan",
                )
            )

            with console.status("[bold cyan]Decomposing task..."):
                plan = asyncio.get_event_loop().run_until_complete(
                    orchestrator._phase_decompose(
                        SwarmPlan(goal=extra),
                        extra,
                        {},
                    )
                )

            with console.status("[bold cyan]Researching..."):
                plan = asyncio.get_event_loop().run_until_complete(
                    orchestrator._phase_research(plan, extra, {})
                )

            with console.status("[bold cyan]Planning..."):
                plan = asyncio.get_event_loop().run_until_complete(
                    orchestrator._phase_plan(plan, extra)
                )

            with console.status("[bold cyan]Reviewing plan..."):
                plan = asyncio.get_event_loop().run_until_complete(
                    orchestrator._phase_review(plan, extra)
                )

            # Display the plan as a Rich table
            task_table = Table(
                title="Decomposed Tasks",
                show_header=True,
                header_style="bold magenta",
            )
            task_table.add_column("#", style="dim", width=4)
            task_table.add_column("Task", min_width=30)
            task_table.add_column("Complexity", width=10)
            task_table.add_column("Files", width=25)

            for idx, task in enumerate(plan.tasks, 1):
                files_str = ", ".join(task.files_changed[:3]) or "-"
                task_table.add_row(
                    str(idx),
                    task.description[:80],
                    task.complexity,
                    files_str,
                )
            console.print(task_table)

            if plan.plan_text:
                console.print(
                    Panel(
                        Markdown(plan.plan_text[:3000]),
                        title="Execution Plan",
                        border_style="green",
                    )
                )

            if plan.review_notes:
                console.print(
                    Panel(
                        plan.review_notes[:2000],
                        title="Review Notes",
                        border_style="yellow",
                    )
                )

            # Cost summary
            cost_table = Table(title="Phase Costs", show_header=True)
            cost_table.add_column("Phase", style="cyan")
            cost_table.add_column("Cost (USD)", justify="right")
            for phase_name, cost in plan.phase_costs.items():
                cost_table.add_row(phase_name, f"${cost:.4f}")
            cost_table.add_row(
                "[bold]Total[/bold]",
                f"[bold]${plan.total_cost:.4f}[/bold]",
            )
            console.print(cost_table)

            console.print(
                "[dim]Plan-only mode. Use "
                "[cyan]/swarm " + extra[:50] + "[/cyan] "
                "to execute.[/dim]"
            )
            return "continue"

        # --- /swarm <goal> (default: full orchestration) ---
        if not args.strip():
            console.print(
                "[yellow]Usage:[/] /swarm <goal>\n"
                "[dim]Subcommands: plan, status[/dim]"
            )
            return "continue"

        goal_raw = args.strip()

        # Parse optional --budget flag: /swarm --budget 5.0 <goal>
        budget_override: float | None = None
        if goal_raw.startswith("--budget"):
            parts = goal_raw.split(maxsplit=2)
            if len(parts) >= 3:
                try:
                    budget_override = float(parts[1])
                except ValueError:
                    console.print(
                        "[red]Invalid --budget value. "
                        "Usage: /swarm --budget 5.0 <goal>[/red]"
                    )
                    return "continue"
                goal_raw = parts[2]
            else:
                console.print(
                    "[yellow]Usage:[/] /swarm --budget <usd> <goal>"
                )
                return "continue"

        goal = goal_raw

        if state.completion_engine is None or state.registry is None:
            console.print(
                "[red]Swarm requires initialized providers. "
                "Configure API keys first.[/red]"
            )
            return "continue"

        swarm_config = SwarmConfig(
            use_debate=True,
            use_moa=True,
            use_cascade=True,
            use_tools=state.tool_registry is not None,
            total_budget=budget_override if budget_override is not None else 1.0,
            auto_scale_budget=budget_override is None,
        )
        orchestrator = SwarmOrchestrator(
            engine=state.completion_engine,
            registry=state.registry,
            config=swarm_config,
            tool_registry=state.tool_registry,
        )

        console.print(
            Panel(
                f"[bold cyan]Swarm Orchestration[/bold cyan]\n"
                f"[dim]{goal}[/dim]",
                title="Swarm Orchestrator",
                border_style="cyan",
            )
        )

        with console.status("[bold cyan]Running swarm pipeline..."):
            plan = asyncio.get_event_loop().run_until_complete(
                orchestrator.orchestrate(goal)
            )

        # Display results: task table
        result_table = Table(
            title="Swarm Results",
            show_header=True,
            header_style="bold magenta",
        )
        result_table.add_column("#", style="dim", width=4)
        result_table.add_column("Task", min_width=30)
        result_table.add_column("Model", width=20)
        result_table.add_column("Status", width=12)
        result_table.add_column("Cost", width=10, justify="right")

        for idx, task in enumerate(plan.tasks, 1):
            status_color = {
                TaskStatus.COMPLETED: "green",
                TaskStatus.FAILED: "red",
                TaskStatus.RUNNING: "yellow",
                TaskStatus.PENDING: "dim",
            }.get(task.status, "white")
            result_table.add_row(
                str(idx),
                task.description[:60],
                task.assigned_model or "-",
                f"[{status_color}]{task.status}[/{status_color}]",
                f"${plan.phase_costs.get(SwarmPhase.EXECUTE, 0.0):.4f}",
            )
        console.print(result_table)

        # Files changed summary
        all_files: list[str] = []
        for task in plan.tasks:
            all_files.extend(task.files_changed)
        if all_files:
            unique_files = sorted(set(all_files))
            console.print(
                f"\n[bold]Files changed:[/bold] "
                f"{', '.join(unique_files[:10])}"
            )
            if len(unique_files) > 10:
                console.print(
                    f"[dim]...and {len(unique_files) - 10} more[/dim]"
                )

        # Phase cost breakdown
        cost_table = Table(title="Cost by Phase", show_header=True)
        cost_table.add_column("Phase", style="cyan")
        cost_table.add_column("Cost (USD)", justify="right")
        for phase_name, cost in plan.phase_costs.items():
            cost_table.add_row(phase_name, f"${cost:.4f}")
        cost_table.add_row(
            "[bold]Total[/bold]",
            f"[bold]${plan.total_cost:.4f}[/bold]",
        )
        console.print(cost_table)

        # Cross-review summary
        if plan.cross_reviews:
            console.print("\n[bold]Cross-Review Summary:[/bold]")
            for review in plan.cross_reviews:
                severity_color = {
                    "info": "cyan",
                    "warning": "yellow",
                    "error": "red",
                }.get(review.severity, "white")
                console.print(
                    f"  [{severity_color}]{review.severity}[/{severity_color}] "
                    f"Task {review.task_id[:8]}... "
                    f"({'approved' if review.approved else 'rejected'}): "
                    f"{review.comments[:100]}"
                )

        # Advanced feature summaries
        if plan.debate_result:
            console.print(
                f"\n[bold]Debate:[/bold] "
                f"{len(plan.debate_result.rounds)} rounds, "
                f"consensus {plan.debate_result.consensus_score:.0%}, "
                f"{len(plan.debate_result.participating_models)} models"
            )
        if plan.cascade_results:
            cascade_tasks = len(plan.cascade_results)
            avg_level = sum(
                r.accepted_at_level for r in plan.cascade_results.values()
            ) / max(cascade_tasks, 1)
            total_saved = sum(
                r.cost_saved_vs_premium for r in plan.cascade_results.values()
            )
            console.print(
                f"[bold]Cascade:[/bold] "
                f"{cascade_tasks} tasks, avg level {avg_level:.1f}, "
                f"saved ${total_saved:.4f} vs premium"
            )
        if plan.moa_results:
            moa_tasks = len(plan.moa_results)
            console.print(
                f"[bold]MoA:[/bold] "
                f"{moa_tasks} complex tasks used parallel generation + fusion"
            )

        completed = sum(
            1 for t in plan.tasks if t.status == TaskStatus.COMPLETED
        )
        console.print(
            f"\n[green]Swarm complete:[/green] "
            f"{completed}/{len(plan.tasks)} tasks completed, "
            f"total cost ${plan.total_cost:.4f}"
        )

    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
    except Exception as exc:
        logger.debug("swarm_error", error=str(exc))
        console.print(f"[red]Swarm error:[/] {exc}")
    return "continue"


def _looks_like_swarm_task(prompt: str) -> bool:
    """Heuristic: does this prompt describe a multi-step coding task?

    Checks for the presence of action keywords (build, create, implement, etc.)
    AND multi-step indicators (and, then, multiple, files, etc.).

    Args:
        prompt: The user's natural-language prompt.

    Returns:
        True if the prompt looks like it would benefit from swarm orchestration.
    """
    keywords = {"build", "create", "implement", "refactor", "add", "design", "develop"}
    multi_indicators = {
        "and", "then", "also", "multiple", "files",
        "modules", "components", "system",
    }
    words = set(prompt.lower().split())
    return bool(words & keywords) and bool(words & multi_indicators)


# ======================================================================
# Prompt processing (non-slash input)
# ======================================================================

def _pick_repl_model() -> str:
    """Pick the best available model based on environment variables.

    Priority order: paid frontier models first (best quality for tool use),
    then reliable free-tier providers, then free aggregators.
    """
    import os

    providers = [
        # Paid frontier — best quality and tool-use support
        ("ANTHROPIC_API_KEY", "anthropic/claude-sonnet-4-20250514"),
        ("OPENAI_API_KEY", "gpt-4o-mini"),
        # Reliable free-tier providers
        ("MISTRAL_API_KEY", "mistral/mistral-small-latest"),
        ("GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
        # Google — only if quota is available (many users hit limit:0)
        ("GOOGLE_API_KEY", "gemini/gemini-2.0-flash"),
        ("GEMINI_API_KEY", "gemini/gemini-2.0-flash"),
        # Free aggregator — rate limits vary
        ("OPENROUTER_API_KEY", "openrouter/meta-llama/llama-3.3-70b-instruct:free"),
    ]
    for env_var, model_id in providers:
        if os.environ.get(env_var):
            return model_id
    return "groq/llama-3.3-70b-versatile"




# ======================================================================
# Agentic loop helpers
# ======================================================================


def _cleanup_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Clean up an event loop, cancelling pending tasks."""
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
    except Exception:  # noqa: S110
        pass  # Intentional — suppressing async cleanup noise
    loop.close()


def _maybe_escalate_model(
    current_model: str,
    state: _SessionState,
) -> str | None:
    """Check whether the current model should be escalated for tool-use.

    When ``escalate_on_tool_use`` is enabled and the current model is a
    lightweight / simple-tier model, this returns the cheapest available
    model that supports tools and is at least MEDIUM tier.  Returns
    ``None`` if no escalation is warranted.

    Args:
        current_model: The model currently selected for the request.
        state: Session state (provides registry and settings).

    Returns:
        A better model id, or ``None`` if the current model is adequate.
    """
    from prism.providers.base import ComplexityTier

    # Guard: need settings + registry
    if state.settings is None or state.registry is None:
        return None

    escalate_flag = getattr(
        state.settings.config.routing, "escalate_on_tool_use", True,
    )
    if not escalate_flag:
        return None

    # If the current model is already at MEDIUM or COMPLEX tier, skip
    model_info = state.registry.get_model_info(current_model)
    if model_info is not None and model_info.tier in (ComplexityTier.MEDIUM, ComplexityTier.COMPLEX):
        return None

    # Find cheapest tool-capable model at MEDIUM or above
    all_models = state.registry.get_available_models()
    tier_order = {"simple": 0, "medium": 1, "complex": 2}
    candidates = [
        m for m in all_models
        if m.supports_tools
        and tier_order.get(m.tier.value, 0) >= 1
        and m.id != current_model
    ]
    if not candidates:
        return None

    # Already sorted cheapest-first by get_available_models
    return candidates[0].id


def _estimate_context_tokens(state: _SessionState) -> int:
    """Estimate tokens in the current conversation context."""
    from prism.context.manager import estimate_tokens

    total = 0
    for msg in state.conversation:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
    return total


def _load_project_instructions() -> str:
    """Load project instructions from .prism.md, CLAUDE.md, or .claude.md.

    Checks the current working directory for project instruction files
    in priority order. Returns the contents of the first file found,
    capped at 4000 characters, or an empty string if none exist.

    Returns:
        The project instructions text, or empty string.
    """
    from pathlib import Path

    instruction_files = (".prism.md", "CLAUDE.md", ".claude.md")
    for name in instruction_files:
        path = Path.cwd() / name
        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8")[:4000]
                logger.debug(
                    "project_instructions_loaded",
                    file=name,
                    length=len(content),
                )
                return content
            except (OSError, UnicodeDecodeError):
                logger.debug(
                    "project_instructions_read_failed", file=name,
                )
                continue
    return ""


def _build_system_prompt() -> dict[str, Any]:
    """Build the system prompt with project context and self-correction rules."""
    from pathlib import Path as _Path
    cwd = str(_Path.cwd())

    content = (
        "You are Prism, an intelligent coding assistant in a terminal REPL. "
        f"Project directory: {cwd}\n\n"
        "## How to respond\n"
        "- Be **conversational**. Talk like a helpful colleague, not a textbook.\n"
        "- Keep responses **short and focused**. 3-8 lines for simple questions. "
        "Never dump a wall of text.\n"
        "- **Ask clarifying questions** when the request is ambiguous. "
        "Don't guess — ask what they mean.\n"
        "- When a task has multiple approaches, briefly describe 2-3 options "
        "and ask which they prefer.\n"
        "- For big tasks, outline your plan in 3-5 bullet points and ask "
        "\"Want me to go ahead?\" before executing.\n"
        "- Use markdown formatting: **bold** for emphasis, `code` for "
        "identifiers, ```blocks``` for code.\n"
        "- Show small code snippets inline. For large changes, describe "
        "what you'll do and use tools.\n\n"
        "## When to use tools\n"
        "- Only use tools when the user asks you to read, write, edit, "
        "search files, or run commands.\n"
        "- For greetings, questions, explanations — just reply with text.\n"
        "- Before editing, read the file first. After editing, confirm what "
        "you changed.\n"
        "- If a tool fails, try to fix it (max 3 retries).\n"
    )

    # --- Repository map (codebase structure) ---
    try:
        from prism.context.repo_map import generate_repo_map
        repo_map = generate_repo_map(_Path.cwd(), max_tokens=3000)
        if repo_map:
            content += (
                "\n\n## Repository Map\n"
                "Below is a compressed view of the codebase structure "
                "(classes, functions, signatures). Use this to understand "
                "the project before making changes.\n\n"
                f"{repo_map}"
            )
    except Exception:
        logger.debug("repo_map_generation_failed")

    # --- Project memory (persistent facts from PRISM_MEMORY.md) ---
    try:
        from prism.context.memory import ProjectMemory
        memory = ProjectMemory(_Path.cwd())
        memory_block = memory.get_context_block()
        if memory_block:
            content += f"\n\n{memory_block}"
    except Exception:
        logger.debug("project_memory_injection_failed")

    # --- Project instructions (.prism.md / CLAUDE.md / .claude.md) ---
    project_instructions = _load_project_instructions()
    if project_instructions:
        content += (
            "\n\n## Project Instructions\n"
            f"{project_instructions}"
        )

    return {"role": "system", "content": content}


def _show_tool_action(
    tool_name: str,
    arguments: dict[str, Any],
    console: Any,
) -> None:
    """Show a dim status line for the tool being used."""
    if tool_name == "read_file":
        console.print(f"  [dim]Reading {arguments.get('path', '?')}[/dim]")
    elif tool_name == "list_directory":
        console.print(
            f"  [dim]Listing {arguments.get('path', '.')}[/dim]"
        )
    elif tool_name == "search_codebase":
        console.print(
            f"  [dim]Searching: {arguments.get('pattern', '?')}[/dim]"
        )
    elif tool_name == "write_file":
        path = arguments.get("path", "?")
        size = len(arguments.get("content", ""))
        console.print(f"  [dim]Writing {path} ({size} chars)[/dim]")
    elif tool_name == "edit_file":
        console.print(
            f"  [dim]Editing {arguments.get('path', '?')}[/dim]"
        )
    elif tool_name == "execute_command":
        cmd = arguments.get("command", "?")
        console.print(f"  [dim]$ {cmd[:100]}[/dim]")
    elif tool_name == "browse_web":
        console.print(
            f"  [dim]Browsing {arguments.get('url', '?')}[/dim]"
        )
    else:
        console.print(f"  [dim]{tool_name}[/dim]")


def _ask_permission(
    tool_name: str,
    arguments: dict[str, Any],
    level: str,
    console: Any,
) -> str:
    """Ask user for permission before executing a tool.

    Returns:
        ``"deny"`` if the user declined, ``"once"`` for a one-time approval,
        or ``"always"`` to approve this tool for the rest of the session.
        DANGEROUS tools never offer the "always" option.
    """
    if level == "DANGEROUS":
        console.print(f"  [red bold]DANGEROUS:[/] {tool_name}")
    else:
        console.print(f"  [yellow]Confirm:[/] {tool_name}")

    if tool_name == "write_file":
        path = arguments.get("path", "?")
        size = len(arguments.get("content", ""))
        console.print(f"    Write {size} chars to {path}")
    elif tool_name == "edit_file":
        console.print(f"    Edit {arguments.get('path', '?')}")
    elif tool_name == "execute_command":
        console.print(f"    $ {arguments.get('command', '?')}")

    try:
        if level == "DANGEROUS":
            response = input("  Allow? [y/N] ").strip().lower()
            if response in ("y", "yes"):
                return "once"
            return "deny"
        # CONFIRM tools offer the "always" option
        response = input("  Allow? [y/N/a(lways)] ").strip().lower()
        if response in ("a", "always"):
            return "always"
        if response in ("y", "yes"):
            return "once"
        return "deny"
    except (EOFError, KeyboardInterrupt):
        return "deny"


def _format_tool_error(
    tool_name: str,
    error_message: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Format a tool error with actionable context for self-correction.

    Delegates to :func:`prism.cli.error_recovery.format_tool_error` which
    classifies the error, selects a recovery strategy, and builds a
    structured prompt for the LLM.

    Args:
        tool_name: Name of the tool that failed.
        error_message: The raw error string.
        arguments: The arguments that were passed to the tool (optional).

    Returns:
        A formatted error string with self-correction hints.
    """
    from prism.cli.error_recovery import format_tool_error

    return format_tool_error(
        tool_name, error_message, arguments=arguments,
    )


def _should_auto_approve(tool_name: str, state: _SessionState) -> bool:
    """Check whether a CONFIRM-level tool can be auto-approved.

    A tool is auto-approved when:
    - It was previously approved with "always" for this session, or
    - The user passed ``--yes`` (``settings.config.tools.auto_approve``).

    DANGEROUS tools must never be auto-approved through this path.

    Args:
        tool_name: Name of the tool to check.
        state: Current session state.

    Returns:
        True if the tool should be auto-approved.
    """
    if tool_name in state.permission_cache:
        return True
    return bool(state.settings and state.settings.config.tools.auto_approve)


def _execute_tool_with_permission(
    tool_name: str,
    arguments: dict[str, Any],
    tool_registry: Any,
    console: Any,
    hook_manager: Any | None = None,
    mcp_client: Any | None = None,
    audit_logger: Any | None = None,
    state: Any | None = None,
) -> str:
    """Execute a tool with permission checking and self-correction context.

    When a tool fails, the error is formatted with actionable hints so
    the LLM can automatically retry with a corrected approach.

    Supports pre/post hooks via ``hook_manager`` and MCP tools via
    ``mcp_client`` (tools prefixed with ``mcp_`` are routed to MCP).
    """
    from prism.tools.base import PermissionLevel

    # --- MCP tool routing ---
    if mcp_client and tool_name.startswith("mcp_"):
        _show_tool_action(tool_name, arguments, console)
        try:
            mcp_result = mcp_client.handle_mcp_tool_call(
                tool_name, arguments,
            )
            output = str(mcp_result)
            if len(output) > 8000:
                output = output[:8000] + "\n... (truncated)"
            return output
        except Exception as exc:
            return _format_tool_error(
                tool_name, str(exc), arguments,
            )

    try:
        tool = tool_registry.get_tool(tool_name)
    except Exception:
        return _format_tool_error(
            tool_name,
            f"Unknown tool '{tool_name}'. Available tools can be "
            "listed with the tool registry.",
        )

    # Determine effective permission level
    perm = tool.permission_level

    # Show what we're about to do
    _show_tool_action(tool_name, arguments, console)

    # --- Diff preview for write/edit tools ---
    if tool_name in ("write_file", "edit_file"):
        try:
            if tool_name == "write_file":
                diff_text, _is_new = tool.generate_preview_diff(arguments)
            else:
                diff_text = tool.generate_preview_diff(arguments)
            if diff_text:
                from prism.cli.ui.display import display_diff

                display_diff(diff_text, arguments.get("path", ""), console)
        except Exception:
            logger.debug("diff_preview_failed", tool=tool_name)

    # --- Pre-hooks ---
    if hook_manager:
        try:
            pre_results = hook_manager.run_pre_hooks(tool_name, arguments)
            for hr in pre_results:
                if hr.blocked:
                    console.print(
                        f"  [yellow]Blocked by hook:[/] {hr.output}"
                    )
                    return f"Blocked by pre-hook: {hr.output}"
        except Exception:
            logger.debug("pre_hook_error", tool=tool_name)

    # Permission gate for writes and commands
    if perm in (PermissionLevel.CONFIRM, PermissionLevel.DANGEROUS):
        # Check permission overrides in priority order
        skip = False

        # 1. --dangerously-skip-permissions → skip everything
        if state is not None and state.skip_all_permissions:
            skip = True

        # 2. Auto-approve for CONFIRM tools only (not DANGEROUS)
        if (
            not skip
            and perm == PermissionLevel.CONFIRM
            and state is not None
            and _should_auto_approve(tool_name, state)
        ):
            console.print(
                f"  [dim]Auto-approved: {tool_name}[/dim]"
            )
            skip = True

        if not skip:
            label = (
                "DANGEROUS"
                if perm == PermissionLevel.DANGEROUS
                else "confirm"
            )
            decision = _ask_permission(
                tool_name, arguments, label, console,
            )
            if decision == "deny":
                return f"Permission denied by user for {tool_name}"
            if (
                decision == "always"
                and state is not None
            ):
                state.permission_cache.add(tool_name)

    # Execute the tool
    try:
        result = tool.execute(arguments)
        result_output = result.output if result.success else (result.error or "")

        # --- Post-hooks ---
        if hook_manager:
            try:
                hook_manager.run_post_hooks(tool_name, result_output)
            except Exception:
                logger.debug("post_hook_error", tool=tool_name)

        # --- Audit log ---
        if audit_logger:
            try:
                audit_logger.log_tool_execution(
                    tool_name=tool_name,
                    args=arguments,
                    success=result.success,
                    error=result.error if not result.success else None,
                )
            except Exception:
                logger.debug("audit_log_error", tool=tool_name)

        if result.success:
            # --- Auto-commit after file writes/edits ---
            if (
                state is not None
                and state.auto_committer
                and tool_name in ("write_file", "edit_file")
            ):
                try:
                    file_path = arguments.get("path", "")
                    state.auto_committer.auto_commit_edit(
                        file_path=file_path,
                        description=f"{tool_name} via REPL",
                    )
                except Exception:
                    logger.debug("auto_commit_failed", tool=tool_name)

            output = result.output
            if len(output) > 8000:
                output = output[:8000] + "\n... (truncated)"
            return output
        return _format_tool_error(
            tool_name, result.error or "Unknown error",
            arguments,
        )
    except Exception as exc:
        return _format_tool_error(
            tool_name, str(exc), arguments,
        )


def _clean_error_message(err_msg: str) -> str:
    """Strip LiteLLM/provider noise from error messages for clean display."""
    for prefix in (
        "GroqException - ", "OpenrouterException - ",
        "DeepseekException - ", "geminiException - ",
        "MistralException - ", "AnthropicException - ",
        "OpenAIException - ",
    ):
        if prefix in err_msg:
            err_msg = err_msg.split(prefix, 1)[-1]
            break
    if "Cannot connect to host" in err_msg or "nodename nor servname" in err_msg:
        return "Network error — check your internet connection."
    if "Invalid API Key" in err_msg or "invalid_api_key" in err_msg:
        return "Invalid API key. Run: prism auth status"
    if "Insufficient Balance" in err_msg:
        return "API credits exhausted for this provider."
    if "rate-limited" in err_msg.lower() or "429" in err_msg:
        return "Rate limited — try again in a moment."
    if "Quota exceeded" in err_msg or "quota" in err_msg.lower():
        return "API quota exceeded. Check your provider dashboard."
    # Truncate very long errors
    if len(err_msg) > 200:
        err_msg = err_msg[:200] + "..."
    return err_msg


def _save_session(state: _SessionState) -> None:
    """Persist current conversation to session file."""
    if state.session_manager is None:
        return
    try:
        state.session_manager.save_session(
            state.session_id,
            {
                "session_id": state.session_id,
                "messages": state.conversation,
                "active_files": state.active_files,
                "pinned_model": state.pinned_model,
            },
        )
    except Exception:
        logger.debug("session_save_failed", session_id=state.session_id)


def _run_completion(
    engine: Any,
    messages: list[dict[str, Any]],
    model: str,
    tool_schemas: list[dict[str, Any]] | None,
    session_id: str,
    tier_value: str,
    console: Any | None = None,
) -> Any:
    """Run async completion in a new event loop with a progress spinner."""
    from rich.live import Live
    from rich.spinner import Spinner

    async def _do_complete() -> Any:
        return await engine.complete(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=4096,
            tools=tool_schemas,
            session_id=session_id,
            complexity_tier=tier_value,
        )

    loop = asyncio.new_event_loop()
    try:
        if console is not None:
            spinner = Spinner("dots", text="[dim]Thinking...[/dim]")
            with Live(spinner, console=console, refresh_per_second=10, transient=True):
                return loop.run_until_complete(_do_complete())
        return loop.run_until_complete(_do_complete())
    finally:
        _cleanup_event_loop(loop)


def _run_completion_streaming(
    engine: Any,
    messages: list[dict[str, Any]],
    model: str,
    session_id: str,
    tier_value: str,
    console: Any,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> Any:
    """Run streaming completion, displaying tokens as they arrive.

    Shows a "Thinking..." spinner while waiting for the first token,
    then switches to Rich Live Markdown rendering for a typing effect.
    Supports tool_calls in streaming responses when tool_schemas are provided.

    Args:
        engine: The :class:`CompletionEngine` instance.
        messages: Chat messages in OpenAI format.
        model: LiteLLM model identifier.
        session_id: Current session id for cost tracking.
        tier_value: Complexity tier label.
        console: Rich console for fallback rendering.
        tool_schemas: Optional tool schemas to send with the request.

    Returns:
        A :class:`CompletionResult` with the full response.
    """
    from prism.cli.stream_handler import StreamHandler

    handler = StreamHandler(console)
    handler.show_thinking()

    extra_kwargs: dict[str, Any] = {}
    if tool_schemas:
        extra_kwargs["tools"] = tool_schemas

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            engine.complete_streaming(
                messages=messages,
                model=model,
                on_token=handler.on_token,
                session_id=session_id,
                complexity_tier=tier_value,
                **extra_kwargs,
            )
        )
        handler.finalize()
        return result
    except Exception:
        # If streaming fails, clean up the handler output and fall back
        handler.finalize()
        raise
    finally:
        _cleanup_event_loop(loop)


def _run_completion_with_fallback(
    engine: Any,
    messages: list[dict[str, Any]],
    models: list[str],
    session_id: str,
    tier_value: str,
) -> Any:
    """Run async completion with fallback in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            engine.complete_with_fallback(
                messages=messages,
                models=models,
                temperature=0.3,
                max_tokens=4096,
                session_id=session_id,
                complexity_tier=tier_value,
            )
        )
    finally:
        _cleanup_event_loop(loop)


# ======================================================================
# Main prompt processor — the agentic core
# ======================================================================


# Token threshold for auto-compaction (roughly 60k tokens ≈ 240k chars)
_AUTO_COMPACT_CHAR_THRESHOLD = 200_000
_AUTO_COMPACT_MAX_TOKENS = 8000


def _auto_compact_conversation(
    state: _SessionState,
    console: Any,
) -> None:
    """Auto-compact conversation when it exceeds the token threshold.

    Replaces older messages with a summary, keeping the most recent
    exchanges intact so the LLM retains working context.
    """
    total_chars = sum(
        len(str(msg.get("content", ""))) for msg in state.conversation
    )
    if total_chars < _AUTO_COMPACT_CHAR_THRESHOLD:
        return

    try:
        from prism.context.summarizer import summarize

        # Only summarize user/assistant messages (not tool messages)
        chat_msgs = [
            m for m in state.conversation
            if m.get("role") in ("user", "assistant")
        ]
        if len(chat_msgs) <= 8:
            return  # Too few to compact

        summary_text = summarize(
            chat_msgs,
            max_tokens=_AUTO_COMPACT_MAX_TOKENS,
            keep_recent=6,
        )

        # Replace conversation with compacted version
        # Keep tool-result messages from last 6 exchanges
        recent_msgs = state.conversation[-12:]
        state.conversation = [
            {"role": "system", "content": f"[Compacted context]\n{summary_text}"},
            *recent_msgs,
        ]

        console.print("[dim]Context auto-compacted.[/dim]")
        logger.debug(
            "auto_compact",
            original_chars=total_chars,
            new_messages=len(state.conversation),
        )
    except Exception:
        logger.debug("auto_compact_failed")


def _execute_tools_parallel(
    tool_calls: list[dict[str, Any]],
    state: _SessionState,
    console: Any,
) -> list[tuple[str, str]]:
    """Execute tool calls, using parallelism for read-only tools.

    Read-only tools (read_file, list_directory, search_codebase) run
    in parallel via ThreadPoolExecutor.  Write tools and commands run
    sequentially to preserve ordering and allow permission prompts.

    Args:
        tool_calls: List of tool call dicts from the LLM response.
        state: Current session state.
        console: Rich console for output.

    Returns:
        List of (tool_call_id, output) tuples in the original order.
    """
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Read-only tools that are safe to parallelize
    parallel_safe = frozenset({
        "read_file", "list_directory", "search_codebase",
    })

    # Parse all tool calls first
    parsed: list[tuple[str, str, dict[str, Any]]] = []
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        fn = tc.get("function", {})
        tool_name = fn.get("name", "")
        try:
            args = _json.loads(fn.get("arguments", "{}"))
        except _json.JSONDecodeError:
            args = {}
        parsed.append((tc_id, tool_name, args))

    # If only 1 tool call, skip parallelism overhead
    if len(parsed) <= 1:
        results: list[tuple[str, str]] = []
        for tc_id, tool_name, args in parsed:
            output = _execute_tool_with_permission(
                tool_name=tool_name,
                arguments=args,
                tool_registry=state.tool_registry,
                console=console,
                hook_manager=state.hook_manager,
                mcp_client=state.mcp_client,
                audit_logger=state.audit_logger,
                state=state,
            )
            results.append((tc_id, output))
        return results

    # Split into parallel-safe and sequential groups
    parallel_batch: list[tuple[int, str, str, dict[str, Any]]] = []
    sequential_batch: list[tuple[int, str, str, dict[str, Any]]] = []

    for idx, (tc_id, tool_name, args) in enumerate(parsed):
        if tool_name in parallel_safe:
            parallel_batch.append((idx, tc_id, tool_name, args))
        else:
            sequential_batch.append((idx, tc_id, tool_name, args))

    # Results indexed by original position
    results_by_idx: dict[int, tuple[str, str]] = {}

    # Execute read-only tools in parallel
    if parallel_batch:
        def _run_tool(item: tuple[int, str, str, dict[str, Any]]) -> tuple[int, str, str]:
            idx, tc_id, tool_name, args = item
            output = _execute_tool_with_permission(
                tool_name=tool_name,
                arguments=args,
                tool_registry=state.tool_registry,
                console=console,
                hook_manager=state.hook_manager,
                mcp_client=state.mcp_client,
                audit_logger=state.audit_logger,
                state=state,
            )
            return idx, tc_id, output

        with ThreadPoolExecutor(max_workers=min(len(parallel_batch), 4)) as pool:
            futures = {pool.submit(_run_tool, item): item for item in parallel_batch}
            for future in as_completed(futures):
                try:
                    idx, tc_id, output = future.result()
                    results_by_idx[idx] = (tc_id, output)
                except Exception as exc:
                    item = futures[future]
                    results_by_idx[item[0]] = (item[1], f"Error: {exc}")

    # Execute write/command tools sequentially
    for idx, tc_id, tool_name, args in sequential_batch:
        output = _execute_tool_with_permission(
            tool_name=tool_name,
            arguments=args,
            tool_registry=state.tool_registry,
            console=console,
            hook_manager=state.hook_manager,
            mcp_client=state.mcp_client,
            audit_logger=state.audit_logger,
            state=state,
        )
        results_by_idx[idx] = (tc_id, output)

    # Return in original order
    return [results_by_idx[i] for i in range(len(parsed))]


def _process_prompt(
    prompt: str,
    console: Console,
    settings: Settings,
    state: _SessionState,
    dry_run: bool,
    offline: bool,
) -> None:
    """Process a user prompt through the full routing + agentic loop.

    Flow:
    1. Classify task complexity (simple/medium/complex)
    2. Select best model for that tier + fallback chain
    3. Send to LLM with tool definitions
    4. If LLM wants tools → execute (with permission) → loop
    5. Display final text response
    6. Track cost, record outcome, persist session
    """
    # --- Dry run mode ---
    if dry_run:
        from prism.router.classifier import TaskClassifier, TaskContext

        classifier = state.classifier or TaskClassifier(settings)
        ctx = TaskContext(active_files=state.active_files)
        result = classifier.classify(prompt, ctx)
        tier_colors = {
            "simple": "green", "medium": "yellow", "complex": "red",
        }
        color = tier_colors.get(result.tier.value, "white")
        console.print(
            f"\n[dim]Tier:[/] [{color}]{result.tier.value.upper()}"
            f"[/{color}] [dim](score: {result.score:.2f})[/dim]"
        )
        console.print(f"[dim]Features:[/] {result.features}")
        console.print(f"[dim]Reasoning:[/] {result.reasoning}")
        console.print("[yellow]Dry-run mode — no API call made.[/]\n")
        return

    # --- Fallback to legacy mode if modules didn't initialize ---
    if state.completion_engine is None:
        _process_prompt_legacy(prompt, console, settings, state)
        return

    import warnings
    warnings.filterwarnings("ignore", message=".*Task was destroyed.*")

    state.conversation.append({"role": "user", "content": prompt})

    # --- Auto-compaction: summarize if conversation is too long ---
    _auto_compact_conversation(state, console)

    # --- Step 1: Classify task complexity ---
    from prism.router.classifier import TaskContext

    tier = None
    if state.classifier:
        task_ctx = TaskContext(
            active_files=state.active_files,
            conversation_turns=len(state.conversation) // 2,
        )
        classification = state.classifier.classify(prompt, task_ctx)
        tier = classification.tier

    # --- Step 2: Select model + fallback chain ---
    chosen_model = state.pinned_model or _pick_repl_model()
    fallback_models: list[str] = []

    # Determine if tools could be invoked (quality floor check)
    tools_could_be_used = state.tool_registry is not None

    if state.selector and tier:
        try:
            selection = state.selector.select(
                tier=tier,
                prompt=prompt,
                context_tokens=_estimate_context_tokens(state),
                tools_enabled=tools_could_be_used,
            )
            chosen_model = selection.model_id
            fallback_models = list(selection.fallback_chain)
        except Exception as sel_exc:
            logger.debug("selector_fallback", error=str(sel_exc))
            # Keep the default from _pick_repl_model()

    # --- Step 2a: Privacy mode enforcement (local-only routing) ---
    if state.privacy_manager and state.privacy_manager.is_private:
        _is_ollama = (
            chosen_model.startswith("ollama/")
            or chosen_model.startswith("ollama_chat/")
        )
        if not _is_ollama:
            recommended = state.privacy_manager.get_recommended_model()
            console.print(
                f"[yellow]Privacy mode:[/] Overriding "
                f"[dim]{chosen_model}[/dim] -> "
                f"[cyan]ollama/{recommended}[/cyan]"
            )
            chosen_model = f"ollama/{recommended}"
            fallback_models = []

    # --- Step 2b: Offline mode fallback (local-only routing) ---
    _force_offline = offline
    if not _force_offline and state.offline_manager:
        _force_offline = state.offline_manager.is_offline
    if _force_offline:
        _is_ollama = (
            chosen_model.startswith("ollama/")
            or chosen_model.startswith("ollama_chat/")
        )
        if not _is_ollama:
            _ollama_fallback = "llama3.1:8b"
            if state.privacy_manager:
                _ollama_fallback = state.privacy_manager.get_recommended_model()
            console.print(
                f"[yellow]Offline mode:[/] Overriding "
                f"[dim]{chosen_model}[/dim] -> "
                f"[cyan]ollama/{_ollama_fallback}[/cyan]"
            )
            chosen_model = f"ollama/{_ollama_fallback}"
            fallback_models = []

    # Routing info — only show in verbose/debug mode
    tier_str = tier.value if tier else "?"
    logger.debug("routing_decision", model=chosen_model, tier=tier_str)

    # --- Swarm suggestion for complex multi-step tasks ---
    if tier and tier.value == "complex" and _looks_like_swarm_task(prompt):
        console.print(
            "[dim]Tip: This looks like a multi-step task. "
            "Try [cyan]/swarm " + prompt[:50] + "...[/cyan] for "
            "multi-model collaboration.[/dim]"
        )

    # --- Step 2.5: Enhance prompt ---
    enhanced_prompt = prompt
    if state.prompt_enhancer:
        try:
            result = state.prompt_enhancer.enhance(prompt)
            enhanced_prompt = result.enhanced_prompt
        except Exception:
            logger.debug("prompt_enhance_failed")

    # --- Step 3: Build tool schemas ---
    tool_schemas: list[dict[str, Any]] | None = None
    if state.tool_registry:
        try:
            raw_schemas = state.tool_registry.all_schemas()
            tool_schemas = [
                {"type": "function", "function": schema}
                for schema in raw_schemas
            ]
        except Exception:
            logger.debug("tool_schema_build_failed")

    # Add MCP tool schemas
    if state.mcp_client:
        try:
            mcp_schemas = state.mcp_client.get_tool_schemas()
            if mcp_schemas:
                if tool_schemas is None:
                    tool_schemas = []
                tool_schemas.extend(mcp_schemas)
        except Exception:
            logger.debug("mcp_schema_build_failed")

    # Check if model supports tools
    if state.registry:
        model_info = state.registry.get_model_info(chosen_model)
        if model_info and not model_info.supports_tools:
            tool_schemas = None

    # --- Step 4: Build messages ---
    system_msg = _build_system_prompt()
    messages: list[dict[str, Any]] = [
        system_msg, *list(state.conversation),
    ]
    # Replace last user message with enhanced prompt if different
    if enhanced_prompt != prompt and messages:
        messages[-1] = {"role": "user", "content": enhanced_prompt}

    # --- Step 4.5: Rate limiter check ---
    if state.rate_limiter:
        provider_name = chosen_model.split("/")[0] if "/" in chosen_model else "openai"
        if state.rate_limiter.is_rate_limited(provider_name):
            wait_secs = state.rate_limiter.get_wait_time(provider_name)
            console.print(
                f"[yellow]Rate limited[/] for {provider_name} — "
                f"retry in {wait_secs:.1f}s"
            )
            return

    # --- Step 5: Agentic loop ---
    max_iterations = 15
    total_cost = 0.0
    did_stream = False

    try:
        for _iteration in range(max_iterations):
            if _iteration == 0:
                # --- First iteration: stream with tools for typing effect ---
                try:
                    result = _run_completion_streaming(
                        engine=state.completion_engine,
                        messages=messages,
                        model=chosen_model,
                        session_id=state.session_id,
                        tier_value=tier_str,
                        console=console,
                        tool_schemas=tool_schemas,
                    )
                    did_stream = True
                except Exception:
                    logger.debug("streaming_fallback_to_regular")
                    # Fall back to non-streaming with tools
                    result = _run_completion(
                        engine=state.completion_engine,
                        messages=messages,
                        model=chosen_model,
                        tool_schemas=tool_schemas,
                        session_id=state.session_id,
                        tier_value=tier_str,
                        console=console,
                    )
                    did_stream = False
            else:
                # --- Subsequent iterations: non-streaming with tools ---
                result = _run_completion(
                    engine=state.completion_engine,
                    messages=messages,
                    model=chosen_model,
                    tool_schemas=tool_schemas,
                    session_id=state.session_id,
                    tier_value=tier_str,
                    console=console,
                )
                did_stream = False

            total_cost += result.cost_usd

            # Record request for rate limiting
            if state.rate_limiter:
                state.rate_limiter.record_request(provider_name)

            # --- Tool calls? Execute them (parallel when possible) ---
            if result.tool_calls:
                # --- Smart escalation on first tool-use detection ---
                if _iteration == 0 and not state.pinned_model:
                    escalated = _maybe_escalate_model(
                        current_model=chosen_model, state=state,
                    )
                    if escalated and escalated != chosen_model:
                        console.print(
                            f"[dim]Escalating: {chosen_model} -> "
                            f"{escalated} (tool-use detected)[/dim]"
                        )
                        chosen_model = escalated

                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": result.content or None,
                    "tool_calls": result.tool_calls,
                }
                messages.append(assistant_msg)

                tool_results = _execute_tools_parallel(
                    tool_calls=result.tool_calls,
                    state=state,
                    console=console,
                )

                for tc_id, tool_output in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": tool_output,
                    })

                continue  # Loop back for model to process results

            # --- Final text response ---
            content = result.content or ""
            if content:
                state.conversation.append(
                    {"role": "assistant", "content": content}
                )
                # If we already streamed the content, skip Rich
                # markdown rendering (the user saw it live).
                if not did_stream:
                    _display_ai_response(console, content)
                else:
                    # Streamed output already printed; just add
                    # trailing spacing for visual consistency.
                    console.print()

            # Show cost (only if non-zero)
            if total_cost > 0.001:
                console.print(f"[dim]${total_cost:.4f}[/dim]")

            # Record outcome for adaptive learning
            if state.learner and tier:
                state.learner.record_outcome(
                    model=chosen_model,
                    tier=tier,
                    outcome="accepted",
                    cost=total_cost,
                )

            # Persist session
            _save_session(state)
            break

    except Exception as exc:
        err_str = str(exc)
        # If tools aren't supported, retry without tools
        if "tool" in err_str.lower() or "function" in err_str.lower():
            logger.debug("tools_unsupported_fallback", error=err_str)
            try:
                fallback_msgs = [system_msg, *list(state.conversation)]
                models = [chosen_model, *fallback_models]
                fb_result = _run_completion_with_fallback(
                    engine=state.completion_engine,
                    messages=fallback_msgs,
                    models=models,
                    session_id=state.session_id,
                    tier_value=tier_str,
                )
                content2 = fb_result.content or ""
                if content2:
                    state.conversation.append(
                        {"role": "assistant", "content": content2}
                    )
                    _display_ai_response(console, content2)
            except Exception as exc2:
                logger.debug("fallback_error", error=str(exc2))
                console.print(
                    f"[red]Error:[/] {_clean_error_message(str(exc2))}\n"
                )
        else:
            logger.debug("completion_error", error=err_str)
            console.print(
                f"[red]Error:[/] {_clean_error_message(err_str)}\n"
            )


def _process_prompt_legacy(
    prompt: str,
    console: Console,
    settings: Settings,
    state: _SessionState,
) -> None:
    """Legacy fallback — direct litellm call without full module stack."""
    import warnings
    warnings.filterwarnings("ignore", message=".*Task was destroyed.*")

    state.conversation.append({"role": "user", "content": prompt})

    try:

        import litellm

        litellm.suppress_debug_info = True
        chosen_model = state.pinned_model or _pick_repl_model()

        system_msg = _build_system_prompt()
        messages: list[dict[str, Any]] = [
            system_msg, *list(state.conversation),
        ]

        # Stream with spinner + Rich Live Markdown
        from prism.cli.stream_handler import StreamHandler

        handler = StreamHandler(console)
        handler.show_thinking()

        loop = asyncio.new_event_loop()
        full_content = ""
        try:
            response = loop.run_until_complete(
                litellm.acompletion(
                    model=chosen_model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2048,
                    stream=True,
                )
            )

            async def _consume_stream() -> str:
                async for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        handler.on_token(delta)
                return handler.buffer

            full_content = loop.run_until_complete(_consume_stream())
            handler.finalize()
        finally:
            _cleanup_event_loop(loop)

        if full_content:
            state.conversation.append(
                {"role": "assistant", "content": full_content}
            )
            # Show cost subtly after response
            console.print()
    except Exception as exc:
        logger.debug("legacy_completion_error", error=str(exc))
        # Show clean error messages — strip LiteLLM/provider noise
        err_msg = str(exc)
        # Extract the useful part from litellm error strings
        for prefix in ("GroqException - ", "OpenrouterException - ",
                        "DeepseekException - ", "geminiException - "):
            if prefix in err_msg:
                err_msg = err_msg.split(prefix, 1)[-1]
                break
        if "Cannot connect to host" in err_msg or "nodename nor servname" in err_msg:
            err_msg = "Network error — check your internet connection."
        elif "Invalid API Key" in err_msg or "invalid_api_key" in err_msg:
            err_msg = "Invalid API key. Run: prism auth status"
        elif "Insufficient Balance" in err_msg:
            err_msg = "API credits exhausted for this provider."
        elif "rate-limited" in err_msg.lower() or "429" in err_msg:
            err_msg = "Rate limited — try again in a moment."
        console.print(f"[red]Error:[/] {err_msg}\n")

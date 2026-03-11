"""Slash command handler for the Prism REPL.

Parses and dispatches ``/command [args]`` inputs to the appropriate
handler method.  Each command returns a :class:`CommandResponse` that
the REPL loop uses to decide what to display and whether to exit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

    from prism.config.settings import Settings
    from prism.context.manager import ContextManager
    from prism.context.memory import ProjectMemory
    from prism.context.session import SessionManager
    from prism.cost.tracker import CostTracker
    from prism.git.operations import GitRepo
    from prism.providers.registry import ProviderRegistry

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Response data-class
# ---------------------------------------------------------------------------


@dataclass
class CommandResponse:
    """Result produced by a slash command handler.

    Attributes:
        output:      Text to display to the user.
        should_exit: When ``True`` the REPL loop should terminate.
        error:       An error message (mutually exclusive with *output* in
                     most cases, but both may be populated).
    """

    output: str
    should_exit: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Command descriptions (used by /help)
# ---------------------------------------------------------------------------

_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/help": "Show available commands",
    "/cost": "Show cost dashboard (session, daily, monthly)",
    "/model": "Show or switch model (e.g. /model claude-sonnet-4-20250514)",
    "/undo": "Rollback last git commit",
    "/compact": "Summarize conversation to save context",
    "/add": "Add file(s) to context (e.g. /add src/main.py)",
    "/drop": "Remove file(s) from context (e.g. /drop src/main.py)",
    "/web": "Toggle web browsing (/web on | /web off)",
    "/status": "Show provider health, rate limits, budget",
    "/budget": "Show or set budget (/budget | /budget set 10.00)",
    "/memory": "View or edit project memory (/memory | /memory set key value)",
    "/feedback": "Rate last response (/feedback up | /feedback down)",
    "/providers": "List configured providers with status",
    "/clear": "Clear conversation history",
    "/save": "Save session (/save [name])",
    "/load": "Load session (/load [session_id])",
    "/logs": "View recent log entries",
    "/exit": "Exit Prism",
    "/quit": "Exit Prism",
}


# ---------------------------------------------------------------------------
# SlashCommandHandler
# ---------------------------------------------------------------------------


class SlashCommandHandler:
    """Handles all REPL slash commands.

    Each ``cmd_*`` method corresponds to a ``/command`` and returns a
    :class:`CommandResponse`.  The :meth:`handle` method routes incoming
    command strings to the correct handler.

    Args:
        settings:          Application settings.
        cost_tracker:      Cost tracking service (may be ``None`` early on).
        context_manager:   Conversation context manager.
        session_manager:   Session persistence manager.
        project_memory:    Project-level persistent memory.
        provider_registry: Provider registry (may be ``None`` if uninitialised).
        git_repo:          Git repository wrapper (``None`` if not in a repo).
    """

    def __init__(
        self,
        settings: Settings,
        cost_tracker: CostTracker | None = None,
        context_manager: ContextManager | None = None,
        session_manager: SessionManager | None = None,
        project_memory: ProjectMemory | None = None,
        provider_registry: ProviderRegistry | None = None,
        git_repo: GitRepo | None = None,
    ) -> None:
        self._settings = settings
        self._cost_tracker = cost_tracker
        self._context_manager = context_manager
        self._session_manager = session_manager
        self._project_memory = project_memory
        self._provider_registry = provider_registry
        self._git_repo = git_repo

        # Web-browsing toggle state
        self._web_enabled: bool = bool(settings.get("tools.web_enabled", False))

        # Track current model
        self._current_model: str | None = settings.config.pinned_model

        # Current session id (set externally or via /save)
        self.session_id: str = ""

        # Last feedback value for testing
        self._last_feedback: str | None = None

        # Build the dispatch table
        self._commands: dict[str, Callable[[str], CommandResponse]] = {
            "/help": self.cmd_help,
            "/cost": self.cmd_cost,
            "/model": self.cmd_model,
            "/undo": self.cmd_undo,
            "/compact": self.cmd_compact,
            "/add": self.cmd_add,
            "/drop": self.cmd_drop,
            "/web": self.cmd_web,
            "/status": self.cmd_status,
            "/budget": self.cmd_budget,
            "/memory": self.cmd_memory,
            "/feedback": self.cmd_feedback,
            "/providers": self.cmd_providers,
            "/clear": self.cmd_clear,
            "/save": self.cmd_save,
            "/load": self.cmd_load,
            "/logs": self.cmd_logs,
            "/exit": self.cmd_exit,
            "/quit": self.cmd_exit,
        }

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def handle(self, command_line: str) -> CommandResponse:
        """Parse and execute a slash command.

        Args:
            command_line: The raw input string starting with ``/``.

        Returns:
            A :class:`CommandResponse` describing the outcome.
        """
        parts = command_line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        handler = self._commands.get(cmd)
        if handler is None:
            return CommandResponse(
                output="",
                error=f"Unknown command: {cmd}. Type /help for available commands.",
            )

        try:
            return handler(args)
        except Exception as exc:
            logger.exception("slash_command_error", command=cmd)
            return CommandResponse(output="", error=f"Command failed: {exc}")

    # ------------------------------------------------------------------
    # Individual command handlers
    # ------------------------------------------------------------------

    def cmd_help(self, args: str) -> CommandResponse:
        """Show all commands with descriptions."""
        lines: list[str] = ["Available Commands:", ""]
        for cmd, desc in _COMMAND_DESCRIPTIONS.items():
            if cmd == "/quit":
                continue  # Don't duplicate /exit and /quit
            lines.append(f"  {cmd:14s} {desc}")
        lines.append("")
        return CommandResponse(output="\n".join(lines))

    def cmd_cost(self, args: str) -> CommandResponse:
        """Full spending dashboard -- session, daily, monthly breakdown."""
        if self._cost_tracker is None:
            return CommandResponse(
                output="Cost dashboard will be available after the first API call."
            )
        from prism.cost.dashboard import render_cost_dashboard

        text = render_cost_dashboard(
            self._cost_tracker,
            session_id=self.session_id or "unknown",
        )
        return CommandResponse(output=text)

    def cmd_model(self, args: str) -> CommandResponse:
        """Switch model mid-session or show the current model."""
        if not args:
            current = self._current_model or "auto (routing)"
            return CommandResponse(output=f"Current model: {current}")

        # Validate model exists if provider registry is available
        if self._provider_registry is not None:
            model_info = self._provider_registry.get_model_info(args)
            if model_info is None:
                known = list(self._provider_registry.all_models.keys())
                return CommandResponse(
                    output="",
                    error=f"Unknown model: {args}. Known models: {', '.join(known[:10])}",
                )

        self._current_model = args
        return CommandResponse(output=f"Model set to: {args}")

    def cmd_undo(self, args: str) -> CommandResponse:
        """Rollback last git commit."""
        if self._git_repo is None:
            return CommandResponse(
                output="",
                error="Not in a git repository. /undo requires git.",
            )
        try:
            self._git_repo._run(["git", "reset", "--soft", "HEAD~1"])
            return CommandResponse(output="Last commit undone (soft reset).")
        except Exception as exc:
            return CommandResponse(output="", error=f"Undo failed: {exc}")

    def cmd_compact(self, args: str) -> CommandResponse:
        """Summarize conversation to save context."""
        if self._context_manager is None:
            return CommandResponse(output="No active context to compact.")

        messages = self._context_manager.messages
        if not messages:
            return CommandResponse(output="No conversation history to compact.")

        from prism.context.summarizer import summarize

        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        max_tokens = self._context_manager.max_tokens // 4
        summary = summarize(msg_dicts, max_tokens=max_tokens)

        # Replace messages with a single summarized system message
        self._context_manager.clear_messages()
        self._context_manager.add_message("system", summary)

        return CommandResponse(
            output=f"Conversation compacted. {len(msg_dicts)} messages summarized."
        )

    def cmd_add(self, args: str) -> CommandResponse:
        """Add file(s) to context."""
        if not args:
            return CommandResponse(output="Usage: /add <file1> [file2] ...")

        if self._context_manager is None:
            return CommandResponse(output="", error="Context manager not available.")

        from pathlib import Path

        files = args.split()
        added: list[str] = []
        errors: list[str] = []

        for filepath in files:
            path = Path(filepath)
            if not path.is_absolute():
                path = self._settings.project_root / path

            if not path.is_file():
                errors.append(f"File not found: {filepath}")
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(f"Cannot read {filepath}: {exc}")
                continue

            self._context_manager.add_active_file(filepath, content)
            added.append(filepath)

        parts: list[str] = []
        for f in added:
            parts.append(f"  + {f}")
        for e in errors:
            parts.append(f"  ! {e}")

        if not parts:
            return CommandResponse(output="No files added.")
        return CommandResponse(output="\n".join(parts))

    def cmd_drop(self, args: str) -> CommandResponse:
        """Remove file(s) from context."""
        if self._context_manager is None:
            return CommandResponse(output="", error="Context manager not available.")

        if not args:
            # Show current active files
            active = self._context_manager.active_files
            if not active:
                return CommandResponse(output="No active files in context.")
            lines = ["Active files:"]
            for path in sorted(active.keys()):
                lines.append(f"  {path}")
            return CommandResponse(output="\n".join(lines))

        files = args.split()
        results: list[str] = []
        for filepath in files:
            removed = self._context_manager.remove_active_file(filepath)
            if removed:
                results.append(f"  - {filepath}")
            else:
                results.append(f"  ! Not in context: {filepath}")

        return CommandResponse(output="\n".join(results))

    def cmd_web(self, args: str) -> CommandResponse:
        """Toggle web browsing."""
        lower_args = args.lower().strip()
        if lower_args == "on":
            self._web_enabled = True
            return CommandResponse(output="Web browsing enabled.")
        elif lower_args == "off":
            self._web_enabled = False
            return CommandResponse(output="Web browsing disabled.")
        elif lower_args == "":
            # Toggle
            self._web_enabled = not self._web_enabled
            state = "enabled" if self._web_enabled else "disabled"
            return CommandResponse(output=f"Web browsing {state}.")
        else:
            return CommandResponse(output="Usage: /web [on|off]")

    def cmd_status(self, args: str) -> CommandResponse:
        """Show provider health, rate limits, budget remaining."""
        lines: list[str] = ["Prism Status", ""]

        # Budget
        if self._cost_tracker is not None:
            remaining = self._cost_tracker.get_budget_remaining()
            if remaining is not None:
                lines.append(f"Budget remaining: ${remaining:.2f}")
            else:
                lines.append("Budget: unlimited")

            daily_cost = self._cost_tracker.get_daily_cost()
            lines.append(f"Today's spend: ${daily_cost:.2f}")
        else:
            lines.append("Cost tracking: not active")

        # Model
        lines.append(f"Current model: {self._current_model or 'auto (routing)'}")
        lines.append(f"Web browsing: {'enabled' if self._web_enabled else 'disabled'}")

        # Providers
        if self._provider_registry is not None:
            providers = self._provider_registry.list_providers()
            lines.append(f"Providers: {len(providers)} registered")
            for p in providers:
                status_icon = "OK" if p.get("available") else "unavailable"
                lines.append(f"  {p['display_name']}: {status_icon}")

        lines.append("")
        return CommandResponse(output="\n".join(lines))

    def cmd_budget(self, args: str) -> CommandResponse:
        """Set or check budget."""
        parts = args.strip().split(maxsplit=1)

        if not parts or not parts[0]:
            # Show current budget
            daily = self._settings.get("budget.daily_limit")
            monthly = self._settings.get("budget.monthly_limit")
            lines: list[str] = []
            if daily is not None:
                lines.append(f"Daily limit: ${daily:.2f}")
            else:
                lines.append("Daily limit: unlimited")
            if monthly is not None:
                lines.append(f"Monthly limit: ${monthly:.2f}")
            else:
                lines.append("Monthly limit: unlimited")

            if self._cost_tracker is not None:
                remaining = self._cost_tracker.get_budget_remaining()
                if remaining is not None:
                    lines.append(f"Remaining: ${remaining:.2f}")

            return CommandResponse(output="\n".join(lines))

        if parts[0].lower() == "set":
            if len(parts) < 2:
                return CommandResponse(
                    output="", error="Usage: /budget set <amount>"
                )
            try:
                amount = float(parts[1])
            except ValueError:
                return CommandResponse(
                    output="",
                    error=f"Invalid amount: {parts[1]}. Must be a number.",
                )
            if amount < 0:
                return CommandResponse(
                    output="", error="Budget cannot be negative."
                )
            self._settings.set_override("budget.daily_limit", amount)
            return CommandResponse(output=f"Daily budget set to ${amount:.2f}")

        return CommandResponse(
            output="", error="Usage: /budget | /budget set <amount>"
        )

    def cmd_memory(self, args: str) -> CommandResponse:
        """View or edit project memory."""
        if self._project_memory is None:
            return CommandResponse(output="", error="Project memory not available.")

        parts = args.strip().split(maxsplit=2)

        if not parts or not parts[0]:
            # Show all facts
            facts = self._project_memory.get_facts()
            if not facts:
                return CommandResponse(output="No project memory entries.")
            lines = ["Project Memory:"]
            for key, value in sorted(facts.items()):
                lines.append(f"  {key}: {value}")
            return CommandResponse(output="\n".join(lines))

        if parts[0].lower() == "set":
            if len(parts) < 3:
                return CommandResponse(
                    output="", error="Usage: /memory set <key> <value>"
                )
            key = parts[1]
            value = parts[2]
            self._project_memory.add_fact(key, value)
            return CommandResponse(output=f"Memory set: {key} = {value}")

        # Treat single arg as key lookup
        fact = self._project_memory.get_fact(parts[0])
        if fact is not None:
            return CommandResponse(output=f"{parts[0]}: {fact}")
        return CommandResponse(output=f"No memory entry for: {parts[0]}")

    def cmd_feedback(self, args: str) -> CommandResponse:
        """Record feedback on the last response."""
        lower_args = args.lower().strip()
        if lower_args not in ("up", "down"):
            return CommandResponse(
                output="", error="Usage: /feedback up | /feedback down"
            )

        self._last_feedback = lower_args
        sentiment = "positive" if lower_args == "up" else "negative"
        logger.info("user_feedback", sentiment=sentiment)
        return CommandResponse(output=f"Feedback recorded: {sentiment}. Thank you!")

    def cmd_providers(self, args: str) -> CommandResponse:
        """List all configured providers with model counts and status."""
        if self._provider_registry is None:
            return CommandResponse(
                output="", error="Provider registry not available."
            )

        providers = self._provider_registry.list_providers()
        if not providers:
            return CommandResponse(output="No providers registered.")

        lines: list[str] = ["Providers:", ""]
        for p in providers:
            status = "available" if p.get("available") else "unavailable"
            configured = "configured" if p.get("configured") else "not configured"
            model_count = p.get("model_count", 0)
            models_str = ", ".join(str(m) for m in p.get("models", []))
            lines.append(
                f"  {p['display_name']} [{status}, {configured}] "
                f"({model_count} models: {models_str})"
            )

        lines.append("")
        return CommandResponse(output="\n".join(lines))

    def cmd_clear(self, args: str) -> CommandResponse:
        """Clear conversation history."""
        if self._context_manager is not None:
            self._context_manager.clear_messages()
        return CommandResponse(output="Conversation history cleared.")

    def cmd_save(self, args: str) -> CommandResponse:
        """Save session."""
        if self._session_manager is None:
            return CommandResponse(output="", error="Session manager not available.")

        if not self.session_id:
            self.session_id = self._session_manager.create_session(
                self._settings.project_root
            )

        # Build session data
        messages: list[dict[str, str]] = []
        if self._context_manager is not None:
            messages = [
                {"role": m.role, "content": m.content}
                for m in self._context_manager.messages
            ]

        session_data: dict[str, Any] = {
            "session_id": self.session_id,
            "project_root": str(self._settings.project_root),
            "messages": messages,
            "metadata": {
                "model": self._current_model,
                "web_enabled": self._web_enabled,
            },
        }

        name = args.strip() if args.strip() else None
        if name:
            session_data["metadata"]["name"] = name

        self._session_manager.save_session(self.session_id, session_data)
        display_name = name or self.session_id[:8]
        return CommandResponse(output=f"Session saved: {display_name}")

    def cmd_load(self, args: str) -> CommandResponse:
        """Load a previous session."""
        if self._session_manager is None:
            return CommandResponse(output="", error="Session manager not available.")

        session_id = args.strip()
        if not session_id:
            # List available sessions
            sessions = self._session_manager.list_sessions(
                self._settings.project_root
            )
            if not sessions:
                return CommandResponse(output="No saved sessions found.")
            lines = ["Available sessions:"]
            for s in sessions[:10]:
                sid = s["session_id"][:8]
                msgs = s.get("message_count", 0)
                updated = s.get("updated_at", "")[:19]
                lines.append(f"  {sid}  ({msgs} messages, updated {updated})")
            return CommandResponse(output="\n".join(lines))

        if not self._session_manager.session_exists(session_id):
            return CommandResponse(
                output="", error=f"Session not found: {session_id}"
            )

        data = self._session_manager.load_session(session_id)
        self.session_id = session_id

        # Restore messages into context manager
        if self._context_manager is not None:
            self._context_manager.clear_messages()
            for msg in data.get("messages", []):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    self._context_manager.add_message(role, content)

        # Restore metadata
        metadata = data.get("metadata", {})
        if metadata.get("model"):
            self._current_model = metadata["model"]
        if "web_enabled" in metadata:
            self._web_enabled = metadata["web_enabled"]

        msg_count = len(data.get("messages", []))
        return CommandResponse(
            output=f"Session loaded: {session_id[:8]} ({msg_count} messages)"
        )

    def cmd_logs(self, args: str) -> CommandResponse:
        """View recent log entries."""
        log_path = self._settings.prism_home / "prism.log"
        if not log_path.is_file():
            return CommandResponse(output="No log file found.")

        try:
            content = log_path.read_text(encoding="utf-8")
            lines = content.strip().splitlines()
            # Show last 20 lines
            recent = lines[-20:] if len(lines) > 20 else lines
            return CommandResponse(output="\n".join(recent))
        except OSError as exc:
            return CommandResponse(output="", error=f"Cannot read logs: {exc}")

    def cmd_exit(self, args: str) -> CommandResponse:
        """Graceful shutdown."""
        return CommandResponse(output="Goodbye!", should_exit=True)

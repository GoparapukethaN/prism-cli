"""Git operations tool -- safe git commands via the agentic loop.

Provides the LLM with structured access to git status, diff, log,
add, commit, branch, and checkout operations.  All commands execute
through :class:`CommandSandbox` for security guardrails (timeout,
output truncation, environment filtering).

Only a curated set of git sub-commands is allowed.  Destructive
operations (``reset --hard``, ``push --force``, ``clean``) are
rejected outright to prevent data loss.
"""

from __future__ import annotations

import re
import shlex
from typing import TYPE_CHECKING, Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

if TYPE_CHECKING:
    from prism.security.sandbox import CommandSandbox

logger = structlog.get_logger(__name__)

# Sub-commands the tool is allowed to run.
_ALLOWED_SUBCOMMANDS: frozenset[str] = frozenset({
    "status",
    "diff",
    "log",
    "add",
    "commit",
    "branch",
    "checkout",
    "show",
    "stash",
    "tag",
    "remote",
    "rev-parse",
    "blame",
})

# Patterns that are *always* blocked, even if the sub-command is allowed.
_BLOCKED_FLAG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"--force\b"),
    re.compile(r"-f\b"),
    re.compile(r"--hard\b"),
    re.compile(r"--delete\b"),
    re.compile(r"-[dD]\b"),
]

# Sub-commands that are inherently dangerous and disallowed.
_DANGEROUS_SUBCOMMANDS: frozenset[str] = frozenset({
    "push",
    "pull",
    "fetch",
    "merge",
    "rebase",
    "reset",
    "clean",
    "rm",
    "mv",
    "cherry-pick",
    "revert",
    "bisect",
    "reflog",
    "gc",
    "filter-branch",
    "replace",
})


class GitTool(Tool):
    """Run safe git operations in the project repository.

    Supports: status, diff, log, add, commit, branch, checkout,
    show, stash, tag, remote, rev-parse, blame.

    Destructive operations (push, pull, reset --hard, clean, etc.)
    are blocked.  The ``message`` parameter is used only for
    ``commit`` and ``tag`` sub-commands.

    Uses :class:`CommandSandbox` for execution, inheriting its
    timeout, output truncation, and environment filtering.
    """

    def __init__(self, sandbox: CommandSandbox) -> None:
        """Initialise the git tool.

        Args:
            sandbox: A :class:`CommandSandbox` for executing git commands.
        """
        self._sandbox = sandbox

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Run git operations: status, diff, log, commit, add, "
            "checkout, branch, show, stash, tag, remote, blame. "
            "Destructive commands (push, reset --hard, clean) are "
            "blocked."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The git sub-command and its arguments "
                        "(e.g. 'status', 'diff --staged', "
                        "'log -5 --oneline', "
                        "'add src/main.py', "
                        "'commit' (use message param for commit msg), "
                        "'branch -a')."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": (
                        "Commit or tag message. Used only with "
                        "'commit' and 'tag' sub-commands."
                    ),
                },
            },
            "required": ["command"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute a git command and return the result.

        Args:
            arguments: Must contain ``command`` (string); optionally
                ``message`` (string) for commit/tag operations.

        Returns:
            A :class:`ToolResult` with the git command output.
        """
        validated = self.validate_arguments(arguments)
        raw_command: str = validated["command"].strip()
        message: str | None = validated.get("message")

        if not raw_command:
            return ToolResult(
                success=False,
                output="",
                error="Empty git command.",
            )

        # Parse the sub-command from the user input.
        sub_command, error = self._parse_subcommand(raw_command)
        if error is not None:
            return ToolResult(
                success=False,
                output="",
                error=error,
            )

        # Check for dangerous sub-commands.
        if sub_command in _DANGEROUS_SUBCOMMANDS:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Git sub-command '{sub_command}' is not "
                    f"allowed for safety reasons. Allowed: "
                    f"{', '.join(sorted(_ALLOWED_SUBCOMMANDS))}."
                ),
            )

        if sub_command not in _ALLOWED_SUBCOMMANDS:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Unknown or disallowed git sub-command: "
                    f"'{sub_command}'. Allowed: "
                    f"{', '.join(sorted(_ALLOWED_SUBCOMMANDS))}."
                ),
            )

        # Check for blocked flags.
        flag_error = self._check_blocked_flags(raw_command)
        if flag_error is not None:
            return ToolResult(
                success=False,
                output="",
                error=flag_error,
            )

        # Build the full shell command.
        full_command = self._build_command(
            raw_command, sub_command, message
        )

        # Execute via sandbox.
        try:
            result = self._sandbox.execute(
                full_command, timeout=30
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Git command failed: {exc}",
                metadata={"command": full_command},
            )

        # Build output.
        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            # Git often writes informational messages to stderr
            # (e.g. "Switched to branch ...").  Include them.
            output_parts.append(result.stderr)

        output = (
            "\n".join(output_parts) if output_parts else "(no output)"
        )

        success = result.exit_code == 0 and not result.timed_out
        error_msg: str | None = None
        if result.timed_out:
            error_msg = "Git command timed out after 30s"
        elif result.exit_code != 0:
            error_msg = (
                f"Git command exited with code {result.exit_code}"
            )

        return ToolResult(
            success=success,
            output=output,
            error=error_msg,
            metadata={
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "timed_out": result.timed_out,
                "sub_command": sub_command,
                "full_command": full_command,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_subcommand(
        raw_command: str,
    ) -> tuple[str, str | None]:
        """Extract the git sub-command from the raw command string.

        Args:
            raw_command: The user-supplied command string (without
                leading ``git``).

        Returns:
            A tuple of ``(sub_command, error_message)``.  On success
            *error_message* is ``None``.
        """
        # Strip a leading "git " if the user included it.
        cleaned = raw_command
        if cleaned.lower().startswith("git "):
            cleaned = cleaned[4:].strip()
        elif cleaned.lower() == "git":
            return "", "Incomplete git command."

        try:
            tokens = shlex.split(cleaned)
        except ValueError as exc:
            return "", f"Failed to parse command: {exc}"

        if not tokens:
            return "", "Empty git sub-command."

        return tokens[0], None

    @staticmethod
    def _check_blocked_flags(raw_command: str) -> str | None:
        """Check for blocked flags in the command string.

        Args:
            raw_command: The full command string to check.

        Returns:
            An error message if a blocked flag is found, else ``None``.
        """
        for pattern in _BLOCKED_FLAG_PATTERNS:
            if pattern.search(raw_command):
                return (
                    f"Blocked flag detected in git command: "
                    f"'{raw_command}'. Destructive flags "
                    f"(--force, --hard, --delete) are not allowed."
                )
        return None

    @staticmethod
    def _build_command(
        raw_command: str,
        sub_command: str,
        message: str | None,
    ) -> str:
        """Build the full shell command string.

        Handles the ``-m`` flag for commits and tags, and ensures the
        command starts with ``git``.

        Args:
            raw_command: User-supplied command string.
            sub_command: The parsed sub-command name.
            message: Optional commit/tag message.

        Returns:
            Complete shell command string ready for execution.
        """
        # Strip leading "git " if present to avoid doubling.
        cleaned = raw_command
        if cleaned.lower().startswith("git "):
            cleaned = cleaned[4:].strip()

        # For commit and tag: inject -m flag if message provided
        # and not already present.
        if message and sub_command in ("commit", "tag") and "-m" not in cleaned:
            escaped_msg = shlex.quote(message)
            cleaned = f"{cleaned} -m {escaped_msg}"

        return f"git {cleaned}"

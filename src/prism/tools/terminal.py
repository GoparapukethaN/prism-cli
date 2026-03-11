"""Terminal command execution tool."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

if TYPE_CHECKING:
    from prism.security.sandbox import CommandSandbox

logger = structlog.get_logger(__name__)

# Commands that elevate the permission level to DANGEROUS.
_DANGEROUS_PATTERNS: list[str] = [
    r"^rm\s",
    r"^git\s+(reset|push\s+.*--force|clean)",
    r"^docker\s+rm",
    r"^docker\s+system\s+prune",
    r"^kill\s",
    r"^pkill\s",
    r"^killall\s",
    r"^shutdown",
    r"^reboot",
]


class ExecuteCommandTool(Tool):
    """Execute a shell command in the project directory.

    Uses :class:`CommandSandbox` for security guardrails including
    blocked-command checking, environment filtering, and output truncation.

    Permission level is CONFIRM by default, escalated to DANGEROUS for
    destructive commands.
    """

    def __init__(self, sandbox: CommandSandbox) -> None:
        self._sandbox = sandbox
        # Pre-compile dangerous patterns
        self._dangerous_compiled: list[re.Pattern[str]] = [
            re.compile(p) for p in _DANGEROUS_PATTERNS
        ]

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "execute_command"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command in the project directory. "
            "Requires confirmation unless in allowlist."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 30, max 300.",
                    "default": 30,
                },
            },
            "required": ["command"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM

    def get_effective_permission(self, command: str) -> PermissionLevel:
        """Determine the effective permission level for a specific command.

        Returns DANGEROUS for destructive commands, otherwise CONFIRM.
        """
        stripped = command.strip()
        for compiled in self._dangerous_compiled:
            if compiled.search(stripped):
                return PermissionLevel.DANGEROUS
        return PermissionLevel.CONFIRM

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute a shell command and return the result."""
        validated = self.validate_arguments(arguments)
        command: str = validated["command"]
        timeout: int = validated.get("timeout", 30)

        # Clamp timeout to [1, 300]
        timeout = max(1, min(300, timeout))

        # Run through sandbox (handles blocked-command check internally)
        try:
            result = self._sandbox.execute(command, timeout=timeout)
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=str(exc),
                metadata={"command": command},
            )

        # Build output
        output_parts: list[str] = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        success = result.exit_code == 0 and not result.timed_out
        error: str | None = None
        if result.timed_out:
            error = f"Command timed out after {timeout}s"
        elif result.exit_code != 0:
            error = f"Command exited with code {result.exit_code}"

        return ToolResult(
            success=success,
            output=output,
            error=error,
            metadata={
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "timed_out": result.timed_out,
                "command": command,
            },
        )

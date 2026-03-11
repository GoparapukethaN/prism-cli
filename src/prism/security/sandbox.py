"""Sandboxed command execution with security guardrails."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from prism.config.defaults import BLOCKED_COMMAND_PATTERNS
from prism.exceptions import BlockedCommandError, SecurityError
from prism.security.secret_filter import SecretFilter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

logger = structlog.get_logger(__name__)

# Defaults matching ToolsConfig schema
_DEFAULT_TIMEOUT: int = 30
_DEFAULT_MAX_STDOUT: int = 102_400   # 100 KB
_DEFAULT_MAX_STDERR: int = 10_240    # 10 KB


@dataclass(frozen=True)
class CommandResult:
    """Result of a sandboxed command execution.

    Attributes:
        stdout:      Captured standard output (may be truncated).
        stderr:      Captured standard error (may be truncated).
        exit_code:   Process exit code.  ``-1`` when the process was killed
                     due to a timeout.
        duration_ms: Wall-clock execution time in milliseconds.
        timed_out:   Whether the process exceeded its timeout.
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0
    timed_out: bool = False


class CommandSandbox:
    """Execute shell commands inside a constrained sandbox.

    Guardrails:
    - Working directory locked to the project root.
    - Configurable timeout (default 30 s).
    - stdout/stderr size limits (default 100 KB / 10 KB) — excess is truncated.
    - Commands are checked against blocked patterns before execution.
    - Environment variables are filtered to strip secrets.

    Usage::

        sandbox = CommandSandbox(project_root=Path("/my/project"))
        result = sandbox.execute("ls -la")
        print(result.stdout)
    """

    def __init__(
        self,
        project_root: Path,
        timeout: int = _DEFAULT_TIMEOUT,
        max_stdout_bytes: int = _DEFAULT_MAX_STDOUT,
        max_stderr_bytes: int = _DEFAULT_MAX_STDERR,
        extra_blocked_patterns: list[str] | None = None,
        secret_filter: SecretFilter | None = None,
    ) -> None:
        """Initialise the sandbox.

        Args:
            project_root:           Directory to use as the working directory
                                    for all executed commands.  Resolved to an
                                    absolute path on construction.
            timeout:                Maximum execution time in seconds.
            max_stdout_bytes:       Maximum captured stdout size in bytes.
            max_stderr_bytes:       Maximum captured stderr size in bytes.
            extra_blocked_patterns: Additional regex patterns to block, on top
                                    of ``BLOCKED_COMMAND_PATTERNS``.
            secret_filter:          A :class:`SecretFilter` instance.  When
                                    ``None`` a default instance is created.
        """
        self._project_root = project_root.resolve()
        self._timeout = timeout
        self._max_stdout_bytes = max_stdout_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._secret_filter = secret_filter or SecretFilter()

        # Compile blocked-command regexes once
        raw_patterns = list(BLOCKED_COMMAND_PATTERNS)
        if extra_blocked_patterns:
            raw_patterns.extend(extra_blocked_patterns)
        self._blocked_patterns: list[tuple[str, re.Pattern[str]]] = []
        for pat in raw_patterns:
            try:
                self._blocked_patterns.append((pat, re.compile(pat)))
            except re.error as exc:
                logger.warning("invalid_blocked_pattern", pattern=pat, error=str(exc))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        """Return the resolved project root."""
        return self._project_root

    @property
    def timeout(self) -> int:
        """Return the configured timeout in seconds."""
        return self._timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_command(self, command: str) -> None:
        """Validate *command* against blocked patterns.

        Raises:
            BlockedCommandError: If the command matches a blocked pattern.
        """
        stripped = command.strip()
        for raw_pattern, compiled in self._blocked_patterns:
            if compiled.search(stripped):
                logger.warning(
                    "blocked_command",
                    command=stripped,
                    pattern=raw_pattern,
                )
                raise BlockedCommandError(stripped, raw_pattern)

    def execute(
        self,
        command: str | Sequence[str],
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Execute *command* inside the sandbox and return the result.

        Args:
            command: A shell command string or a sequence of arguments.
                     When a string is given it is executed via ``shell=True``.
            timeout: Override the default timeout for this invocation.
            env:     Extra environment variables (merged on top of the
                     filtered environment).

        Returns:
            A :class:`CommandResult` with captured output and metadata.

        Raises:
            BlockedCommandError: If the command matches a blocked pattern.
            SecurityError:       If the project root does not exist.
        """
        # Determine shell mode
        if isinstance(command, str):
            shell = True
            cmd_for_check = command
            cmd_arg: str | Sequence[str] = command
        else:
            shell = False
            cmd_for_check = " ".join(command)
            cmd_arg = list(command)

        # Validate the command is not blocked
        self.check_command(cmd_for_check)

        # Validate working directory
        if not self._project_root.is_dir():
            raise SecurityError(
                f"Sandbox project root does not exist: {self._project_root}"
            )

        effective_timeout = timeout if timeout is not None else self._timeout

        # Build filtered environment
        filtered_env = self._secret_filter.filter_env()
        if env:
            # Merge caller-provided env vars (also filter them)
            for key, value in env.items():
                if not self._secret_filter.is_sensitive(key):
                    filtered_env[key] = value

        start = time.monotonic()
        timed_out = False
        exit_code = 0
        raw_stdout = b""
        raw_stderr = b""

        try:
            proc = subprocess.run(
                cmd_arg,
                shell=shell,
                cwd=str(self._project_root),
                env=filtered_env,
                capture_output=True,
                timeout=effective_timeout,
            )
            raw_stdout = proc.stdout
            raw_stderr = proc.stderr
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -1
            raw_stdout = exc.stdout or b""
            raw_stderr = exc.stderr or b""
            logger.warning(
                "command_timed_out",
                command=cmd_for_check,
                timeout=effective_timeout,
            )
        except OSError as exc:
            exit_code = -1
            raw_stderr = str(exc).encode("utf-8", errors="replace")
            logger.error(
                "command_execution_error",
                command=cmd_for_check,
                error=str(exc),
            )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        # Truncate output
        stdout_text = self._truncate(raw_stdout, self._max_stdout_bytes)
        stderr_text = self._truncate(raw_stderr, self._max_stderr_bytes)

        return CommandResult(
            stdout=stdout_text,
            stderr=stderr_text,
            exit_code=exit_code,
            duration_ms=round(elapsed_ms, 2),
            timed_out=timed_out,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(data: bytes, max_bytes: int) -> str:
        """Decode *data* and truncate to *max_bytes*.

        If the data exceeds the limit, a truncation notice is appended.
        """
        if len(data) <= max_bytes:
            return data.decode("utf-8", errors="replace")

        truncated = data[:max_bytes]
        text = truncated.decode("utf-8", errors="replace")
        remaining = len(data) - max_bytes
        return text + f"\n... [truncated {remaining} bytes]"

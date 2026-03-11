"""Code execution sandbox — isolated code execution with Docker or subprocess fallback.

Provides a safe environment for running generated code before applying it to the
main codebase.  Tries Docker first for full isolation (network, memory, PID limits);
falls back to a restricted subprocess when Docker is unavailable.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_TIMEOUT: int = 30
DEFAULT_MEMORY_MB: int = 512
MAX_TIMEOUT: int = 300
MAX_OUTPUT_CHARS: int = 50_000


class SandboxType(Enum):
    """Type of sandbox used for code execution."""

    DOCKER = "docker"
    SUBPROCESS = "subprocess"


@dataclass(frozen=True)
class SandboxResult:
    """Result of sandbox code execution.

    Attributes:
        stdout:            Captured standard output (truncated to 50 000 chars).
        stderr:            Captured standard error (truncated to 50 000 chars).
        exit_code:         Process exit code.  ``-1`` on timeout.
        execution_time_ms: Wall-clock execution time in milliseconds.
        sandbox_type:      The sandbox backend that was used (``"docker"`` or
                           ``"subprocess"``).
        timed_out:         Whether execution exceeded the configured timeout.
        memory_exceeded:   Whether the process was killed due to memory limits
                           (Docker OOM-kill, exit code 137).
    """

    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    sandbox_type: str
    timed_out: bool
    memory_exceeded: bool


class CodeSandbox:
    """Isolated code execution environment with Docker or subprocess fallback.

    Usage::

        sandbox = CodeSandbox(timeout=30, memory_mb=512)
        result = sandbox.execute("print('hello')", language="python")
        print(result.stdout)

    Toggle with ``/sandbox on`` and ``/sandbox off`` in the REPL.
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        memory_mb: int = DEFAULT_MEMORY_MB,
        network_enabled: bool = False,
        enabled: bool = True,
    ) -> None:
        """Initialise the code sandbox.

        Args:
            timeout:         Maximum execution time in seconds (1-300).
            memory_mb:       Memory limit in megabytes for Docker containers.
            network_enabled: Whether to allow network access inside the sandbox.
            enabled:         Whether the sandbox is active.  When disabled,
                             :meth:`execute` raises ``RuntimeError``.
        """
        if timeout < 1:
            raise ValueError("Timeout must be at least 1 second")
        if timeout > MAX_TIMEOUT:
            raise ValueError(f"Timeout cannot exceed {MAX_TIMEOUT} seconds")
        if memory_mb < 1:
            raise ValueError("Memory limit must be at least 1 MB")

        self._timeout = timeout
        self._memory_mb = memory_mb
        self._network_enabled = network_enabled
        self._enabled = enabled
        self._docker_available: bool | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether the sandbox is currently enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        logger.info("sandbox_toggled", enabled=value)

    @property
    def timeout(self) -> int:
        """Configured timeout in seconds."""
        return self._timeout

    @timeout.setter
    def timeout(self, value: int) -> None:
        if value < 1:
            raise ValueError("Timeout must be at least 1 second")
        if value > MAX_TIMEOUT:
            raise ValueError(f"Timeout cannot exceed {MAX_TIMEOUT} seconds")
        self._timeout = value

    @property
    def memory_mb(self) -> int:
        """Configured memory limit in megabytes."""
        return self._memory_mb

    @property
    def network_enabled(self) -> bool:
        """Whether network access is allowed in the sandbox."""
        return self._network_enabled

    @network_enabled.setter
    def network_enabled(self, value: bool) -> None:
        self._network_enabled = value

    # ------------------------------------------------------------------
    # Docker detection
    # ------------------------------------------------------------------

    def check_docker(self) -> bool:
        """Check if Docker is available and running.

        The result is cached after the first call.

        Returns:
            True if ``docker info`` succeeds, False otherwise.
        """
        if self._docker_available is not None:
            return self._docker_available
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            self._docker_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._docker_available = False
        logger.info("docker_check", available=self._docker_available)
        return self._docker_available

    def reset_docker_cache(self) -> None:
        """Clear the cached Docker availability result."""
        self._docker_available = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        code: str,
        language: str = "python",
        working_dir: Path | None = None,
    ) -> SandboxResult:
        """Execute code in the sandbox.

        Tries Docker first for full isolation, falls back to a restricted
        subprocess if Docker is unavailable.

        Args:
            code:        Source code to execute.
            language:    Programming language (``"python"``, ``"javascript"``,
                         ``"bash"``, etc.).
            working_dir: Optional directory for temporary file creation.

        Returns:
            A :class:`SandboxResult` with captured output and metadata.

        Raises:
            RuntimeError: If the sandbox is disabled.
            ValueError:   If *code* is empty.
        """
        if not self._enabled:
            raise RuntimeError("Sandbox is disabled. Enable with /sandbox on")

        if not code or not code.strip():
            raise ValueError("Code must not be empty")

        language = language.lower().strip()

        if self.check_docker():
            return self._execute_docker(code, language, working_dir)
        return self._execute_subprocess(code, language, working_dir)

    # ------------------------------------------------------------------
    # Docker execution
    # ------------------------------------------------------------------

    def _execute_docker(
        self,
        code: str,
        language: str,
        working_dir: Path | None,
    ) -> SandboxResult:
        """Execute code in a Docker container with resource limits."""
        image = self._get_docker_image(language)
        extension = self._get_extension(language)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=extension,
            delete=False,
            dir=working_dir,
        ) as f:
            f.write(code)
            code_path = Path(f.name)

        try:
            cmd: list[str] = ["docker", "run", "--rm"]
            cmd.extend(["--memory", f"{self._memory_mb}m"])
            cmd.extend(["--cpus", "1.0"])
            cmd.extend(["--pids-limit", "100"])

            if not self._network_enabled:
                cmd.extend(["--network", "none"])

            # Read-only mount of the code file into the container
            cmd.extend([
                "-v",
                f"{code_path}:/sandbox/code{extension}:ro",
            ])
            cmd.extend(["-w", "/sandbox"])
            cmd.append(image)
            cmd.extend(self._get_run_command(language))

            logger.debug(
                "docker_execute",
                image=image,
                language=language,
                timeout=self._timeout,
            )

            start = time.monotonic()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            elapsed = (time.monotonic() - start) * 1000

            return SandboxResult(
                stdout=result.stdout[:MAX_OUTPUT_CHARS],
                stderr=result.stderr[:MAX_OUTPUT_CHARS],
                exit_code=result.returncode,
                execution_time_ms=round(elapsed, 2),
                sandbox_type=SandboxType.DOCKER.value,
                timed_out=False,
                memory_exceeded=result.returncode == 137,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "docker_timeout",
                language=language,
                timeout=self._timeout,
            )
            # Kill the container — the process may have been left running
            return SandboxResult(
                stdout="",
                stderr=f"Execution timed out after {self._timeout}s",
                exit_code=-1,
                execution_time_ms=self._timeout * 1000,
                sandbox_type=SandboxType.DOCKER.value,
                timed_out=True,
                memory_exceeded=False,
            )
        finally:
            code_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------

    def _execute_subprocess(
        self,
        code: str,
        language: str,
        working_dir: Path | None,
    ) -> SandboxResult:
        """Execute code in a restricted subprocess (no Docker)."""
        extension = self._get_extension(language)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=extension,
            delete=False,
            dir=working_dir,
        ) as f:
            f.write(code)
            code_path = Path(f.name)

        try:
            interpreter = self._get_interpreter(language)

            # Build a clean, minimal environment
            env: dict[str, str] = {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "HOME": tempfile.gettempdir(),
                "LANG": "en_US.UTF-8",
            }

            cwd = str(working_dir) if working_dir else tempfile.gettempdir()

            logger.debug(
                "subprocess_execute",
                interpreter=interpreter,
                language=language,
                timeout=self._timeout,
            )

            start = time.monotonic()
            result = subprocess.run(
                [interpreter, str(code_path)],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=cwd,
                env=env,
                check=False,
            )
            elapsed = (time.monotonic() - start) * 1000

            return SandboxResult(
                stdout=result.stdout[:MAX_OUTPUT_CHARS],
                stderr=result.stderr[:MAX_OUTPUT_CHARS],
                exit_code=result.returncode,
                execution_time_ms=round(elapsed, 2),
                sandbox_type=SandboxType.SUBPROCESS.value,
                timed_out=False,
                memory_exceeded=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "subprocess_timeout",
                language=language,
                timeout=self._timeout,
            )
            return SandboxResult(
                stdout="",
                stderr=f"Execution timed out after {self._timeout}s",
                exit_code=-1,
                execution_time_ms=self._timeout * 1000,
                sandbox_type=SandboxType.SUBPROCESS.value,
                timed_out=True,
                memory_exceeded=False,
            )
        finally:
            code_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Language helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_docker_image(language: str) -> str:
        """Return the Docker image for the given language."""
        images: dict[str, str] = {
            "python": "python:3.12-slim",
            "javascript": "node:20-slim",
            "typescript": "node:20-slim",
            "ruby": "ruby:3.3-slim",
            "go": "golang:1.22-alpine",
            "rust": "rust:1.77-slim",
            "bash": "bash:5",
            "sh": "bash:5",
        }
        return images.get(language, "python:3.12-slim")

    @staticmethod
    def _get_extension(language: str) -> str:
        """Return the file extension for the given language."""
        extensions: dict[str, str] = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "ruby": ".rb",
            "go": ".go",
            "rust": ".rs",
            "bash": ".sh",
            "sh": ".sh",
        }
        return extensions.get(language, ".py")

    @staticmethod
    def _get_interpreter(language: str) -> str:
        """Return the local interpreter for the given language."""
        interpreters: dict[str, str] = {
            "python": "python3",
            "javascript": "node",
            "ruby": "ruby",
            "bash": "bash",
            "sh": "sh",
        }
        return interpreters.get(language, "python3")

    @staticmethod
    def _get_run_command(language: str) -> list[str]:
        """Return the command to run code inside a Docker container."""
        commands: dict[str, list[str]] = {
            "python": ["python3", "/sandbox/code.py"],
            "javascript": ["node", "/sandbox/code.js"],
            "typescript": ["node", "/sandbox/code.ts"],
            "ruby": ["ruby", "/sandbox/code.rb"],
            "bash": ["bash", "/sandbox/code.sh"],
            "sh": ["sh", "/sandbox/code.sh"],
        }
        return commands.get(language, ["python3", "/sandbox/code.py"])

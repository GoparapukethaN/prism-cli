"""Tests for the code execution sandbox."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from prism.tools.code_sandbox import (
    DEFAULT_MEMORY_MB,
    DEFAULT_TIMEOUT,
    MAX_OUTPUT_CHARS,
    MAX_TIMEOUT,
    CodeSandbox,
    SandboxResult,
    SandboxType,
)

if TYPE_CHECKING:
    from pathlib import Path


# =====================================================================
# SandboxType
# =====================================================================


class TestSandboxType:
    """Tests for the SandboxType enum."""

    def test_docker_value(self) -> None:
        assert SandboxType.DOCKER.value == "docker"

    def test_subprocess_value(self) -> None:
        assert SandboxType.SUBPROCESS.value == "subprocess"

    def test_all_members(self) -> None:
        assert set(SandboxType.__members__) == {"DOCKER", "SUBPROCESS"}


# =====================================================================
# SandboxResult
# =====================================================================


class TestSandboxResult:
    """Tests for the SandboxResult dataclass."""

    def test_fields(self) -> None:
        result = SandboxResult(
            stdout="hello",
            stderr="",
            exit_code=0,
            execution_time_ms=42.5,
            sandbox_type="docker",
            timed_out=False,
            memory_exceeded=False,
        )
        assert result.stdout == "hello"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.execution_time_ms == 42.5
        assert result.sandbox_type == "docker"
        assert result.timed_out is False
        assert result.memory_exceeded is False

    def test_frozen(self) -> None:
        result = SandboxResult(
            stdout="x",
            stderr="",
            exit_code=0,
            execution_time_ms=0.0,
            sandbox_type="subprocess",
            timed_out=False,
            memory_exceeded=False,
        )
        with pytest.raises(AttributeError):
            result.stdout = "changed"  # type: ignore[misc]

    def test_timeout_result(self) -> None:
        result = SandboxResult(
            stdout="",
            stderr="Timed out",
            exit_code=-1,
            execution_time_ms=30000.0,
            sandbox_type="docker",
            timed_out=True,
            memory_exceeded=False,
        )
        assert result.timed_out is True
        assert result.exit_code == -1

    def test_memory_exceeded_result(self) -> None:
        result = SandboxResult(
            stdout="",
            stderr="OOM",
            exit_code=137,
            execution_time_ms=500.0,
            sandbox_type="docker",
            timed_out=False,
            memory_exceeded=True,
        )
        assert result.memory_exceeded is True
        assert result.exit_code == 137


# =====================================================================
# CodeSandbox — Initialisation
# =====================================================================


class TestCodeSandboxInit:
    """Tests for CodeSandbox initialisation."""

    def test_defaults(self) -> None:
        sandbox = CodeSandbox()
        assert sandbox.timeout == DEFAULT_TIMEOUT
        assert sandbox.memory_mb == DEFAULT_MEMORY_MB
        assert sandbox.network_enabled is False
        assert sandbox.enabled is True

    def test_custom_values(self) -> None:
        sandbox = CodeSandbox(
            timeout=60,
            memory_mb=1024,
            network_enabled=True,
            enabled=False,
        )
        assert sandbox.timeout == 60
        assert sandbox.memory_mb == 1024
        assert sandbox.network_enabled is True
        assert sandbox.enabled is False

    def test_timeout_too_low(self) -> None:
        with pytest.raises(ValueError, match="at least 1 second"):
            CodeSandbox(timeout=0)

    def test_timeout_too_high(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            CodeSandbox(timeout=999)

    def test_memory_too_low(self) -> None:
        with pytest.raises(ValueError, match="at least 1 MB"):
            CodeSandbox(memory_mb=0)


# =====================================================================
# CodeSandbox — Properties
# =====================================================================


class TestCodeSandboxProperties:
    """Tests for CodeSandbox property setters."""

    def test_enabled_toggle(self) -> None:
        sandbox = CodeSandbox()
        assert sandbox.enabled is True
        sandbox.enabled = False
        assert sandbox.enabled is False
        sandbox.enabled = True
        assert sandbox.enabled is True

    def test_timeout_setter(self) -> None:
        sandbox = CodeSandbox()
        sandbox.timeout = 120
        assert sandbox.timeout == 120

    def test_timeout_setter_too_low(self) -> None:
        sandbox = CodeSandbox()
        with pytest.raises(ValueError, match="at least 1 second"):
            sandbox.timeout = 0

    def test_timeout_setter_too_high(self) -> None:
        sandbox = CodeSandbox()
        with pytest.raises(ValueError, match="cannot exceed"):
            sandbox.timeout = MAX_TIMEOUT + 1

    def test_network_enabled_setter(self) -> None:
        sandbox = CodeSandbox()
        assert sandbox.network_enabled is False
        sandbox.network_enabled = True
        assert sandbox.network_enabled is True


# =====================================================================
# CodeSandbox — Docker detection
# =====================================================================


class TestDockerDetection:
    """Tests for Docker availability checking."""

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_available(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        sandbox = CodeSandbox()
        assert sandbox.check_docker() is True

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_not_available(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        sandbox = CodeSandbox()
        assert sandbox.check_docker() is False

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_not_installed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("docker not found")
        sandbox = CodeSandbox()
        assert sandbox.check_docker() is False

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker info", timeout=10)
        sandbox = CodeSandbox()
        assert sandbox.check_docker() is False

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_result_cached(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        sandbox = CodeSandbox()
        assert sandbox.check_docker() is True
        assert sandbox.check_docker() is True
        # Called only once due to caching
        assert mock_run.call_count == 1

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_cache_reset(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        sandbox = CodeSandbox()
        assert sandbox.check_docker() is False
        sandbox.reset_docker_cache()
        mock_run.return_value = MagicMock(returncode=0)
        assert sandbox.check_docker() is True
        assert mock_run.call_count == 2


# =====================================================================
# CodeSandbox — Execute (subprocess fallback)
# =====================================================================


class TestSubprocessExecution:
    """Tests for subprocess-based code execution."""

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_successful_execution(self, mock_run: MagicMock, tmp_path: Path) -> None:
        # First call: docker info fails (no Docker)
        # Second call: actual code execution succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1),  # docker info
            MagicMock(returncode=0, stdout="hello\n", stderr=""),  # python3
        ]

        sandbox = CodeSandbox()
        result = sandbox.execute("print('hello')", language="python", working_dir=tmp_path)

        assert result.exit_code == 0
        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.sandbox_type == "subprocess"
        assert result.timed_out is False
        assert result.memory_exceeded is False
        assert result.execution_time_ms >= 0

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_execution_failure(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=1),  # docker info
            MagicMock(returncode=1, stdout="", stderr="SyntaxError: invalid"),
        ]

        sandbox = CodeSandbox()
        result = sandbox.execute("def (", language="python", working_dir=tmp_path)

        assert result.exit_code == 1
        assert "SyntaxError" in result.stderr

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_execution_timeout(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=1),  # docker info
            subprocess.TimeoutExpired(cmd="python3", timeout=30),
        ]

        sandbox = CodeSandbox()
        result = sandbox.execute("import time; time.sleep(999)", working_dir=tmp_path)

        assert result.timed_out is True
        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower()
        assert result.sandbox_type == "subprocess"

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_output_truncation(self, mock_run: MagicMock, tmp_path: Path) -> None:
        big_output = "x" * (MAX_OUTPUT_CHARS + 1000)
        mock_run.side_effect = [
            MagicMock(returncode=1),  # docker info
            MagicMock(returncode=0, stdout=big_output, stderr=""),
        ]

        sandbox = CodeSandbox()
        result = sandbox.execute("print('x' * 100000)", working_dir=tmp_path)

        assert len(result.stdout) == MAX_OUTPUT_CHARS

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_javascript_language(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=1),  # docker info
            MagicMock(returncode=0, stdout="42\n", stderr=""),
        ]

        sandbox = CodeSandbox()
        result = sandbox.execute("console.log(42)", language="javascript", working_dir=tmp_path)

        assert result.exit_code == 0
        # Verify the interpreter used was 'node'
        call_args = mock_run.call_args_list[1]
        assert "node" in call_args[0][0][0]

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_bash_language(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=1),  # docker info
            MagicMock(returncode=0, stdout="ok\n", stderr=""),
        ]

        sandbox = CodeSandbox()
        result = sandbox.execute("echo ok", language="bash", working_dir=tmp_path)

        assert result.exit_code == 0
        call_args = mock_run.call_args_list[1]
        assert "bash" in call_args[0][0][0]


# =====================================================================
# CodeSandbox — Execute (Docker)
# =====================================================================


class TestDockerExecution:
    """Tests for Docker-based code execution."""

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_execution_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=0),  # docker info
            MagicMock(returncode=0, stdout="hello\n", stderr=""),  # docker run
        ]

        sandbox = CodeSandbox()
        result = sandbox.execute("print('hello')", working_dir=tmp_path)

        assert result.exit_code == 0
        assert result.stdout == "hello\n"
        assert result.sandbox_type == "docker"
        assert result.timed_out is False
        assert result.memory_exceeded is False

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_network_disabled(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=0),  # docker info
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        sandbox = CodeSandbox(network_enabled=False)
        sandbox.execute("print(1)", working_dir=tmp_path)

        docker_cmd = mock_run.call_args_list[1][0][0]
        assert "--network" in docker_cmd
        assert "none" in docker_cmd

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_network_enabled(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=0),  # docker info
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        sandbox = CodeSandbox(network_enabled=True)
        sandbox.execute("print(1)", working_dir=tmp_path)

        docker_cmd = mock_run.call_args_list[1][0][0]
        assert "--network" not in docker_cmd

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_memory_limit(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        sandbox = CodeSandbox(memory_mb=256)
        sandbox.execute("print(1)", working_dir=tmp_path)

        docker_cmd = mock_run.call_args_list[1][0][0]
        assert "--memory" in docker_cmd
        mem_idx = docker_cmd.index("--memory")
        assert docker_cmd[mem_idx + 1] == "256m"

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_oom_kill(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=137, stdout="", stderr="Killed"),
        ]

        sandbox = CodeSandbox()
        result = sandbox.execute("x = 'a' * 10**9", working_dir=tmp_path)

        assert result.exit_code == 137
        assert result.memory_exceeded is True
        assert result.sandbox_type == "docker"

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_timeout(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=0),
            subprocess.TimeoutExpired(cmd="docker", timeout=30),
        ]

        sandbox = CodeSandbox()
        result = sandbox.execute("import time; time.sleep(999)", working_dir=tmp_path)

        assert result.timed_out is True
        assert result.exit_code == -1
        assert result.sandbox_type == "docker"

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_docker_pids_limit(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        sandbox = CodeSandbox()
        sandbox.execute("print(1)", working_dir=tmp_path)

        docker_cmd = mock_run.call_args_list[1][0][0]
        assert "--pids-limit" in docker_cmd
        pid_idx = docker_cmd.index("--pids-limit")
        assert docker_cmd[pid_idx + 1] == "100"


# =====================================================================
# CodeSandbox — Edge Cases
# =====================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_disabled_sandbox_raises(self) -> None:
        sandbox = CodeSandbox(enabled=False)
        with pytest.raises(RuntimeError, match="Sandbox is disabled"):
            sandbox.execute("print(1)")

    def test_empty_code_raises(self) -> None:
        sandbox = CodeSandbox()
        with pytest.raises(ValueError, match="must not be empty"):
            sandbox.execute("")

    def test_whitespace_only_code_raises(self) -> None:
        sandbox = CodeSandbox()
        with pytest.raises(ValueError, match="must not be empty"):
            sandbox.execute("   \n  ")

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_temp_file_cleaned_up_on_success(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=1),
            MagicMock(returncode=0, stdout="ok", stderr=""),
        ]

        sandbox = CodeSandbox()
        sandbox.execute("print('ok')", working_dir=tmp_path)

        # No leftover temp files — only the original project files
        py_files = list(tmp_path.glob("*.py"))
        assert len(py_files) == 0

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_temp_file_cleaned_up_on_timeout(
        self, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=1),
            subprocess.TimeoutExpired(cmd="python3", timeout=30),
        ]

        sandbox = CodeSandbox()
        sandbox.execute("import time; time.sleep(999)", working_dir=tmp_path)

        py_files = list(tmp_path.glob("*.py"))
        assert len(py_files) == 0

    @patch("prism.tools.code_sandbox.subprocess.run")
    def test_language_case_insensitive(self, mock_run: MagicMock, tmp_path: Path) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=1),
            MagicMock(returncode=0, stdout="", stderr=""),
        ]

        sandbox = CodeSandbox()
        result = sandbox.execute("print(1)", language="Python", working_dir=tmp_path)

        assert result.sandbox_type == "subprocess"


# =====================================================================
# CodeSandbox — Language helpers
# =====================================================================


class TestLanguageHelpers:
    """Tests for static language helper methods."""

    def test_docker_images(self) -> None:
        assert CodeSandbox._get_docker_image("python") == "python:3.12-slim"
        assert CodeSandbox._get_docker_image("javascript") == "node:20-slim"
        assert CodeSandbox._get_docker_image("typescript") == "node:20-slim"
        assert CodeSandbox._get_docker_image("ruby") == "ruby:3.3-slim"
        assert CodeSandbox._get_docker_image("go") == "golang:1.22-alpine"
        assert CodeSandbox._get_docker_image("rust") == "rust:1.77-slim"
        assert CodeSandbox._get_docker_image("bash") == "bash:5"
        assert CodeSandbox._get_docker_image("sh") == "bash:5"

    def test_docker_image_unknown_defaults_to_python(self) -> None:
        assert CodeSandbox._get_docker_image("brainfuck") == "python:3.12-slim"

    def test_extensions(self) -> None:
        assert CodeSandbox._get_extension("python") == ".py"
        assert CodeSandbox._get_extension("javascript") == ".js"
        assert CodeSandbox._get_extension("typescript") == ".ts"
        assert CodeSandbox._get_extension("ruby") == ".rb"
        assert CodeSandbox._get_extension("go") == ".go"
        assert CodeSandbox._get_extension("rust") == ".rs"
        assert CodeSandbox._get_extension("bash") == ".sh"
        assert CodeSandbox._get_extension("sh") == ".sh"

    def test_extension_unknown_defaults_to_py(self) -> None:
        assert CodeSandbox._get_extension("haskell") == ".py"

    def test_interpreters(self) -> None:
        assert CodeSandbox._get_interpreter("python") == "python3"
        assert CodeSandbox._get_interpreter("javascript") == "node"
        assert CodeSandbox._get_interpreter("ruby") == "ruby"
        assert CodeSandbox._get_interpreter("bash") == "bash"
        assert CodeSandbox._get_interpreter("sh") == "sh"

    def test_interpreter_unknown_defaults_to_python3(self) -> None:
        assert CodeSandbox._get_interpreter("cobol") == "python3"

    def test_run_commands(self) -> None:
        assert CodeSandbox._get_run_command("python") == ["python3", "/sandbox/code.py"]
        assert CodeSandbox._get_run_command("javascript") == ["node", "/sandbox/code.js"]
        assert CodeSandbox._get_run_command("bash") == ["bash", "/sandbox/code.sh"]

    def test_run_command_unknown_defaults_to_python(self) -> None:
        assert CodeSandbox._get_run_command("zig") == ["python3", "/sandbox/code.py"]

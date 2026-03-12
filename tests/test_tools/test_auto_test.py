"""Tests for the auto-test runner tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prism.security.sandbox import CommandResult
from prism.tools.auto_test import AutoTestTool
from prism.tools.base import PermissionLevel

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock CommandSandbox."""
    sandbox = MagicMock(spec=["execute"])
    return sandbox


@pytest.fixture
def auto_test_tool(mock_sandbox: MagicMock) -> AutoTestTool:
    """Create an AutoTestTool with a mock sandbox."""
    return AutoTestTool(mock_sandbox)


def _make_result(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    duration_ms: float = 100.0,
    timed_out: bool = False,
) -> CommandResult:
    """Helper to create a CommandResult."""
    return CommandResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        duration_ms=duration_ms,
        timed_out=timed_out,
    )


# ------------------------------------------------------------------
# Property tests
# ------------------------------------------------------------------


class TestAutoTestToolProperties:
    """Tests for tool metadata properties."""

    def test_name(self, auto_test_tool: AutoTestTool) -> None:
        """Tool name is 'auto_test'."""
        assert auto_test_tool.name == "auto_test"

    def test_description(self, auto_test_tool: AutoTestTool) -> None:
        """Description mentions test discovery and running."""
        desc = auto_test_tool.description
        assert "test" in desc.lower()
        assert "changed files" in desc.lower()

    def test_permission_level(self, auto_test_tool: AutoTestTool) -> None:
        """Permission level is AUTO."""
        assert auto_test_tool.permission_level == PermissionLevel.AUTO

    def test_parameters_schema(self, auto_test_tool: AutoTestTool) -> None:
        """Schema has changed_files (required), run_all, coverage, timeout."""
        schema = auto_test_tool.parameters_schema
        assert schema["type"] == "object"
        assert "changed_files" in schema["properties"]
        assert "run_all" in schema["properties"]
        assert "coverage" in schema["properties"]
        assert "timeout" in schema["properties"]
        assert schema["required"] == ["changed_files"]


# ------------------------------------------------------------------
# Framework detection tests
# ------------------------------------------------------------------


class TestFrameworkDetection:
    """Tests for test framework detection."""

    def test_detects_pytest_from_conftest(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Detects pytest from conftest.py presence."""
        call_count = 0

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            nonlocal call_count
            call_count += 1
            if "conftest.py" in cmd:
                return _make_result(stdout="found")
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        framework = auto_test_tool._detect_framework()
        assert framework == "pytest"

    def test_detects_jest_from_config(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Detects jest from jest.config.js presence."""

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            if "jest.config.js" in cmd:
                return _make_result(stdout="found")
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        framework = auto_test_tool._detect_framework()
        assert framework == "jest"

    def test_detects_mocha_from_config(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Detects mocha from .mocharc.yml presence."""

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            if ".mocharc.yml" in cmd:
                return _make_result(stdout="found")
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        framework = auto_test_tool._detect_framework()
        assert framework == "mocha"

    def test_defaults_to_pytest_with_tests_dir(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Falls back to pytest when tests/ directory exists."""

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            if "test -d tests" in cmd:
                return _make_result(stdout="found")
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        framework = auto_test_tool._detect_framework()
        assert framework == "pytest"

    def test_defaults_to_pytest_when_nothing_found(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Falls back to pytest when no indicators found."""
        mock_sandbox.execute.return_value = _make_result(stdout="missing")
        framework = auto_test_tool._detect_framework()
        assert framework == "pytest"

    def test_detects_jest_from_package_json(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Detects jest from package.json contents."""

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            if "package.json" in cmd and "test -f" in cmd:
                return _make_result(stdout="found")
            if "cat package.json" in cmd:
                return _make_result(stdout='{"devDependencies": {"jest": "^29.0"}}')
            if "test -d tests" in cmd:
                return _make_result(stdout="missing")
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        framework = auto_test_tool._detect_framework()
        assert framework == "jest"


# ------------------------------------------------------------------
# Test file discovery tests
# ------------------------------------------------------------------


class TestTestFileDiscovery:
    """Tests for mapping source files to test files."""

    def test_python_flat_mapping(self) -> None:
        """Maps src/foo.py -> tests/test_foo.py."""
        from pathlib import PurePosixPath

        candidates = AutoTestTool._python_test_candidates(
            PurePosixPath("src/foo.py"), "foo", ".py"
        )
        assert "tests/test_foo.py" in candidates

    def test_python_nested_mapping(self) -> None:
        """Maps src/prism/tools/terminal.py -> tests/test_tools/test_terminal.py."""
        from pathlib import PurePosixPath

        candidates = AutoTestTool._python_test_candidates(
            PurePosixPath("src/prism/tools/terminal.py"), "terminal", ".py"
        )
        assert "tests/test_tools/test_terminal.py" in candidates

    def test_python_non_py_skipped(self) -> None:
        """Non-.py files return no candidates."""
        from pathlib import PurePosixPath

        candidates = AutoTestTool._python_test_candidates(
            PurePosixPath("src/config.yaml"), "config", ".yaml"
        )
        assert candidates == []

    def test_js_test_candidates(self) -> None:
        """Maps src/components/Widget.tsx to multiple candidates."""
        from pathlib import PurePosixPath

        candidates = AutoTestTool._js_test_candidates(
            PurePosixPath("src/components/Widget.tsx"), "Widget", ".tsx"
        )
        assert "src/components/__tests__/Widget.test.tsx" in candidates
        assert "src/components/Widget.test.tsx" in candidates
        assert "src/components/Widget.spec.tsx" in candidates
        assert "tests/Widget.test.tsx" in candidates

    def test_js_non_js_skipped(self) -> None:
        """Non-JS/TS files return no candidates."""
        from pathlib import PurePosixPath

        candidates = AutoTestTool._js_test_candidates(
            PurePosixPath("styles.css"), "styles", ".css"
        )
        assert candidates == []

    def test_already_test_file_kept(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Files starting with test_ are kept as-is."""

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            if "test -f tests/test_foo.py" in cmd:
                return _make_result(stdout="found")
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        result = auto_test_tool._find_test_files(
            ["tests/test_foo.py"], "pytest"
        )
        assert "tests/test_foo.py" in result

    def test_no_existing_test_files(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Returns empty when no test files exist."""
        mock_sandbox.execute.return_value = _make_result(stdout="missing")
        result = auto_test_tool._find_test_files(
            ["src/brand_new.py"], "pytest"
        )
        assert result == []

    def test_deduplication(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Duplicate candidates are deduplicated."""

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            if "test -f tests/test_foo.py" in cmd:
                return _make_result(stdout="found")
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        # Two different source files mapping to the same test file
        result = auto_test_tool._find_test_files(
            ["src/foo.py", "lib/foo.py"], "pytest"
        )
        # Should only appear once
        assert result.count("tests/test_foo.py") == 1


# ------------------------------------------------------------------
# Command building tests
# ------------------------------------------------------------------


class TestCommandBuilding:
    """Tests for test command construction."""

    def test_pytest_basic(self, auto_test_tool: AutoTestTool) -> None:
        """Pytest command with specific test files."""
        cmd = auto_test_tool._build_command(
            "pytest", ["tests/test_foo.py"], coverage=False
        )
        assert "python -m pytest" in cmd
        assert "-v" in cmd
        assert "tests/test_foo.py" in cmd
        assert "--cov" not in cmd

    def test_pytest_with_coverage(self, auto_test_tool: AutoTestTool) -> None:
        """Pytest command with coverage enabled."""
        cmd = auto_test_tool._build_command(
            "pytest", ["tests/test_foo.py"], coverage=True
        )
        assert "--cov" in cmd
        assert "--cov-report=term-missing" in cmd

    def test_pytest_run_all(self, auto_test_tool: AutoTestTool) -> None:
        """Pytest command for full suite (no specific files)."""
        cmd = auto_test_tool._build_command("pytest", [], coverage=False)
        assert "python -m pytest" in cmd
        assert cmd.rstrip().endswith("--no-header")

    def test_unittest_command(self, auto_test_tool: AutoTestTool) -> None:
        """Unittest command with module paths."""
        cmd = auto_test_tool._build_command(
            "unittest", ["tests/test_foo.py"], coverage=False
        )
        assert "python -m unittest" in cmd
        assert "-v" in cmd
        assert "tests.test_foo" in cmd

    def test_unittest_discover(self, auto_test_tool: AutoTestTool) -> None:
        """Unittest discover command when no specific targets."""
        cmd = auto_test_tool._build_command("unittest", [], coverage=False)
        assert "discover" in cmd

    def test_jest_command(self, auto_test_tool: AutoTestTool) -> None:
        """Jest command with specific test files."""
        cmd = auto_test_tool._build_command(
            "jest", ["src/Widget.test.tsx"], coverage=False
        )
        assert "npx jest" in cmd
        assert "--verbose" in cmd
        assert "src/Widget.test.tsx" in cmd

    def test_jest_with_coverage(self, auto_test_tool: AutoTestTool) -> None:
        """Jest command with coverage enabled."""
        cmd = auto_test_tool._build_command(
            "jest", ["test.js"], coverage=True
        )
        assert "--coverage" in cmd

    def test_mocha_command(self, auto_test_tool: AutoTestTool) -> None:
        """Mocha command with test files."""
        cmd = auto_test_tool._build_command(
            "mocha", ["test/foo.spec.js"], coverage=False
        )
        assert "npx mocha" in cmd
        assert "test/foo.spec.js" in cmd


# ------------------------------------------------------------------
# Output parsing tests
# ------------------------------------------------------------------


class TestOutputParsing:
    """Tests for test output parsing."""

    def test_parse_pytest_all_pass(self, auto_test_tool: AutoTestTool) -> None:
        """Parse pytest output with all tests passing."""
        output = (
            "tests/test_foo.py::test_one PASSED\n"
            "tests/test_foo.py::test_two PASSED\n"
            "====== 2 passed in 0.5s ======\n"
        )
        result = auto_test_tool._parse_pytest_output(output, exit_code=0)
        assert result["passed"] == 2
        assert result["failed"] == 0
        assert result["failed_tests"] == []

    def test_parse_pytest_with_failures(self, auto_test_tool: AutoTestTool) -> None:
        """Parse pytest output with failures."""
        output = (
            "tests/test_foo.py::test_one PASSED\n"
            "tests/test_foo.py::test_two FAILED\n"
            "= FAILURES =\n"
            "_____ test_two _____\n"
            "AssertionError: 1 != 2\n"
            "FAILED tests/test_foo.py::test_two\n"
            "====== 1 passed, 1 failed in 0.5s ======\n"
        )
        result = auto_test_tool._parse_pytest_output(output, exit_code=1)
        assert result["passed"] == 1
        assert result["failed"] == 1
        assert "tests/test_foo.py::test_two" in result["failed_tests"]
        assert len(result["error_messages"]) > 0

    def test_parse_pytest_with_skipped(self, auto_test_tool: AutoTestTool) -> None:
        """Parse pytest output with skipped tests."""
        output = "====== 5 passed, 2 skipped in 1.0s ======\n"
        result = auto_test_tool._parse_pytest_output(output, exit_code=0)
        assert result["passed"] == 5
        assert result["skipped"] == 2
        assert result["failed"] == 0

    def test_parse_pytest_no_summary_nonzero_exit(
        self, auto_test_tool: AutoTestTool
    ) -> None:
        """Non-zero exit with no summary still reports failure."""
        result = auto_test_tool._parse_pytest_output(
            "Error: no module named foo", exit_code=1
        )
        assert result["failed"] >= 1

    def test_parse_jest_output(self, auto_test_tool: AutoTestTool) -> None:
        """Parse Jest output with passes and fails."""
        output = (
            "PASS src/Widget.test.tsx\n"
            "Tests: 1 failed, 3 passed, 4 total\n"
        )
        result = auto_test_tool._parse_jest_output(output, exit_code=1)
        assert result["passed"] == 3
        assert result["failed"] == 1

    def test_parse_generic_pass(self, auto_test_tool: AutoTestTool) -> None:
        """Generic parser with zero exit code."""
        result = auto_test_tool._parse_generic_output("all good", exit_code=0)
        assert result["passed"] >= 1
        assert result["failed"] == 0

    def test_parse_generic_fail(self, auto_test_tool: AutoTestTool) -> None:
        """Generic parser with non-zero exit code."""
        result = auto_test_tool._parse_generic_output("boom", exit_code=1)
        assert result["failed"] >= 1
        assert len(result["error_messages"]) > 0

    def test_parse_generic_with_numbers(self, auto_test_tool: AutoTestTool) -> None:
        """Generic parser extracts pass/fail counts from text."""
        output = "Results: 10 passed, 2 failed"
        result = auto_test_tool._parse_generic_output(output, exit_code=1)
        assert result["passed"] == 10
        assert result["failed"] == 2


# ------------------------------------------------------------------
# Full execution tests
# ------------------------------------------------------------------


class TestAutoTestExecution:
    """Tests for the full execute() workflow."""

    def test_no_files_no_run_all(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Error when no files provided and run_all=False."""
        result = auto_test_tool.execute({
            "changed_files": [],
            "run_all": False,
        })
        assert result.success is False
        assert "No changed files" in (result.error or "")

    def test_no_test_files_found(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Returns success with 'no tests found' when no test files match."""
        mock_sandbox.execute.return_value = _make_result(stdout="missing")
        result = auto_test_tool.execute({
            "changed_files": ["src/brand_new.py"],
        })
        assert result.success is True
        assert "No corresponding test files" in result.output
        assert result.metadata is not None
        assert result.metadata["no_tests"] is True

    def test_successful_test_run(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Full workflow: detect framework, find tests, run, parse."""
        pytest_output = (
            "tests/test_foo.py::test_one PASSED\n"
            "====== 1 passed in 0.2s ======\n"
        )

        call_index = 0

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            nonlocal call_index
            call_index += 1
            # Framework detection: conftest.py found
            if "conftest.py" in cmd:
                return _make_result(stdout="found")
            # Test file existence check
            if "test -f" in cmd and "test_foo" in cmd:
                return _make_result(stdout="found")
            if "test -f" in cmd:
                return _make_result(stdout="missing")
            # Test execution
            if "pytest" in cmd:
                return _make_result(stdout=pytest_output)
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        result = auto_test_tool.execute({
            "changed_files": ["src/foo.py"],
        })
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["passed"] == 1
        assert result.metadata["failed"] == 0
        assert result.metadata["framework"] == "pytest"

    def test_failed_test_run(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Workflow with test failures."""
        pytest_output = (
            "FAILED tests/test_bar.py::test_bad\n"
            "= FAILURES =\n"
            "_____ test_bad _____\n"
            "assert False\n"
            "====== 0 passed, 1 failed in 0.1s ======\n"
        )

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            if "conftest.py" in cmd:
                return _make_result(stdout="found")
            if "test -f" in cmd and "test_bar" in cmd:
                return _make_result(stdout="found")
            if "test -f" in cmd:
                return _make_result(stdout="missing")
            if "pytest" in cmd:
                return _make_result(stdout=pytest_output, exit_code=1)
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        result = auto_test_tool.execute({
            "changed_files": ["src/bar.py"],
        })
        assert result.success is False
        assert result.metadata is not None
        assert result.metadata["failed"] == 1
        assert "tests/test_bar.py::test_bad" in result.metadata["failed_tests"]

    def test_run_all_flag(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """run_all=True runs the full suite without file mapping."""
        pytest_output = "====== 10 passed in 2.0s ======\n"

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            if "conftest.py" in cmd:
                return _make_result(stdout="found")
            if "pytest" in cmd:
                return _make_result(stdout=pytest_output)
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        result = auto_test_tool.execute({
            "changed_files": ["anything.py"],
            "run_all": True,
        })
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["passed"] == 10

    def test_coverage_flag(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """coverage=True includes --cov in the command."""
        commands_run: list[str] = []

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            commands_run.append(cmd)
            if "conftest.py" in cmd:
                return _make_result(stdout="found")
            if "test -f" in cmd and "test_foo" in cmd:
                return _make_result(stdout="found")
            if "test -f" in cmd:
                return _make_result(stdout="missing")
            if "pytest" in cmd:
                return _make_result(stdout="====== 1 passed in 0.1s ======")
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        auto_test_tool.execute({
            "changed_files": ["src/foo.py"],
            "coverage": True,
        })
        # Check that --cov was included in the pytest command
        pytest_commands = [c for c in commands_run if "pytest" in c]
        assert any("--cov" in c for c in pytest_commands)

    def test_sandbox_exception_handled(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Sandbox exceptions are caught and returned as errors."""
        call_count = 0

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            nonlocal call_count
            call_count += 1
            if "conftest.py" in cmd:
                return _make_result(stdout="found")
            if "test -f" in cmd and "test_foo" in cmd:
                return _make_result(stdout="found")
            if "test -f" in cmd:
                return _make_result(stdout="missing")
            if "pytest" in cmd:
                raise RuntimeError("sandbox exploded")
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        result = auto_test_tool.execute({
            "changed_files": ["src/foo.py"],
        })
        assert result.success is False
        assert "sandbox exploded" in (result.error or "")

    def test_timeout_clamped(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Timeout is clamped to [1, 300]."""
        mock_sandbox.execute.return_value = _make_result(stdout="missing")
        # Just verify it doesn't crash with extreme values
        auto_test_tool.execute({
            "changed_files": ["src/foo.py"],
            "timeout": 9999,
        })
        auto_test_tool.execute({
            "changed_files": ["src/foo.py"],
            "timeout": -5,
        })

    def test_timed_out_test_run(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Reports timeout when test run exceeds limit."""

        def side_effect(cmd: str, timeout: int = 30) -> CommandResult:
            if "conftest.py" in cmd:
                return _make_result(stdout="found")
            if "test -f" in cmd and "test_foo" in cmd:
                return _make_result(stdout="found")
            if "test -f" in cmd:
                return _make_result(stdout="missing")
            if "pytest" in cmd:
                return _make_result(stdout="", timed_out=True, exit_code=-1)
            return _make_result(stdout="missing")

        mock_sandbox.execute.side_effect = side_effect
        result = auto_test_tool.execute({
            "changed_files": ["src/foo.py"],
        })
        assert result.success is False
        assert "timed out" in (result.error or "").lower()
        assert result.metadata is not None
        assert result.metadata["timed_out"] is True

    def test_missing_required_field(self, auto_test_tool: AutoTestTool) -> None:
        """Missing required 'changed_files' raises ValueError."""
        with pytest.raises(ValueError, match="Missing required"):
            auto_test_tool.execute({})

    def test_metadata_structure(
        self, auto_test_tool: AutoTestTool, mock_sandbox: MagicMock
    ) -> None:
        """Metadata contains all expected fields after a successful run."""
        mock_sandbox.execute.return_value = _make_result(stdout="missing")
        result = auto_test_tool.execute({
            "changed_files": ["src/foo.py"],
        })
        # No test files found case — still has metadata
        assert result.metadata is not None
        assert "framework" in result.metadata

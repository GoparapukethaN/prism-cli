"""Auto-test runner tool — automatically finds and runs relevant tests after code changes.

Unlike a generic terminal tool, this tool is **smart**: given a list of changed
files it automatically discovers corresponding test files, detects the project's
test framework, runs *only* the relevant tests, parses the structured output,
and returns a machine-readable summary the LLM can use to fix failures.

No single-model CLI provides this: it combines framework detection, test-file
mapping, selective execution, and structured output parsing in one atomic tool
call.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

if TYPE_CHECKING:
    from prism.security.sandbox import CommandSandbox

logger = structlog.get_logger(__name__)

# Sentinel files that indicate which test framework is in use.
_FRAMEWORK_INDICATORS: dict[str, list[str]] = {
    "pytest": ["pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py", "tox.ini"],
    "unittest": ["setup.py"],
    "jest": ["jest.config.js", "jest.config.ts", "jest.config.mjs"],
    "mocha": [".mocharc.yml", ".mocharc.json", ".mocharc.js"],
}

# Regex patterns for parsing test output.
_PYTEST_SUMMARY_RE = re.compile(
    r"(?P<passed>\d+) passed"
    r"(?:,\s*(?P<failed>\d+) failed)?"
    r"(?:,\s*(?P<error>\d+) error)?"
    r"(?:,\s*(?P<skipped>\d+) skipped)?",
)

_PYTEST_FAILURE_RE = re.compile(
    r"^FAILED\s+(?P<test_name>\S+)",
    re.MULTILINE,
)

_JEST_SUMMARY_RE = re.compile(
    r"Tests:\s*"
    r"(?:(?P<failed>\d+)\s+failed,?\s*)?"
    r"(?:(?P<passed>\d+)\s+passed,?\s*)?",
)


class AutoTestTool(Tool):
    """Automatically discover and run relevant tests after code changes.

    Given a list of changed source files, this tool:
    1. Maps each source file to its corresponding test file(s).
    2. Detects the project's test framework (pytest, unittest, jest, mocha).
    3. Runs *only* the relevant tests for fast feedback.
    4. Parses the test output into structured pass/fail data.
    5. Returns a machine-readable result the LLM can use to fix failures.

    Supports ``run_all=True`` to execute the entire test suite and
    ``coverage=True`` to include coverage reporting.

    Uses :class:`CommandSandbox` for secure command execution.
    """

    def __init__(self, sandbox: CommandSandbox) -> None:
        """Initialise the auto-test tool.

        Args:
            sandbox: A :class:`CommandSandbox` for executing test commands.
        """
        self._sandbox = sandbox

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "auto_test"

    @property
    def description(self) -> str:
        return (
            "Automatically find and run tests relevant to changed files. "
            "Detects test framework (pytest/unittest/jest/mocha), maps "
            "source files to test files, runs only relevant tests, and "
            "returns structured pass/fail results with error messages."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "changed_files": {
                    "type": "array",
                    "description": (
                        "List of file paths (relative to project root) "
                        "that were modified."
                    ),
                },
                "run_all": {
                    "type": "boolean",
                    "description": "Run the entire test suite instead of just relevant tests.",
                    "default": False,
                },
                "coverage": {
                    "type": "boolean",
                    "description": "Include coverage reporting in the test run.",
                    "default": False,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for the test run. Default 120, max 300.",
                    "default": 120,
                },
            },
            "required": ["changed_files"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.AUTO

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Run relevant tests for the changed files.

        Args:
            arguments: Must contain ``changed_files`` (list of str).
                Optionally ``run_all`` (bool), ``coverage`` (bool),
                ``timeout`` (int).

        Returns:
            A :class:`ToolResult` with structured test output including
            pass count, fail count, failed test names, and error messages.
        """
        validated = self.validate_arguments(arguments)
        changed_files: list[str] = validated["changed_files"]
        run_all: bool = validated.get("run_all", False)
        coverage: bool = validated.get("coverage", False)
        timeout: int = validated.get("timeout", 120)

        # Clamp timeout
        timeout = max(1, min(300, timeout))

        if not changed_files and not run_all:
            return ToolResult(
                success=False,
                output="",
                error="No changed files provided and run_all is False.",
            )

        # Step 1: Detect test framework
        framework = self._detect_framework()
        logger.info("test_framework_detected", framework=framework)

        # Step 2: Find relevant test files
        if run_all:
            test_targets: list[str] = []
        else:
            test_targets = self._find_test_files(changed_files, framework)
            if not test_targets:
                return ToolResult(
                    success=True,
                    output="No corresponding test files found for the changed files.",
                    metadata={
                        "framework": framework,
                        "changed_files": changed_files,
                        "test_files_found": [],
                        "passed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "no_tests": True,
                    },
                )

        # Step 3: Build the test command
        command = self._build_command(framework, test_targets, coverage)
        logger.info("test_command_built", command=command)

        # Step 4: Execute via sandbox
        try:
            result = self._sandbox.execute(command, timeout=timeout)
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Test execution failed: {exc}",
                metadata={"command": command, "framework": framework},
            )

        # Step 5: Parse the output
        full_output = result.stdout
        if result.stderr:
            full_output += f"\n{result.stderr}"

        parsed = self._parse_output(framework, full_output, result.exit_code)

        # Build structured result
        success = parsed["failed"] == 0 and not result.timed_out
        error_msg: str | None = None
        if result.timed_out:
            error_msg = f"Tests timed out after {timeout}s"
        elif parsed["failed"] > 0:
            error_msg = (
                f"{parsed['failed']} test(s) failed. "
                f"See failed_tests and error_messages in metadata."
            )

        metadata: dict[str, Any] = {
            "framework": framework,
            "command": command,
            "passed": parsed["passed"],
            "failed": parsed["failed"],
            "skipped": parsed["skipped"],
            "errors": parsed.get("errors", 0),
            "failed_tests": parsed["failed_tests"],
            "error_messages": parsed["error_messages"],
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "timed_out": result.timed_out,
            "test_files": test_targets,
            "coverage_included": coverage,
        }

        return ToolResult(
            success=success,
            output=full_output,
            error=error_msg,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Framework detection
    # ------------------------------------------------------------------

    def _detect_framework(self) -> str:
        """Detect the test framework by checking for indicator files.

        Runs ``ls`` in the project root and checks for known config files.
        Falls back to ``"pytest"`` as the default for Python projects.

        Returns:
            Framework name: ``"pytest"``, ``"unittest"``, ``"jest"``, or ``"mocha"``.
        """
        # Check for framework indicator files
        for framework, indicators in _FRAMEWORK_INDICATORS.items():
            for indicator in indicators:
                check_result = self._sandbox.execute(
                    f"test -f {indicator} && echo found || echo missing",
                    timeout=5,
                )
                if check_result.stdout.strip() == "found":
                    # For pyproject.toml, check if it contains pytest config
                    if indicator == "pyproject.toml" and framework == "pytest":
                        content_result = self._sandbox.execute(
                            "head -50 pyproject.toml",
                            timeout=5,
                        )
                        if "pytest" in content_result.stdout.lower():
                            return "pytest"
                        # pyproject.toml exists but might not have pytest config
                        continue
                    if indicator == "setup.py" and framework == "unittest":
                        # setup.py alone doesn't confirm unittest; keep looking
                        continue
                    return framework

        # Check for package.json (JS project)
        pkg_result = self._sandbox.execute(
            "test -f package.json && echo found || echo missing",
            timeout=5,
        )
        if pkg_result.stdout.strip() == "found":
            content = self._sandbox.execute("cat package.json", timeout=5)
            if "jest" in content.stdout.lower():
                return "jest"
            if "mocha" in content.stdout.lower():
                return "mocha"

        # Check for conftest.py or tests/ directory with Python files
        tests_result = self._sandbox.execute(
            "test -d tests && echo found || echo missing",
            timeout=5,
        )
        if tests_result.stdout.strip() == "found":
            return "pytest"

        # Default to pytest
        return "pytest"

    # ------------------------------------------------------------------
    # Test file discovery
    # ------------------------------------------------------------------

    def _find_test_files(
        self, changed_files: list[str], framework: str
    ) -> list[str]:
        """Map changed source files to their corresponding test files.

        Mapping strategies:
        - ``src/foo.py`` -> ``tests/test_foo.py``
        - ``src/pkg/bar.py`` -> ``tests/test_pkg/test_bar.py``
        - ``src/components/Widget.tsx`` -> ``src/components/__tests__/Widget.test.tsx``

        Only returns test files that actually exist on disk.

        Args:
            changed_files: List of changed file paths (relative to project root).
            framework: Detected test framework name.

        Returns:
            List of existing test file paths (relative to project root).
        """
        candidates: list[str] = []

        for file_path in changed_files:
            pure = PurePosixPath(file_path)
            stem = pure.stem
            suffix = pure.suffix

            # Skip files that are already test files
            if stem.startswith("test_") or stem.endswith("_test") or ".test." in pure.name:
                candidates.append(file_path)
                continue

            if framework in ("pytest", "unittest"):
                candidates.extend(self._python_test_candidates(pure, stem, suffix))
            elif framework in ("jest", "mocha"):
                candidates.extend(self._js_test_candidates(pure, stem, suffix))

        # Verify which candidates actually exist
        existing: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            check = self._sandbox.execute(
                f"test -f {candidate} && echo found || echo missing",
                timeout=5,
            )
            if check.stdout.strip() == "found":
                existing.append(candidate)

        return existing

    @staticmethod
    def _python_test_candidates(
        pure: PurePosixPath, stem: str, suffix: str
    ) -> list[str]:
        """Generate Python test file candidates for a source file.

        Args:
            pure: The source file path as a PurePosixPath.
            stem: File stem (name without extension).
            suffix: File extension.

        Returns:
            List of candidate test file paths.
        """
        candidates: list[str] = []

        if suffix not in (".py",):
            return candidates

        # Strategy 1: tests/test_<stem>.py (flat tests/)
        candidates.append(f"tests/test_{stem}.py")

        # Strategy 2: tests/<parent_package>/test_<stem>.py
        parts = list(pure.parts)
        if len(parts) > 1:
            # Remove 'src/' prefix if present
            if parts[0] == "src":
                parts = parts[1:]
            if len(parts) > 1:
                # e.g., prism/tools/terminal.py -> tests/test_tools/test_terminal.py
                package_parts = parts[1:-1]  # skip top-level package name
                test_dir = "/".join(f"test_{p}" for p in package_parts)
                if test_dir:
                    candidates.append(f"tests/{test_dir}/test_{stem}.py")
                else:
                    candidates.append(f"tests/test_{stem}.py")

        # Strategy 3: Same directory with test_ prefix
        parent = str(pure.parent)
        candidates.append(f"{parent}/test_{stem}.py")

        return candidates

    @staticmethod
    def _js_test_candidates(
        pure: PurePosixPath, stem: str, suffix: str
    ) -> list[str]:
        """Generate JavaScript/TypeScript test file candidates.

        Args:
            pure: The source file path as a PurePosixPath.
            stem: File stem (name without extension).
            suffix: File extension.

        Returns:
            List of candidate test file paths.
        """
        candidates: list[str] = []

        if suffix not in (".js", ".ts", ".jsx", ".tsx", ".mjs"):
            return candidates

        parent = str(pure.parent)

        # Strategy 1: __tests__/<stem>.test.<ext>
        candidates.append(f"{parent}/__tests__/{stem}.test{suffix}")

        # Strategy 2: <stem>.test.<ext> (co-located)
        candidates.append(f"{parent}/{stem}.test{suffix}")

        # Strategy 3: <stem>.spec.<ext>
        candidates.append(f"{parent}/{stem}.spec{suffix}")

        # Strategy 4: tests/<stem>.test.<ext>
        candidates.append(f"tests/{stem}.test{suffix}")

        return candidates

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _build_command(
        self,
        framework: str,
        test_targets: list[str],
        coverage: bool,
    ) -> str:
        """Build the shell command to run the tests.

        Args:
            framework: Detected test framework name.
            test_targets: List of test file paths to run (empty for run_all).
            coverage: Whether to include coverage reporting.

        Returns:
            Shell command string ready for execution.
        """
        if framework == "pytest":
            parts = ["python", "-m", "pytest", "-v", "--tb=short", "--no-header"]
            if coverage:
                parts.extend(["--cov", "--cov-report=term-missing"])
            parts.extend(test_targets)
            return " ".join(parts)

        if framework == "unittest":
            if test_targets:
                # Convert file paths to module notation
                modules = []
                for target in test_targets:
                    mod = target.replace("/", ".").replace(".py", "")
                    modules.append(mod)
                return f"python -m unittest -v {' '.join(modules)}"
            return "python -m unittest discover -v"

        if framework == "jest":
            parts = ["npx", "jest", "--verbose", "--no-coverage"]
            if coverage:
                parts = ["npx", "jest", "--verbose", "--coverage"]
            parts.extend(test_targets)
            return " ".join(parts)

        if framework == "mocha":
            parts = ["npx", "mocha"]
            if test_targets:
                parts.extend(test_targets)
            return " ".join(parts)

        # Fallback: try pytest
        parts = ["python", "-m", "pytest", "-v", "--tb=short"]
        parts.extend(test_targets)
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(
        self, framework: str, output: str, exit_code: int
    ) -> dict[str, Any]:
        """Parse test output into structured data.

        Args:
            framework: Test framework name.
            output: Combined stdout/stderr output from the test run.
            exit_code: Process exit code.

        Returns:
            Dict with ``passed``, ``failed``, ``skipped``, ``errors``,
            ``failed_tests``, and ``error_messages``.
        """
        if framework in ("pytest", "unittest"):
            return self._parse_pytest_output(output, exit_code)
        if framework == "jest":
            return self._parse_jest_output(output, exit_code)
        # Generic fallback
        return self._parse_generic_output(output, exit_code)

    @staticmethod
    def _parse_pytest_output(output: str, exit_code: int) -> dict[str, Any]:
        """Parse pytest/unittest output.

        Args:
            output: Test runner output.
            exit_code: Process exit code.

        Returns:
            Structured test results dict.
        """
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        failed_tests: list[str] = []
        error_messages: list[str] = []

        # Parse summary line
        summary_match = _PYTEST_SUMMARY_RE.search(output)
        if summary_match:
            passed = int(summary_match.group("passed") or 0)
            failed = int(summary_match.group("failed") or 0)
            skipped = int(summary_match.group("skipped") or 0)
            errors = int(summary_match.group("error") or 0)

        # Find failed test names
        for match in _PYTEST_FAILURE_RE.finditer(output):
            test_name = match.group("test_name")
            failed_tests.append(test_name)

        # Extract error messages from FAILURES section
        failure_section = False
        current_error: list[str] = []
        for line in output.splitlines():
            if "= FAILURES =" in line or "= ERRORS =" in line:
                failure_section = True
                continue
            if failure_section and line.startswith("=") and "passed" in line.lower():
                failure_section = False
                if current_error:
                    error_messages.append("\n".join(current_error))
                    current_error = []
                continue
            if failure_section:
                if line.startswith("_____") and current_error:
                    error_messages.append("\n".join(current_error))
                    current_error = []
                elif line.strip():
                    current_error.append(line)

        if current_error:
            error_messages.append("\n".join(current_error))

        # Fallback: if no summary found but exit code is non-zero
        if not summary_match and exit_code != 0:
            failed = max(1, failed)

        return {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "failed_tests": failed_tests,
            "error_messages": error_messages,
        }

    @staticmethod
    def _parse_jest_output(output: str, exit_code: int) -> dict[str, Any]:
        """Parse Jest output.

        Args:
            output: Jest runner output.
            exit_code: Process exit code.

        Returns:
            Structured test results dict.
        """
        passed = 0
        failed = 0
        skipped = 0
        failed_tests: list[str] = []
        error_messages: list[str] = []

        summary_match = _JEST_SUMMARY_RE.search(output)
        if summary_match:
            passed = int(summary_match.group("passed") or 0)
            failed = int(summary_match.group("failed") or 0)

        # Find failed test names
        fail_pattern = re.compile(r"^\s*[xX]\s+(.+)$", re.MULTILINE)
        for match in fail_pattern.finditer(output):
            failed_tests.append(match.group(1).strip())

        if not summary_match and exit_code != 0:
            failed = max(1, failed)

        return {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": 0,
            "failed_tests": failed_tests,
            "error_messages": error_messages,
        }

    @staticmethod
    def _parse_generic_output(output: str, exit_code: int) -> dict[str, Any]:
        """Generic output parser for unknown frameworks.

        Args:
            output: Test runner output.
            exit_code: Process exit code.

        Returns:
            Structured test results dict with best-effort parsing.
        """
        failed = 1 if exit_code != 0 else 0
        passed = 0 if exit_code != 0 else 1

        # Try to find numeric indicators
        pass_match = re.search(r"(\d+)\s+pass(?:ed|ing)?", output, re.IGNORECASE)
        fail_match = re.search(r"(\d+)\s+fail(?:ed|ing|ure)?", output, re.IGNORECASE)

        if pass_match:
            passed = int(pass_match.group(1))
        if fail_match:
            failed = int(fail_match.group(1))

        return {
            "passed": passed,
            "failed": failed,
            "skipped": 0,
            "errors": 0,
            "failed_tests": [],
            "error_messages": [output] if exit_code != 0 else [],
        }

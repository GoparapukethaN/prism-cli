"""Code quality gate tool — runs lint, security, and type checks on changed files.

No single-model CLI provides an atomic "run all quality checks and return
structured results" primitive.  This tool orchestrates ``ruff``, ``bandit``,
``mypy`` (Python) or ``eslint``, ``tsc`` (JS/TS) across only the changed
files, parses every finding into a structured list, and returns it so the
LLM can auto-fix issues before presenting the final result to the user.

This is a *quality gate* — it runs after the LLM writes code and before the
result is shown, catching problems proactively rather than reactively.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

if TYPE_CHECKING:
    from prism.security.sandbox import CommandSandbox

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class QualityFinding:
    """A single quality issue found by a check tool.

    Attributes:
        check:    Which check produced this finding (``"lint"``, ``"security"``,
                  ``"types"``).
        file:     File path where the issue was found.
        line:     Line number (0 if unknown).
        column:   Column number (0 if unknown).
        severity: ``"error"``, ``"warning"``, or ``"info"``.
        code:     Rule or error code (e.g. ``"E501"``, ``"B101"``).
        message:  Human-readable description.
    """

    check: str
    file: str
    line: int
    column: int
    severity: str
    code: str
    message: str


# Ruff output pattern: file.py:line:col: CODE message
_RUFF_LINE_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<code>[A-Z]+\d+)\s+(?P<message>.+)$"
)

# Bandit output pattern (text format): >> Issue: [CODE:SEVERITY] message
# Location: file.py:line:col
_BANDIT_ISSUE_RE = re.compile(
    r">> Issue:\s+\[(?P<code>[^\]]+)\]\s+(?P<message>.+)"
)
_BANDIT_LOCATION_RE = re.compile(
    r"Location:\s+(?P<file>[^:]+):(?P<line>\d+)(?::(?P<col>\d+))?"
)
_BANDIT_SEVERITY_RE = re.compile(
    r"Severity:\s+(?P<severity>\w+)"
)

# Mypy output pattern: file.py:line: error: message  [code]
_MYPY_LINE_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):\s+"
    r"(?P<severity>error|warning|note):\s+"
    r"(?P<message>.+?)(?:\s+\[(?P<code>[^\]]+)\])?$"
)

# ESLint output pattern (compact): file.py:line:col: message [severity/rule]
_ESLINT_LINE_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<message>.+)\s+(?P<code>\S+)$"
)


class QualityGateTool(Tool):
    """Run code quality checks (lint, security, types) on changed files.

    After the LLM modifies files, this tool can be invoked to catch
    issues *before* the user sees the result.  It supports:

    * **lint** — ``ruff check`` (Python) or ``eslint`` (JS/TS)
    * **security** — ``bandit`` (Python)
    * **types** — ``mypy`` (Python) or ``tsc --noEmit`` (TS)
    * **all** — run every applicable check

    Results are parsed into structured :class:`QualityFinding` objects so
    the LLM can programmatically fix them.

    Uses :class:`CommandSandbox` for secure command execution.
    """

    def __init__(self, sandbox: CommandSandbox) -> None:
        """Initialise the quality gate tool.

        Args:
            sandbox: A :class:`CommandSandbox` for executing check commands.
        """
        self._sandbox = sandbox

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "quality_gate"

    @property
    def description(self) -> str:
        return (
            "Run code quality checks (lint, security, type checks) on "
            "changed files. For Python: ruff, bandit, mypy. For JS/TS: "
            "eslint, tsc. Returns structured findings the LLM can auto-fix."
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
                "checks": {
                    "type": "array",
                    "description": (
                        "List of checks to run: 'lint', 'security', "
                        "'types', or 'all'. Default ['all']."
                    ),
                    "default": ["all"],
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout per check in seconds. Default 60, max 300.",
                    "default": 60,
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
        """Run quality checks on the specified files.

        Args:
            arguments: Must contain ``changed_files`` (list of str).
                Optionally ``checks`` (list of str) and ``timeout`` (int).

        Returns:
            A :class:`ToolResult` with structured findings including
            file, line, severity, code, and message for each issue.
        """
        validated = self.validate_arguments(arguments)
        changed_files: list[str] = validated["changed_files"]
        checks: list[str] = validated.get("checks", ["all"])
        timeout: int = validated.get("timeout", 60)

        # Clamp timeout
        timeout = max(1, min(300, timeout))

        if not changed_files:
            return ToolResult(
                success=False,
                output="",
                error="No changed files provided.",
            )

        # Validate checks list
        valid_checks = {"lint", "security", "types", "all"}
        for check in checks:
            if check not in valid_checks:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        f"Unknown check '{check}'. "
                        f"Valid: {', '.join(sorted(valid_checks))}."
                    ),
                )

        # Expand 'all'
        if "all" in checks:
            checks = ["lint", "security", "types"]

        # Detect language from file extensions
        py_files = [f for f in changed_files if f.endswith(".py")]
        js_ts_files = [
            f for f in changed_files
            if f.endswith((".js", ".ts", ".jsx", ".tsx", ".mjs"))
        ]

        all_findings: list[QualityFinding] = []
        check_results: dict[str, dict[str, Any]] = {}

        # Run checks
        if "lint" in checks:
            if py_files:
                findings, meta = self._run_ruff(py_files, timeout)
                all_findings.extend(findings)
                check_results["ruff"] = meta
            if js_ts_files:
                findings, meta = self._run_eslint(js_ts_files, timeout)
                all_findings.extend(findings)
                check_results["eslint"] = meta

        if "security" in checks and py_files:
            findings, meta = self._run_bandit(py_files, timeout)
            all_findings.extend(findings)
            check_results["bandit"] = meta

        if "types" in checks:
            if py_files:
                findings, meta = self._run_mypy(py_files, timeout)
                all_findings.extend(findings)
                check_results["mypy"] = meta
            if js_ts_files:
                findings, meta = self._run_tsc(js_ts_files, timeout)
                all_findings.extend(findings)
                check_results["tsc"] = meta

        # Build output
        error_count = sum(1 for f in all_findings if f.severity == "error")
        warning_count = sum(1 for f in all_findings if f.severity == "warning")
        info_count = sum(1 for f in all_findings if f.severity == "info")

        lines: list[str] = []
        lines.append("Quality Gate Results")
        lines.append("=" * 50)
        lines.append(
            f"Files checked: {len(changed_files)}"
        )
        lines.append(
            f"Findings: {len(all_findings)} "
            f"({error_count} errors, {warning_count} warnings, {info_count} info)"
        )

        if all_findings:
            lines.append("")
            lines.append("Issues:")
            lines.append("-" * 50)
            for finding in all_findings:
                lines.append(
                    f"  [{finding.severity.upper()}] {finding.file}:{finding.line}:"
                    f"{finding.column} {finding.code}: {finding.message}"
                )
        else:
            lines.append("")
            lines.append("All checks passed! No issues found.")

        output_text = "\n".join(lines)
        all_passed = error_count == 0

        metadata: dict[str, Any] = {
            "total_findings": len(all_findings),
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "all_passed": all_passed,
            "checks_run": list(check_results.keys()),
            "check_details": check_results,
            "findings": [
                {
                    "check": f.check,
                    "file": f.file,
                    "line": f.line,
                    "column": f.column,
                    "severity": f.severity,
                    "code": f.code,
                    "message": f.message,
                }
                for f in all_findings
            ],
            "changed_files": changed_files,
        }

        return ToolResult(
            success=all_passed,
            output=output_text,
            error=(
                f"{error_count} error(s) found. See findings in metadata."
                if not all_passed
                else None
            ),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Ruff (Python lint)
    # ------------------------------------------------------------------

    def _run_ruff(
        self, files: list[str], timeout: int
    ) -> tuple[list[QualityFinding], dict[str, Any]]:
        """Run ``ruff check`` on the given Python files.

        Args:
            files: List of Python file paths.
            timeout: Command timeout in seconds.

        Returns:
            Tuple of (list of findings, metadata dict).
        """
        file_args = " ".join(files)
        command = f"ruff check --output-format=text {file_args}"

        try:
            result = self._sandbox.execute(command, timeout=timeout)
        except Exception as exc:
            logger.warning("ruff_execution_error", error=str(exc))
            return [], {"status": "error", "error": str(exc)}

        findings = self._parse_ruff_output(result.stdout)

        return findings, {
            "status": "completed",
            "exit_code": result.exit_code,
            "finding_count": len(findings),
        }

    @staticmethod
    def _parse_ruff_output(output: str) -> list[QualityFinding]:
        """Parse ruff text output into findings.

        Args:
            output: Raw ruff output text.

        Returns:
            List of :class:`QualityFinding` instances.
        """
        findings: list[QualityFinding] = []
        for line in output.splitlines():
            match = _RUFF_LINE_RE.match(line.strip())
            if match:
                code = match.group("code")
                # E/W prefixes are warnings, F prefix is error
                severity = "error" if code.startswith("F") else "warning"
                findings.append(
                    QualityFinding(
                        check="lint",
                        file=match.group("file"),
                        line=int(match.group("line")),
                        column=int(match.group("col")),
                        severity=severity,
                        code=code,
                        message=match.group("message"),
                    )
                )
        return findings

    # ------------------------------------------------------------------
    # Bandit (Python security)
    # ------------------------------------------------------------------

    def _run_bandit(
        self, files: list[str], timeout: int
    ) -> tuple[list[QualityFinding], dict[str, Any]]:
        """Run ``bandit`` on the given Python files.

        Args:
            files: List of Python file paths.
            timeout: Command timeout in seconds.

        Returns:
            Tuple of (list of findings, metadata dict).
        """
        file_args = " ".join(files)
        command = f"bandit -f txt {file_args}"

        try:
            result = self._sandbox.execute(command, timeout=timeout)
        except Exception as exc:
            logger.warning("bandit_execution_error", error=str(exc))
            return [], {"status": "error", "error": str(exc)}

        findings = self._parse_bandit_output(result.stdout)

        return findings, {
            "status": "completed",
            "exit_code": result.exit_code,
            "finding_count": len(findings),
        }

    @staticmethod
    def _parse_bandit_output(output: str) -> list[QualityFinding]:
        """Parse bandit text output into findings.

        Args:
            output: Raw bandit output text.

        Returns:
            List of :class:`QualityFinding` instances.
        """
        findings: list[QualityFinding] = []
        lines = output.splitlines()
        i = 0
        while i < len(lines):
            issue_match = _BANDIT_ISSUE_RE.match(lines[i].strip())
            if issue_match:
                code = issue_match.group("code")
                message = issue_match.group("message")
                severity = "warning"
                file_path = ""
                line_num = 0
                col_num = 0

                # Look for severity and location on following lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    sev_match = _BANDIT_SEVERITY_RE.match(lines[j].strip())
                    if sev_match:
                        sev_text = sev_match.group("severity").lower()
                        if sev_text in ("high", "critical"):
                            severity = "error"
                        elif sev_text == "medium":
                            severity = "warning"
                        else:
                            severity = "info"

                    loc_match = _BANDIT_LOCATION_RE.match(lines[j].strip())
                    if loc_match:
                        file_path = loc_match.group("file")
                        line_num = int(loc_match.group("line"))
                        col_num = int(loc_match.group("col") or 0)

                if file_path:
                    findings.append(
                        QualityFinding(
                            check="security",
                            file=file_path,
                            line=line_num,
                            column=col_num,
                            severity=severity,
                            code=code,
                            message=message,
                        )
                    )
            i += 1
        return findings

    # ------------------------------------------------------------------
    # Mypy (Python type checking)
    # ------------------------------------------------------------------

    def _run_mypy(
        self, files: list[str], timeout: int
    ) -> tuple[list[QualityFinding], dict[str, Any]]:
        """Run ``mypy`` on the given Python files.

        Args:
            files: List of Python file paths.
            timeout: Command timeout in seconds.

        Returns:
            Tuple of (list of findings, metadata dict).
        """
        file_args = " ".join(files)
        command = f"mypy --no-color-output --no-error-summary {file_args}"

        try:
            result = self._sandbox.execute(command, timeout=timeout)
        except Exception as exc:
            logger.warning("mypy_execution_error", error=str(exc))
            return [], {"status": "error", "error": str(exc)}

        findings = self._parse_mypy_output(result.stdout)

        return findings, {
            "status": "completed",
            "exit_code": result.exit_code,
            "finding_count": len(findings),
        }

    @staticmethod
    def _parse_mypy_output(output: str) -> list[QualityFinding]:
        """Parse mypy output into findings.

        Args:
            output: Raw mypy output text.

        Returns:
            List of :class:`QualityFinding` instances.
        """
        findings: list[QualityFinding] = []
        for line in output.splitlines():
            match = _MYPY_LINE_RE.match(line.strip())
            if match:
                findings.append(
                    QualityFinding(
                        check="types",
                        file=match.group("file"),
                        line=int(match.group("line")),
                        column=0,
                        severity=match.group("severity"),
                        code=match.group("code") or "",
                        message=match.group("message"),
                    )
                )
        return findings

    # ------------------------------------------------------------------
    # ESLint (JS/TS lint)
    # ------------------------------------------------------------------

    def _run_eslint(
        self, files: list[str], timeout: int
    ) -> tuple[list[QualityFinding], dict[str, Any]]:
        """Run ``eslint`` on the given JS/TS files.

        Args:
            files: List of JS/TS file paths.
            timeout: Command timeout in seconds.

        Returns:
            Tuple of (list of findings, metadata dict).
        """
        file_args = " ".join(files)
        command = f"npx eslint --format compact {file_args}"

        try:
            result = self._sandbox.execute(command, timeout=timeout)
        except Exception as exc:
            logger.warning("eslint_execution_error", error=str(exc))
            return [], {"status": "error", "error": str(exc)}

        findings = self._parse_eslint_output(result.stdout)

        return findings, {
            "status": "completed",
            "exit_code": result.exit_code,
            "finding_count": len(findings),
        }

    @staticmethod
    def _parse_eslint_output(output: str) -> list[QualityFinding]:
        """Parse eslint compact output into findings.

        Args:
            output: Raw eslint output text.

        Returns:
            List of :class:`QualityFinding` instances.
        """
        findings: list[QualityFinding] = []
        for line in output.splitlines():
            match = _ESLINT_LINE_RE.match(line.strip())
            if match:
                findings.append(
                    QualityFinding(
                        check="lint",
                        file=match.group("file"),
                        line=int(match.group("line")),
                        column=int(match.group("col")),
                        severity="warning",
                        code=match.group("code"),
                        message=match.group("message"),
                    )
                )
        return findings

    # ------------------------------------------------------------------
    # tsc (TypeScript type checking)
    # ------------------------------------------------------------------

    def _run_tsc(
        self, files: list[str], timeout: int
    ) -> tuple[list[QualityFinding], dict[str, Any]]:
        """Run ``tsc --noEmit`` for TypeScript type checking.

        Args:
            files: List of TS/TSX file paths.
            timeout: Command timeout in seconds.

        Returns:
            Tuple of (list of findings, metadata dict).
        """
        # tsc checks the whole project, not individual files
        command = "npx tsc --noEmit --pretty false"

        try:
            result = self._sandbox.execute(command, timeout=timeout)
        except Exception as exc:
            logger.warning("tsc_execution_error", error=str(exc))
            return [], {"status": "error", "error": str(exc)}

        # Filter findings to only the changed files
        all_findings = self._parse_tsc_output(result.stdout)
        file_set = set(files)
        filtered = [f for f in all_findings if f.file in file_set]

        return filtered, {
            "status": "completed",
            "exit_code": result.exit_code,
            "finding_count": len(filtered),
            "total_project_findings": len(all_findings),
        }

    @staticmethod
    def _parse_tsc_output(output: str) -> list[QualityFinding]:
        """Parse tsc output into findings.

        Args:
            output: Raw tsc output text.

        Returns:
            List of :class:`QualityFinding` instances.
        """
        findings: list[QualityFinding] = []
        # tsc format: file.ts(line,col): error TS2345: message
        tsc_re = re.compile(
            r"^(?P<file>[^(]+)\((?P<line>\d+),(?P<col>\d+)\):\s+"
            r"(?P<severity>error|warning)\s+(?P<code>TS\d+):\s+"
            r"(?P<message>.+)$"
        )
        for line in output.splitlines():
            match = tsc_re.match(line.strip())
            if match:
                findings.append(
                    QualityFinding(
                        check="types",
                        file=match.group("file"),
                        line=int(match.group("line")),
                        column=int(match.group("col")),
                        severity=match.group("severity"),
                        code=match.group("code"),
                        message=match.group("message"),
                    )
                )
        return findings

"""Tests for the code quality gate tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from prism.security.sandbox import CommandResult
from prism.tools.base import PermissionLevel
from prism.tools.quality_gate import QualityFinding, QualityGateTool

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock CommandSandbox."""
    return MagicMock(spec=["execute"])


@pytest.fixture
def quality_gate(mock_sandbox: MagicMock) -> QualityGateTool:
    """Create a QualityGateTool with a mock sandbox."""
    return QualityGateTool(mock_sandbox)


def _make_result(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    duration_ms: float = 50.0,
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


class TestQualityGateProperties:
    """Tests for tool metadata properties."""

    def test_name(self, quality_gate: QualityGateTool) -> None:
        """Tool name is 'quality_gate'."""
        assert quality_gate.name == "quality_gate"

    def test_description(self, quality_gate: QualityGateTool) -> None:
        """Description mentions lint, security, and type checks."""
        desc = quality_gate.description
        assert "lint" in desc.lower()
        assert "security" in desc.lower()
        assert "type" in desc.lower()

    def test_permission_level(self, quality_gate: QualityGateTool) -> None:
        """Permission level is AUTO."""
        assert quality_gate.permission_level == PermissionLevel.AUTO

    def test_parameters_schema(self, quality_gate: QualityGateTool) -> None:
        """Schema has changed_files (required), checks, timeout."""
        schema = quality_gate.parameters_schema
        assert schema["type"] == "object"
        assert "changed_files" in schema["properties"]
        assert "checks" in schema["properties"]
        assert "timeout" in schema["properties"]
        assert schema["required"] == ["changed_files"]


# ------------------------------------------------------------------
# QualityFinding dataclass tests
# ------------------------------------------------------------------


class TestQualityFinding:
    """Tests for the QualityFinding dataclass."""

    def test_creation(self) -> None:
        """QualityFinding can be created with all fields."""
        finding = QualityFinding(
            check="lint",
            file="src/main.py",
            line=42,
            column=10,
            severity="error",
            code="F821",
            message="Undefined name 'foo'",
        )
        assert finding.check == "lint"
        assert finding.file == "src/main.py"
        assert finding.line == 42
        assert finding.severity == "error"

    def test_frozen(self) -> None:
        """QualityFinding is immutable."""
        finding = QualityFinding(
            check="lint", file="a.py", line=1,
            column=0, severity="warning", code="W", message="msg",
        )
        with pytest.raises(AttributeError):
            finding.line = 99  # type: ignore[misc]


# ------------------------------------------------------------------
# Ruff parsing tests
# ------------------------------------------------------------------


class TestRuffParsing:
    """Tests for ruff output parsing."""

    def test_parse_ruff_errors(self) -> None:
        """Parses ruff error output correctly."""
        output = (
            "src/main.py:10:5: F821 Undefined name 'foo'\n"
            "src/main.py:20:1: E501 Line too long (120 > 100 characters)\n"
        )
        findings = QualityGateTool._parse_ruff_output(output)
        assert len(findings) == 2
        assert findings[0].file == "src/main.py"
        assert findings[0].line == 10
        assert findings[0].column == 5
        assert findings[0].code == "F821"
        assert findings[0].severity == "error"  # F prefix -> error
        assert findings[1].severity == "warning"  # E prefix -> warning

    def test_parse_ruff_empty(self) -> None:
        """Empty ruff output returns no findings."""
        findings = QualityGateTool._parse_ruff_output("")
        assert findings == []

    def test_parse_ruff_non_matching_lines(self) -> None:
        """Non-matching lines are skipped."""
        output = (
            "All checks passed!\n"
            "No issues found.\n"
        )
        findings = QualityGateTool._parse_ruff_output(output)
        assert findings == []


# ------------------------------------------------------------------
# Bandit parsing tests
# ------------------------------------------------------------------


class TestBanditParsing:
    """Tests for bandit output parsing."""

    def test_parse_bandit_issue(self) -> None:
        """Parses bandit issues with severity and location."""
        output = (
            ">> Issue: [B101:assert_used] Use of assert detected.\n"
            "   Severity: Low\n"
            "   Location: src/main.py:42:0\n"
        )
        findings = QualityGateTool._parse_bandit_output(output)
        assert len(findings) == 1
        assert findings[0].check == "security"
        assert findings[0].code == "B101:assert_used"
        assert findings[0].file == "src/main.py"
        assert findings[0].line == 42
        assert findings[0].severity == "info"  # Low -> info

    def test_parse_bandit_high_severity(self) -> None:
        """High severity maps to error."""
        output = (
            ">> Issue: [B608:hardcoded_sql] SQL injection possible.\n"
            "   Severity: High\n"
            "   Location: src/db.py:10:5\n"
        )
        findings = QualityGateTool._parse_bandit_output(output)
        assert len(findings) == 1
        assert findings[0].severity == "error"

    def test_parse_bandit_medium_severity(self) -> None:
        """Medium severity maps to warning."""
        output = (
            ">> Issue: [B110:try_except_pass] Try/except/pass detected.\n"
            "   Severity: Medium\n"
            "   Location: src/handler.py:5:0\n"
        )
        findings = QualityGateTool._parse_bandit_output(output)
        assert len(findings) == 1
        assert findings[0].severity == "warning"

    def test_parse_bandit_empty(self) -> None:
        """Empty bandit output returns no findings."""
        findings = QualityGateTool._parse_bandit_output("")
        assert findings == []


# ------------------------------------------------------------------
# Mypy parsing tests
# ------------------------------------------------------------------


class TestMypyParsing:
    """Tests for mypy output parsing."""

    def test_parse_mypy_error(self) -> None:
        """Parses mypy error with code."""
        output = (
            'src/main.py:42: error: Incompatible types in assignment  [assignment]\n'
        )
        findings = QualityGateTool._parse_mypy_output(output)
        assert len(findings) == 1
        assert findings[0].check == "types"
        assert findings[0].file == "src/main.py"
        assert findings[0].line == 42
        assert findings[0].severity == "error"
        assert findings[0].code == "assignment"

    def test_parse_mypy_warning(self) -> None:
        """Parses mypy warning."""
        output = 'src/utils.py:10: warning: Unused variable  [unused]\n'
        findings = QualityGateTool._parse_mypy_output(output)
        assert len(findings) == 1
        assert findings[0].severity == "warning"

    def test_parse_mypy_note(self) -> None:
        """Parses mypy note."""
        output = 'src/utils.py:10: note: See documentation\n'
        findings = QualityGateTool._parse_mypy_output(output)
        assert len(findings) == 1
        assert findings[0].severity == "note"

    def test_parse_mypy_no_code(self) -> None:
        """Parses mypy output without error code."""
        output = 'src/main.py:5: error: Name "foo" is not defined\n'
        findings = QualityGateTool._parse_mypy_output(output)
        assert len(findings) == 1
        assert findings[0].code == ""

    def test_parse_mypy_empty(self) -> None:
        """Empty mypy output returns no findings."""
        findings = QualityGateTool._parse_mypy_output("")
        assert findings == []


# ------------------------------------------------------------------
# ESLint parsing tests
# ------------------------------------------------------------------


class TestESLintParsing:
    """Tests for eslint compact output parsing."""

    def test_parse_eslint_line(self) -> None:
        """Parses eslint compact format."""
        output = "src/app.js:10:5: Unexpected var, use let or const no-var\n"
        findings = QualityGateTool._parse_eslint_output(output)
        assert len(findings) == 1
        assert findings[0].check == "lint"
        assert findings[0].file == "src/app.js"
        assert findings[0].line == 10
        assert findings[0].column == 5

    def test_parse_eslint_empty(self) -> None:
        """Empty eslint output returns no findings."""
        findings = QualityGateTool._parse_eslint_output("")
        assert findings == []


# ------------------------------------------------------------------
# TSC parsing tests
# ------------------------------------------------------------------


class TestTscParsing:
    """Tests for tsc output parsing."""

    def test_parse_tsc_error(self) -> None:
        """Parses tsc error format."""
        output = "src/app.ts(10,5): error TS2345: Argument of type 'string' is not assignable.\n"
        findings = QualityGateTool._parse_tsc_output(output)
        assert len(findings) == 1
        assert findings[0].check == "types"
        assert findings[0].file == "src/app.ts"
        assert findings[0].line == 10
        assert findings[0].column == 5
        assert findings[0].code == "TS2345"

    def test_parse_tsc_empty(self) -> None:
        """Empty tsc output returns no findings."""
        findings = QualityGateTool._parse_tsc_output("")
        assert findings == []


# ------------------------------------------------------------------
# Full execution tests
# ------------------------------------------------------------------


class TestQualityGateExecution:
    """Tests for the full execute() workflow."""

    def test_no_files_error(self, quality_gate: QualityGateTool) -> None:
        """Error when no changed files provided."""
        result = quality_gate.execute({
            "changed_files": [],
            "checks": ["lint"],
        })
        assert result.success is False
        assert "No changed files" in (result.error or "")

    def test_unknown_check_error(self, quality_gate: QualityGateTool) -> None:
        """Error for unknown check type."""
        result = quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["magic"],
        })
        assert result.success is False
        assert "Unknown check" in (result.error or "")

    def test_all_checks_pass(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """All checks passing returns success."""
        mock_sandbox.execute.return_value = _make_result(stdout="", exit_code=0)
        result = quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["all"],
        })
        assert result.success is True
        assert "All checks passed" in result.output
        assert result.metadata is not None
        assert result.metadata["all_passed"] is True
        assert result.metadata["total_findings"] == 0

    def test_lint_only(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """Running only lint check works."""
        ruff_output = "src/main.py:10:5: F821 Undefined name 'foo'\n"
        mock_sandbox.execute.return_value = _make_result(
            stdout=ruff_output, exit_code=1
        )
        result = quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["lint"],
        })
        assert result.success is False  # has errors
        assert result.metadata is not None
        assert result.metadata["error_count"] == 1
        findings = result.metadata["findings"]
        assert len(findings) == 1
        assert findings[0]["code"] == "F821"

    def test_security_only(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """Running only security check works."""
        bandit_output = (
            ">> Issue: [B101:assert_used] Use of assert detected.\n"
            "   Severity: Low\n"
            "   Location: src/main.py:42:0\n"
        )
        mock_sandbox.execute.return_value = _make_result(
            stdout=bandit_output, exit_code=1
        )
        result = quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["security"],
        })
        assert result.success is True  # info severity -> not an error
        assert result.metadata is not None
        assert len(result.metadata["findings"]) == 1
        assert result.metadata["findings"][0]["check"] == "security"

    def test_types_only(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """Running only types check works."""
        mypy_output = (
            'src/main.py:5: error: Name "foo" is not defined  [name-defined]\n'
        )
        mock_sandbox.execute.return_value = _make_result(
            stdout=mypy_output, exit_code=1
        )
        result = quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["types"],
        })
        assert result.success is False
        assert result.metadata is not None
        assert result.metadata["error_count"] == 1

    def test_multiple_file_types(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """Handles mixed Python and JS files."""
        mock_sandbox.execute.return_value = _make_result(stdout="", exit_code=0)
        result = quality_gate.execute({
            "changed_files": ["src/main.py", "src/app.ts"],
            "checks": ["lint"],
        })
        assert result.success is True
        assert result.metadata is not None
        # Should have run both ruff (for .py) and eslint (for .ts)
        checks_run = result.metadata["checks_run"]
        assert "ruff" in checks_run
        assert "eslint" in checks_run

    def test_js_only_files(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """JS-only files skip Python checks."""
        mock_sandbox.execute.return_value = _make_result(stdout="", exit_code=0)
        result = quality_gate.execute({
            "changed_files": ["src/app.js"],
            "checks": ["all"],
        })
        assert result.success is True
        assert result.metadata is not None
        # No ruff, bandit, or mypy for JS files
        checks_run = result.metadata["checks_run"]
        assert "ruff" not in checks_run
        assert "bandit" not in checks_run
        assert "mypy" not in checks_run

    def test_all_expands_to_all_checks(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """'all' in checks expands to lint + security + types."""
        mock_sandbox.execute.return_value = _make_result(stdout="", exit_code=0)
        result = quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["all"],
        })
        assert result.success is True
        assert result.metadata is not None
        checks_run = result.metadata["checks_run"]
        assert "ruff" in checks_run
        assert "bandit" in checks_run
        assert "mypy" in checks_run

    def test_sandbox_exception_handled(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """Sandbox exceptions during checks are caught gracefully."""
        mock_sandbox.execute.side_effect = RuntimeError("command not found")
        result = quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["lint"],
        })
        # Should still succeed (no findings) since the check errored
        assert result.success is True
        assert result.metadata is not None
        check_details = result.metadata["check_details"]
        assert check_details["ruff"]["status"] == "error"

    def test_metadata_structure(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """Metadata contains all expected fields."""
        mock_sandbox.execute.return_value = _make_result(stdout="", exit_code=0)
        result = quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["lint"],
        })
        assert result.metadata is not None
        assert "total_findings" in result.metadata
        assert "error_count" in result.metadata
        assert "warning_count" in result.metadata
        assert "info_count" in result.metadata
        assert "all_passed" in result.metadata
        assert "checks_run" in result.metadata
        assert "findings" in result.metadata
        assert "changed_files" in result.metadata

    def test_missing_required_field(self, quality_gate: QualityGateTool) -> None:
        """Missing required 'changed_files' raises ValueError."""
        with pytest.raises(ValueError, match="Missing required"):
            quality_gate.execute({})

    def test_timeout_clamped(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """Timeout is clamped to valid range."""
        mock_sandbox.execute.return_value = _make_result(stdout="", exit_code=0)
        # Should not crash with extreme values
        quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["lint"],
            "timeout": 9999,
        })
        quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["lint"],
            "timeout": -5,
        })

    def test_tsc_filters_to_changed_files(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """TSC findings are filtered to only changed files."""
        tsc_output = (
            "src/app.ts(10,5): error TS2345: Bad type.\n"
            "src/other.ts(20,1): error TS2339: Not in changed list.\n"
        )
        mock_sandbox.execute.return_value = _make_result(
            stdout=tsc_output, exit_code=1
        )
        result = quality_gate.execute({
            "changed_files": ["src/app.ts"],
            "checks": ["types"],
        })
        assert result.metadata is not None
        findings = result.metadata["findings"]
        # Only src/app.ts findings should be included
        assert all(f["file"] == "src/app.ts" for f in findings)

    def test_output_format(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """Output includes section headers and finding details."""
        ruff_output = "src/main.py:10:5: F821 Undefined name 'foo'\n"
        mock_sandbox.execute.return_value = _make_result(
            stdout=ruff_output, exit_code=1
        )
        result = quality_gate.execute({
            "changed_files": ["src/main.py"],
            "checks": ["lint"],
        })
        assert "Quality Gate Results" in result.output
        assert "Issues:" in result.output
        assert "[ERROR]" in result.output
        assert "F821" in result.output

    def test_default_checks_is_all(
        self, quality_gate: QualityGateTool, mock_sandbox: MagicMock
    ) -> None:
        """Default checks parameter is ['all']."""
        mock_sandbox.execute.return_value = _make_result(stdout="", exit_code=0)
        result = quality_gate.execute({
            "changed_files": ["src/main.py"],
        })
        assert result.success is True
        assert result.metadata is not None
        # All Python checks should have been run
        checks_run = result.metadata["checks_run"]
        assert "ruff" in checks_run
        assert "bandit" in checks_run
        assert "mypy" in checks_run

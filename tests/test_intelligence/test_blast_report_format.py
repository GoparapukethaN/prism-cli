"""Tests for enhanced blast-radius report formatting.

Covers ``BlastRadiusAnalyzer.generate_report_text`` and the display helpers
that render detailed impact reports to the console.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.intelligence.blast_radius import (
    AffectedFile,
    BlastRadiusAnalyzer,
    ImpactReport,
    RiskLevel,
)

if TYPE_CHECKING:
    from pathlib import Path

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture()
def reports_dir(tmp_path: Path) -> Path:
    """Provide a temporary reports directory."""
    d = tmp_path / "impact_reports"
    d.mkdir()
    return d


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """Create a minimal project layout for testing."""
    src = tmp_path / "src" / "prism"
    src.mkdir(parents=True)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    (src / "__init__.py").write_text("")
    (src / "auth.py").write_text(
        "def validate_key(key: str) -> bool:\n"
        "    return bool(key)\n\n"
        "def authenticate(user: str) -> bool:\n"
        "    return True\n"
    )
    (src / "router.py").write_text(
        "from prism.auth import validate_key\n"
        "def route(task: str) -> str:\n"
        "    return task\n"
    )
    (src / "utils.py").write_text(
        "def helper() -> int:\n"
        "    return 42\n"
    )
    (tests_dir / "test_auth.py").write_text(
        "def test_validate_key():\n"
        "    pass\n"
    )

    return tmp_path


@pytest.fixture()
def analyzer(project: Path, reports_dir: Path) -> BlastRadiusAnalyzer:
    """Create a BlastRadiusAnalyzer with temporary paths."""
    return BlastRadiusAnalyzer(project, reports_dir=reports_dir)


def _make_report(
    affected: list[AffectedFile] | None = None,
    risk_score: int = 55,
    missing_tests: list[str] | None = None,
    critical_paths: list[str] | None = None,
    complexity: str = "moderate",
    recommended_test_order: list[str] | None = None,
    execution_order: list[str] | None = None,
) -> ImpactReport:
    """Build an ImpactReport for testing."""
    return ImpactReport(
        description="refactor auth validation",
        risk_score=risk_score,
        affected_files=affected or [],
        missing_tests=missing_tests or [],
        recommended_test_order=(
            ["tests/test_auth.py"]
            if recommended_test_order is None
            else recommended_test_order
        ),
        execution_order=(
            ["src/prism/auth.py"]
            if execution_order is None
            else execution_order
        ),
        estimated_complexity=complexity,
        critical_paths=critical_paths or [],
        created_at="2025-06-15T12:00:00+00:00",
    )


# ======================================================================
# generate_report_text — risk score
# ======================================================================


class TestGenerateReportTextRiskScore:
    """generate_report_text includes the risk score."""

    def test_includes_risk_score(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """The report text contains the numeric risk score."""
        report = _make_report(risk_score=72)
        text = analyzer.generate_report_text(report)
        assert "72/100" in text

    def test_high_risk_label(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Risk score >= 70 shows HIGH label."""
        report = _make_report(risk_score=85)
        text = analyzer.generate_report_text(report)
        assert "[HIGH]" in text

    def test_medium_risk_label(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Risk score 40-69 shows MEDIUM label."""
        report = _make_report(risk_score=55)
        text = analyzer.generate_report_text(report)
        assert "[MEDIUM]" in text

    def test_low_risk_label(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Risk score < 40 shows LOW label."""
        report = _make_report(risk_score=20)
        text = analyzer.generate_report_text(report)
        assert "[LOW]" in text


# ======================================================================
# generate_report_text — groups files by risk level
# ======================================================================


class TestGenerateReportTextGrouping:
    """generate_report_text groups files by risk level."""

    def test_groups_high_risk_files(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """HIGH-risk files appear under the critical areas heading."""
        affected = [
            AffectedFile(
                "src/prism/auth.py", RiskLevel.HIGH,
                "Direct target", ["validate_key"], False, [], 0,
            ),
        ]
        report = _make_report(affected=affected)
        text = analyzer.generate_report_text(report)
        assert "CRITICAL AREAS (HIGH risk)" in text
        assert "src/prism/auth.py" in text

    def test_groups_medium_risk_files(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """MEDIUM-risk files appear under the medium risk heading."""
        affected = [
            AffectedFile(
                "src/prism/router.py", RiskLevel.MEDIUM,
                "Imports auth", [], True, ["tests/test_router.py"], 1,
            ),
        ]
        report = _make_report(affected=affected)
        text = analyzer.generate_report_text(report)
        assert "MEDIUM RISK AREAS" in text
        assert "src/prism/router.py" in text

    def test_groups_low_risk_files(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """LOW-risk files appear under the low risk heading."""
        affected = [
            AffectedFile(
                "src/prism/utils.py", RiskLevel.LOW,
                "Indirect dep", [], True, [], 2,
            ),
        ]
        report = _make_report(affected=affected)
        text = analyzer.generate_report_text(report)
        assert "LOW RISK AREAS" in text
        assert "src/prism/utils.py" in text

    def test_all_three_groups_present(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """When files span all risk levels, all three sections appear."""
        affected = [
            AffectedFile(
                "auth.py", RiskLevel.HIGH, "Target", [], False, [], 0,
            ),
            AffectedFile(
                "router.py", RiskLevel.MEDIUM, "Imports", [], True,
                ["test_router.py"], 1,
            ),
            AffectedFile(
                "utils.py", RiskLevel.LOW, "Indirect", [], True, [], 2,
            ),
        ]
        report = _make_report(affected=affected)
        text = analyzer.generate_report_text(report)
        assert "CRITICAL AREAS (HIGH risk)" in text
        assert "MEDIUM RISK AREAS" in text
        assert "LOW RISK AREAS" in text

    def test_omits_empty_groups(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Groups with no files are omitted from the output."""
        affected = [
            AffectedFile(
                "router.py", RiskLevel.MEDIUM, "Imports", [], True, [], 1,
            ),
        ]
        report = _make_report(affected=affected)
        text = analyzer.generate_report_text(report)
        assert "CRITICAL AREAS" not in text
        assert "LOW RISK AREAS" not in text
        assert "MEDIUM RISK AREAS" in text


# ======================================================================
# generate_report_text — test recommendations
# ======================================================================


class TestGenerateReportTextTestRecommendations:
    """generate_report_text includes test recommendations."""

    def test_includes_pytest_command(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Test recommendations section includes a pytest command."""
        report = _make_report(
            recommended_test_order=[
                "tests/test_auth.py",
                "tests/test_router.py",
            ],
        )
        text = analyzer.generate_report_text(report)
        assert "TEST RECOMMENDATIONS" in text
        assert "pytest tests/test_auth.py tests/test_router.py" in text

    def test_omits_section_when_no_tests(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """The section is omitted when there are no recommended tests."""
        report = _make_report(recommended_test_order=[])
        text = analyzer.generate_report_text(report)
        assert "TEST RECOMMENDATIONS" not in text


# ======================================================================
# generate_report_text — missing tests
# ======================================================================


class TestGenerateReportTextMissingTests:
    """generate_report_text includes missing tests with priority labels."""

    def test_includes_missing_tests_section(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Missing tests appear in the output."""
        report = _make_report(
            missing_tests=["src/prism/router.py", "src/prism/utils.py"],
        )
        text = analyzer.generate_report_text(report)
        assert "MISSING TESTS" in text
        assert "src/prism/router.py" in text
        assert "src/prism/utils.py" in text

    def test_high_priority_label_on_critical_untested(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Untested HIGH-risk files get a HIGH PRIORITY label."""
        affected = [
            AffectedFile(
                "src/prism/auth.py", RiskLevel.HIGH,
                "Direct target", ["validate_key"], False, [], 0,
            ),
            AffectedFile(
                "src/prism/utils.py", RiskLevel.LOW,
                "Indirect dep", [], False, [], 2,
            ),
        ]
        report = _make_report(
            affected=affected,
            missing_tests=["src/prism/auth.py", "src/prism/utils.py"],
        )
        text = analyzer.generate_report_text(report)
        # auth.py is HIGH risk + untested → HIGH PRIORITY
        assert "HIGH PRIORITY" in text
        # The HIGH PRIORITY should appear on the auth.py line
        lines = text.splitlines()
        auth_lines = [ln for ln in lines if "src/prism/auth.py" in ln]
        assert any("HIGH PRIORITY" in ln for ln in auth_lines)

    def test_normal_priority_on_non_critical(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Non-critical untested files get a normal priority label."""
        affected = [
            AffectedFile(
                "src/prism/utils.py", RiskLevel.LOW,
                "Indirect dep", [], False, [], 2,
            ),
        ]
        report = _make_report(
            affected=affected,
            missing_tests=["src/prism/utils.py"],
        )
        text = analyzer.generate_report_text(report)
        utils_lines = [ln for ln in text.splitlines() if "src/prism/utils.py" in ln]
        assert any("normal" in ln for ln in utils_lines)

    def test_omits_section_when_no_missing(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """The section is omitted when there are no missing tests."""
        report = _make_report(missing_tests=[])
        text = analyzer.generate_report_text(report)
        assert "MISSING TESTS" not in text


# ======================================================================
# generate_report_text — complexity-to-effort mapping
# ======================================================================


class TestGenerateReportTextComplexityEffort:
    """generate_report_text maps complexity to effort estimate."""

    @pytest.mark.parametrize(
        ("complexity", "expected_effort"),
        [
            ("trivial", "<1 hour"),
            ("simple", "1-2 hours"),
            ("moderate", "2-4 hours"),
            ("complex", "4-8 hours"),
        ],
    )
    def test_maps_complexity_to_effort(
        self,
        analyzer: BlastRadiusAnalyzer,
        complexity: str,
        expected_effort: str,
    ) -> None:
        """Each complexity level maps to its correct effort estimate."""
        report = _make_report(complexity=complexity)
        text = analyzer.generate_report_text(report)
        assert "EFFORT ESTIMATE" in text
        assert f"Estimated effort: {expected_effort}" in text

    @pytest.mark.parametrize(
        ("complexity", "expected_approach"),
        [
            ("trivial", "direct"),
            ("simple", "direct"),
            ("moderate", "incremental (test-first on critical areas)"),
            ("complex", "incremental (test-first on critical areas)"),
        ],
    )
    def test_maps_complexity_to_approach(
        self,
        analyzer: BlastRadiusAnalyzer,
        complexity: str,
        expected_approach: str,
    ) -> None:
        """Each complexity level maps to its correct recommended approach."""
        report = _make_report(complexity=complexity)
        text = analyzer.generate_report_text(report)
        assert f"Recommended approach: {expected_approach}" in text


# ======================================================================
# generate_report_text — execution order
# ======================================================================


class TestGenerateReportTextExecutionOrder:
    """generate_report_text includes recommended execution order."""

    def test_includes_execution_order(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Execution order section lists files in numbered order."""
        report = _make_report(
            execution_order=[
                "src/prism/utils.py",
                "src/prism/router.py",
                "src/prism/auth.py",
            ],
        )
        text = analyzer.generate_report_text(report)
        assert "RECOMMENDED EXECUTION ORDER" in text
        assert "1. src/prism/utils.py" in text
        assert "2. src/prism/router.py" in text
        assert "3. src/prism/auth.py" in text

    def test_omits_section_when_empty(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """The section is omitted when execution order is empty."""
        report = _make_report(execution_order=[])
        text = analyzer.generate_report_text(report)
        assert "RECOMMENDED EXECUTION ORDER" not in text


# ======================================================================
# generate_report_text — header and structure
# ======================================================================


class TestGenerateReportTextStructure:
    """generate_report_text has correct overall structure."""

    def test_includes_description(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """The report header includes the change description."""
        report = _make_report()
        text = analyzer.generate_report_text(report)
        assert "refactor auth validation" in text

    def test_includes_blast_radius_report_title(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """The report has a clear title banner."""
        report = _make_report()
        text = analyzer.generate_report_text(report)
        assert "BLAST RADIUS REPORT" in text

    def test_includes_file_count(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """The header includes the total file count."""
        affected = [
            AffectedFile(
                "a.py", RiskLevel.LOW, "test", [], True, [], 0,
            ),
            AffectedFile(
                "b.py", RiskLevel.LOW, "test", [], True, [], 1,
            ),
        ]
        report = _make_report(affected=affected)
        text = analyzer.generate_report_text(report)
        assert "Files Affected: 2" in text

    def test_high_risk_shows_functions(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """HIGH-risk files show their affected functions in the text."""
        affected = [
            AffectedFile(
                "src/prism/auth.py", RiskLevel.HIGH,
                "Direct target", ["validate_key", "authenticate"],
                True, ["tests/test_auth.py"], 0,
            ),
        ]
        report = _make_report(affected=affected)
        text = analyzer.generate_report_text(report)
        assert "validate_key" in text
        assert "authenticate" in text

    def test_high_risk_shows_test_files(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """HIGH-risk files with tests show the test file paths."""
        affected = [
            AffectedFile(
                "src/prism/auth.py", RiskLevel.HIGH,
                "Direct target", ["validate_key"],
                True, ["tests/test_auth.py"], 0,
            ),
        ]
        report = _make_report(affected=affected)
        text = analyzer.generate_report_text(report)
        assert "tests/test_auth.py" in text

    def test_high_risk_shows_caller_count(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """HIGH-risk files show the caller count."""
        affected = [
            AffectedFile(
                "src/prism/auth.py", RiskLevel.HIGH,
                "Direct target", ["validate_key", "authenticate"],
                False, [], 0,
            ),
        ]
        report = _make_report(affected=affected)
        text = analyzer.generate_report_text(report)
        assert "callers=2" in text


# ======================================================================
# _risk_label static method
# ======================================================================


class TestRiskLabel:
    """Tests for BlastRadiusAnalyzer._risk_label."""

    def test_high_threshold(self) -> None:
        assert BlastRadiusAnalyzer._risk_label(70) == "HIGH"
        assert BlastRadiusAnalyzer._risk_label(100) == "HIGH"

    def test_medium_threshold(self) -> None:
        assert BlastRadiusAnalyzer._risk_label(40) == "MEDIUM"
        assert BlastRadiusAnalyzer._risk_label(69) == "MEDIUM"

    def test_low_threshold(self) -> None:
        assert BlastRadiusAnalyzer._risk_label(0) == "LOW"
        assert BlastRadiusAnalyzer._risk_label(39) == "LOW"


# ======================================================================
# last_report_path property
# ======================================================================


class TestLastReportPath:
    """Tests for BlastRadiusAnalyzer.last_report_path."""

    def test_none_before_analyze(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Before any analysis, last_report_path is None."""
        assert analyzer.last_report_path is None

    def test_set_after_analyze(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """After analysis, last_report_path points to the saved file."""
        analyzer.analyze(
            "test change",
            target_files=["src/prism/auth.py"],
        )
        path = analyzer.last_report_path
        assert path is not None
        assert path.is_file()
        assert "impact_" in path.name

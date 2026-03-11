"""Tests for Phase 5 blast radius enhancements — load_report, get_summary, affected_functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism.intelligence.blast_radius import (
    AffectedFile,
    BlastRadiusAnalyzer,
    ImpactReport,
    RiskLevel,
)

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
    """Create a minimal project with source files containing public functions."""
    src = tmp_path / "src" / "prism"
    src.mkdir(parents=True)
    tests = tmp_path / "tests"
    tests.mkdir()

    (src / "__init__.py").write_text("")

    (src / "auth.py").write_text(
        "def validate_key(key: str) -> bool:\n"
        "    return bool(key)\n\n"
        "def authenticate(user: str) -> bool:\n"
        "    return True\n\n"
        "def _private_helper() -> None:\n"
        "    pass\n"
    )

    (src / "router.py").write_text(
        "from prism.auth import validate_key\n"
        "def route(task: str) -> str:\n"
        "    return task\n\n"
        "async def async_route(task: str) -> str:\n"
        "    return task\n"
    )

    (src / "utils.py").write_text(
        "def helper() -> int:\n"
        "    return 42\n"
    )

    (tests / "test_auth.py").write_text(
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
) -> ImpactReport:
    """Helper to build an ImpactReport for testing."""
    return ImpactReport(
        description="refactor auth validation",
        risk_score=risk_score,
        affected_files=affected or [],
        missing_tests=missing_tests or [],
        recommended_test_order=["tests/test_auth.py"],
        execution_order=["src/prism/auth.py"],
        estimated_complexity="moderate",
        critical_paths=critical_paths or [],
        created_at="2025-06-15T12:00:00+00:00",
    )


# ======================================================================
# load_report — round-trip with _save_report
# ======================================================================


class TestLoadReport:
    """Tests for BlastRadiusAnalyzer.load_report."""

    def test_round_trip_empty_affected(
        self, analyzer: BlastRadiusAnalyzer, reports_dir: Path,
    ) -> None:
        """A report with no affected files round-trips correctly."""
        report = _make_report()
        path = analyzer._save_report(report)

        loaded = analyzer.load_report(path)
        assert loaded.description == report.description
        assert loaded.risk_score == report.risk_score
        assert loaded.estimated_complexity == report.estimated_complexity
        assert loaded.affected_files == []

    def test_round_trip_with_affected_files(
        self, analyzer: BlastRadiusAnalyzer, reports_dir: Path,
    ) -> None:
        """A report with nested AffectedFile objects round-trips correctly."""
        affected = [
            AffectedFile(
                path="src/prism/auth.py",
                risk_level=RiskLevel.HIGH,
                reason="Direct target",
                affected_functions=["validate_key", "authenticate"],
                has_tests=True,
                test_files=["tests/test_auth.py"],
                depth=0,
            ),
            AffectedFile(
                path="src/prism/router.py",
                risk_level=RiskLevel.MEDIUM,
                reason="Imports auth",
                affected_functions=[],
                has_tests=False,
                test_files=[],
                depth=1,
            ),
        ]
        report = _make_report(affected=affected)
        path = analyzer._save_report(report)

        loaded = analyzer.load_report(path)
        assert loaded.file_count == 2
        assert loaded.affected_files[0].path == "src/prism/auth.py"
        assert loaded.affected_files[0].risk_level == RiskLevel.HIGH
        assert loaded.affected_files[0].affected_functions == [
            "validate_key", "authenticate",
        ]
        assert loaded.affected_files[0].has_tests is True
        assert loaded.affected_files[0].depth == 0
        assert loaded.affected_files[1].path == "src/prism/router.py"
        assert loaded.affected_files[1].depth == 1

    def test_round_trip_preserves_all_fields(
        self, analyzer: BlastRadiusAnalyzer, reports_dir: Path,
    ) -> None:
        """Every top-level field on ImpactReport survives save/load."""
        report = _make_report(
            risk_score=88,
            missing_tests=["src/prism/router.py"],
            critical_paths=["src/prism/auth.py"],
        )
        path = analyzer._save_report(report)

        loaded = analyzer.load_report(path)
        assert loaded.risk_score == 88
        assert loaded.missing_tests == ["src/prism/router.py"]
        assert loaded.critical_paths == ["src/prism/auth.py"]
        assert loaded.recommended_test_order == ["tests/test_auth.py"]
        assert loaded.execution_order == ["src/prism/auth.py"]
        assert loaded.created_at == "2025-06-15T12:00:00+00:00"

    def test_load_report_file_not_found(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Loading a nonexistent report raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            analyzer.load_report(Path("/nonexistent/impact_report.json"))

    def test_load_report_invalid_json(
        self, analyzer: BlastRadiusAnalyzer, reports_dir: Path,
    ) -> None:
        """Loading a file with invalid JSON raises JSONDecodeError."""
        bad_file = reports_dir / "impact_bad.json"
        bad_file.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            analyzer.load_report(bad_file)

    def test_load_report_with_multiple_affected(
        self, analyzer: BlastRadiusAnalyzer, reports_dir: Path,
    ) -> None:
        """Reports with many affected files load all of them."""
        affected = [
            AffectedFile(
                path=f"src/prism/module_{i}.py",
                risk_level=RiskLevel.LOW,
                reason=f"Reason {i}",
                affected_functions=[f"func_{i}"],
                has_tests=i % 2 == 0,
                test_files=[f"tests/test_module_{i}.py"] if i % 2 == 0 else [],
                depth=min(i, 2),
            )
            for i in range(8)
        ]
        report = _make_report(affected=affected)
        path = analyzer._save_report(report)

        loaded = analyzer.load_report(path)
        assert loaded.file_count == 8
        assert loaded.affected_files[3].affected_functions == ["func_3"]


# ======================================================================
# get_summary
# ======================================================================


class TestGetSummary:
    """Tests for BlastRadiusAnalyzer.get_summary."""

    def test_summary_contains_risk_score(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Summary includes the risk score."""
        report = _make_report(risk_score=72)
        summary = analyzer.get_summary(report)
        assert "72/100" in summary

    def test_summary_contains_complexity(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Summary includes the complexity level."""
        report = _make_report()
        summary = analyzer.get_summary(report)
        assert "moderate" in summary

    def test_summary_contains_total_files(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Summary includes total file count."""
        affected = [
            AffectedFile("a.py", RiskLevel.HIGH, "", [], False, [], 0),
            AffectedFile("b.py", RiskLevel.LOW, "", [], False, [], 1),
        ]
        report = _make_report(affected=affected)
        summary = analyzer.get_summary(report)
        assert "Total Files Affected: 2" in summary

    def test_summary_contains_risk_breakdown(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Summary includes high/medium/low risk counts."""
        affected = [
            AffectedFile("a.py", RiskLevel.HIGH, "", [], False, [], 0),
            AffectedFile("b.py", RiskLevel.HIGH, "", [], False, [], 0),
            AffectedFile("c.py", RiskLevel.MEDIUM, "", [], True, [], 1),
            AffectedFile("d.py", RiskLevel.LOW, "", [], True, [], 2),
        ]
        report = _make_report(affected=affected)
        summary = analyzer.get_summary(report)
        assert "HIGH risk:   2" in summary
        assert "MEDIUM risk: 1" in summary
        assert "LOW risk:    1" in summary

    def test_summary_contains_missing_tests_count(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Summary includes missing tests count."""
        report = _make_report(
            missing_tests=["a.py", "b.py", "c.py"],
        )
        summary = analyzer.get_summary(report)
        assert "Missing Tests: 3" in summary

    def test_summary_contains_critical_paths(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Summary includes critical paths when present."""
        report = _make_report(
            critical_paths=["src/prism/auth.py", "src/prism/security.py"],
        )
        summary = analyzer.get_summary(report)
        assert "Critical Paths:" in summary
        assert "src/prism/auth.py" in summary
        assert "src/prism/security.py" in summary

    def test_summary_no_critical_paths(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Summary omits critical paths section when none exist."""
        report = _make_report(critical_paths=[])
        summary = analyzer.get_summary(report)
        assert "Critical Paths:" not in summary

    def test_summary_empty_report(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Summary for a report with no affected files still works."""
        report = _make_report(risk_score=0)
        summary = analyzer.get_summary(report)
        assert "Risk Score: 0/100" in summary
        assert "Total Files Affected: 0" in summary

    def test_summary_contains_description(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Summary includes the report description."""
        report = _make_report()
        summary = analyzer.get_summary(report)
        assert "refactor auth validation" in summary


# ======================================================================
# _extract_public_functions
# ======================================================================


class TestExtractPublicFunctions:
    """Tests for BlastRadiusAnalyzer._extract_public_functions."""

    def test_extracts_public_functions(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Public functions (non-underscore) are extracted."""
        funcs = analyzer._extract_public_functions("src/prism/auth.py")
        assert "validate_key" in funcs
        assert "authenticate" in funcs

    def test_excludes_private_functions(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Functions starting with underscore are excluded."""
        funcs = analyzer._extract_public_functions("src/prism/auth.py")
        assert "_private_helper" not in funcs

    def test_extracts_async_functions(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Async function definitions are also extracted."""
        funcs = analyzer._extract_public_functions("src/prism/router.py")
        assert "route" in funcs
        assert "async_route" in funcs

    def test_nonexistent_file_returns_empty(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """A file that doesn't exist returns an empty list."""
        funcs = analyzer._extract_public_functions("src/prism/nonexistent.py")
        assert funcs == []

    def test_syntax_error_returns_empty(
        self, project: Path, reports_dir: Path,
    ) -> None:
        """A file with a syntax error returns an empty list."""
        bad = project / "src" / "prism" / "broken.py"
        bad.write_text("def broken(\n")
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        funcs = analyzer._extract_public_functions("src/prism/broken.py")
        assert funcs == []


# ======================================================================
# _find_affected includes affected_functions at depth 0
# ======================================================================


class TestFindAffectedFunctions:
    """Tests that _find_affected populates affected_functions at depth 0."""

    def test_depth0_has_functions(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Depth-0 targets get their public functions extracted."""
        analyzer._build_import_graph()
        affected = analyzer._find_affected(["src/prism/auth.py"])
        auth_af = next(
            af for af in affected if af.path == "src/prism/auth.py"
        )
        assert auth_af.depth == 0
        assert "validate_key" in auth_af.affected_functions
        assert "authenticate" in auth_af.affected_functions
        # Private functions should be excluded
        assert "_private_helper" not in auth_af.affected_functions

    def test_depth1_has_empty_functions(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Depth-1 files do not have affected_functions populated."""
        analyzer._build_import_graph()
        affected = analyzer._find_affected(["src/prism/auth.py"])
        depth1 = [af for af in affected if af.depth == 1]
        for af in depth1:
            assert af.affected_functions == []

    def test_depth0_nonexistent_file(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """A nonexistent depth-0 target gets an empty affected_functions list."""
        analyzer._build_import_graph()
        affected = analyzer._find_affected(["src/prism/nonexistent.py"])
        af = affected[0]
        assert af.depth == 0
        assert af.affected_functions == []

    def test_multiple_targets_get_functions(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Multiple depth-0 targets each get their own functions."""
        analyzer._build_import_graph()
        affected = analyzer._find_affected(
            ["src/prism/auth.py", "src/prism/router.py"],
        )
        auth_af = next(af for af in affected if "auth.py" in af.path)
        router_af = next(af for af in affected if "router.py" in af.path)
        assert "validate_key" in auth_af.affected_functions
        assert "route" in router_af.affected_functions


# ======================================================================
# Full integration with load_report + get_summary
# ======================================================================


class TestIntegrationLoadSummary:
    """Integration tests combining analyze, save, load, and summary."""

    def test_analyze_save_load_summary(
        self, analyzer: BlastRadiusAnalyzer,
    ) -> None:
        """Full chain: analyze -> save -> load -> summary."""
        report = analyzer.analyze(
            "refactor auth validation",
            target_files=["src/prism/auth.py"],
        )
        # Report was auto-saved — find it
        saved = analyzer.list_reports()
        assert len(saved) >= 1

        loaded = analyzer.load_report(saved[0])
        assert loaded.description == report.description
        assert loaded.risk_score == report.risk_score

        summary = analyzer.get_summary(loaded)
        assert "Risk Score:" in summary
        assert "Complexity:" in summary

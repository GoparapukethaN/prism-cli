"""Tests for prism.intelligence.blast_radius — Predictive Blast Radius Analysis."""

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
# RiskLevel constants
# ======================================================================


class TestRiskLevel:
    """Tests for the RiskLevel constant container."""

    def test_high_constant(self) -> None:
        assert RiskLevel.HIGH == "high"

    def test_medium_constant(self) -> None:
        assert RiskLevel.MEDIUM == "medium"

    def test_low_constant(self) -> None:
        assert RiskLevel.LOW == "low"

    def test_all_distinct(self) -> None:
        values = {RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW}
        assert len(values) == 3


# ======================================================================
# AffectedFile dataclass
# ======================================================================


class TestAffectedFile:
    """Tests for the AffectedFile dataclass."""

    def test_fields(self) -> None:
        af = AffectedFile(
            path="src/prism/foo.py",
            risk_level=RiskLevel.HIGH,
            reason="Direct target",
            affected_functions=["bar"],
            has_tests=True,
            test_files=["tests/test_foo.py"],
            depth=0,
        )
        assert af.path == "src/prism/foo.py"
        assert af.risk_level == RiskLevel.HIGH
        assert af.reason == "Direct target"
        assert af.affected_functions == ["bar"]
        assert af.has_tests is True
        assert af.test_files == ["tests/test_foo.py"]
        assert af.depth == 0

    def test_depth_levels(self) -> None:
        for depth in (0, 1, 2):
            af = AffectedFile(
                path="x.py",
                risk_level=RiskLevel.LOW,
                reason="test",
                affected_functions=[],
                has_tests=False,
                test_files=[],
                depth=depth,
            )
            assert af.depth == depth

    def test_empty_affected_functions(self) -> None:
        af = AffectedFile(
            path="x.py",
            risk_level=RiskLevel.LOW,
            reason="test",
            affected_functions=[],
            has_tests=False,
            test_files=[],
            depth=0,
        )
        assert af.affected_functions == []


# ======================================================================
# ImpactReport dataclass
# ======================================================================


class TestImpactReport:
    """Tests for the ImpactReport dataclass and its properties."""

    @staticmethod
    def _make_report(
        affected: list[AffectedFile] | None = None,
    ) -> ImpactReport:
        return ImpactReport(
            description="test change",
            risk_score=42,
            affected_files=affected or [],
            missing_tests=[],
            recommended_test_order=[],
            execution_order=[],
            estimated_complexity="simple",
            critical_paths=[],
            created_at="2025-01-01T00:00:00+00:00",
        )

    def test_basic_fields(self) -> None:
        report = self._make_report()
        assert report.description == "test change"
        assert report.risk_score == 42
        assert report.estimated_complexity == "simple"
        assert report.created_at.startswith("2025")

    def test_high_risk_count_zero(self) -> None:
        report = self._make_report()
        assert report.high_risk_count == 0

    def test_high_risk_count_nonzero(self) -> None:
        affected = [
            AffectedFile("a.py", RiskLevel.HIGH, "", [], False, [], 0),
            AffectedFile("b.py", RiskLevel.MEDIUM, "", [], False, [], 0),
            AffectedFile("c.py", RiskLevel.HIGH, "", [], False, [], 0),
        ]
        report = self._make_report(affected)
        assert report.high_risk_count == 2

    def test_file_count(self) -> None:
        affected = [
            AffectedFile("a.py", RiskLevel.LOW, "", [], False, [], 0),
            AffectedFile("b.py", RiskLevel.LOW, "", [], False, [], 1),
        ]
        report = self._make_report(affected)
        assert report.file_count == 2

    def test_file_count_empty(self) -> None:
        report = self._make_report()
        assert report.file_count == 0


# ======================================================================
# BlastRadiusAnalyzer
# ======================================================================


class TestBlastRadiusAnalyzer:
    """Tests for the BlastRadiusAnalyzer engine."""

    @pytest.fixture()
    def project(self, tmp_path: Path) -> Path:
        """Create a minimal project layout for testing."""
        src = tmp_path / "src" / "prism"
        src.mkdir(parents=True)
        tests = tmp_path / "tests"
        tests.mkdir()

        # auth module — high risk
        (src / "auth.py").write_text(
            "from prism.security import check\n"
            "def validate_key(key: str) -> bool:\n"
            "    return bool(key)\n"
        )

        # security module — high risk
        (src / "security.py").write_text(
            "def check(value: str) -> bool:\n"
            "    return len(value) > 0\n"
        )

        # router module — medium risk, imports auth
        (src / "router.py").write_text(
            "from prism.auth import validate_key\n"
            "def route(task: str) -> str:\n"
            "    return task\n"
        )

        # utils module — low risk
        (src / "utils.py").write_text(
            "def helper() -> int:\n"
            "    return 42\n"
        )

        # config module — medium risk, imports router
        (src / "config.py").write_text(
            "from prism.router import route\n"
            "def load_config() -> dict:\n"
            "    return {}\n"
        )

        # __init__.py so directory is a package
        (src / "__init__.py").write_text("")

        # A test file for auth
        (tests / "test_auth.py").write_text(
            "from prism.auth import validate_key\n"
            "def test_validate_key():\n"
            "    assert validate_key('abc')\n"
        )

        return tmp_path

    @pytest.fixture()
    def reports_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "reports"
        d.mkdir()
        return d

    # ----- Initialization -----

    def test_init_creates_reports_dir(self, tmp_path: Path) -> None:
        reports = tmp_path / "custom_reports"
        BlastRadiusAnalyzer(tmp_path, reports_dir=reports)
        assert reports.is_dir()

    def test_init_default_reports_dir(self, project: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project)
        assert analyzer._reports_dir == Path.home() / ".prism" / "impact_reports"

    # ----- Import graph -----

    def test_build_import_graph(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        analyzer._build_import_graph()
        graph = analyzer._import_graph
        assert len(graph) > 0
        # router.py imports auth.py
        router_key = next(
            (k for k in graph if "router.py" in k), None,
        )
        assert router_key is not None
        assert any("auth.py" in dep for dep in graph[router_key])

    def test_build_import_graph_no_src(self, tmp_path: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(tmp_path, reports_dir=reports_dir)
        analyzer._build_import_graph()
        assert analyzer._import_graph == {}

    def test_build_import_graph_syntax_error(
        self, project: Path, reports_dir: Path,
    ) -> None:
        bad = project / "src" / "prism" / "bad.py"
        bad.write_text("def broken(:\n")
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        analyzer._build_import_graph()
        # Should not crash; bad file recorded with empty deps
        bad_key = next((k for k in analyzer._import_graph if "bad.py" in k), None)
        assert bad_key is not None
        assert analyzer._import_graph[bad_key] == set()

    # ----- Call graph -----

    def test_build_call_graph(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        analyzer._build_call_graph()
        assert len(analyzer._call_graph) > 0

    def test_build_call_graph_no_src(self, tmp_path: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(tmp_path, reports_dir=reports_dir)
        analyzer._build_call_graph()
        assert analyzer._call_graph == {}

    # ----- Target discovery -----

    def test_find_targets_by_keyword(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        targets = analyzer._find_targets("change the router logic")
        assert any("router" in t for t in targets)

    def test_find_targets_no_match(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        targets = analyzer._find_targets("xy")
        assert targets == []

    def test_find_targets_no_src(self, tmp_path: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(tmp_path, reports_dir=reports_dir)
        assert analyzer._find_targets("anything") == []

    def test_find_targets_limit_ten(self, project: Path, reports_dir: Path) -> None:
        # Create many files matching keyword
        src = project / "src" / "prism"
        for i in range(15):
            (src / f"widget{i}.py").write_text(f"x = {i}\n")
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        targets = analyzer._find_targets("widget stuff")
        assert len(targets) <= 10

    # ----- Affected files -----

    def test_find_affected_direct(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        analyzer._build_import_graph()
        affected = analyzer._find_affected(["src/prism/auth.py"])
        paths = [af.path for af in affected]
        assert "src/prism/auth.py" in paths
        assert any(af.depth == 0 for af in affected if af.path == "src/prism/auth.py")

    def test_find_affected_depth1(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        analyzer._build_import_graph()
        affected = analyzer._find_affected(["src/prism/auth.py"])
        depths = {af.path: af.depth for af in affected}
        # router.py imports auth.py → depth 1
        router_key = next((k for k in depths if "router.py" in k), None)
        if router_key:
            assert depths[router_key] == 1

    def test_find_affected_depth2(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        analyzer._build_import_graph()
        affected = analyzer._find_affected(["src/prism/auth.py"])
        depths = {af.path: af.depth for af in affected}
        # config.py imports router.py → depth 2
        config_key = next((k for k in depths if "config.py" in k), None)
        if config_key:
            assert depths[config_key] == 2

    def test_find_affected_empty_targets(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        analyzer._build_import_graph()
        affected = analyzer._find_affected([])
        assert affected == []

    # ----- Risk assessment -----

    def test_assess_risk_auth_high(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        af = AffectedFile("src/prism/auth.py", "", "", [], False, [], 0)
        assert analyzer._assess_risk(af) == RiskLevel.HIGH

    def test_assess_risk_security_high(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        af = AffectedFile("src/prism/security.py", "", "", [], False, [], 0)
        assert analyzer._assess_risk(af) == RiskLevel.HIGH

    def test_assess_risk_credential_high(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        af = AffectedFile("src/prism/credential_store.py", "", "", [], False, [], 0)
        assert analyzer._assess_risk(af) == RiskLevel.HIGH

    def test_assess_risk_migration_high(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        af = AffectedFile("src/prism/migration.py", "", "", [], False, [], 0)
        assert analyzer._assess_risk(af) == RiskLevel.HIGH

    def test_assess_risk_router_medium(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        af = AffectedFile("src/prism/router.py", "", "", [], False, [], 0)
        assert analyzer._assess_risk(af) == RiskLevel.MEDIUM

    def test_assess_risk_config_medium(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        af = AffectedFile("src/prism/config.py", "", "", [], False, [], 0)
        assert analyzer._assess_risk(af) == RiskLevel.MEDIUM

    def test_assess_risk_utils_depth0_medium(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        af = AffectedFile("src/prism/utils.py", "", "", [], False, [], 0)
        assert analyzer._assess_risk(af) == RiskLevel.MEDIUM

    def test_assess_risk_deep_low(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        af = AffectedFile("src/prism/helpers.py", "", "", [], False, [], 2)
        assert analyzer._assess_risk(af) == RiskLevel.LOW

    # ----- Find tests -----

    def test_find_tests_for_existing(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        tests = analyzer._find_tests_for("src/prism/auth.py")
        assert any("test_auth" in t for t in tests)

    def test_find_tests_for_missing(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        tests = analyzer._find_tests_for("src/prism/utils.py")
        assert tests == []

    def test_find_tests_no_tests_dir(self, tmp_path: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(tmp_path, reports_dir=reports_dir)
        assert analyzer._find_tests_for("foo.py") == []

    # ----- Risk score -----

    def test_calculate_risk_score_empty(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        assert analyzer._calculate_risk_score([]) == 0

    def test_calculate_risk_score_caps_at_100(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        many = [
            AffectedFile(f"f{i}.py", RiskLevel.HIGH, "", [], False, [], 0)
            for i in range(20)
        ]
        score = analyzer._calculate_risk_score(many)
        assert score == 100

    def test_calculate_risk_score_includes_untested_penalty(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        tested = [
            AffectedFile("a.py", RiskLevel.LOW, "", [], True, ["t.py"], 0),
        ]
        untested = [
            AffectedFile("a.py", RiskLevel.LOW, "", [], False, [], 0),
        ]
        score_tested = analyzer._calculate_risk_score(tested)
        score_untested = analyzer._calculate_risk_score(untested)
        assert score_untested > score_tested

    # ----- Complexity estimation -----

    def test_estimate_complexity_trivial(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        affected = [
            AffectedFile("a.py", RiskLevel.LOW, "", [], True, [], 0),
        ]
        assert analyzer._estimate_complexity(affected) == "trivial"

    def test_estimate_complexity_simple(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        affected = [
            AffectedFile(f"f{i}.py", RiskLevel.LOW, "", [], True, [], 0)
            for i in range(4)
        ]
        assert analyzer._estimate_complexity(affected) == "simple"

    def test_estimate_complexity_moderate(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        affected = [
            AffectedFile(f"f{i}.py", RiskLevel.LOW, "", [], True, [], 0)
            for i in range(10)
        ]
        assert analyzer._estimate_complexity(affected) == "moderate"

    def test_estimate_complexity_complex(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        affected = [
            AffectedFile(f"f{i}.py", RiskLevel.HIGH, "", [], True, [], 0)
            for i in range(20)
        ]
        assert analyzer._estimate_complexity(affected) == "complex"

    # ----- Execution order -----

    def test_build_execution_order_leaves_first(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        affected = [
            AffectedFile("a.py", RiskLevel.LOW, "", [], False, [], 0),
            AffectedFile("b.py", RiskLevel.LOW, "", [], False, [], 2),
            AffectedFile("c.py", RiskLevel.LOW, "", [], False, [], 1),
        ]
        order = analyzer._build_execution_order(affected)
        assert order[0] == "b.py"  # depth 2 first
        assert order[-1] == "a.py"  # depth 0 last

    # ----- Report persistence -----

    def test_save_report(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        report = ImpactReport(
            description="test",
            risk_score=50,
            affected_files=[],
            missing_tests=[],
            recommended_test_order=[],
            execution_order=[],
            estimated_complexity="simple",
            critical_paths=[],
            created_at="2025-01-01T00:00:00+00:00",
        )
        path = analyzer._save_report(report)
        assert path.is_file()
        data = json.loads(path.read_text())
        assert data["risk_score"] == 50
        assert data["description"] == "test"

    def test_list_reports(self, project: Path, reports_dir: Path) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        # Save two reports
        for score in (10, 20):
            report = ImpactReport(
                description="t",
                risk_score=score,
                affected_files=[],
                missing_tests=[],
                recommended_test_order=[],
                execution_order=[],
                estimated_complexity="trivial",
                critical_paths=[],
                created_at=f"2025-01-0{score}T00:00:00+00:00",
            )
            analyzer._save_report(report)
        reports = analyzer.list_reports()
        assert len(reports) >= 2

    # ----- Full integration -----

    def test_analyze_full_integration(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        report = analyzer.analyze(
            "refactor auth validation",
            target_files=["src/prism/auth.py"],
        )
        assert report.risk_score > 0
        assert report.file_count >= 1
        assert report.estimated_complexity in (
            "trivial", "simple", "moderate", "complex",
        )
        assert any("auth" in af.path for af in report.affected_files)

    def test_analyze_infers_targets(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        report = analyzer.analyze("update the router")
        paths = [af.path for af in report.affected_files]
        assert any("router" in p for p in paths)

    def test_analyze_no_targets_found(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        report = analyzer.analyze("do xy")
        assert report.risk_score == 0
        assert report.file_count == 0

    def test_analyze_saves_report_file(
        self, project: Path, reports_dir: Path,
    ) -> None:
        analyzer = BlastRadiusAnalyzer(project, reports_dir=reports_dir)
        analyzer.analyze("change auth", target_files=["src/prism/auth.py"])
        reports = list(reports_dir.glob("impact_*.json"))
        assert len(reports) >= 1

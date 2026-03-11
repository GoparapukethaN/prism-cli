"""Tests for prism.intelligence.test_gaps — Intelligent Test Gap Hunter."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from prism.intelligence.test_gaps import (
    FunctionInfo,
    GapRisk,
    TestGap,
    TestGapHunter,
    TestGapReport,
)

# ======================================================================
# GapRisk constants
# ======================================================================


class TestGapRisk:
    """Tests for the GapRisk constant container."""

    def test_critical_constant(self) -> None:
        assert GapRisk.CRITICAL == "critical"

    def test_high_constant(self) -> None:
        assert GapRisk.HIGH == "high"

    def test_medium_constant(self) -> None:
        assert GapRisk.MEDIUM == "medium"

    def test_low_constant(self) -> None:
        assert GapRisk.LOW == "low"

    def test_all_distinct(self) -> None:
        values = {GapRisk.CRITICAL, GapRisk.HIGH, GapRisk.MEDIUM, GapRisk.LOW}
        assert len(values) == 4


# ======================================================================
# TestGap dataclass
# ======================================================================


class TestTestGap:
    """Tests for the TestGap dataclass."""

    def test_fields(self) -> None:
        gap = TestGap(
            function_name="do_thing",
            file_path="src/prism/foo.py",
            line_number=10,
            risk_level=GapRisk.HIGH,
            reason="needs coverage",
            description="no tests",
            test_stub="def test_do_thing(): ...",
            estimated_effort="minutes",
        )
        assert gap.function_name == "do_thing"
        assert gap.file_path == "src/prism/foo.py"
        assert gap.line_number == 10
        assert gap.risk_level == GapRisk.HIGH
        assert gap.reason == "needs coverage"
        assert gap.description == "no tests"
        assert "test_do_thing" in gap.test_stub
        assert gap.estimated_effort == "minutes"

    def test_hours_effort(self) -> None:
        gap = TestGap(
            function_name="complex_fn",
            file_path="x.py",
            line_number=1,
            risk_level=GapRisk.MEDIUM,
            reason="",
            description="",
            test_stub="",
            estimated_effort="hours",
        )
        assert gap.estimated_effort == "hours"


# ======================================================================
# TestGapReport dataclass
# ======================================================================


class TestTestGapReport:
    """Tests for the TestGapReport dataclass."""

    @staticmethod
    def _make_report(
        total: int = 10,
        tested: int = 7,
        gaps: list[TestGap] | None = None,
        critical: int = 0,
        high: int = 0,
    ) -> TestGapReport:
        untested = total - tested
        return TestGapReport(
            total_functions=total,
            tested_functions=tested,
            untested_functions=untested,
            coverage_percent=(tested / total * 100) if total > 0 else 100.0,
            gaps=gaps or [],
            critical_count=critical,
            high_count=high,
        )

    def test_fields(self) -> None:
        report = self._make_report()
        assert report.total_functions == 10
        assert report.tested_functions == 7
        assert report.untested_functions == 3
        assert report.coverage_percent == 70.0

    def test_has_critical_gaps_true(self) -> None:
        report = self._make_report(critical=2)
        assert report.has_critical_gaps is True

    def test_has_critical_gaps_false(self) -> None:
        report = self._make_report(critical=0)
        assert report.has_critical_gaps is False

    def test_coverage_percent_full(self) -> None:
        report = self._make_report(total=5, tested=5)
        assert report.coverage_percent == 100.0

    def test_coverage_percent_zero(self) -> None:
        report = self._make_report(total=5, tested=0)
        assert report.coverage_percent == 0.0

    def test_coverage_percent_empty(self) -> None:
        report = self._make_report(total=0, tested=0)
        assert report.coverage_percent == 100.0


# ======================================================================
# FunctionInfo dataclass
# ======================================================================


class TestFunctionInfo:
    """Tests for the FunctionInfo dataclass."""

    def test_fields(self) -> None:
        info = FunctionInfo(
            name="my_func",
            file_path="src/prism/mod.py",
            line_number=42,
            is_async=False,
            has_error_handling=True,
            parameters=["a", "b"],
            decorators=["staticmethod"],
            docstring="Does things.",
            complexity=3,
        )
        assert info.name == "my_func"
        assert info.file_path == "src/prism/mod.py"
        assert info.line_number == 42
        assert info.is_async is False
        assert info.has_error_handling is True
        assert info.parameters == ["a", "b"]
        assert info.decorators == ["staticmethod"]
        assert info.docstring == "Does things."
        assert info.complexity == 3

    def test_async_function(self) -> None:
        info = FunctionInfo(
            name="async_fn",
            file_path="x.py",
            line_number=1,
            is_async=True,
            has_error_handling=False,
            parameters=[],
            decorators=[],
            docstring=None,
            complexity=1,
        )
        assert info.is_async is True
        assert info.docstring is None


# ======================================================================
# TestGapHunter
# ======================================================================


class TestTestGapHunter:
    """Tests for the TestGapHunter engine."""

    @pytest.fixture()
    def project(self, tmp_path: Path) -> Path:
        """Create a minimal project layout for testing."""
        src = tmp_path / "src" / "prism"
        src.mkdir(parents=True)
        tests = tmp_path / "tests"
        tests.mkdir()

        # auth module — critical risk, has test
        (src / "auth.py").write_text(
            "def validate_token(token: str) -> bool:\n"
            '    """Validate auth token."""\n'
            "    return bool(token)\n"
        )

        # security module — critical risk, NO test
        (src / "security.py").write_text(
            "def encrypt_data(data: str) -> str:\n"
            '    """Encrypt sensitive data."""\n'
            "    return data[::-1]\n"
            "\n"
            "def _private_helper() -> None:\n"
            "    pass\n"
        )

        # router module — medium risk, has test
        (src / "router.py").write_text(
            "def classify_task(task: str) -> str:\n"
            '    """Classify task."""\n'
            "    return 'simple'\n"
            "\n"
            "async def process_request(req: str) -> str:\n"
            '    """Process request asynchronously."""\n'
            "    return req\n"
        )

        # utils module — low risk, NO test
        (src / "utils.py").write_text(
            "def format_text(text: str) -> str:\n"
            '    """Format text."""\n'
            "    return text.strip()\n"
        )

        # complex module — high risk (database), with error handling
        (src / "database.py").write_text(
            "def save_record(data: dict) -> bool:\n"
            '    """Save a record to the database."""\n'
            "    try:\n"
            "        return True\n"
            "    except Exception:\n"
            "        return False\n"
            "\n"
            "def delete_record(record_id: int) -> bool:\n"
            '    """Delete a record."""\n'
            "    if record_id < 0:\n"
            "        return False\n"
            "    return True\n"
        )

        (src / "__init__.py").write_text("")

        # Test files
        (tests / "test_auth.py").write_text(
            "from prism.auth import validate_token\n"
            "def test_validate_token():\n"
            "    assert validate_token('abc')\n"
        )

        (tests / "test_router.py").write_text(
            "from prism.router import classify_task\n"
            "def test_classify_task():\n"
            "    assert classify_task('x') == 'simple'\n"
        )

        return tmp_path

    # ----- Initialization -----

    def test_init(self, tmp_path: Path) -> None:
        hunter = TestGapHunter(tmp_path)
        assert hunter._root == tmp_path.resolve()
        assert hunter._src == tmp_path.resolve() / "src"
        assert hunter._tests == tmp_path.resolve() / "tests"

    def test_init_custom_dirs(self, tmp_path: Path) -> None:
        hunter = TestGapHunter(tmp_path, src_dir="lib", tests_dir="spec")
        assert hunter._src == tmp_path.resolve() / "lib"
        assert hunter._tests == tmp_path.resolve() / "spec"

    # ----- Function scanning -----

    def test_scan_functions(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        functions = hunter._scan_functions()
        names = [f.name for f in functions]
        assert "validate_token" in names
        assert "encrypt_data" in names
        assert "classify_task" in names
        assert "format_text" in names
        assert "save_record" in names
        assert "delete_record" in names

    def test_scan_functions_skips_private(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        functions = hunter._scan_functions()
        names = [f.name for f in functions]
        assert "_private_helper" not in names

    def test_scan_functions_detects_async(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        functions = hunter._scan_functions()
        process_req = next(
            (f for f in functions if f.name == "process_request"), None,
        )
        assert process_req is not None
        assert process_req.is_async is True

    def test_scan_functions_detects_error_handling(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        functions = hunter._scan_functions()
        save = next((f for f in functions if f.name == "save_record"), None)
        assert save is not None
        assert save.has_error_handling is True

    def test_scan_functions_captures_parameters(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        functions = hunter._scan_functions()
        validate = next(
            (f for f in functions if f.name == "validate_token"), None,
        )
        assert validate is not None
        assert "token" in validate.parameters

    def test_scan_functions_captures_docstring(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        functions = hunter._scan_functions()
        validate = next(
            (f for f in functions if f.name == "validate_token"), None,
        )
        assert validate is not None
        assert validate.docstring is not None
        assert "Validate" in validate.docstring

    def test_scan_functions_no_src_dir(self, tmp_path: Path) -> None:
        hunter = TestGapHunter(tmp_path)
        assert hunter._scan_functions() == []

    def test_scan_functions_syntax_error(self, project: Path) -> None:
        bad = project / "src" / "prism" / "bad.py"
        bad.write_text("def broken(:\n")
        hunter = TestGapHunter(project)
        functions = hunter._scan_functions()
        # Should not crash; bad file is simply skipped
        names = [f.name for f in functions]
        assert "broken" not in names

    # ----- Tested function detection -----

    def test_find_tested_functions(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        tested = hunter._find_tested_functions()
        assert "validate_token" in tested
        assert "classify_task" in tested

    def test_find_tested_functions_no_tests_dir(self, tmp_path: Path) -> None:
        hunter = TestGapHunter(tmp_path)
        assert hunter._find_tested_functions() == set()

    # ----- is_tested -----

    def test_is_tested_by_name(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        tested = hunter._find_tested_functions()
        func = FunctionInfo(
            "validate_token", "src/prism/auth.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._is_tested(func, tested) is True

    def test_is_tested_by_test_file(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        # "auth" stem → test_auth.py exists
        func = FunctionInfo(
            "some_other_func", "src/prism/auth.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._is_tested(func, set()) is True

    def test_is_tested_false(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "nonexistent_fn", "src/prism/utils.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._is_tested(func, set()) is False

    # ----- Risk assessment -----

    def test_assess_risk_auth_critical(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "check", "src/prism/auth.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._assess_risk(func) == GapRisk.CRITICAL

    def test_assess_risk_security_critical(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "encrypt_data", "src/prism/security.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._assess_risk(func) == GapRisk.CRITICAL

    def test_assess_risk_password_in_name_critical(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "reset_password", "src/prism/utils.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._assess_risk(func) == GapRisk.CRITICAL

    def test_assess_risk_database_high(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "query", "src/prism/database.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._assess_risk(func) == GapRisk.HIGH

    def test_assess_risk_save_in_name_high(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "save_settings", "src/prism/foo.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._assess_risk(func) == GapRisk.HIGH

    def test_assess_risk_delete_in_name_high(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "delete_user", "src/prism/foo.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._assess_risk(func) == GapRisk.HIGH

    def test_assess_risk_router_medium(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "dispatch", "src/prism/router.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._assess_risk(func) == GapRisk.MEDIUM

    def test_assess_risk_classify_in_name_medium(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "classify_task", "src/prism/foo.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._assess_risk(func) == GapRisk.MEDIUM

    def test_assess_risk_utility_low(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "format_text", "src/prism/helpers.py", 1,
            False, False, [], [], None, 1,
        )
        assert hunter._assess_risk(func) == GapRisk.LOW

    # ----- Risk reason -----

    def test_get_risk_reason_critical(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "check", "src/prism/auth.py", 1,
            False, False, [], [], None, 1,
        )
        reason = hunter._get_risk_reason(func, GapRisk.CRITICAL)
        assert "Security" in reason

    def test_get_risk_reason_with_error_handling(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "check", "src/prism/auth.py", 1,
            False, True, [], [], None, 1,
        )
        reason = hunter._get_risk_reason(func, GapRisk.HIGH)
        assert "error handling" in reason

    def test_get_risk_reason_with_high_complexity(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "check", "src/prism/auth.py", 1,
            False, False, [], [], None, 8,
        )
        reason = hunter._get_risk_reason(func, GapRisk.MEDIUM)
        assert "complexity" in reason

    # ----- Test stub generation -----

    def test_generate_test_stub_sync(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "do_thing", "src/prism/foo.py", 1,
            False, False, ["a", "b"], [], None, 1,
        )
        stub = hunter._generate_test_stub(func)
        assert "def test_do_thing()" in stub
        assert "a, b" in stub
        assert "assert result is not None" in stub

    def test_generate_test_stub_async(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "do_async", "src/prism/foo.py", 1,
            True, False, ["req"], [], None, 1,
        )
        stub = hunter._generate_test_stub(func)
        assert "async def test_do_async()" in stub
        assert "await do_async(req)" in stub

    def test_generate_test_stub_no_params(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "get_version", "src/prism/foo.py", 1,
            False, False, [], [], None, 1,
        )
        stub = hunter._generate_test_stub(func)
        assert "get_version()" in stub

    # ----- Effort estimation -----

    def test_estimate_effort_simple(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "simple_fn", "x.py", 1, False, False, [], [], None, 2,
        )
        assert hunter._estimate_effort(func) == "minutes"

    def test_estimate_effort_complex(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "complex_fn", "x.py", 1, False, True, [], [], None, 8,
        )
        assert hunter._estimate_effort(func) == "hours"

    def test_estimate_effort_high_complexity_no_error(
        self, project: Path,
    ) -> None:
        hunter = TestGapHunter(project)
        func = FunctionInfo(
            "fn", "x.py", 1, False, False, [], [], None, 5,
        )
        assert hunter._estimate_effort(func) == "hours"

    # ----- Complexity estimation -----

    def test_estimate_complexity_minimal(self) -> None:
        code = "def f(): return 1"
        tree = ast.parse(code)
        func_node = tree.body[0]
        assert TestGapHunter._estimate_complexity(func_node) == 1

    def test_estimate_complexity_with_if(self) -> None:
        code = "def f(x):\n  if x: return 1\n  return 0"
        tree = ast.parse(code)
        func_node = tree.body[0]
        assert TestGapHunter._estimate_complexity(func_node) == 2

    def test_estimate_complexity_with_loop_and_exception(self) -> None:
        code = (
            "def f(items):\n"
            "  for i in items:\n"
            "    try:\n"
            "      pass\n"
            "    except:\n"
            "      pass\n"
        )
        tree = ast.parse(code)
        func_node = tree.body[0]
        # base + for + except = 3
        assert TestGapHunter._estimate_complexity(func_node) == 3

    def test_estimate_complexity_with_bool_op(self) -> None:
        code = "def f(a, b, c):\n  if a and b and c: return 1\n  return 0"
        tree = ast.parse(code)
        func_node = tree.body[0]
        # base + if + (BoolOp with 3 values → +2) = 4
        assert TestGapHunter._estimate_complexity(func_node) == 4

    # ----- Full integration -----

    def test_analyze_full_integration(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        assert report.total_functions > 0
        assert report.coverage_percent > 0
        assert isinstance(report.gaps, list)
        # All gaps should be sorted by risk
        risk_order = {
            GapRisk.CRITICAL: 0,
            GapRisk.HIGH: 1,
            GapRisk.MEDIUM: 2,
            GapRisk.LOW: 3,
        }
        for i in range(len(report.gaps) - 1):
            current = risk_order.get(report.gaps[i].risk_level, 4)
            nxt = risk_order.get(report.gaps[i + 1].risk_level, 4)
            assert current <= nxt

    def test_analyze_counts_match(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        assert (
            report.tested_functions + report.untested_functions
            == report.total_functions
        )

    def test_analyze_empty_project(self, tmp_path: Path) -> None:
        hunter = TestGapHunter(tmp_path)
        report = hunter.analyze()
        assert report.total_functions == 0
        assert report.coverage_percent == 100.0
        assert report.gaps == []
        assert report.has_critical_gaps is False

    def test_analyze_detects_critical_gaps(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        # encrypt_data in security.py has no test → critical
        critical_names = [
            g.function_name for g in report.gaps
            if g.risk_level == GapRisk.CRITICAL
        ]
        assert "encrypt_data" in critical_names

    def test_analyze_detects_high_gaps(self, project: Path) -> None:
        hunter = TestGapHunter(project)
        report = hunter.analyze()
        high_names = [
            g.function_name for g in report.gaps
            if g.risk_level == GapRisk.HIGH
        ]
        # save_record or delete_record in database.py should be high
        assert any(n in high_names for n in ("save_record", "delete_record"))

"""Tests for the ``prism test-gaps`` CLI command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from prism.cli.app import app
from prism.intelligence.test_gaps import (
    GapRisk,
    TestGap,
    TestGapReport,
)

runner = CliRunner()


# ======================================================================
# Helpers
# ======================================================================


def _make_gap(
    name: str = "do_thing",
    file_path: str = "src/prism/foo.py",
    line: int = 10,
    risk: str = GapRisk.LOW,
    scenarios: list[str] | None = None,
) -> TestGap:
    """Create a TestGap for testing."""
    return TestGap(
        function_name=name,
        file_path=file_path,
        line_number=line,
        risk_level=risk,
        reason="needs coverage",
        description=f"Function '{name}' has no test coverage",
        test_stub=f"def test_{name}(): ...",
        estimated_effort="minutes",
        scenarios=scenarios or [],
    )


def _make_report(
    gaps: list[TestGap] | None = None,
    total: int = 10,
    tested: int = 7,
) -> TestGapReport:
    """Create a TestGapReport for testing."""
    gaps = gaps or []
    critical = sum(1 for g in gaps if g.risk_level == GapRisk.CRITICAL)
    high = sum(1 for g in gaps if g.risk_level == GapRisk.HIGH)
    untested = total - tested
    return TestGapReport(
        total_functions=total,
        tested_functions=tested,
        untested_functions=untested,
        coverage_percent=(tested / total * 100) if total > 0 else 100.0,
        gaps=gaps,
        critical_count=critical,
        high_count=high,
    )


# ======================================================================
# CLI: prism test-gaps
# ======================================================================


class TestTestGapsCommand:
    """Tests for the prism test-gaps CLI command."""

    @patch("prism.intelligence.test_gaps.TestGapHunter")
    def test_no_options_displays_summary(
        self,
        mock_hunter_cls: MagicMock,
    ) -> None:
        """prism test-gaps with no options should display summary panel."""
        gaps = [
            _make_gap("encrypt_data", risk=GapRisk.CRITICAL),
            _make_gap("save_record", risk=GapRisk.HIGH),
            _make_gap("format_text", risk=GapRisk.LOW),
        ]
        report = _make_report(gaps=gaps, total=10, tested=7)
        mock_hunter_cls.return_value.analyze.return_value = report

        result = runner.invoke(app, ["test-gaps"])
        assert result.exit_code == 0
        assert "Test Gap Analysis" in result.output
        assert "Total functions: 10" in result.output
        assert "Coverage: 70.0%" in result.output

    @patch("prism.intelligence.test_gaps.TestGapHunter")
    def test_critical_flag_filters_gaps(
        self,
        mock_hunter_cls: MagicMock,
    ) -> None:
        """prism test-gaps --critical should filter to critical only."""
        gaps = [
            _make_gap("encrypt_data", risk=GapRisk.CRITICAL),
            _make_gap("save_record", risk=GapRisk.HIGH),
            _make_gap("format_text", risk=GapRisk.LOW),
        ]
        report = _make_report(gaps=gaps, total=10, tested=7)
        mock_hunter_cls.return_value.analyze.return_value = report

        result = runner.invoke(app, ["test-gaps", "--critical"])
        assert result.exit_code == 0
        assert "Filtered to critical risk only" in result.output

    @patch("prism.intelligence.test_gaps.TestGapHunter")
    def test_ci_exits_nonzero_with_critical_gaps(
        self,
        mock_hunter_cls: MagicMock,
    ) -> None:
        """prism test-gaps --ci should exit 1 if critical gaps exist."""
        gaps = [
            _make_gap("encrypt_data", risk=GapRisk.CRITICAL),
        ]
        report = _make_report(gaps=gaps, total=5, tested=4)
        mock_hunter_cls.return_value.analyze.return_value = report

        result = runner.invoke(app, ["test-gaps", "--ci"])
        assert result.exit_code == 1
        assert "CI check failed" in result.output

    @patch("prism.intelligence.test_gaps.TestGapHunter")
    def test_ci_exits_zero_without_critical_gaps(
        self,
        mock_hunter_cls: MagicMock,
    ) -> None:
        """prism test-gaps --ci should exit 0 if no critical gaps."""
        gaps = [
            _make_gap("format_text", risk=GapRisk.LOW),
        ]
        report = _make_report(gaps=gaps, total=5, tested=4)
        mock_hunter_cls.return_value.analyze.return_value = report

        result = runner.invoke(app, ["test-gaps", "--ci"])
        assert result.exit_code == 0

    @patch("prism.intelligence.test_gaps.TestGapHunter")
    def test_fix_writes_files(
        self,
        mock_hunter_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """prism test-gaps --fix should generate and write test files."""
        gaps = [
            _make_gap("encrypt_data", risk=GapRisk.CRITICAL),
        ]
        report = _make_report(gaps=gaps, total=5, tested=4)
        mock_hunter_cls.return_value.analyze.return_value = report

        test_file_path = str(
            tmp_path / "tests" / "test_security_generated.py"
        )
        test_content = (
            '"""Generated tests."""\n'
            "import pytest\n\n"
            "def test_encrypt_data() -> None:\n"
            '    """Test encrypt_data."""\n'
            "    pass\n"
        )
        mock_hunter_cls.return_value.generate_tests.return_value = {
            test_file_path: test_content,
        }

        result = runner.invoke(app, ["test-gaps", "--fix"])
        assert result.exit_code == 0
        assert "Generated" in result.output
        # Verify file was written
        assert Path(test_file_path).exists()
        assert "test_encrypt_data" in Path(test_file_path).read_text()

    @patch("prism.intelligence.test_gaps.TestGapHunter")
    def test_module_flag_uses_analyze_module(
        self,
        mock_hunter_cls: MagicMock,
    ) -> None:
        """prism test-gaps --module auth should call analyze_module."""
        report = _make_report(total=2, tested=2)
        mock_hunter_cls.return_value.analyze_module.return_value = report

        result = runner.invoke(
            app, ["test-gaps", "--module", "auth"],
        )
        assert result.exit_code == 0
        mock_hunter_cls.return_value.analyze_module.assert_called_once_with(
            "auth"
        )
        assert "Module: auth" in result.output

    @patch("prism.intelligence.test_gaps.TestGapHunter")
    def test_no_gaps_shows_green_message(
        self,
        mock_hunter_cls: MagicMock,
    ) -> None:
        """No gaps should show a positive message."""
        report = _make_report(total=5, tested=5)
        mock_hunter_cls.return_value.analyze.return_value = report

        result = runner.invoke(app, ["test-gaps"])
        assert result.exit_code == 0
        assert "No test gaps found" in result.output

    @patch("prism.intelligence.test_gaps.TestGapHunter")
    def test_displays_scenarios_count(
        self,
        mock_hunter_cls: MagicMock,
    ) -> None:
        """Table should show scenarios count column."""
        scenarios = [
            "Error path: except block at line 5",
            "Boundary: None check at line 3",
        ]
        gaps = [
            _make_gap(
                "encrypt_data",
                risk=GapRisk.CRITICAL,
                scenarios=scenarios,
            ),
        ]
        report = _make_report(gaps=gaps, total=5, tested=4)
        mock_hunter_cls.return_value.analyze.return_value = report

        result = runner.invoke(app, ["test-gaps"])
        assert result.exit_code == 0
        assert "encrypt_data" in result.output

    @patch("prism.intelligence.test_gaps.TestGapHunter")
    def test_displays_scenario_details(
        self,
        mock_hunter_cls: MagicMock,
    ) -> None:
        """Should display scenario details indented under gaps."""
        scenarios = [
            "Error path: except block at line 5",
        ]
        gaps = [
            _make_gap(
                "encrypt_data",
                risk=GapRisk.CRITICAL,
                scenarios=scenarios,
            ),
        ]
        report = _make_report(gaps=gaps, total=5, tested=4)
        mock_hunter_cls.return_value.analyze.return_value = report

        result = runner.invoke(app, ["test-gaps"])
        assert result.exit_code == 0
        assert "except block at line 5" in result.output

    @patch("prism.intelligence.test_gaps.TestGapHunter")
    def test_fix_no_gaps_shows_message(
        self,
        mock_hunter_cls: MagicMock,
    ) -> None:
        """--fix with no gaps should show informational message."""
        report = _make_report(total=5, tested=5)
        mock_hunter_cls.return_value.analyze.return_value = report

        result = runner.invoke(app, ["test-gaps", "--fix"])
        assert result.exit_code == 0
        assert "No gaps to generate" in result.output

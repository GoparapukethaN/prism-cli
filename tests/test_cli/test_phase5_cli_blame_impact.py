"""Tests for Phase 5 CLI commands — prism blame and prism impact."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from prism.cli.app import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


# ======================================================================
# Helpers
# ======================================================================


def _make_blame_report() -> MagicMock:
    """Create a mock BlameReport with realistic fields."""
    report = MagicMock()
    report.bug_description = "Widget crashes on startup"
    report.breaking_commit = "abc123def45678"
    report.breaking_author = "Alice"
    report.breaking_date = "2025-06-15T10:00:00+00:00"
    report.breaking_message = "feat: add startup logic"
    report.confidence = 0.7
    report.bisect_steps = 5
    report.affected_files = ["src/widget.py", "src/startup.py"]
    report.related_tests = ["tests/test_widget.py"]
    report.causal_narrative = "The crash was introduced in commit abc123."
    report.affected_lines = ["widget.py: +new_line"]
    report.proposed_fix = ""
    report.created_at = "2025-06-15T12:00:00+00:00"
    return report


def _make_impact_report() -> MagicMock:
    """Create a mock ImpactReport with realistic fields."""
    report = MagicMock()
    report.description = "refactor auth validation"
    report.risk_score = 65
    report.estimated_complexity = "moderate"
    report.file_count = 3
    report.high_risk_count = 1

    af1 = MagicMock()
    af1.path = "src/prism/auth.py"
    af1.risk_level = "high"
    af1.depth = 0
    af1.reason = "Direct target"
    af1.has_tests = True

    af2 = MagicMock()
    af2.path = "src/prism/router.py"
    af2.risk_level = "medium"
    af2.depth = 1
    af2.reason = "Imports auth"
    af2.has_tests = False

    report.affected_files = [af1, af2]
    report.missing_tests = ["src/prism/router.py"]
    report.recommended_test_order = ["tests/test_auth.py"]
    report.execution_order = ["src/prism/auth.py"]
    report.critical_paths = ["src/prism/auth.py"]
    report.created_at = "2025-06-15T12:00:00+00:00"
    return report


# ======================================================================
# Blame command — --list
# ======================================================================


class TestBlameListReports:
    """Tests for 'prism blame --list'."""

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_list_no_reports(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """--list with no reports shows a message."""
        mock_tracer = MagicMock()
        mock_tracer.list_reports.return_value = []
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app, ["blame", "--list", "placeholder"],
        )
        assert result.exit_code == 0
        assert "No blame reports found" in result.output

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_list_with_reports(
        self, mock_tracer_cls: MagicMock, tmp_path: Path,
    ) -> None:
        """--list with existing reports shows a table."""
        # Create a fake report file
        report_file = tmp_path / "blame_abc123_2025-06-15.json"
        report_file.write_text("{}")

        mock_tracer = MagicMock()
        mock_tracer.list_reports.return_value = [report_file]
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app, ["blame", "--list", "placeholder"],
        )
        assert result.exit_code == 0
        assert "blame_abc123" in result.output

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_list_shows_file_size(
        self, mock_tracer_cls: MagicMock, tmp_path: Path,
    ) -> None:
        """--list shows file sizes in KB."""
        report_file = tmp_path / "blame_def456_2025-06-16.json"
        report_file.write_text('{"data": "' + "x" * 1000 + '"}')

        mock_tracer = MagicMock()
        mock_tracer.list_reports.return_value = [report_file]
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app, ["blame", "--list", "placeholder"],
        )
        assert result.exit_code == 0
        assert "KB" in result.output


# ======================================================================
# Blame command — trace with description
# ======================================================================


class TestBlameTrace:
    """Tests for 'prism blame <description>'."""

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_trace_displays_report(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """A blame trace displays the report panel."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _make_blame_report()
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app, ["blame", "Widget crashes on startup"],
        )
        assert result.exit_code == 0
        assert "abc123def456" in result.output
        assert "Alice" in result.output
        assert "Blame Report" in result.output

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_trace_shows_affected_files(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """Blame trace shows affected files."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _make_blame_report()
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app, ["blame", "startup crash"],
        )
        assert result.exit_code == 0
        assert "src/widget.py" in result.output
        assert "src/startup.py" in result.output

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_trace_shows_related_tests(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """Blame trace shows related tests."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _make_blame_report()
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app, ["blame", "startup crash"],
        )
        assert result.exit_code == 0
        assert "tests/test_widget.py" in result.output

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_trace_shows_confidence(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """Blame trace shows confidence percentage."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _make_blame_report()
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app, ["blame", "crash bug"],
        )
        assert result.exit_code == 0
        assert "70%" in result.output

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_trace_shows_causal_narrative(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """Blame trace shows the causal narrative."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _make_blame_report()
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app, ["blame", "crash bug"],
        )
        assert result.exit_code == 0
        assert "Causal Narrative" in result.output


# ======================================================================
# Blame command — with --test and --good
# ======================================================================


class TestBlameWithBisect:
    """Tests for 'prism blame --test ... --good ...'."""

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_trace_with_test_and_good(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """Passing --test and --good invokes trace with those parameters."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _make_blame_report()
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app,
            [
                "blame",
                "test failure in auth",
                "--test", "pytest tests/",
                "--good", "abc123",
            ],
        )
        assert result.exit_code == 0

        # Verify the tracer was called with the right parameters
        mock_tracer.trace.assert_called_once_with(
            bug_description="test failure in auth",
            test_command="pytest tests/",
            good_commit="abc123",
        )

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_trace_with_test_only(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """Passing --test without --good still works (bisect skipped internally)."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _make_blame_report()
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app,
            ["blame", "auth bug", "--test", "pytest tests/"],
        )
        assert result.exit_code == 0
        mock_tracer.trace.assert_called_once_with(
            bug_description="auth bug",
            test_command="pytest tests/",
            good_commit=None,
        )

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_shows_test_command_in_output(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """Output includes the test command when provided."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _make_blame_report()
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app,
            ["blame", "auth bug", "--test", "pytest tests/"],
        )
        assert result.exit_code == 0
        assert "pytest tests/" in result.output


# ======================================================================
# Blame command — error handling
# ======================================================================


class TestBlameErrors:
    """Tests for blame command error handling."""

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_trace_failure_shows_error(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """If trace() raises, an error message is shown."""
        mock_tracer = MagicMock()
        mock_tracer.trace.side_effect = RuntimeError("git not found")
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app, ["blame", "some bug"],
        )
        assert result.exit_code == 1
        assert "Blame trace failed" in result.output

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_init_failure_shows_error(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """If CausalBlameTracer init fails, error is shown."""
        mock_tracer_cls.side_effect = OSError("Permission denied")

        result = runner.invoke(
            app, ["blame", "some bug"],
        )
        assert result.exit_code == 1
        assert "Error initializing" in result.output

    @patch("prism.intelligence.blame.CausalBlameTracer")
    def test_low_confidence_style(
        self, mock_tracer_cls: MagicMock,
    ) -> None:
        """Low confidence blame report is still displayed."""
        mock_tracer = MagicMock()
        report = _make_blame_report()
        report.confidence = 0.2
        mock_tracer.trace.return_value = report
        mock_tracer_cls.return_value = mock_tracer

        result = runner.invoke(
            app, ["blame", "intermittent bug"],
        )
        assert result.exit_code == 0
        assert "20%" in result.output


# ======================================================================
# Impact command — --list
# ======================================================================


class TestImpactListReports:
    """Tests for 'prism impact --list'."""

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_list_no_reports(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """--list with no reports shows a message."""
        mock_analyzer = MagicMock()
        mock_analyzer.list_reports.return_value = []
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "--list", "placeholder"],
        )
        assert result.exit_code == 0
        assert "No impact reports found" in result.output

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_list_with_reports(
        self, mock_analyzer_cls: MagicMock, tmp_path: Path,
    ) -> None:
        """--list with reports shows a table."""
        report_file = tmp_path / "impact_2025-06-15_65.json"
        report_file.write_text("{}")

        mock_analyzer = MagicMock()
        mock_analyzer.list_reports.return_value = [report_file]
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "--list", "placeholder"],
        )
        assert result.exit_code == 0
        assert "impact_2025-06-15" in result.output

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_list_shows_file_size(
        self, mock_analyzer_cls: MagicMock, tmp_path: Path,
    ) -> None:
        """--list shows file sizes in KB."""
        report_file = tmp_path / "impact_2025-06-15_50.json"
        report_file.write_text('{"big": "' + "y" * 2000 + '"}')

        mock_analyzer = MagicMock()
        mock_analyzer.list_reports.return_value = [report_file]
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "--list", "placeholder"],
        )
        assert result.exit_code == 0
        assert "KB" in result.output


# ======================================================================
# Impact command — with description
# ======================================================================


class TestImpactAnalyze:
    """Tests for 'prism impact <description>'."""

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_analyze_displays_results(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """Impact analysis displays risk score and complexity."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = _make_impact_report()
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "refactor auth validation"],
        )
        assert result.exit_code == 0
        assert "Blast Radius Report" in result.output
        assert "65" in result.output
        assert "moderate" in result.output

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_analyze_shows_affected_files_table(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """Impact analysis shows the affected files table."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = _make_impact_report()
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "refactor auth"],
        )
        assert result.exit_code == 0
        assert "src/prism/auth.py" in result.output
        assert "src/prism/router.py" in result.output

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_analyze_shows_missing_tests(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """Impact analysis shows missing tests warning."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = _make_impact_report()
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "refactor auth"],
        )
        assert result.exit_code == 0
        assert "Missing Tests" in result.output

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_analyze_calls_with_description(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """Verify analyze is called with the provided description."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = _make_impact_report()
        mock_analyzer_cls.return_value = mock_analyzer

        runner.invoke(app, ["impact", "update the router"])
        mock_analyzer.analyze.assert_called_once_with(
            description="update the router",
            target_files=None,
        )


# ======================================================================
# Impact command — with --file flags
# ======================================================================


class TestImpactWithFiles:
    """Tests for 'prism impact --file <path>'."""

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_single_file(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """Passing a single --file works correctly."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = _make_impact_report()
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app,
            [
                "impact", "change auth",
                "--file", "src/prism/auth.py",
            ],
        )
        assert result.exit_code == 0
        mock_analyzer.analyze.assert_called_once_with(
            description="change auth",
            target_files=["src/prism/auth.py"],
        )

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_multiple_files(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """Passing multiple --file flags works correctly."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = _make_impact_report()
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app,
            [
                "impact", "change auth and router",
                "-f", "src/prism/auth.py",
                "-f", "src/prism/router.py",
            ],
        )
        assert result.exit_code == 0
        mock_analyzer.analyze.assert_called_once_with(
            description="change auth and router",
            target_files=["src/prism/auth.py", "src/prism/router.py"],
        )

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_files_shown_in_output(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """Target files are displayed in the output."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = _make_impact_report()
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app,
            [
                "impact", "auth change",
                "--file", "src/prism/auth.py",
            ],
        )
        assert result.exit_code == 0
        assert "src/prism/auth.py" in result.output


# ======================================================================
# Impact command — error handling
# ======================================================================


class TestImpactErrors:
    """Tests for impact command error handling."""

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_analyze_failure_shows_error(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """If analyze() raises, an error message is shown."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = RuntimeError("parse error")
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "some change"],
        )
        assert result.exit_code == 1
        assert "Impact analysis failed" in result.output

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_init_failure_shows_error(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """If BlastRadiusAnalyzer init fails, error is shown."""
        mock_analyzer_cls.side_effect = OSError("Permission denied")

        result = runner.invoke(
            app, ["impact", "some change"],
        )
        assert result.exit_code == 1
        assert "Error initializing" in result.output

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_no_affected_files_still_succeeds(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """Impact analysis with zero affected files doesn't crash."""
        mock_analyzer = MagicMock()
        report = _make_impact_report()
        report.affected_files = []
        report.missing_tests = []
        report.risk_score = 0
        mock_analyzer.analyze.return_value = report
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "trivial change"],
        )
        assert result.exit_code == 0
        assert "Blast Radius Report" in result.output


# ======================================================================
# Impact command — risk score styling
# ======================================================================


class TestImpactRiskStyling:
    """Tests for risk score display styling in impact command."""

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_high_risk_score(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """A risk score >= 70 is displayed (red styling applied)."""
        mock_analyzer = MagicMock()
        report = _make_impact_report()
        report.risk_score = 85
        mock_analyzer.analyze.return_value = report
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "dangerous change"],
        )
        assert result.exit_code == 0
        assert "85" in result.output

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_medium_risk_score(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """A risk score between 40-69 is displayed (yellow styling applied)."""
        mock_analyzer = MagicMock()
        report = _make_impact_report()
        report.risk_score = 50
        mock_analyzer.analyze.return_value = report
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "moderate change"],
        )
        assert result.exit_code == 0
        assert "50" in result.output

    @patch("prism.intelligence.blast_radius.BlastRadiusAnalyzer")
    def test_low_risk_score(
        self, mock_analyzer_cls: MagicMock,
    ) -> None:
        """A risk score below 40 is displayed (green styling applied)."""
        mock_analyzer = MagicMock()
        report = _make_impact_report()
        report.risk_score = 15
        report.affected_files = []
        report.missing_tests = []
        mock_analyzer.analyze.return_value = report
        mock_analyzer_cls.return_value = mock_analyzer

        result = runner.invoke(
            app, ["impact", "safe change"],
        )
        assert result.exit_code == 0
        assert "15" in result.output

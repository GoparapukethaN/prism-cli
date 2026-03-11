"""Tests for Phase 5 CLI commands — prism debate and prism why."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from prism.cli.app import app

runner = CliRunner()


# ======================================================================
# Helpers
# ======================================================================


def _make_debate_result() -> MagicMock:
    """Create a mock DebateResult with realistic fields."""
    result = MagicMock()
    result.question = "Should we use microservices?"
    result.consensus = "All agree on modularity"
    result.disagreements = "Complexity vs simplicity"
    result.tradeoffs = "Speed vs maintainability"
    result.recommendation = "Use microservices for large teams"
    result.confidence = 0.85
    result.blind_spots = {"model-a": "missed cost implications"}
    result.total_cost = 0.05
    result.created_at = "2025-06-15T10:00:00+00:00"

    round1 = MagicMock()
    round1.round_type = "position"
    round1.round_number = 1
    round1.positions = {
        "model-a": "Position A text",
        "model-b": "Position B text",
    }

    round3 = MagicMock()
    round3.round_type = "synthesis"
    round3.round_number = 3
    round3.positions = {"model-a": "Synthesis text"}

    result.rounds = [round1, round3]
    return result


def _make_archaeology_report() -> MagicMock:
    """Create a mock ArchaeologyReport with realistic fields."""
    report = MagicMock()
    report.target = "src/parser.py:42"
    report.primary_author = "Alice"
    report.stability_score = 0.85
    report.narrative = "## History of parser.py\n\nCreated by Alice"
    report.risks = ["Low risk: stable code"]
    report.created_at = "2025-06-15T12:00:00+00:00"
    report.author_distribution = {"Alice": 5, "Bob": 2}
    report.co_evolution = [("tests/test_parser.py", 0.67)]

    commit = MagicMock()
    commit.hash = "abc123def456789012345678901234567890abcd"
    commit.author = "Alice"
    commit.date = "2025-06-15T10:00:00+00:00"
    commit.subject = "feat: add parser"

    report.timeline = [commit]
    return report


# ======================================================================
# Debate command tests
# ======================================================================


class TestDebateCommand:
    """Tests for 'prism debate' command."""

    @patch("prism.intelligence.debate.debate")
    @patch("prism.intelligence.debate.DebateConfig")
    def test_debate_basic(
        self, mock_config_cls: MagicMock, mock_debate: MagicMock,
    ) -> None:
        """Basic debate command invocation succeeds."""
        mock_debate.return_value = _make_debate_result()
        result = runner.invoke(app, ["debate", "Is Python better?"])
        assert result.exit_code == 0

    @patch("prism.intelligence.debate.debate")
    @patch("prism.intelligence.debate.DebateConfig")
    def test_debate_quick_mode(
        self, mock_config_cls: MagicMock, mock_debate: MagicMock,
    ) -> None:
        """Quick mode flag is passed through."""
        mock_debate.return_value = _make_debate_result()
        result = runner.invoke(app, ["debate", "--quick", "Question?"])
        assert result.exit_code == 0
        # The config was created with quick_mode=True
        mock_config_cls.assert_called_once_with(quick_mode=True)

    @patch("prism.intelligence.debate.debate")
    @patch("prism.intelligence.debate.DebateConfig")
    def test_debate_custom_models(
        self, mock_config_cls: MagicMock, mock_debate: MagicMock,
    ) -> None:
        """Custom models are parsed from comma-separated string."""
        mock_debate.return_value = _make_debate_result()
        result = runner.invoke(
            app,
            ["debate", "--models", "gpt-4o,claude-3", "Question?"],
        )
        assert result.exit_code == 0

    @patch("prism.intelligence.debate.debate")
    @patch("prism.intelligence.debate.DebateConfig")
    def test_debate_displays_rounds(
        self, mock_config_cls: MagicMock, mock_debate: MagicMock,
    ) -> None:
        """Output includes round labels and model positions."""
        mock_debate.return_value = _make_debate_result()
        result = runner.invoke(app, ["debate", "Test question"])
        assert result.exit_code == 0
        assert "Round 1" in result.output

    @patch("prism.intelligence.debate.debate")
    @patch("prism.intelligence.debate.DebateConfig")
    def test_debate_displays_synthesis(
        self, mock_config_cls: MagicMock, mock_debate: MagicMock,
    ) -> None:
        """Output includes synthesis summary."""
        mock_debate.return_value = _make_debate_result()
        result = runner.invoke(app, ["debate", "Test question"])
        assert result.exit_code == 0
        assert "Synthesis" in result.output

    @patch("prism.intelligence.debate.debate")
    @patch("prism.intelligence.debate.DebateConfig")
    def test_debate_displays_cost(
        self, mock_config_cls: MagicMock, mock_debate: MagicMock,
    ) -> None:
        """Output includes cost summary."""
        mock_debate.return_value = _make_debate_result()
        result = runner.invoke(app, ["debate", "Cost test"])
        assert result.exit_code == 0
        assert "cost" in result.output.lower()

    @patch("prism.intelligence.debate.debate")
    @patch("prism.intelligence.debate.DebateConfig")
    def test_debate_value_error(
        self, mock_config_cls: MagicMock, mock_debate: MagicMock,
    ) -> None:
        """ValueError is displayed gracefully."""
        mock_debate.side_effect = ValueError("Question cannot be empty")
        result = runner.invoke(app, ["debate", "empty"])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    @patch("prism.intelligence.debate.debate")
    @patch("prism.intelligence.debate.DebateConfig")
    def test_debate_generic_error(
        self, mock_config_cls: MagicMock, mock_debate: MagicMock,
    ) -> None:
        """Generic exceptions are displayed as errors."""
        mock_debate.side_effect = RuntimeError("Network down")
        result = runner.invoke(app, ["debate", "Error test"])
        assert result.exit_code == 0
        assert "error" in result.output.lower()


# ======================================================================
# Why command tests
# ======================================================================


class TestWhyCommand:
    """Tests for 'prism why' command."""

    @patch("prism.intelligence.archaeologist.investigate")
    def test_why_basic(self, mock_investigate: MagicMock) -> None:
        """Basic why command invocation succeeds."""
        mock_investigate.return_value = _make_archaeology_report()
        result = runner.invoke(app, ["why", "src/parser.py:42"])
        assert result.exit_code == 0

    @patch("prism.intelligence.archaeologist.investigate")
    def test_why_displays_summary(self, mock_investigate: MagicMock) -> None:
        """Output includes summary panel."""
        mock_investigate.return_value = _make_archaeology_report()
        result = runner.invoke(app, ["why", "src/parser.py"])
        assert result.exit_code == 0
        assert "Code Archaeology" in result.output

    @patch("prism.intelligence.archaeologist.investigate")
    def test_why_displays_timeline(self, mock_investigate: MagicMock) -> None:
        """Output includes timeline table."""
        mock_investigate.return_value = _make_archaeology_report()
        result = runner.invoke(app, ["why", "src/parser.py"])
        assert result.exit_code == 0
        assert "Timeline" in result.output

    @patch("prism.intelligence.archaeologist.investigate")
    def test_why_displays_contributors(
        self, mock_investigate: MagicMock,
    ) -> None:
        """Output includes contributors table."""
        mock_investigate.return_value = _make_archaeology_report()
        result = runner.invoke(app, ["why", "src/parser.py"])
        assert result.exit_code == 0
        assert "Contributors" in result.output

    @patch("prism.intelligence.archaeologist.investigate")
    def test_why_with_module_option(
        self, mock_investigate: MagicMock,
    ) -> None:
        """Module option prefixes the target path."""
        mock_investigate.return_value = _make_archaeology_report()
        result = runner.invoke(
            app, ["why", "--module", "auth", "validator"],
        )
        assert result.exit_code == 0
        # Should have been called with a path containing "auth"
        call_args = mock_investigate.call_args
        target_val = (
            call_args.kwargs.get("target")
            or (call_args.args[0] if call_args.args else "")
        )
        assert "auth" in target_val

    @patch("prism.intelligence.archaeologist.investigate")
    def test_why_value_error(self, mock_investigate: MagicMock) -> None:
        """ValueError is displayed gracefully."""
        mock_investigate.side_effect = ValueError("Could not find target")
        result = runner.invoke(app, ["why", "nonexistent"])
        assert result.exit_code == 0
        assert "Could not find" in result.output

    @patch("prism.intelligence.archaeologist.investigate")
    def test_why_generic_error(self, mock_investigate: MagicMock) -> None:
        """Generic exceptions are displayed as errors."""
        mock_investigate.side_effect = RuntimeError("Git error")
        result = runner.invoke(app, ["why", "broken.py"])
        assert result.exit_code == 0
        assert "error" in result.output.lower()

    @patch("prism.intelligence.archaeologist.investigate")
    def test_why_with_root_option(self, mock_investigate: MagicMock) -> None:
        """Root option is passed through."""
        mock_investigate.return_value = _make_archaeology_report()
        result = runner.invoke(app, ["why", "--root", ".", "test.py"])
        assert result.exit_code == 0

    @patch("prism.intelligence.archaeologist.investigate")
    def test_why_displays_risks(self, mock_investigate: MagicMock) -> None:
        """Output includes risks panel."""
        mock_investigate.return_value = _make_archaeology_report()
        result = runner.invoke(app, ["why", "src/parser.py"])
        assert result.exit_code == 0
        assert "Risk" in result.output

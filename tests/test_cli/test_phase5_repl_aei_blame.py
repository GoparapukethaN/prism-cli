"""Tests for Phase 5 REPL commands: /aei and /blame.

Covers _cmd_aei (stats, reset, explain) and _cmd_blame (trace,
list, bisect flags) with comprehensive mocking of all external
dependencies.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

from rich.console import Console

from prism.cli.repl import (
    _dispatch_command,
    _SessionState,
)
from prism.config.schema import PrismConfig
from prism.config.settings import Settings

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _make_settings(tmp_path: Path) -> Settings:
    """Create a minimal Settings pointing at *tmp_path*."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(
        config=config, project_root=tmp_path,
    )
    settings.ensure_directories()
    return settings


def _make_console(width: int = 300) -> Console:
    """In-memory console for capturing output."""
    buf = io.StringIO()
    return Console(
        file=buf,
        force_terminal=False,
        no_color=True,
        width=width,
    )


def _get_output(console: Console) -> str:
    """Extract text from an in-memory console."""
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


def _make_state(
    pinned_model: str | None = None,
) -> _SessionState:
    """Create a SessionState with sane defaults."""
    state = _SessionState(pinned_model=pinned_model)
    state.session_id = "test-session"
    return state


def _cmd(
    command: str,
    tmp_path: Path,
    console: Console | None = None,
    settings: Settings | None = None,
    state: _SessionState | None = None,
) -> tuple[str, str, Console, Settings, _SessionState]:
    """Run a slash command, return (action, output, ...)."""
    con = console or _make_console()
    stg = settings or _make_settings(tmp_path)
    st = state or _make_state()
    action = _dispatch_command(
        command,
        console=con,
        settings=stg,
        state=st,
        dry_run=False,
        offline=False,
    )
    return action, _get_output(con), con, stg, st


# -----------------------------------------------------------
# Mock data classes for AEI
# -----------------------------------------------------------

@dataclass
class _MockAEIStats:
    total_attempts: int = 10
    total_successes: int = 7
    total_failures: int = 3
    success_rate: float = 0.7
    strategies_used: dict[str, int] = field(
        default_factory=lambda: {
            "regex_patch": 5,
            "ast_diff": 3,
            "full_rewrite": 2,
        }
    )
    escalation_count: int = 1
    top_error_types: list[tuple[str, int]] = field(
        default_factory=lambda: [
            ("TypeError", 4),
            ("KeyError", 3),
        ]
    )


@dataclass
class _MockBlameReport:
    bug_description: str = "Login fails with 500 error"
    breaking_commit: str = "abc12345def67890"
    breaking_author: str = "dev@example.com"
    breaking_date: str = "2026-01-15T10:30:00+00:00"
    breaking_message: str = "refactor: change auth middleware"
    affected_files: list[str] = field(
        default_factory=lambda: ["auth.py", "middleware.py"]
    )
    affected_lines: list[str] = field(default_factory=list)
    causal_narrative: str = "The breaking change was in commit abc12345."
    confidence: float = 0.7
    proposed_fix: str = ""
    related_tests: list[str] = field(
        default_factory=lambda: ["tests/test_auth.py"]
    )
    bisect_steps: int = 0
    created_at: str = "2026-01-15T12:00:00+00:00"


# ===========================================================
# TestAEICommand
# ===========================================================


class TestAEICommand:
    """Tests for the /aei REPL command."""

    def test_aei_no_args_shows_stats(
        self, tmp_path: Path,
    ) -> None:
        """``/aei`` with no args shows statistics panel."""
        mock_aei = MagicMock()
        mock_aei.get_stats.return_value = _MockAEIStats()

        with patch(
            "prism.intelligence.aei.AdaptiveExecutionIntelligence",
            return_value=mock_aei,
        ):
            action, output, *_ = _cmd("/aei", tmp_path)

        assert action == "continue"
        assert "AEI Statistics" in output
        assert "10" in output  # total_attempts
        assert "regex_patch" in output

    def test_aei_stats_subcommand(
        self, tmp_path: Path,
    ) -> None:
        """``/aei stats`` shows statistics panel."""
        mock_aei = MagicMock()
        mock_aei.get_stats.return_value = _MockAEIStats()

        with patch(
            "prism.intelligence.aei.AdaptiveExecutionIntelligence",
            return_value=mock_aei,
        ):
            action, output, *_ = _cmd("/aei stats", tmp_path)

        assert action == "continue"
        assert "AEI Statistics" in output
        assert "Success rate" in output

    def test_aei_reset(
        self, tmp_path: Path,
    ) -> None:
        """``/aei reset`` clears history and shows confirmation."""
        mock_aei = MagicMock()
        mock_aei.reset.return_value = 15

        with patch(
            "prism.intelligence.aei.AdaptiveExecutionIntelligence",
            return_value=mock_aei,
        ):
            action, output, *_ = _cmd("/aei reset", tmp_path)

        assert action == "continue"
        assert "15" in output
        assert "cleared" in output.lower()
        mock_aei.reset.assert_called_once()
        mock_aei.close.assert_called_once()

    def test_aei_explain_no_hash(
        self, tmp_path: Path,
    ) -> None:
        """``/aei explain`` without a hash shows usage."""
        mock_aei = MagicMock()

        with patch(
            "prism.intelligence.aei.AdaptiveExecutionIntelligence",
            return_value=mock_aei,
        ):
            action, output, *_ = _cmd("/aei explain", tmp_path)

        assert action == "continue"
        assert "Usage" in output

    def test_aei_explain_with_hash(
        self, tmp_path: Path,
    ) -> None:
        """``/aei explain hash123`` shows explanation panel."""
        mock_aei = MagicMock()
        mock_aei.explain.return_value = (
            "Error fingerprint: hash123\n"
            "Past attempts: 5\n"
            "Current recommendation: ast_diff\n"
            "Confidence: 60% (moderate)"
        )

        with patch(
            "prism.intelligence.aei.AdaptiveExecutionIntelligence",
            return_value=mock_aei,
        ):
            action, output, *_ = _cmd(
                "/aei explain hash123", tmp_path,
            )

        assert action == "continue"
        assert "AEI Explanation" in output
        assert "hash123" in output
        mock_aei.explain.assert_called_once()
        mock_aei.close.assert_called_once()

    def test_aei_stats_empty_db(
        self, tmp_path: Path,
    ) -> None:
        """``/aei`` handles an empty database gracefully."""
        mock_aei = MagicMock()
        mock_aei.get_stats.return_value = _MockAEIStats(
            total_attempts=0,
            total_successes=0,
            total_failures=0,
            success_rate=0.0,
            strategies_used={},
            escalation_count=0,
            top_error_types=[],
        )

        with patch(
            "prism.intelligence.aei.AdaptiveExecutionIntelligence",
            return_value=mock_aei,
        ):
            action, output, *_ = _cmd("/aei", tmp_path)

        assert action == "continue"
        assert "AEI Statistics" in output
        assert "0" in output

    def test_aei_handles_exception(
        self, tmp_path: Path,
    ) -> None:
        """``/aei`` handles errors gracefully."""
        with patch(
            "prism.intelligence.aei.AdaptiveExecutionIntelligence",
            side_effect=RuntimeError("DB locked"),
        ):
            action, output, *_ = _cmd("/aei", tmp_path)

        assert action == "continue"
        assert "AEI error" in output

    def test_aei_stats_shows_strategies_used(
        self, tmp_path: Path,
    ) -> None:
        """``/aei`` displays strategy usage breakdown."""
        mock_aei = MagicMock()
        mock_aei.get_stats.return_value = _MockAEIStats(
            strategies_used={
                "regex_patch": 10,
                "ast_diff": 5,
                "add_defensive_code": 2,
            },
        )

        with patch(
            "prism.intelligence.aei.AdaptiveExecutionIntelligence",
            return_value=mock_aei,
        ):
            _action, output, *_ = _cmd("/aei", tmp_path)

        assert "regex_patch" in output
        assert "ast_diff" in output
        assert "add_defensive_code" in output

    def test_aei_stats_shows_error_types(
        self, tmp_path: Path,
    ) -> None:
        """``/aei`` displays top error types."""
        mock_aei = MagicMock()
        mock_aei.get_stats.return_value = _MockAEIStats(
            top_error_types=[
                ("ValueError", 12),
                ("KeyError", 8),
            ],
        )

        with patch(
            "prism.intelligence.aei.AdaptiveExecutionIntelligence",
            return_value=mock_aei,
        ):
            _action, output, *_ = _cmd("/aei", tmp_path)

        assert "ValueError" in output
        assert "KeyError" in output


# ===========================================================
# TestBlameCommand
# ===========================================================


class TestBlameCommand:
    """Tests for the /blame REPL command."""

    def test_blame_no_args_shows_usage(
        self, tmp_path: Path,
    ) -> None:
        """``/blame`` with no args shows usage message."""
        action, output, *_ = _cmd("/blame", tmp_path)
        assert action == "continue"
        assert "Usage" in output

    def test_blame_with_description_runs_trace(
        self, tmp_path: Path,
    ) -> None:
        """``/blame login bug`` runs a trace and displays the report."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _MockBlameReport()

        with patch(
            "prism.intelligence.blame.CausalBlameTracer",
            return_value=mock_tracer,
        ):
            action, output, *_ = _cmd(
                "/blame login fails with 500 error", tmp_path,
            )

        assert action == "continue"
        assert "Blame Report" in output
        assert "abc12345" in output  # breaking_commit prefix
        assert "dev@example.com" in output
        mock_tracer.trace.assert_called_once()

    def test_blame_list_no_reports(
        self, tmp_path: Path,
    ) -> None:
        """``/blame list`` with no reports shows empty message."""
        mock_tracer = MagicMock()
        mock_tracer.list_reports.return_value = []

        with patch(
            "prism.intelligence.blame.CausalBlameTracer",
            return_value=mock_tracer,
        ):
            action, output, *_ = _cmd("/blame list", tmp_path)

        assert action == "continue"
        assert "No blame reports" in output

    def test_blame_list_with_reports(
        self, tmp_path: Path,
    ) -> None:
        """``/blame list`` shows a table of saved reports."""
        mock_tracer = MagicMock()
        mock_report = _MockBlameReport()
        mock_tracer.list_reports.return_value = [
            Path("/fake/blame_abc12345_2026-01-15.json"),
        ]
        mock_tracer.load_report.return_value = mock_report

        with patch(
            "prism.intelligence.blame.CausalBlameTracer",
            return_value=mock_tracer,
        ):
            action, output, *_ = _cmd("/blame list", tmp_path)

        assert action == "continue"
        assert "Blame Reports" in output
        assert "abc12345" in output

    def test_blame_with_bisect_flags(
        self, tmp_path: Path,
    ) -> None:
        """``/blame --test ... --good ... desc`` passes correct args."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _MockBlameReport(
            bisect_steps=5,
            confidence=0.7,
        )

        with patch(
            "prism.intelligence.blame.CausalBlameTracer",
            return_value=mock_tracer,
        ):
            action, output, *_ = _cmd(
                '/blame --test "pytest tests/" --good abc123 '
                "login breaks after refactor",
                tmp_path,
            )

        assert action == "continue"
        assert "Blame Report" in output
        # Verify the trace was called with test_command and good_commit
        call_kwargs = mock_tracer.trace.call_args
        assert call_kwargs.kwargs["test_command"] == "pytest tests/"
        assert call_kwargs.kwargs["good_commit"] == "abc123"

    def test_blame_displays_affected_files(
        self, tmp_path: Path,
    ) -> None:
        """The blame report shows affected files."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _MockBlameReport(
            affected_files=["auth.py", "middleware.py"],
        )

        with patch(
            "prism.intelligence.blame.CausalBlameTracer",
            return_value=mock_tracer,
        ):
            _action, output, *_ = _cmd(
                "/blame auth breaks", tmp_path,
            )

        assert "auth.py" in output
        assert "middleware.py" in output

    def test_blame_displays_causal_narrative(
        self, tmp_path: Path,
    ) -> None:
        """The blame report shows the causal narrative."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _MockBlameReport(
            causal_narrative="The middleware change broke login.",
        )

        with patch(
            "prism.intelligence.blame.CausalBlameTracer",
            return_value=mock_tracer,
        ):
            _action, output, *_ = _cmd(
                "/blame login bug", tmp_path,
            )

        assert "Causal Narrative" in output
        assert "middleware change broke login" in output

    def test_blame_displays_confidence(
        self, tmp_path: Path,
    ) -> None:
        """The blame report shows the confidence level."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _MockBlameReport(
            confidence=0.7,
        )

        with patch(
            "prism.intelligence.blame.CausalBlameTracer",
            return_value=mock_tracer,
        ):
            _action, output, *_ = _cmd(
                "/blame login bug", tmp_path,
            )

        assert "70%" in output

    def test_blame_displays_related_tests(
        self, tmp_path: Path,
    ) -> None:
        """The blame report shows related tests."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _MockBlameReport(
            related_tests=["tests/test_auth.py"],
        )

        with patch(
            "prism.intelligence.blame.CausalBlameTracer",
            return_value=mock_tracer,
        ):
            _action, output, *_ = _cmd(
                "/blame auth issue", tmp_path,
            )

        assert "test_auth.py" in output

    def test_blame_handles_exception(
        self, tmp_path: Path,
    ) -> None:
        """``/blame`` handles errors gracefully."""
        with patch(
            "prism.intelligence.blame.CausalBlameTracer",
            side_effect=RuntimeError("git not found"),
        ):
            action, output, *_ = _cmd(
                "/blame some bug", tmp_path,
            )

        assert action == "continue"
        assert "Blame error" in output

    def test_blame_bisect_steps_displayed(
        self, tmp_path: Path,
    ) -> None:
        """When bisect was used, the step count is shown."""
        mock_tracer = MagicMock()
        mock_tracer.trace.return_value = _MockBlameReport(
            bisect_steps=7,
        )

        with patch(
            "prism.intelligence.blame.CausalBlameTracer",
            return_value=mock_tracer,
        ):
            _action, output, *_ = _cmd(
                "/blame regression in parser", tmp_path,
            )

        assert "Bisect steps" in output
        assert "7" in output

"""Tests for the ``/context`` REPL command and ``prism context`` CLI command.

Covers:
- /context show display format
- /context add/drop functionality
- /context stats output
- prism context CLI command
- Error handling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from prism.cli.app import app
from prism.intelligence.context_budget import (
    BudgetAllocation,
    ContextItem,
    EfficiencyStats,
    RelevanceLevel,
    SmartContextBudgetManager,
)

runner = CliRunner()


# ======================================================================
# Helpers
# ======================================================================


def _make_allocation(
    tokens_used: int = 2341,
    items_included: list[ContextItem] | None = None,
    items_excluded: list[ContextItem] | None = None,
) -> BudgetAllocation:
    """Create a BudgetAllocation for testing."""
    return BudgetAllocation(
        total_tokens=128_000,
        response_tokens=51_200,
        system_tokens=12_800,
        context_tokens=64_000,
        items_included=items_included or [
            ContextItem(
                path="src/auth/token.py",
                relevance=1.0,
                token_count=2341,
                reason="directly mentioned",
                level=RelevanceLevel.DIRECT,
            ),
        ],
        items_excluded=items_excluded or [],
        tokens_used=tokens_used,
        tokens_remaining=64_000 - tokens_used,
        efficiency_pct=(tokens_used / 64_000) * 100,
    )


def _make_stats(
    total_records: int = 10,
    avg_tokens: float = 5000.0,
    success_rate: float = 0.9,
) -> EfficiencyStats:
    """Create EfficiencyStats for testing."""
    return EfficiencyStats(
        avg_tokens_used=avg_tokens,
        avg_efficiency_pct=75.0,
        success_rate=success_rate,
        total_records=total_records,
        total_tokens_saved=50_000,
    )


class _FakeConsole:
    """Minimal fake console that captures print output."""

    def __init__(self) -> None:
        self.output: list[str] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        for a in args:
            self.output.append(str(a))


class _FakeSettings:
    """Minimal fake settings."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root
        self.db_path = project_root / "prism.db"

    @property
    def project_root(self) -> Path:
        return self._project_root

    def ensure_directories(self) -> None:
        pass


class _FakeState:
    """Minimal fake session state."""

    def __init__(self) -> None:
        self.active_files: list[str] = ["src/auth.py", "src/router.py"]
        self.conversation: list[dict[str, str]] = []
        self.pinned_model: str | None = None


# ======================================================================
# REPL /context command tests
# ======================================================================


class TestReplContextShow:
    """/context show display format tests."""

    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    def test_context_show_default(
        self, mock_mgr_cls: MagicMock,
    ) -> None:
        """'/context' with no args should show current context."""
        from prism.cli.repl import _cmd_context

        mock_mgr = MagicMock()
        mock_mgr.allocate.return_value = _make_allocation()
        mock_mgr_cls.return_value = mock_mgr
        mock_mgr_cls.generate_context_display = (
            SmartContextBudgetManager.generate_context_display
        )

        console = _FakeConsole()
        settings = _FakeSettings(Path("/tmp/test"))
        state = _FakeState()

        result = _cmd_context(
            args="",
            console=console,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
        )
        assert result == "continue"

    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    def test_context_show_explicit(
        self, mock_mgr_cls: MagicMock,
    ) -> None:
        """'/context show' should show current context."""
        from prism.cli.repl import _cmd_context

        mock_mgr = MagicMock()
        mock_mgr.allocate.return_value = _make_allocation()
        mock_mgr_cls.return_value = mock_mgr
        mock_mgr_cls.generate_context_display = (
            SmartContextBudgetManager.generate_context_display
        )

        console = _FakeConsole()
        settings = _FakeSettings(Path("/tmp/test"))
        state = _FakeState()

        result = _cmd_context(
            args="show",
            console=console,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
        )
        assert result == "continue"


class TestReplContextAdd:
    """/context add functionality tests."""

    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    def test_add_file(
        self, mock_mgr_cls: MagicMock,
    ) -> None:
        """'/context add src/utils.py' should force-include the file."""
        from prism.cli.repl import _cmd_context

        mock_mgr = MagicMock()
        mock_mgr_cls.return_value = mock_mgr

        console = _FakeConsole()
        settings = _FakeSettings(Path("/tmp/test"))
        state = _FakeState()

        result = _cmd_context(
            args="add src/utils.py",
            console=console,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
        )
        assert result == "continue"
        mock_mgr.add_file.assert_called_once_with("src/utils.py")
        assert "src/utils.py" in state.active_files

    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    def test_add_no_file_shows_usage(
        self, mock_mgr_cls: MagicMock,
    ) -> None:
        """'/context add' without a file should show usage."""
        from prism.cli.repl import _cmd_context

        mock_mgr = MagicMock()
        mock_mgr_cls.return_value = mock_mgr

        console = _FakeConsole()
        settings = _FakeSettings(Path("/tmp/test"))
        state = _FakeState()

        result = _cmd_context(
            args="add",
            console=console,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
        )
        assert result == "continue"
        assert any("Usage" in o for o in console.output)

    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    def test_add_duplicate_file_not_duplicated(
        self, mock_mgr_cls: MagicMock,
    ) -> None:
        """Adding an already-active file should not duplicate it."""
        from prism.cli.repl import _cmd_context

        mock_mgr = MagicMock()
        mock_mgr_cls.return_value = mock_mgr

        console = _FakeConsole()
        settings = _FakeSettings(Path("/tmp/test"))
        state = _FakeState()
        state.active_files = ["src/auth.py"]

        _cmd_context(
            args="add src/auth.py",
            console=console,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
        )
        assert state.active_files.count("src/auth.py") == 1


class TestReplContextDrop:
    """/context drop functionality tests."""

    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    def test_drop_file(
        self, mock_mgr_cls: MagicMock,
    ) -> None:
        """'/context drop src/auth.py' should exclude the file."""
        from prism.cli.repl import _cmd_context

        mock_mgr = MagicMock()
        mock_mgr_cls.return_value = mock_mgr

        console = _FakeConsole()
        settings = _FakeSettings(Path("/tmp/test"))
        state = _FakeState()

        result = _cmd_context(
            args="drop src/auth.py",
            console=console,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
        )
        assert result == "continue"
        mock_mgr.drop_file.assert_called_once_with("src/auth.py")
        assert "src/auth.py" not in state.active_files

    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    def test_drop_no_file_shows_usage(
        self, mock_mgr_cls: MagicMock,
    ) -> None:
        """'/context drop' without a file should show usage."""
        from prism.cli.repl import _cmd_context

        mock_mgr = MagicMock()
        mock_mgr_cls.return_value = mock_mgr

        console = _FakeConsole()
        settings = _FakeSettings(Path("/tmp/test"))
        state = _FakeState()

        result = _cmd_context(
            args="drop",
            console=console,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
        )
        assert result == "continue"
        assert any("Usage" in o for o in console.output)


class TestReplContextStats:
    """/context stats output tests."""

    @patch("prism.db.database.Database")
    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    def test_stats_with_data(
        self,
        mock_mgr_cls: MagicMock,
        mock_db_cls: MagicMock,
    ) -> None:
        """'/context stats' should display efficiency metrics."""
        from prism.cli.repl import _cmd_context

        mock_mgr = MagicMock()
        mock_mgr.get_efficiency_stats.return_value = _make_stats()
        mock_mgr_cls.return_value = mock_mgr

        console = _FakeConsole()
        settings = _FakeSettings(Path("/tmp/test"))
        state = _FakeState()

        result = _cmd_context(
            args="stats",
            console=console,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
        )
        assert result == "continue"


class TestReplContextErrors:
    """Error handling tests."""

    def test_context_exception_handled(self) -> None:
        """Exceptions in /context should be caught and displayed."""
        from prism.cli.repl import _cmd_context

        console = _FakeConsole()
        # Settings with non-existent path — should not crash
        settings = _FakeSettings(Path("/nonexistent/path"))
        state = _FakeState()

        result = _cmd_context(
            args="show",
            console=console,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            state=state,  # type: ignore[arg-type]
        )
        assert result == "continue"


# ======================================================================
# CLI: prism context
# ======================================================================


class TestContextCommand:
    """Tests for the ``prism context`` CLI command."""

    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    @patch("prism.config.settings.load_settings")
    def test_context_show_command(
        self,
        mock_settings: MagicMock,
        mock_mgr_cls: MagicMock,
    ) -> None:
        """'prism context' should invoke the show action."""
        mock_s = MagicMock()
        mock_s.project_root = Path("/tmp/test")
        mock_s.db_path = Path("/tmp/test/prism.db")
        mock_settings.return_value = mock_s

        # Make rglob return empty to avoid filesystem issues
        with patch.object(Path, "rglob", return_value=[]):
            mock_mgr = MagicMock()
            mock_mgr.allocate.return_value = _make_allocation()
            mock_mgr_cls.return_value = mock_mgr
            mock_mgr_cls.generate_context_display = (
                SmartContextBudgetManager.generate_context_display
            )

            result = runner.invoke(app, ["context", "show"])
            assert result.exit_code == 0

    @patch("prism.db.database.Database")
    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    @patch("prism.config.settings.load_settings")
    def test_context_stats_command(
        self,
        mock_settings: MagicMock,
        mock_mgr_cls: MagicMock,
        mock_db_cls: MagicMock,
    ) -> None:
        """'prism context stats' should show efficiency stats."""
        mock_s = MagicMock()
        mock_s.project_root = Path("/tmp/test")
        mock_s.db_path = Path("/tmp/test/prism.db")
        mock_settings.return_value = mock_s

        mock_mgr = MagicMock()
        mock_mgr.get_efficiency_stats.return_value = _make_stats()
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(app, ["context", "stats"])
        assert result.exit_code == 0

    @patch("prism.config.settings.load_settings")
    def test_context_settings_error(
        self,
        mock_settings: MagicMock,
    ) -> None:
        """'prism context' should handle settings errors gracefully."""
        mock_settings.side_effect = RuntimeError("Settings failed")

        result = runner.invoke(app, ["context", "show"])
        assert result.exit_code == 1

    @patch("prism.db.database.Database")
    @patch("prism.intelligence.context_budget.SmartContextBudgetManager")
    @patch("prism.config.settings.load_settings")
    def test_context_stats_db_error(
        self,
        mock_settings: MagicMock,
        mock_mgr_cls: MagicMock,
        mock_db_cls: MagicMock,
    ) -> None:
        """'prism context stats' should handle DB errors."""
        mock_s = MagicMock()
        mock_s.project_root = Path("/tmp/test")
        mock_s.db_path = Path("/tmp/test/prism.db")
        mock_settings.return_value = mock_s

        mock_mgr = MagicMock()
        mock_mgr.get_efficiency_stats.side_effect = RuntimeError("DB error")
        mock_mgr_cls.return_value = mock_mgr

        result = runner.invoke(app, ["context", "stats"])
        assert result.exit_code == 1


# ======================================================================
# Test generate_context_display integration
# ======================================================================


class TestDisplayIntegration:
    """Integration tests for display formatting."""

    def test_display_with_included_and_excluded(self) -> None:
        """Display should render both included and excluded items."""
        included = [
            ContextItem(
                path="src/auth.py",
                relevance=1.0,
                token_count=2341,
                reason="directly mentioned",
                level=RelevanceLevel.DIRECT,
            ),
            ContextItem(
                path="src/middleware.py",
                relevance=0.85,
                token_count=1876,
                reason="calls validate_token",
                level=RelevanceLevel.RELATED,
            ),
        ]
        excluded = [
            ContextItem(
                path="src/utils/jwt.py",
                relevance=0.6,
                token_count=4200,
                reason="too large",
                level=RelevanceLevel.INDIRECT,
            ),
        ]

        alloc = _make_allocation(
            tokens_used=4217,
            items_included=included,
            items_excluded=excluded,
        )

        display = SmartContextBudgetManager.generate_context_display(alloc)

        assert "score 1.00" in display
        assert "score 0.85" in display
        assert "src/auth.py" in display
        assert "EXCLUDED" in display
        assert "src/utils/jwt.py" in display

    def test_display_token_count_formatting(self) -> None:
        """Display should format token counts with commas."""
        item = ContextItem(
            path="src/big_file.py",
            relevance=1.0,
            token_count=12_450,
            reason="directly mentioned",
            level=RelevanceLevel.DIRECT,
        )
        alloc = _make_allocation(
            tokens_used=12_450,
            items_included=[item],
        )
        display = SmartContextBudgetManager.generate_context_display(alloc)
        assert "12,450 tokens" in display

    def test_display_efficiency_percentage(self) -> None:
        """Display should show the efficiency percentage in the header."""
        alloc = BudgetAllocation(
            total_tokens=128_000,
            response_tokens=51_200,
            system_tokens=12_800,
            context_tokens=64_000,
            items_included=[],
            items_excluded=[],
            tokens_used=32_000,
            tokens_remaining=32_000,
            efficiency_pct=50.0,
        )
        display = SmartContextBudgetManager.generate_context_display(alloc)
        assert "50% of budget" in display

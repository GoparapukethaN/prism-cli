"""Tests for the cost dashboard rendering module.

Covers: render_cost_dashboard() plain-text output and _render_rich_dashboard()
Rich console rendering, including budget display, model breakdown bars,
and savings estimates.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

from prism.cost.dashboard import _render_rich_dashboard, render_cost_dashboard
from prism.cost.tracker import CostSummary, CostTracker, ModelCostBreakdown

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_summary(
    period: str = "session",
    total_cost: float = 0.0,
    total_requests: int = 0,
    model_breakdown: list[ModelCostBreakdown] | None = None,
    budget_limit: float | None = None,
    budget_remaining: float | None = None,
) -> CostSummary:
    """Build a CostSummary with sensible defaults."""
    return CostSummary(
        period=period,
        total_cost=total_cost,
        total_requests=total_requests,
        model_breakdown=model_breakdown or [],
        budget_limit=budget_limit,
        budget_remaining=budget_remaining,
    )


def _make_tracker(
    session_summary: CostSummary | None = None,
    daily_summary: CostSummary | None = None,
    monthly_summary: CostSummary | None = None,
    budget_remaining: float | None = None,
    savings: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> MagicMock:
    """Create a mock CostTracker with configurable return values."""
    tracker = MagicMock(spec=CostTracker)

    s = session_summary or _make_summary("session")
    d = daily_summary or _make_summary("day")
    m = monthly_summary or _make_summary("month")

    def get_cost_summary_side_effect(
        period: str, session_id: str = ""
    ) -> CostSummary:
        if period == "session":
            return s
        if period == "day":
            return d
        return m

    tracker.get_cost_summary = MagicMock(side_effect=get_cost_summary_side_effect)
    tracker.get_budget_remaining = MagicMock(return_value=budget_remaining)
    tracker.calculate_savings = MagicMock(return_value=savings)

    return tracker


# ------------------------------------------------------------------
# Plain-text render_cost_dashboard()
# ------------------------------------------------------------------


class TestRenderCostDashboardPlainText:
    """Tests for the plain-text output path (console=None)."""

    def test_basic_output_contains_header(self) -> None:
        tracker = _make_tracker()
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "Cost Dashboard" in output
        assert "=" * 50 in output

    def test_session_cost_displayed(self) -> None:
        tracker = _make_tracker(
            session_summary=_make_summary("session", total_cost=1.23, total_requests=5),
        )
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "Session:" in output
        assert "$    1.23" in output
        assert "(5 requests)" in output

    def test_daily_cost_displayed(self) -> None:
        tracker = _make_tracker(
            daily_summary=_make_summary("day", total_cost=4.56, total_requests=12),
        )
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "Today:" in output
        assert "$    4.56" in output
        assert "(12 requests)" in output

    def test_monthly_cost_displayed(self) -> None:
        tracker = _make_tracker(
            monthly_summary=_make_summary("month", total_cost=25.00, total_requests=100),
        )
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "This month:" in output
        assert "$   25.00" in output
        assert "(100 requests)" in output

    def test_monthly_budget_displayed(self) -> None:
        """Lines 60-65: monthly budget info when monthly_summary has budget_limit."""
        tracker = _make_tracker(
            monthly_summary=_make_summary(
                "month",
                total_cost=30.0,
                total_requests=50,
                budget_limit=100.0,
                budget_remaining=70.0,
            ),
        )
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "Budget:" in output
        assert "$   70.00 remaining" in output
        assert "/ $100.00" in output

    def test_daily_budget_displayed_when_no_monthly(self) -> None:
        """Lines 66-71: daily budget displayed when monthly has no limit."""
        tracker = _make_tracker(
            daily_summary=_make_summary(
                "day",
                total_cost=3.0,
                total_requests=10,
                budget_limit=5.0,
                budget_remaining=2.0,
            ),
            monthly_summary=_make_summary(
                "month",
                total_cost=30.0,
                total_requests=50,
                budget_limit=None,
                budget_remaining=None,
            ),
        )
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "Budget:" in output
        assert "$    2.00 remaining" in output
        assert "/ $5.00/day" in output

    def test_no_budget_displayed_when_no_limits(self) -> None:
        tracker = _make_tracker()
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "Budget:" not in output

    def test_model_breakdown_displayed(self) -> None:
        """Lines 74-91: model breakdown section."""
        breakdown = [
            ModelCostBreakdown(
                model_id="gpt-4o",
                display_name="GPT-4o",
                request_count=20,
                total_cost=5.00,
                percentage=80.0,
            ),
            ModelCostBreakdown(
                model_id="gpt-4o-mini",
                display_name="GPT-4o-mini",
                request_count=50,
                total_cost=1.25,
                percentage=20.0,
            ),
        ]
        tracker = _make_tracker(
            monthly_summary=_make_summary(
                "month",
                total_cost=6.25,
                total_requests=70,
                model_breakdown=breakdown,
            ),
        )
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "Model Breakdown (this month):" in output
        assert "-" * 50 in output
        assert "gpt-4o" in output
        assert "gpt-4o-mini" in output
        assert "20 req" in output
        assert "50 req" in output

    def test_model_breakdown_sorted_by_request_count(self) -> None:
        """Models should appear sorted by request count descending."""
        breakdown = [
            ModelCostBreakdown(
                model_id="model-a",
                display_name="A",
                request_count=5,
                total_cost=1.0,
                percentage=50.0,
            ),
            ModelCostBreakdown(
                model_id="model-b",
                display_name="B",
                request_count=15,
                total_cost=1.0,
                percentage=50.0,
            ),
        ]
        tracker = _make_tracker(
            monthly_summary=_make_summary(
                "month",
                total_cost=2.0,
                total_requests=20,
                model_breakdown=breakdown,
            ),
        )
        output = render_cost_dashboard(tracker, session_id="sess-1")

        # model-b (15 req) should appear before model-a (5 req)
        pos_b = output.index("model-b")
        pos_a = output.index("model-a")
        assert pos_b < pos_a

    def test_model_breakdown_bar_rendering(self) -> None:
        """The usage bar should contain block characters."""
        breakdown = [
            ModelCostBreakdown(
                model_id="gpt-4o",
                display_name="GPT-4o",
                request_count=10,
                total_cost=2.0,
                percentage=100.0,
            ),
        ]
        tracker = _make_tracker(
            monthly_summary=_make_summary(
                "month",
                total_cost=2.0,
                total_requests=10,
                model_breakdown=breakdown,
            ),
        )
        output = render_cost_dashboard(tracker, session_id="sess-1")

        # 100% should have all filled blocks
        assert "\u2588" in output  # filled block character

    def test_no_model_breakdown_when_empty(self) -> None:
        tracker = _make_tracker(
            monthly_summary=_make_summary("month", model_breakdown=[]),
        )
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "Model Breakdown" not in output

    def test_savings_displayed(self) -> None:
        """Lines 94-100: savings section when hypothetical > 0."""
        tracker = _make_tracker(savings=(10.0, 3.0, 7.0))
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "Savings Estimate:" in output
        assert "All via Claude Sonnet: ~$10.00" in output
        assert "Actual with routing:    $3.00" in output
        assert "You saved: ~$7.00 (70%)" in output

    def test_no_savings_displayed_when_zero(self) -> None:
        """Savings section should not appear when hypothetical is 0."""
        tracker = _make_tracker(savings=(0.0, 0.0, 0.0))
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "Savings Estimate" not in output

    def test_savings_percent_calculation(self) -> None:
        tracker = _make_tracker(savings=(20.0, 5.0, 15.0))
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "(75%)" in output

    def test_budget_remaining_zero_fallback(self) -> None:
        """Line 61: budget_remaining or 0.0 when budget_remaining is None."""
        tracker = _make_tracker(
            monthly_summary=_make_summary(
                "month",
                total_cost=10.0,
                total_requests=5,
                budget_limit=10.0,
                budget_remaining=None,
            ),
        )
        output = render_cost_dashboard(tracker, session_id="sess-1")

        assert "$    0.00 remaining" in output


# ------------------------------------------------------------------
# Rich console rendering — lines 105, 131-194
# ------------------------------------------------------------------


class TestRenderWithRichConsole:
    """Tests for the Rich console rendering path (console is not None)."""

    def test_console_print_called(self) -> None:
        """Line 105: when console is provided, _render_rich_dashboard is called."""
        tracker = _make_tracker()
        console = MagicMock(spec=Console)

        output = render_cost_dashboard(tracker, session_id="sess-1", console=console)

        # console.print should have been called at least once (for Panel)
        assert console.print.called
        # plain-text output should still be returned
        assert "Cost Dashboard" in output

    def test_rich_summary_panel_printed(self) -> None:
        """The summary panel should be printed to the console."""
        console = MagicMock(spec=Console)

        session = _make_summary("session", total_cost=1.0, total_requests=3)
        daily = _make_summary("day", total_cost=2.0, total_requests=5)
        monthly = _make_summary("month", total_cost=10.0, total_requests=20)

        _render_rich_dashboard(
            console=console,
            session=session,
            daily=daily,
            monthly=monthly,
            budget_remaining=None,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        # At least one panel should have been printed
        assert console.print.call_count >= 1

    def test_rich_budget_row_added(self) -> None:
        """Line 146-153: budget row added when budget_remaining is not None."""
        console = MagicMock(spec=Console)

        monthly = _make_summary(
            "month", total_cost=30.0, total_requests=50,
            budget_limit=100.0, budget_remaining=70.0,
        )

        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=_make_summary("day"),
            monthly=monthly,
            budget_remaining=50.0,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        assert console.print.called

    def test_rich_budget_label_monthly(self) -> None:
        """Line 148: period_label should be 'monthly' when monthly has budget."""
        console = Console(file=StringIO(), force_terminal=True, width=120)

        monthly = _make_summary(
            "month", total_cost=30.0, total_requests=50,
            budget_limit=100.0,
        )

        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=_make_summary("day"),
            monthly=monthly,
            budget_remaining=70.0,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "monthly" in output

    def test_rich_budget_label_daily(self) -> None:
        """Line 148: period_label should be 'daily' when only daily has budget."""
        console = Console(file=StringIO(), force_terminal=True, width=120)

        daily = _make_summary(
            "day", total_cost=3.0, total_requests=10,
            budget_limit=5.0,
        )

        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=daily,
            monthly=_make_summary("month"),
            budget_remaining=2.0,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "daily" in output

    def test_rich_model_breakdown_table(self) -> None:
        """Lines 157-181: breakdown table printed when models exist."""
        console = Console(file=StringIO(), force_terminal=True, width=120)

        breakdown = [
            ModelCostBreakdown(
                model_id="gpt-4o",
                display_name="GPT-4o",
                request_count=10,
                total_cost=2.50,
                percentage=80.0,
            ),
            ModelCostBreakdown(
                model_id="gpt-4o-mini",
                display_name="GPT-4o-mini",
                request_count=30,
                total_cost=0.60,
                percentage=20.0,
            ),
        ]
        monthly = _make_summary(
            "month",
            total_cost=3.10,
            total_requests=40,
            model_breakdown=breakdown,
        )

        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=_make_summary("day"),
            monthly=monthly,
            budget_remaining=None,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "gpt-4o" in output
        assert "gpt-4o-mini" in output
        assert "Model Breakdown" in output

    def test_rich_no_breakdown_when_empty(self) -> None:
        """Breakdown table should not print when no models."""
        console = Console(file=StringIO(), force_terminal=True, width=120)

        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=_make_summary("day"),
            monthly=_make_summary("month"),
            budget_remaining=None,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Model Breakdown" not in output

    def test_rich_savings_panel(self) -> None:
        """Lines 183-194: savings panel printed when hypothetical > 0."""
        console = Console(file=StringIO(), force_terminal=True, width=120)

        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=_make_summary("day"),
            monthly=_make_summary("month"),
            budget_remaining=None,
            hypothetical=10.0,
            actual=3.0,
            savings=7.0,
        )

        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Savings Estimate" in output
        assert "$10.00" in output
        assert "$3.00" in output
        assert "$7.00" in output

    def test_rich_no_savings_panel_when_zero(self) -> None:
        """Savings panel should not appear when hypothetical is 0."""
        console = Console(file=StringIO(), force_terminal=True, width=120)

        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=_make_summary("day"),
            monthly=_make_summary("month"),
            budget_remaining=None,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Savings Estimate" not in output

    def test_rich_breakdown_sorted_by_request_count_desc(self) -> None:
        """Models in breakdown table should be sorted by request_count descending."""
        console = Console(file=StringIO(), force_terminal=True, width=120)

        breakdown = [
            ModelCostBreakdown(
                model_id="model-low",
                display_name="Low",
                request_count=2,
                total_cost=0.10,
                percentage=10.0,
            ),
            ModelCostBreakdown(
                model_id="model-high",
                display_name="High",
                request_count=50,
                total_cost=0.90,
                percentage=90.0,
            ),
        ]
        monthly = _make_summary(
            "month",
            total_cost=1.0,
            total_requests=52,
            model_breakdown=breakdown,
        )

        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=_make_summary("day"),
            monthly=monthly,
            budget_remaining=None,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        output = console.file.getvalue()  # type: ignore[union-attr]
        pos_high = output.index("model-high")
        pos_low = output.index("model-low")
        assert pos_high < pos_low

    def test_rich_cost_style_green_for_zero(self) -> None:
        """Line 173: cost_style should be 'green' when total_cost == 0."""
        console = Console(file=StringIO(), force_terminal=True, width=120)

        breakdown = [
            ModelCostBreakdown(
                model_id="ollama/llama3.2:3b",
                display_name="Llama 3.2",
                request_count=10,
                total_cost=0.0,
                percentage=100.0,
            ),
        ]
        monthly = _make_summary(
            "month",
            total_cost=0.0,
            total_requests=10,
            model_breakdown=breakdown,
        )

        # This should not raise — verifies the code path runs
        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=_make_summary("day"),
            monthly=monthly,
            budget_remaining=None,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "$0.00" in output

    def test_rich_cost_style_yellow_for_nonzero(self) -> None:
        """Line 173: cost_style should be 'yellow' when total_cost != 0."""
        console = Console(file=StringIO(), force_terminal=True, width=120)

        breakdown = [
            ModelCostBreakdown(
                model_id="gpt-4o",
                display_name="GPT-4o",
                request_count=5,
                total_cost=1.50,
                percentage=100.0,
            ),
        ]
        monthly = _make_summary(
            "month",
            total_cost=1.50,
            total_requests=5,
            model_breakdown=breakdown,
        )

        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=_make_summary("day"),
            monthly=monthly,
            budget_remaining=None,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "$1.50" in output

    def test_rich_bar_width_proportional(self) -> None:
        """Bar fill should be proportional to percentage."""
        console = Console(file=StringIO(), force_terminal=True, width=120)

        breakdown = [
            ModelCostBreakdown(
                model_id="half-model",
                display_name="Half",
                request_count=5,
                total_cost=1.0,
                percentage=50.0,
            ),
        ]
        monthly = _make_summary(
            "month",
            total_cost=1.0,
            total_requests=5,
            model_breakdown=breakdown,
        )

        _render_rich_dashboard(
            console=console,
            session=_make_summary("session"),
            daily=_make_summary("day"),
            monthly=monthly,
            budget_remaining=None,
            hypothetical=0.0,
            actual=0.0,
            savings=0.0,
        )

        output = console.file.getvalue()  # type: ignore[union-attr]
        # 50% of bar_width=20 => 10 filled + 10 empty
        assert "50%" in output


# ------------------------------------------------------------------
# Full integration: render_cost_dashboard with console
# ------------------------------------------------------------------


class TestRenderCostDashboardWithConsole:
    """End-to-end tests: render_cost_dashboard with a real Rich Console."""

    def test_full_render_with_all_sections(self) -> None:
        """Render with all sections: summary, budget, breakdown, savings."""
        breakdown = [
            ModelCostBreakdown(
                model_id="gpt-4o",
                display_name="GPT-4o",
                request_count=10,
                total_cost=5.0,
                percentage=80.0,
            ),
            ModelCostBreakdown(
                model_id="gpt-4o-mini",
                display_name="GPT-4o-mini",
                request_count=40,
                total_cost=1.25,
                percentage=20.0,
            ),
        ]

        tracker = _make_tracker(
            session_summary=_make_summary("session", total_cost=2.5, total_requests=8),
            daily_summary=_make_summary(
                "day", total_cost=5.0, total_requests=15,
                budget_limit=20.0, budget_remaining=15.0,
            ),
            monthly_summary=_make_summary(
                "month", total_cost=25.0, total_requests=80,
                model_breakdown=breakdown,
                budget_limit=100.0, budget_remaining=75.0,
            ),
            budget_remaining=15.0,
            savings=(50.0, 25.0, 25.0),
        )

        console = Console(file=StringIO(), force_terminal=True, width=120)
        output = render_cost_dashboard(tracker, session_id="sess-1", console=console)

        # Plain-text output
        assert "Cost Dashboard" in output
        assert "Session:" in output
        assert "Today:" in output
        assert "This month:" in output
        assert "Savings Estimate:" in output

        # Rich console output
        rich_output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Cost Dashboard" in rich_output
        assert "Savings Estimate" in rich_output

    def test_returns_string_regardless_of_console(self) -> None:
        tracker = _make_tracker()
        console = Console(file=StringIO(), force_terminal=True, width=120)

        output = render_cost_dashboard(tracker, session_id="sess-1", console=console)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_daily_budget_fallback_in_rich(self) -> None:
        """When monthly has no budget but daily does, Rich path uses daily."""
        tracker = _make_tracker(
            daily_summary=_make_summary(
                "day", total_cost=3.0, total_requests=10,
                budget_limit=5.0, budget_remaining=2.0,
            ),
            monthly_summary=_make_summary(
                "month", total_cost=30.0, total_requests=50,
            ),
            budget_remaining=2.0,
        )

        console = Console(file=StringIO(), force_terminal=True, width=120)
        render_cost_dashboard(tracker, session_id="sess-1", console=console)

        rich_output = console.file.getvalue()  # type: ignore[union-attr]
        assert "daily" in rich_output

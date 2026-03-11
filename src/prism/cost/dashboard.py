"""Cost dashboard rendering for the /cost command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

    from prism.cost.tracker import CostSummary, CostTracker


def render_cost_dashboard(
    tracker: CostTracker,
    session_id: str,
    console: Console | None = None,
) -> str:
    """Render the cost dashboard and optionally print to console.

    Args:
        tracker: CostTracker instance.
        session_id: Current session ID.
        console: Optional Rich console for direct rendering.

    Returns:
        Plain text representation of the dashboard.
    """
    session_summary = tracker.get_cost_summary("session", session_id)
    daily_summary = tracker.get_cost_summary("day")
    monthly_summary = tracker.get_cost_summary("month")

    budget_remaining = tracker.get_budget_remaining()
    hypothetical, actual, savings = tracker.calculate_savings()

    lines: list[str] = []

    # Header
    lines.append("Cost Dashboard")
    lines.append("=" * 50)
    lines.append("")

    # Summary
    lines.append(
        f"Session:      ${session_summary.total_cost:>8.2f}  "
        f"({session_summary.total_requests} requests)"
    )
    lines.append(
        f"Today:        ${daily_summary.total_cost:>8.2f}  "
        f"({daily_summary.total_requests} requests)"
    )
    lines.append(
        f"This month:   ${monthly_summary.total_cost:>8.2f}  "
        f"({monthly_summary.total_requests} requests)"
    )

    if monthly_summary.budget_limit is not None:
        remaining = monthly_summary.budget_remaining or 0.0
        lines.append(
            f"Budget:       ${remaining:>8.2f} remaining "
            f"/ ${monthly_summary.budget_limit:.2f}"
        )
    elif daily_summary.budget_limit is not None:
        remaining = daily_summary.budget_remaining or 0.0
        lines.append(
            f"Budget:       ${remaining:>8.2f} remaining "
            f"/ ${daily_summary.budget_limit:.2f}/day"
        )

    # Model breakdown
    if monthly_summary.model_breakdown:
        lines.append("")
        lines.append("Model Breakdown (this month):")
        lines.append("-" * 50)

        for entry in sorted(
            monthly_summary.model_breakdown,
            key=lambda x: x.request_count,
            reverse=True,
        ):
            bar_width = 18
            filled = int(entry.percentage / 100 * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            lines.append(
                f"  {entry.model_id:<25s} │ {entry.request_count:>5d} req │ "
                f"${entry.total_cost:>8.2f} │ {bar} {entry.percentage:>4.0f}%"
            )

    # Savings
    if hypothetical > 0:
        savings_percent = (savings / hypothetical * 100) if hypothetical > 0 else 0
        lines.append("")
        lines.append("Savings Estimate:")
        lines.append(f"  All via Claude Sonnet: ~${hypothetical:.2f}")
        lines.append(f"  Actual with routing:    ${actual:.2f}")
        lines.append(f"  You saved: ~${savings:.2f} ({savings_percent:.0f}%)")

    output = "\n".join(lines)

    if console is not None:
        _render_rich_dashboard(
            console,
            session_summary,
            daily_summary,
            monthly_summary,
            budget_remaining,
            hypothetical,
            actual,
            savings,
        )

    return output


def _render_rich_dashboard(
    console: Console,
    session: CostSummary,
    daily: CostSummary,
    monthly: CostSummary,
    budget_remaining: float | None,
    hypothetical: float,
    actual: float,
    savings: float,
) -> None:
    """Render a Rich-formatted cost dashboard to the console."""
    # Summary table
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Label", style="bold")
    summary_table.add_column("Cost", justify="right")
    summary_table.add_column("Requests")

    summary_table.add_row(
        "Session:", f"${session.total_cost:.2f}", f"({session.total_requests} requests)"
    )
    summary_table.add_row(
        "Today:", f"${daily.total_cost:.2f}", f"({daily.total_requests} requests)"
    )
    summary_table.add_row(
        "This month:", f"${monthly.total_cost:.2f}", f"({monthly.total_requests} requests)"
    )

    if budget_remaining is not None:
        limit = monthly.budget_limit or daily.budget_limit or 0
        period_label = "monthly" if monthly.budget_limit else "daily"
        summary_table.add_row(
            "Budget:",
            f"${budget_remaining:.2f} remaining",
            f"/ ${limit:.2f} {period_label}",
        )

    console.print(Panel(summary_table, title="[bold]Cost Dashboard[/bold]", border_style="blue"))

    # Model breakdown
    if monthly.model_breakdown:
        breakdown_table = Table(title="Model Breakdown (this month)", border_style="dim")
        breakdown_table.add_column("Model", style="cyan")
        breakdown_table.add_column("Requests", justify="right")
        breakdown_table.add_column("Cost", justify="right", style="green")
        breakdown_table.add_column("Usage", justify="left")

        for entry in sorted(
            monthly.model_breakdown, key=lambda x: x.request_count, reverse=True
        ):
            bar_width = 20
            filled = int(entry.percentage / 100 * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            bar_text = Text(f"{bar} {entry.percentage:.0f}%")

            cost_style = "green" if entry.total_cost == 0 else "yellow"
            breakdown_table.add_row(
                entry.model_id,
                str(entry.request_count),
                Text(f"${entry.total_cost:.2f}", style=cost_style),
                bar_text,
            )

        console.print(breakdown_table)

    # Savings
    if hypothetical > 0:
        savings_percent = (savings / hypothetical * 100) if hypothetical > 0 else 0
        savings_text = Text()
        savings_text.append("If all routed to Claude Sonnet: ", style="dim")
        savings_text.append(f"~${hypothetical:.2f}\n", style="red")
        savings_text.append("Actual cost with Prism routing: ", style="dim")
        savings_text.append(f"${actual:.2f}\n", style="green")
        savings_text.append("You saved: ", style="dim")
        savings_text.append(f"~${savings:.2f} ({savings_percent:.0f}%)", style="bold green")

        console.print(Panel(savings_text, title="Savings Estimate", border_style="green"))

"""Rich display helpers for Architect Mode.

Renders plans, step progress, cost estimates, and rollback results to
the terminal using the ``rich`` library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from prism.architect.planner import Plan, PlanStep, StepStatus

if TYPE_CHECKING:
    from rich.console import Console


# ------------------------------------------------------------------
# Status badge mapping
# ------------------------------------------------------------------

_STATUS_STYLES: dict[StepStatus, tuple[str, str]] = {
    StepStatus.PENDING: ("PENDING", "dim"),
    StepStatus.IN_PROGRESS: ("IN PROGRESS", "yellow"),
    StepStatus.COMPLETED: ("COMPLETED", "green"),
    StepStatus.FAILED: ("FAILED", "red bold"),
    StepStatus.SKIPPED: ("SKIPPED", "dim italic"),
}

_MAX_DESCRIPTION_LEN = 72


def _get_console(console: Console | None = None) -> Console:
    """Return the given console or create a default one."""
    if console is not None:
        return console
    from rich.console import Console as _Console

    return _Console()


def _truncate(text: str, max_len: int = _MAX_DESCRIPTION_LEN) -> str:
    """Truncate *text* to *max_len* characters, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ------------------------------------------------------------------
# Public display functions
# ------------------------------------------------------------------


def display_plan(plan: Plan, console: Console | None = None) -> None:
    """Display a plan with steps, estimated costs, and status badges.

    Renders a Rich panel containing a summary header and a table of
    steps with their statuses.

    Args:
        plan: The plan to display.
        console: Optional Rich console (a default is created if omitted).
    """
    con = _get_console(console)

    # Build steps table
    table = Table(
        show_header=True,
        header_style="bold",
        border_style="dim",
        expand=True,
    )
    table.add_column("#", justify="right", style="cyan", width=4)
    table.add_column("Description", ratio=3)
    table.add_column("Status", justify="center", width=14)
    table.add_column("Tokens", justify="right", width=8)

    for step in sorted(plan.steps, key=lambda s: s.order):
        label, style = _STATUS_STYLES.get(
            step.status, ("UNKNOWN", "dim")
        )
        status_text = Text(label, style=style)
        desc = _truncate(step.description)
        table.add_row(
            str(step.order),
            desc,
            status_text,
            str(step.estimated_tokens),
        )

    # Header text
    header = Text()
    header.append("Plan: ", style="bold")
    header.append(_truncate(plan.description, 60))
    header.append("\nStatus: ", style="bold")
    header.append(plan.status.upper(), style="cyan")
    header.append(f"  |  Steps: {len(plan.steps)}", style="dim")
    header.append(
        f"  |  Est. cost: ${plan.estimated_total_cost:.4f}", style="dim"
    )
    header.append("\nExecution model: ", style="bold")
    header.append(plan.execution_model, style="dim")

    con.print(Panel(header, border_style="blue"))
    con.print(table)


def display_step_progress(
    step: PlanStep, console: Console | None = None
) -> None:
    """Show step execution progress.

    Renders a compact single-line or two-line display for the current
    step's status.

    Args:
        step: The step to display.
        console: Optional Rich console.
    """
    con = _get_console(console)
    label, style = _STATUS_STYLES.get(step.status, ("UNKNOWN", "dim"))

    line = Text()
    line.append(f"  Step {step.order}: ", style="bold")
    line.append(_truncate(step.description, 50))
    line.append(f"  [{label}]", style=style)

    if step.status == StepStatus.FAILED and step.error:
        line.append(f"\n    Error: {_truncate(step.error, 60)}", style="red")
    elif step.status == StepStatus.COMPLETED and step.result:
        line.append(
            f"\n    Result: {_truncate(step.result, 60)}", style="green"
        )

    con.print(line)


def display_cost_estimate(
    plan: Plan, console: Console | None = None
) -> None:
    """Show cost estimate breakdown before execution.

    Renders a table with per-step token estimates and total cost.

    Args:
        plan: The plan to estimate.
        console: Optional Rich console.
    """
    con = _get_console(console)

    table = Table(
        title="Cost Estimate",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    table.add_column("Step", justify="right", width=6)
    table.add_column("Description", ratio=3)
    table.add_column("Est. Tokens", justify="right", width=12)

    total_tokens = 0
    for step in sorted(plan.steps, key=lambda s: s.order):
        table.add_row(
            str(step.order),
            _truncate(step.description),
            str(step.estimated_tokens),
        )
        total_tokens += step.estimated_tokens

    # Footer row
    table.add_row(
        "",
        Text("Total", style="bold"),
        Text(str(total_tokens), style="bold"),
    )

    con.print(table)

    cost_text = Text()
    cost_text.append("Execution model: ", style="dim")
    cost_text.append(plan.execution_model, style="cyan")
    cost_text.append("  |  Estimated cost: ", style="dim")
    cost_text.append(f"${plan.estimated_total_cost:.4f}", style="bold green")
    con.print(cost_text)


def display_rollback_result(
    success: bool, console: Console | None = None
) -> None:
    """Show rollback result.

    Renders a success or failure message.

    Args:
        success: Whether the rollback succeeded.
        console: Optional Rich console.
    """
    con = _get_console(console)

    if success:
        con.print(
            Panel(
                Text("Rollback successful. Repository restored to checkpoint.", style="green"),
                title="Rollback",
                border_style="green",
            )
        )
    else:
        con.print(
            Panel(
                Text(
                    "Rollback failed. Manual intervention may be required.",
                    style="red bold",
                ),
                title="Rollback",
                border_style="red",
            )
        )

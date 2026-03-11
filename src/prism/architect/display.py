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


def display_execution_progress(
    current: int,
    total: int,
    description: str,
    console: Console | None = None,
) -> None:
    """Display step execution progress.

    Shows a compact progress line like:
    [3/10] Creating database migration...

    Args:
        current: Current step number (1-indexed).
        total: Total number of steps.
        description: Current step description.
        console: Optional Rich console.
    """
    con = _get_console(console)
    desc = _truncate(description, 60)
    con.print(
        f"  [{current}/{total}] {desc}",
        style="cyan",
    )


def display_execution_summary(
    summary: object,
    console: Console | None = None,
) -> None:
    """Display a comprehensive execution summary with cost breakdown.

    Shows:
    - Overall status (completed/failed/interrupted)
    - Step results with pass/fail indicators
    - Total cost and token usage
    - Git checkpoint info

    Args:
        summary: ExecutionSummary object.
        console: Optional Rich console.
    """
    con = _get_console(console)

    # Determine overall status style
    if getattr(summary, "interrupted", False):
        status_text = "INTERRUPTED"
        status_style = "yellow bold"
    elif getattr(summary, "was_rolled_back", False):
        status_text = "ROLLED BACK"
        status_style = "red bold"
    elif getattr(summary, "failed_steps", 0) > 0:
        status_text = "FAILED"
        status_style = "red bold"
    else:
        status_text = "COMPLETED"
        status_style = "green bold"

    # Header panel
    desc = _truncate(getattr(summary, "plan_description", ""), 60)
    con.print()
    con.print(
        Panel(
            f"[bold]{desc}[/bold]\n"
            f"Status: [{status_style}]{status_text}[/{status_style}]",
            title="Execution Summary",
            border_style="blue",
        )
    )

    # Step results table
    step_results = getattr(summary, "step_results", [])
    if step_results:
        table = Table(title="Step Results", show_lines=False)
        table.add_column("#", style="dim", width=4)
        table.add_column("Status", width=8)
        table.add_column("Model", width=30)
        table.add_column("Strategy", width=16)
        table.add_column("Attempts", justify="right", width=8)
        table.add_column("Cost", justify="right", width=10)

        for i, result in enumerate(step_results, 1):
            success = getattr(result, "success", False)
            status_icon = "[green]PASS[/green]" if success else "[red]FAIL[/red]"
            model = getattr(result, "model_used", "")
            strategy = getattr(result, "strategy", "default")
            attempts = str(getattr(result, "attempts", 1))
            cost = f"${getattr(result, 'cost_usd', 0.0):.4f}"

            table.add_row(str(i), status_icon, model, strategy, attempts, cost)

        con.print(table)

    # Cost summary
    total_cost = getattr(summary, "total_cost_usd", 0.0)
    total_tokens = getattr(summary, "total_tokens", 0)
    completed = getattr(summary, "completed_steps", 0)
    total = getattr(summary, "total_steps", 0)
    success_rate = getattr(summary, "success_rate", 0.0)

    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Label", style="bold")
    stats_table.add_column("Value")
    stats_table.add_row("Steps completed", f"{completed}/{total} ({success_rate:.0f}%)")
    stats_table.add_row("Total tokens", f"{total_tokens:,}")
    stats_table.add_row("Total cost", f"[bold green]${total_cost:.4f}[/bold green]")

    checkpoint = getattr(summary, "git_checkpoint", None)
    if checkpoint:
        stats_table.add_row("Git checkpoint", f"[dim]{checkpoint}[/dim]")

    con.print(stats_table)
    con.print()


def display_step_retry(
    step_order: int,
    attempt: int,
    max_attempts: int,
    strategy: str,
    model: str,
    console: Console | None = None,
) -> None:
    """Display a retry notification for a failed step.

    Args:
        step_order: Step number.
        attempt: Current attempt number.
        max_attempts: Maximum attempts before giving up.
        strategy: Current retry strategy name.
        model: Model being used for this attempt.
        console: Optional Rich console.
    """
    con = _get_console(console)
    con.print(
        f"  [yellow]Retry step {step_order}[/yellow] "
        f"(attempt {attempt}/{max_attempts}, "
        f"strategy: {strategy}, model: {model})",
    )


def display_pause_notification(
    plan_id: str,
    current_step: int,
    total_steps: int,
    console: Console | None = None,
) -> None:
    """Display notification when execution is paused by Ctrl+C.

    Args:
        plan_id: The plan ID for resume command.
        current_step: Step number where paused.
        total_steps: Total steps in the plan.
        console: Optional Rich console.
    """
    con = _get_console(console)
    con.print()
    con.print(
        Panel(
            f"[yellow bold]Execution paused[/yellow bold] at step {current_step}/{total_steps}\n\n"
            f"Resume with: [cyan]/architect resume {plan_id}[/cyan]",
            title="Paused",
            border_style="yellow",
        )
    )

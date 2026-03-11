"""Tests for prism.architect.display — rich display helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from io import StringIO

from rich.console import Console

from prism.architect.display import (
    display_cost_estimate,
    display_plan,
    display_rollback_result,
    display_step_progress,
)
from prism.architect.planner import Plan, PlanStep, StepStatus

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _capture_console() -> Console:
    """Create a Rich Console that writes to a string buffer."""
    return Console(file=StringIO(), force_terminal=True, width=120)


def _captured_output(console: Console) -> str:
    """Extract captured text from a Console backed by StringIO."""
    file = console.file
    assert isinstance(file, StringIO)
    return file.getvalue()


def _make_step(
    order: int,
    description: str = "",
    status: StepStatus = StepStatus.PENDING,
    *,
    result: str | None = None,
    error: str | None = None,
    tokens: int = 500,
) -> PlanStep:
    return PlanStep(
        id=str(uuid.uuid4()),
        order=order,
        description=description or f"Step {order}",
        tool_calls=[],
        estimated_tokens=tokens,
        status=status,
        result=result,
        error=error,
    )


def _make_plan(
    steps: list[PlanStep] | None = None,
    *,
    description: str = "Test plan",
    status: str = "draft",
    cost: float = 0.0025,
) -> Plan:
    if steps is None:
        steps = [_make_step(1), _make_step(2)]
    return Plan(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        description=description,
        steps=steps,
        planning_model="claude-sonnet-4-20250514",
        execution_model="deepseek/deepseek-chat",
        estimated_total_cost=cost,
        status=status,
    )


# ------------------------------------------------------------------
# Tests: display_plan
# ------------------------------------------------------------------


class TestDisplayPlan:
    """Tests for display_plan."""

    def test_displays_plan_description(self) -> None:
        """Plan description should appear in output."""
        console = _capture_console()
        plan = _make_plan(description="Refactor the auth module")
        display_plan(plan, console=console)
        output = _captured_output(console)
        assert "Refactor the auth module" in output

    def test_displays_all_statuses(self) -> None:
        """Steps with every status should render their status label."""
        steps = [
            _make_step(1, "Pending step", StepStatus.PENDING),
            _make_step(2, "In-progress step", StepStatus.IN_PROGRESS),
            _make_step(3, "Completed step", StepStatus.COMPLETED),
            _make_step(4, "Failed step", StepStatus.FAILED),
            _make_step(5, "Skipped step", StepStatus.SKIPPED),
        ]
        console = _capture_console()
        plan = _make_plan(steps=steps)
        display_plan(plan, console=console)
        output = _captured_output(console)
        assert "PENDING" in output
        assert "IN PROGRESS" in output
        assert "COMPLETED" in output
        assert "FAILED" in output
        assert "SKIPPED" in output

    def test_displays_step_count(self) -> None:
        """Step count should appear in the output."""
        console = _capture_console()
        plan = _make_plan(steps=[_make_step(1), _make_step(2), _make_step(3)])
        display_plan(plan, console=console)
        output = _captured_output(console)
        assert "3" in output

    def test_displays_estimated_cost(self) -> None:
        """Estimated cost should appear in the output."""
        console = _capture_console()
        plan = _make_plan(cost=0.0123)
        display_plan(plan, console=console)
        output = _captured_output(console)
        assert "0.0123" in output

    def test_displays_execution_model(self) -> None:
        """Execution model should appear in the output."""
        console = _capture_console()
        plan = _make_plan()
        display_plan(plan, console=console)
        output = _captured_output(console)
        assert "deepseek/deepseek-chat" in output

    def test_long_description_truncated(self) -> None:
        """Step descriptions longer than the max should be truncated."""
        long = "A" * 200
        steps = [_make_step(1, long)]
        console = _capture_console()
        plan = _make_plan(steps=steps)
        display_plan(plan, console=console)
        output = _captured_output(console)
        assert "..." in output

    def test_default_console_works(self) -> None:
        """display_plan should work without an explicit console (no crash)."""
        plan = _make_plan()
        # Just verify no exception is raised
        display_plan(plan)


# ------------------------------------------------------------------
# Tests: display_step_progress
# ------------------------------------------------------------------


class TestDisplayStepProgress:
    """Tests for display_step_progress."""

    def test_displays_step_order(self) -> None:
        """Step order number should appear in the output."""
        console = _capture_console()
        step = _make_step(3, "Do something")
        display_step_progress(step, console=console)
        output = _captured_output(console)
        assert "Step 3" in output

    def test_completed_step_shows_result(self) -> None:
        """A completed step should show its result."""
        console = _capture_console()
        step = _make_step(
            1, "Task done", StepStatus.COMPLETED, result="All good"
        )
        display_step_progress(step, console=console)
        output = _captured_output(console)
        assert "All good" in output

    def test_failed_step_shows_error(self) -> None:
        """A failed step should show its error message."""
        console = _capture_console()
        step = _make_step(
            1, "Task failed", StepStatus.FAILED, error="Connection timeout"
        )
        display_step_progress(step, console=console)
        output = _captured_output(console)
        assert "Connection timeout" in output

    def test_pending_step_no_extra_info(self) -> None:
        """A pending step should not show result or error lines."""
        console = _capture_console()
        step = _make_step(1, "Waiting")
        display_step_progress(step, console=console)
        output = _captured_output(console)
        assert "PENDING" in output
        assert "Error:" not in output
        assert "Result:" not in output


# ------------------------------------------------------------------
# Tests: display_cost_estimate
# ------------------------------------------------------------------


class TestDisplayCostEstimate:
    """Tests for display_cost_estimate."""

    def test_shows_total_tokens(self) -> None:
        """Total token count should appear in the output."""
        console = _capture_console()
        steps = [_make_step(1, tokens=300), _make_step(2, tokens=700)]
        plan = _make_plan(steps=steps)
        display_cost_estimate(plan, console=console)
        output = _captured_output(console)
        assert "1000" in output

    def test_shows_execution_model(self) -> None:
        """Execution model name should appear in the output."""
        console = _capture_console()
        plan = _make_plan()
        display_cost_estimate(plan, console=console)
        output = _captured_output(console)
        assert "deepseek/deepseek-chat" in output

    def test_shows_estimated_cost(self) -> None:
        """Estimated cost should appear in the output."""
        console = _capture_console()
        plan = _make_plan(cost=0.0042)
        display_cost_estimate(plan, console=console)
        output = _captured_output(console)
        assert "0.0042" in output


# ------------------------------------------------------------------
# Tests: display_rollback_result
# ------------------------------------------------------------------


class TestDisplayRollbackResult:
    """Tests for display_rollback_result."""

    def test_success_message(self) -> None:
        """Successful rollback should display a success message."""
        console = _capture_console()
        display_rollback_result(True, console=console)
        output = _captured_output(console)
        assert "successful" in output.lower()

    def test_failure_message(self) -> None:
        """Failed rollback should display a failure message."""
        console = _capture_console()
        display_rollback_result(False, console=console)
        output = _captured_output(console)
        assert "failed" in output.lower()

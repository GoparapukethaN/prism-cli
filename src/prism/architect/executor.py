"""Plan execution engine for Architect Mode.

Executes an approved :class:`Plan` step by step using the cheap execution
model.  Supports git checkpoints for safe rollback, step-level callbacks,
failure handling, and resumption from the first pending step.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Protocol

import structlog

from prism.architect.planner import Plan, PlanStep, StepStatus
from prism.exceptions import GitError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from prism.config.settings import Settings
    from prism.cost.tracker import CostTracker

logger = structlog.get_logger(__name__)


class GitRepoLike(Protocol):
    """Minimal interface expected from a git repository object."""

    @property
    def root(self) -> Path: ...  # pragma: no cover

    def commit(self, message: str) -> str: ...  # pragma: no cover

    def add(self, files: list[str] | None = None) -> None: ...  # pragma: no cover


class ArchitectExecutor:
    """Executes a plan step by step using the cheap execution model.

    Lifecycle:
        1. :meth:`execute_plan` creates a git checkpoint.
        2. Each step is executed in order via :meth:`execute_step`.
        3. On first failure the plan is marked ``failed`` and execution stops.
        4. :meth:`rollback` resets the repository to the git checkpoint.
        5. :meth:`resume` restarts execution from the first ``PENDING`` step.
    """

    def __init__(
        self,
        settings: Settings,
        cost_tracker: CostTracker,
        git_repo: GitRepoLike | None = None,
    ) -> None:
        """Initialise the executor.

        Args:
            settings: Application settings.
            cost_tracker: Cost tracker.
            git_repo: Optional git repository for checkpoints/rollback.
        """
        self._settings = settings
        self._cost_tracker = cost_tracker
        self._git_repo = git_repo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        plan: Plan,
        on_step_start: Callable[[PlanStep], None] | None = None,
        on_step_complete: Callable[[PlanStep], None] | None = None,
    ) -> Plan:
        """Execute all pending steps in order.

        Creates a git checkpoint before starting.  On first failure the
        plan status is set to ``failed`` and execution stops.

        Args:
            plan: Plan to execute (must be in ``approved`` or ``in_progress`` status).
            on_step_start: Optional callback invoked before each step.
            on_step_complete: Optional callback invoked after each step.

        Returns:
            The updated plan (mutated in place).

        Raises:
            ValueError: If the plan is not in an executable status.
        """
        if plan.status not in ("approved", "in_progress"):
            raise ValueError(
                f"Plan must be 'approved' or 'in_progress' to execute, got '{plan.status}'"
            )

        # Create git checkpoint
        if self._git_repo is not None and plan.git_checkpoint is None:
            plan.git_checkpoint = self._create_checkpoint()

        plan.status = "in_progress"

        for step in sorted(plan.steps, key=lambda s: s.order):
            if step.status in (StepStatus.COMPLETED, StepStatus.SKIPPED):
                continue

            if on_step_start is not None:
                on_step_start(step)

            self.execute_step(step, plan)

            if on_step_complete is not None:
                on_step_complete(step)

            if step.status == StepStatus.FAILED:
                plan.status = "failed"
                logger.warning(
                    "plan_execution_failed",
                    plan_id=plan.id,
                    failed_step=step.id,
                    error=step.error,
                )
                return plan

        plan.status = "completed"
        logger.info("plan_execution_completed", plan_id=plan.id)
        return plan

    def execute_step(self, step: PlanStep, plan: Plan) -> PlanStep:
        """Execute a single step and update its status.

        The current implementation simulates execution by recording a
        success result.  In production this would dispatch to the
        execution model and tool runner.

        Args:
            step: The step to execute.
            plan: The owning plan (for context).

        Returns:
            The updated step (mutated in place).
        """
        step.status = StepStatus.IN_PROGRESS

        try:
            # Simulated execution — in production this calls the
            # execution model and processes tool_calls.
            result = self._simulate_execution(step, plan)
            step.status = StepStatus.COMPLETED
            step.result = result
            logger.info(
                "step_completed",
                plan_id=plan.id,
                step_id=step.id,
                order=step.order,
            )
        except Exception as exc:
            step.status = StepStatus.FAILED
            step.error = str(exc)
            logger.error(
                "step_failed",
                plan_id=plan.id,
                step_id=step.id,
                order=step.order,
                error=str(exc),
            )

        return step

    def rollback(self, plan: Plan) -> bool:
        """Rollback all changes by resetting to the git checkpoint.

        Args:
            plan: Plan whose checkpoint to revert to.

        Returns:
            ``True`` if rollback succeeded, ``False`` otherwise.
        """
        if plan.git_checkpoint is None:
            logger.warning("rollback_no_checkpoint", plan_id=plan.id)
            return False

        if self._git_repo is None:
            logger.warning("rollback_no_git_repo", plan_id=plan.id)
            return False

        try:
            self._git_reset(plan.git_checkpoint)
            plan.status = "rolled_back"
            logger.info(
                "plan_rolled_back",
                plan_id=plan.id,
                checkpoint=plan.git_checkpoint,
            )
            return True
        except (GitError, subprocess.SubprocessError, OSError) as exc:
            logger.error(
                "rollback_failed",
                plan_id=plan.id,
                error=str(exc),
            )
            return False

    def resume(self, plan: Plan) -> Plan:
        """Resume execution from the first PENDING step.

        Re-invokes :meth:`execute_plan` so that callbacks and failure
        semantics are preserved.

        Args:
            plan: Plan to resume.

        Returns:
            The updated plan.

        Raises:
            ValueError: If no pending steps remain.
        """
        pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
        if not pending:
            raise ValueError("No pending steps to resume")

        # Reset failed steps back to pending so they can be retried
        for step in plan.steps:
            if step.status == StepStatus.FAILED:
                step.status = StepStatus.PENDING
                step.error = None

        plan.status = "in_progress"
        return self.execute_plan(plan)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_checkpoint(self) -> str | None:
        """Create a git checkpoint commit and return its hash."""
        if self._git_repo is None:
            return None
        try:
            self._git_repo.add()
            commit_hash = self._git_repo.commit(
                "[prism] architect checkpoint (auto)"
            )
            logger.info("git_checkpoint_created", commit=commit_hash)
            return commit_hash
        except GitError:
            logger.warning("git_checkpoint_failed")
            return None

    def _git_reset(self, commit_hash: str) -> None:
        """Reset the repository to a commit hash.

        Uses ``git reset --hard`` via subprocess so we don't need to
        extend the GitRepo API just for this edge case.
        """
        if self._git_repo is None:
            raise GitError("No git repository configured")

        result = subprocess.run(
            ["git", "reset", "--hard", commit_hash],
            cwd=str(self._git_repo.root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise GitError(f"git reset failed: {result.stderr.strip()}")

    def _simulate_execution(self, step: PlanStep, plan: Plan) -> str:
        """Simulate step execution.

        Returns a success message.  In production this would dispatch
        to the LLM and tool runner.
        """
        return f"Step {step.order} completed: {step.description}"

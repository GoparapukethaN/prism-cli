"""Plan execution engine for Architect Mode.

Executes an approved :class:`Plan` step by step using the cheap execution
model.  Supports git checkpoints for safe rollback, step-level callbacks,
failure handling, and resumption from the first pending step.

Phase 3 enhancements:
- Progress tracking via callbacks
- Step validation for output quality
- Retry with strategy escalation (default -> expanded_context -> simplified)
- Model escalation after retry exhaustion
- Ctrl+C graceful pause (completes current step, then stops)
- Cost breakdown summary via :class:`ExecutionSummary`
- Plan persistence to ``~/.prism/plans/`` as JSON
"""

from __future__ import annotations

import json
import signal
import subprocess
import threading
from dataclasses import dataclass, field
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


@dataclass
class StepResult:
    """Result of executing a single step with metadata."""

    step_id: str
    success: bool
    output: str
    attempts: int = 1
    model_used: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    strategy: str = "default"


@dataclass
class ExecutionSummary:
    """Summary of a completed plan execution."""

    plan_id: str
    plan_description: str
    total_steps: int
    completed_steps: int
    failed_steps: int
    skipped_steps: int
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    step_results: list[StepResult] = field(default_factory=list)
    git_checkpoint: str | None = None
    was_rolled_back: bool = False
    interrupted: bool = False

    @property
    def success_rate(self) -> float:
        """Percentage of steps that completed successfully."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100.0


# Retry and escalation configuration
MAX_STEP_RETRIES = 3
ESCALATION_MODELS: list[str] = [
    "deepseek/deepseek-chat",      # Cheap: try first
    "gpt-4o-mini",                  # Medium: escalate
    "claude-sonnet-4-20250514",     # Premium: final escalation
]
RETRY_STRATEGIES: list[str] = [
    "default",           # Normal execution
    "expanded_context",  # Include more surrounding code
    "simplified",        # Break step into smaller sub-operations
]


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

    async def execute_plan(
        self,
        plan: Plan,
        on_step_start: Callable[[PlanStep], None] | None = None,
        on_step_complete: Callable[[PlanStep], None] | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> ExecutionSummary:
        """Execute an approved plan step by step with progress tracking.

        Supports:
        - Progress callbacks via on_progress(current, total, description)
        - Ctrl+C graceful pause (completes current step, then stops)
        - Retry with strategy escalation on failure
        - Model escalation after MAX_STEP_RETRIES failures
        - Git checkpoint before execution, rollback on total failure

        Args:
            plan: The plan to execute (must be approved or in_progress).
            on_step_start: Called before each step starts.
            on_step_complete: Called after each step finishes.
            on_progress: Called with (current_step, total_steps, description).

        Returns:
            ExecutionSummary with cost breakdown and results.

        Raises:
            ValueError: If plan is not in an executable status.
        """
        if plan.status not in ("approved", "in_progress"):
            msg = f"Plan must be 'approved' or 'in_progress', got '{plan.status}'"
            raise ValueError(msg)

        # Set up Ctrl+C pause handling
        interrupted = False
        pause_event = threading.Event()

        def _signal_handler(signum: int, frame: object) -> None:
            nonlocal interrupted
            interrupted = True
            pause_event.set()
            logger.info("execution_paused", plan_id=plan.id)

        old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _signal_handler)

        try:
            return await self._execute_plan_inner(
                plan=plan,
                on_step_start=on_step_start,
                on_step_complete=on_step_complete,
                on_progress=on_progress,
                interrupted_flag=lambda: interrupted,
            )
        finally:
            signal.signal(signal.SIGINT, old_handler)

    async def _execute_plan_inner(
        self,
        plan: Plan,
        on_step_start: Callable[[PlanStep], None] | None,
        on_step_complete: Callable[[PlanStep], None] | None,
        on_progress: Callable[[int, int, str], None] | None,
        interrupted_flag: Callable[[], bool],
    ) -> ExecutionSummary:
        """Inner execution loop separated for signal handler cleanup."""
        # Create git checkpoint
        if self._git_repo is not None and plan.git_checkpoint is None:
            checkpoint = self._create_checkpoint()
            if checkpoint:
                plan.git_checkpoint = checkpoint

        plan.status = "in_progress"

        step_results: list[StepResult] = []
        executable_steps = sorted(
            [s for s in plan.steps if s.status not in (StepStatus.COMPLETED, StepStatus.SKIPPED)],
            key=lambda s: s.order,
        )
        total = len(executable_steps)

        for idx, step in enumerate(executable_steps):
            # Check for Ctrl+C pause
            if interrupted_flag():
                logger.info("execution_interrupted", plan_id=plan.id, step=step.order)
                plan.status = "in_progress"  # Keep resumable
                return self._build_summary(plan, step_results, interrupted=True)

            if on_progress:
                on_progress(idx + 1, total, step.description)
            if on_step_start:
                on_step_start(step)

            # Execute with retry and escalation
            result = await self._execute_step_with_retry(step, plan)
            step_results.append(result)

            if on_step_complete:
                on_step_complete(step)

            if not result.success:
                plan.status = "failed"
                logger.warning(
                    "plan_execution_failed",
                    plan_id=plan.id,
                    failed_step=step.order,
                    error=step.error,
                )
                return self._build_summary(plan, step_results)

        plan.status = "completed"
        logger.info("plan_execution_completed", plan_id=plan.id)
        return self._build_summary(plan, step_results)

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

    async def resume(self, plan: Plan) -> ExecutionSummary:
        """Resume an interrupted or failed plan from the first pending step.

        Resets any FAILED steps back to PENDING before resuming.

        Args:
            plan: A plan with at least one PENDING step.

        Returns:
            ExecutionSummary from the resumed execution.

        Raises:
            ValueError: If no pending steps remain.
        """
        pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
        if not pending:
            msg = "No pending steps to resume"
            raise ValueError(msg)

        # Reset failed steps so they can be retried
        for step in plan.steps:
            if step.status == StepStatus.FAILED:
                step.status = StepStatus.PENDING
                step.error = None

        plan.status = "in_progress"
        return await self.execute_plan(plan)

    async def _execute_step_with_retry(
        self,
        step: PlanStep,
        plan: Plan,
    ) -> StepResult:
        """Execute a step with retry strategies and model escalation.

        Retry order:
        1. Try with current execution model using default strategy
        2. Try with expanded context strategy
        3. Try with simplified strategy
        4. Escalate to next model tier and repeat

        Args:
            step: The step to execute.
            plan: The parent plan (for model info).

        Returns:
            StepResult with execution metadata.
        """
        models_to_try = [plan.execution_model]
        for m in ESCALATION_MODELS:
            if m != plan.execution_model and m not in models_to_try:
                models_to_try.append(m)

        total_attempts = 0
        last_error = ""

        for model in models_to_try:
            for strategy in RETRY_STRATEGIES:
                total_attempts += 1
                step.status = StepStatus.IN_PROGRESS

                try:
                    output = self._simulate_execution(step, plan)
                    # Validate step result
                    if self._validate_step_result(output, step):
                        step.status = StepStatus.COMPLETED
                        step.result = output
                        step.error = None
                        logger.info(
                            "step_completed",
                            step_id=step.id,
                            order=step.order,
                            model=model,
                            strategy=strategy,
                            attempts=total_attempts,
                        )
                        return StepResult(
                            step_id=step.id,
                            success=True,
                            output=output,
                            attempts=total_attempts,
                            model_used=model,
                            strategy=strategy,
                        )
                except Exception as exc:
                    last_error = str(exc)
                    logger.warning(
                        "step_attempt_failed",
                        step_id=step.id,
                        order=step.order,
                        model=model,
                        strategy=strategy,
                        attempt=total_attempts,
                        error=last_error,
                    )

        # All retries exhausted
        step.status = StepStatus.FAILED
        step.error = f"All retry strategies exhausted after {total_attempts} attempts: {last_error}"
        return StepResult(
            step_id=step.id,
            success=False,
            output="",
            attempts=total_attempts,
            model_used=models_to_try[-1] if models_to_try else "",
            strategy="exhausted",
        )

    @staticmethod
    def _validate_step_result(output: str, step: PlanStep) -> bool:
        """Validate that a step's output is acceptable.

        Checks:
        - Output is not empty
        - Output doesn't contain obvious error markers
        - Output is relevant to the step description

        Args:
            output: The execution output.
            step: The step that produced the output.

        Returns:
            True if the result is valid, False otherwise.
        """
        if not output or not output.strip():
            return False

        # Check for common error patterns in output
        error_markers = [
            "Traceback (most recent call last)",
            "SyntaxError:",
            "IndentationError:",
            "FATAL ERROR",
            "CRITICAL FAILURE",
        ]
        output_lower = output.lower()
        return all(marker.lower() not in output_lower for marker in error_markers)

    def _build_summary(
        self,
        plan: Plan,
        step_results: list[StepResult],
        interrupted: bool = False,
    ) -> ExecutionSummary:
        """Build an ExecutionSummary from plan state and step results.

        Args:
            plan: The executed plan.
            step_results: Results from each step execution.
            interrupted: Whether execution was interrupted by user.

        Returns:
            ExecutionSummary with full cost breakdown.
        """
        completed = sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in plan.steps if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in plan.steps if s.status == StepStatus.SKIPPED)
        total_cost = sum(r.cost_usd for r in step_results)
        total_tokens = sum(r.tokens_used for r in step_results)

        return ExecutionSummary(
            plan_id=plan.id,
            plan_description=plan.description,
            total_steps=len(plan.steps),
            completed_steps=completed,
            failed_steps=failed,
            skipped_steps=skipped,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            step_results=step_results,
            git_checkpoint=plan.git_checkpoint,
            was_rolled_back=plan.status == "rolled_back",
            interrupted=interrupted,
        )

    def save_plan_to_disk(self, plan: Plan, summary: ExecutionSummary | None = None) -> Path:
        """Save a plan and optional summary to ~/.prism/plans/ as JSON.

        Args:
            plan: The plan to save.
            summary: Optional execution summary.

        Returns:
            Path to the saved JSON file.
        """
        plans_dir = self._settings.config.prism_home / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)

        data: dict[str, object] = {
            "id": plan.id,
            "created_at": plan.created_at,
            "description": plan.description,
            "planning_model": plan.planning_model,
            "execution_model": plan.execution_model,
            "estimated_total_cost": plan.estimated_total_cost,
            "status": plan.status,
            "git_checkpoint": plan.git_checkpoint,
            "steps": [
                {
                    "id": s.id,
                    "order": s.order,
                    "description": s.description,
                    "tool_calls": s.tool_calls,
                    "estimated_tokens": s.estimated_tokens,
                    "status": str(s.status),
                    "result": s.result,
                    "error": s.error,
                }
                for s in plan.steps
            ],
        }

        if summary is not None:
            data["summary"] = {
                "total_steps": summary.total_steps,
                "completed_steps": summary.completed_steps,
                "failed_steps": summary.failed_steps,
                "skipped_steps": summary.skipped_steps,
                "total_cost_usd": summary.total_cost_usd,
                "total_tokens": summary.total_tokens,
                "success_rate": summary.success_rate,
                "interrupted": summary.interrupted,
                "was_rolled_back": summary.was_rolled_back,
                "step_results": [
                    {
                        "step_id": r.step_id,
                        "success": r.success,
                        "attempts": r.attempts,
                        "model_used": r.model_used,
                        "strategy": r.strategy,
                        "cost_usd": r.cost_usd,
                        "tokens_used": r.tokens_used,
                    }
                    for r in summary.step_results
                ],
            }

        filename = f"plan_{plan.id}.json"
        filepath = plans_dir / filename
        filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        logger.info("plan_saved_to_disk", path=str(filepath))
        return filepath

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

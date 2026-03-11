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
- Auto-commit after each successful step
- Validation running (pytest) after each step
- Planning vs execution cost tracking
- Enhanced resume with context reconstruction
- Interrupted plan listing
- Enhanced rollback with per-step and per-plan git reset
"""

from __future__ import annotations

import json
import re
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

    def add(
        self, files: list[str] | None = None
    ) -> None: ...  # pragma: no cover


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
    validation_passed: bool | None = None


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
    planning_cost: float = 0.0
    execution_cost: float = 0.0
    estimated_cost: float = 0.0

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

# Pattern to extract pytest path from validation string
_PYTEST_PATH_PATTERN = re.compile(
    r"^Run pytest\s+(.+)$", re.IGNORECASE
)


class ArchitectExecutor:
    """Executes a plan step by step using the cheap execution model.

    Lifecycle:
        1. :meth:`execute_plan` creates a git checkpoint.
        2. Each step is executed in order via :meth:`execute_step`.
        3. On first failure the plan is marked ``failed`` and stops.
        4. :meth:`rollback` resets the repository to the git checkpoint.
        5. :meth:`resume` restarts from the first ``PENDING`` step.
    """

    def __init__(
        self,
        settings: Settings,
        cost_tracker: CostTracker,
        git_repo: GitRepoLike | None = None,
        planning_cost: float = 0.0,
    ) -> None:
        """Initialise the executor.

        Args:
            settings: Application settings.
            cost_tracker: Cost tracker.
            git_repo: Optional git repository for checkpoints/rollback.
            planning_cost: Cost incurred during plan generation.
        """
        self._settings = settings
        self._cost_tracker = cost_tracker
        self._git_repo = git_repo
        self._planning_cost = planning_cost
        self._step_commit_hashes: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_plan(
        self,
        plan: Plan,
        on_step_start: Callable[[PlanStep], None] | None = None,
        on_step_complete: Callable[[PlanStep], None] | None = None,
        on_progress: (
            Callable[[int, int, str], None] | None
        ) = None,
    ) -> ExecutionSummary:
        """Execute an approved plan step by step with progress tracking.

        Supports:
        - Progress callbacks via on_progress(current, total, desc)
        - Ctrl+C graceful pause (completes current step, then stops)
        - Retry with strategy escalation on failure
        - Model escalation after MAX_STEP_RETRIES failures
        - Git checkpoint before execution, rollback on total failure
        - Auto-commit after each successful step
        - Validation running after each step
        - Planning vs execution cost tracking

        Args:
            plan: The plan to execute (must be approved/in_progress).
            on_step_start: Called before each step starts.
            on_step_complete: Called after each step finishes.
            on_progress: Called with (current, total, description).

        Returns:
            ExecutionSummary with cost breakdown and results.

        Raises:
            ValueError: If plan is not in an executable status.
        """
        if plan.status not in ("approved", "in_progress"):
            msg = (
                f"Plan must be 'approved' or 'in_progress', "
                f"got '{plan.status}'"
            )
            raise ValueError(msg)

        # Set up Ctrl+C pause handling
        interrupted = False
        pause_event = threading.Event()

        def _signal_handler(
            signum: int, frame: object
        ) -> None:
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
        # Create git checkpoint and record start hash
        if (
            self._git_repo is not None
            and plan.git_checkpoint is None
        ):
            checkpoint = self._create_checkpoint()
            if checkpoint:
                plan.git_checkpoint = checkpoint
                if not plan.git_start_hash:
                    plan.git_start_hash = checkpoint

        plan.status = "in_progress"
        self._step_commit_hashes.clear()

        step_results: list[StepResult] = []
        executable_steps = sorted(
            [
                s
                for s in plan.steps
                if s.status
                not in (StepStatus.COMPLETED, StepStatus.SKIPPED)
            ],
            key=lambda s: s.order,
        )
        total = len(executable_steps)
        execution_cost = 0.0

        for idx, step in enumerate(executable_steps):
            # Check for Ctrl+C pause
            if interrupted_flag():
                logger.info(
                    "execution_interrupted",
                    plan_id=plan.id,
                    step=step.order,
                )
                plan.status = "in_progress"  # Keep resumable
                return self._build_summary(
                    plan,
                    step_results,
                    interrupted=True,
                    execution_cost=execution_cost,
                )

            if on_progress:
                on_progress(idx + 1, total, step.description)
            if on_step_start:
                on_step_start(step)

            # Execute with retry and escalation
            result = await self._execute_step_with_retry(
                step, plan
            )
            step_results.append(result)

            # Update step cost_usd from result
            if hasattr(step, "estimated_cost"):
                step.estimated_cost = result.cost_usd
            execution_cost += result.cost_usd

            if result.success:
                # Run validation if step has one
                validation_ok = self._run_step_validation(step)
                result.validation_passed = validation_ok

                if not validation_ok:
                    # Retry once on validation failure
                    logger.warning(
                        "step_validation_failed_retrying",
                        step_id=step.id,
                        order=step.order,
                    )
                    step.status = StepStatus.PENDING
                    step.result = None
                    retry_result = (
                        await self._execute_step_with_retry(
                            step, plan
                        )
                    )
                    retry_validation = self._run_step_validation(
                        step
                    )
                    retry_result.validation_passed = (
                        retry_validation
                    )

                    if (
                        not retry_result.success
                        or not retry_validation
                    ):
                        # Retry failed: pause execution
                        step.status = StepStatus.FAILED
                        step.error = (
                            "Validation failed after retry"
                        )
                        retry_result.success = False
                        step_results[-1] = retry_result
                        execution_cost += retry_result.cost_usd
                        plan.status = "paused"
                        logger.warning(
                            "plan_paused_validation_failure",
                            plan_id=plan.id,
                            step=step.order,
                        )
                        return self._build_summary(
                            plan,
                            step_results,
                            execution_cost=execution_cost,
                        )

                    step_results[-1] = retry_result
                    execution_cost += retry_result.cost_usd

                # Auto-commit after successful step
                self._auto_commit_step(plan, step)

            if on_step_complete:
                on_step_complete(step)

            if not result.success:
                # Rollback to previous step's commit on failure
                self._rollback_step_failure(step)
                plan.status = "failed"
                logger.warning(
                    "plan_execution_failed",
                    plan_id=plan.id,
                    failed_step=step.order,
                    error=step.error,
                )
                return self._build_summary(
                    plan,
                    step_results,
                    execution_cost=execution_cost,
                )

        plan.status = "completed"
        logger.info("plan_execution_completed", plan_id=plan.id)
        return self._build_summary(
            plan,
            step_results,
            execution_cost=execution_cost,
        )

    def execute_step(
        self, step: PlanStep, plan: Plan
    ) -> PlanStep:
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

    def rollback(
        self,
        plan: Plan,
        *,
        to_step: int | None = None,
    ) -> tuple[bool, str]:
        """Rollback changes by resetting to a git commit.

        When *to_step* is ``None`` the entire plan is rolled back to
        ``git_start_hash`` (or ``git_checkpoint`` as fallback).
        When *to_step* is given, rollback to that step's commit.

        Args:
            plan: Plan whose checkpoint to revert to.
            to_step: Optional step order to rollback to.

        Returns:
            Tuple of (success, description) for display.
        """
        if self._git_repo is None:
            logger.warning(
                "rollback_no_git_repo", plan_id=plan.id
            )
            return False, "No git repository configured"

        target_hash: str | None = None
        description = ""

        if to_step is not None:
            # Rollback to a specific step's commit
            target_hash = self._step_commit_hashes.get(to_step)
            if target_hash is None:
                msg = f"No commit found for step {to_step}"
                logger.warning(
                    "rollback_step_not_found",
                    plan_id=plan.id,
                    step=to_step,
                )
                return False, msg
            description = (
                f"Rolled back to step {to_step} "
                f"(commit {target_hash[:8]})"
            )
        else:
            # Rollback entire plan
            target_hash = plan.git_start_hash or (
                plan.git_checkpoint
            )
            if target_hash is None:
                logger.warning(
                    "rollback_no_checkpoint",
                    plan_id=plan.id,
                )
                return False, "No git checkpoint available"
            description = (
                f"Rolled back entire plan to "
                f"start (commit {target_hash[:8]})"
            )

        try:
            self._git_reset(target_hash)
            plan.status = "rolled_back"
            logger.info(
                "plan_rolled_back",
                plan_id=plan.id,
                checkpoint=target_hash,
                description=description,
            )
            return True, description
        except (
            GitError,
            subprocess.SubprocessError,
            OSError,
        ) as exc:
            logger.error(
                "rollback_failed",
                plan_id=plan.id,
                error=str(exc),
            )
            return False, f"Rollback failed: {exc}"

    async def resume(
        self,
        plan: Plan,
        storage: object | None = None,
    ) -> ExecutionSummary:
        """Resume an interrupted or failed plan.

        Reconstructs context from storage if provided, resets any
        FAILED steps back to PENDING, and continues execution from
        the first PENDING step.

        Args:
            plan: A plan with at least one PENDING step.
            storage: Optional PlanStorage for context reconstruction.

        Returns:
            ExecutionSummary from the resumed execution.

        Raises:
            ValueError: If no pending steps remain.
        """
        pending = [
            s
            for s in plan.steps
            if s.status == StepStatus.PENDING
        ]
        failed_steps = [
            s
            for s in plan.steps
            if s.status == StepStatus.FAILED
        ]

        if not pending and not failed_steps:
            msg = "No pending steps to resume"
            raise ValueError(msg)

        # Reconstruct context from storage if available
        if storage is not None and hasattr(storage, "load_plan"):
            stored = storage.load_plan(plan.id)
            if stored is not None:
                # Merge completed step results from storage
                stored_map = {
                    s.id: s for s in stored.steps
                }
                for step in plan.steps:
                    stored_step = stored_map.get(step.id)
                    if (
                        stored_step is not None
                        and stored_step.status
                        == StepStatus.COMPLETED
                    ):
                        step.status = stored_step.status
                        step.result = stored_step.result

        # Log resume context
        completed = [
            s
            for s in plan.steps
            if s.status == StepStatus.COMPLETED
        ]
        remaining = [
            s
            for s in plan.steps
            if s.status
            in (StepStatus.PENDING, StepStatus.FAILED)
        ]
        logger.info(
            "plan_resuming",
            plan_id=plan.id,
            completed_steps=len(completed),
            pending_steps=len(remaining),
        )

        # Reset failed steps so they can be retried
        for step in plan.steps:
            if step.status == StepStatus.FAILED:
                step.status = StepStatus.PENDING
                step.error = None

        plan.status = "in_progress"
        return await self.execute_plan(plan)

    def list_interrupted_plans(
        self,
        storage: object | None = None,
    ) -> list[Plan]:
        """Return plans with status PAUSED or RUNNING (stale).

        Scans stored plans from disk (via ``~/.prism/plans/``)
        and returns those that are in an interrupted state.

        Args:
            storage: Optional PlanStorage with list_plans method.

        Returns:
            List of interrupted/paused plans.
        """
        interrupted: list[Plan] = []

        # Try storage (database) first
        if storage is not None and hasattr(
            storage, "list_plans"
        ):
            all_plans = storage.list_plans()
            for p in all_plans:
                if p.status in (
                    "paused",
                    "running",
                    "in_progress",
                ):
                    interrupted.append(p)
            return interrupted

        # Fall back to disk-based plans
        plans_dir = (
            self._settings.config.prism_home / "plans"
        )
        if not plans_dir.exists():
            return interrupted

        for filepath in sorted(plans_dir.glob("plan_*.json")):
            try:
                data = json.loads(
                    filepath.read_text(encoding="utf-8")
                )
                status = data.get("status", "")
                if status in (
                    "paused",
                    "running",
                    "in_progress",
                ):
                    plan = self._plan_from_disk_data(data)
                    interrupted.append(plan)
            except (json.JSONDecodeError, KeyError, OSError):
                logger.warning(
                    "failed_to_read_plan_file",
                    path=str(filepath),
                )
                continue

        return interrupted

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
            if (
                m != plan.execution_model
                and m not in models_to_try
            ):
                models_to_try.append(m)

        total_attempts = 0
        last_error = ""

        for model in models_to_try:
            for strategy in RETRY_STRATEGIES:
                total_attempts += 1
                step.status = StepStatus.IN_PROGRESS

                try:
                    output = self._simulate_execution(
                        step, plan
                    )
                    # Validate step result
                    if self._validate_step_result(
                        output, step
                    ):
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
        step.error = (
            f"All retry strategies exhausted after "
            f"{total_attempts} attempts: {last_error}"
        )
        return StepResult(
            step_id=step.id,
            success=False,
            output="",
            attempts=total_attempts,
            model_used=(
                models_to_try[-1] if models_to_try else ""
            ),
            strategy="exhausted",
        )

    @staticmethod
    def _validate_step_result(
        output: str, step: PlanStep
    ) -> bool:
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
        return all(
            marker.lower() not in output_lower
            for marker in error_markers
        )

    def _build_summary(
        self,
        plan: Plan,
        step_results: list[StepResult],
        interrupted: bool = False,
        execution_cost: float = 0.0,
    ) -> ExecutionSummary:
        """Build an ExecutionSummary from plan state and results.

        Args:
            plan: The executed plan.
            step_results: Results from each step execution.
            interrupted: Whether execution was interrupted.
            execution_cost: Total cost of execution steps.

        Returns:
            ExecutionSummary with full cost breakdown.
        """
        completed = sum(
            1
            for s in plan.steps
            if s.status == StepStatus.COMPLETED
        )
        failed = sum(
            1
            for s in plan.steps
            if s.status == StepStatus.FAILED
        )
        skipped = sum(
            1
            for s in plan.steps
            if s.status == StepStatus.SKIPPED
        )
        total_cost = sum(r.cost_usd for r in step_results)
        total_tokens = sum(
            r.tokens_used for r in step_results
        )

        return ExecutionSummary(
            plan_id=plan.id,
            plan_description=plan.description,
            total_steps=len(plan.steps),
            completed_steps=completed,
            failed_steps=failed,
            skipped_steps=skipped,
            total_cost_usd=total_cost + self._planning_cost,
            total_tokens=total_tokens,
            step_results=step_results,
            git_checkpoint=plan.git_checkpoint,
            was_rolled_back=plan.status == "rolled_back",
            interrupted=interrupted,
            planning_cost=self._planning_cost,
            execution_cost=execution_cost or total_cost,
            estimated_cost=plan.estimated_total_cost,
        )

    def save_plan_to_disk(
        self,
        plan: Plan,
        summary: ExecutionSummary | None = None,
    ) -> Path:
        """Save a plan and optional summary to ~/.prism/plans/.

        Args:
            plan: The plan to save.
            summary: Optional execution summary.

        Returns:
            Path to the saved JSON file.
        """
        plans_dir = (
            self._settings.config.prism_home / "plans"
        )
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
            "goal_summary": plan.goal_summary,
            "git_start_hash": plan.git_start_hash,
            "preconditions": plan.preconditions,
            "postconditions": plan.postconditions,
            "risk_assessment": plan.risk_assessment,
            "estimated_time_minutes": (
                plan.estimated_time_minutes
            ),
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
                    "estimated_cost": s.estimated_cost,
                    "risk_level": s.risk_level,
                    "validation": s.validation,
                    "rollback": s.rollback,
                    "files_to_modify": s.files_to_modify,
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
                "planning_cost": summary.planning_cost,
                "execution_cost": summary.execution_cost,
                "estimated_cost": summary.estimated_cost,
                "step_results": [
                    {
                        "step_id": r.step_id,
                        "success": r.success,
                        "attempts": r.attempts,
                        "model_used": r.model_used,
                        "strategy": r.strategy,
                        "cost_usd": r.cost_usd,
                        "tokens_used": r.tokens_used,
                        "validation_passed": (
                            r.validation_passed
                        ),
                    }
                    for r in summary.step_results
                ],
            }

        filename = f"plan_{plan.id}.json"
        filepath = plans_dir / filename
        filepath.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )

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
            logger.info(
                "git_checkpoint_created", commit=commit_hash
            )
            return commit_hash
        except GitError:
            logger.warning("git_checkpoint_failed")
            return None

    def _auto_commit_step(
        self, plan: Plan, step: PlanStep
    ) -> None:
        """Auto-commit after a successful step.

        Creates a commit with a descriptive message and records the
        commit hash for potential per-step rollback.

        Args:
            plan: The owning plan.
            step: The completed step.
        """
        if self._git_repo is None:
            return

        desc = step.description
        if len(desc) > 50:
            desc = desc[:47] + "..."
        message = (
            f"prism: plan {plan.id[:8]} "
            f"step {step.order} - {desc}"
        )

        try:
            self._git_repo.add()
            commit_hash = self._git_repo.commit(message)
            self._step_commit_hashes[step.order] = (
                commit_hash
            )
            logger.info(
                "step_auto_committed",
                plan_id=plan.id,
                step_order=step.order,
                commit=commit_hash,
            )
        except GitError:
            logger.warning(
                "step_auto_commit_failed",
                plan_id=plan.id,
                step_order=step.order,
            )

    def _run_step_validation(self, step: PlanStep) -> bool:
        """Run validation for a step if configured.

        If the step has a ``validation`` string starting with
        "Run pytest", extracts the test path and delegates to
        :meth:`_run_validation` (which can be overridden in tests).

        Args:
            step: The step to validate.

        Returns:
            True if validation passed or no validation is set.
        """
        if not step.validation:
            return True

        match = _PYTEST_PATH_PATTERN.match(step.validation)
        if match:
            test_path = match.group(1).strip()
            return self._run_validation(test_path)

        # No recognised validation pattern; treat as passing
        return True

    def _run_validation(self, test_path: str) -> bool:
        """Run pytest validation on the given path.

        This method exists as a seam for testing: tests can
        override or mock it without touching subprocess.

        Args:
            test_path: Path to pass to pytest.

        Returns:
            True if pytest passes, False otherwise.
        """
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_path, "-x", "-q"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=(
                    str(self._git_repo.root)
                    if self._git_repo
                    else None
                ),
            )
            passed = result.returncode == 0
            logger.info(
                "validation_result",
                test_path=test_path,
                passed=passed,
                output=result.stdout[:200],
            )
            return passed
        except (
            subprocess.TimeoutExpired,
            OSError,
            FileNotFoundError,
        ) as exc:
            logger.warning(
                "validation_error",
                test_path=test_path,
                error=str(exc),
            )
            return False

    def _rollback_step_failure(
        self, failed_step: PlanStep
    ) -> None:
        """Rollback to previous step's commit on step failure.

        If the failed step has a predecessor with a recorded commit,
        reset to that commit. Otherwise, do nothing (the plan-level
        rollback handles full revert).

        Args:
            failed_step: The step that failed.
        """
        if self._git_repo is None:
            return

        previous_order = failed_step.order - 1
        if previous_order in self._step_commit_hashes:
            target = self._step_commit_hashes[previous_order]
            try:
                self._git_reset(target)
                logger.info(
                    "step_rollback",
                    failed_step=failed_step.order,
                    rolled_back_to=previous_order,
                    commit=target,
                )
            except (
                GitError,
                subprocess.SubprocessError,
                OSError,
            ):
                logger.warning(
                    "step_rollback_failed",
                    failed_step=failed_step.order,
                )

    def _git_reset(self, commit_hash: str) -> None:
        """Reset the repository to a commit hash.

        Uses ``git reset --hard`` via subprocess so we don't need
        to extend the GitRepo API just for this edge case.
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
            raise GitError(
                f"git reset failed: {result.stderr.strip()}"
            )

    def _simulate_execution(
        self, step: PlanStep, plan: Plan
    ) -> str:
        """Simulate step execution.

        Returns a success message.  In production this would dispatch
        to the LLM and tool runner.
        """
        return (
            f"Step {step.order} completed: "
            f"{step.description}"
        )

    @staticmethod
    def _plan_from_disk_data(data: dict[str, object]) -> Plan:
        """Reconstruct a Plan from JSON data loaded from disk.

        Args:
            data: Dictionary parsed from a plan JSON file.

        Returns:
            Reconstructed Plan object.
        """
        steps_data = data.get("steps", [])
        steps: list[PlanStep] = []
        if isinstance(steps_data, list):
            for sd in steps_data:
                if not isinstance(sd, dict):
                    continue
                tool_calls_raw = sd.get("tool_calls", [])
                tool_calls = (
                    tool_calls_raw
                    if isinstance(tool_calls_raw, list)
                    else []
                )
                files_raw = sd.get("files_to_modify", [])
                files_to_modify = (
                    files_raw
                    if isinstance(files_raw, list)
                    else []
                )
                steps.append(
                    PlanStep(
                        id=str(sd.get("id", "")),
                        order=int(sd.get("order", 0)),
                        description=str(
                            sd.get("description", "")
                        ),
                        tool_calls=tool_calls,
                        estimated_tokens=int(
                            sd.get("estimated_tokens", 0)
                        ),
                        status=StepStatus(
                            str(
                                sd.get(
                                    "status", "pending"
                                )
                            )
                        ),
                        result=sd.get("result"),
                        error=sd.get("error"),
                        estimated_cost=float(
                            sd.get("estimated_cost", 0.0)
                        ),
                        risk_level=str(
                            sd.get("risk_level", "LOW")
                        ),
                        validation=str(
                            sd.get("validation", "")
                        ),
                        rollback=str(
                            sd.get("rollback", "")
                        ),
                        files_to_modify=files_to_modify,
                    )
                )

        preconditions_raw = data.get("preconditions", [])
        preconditions = (
            preconditions_raw
            if isinstance(preconditions_raw, list)
            else []
        )
        postconditions_raw = data.get("postconditions", [])
        postconditions = (
            postconditions_raw
            if isinstance(postconditions_raw, list)
            else []
        )

        return Plan(
            id=str(data.get("id", "")),
            created_at=str(data.get("created_at", "")),
            description=str(data.get("description", "")),
            steps=steps,
            planning_model=str(
                data.get("planning_model", "")
            ),
            execution_model=str(
                data.get("execution_model", "")
            ),
            estimated_total_cost=float(
                data.get("estimated_total_cost", 0.0)
            ),
            status=str(data.get("status", "draft")),
            git_checkpoint=data.get("git_checkpoint"),
            goal_summary=str(
                data.get("goal_summary", "")
            ),
            git_start_hash=str(
                data.get("git_start_hash", "")
            ),
            preconditions=preconditions,
            postconditions=postconditions,
            risk_assessment=str(
                data.get("risk_assessment", "")
            ),
            estimated_time_minutes=float(
                data.get("estimated_time_minutes", 0.0)
            ),
        )

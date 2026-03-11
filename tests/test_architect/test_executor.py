"""Tests for prism.architect.executor — plan execution engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prism.architect.executor import ArchitectExecutor
from prism.architect.planner import Plan, PlanStep, StepStatus

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _pending_count(plan: Plan) -> int:
    return sum(1 for s in plan.steps if s.status == StepStatus.PENDING)


def _completed_count(plan: Plan) -> int:
    return sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestExecutePlan:
    """Tests for ArchitectExecutor.execute_plan."""

    def test_execute_single_step_plan(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """A single-step approved plan should complete successfully."""
        # Keep only one step
        approved_plan.steps = [approved_plan.steps[0]]
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        result = executor.execute_plan(approved_plan)
        assert result.status == "completed"
        assert result.steps[0].status == StepStatus.COMPLETED

    def test_execute_multi_step_plan(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """All steps should be executed and completed in order."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        result = executor.execute_plan(approved_plan)
        assert result.status == "completed"
        assert all(s.status == StepStatus.COMPLETED for s in result.steps)

    def test_plan_must_be_approved_or_in_progress(
        self, mock_settings, mock_cost_tracker, sample_plan
    ) -> None:
        """Executing a 'draft' plan should raise ValueError."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        with pytest.raises(ValueError, match="approved"):
            executor.execute_plan(sample_plan)

    def test_stop_on_failure(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Execution should stop on the first failed step."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        original = ArchitectExecutor._simulate_execution

        def failing_sim(self_inner, step, plan):
            if step.order == 2:
                raise RuntimeError("Simulated failure")
            return original(self_inner, step, plan)

        with patch.object(ArchitectExecutor, "_simulate_execution", failing_sim):
            result = executor.execute_plan(approved_plan)

        assert result.status == "failed"
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[1].status == StepStatus.FAILED
        # Third step should still be PENDING (not executed)
        assert result.steps[2].status == StepStatus.PENDING

    def test_skip_completed_steps(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Already-completed steps should be skipped."""
        approved_plan.steps[0].status = StepStatus.COMPLETED
        approved_plan.steps[0].result = "Previously done"
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        result = executor.execute_plan(approved_plan)
        assert result.status == "completed"
        # First step should still have original result
        assert result.steps[0].result == "Previously done"

    def test_git_checkpoint_created(
        self, mock_settings, mock_cost_tracker, mock_git_repo, approved_plan
    ) -> None:
        """A git checkpoint should be created before execution starts."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker, git_repo=mock_git_repo
        )
        result = executor.execute_plan(approved_plan)
        assert result.git_checkpoint is not None
        mock_git_repo.add.assert_called()
        mock_git_repo.commit.assert_called()

    def test_plan_status_transitions(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Plan status should transition: approved -> in_progress -> completed."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        assert approved_plan.status == "approved"
        result = executor.execute_plan(approved_plan)
        assert result.status == "completed"

    def test_step_callbacks_called(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """on_step_start and on_step_complete should be called for each step."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        start_calls: list[PlanStep] = []
        complete_calls: list[PlanStep] = []

        executor.execute_plan(
            approved_plan,
            on_step_start=start_calls.append,
            on_step_complete=complete_calls.append,
        )

        assert len(start_calls) == len(approved_plan.steps)
        assert len(complete_calls) == len(approved_plan.steps)
        # Check order
        assert [s.order for s in start_calls] == [1, 2, 3]

    def test_step_callbacks_on_failure(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Callbacks should still be called on the failing step."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        original = ArchitectExecutor._simulate_execution

        def failing_sim(self_inner, step, plan):
            if step.order == 1:
                raise RuntimeError("Boom")
            return original(self_inner, step, plan)

        with patch.object(ArchitectExecutor, "_simulate_execution", failing_sim):
            complete_calls: list[PlanStep] = []
            executor.execute_plan(
                approved_plan,
                on_step_complete=complete_calls.append,
            )

        assert len(complete_calls) == 1
        assert complete_calls[0].status == StepStatus.FAILED

    def test_no_git_repo_skips_checkpoint(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Without a git repo, no checkpoint should be set."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        result = executor.execute_plan(approved_plan)
        assert result.git_checkpoint is None

    def test_skip_skipped_steps(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Steps with SKIPPED status should not be re-executed."""
        approved_plan.steps[1].status = StepStatus.SKIPPED
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        result = executor.execute_plan(approved_plan)
        assert result.status == "completed"
        assert result.steps[1].status == StepStatus.SKIPPED


class TestExecuteStep:
    """Tests for ArchitectExecutor.execute_step."""

    def test_step_completed(
        self, mock_settings, mock_cost_tracker, sample_step, sample_plan
    ) -> None:
        """A successfully executed step should be COMPLETED with a result."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        executor.execute_step(sample_step, sample_plan)
        assert sample_step.status == StepStatus.COMPLETED
        assert sample_step.result is not None
        assert sample_step.error is None

    def test_step_result_contains_description(
        self, mock_settings, mock_cost_tracker, sample_step, sample_plan
    ) -> None:
        """Step result should reference the step description."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        executor.execute_step(sample_step, sample_plan)
        assert sample_step.description in (sample_step.result or "")


class TestRollback:
    """Tests for ArchitectExecutor.rollback."""

    def test_rollback_no_checkpoint(
        self, mock_settings, mock_cost_tracker, sample_plan
    ) -> None:
        """Rollback without a checkpoint should return False."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        sample_plan.git_checkpoint = None
        assert executor.rollback(sample_plan) is False

    def test_rollback_no_git_repo(
        self, mock_settings, mock_cost_tracker, sample_plan
    ) -> None:
        """Rollback without a git repo should return False."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        sample_plan.git_checkpoint = "abc123"
        assert executor.rollback(sample_plan) is False

    def test_rollback_success(
        self, mock_settings, mock_cost_tracker, mock_git_repo, failed_plan
    ) -> None:
        """Successful rollback should reset plan status to rolled_back."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker, git_repo=mock_git_repo
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("prism.architect.executor.subprocess.run", return_value=mock_result):
            result = executor.rollback(failed_plan)

        assert result is True
        assert failed_plan.status == "rolled_back"


class TestResume:
    """Tests for ArchitectExecutor.resume."""

    def test_resume_from_pending(
        self, mock_settings, mock_cost_tracker, failed_plan
    ) -> None:
        """Resume should re-execute from the first pending step."""
        failed_plan.status = "in_progress"
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        result = executor.resume(failed_plan)
        # The failed step should be retried and the pending step executed
        assert result.status == "completed"

    def test_resume_no_pending_raises(
        self, mock_settings, mock_cost_tracker, completed_plan
    ) -> None:
        """Resume with no pending steps should raise ValueError."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        with pytest.raises(ValueError, match="pending"):
            executor.resume(completed_plan)

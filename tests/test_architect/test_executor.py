"""Tests for prism.architect.executor — plan execution engine."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from prism.architect.executor import (
    ESCALATION_MODELS,
    MAX_STEP_RETRIES,
    RETRY_STRATEGIES,
    ArchitectExecutor,
    ExecutionSummary,
    StepResult,
)
from prism.architect.planner import Plan, PlanStep, StepStatus

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _pending_count(plan: Plan) -> int:
    return sum(1 for s in plan.steps if s.status == StepStatus.PENDING)


def _completed_count(plan: Plan) -> int:
    return sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED)


# ------------------------------------------------------------------
# Tests — ExecutionSummary and StepResult dataclasses
# ------------------------------------------------------------------


class TestStepResult:
    """Tests for the StepResult dataclass."""

    def test_default_values(self) -> None:
        """StepResult should have sensible defaults."""
        sr = StepResult(step_id="s1", success=True, output="ok")
        assert sr.attempts == 1
        assert sr.model_used == ""
        assert sr.tokens_used == 0
        assert sr.cost_usd == 0.0
        assert sr.strategy == "default"


class TestExecutionSummary:
    """Tests for the ExecutionSummary dataclass."""

    def test_success_rate_all_completed(self) -> None:
        """Success rate should be 100% when all steps completed."""
        summary = ExecutionSummary(
            plan_id="p1",
            plan_description="test",
            total_steps=3,
            completed_steps=3,
            failed_steps=0,
            skipped_steps=0,
        )
        assert summary.success_rate == 100.0

    def test_success_rate_partial(self) -> None:
        """Success rate should reflect partial completion."""
        summary = ExecutionSummary(
            plan_id="p1",
            plan_description="test",
            total_steps=4,
            completed_steps=2,
            failed_steps=1,
            skipped_steps=1,
        )
        assert summary.success_rate == 50.0

    def test_success_rate_zero_steps(self) -> None:
        """Success rate should be 0 when there are no steps."""
        summary = ExecutionSummary(
            plan_id="p1",
            plan_description="test",
            total_steps=0,
            completed_steps=0,
            failed_steps=0,
            skipped_steps=0,
        )
        assert summary.success_rate == 0.0

    def test_default_fields(self) -> None:
        """Default fields should be set correctly."""
        summary = ExecutionSummary(
            plan_id="p1",
            plan_description="test",
            total_steps=1,
            completed_steps=1,
            failed_steps=0,
            skipped_steps=0,
        )
        assert summary.total_cost_usd == 0.0
        assert summary.total_tokens == 0
        assert summary.step_results == []
        assert summary.git_checkpoint is None
        assert summary.was_rolled_back is False
        assert summary.interrupted is False


# ------------------------------------------------------------------
# Tests — Constants
# ------------------------------------------------------------------


class TestConstants:
    """Tests for retry/escalation constants."""

    def test_max_step_retries(self) -> None:
        """MAX_STEP_RETRIES should be a positive integer."""
        assert MAX_STEP_RETRIES == 3

    def test_escalation_models(self) -> None:
        """ESCALATION_MODELS should contain expected models."""
        assert len(ESCALATION_MODELS) == 3
        assert "deepseek/deepseek-chat" in ESCALATION_MODELS
        assert "gpt-4o-mini" in ESCALATION_MODELS
        assert "claude-sonnet-4-20250514" in ESCALATION_MODELS

    def test_retry_strategies(self) -> None:
        """RETRY_STRATEGIES should contain expected strategies."""
        assert len(RETRY_STRATEGIES) == 3
        assert "default" in RETRY_STRATEGIES
        assert "expanded_context" in RETRY_STRATEGIES
        assert "simplified" in RETRY_STRATEGIES


# ------------------------------------------------------------------
# Tests — execute_plan (async)
# ------------------------------------------------------------------


class TestExecutePlan:
    """Tests for ArchitectExecutor.execute_plan."""

    async def test_execute_single_step_plan(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """A single-step approved plan should complete successfully."""
        approved_plan.steps = [approved_plan.steps[0]]
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        summary = await executor.execute_plan(approved_plan)
        assert isinstance(summary, ExecutionSummary)
        assert approved_plan.status == "completed"
        assert approved_plan.steps[0].status == StepStatus.COMPLETED
        assert summary.completed_steps == 1
        assert summary.failed_steps == 0

    async def test_execute_multi_step_plan(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """All steps should be executed and completed in order."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        summary = await executor.execute_plan(approved_plan)
        assert approved_plan.status == "completed"
        assert all(s.status == StepStatus.COMPLETED for s in approved_plan.steps)
        assert summary.completed_steps == 3
        assert summary.total_steps == 3

    async def test_plan_must_be_approved_or_in_progress(
        self, mock_settings, mock_cost_tracker, sample_plan
    ) -> None:
        """Executing a 'draft' plan should raise ValueError."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        with pytest.raises(ValueError, match="approved"):
            await executor.execute_plan(sample_plan)

    async def test_stop_on_failure(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Execution should stop after all retry strategies are exhausted."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)

        def always_failing_sim(self_inner: object, step: PlanStep, plan: Plan) -> str:
            if step.order == 2:
                raise RuntimeError("Simulated failure")
            return f"Step {step.order} completed: {step.description}"

        with patch.object(ArchitectExecutor, "_simulate_execution", always_failing_sim):
            summary = await executor.execute_plan(approved_plan)

        assert approved_plan.status == "failed"
        assert approved_plan.steps[0].status == StepStatus.COMPLETED
        assert approved_plan.steps[1].status == StepStatus.FAILED
        # Third step should still be PENDING (not executed)
        assert approved_plan.steps[2].status == StepStatus.PENDING
        assert summary.failed_steps == 1
        assert summary.completed_steps == 1

    async def test_skip_completed_steps(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Already-completed steps should be skipped."""
        approved_plan.steps[0].status = StepStatus.COMPLETED
        approved_plan.steps[0].result = "Previously done"
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        summary = await executor.execute_plan(approved_plan)
        assert approved_plan.status == "completed"
        # First step should still have original result
        assert approved_plan.steps[0].result == "Previously done"
        assert summary.completed_steps == 3

    async def test_git_checkpoint_created(
        self, mock_settings, mock_cost_tracker, mock_git_repo, approved_plan
    ) -> None:
        """A git checkpoint should be created before execution starts."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker, git_repo=mock_git_repo
        )
        summary = await executor.execute_plan(approved_plan)
        assert approved_plan.git_checkpoint is not None
        assert summary.git_checkpoint is not None
        mock_git_repo.add.assert_called()
        mock_git_repo.commit.assert_called()

    async def test_plan_status_transitions(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Plan status should transition: approved -> in_progress -> completed."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        assert approved_plan.status == "approved"
        summary = await executor.execute_plan(approved_plan)
        assert approved_plan.status == "completed"
        assert summary.completed_steps == summary.total_steps

    async def test_step_callbacks_called(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """on_step_start and on_step_complete should be called for each step."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        start_calls: list[PlanStep] = []
        complete_calls: list[PlanStep] = []

        await executor.execute_plan(
            approved_plan,
            on_step_start=start_calls.append,
            on_step_complete=complete_calls.append,
        )

        assert len(start_calls) == len(approved_plan.steps)
        assert len(complete_calls) == len(approved_plan.steps)
        # Check order
        assert [s.order for s in start_calls] == [1, 2, 3]

    async def test_step_callbacks_on_failure(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Callbacks should still be called on the failing step."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)

        def always_failing_sim(self_inner: object, step: PlanStep, plan: Plan) -> str:
            if step.order == 1:
                raise RuntimeError("Boom")
            return f"Step {step.order} completed: {step.description}"

        with patch.object(ArchitectExecutor, "_simulate_execution", always_failing_sim):
            complete_calls: list[PlanStep] = []
            await executor.execute_plan(
                approved_plan,
                on_step_complete=complete_calls.append,
            )

        assert len(complete_calls) == 1
        assert complete_calls[0].status == StepStatus.FAILED

    async def test_no_git_repo_skips_checkpoint(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Without a git repo, no checkpoint should be set."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        summary = await executor.execute_plan(approved_plan)
        assert approved_plan.git_checkpoint is None
        assert summary.git_checkpoint is None

    async def test_skip_skipped_steps(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Steps with SKIPPED status should not be re-executed."""
        approved_plan.steps[1].status = StepStatus.SKIPPED
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        summary = await executor.execute_plan(approved_plan)
        assert approved_plan.status == "completed"
        assert approved_plan.steps[1].status == StepStatus.SKIPPED
        assert summary.skipped_steps == 1

    async def test_progress_callback_called(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """on_progress should be called for each executable step."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        progress_calls: list[tuple[int, int, str]] = []

        def on_progress(current: int, total: int, desc: str) -> None:
            progress_calls.append((current, total, desc))

        await executor.execute_plan(approved_plan, on_progress=on_progress)

        assert len(progress_calls) == 3
        assert progress_calls[0][0] == 1  # current
        assert progress_calls[0][1] == 3  # total
        assert progress_calls[2][0] == 3

    async def test_summary_step_results(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """ExecutionSummary should contain StepResult for each executed step."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        summary = await executor.execute_plan(approved_plan)
        assert len(summary.step_results) == 3
        for sr in summary.step_results:
            assert isinstance(sr, StepResult)
            assert sr.success is True
            assert sr.attempts == 1
            assert sr.strategy == "default"

    async def test_summary_success_rate(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """ExecutionSummary should report correct success rate."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        summary = await executor.execute_plan(approved_plan)
        assert summary.success_rate == 100.0


# ------------------------------------------------------------------
# Tests — execute_step (sync, unchanged)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Tests — rollback (sync, unchanged)
# ------------------------------------------------------------------


class TestRollback:
    """Tests for ArchitectExecutor.rollback."""

    def test_rollback_no_checkpoint(
        self, mock_settings, mock_cost_tracker, sample_plan
    ) -> None:
        """Rollback without a checkpoint should return False."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        sample_plan.git_checkpoint = None
        success, _desc = executor.rollback(sample_plan)
        assert success is False

    def test_rollback_no_git_repo(
        self, mock_settings, mock_cost_tracker, sample_plan
    ) -> None:
        """Rollback without a git repo should return False."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        sample_plan.git_checkpoint = "abc123"
        success, _desc = executor.rollback(sample_plan)
        assert success is False

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
            success, desc = executor.rollback(failed_plan)

        assert success is True
        assert desc  # Description should be non-empty
        assert failed_plan.status == "rolled_back"


# ------------------------------------------------------------------
# Tests — resume (async)
# ------------------------------------------------------------------


class TestResume:
    """Tests for ArchitectExecutor.resume."""

    async def test_resume_from_pending(
        self, mock_settings, mock_cost_tracker, failed_plan
    ) -> None:
        """Resume should re-execute from the first pending step."""
        failed_plan.status = "in_progress"
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        summary = await executor.resume(failed_plan)
        # The failed step should be retried and the pending step executed
        assert isinstance(summary, ExecutionSummary)
        assert failed_plan.status == "completed"

    async def test_resume_no_pending_raises(
        self, mock_settings, mock_cost_tracker, completed_plan
    ) -> None:
        """Resume with no pending steps should raise ValueError."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        with pytest.raises(ValueError, match="pending"):
            await executor.resume(completed_plan)


# ------------------------------------------------------------------
# Tests — _validate_step_result
# ------------------------------------------------------------------


class TestValidateStepResult:
    """Tests for ArchitectExecutor._validate_step_result."""

    def test_valid_output(self, sample_step: PlanStep) -> None:
        """Normal output should be valid."""
        assert ArchitectExecutor._validate_step_result("Step completed OK", sample_step) is True

    def test_empty_output(self, sample_step: PlanStep) -> None:
        """Empty output should be invalid."""
        assert ArchitectExecutor._validate_step_result("", sample_step) is False

    def test_whitespace_only_output(self, sample_step: PlanStep) -> None:
        """Whitespace-only output should be invalid."""
        assert ArchitectExecutor._validate_step_result("   \n  ", sample_step) is False

    def test_traceback_output(self, sample_step: PlanStep) -> None:
        """Output containing a traceback should be invalid."""
        output = "Traceback (most recent call last):\n  File 'x.py', line 1"
        assert ArchitectExecutor._validate_step_result(output, sample_step) is False

    def test_syntax_error_output(self, sample_step: PlanStep) -> None:
        """Output containing SyntaxError should be invalid."""
        assert ArchitectExecutor._validate_step_result("SyntaxError: invalid", sample_step) is False

    def test_fatal_error_output(self, sample_step: PlanStep) -> None:
        """Output containing FATAL ERROR should be invalid."""
        assert ArchitectExecutor._validate_step_result("FATAL ERROR occurred", sample_step) is False

    def test_critical_failure_output(self, sample_step: PlanStep) -> None:
        """Output containing CRITICAL FAILURE should be invalid."""
        output = "CRITICAL FAILURE: system down"
        assert ArchitectExecutor._validate_step_result(output, sample_step) is False

    def test_indentation_error_output(self, sample_step: PlanStep) -> None:
        """Output containing IndentationError should be invalid."""
        output = "IndentationError: unexpected indent"
        assert ArchitectExecutor._validate_step_result(output, sample_step) is False


# ------------------------------------------------------------------
# Tests — _execute_step_with_retry
# ------------------------------------------------------------------


class TestExecuteStepWithRetry:
    """Tests for retry and model escalation logic."""

    async def test_success_on_first_try(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """A step that succeeds on first try should return attempts=1."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        step = approved_plan.steps[0]
        result = await executor._execute_step_with_retry(step, approved_plan)
        assert result.success is True
        assert result.attempts == 1
        assert result.strategy == "default"

    async def test_retry_on_exception(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """A step that fails some attempts should eventually succeed or exhaust retries."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        step = approved_plan.steps[0]
        call_count = 0

        def intermittent_failure(self_inner: object, s: PlanStep, p: Plan) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Transient failure")
            return f"Step {s.order} completed: {s.description}"

        with patch.object(ArchitectExecutor, "_simulate_execution", intermittent_failure):
            result = await executor._execute_step_with_retry(step, approved_plan)

        assert result.success is True
        assert result.attempts == 3

    async def test_all_retries_exhausted(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """When all retries fail, the step should be marked FAILED."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        step = approved_plan.steps[0]

        def always_fail(self_inner: object, s: PlanStep, p: Plan) -> str:
            raise RuntimeError("Permanent failure")

        with patch.object(ArchitectExecutor, "_simulate_execution", always_fail):
            result = await executor._execute_step_with_retry(step, approved_plan)

        assert result.success is False
        assert result.strategy == "exhausted"
        assert step.status == StepStatus.FAILED
        assert "exhausted" in (step.error or "").lower()
        # Should have tried all models x all strategies
        expected = len(ESCALATION_MODELS) * len(RETRY_STRATEGIES)
        assert result.attempts == expected

    async def test_validation_failure_triggers_retry(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """If output fails validation, the step should retry."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        step = approved_plan.steps[0]
        call_count = 0

        def bad_then_good(self_inner: object, s: PlanStep, p: Plan) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ""  # Empty output fails validation
            return f"Step {s.order} completed: {s.description}"

        with patch.object(ArchitectExecutor, "_simulate_execution", bad_then_good):
            result = await executor._execute_step_with_retry(step, approved_plan)

        assert result.success is True
        assert result.attempts == 2


# ------------------------------------------------------------------
# Tests — _build_summary
# ------------------------------------------------------------------


class TestBuildSummary:
    """Tests for _build_summary."""

    def test_build_summary_completed(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Summary should reflect completed plan state."""
        approved_plan.status = "completed"
        for s in approved_plan.steps:
            s.status = StepStatus.COMPLETED

        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        step_results = [
            StepResult(step_id=s.id, success=True, output="ok", cost_usd=0.001)
            for s in approved_plan.steps
        ]
        summary = executor._build_summary(approved_plan, step_results)

        assert summary.plan_id == approved_plan.id
        assert summary.total_steps == 3
        assert summary.completed_steps == 3
        assert summary.failed_steps == 0
        assert summary.total_cost_usd == pytest.approx(0.003)
        assert summary.interrupted is False

    def test_build_summary_interrupted(
        self, mock_settings, mock_cost_tracker, approved_plan
    ) -> None:
        """Summary should report interrupted=True when flagged."""
        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        summary = executor._build_summary(approved_plan, [], interrupted=True)
        assert summary.interrupted is True


# ------------------------------------------------------------------
# Tests — save_plan_to_disk
# ------------------------------------------------------------------


class TestSavePlanToDisk:
    """Tests for save_plan_to_disk."""

    def test_save_plan_without_summary(
        self, mock_settings, mock_cost_tracker, approved_plan, tmp_path: Path
    ) -> None:
        """Save should create a JSON file in the plans directory."""
        mock_prism_home = MagicMock()
        mock_prism_home.__truediv__ = lambda self, other: tmp_path / other
        mock_settings.config.prism_home = tmp_path

        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        filepath = executor.save_plan_to_disk(approved_plan)

        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["id"] == approved_plan.id
        assert data["description"] == approved_plan.description
        assert len(data["steps"]) == len(approved_plan.steps)
        assert "summary" not in data

    def test_save_plan_with_summary(
        self, mock_settings, mock_cost_tracker, approved_plan, tmp_path: Path
    ) -> None:
        """Save with summary should include summary data in JSON."""
        mock_settings.config.prism_home = tmp_path

        summary = ExecutionSummary(
            plan_id=approved_plan.id,
            plan_description=approved_plan.description,
            total_steps=3,
            completed_steps=3,
            failed_steps=0,
            skipped_steps=0,
            total_cost_usd=0.005,
            total_tokens=1500,
            step_results=[
                StepResult(
                    step_id=s.id,
                    success=True,
                    output="ok",
                    model_used="deepseek/deepseek-chat",
                )
                for s in approved_plan.steps
            ],
        )

        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        filepath = executor.save_plan_to_disk(approved_plan, summary=summary)

        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert "summary" in data
        assert data["summary"]["total_steps"] == 3
        assert data["summary"]["completed_steps"] == 3
        assert data["summary"]["total_cost_usd"] == 0.005
        assert data["summary"]["success_rate"] == 100.0
        assert len(data["summary"]["step_results"]) == 3

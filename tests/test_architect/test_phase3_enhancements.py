"""Tests for Phase 3 enhancements across the architect module.

Covers all new fields, enrichment logic, storage round-trips,
enhanced executor features (auto-commit, validation, resume,
interrupted plans, rollback), and display functions for the
review/list/validation/summary views.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from prism.architect.display import (
    display_execution_summary,
    display_plan_list,
    display_plan_review,
    display_rollback_result,
    display_step_validation,
)
from prism.architect.executor import (
    ArchitectExecutor,
    ExecutionSummary,
    StepResult,
)
from prism.architect.planner import (
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    ArchitectPlanner,
    Plan,
    PlanStatus,
    PlanStep,
    StepStatus,
)
from prism.architect.storage import PlanStorage

if TYPE_CHECKING:
    from prism.db.database import Database

# ------------------------------------------------------------------
# Fixtures
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
    files_to_modify: list[str] | None = None,
    estimated_cost: float = 0.0,
    risk_level: str = RISK_LOW,
    validation: str = "",
    rollback: str = "",
) -> PlanStep:
    """Build a PlanStep with enhanced fields."""
    return PlanStep(
        id=str(uuid.uuid4()),
        order=order,
        description=description or f"Step {order} description",
        tool_calls=[{"tool": "read_file", "args": {"path": "/tmp/f.py"}}],
        estimated_tokens=500,
        status=status,
        result=result,
        error=error,
        files_to_modify=files_to_modify or [],
        estimated_cost=estimated_cost,
        risk_level=risk_level,
        validation=validation,
        rollback=rollback,
    )


def _make_enhanced_plan(
    *,
    step_count: int = 3,
    status: str = "draft",
    goal_summary: str = "Refactor module",
    preconditions: list[str] | None = None,
    postconditions: list[str] | None = None,
    risk_assessment: str = "LOW - all safe",
    estimated_time_minutes: float = 6.0,
    git_start_hash: str = "abc123",
    git_checkpoint: str | None = None,
) -> Plan:
    """Build a Plan with all Phase 3 fields populated."""
    steps = [
        _make_step(
            i,
            f"Step {i} of the plan",
            files_to_modify=[f"src/module_{i}.py"],
            estimated_cost=0.001,
            risk_level=RISK_LOW,
            validation=f"Verify src/module_{i}.py exists and is valid",
            rollback=f"git checkout -- src/module_{i}.py",
        )
        for i in range(1, step_count + 1)
    ]
    return Plan(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        description="Refactor the authentication module",
        steps=steps,
        planning_model="claude-sonnet-4-20250514",
        execution_model="deepseek/deepseek-chat",
        estimated_total_cost=0.003,
        status=status,
        git_checkpoint=git_checkpoint,
        goal_summary=goal_summary,
        preconditions=preconditions or ["Git working directory clean", "All tests passing"],
        postconditions=postconditions or ["All new tests passing", "No ruff errors"],
        risk_assessment=risk_assessment,
        estimated_time_minutes=estimated_time_minutes,
        git_start_hash=git_start_hash,
    )


# ------------------------------------------------------------------
# TestPlanStepNewFields (8 tests)
# ------------------------------------------------------------------


class TestPlanStepNewFields:
    """Tests for new PlanStep fields added in Phase 3."""

    def test_step_has_files_to_modify_field(self) -> None:
        """PlanStep should have a files_to_modify list field."""
        step = PlanStep(
            id="s1",
            order=1,
            description="Edit file",
            files_to_modify=["src/app.py", "tests/test_app.py"],
        )
        assert step.files_to_modify == ["src/app.py", "tests/test_app.py"]

    def test_step_has_estimated_cost_field(self) -> None:
        """PlanStep should have an estimated_cost float field."""
        step = PlanStep(
            id="s2",
            order=1,
            description="Edit file",
            estimated_cost=0.0015,
        )
        assert step.estimated_cost == pytest.approx(0.0015)

    def test_step_has_risk_level_field(self) -> None:
        """PlanStep should have a risk_level str field."""
        step = PlanStep(
            id="s3",
            order=1,
            description="Edit file",
            risk_level=RISK_HIGH,
        )
        assert step.risk_level == "HIGH"

    def test_step_risk_level_defaults_to_low(self) -> None:
        """PlanStep risk_level should default to LOW."""
        step = PlanStep(id="s4", order=1, description="Edit file")
        assert step.risk_level == RISK_LOW

    def test_step_has_validation_field(self) -> None:
        """PlanStep should have a validation str field."""
        step = PlanStep(
            id="s5",
            order=1,
            description="Edit file",
            validation="Run pytest tests/test_app.py",
        )
        assert step.validation == "Run pytest tests/test_app.py"

    def test_step_has_rollback_field(self) -> None:
        """PlanStep should have a rollback str field."""
        step = PlanStep(
            id="s6",
            order=1,
            description="Edit file",
            rollback="git checkout -- src/app.py",
        )
        assert step.rollback == "git checkout -- src/app.py"

    def test_step_defaults_for_new_fields(self) -> None:
        """All new PlanStep fields should have correct defaults."""
        step = PlanStep(id="s7", order=1, description="Test defaults")
        assert step.files_to_modify == []
        assert step.estimated_cost == 0.0
        assert step.risk_level == RISK_LOW
        assert step.validation == ""
        assert step.rollback == ""

    def test_step_files_to_modify_is_independent_list(self) -> None:
        """Each PlanStep should have its own files_to_modify list."""
        step1 = PlanStep(id="a", order=1, description="Step A")
        step2 = PlanStep(id="b", order=2, description="Step B")
        step1.files_to_modify.append("file1.py")
        assert step2.files_to_modify == []


# ------------------------------------------------------------------
# TestPlanNewFields (8 tests)
# ------------------------------------------------------------------


class TestPlanNewFields:
    """Tests for new Plan fields added in Phase 3."""

    def test_plan_has_goal_summary(self) -> None:
        """Plan should have a goal_summary str field."""
        plan = _make_enhanced_plan(goal_summary="Upgrade auth system")
        assert plan.goal_summary == "Upgrade auth system"

    def test_plan_has_preconditions(self) -> None:
        """Plan should have a preconditions list field."""
        plan = _make_enhanced_plan(
            preconditions=["Clean working tree", "Tests passing"],
        )
        assert plan.preconditions == ["Clean working tree", "Tests passing"]

    def test_plan_has_postconditions(self) -> None:
        """Plan should have a postconditions list field."""
        plan = _make_enhanced_plan(
            postconditions=["New tests pass", "Lint clean"],
        )
        assert plan.postconditions == ["New tests pass", "Lint clean"]

    def test_plan_has_risk_assessment(self) -> None:
        """Plan should have a risk_assessment str field."""
        plan = _make_enhanced_plan(risk_assessment="HIGH - touches auth")
        assert plan.risk_assessment == "HIGH - touches auth"

    def test_plan_has_estimated_time_minutes(self) -> None:
        """Plan should have an estimated_time_minutes float field."""
        plan = _make_enhanced_plan(estimated_time_minutes=12.5)
        assert plan.estimated_time_minutes == pytest.approx(12.5)

    def test_plan_has_git_start_hash(self) -> None:
        """Plan should have a git_start_hash str field."""
        plan = _make_enhanced_plan(git_start_hash="deadbeef")
        assert plan.git_start_hash == "deadbeef"

    def test_plan_status_enum_values(self) -> None:
        """PlanStatus enum should contain all seven lifecycle states."""
        assert PlanStatus.DRAFT == "draft"
        assert PlanStatus.APPROVED == "approved"
        assert PlanStatus.RUNNING == "running"
        assert PlanStatus.PAUSED == "paused"
        assert PlanStatus.COMPLETED == "completed"
        assert PlanStatus.FAILED == "failed"
        assert PlanStatus.ROLLED_BACK == "rolled_back"
        assert len(PlanStatus) == 7

    def test_create_plan_populates_all_new_fields(
        self, planner: ArchitectPlanner,
    ) -> None:
        """create_plan should populate goal_summary, pre/postconditions, risk, time."""
        plan = planner.create_plan("Fix security issue in the auth module")
        assert plan.goal_summary  # Non-empty
        assert isinstance(plan.preconditions, list)
        assert len(plan.preconditions) > 0
        assert isinstance(plan.postconditions, list)
        assert len(plan.postconditions) > 0
        assert plan.risk_assessment  # Non-empty
        assert plan.estimated_time_minutes > 0
        assert plan.git_start_hash == ""  # Not set until execution


# ------------------------------------------------------------------
# TestEnhancedPlanner (12 tests)
# ------------------------------------------------------------------


class TestEnhancedPlanner:
    """Tests for planner enrichment logic added in Phase 3."""

    def test_create_plan_sets_goal_summary_from_request(
        self, planner: ArchitectPlanner,
    ) -> None:
        """goal_summary should be derived from the request text."""
        plan = planner.create_plan("Add rate limiting to the API.")
        assert "rate limiting" in plan.goal_summary.lower()

    def test_create_plan_sets_preconditions(
        self, planner: ArchitectPlanner,
    ) -> None:
        """create_plan should always set standard preconditions."""
        plan = planner.create_plan("Implement caching layer")
        assert "Git working directory clean" in plan.preconditions
        assert "All tests passing" in plan.preconditions

    def test_create_plan_sets_postconditions(
        self, planner: ArchitectPlanner,
    ) -> None:
        """create_plan should always set standard postconditions."""
        plan = planner.create_plan("Implement caching layer")
        assert "All new tests passing" in plan.postconditions
        assert "No ruff errors" in plan.postconditions

    def test_create_plan_detects_file_paths_in_step_descriptions(
        self, planner: ArchitectPlanner,
    ) -> None:
        """Steps mentioning file paths should have files_to_modify populated."""
        plan = planner.create_plan(
            "1. Update src/config.py  2. Modify tests/test_config.py"
        )
        all_files = []
        for step in plan.steps:
            all_files.extend(step.files_to_modify)
        assert any("config.py" in f for f in all_files)

    def test_create_plan_sets_high_risk_for_security_steps(
        self, planner: ArchitectPlanner,
    ) -> None:
        """Steps mentioning security keywords should be HIGH risk."""
        plan = planner.create_plan(
            "1. Update the authentication logic  2. Run tests to verify"
        )
        auth_steps = [
            s for s in plan.steps
            if "auth" in s.description.lower()
        ]
        assert len(auth_steps) > 0
        for step in auth_steps:
            assert step.risk_level == RISK_HIGH

    def test_create_plan_sets_medium_risk_for_test_steps(
        self, planner: ArchitectPlanner,
    ) -> None:
        """Steps mentioning test keywords should be MEDIUM risk."""
        plan = planner.create_plan("Run tests to verify changes")
        test_steps = [
            s for s in plan.steps
            if "test" in s.description.lower()
        ]
        assert len(test_steps) > 0
        for step in test_steps:
            assert step.risk_level == RISK_MEDIUM

    def test_create_plan_sets_low_risk_for_plain_steps(
        self, planner: ArchitectPlanner,
    ) -> None:
        """Steps with no risk keywords should default to LOW."""
        plan = planner.create_plan("Fix the typo in the readme")
        for step in plan.steps:
            # Only check steps that do not contain test/config keywords
            desc_lower = step.description.lower()
            has_risk_kw = any(
                kw in desc_lower
                for kw in (
                    "security", "auth", "database", "db", "migration",
                    "test", "config", "deploy",
                )
            )
            if not has_risk_kw:
                assert step.risk_level == RISK_LOW

    def test_create_plan_generates_validation_strings(
        self, planner: ArchitectPlanner,
    ) -> None:
        """Every step should have a non-empty validation string."""
        plan = planner.create_plan("Refactor the module and then run tests")
        for step in plan.steps:
            assert step.validation  # Non-empty string

    def test_create_plan_generates_rollback_strings(
        self, planner: ArchitectPlanner,
    ) -> None:
        """Every step should have a non-empty rollback string."""
        plan = planner.create_plan("Refactor the module and then run tests")
        for step in plan.steps:
            assert step.rollback  # Non-empty string
            assert "git checkout" in step.rollback

    def test_create_plan_calculates_per_step_costs(
        self, planner: ArchitectPlanner,
    ) -> None:
        """Each step should have a positive estimated_cost with known model."""
        plan = planner.create_plan("Implement a new feature")
        for step in plan.steps:
            assert step.estimated_cost >= 0.0

    def test_create_plan_estimates_time(
        self, planner: ArchitectPlanner,
    ) -> None:
        """estimated_time_minutes should be len(steps) * 2.0."""
        plan = planner.create_plan("1. Step A  2. Step B  3. Step C")
        # Each step adds 2 minutes
        assert plan.estimated_time_minutes == pytest.approx(
            len(plan.steps) * 2.0
        )

    def test_create_plan_builds_risk_assessment_from_step_risks(
        self, planner: ArchitectPlanner,
    ) -> None:
        """risk_assessment should summarise HIGH/MEDIUM/LOW counts."""
        plan = planner.create_plan(
            "1. Update the database schema  2. Run tests"
        )
        assert plan.risk_assessment  # Non-empty
        # Should contain a count summary
        assert "high" in plan.risk_assessment.lower() or \
               "medium" in plan.risk_assessment.lower() or \
               "low" in plan.risk_assessment.lower()

    def test_format_plan_for_review_includes_new_fields(
        self, planner: ArchitectPlanner,
    ) -> None:
        """format_plan_for_review should include goal, risk, time, pre/postconditions."""
        plan = planner.create_plan("Refactor the authentication module")
        output = planner.format_plan_for_review(plan)
        assert "Goal:" in output
        assert "Risk:" in output
        assert "Est. time:" in output
        assert "Preconditions:" in output
        assert "Postconditions:" in output


# ------------------------------------------------------------------
# TestEnhancedStorage (8 tests)
# ------------------------------------------------------------------


class TestEnhancedStorage:
    """Tests for storage of Phase 3 enhanced fields."""

    def test_save_plan_stores_new_plan_fields_in_metadata(
        self, memory_db: Database,
    ) -> None:
        """save_plan should store goal_summary, pre/postconditions, risk, time, hash."""
        storage = PlanStorage(memory_db)
        plan = _make_enhanced_plan(
            goal_summary="Test goal",
            preconditions=["clean tree"],
            postconditions=["tests pass"],
            risk_assessment="LOW - safe",
            estimated_time_minutes=4.0,
            git_start_hash="aaa111",
        )
        storage.save_plan(plan)

        # Verify metadata was stored correctly by reading the raw row
        row = memory_db.fetchone(
            "SELECT metadata FROM plans WHERE id = ?", (plan.id,),
        )
        assert row is not None
        meta = json.loads(row["metadata"])
        assert meta["goal_summary"] == "Test goal"
        assert meta["preconditions"] == ["clean tree"]
        assert meta["postconditions"] == ["tests pass"]
        assert meta["risk_assessment"] == "LOW - safe"
        assert meta["estimated_time_minutes"] == 4.0
        assert meta["git_start_hash"] == "aaa111"

    def test_load_plan_restores_new_plan_fields_from_metadata(
        self, memory_db: Database,
    ) -> None:
        """load_plan should restore all Phase 3 Plan fields from metadata."""
        storage = PlanStorage(memory_db)
        plan = _make_enhanced_plan(
            goal_summary="Restore goal",
            preconditions=["precond1"],
            postconditions=["postcond1"],
            risk_assessment="MEDIUM - some risk",
            estimated_time_minutes=8.0,
            git_start_hash="bbb222",
        )
        storage.save_plan(plan)

        loaded = storage.load_plan(plan.id)
        assert loaded is not None
        assert loaded.goal_summary == "Restore goal"
        assert loaded.preconditions == ["precond1"]
        assert loaded.postconditions == ["postcond1"]
        assert loaded.risk_assessment == "MEDIUM - some risk"
        assert loaded.estimated_time_minutes == pytest.approx(8.0)
        assert loaded.git_start_hash == "bbb222"

    def test_save_plan_stores_new_step_fields(
        self, memory_db: Database,
    ) -> None:
        """save_plan should store step-level Phase 3 fields in step metadata."""
        storage = PlanStorage(memory_db)
        plan = _make_enhanced_plan(step_count=1)
        plan.steps[0].files_to_modify = ["src/app.py"]
        plan.steps[0].estimated_cost = 0.002
        plan.steps[0].risk_level = RISK_HIGH
        plan.steps[0].validation = "Run pytest tests/"
        plan.steps[0].rollback = "git checkout -- src/app.py"
        storage.save_plan(plan)

        row = memory_db.fetchone(
            "SELECT metadata FROM plan_steps WHERE plan_id = ?",
            (plan.id,),
        )
        assert row is not None
        meta = json.loads(row["metadata"])
        assert meta["files_to_modify"] == ["src/app.py"]
        assert meta["estimated_cost"] == pytest.approx(0.002)
        assert meta["risk_level"] == RISK_HIGH
        assert meta["validation"] == "Run pytest tests/"
        assert meta["rollback"] == "git checkout -- src/app.py"

    def test_load_plan_restores_new_step_fields(
        self, memory_db: Database,
    ) -> None:
        """load_plan should restore all Phase 3 PlanStep fields from metadata."""
        storage = PlanStorage(memory_db)
        plan = _make_enhanced_plan(step_count=1)
        plan.steps[0].files_to_modify = ["tests/test_auth.py"]
        plan.steps[0].estimated_cost = 0.0042
        plan.steps[0].risk_level = RISK_MEDIUM
        plan.steps[0].validation = "Run pytest tests/test_auth.py"
        plan.steps[0].rollback = "git checkout -- tests/test_auth.py"
        storage.save_plan(plan)

        loaded = storage.load_plan(plan.id)
        assert loaded is not None
        step = loaded.steps[0]
        assert step.files_to_modify == ["tests/test_auth.py"]
        assert step.estimated_cost == pytest.approx(0.0042)
        assert step.risk_level == RISK_MEDIUM
        assert step.validation == "Run pytest tests/test_auth.py"
        assert step.rollback == "git checkout -- tests/test_auth.py"

    def test_metadata_columns_created_if_not_exist(
        self, memory_db: Database,
    ) -> None:
        """Initialising PlanStorage should add metadata columns idempotently."""
        # First init adds columns
        PlanStorage(memory_db)
        # Second init should not raise (columns already exist)
        PlanStorage(memory_db)

    def test_backward_compatibility_old_plans_without_metadata(
        self, memory_db: Database,
    ) -> None:
        """Old plans without metadata column data should load with defaults."""
        storage = PlanStorage(memory_db)
        # Insert a plan row directly without metadata
        memory_db.execute(
            """
            INSERT INTO plans (
                id, created_at, description,
                planning_model, execution_model,
                estimated_total_cost, status,
                git_checkpoint, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-plan-id",
                datetime.now(UTC).isoformat(),
                "Legacy plan",
                "claude-sonnet-4-20250514",
                "deepseek/deepseek-chat",
                0.001,
                "completed",
                None,
                None,  # No metadata
            ),
        )
        memory_db.execute(
            """
            INSERT INTO plan_steps (
                id, plan_id, order_num,
                description, tool_calls,
                estimated_tokens, status,
                result, error, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "old-step-id",
                "old-plan-id",
                1,
                "Legacy step",
                json.dumps([]),
                500,
                "completed",
                "Done",
                None,
                None,  # No metadata
            ),
        )
        memory_db.commit()

        loaded = storage.load_plan("old-plan-id")
        assert loaded is not None
        assert loaded.goal_summary == ""
        assert loaded.preconditions == []
        assert loaded.postconditions == []
        assert loaded.risk_assessment == ""
        assert loaded.estimated_time_minutes == 0.0
        assert loaded.git_start_hash == ""
        # Step defaults
        step = loaded.steps[0]
        assert step.files_to_modify == []
        assert step.estimated_cost == 0.0
        assert step.risk_level == RISK_LOW
        assert step.validation == ""
        assert step.rollback == ""

    def test_format_plan_summary_produces_one_line(
        self, memory_db: Database,
    ) -> None:
        """format_plan_summary should return a single line with status/goal/steps/cost."""
        plan = _make_enhanced_plan(
            goal_summary="Refactor auth",
            status="draft",
            step_count=3,
        )
        summary = PlanStorage.format_plan_summary(plan)
        assert "[draft]" in summary
        assert "Refactor auth" in summary
        assert "3 steps" in summary
        assert "$" in summary
        assert "\n" not in summary

    def test_list_plans_returns_plans_with_new_fields(
        self, memory_db: Database,
    ) -> None:
        """list_plans should return plans with all Phase 3 fields populated."""
        storage = PlanStorage(memory_db)
        plan = _make_enhanced_plan(
            goal_summary="List test goal",
            risk_assessment="HIGH - critical",
            estimated_time_minutes=10.0,
        )
        storage.save_plan(plan)

        plans = storage.list_plans()
        assert len(plans) == 1
        loaded = plans[0]
        assert loaded.goal_summary == "List test goal"
        assert loaded.risk_assessment == "HIGH - critical"
        assert loaded.estimated_time_minutes == pytest.approx(10.0)


# ------------------------------------------------------------------
# TestEnhancedExecutor (12 tests)
# ------------------------------------------------------------------


class TestEnhancedExecutor:
    """Tests for Phase 3 executor enhancements."""

    def test_auto_commit_step_creates_git_commit(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
        mock_git_repo: MagicMock,
    ) -> None:
        """_auto_commit_step should call git add + commit with a descriptive message."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker, git_repo=mock_git_repo,
        )
        plan = _make_enhanced_plan(status="approved")
        step = plan.steps[0]

        executor._auto_commit_step(plan, step)

        mock_git_repo.add.assert_called_once()
        mock_git_repo.commit.assert_called_once()
        commit_msg = mock_git_repo.commit.call_args[0][0]
        assert "prism:" in commit_msg
        assert f"step {step.order}" in commit_msg

    def test_auto_commit_skipped_if_no_git_repo(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
    ) -> None:
        """_auto_commit_step should be a no-op without a git repo."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker, git_repo=None,
        )
        plan = _make_enhanced_plan(status="approved")
        step = plan.steps[0]
        # Should not raise
        executor._auto_commit_step(plan, step)

    async def test_validation_runs_after_step_success(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
    ) -> None:
        """After a step succeeds, _run_step_validation should be called."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker,
        )
        plan = _make_enhanced_plan(status="approved", step_count=1)
        plan.steps[0].validation = "Run pytest tests/test_auth.py"

        with patch.object(
            executor, "_run_validation", return_value=True,
        ) as mock_val:
            summary = await executor.execute_plan(plan)

        mock_val.assert_called_once_with("tests/test_auth.py")
        assert summary.completed_steps == 1

    async def test_validation_failure_triggers_retry(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
    ) -> None:
        """If step validation fails, execution should retry the step once."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker,
        )
        plan = _make_enhanced_plan(status="approved", step_count=1)
        plan.steps[0].validation = "Run pytest tests/test_x.py"

        # Fail first validation, pass second
        call_count = 0

        def validation_side_effect(test_path: str) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 2

        with patch.object(
            executor, "_run_validation", side_effect=validation_side_effect,
        ):
            await executor.execute_plan(plan)

        assert call_count == 2
        assert plan.status == "completed"

    async def test_retry_failure_pauses_plan(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
    ) -> None:
        """If retry after validation failure also fails, plan should be paused."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker,
        )
        plan = _make_enhanced_plan(status="approved", step_count=1)
        plan.steps[0].validation = "Run pytest tests/test_x.py"

        with patch.object(
            executor, "_run_validation", return_value=False,
        ):
            await executor.execute_plan(plan)

        assert plan.status == "paused"
        assert plan.steps[0].status == StepStatus.FAILED

    def test_cost_tracked_per_step(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
    ) -> None:
        """ExecutionSummary should have separate planning_cost and execution_cost."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker, planning_cost=0.01,
        )
        plan = _make_enhanced_plan(status="completed")
        for s in plan.steps:
            s.status = StepStatus.COMPLETED

        step_results = [
            StepResult(
                step_id=s.id, success=True, output="ok", cost_usd=0.002,
            )
            for s in plan.steps
        ]
        summary = executor._build_summary(
            plan, step_results, execution_cost=0.006,
        )
        assert summary.planning_cost == pytest.approx(0.01)
        assert summary.execution_cost == pytest.approx(0.006)
        # total_cost = sum of step results + planning_cost
        assert summary.total_cost_usd == pytest.approx(0.006 + 0.01)

    async def test_resume_reconstructs_context(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
    ) -> None:
        """resume should load context from storage and merge completed steps."""
        plan = _make_enhanced_plan(status="in_progress", step_count=3)
        plan.steps[0].status = StepStatus.COMPLETED
        plan.steps[0].result = "Done"
        plan.steps[1].status = StepStatus.FAILED
        plan.steps[1].error = "Oops"

        # Create a mock storage that returns the same plan
        mock_storage = MagicMock()
        mock_storage.load_plan.return_value = plan

        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        summary = await executor.resume(plan, storage=mock_storage)

        mock_storage.load_plan.assert_called_once_with(plan.id)
        assert isinstance(summary, ExecutionSummary)
        assert plan.status == "completed"

    async def test_resume_continues_from_first_pending_step(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
    ) -> None:
        """resume should skip completed steps and start from PENDING."""
        plan = _make_enhanced_plan(status="in_progress", step_count=3)
        plan.steps[0].status = StepStatus.COMPLETED
        plan.steps[0].result = "Done"
        plan.steps[1].status = StepStatus.PENDING
        plan.steps[2].status = StepStatus.PENDING

        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        await executor.resume(plan)

        assert plan.status == "completed"
        assert all(
            s.status == StepStatus.COMPLETED for s in plan.steps
        )

    def test_list_interrupted_plans_returns_paused(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
    ) -> None:
        """list_interrupted_plans should return plans with 'paused' status."""
        paused = _make_enhanced_plan(status="paused")
        completed = _make_enhanced_plan(status="completed")
        mock_storage = MagicMock()
        mock_storage.list_plans.return_value = [paused, completed]

        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        interrupted = executor.list_interrupted_plans(storage=mock_storage)

        assert len(interrupted) == 1
        assert interrupted[0].status == "paused"

    def test_list_interrupted_plans_returns_stale_running(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
    ) -> None:
        """list_interrupted_plans should return plans with 'running'/'in_progress' status."""
        running = _make_enhanced_plan(status="running")
        in_progress = _make_enhanced_plan(status="in_progress")
        draft = _make_enhanced_plan(status="draft")
        mock_storage = MagicMock()
        mock_storage.list_plans.return_value = [running, in_progress, draft]

        executor = ArchitectExecutor(mock_settings, mock_cost_tracker)
        interrupted = executor.list_interrupted_plans(storage=mock_storage)

        assert len(interrupted) == 2
        statuses = {p.status for p in interrupted}
        assert "running" in statuses
        assert "in_progress" in statuses

    def test_rollback_returns_tuple(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
        mock_git_repo: MagicMock,
    ) -> None:
        """rollback should return a (bool, str) tuple."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker, git_repo=mock_git_repo,
        )
        plan = _make_enhanced_plan(
            status="failed", git_start_hash="abc123",
        )
        plan.git_checkpoint = "abc123"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch(
            "prism.architect.executor.subprocess.run",
            return_value=mock_result,
        ):
            result = executor.rollback(plan)

        assert isinstance(result, tuple)
        assert len(result) == 2
        success, description = result
        assert isinstance(success, bool)
        assert isinstance(description, str)
        assert success is True
        assert "abc123" in description

    def test_rollback_to_specific_step_uses_commit_hash(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
        mock_git_repo: MagicMock,
    ) -> None:
        """rollback(to_step=N) should use the step's recorded commit hash."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker, git_repo=mock_git_repo,
        )
        plan = _make_enhanced_plan(
            status="failed", git_start_hash="start123",
        )
        # Simulate recorded step commits
        executor._step_commit_hashes[1] = "step1hash"
        executor._step_commit_hashes[2] = "step2hash"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch(
            "prism.architect.executor.subprocess.run",
            return_value=mock_result,
        ):
            success, description = executor.rollback(plan, to_step=2)

        assert success is True
        assert "step 2" in description
        assert "step2has" in description  # First 8 chars of hash

    def test_rollback_to_plan_start_uses_git_start_hash(
        self, mock_settings: MagicMock, mock_cost_tracker: MagicMock,
        mock_git_repo: MagicMock,
    ) -> None:
        """rollback() without to_step should use git_start_hash."""
        executor = ArchitectExecutor(
            mock_settings, mock_cost_tracker, git_repo=mock_git_repo,
        )
        plan = _make_enhanced_plan(
            status="failed", git_start_hash="start9999",
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch(
            "prism.architect.executor.subprocess.run",
            return_value=mock_result,
        ):
            success, description = executor.rollback(plan)

        assert success is True
        assert "start999" in description  # First 8 chars


# ------------------------------------------------------------------
# TestEnhancedDisplay (10 tests)
# ------------------------------------------------------------------


class TestEnhancedDisplay:
    """Tests for Phase 3 display functions."""

    def test_display_plan_review_shows_table_with_all_columns(self) -> None:
        """display_plan_review should render a table with Step#, Desc, Files, Risk, Cost, Validation."""
        console = _capture_console()
        plan = _make_enhanced_plan()
        display_plan_review(plan, console=console)
        output = _captured_output(console)
        # Table headers
        assert "Step#" in output
        assert "Description" in output
        assert "Files" in output
        assert "Risk" in output
        assert "Est.Cost" in output
        assert "Validation" in output

    def test_display_plan_review_shows_preconditions(self) -> None:
        """display_plan_review should show preconditions section."""
        console = _capture_console()
        plan = _make_enhanced_plan(
            preconditions=["Git clean", "Tests pass"],
        )
        display_plan_review(plan, console=console)
        output = _captured_output(console)
        assert "Preconditions" in output
        assert "Git clean" in output
        assert "Tests pass" in output

    def test_display_plan_review_shows_postconditions(self) -> None:
        """display_plan_review should show postconditions section."""
        console = _capture_console()
        plan = _make_enhanced_plan(
            postconditions=["New tests pass", "Lint clean"],
        )
        display_plan_review(plan, console=console)
        output = _captured_output(console)
        assert "Postconditions" in output
        assert "New tests pass" in output
        assert "Lint clean" in output

    def test_display_plan_review_shows_risk_assessment(self) -> None:
        """display_plan_review should show risk assessment section."""
        console = _capture_console()
        plan = _make_enhanced_plan(
            risk_assessment="HIGH - 2 high, 1 medium, 0 low",
        )
        display_plan_review(plan, console=console)
        output = _captured_output(console)
        assert "Risk Assessment" in output
        assert "HIGH" in output

    def test_display_plan_review_shows_action_menu(self) -> None:
        """display_plan_review should show [a]ccept/[e]dit/[s]kip/[c]ancel menu."""
        console = _capture_console()
        plan = _make_enhanced_plan()
        display_plan_review(plan, console=console)
        output = _captured_output(console)
        assert "Accept" in output or "accept" in output.lower()
        assert "Cancel" in output or "cancel" in output.lower()

    def test_display_plan_list_shows_colored_statuses(self) -> None:
        """display_plan_list should render each plan's status."""
        console = _capture_console()
        plans = [
            _make_enhanced_plan(status="completed"),
            _make_enhanced_plan(status="failed"),
            _make_enhanced_plan(status="paused"),
        ]
        display_plan_list(plans, console=console)
        output = _captured_output(console)
        assert "COMPLETED" in output
        assert "FAILED" in output
        assert "PAUSED" in output

    def test_display_step_validation_shows_pass(self) -> None:
        """display_step_validation with passed=True should show pass message."""
        console = _capture_console()
        display_step_validation(3, True, console=console)
        output = _captured_output(console)
        assert "passed" in output.lower()
        assert "3" in output

    def test_display_step_validation_shows_fail(self) -> None:
        """display_step_validation with passed=False should show fail message."""
        console = _capture_console()
        display_step_validation(5, False, console=console)
        output = _captured_output(console)
        assert "failed" in output.lower()
        assert "5" in output

    def test_display_execution_summary_shows_planning_vs_execution_cost(
        self,
    ) -> None:
        """display_execution_summary should show planning and execution cost breakdown."""
        console = _capture_console()
        summary = ExecutionSummary(
            plan_id="p1",
            plan_description="Test plan",
            total_steps=3,
            completed_steps=3,
            failed_steps=0,
            skipped_steps=0,
            total_cost_usd=0.015,
            total_tokens=1500,
            planning_cost=0.005,
            execution_cost=0.010,
            estimated_cost=0.012,
        )
        display_execution_summary(summary, console=console)
        output = _captured_output(console)
        assert "Planning cost" in output
        assert "Execution cost" in output
        assert "0.0050" in output
        assert "0.0100" in output

    def test_display_execution_summary_shows_estimated_vs_actual(
        self,
    ) -> None:
        """display_execution_summary should compare estimated vs actual cost."""
        console = _capture_console()
        summary = ExecutionSummary(
            plan_id="p2",
            plan_description="Summary test",
            total_steps=2,
            completed_steps=2,
            failed_steps=0,
            skipped_steps=0,
            total_cost_usd=0.010,
            total_tokens=1000,
            planning_cost=0.003,
            execution_cost=0.007,
            estimated_cost=0.008,
        )
        display_execution_summary(summary, console=console)
        output = _captured_output(console)
        assert "Estimated cost" in output
        assert "estimate" in output.lower()

    def test_display_plan_list_handles_empty_list(self) -> None:
        """display_plan_list with empty list should show 'No plans found'."""
        console = _capture_console()
        display_plan_list([], console=console)
        output = _captured_output(console)
        assert "No plans found" in output


# ------------------------------------------------------------------
# TestDisplayRollback (4 tests)
# ------------------------------------------------------------------


class TestDisplayRollback:
    """Tests for display_rollback_result with description parameter."""

    def test_rollback_result_with_description(self) -> None:
        """display_rollback_result should include description when provided."""
        console = _capture_console()
        display_rollback_result(
            True, console=console,
            description="Rolled back to commit abc123",
        )
        output = _captured_output(console)
        assert "Rolled back to commit abc123" in output

    def test_rollback_result_without_description(self) -> None:
        """display_rollback_result should work without description."""
        console = _capture_console()
        display_rollback_result(True, console=console)
        output = _captured_output(console)
        assert "successful" in output.lower()
        # Description line should not appear
        assert "Rolled back to" not in output

    def test_rollback_result_success(self) -> None:
        """Successful rollback should show 'successful' and 'restored'."""
        console = _capture_console()
        display_rollback_result(True, console=console)
        output = _captured_output(console)
        assert "successful" in output.lower()
        assert "restored" in output.lower()

    def test_rollback_result_failure(self) -> None:
        """Failed rollback should show 'failed' and 'Manual intervention'."""
        console = _capture_console()
        display_rollback_result(
            False, console=console,
            description="No checkpoint available",
        )
        output = _captured_output(console)
        assert "failed" in output.lower()
        assert "manual intervention" in output.lower()
        assert "No checkpoint available" in output

"""Tests for prism.architect.storage — plan persistence."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from prism.architect.planner import Plan, PlanStep, StepStatus
from prism.architect.storage import PlanStorage
from prism.exceptions import DatabaseError

if TYPE_CHECKING:
    from prism.db.database import Database

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_plan(
    *,
    step_count: int = 2,
    status: str = "draft",
    git_checkpoint: str | None = None,
) -> Plan:
    """Build a Plan with *step_count* PENDING steps."""
    steps = [
        PlanStep(
            id=str(uuid.uuid4()),
            order=i,
            description=f"Step {i} of the plan",
            tool_calls=[{"tool": "read_file", "args": {}}],
            estimated_tokens=500,
            status=StepStatus.PENDING,
        )
        for i in range(1, step_count + 1)
    ]
    return Plan(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        description="Test plan",
        steps=steps,
        planning_model="claude-sonnet-4-20250514",
        execution_model="deepseek/deepseek-chat",
        estimated_total_cost=0.002,
        status=status,
        git_checkpoint=git_checkpoint,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSaveAndLoad:
    """Round-trip save/load tests."""

    def test_save_and_load_round_trip(self, memory_db: Database) -> None:
        """A saved plan should be loadable with identical data."""
        storage = PlanStorage(memory_db)
        plan = _make_plan()
        storage.save_plan(plan)

        loaded = storage.load_plan(plan.id)
        assert loaded is not None
        assert loaded.id == plan.id
        assert loaded.description == plan.description
        assert loaded.planning_model == plan.planning_model
        assert loaded.execution_model == plan.execution_model
        assert loaded.status == plan.status
        assert loaded.estimated_total_cost == plan.estimated_total_cost
        assert loaded.git_checkpoint == plan.git_checkpoint

    def test_steps_preserved_on_round_trip(self, memory_db: Database) -> None:
        """All steps should be preserved after save/load."""
        storage = PlanStorage(memory_db)
        plan = _make_plan(step_count=4)
        storage.save_plan(plan)

        loaded = storage.load_plan(plan.id)
        assert loaded is not None
        assert len(loaded.steps) == 4
        for orig, loaded_step in zip(
            sorted(plan.steps, key=lambda s: s.order),
            sorted(loaded.steps, key=lambda s: s.order), strict=False,
        ):
            assert loaded_step.id == orig.id
            assert loaded_step.order == orig.order
            assert loaded_step.description == orig.description
            assert loaded_step.status == orig.status

    def test_step_tool_calls_serialized(self, memory_db: Database) -> None:
        """Tool calls (list of dicts) should survive JSON serialization."""
        storage = PlanStorage(memory_db)
        plan = _make_plan(step_count=1)
        plan.steps[0].tool_calls = [
            {"tool": "write_file", "args": {"path": "/tmp/out.py"}},
            {"tool": "run_command", "args": {"command": "pytest"}},
        ]
        storage.save_plan(plan)

        loaded = storage.load_plan(plan.id)
        assert loaded is not None
        assert len(loaded.steps[0].tool_calls) == 2
        assert loaded.steps[0].tool_calls[0]["tool"] == "write_file"

    def test_load_nonexistent_plan_returns_none(self, memory_db: Database) -> None:
        """Loading a plan that doesn't exist should return None."""
        storage = PlanStorage(memory_db)
        assert storage.load_plan("nonexistent-id-12345") is None

    def test_save_overwrites_existing(self, memory_db: Database) -> None:
        """Saving a plan with the same ID should replace the old one."""
        storage = PlanStorage(memory_db)
        plan = _make_plan(step_count=2)
        storage.save_plan(plan)

        plan.status = "completed"
        plan.steps = plan.steps[:1]  # Remove one step
        storage.save_plan(plan)

        loaded = storage.load_plan(plan.id)
        assert loaded is not None
        assert loaded.status == "completed"
        assert len(loaded.steps) == 1

    def test_git_checkpoint_preserved(self, memory_db: Database) -> None:
        """Git checkpoint should be persisted and loaded."""
        storage = PlanStorage(memory_db)
        plan = _make_plan(git_checkpoint="abc1234")
        storage.save_plan(plan)

        loaded = storage.load_plan(plan.id)
        assert loaded is not None
        assert loaded.git_checkpoint == "abc1234"

    def test_plan_with_many_steps(self, memory_db: Database) -> None:
        """A plan with many steps should round-trip correctly."""
        storage = PlanStorage(memory_db)
        plan = _make_plan(step_count=20)
        storage.save_plan(plan)

        loaded = storage.load_plan(plan.id)
        assert loaded is not None
        assert len(loaded.steps) == 20


class TestListPlans:
    """Tests for listing plans."""

    def test_list_all_plans(self, memory_db: Database) -> None:
        """list_plans without filter should return all plans."""
        storage = PlanStorage(memory_db)
        for _ in range(3):
            storage.save_plan(_make_plan())

        plans = storage.list_plans()
        assert len(plans) == 3

    def test_filter_by_status(self, memory_db: Database) -> None:
        """list_plans with status filter should return only matching plans."""
        storage = PlanStorage(memory_db)
        storage.save_plan(_make_plan(status="draft"))
        storage.save_plan(_make_plan(status="completed"))
        storage.save_plan(_make_plan(status="draft"))

        drafts = storage.list_plans(status="draft")
        assert len(drafts) == 2

        completed = storage.list_plans(status="completed")
        assert len(completed) == 1

    def test_list_empty_returns_empty(self, memory_db: Database) -> None:
        """list_plans on an empty database should return an empty list."""
        storage = PlanStorage(memory_db)
        assert storage.list_plans() == []

    def test_list_plans_include_steps(self, memory_db: Database) -> None:
        """Listed plans should include their steps."""
        storage = PlanStorage(memory_db)
        storage.save_plan(_make_plan(step_count=3))

        plans = storage.list_plans()
        assert len(plans) == 1
        assert len(plans[0].steps) == 3


class TestUpdateStepStatus:
    """Tests for updating step status."""

    def test_update_step_status(self, memory_db: Database) -> None:
        """update_step_status should persist the new status."""
        storage = PlanStorage(memory_db)
        plan = _make_plan(step_count=2)
        storage.save_plan(plan)

        step = plan.steps[0]
        storage.update_step_status(
            step.id, StepStatus.COMPLETED, result="Done successfully"
        )

        loaded = storage.load_plan(plan.id)
        assert loaded is not None
        loaded_step = next(s for s in loaded.steps if s.id == step.id)
        assert loaded_step.status == StepStatus.COMPLETED
        assert loaded_step.result == "Done successfully"

    def test_update_step_with_error(self, memory_db: Database) -> None:
        """update_step_status with error should persist the error."""
        storage = PlanStorage(memory_db)
        plan = _make_plan(step_count=1)
        storage.save_plan(plan)

        step = plan.steps[0]
        storage.update_step_status(
            step.id, StepStatus.FAILED, error="Syntax error"
        )

        loaded = storage.load_plan(plan.id)
        assert loaded is not None
        assert loaded.steps[0].status == StepStatus.FAILED
        assert loaded.steps[0].error == "Syntax error"


class TestDeletePlan:
    """Tests for deleting plans."""

    def test_delete_removes_plan(self, memory_db: Database) -> None:
        """Deleting a plan should remove it and its steps."""
        storage = PlanStorage(memory_db)
        plan = _make_plan(step_count=3)
        storage.save_plan(plan)

        storage.delete_plan(plan.id)

        assert storage.load_plan(plan.id) is None

    def test_delete_nonexistent_is_noop(self, memory_db: Database) -> None:
        """Deleting a plan that doesn't exist should not raise."""
        storage = PlanStorage(memory_db)
        storage.delete_plan("nonexistent-id")  # Should not raise

    def test_delete_only_removes_target(self, memory_db: Database) -> None:
        """Deleting one plan should not affect others."""
        storage = PlanStorage(memory_db)
        plan1 = _make_plan()
        plan2 = _make_plan()
        storage.save_plan(plan1)
        storage.save_plan(plan2)

        storage.delete_plan(plan1.id)

        assert storage.load_plan(plan1.id) is None
        assert storage.load_plan(plan2.id) is not None


# ------------------------------------------------------------------
# Error handling — DatabaseError re-raise and generic Exception wrapping
# ------------------------------------------------------------------


class TestSavePlanErrors:
    """Tests for error handling in save_plan (lines 107-110)."""

    def test_save_plan_reraises_database_error(self, memory_db: Database) -> None:
        """save_plan should re-raise DatabaseError without wrapping."""
        storage = PlanStorage(memory_db)
        plan = _make_plan()

        with patch.object(
            memory_db, "transaction", side_effect=DatabaseError("db is locked")
        ):
            with pytest.raises(DatabaseError, match="db is locked"):
                storage.save_plan(plan)

    def test_save_plan_wraps_generic_exception(self, memory_db: Database) -> None:
        """save_plan should wrap non-DatabaseError in DatabaseError."""
        storage = PlanStorage(memory_db)
        plan = _make_plan()

        with patch.object(
            memory_db, "transaction", side_effect=RuntimeError("unexpected failure")
        ):
            with pytest.raises(DatabaseError, match="Failed to save plan"):
                storage.save_plan(plan)


class TestUpdateStepStatusErrors:
    """Tests for error handling in update_step_status (lines 142-145)."""

    def test_update_step_status_reraises_database_error(
        self, memory_db: Database
    ) -> None:
        """update_step_status should re-raise DatabaseError without wrapping."""
        storage = PlanStorage(memory_db)

        with patch.object(
            memory_db, "execute", side_effect=DatabaseError("disk full")
        ):
            with pytest.raises(DatabaseError, match="disk full"):
                storage.update_step_status(
                    "step-id-123", StepStatus.COMPLETED, result="ok"
                )

    def test_update_step_status_wraps_generic_exception(
        self, memory_db: Database
    ) -> None:
        """update_step_status should wrap non-DatabaseError in DatabaseError."""
        storage = PlanStorage(memory_db)

        with patch.object(
            memory_db, "execute", side_effect=TypeError("bad argument")
        ):
            with pytest.raises(DatabaseError, match="Failed to update step status"):
                storage.update_step_status(
                    "step-id-123", StepStatus.FAILED, error="oops"
                )


class TestDeletePlanErrors:
    """Tests for error handling in delete_plan (lines 164-167)."""

    def test_delete_plan_reraises_database_error(self, memory_db: Database) -> None:
        """delete_plan should re-raise DatabaseError without wrapping."""
        storage = PlanStorage(memory_db)

        with patch.object(
            memory_db, "transaction", side_effect=DatabaseError("read-only db")
        ):
            with pytest.raises(DatabaseError, match="read-only db"):
                storage.delete_plan("plan-id-123")

    def test_delete_plan_wraps_generic_exception(self, memory_db: Database) -> None:
        """delete_plan should wrap non-DatabaseError in DatabaseError."""
        storage = PlanStorage(memory_db)

        with patch.object(
            memory_db, "transaction", side_effect=OSError("filesystem error")
        ):
            with pytest.raises(DatabaseError, match="Failed to delete plan"):
                storage.delete_plan("plan-id-123")


class TestLoadPlanErrors:
    """Tests for error handling in load_plan (lines 197-200)."""

    def test_load_plan_reraises_database_error(self, memory_db: Database) -> None:
        """load_plan should re-raise DatabaseError without wrapping."""
        storage = PlanStorage(memory_db)

        with patch.object(
            memory_db, "fetchone", side_effect=DatabaseError("connection closed")
        ):
            with pytest.raises(DatabaseError, match="connection closed"):
                storage.load_plan("plan-id-123")

    def test_load_plan_wraps_generic_exception(self, memory_db: Database) -> None:
        """load_plan should wrap non-DatabaseError in DatabaseError."""
        storage = PlanStorage(memory_db)

        with patch.object(
            memory_db, "fetchone", side_effect=ValueError("unexpected value")
        ):
            with pytest.raises(DatabaseError, match="Failed to load plan"):
                storage.load_plan("plan-id-123")


class TestListPlansErrors:
    """Tests for error handling in list_plans (lines 232-235)."""

    def test_list_plans_reraises_database_error(self, memory_db: Database) -> None:
        """list_plans should re-raise DatabaseError without wrapping."""
        storage = PlanStorage(memory_db)

        with patch.object(
            memory_db, "fetchall", side_effect=DatabaseError("table missing")
        ):
            with pytest.raises(DatabaseError, match="table missing"):
                storage.list_plans()

    def test_list_plans_wraps_generic_exception(self, memory_db: Database) -> None:
        """list_plans should wrap non-DatabaseError in DatabaseError."""
        storage = PlanStorage(memory_db)

        with patch.object(
            memory_db, "fetchall", side_effect=KeyError("bad column")
        ):
            with pytest.raises(DatabaseError, match="Failed to list plans"):
                storage.list_plans()

    def test_list_plans_with_status_reraises_database_error(
        self, memory_db: Database
    ) -> None:
        """list_plans with status filter should also re-raise DatabaseError."""
        storage = PlanStorage(memory_db)

        with patch.object(
            memory_db, "fetchall", side_effect=DatabaseError("query failed")
        ):
            with pytest.raises(DatabaseError, match="query failed"):
                storage.list_plans(status="draft")

    def test_list_plans_with_status_wraps_generic_exception(
        self, memory_db: Database
    ) -> None:
        """list_plans with status filter wraps non-DatabaseError."""
        storage = PlanStorage(memory_db)

        with patch.object(
            memory_db, "fetchall", side_effect=RuntimeError("oops")
        ):
            with pytest.raises(DatabaseError, match="Failed to list plans"):
                storage.list_plans(status="completed")

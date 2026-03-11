"""Tests for prism.architect.planner — plan generation."""

from __future__ import annotations

import uuid

import pytest

from prism.architect.planner import ArchitectPlanner, Plan, StepStatus


class TestCreatePlan:
    """Tests for ArchitectPlanner.create_plan."""

    def test_simple_request_creates_plan(self, planner: ArchitectPlanner) -> None:
        """A simple request should produce a plan with at least one step."""
        plan = planner.create_plan("Fix the typo in README.md")
        assert isinstance(plan, Plan)
        assert len(plan.steps) >= 1
        assert plan.status == "draft"

    def test_multi_step_request(self, planner: ArchitectPlanner) -> None:
        """A request with multi-step indicators should produce multiple steps."""
        plan = planner.create_plan(
            "1. Read the config file  2. Update the timeout value  3. Run tests"
        )
        assert len(plan.steps) >= 3

    def test_refactor_request_adds_analysis_and_test_steps(
        self, planner: ArchitectPlanner
    ) -> None:
        """A refactor request should include analysis and test steps."""
        plan = planner.create_plan("Refactor the authentication module")
        descriptions = [s.description.lower() for s in plan.steps]
        assert any("analyze" in d for d in descriptions)
        assert any("test" in d for d in descriptions)

    def test_empty_request_raises(self, planner: ArchitectPlanner) -> None:
        """An empty request should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            planner.create_plan("")

    def test_whitespace_only_request_raises(self, planner: ArchitectPlanner) -> None:
        """A whitespace-only request should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            planner.create_plan("   \n  ")

    def test_plan_has_unique_id(self, planner: ArchitectPlanner) -> None:
        """Each plan should have a unique UUID."""
        plan1 = planner.create_plan("Do task A")
        plan2 = planner.create_plan("Do task B")
        assert plan1.id != plan2.id
        # Verify they're valid UUIDs
        uuid.UUID(plan1.id)
        uuid.UUID(plan2.id)

    def test_step_ids_are_unique(self, planner: ArchitectPlanner) -> None:
        """All step IDs within a plan should be unique."""
        plan = planner.create_plan(
            "1. Read file  2. Modify it  3. Write file  4. Run tests"
        )
        ids = [s.id for s in plan.steps]
        assert len(ids) == len(set(ids))

    def test_step_ordering_is_sequential(self, planner: ArchitectPlanner) -> None:
        """Steps should have sequential order values starting at 1."""
        plan = planner.create_plan(
            "1. First task  2. Second task  3. Third task"
        )
        orders = sorted(s.order for s in plan.steps)
        assert orders == list(range(1, len(orders) + 1))

    def test_all_steps_start_pending(self, planner: ArchitectPlanner) -> None:
        """All steps in a new plan should be PENDING."""
        plan = planner.create_plan("Implement feature X")
        for step in plan.steps:
            assert step.status == StepStatus.PENDING

    def test_plan_description_matches_request(self, planner: ArchitectPlanner) -> None:
        """Plan description should match the original request (stripped)."""
        request = "  Add logging to the API handler  "
        plan = planner.create_plan(request)
        assert plan.description == request.strip()

    def test_plan_created_at_is_set(self, planner: ArchitectPlanner) -> None:
        """Plan should have a created_at timestamp."""
        plan = planner.create_plan("Some task")
        assert plan.created_at
        assert "T" in plan.created_at  # ISO 8601 format

    def test_context_is_optional(self, planner: ArchitectPlanner) -> None:
        """create_plan should work without context."""
        plan = planner.create_plan("Do something")
        assert plan is not None

    def test_multi_file_request_with_conjunction(self, planner: ArchitectPlanner) -> None:
        """Multi-step request using 'and then' should split correctly."""
        plan = planner.create_plan(
            "Read the config and then update the database and then restart the service"
        )
        assert len(plan.steps) >= 3


class TestPlanModels:
    """Tests for Plan and PlanStep model defaults."""

    def test_planning_model_default(self, planner: ArchitectPlanner) -> None:
        """Planning model should default to claude-sonnet-4-20250514."""
        plan = planner.create_plan("task")
        assert plan.planning_model == "claude-sonnet-4-20250514"

    def test_execution_model_default(self, planner: ArchitectPlanner) -> None:
        """Execution model should default to deepseek/deepseek-chat."""
        plan = planner.create_plan("task")
        assert plan.execution_model == "deepseek/deepseek-chat"

    def test_custom_models(self, mock_settings, mock_cost_tracker) -> None:
        """Custom model overrides should be respected."""
        planner = ArchitectPlanner(
            mock_settings,
            mock_cost_tracker,
            planning_model="gpt-4o",
            execution_model="gpt-4o-mini",
        )
        plan = planner.create_plan("task")
        assert plan.planning_model == "gpt-4o"
        assert plan.execution_model == "gpt-4o-mini"


class TestCostEstimation:
    """Tests for cost estimation."""

    def test_estimate_cost_positive(self, planner: ArchitectPlanner) -> None:
        """Cost estimate for deepseek model should be positive."""
        plan = planner.create_plan("Implement a feature with 3 files")
        assert plan.estimated_total_cost > 0.0

    def test_estimate_cost_matches_plan(self, planner: ArchitectPlanner) -> None:
        """estimate_cost should return the same value as the plan's field."""
        plan = planner.create_plan("Do something")
        assert planner.estimate_cost(plan) == plan.estimated_total_cost

    def test_estimate_cost_unknown_model(self, mock_settings, mock_cost_tracker) -> None:
        """Unknown model should result in zero cost."""
        planner = ArchitectPlanner(
            mock_settings,
            mock_cost_tracker,
            execution_model="unknown/nonexistent-model",
        )
        plan = planner.create_plan("Do something")
        assert plan.estimated_total_cost == 0.0


class TestFormatPlanForReview:
    """Tests for plan formatting."""

    def test_format_includes_description(self, planner: ArchitectPlanner) -> None:
        """Formatted output should contain the plan description."""
        plan = planner.create_plan("Refactor the user service")
        output = planner.format_plan_for_review(plan)
        assert "Refactor the user service" in output

    def test_format_includes_step_descriptions(self, planner: ArchitectPlanner) -> None:
        """Formatted output should contain step text."""
        plan = planner.create_plan("1. Read file  2. Modify file")
        output = planner.format_plan_for_review(plan)
        assert "PENDING" in output

    def test_format_includes_models(self, planner: ArchitectPlanner) -> None:
        """Formatted output should mention both models."""
        plan = planner.create_plan("task")
        output = planner.format_plan_for_review(plan)
        assert plan.planning_model in output
        assert plan.execution_model in output

    def test_format_truncates_long_descriptions(self, planner: ArchitectPlanner) -> None:
        """Descriptions longer than 80 chars should be truncated."""
        long_desc = "A" * 200
        plan = planner.create_plan(long_desc)
        output = planner.format_plan_for_review(plan)
        assert "..." in output

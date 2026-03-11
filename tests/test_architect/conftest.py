"""Shared fixtures for architect module tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prism.architect.planner import ArchitectPlanner, Plan, PlanStep, StepStatus
from prism.db.database import Database

# ------------------------------------------------------------------
# Database fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def memory_db() -> Database:
    """Return an initialised in-memory database with all migrations applied."""
    db = Database(Path(":memory:"))
    db.initialize()
    return db


# ------------------------------------------------------------------
# Mock settings / cost tracker
# ------------------------------------------------------------------


@pytest.fixture()
def mock_settings() -> MagicMock:
    """Create a mock Settings with sensible defaults."""
    settings = MagicMock(spec_set=["get", "config", "project_root"])
    settings.get.return_value = None
    settings.project_root = Path("/tmp/fake-project")
    return settings


@pytest.fixture()
def mock_cost_tracker() -> MagicMock:
    """Create a mock CostTracker."""
    tracker = MagicMock(spec_set=["track", "get_budget_remaining"])
    tracker.get_budget_remaining.return_value = None
    return tracker


# ------------------------------------------------------------------
# Mock git repo
# ------------------------------------------------------------------


@pytest.fixture()
def mock_git_repo() -> MagicMock:
    """Create a mock git repository object."""
    repo = MagicMock()
    repo.root = Path("/tmp/fake-project")
    repo.add.return_value = None
    repo.commit.return_value = "abc1234"
    return repo


# ------------------------------------------------------------------
# Sample plan data
# ------------------------------------------------------------------


def _make_step(
    order: int,
    description: str = "",
    status: StepStatus = StepStatus.PENDING,
    *,
    result: str | None = None,
    error: str | None = None,
) -> PlanStep:
    """Helper to build a PlanStep with defaults."""
    return PlanStep(
        id=str(uuid.uuid4()),
        order=order,
        description=description or f"Step {order} description",
        tool_calls=[{"tool": "read_file", "args": {"path": "/tmp/f.py"}}],
        estimated_tokens=500,
        status=status,
        result=result,
        error=error,
    )


@pytest.fixture()
def sample_step() -> PlanStep:
    """A single sample PlanStep."""
    return _make_step(1, "Read the main module")


@pytest.fixture()
def sample_plan() -> Plan:
    """A sample Plan with three steps, all PENDING."""
    return Plan(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        description="Refactor the authentication module",
        steps=[
            _make_step(1, "Analyze current auth implementation"),
            _make_step(2, "Refactor auth module"),
            _make_step(3, "Run tests to verify changes"),
        ],
        planning_model="claude-sonnet-4-20250514",
        execution_model="deepseek/deepseek-chat",
        estimated_total_cost=0.002,
        status="draft",
        git_checkpoint=None,
    )


@pytest.fixture()
def approved_plan(sample_plan: Plan) -> Plan:
    """A sample Plan in 'approved' status, ready for execution."""
    sample_plan.status = "approved"
    return sample_plan


@pytest.fixture()
def completed_plan() -> Plan:
    """A Plan where all steps are COMPLETED."""
    return Plan(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        description="Add logging to utils module",
        steps=[
            _make_step(1, "Add import for structlog", StepStatus.COMPLETED, result="Done"),
            _make_step(2, "Add logger calls", StepStatus.COMPLETED, result="Done"),
        ],
        planning_model="claude-sonnet-4-20250514",
        execution_model="deepseek/deepseek-chat",
        estimated_total_cost=0.001,
        status="completed",
        git_checkpoint="def5678",
    )


@pytest.fixture()
def failed_plan() -> Plan:
    """A Plan where one step FAILED."""
    return Plan(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        description="Migrate database schema",
        steps=[
            _make_step(1, "Backup database", StepStatus.COMPLETED, result="Backup created"),
            _make_step(2, "Apply migration", StepStatus.FAILED, error="Syntax error in SQL"),
            _make_step(3, "Verify migration"),
        ],
        planning_model="claude-sonnet-4-20250514",
        execution_model="deepseek/deepseek-chat",
        estimated_total_cost=0.003,
        status="failed",
        git_checkpoint="aaa1111",
    )


@pytest.fixture()
def planner(mock_settings: MagicMock, mock_cost_tracker: MagicMock) -> ArchitectPlanner:
    """Create an ArchitectPlanner with mock dependencies."""
    return ArchitectPlanner(mock_settings, mock_cost_tracker)

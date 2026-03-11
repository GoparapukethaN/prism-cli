"""Shared fixtures for database tests."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from prism.db.database import Database
from prism.db.models import (
    ComplexityTier,
    CostEntry,
    Outcome,
    RoutingDecision,
    Session,
    ToolExecution,
)


@pytest.fixture()
def memory_db() -> Database:
    """Return an initialised in-memory database (no disk I/O)."""
    db = Database(Path(":memory:"))
    db.initialize()
    return db


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Database:
    """Return an initialised database backed by a file in *tmp_path*."""
    db = Database(tmp_path / "test.db")
    db.initialize()
    return db


@pytest.fixture()
def sample_routing_decision() -> RoutingDecision:
    """Return a fully-populated RoutingDecision for testing."""
    return RoutingDecision(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        session_id=str(uuid.uuid4()),
        prompt_hash="a" * 64,
        complexity_tier=ComplexityTier.MEDIUM,
        complexity_score=0.55,
        model_selected="gpt-4o",
        model_actual=None,
        fallback_chain=json.dumps(["gpt-4o", "claude-sonnet-4-20250514"]),
        estimated_cost=0.025,
        actual_cost=None,
        input_tokens=500,
        output_tokens=200,
        cached_tokens=0,
        latency_ms=1200.5,
        outcome=Outcome.UNKNOWN,
        features=json.dumps({"code_ratio": 0.3, "question_marks": 1}),
        error=None,
    )


@pytest.fixture()
def sample_cost_entry() -> CostEntry:
    """Return a fully-populated CostEntry for testing."""
    return CostEntry(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        session_id=str(uuid.uuid4()),
        model_id="gpt-4o",
        provider="openai",
        input_tokens=500,
        output_tokens=200,
        cached_tokens=0,
        cost_usd=0.025,
        complexity_tier=ComplexityTier.MEDIUM,
    )


@pytest.fixture()
def sample_session() -> Session:
    """Return a fully-populated Session for testing."""
    now = datetime.now(UTC).isoformat()
    return Session(
        id=str(uuid.uuid4()),
        created_at=now,
        updated_at=now,
        project_root="/tmp/test-project",
        total_cost=0.0,
        total_requests=0,
        summary=None,
        active=True,
    )


@pytest.fixture()
def sample_tool_execution() -> ToolExecution:
    """Return a fully-populated ToolExecution for testing."""
    return ToolExecution(
        id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        session_id=str(uuid.uuid4()),
        tool_name="read_file",
        arguments=json.dumps({"path": "/tmp/test.py"}),
        result_success=True,
        result_error=None,
        duration_ms=45.2,
        metadata=json.dumps({"bytes_read": 1024}),
    )

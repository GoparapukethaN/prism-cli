"""Tests for prism.db.queries — every query function with assertions."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from prism.db.models import (
    ComplexityTier,
    CostEntry,
    Outcome,
    RoutingDecision,
    ToolExecution,
)
from prism.db.queries import (
    cleanup_old_data,
    create_session,
    get_cost_breakdown,
    get_daily_cost,
    get_free_tier_remaining,
    get_model_success_rate,
    get_monthly_cost,
    get_routing_history,
    get_session_cost,
    increment_free_tier_usage,
    is_rate_limited,
    save_cost_entry,
    save_routing_decision,
    save_tool_execution,
    set_rate_limited,
    update_provider_status,
    update_routing_outcome,
    update_session,
)

if TYPE_CHECKING:
    from prism.db.database import Database

# =====================================================================
# Helpers
# =====================================================================


def _make_routing_decision(**overrides: object) -> RoutingDecision:
    """Create a RoutingDecision with sensible defaults, applying overrides."""
    defaults: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "session_id": str(uuid.uuid4()),
        "prompt_hash": "b" * 64,
        "complexity_tier": ComplexityTier.MEDIUM,
        "complexity_score": 0.5,
        "model_selected": "gpt-4o",
        "model_actual": None,
        "fallback_chain": json.dumps(["gpt-4o"]),
        "estimated_cost": 0.01,
        "actual_cost": None,
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_tokens": 0,
        "latency_ms": 500.0,
        "outcome": Outcome.UNKNOWN,
        "features": json.dumps({"code_ratio": 0.1}),
        "error": None,
    }
    defaults.update(overrides)
    return RoutingDecision(**defaults)


def _make_cost_entry(**overrides: object) -> CostEntry:
    defaults: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "session_id": str(uuid.uuid4()),
        "model_id": "gpt-4o",
        "provider": "openai",
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_tokens": 0,
        "cost_usd": 0.005,
        "complexity_tier": ComplexityTier.MEDIUM,
    }
    defaults.update(overrides)
    return CostEntry(**defaults)


def _make_tool_execution(**overrides: object) -> ToolExecution:
    defaults: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "session_id": str(uuid.uuid4()),
        "tool_name": "read_file",
        "arguments": json.dumps({"path": "/tmp/x"}),
        "result_success": True,
        "result_error": None,
        "duration_ms": 10.0,
        "metadata": None,
    }
    defaults.update(overrides)
    return ToolExecution(**defaults)


# =====================================================================
# Routing decisions
# =====================================================================


class TestSaveRoutingDecision:
    def test_save_and_retrieve(
        self,
        memory_db: Database,
        sample_routing_decision: RoutingDecision,
    ) -> None:
        save_routing_decision(memory_db, sample_routing_decision)
        row = memory_db.fetchone(
            "SELECT * FROM routing_decisions WHERE id = ?",
            (sample_routing_decision.id,),
        )
        assert row is not None
        assert row["model_selected"] == sample_routing_decision.model_selected
        assert row["complexity_tier"] == sample_routing_decision.complexity_tier.value

    def test_duplicate_id_raises(
        self,
        memory_db: Database,
        sample_routing_decision: RoutingDecision,
    ) -> None:
        save_routing_decision(memory_db, sample_routing_decision)
        with pytest.raises(Exception):
            save_routing_decision(memory_db, sample_routing_decision)


class TestUpdateRoutingOutcome:
    def test_update_outcome(self, memory_db: Database) -> None:
        decision = _make_routing_decision()
        save_routing_decision(memory_db, decision)
        update_routing_outcome(memory_db, decision.id, Outcome.ACCEPTED, actual_cost=0.03)

        row = memory_db.fetchone(
            "SELECT outcome, actual_cost FROM routing_decisions WHERE id = ?",
            (decision.id,),
        )
        assert row is not None
        assert row["outcome"] == "accepted"
        assert row["actual_cost"] == pytest.approx(0.03)

    def test_update_outcome_without_cost(self, memory_db: Database) -> None:
        decision = _make_routing_decision()
        save_routing_decision(memory_db, decision)
        update_routing_outcome(memory_db, decision.id, Outcome.REJECTED)

        row = memory_db.fetchone(
            "SELECT outcome, actual_cost FROM routing_decisions WHERE id = ?",
            (decision.id,),
        )
        assert row is not None
        assert row["outcome"] == "rejected"
        assert row["actual_cost"] is None


class TestGetRoutingHistory:
    def test_empty_history(self, memory_db: Database) -> None:
        history = get_routing_history(memory_db)
        assert history == []

    def test_returns_most_recent_first(self, memory_db: Database) -> None:
        session_id = str(uuid.uuid4())
        for i in range(3):
            d = _make_routing_decision(
                created_at=f"2025-01-0{i + 1}T00:00:00+00:00",
                session_id=session_id,
            )
            save_routing_decision(memory_db, d)

        history = get_routing_history(memory_db, limit=10)
        assert len(history) == 3
        # Most recent first
        assert history[0].created_at > history[1].created_at

    def test_filter_by_session(self, memory_db: Database) -> None:
        s1 = str(uuid.uuid4())
        s2 = str(uuid.uuid4())
        save_routing_decision(memory_db, _make_routing_decision(session_id=s1))
        save_routing_decision(memory_db, _make_routing_decision(session_id=s2))

        history = get_routing_history(memory_db, session_id=s1)
        assert len(history) == 1
        assert history[0].session_id == s1

    def test_limit_honoured(self, memory_db: Database) -> None:
        for _ in range(5):
            save_routing_decision(memory_db, _make_routing_decision())
        history = get_routing_history(memory_db, limit=2)
        assert len(history) == 2


class TestGetModelSuccessRate:
    def test_insufficient_data_returns_0_5(self, memory_db: Database) -> None:
        rate = get_model_success_rate(memory_db, "gpt-4o", ComplexityTier.MEDIUM, min_entries=10)
        assert rate == 0.5

    def test_all_accepted(self, memory_db: Database) -> None:
        for _ in range(12):
            d = _make_routing_decision(
                model_selected="gpt-4o",
                complexity_tier=ComplexityTier.SIMPLE,
                outcome=Outcome.ACCEPTED,
            )
            save_routing_decision(memory_db, d)
        rate = get_model_success_rate(
            memory_db, "gpt-4o", ComplexityTier.SIMPLE, min_entries=10
        )
        assert rate == pytest.approx(1.0)

    def test_mixed_outcomes(self, memory_db: Database) -> None:
        for i in range(20):
            outcome = Outcome.ACCEPTED if i < 15 else Outcome.REJECTED
            d = _make_routing_decision(
                model_selected="claude-sonnet",
                complexity_tier=ComplexityTier.COMPLEX,
                outcome=outcome,
            )
            save_routing_decision(memory_db, d)
        rate = get_model_success_rate(
            memory_db, "claude-sonnet", ComplexityTier.COMPLEX, min_entries=10
        )
        assert rate == pytest.approx(15 / 20)

    def test_unknown_outcomes_excluded(self, memory_db: Database) -> None:
        """Rows with outcome='unknown' should not count toward the rate."""
        for i in range(15):
            outcome = Outcome.ACCEPTED if i < 10 else Outcome.UNKNOWN
            d = _make_routing_decision(
                model_selected="test-model",
                complexity_tier=ComplexityTier.MEDIUM,
                outcome=outcome,
            )
            save_routing_decision(memory_db, d)
        rate = get_model_success_rate(
            memory_db, "test-model", ComplexityTier.MEDIUM, min_entries=5
        )
        assert rate == pytest.approx(1.0)  # 10 accepted out of 10 non-unknown


# =====================================================================
# Cost entries
# =====================================================================


class TestSaveCostEntry:
    def test_save_and_retrieve(
        self,
        memory_db: Database,
        sample_cost_entry: CostEntry,
    ) -> None:
        save_cost_entry(memory_db, sample_cost_entry)
        row = memory_db.fetchone(
            "SELECT * FROM cost_entries WHERE id = ?", (sample_cost_entry.id,)
        )
        assert row is not None
        assert row["model_id"] == sample_cost_entry.model_id
        assert row["cost_usd"] == pytest.approx(sample_cost_entry.cost_usd)


class TestGetSessionCost:
    def test_empty_session(self, memory_db: Database) -> None:
        cost = get_session_cost(memory_db, "no-such-session")
        assert cost == 0.0

    def test_accumulates_cost(self, memory_db: Database) -> None:
        sid = str(uuid.uuid4())
        for _ in range(3):
            entry = _make_cost_entry(session_id=sid, cost_usd=0.10)
            save_cost_entry(memory_db, entry)
        total = get_session_cost(memory_db, sid)
        assert total == pytest.approx(0.30)


class TestGetDailyCost:
    def test_no_entries(self, memory_db: Database) -> None:
        cost = get_daily_cost(memory_db, "2099-12-31")
        assert cost == 0.0

    def test_sums_for_date(self, memory_db: Database) -> None:
        date_str = "2025-06-15"
        for i in range(4):
            entry = _make_cost_entry(
                created_at=f"{date_str}T{10 + i}:00:00+00:00",
                cost_usd=0.25,
            )
            save_cost_entry(memory_db, entry)
        # Also add one entry on a different date
        other = _make_cost_entry(
            created_at="2025-06-14T10:00:00+00:00",
            cost_usd=1.00,
        )
        save_cost_entry(memory_db, other)

        total = get_daily_cost(memory_db, date_str)
        assert total == pytest.approx(1.00)


class TestGetMonthlyCost:
    def test_no_entries(self, memory_db: Database) -> None:
        cost = get_monthly_cost(memory_db, year=2099, month=1)
        assert cost == 0.0

    def test_sums_for_month(self, memory_db: Database) -> None:
        for day in range(1, 4):
            entry = _make_cost_entry(
                created_at=f"2025-07-{day:02d}T12:00:00+00:00",
                cost_usd=1.00,
            )
            save_cost_entry(memory_db, entry)
        # Entry in a different month
        other = _make_cost_entry(
            created_at="2025-08-01T12:00:00+00:00", cost_usd=5.00
        )
        save_cost_entry(memory_db, other)

        total = get_monthly_cost(memory_db, year=2025, month=7)
        assert total == pytest.approx(3.00)

    def test_december_boundary(self, memory_db: Database) -> None:
        """Cost in December should not bleed into January of next year."""
        dec = _make_cost_entry(
            created_at="2025-12-15T00:00:00+00:00", cost_usd=2.00
        )
        jan = _make_cost_entry(
            created_at="2026-01-01T00:00:00+00:00", cost_usd=3.00
        )
        save_cost_entry(memory_db, dec)
        save_cost_entry(memory_db, jan)

        assert get_monthly_cost(memory_db, 2025, 12) == pytest.approx(2.00)
        assert get_monthly_cost(memory_db, 2026, 1) == pytest.approx(3.00)


class TestGetCostBreakdown:
    def test_session_breakdown(self, memory_db: Database) -> None:
        sid = str(uuid.uuid4())
        for model, cost in [("gpt-4o", 0.50), ("gpt-4o", 0.30), ("claude-sonnet", 0.20)]:
            entry = _make_cost_entry(session_id=sid, model_id=model, cost_usd=cost)
            save_cost_entry(memory_db, entry)

        breakdown = get_cost_breakdown(memory_db, "session", session_id=sid)
        assert len(breakdown) == 2
        # Ordered by total_cost desc
        assert breakdown[0]["model_id"] == "gpt-4o"
        assert breakdown[0]["request_count"] == 2
        assert breakdown[0]["total_cost"] == pytest.approx(0.80)

    def test_session_breakdown_requires_session_id(
        self, memory_db: Database
    ) -> None:
        with pytest.raises(ValueError, match="session_id"):
            get_cost_breakdown(memory_db, "session")

    def test_invalid_period_raises(self, memory_db: Database) -> None:
        with pytest.raises(ValueError, match="Invalid period"):
            get_cost_breakdown(memory_db, "week")

    def test_day_breakdown(self, memory_db: Database) -> None:
        datetime.now(UTC).strftime("%Y-%m-%d")
        entry = _make_cost_entry(
            created_at=datetime.now(UTC).isoformat(),
            model_id="test-model",
            cost_usd=0.42,
        )
        save_cost_entry(memory_db, entry)
        breakdown = get_cost_breakdown(memory_db, "day")
        assert len(breakdown) >= 1
        assert any(b["model_id"] == "test-model" for b in breakdown)


# =====================================================================
# Sessions
# =====================================================================


class TestCreateSession:
    def test_create_and_retrieve(self, memory_db: Database) -> None:
        sid = str(uuid.uuid4())
        create_session(memory_db, sid, "/tmp/project")
        row = memory_db.fetchone("SELECT * FROM sessions WHERE id = ?", (sid,))
        assert row is not None
        assert row["project_root"] == "/tmp/project"
        assert row["total_cost"] == pytest.approx(0.0)
        assert row["total_requests"] == 0
        assert row["active"] == 1


class TestUpdateSession:
    def test_increment_cost_and_requests(self, memory_db: Database) -> None:
        sid = str(uuid.uuid4())
        create_session(memory_db, sid, "/tmp/project")

        update_session(memory_db, sid, cost_delta=0.15, request_delta=1)
        update_session(memory_db, sid, cost_delta=0.10, request_delta=1)

        row = memory_db.fetchone("SELECT * FROM sessions WHERE id = ?", (sid,))
        assert row is not None
        assert row["total_cost"] == pytest.approx(0.25)
        assert row["total_requests"] == 2

    def test_update_nonexistent_session_is_noop(
        self, memory_db: Database
    ) -> None:
        """Updating a non-existent session should silently succeed (0 rows updated)."""
        update_session(memory_db, "nonexistent", cost_delta=1.0, request_delta=1)
        # No exception; no row created


# =====================================================================
# Provider status
# =====================================================================


class TestUpdateProviderStatus:
    def test_insert_new_provider(self, memory_db: Database) -> None:
        update_provider_status(memory_db, "openai", available=True)
        row = memory_db.fetchone(
            "SELECT * FROM provider_status WHERE provider = ?", ("openai",)
        )
        assert row is not None
        assert row["is_available"] == 1
        assert row["consecutive_failures"] == 0

    def test_upsert_on_conflict(self, memory_db: Database) -> None:
        update_provider_status(memory_db, "openai", available=True)
        update_provider_status(memory_db, "openai", available=False, error="timeout")

        row = memory_db.fetchone(
            "SELECT * FROM provider_status WHERE provider = ?", ("openai",)
        )
        assert row is not None
        assert row["is_available"] == 0
        assert row["last_error"] == "timeout"
        assert row["consecutive_failures"] == 1

    def test_consecutive_failures_reset_on_success(
        self, memory_db: Database
    ) -> None:
        update_provider_status(memory_db, "openai", available=False, error="err")
        update_provider_status(memory_db, "openai", available=False, error="err")
        update_provider_status(memory_db, "openai", available=True)

        row = memory_db.fetchone(
            "SELECT * FROM provider_status WHERE provider = ?", ("openai",)
        )
        assert row["consecutive_failures"] == 0


class TestRateLimiting:
    def test_not_rate_limited_by_default(self, memory_db: Database) -> None:
        assert is_rate_limited(memory_db, "openai") is False

    def test_set_and_check_rate_limited(self, memory_db: Database) -> None:
        future = datetime.now(UTC) + timedelta(hours=1)
        set_rate_limited(memory_db, "openai", until=future)
        assert is_rate_limited(memory_db, "openai") is True

    def test_expired_rate_limit(self, memory_db: Database) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        set_rate_limited(memory_db, "openai", until=past)
        assert is_rate_limited(memory_db, "openai") is False


class TestFreeTierUsage:
    def test_first_increment(self, memory_db: Database) -> None:
        count = increment_free_tier_usage(memory_db, "google")
        assert count == 1

    def test_multiple_increments(self, memory_db: Database) -> None:
        for expected in range(1, 6):
            count = increment_free_tier_usage(memory_db, "groq")
            assert count == expected

    def test_get_remaining(self, memory_db: Database) -> None:
        daily_limit = 100
        remaining = get_free_tier_remaining(memory_db, "google", daily_limit)
        assert remaining == daily_limit

    def test_remaining_decreases(self, memory_db: Database) -> None:
        daily_limit = 10
        increment_free_tier_usage(memory_db, "google")
        increment_free_tier_usage(memory_db, "google")
        remaining = get_free_tier_remaining(memory_db, "google", daily_limit)
        assert remaining == 8

    def test_remaining_never_negative(self, memory_db: Database) -> None:
        daily_limit = 2
        for _ in range(5):
            increment_free_tier_usage(memory_db, "google")
        remaining = get_free_tier_remaining(memory_db, "google", daily_limit)
        assert remaining == 0


# =====================================================================
# Tool executions
# =====================================================================


class TestSaveToolExecution:
    def test_save_and_retrieve(
        self,
        memory_db: Database,
        sample_tool_execution: ToolExecution,
    ) -> None:
        save_tool_execution(memory_db, sample_tool_execution)
        row = memory_db.fetchone(
            "SELECT * FROM tool_executions WHERE id = ?",
            (sample_tool_execution.id,),
        )
        assert row is not None
        assert row["tool_name"] == "read_file"
        assert row["result_success"] == 1
        assert row["duration_ms"] == pytest.approx(45.2)

    def test_save_failed_execution(self, memory_db: Database) -> None:
        execution = _make_tool_execution(
            result_success=False,
            result_error="Permission denied",
        )
        save_tool_execution(memory_db, execution)
        row = memory_db.fetchone(
            "SELECT * FROM tool_executions WHERE id = ?", (execution.id,)
        )
        assert row is not None
        assert row["result_success"] == 0
        assert row["result_error"] == "Permission denied"


# =====================================================================
# Cleanup
# =====================================================================


class TestCleanupOldData:
    def test_cleanup_deletes_old_routing_decisions(
        self, memory_db: Database
    ) -> None:
        old = _make_routing_decision(
            created_at="2020-01-01T00:00:00+00:00",
        )
        recent = _make_routing_decision()
        save_routing_decision(memory_db, old)
        save_routing_decision(memory_db, recent)

        deleted = cleanup_old_data(memory_db, routing_days=90)
        assert deleted["routing_decisions"] == 1

        rows = memory_db.fetchall("SELECT * FROM routing_decisions")
        assert len(rows) == 1

    def test_cleanup_deletes_old_tool_executions(
        self, memory_db: Database
    ) -> None:
        old = _make_tool_execution(created_at="2020-01-01T00:00:00+00:00")
        recent = _make_tool_execution()
        save_tool_execution(memory_db, old)
        save_tool_execution(memory_db, recent)

        deleted = cleanup_old_data(memory_db, tool_days=30)
        assert deleted["tool_executions"] == 1

    def test_cleanup_deletes_old_inactive_sessions(
        self, memory_db: Database
    ) -> None:
        sid_old = str(uuid.uuid4())
        sid_recent = str(uuid.uuid4())

        # Create old inactive session
        memory_db.execute(
            "INSERT INTO sessions (id, created_at, updated_at, project_root, active) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid_old, "2020-01-01T00:00:00", "2020-01-01T00:00:00", "/tmp", 0),
        )
        # Create recent session (active)
        create_session(memory_db, sid_recent, "/tmp")
        memory_db.commit()

        deleted = cleanup_old_data(memory_db, session_days=30)
        assert deleted["sessions"] == 1

        row = memory_db.fetchone("SELECT id FROM sessions WHERE id = ?", (sid_old,))
        assert row is None
        row = memory_db.fetchone("SELECT id FROM sessions WHERE id = ?", (sid_recent,))
        assert row is not None

    def test_cleanup_preserves_active_old_sessions(
        self, memory_db: Database
    ) -> None:
        """Active sessions should not be deleted even if old."""
        sid = str(uuid.uuid4())
        memory_db.execute(
            "INSERT INTO sessions (id, created_at, updated_at, project_root, active) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, "2020-01-01T00:00:00", "2020-01-01T00:00:00", "/tmp", 1),
        )
        memory_db.commit()

        deleted = cleanup_old_data(memory_db, session_days=30)
        assert deleted["sessions"] == 0

    def test_cleanup_deletes_old_cost_entries(self, memory_db: Database) -> None:
        old = _make_cost_entry(created_at="2020-01-01T00:00:00+00:00")
        recent = _make_cost_entry()
        save_cost_entry(memory_db, old)
        save_cost_entry(memory_db, recent)

        deleted = cleanup_old_data(memory_db, cost_days=365)
        assert deleted["cost_entries"] == 1

    def test_cleanup_returns_all_zero_when_nothing_to_delete(
        self, memory_db: Database
    ) -> None:
        deleted = cleanup_old_data(memory_db)
        assert all(v == 0 for v in deleted.values())


# =====================================================================
# Transaction integration
# =====================================================================


class TestTransactionIntegration:
    def test_atomic_save_and_update(self, memory_db: Database) -> None:
        """Saving a decision + cost entry + session update atomically."""
        sid = str(uuid.uuid4())
        create_session(memory_db, sid, "/tmp/project")

        decision = _make_routing_decision(session_id=sid)
        entry = _make_cost_entry(session_id=sid, cost_usd=0.50)

        with memory_db.transaction():
            memory_db.execute(
                """
                INSERT INTO routing_decisions (
                    id, created_at, session_id, prompt_hash, complexity_tier,
                    complexity_score, model_selected, fallback_chain,
                    estimated_cost, features, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.id,
                    decision.created_at,
                    decision.session_id,
                    decision.prompt_hash,
                    decision.complexity_tier.value,
                    decision.complexity_score,
                    decision.model_selected,
                    decision.fallback_chain,
                    decision.estimated_cost,
                    decision.features,
                    decision.outcome.value,
                ),
            )
            memory_db.execute(
                """
                INSERT INTO cost_entries (
                    id, created_at, session_id, model_id, provider,
                    input_tokens, output_tokens, cost_usd, complexity_tier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.created_at,
                    entry.session_id,
                    entry.model_id,
                    entry.provider,
                    entry.input_tokens,
                    entry.output_tokens,
                    entry.cost_usd,
                    entry.complexity_tier.value,
                ),
            )
            memory_db.execute(
                "UPDATE sessions SET total_cost = total_cost + ?, total_requests = total_requests + 1 WHERE id = ?",
                (entry.cost_usd, sid),
            )

        # Verify all three writes committed
        assert memory_db.fetchone("SELECT id FROM routing_decisions WHERE id = ?", (decision.id,)) is not None
        assert memory_db.fetchone("SELECT id FROM cost_entries WHERE id = ?", (entry.id,)) is not None
        session_row = memory_db.fetchone("SELECT total_cost, total_requests FROM sessions WHERE id = ?", (sid,))
        assert session_row["total_cost"] == pytest.approx(0.50)
        assert session_row["total_requests"] == 1


# =====================================================================
# Exception handling — DatabaseError re-raise and generic wrapping
# =====================================================================


def _make_broken_db() -> MagicMock:
    """Return a mock Database whose execute/fetchone/fetchall raise RuntimeError."""
    db = MagicMock()
    db.execute.side_effect = RuntimeError("unexpected failure")
    db.fetchone.side_effect = RuntimeError("unexpected failure")
    db.fetchall.side_effect = RuntimeError("unexpected failure")
    db.commit.side_effect = RuntimeError("unexpected failure")
    return db


def _make_db_error_db() -> MagicMock:
    """Return a mock Database whose execute/fetchone/fetchall raise DatabaseError."""
    from prism.exceptions import DatabaseError

    db = MagicMock()
    db.execute.side_effect = DatabaseError("db layer error")
    db.fetchone.side_effect = DatabaseError("db layer error")
    db.fetchall.side_effect = DatabaseError("db layer error")
    db.commit.side_effect = DatabaseError("db layer error")
    return db


class TestSaveRoutingDecisionErrors:
    """Cover lines 72-73: generic exception wrapping in save_routing_decision."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        decision = _make_routing_decision()
        with pytest.raises(DatabaseError, match="Failed to save routing decision"):
            save_routing_decision(broken, decision)

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        decision = _make_routing_decision()
        with pytest.raises(DatabaseError, match="db layer error"):
            save_routing_decision(db, decision)


class TestUpdateRoutingOutcomeErrors:
    """Cover lines 104-107: exception handling in update_routing_outcome."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to update routing outcome"):
            update_routing_outcome(broken, "some-id", Outcome.ACCEPTED, actual_cost=0.01)

    def test_generic_exception_without_cost(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to update routing outcome"):
            update_routing_outcome(broken, "some-id", Outcome.REJECTED)

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            update_routing_outcome(db, "some-id", Outcome.ACCEPTED)


class TestGetRoutingHistoryErrors:
    """Cover lines 145-148: exception handling in get_routing_history."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to get routing history"):
            get_routing_history(broken)

    def test_generic_exception_with_session_filter(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to get routing history"):
            get_routing_history(broken, session_id="some-session")

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            get_routing_history(db)


class TestGetModelSuccessRateErrors:
    """Cover lines 202-205: exception handling in get_model_success_rate."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to get model success rate"):
            get_model_success_rate(broken, "gpt-4o", ComplexityTier.MEDIUM)

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            get_model_success_rate(db, "gpt-4o", ComplexityTier.MEDIUM)


class TestSaveCostEntryErrors:
    """Cover lines 239-242: exception handling in save_cost_entry."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        entry = _make_cost_entry()
        with pytest.raises(DatabaseError, match="Failed to save cost entry"):
            save_cost_entry(broken, entry)

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        entry = _make_cost_entry()
        with pytest.raises(DatabaseError, match="db layer error"):
            save_cost_entry(db, entry)


class TestGetSessionCostErrors:
    """Cover lines 253-256: exception handling in get_session_cost."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to get session cost"):
            get_session_cost(broken, "some-session")

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            get_session_cost(db, "some-session")


class TestGetDailyCostErrors:
    """Cover lines 272-275: exception handling in get_daily_cost."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to get daily cost"):
            get_daily_cost(broken, "2025-06-15")

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            get_daily_cost(db, "2025-06-15")


class TestGetMonthlyCostErrors:
    """Cover lines 304-307: exception handling in get_monthly_cost."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to get monthly cost"):
            get_monthly_cost(broken, year=2025, month=6)

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            get_monthly_cost(db, year=2025, month=6)


class TestGetCostBreakdownMonthPeriod:
    """Cover lines 353-359: the month period branch of get_cost_breakdown."""

    def test_month_breakdown(self, memory_db: Database) -> None:
        """Entries for the current month should appear in the breakdown."""
        now = datetime.now(UTC)
        entry = _make_cost_entry(
            created_at=now.isoformat(),
            model_id="test-monthly-model",
            cost_usd=0.75,
        )
        save_cost_entry(memory_db, entry)
        breakdown = get_cost_breakdown(memory_db, "month")
        assert len(breakdown) >= 1
        assert any(b["model_id"] == "test-monthly-model" for b in breakdown)
        assert any(b["total_cost"] == pytest.approx(0.75) for b in breakdown)

    def test_month_breakdown_multiple_models(self, memory_db: Database) -> None:
        """Multiple models in the same month should be grouped correctly."""
        now = datetime.now(UTC)
        for model, cost in [("model-a", 0.10), ("model-a", 0.20), ("model-b", 0.50)]:
            entry = _make_cost_entry(
                created_at=now.isoformat(),
                model_id=model,
                cost_usd=cost,
            )
            save_cost_entry(memory_db, entry)
        breakdown = get_cost_breakdown(memory_db, "month")
        assert len(breakdown) == 2
        # model-b is more expensive, should be first (ORDER BY total_cost DESC)
        assert breakdown[0]["model_id"] == "model-b"
        assert breakdown[0]["total_cost"] == pytest.approx(0.50)
        assert breakdown[1]["model_id"] == "model-a"
        assert breakdown[1]["total_cost"] == pytest.approx(0.30)
        assert breakdown[1]["request_count"] == 2

    def test_month_breakdown_december(self, memory_db: Database) -> None:
        """December breakdown should work correctly (month == 12 boundary)."""
        # Insert entries for December
        entry = _make_cost_entry(
            created_at="2025-12-15T12:00:00+00:00",
            model_id="dec-model",
            cost_usd=1.50,
        )
        save_cost_entry(memory_db, entry)

        # We need to mock datetime.now to return a December date
        fake_now = datetime(2025, 12, 20, tzinfo=UTC)
        with patch("prism.db.queries.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = datetime
            breakdown = get_cost_breakdown(memory_db, "month")
        assert len(breakdown) == 1
        assert breakdown[0]["model_id"] == "dec-model"
        assert breakdown[0]["total_cost"] == pytest.approx(1.50)


class TestGetCostBreakdownErrors:
    """Cover lines 384-385: generic exception wrapping in get_cost_breakdown."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to get cost breakdown"):
            get_cost_breakdown(broken, "session", session_id="some-sid")

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            get_cost_breakdown(db, "session", session_id="some-sid")


class TestCreateSessionErrors:
    """Cover lines 406-409: exception handling in create_session."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to create session"):
            create_session(broken, "some-id", "/tmp/project")

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            create_session(db, "some-id", "/tmp/project")


class TestUpdateSessionErrors:
    """Cover lines 438-441: exception handling in update_session."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to update session"):
            update_session(broken, "some-id", cost_delta=0.5, request_delta=1)

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            update_session(db, "some-id")


class TestUpdateProviderStatusErrors:
    """Cover lines 485-488: exception handling in update_provider_status."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to update provider status"):
            update_provider_status(broken, "openai", available=True)

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            update_provider_status(db, "openai", available=True)


class TestSetRateLimitedErrors:
    """Cover lines 506-509: exception handling in set_rate_limited."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to set rate limit"):
            set_rate_limited(broken, "openai", until=datetime.now(UTC))

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            set_rate_limited(db, "openai", until=datetime.now(UTC))


class TestIsRateLimitedEdgeCases:
    """Cover lines 525, 527-530: naive datetime handling and exception wrapping."""

    def test_naive_datetime_rate_limit(self, memory_db: Database) -> None:
        """A naive (no tzinfo) rate_limited_until should still work."""
        # Insert a rate limit with a naive datetime (no timezone)
        future_naive = (datetime.now(UTC) + timedelta(hours=1)).replace(tzinfo=None)
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, rate_limited_until)
            VALUES (?, ?)
            """,
            ("test-provider", future_naive.isoformat()),
        )
        memory_db.commit()
        assert is_rate_limited(memory_db, "test-provider") is True

    def test_naive_datetime_expired_rate_limit(self, memory_db: Database) -> None:
        """A naive expired rate_limited_until should return False."""
        past_naive = (datetime.now(UTC) - timedelta(hours=1)).replace(tzinfo=None)
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, rate_limited_until)
            VALUES (?, ?)
            """,
            ("test-provider-past", past_naive.isoformat()),
        )
        memory_db.commit()
        assert is_rate_limited(memory_db, "test-provider-past") is False

    def test_null_rate_limited_until(self, memory_db: Database) -> None:
        """Provider with NULL rate_limited_until should not be rate limited."""
        memory_db.execute(
            "INSERT INTO provider_status (provider) VALUES (?)",
            ("null-provider",),
        )
        memory_db.commit()
        assert is_rate_limited(memory_db, "null-provider") is False

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to check rate limit"):
            is_rate_limited(broken, "openai")

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            is_rate_limited(db, "openai")


class TestIncrementFreeTierUsageEdgeCases:
    """Cover lines 546-552: December/end-of-month boundary for tomorrow_iso.
    Cover lines 579-586, 582, 584: reset counter logic when reset_at is past.
    Cover lines 598-601: exception handling.
    """

    def test_end_of_month_boundary(self, memory_db: Database) -> None:
        """Last day of a non-December month should roll to next month."""
        # Simulate being on January 31
        fake_now = datetime(2025, 1, 31, 15, 0, 0, tzinfo=UTC)
        with patch("prism.db.queries.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = datetime
            count = increment_free_tier_usage(memory_db, "test-eom")
        assert count == 1

        # Verify the reset_at is February 1
        row = memory_db.fetchone(
            "SELECT free_tier_reset_at FROM provider_status WHERE provider = ?",
            ("test-eom",),
        )
        assert row is not None
        assert "2025-02-01" in row["free_tier_reset_at"]

    def test_december_31_boundary(self, memory_db: Database) -> None:
        """December 31 should roll to January 1 of the next year."""
        fake_now = datetime(2025, 12, 31, 23, 0, 0, tzinfo=UTC)
        with patch("prism.db.queries.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = datetime
            count = increment_free_tier_usage(memory_db, "test-dec31")
        assert count == 1

        row = memory_db.fetchone(
            "SELECT free_tier_reset_at FROM provider_status WHERE provider = ?",
            ("test-dec31",),
        )
        assert row is not None
        assert "2026-01-01" in row["free_tier_reset_at"]

    def test_end_of_non_last_day_month(self, memory_db: Database) -> None:
        """A mid-month day should roll to the next day normally."""
        fake_now = datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC)
        with patch("prism.db.queries.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_dt.side_effect = datetime
            count = increment_free_tier_usage(memory_db, "test-mid-month")
        assert count == 1

        row = memory_db.fetchone(
            "SELECT free_tier_reset_at FROM provider_status WHERE provider = ?",
            ("test-mid-month",),
        )
        assert row is not None
        assert "2025-06-16" in row["free_tier_reset_at"]

    def test_counter_resets_when_past_reset_time(self, memory_db: Database) -> None:
        """If free_tier_reset_at is in the past, counter should reset to 0 before incrementing."""
        # Insert provider with a past reset time and existing count
        past_reset = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
            VALUES (?, ?, ?)
            """,
            ("reset-provider", 42, past_reset),
        )
        memory_db.commit()

        count = increment_free_tier_usage(memory_db, "reset-provider")
        # Counter should have been reset to 0, then incremented to 1
        assert count == 1

    def test_counter_does_not_reset_when_before_reset_time(
        self, memory_db: Database
    ) -> None:
        """If free_tier_reset_at is in the future, counter should continue incrementing."""
        future_reset = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
            VALUES (?, ?, ?)
            """,
            ("no-reset-provider", 5, future_reset),
        )
        memory_db.commit()

        count = increment_free_tier_usage(memory_db, "no-reset-provider")
        assert count == 6

    def test_counter_with_naive_reset_timestamp(self, memory_db: Database) -> None:
        """A naive reset_at timestamp (no tzinfo) should still work."""
        past_naive = (datetime.now(UTC) - timedelta(hours=1)).replace(tzinfo=None)
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
            VALUES (?, ?, ?)
            """,
            ("naive-reset", 10, past_naive.isoformat()),
        )
        memory_db.commit()

        count = increment_free_tier_usage(memory_db, "naive-reset")
        # Naive past reset should trigger reset, so count = 1
        assert count == 1

    def test_counter_with_null_reset_at(self, memory_db: Database) -> None:
        """Provider with NULL free_tier_reset_at should continue without reset."""
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
            VALUES (?, ?, ?)
            """,
            ("null-reset", 3, None),
        )
        memory_db.commit()

        count = increment_free_tier_usage(memory_db, "null-reset")
        assert count == 4

    def test_counter_with_null_requests_today(self, memory_db: Database) -> None:
        """Provider with NULL free_tier_requests_today should treat it as 0."""
        future_reset = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
            VALUES (?, ?, ?)
            """,
            ("null-count", None, future_reset),
        )
        memory_db.commit()

        count = increment_free_tier_usage(memory_db, "null-count")
        assert count == 1

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to increment free tier usage"):
            increment_free_tier_usage(broken, "openai")

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            increment_free_tier_usage(db, "openai")


class TestGetFreeTierRemainingEdgeCases:
    """Cover lines 621-629, 625, 627: reset logic with timezone handling.
    Cover lines 631-634: exception handling.
    """

    def test_remaining_resets_when_past_reset_time(
        self, memory_db: Database
    ) -> None:
        """If reset_at is in the past, should return full daily_limit."""
        past_reset = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
            VALUES (?, ?, ?)
            """,
            ("reset-remaining", 50, past_reset),
        )
        memory_db.commit()

        remaining = get_free_tier_remaining(memory_db, "reset-remaining", 100)
        assert remaining == 100

    def test_remaining_with_naive_reset_timestamp(
        self, memory_db: Database
    ) -> None:
        """A naive reset_at timestamp should still work for the reset check."""
        past_naive = (datetime.now(UTC) - timedelta(hours=1)).replace(tzinfo=None)
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
            VALUES (?, ?, ?)
            """,
            ("naive-remaining", 30, past_naive.isoformat()),
        )
        memory_db.commit()

        remaining = get_free_tier_remaining(memory_db, "naive-remaining", 100)
        assert remaining == 100

    def test_remaining_with_future_naive_reset(
        self, memory_db: Database
    ) -> None:
        """A naive future reset_at should NOT trigger reset."""
        future_naive = (datetime.now(UTC) + timedelta(hours=2)).replace(tzinfo=None)
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
            VALUES (?, ?, ?)
            """,
            ("future-naive", 30, future_naive.isoformat()),
        )
        memory_db.commit()

        remaining = get_free_tier_remaining(memory_db, "future-naive", 100)
        assert remaining == 70

    def test_remaining_with_null_reset_at(self, memory_db: Database) -> None:
        """Provider with NULL free_tier_reset_at should not reset."""
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
            VALUES (?, ?, ?)
            """,
            ("null-reset-rem", 20, None),
        )
        memory_db.commit()

        remaining = get_free_tier_remaining(memory_db, "null-reset-rem", 100)
        assert remaining == 80

    def test_remaining_with_null_requests_today(
        self, memory_db: Database
    ) -> None:
        """Provider with NULL free_tier_requests_today should treat as 0 used."""
        future_reset = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        memory_db.execute(
            """
            INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
            VALUES (?, ?, ?)
            """,
            ("null-used", None, future_reset),
        )
        memory_db.commit()

        remaining = get_free_tier_remaining(memory_db, "null-used", 50)
        assert remaining == 50

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        with pytest.raises(DatabaseError, match="Failed to get free tier remaining"):
            get_free_tier_remaining(broken, "openai", 100)

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        with pytest.raises(DatabaseError, match="db layer error"):
            get_free_tier_remaining(db, "openai", 100)


class TestSaveToolExecutionErrors:
    """Cover lines 666-669: exception handling in save_tool_execution."""

    def test_generic_exception_wraps_as_database_error(self) -> None:
        from prism.exceptions import DatabaseError

        broken = _make_broken_db()
        execution = _make_tool_execution()
        with pytest.raises(DatabaseError, match="Failed to save tool execution"):
            save_tool_execution(broken, execution)

    def test_database_error_reraised_directly(self) -> None:
        from prism.exceptions import DatabaseError

        db = _make_db_error_db()
        execution = _make_tool_execution()
        with pytest.raises(DatabaseError, match="db layer error"):
            save_tool_execution(db, execution)


class TestCleanupOldDataErrors:
    """Cover lines 717-720: exception handling in cleanup_old_data."""

    def test_generic_exception_wraps_as_database_error(
        self, memory_db: Database
    ) -> None:
        from prism.exceptions import DatabaseError

        original_execute = memory_db.execute
        call_count = 0

        def failing_execute(sql: str, params: object = ()) -> object:
            nonlocal call_count
            if "DELETE FROM routing_decisions" in sql:
                call_count += 1
                raise RuntimeError("unexpected delete failure")
            return original_execute(sql, params)

        with patch.object(memory_db, "execute", side_effect=failing_execute):
            with pytest.raises(DatabaseError, match="Failed to cleanup old data"):
                cleanup_old_data(memory_db)

    def test_database_error_reraised_directly(self, memory_db: Database) -> None:
        from prism.exceptions import DatabaseError

        # Use a real db but patch execute to raise DatabaseError inside transaction
        original_execute = memory_db.execute

        def failing_execute(sql: str, params: object = ()) -> object:
            if "DELETE FROM routing_decisions" in sql:
                raise DatabaseError("db delete error")
            return original_execute(sql, params)

        with patch.object(memory_db, "execute", side_effect=failing_execute):
            with pytest.raises(DatabaseError, match="db delete error"):
                cleanup_old_data(memory_db)


class TestGetDailyCostDefaultDate:
    """Cover the default date branch of get_daily_cost (target_date=None)."""

    def test_default_date_uses_today(self, memory_db: Database) -> None:
        """When target_date is None, should use today's date."""
        entry = _make_cost_entry(
            created_at=datetime.now(UTC).isoformat(),
            cost_usd=0.42,
        )
        save_cost_entry(memory_db, entry)
        cost = get_daily_cost(memory_db)
        assert cost == pytest.approx(0.42)


class TestGetMonthlyCostDefaults:
    """Cover the default year/month branches of get_monthly_cost."""

    def test_default_year_and_month(self, memory_db: Database) -> None:
        """When year/month are None, should use current month."""
        entry = _make_cost_entry(
            created_at=datetime.now(UTC).isoformat(),
            cost_usd=0.99,
        )
        save_cost_entry(memory_db, entry)
        cost = get_monthly_cost(memory_db)
        assert cost == pytest.approx(0.99)

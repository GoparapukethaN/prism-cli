"""Tests for prism.db.migrations — migration application and idempotency."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prism.db.database import Database
from prism.db.migrations import (
    MIGRATIONS,
    apply_migrations,
    get_current_version,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestSchemaVersionTracking:
    """schema_version table and version queries."""

    def test_fresh_database_version_is_zero(self, tmp_path: Path) -> None:
        """A brand-new database (no migrations) should report version 0."""
        db = Database(tmp_path / "fresh.db")
        version = get_current_version(db)
        assert version == 0
        db.close()

    def test_after_initialize_version_matches_latest(
        self, memory_db: Database
    ) -> None:
        """After initialize(), version should equal the highest migration key."""
        latest = max(MIGRATIONS.keys())
        assert get_current_version(memory_db) == latest

    def test_schema_version_table_exists(self, memory_db: Database) -> None:
        """The schema_version table should exist after initialization."""
        row = memory_db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        assert row is not None

    def test_schema_version_has_record(self, memory_db: Database) -> None:
        """After migration 1, there should be a row in schema_version."""
        row = memory_db.fetchone("SELECT * FROM schema_version WHERE version = 1")
        assert row is not None
        assert row["version"] == 1
        assert row["applied_at"] is not None
        assert row["description"] == "Initial schema"


class TestMigrationApplication:
    """Applying migrations to a database."""

    def test_apply_creates_all_tables(self, memory_db: Database) -> None:
        """Migration 1 should create all expected tables."""
        expected_tables = {
            "routing_decisions",
            "cost_entries",
            "sessions",
            "provider_status",
            "budget_config",
            "tool_executions",
            "schema_version",
        }
        rows = memory_db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        actual_tables = {row["name"] for row in rows}
        assert expected_tables.issubset(actual_tables)

    def test_apply_creates_indexes(self, memory_db: Database) -> None:
        """Migration 1 should create all expected indexes."""
        expected_indexes = {
            "idx_routing_created_at",
            "idx_routing_created_date",
            "idx_routing_model",
            "idx_routing_tier",
            "idx_routing_session",
            "idx_routing_outcome",
            "idx_cost_created_at",
            "idx_cost_created_date",
            "idx_cost_session",
            "idx_cost_model",
            "idx_cost_provider",
            "idx_sessions_created_at",
            "idx_sessions_project",
            "idx_tools_created_at",
            "idx_tools_session",
            "idx_tools_name",
        }
        rows = memory_db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
        actual_indexes = {row["name"] for row in rows}
        assert expected_indexes.issubset(actual_indexes)


class TestMigrationIdempotency:
    """Calling initialize() or apply_migrations() more than once."""

    def test_double_initialize_is_safe(self, memory_db: Database) -> None:
        """Calling initialize() a second time should not fail or duplicate rows."""
        rows_before = memory_db.fetchall("SELECT * FROM schema_version")
        memory_db.initialize()  # second call
        rows_after = memory_db.fetchall("SELECT * FROM schema_version")
        assert len(rows_after) == len(rows_before)  # no duplicates from double init

    def test_apply_returns_zero_when_up_to_date(
        self, memory_db: Database
    ) -> None:
        """apply_migrations on an already-migrated DB should return 0."""
        count = apply_migrations(memory_db)
        assert count == 0

    def test_version_unchanged_after_double_apply(
        self, memory_db: Database
    ) -> None:
        """Version should not change after a redundant apply."""
        v_before = get_current_version(memory_db)
        apply_migrations(memory_db)
        v_after = get_current_version(memory_db)
        assert v_before == v_after


class TestMigrationOnFileDB:
    """Test migrations persist across reopens of a file-backed database."""

    def test_migration_persists_across_reopen(self, tmp_path: Path) -> None:
        """Close and reopen a DB — migrations should not be re-applied."""
        db_path = tmp_path / "persist.db"

        db1 = Database(db_path)
        db1.initialize()
        v1 = get_current_version(db1)
        db1.close()

        db2 = Database(db_path)
        v2 = get_current_version(db2)
        count = apply_migrations(db2)
        db2.close()

        assert v1 == v2
        assert count == 0


class TestGeneratedColumn:
    """Verify the GENERATED ALWAYS AS columns work correctly."""

    def test_routing_decisions_created_date(self, memory_db: Database) -> None:
        """created_date should be auto-populated from created_at."""
        memory_db.execute(
            """
            INSERT INTO routing_decisions (
                id, created_at, session_id, prompt_hash, complexity_tier,
                complexity_score, model_selected, fallback_chain,
                estimated_cost, features
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "gen-col-1",
                "2025-06-15T10:30:00+00:00",
                "sess-1",
                "a" * 64,
                "simple",
                0.2,
                "gpt-4o",
                "[]",
                0.01,
                "{}",
            ),
        )
        memory_db.commit()

        row = memory_db.fetchone(
            "SELECT created_date FROM routing_decisions WHERE id = ?", ("gen-col-1",)
        )
        assert row is not None
        assert row["created_date"] == "2025-06-15"

    def test_cost_entries_created_date(self, memory_db: Database) -> None:
        """cost_entries.created_date should be derived from created_at."""
        memory_db.execute(
            """
            INSERT INTO cost_entries (
                id, created_at, session_id, model_id, provider,
                input_tokens, output_tokens, cost_usd, complexity_tier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "cost-gen-1",
                "2025-03-22T14:00:00+00:00",
                "sess-1",
                "gpt-4o",
                "openai",
                100,
                50,
                0.005,
                "simple",
            ),
        )
        memory_db.commit()

        row = memory_db.fetchone(
            "SELECT created_date FROM cost_entries WHERE id = ?", ("cost-gen-1",)
        )
        assert row is not None
        assert row["created_date"] == "2025-03-22"

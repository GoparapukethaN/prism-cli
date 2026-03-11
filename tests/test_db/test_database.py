"""Tests for prism.db.database — connection management, pragmas, WAL mode."""

from __future__ import annotations

import sqlite3
import threading
from typing import TYPE_CHECKING

import pytest

from prism.db.database import Database
from prism.exceptions import DatabaseError

if TYPE_CHECKING:
    from pathlib import Path


class TestDatabaseConnection:
    """Basic connection lifecycle tests."""

    def test_in_memory_connection(self, memory_db: Database) -> None:
        """An in-memory database should be usable immediately after init."""
        cursor = memory_db.execute("SELECT 1 AS val")
        row = cursor.fetchone()
        assert row["val"] == 1

    def test_file_connection(self, tmp_db: Database) -> None:
        """A file-backed database should create the .db file on disk."""
        assert tmp_db.path.exists()
        cursor = tmp_db.execute("SELECT 1 AS val")
        row = cursor.fetchone()
        assert row["val"] == 1

    def test_parent_directory_created(self, tmp_path: Path) -> None:
        """Database constructor should create missing parent directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "prism.db"
        db = Database(deep_path)
        assert deep_path.parent.exists()
        db.close()

    def test_close_clears_connection(self, memory_db: Database) -> None:
        """After close(), the next access should create a fresh connection."""
        memory_db.execute("SELECT 1")
        memory_db.close()
        # Connection is None after close; next call should re-create
        cursor = memory_db.execute("SELECT 1 AS val")
        assert cursor.fetchone()["val"] == 1

    def test_double_close_is_safe(self, memory_db: Database) -> None:
        """Calling close() twice should not raise."""
        memory_db.close()
        memory_db.close()  # should not raise


class TestPragmas:
    """Verify that performance pragmas are applied correctly."""

    def _pragma_value(self, db: Database, pragma: str) -> str:
        """Helper to read a PRAGMA value."""
        row = db.fetchone(f"PRAGMA {pragma}")
        assert row is not None
        return str(row[0])

    def test_wal_mode(self, tmp_db: Database) -> None:
        """WAL journal mode should be active (on file-backed databases)."""
        assert self._pragma_value(tmp_db, "journal_mode").lower() == "wal"

    def test_synchronous_normal(self, memory_db: Database) -> None:
        """PRAGMA synchronous should be NORMAL (1)."""
        val = self._pragma_value(memory_db, "synchronous")
        assert val == "1"  # NORMAL = 1

    def test_cache_size(self, memory_db: Database) -> None:
        """PRAGMA cache_size should be -64000 (64 MB)."""
        val = int(self._pragma_value(memory_db, "cache_size"))
        assert val == -64000

    def test_foreign_keys_on(self, memory_db: Database) -> None:
        """PRAGMA foreign_keys should be ON (1)."""
        assert self._pragma_value(memory_db, "foreign_keys") == "1"

    def test_temp_store_memory(self, memory_db: Database) -> None:
        """PRAGMA temp_store should be MEMORY (2)."""
        assert self._pragma_value(memory_db, "temp_store") == "2"

    def test_mmap_size(self, tmp_db: Database) -> None:
        """PRAGMA mmap_size should be 256 MB (only works on file-backed DBs)."""
        val = int(self._pragma_value(tmp_db, "mmap_size"))
        assert val == 268435456

    def test_row_factory_is_row(self, memory_db: Database) -> None:
        """Connections should use sqlite3.Row as row_factory."""
        conn = memory_db.connection
        assert conn.row_factory is sqlite3.Row


class TestContextManager:
    """Database as a context manager."""

    def test_context_manager_returns_db(self, tmp_path: Path) -> None:
        """__enter__ should return the Database instance."""
        with Database(tmp_path / "ctx.db") as db:
            db.initialize()
            cursor = db.execute("SELECT 1 AS val")
            assert cursor.fetchone()["val"] == 1

    def test_context_manager_closes_on_exit(self, tmp_path: Path) -> None:
        """Exiting the context should close the connection."""
        db_ref: Database | None = None
        with Database(tmp_path / "ctx2.db") as db:
            db.initialize()
            db_ref = db
        # After exiting the context, _local.conn should be None
        assert getattr(db_ref._local, "conn", None) is None


class TestTransaction:
    """Transaction context manager behaviour."""

    def test_commit_on_success(self, memory_db: Database) -> None:
        """Successful block should commit."""
        with memory_db.transaction():
            memory_db.execute(
                "INSERT INTO sessions (id, created_at, updated_at, project_root) "
                "VALUES (?, ?, ?, ?)",
                ("txn-1", "2025-01-01T00:00:00", "2025-01-01T00:00:00", "/tmp"),
            )
        row = memory_db.fetchone("SELECT id FROM sessions WHERE id = ?", ("txn-1",))
        assert row is not None
        assert row["id"] == "txn-1"

    def test_rollback_on_exception(self, memory_db: Database) -> None:
        """Exception inside a transaction block should roll back."""
        with pytest.raises(RuntimeError, match="boom"), memory_db.transaction():
            memory_db.execute(
                "INSERT INTO sessions (id, created_at, updated_at, project_root) "
                "VALUES (?, ?, ?, ?)",
                ("txn-2", "2025-01-01T00:00:00", "2025-01-01T00:00:00", "/tmp"),
            )
            raise RuntimeError("boom")

        row = memory_db.fetchone("SELECT id FROM sessions WHERE id = ?", ("txn-2",))
        assert row is None

    def test_nested_transaction_not_supported(self, memory_db: Database) -> None:
        """Nested transaction() calls should raise DatabaseError."""
        with pytest.raises(DatabaseError, match="Nested transactions"):
            with memory_db.transaction():
                with memory_db.transaction():
                    pass


class TestThreadSafety:
    """Thread-local connection isolation."""

    def test_different_threads_get_different_connections(
        self, tmp_path: Path
    ) -> None:
        """Two threads should not share the same sqlite3.Connection object."""
        db = Database(tmp_path / "threads.db")
        db.initialize()

        connections: list[int] = []
        barrier = threading.Barrier(2)

        def worker() -> None:
            conn = db.connection
            connections.append(id(conn))
            barrier.wait(timeout=5)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(connections) == 2
        assert connections[0] != connections[1]

        db.close()


class TestHelperMethods:
    """Test fetchone, fetchall, executemany, commit."""

    def test_fetchone_returns_none_for_empty(self, memory_db: Database) -> None:
        """fetchone on an empty table should return None."""
        row = memory_db.fetchone("SELECT id FROM sessions WHERE id = ?", ("nope",))
        assert row is None

    def test_fetchall_returns_empty_list(self, memory_db: Database) -> None:
        """fetchall on an empty table should return []."""
        rows = memory_db.fetchall("SELECT * FROM sessions")
        assert rows == []

    def test_execute_raises_database_error_on_bad_sql(
        self, memory_db: Database
    ) -> None:
        """Bad SQL should raise DatabaseError, not raw sqlite3.Error."""
        with pytest.raises(DatabaseError):
            memory_db.execute("SELECT * FROM nonexistent_table_xyz")

    def test_commit_after_manual_insert(self, memory_db: Database) -> None:
        """Manual insert + commit should persist."""
        memory_db.execute(
            "INSERT INTO sessions (id, created_at, updated_at, project_root) "
            "VALUES (?, ?, ?, ?)",
            ("manual-1", "2025-01-01T00:00:00", "2025-01-01T00:00:00", "/tmp"),
        )
        memory_db.commit()
        row = memory_db.fetchone("SELECT id FROM sessions WHERE id = ?", ("manual-1",))
        assert row is not None

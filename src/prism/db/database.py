"""Database connection management with thread-local SQLite connections.

Usage::

    db = Database(Path("~/.prism/prism.db"))
    db.initialize()          # creates tables / runs migrations

    with db.transaction():
        save_routing_decision(db, decision)
        save_cost_entry(db, entry)

    db.close()

Or as a context manager::

    with Database(Path("~/.prism/prism.db")) as db:
        db.initialize()
        ...
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from prism.exceptions import DatabaseError

if TYPE_CHECKING:
    from collections.abc import Generator

logger = structlog.get_logger(__name__)


class Database:
    """Thread-safe SQLite database wrapper.

    Each thread gets its own ``sqlite3.Connection`` stored on a
    ``threading.local()`` instance.  Connections are opened with
    ``isolation_level=None`` (autocommit) so that we control transactions
    explicitly via :meth:`transaction`.

    Query functions call :meth:`commit` after standalone writes.  Inside a
    :meth:`transaction` block the flag ``_in_transaction`` suppresses those
    inner commits so that the whole block is atomic.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._local = threading.local()

        # Ensure parent directory exists (skip for :memory: databases)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return the thread-local connection, creating one if needed."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            try:
                conn = sqlite3.connect(
                    str(self.path),
                    detect_types=sqlite3.PARSE_DECLTYPES,
                    isolation_level=None,  # autocommit — we manage transactions explicitly
                )
                conn.row_factory = sqlite3.Row
                self._configure(conn)
                self._local.conn = conn
            except sqlite3.Error as exc:
                raise DatabaseError(f"Failed to open database at {self.path}: {exc}") from exc
        return conn

    @property
    def connection(self) -> sqlite3.Connection:
        """Public access to the thread-local connection."""
        return self._get_connection()

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        """Apply WAL mode and performance pragmas to a fresh connection."""
        pragmas = [
            "PRAGMA journal_mode = WAL",
            "PRAGMA synchronous = NORMAL",
            "PRAGMA cache_size = -64000",
            "PRAGMA foreign_keys = ON",
            "PRAGMA temp_store = MEMORY",
            "PRAGMA mmap_size = 268435456",
        ]
        for pragma in pragmas:
            conn.execute(pragma)

    def close(self) -> None:
        """Close the thread-local connection if open."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error as exc:
                logger.warning("db_close_error", error=str(exc))
            finally:
                self._local.conn = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Transaction helper
    # ------------------------------------------------------------------

    @property
    def in_transaction(self) -> bool:
        """Whether the current thread is inside a ``transaction()`` block."""
        return bool(getattr(self._local, "in_transaction", False))

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that wraps a block in a SQL transaction.

        Commits on success, rolls back on exception, then re-raises.
        Inner calls to :meth:`commit` are suppressed so the block stays
        atomic.
        """
        if self.in_transaction:
            raise DatabaseError("Nested transactions are not supported")
        conn = self._get_connection()
        self._local.in_transaction = True
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error as rollback_exc:
                logger.error("transaction_rollback_failed", error=str(rollback_exc))
            raise
        finally:
            self._local.in_transaction = False

    # ------------------------------------------------------------------
    # Initialisation (migrations)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create tables and apply pending migrations.

        Safe to call multiple times — already-applied migrations are
        skipped.
        """
        from prism.db.migrations import apply_migrations

        try:
            apply_migrations(self)
            logger.info("database_initialized", path=str(self.path))
        except Exception as exc:
            raise DatabaseError(f"Database initialization failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def execute(
        self,
        sql: str,
        params: tuple[object, ...] | dict[str, object] = (),
    ) -> sqlite3.Cursor:
        """Execute a single SQL statement and return the cursor."""
        try:
            return self._get_connection().execute(sql, params)
        except sqlite3.Error as exc:
            raise DatabaseError(f"SQL execution failed: {exc}") from exc

    def executemany(
        self,
        sql: str,
        params_seq: list[tuple[object, ...]] | list[dict[str, object]],
    ) -> sqlite3.Cursor:
        """Execute a SQL statement against many parameter sets."""
        try:
            return self._get_connection().executemany(sql, params_seq)
        except sqlite3.Error as exc:
            raise DatabaseError(f"SQL executemany failed: {exc}") from exc

    def fetchone(
        self,
        sql: str,
        params: tuple[object, ...] | dict[str, object] = (),
    ) -> sqlite3.Row | None:
        """Execute SQL and return a single row or ``None``."""
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetchall(
        self,
        sql: str,
        params: tuple[object, ...] | dict[str, object] = (),
    ) -> list[sqlite3.Row]:
        """Execute SQL and return all rows."""
        cursor = self.execute(sql, params)
        return cursor.fetchall()

    def commit(self) -> None:
        """Commit the current implicit transaction.

        When called inside a :meth:`transaction` block this is a no-op;
        the block's ``COMMIT`` handles persistence.
        """
        if self.in_transaction:
            return  # transaction() will commit at the end
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.commit()

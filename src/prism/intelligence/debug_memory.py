"""Cross-session debugging memory — learn from past bug fixes across projects.

Provides:
- :class:`DebugMemory` — SQLite-backed store that records bug fingerprints and
  successful fix patterns, then retrieves similar fixes from any project/session.
- Data classes: :class:`BugFingerprint`, :class:`FixRecord`, :class:`FixSuggestion`.

Usage::

    db = DebugMemory(db_path=Path("~/.prism/debug_memory.db"))
    fp = BugFingerprint(error_type="TypeError", stack_pattern="...", ...)
    db.store_fix(fp, fix_pattern="Add None check", fix_diff="...",
                 project="myapp", model_used="gpt-4o")
    suggestions = db.search_similar(fp)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BugFingerprint:
    """A fingerprint that identifies a class of bugs by error type, stack, and files."""

    error_type: str
    stack_pattern: str
    affected_files: list[str]
    affected_functions: list[str]
    language: str
    framework: str
    fingerprint_hash: str = ""

    def __post_init__(self) -> None:
        """Compute the fingerprint hash if not already set."""
        if not self.fingerprint_hash:
            raw = (
                f"{self.error_type}:{self.stack_pattern}"
                f":{','.join(sorted(self.affected_files))}"
            )
            computed = hashlib.sha256(raw.encode()).hexdigest()[:16]
            object.__setattr__(self, "fingerprint_hash", computed)


@dataclass
class FixRecord:
    """A stored record of a successful bug fix."""

    id: int
    fingerprint: str
    error_type: str
    stack_pattern: str
    fix_pattern: str
    fix_diff: str
    confidence: float
    project: str
    model_used: str
    timestamp: str
    language: str = ""
    framework: str = ""
    affected_files_json: str = "[]"
    affected_functions_json: str = "[]"


@dataclass
class FixSuggestion:
    """A suggested fix retrieved from debug memory, ranked by similarity."""

    original_fix: FixRecord
    similarity: float
    adapted_description: str
    original_context: str


class DebugMemory:
    """Cross-session, cross-project debugging memory with similarity search.

    Stores bug fingerprints and their successful fixes in a local SQLite database.
    On future bugs, searches for similar past fixes by fingerprint hash, error type,
    affected files, and stack pattern keywords.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local = threading.local()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    # ------------------------------------------------------------------
    # Connection management (thread-local)
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local SQLite connection, creating one if needed."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _initialize_db(self) -> None:
        """Create tables and indices if they don't already exist."""
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fix_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                error_type TEXT NOT NULL,
                stack_pattern TEXT DEFAULT '',
                fix_pattern TEXT NOT NULL,
                fix_diff TEXT DEFAULT '',
                confidence REAL DEFAULT 0.5,
                project TEXT DEFAULT '',
                model_used TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                language TEXT DEFAULT '',
                framework TEXT DEFAULT '',
                affected_files_json TEXT DEFAULT '[]',
                affected_functions_json TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_fix_fingerprint
                ON fix_records(fingerprint);
            CREATE INDEX IF NOT EXISTS idx_fix_error_type
                ON fix_records(error_type);
            CREATE INDEX IF NOT EXISTS idx_fix_language
                ON fix_records(language);
            """
        )
        logger.debug("debug_memory.initialized", db=str(self._db_path))

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store_fix(
        self,
        fingerprint: BugFingerprint,
        fix_pattern: str,
        fix_diff: str,
        project: str,
        model_used: str,
        confidence: float = 0.5,
    ) -> FixRecord:
        """Store a successful bug fix for future reference.

        Args:
            fingerprint: The bug fingerprint identifying the class of bug.
            fix_pattern: A short human-readable description of the fix.
            fix_diff: The actual diff or patch that fixed the bug.
            project: Name of the project where the fix was applied.
            model_used: Which AI model produced or helped with the fix.
            confidence: Confidence score in ``[0, 1]``.

        Returns:
            The persisted :class:`FixRecord`.
        """
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()

        cursor = conn.execute(
            """INSERT INTO fix_records
               (fingerprint, error_type, stack_pattern, fix_pattern, fix_diff,
                confidence, project, model_used, timestamp, language, framework,
                affected_files_json, affected_functions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fingerprint.fingerprint_hash,
                fingerprint.error_type,
                fingerprint.stack_pattern,
                fix_pattern,
                fix_diff,
                confidence,
                project,
                model_used,
                now,
                fingerprint.language,
                fingerprint.framework,
                json.dumps(fingerprint.affected_files),
                json.dumps(fingerprint.affected_functions),
            ),
        )
        conn.commit()

        record = FixRecord(
            id=cursor.lastrowid or 0,
            fingerprint=fingerprint.fingerprint_hash,
            error_type=fingerprint.error_type,
            stack_pattern=fingerprint.stack_pattern,
            fix_pattern=fix_pattern,
            fix_diff=fix_diff,
            confidence=confidence,
            project=project,
            model_used=model_used,
            timestamp=now,
            language=fingerprint.language,
            framework=fingerprint.framework,
            affected_files_json=json.dumps(fingerprint.affected_files),
            affected_functions_json=json.dumps(fingerprint.affected_functions),
        )
        logger.info(
            "debug_memory.fix_stored",
            id=record.id,
            error_type=record.error_type,
            project=project,
        )
        return record

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_similar(
        self, fingerprint: BugFingerprint, limit: int = 5
    ) -> list[FixSuggestion]:
        """Search for similar past bug fixes by fingerprint then error type.

        First returns exact fingerprint-hash matches (similarity 1.0), then
        same-error-type matches ranked by a heuristic similarity score.

        Args:
            fingerprint: The bug fingerprint to search against.
            limit: Maximum number of suggestions to return.

        Returns:
            A list of :class:`FixSuggestion` sorted by decreasing similarity.
        """
        conn = self._get_conn()
        suggestions: list[FixSuggestion] = []

        # --- exact fingerprint match ------------------------------------------
        exact_rows = conn.execute(
            "SELECT * FROM fix_records WHERE fingerprint = ? "
            "ORDER BY confidence DESC, timestamp DESC",
            (fingerprint.fingerprint_hash,),
        ).fetchall()

        for row in exact_rows[:limit]:
            record = self._row_to_record(row)
            suggestions.append(
                FixSuggestion(
                    original_fix=record,
                    similarity=1.0,
                    adapted_description=f"Exact match: {record.fix_pattern}",
                    original_context=f"Project: {record.project}, Model: {record.model_used}",
                )
            )

        if len(suggestions) >= limit:
            return suggestions

        # --- same error-type match --------------------------------------------
        remaining = limit - len(suggestions)
        similar_rows = conn.execute(
            "SELECT * FROM fix_records WHERE error_type = ? AND fingerprint != ? "
            "ORDER BY confidence DESC, timestamp DESC LIMIT ?",
            (fingerprint.error_type, fingerprint.fingerprint_hash, remaining),
        ).fetchall()

        for row in similar_rows:
            record = self._row_to_record(row)
            sim = self._compute_similarity(fingerprint, record)
            suggestions.append(
                FixSuggestion(
                    original_fix=record,
                    similarity=sim,
                    adapted_description=(
                        f"Similar error ({fingerprint.error_type}): {record.fix_pattern}"
                    ),
                    original_context=f"Project: {record.project}, Model: {record.model_used}",
                )
            )

        return sorted(suggestions, key=lambda s: s.similarity, reverse=True)[:limit]

    def browse_fixes(
        self, limit: int = 20, project: str | None = None
    ) -> list[FixRecord]:
        """Browse stored fixes, optionally filtered by *project*.

        Args:
            limit: Maximum rows to return.
            project: If given, only return fixes from this project.

        Returns:
            A list of :class:`FixRecord` ordered by most recent first.
        """
        conn = self._get_conn()
        if project:
            rows = conn.execute(
                "SELECT * FROM fix_records WHERE project = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (project, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM fix_records ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search_by_description(self, query: str, limit: int = 10) -> list[FixRecord]:
        """Search fixes by keyword match against fix_pattern and error_type.

        Args:
            query: Search keyword or phrase.
            limit: Maximum rows to return.

        Returns:
            A list of matching :class:`FixRecord`.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM fix_records "
            "WHERE fix_pattern LIKE ? OR error_type LIKE ? "
            "ORDER BY confidence DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def forget(self, fix_id: int) -> bool:
        """Remove a stored fix by *fix_id*. Returns ``True`` if a row was deleted."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM fix_records WHERE id = ?", (fix_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("debug_memory.fix_forgotten", id=fix_id)
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the debug memory store."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM fix_records").fetchone()[0]
        projects = conn.execute(
            "SELECT COUNT(DISTINCT project) FROM fix_records"
        ).fetchone()[0]
        error_types = conn.execute(
            "SELECT COUNT(DISTINCT error_type) FROM fix_records"
        ).fetchone()[0]
        avg_confidence = conn.execute(
            "SELECT AVG(confidence) FROM fix_records"
        ).fetchone()[0]

        return {
            "total_fixes": total,
            "projects": projects,
            "error_types": error_types,
            "avg_confidence": round(avg_confidence, 4) if avg_confidence else 0.0,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the thread-local database connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
            logger.debug("debug_memory.closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FixRecord:
        """Convert a :class:`sqlite3.Row` to a :class:`FixRecord`."""
        return FixRecord(**dict(row))

    @staticmethod
    def _compute_similarity(fingerprint: BugFingerprint, record: FixRecord) -> float:
        """Compute a rough similarity score in ``[0, 1]`` between *fingerprint* and *record*.

        Factors considered:
        - Error type match (0.4)
        - Language match (0.1)
        - Framework match (0.1)
        - Affected-file overlap (0.2)
        - Stack-pattern keyword overlap (0.2)
        """
        score = 0.0

        # Error type
        if fingerprint.error_type == record.error_type:
            score += 0.4

        # Language
        if fingerprint.language and fingerprint.language == record.language:
            score += 0.1

        # Framework
        if fingerprint.framework and fingerprint.framework == record.framework:
            score += 0.1

        # File overlap
        try:
            stored_files = set(json.loads(record.affected_files_json))
            current_files = set(fingerprint.affected_files)
            if stored_files and current_files:
                union = stored_files | current_files
                overlap = len(stored_files & current_files) / max(len(union), 1)
                score += overlap * 0.2
        except (json.JSONDecodeError, TypeError):
            pass

        # Stack pattern keyword overlap
        if fingerprint.stack_pattern and record.stack_pattern:
            fp_words = set(fingerprint.stack_pattern.lower().split())
            rec_words = set(record.stack_pattern.lower().split())
            if fp_words and rec_words:
                union = fp_words | rec_words
                overlap = len(fp_words & rec_words) / max(len(union), 1)
                score += overlap * 0.2

        return min(score, 1.0)

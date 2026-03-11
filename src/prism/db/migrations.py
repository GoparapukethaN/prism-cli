"""Forward-only numbered migration system for the Prism database.

Migrations are applied in order and tracked in the ``schema_version``
table so each migration runs exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from prism.exceptions import MigrationError

if TYPE_CHECKING:
    from prism.db.database import Database

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Migration:
    """A single, immutable database migration."""

    version: int
    description: str
    sql: str


# ------------------------------------------------------------------
# Migration registry — append-only, never edit existing entries
# ------------------------------------------------------------------

MIGRATIONS: dict[int, Migration] = {
    1: Migration(
        version=1,
        description="Initial schema",
        sql="""
        CREATE TABLE IF NOT EXISTS routing_decisions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            session_id TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            complexity_tier TEXT NOT NULL,
            complexity_score REAL NOT NULL,
            model_selected TEXT NOT NULL,
            model_actual TEXT,
            fallback_chain TEXT NOT NULL,
            estimated_cost REAL NOT NULL,
            actual_cost REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cached_tokens INTEGER DEFAULT 0,
            latency_ms REAL,
            outcome TEXT DEFAULT 'unknown',
            features TEXT NOT NULL,
            error TEXT,
            created_date TEXT GENERATED ALWAYS AS (date(created_at)) STORED
        );

        CREATE INDEX IF NOT EXISTS idx_routing_created_at
            ON routing_decisions(created_at);
        CREATE INDEX IF NOT EXISTS idx_routing_created_date
            ON routing_decisions(created_date);
        CREATE INDEX IF NOT EXISTS idx_routing_model
            ON routing_decisions(model_selected);
        CREATE INDEX IF NOT EXISTS idx_routing_tier
            ON routing_decisions(complexity_tier);
        CREATE INDEX IF NOT EXISTS idx_routing_session
            ON routing_decisions(session_id);
        CREATE INDEX IF NOT EXISTS idx_routing_outcome
            ON routing_decisions(outcome);

        CREATE TABLE IF NOT EXISTS cost_entries (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            session_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cached_tokens INTEGER DEFAULT 0,
            cost_usd REAL NOT NULL,
            complexity_tier TEXT NOT NULL,
            created_date TEXT GENERATED ALWAYS AS (date(created_at)) STORED
        );

        CREATE INDEX IF NOT EXISTS idx_cost_created_at
            ON cost_entries(created_at);
        CREATE INDEX IF NOT EXISTS idx_cost_created_date
            ON cost_entries(created_date);
        CREATE INDEX IF NOT EXISTS idx_cost_session
            ON cost_entries(session_id);
        CREATE INDEX IF NOT EXISTS idx_cost_model
            ON cost_entries(model_id);
        CREATE INDEX IF NOT EXISTS idx_cost_provider
            ON cost_entries(provider);

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            project_root TEXT NOT NULL,
            total_cost REAL DEFAULT 0.0,
            total_requests INTEGER DEFAULT 0,
            summary TEXT,
            active BOOLEAN DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_created_at
            ON sessions(created_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_project
            ON sessions(project_root);

        CREATE TABLE IF NOT EXISTS provider_status (
            provider TEXT PRIMARY KEY,
            last_check_at TEXT,
            is_available BOOLEAN DEFAULT 1,
            last_error TEXT,
            rate_limited_until TEXT,
            consecutive_failures INTEGER DEFAULT 0,
            free_tier_requests_today INTEGER DEFAULT 0,
            free_tier_reset_at TEXT
        );

        CREATE TABLE IF NOT EXISTS budget_config (
            id TEXT PRIMARY KEY DEFAULT 'default',
            daily_limit REAL,
            monthly_limit REAL,
            warn_at_percent REAL DEFAULT 80.0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tool_executions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            session_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            arguments TEXT NOT NULL,
            result_success BOOLEAN NOT NULL,
            result_error TEXT,
            duration_ms REAL,
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_tools_created_at
            ON tool_executions(created_at);
        CREATE INDEX IF NOT EXISTS idx_tools_session
            ON tool_executions(session_id);
        CREATE INDEX IF NOT EXISTS idx_tools_name
            ON tool_executions(tool_name);

        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT
        );
        """,
    ),
    2: Migration(
        version=2,
        description="Add architect mode tables (plans, plan_steps)",
        sql="""
        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            description TEXT NOT NULL,
            planning_model TEXT NOT NULL,
            execution_model TEXT NOT NULL,
            estimated_total_cost REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'draft',
            git_checkpoint TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_plans_created_at
            ON plans(created_at);
        CREATE INDEX IF NOT EXISTS idx_plans_status
            ON plans(status);

        CREATE TABLE IF NOT EXISTS plan_steps (
            id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL,
            order_num INTEGER NOT NULL,
            description TEXT NOT NULL,
            tool_calls TEXT NOT NULL DEFAULT '[]',
            estimated_tokens INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            result TEXT,
            error TEXT,
            FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_plan_steps_plan_id
            ON plan_steps(plan_id);
        CREATE INDEX IF NOT EXISTS idx_plan_steps_status
            ON plan_steps(status);
        """,
    ),
    3: Migration(
        version=3,
        description="Add user_feedback table for explicit thumbs up/down tracking",
        sql="""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            session_id TEXT NOT NULL,
            model TEXT NOT NULL,
            tier TEXT NOT NULL,
            feedback TEXT NOT NULL,
            routing_decision_id TEXT,
            FOREIGN KEY (routing_decision_id) REFERENCES routing_decisions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_feedback_session
            ON user_feedback(session_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_model
            ON user_feedback(model);
        CREATE INDEX IF NOT EXISTS idx_feedback_tier
            ON user_feedback(tier);
        CREATE INDEX IF NOT EXISTS idx_feedback_created_at
            ON user_feedback(created_at);
        """,
    ),
}


# ------------------------------------------------------------------
# Migration runner
# ------------------------------------------------------------------


def _ensure_schema_version_table(db: Database) -> None:
    """Create the schema_version table if it doesn't exist yet.

    This is needed so we can query current version even on a brand-new
    database.
    """
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT
        )
        """
    )
    db.commit()


def get_current_version(db: Database) -> int:
    """Return the highest migration version that has been applied.

    Returns 0 if the database has never been migrated.
    """
    _ensure_schema_version_table(db)
    row = db.fetchone("SELECT MAX(version) AS v FROM schema_version")
    if row is None or row["v"] is None:
        return 0
    return int(row["v"])


def apply_migrations(db: Database) -> int:
    """Apply all pending migrations in order.

    Returns the number of migrations applied (0 means already up-to-date).

    Raises:
        MigrationError: If any migration fails.
    """
    current = get_current_version(db)
    target = max(MIGRATIONS.keys()) if MIGRATIONS else 0
    applied_count = 0

    if current >= target:
        logger.debug("migrations_up_to_date", current_version=current)
        return 0

    for version in sorted(MIGRATIONS.keys()):
        if version <= current:
            continue

        migration = MIGRATIONS[version]
        logger.info(
            "applying_migration",
            version=version,
            description=migration.description,
        )

        try:
            conn = db.connection
            # executescript() implicitly commits any open transaction and
            # runs each statement in autocommit mode, so we cannot wrap it
            # inside our transaction() helper.  Instead we run the DDL
            # first, then record the version in a separate transaction.
            conn.executescript(migration.sql)
            conn.execute(
                "INSERT INTO schema_version (version, applied_at, description) "
                "VALUES (?, ?, ?)",
                (
                    version,
                    datetime.now(UTC).isoformat(),
                    migration.description,
                ),
            )
            conn.commit()
            applied_count += 1
            logger.info("migration_applied", version=version)
        except Exception as exc:
            raise MigrationError(version, str(exc)) from exc

    return applied_count

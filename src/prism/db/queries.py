"""All database query functions for the Prism database.

Every function takes a ``Database`` instance as first argument and uses
parameterised queries exclusively — no string interpolation for SQL.
"""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from prism.db.models import (
    ComplexityTier,
    CostEntry,
    Outcome,
    RoutingDecision,
    ToolExecution,
)
from prism.exceptions import DatabaseError

if TYPE_CHECKING:
    from prism.db.database import Database

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Routing decisions
# ------------------------------------------------------------------


def save_routing_decision(db: Database, decision: RoutingDecision) -> None:
    """Insert a routing decision record."""
    try:
        db.execute(
            """
            INSERT INTO routing_decisions (
                id, created_at, session_id, prompt_hash, complexity_tier,
                complexity_score, model_selected, model_actual, fallback_chain,
                estimated_cost, actual_cost, input_tokens, output_tokens,
                cached_tokens, latency_ms, outcome, features, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.id,
                decision.created_at,
                decision.session_id,
                decision.prompt_hash,
                decision.complexity_tier.value,
                decision.complexity_score,
                decision.model_selected,
                decision.model_actual,
                decision.fallback_chain,
                decision.estimated_cost,
                decision.actual_cost,
                decision.input_tokens,
                decision.output_tokens,
                decision.cached_tokens,
                decision.latency_ms,
                decision.outcome.value,
                decision.features,
                decision.error,
            ),
        )
        db.commit()
        logger.debug("routing_decision_saved", decision_id=decision.id)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to save routing decision: {exc}") from exc


def update_routing_outcome(
    db: Database,
    decision_id: str,
    outcome: Outcome,
    actual_cost: float | None = None,
) -> None:
    """Update the outcome and optionally the actual cost of a routing decision."""
    try:
        if actual_cost is not None:
            db.execute(
                """
                UPDATE routing_decisions
                SET outcome = ?, actual_cost = ?
                WHERE id = ?
                """,
                (outcome.value, actual_cost, decision_id),
            )
        else:
            db.execute(
                "UPDATE routing_decisions SET outcome = ? WHERE id = ?",
                (outcome.value, decision_id),
            )
        db.commit()
        logger.debug(
            "routing_outcome_updated",
            decision_id=decision_id,
            outcome=outcome.value,
        )
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to update routing outcome: {exc}") from exc


def get_routing_history(
    db: Database,
    limit: int = 100,
    session_id: str | None = None,
) -> list[RoutingDecision]:
    """Return recent routing decisions, optionally filtered by session."""
    try:
        if session_id is not None:
            rows = db.fetchall(
                """
                SELECT id, created_at, session_id, prompt_hash, complexity_tier,
                       complexity_score, model_selected, model_actual, fallback_chain,
                       estimated_cost, actual_cost, input_tokens, output_tokens,
                       cached_tokens, latency_ms, outcome, features, error
                FROM routing_decisions
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
        else:
            rows = db.fetchall(
                """
                SELECT id, created_at, session_id, prompt_hash, complexity_tier,
                       complexity_score, model_selected, model_actual, fallback_chain,
                       estimated_cost, actual_cost, input_tokens, output_tokens,
                       cached_tokens, latency_ms, outcome, features, error
                FROM routing_decisions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [_row_to_routing_decision(row) for row in rows]
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to get routing history: {exc}") from exc


def _row_to_routing_decision(row: object) -> RoutingDecision:
    """Convert a sqlite3.Row to a RoutingDecision model."""
    return RoutingDecision(
        id=row["id"],
        created_at=row["created_at"],
        session_id=row["session_id"],
        prompt_hash=row["prompt_hash"],
        complexity_tier=ComplexityTier(row["complexity_tier"]),
        complexity_score=row["complexity_score"],
        model_selected=row["model_selected"],
        model_actual=row["model_actual"],
        fallback_chain=row["fallback_chain"],
        estimated_cost=row["estimated_cost"],
        actual_cost=row["actual_cost"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cached_tokens=row["cached_tokens"] or 0,
        latency_ms=row["latency_ms"],
        outcome=Outcome(row["outcome"]) if row["outcome"] else Outcome.UNKNOWN,
        features=row["features"],
        error=row["error"],
    )


def get_model_success_rate(
    db: Database,
    model_id: str,
    tier: ComplexityTier,
    min_entries: int = 10,
) -> float:
    """Calculate the historical success rate for *model_id* on *tier*.

    Returns 0.5 (agnostic prior) when fewer than *min_entries* records
    with a known outcome exist.
    """
    try:
        row = db.fetchone(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN outcome = 'accepted' THEN 1 ELSE 0 END) AS accepted
            FROM routing_decisions
            WHERE model_selected = ?
              AND complexity_tier = ?
              AND outcome != 'unknown'
            """,
            (model_id, tier.value),
        )
        if row is None or row["total"] < min_entries:
            return 0.5
        return float(row["accepted"]) / float(row["total"])
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to get model success rate: {exc}") from exc


# ------------------------------------------------------------------
# Cost entries
# ------------------------------------------------------------------


def save_cost_entry(db: Database, entry: CostEntry) -> None:
    """Insert a cost tracking record."""
    try:
        db.execute(
            """
            INSERT INTO cost_entries (
                id, created_at, session_id, model_id, provider,
                input_tokens, output_tokens, cached_tokens, cost_usd,
                complexity_tier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.created_at,
                entry.session_id,
                entry.model_id,
                entry.provider,
                entry.input_tokens,
                entry.output_tokens,
                entry.cached_tokens,
                entry.cost_usd,
                entry.complexity_tier.value if hasattr(entry.complexity_tier, "value") else str(entry.complexity_tier),
            ),
        )
        db.commit()
        logger.debug("cost_entry_saved", entry_id=entry.id)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to save cost entry: {exc}") from exc


def get_session_cost(db: Database, session_id: str) -> float:
    """Return the total cost in USD for a given session."""
    try:
        row = db.fetchone(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM cost_entries WHERE session_id = ?",
            (session_id,),
        )
        return float(row["total"]) if row else 0.0
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to get session cost: {exc}") from exc


def get_daily_cost(db: Database, target_date: str | None = None) -> float:
    """Return the total cost in USD for *target_date* (``YYYY-MM-DD``).

    Defaults to today (UTC).
    """
    if target_date is None:
        target_date = datetime.now(UTC).strftime("%Y-%m-%d")
    try:
        row = db.fetchone(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM cost_entries WHERE created_date = ?",
            (target_date,),
        )
        return float(row["total"]) if row else 0.0
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to get daily cost: {exc}") from exc


def get_monthly_cost(
    db: Database,
    year: int | None = None,
    month: int | None = None,
) -> float:
    """Return the total cost in USD for a given month.

    Defaults to the current month (UTC).
    """
    now = datetime.now(UTC)
    if year is None:
        year = now.year
    if month is None:
        month = now.month
    month_start = f"{year:04d}-{month:02d}-01"
    month_end = f"{year + 1:04d}-01-01" if month == 12 else f"{year:04d}-{month + 1:02d}-01"
    try:
        row = db.fetchone(
            """
            SELECT COALESCE(SUM(cost_usd), 0.0) AS total
            FROM cost_entries
            WHERE created_date >= ? AND created_date < ?
            """,
            (month_start, month_end),
        )
        return float(row["total"]) if row else 0.0
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to get monthly cost: {exc}") from exc


def get_cost_breakdown(
    db: Database,
    period: str,
    session_id: str | None = None,
) -> list[dict[str, object]]:
    """Return a cost breakdown by model for *period*.

    *period* is one of ``"session"``, ``"day"``, ``"month"``.
    When ``period="session"``, *session_id* is required.

    Returns a list of dicts with keys: model_id, request_count, total_cost.
    """
    try:
        if period == "session":
            if session_id is None:
                raise ValueError("session_id is required when period='session'")
            rows = db.fetchall(
                """
                SELECT model_id,
                       COUNT(*) AS request_count,
                       SUM(cost_usd) AS total_cost
                FROM cost_entries
                WHERE session_id = ?
                GROUP BY model_id
                ORDER BY total_cost DESC
                """,
                (session_id,),
            )
        elif period == "day":
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            rows = db.fetchall(
                """
                SELECT model_id,
                       COUNT(*) AS request_count,
                       SUM(cost_usd) AS total_cost
                FROM cost_entries
                WHERE created_date = ?
                GROUP BY model_id
                ORDER BY total_cost DESC
                """,
                (today,),
            )
        elif period == "month":
            now = datetime.now(UTC)
            month_start = f"{now.year:04d}-{now.month:02d}-01"
            if now.month == 12:
                month_end = f"{now.year + 1:04d}-01-01"
            else:
                month_end = f"{now.year:04d}-{now.month + 1:02d}-01"
            rows = db.fetchall(
                """
                SELECT model_id,
                       COUNT(*) AS request_count,
                       SUM(cost_usd) AS total_cost
                FROM cost_entries
                WHERE created_date >= ? AND created_date < ?
                GROUP BY model_id
                ORDER BY total_cost DESC
                """,
                (month_start, month_end),
            )
        else:
            raise ValueError(f"Invalid period: {period!r}. Must be 'session', 'day', or 'month'.")

        return [
            {
                "model_id": row["model_id"],
                "request_count": row["request_count"],
                "total_cost": row["total_cost"],
            }
            for row in rows
        ]
    except (DatabaseError, ValueError):
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to get cost breakdown: {exc}") from exc


# ------------------------------------------------------------------
# Sessions
# ------------------------------------------------------------------


def create_session(db: Database, session_id: str, project_root: str) -> None:
    """Create a new session record."""
    now = datetime.now(UTC).isoformat()
    try:
        db.execute(
            """
            INSERT INTO sessions (id, created_at, updated_at, project_root, total_cost, total_requests, active)
            VALUES (?, ?, ?, ?, 0.0, 0, 1)
            """,
            (session_id, now, now, project_root),
        )
        db.commit()
        logger.debug("session_created", session_id=session_id)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to create session: {exc}") from exc


def update_session(
    db: Database,
    session_id: str,
    cost_delta: float = 0.0,
    request_delta: int = 0,
) -> None:
    """Increment session cost and request count."""
    now = datetime.now(UTC).isoformat()
    try:
        db.execute(
            """
            UPDATE sessions
            SET total_cost = total_cost + ?,
                total_requests = total_requests + ?,
                updated_at = ?
            WHERE id = ?
            """,
            (cost_delta, request_delta, now, session_id),
        )
        db.commit()
        logger.debug(
            "session_updated",
            session_id=session_id,
            cost_delta=cost_delta,
            request_delta=request_delta,
        )
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to update session: {exc}") from exc


# ------------------------------------------------------------------
# Provider status
# ------------------------------------------------------------------


def update_provider_status(
    db: Database,
    provider: str,
    available: bool,
    error: str | None = None,
) -> None:
    """Insert or update a provider's availability status."""
    now = datetime.now(UTC).isoformat()
    try:
        db.execute(
            """
            INSERT INTO provider_status (provider, last_check_at, is_available, last_error, consecutive_failures)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                last_check_at = excluded.last_check_at,
                is_available = excluded.is_available,
                last_error = excluded.last_error,
                consecutive_failures = CASE
                    WHEN excluded.is_available = 1 THEN 0
                    ELSE provider_status.consecutive_failures + 1
                END
            """,
            (
                provider,
                now,
                available,
                error,
                0 if available else 1,
            ),
        )
        db.commit()
        logger.debug(
            "provider_status_updated",
            provider=provider,
            available=available,
        )
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to update provider status: {exc}") from exc


def set_rate_limited(db: Database, provider: str, until: datetime) -> None:
    """Mark a provider as rate-limited until the given UTC datetime."""
    until_iso = until.isoformat()
    try:
        db.execute(
            """
            INSERT INTO provider_status (provider, rate_limited_until)
            VALUES (?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                rate_limited_until = excluded.rate_limited_until
            """,
            (provider, until_iso),
        )
        db.commit()
        logger.debug("provider_rate_limited", provider=provider, until=until_iso)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to set rate limit: {exc}") from exc


def is_rate_limited(db: Database, provider: str) -> bool:
    """Return ``True`` if the provider is currently rate-limited."""
    try:
        row = db.fetchone(
            "SELECT rate_limited_until FROM provider_status WHERE provider = ?",
            (provider,),
        )
        if row is None or row["rate_limited_until"] is None:
            return False
        limit_until = datetime.fromisoformat(row["rate_limited_until"])
        # Ensure comparison is timezone-aware
        now = datetime.now(UTC)
        if limit_until.tzinfo is None:
            limit_until = limit_until.replace(tzinfo=UTC)
        return now < limit_until
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to check rate limit: {exc}") from exc


def increment_free_tier_usage(db: Database, provider: str) -> int:
    """Increment and return the free-tier request count for today.

    Resets the counter if the reset timestamp is in the past.
    """
    now = datetime.now(UTC)

    # Calculate tomorrow midnight for reset timestamp
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    if now.day < days_in_month:
        tomorrow_iso = now.replace(
            day=now.day + 1, hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
    elif now.month == 12:
        tomorrow_iso = now.replace(
            year=now.year + 1, month=1, day=1,
            hour=0, minute=0, second=0, microsecond=0,
        ).isoformat()
    else:
        tomorrow_iso = now.replace(
            month=now.month + 1, day=1,
            hour=0, minute=0, second=0, microsecond=0,
        ).isoformat()

    try:
        row = db.fetchone(
            "SELECT free_tier_requests_today, free_tier_reset_at FROM provider_status WHERE provider = ?",
            (provider,),
        )

        if row is None:
            # First time seeing this provider
            db.execute(
                """
                INSERT INTO provider_status (provider, free_tier_requests_today, free_tier_reset_at)
                VALUES (?, 1, ?)
                """,
                (provider, tomorrow_iso),
            )
            db.commit()
            return 1

        reset_at = row["free_tier_reset_at"]
        current_count = row["free_tier_requests_today"] or 0

        # Check if counter should be reset
        if reset_at is not None:
            reset_dt = datetime.fromisoformat(reset_at)
            if reset_dt.tzinfo is None:
                reset_dt = reset_dt.replace(tzinfo=UTC)
            if now >= reset_dt:
                current_count = 0

        new_count = current_count + 1
        db.execute(
            """
            UPDATE provider_status
            SET free_tier_requests_today = ?,
                free_tier_reset_at = ?
            WHERE provider = ?
            """,
            (new_count, tomorrow_iso, provider),
        )
        db.commit()
        return new_count
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to increment free tier usage: {exc}") from exc


def get_free_tier_remaining(db: Database, provider: str, daily_limit: int) -> int:
    """Return the number of free-tier requests remaining today.

    If the provider has no record yet, the full *daily_limit* is returned.
    """
    try:
        row = db.fetchone(
            "SELECT free_tier_requests_today, free_tier_reset_at FROM provider_status WHERE provider = ?",
            (provider,),
        )
        if row is None:
            return daily_limit

        used = row["free_tier_requests_today"] or 0

        # Check if counter should be reset
        reset_at = row["free_tier_reset_at"]
        if reset_at is not None:
            now = datetime.now(UTC)
            reset_dt = datetime.fromisoformat(reset_at)
            if reset_dt.tzinfo is None:
                reset_dt = reset_dt.replace(tzinfo=UTC)
            if now >= reset_dt:
                return daily_limit

        remaining = daily_limit - used
        return max(remaining, 0)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to get free tier remaining: {exc}") from exc


# ------------------------------------------------------------------
# Tool executions
# ------------------------------------------------------------------


def save_tool_execution(db: Database, execution: ToolExecution) -> None:
    """Insert a tool execution audit record."""
    try:
        db.execute(
            """
            INSERT INTO tool_executions (
                id, created_at, session_id, tool_name, arguments,
                result_success, result_error, duration_ms, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                execution.id,
                execution.created_at,
                execution.session_id,
                execution.tool_name,
                execution.arguments,
                execution.result_success,
                execution.result_error,
                execution.duration_ms,
                execution.metadata,
            ),
        )
        db.commit()
        logger.debug("tool_execution_saved", execution_id=execution.id)
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to save tool execution: {exc}") from exc


# ------------------------------------------------------------------
# Data maintenance
# ------------------------------------------------------------------


def cleanup_old_data(
    db: Database,
    routing_days: int = 90,
    tool_days: int = 30,
    session_days: int = 30,
    cost_days: int = 365,
) -> dict[str, int]:
    """Delete records older than the specified retention periods.

    Returns a dict mapping table name to the number of rows deleted.
    """
    deleted: dict[str, int] = {}
    try:
        with db.transaction():
            cursor = db.execute(
                "DELETE FROM routing_decisions WHERE created_at < datetime('now', ?)",
                (f"-{routing_days} days",),
            )
            deleted["routing_decisions"] = cursor.rowcount

            cursor = db.execute(
                "DELETE FROM tool_executions WHERE created_at < datetime('now', ?)",
                (f"-{tool_days} days",),
            )
            deleted["tool_executions"] = cursor.rowcount

            cursor = db.execute(
                "DELETE FROM sessions WHERE created_at < datetime('now', ?) AND active = 0",
                (f"-{session_days} days",),
            )
            deleted["sessions"] = cursor.rowcount

            cursor = db.execute(
                "DELETE FROM cost_entries WHERE created_at < datetime('now', ?)",
                (f"-{cost_days} days",),
            )
            deleted["cost_entries"] = cursor.rowcount

        logger.info("cleanup_complete", deleted=deleted)
        return deleted
    except DatabaseError:
        raise
    except Exception as exc:
        raise DatabaseError(f"Failed to cleanup old data: {exc}") from exc

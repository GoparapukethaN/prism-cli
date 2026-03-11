"""Plan persistence for Architect Mode.

Stores and retrieves :class:`Plan` and :class:`PlanStep` objects in the
SQLite database managed by :class:`prism.db.database.Database`.

New fields introduced in the enhanced spec are stored in a ``metadata``
TEXT column (JSON blob) on both ``plans`` and ``plan_steps`` tables.
The column is added idempotently via ``ALTER TABLE ... ADD COLUMN``
with a try/except guard so existing databases are upgraded
transparently.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

import structlog

from prism.architect.planner import (
    RISK_LOW,
    Plan,
    PlanStep,
    StepStatus,
)
from prism.exceptions import DatabaseError

if TYPE_CHECKING:
    from prism.db.database import Database

logger = structlog.get_logger(__name__)


def _ensure_metadata_columns(db: Database) -> None:
    """Idempotently add ``metadata`` TEXT column to plans/plan_steps.

    Uses ``ALTER TABLE ... ADD COLUMN`` wrapped in try/except so
    it is safe to call repeatedly (the column already existing
    raises an OperationalError which is silently ignored).

    Args:
        db: Initialised database instance.
    """
    for table in ("plans", "plan_steps"):
        try:
            db.execute(
                f"ALTER TABLE {table} ADD COLUMN metadata TEXT"
            )
            db.commit()
            logger.debug(
                "metadata_column_added", table=table,
            )
        except Exception:
            # Column already exists — expected on subsequent runs
            logger.debug(
                "metadata_column_exists", table=table,
            )


class PlanStorage:
    """Store and retrieve plans from SQLite.

    Plans are stored in the ``plans`` table and their steps in
    ``plan_steps``.  Both tables are created by migration 2 in
    :mod:`prism.db.migrations`.  Enhanced fields are stored in a
    ``metadata`` TEXT column as a JSON blob.
    """

    def __init__(self, db: Database) -> None:
        """Initialise with a database connection.

        Ensures the ``metadata`` column exists on both tables.

        Args:
            db: Initialised :class:`Database` instance.
        """
        self.db = db
        _ensure_metadata_columns(db)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_plan(self, plan: Plan) -> None:
        """Save a plan and all its steps to the database.

        If a plan with the same ID already exists it is replaced
        (upsert semantics via ``INSERT OR REPLACE``).  Enhanced
        fields (goal_summary, preconditions, postconditions,
        risk_assessment, estimated_time_minutes, git_start_hash)
        are stored in the ``metadata`` JSON column.

        Args:
            plan: The plan to persist.
        """
        try:
            plan_metadata = json.dumps({
                "goal_summary": plan.goal_summary,
                "preconditions": plan.preconditions,
                "postconditions": plan.postconditions,
                "risk_assessment": plan.risk_assessment,
                "estimated_time_minutes": (
                    plan.estimated_time_minutes
                ),
                "git_start_hash": plan.git_start_hash,
            })

            with self.db.transaction():
                self.db.execute(
                    """
                    INSERT OR REPLACE INTO plans (
                        id, created_at, description,
                        planning_model, execution_model,
                        estimated_total_cost, status,
                        git_checkpoint, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        plan.id,
                        plan.created_at,
                        plan.description,
                        plan.planning_model,
                        plan.execution_model,
                        plan.estimated_total_cost,
                        plan.status,
                        plan.git_checkpoint,
                        plan_metadata,
                    ),
                )

                # Remove existing steps and re-insert
                self.db.execute(
                    "DELETE FROM plan_steps WHERE plan_id = ?",
                    (plan.id,),
                )

                for step in plan.steps:
                    step_metadata = json.dumps({
                        "files_to_modify": step.files_to_modify,
                        "estimated_cost": step.estimated_cost,
                        "risk_level": step.risk_level,
                        "validation": step.validation,
                        "rollback": step.rollback,
                    })
                    self.db.execute(
                        """
                        INSERT INTO plan_steps (
                            id, plan_id, order_num,
                            description, tool_calls,
                            estimated_tokens, status,
                            result, error, metadata
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            step.id,
                            plan.id,
                            step.order,
                            step.description,
                            json.dumps(step.tool_calls),
                            step.estimated_tokens,
                            step.status.value,
                            step.result,
                            step.error,
                            step_metadata,
                        ),
                    )

            logger.debug(
                "plan_saved",
                plan_id=plan.id,
                step_count=len(plan.steps),
            )
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(
                f"Failed to save plan: {exc}"
            ) from exc

    def update_step_status(
        self,
        step_id: str,
        status: StepStatus,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        """Update a single step's status, result, and error fields.

        Args:
            step_id: The step's UUID.
            status: New status value.
            result: Optional result text.
            error: Optional error text.
        """
        try:
            self.db.execute(
                """
                UPDATE plan_steps
                SET status = ?, result = ?, error = ?
                WHERE id = ?
                """,
                (status.value, result, error, step_id),
            )
            self.db.commit()
            logger.debug(
                "step_status_updated",
                step_id=step_id,
                status=status.value,
            )
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Failed to update step status: {exc}") from exc

    def delete_plan(self, plan_id: str) -> None:
        """Delete a plan and all its steps from the database.

        Args:
            plan_id: The plan's UUID.
        """
        try:
            with self.db.transaction():
                self.db.execute(
                    "DELETE FROM plan_steps WHERE plan_id = ?",
                    (plan_id,),
                )
                self.db.execute(
                    "DELETE FROM plans WHERE id = ?",
                    (plan_id,),
                )
            logger.debug("plan_deleted", plan_id=plan_id)
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Failed to delete plan: {exc}") from exc

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def load_plan(self, plan_id: str) -> Plan | None:
        """Load a plan by ID.

        Args:
            plan_id: The plan's UUID.

        Returns:
            The plan with all its steps, or ``None`` if not found.
        """
        try:
            row = self.db.fetchone(
                "SELECT * FROM plans WHERE id = ?",
                (plan_id,),
            )
            if row is None:
                return None

            step_rows = self.db.fetchall(
                "SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY order_num",
                (plan_id,),
            )

            steps = [_row_to_step(sr) for sr in step_rows]
            return _row_to_plan(row, steps)
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Failed to load plan: {exc}") from exc

    def list_plans(
        self, status: str | None = None,
    ) -> list[Plan]:
        """List all plans, optionally filtered by status.

        Each plan includes its steps and enhanced metadata
        (goal_summary, risk_assessment, etc.).

        Args:
            status: Optional status filter
                (e.g. ``"draft"``, ``"completed"``).

        Returns:
            List of plans ordered by creation time (newest first).
        """
        try:
            if status is not None:
                plan_rows = self.db.fetchall(
                    "SELECT * FROM plans "
                    "WHERE status = ? "
                    "ORDER BY created_at DESC",
                    (status,),
                )
            else:
                plan_rows = self.db.fetchall(
                    "SELECT * FROM plans "
                    "ORDER BY created_at DESC",
                )

            plans: list[Plan] = []
            for pr in plan_rows:
                step_rows = self.db.fetchall(
                    "SELECT * FROM plan_steps "
                    "WHERE plan_id = ? "
                    "ORDER BY order_num",
                    (pr["id"],),
                )
                steps = [_row_to_step(sr) for sr in step_rows]
                plans.append(_row_to_plan(pr, steps))

            return plans
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(
                f"Failed to list plans: {exc}"
            ) from exc

    @staticmethod
    def format_plan_summary(plan: Plan) -> str:
        """Format a plan as a one-line summary string.

        Useful for listing plans in a compact table view.

        Args:
            plan: The plan to summarise.

        Returns:
            A string like ``"[draft] Refactor auth (3 steps, ..."``.
        """
        goal = plan.goal_summary or plan.description
        if len(goal) > 50:
            goal = goal[:47] + "..."
        return (
            f"[{plan.status}] {goal} "
            f"({len(plan.steps)} steps, "
            f"${plan.estimated_total_cost:.4f}, "
            f"{plan.created_at[:10]})"
        )


# ------------------------------------------------------------------
# Row-to-model converters
# ------------------------------------------------------------------


def _safe_json_load(raw: object, default: object = None) -> object:
    """Parse a JSON string, returning *default* on failure.

    Args:
        raw: Value that may be a JSON string or ``None``.
        default: Fallback value if parsing fails.

    Returns:
        Parsed object or *default*.
    """
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return default
    return default


def _row_to_step(row: object) -> PlanStep:
    """Convert a ``sqlite3.Row`` to a :class:`PlanStep`.

    Enhanced fields are read from the ``metadata`` JSON column
    if present; otherwise defaults are used.
    """
    tool_calls_raw = row["tool_calls"]
    tool_calls: list[dict[str, object]] = (
        json.loads(tool_calls_raw)
        if isinstance(tool_calls_raw, str)
        else tool_calls_raw or []
    )

    # Parse step metadata (may be absent on old databases)
    meta_raw = None
    with contextlib.suppress(KeyError, IndexError):
        meta_raw = row["metadata"]
    meta: dict[str, object] = (
        _safe_json_load(meta_raw, {}) or {}  # type: ignore[assignment]
    )

    return PlanStep(
        id=row["id"],
        order=row["order_num"],
        description=row["description"],
        tool_calls=tool_calls,
        estimated_tokens=row["estimated_tokens"],
        status=StepStatus(row["status"]),
        result=row["result"],
        error=row["error"],
        files_to_modify=meta.get(  # type: ignore[union-attr]
            "files_to_modify", [],
        ),
        estimated_cost=float(
            meta.get("estimated_cost", 0.0),  # type: ignore[union-attr]
        ),
        risk_level=str(
            meta.get("risk_level", RISK_LOW),  # type: ignore[union-attr]
        ),
        validation=str(
            meta.get("validation", ""),  # type: ignore[union-attr]
        ),
        rollback=str(
            meta.get("rollback", ""),  # type: ignore[union-attr]
        ),
    )


def _row_to_plan(
    row: object, steps: list[PlanStep],
) -> Plan:
    """Convert a ``sqlite3.Row`` to a :class:`Plan`.

    Enhanced fields are read from the ``metadata`` JSON column
    if present; otherwise defaults are used.
    """
    # Parse plan metadata (may be absent on old databases)
    meta_raw = None
    with contextlib.suppress(KeyError, IndexError):
        meta_raw = row["metadata"]
    meta: dict[str, object] = (
        _safe_json_load(meta_raw, {}) or {}  # type: ignore[assignment]
    )

    return Plan(
        id=row["id"],
        created_at=row["created_at"],
        description=row["description"],
        steps=steps,
        planning_model=row["planning_model"],
        execution_model=row["execution_model"],
        estimated_total_cost=row["estimated_total_cost"],
        status=row["status"],
        git_checkpoint=row["git_checkpoint"],
        goal_summary=str(
            meta.get("goal_summary", ""),  # type: ignore[union-attr]
        ),
        preconditions=meta.get(  # type: ignore[union-attr]
            "preconditions", [],
        ),
        postconditions=meta.get(  # type: ignore[union-attr]
            "postconditions", [],
        ),
        risk_assessment=str(
            meta.get("risk_assessment", ""),  # type: ignore[union-attr]
        ),
        estimated_time_minutes=float(
            meta.get("estimated_time_minutes", 0.0),  # type: ignore[union-attr]
        ),
        git_start_hash=str(
            meta.get("git_start_hash", ""),  # type: ignore[union-attr]
        ),
    )

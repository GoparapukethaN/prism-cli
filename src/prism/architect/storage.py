"""Plan persistence for Architect Mode.

Stores and retrieves :class:`Plan` and :class:`PlanStep` objects in the
SQLite database managed by :class:`prism.db.database.Database`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from prism.architect.planner import Plan, PlanStep, StepStatus
from prism.exceptions import DatabaseError

if TYPE_CHECKING:
    from prism.db.database import Database

logger = structlog.get_logger(__name__)


class PlanStorage:
    """Store and retrieve plans from SQLite.

    Plans are stored in the ``plans`` table and their steps in
    ``plan_steps``.  Both tables are created by migration 2 in
    :mod:`prism.db.migrations`.
    """

    def __init__(self, db: Database) -> None:
        """Initialise with a database connection.

        Args:
            db: Initialised :class:`Database` instance.
        """
        self.db = db

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_plan(self, plan: Plan) -> None:
        """Save a plan and all its steps to the database.

        If a plan with the same ID already exists it is replaced (upsert
        semantics via ``INSERT OR REPLACE``).

        Args:
            plan: The plan to persist.
        """
        try:
            with self.db.transaction():
                self.db.execute(
                    """
                    INSERT OR REPLACE INTO plans (
                        id, created_at, description, planning_model,
                        execution_model, estimated_total_cost, status,
                        git_checkpoint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                    ),
                )

                # Remove existing steps and re-insert
                self.db.execute(
                    "DELETE FROM plan_steps WHERE plan_id = ?",
                    (plan.id,),
                )

                for step in plan.steps:
                    self.db.execute(
                        """
                        INSERT INTO plan_steps (
                            id, plan_id, order_num, description,
                            tool_calls, estimated_tokens, status,
                            result, error
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            raise DatabaseError(f"Failed to save plan: {exc}") from exc

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

    def list_plans(self, status: str | None = None) -> list[Plan]:
        """List all plans, optionally filtered by status.

        Args:
            status: Optional status filter (e.g. ``"draft"``, ``"completed"``).

        Returns:
            List of plans ordered by creation time (most recent first).
        """
        try:
            if status is not None:
                plan_rows = self.db.fetchall(
                    "SELECT * FROM plans WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                )
            else:
                plan_rows = self.db.fetchall(
                    "SELECT * FROM plans ORDER BY created_at DESC",
                )

            plans: list[Plan] = []
            for pr in plan_rows:
                step_rows = self.db.fetchall(
                    "SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY order_num",
                    (pr["id"],),
                )
                steps = [_row_to_step(sr) for sr in step_rows]
                plans.append(_row_to_plan(pr, steps))

            return plans
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(f"Failed to list plans: {exc}") from exc


# ------------------------------------------------------------------
# Row-to-model converters
# ------------------------------------------------------------------


def _row_to_step(row: object) -> PlanStep:
    """Convert a ``sqlite3.Row`` to a :class:`PlanStep`."""
    tool_calls_raw = row["tool_calls"]
    tool_calls: list[dict[str, object]] = (
        json.loads(tool_calls_raw)
        if isinstance(tool_calls_raw, str)
        else tool_calls_raw or []
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
    )


def _row_to_plan(row: object, steps: list[PlanStep]) -> Plan:
    """Convert a ``sqlite3.Row`` to a :class:`Plan`."""
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
    )

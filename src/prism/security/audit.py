"""Structured audit logging for tool executions and routing decisions."""

from __future__ import annotations

import datetime
import json
import logging
import logging.handlers
from typing import TYPE_CHECKING, Any

import structlog

from prism.config.defaults import AUDIT_LOG_BACKUP_COUNT, AUDIT_LOG_MAX_BYTES
from prism.security.secret_filter import SecretFilter

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)


class AuditLogger:
    """Append-only audit log backed by a :class:`RotatingFileHandler`.

    Each log entry is a single JSON line containing:

    - ``timestamp`` — ISO-8601 UTC timestamp
    - ``event`` — event type string (e.g. ``tool_execution``, ``routing_decision``)
    - ``data`` — event-specific payload (sanitized of secrets before writing)

    The audit log rotates when it exceeds ``max_bytes``; up to
    ``backup_count`` old files are kept.

    Usage::

        audit = AuditLogger(Path("/tmp/audit.log"))
        audit.log_tool_execution(
            tool_name="bash",
            command="ls -la",
            exit_code=0,
            duration_ms=42.5,
        )
    """

    def __init__(
        self,
        log_path: Path,
        max_bytes: int = AUDIT_LOG_MAX_BYTES,
        backup_count: int = AUDIT_LOG_BACKUP_COUNT,
    ) -> None:
        """Initialise the audit logger.

        Args:
            log_path: Path to the audit log file.  Parent directories are
                      created automatically.
            max_bytes: Maximum size of a single log file before rotation.
            backup_count: Number of rotated backup files to keep.
        """
        self._log_path = log_path
        self._secret_filter = SecretFilter()

        # Ensure parent directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Set up a dedicated Python logger with a RotatingFileHandler
        self._logger = logging.getLogger(f"prism.audit.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        # Remove any existing handlers (safety for re-initialisation)
        self._logger.handlers.clear()

        handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def log_path(self) -> Path:
        """Return the path to the audit log file."""
        return self._log_path

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_tool_execution(
        self,
        *,
        tool_name: str,
        command: str | None = None,
        args: dict[str, Any] | None = None,
        exit_code: int | None = None,
        duration_ms: float | None = None,
        success: bool = True,
        error: str | None = None,
        user: str | None = None,
    ) -> None:
        """Log a tool execution event.

        Args:
            tool_name:   Name of the tool that was executed.
            command:     Shell command (for command-based tools).
            args:        Tool arguments (sanitized before logging).
            exit_code:   Process exit code (for subprocess tools).
            duration_ms: Wall-clock execution time in milliseconds.
            success:     Whether the execution succeeded.
            error:       Error message, if any.
            user:        Optional user identifier.
        """
        data: dict[str, Any] = {
            "tool_name": tool_name,
            "success": success,
        }
        if command is not None:
            data["command"] = command
        if args is not None:
            data["args"] = self._sanitize(args)
        if exit_code is not None:
            data["exit_code"] = exit_code
        if duration_ms is not None:
            data["duration_ms"] = round(duration_ms, 2)
        if error is not None:
            data["error"] = error
        if user is not None:
            data["user"] = user

        self._write_entry("tool_execution", data)

    def log_routing_decision(
        self,
        *,
        task_tier: str,
        selected_model: str,
        candidates: list[str] | None = None,
        scores: dict[str, float] | None = None,
        reason: str | None = None,
        budget_remaining: float | None = None,
        estimated_cost: float | None = None,
    ) -> None:
        """Log a routing decision event.

        Args:
            task_tier:        Classified task tier (simple / medium / complex).
            selected_model:   The model that was chosen.
            candidates:       All candidate models considered.
            scores:           Per-model scores used for ranking.
            reason:           Human-readable reason for the decision.
            budget_remaining: Remaining budget in USD.
            estimated_cost:   Estimated cost for the chosen model.
        """
        data: dict[str, Any] = {
            "task_tier": task_tier,
            "selected_model": selected_model,
        }
        if candidates is not None:
            data["candidates"] = candidates
        if scores is not None:
            data["scores"] = scores
        if reason is not None:
            data["reason"] = reason
        if budget_remaining is not None:
            data["budget_remaining"] = round(budget_remaining, 4)
        if estimated_cost is not None:
            data["estimated_cost"] = round(estimated_cost, 6)

        self._write_entry("routing_decision", data)

    def log_security_event(
        self,
        *,
        event_type: str,
        detail: str,
        path: str | None = None,
        command: str | None = None,
        pattern: str | None = None,
    ) -> None:
        """Log a security-related event (blocked path, blocked command, etc.).

        Args:
            event_type: Sub-type such as ``path_traversal``, ``blocked_command``.
            detail:     Human-readable description.
            path:       Offending path, if applicable.
            command:    Offending command, if applicable.
            pattern:    The security pattern that triggered the block.
        """
        data: dict[str, Any] = {
            "event_type": event_type,
            "detail": detail,
        }
        if path is not None:
            data["path"] = path
        if command is not None:
            data["command"] = command
        if pattern is not None:
            data["pattern"] = pattern

        self._write_entry("security_event", data)

    def log_raw(self, event: str, data: dict[str, Any]) -> None:
        """Log an arbitrary structured event.

        Args:
            event: Event type string.
            data:  Event payload (will be sanitized).
        """
        self._write_entry(event, self._sanitize(data))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_entry(self, event: str, data: dict[str, Any]) -> None:
        """Format and write a single JSON-line audit entry."""
        entry: dict[str, Any] = {
            "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            "event": event,
            "data": data,
        }
        try:
            line = json.dumps(entry, default=str, ensure_ascii=False)
            self._logger.info(line)
        except Exception:
            # Audit logging must never crash the application.
            logger.exception("audit_log_write_failed", event=event)

    def _sanitize(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize *data* by redacting values whose keys look sensitive."""
        return self._secret_filter.sanitize_dict(data)  # type: ignore[return-value]

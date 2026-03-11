"""Session persistence — save and load conversation sessions to disk.

Sessions are stored as JSON files in ``~/.prism/sessions/``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from prism.exceptions import ContextError

logger = structlog.get_logger(__name__)


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class SessionManager:
    """Manages session persistence to disk.

    Sessions are stored as JSON files in *sessions_dir* (typically
    ``~/.prism/sessions/``).  Each session file contains conversation
    messages, metadata, and project information.

    Args:
        sessions_dir: Directory for session files.
    """

    def __init__(self, sessions_dir: Path) -> None:
        self._sessions_dir = Path(sessions_dir)
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    @property
    def sessions_dir(self) -> Path:
        return self._sessions_dir

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_session(self, project_root: Path | str) -> str:
        """Create a new empty session and return its ID.

        Args:
            project_root: The project directory this session belongs to.

        Returns:
            A newly generated session ID (UUID).
        """
        session_id = str(uuid.uuid4())
        now = _utcnow_iso()

        session_data: dict[str, Any] = {
            "session_id": session_id,
            "project_root": str(project_root),
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "metadata": {},
        }

        self._write_session(session_id, session_data)
        logger.debug("session_created", session_id=session_id)
        return session_id

    def save_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Persist session data to disk.

        The ``updated_at`` timestamp is refreshed automatically.

        Args:
            session_id: The session identifier.
            data: Session data dictionary (must contain at least
                ``session_id``).

        Raises:
            ContextError: If *session_id* is empty.
        """
        if not session_id:
            raise ContextError("session_id must not be empty")

        data["session_id"] = session_id
        data["updated_at"] = _utcnow_iso()

        self._write_session(session_id, data)
        logger.debug("session_saved", session_id=session_id)

    def load_session(self, session_id: str) -> dict[str, Any]:
        """Load session data from disk.

        Args:
            session_id: The session identifier.

        Returns:
            The session data dictionary.

        Raises:
            ContextError: If the session file does not exist or is corrupt.
        """
        path = self._session_path(session_id)
        if not path.is_file():
            raise ContextError(f"Session not found: {session_id}")

        try:
            raw = path.read_text(encoding="utf-8")
            data: dict[str, Any] = json.loads(raw)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            raise ContextError(
                f"Failed to load session {session_id}: {exc}"
            ) from exc

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file from disk.

        Args:
            session_id: The session identifier.

        Returns:
            *True* if the file was deleted, *False* if it did not exist.
        """
        path = self._session_path(session_id)
        if path.is_file():
            try:
                path.unlink()
                logger.debug("session_deleted", session_id=session_id)
                return True
            except OSError as exc:
                raise ContextError(
                    f"Failed to delete session {session_id}: {exc}"
                ) from exc
        return False

    def list_sessions(
        self,
        project_root: Path | str | None = None,
    ) -> list[dict[str, Any]]:
        """List all sessions, optionally filtered by *project_root*.

        Args:
            project_root: If provided, only return sessions belonging to
                this project directory.

        Returns:
            A list of session metadata dicts, sorted by ``updated_at``
            descending (most recent first).
        """
        sessions: list[dict[str, Any]] = []
        project_str = str(project_root) if project_root else None

        for path in self._sessions_dir.glob("*.json"):
            try:
                raw = path.read_text(encoding="utf-8")
                data: dict[str, Any] = json.loads(raw)
            except (json.JSONDecodeError, OSError):
                logger.warning("session_file_corrupt", path=str(path))
                continue

            if project_str and data.get("project_root") != project_str:
                continue

            sessions.append({
                "session_id": data.get("session_id", path.stem),
                "project_root": data.get("project_root", ""),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def session_exists(self, session_id: str) -> bool:
        """Return *True* if a session file exists for *session_id*."""
        return self._session_path(session_id).is_file()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_path(self, session_id: str) -> Path:
        """Return the file path for a given session ID."""
        return self._sessions_dir / f"{session_id}.json"

    def _write_session(self, session_id: str, data: dict[str, Any]) -> None:
        """Write session data to disk as JSON."""
        path = self._session_path(session_id)
        try:
            path.write_text(
                json.dumps(data, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ContextError(
                f"Failed to write session {session_id}: {exc}"
            ) from exc

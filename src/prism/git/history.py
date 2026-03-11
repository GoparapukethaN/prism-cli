"""Full rollback history -- session timeline, diff viewing, and state restoration.

Every file edit creates a git commit automatically (via AutoCommitter).  This
module tracks those commits per session and provides:

- ``/undo [N]`` -- revert the last *N* changes.
- ``/history`` -- display a coloured timeline of all session changes.
- ``/history diff <n>`` -- show the full diff of the *n*-th change.
- ``/restore <hash>`` -- restore the working tree to any previous state.

All operations preserve history by creating new commits (never rewriting).
"""

from __future__ import annotations

import contextlib
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangeRecord:
    """A single change in the session timeline.

    Attributes:
        index: 1-based position in the session timeline.
        commit_hash: Full 40-character commit SHA.
        short_hash: First 8 characters of the commit SHA.
        message: Commit message text.
        timestamp: ISO-8601 timestamp of the commit.
        files_changed: List of file paths touched by this commit.
        insertions: Number of lines added.
        deletions: Number of lines removed.
    """

    index: int
    commit_hash: str
    short_hash: str
    message: str
    timestamp: str
    files_changed: list[str]
    insertions: int
    deletions: int


@dataclass
class SessionTimeline:
    """Timeline of all changes in the current session.

    Attributes:
        session_id: Unique identifier for the session.
        changes: Ordered list of changes (oldest first).
        start_commit: Commit hash when the session was started.
    """

    session_id: str
    changes: list[ChangeRecord] = field(default_factory=list)
    start_commit: str = ""


# ---------------------------------------------------------------------------
# RollbackManager
# ---------------------------------------------------------------------------


class RollbackManager:
    """Manages full rollback history with session timeline and restore operations.

    All restore/undo operations create new commits so that no work is ever
    lost -- the full git history is always preserved.

    Args:
        project_root: Root directory of the git repository.
    """

    def __init__(self, project_root: Path) -> None:
        self._root = project_root.resolve()
        self._session_start_commit: str = ""
        self._session_changes: list[ChangeRecord] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_start_commit(self) -> str:
        """Return the commit hash recorded at session start."""
        return self._session_start_commit

    @property
    def change_count(self) -> int:
        """Return the number of changes recorded this session."""
        return len(self._session_changes)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> str:
        """Record the current HEAD as the session start point.

        Returns:
            The current HEAD commit hash.

        Raises:
            RuntimeError: If the git command fails (e.g. empty repo).
        """
        result = self._git("rev-parse", "HEAD")
        self._session_start_commit = result.strip()
        self._session_changes = []
        logger.info(
            "session_started",
            start_commit=self._session_start_commit[:8],
        )
        return self._session_start_commit

    # ------------------------------------------------------------------
    # Recording changes
    # ------------------------------------------------------------------

    def record_change(self, commit_hash: str, message: str = "") -> ChangeRecord:
        """Record a new change after an auto-commit.

        Args:
            commit_hash: The full SHA of the new commit.
            message: Optional override for the commit message.  If empty,
                the message is read from the commit itself.

        Returns:
            The newly created :class:`ChangeRecord`.

        Raises:
            RuntimeError: If the git command fails.
        """
        if not commit_hash or not commit_hash.strip():
            raise ValueError("commit_hash must not be empty")

        commit_hash = commit_hash.strip()

        if not message:
            msg_result = self._git("log", "-1", "--format=%s", commit_hash)
            message = msg_result.strip()

        files, insertions, deletions = self._parse_stat(commit_hash)

        ts_result = self._git("log", "-1", "--format=%aI", commit_hash)

        record = ChangeRecord(
            index=len(self._session_changes) + 1,
            commit_hash=commit_hash,
            short_hash=commit_hash[:8],
            message=message,
            timestamp=ts_result.strip(),
            files_changed=files,
            insertions=insertions,
            deletions=deletions,
        )
        self._session_changes.append(record)

        logger.info(
            "change_recorded",
            index=record.index,
            short_hash=record.short_hash,
            message=record.message,
        )
        return record

    # ------------------------------------------------------------------
    # Timeline / history
    # ------------------------------------------------------------------

    def get_timeline(self) -> SessionTimeline:
        """Return the full session timeline.

        Returns:
            A :class:`SessionTimeline` containing all recorded changes.
        """
        return SessionTimeline(
            session_id="",
            changes=list(self._session_changes),
            start_commit=self._session_start_commit,
        )

    def get_change(self, change_index: int) -> ChangeRecord:
        """Return a specific change by its 1-based index.

        Args:
            change_index: 1-based position in the timeline.

        Returns:
            The matching :class:`ChangeRecord`.

        Raises:
            ValueError: If *change_index* is out of range.
        """
        self._validate_index(change_index)
        return self._session_changes[change_index - 1]

    # ------------------------------------------------------------------
    # Diff viewing
    # ------------------------------------------------------------------

    def get_diff(self, change_index: int) -> str:
        """Return the full diff for a specific change by its 1-based index.

        Args:
            change_index: 1-based position in the timeline.

        Returns:
            Coloured unified diff text.

        Raises:
            ValueError: If *change_index* is out of range.
            RuntimeError: If the git diff command fails.
        """
        self._validate_index(change_index)
        change = self._session_changes[change_index - 1]
        return self._git(
            "diff",
            "--color=always",
            f"{change.commit_hash}~1..{change.commit_hash}",
        )

    def get_restore_preview(self, commit_hash: str) -> str:
        """Preview the diff that restoring to *commit_hash* would produce.

        Args:
            commit_hash: Target commit hash.

        Returns:
            A ``--stat`` summary of what would change.

        Raises:
            RuntimeError: If *commit_hash* is invalid.
        """
        self._validate_commit(commit_hash)
        return self._git("diff", "--color=always", "--stat", f"HEAD..{commit_hash}")

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def undo(self, count: int = 1) -> list[str]:
        """Undo the last *count* changes by reverting their commits.

        Each revert creates a new commit, preserving full history.

        Args:
            count: Number of changes to undo (default 1).

        Returns:
            List of commit hashes that were reverted (most recent first).

        Raises:
            ValueError: If *count* < 1 or exceeds available changes.
            RuntimeError: If a ``git revert`` fails.
        """
        if count < 1:
            raise ValueError("Count must be at least 1")
        if count > len(self._session_changes):
            raise ValueError(
                f"Only {len(self._session_changes)} changes to undo, "
                f"requested {count}"
            )

        reverted: list[str] = []
        for _ in range(count):
            change = self._session_changes.pop()
            self._git("revert", "--no-edit", change.commit_hash)
            reverted.append(change.commit_hash)
            logger.info("change_undone", short_hash=change.short_hash)

        return reverted

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self, commit_hash: str) -> str:
        """Restore the working tree to the state at *commit_hash*.

        Creates a **new** commit with the restored state so that no
        history is lost.

        Args:
            commit_hash: Target commit hash (full or abbreviated).

        Returns:
            The full hash of the new restore commit.

        Raises:
            RuntimeError: If the commit hash is invalid or git fails.
        """
        self._validate_commit(commit_hash)

        # Restore all files from the target commit
        self._git("checkout", commit_hash, "--", ".")
        self._git("add", "-A")

        restore_msg = f"prism: restore to {commit_hash[:8]}"
        self._git("commit", "-m", restore_msg, "--allow-empty")

        result = self._git("rev-parse", "HEAD")
        new_hash = result.strip()

        logger.info(
            "state_restored",
            target=commit_hash[:8],
            new_commit=new_hash[:8],
        )
        return new_hash

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_index(self, change_index: int) -> None:
        """Raise ValueError if *change_index* is out of range."""
        if change_index < 1 or change_index > len(self._session_changes):
            raise ValueError(
                f"Invalid change index: {change_index}. "
                f"Valid range: 1-{len(self._session_changes)}"
            )

    def _validate_commit(self, commit_hash: str) -> None:
        """Raise RuntimeError if *commit_hash* does not exist."""
        if not commit_hash or not commit_hash.strip():
            raise ValueError("commit_hash must not be empty")
        self._git("cat-file", "-t", commit_hash)

    def _parse_stat(self, commit_hash: str) -> tuple[list[str], int, int]:
        """Parse ``git diff --stat`` output for a single commit.

        Args:
            commit_hash: The commit to inspect.

        Returns:
            Tuple of (files_changed, insertions, deletions).
        """
        stat_result = self._git(
            "diff", "--stat", f"{commit_hash}~1..{commit_hash}"
        )
        files: list[str] = []
        insertions = 0
        deletions = 0

        for line in stat_result.strip().split("\n"):
            if "|" in line:
                fname = line.split("|")[0].strip()
                if fname:
                    files.append(fname)
            # Summary line: "2 files changed, 5 insertions(+), 3 deletions(-)"
            if "insertion" in line or "deletion" in line:
                parts = line.split(",")
                for raw_part in parts:
                    segment = raw_part.strip()
                    if "insertion" in segment:
                        with contextlib.suppress(ValueError, IndexError):
                            insertions = int(segment.split()[0])
                    if "deletion" in segment:
                        with contextlib.suppress(ValueError, IndexError):
                            deletions = int(segment.split()[0])

        return files, insertions, deletions

    def _git(self, *args: str) -> str:
        """Run a git command in the project root.

        Args:
            *args: Git subcommand and arguments.

        Returns:
            Standard output as a string.

        Raises:
            RuntimeError: If the command exits with a non-zero status
                (unless the subcommand is ``revert``, which may produce
                benign warnings).
        """
        result = subprocess.run(
            ["git", *args],
            cwd=str(self._root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0 and args[0] != "revert":
            raise RuntimeError(f"Git error: {result.stderr.strip()}")
        return result.stdout

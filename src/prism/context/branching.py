"""Conversation branching -- create, switch, and merge conversation branches.

Allows users to explore multiple approaches in parallel by forking the
conversation at any point and switching between named branches.

Commands:

- ``/branch <name>`` -- create a named branch from the current conversation.
- ``/branches`` -- list all conversation branches.
- ``/switch <name>`` -- switch to a different branch.
- ``/merge <name>`` -- merge a branch back into the current conversation.

Branch metadata is persisted as JSON files under ``~/.prism/branches/``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_BRANCH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _validate_branch_name(name: str) -> None:
    """Raise ValueError if *name* is not a valid branch name.

    Valid names start with an alphanumeric character and contain only
    alphanumerics, hyphens, and underscores.

    Args:
        name: Proposed branch name.

    Raises:
        ValueError: If *name* is empty or contains invalid characters.
    """
    if not name:
        raise ValueError("Branch name must not be empty")
    if not _BRANCH_NAME_RE.match(name):
        raise ValueError(
            f"Invalid branch name: '{name}'. "
            "Use alphanumeric characters, hyphens, and underscores only. "
            "Must start with an alphanumeric character."
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BranchMetadata:
    """Metadata for a conversation branch.

    Attributes:
        name: Unique branch name.
        created_at: ISO-8601 creation timestamp.
        parent_branch: Name of the branch this was forked from.
        fork_point_index: Message index where the branch was created.
        git_commit_at_fork: Git commit hash at the moment of branching.
        message_count: Total number of messages in the branch.
        is_active: Whether this branch is currently active.
        description: Optional human-readable description.
    """

    name: str
    created_at: str
    parent_branch: str
    fork_point_index: int
    git_commit_at_fork: str
    message_count: int = 0
    is_active: bool = False
    description: str = ""


@dataclass
class ConversationBranch:
    """A complete conversation branch with messages and metadata.

    Attributes:
        metadata: Branch metadata.
        messages: Conversation messages (list of ``{"role": ..., "content": ...}``).
        file_edits: List of commit hashes for file edits made on this branch.
    """

    metadata: BranchMetadata
    messages: list[dict[str, Any]] = field(default_factory=list)
    file_edits: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# BranchManager
# ---------------------------------------------------------------------------


class BranchManager:
    """Manages conversation branches with disk persistence.

    Branches are stored as individual JSON files in *branches_dir*.
    Each file contains the full branch metadata, message history, and
    file edit tracking.

    Args:
        branches_dir: Directory for branch JSON files (created if missing).
    """

    def __init__(self, branches_dir: Path) -> None:
        self._dir = Path(branches_dir).resolve()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._branches: dict[str, ConversationBranch] = {}
        self._active_branch: str = "main"
        self._load_all()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_branch(self) -> str:
        """Return the name of the currently active branch."""
        return self._active_branch

    @property
    def branch_count(self) -> int:
        """Return the total number of branches."""
        return len(self._branches)

    # ------------------------------------------------------------------
    # Branch CRUD
    # ------------------------------------------------------------------

    def create_branch(
        self,
        name: str,
        current_messages: list[dict[str, Any]],
        git_commit: str = "",
        description: str = "",
    ) -> BranchMetadata:
        """Create a new conversation branch from the current state.

        The current branch state is saved before the new branch is
        activated.  The new branch inherits a copy of *current_messages*.

        Args:
            name: Unique branch name.
            current_messages: Messages in the current conversation.
            git_commit: Git commit hash at the moment of branching.
            description: Optional human-readable description.

        Returns:
            Metadata for the newly created branch.

        Raises:
            ValueError: If *name* already exists or is invalid.
        """
        _validate_branch_name(name)

        if name in self._branches:
            raise ValueError(f"Branch '{name}' already exists")

        now = datetime.now(UTC).isoformat()

        # Save current branch state before creating a new one
        if self._active_branch in self._branches:
            self._branches[self._active_branch].messages = list(current_messages)
            self._branches[self._active_branch].metadata.is_active = False
            self._save_branch(self._active_branch)

        metadata = BranchMetadata(
            name=name,
            created_at=now,
            parent_branch=self._active_branch,
            fork_point_index=len(current_messages),
            git_commit_at_fork=git_commit,
            message_count=len(current_messages),
            is_active=True,
            description=description,
        )

        branch = ConversationBranch(
            metadata=metadata,
            messages=list(current_messages),
            file_edits=[],
        )

        self._branches[name] = branch
        self._active_branch = name
        self._save_branch(name)

        logger.info(
            "branch_created",
            name=name,
            parent=metadata.parent_branch,
            messages=len(current_messages),
        )
        return metadata

    def list_branches(self) -> list[BranchMetadata]:
        """List all branches sorted by creation time (oldest first).

        Returns:
            List of :class:`BranchMetadata` for every branch.
        """
        branches = [b.metadata for b in self._branches.values()]
        branches.sort(key=lambda m: m.created_at)
        return branches

    def get_branch(self, name: str) -> ConversationBranch:
        """Return a specific branch by name.

        Args:
            name: Branch name.

        Returns:
            The :class:`ConversationBranch`.

        Raises:
            ValueError: If *name* does not exist.
        """
        if name not in self._branches:
            raise ValueError(f"Branch '{name}' not found")
        return self._branches[name]

    def switch_branch(
        self,
        name: str,
        current_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Switch to a different conversation branch.

        Saves the current branch state, then activates the target branch
        and returns its message history.

        Args:
            name: Name of the target branch.
            current_messages: Messages in the current conversation
                (saved to the current branch before switching).

        Returns:
            The message history of the target branch.

        Raises:
            ValueError: If *name* does not exist.
        """
        if name not in self._branches:
            raise ValueError(f"Branch '{name}' not found")

        # Save current branch
        if self._active_branch in self._branches:
            cur = self._branches[self._active_branch]
            cur.messages = list(current_messages)
            cur.metadata.is_active = False
            cur.metadata.message_count = len(current_messages)
            self._save_branch(self._active_branch)

        # Activate target
        target = self._branches[name]
        target.metadata.is_active = True
        self._active_branch = name
        self._save_branch(name)

        logger.info(
            "branch_switched",
            name=name,
            messages=len(target.messages),
        )
        return list(target.messages)

    def merge_branch(
        self,
        source_name: str,
        current_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge messages from *source_name* into the active branch.

        Messages that are unique to the source branch (created after the
        fork point) are appended to the current conversation.

        Args:
            source_name: Name of the branch to merge from.
            current_messages: Current conversation messages.

        Returns:
            The merged message list.

        Raises:
            ValueError: If *source_name* does not exist or is the
                active branch.
        """
        if source_name not in self._branches:
            raise ValueError(f"Branch '{source_name}' not found")
        if source_name == self._active_branch:
            raise ValueError("Cannot merge a branch into itself")

        source = self._branches[source_name]
        fork_idx = source.metadata.fork_point_index

        # Messages unique to the source branch (after fork)
        new_messages = source.messages[fork_idx:]
        merged = list(current_messages) + new_messages

        # Update current branch
        if self._active_branch in self._branches:
            cur = self._branches[self._active_branch]
            cur.messages = merged
            cur.metadata.message_count = len(merged)
            self._save_branch(self._active_branch)

        logger.info(
            "branch_merged",
            source=source_name,
            target=self._active_branch,
            new_messages=len(new_messages),
        )
        return merged

    def delete_branch(self, name: str) -> None:
        """Delete a conversation branch.

        Args:
            name: Name of the branch to delete.

        Raises:
            ValueError: If *name* does not exist or is the active branch.
        """
        if name not in self._branches:
            raise ValueError(f"Branch '{name}' not found")
        if name == self._active_branch:
            raise ValueError("Cannot delete the active branch")

        del self._branches[name]
        path = self._dir / f"{name}.json"
        path.unlink(missing_ok=True)

        logger.info("branch_deleted", name=name)

    # ------------------------------------------------------------------
    # File edit tracking
    # ------------------------------------------------------------------

    def add_file_edit(self, commit_hash: str) -> None:
        """Record a file-edit commit hash on the active branch.

        Args:
            commit_hash: The git commit hash of the file edit.
        """
        if not commit_hash:
            raise ValueError("commit_hash must not be empty")

        if self._active_branch in self._branches:
            self._branches[self._active_branch].file_edits.append(commit_hash)
            self._save_branch(self._active_branch)
            logger.debug(
                "file_edit_recorded",
                branch=self._active_branch,
                commit=commit_hash[:8],
            )

    def get_file_edits(self, name: str | None = None) -> list[str]:
        """Return file edit commit hashes for a branch.

        Args:
            name: Branch name. Defaults to the active branch.

        Returns:
            List of commit hashes.

        Raises:
            ValueError: If *name* does not exist.
        """
        target = name or self._active_branch
        if target not in self._branches:
            raise ValueError(f"Branch '{target}' not found")
        return list(self._branches[target].file_edits)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all branch data from disk."""
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                meta = BranchMetadata(**data["metadata"])
                branch = ConversationBranch(
                    metadata=meta,
                    messages=data.get("messages", []),
                    file_edits=data.get("file_edits", []),
                )
                self._branches[meta.name] = branch
                if meta.is_active:
                    self._active_branch = meta.name
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "branch_load_error",
                    path=str(path),
                    error=str(exc),
                )

    def _save_branch(self, name: str) -> None:
        """Persist a single branch to disk as JSON.

        Args:
            name: Branch name to save.
        """
        if name not in self._branches:
            return

        branch = self._branches[name]
        data: dict[str, Any] = {
            "metadata": asdict(branch.metadata),
            "messages": branch.messages,
            "file_edits": branch.file_edits,
        }
        path = self._dir / f"{name}.json"
        path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )

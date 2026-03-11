"""Git repository operations.

Wraps the git CLI to provide structured access to repository status,
diffs, logs, and branch information.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from prism.exceptions import GitError, NotAGitRepoError


@dataclass(frozen=True)
class GitStatus:
    """Structured representation of ``git status``."""

    staged: list[str] = field(default_factory=list)
    unstaged: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Return True when the working tree has no changes."""
        return not self.staged and not self.unstaged and not self.untracked

    @property
    def total_changes(self) -> int:
        """Total number of changed files across all categories."""
        return len(self.staged) + len(self.unstaged) + len(self.untracked)


@dataclass(frozen=True)
class GitLogEntry:
    """A single git log entry."""

    hash: str
    author: str
    date: str
    message: str


class GitRepo:
    """High-level wrapper around the git CLI for a single repository.

    All operations use ``subprocess.run`` with a configurable timeout
    and raise :class:`GitError` on failures.
    """

    DEFAULT_TIMEOUT: int = 30  # seconds

    def __init__(self, project_root: Path, *, timeout: int | None = None) -> None:
        """Initialise and validate a git repository path.

        Args:
            project_root: Path to the root of the git repository.
            timeout: Subprocess timeout in seconds (default 30).

        Raises:
            NotAGitRepoError: If *project_root* is not inside a git repository.
        """
        self._root = Path(project_root).resolve()
        self._timeout = timeout or self.DEFAULT_TIMEOUT

        if not GitRepo.is_git_repo(self._root):
            raise NotAGitRepoError(str(self._root))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Return the resolved repository root path."""
        return self._root

    def status(self) -> GitStatus:
        """Return structured repository status.

        Returns:
            A :class:`GitStatus` with staged, unstaged and untracked files.
        """
        output = self._run(["git", "status", "--porcelain", "-z"])
        return self._parse_status(output)

    def diff(self, *, staged: bool = False) -> str:
        """Return the diff output.

        Args:
            staged: If True return the staged (cached) diff.

        Returns:
            Raw diff text.
        """
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")
        return self._run(cmd)

    def log(self, n: int = 10) -> list[GitLogEntry]:
        """Return recent log entries.

        Args:
            n: Maximum number of entries to return.

        Returns:
            List of :class:`GitLogEntry` (most recent first).
        """
        # Use a delimiter unlikely to appear in messages
        sep = "---PRISM_SEP---"
        fmt = f"%H{sep}%an{sep}%ai{sep}%s"
        output = self._run(["git", "log", f"-{n}", f"--format={fmt}"])
        if not output.strip():
            return []

        entries: list[GitLogEntry] = []
        for line in output.strip().splitlines():
            parts = line.split(sep)
            if len(parts) >= 4:
                entries.append(
                    GitLogEntry(
                        hash=parts[0].strip(),
                        author=parts[1].strip(),
                        date=parts[2].strip(),
                        message=parts[3].strip(),
                    )
                )
        return entries

    def current_branch(self) -> str:
        """Return the name of the current branch.

        Returns:
            Branch name string.

        Raises:
            GitError: If the branch name cannot be determined.
        """
        output = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        branch = output.strip()
        if not branch:
            raise GitError("Could not determine current branch")
        return branch

    def add(self, files: list[str] | None = None) -> None:
        """Stage files for commit.

        Args:
            files: Specific files to stage. ``None`` stages all changes.
        """
        cmd = ["git", "add"]
        if files:
            cmd.extend(files)
        else:
            cmd.append("-A")
        self._run(cmd)

    def commit(self, message: str) -> str:
        """Create a commit.

        Args:
            message: Commit message.

        Returns:
            The short hash of the new commit.

        Raises:
            GitError: If the commit fails.
        """
        self._run(["git", "commit", "-m", message])
        return self._run(["git", "rev-parse", "--short", "HEAD"]).strip()

    def undo_last_commit(self) -> str:
        """Revert the last commit (git reset --soft HEAD~1).

        The changes from the undone commit remain staged so the user can
        re-commit or discard them.

        Returns:
            The full hash of the undone commit.

        Raises:
            GitError: If there is no commit to undo or the reset fails.
        """
        commit_hash = self._run(["git", "rev-parse", "HEAD"]).strip()
        self._run(["git", "reset", "--soft", "HEAD~1"])
        return commit_hash

    def get_dirty_files(self) -> list[str]:
        """List all modified and untracked files in the working tree.

        Returns:
            Sorted list of file paths that have been modified, staged,
            or are untracked.
        """
        status = self.status()
        all_files: set[str] = set()
        all_files.update(status.staged)
        all_files.update(status.unstaged)
        all_files.update(status.untracked)
        return sorted(all_files)

    def is_dirty(self) -> bool:
        """Check whether the working tree has uncommitted changes.

        Returns:
            True if there are staged, unstaged, or untracked changes.
        """
        return not self.status().is_clean

    def create_checkpoint(self) -> str:
        """Create a checkpoint commit for later rollback.

        Stages all changes and creates a commit with a ``prism: checkpoint``
        message.  The returned hash can be passed to
        :meth:`reset_to_checkpoint` to restore this state.

        Returns:
            The full hash of the checkpoint commit.

        Raises:
            GitError: If there are no changes to checkpoint.
        """
        self.add()
        self._run(["git", "commit", "-m", "prism: checkpoint"])
        return self._run(["git", "rev-parse", "HEAD"]).strip()

    def reset_to_checkpoint(self, commit_hash: str) -> bool:
        """Reset the repository to a previous checkpoint commit.

        Performs a hard reset to the given commit hash, discarding all
        changes made after the checkpoint.

        Args:
            commit_hash: The commit hash to reset to (as returned by
                :meth:`create_checkpoint`).

        Returns:
            True if the reset succeeded.

        Raises:
            GitError: If the reset fails (e.g. invalid hash).
        """
        self._run(["git", "reset", "--hard", commit_hash])
        return True

    def get_current_commit(self) -> str:
        """Return the full SHA hash of the current HEAD commit.

        Returns:
            40-character commit hash.

        Raises:
            GitError: If HEAD cannot be resolved (e.g. empty repository).
        """
        return self._run(["git", "rev-parse", "HEAD"]).strip()

    def ensure_gitignore_has_prism(self) -> bool:
        """Add ``.prism/`` to ``.gitignore`` if it is not already present.

        Creates the ``.gitignore`` file if it does not exist.

        Returns:
            True if the file was modified, False if ``.prism/`` was
            already listed.
        """
        gitignore_path = self._root / ".gitignore"
        entry = ".prism/"

        if gitignore_path.is_file():
            content = gitignore_path.read_text()
            # Check if .prism/ is already in the file (as its own line)
            for line in content.splitlines():
                if line.strip() == entry:
                    return False
            # Append with a newline if the file doesn't end with one
            if content and not content.endswith("\n"):
                content += "\n"
            content += entry + "\n"
            gitignore_path.write_text(content)
        else:
            gitignore_path.write_text(entry + "\n")

        return True

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_git_repo(path: Path) -> bool:
        """Check whether *path* is inside a git repository.

        Args:
            path: Directory path to test.

        Returns:
            True if the path is inside a git working tree.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, cmd: list[str]) -> str:
        """Execute a git command and return its stdout.

        Args:
            cmd: Command and arguments.

        Returns:
            Standard output as a string.

        Raises:
            GitError: On non-zero exit code or timeout.
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise GitError(f"Git command timed out: {' '.join(cmd)}") from exc
        except FileNotFoundError as exc:
            raise GitError("git executable not found") from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise GitError(f"git command failed: {stderr or 'unknown error'}")

        return result.stdout

    @staticmethod
    def _parse_status(raw: str) -> GitStatus:
        """Parse ``git status --porcelain -z`` output.

        The ``-z`` flag uses NUL terminators so filenames with spaces are safe.

        Args:
            raw: Raw output from ``git status --porcelain -z``.

        Returns:
            Populated :class:`GitStatus`.
        """
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []

        if not raw:
            return GitStatus(staged=staged, unstaged=unstaged, untracked=untracked)

        # Split by NUL; the last element may be empty.
        entries = raw.split("\0")
        i = 0
        while i < len(entries):
            entry = entries[i]
            if len(entry) < 3:
                i += 1
                continue

            index_status = entry[0]
            worktree_status = entry[1]
            filename = entry[3:]

            # Renames have an extra NUL-separated field (the old name)
            if index_status == "R":
                i += 1  # skip the old-name entry

            if index_status == "?" and worktree_status == "?":
                untracked.append(filename)
            else:
                if index_status not in (" ", "?"):
                    staged.append(filename)
                if worktree_status not in (" ", "?"):
                    unstaged.append(filename)

            i += 1

        return GitStatus(staged=staged, unstaged=unstaged, untracked=untracked)

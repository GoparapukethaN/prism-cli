"""Automatic commit message generation and commit heuristics.

Uses rule-based analysis of diffs and file patterns to produce
`Conventional Commits <https://www.conventionalcommits.org/>`_-style
messages without requiring an LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from prism.exceptions import GitError

if TYPE_CHECKING:
    from prism.git.operations import GitRepo, GitStatus

# ---------------------------------------------------------------------------
# File-pattern → prefix mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PrefixRule:
    """Maps a set of file globs / directory names to a commit prefix."""

    prefix: str
    patterns: tuple[str, ...]


_PREFIX_RULES: list[_PrefixRule] = [
    _PrefixRule(prefix="test", patterns=("test_", "_test.py", "tests/", "conftest.py", "spec/")),
    _PrefixRule(prefix="docs", patterns=("docs/", ".md", "README", "CHANGELOG", "LICENSE", ".rst")),
    _PrefixRule(prefix="chore", patterns=(
        ".gitignore", "Makefile", "pyproject.toml", "setup.cfg", "setup.py",
        "requirements", ".pre-commit", "tox.ini", "Dockerfile", "docker-compose",
        ".github/", ".ci/", "Jenkinsfile",
    )),
    _PrefixRule(prefix="fix", patterns=("fix", "bugfix", "hotfix", "patch")),
    _PrefixRule(prefix="refactor", patterns=("refactor",)),
]


def _detect_prefix(changed_files: list[str]) -> str:
    """Detect the conventional-commit prefix from a list of changed paths.

    The first matching rule wins.  Falls back to ``"feat"`` when no rule
    matches.

    Args:
        changed_files: List of file paths that changed.

    Returns:
        A conventional-commit prefix (``feat``, ``fix``, ``test``, etc.).
    """
    if not changed_files:
        return "chore"

    # Count how many files match each prefix rule
    scores: dict[str, int] = {}
    for filepath in changed_files:
        lower = filepath.lower()
        for rule in _PREFIX_RULES:
            for pattern in rule.patterns:
                if pattern in lower:
                    scores[rule.prefix] = scores.get(rule.prefix, 0) + 1
                    break

    if scores:
        # Return the prefix with the most matching files
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    return "feat"


def _summarise_files(changed_files: list[str]) -> str:
    """Build a short summary of the changed files.

    If only one file changed, name it.  Otherwise group by directory or
    state a count.

    Args:
        changed_files: All changed file paths.

    Returns:
        A human-readable, one-line summary.
    """
    if not changed_files:
        return "no files changed"

    if len(changed_files) == 1:
        return PurePosixPath(changed_files[0]).name

    # Group by top-level directory
    dirs: dict[str, int] = {}
    for f in changed_files:
        parts = PurePosixPath(f).parts
        top = parts[0] if parts else "."
        dirs[top] = dirs.get(top, 0) + 1

    if len(dirs) == 1:
        dirname = next(iter(dirs))
        return f"{len(changed_files)} files in {dirname}"

    return f"{len(changed_files)} files across {len(dirs)} directories"


def _detect_changes_from_diff(diff: str) -> tuple[int, int]:
    """Count insertions and deletions from a unified diff.

    Args:
        diff: The raw diff text.

    Returns:
        Tuple of (insertions, deletions).
    """
    insertions = 0
    deletions = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            insertions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return insertions, deletions


class AutoCommitter:
    """Rule-based auto-commit message generation and commit execution.

    Does **not** call any LLM — uses file patterns, diff statistics, and
    simple heuristics instead.
    """

    # Heuristic thresholds
    MAX_AUTO_COMMIT_FILES: int = 20
    MAX_AUTO_COMMIT_LINES: int = 500

    def __init__(self, repo: GitRepo) -> None:
        """Initialise the auto-committer.

        Args:
            repo: The :class:`GitRepo` instance to work with.
        """
        self._repo = repo

    # ------------------------------------------------------------------
    # Message generation
    # ------------------------------------------------------------------

    def generate_commit_message(self, diff: str, changed_files: list[str] | None = None) -> str:
        """Generate a conventional-commit message from a diff.

        Args:
            diff: Unified diff text (``git diff`` output).
            changed_files: Optional explicit list of changed file paths.
                If ``None``, file paths are extracted from the diff.

        Returns:
            A commit message like ``feat: update 3 files in src``.
        """
        if changed_files is None:
            changed_files = self._extract_files_from_diff(diff)

        prefix = _detect_prefix(changed_files)
        summary = _summarise_files(changed_files)
        insertions, deletions = _detect_changes_from_diff(diff)

        message = f"{prefix}: {summary}"

        # Add a short stats suffix for non-trivial changes
        stats_parts: list[str] = []
        if insertions:
            stats_parts.append(f"+{insertions}")
        if deletions:
            stats_parts.append(f"-{deletions}")
        if stats_parts:
            message += f" ({', '.join(stats_parts)})"

        return message

    # ------------------------------------------------------------------
    # Commit execution
    # ------------------------------------------------------------------

    def commit(self, message: str, files: list[str] | None = None) -> str:
        """Stage files and create a commit.

        Args:
            message: Commit message.
            files: Specific files to stage. ``None`` stages all changes.

        Returns:
            The short hash of the new commit.

        Raises:
            GitError: If the commit fails.
        """
        self._repo.add(files)
        return self._repo.commit(message)

    # ------------------------------------------------------------------
    # Auto-commit heuristics
    # ------------------------------------------------------------------

    def should_auto_commit(self, changes: GitStatus) -> bool:
        """Decide whether an automatic commit is appropriate.

        Returns ``False`` when:

        * There are no changes at all.
        * The change set is too large (more than *MAX_AUTO_COMMIT_FILES*
          files or *MAX_AUTO_COMMIT_LINES* changed lines).
        * There are only untracked files (user should review first).

        Args:
            changes: Current :class:`GitStatus`.

        Returns:
            ``True`` if an auto-commit should proceed.
        """
        if changes.is_clean:
            return False

        # Don't auto-commit if only untracked files
        if not changes.staged and not changes.unstaged and changes.untracked:
            return False

        # Don't auto-commit large change sets
        if changes.total_changes > self.MAX_AUTO_COMMIT_FILES:
            return False

        # Check diff size
        try:
            diff_text = self._repo.diff()
            staged_diff = self._repo.diff(staged=True)
            combined = diff_text + staged_diff
            insertions, deletions = _detect_changes_from_diff(combined)
            if insertions + deletions > self.MAX_AUTO_COMMIT_LINES:
                return False
        except GitError:
            return False

        return True

    def auto_commit_edit(self, file_path: str, description: str) -> str | None:
        """Auto-commit after a file edit if auto-commit is enabled.

        Stages the given file and creates a commit with a descriptive
        message that includes the current branch name.

        Args:
            file_path: Path to the edited file (relative to repo root).
            description: Short description of the edit.

        Returns:
            The short commit hash if a commit was made, or ``None`` if
            skipped (e.g. auto-commit disabled or no changes).
        """
        status = self._repo.status()
        if status.is_clean:
            return None

        branch = self._repo.current_branch()
        file_name = PurePosixPath(file_path).name
        message = f"prism({branch}): edit {file_name} - {description}"

        try:
            self._repo.add([file_path])
            return self._repo.commit(message)
        except GitError:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_files_from_diff(diff: str) -> list[str]:
        """Extract file paths from a unified diff.

        Args:
            diff: Raw diff text.

        Returns:
            De-duplicated list of file paths.
        """
        files: list[str] = []
        pattern = re.compile(r"^diff --git a/(.*?) b/", re.MULTILINE)
        for match in pattern.finditer(diff):
            path = match.group(1)
            if path not in files:
                files.append(path)
        return files

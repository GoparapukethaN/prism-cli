"""Tests for prism.git.operations — GitRepo, GitStatus, GitLogEntry."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from prism.exceptions import GitError, NotAGitRepoError
from prism.git.operations import GitLogEntry, GitRepo, GitStatus

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# GitRepo initialisation
# ------------------------------------------------------------------


class TestGitRepoInit:
    """Tests for GitRepo construction and validation."""

    def test_init_valid_repo(self, git_repo: Path) -> None:
        repo = GitRepo(git_repo)
        assert repo.root == git_repo.resolve()

    def test_init_non_repo_raises(self, non_git_dir: Path) -> None:
        with pytest.raises(NotAGitRepoError):
            GitRepo(non_git_dir)

    def test_is_git_repo_true(self, git_repo: Path) -> None:
        assert GitRepo.is_git_repo(git_repo) is True

    def test_is_git_repo_false(self, non_git_dir: Path) -> None:
        assert GitRepo.is_git_repo(non_git_dir) is False

    def test_is_git_repo_nonexistent_path(self, tmp_path: Path) -> None:
        assert GitRepo.is_git_repo(tmp_path / "does_not_exist") is False


# ------------------------------------------------------------------
# status()
# ------------------------------------------------------------------


class TestStatus:
    """Tests for GitRepo.status()."""

    def test_clean_repo(self, git_repo: Path) -> None:
        repo = GitRepo(git_repo)
        status = repo.status()
        assert status.is_clean
        assert status.total_changes == 0

    def test_unstaged_modification(self, git_repo: Path) -> None:
        (git_repo / "hello.py").write_text('print("modified")\n')
        repo = GitRepo(git_repo)
        status = repo.status()
        assert "hello.py" in status.unstaged
        assert not status.is_clean

    def test_staged_new_file(self, git_repo: Path) -> None:
        (git_repo / "new.py").write_text("new\n")
        subprocess.run(
            ["git", "add", "new.py"], cwd=str(git_repo), capture_output=True
        )
        repo = GitRepo(git_repo)
        status = repo.status()
        assert "new.py" in status.staged

    def test_untracked_file(self, git_repo: Path) -> None:
        (git_repo / "random.txt").write_text("surprise\n")
        repo = GitRepo(git_repo)
        status = repo.status()
        assert "random.txt" in status.untracked

    def test_mixed_changes(self, git_repo_with_changes: Path) -> None:
        repo = GitRepo(git_repo_with_changes)
        status = repo.status()
        assert "staged.py" in status.staged
        assert "hello.py" in status.unstaged
        assert "untracked.txt" in status.untracked
        assert status.total_changes >= 3

    def test_total_changes_count(self, git_repo_with_changes: Path) -> None:
        repo = GitRepo(git_repo_with_changes)
        status = repo.status()
        expected = len(status.staged) + len(status.unstaged) + len(status.untracked)
        assert status.total_changes == expected


# ------------------------------------------------------------------
# diff()
# ------------------------------------------------------------------


class TestDiff:
    """Tests for GitRepo.diff()."""

    def test_diff_clean_repo(self, git_repo: Path) -> None:
        repo = GitRepo(git_repo)
        assert repo.diff() == ""

    def test_diff_unstaged(self, git_repo: Path) -> None:
        (git_repo / "hello.py").write_text('print("changed")\n')
        repo = GitRepo(git_repo)
        diff = repo.diff()
        assert "hello.py" in diff
        assert "+print" in diff or '-print' in diff

    def test_diff_staged(self, git_repo: Path) -> None:
        (git_repo / "hello.py").write_text('print("staged change")\n')
        subprocess.run(
            ["git", "add", "hello.py"], cwd=str(git_repo), capture_output=True
        )
        repo = GitRepo(git_repo)
        # Unstaged diff should be empty now
        assert repo.diff() == ""
        # Staged diff should show the change
        staged_diff = repo.diff(staged=True)
        assert "hello.py" in staged_diff


# ------------------------------------------------------------------
# log()
# ------------------------------------------------------------------


class TestLog:
    """Tests for GitRepo.log()."""

    def test_log_returns_entries(self, git_repo: Path) -> None:
        repo = GitRepo(git_repo)
        entries = repo.log()
        assert len(entries) >= 1
        assert entries[0].message == "initial commit"

    def test_log_entry_fields(self, git_repo: Path) -> None:
        repo = GitRepo(git_repo)
        entry = repo.log()[0]
        assert isinstance(entry, GitLogEntry)
        assert len(entry.hash) == 40  # full SHA
        assert entry.author == "Prism Test"
        assert entry.date  # non-empty

    def test_log_multiple_commits(self, git_repo_multiple_commits: Path) -> None:
        repo = GitRepo(git_repo_multiple_commits)
        entries = repo.log(n=10)
        assert len(entries) == 3  # initial + 2 extra
        # Most recent first
        assert entries[0].message == "commit number 2"

    def test_log_limit(self, git_repo_multiple_commits: Path) -> None:
        repo = GitRepo(git_repo_multiple_commits)
        entries = repo.log(n=1)
        assert len(entries) == 1


# ------------------------------------------------------------------
# current_branch()
# ------------------------------------------------------------------


class TestCurrentBranch:
    """Tests for GitRepo.current_branch()."""

    def test_main_branch(self, git_repo: Path) -> None:
        repo = GitRepo(git_repo)
        assert repo.current_branch() == "main"

    def test_new_branch(self, git_repo: Path) -> None:
        subprocess.run(
            ["git", "checkout", "-b", "feature/test"],
            cwd=str(git_repo),
            capture_output=True,
        )
        repo = GitRepo(git_repo)
        assert repo.current_branch() == "feature/test"


# ------------------------------------------------------------------
# commit() via GitRepo
# ------------------------------------------------------------------


class TestCommit:
    """Tests for GitRepo.add() and GitRepo.commit()."""

    def test_commit_returns_short_hash(self, git_repo: Path) -> None:
        (git_repo / "new.py").write_text("x = 1\n")
        repo = GitRepo(git_repo)
        repo.add(["new.py"])
        short_hash = repo.commit("add new.py")
        assert len(short_hash) >= 7
        # Verify commit happened
        entries = repo.log(n=1)
        assert entries[0].message == "add new.py"

    def test_commit_no_changes_raises(self, git_repo: Path) -> None:
        repo = GitRepo(git_repo)
        with pytest.raises(GitError):
            repo.commit("empty commit")


# ------------------------------------------------------------------
# GitStatus dataclass
# ------------------------------------------------------------------


class TestGitStatusDataclass:
    """Tests for the GitStatus dataclass behaviour."""

    def test_empty_status_is_clean(self) -> None:
        status = GitStatus()
        assert status.is_clean
        assert status.total_changes == 0

    def test_staged_not_clean(self) -> None:
        status = GitStatus(staged=["a.py"])
        assert not status.is_clean
        assert status.total_changes == 1

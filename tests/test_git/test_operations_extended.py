"""Tests for extended GitRepo operations — undo, dirty, checkpoint, gitignore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.exceptions import GitError
from prism.git.auto_commit import AutoCommitter
from prism.git.operations import GitRepo

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# undo_last_commit()
# ------------------------------------------------------------------


class TestUndoLastCommit:
    """Tests for GitRepo.undo_last_commit()."""

    def test_undo_last_commit(self, git_repo: Path) -> None:
        """Undoing a commit returns the undone hash and leaves changes staged."""
        (git_repo / "new.py").write_text("x = 1\n")
        repo = GitRepo(git_repo)
        repo.add(["new.py"])
        repo.commit("add new.py")

        head_before = repo.get_current_commit()
        undone_hash = repo.undo_last_commit()

        assert undone_hash == head_before
        # The file should still exist and be staged
        status = repo.status()
        assert "new.py" in status.staged

    def test_undo_on_initial_commit(self, git_repo: Path) -> None:
        """Undoing the only commit should raise GitError (no HEAD~1)."""
        repo = GitRepo(git_repo)
        # The git_repo fixture has exactly one commit ("initial commit").
        # Undoing it means HEAD~1 doesn't exist.
        with pytest.raises(GitError):
            repo.undo_last_commit()


# ------------------------------------------------------------------
# get_dirty_files()
# ------------------------------------------------------------------


class TestGetDirtyFiles:
    """Tests for GitRepo.get_dirty_files()."""

    def test_get_dirty_files(self, git_repo: Path) -> None:
        """Modified and untracked files are both returned."""
        (git_repo / "hello.py").write_text('print("changed")\n')
        (git_repo / "untracked.txt").write_text("data\n")
        repo = GitRepo(git_repo)
        dirty = repo.get_dirty_files()
        assert "hello.py" in dirty
        assert "untracked.txt" in dirty

    def test_get_dirty_files_clean_repo(self, git_repo: Path) -> None:
        """A clean repo returns an empty list."""
        repo = GitRepo(git_repo)
        assert repo.get_dirty_files() == []


# ------------------------------------------------------------------
# is_dirty()
# ------------------------------------------------------------------


class TestIsDirty:
    """Tests for GitRepo.is_dirty()."""

    def test_is_dirty_clean_repo(self, git_repo: Path) -> None:
        """A fresh repo with no changes is not dirty."""
        repo = GitRepo(git_repo)
        assert repo.is_dirty() is False

    def test_is_dirty_with_changes(self, git_repo: Path) -> None:
        """Modifying a tracked file makes the repo dirty."""
        (git_repo / "hello.py").write_text('print("dirty")\n')
        repo = GitRepo(git_repo)
        assert repo.is_dirty() is True


# ------------------------------------------------------------------
# create_checkpoint() / reset_to_checkpoint()
# ------------------------------------------------------------------


class TestCheckpointReset:
    """Tests for checkpoint creation and rollback."""

    def test_create_checkpoint(self, git_repo: Path) -> None:
        """Creating a checkpoint stages and commits all changes."""
        (git_repo / "hello.py").write_text('print("checkpoint me")\n')
        repo = GitRepo(git_repo)
        checkpoint = repo.create_checkpoint()

        assert len(checkpoint) == 40  # full SHA
        assert not repo.is_dirty()

        # Verify the commit message
        entries = repo.log(n=1)
        assert entries[0].message == "prism: checkpoint"

    def test_reset_to_checkpoint(self, git_repo: Path) -> None:
        """Resetting to a checkpoint discards subsequent changes."""
        repo = GitRepo(git_repo)
        original_hash = repo.get_current_commit()

        # Make a change and checkpoint it
        (git_repo / "hello.py").write_text('print("checkpoint")\n')
        checkpoint = repo.create_checkpoint()
        assert checkpoint != original_hash

        # Make more changes after the checkpoint
        (git_repo / "hello.py").write_text('print("after checkpoint")\n')
        repo.add(["hello.py"])
        repo.commit("post-checkpoint work")

        # Reset to the checkpoint
        result = repo.reset_to_checkpoint(checkpoint)
        assert result is True
        assert repo.get_current_commit() == checkpoint

        # The file should contain the checkpoint content
        content = (git_repo / "hello.py").read_text()
        assert "checkpoint" in content
        assert "after checkpoint" not in content


# ------------------------------------------------------------------
# get_current_commit()
# ------------------------------------------------------------------


class TestGetCurrentCommit:
    """Tests for GitRepo.get_current_commit()."""

    def test_get_current_commit(self, git_repo: Path) -> None:
        """Returns a 40-character SHA hash."""
        repo = GitRepo(git_repo)
        commit_hash = repo.get_current_commit()
        assert len(commit_hash) == 40
        assert all(c in "0123456789abcdef" for c in commit_hash)

    def test_get_current_commit_changes_after_commit(self, git_repo: Path) -> None:
        """The hash changes after a new commit is made."""
        repo = GitRepo(git_repo)
        hash_before = repo.get_current_commit()

        (git_repo / "new.py").write_text("y = 2\n")
        repo.add(["new.py"])
        repo.commit("add new.py")

        hash_after = repo.get_current_commit()
        assert hash_before != hash_after


# ------------------------------------------------------------------
# ensure_gitignore_has_prism()
# ------------------------------------------------------------------


class TestEnsureGitignoreHasPrism:
    """Tests for GitRepo.ensure_gitignore_has_prism()."""

    def test_ensure_gitignore_prism_added(self, git_repo: Path) -> None:
        """Creates .gitignore with .prism/ when no gitignore exists."""
        gitignore = git_repo / ".gitignore"
        if gitignore.exists():
            gitignore.unlink()

        repo = GitRepo(git_repo)
        result = repo.ensure_gitignore_has_prism()

        assert result is True
        assert gitignore.is_file()
        assert ".prism/" in gitignore.read_text()

    def test_ensure_gitignore_prism_already_present(self, git_repo: Path) -> None:
        """Returns False when .prism/ is already in .gitignore."""
        gitignore = git_repo / ".gitignore"
        gitignore.write_text(".prism/\n")

        repo = GitRepo(git_repo)
        result = repo.ensure_gitignore_has_prism()

        assert result is False
        # Should not duplicate the entry
        lines = gitignore.read_text().splitlines()
        assert lines.count(".prism/") == 1

    def test_ensure_gitignore_appends_to_existing(self, git_repo: Path) -> None:
        """Appends .prism/ to an existing .gitignore without .prism/."""
        gitignore = git_repo / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n")

        repo = GitRepo(git_repo)
        result = repo.ensure_gitignore_has_prism()

        assert result is True
        content = gitignore.read_text()
        assert "*.pyc" in content
        assert ".prism/" in content


# ------------------------------------------------------------------
# auto_commit_edit() — integration
# ------------------------------------------------------------------


class TestAutoCommitEdit:
    """Tests for AutoCommitter.auto_commit_edit()."""

    def test_auto_commit_edit(self, git_repo: Path) -> None:
        """auto_commit_edit stages and commits the file with a descriptive message."""
        (git_repo / "hello.py").write_text('print("edited")\n')
        repo = GitRepo(git_repo)
        committer = AutoCommitter(repo)
        result = committer.auto_commit_edit("hello.py", "update greeting")

        assert result is not None
        assert len(result) >= 7  # short hash

        # Verify commit message format includes branch and description
        entries = repo.log(n=1)
        assert "prism(main)" in entries[0].message
        assert "hello.py" in entries[0].message
        assert "update greeting" in entries[0].message

    def test_auto_commit_edit_clean_repo_returns_none(self, git_repo: Path) -> None:
        """auto_commit_edit returns None when there are no changes."""
        repo = GitRepo(git_repo)
        committer = AutoCommitter(repo)
        result = committer.auto_commit_edit("hello.py", "no changes")
        assert result is None

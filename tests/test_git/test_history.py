"""Tests for prism.git.history -- RollbackManager, ChangeRecord, SessionTimeline."""

from __future__ import annotations

import contextlib
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from prism.git.history import ChangeRecord, RollbackManager, SessionTimeline

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_git(cwd: Path, *args: str) -> str:
    """Run a git command inside *cwd* and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout


def _make_commit(repo: Path, filename: str, content: str, message: str) -> str:
    """Create a file, stage it, commit, and return the full commit hash."""
    (repo / filename).write_text(content)
    _run_git(repo, "add", filename)
    _run_git(repo, "commit", "-m", message)
    return _run_git(repo, "rev-parse", "HEAD").strip()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    """Create a fresh git repository with one initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "test@prism.dev")
    _run_git(repo, "config", "user.name", "Prism Test")
    # Ensure branch is "main"
    with contextlib.suppress(RuntimeError):
        _run_git(repo, "checkout", "-b", "main")
    (repo / "init.py").write_text('print("init")\n')
    _run_git(repo, "add", "init.py")
    _run_git(repo, "commit", "-m", "initial commit")
    return repo


@pytest.fixture
def manager(repo_dir: Path) -> RollbackManager:
    """Create a RollbackManager pointed at the test repo."""
    return RollbackManager(repo_dir)


# ===================================================================
# ChangeRecord dataclass
# ===================================================================


class TestChangeRecord:
    """Tests for the ChangeRecord frozen dataclass."""

    def test_fields(self) -> None:
        rec = ChangeRecord(
            index=1,
            commit_hash="a" * 40,
            short_hash="a" * 8,
            message="test commit",
            timestamp="2026-01-01T00:00:00+00:00",
            files_changed=["foo.py"],
            insertions=5,
            deletions=2,
        )
        assert rec.index == 1
        assert rec.commit_hash == "a" * 40
        assert rec.short_hash == "a" * 8
        assert rec.message == "test commit"
        assert rec.files_changed == ["foo.py"]
        assert rec.insertions == 5
        assert rec.deletions == 2

    def test_frozen(self) -> None:
        rec = ChangeRecord(
            index=1,
            commit_hash="a" * 40,
            short_hash="a" * 8,
            message="msg",
            timestamp="",
            files_changed=[],
            insertions=0,
            deletions=0,
        )
        with pytest.raises(AttributeError):
            rec.index = 99  # type: ignore[misc]

    def test_equality(self) -> None:
        kwargs = {
            "index": 1,
            "commit_hash": "b" * 40,
            "short_hash": "b" * 8,
            "message": "same",
            "timestamp": "ts",
            "files_changed": [],
            "insertions": 0,
            "deletions": 0,
        }
        assert ChangeRecord(**kwargs) == ChangeRecord(**kwargs)  # type: ignore[arg-type]


# ===================================================================
# SessionTimeline dataclass
# ===================================================================


class TestSessionTimeline:
    """Tests for the SessionTimeline dataclass."""

    def test_defaults(self) -> None:
        tl = SessionTimeline(session_id="abc")
        assert tl.session_id == "abc"
        assert tl.changes == []
        assert tl.start_commit == ""

    def test_with_changes(self) -> None:
        rec = ChangeRecord(
            index=1,
            commit_hash="c" * 40,
            short_hash="c" * 8,
            message="m",
            timestamp="t",
            files_changed=[],
            insertions=0,
            deletions=0,
        )
        tl = SessionTimeline(session_id="x", changes=[rec], start_commit="abc123")
        assert len(tl.changes) == 1
        assert tl.start_commit == "abc123"


# ===================================================================
# RollbackManager -- start_session
# ===================================================================


class TestStartSession:
    """Tests for RollbackManager.start_session()."""

    def test_returns_commit_hash(self, manager: RollbackManager, repo_dir: Path) -> None:
        result = manager.start_session()
        expected = _run_git(repo_dir, "rev-parse", "HEAD").strip()
        assert result == expected

    def test_records_start_commit(self, manager: RollbackManager) -> None:
        result = manager.start_session()
        assert manager.session_start_commit == result

    def test_clears_previous_changes(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "a.py", "x = 1\n", "add a.py")
        manager.record_change(commit)
        assert manager.change_count == 1

        # Second start_session should reset
        manager.start_session()
        assert manager.change_count == 0


# ===================================================================
# RollbackManager -- record_change
# ===================================================================


class TestRecordChange:
    """Tests for RollbackManager.record_change()."""

    def test_records_single_change(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "file1.py", "x = 1\n", "add file1")
        rec = manager.record_change(commit)
        assert rec.index == 1
        assert rec.commit_hash == commit
        assert rec.short_hash == commit[:8]
        assert rec.message == "add file1"

    def test_records_multiple_changes(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        c1 = _make_commit(repo_dir, "f1.py", "a\n", "first")
        c2 = _make_commit(repo_dir, "f2.py", "b\n", "second")
        r1 = manager.record_change(c1)
        r2 = manager.record_change(c2)
        assert r1.index == 1
        assert r2.index == 2
        assert manager.change_count == 2

    def test_custom_message_override(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "file.py", "y = 2\n", "original msg")
        rec = manager.record_change(commit, message="override msg")
        assert rec.message == "override msg"

    def test_empty_hash_raises(self, manager: RollbackManager) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            manager.record_change("")

    def test_whitespace_hash_raises(self, manager: RollbackManager) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            manager.record_change("   ")

    def test_files_changed_populated(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "new_module.py", "z = 3\n", "add module")
        rec = manager.record_change(commit)
        assert "new_module.py" in rec.files_changed

    def test_timestamp_populated(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "ts.py", "t = 0\n", "ts test")
        rec = manager.record_change(commit)
        assert rec.timestamp  # non-empty


# ===================================================================
# RollbackManager -- get_timeline
# ===================================================================


class TestGetTimeline:
    """Tests for RollbackManager.get_timeline()."""

    def test_empty_timeline(self, manager: RollbackManager) -> None:
        manager.start_session()
        tl = manager.get_timeline()
        assert isinstance(tl, SessionTimeline)
        assert tl.changes == []
        assert tl.start_commit == manager.session_start_commit

    def test_timeline_with_changes(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        c1 = _make_commit(repo_dir, "a.py", "1\n", "first")
        c2 = _make_commit(repo_dir, "b.py", "2\n", "second")
        manager.record_change(c1)
        manager.record_change(c2)

        tl = manager.get_timeline()
        assert len(tl.changes) == 2
        assert tl.changes[0].index == 1
        assert tl.changes[1].index == 2

    def test_timeline_is_copy(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        """Modifying returned timeline should not affect internal state."""
        manager.start_session()
        commit = _make_commit(repo_dir, "c.py", "3\n", "third")
        manager.record_change(commit)

        tl = manager.get_timeline()
        tl.changes.clear()
        assert manager.change_count == 1  # internal unchanged


# ===================================================================
# RollbackManager -- get_change
# ===================================================================


class TestGetChange:
    """Tests for RollbackManager.get_change()."""

    def test_valid_index(self, manager: RollbackManager, repo_dir: Path) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "x.py", "x\n", "test get_change")
        manager.record_change(commit)
        rec = manager.get_change(1)
        assert rec.commit_hash == commit

    def test_index_zero_raises(self, manager: RollbackManager) -> None:
        manager.start_session()
        with pytest.raises(ValueError, match="Invalid change index"):
            manager.get_change(0)

    def test_index_out_of_range_raises(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "q.py", "q\n", "q")
        manager.record_change(commit)
        with pytest.raises(ValueError, match="Invalid change index"):
            manager.get_change(5)


# ===================================================================
# RollbackManager -- get_diff (mocked)
# ===================================================================


class TestGetDiff:
    """Tests for RollbackManager.get_diff()."""

    def test_returns_diff_string(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "diff_test.py", "val = 42\n", "diff me")
        manager.record_change(commit)
        diff = manager.get_diff(1)
        assert isinstance(diff, str)
        # Should contain something about the file
        assert "diff_test.py" in diff or "val" in diff

    def test_invalid_index_raises(self, manager: RollbackManager) -> None:
        manager.start_session()
        with pytest.raises(ValueError, match="Invalid change index"):
            manager.get_diff(1)

    def test_negative_index_raises(self, manager: RollbackManager) -> None:
        manager.start_session()
        with pytest.raises(ValueError, match="Invalid change index"):
            manager.get_diff(-1)


# ===================================================================
# RollbackManager -- undo
# ===================================================================


class TestUndo:
    """Tests for RollbackManager.undo()."""

    def test_undo_single(self, manager: RollbackManager, repo_dir: Path) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "undo_me.py", "content\n", "to undo")
        manager.record_change(commit)

        reverted = manager.undo(1)
        assert reverted == [commit]
        assert manager.change_count == 0

    def test_undo_multiple(self, manager: RollbackManager, repo_dir: Path) -> None:
        manager.start_session()
        c1 = _make_commit(repo_dir, "u1.py", "a\n", "first undo")
        c2 = _make_commit(repo_dir, "u2.py", "b\n", "second undo")
        manager.record_change(c1)
        manager.record_change(c2)

        reverted = manager.undo(2)
        # Most recent first
        assert reverted == [c2, c1]
        assert manager.change_count == 0

    def test_undo_zero_raises(self, manager: RollbackManager) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            manager.undo(0)

    def test_undo_negative_raises(self, manager: RollbackManager) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            manager.undo(-3)

    def test_undo_more_than_available_raises(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "only.py", "x\n", "only one")
        manager.record_change(commit)

        with pytest.raises(ValueError, match="Only 1 changes to undo"):
            manager.undo(5)

    def test_undo_no_changes_raises(self, manager: RollbackManager) -> None:
        manager.start_session()
        with pytest.raises(ValueError, match="Only 0 changes to undo"):
            manager.undo(1)

    def test_undo_preserves_remaining(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        """Undoing 1 of 3 changes should leave 2."""
        manager.start_session()
        c1 = _make_commit(repo_dir, "k1.py", "1\n", "keep1")
        c2 = _make_commit(repo_dir, "k2.py", "2\n", "keep2")
        c3 = _make_commit(repo_dir, "k3.py", "3\n", "remove")
        manager.record_change(c1)
        manager.record_change(c2)
        manager.record_change(c3)

        reverted = manager.undo(1)
        assert reverted == [c3]
        assert manager.change_count == 2


# ===================================================================
# RollbackManager -- restore
# ===================================================================


class TestRestore:
    """Tests for RollbackManager.restore()."""

    def test_restore_creates_new_commit(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        start = manager.session_start_commit
        _make_commit(repo_dir, "temp.py", "temp\n", "will revert")

        new_hash = manager.restore(start)
        assert new_hash != start
        # New commit message should mention restore
        msg = _run_git(repo_dir, "log", "-1", "--format=%s").strip()
        assert "restore" in msg.lower()

    def test_restore_invalid_hash_raises(
        self, manager: RollbackManager
    ) -> None:
        with pytest.raises(RuntimeError, match="Git error"):
            manager.restore("0000000000000000000000000000000000000000")

    def test_restore_empty_hash_raises(self, manager: RollbackManager) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            manager.restore("")


# ===================================================================
# RollbackManager -- get_restore_preview
# ===================================================================


class TestGetRestorePreview:
    """Tests for RollbackManager.get_restore_preview()."""

    def test_preview_returns_string(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        start = manager.session_start_commit
        _make_commit(repo_dir, "preview.py", "p\n", "added preview")

        preview = manager.get_restore_preview(start)
        assert isinstance(preview, str)

    def test_preview_invalid_hash_raises(
        self, manager: RollbackManager
    ) -> None:
        with pytest.raises(RuntimeError, match="Git error"):
            manager.get_restore_preview("deadbeef" * 5)

    def test_preview_empty_hash_raises(self, manager: RollbackManager) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            manager.get_restore_preview("")


# ===================================================================
# RollbackManager -- _git (mocked)
# ===================================================================


class TestGitHelper:
    """Tests for the internal _git helper via mocking."""

    def test_timeout_raises(self, manager: RollbackManager) -> None:
        with patch("prism.git.history.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=30)
            with pytest.raises(subprocess.TimeoutExpired):
                manager._git("status")

    def test_nonzero_exit_raises_runtime(self, manager: RollbackManager) -> None:
        with patch("prism.git.history.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="fatal: bad object", stdout=""
            )
            with pytest.raises(RuntimeError, match="Git error"):
                manager._git("log")

    def test_revert_nonzero_does_not_raise(self, manager: RollbackManager) -> None:
        """The revert subcommand is allowed to have non-zero exit."""
        with patch("prism.git.history.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="conflict", stdout="reverted"
            )
            result = manager._git("revert", "--no-edit", "abc123")
            assert result == "reverted"


# ===================================================================
# RollbackManager -- properties
# ===================================================================


class TestProperties:
    """Tests for RollbackManager properties."""

    def test_session_start_commit_default(self, manager: RollbackManager) -> None:
        assert manager.session_start_commit == ""

    def test_change_count_default(self, manager: RollbackManager) -> None:
        assert manager.change_count == 0

    def test_change_count_after_recording(
        self, manager: RollbackManager, repo_dir: Path
    ) -> None:
        manager.start_session()
        commit = _make_commit(repo_dir, "cc.py", "cc\n", "count test")
        manager.record_change(commit)
        assert manager.change_count == 1

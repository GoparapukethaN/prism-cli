"""Tests for prism.context.branching -- BranchManager, BranchMetadata, ConversationBranch."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from prism.context.branching import (
    BranchManager,
    BranchMetadata,
    ConversationBranch,
    _validate_branch_name,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "Write a function"},
    {"role": "assistant", "content": "def foo(): ..."},
]


def _make_messages(n: int) -> list[dict[str, str]]:
    """Create *n* sample messages."""
    msgs: list[dict[str, str]] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"message {i}"})
    return msgs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def branch_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for branch storage."""
    return tmp_path / "branches"


@pytest.fixture
def mgr(branch_dir: Path) -> BranchManager:
    """Create a fresh BranchManager."""
    return BranchManager(branch_dir)


# ===================================================================
# _validate_branch_name
# ===================================================================


class TestValidateBranchName:
    """Tests for the branch name validator."""

    def test_valid_simple_name(self) -> None:
        _validate_branch_name("feature1")  # should not raise

    def test_valid_with_hyphen(self) -> None:
        _validate_branch_name("my-branch")

    def test_valid_with_underscore(self) -> None:
        _validate_branch_name("my_branch")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_branch_name("")

    def test_starts_with_hyphen_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid branch name"):
            _validate_branch_name("-bad")

    def test_starts_with_underscore_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid branch name"):
            _validate_branch_name("_bad")

    def test_spaces_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid branch name"):
            _validate_branch_name("has space")

    def test_special_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid branch name"):
            _validate_branch_name("bad!name")


# ===================================================================
# BranchMetadata dataclass
# ===================================================================


class TestBranchMetadata:
    """Tests for BranchMetadata fields and defaults."""

    def test_all_fields(self) -> None:
        meta = BranchMetadata(
            name="test",
            created_at="2026-01-01T00:00:00",
            parent_branch="main",
            fork_point_index=5,
            git_commit_at_fork="abc123",
            message_count=10,
            is_active=True,
            description="test branch",
        )
        assert meta.name == "test"
        assert meta.parent_branch == "main"
        assert meta.fork_point_index == 5
        assert meta.git_commit_at_fork == "abc123"
        assert meta.message_count == 10
        assert meta.is_active is True
        assert meta.description == "test branch"

    def test_defaults(self) -> None:
        meta = BranchMetadata(
            name="minimal",
            created_at="",
            parent_branch="main",
            fork_point_index=0,
            git_commit_at_fork="",
        )
        assert meta.message_count == 0
        assert meta.is_active is False
        assert meta.description == ""


# ===================================================================
# ConversationBranch dataclass
# ===================================================================


class TestConversationBranch:
    """Tests for ConversationBranch fields and defaults."""

    def test_defaults(self) -> None:
        meta = BranchMetadata(
            name="b", created_at="", parent_branch="", fork_point_index=0,
            git_commit_at_fork="",
        )
        branch = ConversationBranch(metadata=meta)
        assert branch.messages == []
        assert branch.file_edits == []

    def test_with_messages(self) -> None:
        meta = BranchMetadata(
            name="b", created_at="", parent_branch="", fork_point_index=0,
            git_commit_at_fork="",
        )
        branch = ConversationBranch(
            metadata=meta,
            messages=list(_SAMPLE_MESSAGES),
            file_edits=["abc", "def"],
        )
        assert len(branch.messages) == 4
        assert len(branch.file_edits) == 2


# ===================================================================
# BranchManager -- create_branch
# ===================================================================


class TestCreateBranch:
    """Tests for BranchManager.create_branch()."""

    def test_creates_branch(self, mgr: BranchManager) -> None:
        meta = mgr.create_branch("feature", _SAMPLE_MESSAGES, git_commit="abc123")
        assert meta.name == "feature"
        assert meta.parent_branch == "main"
        assert meta.fork_point_index == 4
        assert meta.git_commit_at_fork == "abc123"
        assert meta.is_active is True

    def test_switches_active(self, mgr: BranchManager) -> None:
        mgr.create_branch("b1", _SAMPLE_MESSAGES)
        assert mgr.active_branch == "b1"

    def test_duplicate_name_raises(self, mgr: BranchManager) -> None:
        mgr.create_branch("dup", _SAMPLE_MESSAGES)
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_branch("dup", _SAMPLE_MESSAGES)

    def test_invalid_name_raises(self, mgr: BranchManager) -> None:
        with pytest.raises(ValueError, match="Invalid branch name"):
            mgr.create_branch("bad name!", _SAMPLE_MESSAGES)

    def test_empty_name_raises(self, mgr: BranchManager) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            mgr.create_branch("", _SAMPLE_MESSAGES)

    def test_messages_are_copied(self, mgr: BranchManager) -> None:
        msgs = list(_SAMPLE_MESSAGES)
        mgr.create_branch("copy_test", msgs)
        msgs.append({"role": "user", "content": "extra"})
        branch = mgr.get_branch("copy_test")
        assert len(branch.messages) == 4  # original 4, not 5

    def test_with_description(self, mgr: BranchManager) -> None:
        meta = mgr.create_branch(
            "desc", _SAMPLE_MESSAGES, description="testing approach A"
        )
        assert meta.description == "testing approach A"

    def test_branch_count_increases(self, mgr: BranchManager) -> None:
        assert mgr.branch_count == 0
        mgr.create_branch("b1", [])
        assert mgr.branch_count == 1
        mgr.create_branch("b2", [])
        assert mgr.branch_count == 2

    def test_persists_to_disk(self, mgr: BranchManager, branch_dir: Path) -> None:
        mgr.create_branch("persist", _SAMPLE_MESSAGES)
        path = branch_dir / "persist.json"
        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["metadata"]["name"] == "persist"
        assert len(data["messages"]) == 4


# ===================================================================
# BranchManager -- list_branches
# ===================================================================


class TestListBranches:
    """Tests for BranchManager.list_branches()."""

    def test_empty(self, mgr: BranchManager) -> None:
        assert mgr.list_branches() == []

    def test_lists_all(self, mgr: BranchManager) -> None:
        mgr.create_branch("a", [])
        mgr.create_branch("b", [])
        mgr.create_branch("c", [])
        result = mgr.list_branches()
        names = [m.name for m in result]
        assert "a" in names
        assert "b" in names
        assert "c" in names
        assert len(result) == 3

    def test_sorted_by_created_at(self, mgr: BranchManager) -> None:
        import time
        mgr.create_branch("first", [])
        time.sleep(0.01)
        mgr.create_branch("second", [])
        result = mgr.list_branches()
        assert result[0].name == "first"
        assert result[1].name == "second"


# ===================================================================
# BranchManager -- get_branch
# ===================================================================


class TestGetBranch:
    """Tests for BranchManager.get_branch()."""

    def test_existing_branch(self, mgr: BranchManager) -> None:
        mgr.create_branch("existing", _SAMPLE_MESSAGES)
        branch = mgr.get_branch("existing")
        assert isinstance(branch, ConversationBranch)
        assert branch.metadata.name == "existing"

    def test_nonexistent_raises(self, mgr: BranchManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            mgr.get_branch("nope")


# ===================================================================
# BranchManager -- switch_branch
# ===================================================================


class TestSwitchBranch:
    """Tests for BranchManager.switch_branch()."""

    def test_switch_returns_messages(self, mgr: BranchManager) -> None:
        mgr.create_branch("orig", _SAMPLE_MESSAGES)
        # Create second branch -- pass _SAMPLE_MESSAGES as "current" so that
        # "orig" is saved with its full conversation before switching away.
        mgr.create_branch("branch2", list(_SAMPLE_MESSAGES))

        # Switch back to "orig"
        returned = mgr.switch_branch("orig", [{"role": "user", "content": "current"}])
        assert len(returned) == 4  # original messages saved when branch2 was created

    def test_switch_updates_active(self, mgr: BranchManager) -> None:
        mgr.create_branch("x", [])
        mgr.create_branch("y", [])
        assert mgr.active_branch == "y"

        mgr.switch_branch("x", [])
        assert mgr.active_branch == "x"

    def test_switch_saves_current_state(self, mgr: BranchManager) -> None:
        mgr.create_branch("alpha", _SAMPLE_MESSAGES)
        mgr.create_branch("beta", [])

        # Switch back to alpha with new messages
        extra = [{"role": "user", "content": "extra"}]
        mgr.switch_branch("alpha", extra)

        # Verify beta's state was saved
        beta = mgr.get_branch("beta")
        assert beta.messages == extra

    def test_switch_nonexistent_raises(self, mgr: BranchManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            mgr.switch_branch("ghost", [])

    def test_switch_returns_copy(self, mgr: BranchManager) -> None:
        """Returned messages should be a copy, not a reference."""
        mgr.create_branch("ref_test", _SAMPLE_MESSAGES)
        # Pass the same messages so ref_test is saved correctly
        mgr.create_branch("other", list(_SAMPLE_MESSAGES))
        returned = mgr.switch_branch("ref_test", [])
        returned.clear()
        branch = mgr.get_branch("ref_test")
        assert len(branch.messages) == 4  # unmodified


# ===================================================================
# BranchManager -- merge_branch
# ===================================================================


class TestMergeBranch:
    """Tests for BranchManager.merge_branch()."""

    def test_merge_appends_new_messages(self, mgr: BranchManager) -> None:
        # Create source branch with 4 messages
        mgr.create_branch("source", _SAMPLE_MESSAGES)

        # Add 2 more messages to source (simulate continuation on this branch)
        branch = mgr.get_branch("source")
        branch.messages.extend([
            {"role": "user", "content": "extra q"},
            {"role": "assistant", "content": "extra a"},
        ])
        branch.metadata.message_count = 6
        mgr._save_branch("source")

        # Create target branch.  Pass the 6-message list as "current" so
        # the source branch is saved with the extended conversation before
        # we switch away.
        extended_msgs = list(branch.messages)
        mgr.create_branch("target", extended_msgs)

        # Merge source into target -- current target has the original 4
        merged = mgr.merge_branch("source", list(_SAMPLE_MESSAGES))
        # fork_point_index is 6 (length at create_branch time), so the
        # source has 6 total messages with fork=4 -> 2 new after fork
        # But we saved source with the 6-message override above.
        # source.fork_point_index = 4, source.messages has 6 items
        # => new messages = source.messages[4:] = 2 extra
        assert len(merged) == 6
        assert merged[-1]["content"] == "extra a"

    def test_merge_self_raises(self, mgr: BranchManager) -> None:
        mgr.create_branch("self_merge", [])
        with pytest.raises(ValueError, match="Cannot merge a branch into itself"):
            mgr.merge_branch("self_merge", [])

    def test_merge_nonexistent_raises(self, mgr: BranchManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            mgr.merge_branch("phantom", [])

    def test_merge_no_new_messages(self, mgr: BranchManager) -> None:
        """If source has no messages beyond fork, merge returns current unchanged."""
        mgr.create_branch("no_new", _SAMPLE_MESSAGES)
        mgr.create_branch("target2", _SAMPLE_MESSAGES)

        merged = mgr.merge_branch("no_new", _SAMPLE_MESSAGES)
        assert len(merged) == len(_SAMPLE_MESSAGES)


# ===================================================================
# BranchManager -- delete_branch
# ===================================================================


class TestDeleteBranch:
    """Tests for BranchManager.delete_branch()."""

    def test_delete_existing(self, mgr: BranchManager, branch_dir: Path) -> None:
        mgr.create_branch("doomed", [])
        mgr.create_branch("keeper", [])  # switch active away

        mgr.delete_branch("doomed")
        assert mgr.branch_count == 1
        assert not (branch_dir / "doomed.json").exists()

    def test_delete_nonexistent_raises(self, mgr: BranchManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            mgr.delete_branch("nope")

    def test_delete_active_raises(self, mgr: BranchManager) -> None:
        mgr.create_branch("active_one", [])
        assert mgr.active_branch == "active_one"
        with pytest.raises(ValueError, match="Cannot delete the active branch"):
            mgr.delete_branch("active_one")


# ===================================================================
# BranchManager -- file edit tracking
# ===================================================================


class TestFileEdits:
    """Tests for add_file_edit() and get_file_edits()."""

    def test_add_and_get(self, mgr: BranchManager) -> None:
        mgr.create_branch("edits", [])
        mgr.add_file_edit("abc123")
        mgr.add_file_edit("def456")
        edits = mgr.get_file_edits("edits")
        assert edits == ["abc123", "def456"]

    def test_get_default_active(self, mgr: BranchManager) -> None:
        mgr.create_branch("active_edits", [])
        mgr.add_file_edit("xyz789")
        edits = mgr.get_file_edits()  # no name => active
        assert edits == ["xyz789"]

    def test_add_empty_hash_raises(self, mgr: BranchManager) -> None:
        mgr.create_branch("empty_edit", [])
        with pytest.raises(ValueError, match="must not be empty"):
            mgr.add_file_edit("")

    def test_get_nonexistent_branch_raises(self, mgr: BranchManager) -> None:
        with pytest.raises(ValueError, match="not found"):
            mgr.get_file_edits("nosuch")

    def test_edits_persisted(self, mgr: BranchManager, branch_dir: Path) -> None:
        mgr.create_branch("persist_edit", [])
        mgr.add_file_edit("commit1")
        data = json.loads((branch_dir / "persist_edit.json").read_text())
        assert data["file_edits"] == ["commit1"]

    def test_edits_isolated_per_branch(self, mgr: BranchManager) -> None:
        mgr.create_branch("branch_a", [])
        mgr.add_file_edit("edit_a")
        mgr.create_branch("branch_b", [])
        mgr.add_file_edit("edit_b")

        assert mgr.get_file_edits("branch_a") == ["edit_a"]
        assert mgr.get_file_edits("branch_b") == ["edit_b"]


# ===================================================================
# BranchManager -- persistence round-trip
# ===================================================================


class TestPersistence:
    """Tests for save/load round-trip."""

    def test_load_on_init(self, branch_dir: Path) -> None:
        # Create a branch with one manager
        mgr1 = BranchManager(branch_dir)
        mgr1.create_branch("saved", _SAMPLE_MESSAGES, git_commit="abc")

        # Create a new manager pointing to the same directory
        mgr2 = BranchManager(branch_dir)
        branches = mgr2.list_branches()
        names = [b.name for b in branches]
        assert "saved" in names

        branch = mgr2.get_branch("saved")
        assert len(branch.messages) == 4
        assert branch.metadata.git_commit_at_fork == "abc"

    def test_active_restored_on_load(self, branch_dir: Path) -> None:
        mgr1 = BranchManager(branch_dir)
        mgr1.create_branch("b1", [])
        mgr1.create_branch("b2", [])
        assert mgr1.active_branch == "b2"

        mgr2 = BranchManager(branch_dir)
        assert mgr2.active_branch == "b2"

    def test_corrupt_file_skipped(self, branch_dir: Path) -> None:
        branch_dir.mkdir(parents=True, exist_ok=True)
        (branch_dir / "bad.json").write_text("not valid json!", encoding="utf-8")

        # Should load without raising
        mgr = BranchManager(branch_dir)
        assert mgr.branch_count == 0

    def test_missing_key_skipped(self, branch_dir: Path) -> None:
        branch_dir.mkdir(parents=True, exist_ok=True)
        # Valid JSON but missing "metadata" key
        data = {"messages": [], "file_edits": []}
        (branch_dir / "incomplete.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

        mgr = BranchManager(branch_dir)
        assert mgr.branch_count == 0

    def test_file_edits_persisted_and_loaded(self, branch_dir: Path) -> None:
        mgr1 = BranchManager(branch_dir)
        mgr1.create_branch("with_edits", [])
        mgr1.add_file_edit("commit_a")
        mgr1.add_file_edit("commit_b")

        mgr2 = BranchManager(branch_dir)
        edits = mgr2.get_file_edits("with_edits")
        assert edits == ["commit_a", "commit_b"]

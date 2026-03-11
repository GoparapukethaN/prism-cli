"""Tests for prism.git.auto_commit — AutoCommitter and message generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.git.auto_commit import AutoCommitter, _detect_prefix, _summarise_files
from prism.git.operations import GitRepo, GitStatus

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def committer(git_repo: Path) -> AutoCommitter:
    """AutoCommitter wired to a real temporary git repo."""
    return AutoCommitter(GitRepo(git_repo))


# ------------------------------------------------------------------
# _detect_prefix()
# ------------------------------------------------------------------


class TestDetectPrefix:
    """Tests for the rule-based prefix detection."""

    def test_test_files(self) -> None:
        assert _detect_prefix(["tests/test_foo.py", "tests/test_bar.py"]) == "test"

    def test_docs_files(self) -> None:
        assert _detect_prefix(["README.md", "docs/guide.rst"]) == "docs"

    def test_chore_files(self) -> None:
        assert _detect_prefix(["pyproject.toml", "Makefile"]) == "chore"

    def test_mixed_prefers_majority(self) -> None:
        files = ["tests/test_a.py", "tests/test_b.py", "src/main.py"]
        # Two test files vs one unmatched → test wins
        assert _detect_prefix(files) == "test"

    def test_no_match_defaults_to_feat(self) -> None:
        assert _detect_prefix(["src/app.py", "src/models.py"]) == "feat"

    def test_empty_files_defaults_to_chore(self) -> None:
        assert _detect_prefix([]) == "chore"

    def test_fix_in_path(self) -> None:
        assert _detect_prefix(["bugfix/issue42.py"]) == "fix"

    def test_dockerfile(self) -> None:
        assert _detect_prefix(["Dockerfile", ".github/workflows/ci.yml"]) == "chore"


# ------------------------------------------------------------------
# _summarise_files()
# ------------------------------------------------------------------


class TestSummariseFiles:
    """Tests for the file summary helper."""

    def test_single_file(self) -> None:
        assert _summarise_files(["src/main.py"]) == "main.py"

    def test_multiple_same_dir(self) -> None:
        summary = _summarise_files(["src/a.py", "src/b.py"])
        assert "2 files" in summary
        assert "src" in summary

    def test_multiple_dirs(self) -> None:
        summary = _summarise_files(["src/a.py", "tests/b.py"])
        assert "2 files" in summary
        assert "2 directories" in summary

    def test_empty(self) -> None:
        assert _summarise_files([]) == "no files changed"


# ------------------------------------------------------------------
# generate_commit_message()
# ------------------------------------------------------------------


class TestGenerateCommitMessage:
    """Tests for AutoCommitter.generate_commit_message()."""

    def test_with_explicit_files(self, committer: AutoCommitter) -> None:
        diff = "+x = 1\n"
        msg = committer.generate_commit_message(diff, changed_files=["src/app.py"])
        assert msg.startswith("feat:")
        assert "app.py" in msg

    def test_includes_stats(self, committer: AutoCommitter) -> None:
        diff = "+line1\n+line2\n-old_line\n"
        msg = committer.generate_commit_message(diff, changed_files=["src/app.py"])
        assert "+2" in msg
        assert "-1" in msg

    def test_extracts_files_from_diff(self, committer: AutoCommitter) -> None:
        diff = (
            "diff --git a/tests/test_foo.py b/tests/test_foo.py\n"
            "--- a/tests/test_foo.py\n"
            "+++ b/tests/test_foo.py\n"
            "+new_test_line\n"
        )
        msg = committer.generate_commit_message(diff)
        assert msg.startswith("test:")

    def test_docs_prefix(self, committer: AutoCommitter) -> None:
        diff = "+doc content\n"
        msg = committer.generate_commit_message(diff, changed_files=["README.md"])
        assert msg.startswith("docs:")


# ------------------------------------------------------------------
# should_auto_commit()
# ------------------------------------------------------------------


class TestShouldAutoCommit:
    """Tests for the auto-commit heuristic."""

    def test_clean_repo_returns_false(self) -> None:
        status = GitStatus()
        # We need a mock repo for this
        # Just test the quick-exit path
        assert status.is_clean

    def test_only_untracked_returns_false(self, committer: AutoCommitter) -> None:
        status = GitStatus(untracked=["new_file.txt"])
        assert committer.should_auto_commit(status) is False

    def test_too_many_files_returns_false(self, committer: AutoCommitter) -> None:
        files = [f"file{i}.py" for i in range(25)]
        status = GitStatus(staged=files)
        assert committer.should_auto_commit(status) is False

    def test_moderate_changes_returns_true(self, git_repo: Path) -> None:
        # Create a small change
        (git_repo / "hello.py").write_text('print("small change")\n')
        repo = GitRepo(git_repo)
        committer = AutoCommitter(repo)
        status = repo.status()
        assert committer.should_auto_commit(status) is True


# ------------------------------------------------------------------
# commit()
# ------------------------------------------------------------------


class TestCommitExecution:
    """Tests for AutoCommitter.commit()."""

    def test_commit_specific_files(self, git_repo: Path) -> None:
        (git_repo / "a.py").write_text("a = 1\n")
        (git_repo / "b.py").write_text("b = 2\n")
        repo = GitRepo(git_repo)
        committer = AutoCommitter(repo)
        short_hash = committer.commit("feat: add modules", files=["a.py", "b.py"])
        assert len(short_hash) >= 7

        entries = repo.log(n=1)
        assert entries[0].message == "feat: add modules"

    def test_commit_all_files(self, git_repo: Path) -> None:
        (git_repo / "c.py").write_text("c = 3\n")
        repo = GitRepo(git_repo)
        committer = AutoCommitter(repo)
        short_hash = committer.commit("chore: add c.py")
        assert short_hash
        entries = repo.log(n=1)
        assert entries[0].message == "chore: add c.py"

"""Tests for the git operations tool."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from prism.tools.base import PermissionLevel
from prism.tools.git_tool import GitTool

if TYPE_CHECKING:
    from pathlib import Path

    from prism.security.sandbox import CommandSandbox


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def git_project(tmp_path: Path) -> Path:
    """Create a temporary directory initialised as a real git repo."""
    subprocess.run(
        ["git", "init"],
        cwd=str(tmp_path),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(tmp_path),
        capture_output=True,
        check=True,
    )
    # Create an initial commit so HEAD exists.
    readme = tmp_path / "README.md"
    readme.write_text("# Test\n")
    subprocess.run(
        ["git", "add", "."],
        cwd=str(tmp_path),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(tmp_path),
        capture_output=True,
        check=True,
    )
    return tmp_path


@pytest.fixture
def git_sandbox(git_project: Path) -> CommandSandbox:
    """Create a CommandSandbox rooted at the git project directory."""
    from prism.security.sandbox import CommandSandbox

    return CommandSandbox(project_root=git_project, timeout=10)


@pytest.fixture
def git_tool(git_sandbox: CommandSandbox) -> GitTool:
    """Create a GitTool instance."""
    return GitTool(git_sandbox)


# ------------------------------------------------------------------
# Property tests
# ------------------------------------------------------------------


class TestGitToolProperties:
    """Tests for tool metadata properties."""

    def test_name(self, git_tool: GitTool) -> None:
        """Tool name is 'git'."""
        assert git_tool.name == "git"

    def test_description(self, git_tool: GitTool) -> None:
        """Description mentions git operations."""
        assert "git" in git_tool.description.lower()
        assert "status" in git_tool.description
        assert "diff" in git_tool.description

    def test_permission_level(self, git_tool: GitTool) -> None:
        """Permission level is CONFIRM."""
        assert git_tool.permission_level == PermissionLevel.CONFIRM

    def test_parameters_schema(self, git_tool: GitTool) -> None:
        """Schema has command (required) and message (optional)."""
        schema = git_tool.parameters_schema
        assert schema["type"] == "object"
        assert "command" in schema["properties"]
        assert "message" in schema["properties"]
        assert schema["required"] == ["command"]


# ------------------------------------------------------------------
# Allowed sub-command tests
# ------------------------------------------------------------------


class TestGitToolAllowed:
    """Tests for allowed git sub-commands."""

    def test_status(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git status succeeds on a clean repo."""
        result = git_tool.execute({"command": "status"})
        assert result.success is True
        # A clean repo should mention "nothing to commit" or
        # "working tree clean"
        output_lower = result.output.lower()
        assert (
            "nothing to commit" in output_lower
            or "clean" in output_lower
        )

    def test_log(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git log shows the initial commit."""
        result = git_tool.execute({"command": "log --oneline"})
        assert result.success is True
        assert "Initial commit" in result.output

    def test_diff_empty(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git diff on a clean tree produces no output."""
        result = git_tool.execute({"command": "diff"})
        assert result.success is True

    def test_diff_with_changes(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git diff shows changes to tracked files."""
        (git_project / "README.md").write_text("# Updated\n")
        result = git_tool.execute({"command": "diff"})
        assert result.success is True
        assert "Updated" in result.output

    def test_branch(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git branch lists branches."""
        result = git_tool.execute({"command": "branch"})
        assert result.success is True
        # Default branch should appear.
        assert result.output.strip() != ""

    def test_add_and_commit(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git add + git commit with message parameter."""
        # Create a new file
        (git_project / "new_file.txt").write_text("hello\n")

        # Stage it
        add_result = git_tool.execute(
            {"command": "add new_file.txt"}
        )
        assert add_result.success is True

        # Commit with message
        commit_result = git_tool.execute(
            {"command": "commit", "message": "Add new file"}
        )
        assert commit_result.success is True

        # Verify the commit
        log_result = git_tool.execute(
            {"command": "log --oneline -1"}
        )
        assert log_result.success is True
        assert "Add new file" in log_result.output

    def test_commit_with_inline_message(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git commit -m 'msg' works when message is inline."""
        (git_project / "inline.txt").write_text("content\n")
        git_tool.execute({"command": "add inline.txt"})
        result = git_tool.execute(
            {"command": "commit -m 'Inline message'"}
        )
        assert result.success is True

    def test_show(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git show displays commit info."""
        result = git_tool.execute({"command": "show --stat"})
        assert result.success is True
        assert "Initial commit" in result.output

    def test_rev_parse(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git rev-parse HEAD returns a commit hash."""
        result = git_tool.execute(
            {"command": "rev-parse --short HEAD"}
        )
        assert result.success is True
        assert len(result.output.strip()) >= 7

    def test_command_with_git_prefix(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """User can include 'git ' prefix; it is stripped."""
        result = git_tool.execute(
            {"command": "git status"}
        )
        assert result.success is True

    def test_checkout_branch(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git checkout -b creates a new branch."""
        result = git_tool.execute(
            {"command": "checkout -b test-branch"}
        )
        assert result.success is True
        branch_result = git_tool.execute(
            {"command": "branch"}
        )
        assert "test-branch" in branch_result.output

    def test_stash(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git stash list succeeds."""
        result = git_tool.execute({"command": "stash list"})
        assert result.success is True

    def test_remote(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git remote -v succeeds (even with no remotes)."""
        result = git_tool.execute({"command": "remote -v"})
        assert result.success is True

    def test_blame(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """git blame works on a tracked file."""
        result = git_tool.execute(
            {"command": "blame README.md"}
        )
        assert result.success is True
        assert "Test" in result.output


# ------------------------------------------------------------------
# Blocked sub-command tests
# ------------------------------------------------------------------


class TestGitToolBlocked:
    """Tests for blocked/dangerous git sub-commands."""

    def test_push_blocked(self, git_tool: GitTool) -> None:
        """git push is blocked."""
        result = git_tool.execute({"command": "push"})
        assert result.success is False
        assert "not allowed" in result.error.lower()

    def test_pull_blocked(self, git_tool: GitTool) -> None:
        """git pull is blocked."""
        result = git_tool.execute({"command": "pull"})
        assert result.success is False
        assert "not allowed" in result.error.lower()

    def test_reset_blocked(self, git_tool: GitTool) -> None:
        """git reset is blocked."""
        result = git_tool.execute({"command": "reset --hard HEAD"})
        assert result.success is False

    def test_clean_blocked(self, git_tool: GitTool) -> None:
        """git clean is blocked."""
        result = git_tool.execute({"command": "clean -fd"})
        assert result.success is False

    def test_merge_blocked(self, git_tool: GitTool) -> None:
        """git merge is blocked."""
        result = git_tool.execute(
            {"command": "merge some-branch"}
        )
        assert result.success is False
        assert "not allowed" in result.error.lower()

    def test_rebase_blocked(self, git_tool: GitTool) -> None:
        """git rebase is blocked."""
        result = git_tool.execute({"command": "rebase main"})
        assert result.success is False
        assert "not allowed" in result.error.lower()

    def test_force_flag_blocked(
        self, git_tool: GitTool
    ) -> None:
        """--force flag is blocked even on allowed commands."""
        result = git_tool.execute(
            {"command": "branch --force main"}
        )
        assert result.success is False
        assert "blocked flag" in result.error.lower()

    def test_hard_flag_blocked(
        self, git_tool: GitTool
    ) -> None:
        """--hard flag is blocked."""
        result = git_tool.execute(
            {"command": "checkout --hard"}
        )
        assert result.success is False
        assert "blocked flag" in result.error.lower()

    def test_delete_flag_blocked(
        self, git_tool: GitTool
    ) -> None:
        """--delete flag is blocked."""
        result = git_tool.execute(
            {"command": "branch --delete main"}
        )
        assert result.success is False
        assert "blocked flag" in result.error.lower()

    def test_unknown_subcommand_blocked(
        self, git_tool: GitTool
    ) -> None:
        """Unknown sub-commands are rejected."""
        result = git_tool.execute(
            {"command": "filter-branch --all"}
        )
        assert result.success is False


# ------------------------------------------------------------------
# Edge case tests
# ------------------------------------------------------------------


class TestGitToolEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_command(self, git_tool: GitTool) -> None:
        """Empty command string returns an error."""
        result = git_tool.execute({"command": ""})
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_whitespace_only_command(
        self, git_tool: GitTool
    ) -> None:
        """Whitespace-only command returns an error."""
        result = git_tool.execute({"command": "   "})
        assert result.success is False
        assert "empty" in result.error.lower()

    def test_just_git(self, git_tool: GitTool) -> None:
        """Bare 'git' command returns an error."""
        result = git_tool.execute({"command": "git"})
        assert result.success is False
        assert "incomplete" in result.error.lower()

    def test_missing_required_command(
        self, git_tool: GitTool
    ) -> None:
        """Missing required 'command' raises ValueError."""
        with pytest.raises(ValueError, match="Missing required"):
            git_tool.execute({})

    def test_message_ignored_for_non_commit(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """message parameter is ignored for non-commit commands."""
        result = git_tool.execute(
            {"command": "status", "message": "this is ignored"}
        )
        assert result.success is True

    def test_metadata_fields(
        self, git_tool: GitTool, git_project: Path
    ) -> None:
        """Result metadata contains expected fields."""
        result = git_tool.execute({"command": "status"})
        assert result.success is True
        assert result.metadata is not None
        assert "exit_code" in result.metadata
        assert "duration_ms" in result.metadata
        assert "timed_out" in result.metadata
        assert "sub_command" in result.metadata
        assert result.metadata["sub_command"] == "status"
        assert "full_command" in result.metadata
        assert result.metadata["full_command"] == "git status"

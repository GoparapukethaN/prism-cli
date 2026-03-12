"""Tests for the edit_file tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.tools.base import PermissionLevel

if TYPE_CHECKING:
    from pathlib import Path

    from prism.tools.file_edit import EditFileTool


class TestEditFileTool:
    """Tests for EditFileTool."""

    def test_name_and_permission(self, edit_tool: EditFileTool) -> None:
        """Tool has correct name and CONFIRM permission."""
        assert edit_tool.name == "edit_file"
        assert edit_tool.permission_level == PermissionLevel.CONFIRM

    def test_basic_search_replace(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Basic single occurrence search/replace works."""
        result = edit_tool.execute({
            "path": "src/main.py",
            "search": 'print("hello world")',
            "replace": 'print("goodbye world")',
        })
        assert result.success is True
        content = (project_dir / "src" / "main.py").read_text()
        assert 'print("goodbye world")' in content
        assert 'print("hello world")' not in content
        # Output should contain a diff
        assert "---" in result.output or "no visible diff" in result.output

    def test_search_string_not_found(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Returns error when search string is not found."""
        result = edit_tool.execute({
            "path": "src/main.py",
            "search": "this does not exist anywhere",
            "replace": "replacement",
        })
        assert result.success is False
        assert "not found" in result.error.lower()
        assert result.metadata is not None
        assert result.metadata["occurrences"] == 0

    def test_multiple_occurrences_without_replace_all(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Errors when search string appears multiple times without replace_all."""
        # Write a file with duplicated content
        target = project_dir / "dupes.py"
        target.write_text("x = 1\nx = 1\nx = 1\n")

        result = edit_tool.execute({
            "path": "dupes.py",
            "search": "x = 1",
            "replace": "x = 2",
        })
        assert result.success is False
        assert "3 times" in result.error
        assert result.metadata is not None
        assert result.metadata["occurrences"] == 3

    def test_replace_all(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """replace_all replaces every occurrence."""
        target = project_dir / "multi.py"
        target.write_text("x = 1\ny = 2\nx = 1\n")

        result = edit_tool.execute({
            "path": "multi.py",
            "search": "x = 1",
            "replace": "x = 99",
            "replace_all": True,
        })
        assert result.success is True
        content = target.read_text()
        assert content.count("x = 99") == 2
        assert "x = 1" not in content
        assert result.metadata is not None
        assert result.metadata["replacements"] == 2

    def test_file_not_found(self, edit_tool: EditFileTool) -> None:
        """Returns error when file does not exist."""
        result = edit_tool.execute({
            "path": "no_such_file.py",
            "search": "x",
            "replace": "y",
        })
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_path_traversal_rejected(self, edit_tool: EditFileTool) -> None:
        """Path traversal is blocked."""
        result = edit_tool.execute({
            "path": "../../etc/passwd",
            "search": "root",
            "replace": "hacked",
        })
        assert result.success is False
        assert result.error is not None

    def test_diff_output(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Output contains a unified diff."""
        result = edit_tool.execute({
            "path": "src/main.py",
            "search": "return 42",
            "replace": "return 100",
        })
        assert result.success is True
        # Unified diff markers
        assert "-" in result.output
        assert "+" in result.output

    def test_validate_missing_search(self, edit_tool: EditFileTool) -> None:
        """Missing search argument raises ValueError."""
        with pytest.raises(ValueError, match="Missing required argument"):
            edit_tool.validate_arguments({"path": "x.py", "replace": "y"})

    def test_edit_preserves_other_content(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Editing one part does not change unrelated content."""
        original = (project_dir / "src" / "main.py").read_text()
        assert "def helper():" in original

        edit_tool.execute({
            "path": "src/main.py",
            "search": 'print("hello world")',
            "replace": 'print("changed")',
        })
        updated = (project_dir / "src" / "main.py").read_text()
        assert "def helper():" in updated
        assert "return 42" in updated

    def test_not_a_file(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Returns error when path is a directory."""
        result = edit_tool.execute({
            "path": "src",
            "search": "x",
            "replace": "y",
        })
        assert result.success is False
        assert "Not a file" in result.error

    def test_replace_with_empty_string(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Can replace with an empty string (delete text)."""
        target = project_dir / "deleteme.py"
        target.write_text("keep this\ndelete this\nkeep this too\n")

        result = edit_tool.execute({
            "path": "deleteme.py",
            "search": "delete this\n",
            "replace": "",
        })
        assert result.success is True
        content = target.read_text()
        assert "delete this" not in content
        assert "keep this" in content


class TestEditFilePreviewDiff:
    """Tests for EditFileTool.generate_preview_diff()."""

    def test_preview_shows_replacement(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Preview diff shows the search/replace change."""
        diff_text = edit_tool.generate_preview_diff({
            "path": "src/main.py",
            "search": 'print("hello world")',
            "replace": 'print("goodbye world")',
        })
        assert diff_text is not None
        assert '-    print("hello world")' in diff_text
        assert '+    print("goodbye world")' in diff_text

    def test_preview_does_not_write(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """generate_preview_diff must not modify the file."""
        original = (project_dir / "src" / "main.py").read_text()
        edit_tool.generate_preview_diff({
            "path": "src/main.py",
            "search": 'print("hello world")',
            "replace": 'print("changed")',
        })
        after = (project_dir / "src" / "main.py").read_text()
        assert original == after

    def test_preview_returns_none_when_not_found(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Returns None when search string is not in the file."""
        diff_text = edit_tool.generate_preview_diff({
            "path": "src/main.py",
            "search": "this text does not exist",
            "replace": "replacement",
        })
        assert diff_text is None

    def test_preview_returns_none_for_missing_file(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Returns None when the file does not exist."""
        diff_text = edit_tool.generate_preview_diff({
            "path": "no_such_file.py",
            "search": "x",
            "replace": "y",
        })
        assert diff_text is None

    def test_preview_multiline_replacement(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """Preview works for multi-line search and replace strings."""
        target = project_dir / "multi.py"
        target.write_text("line1\nline2\nline3\nline4\n")

        diff_text = edit_tool.generate_preview_diff({
            "path": "multi.py",
            "search": "line2\nline3",
            "replace": "replaced2\nreplaced3\nextra_line",
        })
        assert diff_text is not None
        assert "-line2" in diff_text
        assert "-line3" in diff_text
        assert "+replaced2" in diff_text
        assert "+replaced3" in diff_text
        assert "+extra_line" in diff_text

    def test_preview_identical_replacement_returns_none(
        self, edit_tool: EditFileTool, project_dir: Path
    ) -> None:
        """When search and replace are the same, returns None (no diff)."""
        diff_text = edit_tool.generate_preview_diff({
            "path": "src/main.py",
            "search": "return 42",
            "replace": "return 42",
        })
        assert diff_text is None

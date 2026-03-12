"""Tests for the write_file tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.tools.base import PermissionLevel

if TYPE_CHECKING:
    from pathlib import Path

    from prism.tools.file_write import WriteFileTool


class TestWriteFileTool:
    """Tests for WriteFileTool."""

    def test_name_and_permission(self, write_tool: WriteFileTool) -> None:
        """Tool has correct name and CONFIRM permission."""
        assert write_tool.name == "write_file"
        assert write_tool.permission_level == PermissionLevel.CONFIRM

    def test_write_new_file(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """Can create a new file."""
        result = write_tool.execute(
            {"path": "new_file.py", "content": "print('hello')\n"}
        )
        assert result.success is True
        assert "Created" in result.output

        written = project_dir / "new_file.py"
        assert written.exists()
        assert written.read_text() == "print('hello')\n"

    def test_overwrite_existing_file(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """Can overwrite an existing file."""
        result = write_tool.execute(
            {"path": "README.md", "content": "# Updated\n"}
        )
        assert result.success is True
        assert "Overwritten" in result.output
        assert (project_dir / "README.md").read_text() == "# Updated\n"

    def test_creates_parent_directories(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """Creates parent directories when they don't exist."""
        result = write_tool.execute(
            {"path": "new/nested/dir/file.py", "content": "x = 1\n"}
        )
        assert result.success is True
        target = project_dir / "new" / "nested" / "dir" / "file.py"
        assert target.exists()
        assert target.read_text() == "x = 1\n"

    def test_path_traversal_rejected(self, write_tool: WriteFileTool) -> None:
        """Path traversal attempts are blocked."""
        result = write_tool.execute(
            {"path": "../../../tmp/evil.py", "content": "evil"}
        )
        assert result.success is False
        assert result.error is not None

    def test_bytes_written_metadata(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """Result metadata includes bytes_written."""
        content = "hello world\n"
        result = write_tool.execute(
            {"path": "count.txt", "content": content}
        )
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["bytes_written"] == len(content.encode("utf-8"))

    def test_new_file_flag_metadata(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """Metadata correctly reports new_file flag."""
        # New file
        result = write_tool.execute(
            {"path": "brand_new.txt", "content": "data"}
        )
        assert result.metadata is not None
        assert result.metadata["new_file"] is True

        # Overwrite
        result = write_tool.execute(
            {"path": "brand_new.txt", "content": "updated"}
        )
        assert result.metadata is not None
        assert result.metadata["new_file"] is False

    def test_validate_missing_content(self, write_tool: WriteFileTool) -> None:
        """Missing content argument raises ValueError."""
        with pytest.raises(ValueError, match="Missing required argument"):
            write_tool.validate_arguments({"path": "foo.py"})

    def test_validate_missing_path(self, write_tool: WriteFileTool) -> None:
        """Missing path argument raises ValueError."""
        with pytest.raises(ValueError, match="Missing required argument"):
            write_tool.validate_arguments({"content": "hello"})

    def test_write_empty_content(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """Can write empty content."""
        result = write_tool.execute({"path": "empty.txt", "content": ""})
        assert result.success is True
        assert (project_dir / "empty.txt").read_text() == ""
        assert result.metadata is not None
        assert result.metadata["bytes_written"] == 0

    def test_write_unicode_content(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """Can write Unicode content."""
        content = "# Unicode test\ncafe = 'caf\u00e9'\nemoji = '\u2728'\n"
        result = write_tool.execute({"path": "unicode.py", "content": content})
        assert result.success is True
        assert (project_dir / "unicode.py").read_text(encoding="utf-8") == content

    def test_null_byte_path_rejected(self, write_tool: WriteFileTool) -> None:
        """Null bytes in path are rejected."""
        result = write_tool.execute(
            {"path": "file\x00.py", "content": "x"}
        )
        assert result.success is False
        assert result.error is not None


class TestWriteFilePreviewDiff:
    """Tests for WriteFileTool.generate_preview_diff()."""

    def test_new_file_diff_shows_all_additions(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """New file diff shows every line as an addition."""
        diff_text, is_new = write_tool.generate_preview_diff(
            {"path": "brand_new.py", "content": "x = 1\ny = 2\n"}
        )
        assert is_new is True
        assert "+x = 1" in diff_text
        assert "+y = 2" in diff_text
        assert "/dev/null" in diff_text
        assert "b/brand_new.py" in diff_text

    def test_new_file_does_not_write(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """generate_preview_diff must not create the file on disk."""
        write_tool.generate_preview_diff(
            {"path": "should_not_exist.py", "content": "data"}
        )
        assert not (project_dir / "should_not_exist.py").exists()

    def test_overwrite_diff_shows_changes(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """Overwriting an existing file produces a proper unified diff."""
        diff_text, is_new = write_tool.generate_preview_diff(
            {"path": "README.md", "content": "# Updated\n"}
        )
        assert is_new is False
        assert "---" in diff_text
        assert "+++" in diff_text
        assert "-# Test Project" in diff_text
        assert "+# Updated" in diff_text

    def test_overwrite_identical_content_returns_empty(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """When content is identical, the diff text is empty."""
        original_content = (project_dir / "README.md").read_text()
        diff_text, is_new = write_tool.generate_preview_diff(
            {"path": "README.md", "content": original_content}
        )
        assert is_new is False
        assert diff_text == ""

    def test_new_file_single_line_no_trailing_newline(
        self, write_tool: WriteFileTool, project_dir: Path
    ) -> None:
        """A single-line file without trailing newline still produces valid diff."""
        diff_text, is_new = write_tool.generate_preview_diff(
            {"path": "one_liner.txt", "content": "hello"}
        )
        assert is_new is True
        assert "+hello" in diff_text

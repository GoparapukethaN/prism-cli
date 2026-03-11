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

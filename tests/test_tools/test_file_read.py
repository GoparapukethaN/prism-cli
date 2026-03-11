"""Tests for the read_file tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.tools.base import PermissionLevel

if TYPE_CHECKING:
    from pathlib import Path

    from prism.tools.file_read import ReadFileTool


class TestReadFileTool:
    """Tests for ReadFileTool."""

    def test_name_and_permission(self, read_tool: ReadFileTool) -> None:
        """Tool has correct name and AUTO permission."""
        assert read_tool.name == "read_file"
        assert read_tool.permission_level == PermissionLevel.AUTO

    def test_read_existing_file(
        self, read_tool: ReadFileTool, project_dir: Path
    ) -> None:
        """Can read an existing text file with line numbers."""
        result = read_tool.execute({"path": "src/main.py"})
        assert result.success is True
        assert "1: def main():" in result.output
        assert '2:     print("hello world")' in result.output
        assert result.error is None

    def test_read_missing_file(self, read_tool: ReadFileTool) -> None:
        """Returns error for missing files."""
        result = read_tool.execute({"path": "nonexistent.py"})
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_read_directory_not_file(
        self, read_tool: ReadFileTool, project_dir: Path
    ) -> None:
        """Returns error when path points to a directory."""
        result = read_tool.execute({"path": "src"})
        assert result.success is False
        assert "Not a file" in result.error

    def test_read_with_start_and_end_line(
        self, read_tool: ReadFileTool, project_dir: Path
    ) -> None:
        """Supports start_line and end_line for partial reads."""
        result = read_tool.execute(
            {"path": "src/main.py", "start_line": 1, "end_line": 2}
        )
        assert result.success is True
        assert "1:" in result.output
        assert "2:" in result.output
        # Should not contain line 5 (helper function)
        assert "5:" not in result.output

    def test_read_binary_file_detection(
        self, read_tool: ReadFileTool, project_dir: Path
    ) -> None:
        """Binary files are detected and rejected."""
        binary_path = project_dir / "image.bin"
        binary_path.write_bytes(b"\x89PNG\x00\x00\x00" + b"\x00" * 100)

        result = read_tool.execute({"path": "image.bin"})
        assert result.success is False
        assert "binary" in result.error.lower()
        assert result.metadata is not None
        assert result.metadata.get("binary") is True

    def test_read_large_file_truncation(
        self, read_tool: ReadFileTool, project_dir: Path
    ) -> None:
        """Large files are truncated with a warning."""
        large_file = project_dir / "big.txt"
        # Write 2MB of text
        large_file.write_text("x" * (2 * 1024 * 1024))

        result = read_tool.execute({"path": "big.txt"})
        assert result.success is True
        assert "truncated" in result.output.lower()
        assert result.metadata is not None
        assert result.metadata.get("truncated") is True

    def test_path_traversal_rejected(self, read_tool: ReadFileTool) -> None:
        """Path traversal attempts are blocked."""
        result = read_tool.execute({"path": "../../../etc/passwd"})
        assert result.success is False
        assert result.error is not None

    def test_null_byte_rejected(self, read_tool: ReadFileTool) -> None:
        """Paths with null bytes are rejected."""
        result = read_tool.execute({"path": "src/main.py\x00.txt"})
        assert result.success is False
        assert result.error is not None

    def test_read_metadata(
        self, read_tool: ReadFileTool, project_dir: Path
    ) -> None:
        """Result metadata includes file size and line count."""
        result = read_tool.execute({"path": "src/main.py"})
        assert result.success is True
        assert result.metadata is not None
        assert "size" in result.metadata
        assert "lines" in result.metadata
        assert result.metadata["lines"] > 0

    def test_validate_missing_required_arg(self, read_tool: ReadFileTool) -> None:
        """Missing required argument raises ValueError."""
        with pytest.raises(ValueError, match="Missing required argument"):
            read_tool.validate_arguments({})

    def test_read_utf8_fallback_latin1(
        self, read_tool: ReadFileTool, project_dir: Path
    ) -> None:
        """Falls back to latin-1 when UTF-8 decode fails."""
        latin_file = project_dir / "latin.txt"
        latin_file.write_bytes("caf\xe9\nna\xefve\n".encode("latin-1"))

        result = read_tool.execute({"path": "latin.txt"})
        assert result.success is True
        assert "caf" in result.output

    def test_read_empty_file(
        self, read_tool: ReadFileTool, project_dir: Path
    ) -> None:
        """Can read empty files without error."""
        empty = project_dir / "empty.txt"
        empty.write_text("")

        result = read_tool.execute({"path": "empty.txt"})
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["size"] == 0

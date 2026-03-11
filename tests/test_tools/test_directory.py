"""Tests for the list_directory tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prism.tools.base import PermissionLevel

if TYPE_CHECKING:
    from pathlib import Path

    from prism.tools.directory import ListDirectoryTool


class TestListDirectoryTool:
    """Tests for ListDirectoryTool."""

    def test_name_and_permission(self, directory_tool: ListDirectoryTool) -> None:
        """Tool has correct name and AUTO permission."""
        assert directory_tool.name == "list_directory"
        assert directory_tool.permission_level == PermissionLevel.AUTO

    def test_list_project_root(
        self, directory_tool: ListDirectoryTool, project_dir: Path
    ) -> None:
        """Can list the project root."""
        result = directory_tool.execute({"path": "."})
        assert result.success is True
        assert "README.md" in result.output
        assert "src" in result.output
        assert result.metadata is not None
        assert result.metadata["entries"] > 0

    def test_list_subdirectory(
        self, directory_tool: ListDirectoryTool, project_dir: Path
    ) -> None:
        """Can list a subdirectory."""
        result = directory_tool.execute({"path": "src"})
        assert result.success is True
        assert "main.py" in result.output
        assert "utils.py" in result.output

    def test_nonexistent_directory(
        self, directory_tool: ListDirectoryTool
    ) -> None:
        """Returns error for nonexistent directory."""
        result = directory_tool.execute({"path": "no_such_dir"})
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_not_a_directory(
        self, directory_tool: ListDirectoryTool, project_dir: Path
    ) -> None:
        """Returns error when path is a file, not directory."""
        result = directory_tool.execute({"path": "README.md"})
        assert result.success is False
        assert "Not a directory" in result.error

    def test_glob_pattern_filter(
        self, directory_tool: ListDirectoryTool, project_dir: Path
    ) -> None:
        """Glob pattern filters entries."""
        result = directory_tool.execute({"path": "src", "pattern": "*.py"})
        assert result.success is True
        assert "main.py" in result.output
        assert "utils.py" in result.output

    def test_glob_pattern_no_match(
        self, directory_tool: ListDirectoryTool, project_dir: Path
    ) -> None:
        """Glob pattern that matches nothing still succeeds."""
        result = directory_tool.execute({"path": "src", "pattern": "*.java"})
        assert result.success is True
        # Output might be empty or show directories only
        assert "main.py" not in result.output

    def test_excluded_patterns(
        self, directory_tool: ListDirectoryTool, project_dir: Path
    ) -> None:
        """Excluded patterns (e.g., __pycache__) are hidden."""
        pycache = project_dir / "src" / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-312.pyc").write_bytes(b"\x00" * 10)

        result = directory_tool.execute({"path": "src"})
        assert result.success is True
        assert "__pycache__" not in result.output

    def test_recursive_listing(
        self, directory_tool: ListDirectoryTool, project_dir: Path
    ) -> None:
        """Recursive listing includes nested entries."""
        result = directory_tool.execute(
            {"path": ".", "recursive": True, "max_depth": 3}
        )
        assert result.success is True
        # Should see files from nested dirs
        assert "deep.py" in result.output

    def test_path_traversal_rejected(
        self, directory_tool: ListDirectoryTool
    ) -> None:
        """Path traversal is blocked."""
        result = directory_tool.execute({"path": "../../.."})
        assert result.success is False
        assert result.error is not None

    def test_shows_file_metadata(
        self, directory_tool: ListDirectoryTool, project_dir: Path
    ) -> None:
        """Entries include type (file/dir) information."""
        result = directory_tool.execute({"path": "."})
        assert result.success is True
        # Should see 'dir' and 'file' markers
        assert "dir" in result.output
        assert "file" in result.output

    def test_truncation_on_large_directory(
        self, project_dir: Path
    ) -> None:
        """Listings beyond 500 entries are truncated."""
        from prism.security.path_guard import PathGuard
        from prism.tools.directory import ListDirectoryTool

        # Create many files
        big_dir = project_dir / "many"
        big_dir.mkdir()
        for i in range(550):
            (big_dir / f"file_{i:04d}.txt").write_text(f"content {i}\n")

        pg = PathGuard(project_root=project_dir)
        tool = ListDirectoryTool(pg, excluded_patterns=[])
        result = tool.execute({"path": "many"})
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["truncated"] is True
        assert "more entries not shown" in result.output

    def test_custom_excluded_patterns(
        self, project_dir: Path
    ) -> None:
        """Custom excluded patterns are respected."""
        from prism.security.path_guard import PathGuard
        from prism.tools.directory import ListDirectoryTool

        pg = PathGuard(project_root=project_dir)
        tool = ListDirectoryTool(pg, excluded_patterns=["*.md"])
        result = tool.execute({"path": "."})
        assert result.success is True
        assert "README.md" not in result.output

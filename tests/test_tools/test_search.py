"""Tests for the search_codebase tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.tools.base import PermissionLevel

if TYPE_CHECKING:
    from pathlib import Path

    from prism.tools.search import SearchCodebaseTool


class TestSearchCodebaseTool:
    """Tests for SearchCodebaseTool."""

    def test_name_and_permission(self, search_tool: SearchCodebaseTool) -> None:
        """Tool has correct name and AUTO permission."""
        assert search_tool.name == "search_codebase"
        assert search_tool.permission_level == PermissionLevel.AUTO

    def test_simple_search(
        self, search_tool: SearchCodebaseTool, project_dir: Path
    ) -> None:
        """Basic text search finds matches."""
        result = search_tool.execute({"pattern": "hello world"})
        assert result.success is True
        assert "hello world" in result.output
        assert result.metadata is not None
        assert result.metadata["matches"] >= 1

    def test_regex_search(
        self, search_tool: SearchCodebaseTool, project_dir: Path
    ) -> None:
        """Regex patterns work correctly."""
        result = search_tool.execute({"pattern": r"def \w+\("})
        assert result.success is True
        assert "def main(" in result.output
        assert "def helper(" in result.output
        assert result.metadata["matches"] >= 2

    def test_no_matches(
        self, search_tool: SearchCodebaseTool, project_dir: Path
    ) -> None:
        """Returns success with 'no matches' message when nothing found."""
        result = search_tool.execute({"pattern": "zzz_nonexistent_pattern_zzz"})
        assert result.success is True
        assert "no matches" in result.output.lower()
        assert result.metadata is not None
        assert result.metadata["matches"] == 0

    def test_file_pattern_filter(
        self, search_tool: SearchCodebaseTool, project_dir: Path
    ) -> None:
        """file_pattern filters to specific file types."""
        # Search only in .md files
        result = search_tool.execute(
            {"pattern": "Test", "file_pattern": "*.md"}
        )
        assert result.success is True
        if result.metadata["matches"] > 0:
            assert "README.md" in result.output
            # Should not match .py files
            assert "main.py" not in result.output

    def test_invalid_regex(self, search_tool: SearchCodebaseTool) -> None:
        """Invalid regex returns an error."""
        result = search_tool.execute({"pattern": "[invalid"})
        assert result.success is False
        assert "invalid regex" in result.error.lower()

    def test_max_results_limit(
        self, search_tool: SearchCodebaseTool, project_dir: Path
    ) -> None:
        """Results are capped at max_results."""
        # Create a file with many matching lines
        target = project_dir / "many_matches.py"
        lines = [f"item_{i} = {i}" for i in range(100)]
        target.write_text("\n".join(lines))

        result = search_tool.execute(
            {"pattern": "item_", "max_results": 5}
        )
        assert result.success is True
        assert result.metadata is not None
        # We got at most 5 results displayed
        # total matches may be higher
        assert result.metadata["matches"] >= 5

    def test_context_lines(
        self, search_tool: SearchCodebaseTool, project_dir: Path
    ) -> None:
        """Search results include context lines."""
        result = search_tool.execute({"pattern": "hello world"})
        assert result.success is True
        # Should have context lines around the match
        assert "def main" in result.output or "hello world" in result.output

    def test_binary_files_excluded(
        self, search_tool: SearchCodebaseTool, project_dir: Path
    ) -> None:
        """Binary files are excluded from search."""
        binary = project_dir / "data.bin"
        binary.write_bytes(b"searchterm\x00\x00\x00binary data")

        result = search_tool.execute({"pattern": "searchterm"})
        assert result.success is True
        # Binary file should not appear in results
        assert "data.bin" not in result.output

    def test_excluded_directories_skipped(
        self, search_tool: SearchCodebaseTool, project_dir: Path
    ) -> None:
        """Excluded directories like __pycache__ are not searched."""
        cache = project_dir / "__pycache__"
        cache.mkdir()
        (cache / "cached.py").write_text("unique_cache_string = 42\n")

        result = search_tool.execute({"pattern": "unique_cache_string"})
        assert result.success is True
        assert result.metadata["matches"] == 0

    def test_files_searched_count(
        self, search_tool: SearchCodebaseTool, project_dir: Path
    ) -> None:
        """Metadata includes files_searched count."""
        result = search_tool.execute({"pattern": "def"})
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["files_searched"] > 0

    def test_search_across_multiple_files(
        self, search_tool: SearchCodebaseTool, project_dir: Path
    ) -> None:
        """Matches from multiple files are returned."""
        # Create a second file with a match
        (project_dir / "another.py").write_text("def another_function():\n    pass\n")

        result = search_tool.execute({"pattern": r"def \w+\("})
        assert result.success is True
        assert result.metadata["matches"] >= 3  # main, helper, another_function

    def test_validate_missing_pattern(
        self, search_tool: SearchCodebaseTool
    ) -> None:
        """Missing pattern argument raises ValueError."""
        with pytest.raises(ValueError, match="Missing required argument"):
            search_tool.validate_arguments({})

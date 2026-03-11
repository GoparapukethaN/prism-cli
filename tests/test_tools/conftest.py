"""Fixtures for tools tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.security.path_guard import PathGuard
from prism.security.sandbox import CommandSandbox
from prism.tools.directory import ListDirectoryTool
from prism.tools.file_edit import EditFileTool
from prism.tools.file_read import ReadFileTool
from prism.tools.file_write import WriteFileTool
from prism.tools.registry import ToolRegistry
from prism.tools.search import SearchCodebaseTool
from prism.tools.terminal import ExecuteCommandTool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a minimal project directory for tool tests."""
    # Source files
    src = tmp_path / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "main.py").write_text(
        'def main():\n    print("hello world")\n\n\ndef helper():\n    return 42\n'
    )
    (src / "utils.py").write_text(
        "import os\n\ndef get_path():\n    return os.getcwd()\n"
    )

    # Config files
    (tmp_path / "README.md").write_text("# Test Project\n")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')

    # Nested directory
    nested = src / "sub"
    nested.mkdir()
    (nested / "deep.py").write_text("x = 1\n")

    return tmp_path


@pytest.fixture
def path_guard(project_dir: Path) -> PathGuard:
    """Create a PathGuard rooted at the project directory."""
    return PathGuard(project_root=project_dir)


@pytest.fixture
def sandbox(project_dir: Path) -> CommandSandbox:
    """Create a CommandSandbox rooted at the project directory."""
    return CommandSandbox(project_root=project_dir, timeout=10)


@pytest.fixture
def read_tool(path_guard: PathGuard) -> ReadFileTool:
    """Create a ReadFileTool instance."""
    return ReadFileTool(path_guard)


@pytest.fixture
def write_tool(path_guard: PathGuard) -> WriteFileTool:
    """Create a WriteFileTool instance."""
    return WriteFileTool(path_guard)


@pytest.fixture
def edit_tool(path_guard: PathGuard) -> EditFileTool:
    """Create an EditFileTool instance."""
    return EditFileTool(path_guard)


@pytest.fixture
def directory_tool(path_guard: PathGuard) -> ListDirectoryTool:
    """Create a ListDirectoryTool instance."""
    return ListDirectoryTool(path_guard)


@pytest.fixture
def search_tool(path_guard: PathGuard) -> SearchCodebaseTool:
    """Create a SearchCodebaseTool instance."""
    return SearchCodebaseTool(path_guard)


@pytest.fixture
def terminal_tool(sandbox: CommandSandbox) -> ExecuteCommandTool:
    """Create an ExecuteCommandTool instance."""
    return ExecuteCommandTool(sandbox)


@pytest.fixture
def registry(path_guard: PathGuard, sandbox: CommandSandbox) -> ToolRegistry:
    """Create a ToolRegistry with all built-in tools."""
    return ToolRegistry.create_default(path_guard, sandbox)

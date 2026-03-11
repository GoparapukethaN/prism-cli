"""Fixtures for context management tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.context.manager import ContextManager
from prism.context.memory import ProjectMemory
from prism.context.session import SessionManager

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ctx() -> ContextManager:
    """Create a ContextManager with a small token budget for testing."""
    return ContextManager(
        system_prompt="You are a test assistant.",
        max_tokens=10_000,
    )


@pytest.fixture
def large_ctx() -> ContextManager:
    """Create a ContextManager with a large token budget."""
    return ContextManager(
        system_prompt="You are a test assistant.",
        max_tokens=200_000,
    )


@pytest.fixture
def session_mgr(tmp_path: Path) -> SessionManager:
    """Create a SessionManager backed by a temporary directory."""
    sessions_dir = tmp_path / "sessions"
    return SessionManager(sessions_dir)


@pytest.fixture
def memory(tmp_path: Path) -> ProjectMemory:
    """Create a ProjectMemory backed by a temporary directory."""
    return ProjectMemory(project_root=tmp_path)


@pytest.fixture
def python_project(tmp_path: Path) -> Path:
    """Create a small Python project structure for repo map tests."""
    src = tmp_path / "src"
    src.mkdir()

    # main.py
    (src / "main.py").write_text(
        '"""Main module."""\n\n'
        'def main() -> None:\n'
        '    print("hello")\n\n\n'
        'def helper(x: int, y: int) -> int:\n'
        '    return x + y\n'
    )

    # models.py with a class
    (src / "models.py").write_text(
        '"""Data models."""\n\n'
        'class User:\n'
        '    def __init__(self, name: str, email: str) -> None:\n'
        '        self.name = name\n'
        '        self.email = email\n\n'
        '    def display(self) -> str:\n'
        '        return f"{self.name} <{self.email}>"\n\n\n'
        'class Admin(User):\n'
        '    def __init__(self, name: str, email: str, role: str) -> None:\n'
        '        super().__init__(name, email)\n'
        '        self.role = role\n\n'
        '    def has_permission(self, perm: str) -> bool:\n'
        '        return True\n'
    )

    # __init__.py
    (src / "__init__.py").write_text("")

    # utils.py with plain functions
    (src / "utils.py").write_text(
        'def format_name(first: str, last: str) -> str:\n'
        '    return f"{first} {last}"\n\n\n'
        'def parse_int(value: str) -> int:\n'
        '    return int(value)\n'
    )

    # Create a .gitignore
    (tmp_path / ".gitignore").write_text(
        "__pycache__/\n"
        "*.pyc\n"
        ".venv/\n"
        "build/\n"
        "secret_config.py\n"
    )

    # Create an ignored directory
    pycache = src / "__pycache__"
    pycache.mkdir()
    (pycache / "main.cpython-312.pyc").write_bytes(b"\x00")

    # Create an ignored file
    (src / "secret_config.py").write_text("SECRET_KEY = 'abc123'\n")

    return tmp_path

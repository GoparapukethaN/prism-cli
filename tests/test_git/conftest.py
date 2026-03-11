"""Fixtures for git module tests.

Creates real temporary git repositories via ``git init`` and ``git commit``
so that tests exercise actual CLI interactions without touching the
host repo.
"""

from __future__ import annotations

import contextlib
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _run_git(cwd: Path, *args: str) -> str:
    """Run a git command inside *cwd* and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create an initialised git repository with one commit.

    The repo contains a single ``hello.py`` file committed on branch
    ``main``.  Returns the repo root directory.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "test@prism.dev")
    _run_git(repo, "config", "user.name", "Prism Test")
    # Rename default branch to "main" (compat with git < 2.28 which lacks -b flag)
    with contextlib.suppress(RuntimeError):
        _run_git(repo, "checkout", "-b", "main")

    (repo / "hello.py").write_text('print("hello")\n')
    _run_git(repo, "add", "hello.py")
    _run_git(repo, "commit", "-m", "initial commit")

    return repo


@pytest.fixture
def git_repo_with_changes(git_repo: Path) -> Path:
    """Repository with both staged and unstaged changes.

    * ``hello.py`` — modified (unstaged)
    * ``staged.py`` — new file, staged
    * ``untracked.txt`` — untracked
    """
    (git_repo / "hello.py").write_text('print("hello world")\n')
    (git_repo / "staged.py").write_text("x = 1\n")
    _run_git(git_repo, "add", "staged.py")
    (git_repo / "untracked.txt").write_text("notes\n")
    return git_repo


@pytest.fixture
def non_git_dir(tmp_path: Path) -> Path:
    """A plain directory that is *not* a git repository."""
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "file.txt").write_text("not a repo\n")
    return plain


@pytest.fixture
def git_repo_multiple_commits(git_repo: Path) -> Path:
    """Repository with three commits for log testing."""
    for i in range(1, 3):
        (git_repo / f"file{i}.py").write_text(f"content {i}\n")
        _run_git(git_repo, "add", f"file{i}.py")
        _run_git(git_repo, "commit", "-m", f"commit number {i}")
    return git_repo

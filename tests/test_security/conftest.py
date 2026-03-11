"""Shared fixtures for security module tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.security.audit import AuditLogger
from prism.security.path_guard import PathGuard
from prism.security.sandbox import CommandSandbox
from prism.security.secret_filter import SecretFilter

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project directory tree for testing."""
    # Create some files and directories
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_ok(): pass\n")
    (tmp_path / ".env").write_text("SECRET=123\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "input.txt").write_text("sample data\n")
    return tmp_path


@pytest.fixture()
def path_guard(project_root: Path) -> PathGuard:
    """PathGuard instance bound to the test project root."""
    return PathGuard(
        project_root=project_root,
        excluded_patterns=[
            ".env",
            ".env.*",
            "*.env",
            "**/*.pem",
            "**/*.key",
            "**/credentials.json",
        ],
    )


@pytest.fixture()
def secret_filter() -> SecretFilter:
    """SecretFilter instance with default patterns."""
    return SecretFilter()


@pytest.fixture()
def sandbox(project_root: Path) -> CommandSandbox:
    """CommandSandbox instance bound to the test project root."""
    return CommandSandbox(
        project_root=project_root,
        timeout=10,
    )


@pytest.fixture()
def audit_logger(tmp_path: Path) -> AuditLogger:
    """AuditLogger writing to a temporary file."""
    log_path = tmp_path / "audit.log"
    return AuditLogger(
        log_path=log_path,
        max_bytes=1_000_000,
        backup_count=2,
    )


@pytest.fixture()
def env_with_secrets() -> dict[str, str]:
    """Sample environment dict containing sensitive and non-sensitive vars."""
    return {
        "HOME": "/home/user",
        "PATH": "/usr/bin:/bin",
        "OPENAI_API_KEY": "sk-secret-key-1234",
        "AWS_SECRET": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "DATABASE_URL": "postgres://user:pass@localhost/db",
        "MY_APP_PASSWORD": "hunter2",
        "REDIS_URL": "redis://localhost:6379",
        "PRISM_HOME": "/tmp/prism",
        "EDITOR": "vim",
        "LANG": "en_US.UTF-8",
    }

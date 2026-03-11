"""Global test fixtures for Prism."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from prism.config.schema import PrismConfig
from prism.config.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def default_settings(tmp_path: Path) -> Settings:
    """Create default settings for testing."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    """Create a minimal project directory for testing."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text('def main():\n    print("hello")\n')
    (src / "__init__.py").write_text("")
    (tmp_path / "README.md").write_text("# Test Project\n")
    (tmp_path / ".prism.md").write_text("# Test Project\n## Stack\nPython\n")

    # Create a fake .git directory
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")

    return tmp_path


@pytest.fixture
def mock_auth_manager() -> MagicMock:
    """Create a mock AuthManager that returns keys for common providers."""
    manager = MagicMock()
    manager.get_key.side_effect = {
        "anthropic": "sk-ant-test-key-1234",
        "openai": "sk-test-key-1234",
        "google": "test-google-key-1234",
        "deepseek": "sk-deepseek-test-1234",
        "groq": "gsk_test_key_1234",
        "mistral": "test-mistral-key-1234",
        "ollama": None,  # Ollama doesn't need a key
    }.get
    return manager

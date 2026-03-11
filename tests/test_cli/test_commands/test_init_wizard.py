"""Tests for the InitWizard — interactive setup wizard."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from prism.cli.commands.init_wizard import (
    EnvInfo,
    InitResult,
    InitWizard,
)
from prism.config.schema import PrismConfig
from prism.config.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings backed by a temporary directory."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


@pytest.fixture
def mock_auth() -> MagicMock:
    """Mock AuthManager that stores keys in a dict."""
    auth = MagicMock()
    stored: dict[str, str] = {}

    def _store(provider: str, key: str, **kwargs: object) -> str:
        stored[provider] = key
        return "keyring"

    auth.store_key.side_effect = _store
    auth._stored = stored
    return auth


@pytest.fixture
def wizard(mock_auth: MagicMock, tmp_settings: Settings) -> InitWizard:
    """InitWizard with mocked dependencies and a quiet console."""
    console = Console(quiet=True)
    return InitWizard(
        auth_manager=mock_auth,
        provider_registry=None,
        settings=tmp_settings,
        console=console,
    )


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def test_detect_environment(wizard: InitWizard) -> None:
    """_detect_environment returns an EnvInfo with all fields populated."""
    env = wizard._detect_environment()
    assert isinstance(env, EnvInfo)
    assert env.os_name  # non-empty
    assert env.python_version  # non-empty
    assert isinstance(env.git_available, bool)
    assert isinstance(env.ollama_available, bool)


def test_env_info_fields() -> None:
    """EnvInfo dataclass stores all expected fields."""
    info = EnvInfo(
        os_name="Darwin",
        os_version="23.5.0",
        python_version="3.12.4",
        git_available=True,
        ollama_available=False,
    )
    assert info.os_name == "Darwin"
    assert info.os_version == "23.5.0"
    assert info.python_version == "3.12.4"
    assert info.git_available is True
    assert info.ollama_available is False


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------


def test_setup_no_providers(wizard: InitWizard) -> None:
    """User skips all providers -> empty list returned."""
    with patch("prism.cli.commands.init_wizard.Confirm") as mock_confirm:
        mock_confirm.ask.return_value = False
        result = wizard._setup_providers()
    assert result == []


def test_setup_one_provider(wizard: InitWizard, mock_auth: MagicMock) -> None:
    """User configures exactly one provider."""
    call_count = 0

    def _confirm_side_effect(*args: object, **kwargs: object) -> bool:
        nonlocal call_count
        call_count += 1
        # Only say yes to the first provider (anthropic)
        return call_count == 1

    with (
        patch("prism.cli.commands.init_wizard.Confirm") as mock_confirm,
        patch("prism.cli.commands.init_wizard.Prompt") as mock_prompt,
    ):
        mock_confirm.ask.side_effect = _confirm_side_effect
        mock_prompt.ask.return_value = "sk-ant-test-key-1234"

        result = wizard._setup_providers()

    assert result == ["anthropic"]
    mock_auth.store_key.assert_called_once()


def test_setup_multiple_providers(wizard: InitWizard, mock_auth: MagicMock) -> None:
    """User configures two providers."""
    provider_idx = 0
    keys = {
        "anthropic": "sk-ant-test-key-1234",
        "openai": "sk-test-openai-key-5678",
    }
    target_providers = list(keys.keys())

    def _confirm_side_effect(*args: object, **kwargs: object) -> bool:
        nonlocal provider_idx
        # Confirm asks are interleaved: configure? yes/no, then possibly store_anyway
        prompt_str = str(args[0]) if args else ""
        for tp in target_providers:
            if tp.lower() in prompt_str.lower() or "anthropic" in prompt_str.lower() or "openai" in prompt_str.lower():
                pass
        provider_idx += 1
        return provider_idx <= 2  # Say yes to first two providers

    prompt_call_count = 0

    def _prompt_side_effect(*args: object, **kwargs: object) -> str:
        nonlocal prompt_call_count
        prompt_call_count += 1
        if prompt_call_count == 1:
            return "sk-ant-test-key-1234"
        return "sk-test-openai-key-5678"

    with (
        patch("prism.cli.commands.init_wizard.Confirm") as mock_confirm,
        patch("prism.cli.commands.init_wizard.Prompt") as mock_prompt,
    ):
        mock_confirm.ask.side_effect = _confirm_side_effect
        mock_prompt.ask.side_effect = _prompt_side_effect

        result = wizard._setup_providers()

    assert len(result) == 2
    assert "anthropic" in result
    assert "openai" in result


def test_invalid_key_rejected(wizard: InitWizard, mock_auth: MagicMock) -> None:
    """Invalid key format triggers a warning; user declines to store."""
    call_count = 0

    def _confirm_side_effect(*args: object, **kwargs: object) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return True  # Yes, configure anthropic
        if call_count == 2:
            return False  # No, don't store invalid key
        return False  # Skip rest

    with (
        patch("prism.cli.commands.init_wizard.Confirm") as mock_confirm,
        patch("prism.cli.commands.init_wizard.Prompt") as mock_prompt,
    ):
        mock_confirm.ask.side_effect = _confirm_side_effect
        # Invalid key for anthropic (doesn't start with sk-ant-)
        mock_prompt.ask.return_value = "invalid-key-format"

        result = wizard._setup_providers()

    assert result == []
    mock_auth.store_key.assert_not_called()


# ---------------------------------------------------------------------------
# Ollama detection
# ---------------------------------------------------------------------------


def test_ollama_detection_installed(wizard: InitWizard) -> None:
    """When Ollama binary is found, installed=True."""
    with (
        patch("prism.cli.commands.init_wizard.shutil.which", return_value="/usr/local/bin/ollama"),
        patch.object(wizard, "_probe_ollama", return_value=True),
        patch.object(wizard, "_list_ollama_models", return_value=["llama3.2:3b", "qwen2.5-coder:7b"]),
    ):
        info = wizard._detect_ollama()

    assert info.installed is True
    assert info.running is True
    assert "llama3.2:3b" in info.models


def test_ollama_not_installed(wizard: InitWizard) -> None:
    """When Ollama binary is not found, installed=False."""
    with patch("prism.cli.commands.init_wizard.shutil.which", return_value=None):
        info = wizard._detect_ollama()

    assert info.installed is False
    assert info.running is False
    assert info.models == []


# ---------------------------------------------------------------------------
# Config file creation
# ---------------------------------------------------------------------------


def test_config_file_created(wizard: InitWizard, tmp_settings: Settings) -> None:
    """_create_config creates a YAML config file."""
    config_path = wizard._create_config(["anthropic"])
    assert config_path.exists()

    import yaml

    with config_path.open() as f:
        data = yaml.safe_load(f)

    assert "routing" in data
    assert data["routing"]["simple_threshold"] == 0.3
    assert data["preferred_provider"] == "anthropic"


def test_existing_config_not_overwritten(wizard: InitWizard, tmp_settings: Settings) -> None:
    """If config already exists, it is not overwritten."""
    config_path = tmp_settings.config_file_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("existing: true\n")

    returned = wizard._create_config(["openai"])
    assert returned == config_path

    content = config_path.read_text()
    assert "existing: true" in content
    assert "openai" not in content


def test_project_config_created(wizard: InitWizard, tmp_path: Path) -> None:
    """_create_project_config creates .prism.yaml in project root."""
    project_config = wizard._create_project_config(tmp_path)
    assert project_config.exists()
    assert project_config.name == ".prism.yaml"

    import yaml

    with project_config.open() as f:
        data = yaml.safe_load(f)

    assert "project" in data
    assert "routing" in data


# ---------------------------------------------------------------------------
# Cost comparison
# ---------------------------------------------------------------------------


def test_cost_comparison_display(wizard: InitWizard) -> None:
    """_show_cost_comparison runs without error for valid providers."""
    # Should not raise
    wizard._show_cost_comparison(["anthropic", "openai"])


def test_cost_comparison_empty_providers(wizard: InitWizard) -> None:
    """_show_cost_comparison handles empty list gracefully."""
    # Should not raise
    wizard._show_cost_comparison([])


# ---------------------------------------------------------------------------
# Budget suggestion
# ---------------------------------------------------------------------------


def test_budget_suggestion(wizard: InitWizard) -> None:
    """_suggest_budget returns a float for configured providers."""
    result = wizard._suggest_budget(["anthropic"])
    assert result is not None
    assert isinstance(result, float)
    assert result > 0


def test_budget_suggestion_no_providers(wizard: InitWizard) -> None:
    """_suggest_budget returns None when no providers configured."""
    result = wizard._suggest_budget([])
    assert result is None


# ---------------------------------------------------------------------------
# Full wizard run (integration)
# ---------------------------------------------------------------------------


def test_run_full_wizard(
    mock_auth: MagicMock,
    tmp_path: Path,
) -> None:
    """run() completes end-to-end with mocked inputs."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()

    console = Console(quiet=True)
    wizard = InitWizard(
        auth_manager=mock_auth,
        provider_registry=None,
        settings=settings,
        console=console,
    )

    with (
        patch("prism.cli.commands.init_wizard.Confirm") as mock_confirm,
        patch("prism.cli.commands.init_wizard.Prompt") as mock_prompt,
        patch("prism.cli.commands.init_wizard.shutil.which", return_value=None),
    ):
        # Skip all providers
        mock_confirm.ask.return_value = False
        mock_prompt.ask.return_value = ""

        result = wizard.run(project_root=tmp_path)

    assert isinstance(result, InitResult)
    assert result.config_path.exists()
    assert result.ollama_available is False
    assert isinstance(result.configured_providers, list)


def test_run_full_wizard_with_provider(
    mock_auth: MagicMock,
    tmp_path: Path,
) -> None:
    """run() configures a provider end-to-end."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()

    console = Console(quiet=True)
    wizard = InitWizard(
        auth_manager=mock_auth,
        provider_registry=None,
        settings=settings,
        console=console,
    )

    confirm_calls = 0

    def _confirm(*args: object, **kwargs: object) -> bool:
        nonlocal confirm_calls
        confirm_calls += 1
        return confirm_calls == 1  # Only first provider

    with (
        patch("prism.cli.commands.init_wizard.Confirm") as mock_confirm,
        patch("prism.cli.commands.init_wizard.Prompt") as mock_prompt,
        patch("prism.cli.commands.init_wizard.shutil.which", return_value=None),
    ):
        mock_confirm.ask.side_effect = _confirm
        mock_prompt.ask.return_value = "sk-ant-test-key-1234"

        result = wizard.run(project_root=tmp_path)

    assert "anthropic" in result.configured_providers
    assert result.config_path.exists()

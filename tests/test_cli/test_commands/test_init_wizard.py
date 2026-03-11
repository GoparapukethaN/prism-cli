"""Tests for the InitWizard — enhanced setup wizard.

This file contains integration-style tests that exercise the InitWizard
from ``prism.cli.commands.init_wizard`` end-to-end with mocked subprocess
calls and environment variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from prism.cli.commands.init_wizard import (
    PROVIDER_CONFIGS,
    InitWizard,
    ProviderSetup,
    SystemInfo,
    WizardResult,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wizard(tmp_path: Path) -> InitWizard:
    """InitWizard pointed at a temporary directory."""
    return InitWizard(project_root=tmp_path)


# ---------------------------------------------------------------------------
# Environment / system detection
# ---------------------------------------------------------------------------


@patch("prism.cli.commands.init_wizard.InitWizard._check_docker", return_value=False)
@patch(
    "prism.cli.commands.init_wizard.InitWizard._check_ollama_installed",
    return_value=False,
)
@patch(
    "prism.cli.commands.init_wizard.InitWizard._detect_gpu",
    return_value=(False, ""),
)
@patch("prism.cli.commands.init_wizard.InitWizard._detect_ram", return_value=16.0)
def test_detect_system(
    _ram: MagicMock,
    _gpu: MagicMock,
    _ollama: MagicMock,
    _docker: MagicMock,
    wizard: InitWizard,
) -> None:
    """detect_system returns a SystemInfo with all fields populated."""
    info = wizard.detect_system()
    assert isinstance(info, SystemInfo)
    assert info.os_name  # non-empty
    assert info.python_version  # non-empty
    assert isinstance(info.ollama_installed, bool)


def test_system_info_fields() -> None:
    """SystemInfo dataclass stores all expected fields."""
    info = SystemInfo(
        os_name="Darwin",
        os_version="23.5.0",
        python_version="3.12.4",
        ram_gb=16.0,
        gpu_detected=True,
        gpu_name="Apple M1",
        cpu_cores=8,
        ollama_installed=False,
        ollama_models=[],
        docker_installed=False,
    )
    assert info.os_name == "Darwin"
    assert info.os_version == "23.5.0"
    assert info.python_version == "3.12.4"
    assert info.gpu_detected is True
    assert info.ollama_installed is False


# ---------------------------------------------------------------------------
# Provider checking
# ---------------------------------------------------------------------------


def test_check_provider_not_configured(
    wizard: InitWizard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no env var is set, provider is not configured."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = ProviderSetup(
        name="anthropic",
        display_name="Anthropic (Claude)",
        env_var="ANTHROPIC_API_KEY",
    )
    assert wizard.check_provider(p) is False
    assert p.is_configured is False


def test_check_provider_configured(
    wizard: InitWizard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When env var is set to a valid-length key, provider is configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-long-key-1234567890")
    p = ProviderSetup(
        name="openai",
        display_name="OpenAI",
        env_var="OPENAI_API_KEY",
    )
    assert wizard.check_provider(p) is True
    assert p.is_configured is True


def test_get_configured_providers(
    wizard: InitWizard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_configured_providers returns all providers with status."""
    for p in PROVIDER_CONFIGS:
        monkeypatch.delenv(p.env_var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-1234567890")

    providers = wizard.get_configured_providers()
    assert len(providers) == len(PROVIDER_CONFIGS)

    configured_names = [p.name for p in providers if p.is_configured]
    assert "anthropic" in configured_names


# ---------------------------------------------------------------------------
# Ollama detection
# ---------------------------------------------------------------------------


@patch("prism.cli.commands.init_wizard.subprocess.run")
def test_ollama_installed(mock_run: MagicMock) -> None:
    """When Ollama binary is found, _check_ollama_installed returns True."""
    mock_run.return_value = MagicMock(returncode=0, stdout="ollama v0.1.27")
    assert InitWizard._check_ollama_installed() is True


@patch("prism.cli.commands.init_wizard.subprocess.run")
def test_ollama_not_installed(mock_run: MagicMock) -> None:
    """When Ollama binary is not found, returns False."""
    mock_run.side_effect = FileNotFoundError("ollama not found")
    assert InitWizard._check_ollama_installed() is False


# ---------------------------------------------------------------------------
# Config file creation
# ---------------------------------------------------------------------------


def test_config_file_created(wizard: InitWizard, tmp_path: Path) -> None:
    """create_config creates a YAML config file with budget settings."""
    prism_home = tmp_path / ".prism"
    config_path = wizard.create_config(prism_home=prism_home)
    assert config_path.exists()

    content = config_path.read_text()
    assert "budget:" in content
    assert "daily_limit: 10.0" in content
    assert "monthly_limit: 50.0" in content
    assert "routing:" in content


def test_config_custom_budget(wizard: InitWizard, tmp_path: Path) -> None:
    """create_config respects custom budget values."""
    prism_home = tmp_path / ".prism"
    config_path = wizard.create_config(
        prism_home=prism_home, daily_budget=25.0, monthly_budget=200.0
    )
    content = config_path.read_text()
    assert "daily_limit: 25.0" in content
    assert "monthly_limit: 200.0" in content


def test_project_memory_created(wizard: InitWizard, tmp_path: Path) -> None:
    """create_project_memory creates .prism.md in project root."""
    memory_path = wizard.create_project_memory()
    assert memory_path.exists()
    assert memory_path.name == ".prism.md"
    content = memory_path.read_text()
    assert "Project Overview" in content


def test_project_memory_not_overwritten(wizard: InitWizard, tmp_path: Path) -> None:
    """create_project_memory does not overwrite existing .prism.md."""
    existing = tmp_path / ".prism.md"
    existing.write_text("# My project\n")
    memory_path = wizard.create_project_memory()
    assert memory_path.read_text() == "# My project\n"


# ---------------------------------------------------------------------------
# Cost comparison
# ---------------------------------------------------------------------------


def test_cost_comparison(
    wizard: InitWizard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_cost_comparison returns rows with expected keys."""
    for p in PROVIDER_CONFIGS:
        monkeypatch.delenv(p.env_var, raising=False)
    rows = wizard.get_cost_comparison()
    assert isinstance(rows, list)
    assert len(rows) == len(PROVIDER_CONFIGS)
    for row in rows:
        assert "provider" in row
        assert "configured" in row
        assert "cost" in row


# ---------------------------------------------------------------------------
# Full wizard run (integration)
# ---------------------------------------------------------------------------


@patch("prism.cli.commands.init_wizard.InitWizard._check_docker", return_value=False)
@patch(
    "prism.cli.commands.init_wizard.InitWizard._check_ollama_installed",
    return_value=False,
)
@patch(
    "prism.cli.commands.init_wizard.InitWizard._detect_gpu",
    return_value=(False, ""),
)
@patch("prism.cli.commands.init_wizard.InitWizard._detect_ram", return_value=16.0)
def test_run_full_wizard(
    _ram: MagicMock,
    _gpu: MagicMock,
    _ollama: MagicMock,
    _docker: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run() completes end-to-end and returns WizardResult."""
    for p in PROVIDER_CONFIGS:
        monkeypatch.delenv(p.env_var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    wizard = InitWizard(project_root=tmp_path)
    result = wizard.run()

    assert isinstance(result, WizardResult)
    assert result.config_path is not None
    assert result.config_path.exists()
    assert result.memory_path is not None
    assert result.memory_path.exists()
    assert result.ignore_path is not None
    assert result.ignore_path.exists()
    assert isinstance(result.providers_configured, list)


@patch("prism.cli.commands.init_wizard.InitWizard._check_docker", return_value=False)
@patch(
    "prism.cli.commands.init_wizard.InitWizard._check_ollama_installed",
    return_value=False,
)
@patch(
    "prism.cli.commands.init_wizard.InitWizard._detect_gpu",
    return_value=(False, ""),
)
@patch("prism.cli.commands.init_wizard.InitWizard._detect_ram", return_value=16.0)
def test_run_full_wizard_with_provider(
    _ram: MagicMock,
    _gpu: MagicMock,
    _ollama: MagicMock,
    _docker: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run() detects configured providers."""
    for p in PROVIDER_CONFIGS:
        monkeypatch.delenv(p.env_var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234567890")
    monkeypatch.setenv("HOME", str(tmp_path))

    wizard = InitWizard(project_root=tmp_path)
    result = wizard.run()

    assert "anthropic" in result.providers_configured
    assert result.config_path is not None
    assert result.config_path.exists()

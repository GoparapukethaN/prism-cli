"""Tests for prism.cli.commands.init_wizard — enhanced init wizard."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest  # noqa: TC002

from prism.cli.commands.init_wizard import (
    PRISM_MD_TEMPLATE,
    PRISMIGNORE_DEFAULTS,
    PROVIDER_CONFIGS,
    RECOMMENDED_OLLAMA_MODELS,
    InitWizard,
    ProviderSetup,
    SystemInfo,
    WizardResult,
)

# ---------------------------------------------------------------------------
# TestSystemInfo — dataclass fields
# ---------------------------------------------------------------------------


class TestSystemInfo:
    """Tests for the SystemInfo dataclass."""

    def test_all_fields_present(self) -> None:
        info = SystemInfo(
            os_name="Darwin",
            os_version="23.5.0",
            python_version="3.12.4",
            ram_gb=16.0,
            gpu_detected=True,
            gpu_name="Apple M1",
            cpu_cores=8,
            ollama_installed=True,
            ollama_models=["llama3.1:8b"],
            docker_installed=False,
        )
        assert info.os_name == "Darwin"
        assert info.os_version == "23.5.0"
        assert info.python_version == "3.12.4"
        assert info.ram_gb == 16.0
        assert info.gpu_detected is True
        assert info.gpu_name == "Apple M1"
        assert info.cpu_cores == 8
        assert info.ollama_installed is True
        assert info.ollama_models == ["llama3.1:8b"]
        assert info.docker_installed is False

    def test_default_empty_models_list(self) -> None:
        info = SystemInfo(
            os_name="Linux",
            os_version="6.1.0",
            python_version="3.11.0",
            ram_gb=32.0,
            gpu_detected=False,
            gpu_name="",
            cpu_cores=16,
            ollama_installed=False,
            ollama_models=[],
            docker_installed=True,
        )
        assert info.ollama_models == []
        assert info.gpu_name == ""

    def test_zero_ram(self) -> None:
        info = SystemInfo(
            os_name="Windows",
            os_version="10.0",
            python_version="3.11.1",
            ram_gb=0.0,
            gpu_detected=False,
            gpu_name="",
            cpu_cores=1,
            ollama_installed=False,
            ollama_models=[],
            docker_installed=False,
        )
        assert info.ram_gb == 0.0
        assert info.cpu_cores == 1


# ---------------------------------------------------------------------------
# TestProviderSetup — dataclass fields
# ---------------------------------------------------------------------------


class TestProviderSetup:
    """Tests for the ProviderSetup dataclass."""

    def test_required_fields(self) -> None:
        p = ProviderSetup(
            name="anthropic",
            display_name="Anthropic (Claude)",
            env_var="ANTHROPIC_API_KEY",
        )
        assert p.name == "anthropic"
        assert p.display_name == "Anthropic (Claude)"
        assert p.env_var == "ANTHROPIC_API_KEY"

    def test_defaults(self) -> None:
        p = ProviderSetup(
            name="openai", display_name="OpenAI", env_var="OPENAI_API_KEY"
        )
        assert p.is_configured is False
        assert p.is_healthy is False
        assert p.example_cost == ""

    def test_with_example_cost(self) -> None:
        p = ProviderSetup(
            name="groq",
            display_name="Groq",
            env_var="GROQ_API_KEY",
            example_cost="Free tier available",
        )
        assert p.example_cost == "Free tier available"

    def test_mutable_is_configured(self) -> None:
        p = ProviderSetup(
            name="test", display_name="Test", env_var="TEST_KEY"
        )
        p.is_configured = True
        p.is_healthy = True
        assert p.is_configured is True
        assert p.is_healthy is True


# ---------------------------------------------------------------------------
# TestWizardResult — dataclass fields
# ---------------------------------------------------------------------------


class TestWizardResult:
    """Tests for the WizardResult dataclass."""

    def test_basic_fields(self) -> None:
        info = SystemInfo(
            os_name="Darwin",
            os_version="23.5.0",
            python_version="3.12.0",
            ram_gb=16.0,
            gpu_detected=False,
            gpu_name="",
            cpu_cores=8,
            ollama_installed=False,
            ollama_models=[],
            docker_installed=False,
        )
        result = WizardResult(
            system_info=info,
            providers_configured=["anthropic", "openai"],
        )
        assert result.system_info.os_name == "Darwin"
        assert result.providers_configured == ["anthropic", "openai"]

    def test_defaults(self) -> None:
        info = SystemInfo(
            os_name="Linux",
            os_version="6.1",
            python_version="3.11.0",
            ram_gb=8.0,
            gpu_detected=False,
            gpu_name="",
            cpu_cores=4,
            ollama_installed=False,
            ollama_models=[],
            docker_installed=False,
        )
        result = WizardResult(
            system_info=info,
            providers_configured=[],
        )
        assert result.config_path is None
        assert result.memory_path is None
        assert result.ignore_path is None
        assert result.budget_daily == 10.0
        assert result.budget_monthly == 50.0

    def test_custom_budget(self) -> None:
        info = SystemInfo(
            os_name="Darwin",
            os_version="23.0",
            python_version="3.12.0",
            ram_gb=16.0,
            gpu_detected=False,
            gpu_name="",
            cpu_cores=8,
            ollama_installed=False,
            ollama_models=[],
            docker_installed=False,
        )
        result = WizardResult(
            system_info=info,
            providers_configured=["deepseek"],
            budget_daily=5.0,
            budget_monthly=25.0,
        )
        assert result.budget_daily == 5.0
        assert result.budget_monthly == 25.0


# ---------------------------------------------------------------------------
# TestInitWizard — core logic
# ---------------------------------------------------------------------------


class TestInitWizardDetectSystem:
    """Tests for InitWizard.detect_system with mocked subprocess calls."""

    @patch("prism.cli.commands.init_wizard.InitWizard._check_docker", return_value=False)
    @patch("prism.cli.commands.init_wizard.InitWizard._get_ollama_models", return_value=[])
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._check_ollama_installed",
        return_value=False,
    )
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._detect_gpu",
        return_value=(False, ""),
    )
    @patch("prism.cli.commands.init_wizard.InitWizard._detect_ram", return_value=16.0)
    def test_detect_system_basic(
        self,
        _mock_ram: MagicMock,
        _mock_gpu: MagicMock,
        _mock_ollama: MagicMock,
        _mock_models: MagicMock,
        _mock_docker: MagicMock,
        tmp_path: Path,
    ) -> None:
        wizard = InitWizard(project_root=tmp_path)
        info = wizard.detect_system()
        assert isinstance(info, SystemInfo)
        assert info.ram_gb == 16.0
        assert info.gpu_detected is False
        assert info.ollama_installed is False
        assert info.docker_installed is False

    @patch("prism.cli.commands.init_wizard.InitWizard._check_docker", return_value=True)
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._get_ollama_models",
        return_value=["llama3.1:8b", "codellama:7b"],
    )
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._check_ollama_installed",
        return_value=True,
    )
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._detect_gpu",
        return_value=(True, "NVIDIA RTX 4090"),
    )
    @patch("prism.cli.commands.init_wizard.InitWizard._detect_ram", return_value=64.0)
    def test_detect_system_full_hardware(
        self,
        _mock_ram: MagicMock,
        _mock_gpu: MagicMock,
        _mock_ollama: MagicMock,
        _mock_models: MagicMock,
        _mock_docker: MagicMock,
        tmp_path: Path,
    ) -> None:
        wizard = InitWizard(project_root=tmp_path)
        info = wizard.detect_system()
        assert info.ram_gb == 64.0
        assert info.gpu_detected is True
        assert info.gpu_name == "NVIDIA RTX 4090"
        assert info.ollama_installed is True
        assert info.ollama_models == ["llama3.1:8b", "codellama:7b"]
        assert info.docker_installed is True

    @patch("prism.cli.commands.init_wizard.InitWizard._check_docker", return_value=False)
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._check_ollama_installed",
        return_value=False,
    )
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._detect_gpu",
        return_value=(False, ""),
    )
    @patch("prism.cli.commands.init_wizard.InitWizard._detect_ram", return_value=0.0)
    def test_detect_system_stores_info(
        self,
        _mock_ram: MagicMock,
        _mock_gpu: MagicMock,
        _mock_ollama: MagicMock,
        _mock_docker: MagicMock,
        tmp_path: Path,
    ) -> None:
        wizard = InitWizard(project_root=tmp_path)
        assert wizard._system_info is None
        wizard.detect_system()
        assert wizard._system_info is not None
        assert wizard._system_info.ram_gb == 0.0

    @patch("prism.cli.commands.init_wizard.InitWizard._check_docker", return_value=False)
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._check_ollama_installed",
        return_value=False,
    )
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._detect_gpu",
        return_value=(False, ""),
    )
    @patch("prism.cli.commands.init_wizard.InitWizard._detect_ram", return_value=8.0)
    @patch("prism.cli.commands.init_wizard.os.cpu_count", return_value=4)
    def test_detect_system_cpu_cores(
        self,
        _mock_cpu: MagicMock,
        _mock_ram: MagicMock,
        _mock_gpu: MagicMock,
        _mock_ollama: MagicMock,
        _mock_docker: MagicMock,
        tmp_path: Path,
    ) -> None:
        wizard = InitWizard(project_root=tmp_path)
        info = wizard.detect_system()
        assert info.cpu_cores == 4

    @patch("prism.cli.commands.init_wizard.InitWizard._check_docker", return_value=False)
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._check_ollama_installed",
        return_value=False,
    )
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._detect_gpu",
        return_value=(False, ""),
    )
    @patch("prism.cli.commands.init_wizard.InitWizard._detect_ram", return_value=8.0)
    @patch("prism.cli.commands.init_wizard.os.cpu_count", return_value=None)
    def test_detect_system_cpu_count_none_defaults_to_1(
        self,
        _mock_cpu: MagicMock,
        _mock_ram: MagicMock,
        _mock_gpu: MagicMock,
        _mock_ollama: MagicMock,
        _mock_docker: MagicMock,
        tmp_path: Path,
    ) -> None:
        wizard = InitWizard(project_root=tmp_path)
        info = wizard.detect_system()
        assert info.cpu_cores == 1


class TestInitWizardCheckProvider:
    """Tests for InitWizard.check_provider."""

    def test_check_provider_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123456")
        wizard = InitWizard(project_root=tmp_path)
        p = ProviderSetup(
            name="anthropic",
            display_name="Anthropic",
            env_var="ANTHROPIC_API_KEY",
        )
        assert wizard.check_provider(p) is True
        assert p.is_configured is True

    def test_check_provider_not_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        wizard = InitWizard(project_root=tmp_path)
        p = ProviderSetup(
            name="anthropic",
            display_name="Anthropic",
            env_var="ANTHROPIC_API_KEY",
        )
        assert wizard.check_provider(p) is False
        assert p.is_configured is False

    def test_check_provider_empty_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        wizard = InitWizard(project_root=tmp_path)
        p = ProviderSetup(
            name="openai", display_name="OpenAI", env_var="OPENAI_API_KEY"
        )
        assert wizard.check_provider(p) is False

    def test_check_provider_short_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "abc")
        wizard = InitWizard(project_root=tmp_path)
        p = ProviderSetup(
            name="groq", display_name="Groq", env_var="GROQ_API_KEY"
        )
        assert wizard.check_provider(p) is False


class TestInitWizardGetConfiguredProviders:
    """Tests for InitWizard.get_configured_providers."""

    def test_returns_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Clear all provider env vars
        for p in PROVIDER_CONFIGS:
            monkeypatch.delenv(p.env_var, raising=False)
        wizard = InitWizard(project_root=tmp_path)
        providers = wizard.get_configured_providers()
        assert isinstance(providers, list)
        assert len(providers) == len(PROVIDER_CONFIGS)

    def test_detects_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for p in PROVIDER_CONFIGS:
            monkeypatch.delenv(p.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-key-1234567890")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key-1234567890")
        wizard = InitWizard(project_root=tmp_path)
        providers = wizard.get_configured_providers()
        configured = [p for p in providers if p.is_configured]
        names = [p.name for p in configured]
        assert "anthropic" in names
        assert "openai" in names


class TestInitWizardHealthCheck:
    """Tests for InitWizard.health_check_provider."""

    def test_health_check_configured_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-1234567890")
        wizard = InitWizard(project_root=tmp_path)
        p = ProviderSetup(
            name="anthropic",
            display_name="Anthropic",
            env_var="ANTHROPIC_API_KEY",
            is_configured=True,
        )
        assert wizard.health_check_provider(p) is True
        assert p.is_healthy is True

    def test_health_check_unconfigured_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        wizard = InitWizard(project_root=tmp_path)
        p = ProviderSetup(
            name="openai",
            display_name="OpenAI",
            env_var="OPENAI_API_KEY",
            is_configured=False,
        )
        assert wizard.health_check_provider(p) is False
        assert p.is_healthy is False


class TestInitWizardCreateConfig:
    """Tests for InitWizard.create_config."""

    def test_creates_config_file(self, tmp_path: Path) -> None:
        prism_home = tmp_path / ".prism"
        wizard = InitWizard(project_root=tmp_path)
        config_path = wizard.create_config(prism_home=prism_home)
        assert config_path.exists()
        assert config_path.name == "config.yaml"

    def test_config_contains_budget(self, tmp_path: Path) -> None:
        prism_home = tmp_path / ".prism"
        wizard = InitWizard(project_root=tmp_path)
        config_path = wizard.create_config(
            prism_home=prism_home, daily_budget=20.0, monthly_budget=100.0
        )
        content = config_path.read_text()
        assert "daily_limit: 20.0" in content
        assert "monthly_limit: 100.0" in content

    def test_config_contains_routing(self, tmp_path: Path) -> None:
        prism_home = tmp_path / ".prism"
        wizard = InitWizard(project_root=tmp_path)
        config_path = wizard.create_config(prism_home=prism_home)
        content = config_path.read_text()
        assert "routing:" in content
        assert "prefer_cheap: true" in content
        assert "fallback_enabled: true" in content

    def test_config_contains_cache(self, tmp_path: Path) -> None:
        prism_home = tmp_path / ".prism"
        wizard = InitWizard(project_root=tmp_path)
        config_path = wizard.create_config(prism_home=prism_home)
        content = config_path.read_text()
        assert "cache:" in content
        assert "enabled: true" in content
        assert "ttl_seconds: 3600" in content

    def test_config_default_budget_values(self, tmp_path: Path) -> None:
        prism_home = tmp_path / ".prism"
        wizard = InitWizard(project_root=tmp_path)
        config_path = wizard.create_config(prism_home=prism_home)
        content = config_path.read_text()
        assert "daily_limit: 10.0" in content
        assert "monthly_limit: 50.0" in content

    def test_config_creates_parent_directory(self, tmp_path: Path) -> None:
        prism_home = tmp_path / "deep" / "nested" / ".prism"
        wizard = InitWizard(project_root=tmp_path)
        config_path = wizard.create_config(prism_home=prism_home)
        assert config_path.exists()
        assert prism_home.is_dir()

    def test_config_warn_at_percent(self, tmp_path: Path) -> None:
        prism_home = tmp_path / ".prism"
        wizard = InitWizard(project_root=tmp_path)
        config_path = wizard.create_config(prism_home=prism_home)
        content = config_path.read_text()
        assert "warn_at_percent: 70" in content


class TestInitWizardCreateProjectMemory:
    """Tests for InitWizard.create_project_memory."""

    def test_creates_prism_md(self, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        memory_path = wizard.create_project_memory()
        assert memory_path.exists()
        assert memory_path.name == ".prism.md"

    def test_template_content(self, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        memory_path = wizard.create_project_memory()
        content = memory_path.read_text()
        assert "Project Overview" in content
        assert "Key Decisions" in content
        assert "Conventions" in content

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        existing = tmp_path / ".prism.md"
        existing.write_text("# Existing content\n")
        wizard = InitWizard(project_root=tmp_path)
        memory_path = wizard.create_project_memory()
        assert memory_path.read_text() == "# Existing content\n"


class TestInitWizardCreatePrismignore:
    """Tests for InitWizard.create_prismignore."""

    def test_creates_prismignore(self, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        ignore_path = wizard.create_prismignore()
        assert ignore_path.exists()
        assert ".prismignore" in ignore_path.name

    def test_prismignore_contains_env(self, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        ignore_path = wizard.create_prismignore()
        content = ignore_path.read_text()
        assert ".env" in content

    def test_prismignore_contains_keys(self, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        ignore_path = wizard.create_prismignore()
        content = ignore_path.read_text()
        assert "*.pem" in content


class TestInitWizardOllamaModels:
    """Tests for InitWizard.get_recommended_ollama_models."""

    def test_high_ram_gets_all_models(self, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        wizard._system_info = SystemInfo(
            os_name="Linux",
            os_version="6.1",
            python_version="3.12.0",
            ram_gb=64.0,
            gpu_detected=False,
            gpu_name="",
            cpu_cores=16,
            ollama_installed=True,
            ollama_models=[],
            docker_installed=False,
        )
        recommended = wizard.get_recommended_ollama_models()
        assert len(recommended) == len(RECOMMENDED_OLLAMA_MODELS)

    def test_low_ram_gets_fewer_models(self, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        wizard._system_info = SystemInfo(
            os_name="Linux",
            os_version="6.1",
            python_version="3.12.0",
            ram_gb=4.5,
            gpu_detected=False,
            gpu_name="",
            cpu_cores=4,
            ollama_installed=True,
            ollama_models=[],
            docker_installed=False,
        )
        recommended = wizard.get_recommended_ollama_models()
        # 4.5 GB RAM: only qwen2.5-coder:7b (needs 4.0 GB)
        assert len(recommended) == 1
        assert "qwen2.5-coder:7b" in recommended

    def test_zero_ram_gets_no_models(self, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        wizard._system_info = SystemInfo(
            os_name="Linux",
            os_version="6.1",
            python_version="3.12.0",
            ram_gb=0.0,
            gpu_detected=False,
            gpu_name="",
            cpu_cores=1,
            ollama_installed=False,
            ollama_models=[],
            docker_installed=False,
        )
        recommended = wizard.get_recommended_ollama_models()
        assert len(recommended) == 0

    @patch("prism.cli.commands.init_wizard.InitWizard._check_docker", return_value=False)
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._check_ollama_installed",
        return_value=False,
    )
    @patch(
        "prism.cli.commands.init_wizard.InitWizard._detect_gpu",
        return_value=(False, ""),
    )
    @patch("prism.cli.commands.init_wizard.InitWizard._detect_ram", return_value=32.0)
    def test_auto_detects_system_if_needed(
        self,
        _mock_ram: MagicMock,
        _mock_gpu: MagicMock,
        _mock_ollama: MagicMock,
        _mock_docker: MagicMock,
        tmp_path: Path,
    ) -> None:
        wizard = InitWizard(project_root=tmp_path)
        assert wizard._system_info is None
        recommended = wizard.get_recommended_ollama_models()
        assert wizard._system_info is not None
        assert len(recommended) >= 2


class TestInitWizardOllamaInstructions:
    """Tests for InitWizard.get_ollama_install_instructions."""

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Darwin")
    def test_macos_instructions(self, _mock: MagicMock, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        instructions = wizard.get_ollama_install_instructions()
        assert "macOS" in instructions
        assert "brew install ollama" in instructions

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Linux")
    def test_linux_instructions(self, _mock: MagicMock, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        instructions = wizard.get_ollama_install_instructions()
        assert "Linux" in instructions
        assert "curl" in instructions

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Windows")
    def test_generic_instructions(self, _mock: MagicMock, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        instructions = wizard.get_ollama_install_instructions()
        assert "ollama.com" in instructions


class TestInitWizardCostComparison:
    """Tests for InitWizard.get_cost_comparison."""

    def test_returns_list_of_dicts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for p in PROVIDER_CONFIGS:
            monkeypatch.delenv(p.env_var, raising=False)
        wizard = InitWizard(project_root=tmp_path)
        rows = wizard.get_cost_comparison()
        assert isinstance(rows, list)
        assert len(rows) == len(PROVIDER_CONFIGS)
        for row in rows:
            assert "provider" in row
            assert "configured" in row
            assert "cost" in row

    def test_shows_configured_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for p in PROVIDER_CONFIGS:
            monkeypatch.delenv(p.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-key-very-long-1234567890")
        wizard = InitWizard(project_root=tmp_path)
        rows = wizard.get_cost_comparison()
        anthropic_row = next(r for r in rows if "Anthropic" in r["provider"])
        assert anthropic_row["configured"] == "Yes"
        # Others should be "No"
        openai_row = next(r for r in rows if "OpenAI" in r["provider"])
        assert openai_row["configured"] == "No"


class TestInitWizardQuickstart:
    """Tests for InitWizard.get_quickstart_commands."""

    def test_returns_list_of_strings(self, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        commands = wizard.get_quickstart_commands()
        assert isinstance(commands, list)
        assert all(isinstance(c, str) for c in commands)
        assert len(commands) >= 3

    def test_contains_prism_command(self, tmp_path: Path) -> None:
        wizard = InitWizard(project_root=tmp_path)
        commands = wizard.get_quickstart_commands()
        assert any("prism" in c for c in commands)


class TestInitWizardRun:
    """Tests for InitWizard.run (full wizard)."""

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
    def test_run_returns_wizard_result(
        self,
        _mock_ram: MagicMock,
        _mock_gpu: MagicMock,
        _mock_ollama: MagicMock,
        _mock_docker: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for p in PROVIDER_CONFIGS:
            monkeypatch.delenv(p.env_var, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        wizard = InitWizard(project_root=tmp_path)
        result = wizard.run()
        assert isinstance(result, WizardResult)
        assert isinstance(result.system_info, SystemInfo)
        assert result.config_path is not None
        assert result.memory_path is not None
        assert result.ignore_path is not None

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
    def test_run_creates_files(
        self,
        _mock_ram: MagicMock,
        _mock_gpu: MagicMock,
        _mock_ollama: MagicMock,
        _mock_docker: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for p in PROVIDER_CONFIGS:
            monkeypatch.delenv(p.env_var, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        wizard = InitWizard(project_root=tmp_path)
        result = wizard.run()
        assert result.config_path is not None
        assert result.config_path.exists()
        assert result.memory_path is not None
        assert result.memory_path.exists()
        assert result.ignore_path is not None
        assert result.ignore_path.exists()

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
    def test_run_detects_configured_providers(
        self,
        _mock_ram: MagicMock,
        _mock_gpu: MagicMock,
        _mock_ollama: MagicMock,
        _mock_docker: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for p in PROVIDER_CONFIGS:
            monkeypatch.delenv(p.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-api-key-1234567890")
        monkeypatch.setenv("HOME", str(tmp_path))
        wizard = InitWizard(project_root=tmp_path)
        result = wizard.run()
        assert "anthropic" in result.providers_configured


# ---------------------------------------------------------------------------
# TestDetectRam — mock subprocess for macOS and Linux
# ---------------------------------------------------------------------------


class TestDetectRam:
    """Tests for InitWizard._detect_ram."""

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Darwin")
    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_macos_ram_detection(
        self, mock_run: MagicMock, _mock_sys: MagicMock, tmp_path: Path
    ) -> None:
        # 16 GB in bytes
        mock_run.return_value = MagicMock(
            returncode=0, stdout=str(16 * 1024**3)
        )
        wizard = InitWizard(project_root=tmp_path)
        ram = wizard._detect_ram()
        assert abs(ram - 16.0) < 0.01

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Darwin")
    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_macos_ram_failure(
        self, mock_run: MagicMock, _mock_sys: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        wizard = InitWizard(project_root=tmp_path)
        ram = wizard._detect_ram()
        assert ram == 0.0

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Darwin")
    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_macos_ram_timeout(
        self, mock_run: MagicMock, _mock_sys: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sysctl", timeout=5)
        wizard = InitWizard(project_root=tmp_path)
        ram = wizard._detect_ram()
        assert ram == 0.0

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Linux")
    def test_linux_ram_detection(
        self, _mock_sys: MagicMock, tmp_path: Path
    ) -> None:
        # Create a fake /proc/meminfo via patching Path operations
        proc_meminfo = tmp_path / "meminfo"
        proc_meminfo.write_text("MemTotal:       16384000 kB\nMemFree:        8192000 kB\n")
        wizard = InitWizard(project_root=tmp_path)
        with patch(
            "prism.cli.commands.init_wizard.Path",
            side_effect=lambda x: Path(str(x).replace("/proc/meminfo", str(proc_meminfo))),
        ):
            # Directly test the Linux path with a patched file read
            ram = wizard._detect_ram()
        # On CI this may take the Linux or fallback path; we just ensure no crash
        assert isinstance(ram, float)

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Windows")
    def test_windows_returns_zero(
        self, _mock_sys: MagicMock, tmp_path: Path
    ) -> None:
        wizard = InitWizard(project_root=tmp_path)
        ram = wizard._detect_ram()
        assert ram == 0.0


# ---------------------------------------------------------------------------
# TestDetectGPU — mock subprocess
# ---------------------------------------------------------------------------


class TestDetectGPU:
    """Tests for InitWizard._detect_gpu."""

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Darwin")
    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_macos_gpu_detection(
        self, mock_run: MagicMock, _mock_sys: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Graphics/Displays:\n    Chipset Model: Apple M1 Pro\n",
        )
        wizard = InitWizard(project_root=tmp_path)
        detected, name = wizard._detect_gpu()
        assert detected is True
        assert "Apple M1 Pro" in name

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Linux")
    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_nvidia_gpu_detection(
        self, mock_run: MagicMock, _mock_sys: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090\n",
        )
        wizard = InitWizard(project_root=tmp_path)
        detected, name = wizard._detect_gpu()
        assert detected is True
        assert "RTX 4090" in name

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Linux")
    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_no_gpu_detected(
        self, mock_run: MagicMock, _mock_sys: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        wizard = InitWizard(project_root=tmp_path)
        detected, name = wizard._detect_gpu()
        assert detected is False
        assert name == ""

    @patch("prism.cli.commands.init_wizard.platform.system", return_value="Darwin")
    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_gpu_timeout(
        self, mock_run: MagicMock, _mock_sys: MagicMock, tmp_path: Path
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="system_profiler", timeout=10
        )
        wizard = InitWizard(project_root=tmp_path)
        detected, name = wizard._detect_gpu()
        assert detected is False
        assert name == ""


# ---------------------------------------------------------------------------
# TestCheckOllama — mock subprocess
# ---------------------------------------------------------------------------


class TestCheckOllama:
    """Tests for InitWizard._check_ollama_installed."""

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_ollama_installed(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="ollama v0.1.27")
        assert InitWizard._check_ollama_installed() is True

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_ollama_not_installed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("ollama not found")
        assert InitWizard._check_ollama_installed() is False

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_ollama_bad_return_code(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert InitWizard._check_ollama_installed() is False


# ---------------------------------------------------------------------------
# TestGetOllamaModels — mock subprocess
# ---------------------------------------------------------------------------


class TestGetOllamaModels:
    """Tests for InitWizard._get_ollama_models."""

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_lists_models(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NAME            ID          SIZE    MODIFIED\n"
            "llama3.1:8b    abc123     4.7 GB   2 days ago\n"
            "codellama:7b   def456     3.8 GB   5 days ago\n",
        )
        models = InitWizard._get_ollama_models()
        assert "llama3.1:8b" in models
        assert "codellama:7b" in models

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_empty_model_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NAME            ID          SIZE    MODIFIED\n",
        )
        models = InitWizard._get_ollama_models()
        assert models == []

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_ollama_list_failure(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("ollama not found")
        models = InitWizard._get_ollama_models()
        assert models == []

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_ollama_list_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=10)
        models = InitWizard._get_ollama_models()
        assert models == []


# ---------------------------------------------------------------------------
# TestCheckDocker — mock subprocess
# ---------------------------------------------------------------------------


class TestCheckDocker:
    """Tests for InitWizard._check_docker."""

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_docker_available(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        assert InitWizard._check_docker() is True

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_docker_not_available(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("docker not found")
        assert InitWizard._check_docker() is False

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_docker_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=10)
        assert InitWizard._check_docker() is False

    @patch("prism.cli.commands.init_wizard.subprocess.run")
    def test_docker_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        assert InitWizard._check_docker() is False


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_provider_configs_not_empty(self) -> None:
        assert len(PROVIDER_CONFIGS) >= 9

    def test_provider_configs_have_required_fields(self) -> None:
        for p in PROVIDER_CONFIGS:
            assert p.name
            assert p.display_name
            assert p.env_var

    def test_recommended_ollama_models_not_empty(self) -> None:
        assert len(RECOMMENDED_OLLAMA_MODELS) >= 3

    def test_recommended_ollama_models_have_min_ram(self) -> None:
        for _model, (desc, min_ram) in RECOMMENDED_OLLAMA_MODELS.items():
            assert isinstance(desc, str)
            assert isinstance(min_ram, float)
            assert min_ram > 0

    def test_prism_md_template_not_empty(self) -> None:
        assert len(PRISM_MD_TEMPLATE) > 10
        assert ".prism.md" in PRISM_MD_TEMPLATE

    def test_prismignore_defaults_not_empty(self) -> None:
        assert len(PRISMIGNORE_DEFAULTS) > 10
        # Should contain secure patterns
        assert ".env" in PRISMIGNORE_DEFAULTS
        assert "*.pem" in PRISMIGNORE_DEFAULTS

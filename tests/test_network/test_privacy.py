"""Tests for prism.network.privacy — Ollama-only privacy mode."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from prism.network.privacy import (
    CLOUD_PROVIDERS,
    RECOMMENDED_MODELS,
    OllamaModel,
    PrivacyLevel,
    PrivacyManager,
    PrivacyStatus,
    PrivacyViolationError,
)

# =====================================================================
# PrivacyLevel enum
# =====================================================================


class TestPrivacyLevel:
    """Verify PrivacyLevel enum values."""

    def test_normal_value(self) -> None:
        assert PrivacyLevel.NORMAL.value == "normal"

    def test_private_value(self) -> None:
        assert PrivacyLevel.PRIVATE.value == "private"

    def test_all_members(self) -> None:
        assert set(PrivacyLevel) == {PrivacyLevel.NORMAL, PrivacyLevel.PRIVATE}


# =====================================================================
# OllamaModel dataclass
# =====================================================================


class TestOllamaModel:
    """Verify OllamaModel fields."""

    def test_fields(self) -> None:
        model = OllamaModel(
            name="llama3.1:8b",
            size_bytes=4_800_000_000,
            modified_at="2024-01-15",
            digest="abc123",
        )
        assert model.name == "llama3.1:8b"
        assert model.size_bytes == 4_800_000_000
        assert model.modified_at == "2024-01-15"
        assert model.digest == "abc123"

    def test_equality(self) -> None:
        m1 = OllamaModel(name="a", size_bytes=0, modified_at="", digest="x")
        m2 = OllamaModel(name="a", size_bytes=0, modified_at="", digest="x")
        assert m1 == m2


# =====================================================================
# PrivacyStatus dataclass
# =====================================================================


class TestPrivacyStatus:
    """Verify PrivacyStatus fields and defaults."""

    def test_fields(self) -> None:
        status = PrivacyStatus(
            level=PrivacyLevel.PRIVATE,
            ollama_running=True,
            available_models=[
                OllamaModel(name="m", size_bytes=0, modified_at="", digest="")
            ],
            active_model="m",
        )
        assert status.level == PrivacyLevel.PRIVATE
        assert status.ollama_running is True
        assert len(status.available_models) == 1
        assert status.active_model == "m"

    def test_default_empty_models(self) -> None:
        status = PrivacyStatus(level=PrivacyLevel.NORMAL, ollama_running=False)
        assert status.available_models == []
        assert status.active_model == ""


# =====================================================================
# PrivacyViolationError
# =====================================================================


class TestPrivacyViolationError:
    """Verify exception behaviour."""

    def test_is_exception(self) -> None:
        err = PrivacyViolationError("blocked")
        assert isinstance(err, Exception)

    def test_message_preserved(self) -> None:
        err = PrivacyViolationError("Cloud provider 'openai' blocked")
        assert "openai" in str(err)
        assert "blocked" in str(err)

    def test_raised_and_caught(self) -> None:
        with pytest.raises(PrivacyViolationError, match="blocked"):
            raise PrivacyViolationError("request blocked")


# =====================================================================
# PrivacyManager — init and properties
# =====================================================================


class TestPrivacyManagerInit:
    """Verify initial state of PrivacyManager."""

    def test_default_level_is_normal(self) -> None:
        pm = PrivacyManager()
        assert pm.level == PrivacyLevel.NORMAL

    def test_is_private_false_initially(self) -> None:
        pm = PrivacyManager()
        assert pm.is_private is False


# =====================================================================
# PrivacyManager — enable/disable
# =====================================================================


class TestPrivacyModeToggle:
    """Verify mode switching."""

    def test_enable_sets_private(self) -> None:
        pm = PrivacyManager()
        with patch.object(pm, "check_ollama", return_value=True):
            with patch.object(pm, "list_models", return_value=[]):
                status = pm.enable_private_mode()

        assert pm.is_private is True
        assert pm.level == PrivacyLevel.PRIVATE
        assert status.level == PrivacyLevel.PRIVATE

    def test_enable_starts_ollama_if_not_running(self) -> None:
        pm = PrivacyManager()
        with patch.object(pm, "check_ollama", side_effect=[False, True]):
            with patch.object(pm, "start_ollama", return_value=True) as mock_start:
                with patch.object(pm, "list_models", return_value=[]):
                    pm.enable_private_mode()

        mock_start.assert_called_once()

    def test_enable_handles_ollama_unavailable(self) -> None:
        pm = PrivacyManager()
        with patch.object(pm, "check_ollama", return_value=False):
            with patch.object(pm, "start_ollama", return_value=False):
                with patch.object(pm, "list_models", return_value=[]):
                    status = pm.enable_private_mode()

        # Still enables private mode, just with warning
        assert pm.is_private is True
        assert status.ollama_running is False

    def test_disable_restores_normal(self) -> None:
        pm = PrivacyManager()
        with patch.object(pm, "check_ollama", return_value=True):
            with patch.object(pm, "list_models", return_value=[]):
                pm.enable_private_mode()

        pm.disable_private_mode()
        assert pm.is_private is False
        assert pm.level == PrivacyLevel.NORMAL


# =====================================================================
# PrivacyManager — Ollama lifecycle
# =====================================================================


class TestOllamaLifecycle:
    """Verify Ollama check/start/list/pull operations (all mocked)."""

    def test_check_ollama_running(self) -> None:
        pm = PrivacyManager()
        with patch("prism.network.privacy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert pm.check_ollama() is True

    def test_check_ollama_not_running(self) -> None:
        pm = PrivacyManager()
        with patch("prism.network.privacy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert pm.check_ollama() is False

    def test_check_ollama_not_installed(self) -> None:
        pm = PrivacyManager()
        with patch(
            "prism.network.privacy.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert pm.check_ollama() is False

    def test_check_ollama_timeout(self) -> None:
        pm = PrivacyManager()
        with patch(
            "prism.network.privacy.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ollama", timeout=10),
        ):
            assert pm.check_ollama() is False

    def test_start_ollama_success(self) -> None:
        pm = PrivacyManager()
        with patch("prism.network.privacy.subprocess.Popen") as mock_popen:
            with patch("prism.network.privacy.time.sleep"):
                with patch.object(pm, "check_ollama", return_value=True):
                    assert pm.start_ollama() is True

        mock_popen.assert_called_once()

    def test_start_ollama_failure(self) -> None:
        pm = PrivacyManager()
        with patch(
            "prism.network.privacy.subprocess.Popen",
            side_effect=FileNotFoundError,
        ):
            assert pm.start_ollama() is False

    def test_start_ollama_os_error(self) -> None:
        pm = PrivacyManager()
        with patch(
            "prism.network.privacy.subprocess.Popen",
            side_effect=OSError("Permission denied"),
        ):
            assert pm.start_ollama() is False

    def test_list_models_parses_output(self) -> None:
        pm = PrivacyManager()
        ollama_output = (
            "NAME                    ID              SIZE    MODIFIED\n"
            "llama3.1:8b             abc123def456    4.7     GB      2 days ago\n"
            "qwen2.5-coder:7b       def789abc012    4.1     GB      5 days ago\n"
        )
        with patch("prism.network.privacy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=ollama_output
            )
            models = pm.list_models()

        assert len(models) == 2
        assert models[0].name == "llama3.1:8b"
        assert models[0].digest == "abc123def456"
        assert models[1].name == "qwen2.5-coder:7b"

    def test_list_models_empty(self) -> None:
        pm = PrivacyManager()
        with patch("prism.network.privacy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="NAME  ID  SIZE  MODIFIED\n",
            )
            models = pm.list_models()

        assert models == []

    def test_list_models_command_fails(self) -> None:
        pm = PrivacyManager()
        with patch("prism.network.privacy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert pm.list_models() == []

    def test_list_models_not_installed(self) -> None:
        pm = PrivacyManager()
        with patch(
            "prism.network.privacy.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert pm.list_models() == []

    def test_list_models_timeout(self) -> None:
        pm = PrivacyManager()
        with patch(
            "prism.network.privacy.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ollama", timeout=15),
        ):
            assert pm.list_models() == []

    def test_pull_model_success(self) -> None:
        pm = PrivacyManager()
        with patch("prism.network.privacy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert pm.pull_model("llama3.1:8b") is True

    def test_pull_model_failure(self) -> None:
        pm = PrivacyManager()
        with patch("prism.network.privacy.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="not found")
            assert pm.pull_model("nonexistent:latest") is False

    def test_pull_model_not_installed(self) -> None:
        pm = PrivacyManager()
        with patch(
            "prism.network.privacy.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert pm.pull_model("llama3.1:8b") is False

    def test_pull_model_timeout(self) -> None:
        pm = PrivacyManager()
        with patch(
            "prism.network.privacy.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ollama", timeout=600),
        ):
            assert pm.pull_model("llama3.1:8b") is False

    def test_pull_model_empty_name(self) -> None:
        pm = PrivacyManager()
        assert pm.pull_model("") is False
        assert pm.pull_model("   ") is False


# =====================================================================
# PrivacyManager — cloud provider detection
# =====================================================================


class TestCloudProviderDetection:
    """Verify is_cloud_provider for known cloud providers."""

    @pytest.mark.parametrize("provider", list(CLOUD_PROVIDERS))
    def test_known_cloud_providers(self, provider: str) -> None:
        pm = PrivacyManager()
        assert pm.is_cloud_provider(provider) is True

    def test_ollama_is_not_cloud(self) -> None:
        pm = PrivacyManager()
        assert pm.is_cloud_provider("ollama") is False

    def test_case_insensitive(self) -> None:
        pm = PrivacyManager()
        assert pm.is_cloud_provider("OpenAI") is True
        assert pm.is_cloud_provider("ANTHROPIC") is True
        assert pm.is_cloud_provider("Google") is True

    def test_whitespace_handling(self) -> None:
        pm = PrivacyManager()
        assert pm.is_cloud_provider("  openai  ") is True

    def test_unknown_provider(self) -> None:
        pm = PrivacyManager()
        assert pm.is_cloud_provider("unknown_local_provider") is False

    def test_empty_string(self) -> None:
        pm = PrivacyManager()
        assert pm.is_cloud_provider("") is False


# =====================================================================
# PrivacyManager — validate_request
# =====================================================================


class TestValidateRequest:
    """Verify request validation against privacy mode."""

    def test_normal_mode_allows_cloud(self) -> None:
        pm = PrivacyManager()
        # Should not raise
        pm.validate_request("openai", "gpt-4o")
        pm.validate_request("anthropic", "claude-3-opus")

    def test_private_mode_blocks_cloud_anthropic(self) -> None:
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        with pytest.raises(PrivacyViolationError, match="anthropic"):
            pm.validate_request("anthropic", "claude-3-opus")

    def test_private_mode_blocks_cloud_openai(self) -> None:
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        with pytest.raises(PrivacyViolationError, match="openai"):
            pm.validate_request("openai", "gpt-4o")

    def test_private_mode_blocks_cloud_google(self) -> None:
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        with pytest.raises(PrivacyViolationError, match="google"):
            pm.validate_request("google", "gemini-pro")

    def test_private_mode_blocks_cloud_deepseek(self) -> None:
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        with pytest.raises(PrivacyViolationError, match="deepseek"):
            pm.validate_request("deepseek", "deepseek-chat")

    def test_private_mode_blocks_cloud_groq(self) -> None:
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        with pytest.raises(PrivacyViolationError, match="groq"):
            pm.validate_request("groq", "llama-3-70b")

    def test_private_mode_allows_ollama(self) -> None:
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        # Should not raise
        pm.validate_request("ollama", "llama3.1:8b")

    def test_private_mode_allows_ollama_prefixed_model(self) -> None:
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        # Provider is something else but model has ollama/ prefix
        pm.validate_request("ollama", "ollama/llama3.1:8b")

    def test_private_mode_blocks_non_ollama_local_provider(self) -> None:
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        # Provider is not ollama and model doesn't have ollama/ prefix
        with pytest.raises(PrivacyViolationError, match="not routed through Ollama"):
            pm.validate_request("custom_local", "some-model")

    def test_error_message_mentions_private_mode(self) -> None:
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        with pytest.raises(PrivacyViolationError) as exc_info:
            pm.validate_request("openai", "gpt-4o")

        assert "private mode" in str(exc_info.value).lower()


# =====================================================================
# PrivacyManager — get_status
# =====================================================================


class TestGetStatus:
    """Verify get_status returns correct snapshots."""

    def test_status_reflects_level(self) -> None:
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        with patch.object(pm, "check_ollama", return_value=False):
            status = pm.get_status()

        assert status.level == PrivacyLevel.PRIVATE

    def test_status_shows_ollama_running(self) -> None:
        pm = PrivacyManager()
        with patch.object(pm, "check_ollama", return_value=True):
            with patch.object(pm, "list_models", return_value=[]):
                status = pm.get_status()

        assert status.ollama_running is True

    def test_status_shows_ollama_not_running(self) -> None:
        pm = PrivacyManager()
        with patch.object(pm, "check_ollama", return_value=False):
            status = pm.get_status()

        assert status.ollama_running is False
        assert status.available_models == []

    def test_status_includes_models(self) -> None:
        pm = PrivacyManager()
        mock_models = [
            OllamaModel(
                name="llama3.1:8b", size_bytes=4_800_000_000,
                modified_at="2024-01-15", digest="abc",
            )
        ]
        with patch.object(pm, "check_ollama", return_value=True):
            with patch.object(pm, "list_models", return_value=mock_models):
                status = pm.get_status()

        assert len(status.available_models) == 1
        assert status.available_models[0].name == "llama3.1:8b"


# =====================================================================
# PrivacyManager — get_recommended_model
# =====================================================================


class TestGetRecommendedModel:
    """Verify recommended model selection."""

    def test_returns_first_installed_recommended(self) -> None:
        pm = PrivacyManager()
        mock_models = [
            OllamaModel(
                name="llama3.1:8b", size_bytes=0, modified_at="", digest=""
            ),
        ]
        with patch.object(pm, "list_models", return_value=mock_models):
            model = pm.get_recommended_model()

        # llama3.1:8b is in RECOMMENDED_MODELS
        assert model == "llama3.1:8b"

    def test_prefers_earlier_recommended(self) -> None:
        pm = PrivacyManager()
        # Install both first and second recommended
        mock_models = [
            OllamaModel(
                name="qwen2.5-coder:7b", size_bytes=0, modified_at="", digest=""
            ),
            OllamaModel(
                name="llama3.1:8b", size_bytes=0, modified_at="", digest=""
            ),
        ]
        with patch.object(pm, "list_models", return_value=mock_models):
            model = pm.get_recommended_model()

        # qwen2.5-coder:7b is listed first in RECOMMENDED_MODELS
        assert model == "qwen2.5-coder:7b"

    def test_falls_back_to_default(self) -> None:
        pm = PrivacyManager()
        with patch.object(pm, "list_models", return_value=[]):
            model = pm.get_recommended_model()

        assert model == "llama3.1:8b"

    def test_non_recommended_model_not_selected(self) -> None:
        pm = PrivacyManager()
        mock_models = [
            OllamaModel(
                name="custom-model:latest", size_bytes=0, modified_at="", digest=""
            ),
        ]
        with patch.object(pm, "list_models", return_value=mock_models):
            model = pm.get_recommended_model()

        # Falls back because custom-model is not recommended
        assert model == "llama3.1:8b"


# =====================================================================
# PrivacyManager — _parse_size
# =====================================================================


class TestParseSize:
    """Verify size string parsing."""

    def test_gigabytes(self) -> None:
        assert PrivacyManager._parse_size("4.1", "GB") == int(4.1 * 1_073_741_824)

    def test_megabytes(self) -> None:
        assert PrivacyManager._parse_size("512", "MB") == (512 * 1_048_576)

    def test_kilobytes(self) -> None:
        assert PrivacyManager._parse_size("100", "KB") == (100 * 1024)

    def test_bytes_no_unit(self) -> None:
        assert PrivacyManager._parse_size("1024", "") == 1024

    def test_case_insensitive_unit(self) -> None:
        assert PrivacyManager._parse_size("2.0", "gb") == int(2.0 * 1_073_741_824)

    def test_invalid_value(self) -> None:
        assert PrivacyManager._parse_size("not_a_number", "GB") == 0

    def test_empty_value(self) -> None:
        assert PrivacyManager._parse_size("", "GB") == 0


# =====================================================================
# Constants
# =====================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_recommended_models_not_empty(self) -> None:
        assert len(RECOMMENDED_MODELS) >= 3

    def test_recommended_models_have_descriptions(self) -> None:
        for model, desc in RECOMMENDED_MODELS.items():
            assert model, "Model name must not be empty"
            assert desc, f"Description for {model} must not be empty"

    def test_cloud_providers_not_empty(self) -> None:
        assert len(CLOUD_PROVIDERS) >= 5

    def test_cloud_providers_contains_major(self) -> None:
        assert "anthropic" in CLOUD_PROVIDERS
        assert "openai" in CLOUD_PROVIDERS
        assert "google" in CLOUD_PROVIDERS

    def test_cloud_providers_does_not_contain_ollama(self) -> None:
        assert "ollama" not in CLOUD_PROVIDERS

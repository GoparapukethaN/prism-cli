"""Tests for prism.security.secret_filter.SecretFilter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prism.security.secret_filter import REDACTED, SecretFilter

if TYPE_CHECKING:
    import pytest


class TestIsSensitive:
    """Verify pattern matching on individual keys."""

    def test_api_key_pattern(self, secret_filter: SecretFilter) -> None:
        assert secret_filter.is_sensitive("OPENAI_API_KEY") is True
        assert secret_filter.is_sensitive("MY_SERVICE_API_KEY") is True

    def test_secret_pattern(self, secret_filter: SecretFilter) -> None:
        assert secret_filter.is_sensitive("AWS_SECRET") is True

    def test_token_pattern(self, secret_filter: SecretFilter) -> None:
        assert secret_filter.is_sensitive("GITHUB_TOKEN") is True
        assert secret_filter.is_sensitive("AUTH_TOKEN") is True

    def test_password_pattern(self, secret_filter: SecretFilter) -> None:
        assert secret_filter.is_sensitive("DB_PASSWORD") is True
        assert secret_filter.is_sensitive("MY_APP_PASSWORD") is True

    def test_credential_pattern(self, secret_filter: SecretFilter) -> None:
        assert secret_filter.is_sensitive("GCP_CREDENTIAL") is True

    def test_private_key_pattern(self, secret_filter: SecretFilter) -> None:
        assert secret_filter.is_sensitive("SSH_PRIVATE_KEY") is True

    def test_database_url(self, secret_filter: SecretFilter) -> None:
        assert secret_filter.is_sensitive("DATABASE_URL") is True

    def test_redis_url(self, secret_filter: SecretFilter) -> None:
        assert secret_filter.is_sensitive("REDIS_URL") is True

    def test_non_sensitive_keys(self, secret_filter: SecretFilter) -> None:
        assert secret_filter.is_sensitive("HOME") is False
        assert secret_filter.is_sensitive("PATH") is False
        assert secret_filter.is_sensitive("LANG") is False
        assert secret_filter.is_sensitive("EDITOR") is False
        assert secret_filter.is_sensitive("PRISM_HOME") is False
        assert secret_filter.is_sensitive("TERM") is False

    def test_case_insensitive_matching(self, secret_filter: SecretFilter) -> None:
        """Patterns should match regardless of case."""
        assert secret_filter.is_sensitive("openai_api_key") is True
        assert secret_filter.is_sensitive("Aws_Secret") is True
        assert secret_filter.is_sensitive("github_token") is True


class TestFilterEnv:
    """Verify that filter_env strips sensitive keys entirely."""

    def test_removes_sensitive_keys(
        self, secret_filter: SecretFilter, env_with_secrets: dict[str, str]
    ) -> None:
        result = secret_filter.filter_env(env_with_secrets)

        # Sensitive keys must be absent
        assert "OPENAI_API_KEY" not in result
        assert "AWS_SECRET" not in result
        assert "GITHUB_TOKEN" not in result
        assert "DATABASE_URL" not in result
        assert "MY_APP_PASSWORD" not in result
        assert "REDIS_URL" not in result

        # Non-sensitive keys must remain
        assert result["HOME"] == "/home/user"
        assert result["PATH"] == "/usr/bin:/bin"
        assert result["PRISM_HOME"] == "/tmp/prism"
        assert result["EDITOR"] == "vim"
        assert result["LANG"] == "en_US.UTF-8"

    def test_empty_env(self, secret_filter: SecretFilter) -> None:
        result = secret_filter.filter_env({})
        assert result == {}

    def test_all_sensitive(self, secret_filter: SecretFilter) -> None:
        env = {
            "OPENAI_API_KEY": "sk-123",
            "MY_SECRET": "abc",
            "AUTH_TOKEN": "tok",
        }
        result = secret_filter.filter_env(env)
        assert result == {}

    def test_no_sensitive(self, secret_filter: SecretFilter) -> None:
        env = {"HOME": "/home", "SHELL": "/bin/bash"}
        result = secret_filter.filter_env(env)
        assert result == env

    def test_none_defaults_to_os_environ(
        self, secret_filter: SecretFilter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", "/test/home")
        monkeypatch.setenv("MY_API_KEY", "super-secret")
        result = secret_filter.filter_env(None)
        assert "HOME" in result
        assert "MY_API_KEY" not in result


class TestRedactEnv:
    """Verify that redact_env keeps keys but masks values."""

    def test_redacts_values(
        self, secret_filter: SecretFilter, env_with_secrets: dict[str, str]
    ) -> None:
        result = secret_filter.redact_env(env_with_secrets)

        assert result["OPENAI_API_KEY"] == REDACTED
        assert result["AWS_SECRET"] == REDACTED
        assert result["GITHUB_TOKEN"] == REDACTED
        assert result["DATABASE_URL"] == REDACTED
        assert result["MY_APP_PASSWORD"] == REDACTED
        assert result["REDIS_URL"] == REDACTED

        # Non-sensitive values must remain unchanged
        assert result["HOME"] == "/home/user"
        assert result["PATH"] == "/usr/bin:/bin"


class TestSanitizeDict:
    """Verify recursive dict sanitization."""

    def test_flat_dict(self, secret_filter: SecretFilter) -> None:
        data = {
            "openai_api_key": "sk-secret",
            "model": "gpt-4",
        }
        result = secret_filter.sanitize_dict(data)
        assert result["openai_api_key"] == REDACTED
        assert result["model"] == "gpt-4"

    def test_nested_dict(self, secret_filter: SecretFilter) -> None:
        data = {
            "provider": {
                "name": "openai",
                "auth_token": "tok-123",
                "endpoint": "https://api.openai.com",
            },
            "count": 5,
        }
        result = secret_filter.sanitize_dict(data)
        assert result["provider"]["auth_token"] == REDACTED  # type: ignore[index]
        assert result["provider"]["name"] == "openai"  # type: ignore[index]
        assert result["provider"]["endpoint"] == "https://api.openai.com"  # type: ignore[index]
        assert result["count"] == 5

    def test_non_string_values_are_preserved(
        self, secret_filter: SecretFilter
    ) -> None:
        """Non-string values under sensitive keys are NOT redacted (only strings)."""
        data = {
            "my_api_key": 12345,
            "status": True,
        }
        result = secret_filter.sanitize_dict(data)
        # Integer under sensitive key stays (only strings get redacted)
        assert result["my_api_key"] == 12345
        assert result["status"] is True


class TestExtraPatterns:
    """Verify that extra_patterns extend the default set."""

    def test_extra_pattern_is_applied(self) -> None:
        sf = SecretFilter(extra_patterns=["*_CUSTOM_CRED"])
        assert sf.is_sensitive("MY_CUSTOM_CRED") is True
        # Default patterns still work
        assert sf.is_sensitive("OPENAI_API_KEY") is True

    def test_extra_pattern_in_filter_env(self) -> None:
        sf = SecretFilter(extra_patterns=["*_CUSTOM_CRED"])
        env = {
            "FOO_CUSTOM_CRED": "secret",
            "HOME": "/home/user",
        }
        result = sf.filter_env(env)
        assert "FOO_CUSTOM_CRED" not in result
        assert result["HOME"] == "/home/user"

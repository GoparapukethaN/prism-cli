"""Tests for prism.auth.env_store."""

from __future__ import annotations

from unittest.mock import patch

from prism.auth.env_store import PROVIDER_ENV_MAP, EnvStore

# ---------------------------------------------------------------
# get_key
# ---------------------------------------------------------------


class TestGetKey:
    def test_returns_key_from_env(self, fake_keys):
        store = EnvStore()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": fake_keys["anthropic"]}):
            result = store.get_key("anthropic")
        assert result == fake_keys["anthropic"]

    def test_returns_none_when_var_unset(self):
        store = EnvStore()
        with patch.dict("os.environ", {}, clear=True):
            result = store.get_key("anthropic")
        assert result is None

    def test_returns_none_for_unknown_provider(self):
        store = EnvStore()
        assert store.get_key("unknown_provider_xyz") is None

    def test_each_known_provider(self, fake_keys):
        """Every provider in PROVIDER_ENV_MAP should resolve correctly."""
        store = EnvStore()
        for provider, env_var in PROVIDER_ENV_MAP.items():
            key = fake_keys.get(provider, "test-key-value")
            with patch.dict("os.environ", {env_var: key}):
                assert store.get_key(provider) == key

    def test_custom_env_map(self):
        """EnvStore can be initialised with a custom mapping."""
        custom_map = {"custom_provider": "CUSTOM_KEY_VAR"}
        store = EnvStore(env_map=custom_map)
        with patch.dict("os.environ", {"CUSTOM_KEY_VAR": "my-secret-key"}):
            assert store.get_key("custom_provider") == "my-secret-key"
        assert store.get_key("anthropic") is None  # not in custom map


# ---------------------------------------------------------------
# list_configured
# ---------------------------------------------------------------


class TestListConfigured:
    def test_lists_set_providers(self, fake_keys):
        store = EnvStore()
        env = {
            "ANTHROPIC_API_KEY": fake_keys["anthropic"],
            "OPENAI_API_KEY": fake_keys["openai"],
        }
        with patch.dict("os.environ", env, clear=True):
            result = store.list_configured()
        assert result == ["anthropic", "openai"]

    def test_empty_when_nothing_set(self):
        store = EnvStore()
        with patch.dict("os.environ", {}, clear=True):
            result = store.list_configured()
        assert result == []

    def test_ignores_empty_values(self):
        store = EnvStore()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=True):
            result = store.list_configured()
        assert result == []

    def test_returns_sorted(self, fake_keys):
        store = EnvStore()
        env = {
            "GROQ_API_KEY": fake_keys["groq"],
            "ANTHROPIC_API_KEY": fake_keys["anthropic"],
            "DEEPSEEK_API_KEY": fake_keys["deepseek"],
        }
        with patch.dict("os.environ", env, clear=True):
            result = store.list_configured()
        assert result == sorted(result)


# ---------------------------------------------------------------
# env_map property
# ---------------------------------------------------------------


class TestEnvMapProperty:
    def test_returns_copy(self):
        store = EnvStore()
        map1 = store.env_map
        map2 = store.env_map
        assert map1 == map2
        assert map1 is not map2  # must be a copy

    def test_default_mapping(self):
        store = EnvStore()
        assert store.env_map == PROVIDER_ENV_MAP


# ---------------------------------------------------------------
# Masking
# ---------------------------------------------------------------


class TestMaskKey:
    def test_short_key_fully_masked(self):
        from prism.auth.env_store import _mask_key

        assert _mask_key("ab") == "****"

    def test_normal_key_shows_last_four(self):
        from prism.auth.env_store import _mask_key

        assert _mask_key("sk-ant-long-key-abcd") == "****abcd"

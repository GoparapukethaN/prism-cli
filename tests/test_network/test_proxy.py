"""Tests for ProxyConfig and ProxyManager."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from prism.network.proxy import ProxyConfig, ProxyManager

# ---------------------------------------------------------------------------
# TestProxyConfig
# ---------------------------------------------------------------------------


class TestProxyConfig:
    """Tests for ProxyConfig dataclass."""

    def test_defaults(self) -> None:
        """Default ProxyConfig has no proxy and standard timeouts."""
        config = ProxyConfig()
        assert config.http_proxy is None
        assert config.https_proxy is None
        assert config.socks5_proxy is None
        assert config.no_proxy == []
        assert config.ssl_verify is True
        assert config.ssl_ca_bundle is None
        assert config.connect_timeout == 10.0
        assert config.read_timeout == 60.0

    def test_custom_values(self) -> None:
        """ProxyConfig accepts custom values."""
        config = ProxyConfig(
            http_proxy="http://proxy:8080",
            https_proxy="https://proxy:8443",
            socks5_proxy="socks5://proxy:1080",
            no_proxy=["localhost", "127.0.0.1"],
            ssl_verify=False,
            ssl_ca_bundle="/path/to/ca-bundle.crt",
            connect_timeout=5.0,
            read_timeout=120.0,
        )
        assert config.http_proxy == "http://proxy:8080"
        assert config.https_proxy == "https://proxy:8443"
        assert config.socks5_proxy == "socks5://proxy:1080"
        assert config.no_proxy == ["localhost", "127.0.0.1"]
        assert config.ssl_verify is False
        assert config.ssl_ca_bundle == "/path/to/ca-bundle.crt"
        assert config.connect_timeout == 5.0
        assert config.read_timeout == 120.0

    def test_no_proxy_list_is_independent(self) -> None:
        """Each ProxyConfig gets its own no_proxy list."""
        c1 = ProxyConfig()
        c2 = ProxyConfig()
        c1.no_proxy.append("example.com")
        assert c2.no_proxy == []

    def test_partial_proxy(self) -> None:
        """Only http_proxy set, others remain None."""
        config = ProxyConfig(http_proxy="http://proxy:8080")
        assert config.http_proxy == "http://proxy:8080"
        assert config.https_proxy is None
        assert config.socks5_proxy is None


# ---------------------------------------------------------------------------
# TestProxyManager — from env
# ---------------------------------------------------------------------------


class TestProxyManagerFromEnv:
    """Tests for ProxyManager reading environment variables."""

    def test_no_env_vars(self) -> None:
        """When no proxy env vars are set, global config has no proxy."""
        env = {"HOME": "/tmp", "PATH": "/usr/bin"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.has_proxy is False
            assert pm.global_config.http_proxy is None
            assert pm.global_config.https_proxy is None

    def test_http_proxy_upper(self) -> None:
        """HTTP_PROXY (upper) is read."""
        env = {"HTTP_PROXY": "http://corp-proxy:3128"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.http_proxy == "http://corp-proxy:3128"
            assert pm.has_proxy is True

    def test_http_proxy_lower(self) -> None:
        """http_proxy (lower) is read when upper is absent."""
        env = {"http_proxy": "http://lower-proxy:3128"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.http_proxy == "http://lower-proxy:3128"

    def test_upper_takes_precedence(self) -> None:
        """HTTP_PROXY takes precedence over http_proxy."""
        env = {
            "HTTP_PROXY": "http://upper:3128",
            "http_proxy": "http://lower:3128",
        }
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.http_proxy == "http://upper:3128"

    def test_https_proxy(self) -> None:
        """HTTPS_PROXY is read."""
        env = {"HTTPS_PROXY": "https://secure-proxy:443"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.https_proxy == "https://secure-proxy:443"
            assert pm.has_proxy is True

    def test_socks5_proxy(self) -> None:
        """SOCKS5_PROXY is read."""
        env = {"SOCKS5_PROXY": "socks5://socks-proxy:1080"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.socks5_proxy == "socks5://socks-proxy:1080"
            assert pm.has_proxy is True

    def test_no_proxy_parsed(self) -> None:
        """NO_PROXY is split on commas."""
        env = {"NO_PROXY": "localhost, 127.0.0.1, .internal.corp"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.no_proxy == [
                "localhost",
                "127.0.0.1",
                ".internal.corp",
            ]

    def test_no_proxy_empty(self) -> None:
        """Empty NO_PROXY produces empty list."""
        env = {"NO_PROXY": ""}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.no_proxy == []

    def test_ssl_verify_false(self) -> None:
        """PRISM_SSL_VERIFY=false disables verification."""
        env = {"PRISM_SSL_VERIFY": "false"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.ssl_verify is False

    def test_ssl_verify_zero(self) -> None:
        """PRISM_SSL_VERIFY=0 disables verification."""
        env = {"PRISM_SSL_VERIFY": "0"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.ssl_verify is False

    def test_ssl_verify_default_true(self) -> None:
        """Without PRISM_SSL_VERIFY, verification is enabled."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            assert pm.global_config.ssl_verify is True

    def test_ca_bundle(self) -> None:
        """PRISM_CA_BUNDLE is read."""
        env = {"PRISM_CA_BUNDLE": "/etc/ssl/custom-ca.crt"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.ssl_ca_bundle == "/etc/ssl/custom-ca.crt"

    def test_connect_timeout_from_env(self) -> None:
        """PRISM_CONNECT_TIMEOUT overrides default."""
        env = {"PRISM_CONNECT_TIMEOUT": "5.0"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.connect_timeout == 5.0

    def test_read_timeout_from_env(self) -> None:
        """PRISM_READ_TIMEOUT overrides default."""
        env = {"PRISM_READ_TIMEOUT": "120"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.read_timeout == 120.0

    def test_invalid_timeout_falls_back(self) -> None:
        """Non-numeric timeout values fall back to defaults."""
        env = {
            "PRISM_CONNECT_TIMEOUT": "not-a-number",
            "PRISM_READ_TIMEOUT": "bad",
        }
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.connect_timeout == 10.0
            assert pm.global_config.read_timeout == 60.0

    def test_negative_timeout_falls_back(self) -> None:
        """Negative timeout values fall back to defaults."""
        env = {"PRISM_CONNECT_TIMEOUT": "-5", "PRISM_READ_TIMEOUT": "-10"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.global_config.connect_timeout == 10.0
            assert pm.global_config.read_timeout == 60.0


# ---------------------------------------------------------------------------
# TestProxyManager — per-provider config
# ---------------------------------------------------------------------------


class TestProxyManagerProviderConfig:
    """Tests for per-provider proxy configuration."""

    def test_get_config_fallback_to_global(self) -> None:
        """get_config returns global when no provider config set."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            config = pm.get_config("openai")
            assert config is pm.global_config

    def test_get_config_with_provider(self) -> None:
        """get_config returns provider-specific config when set."""
        provider_config = ProxyConfig(http_proxy="http://openai-proxy:8080")
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager(provider_configs={"openai": provider_config})
            config = pm.get_config("openai")
            assert config.http_proxy == "http://openai-proxy:8080"

    def test_get_config_none_returns_global(self) -> None:
        """get_config(None) always returns global."""
        provider_config = ProxyConfig(http_proxy="http://proxy:8080")
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager(provider_configs={"openai": provider_config})
            config = pm.get_config(None)
            assert config is pm.global_config

    def test_set_provider_config(self) -> None:
        """set_provider_config stores config for a provider."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            custom = ProxyConfig(
                https_proxy="https://custom:443",
                connect_timeout=3.0,
            )
            pm.set_provider_config("anthropic", custom)
            config = pm.get_config("anthropic")
            assert config.https_proxy == "https://custom:443"
            assert config.connect_timeout == 3.0

    def test_set_provider_config_empty_name_raises(self) -> None:
        """set_provider_config raises ValueError for empty name."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            with pytest.raises(ValueError, match="must not be empty"):
                pm.set_provider_config("", ProxyConfig())

    def test_set_provider_config_whitespace_name_raises(self) -> None:
        """set_provider_config raises ValueError for whitespace-only name."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            with pytest.raises(ValueError, match="must not be empty"):
                pm.set_provider_config("   ", ProxyConfig())

    def test_remove_provider_config(self) -> None:
        """remove_provider_config removes a provider config."""
        config = ProxyConfig(http_proxy="http://proxy:8080")
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager(provider_configs={"openai": config})
            assert pm.remove_provider_config("openai") is True
            assert pm.get_config("openai") is pm.global_config

    def test_remove_nonexistent_provider_config(self) -> None:
        """remove_provider_config returns False for unknown provider."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            assert pm.remove_provider_config("nonexistent") is False

    def test_provider_names(self) -> None:
        """provider_names returns sorted list of configured providers."""
        configs = {
            "google": ProxyConfig(),
            "anthropic": ProxyConfig(),
            "openai": ProxyConfig(),
        }
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager(provider_configs=configs)
            assert pm.provider_names == ["anthropic", "google", "openai"]

    def test_provider_names_empty(self) -> None:
        """provider_names returns empty list when no providers configured."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            assert pm.provider_names == []


# ---------------------------------------------------------------------------
# TestProxyManager — get_httpx_kwargs
# ---------------------------------------------------------------------------


class TestProxyManagerHttpxKwargs:
    """Tests for get_httpx_kwargs."""

    def test_no_proxy_kwargs(self) -> None:
        """Without proxy, kwargs have timeout but no proxy."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            kwargs = pm.get_httpx_kwargs()
            assert "proxy" not in kwargs
            assert kwargs["timeout"]["connect"] == 10.0
            assert kwargs["timeout"]["read"] == 60.0

    def test_http_proxy_in_kwargs(self) -> None:
        """HTTP proxy appears in kwargs."""
        env = {"HTTP_PROXY": "http://proxy:3128"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            kwargs = pm.get_httpx_kwargs()
            assert kwargs["proxy"] == "http://proxy:3128"

    def test_socks5_takes_precedence(self) -> None:
        """SOCKS5 proxy takes precedence over HTTP/HTTPS."""
        env = {
            "HTTP_PROXY": "http://http-proxy:3128",
            "HTTPS_PROXY": "https://https-proxy:443",
            "SOCKS5_PROXY": "socks5://socks:1080",
        }
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            kwargs = pm.get_httpx_kwargs()
            assert kwargs["proxy"] == "socks5://socks:1080"

    def test_https_over_http_precedence(self) -> None:
        """HTTPS proxy takes precedence over HTTP when no SOCKS5."""
        env = {
            "HTTP_PROXY": "http://http-proxy:3128",
            "HTTPS_PROXY": "https://https-proxy:443",
        }
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            kwargs = pm.get_httpx_kwargs()
            assert kwargs["proxy"] == "https://https-proxy:443"

    def test_ssl_verify_false_in_kwargs(self) -> None:
        """verify=False when SSL verification is disabled."""
        env = {"PRISM_SSL_VERIFY": "false"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            kwargs = pm.get_httpx_kwargs()
            assert kwargs["verify"] is False

    def test_ca_bundle_in_kwargs(self) -> None:
        """Custom CA bundle path in kwargs."""
        env = {"PRISM_CA_BUNDLE": "/path/to/cert.pem"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            kwargs = pm.get_httpx_kwargs()
            assert kwargs["verify"] == "/path/to/cert.pem"

    def test_provider_specific_httpx_kwargs(self) -> None:
        """Provider-specific config produces different kwargs."""
        provider_config = ProxyConfig(
            http_proxy="http://openai-proxy:8080",
            connect_timeout=3.0,
            read_timeout=30.0,
        )
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager(provider_configs={"openai": provider_config})
            kwargs = pm.get_httpx_kwargs("openai")
            assert kwargs["proxy"] == "http://openai-proxy:8080"
            assert kwargs["timeout"]["connect"] == 3.0
            assert kwargs["timeout"]["read"] == 30.0

    def test_timeout_structure(self) -> None:
        """Timeout dict has all four keys."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            kwargs = pm.get_httpx_kwargs()
            timeout = kwargs["timeout"]
            assert "connect" in timeout
            assert "read" in timeout
            assert "write" in timeout
            assert "pool" in timeout


# ---------------------------------------------------------------------------
# TestProxyManager — get_litellm_kwargs
# ---------------------------------------------------------------------------


class TestProxyManagerLitellmKwargs:
    """Tests for get_litellm_kwargs."""

    def test_default_litellm_kwargs(self) -> None:
        """Default kwargs have timeout."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            kwargs = pm.get_litellm_kwargs()
            assert kwargs["timeout"] == 60.0
            assert "ssl_verify" not in kwargs

    def test_ssl_verify_false_litellm(self) -> None:
        """ssl_verify=False passed to LiteLLM when disabled."""
        env = {"PRISM_SSL_VERIFY": "false"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            kwargs = pm.get_litellm_kwargs()
            assert kwargs["ssl_verify"] is False

    def test_proxy_in_litellm_kwargs(self) -> None:
        """Proxy URL passed to LiteLLM."""
        env = {"HTTPS_PROXY": "https://proxy:443"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            kwargs = pm.get_litellm_kwargs()
            assert kwargs["proxy"] == "https://proxy:443"

    def test_provider_specific_litellm_kwargs(self) -> None:
        """Provider-specific timeout in LiteLLM kwargs."""
        config = ProxyConfig(read_timeout=120.0)
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager(provider_configs={"anthropic": config})
            kwargs = pm.get_litellm_kwargs("anthropic")
            assert kwargs["timeout"] == 120.0


# ---------------------------------------------------------------------------
# TestProxyManager — is_no_proxy
# ---------------------------------------------------------------------------


class TestProxyManagerNoProxy:
    """Tests for is_no_proxy hostname matching."""

    def test_empty_hostname(self) -> None:
        """Empty hostname returns False."""
        env = {"NO_PROXY": "localhost"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.is_no_proxy("") is False

    def test_exact_match(self) -> None:
        """Exact hostname match."""
        env = {"NO_PROXY": "localhost,api.internal.com"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.is_no_proxy("localhost") is True
            assert pm.is_no_proxy("api.internal.com") is True
            assert pm.is_no_proxy("other.com") is False

    def test_suffix_match_with_dot(self) -> None:
        """Suffix match with leading dot."""
        env = {"NO_PROXY": ".example.com"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.is_no_proxy("api.example.com") is True
            assert pm.is_no_proxy("deep.api.example.com") is True
            assert pm.is_no_proxy("example.com") is True
            assert pm.is_no_proxy("notexample.com") is False

    def test_implicit_suffix_match(self) -> None:
        """Implicit suffix match without leading dot."""
        env = {"NO_PROXY": "example.com"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.is_no_proxy("example.com") is True
            assert pm.is_no_proxy("api.example.com") is True
            assert pm.is_no_proxy("notexample.com") is False

    def test_wildcard_matches_all(self) -> None:
        """Wildcard '*' matches everything."""
        env = {"NO_PROXY": "*"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.is_no_proxy("anything.com") is True
            assert pm.is_no_proxy("localhost") is True

    def test_case_insensitive(self) -> None:
        """Hostname matching is case-insensitive."""
        env = {"NO_PROXY": "Example.COM"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.is_no_proxy("example.com") is True
            assert pm.is_no_proxy("EXAMPLE.COM") is True

    def test_no_proxy_empty_list(self) -> None:
        """Empty no_proxy list matches nothing."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            assert pm.is_no_proxy("localhost") is False


# ---------------------------------------------------------------------------
# TestProxyManager — reload
# ---------------------------------------------------------------------------


class TestProxyManagerReload:
    """Tests for reload_from_env."""

    def test_reload_picks_up_new_vars(self) -> None:
        """reload_from_env re-reads proxy env vars."""
        with patch.dict("os.environ", {}, clear=True):
            pm = ProxyManager()
            assert pm.has_proxy is False

        with patch.dict(
            "os.environ",
            {"HTTP_PROXY": "http://new-proxy:3128"},
            clear=True,
        ):
            pm.reload_from_env()
            assert pm.has_proxy is True
            assert pm.global_config.http_proxy == "http://new-proxy:3128"

    def test_reload_clears_old_proxy(self) -> None:
        """reload_from_env clears proxy when env var removed."""
        env = {"HTTP_PROXY": "http://proxy:3128"}
        with patch.dict("os.environ", env, clear=True):
            pm = ProxyManager()
            assert pm.has_proxy is True

        with patch.dict("os.environ", {}, clear=True):
            pm.reload_from_env()
            assert pm.has_proxy is False

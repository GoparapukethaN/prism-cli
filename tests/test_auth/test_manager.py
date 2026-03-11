"""Tests for prism.auth.manager — integration of all stores."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from prism.auth.encrypted_store import EncryptedStore
from prism.auth.env_store import EnvStore
from prism.auth.manager import (
    KNOWN_PROVIDERS,
    AuthManager,
    ProviderAuthStatus,
    _mask_key,
)
from prism.exceptions import (
    AuthError,
    KeyInvalidError,
    KeyNotFoundError,
    KeyringUnavailableError,
)

if TYPE_CHECKING:
    from pathlib import Path

try:
    import cryptography  # noqa: F401
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture()
def env_store():
    return EnvStore()


@pytest.fixture()
def encrypted_store(tmp_path: Path):
    return EncryptedStore(path=tmp_path / "credentials.enc")


@pytest.fixture()
def mock_keyring_store():
    """A mock KeyringStore that behaves like a dict-based store."""
    store_data: dict[str, str] = {}
    mock_ks = MagicMock()
    mock_ks.is_available = MagicMock(return_value=True)
    mock_ks.get_key = MagicMock(side_effect=store_data.get)
    mock_ks.set_key = MagicMock(side_effect=store_data.__setitem__)
    mock_ks.delete_key = MagicMock(
        side_effect=lambda p: store_data.pop(p, None)
    )
    mock_ks._data = store_data  # for test inspection
    return mock_ks


@pytest.fixture()
def mock_keyring_unavailable_store():
    """A mock KeyringStore where keyring is unavailable."""
    mock_ks = MagicMock()
    mock_ks.is_available = MagicMock(return_value=False)
    mock_ks.get_key = MagicMock(return_value=None)
    mock_ks.set_key = MagicMock(side_effect=Exception("keyring unavailable"))
    mock_ks.delete_key = MagicMock()
    return mock_ks


# ---------------------------------------------------------------
# get_key — resolution order
# ---------------------------------------------------------------


class TestGetKey:
    def test_keyring_first(self, mock_keyring_store, env_store, encrypted_store, fake_keys):
        """Keyring has highest priority."""
        mock_keyring_store._data["anthropic"] = fake_keys["anthropic"]
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key-1234"}):
            mgr = AuthManager(
                keyring_store=mock_keyring_store,
                env_store=env_store,
                encrypted_store=encrypted_store,
            )
            # Need to patch is_available at class level for static method calls
            with patch("prism.auth.manager.KeyringStore.is_available", return_value=True):
                with patch("prism.auth.manager.KeyringStore.get_key", return_value=fake_keys["anthropic"]):
                    result = mgr.get_key("anthropic")
        assert result == fake_keys["anthropic"]

    def test_falls_back_to_env(self, env_store, encrypted_store, fake_keys):
        """If keyring is unavailable, env is next."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": fake_keys["openai"]}):
            with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                result = mgr.get_key("openai")
        assert result == fake_keys["openai"]

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_falls_back_to_encrypted(self, env_store, encrypted_store, fake_keys):
        """If keyring and env both miss, try encrypted store."""
        encrypted_store.set_key("google", fake_keys["google"], "my-pass")

        with patch.dict("os.environ", {}, clear=True):
            with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                    passphrase="my-pass",
                )
                result = mgr.get_key("google")
        assert result == fake_keys["google"]

    def test_raises_key_not_found(self, env_store, encrypted_store):
        """If all stores miss, raise KeyNotFoundError."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                with pytest.raises(KeyNotFoundError):
                    mgr.get_key("anthropic")

    def test_encrypted_not_tried_without_passphrase(self, env_store, encrypted_store):
        """Without a passphrase, encrypted store is skipped."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                    passphrase=None,
                )
                with pytest.raises(KeyNotFoundError):
                    mgr.get_key("anthropic")


# ---------------------------------------------------------------
# store_key
# ---------------------------------------------------------------


class TestStoreKey:
    def test_stores_in_keyring_when_available(self, env_store, encrypted_store, fake_keys):
        with patch("prism.auth.manager.KeyringStore.is_available", return_value=True):
            with patch("prism.auth.manager.KeyringStore.set_key") as mock_set:
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                result = mgr.store_key("anthropic", fake_keys["anthropic"])
        assert result == "keyring"
        mock_set.assert_called_once_with("anthropic", fake_keys["anthropic"])

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_falls_back_to_encrypted(self, env_store, encrypted_store, fake_keys):

        with patch("prism.auth.manager.KeyringStore.is_available", return_value=True):
            with patch(
                "prism.auth.manager.KeyringStore.set_key",
                side_effect=KeyringUnavailableError("broken"),
            ):
                with patch("prism.auth.manager.EncryptedStore.is_available", return_value=True):
                    mgr = AuthManager(
                        env_store=env_store,
                        encrypted_store=encrypted_store,
                        passphrase="test-pass",
                    )
                    result = mgr.store_key("anthropic", fake_keys["anthropic"])
        assert result == "encrypted"
        # Verify we can read it back.
        assert encrypted_store.get_key("anthropic", "test-pass") == fake_keys["anthropic"]

    def test_rejects_invalid_format(self, env_store, encrypted_store):
        """Validation failure should raise KeyInvalidError."""
        with patch("prism.auth.manager.KeyringStore.is_available", return_value=True):
            mgr = AuthManager(
                env_store=env_store,
                encrypted_store=encrypted_store,
            )
            with pytest.raises(KeyInvalidError):
                mgr.store_key("anthropic", "bad-key-no-prefix")

    def test_skip_validation(self, env_store, encrypted_store):
        """validate=False skips format check."""
        with patch("prism.auth.manager.KeyringStore.is_available", return_value=True):
            with patch("prism.auth.manager.KeyringStore.set_key"):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                result = mgr.store_key("anthropic", "bad-key", validate=False)
        assert result == "keyring"

    def test_raises_when_no_writable_store(self, env_store, encrypted_store):
        """No keyring, no passphrase -> AuthError."""
        with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
            with patch("prism.auth.manager.EncryptedStore.is_available", return_value=False):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                    passphrase=None,
                )
                with pytest.raises(AuthError, match="No writable"):
                    mgr.store_key("google", "some-key", validate=False)


# ---------------------------------------------------------------
# remove_key
# ---------------------------------------------------------------


class TestRemoveKey:
    def test_removes_from_keyring(self, env_store, encrypted_store):
        with patch("prism.auth.manager.KeyringStore.is_available", return_value=True):
            with patch("prism.auth.manager.KeyringStore.delete_key") as mock_del:
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                mgr.remove_key("openai")
        mock_del.assert_called_once_with("openai")

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_removes_from_encrypted(self, env_store, encrypted_store, fake_keys):
        encrypted_store.set_key("openai", fake_keys["openai"], "pass")

        with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
            with patch("prism.auth.manager.EncryptedStore.is_available", return_value=True):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                    passphrase="pass",
                )
                mgr.remove_key("openai")

        assert encrypted_store.get_key("openai", "pass") is None


# ---------------------------------------------------------------
# list_configured
# ---------------------------------------------------------------


class TestListConfigured:
    def test_returns_status_for_all_known_providers(self, env_store, encrypted_store):
        with patch.dict("os.environ", {}, clear=True):
            with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                statuses = mgr.list_configured()

        assert len(statuses) == len(KNOWN_PROVIDERS)
        providers = {s.provider for s in statuses}
        assert providers == KNOWN_PROVIDERS

    def test_sorted_by_provider(self, env_store, encrypted_store):
        with patch.dict("os.environ", {}, clear=True):
            with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                statuses = mgr.list_configured()
        names = [s.provider for s in statuses]
        assert names == sorted(names)

    def test_detects_env_key(self, env_store, encrypted_store, fake_keys):
        with patch.dict("os.environ", {"OPENAI_API_KEY": fake_keys["openai"]}, clear=True):
            with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                statuses = mgr.list_configured()

        openai_status = next(s for s in statuses if s.provider == "openai")
        assert openai_status.has_key is True
        assert openai_status.source == "env"
        assert openai_status.is_valid_format is True

    def test_detects_invalid_format(self, env_store, encrypted_store):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "bad-no-prefix"}, clear=True):
            with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                statuses = mgr.list_configured()

        anthropic_status = next(s for s in statuses if s.provider == "anthropic")
        assert anthropic_status.has_key is True
        assert anthropic_status.is_valid_format is False
        assert len(anthropic_status.errors) > 0

    def test_no_key_status(self, env_store, encrypted_store):
        with patch.dict("os.environ", {}, clear=True):
            with patch("prism.auth.manager.KeyringStore.is_available", return_value=False):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                statuses = mgr.list_configured()

        for status in statuses:
            assert status.has_key is False
            assert status.source is None
            assert status.is_valid_format is None


# ---------------------------------------------------------------
# validate_key (convenience proxy)
# ---------------------------------------------------------------


class TestValidateKey:
    def test_delegates_to_validator(self, env_store, encrypted_store):
        mgr = AuthManager(
            env_store=env_store,
            encrypted_store=encrypted_store,
        )
        assert mgr.validate_key("anthropic", "sk-ant-valid-key") is True
        assert mgr.validate_key("anthropic", "bad-key") is False


# ---------------------------------------------------------------
# ProviderAuthStatus dataclass
# ---------------------------------------------------------------


class TestProviderAuthStatus:
    def test_frozen(self):
        status = ProviderAuthStatus(provider="anthropic", has_key=False)
        with pytest.raises(AttributeError):
            status.provider = "openai"  # type: ignore[misc]

    def test_defaults(self):
        status = ProviderAuthStatus(provider="openai", has_key=True)
        assert status.source is None
        assert status.is_valid_format is None
        assert status.errors == []

    def test_full_construction(self):
        status = ProviderAuthStatus(
            provider="google",
            has_key=True,
            source="env",
            is_valid_format=True,
            errors=[],
        )
        assert status.provider == "google"
        assert status.has_key is True
        assert status.source == "env"
        assert status.is_valid_format is True


# ---------------------------------------------------------------
# _mask_key helper (line 329)
# ---------------------------------------------------------------


class TestMaskKey:
    def test_short_key_returns_stars(self):
        """Keys with 4 or fewer chars should return '****'."""
        assert _mask_key("ab") == "****"
        assert _mask_key("abcd") == "****"
        assert _mask_key("") == "****"

    def test_long_key_shows_last_four(self):
        """Keys longer than 4 chars should show ****<last4>."""
        assert _mask_key("sk-ant-12345678") == "****5678"
        assert _mask_key("12345") == "****2345"


# ---------------------------------------------------------------
# get_key — encrypted store resolution (lines 117-118)
# ---------------------------------------------------------------


class TestGetKeyEncrypted:
    def test_get_key_resolves_from_encrypted_store(
        self, env_store, encrypted_store
    ):
        """get_key should resolve from encrypted store when keyring and env miss."""
        mock_enc = MagicMock()
        mock_enc.get_key.return_value = "enc-secret-key-1234"

        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "prism.auth.manager.KeyringStore.is_available", return_value=False
            ):
                with patch(
                    "prism.auth.manager.EncryptedStore.is_available",
                    return_value=True,
                ):
                    mgr = AuthManager(
                        env_store=env_store,
                        encrypted_store=mock_enc,
                        passphrase="test-pass",
                    )
                    result = mgr.get_key("anthropic")

        assert result == "enc-secret-key-1234"
        mock_enc.get_key.assert_called_once_with("anthropic", "test-pass")


# ---------------------------------------------------------------
# _try_keyring — KeyringUnavailableError handling (lines 264-266)
# ---------------------------------------------------------------


class TestTryKeyring:
    def test_try_keyring_returns_none_on_keyring_error(self, env_store, encrypted_store):
        """_try_keyring should return None when KeyringStore.get_key raises."""
        with patch(
            "prism.auth.manager.KeyringStore.is_available", return_value=True
        ):
            with patch(
                "prism.auth.manager.KeyringStore.get_key",
                side_effect=KeyringUnavailableError("backend broken"),
            ):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                result = mgr._try_keyring("anthropic")

        assert result is None


# ---------------------------------------------------------------
# _try_encrypted — passphrase/availability/error handling (lines 272-278)
# ---------------------------------------------------------------


class TestTryEncrypted:
    def test_try_encrypted_returns_none_without_passphrase(
        self, env_store, encrypted_store
    ):
        """_try_encrypted should return None if passphrase is None."""
        mgr = AuthManager(
            env_store=env_store,
            encrypted_store=encrypted_store,
            passphrase=None,
        )
        result = mgr._try_encrypted("anthropic")
        assert result is None

    def test_try_encrypted_returns_none_when_unavailable(
        self, env_store, encrypted_store
    ):
        """_try_encrypted should return None if EncryptedStore is not available."""
        with patch(
            "prism.auth.manager.EncryptedStore.is_available", return_value=False
        ):
            mgr = AuthManager(
                env_store=env_store,
                encrypted_store=encrypted_store,
                passphrase="my-pass",
            )
            result = mgr._try_encrypted("anthropic")
        assert result is None

    def test_try_encrypted_returns_key_on_success(self, env_store):
        """_try_encrypted should return the key from encrypted store."""
        mock_enc = MagicMock()
        mock_enc.get_key.return_value = "secret-from-encrypted"

        with patch(
            "prism.auth.manager.EncryptedStore.is_available", return_value=True
        ):
            mgr = AuthManager(
                env_store=env_store,
                encrypted_store=mock_enc,
                passphrase="my-pass",
            )
            result = mgr._try_encrypted("google")

        assert result == "secret-from-encrypted"

    def test_try_encrypted_returns_none_on_auth_error(self, env_store):
        """_try_encrypted should return None when encrypted store raises AuthError."""
        mock_enc = MagicMock()
        mock_enc.get_key.side_effect = AuthError("decryption failed")

        with patch(
            "prism.auth.manager.EncryptedStore.is_available", return_value=True
        ):
            mgr = AuthManager(
                env_store=env_store,
                encrypted_store=mock_enc,
                passphrase="my-pass",
            )
            result = mgr._try_encrypted("google")

        assert result is None


# ---------------------------------------------------------------
# store_key — keyring fallback to encrypted (lines 165-166, 175-182)
# ---------------------------------------------------------------


class TestStoreKeyFallback:
    def test_store_key_falls_back_on_keyring_write_failure(
        self, env_store
    ):
        """store_key should fall back to encrypted when keyring write raises."""
        mock_enc = MagicMock()

        with patch(
            "prism.auth.manager.KeyringStore.is_available", return_value=True
        ):
            with patch(
                "prism.auth.manager.KeyringStore.set_key",
                side_effect=KeyringUnavailableError("write failed"),
            ):
                with patch(
                    "prism.auth.manager.EncryptedStore.is_available",
                    return_value=True,
                ):
                    mgr = AuthManager(
                        env_store=env_store,
                        encrypted_store=mock_enc,
                        passphrase="test-pass",
                    )
                    result = mgr.store_key(
                        "anthropic", "bad-key", validate=False
                    )

        assert result == "encrypted"
        mock_enc.set_key.assert_called_once_with(
            "anthropic", "bad-key", "test-pass"
        )

    def test_store_key_encrypted_uses_explicit_passphrase(
        self, env_store
    ):
        """store_key should prefer the explicit passphrase over instance one."""
        mock_enc = MagicMock()

        with patch(
            "prism.auth.manager.KeyringStore.is_available", return_value=False
        ):
            with patch(
                "prism.auth.manager.EncryptedStore.is_available",
                return_value=True,
            ):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=mock_enc,
                    passphrase="instance-pass",
                )
                result = mgr.store_key(
                    "openai", "sk-key-val", validate=False, passphrase="explicit-pass"
                )

        assert result == "encrypted"
        mock_enc.set_key.assert_called_once_with(
            "openai", "sk-key-val", "explicit-pass"
        )


# ---------------------------------------------------------------
# remove_key — keyring and encrypted error paths (lines 203-215)
# ---------------------------------------------------------------


class TestRemoveKeyErrorPaths:
    def test_remove_key_catches_keyring_unavailable(self, env_store, encrypted_store):
        """remove_key should not raise when keyring delete raises KeyringUnavailableError."""
        with patch(
            "prism.auth.manager.KeyringStore.is_available", return_value=True
        ):
            with patch(
                "prism.auth.manager.KeyringStore.delete_key",
                side_effect=KeyringUnavailableError("delete failed"),
            ):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=encrypted_store,
                )
                # Should not raise
                mgr.remove_key("anthropic")

    def test_remove_key_catches_encrypted_auth_error(self, env_store):
        """remove_key should not raise when encrypted delete raises AuthError."""
        mock_enc = MagicMock()
        mock_enc.delete_key.side_effect = AuthError("decryption failed")

        with patch(
            "prism.auth.manager.KeyringStore.is_available", return_value=False
        ):
            with patch(
                "prism.auth.manager.EncryptedStore.is_available",
                return_value=True,
            ):
                mgr = AuthManager(
                    env_store=env_store,
                    encrypted_store=mock_enc,
                    passphrase="test-pass",
                )
                # Should not raise
                mgr.remove_key("anthropic")

        mock_enc.delete_key.assert_called_once_with("anthropic", "test-pass")

    def test_remove_key_skips_encrypted_without_passphrase(
        self, env_store
    ):
        """remove_key should skip encrypted store when no passphrase is available."""
        mock_enc = MagicMock()

        with patch(
            "prism.auth.manager.KeyringStore.is_available", return_value=False
        ):
            mgr = AuthManager(
                env_store=env_store,
                encrypted_store=mock_enc,
                passphrase=None,
            )
            mgr.remove_key("anthropic")

        mock_enc.delete_key.assert_not_called()

    def test_remove_key_removes_from_both_stores(self, env_store):
        """remove_key should attempt removal from both keyring and encrypted."""
        mock_enc = MagicMock()

        with patch(
            "prism.auth.manager.KeyringStore.is_available", return_value=True
        ):
            with patch(
                "prism.auth.manager.KeyringStore.delete_key"
            ) as mock_kr_del:
                with patch(
                    "prism.auth.manager.EncryptedStore.is_available",
                    return_value=True,
                ):
                    mgr = AuthManager(
                        env_store=env_store,
                        encrypted_store=mock_enc,
                        passphrase="pass",
                    )
                    mgr.remove_key("openai")

        mock_kr_del.assert_called_once_with("openai")
        mock_enc.delete_key.assert_called_once_with("openai", "pass")


# ---------------------------------------------------------------
# _provider_status — keyring and encrypted source paths (lines 291, 303)
# ---------------------------------------------------------------


class TestProviderStatusSources:
    def test_provider_status_keyring_source(self, env_store, encrypted_store, fake_keys):
        """_provider_status should report source='keyring' when keyring has key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "prism.auth.manager.KeyringStore.is_available", return_value=True
            ):
                with patch(
                    "prism.auth.manager.KeyringStore.get_key",
                    return_value=fake_keys["anthropic"],
                ):
                    mgr = AuthManager(
                        env_store=env_store,
                        encrypted_store=encrypted_store,
                    )
                    status = mgr._provider_status("anthropic")

        assert status.has_key is True
        assert status.source == "keyring"

    def test_provider_status_encrypted_source(self, env_store):
        """_provider_status should report source='encrypted' when encrypted has key."""
        mock_enc = MagicMock()
        mock_enc.get_key.return_value = "enc-key-secret-1234"

        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "prism.auth.manager.KeyringStore.is_available", return_value=False
            ):
                with patch(
                    "prism.auth.manager.EncryptedStore.is_available",
                    return_value=True,
                ):
                    mgr = AuthManager(
                        env_store=EnvStore(),
                        encrypted_store=mock_enc,
                        passphrase="pass",
                    )
                    status = mgr._provider_status("deepseek")

        assert status.has_key is True
        assert status.source == "encrypted"

    def test_list_configured_detects_keyring_key(
        self, env_store, encrypted_store, fake_keys
    ):
        """list_configured should detect keyring keys and report source."""
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "prism.auth.manager.KeyringStore.is_available", return_value=True
            ):
                with patch(
                    "prism.auth.manager.KeyringStore.get_key",
                    side_effect=fake_keys.get,
                ):
                    mgr = AuthManager(
                        env_store=env_store,
                        encrypted_store=encrypted_store,
                    )
                    statuses = mgr.list_configured()

        anthropic_status = next(
            s for s in statuses if s.provider == "anthropic"
        )
        assert anthropic_status.has_key is True
        assert anthropic_status.source == "keyring"

    def test_list_configured_detects_encrypted_key(self, env_store):
        """list_configured should detect encrypted keys and report source."""
        mock_enc = MagicMock()

        def _enc_get(provider, passphrase):
            if provider == "google":
                return "AIzaSyDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"
            return None

        mock_enc.get_key.side_effect = _enc_get

        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "prism.auth.manager.KeyringStore.is_available", return_value=False
            ):
                with patch(
                    "prism.auth.manager.EncryptedStore.is_available",
                    return_value=True,
                ):
                    mgr = AuthManager(
                        env_store=EnvStore(),
                        encrypted_store=mock_enc,
                        passphrase="pass",
                    )
                    statuses = mgr.list_configured()

        google_status = next(s for s in statuses if s.provider == "google")
        assert google_status.has_key is True
        assert google_status.source == "encrypted"

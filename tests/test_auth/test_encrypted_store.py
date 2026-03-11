"""Tests for prism.auth.encrypted_store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.exceptions import AuthError

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------
# Helpers — skip when cryptography is not installed
# ---------------------------------------------------------------

try:
    import cryptography  # noqa: F401
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

skip_no_crypto = pytest.mark.skipif(
    not HAS_CRYPTO,
    reason="cryptography package not installed",
)


# ---------------------------------------------------------------
# is_available
# ---------------------------------------------------------------


class TestIsAvailable:
    @skip_no_crypto
    def test_available_when_crypto_installed(self):
        from prism.auth.encrypted_store import EncryptedStore

        assert EncryptedStore.is_available() is True

    def test_reflects_import_status(self):
        """is_available() matches whether cryptography is importable."""
        from prism.auth.encrypted_store import EncryptedStore

        assert EncryptedStore.is_available() is HAS_CRYPTO


# ---------------------------------------------------------------
# Round-trip: set / get / delete
# ---------------------------------------------------------------


@skip_no_crypto
class TestRoundTrip:
    def test_set_then_get(self, enc_store_path: Path, passphrase: str, fake_keys):
        from prism.auth.encrypted_store import EncryptedStore

        store = EncryptedStore(path=enc_store_path)
        store.set_key("anthropic", fake_keys["anthropic"], passphrase)

        result = store.get_key("anthropic", passphrase)
        assert result == fake_keys["anthropic"]

    def test_get_missing_returns_none(self, enc_store_path: Path, passphrase: str):
        from prism.auth.encrypted_store import EncryptedStore

        store = EncryptedStore(path=enc_store_path)
        # File does not exist yet.
        assert store.get_key("openai", passphrase) is None

    def test_multiple_providers(self, enc_store_path: Path, passphrase: str, fake_keys):
        from prism.auth.encrypted_store import EncryptedStore

        store = EncryptedStore(path=enc_store_path)

        store.set_key("anthropic", fake_keys["anthropic"], passphrase)
        store.set_key("openai", fake_keys["openai"], passphrase)
        store.set_key("google", fake_keys["google"], passphrase)

        assert store.get_key("anthropic", passphrase) == fake_keys["anthropic"]
        assert store.get_key("openai", passphrase) == fake_keys["openai"]
        assert store.get_key("google", passphrase) == fake_keys["google"]

    def test_overwrite_key(self, enc_store_path: Path, passphrase: str):
        from prism.auth.encrypted_store import EncryptedStore

        store = EncryptedStore(path=enc_store_path)

        store.set_key("openai", "sk-old-value-12345678", passphrase)
        store.set_key("openai", "sk-new-value-87654321", passphrase)

        assert store.get_key("openai", passphrase) == "sk-new-value-87654321"

    def test_delete_key(self, enc_store_path: Path, passphrase: str, fake_keys):
        from prism.auth.encrypted_store import EncryptedStore

        store = EncryptedStore(path=enc_store_path)
        store.set_key("anthropic", fake_keys["anthropic"], passphrase)
        store.delete_key("anthropic", passphrase)

        assert store.get_key("anthropic", passphrase) is None

    def test_delete_nonexistent_is_noop(self, enc_store_path: Path, passphrase: str):
        from prism.auth.encrypted_store import EncryptedStore

        store = EncryptedStore(path=enc_store_path)
        # Should not raise.
        store.delete_key("nonexistent", passphrase)


# ---------------------------------------------------------------
# Wrong passphrase
# ---------------------------------------------------------------


@skip_no_crypto
class TestWrongPassphrase:
    def test_wrong_passphrase_raises(self, enc_store_path: Path, fake_keys):
        from prism.auth.encrypted_store import EncryptedStore

        store = EncryptedStore(path=enc_store_path)
        store.set_key("anthropic", fake_keys["anthropic"], "correct-passphrase")

        with pytest.raises(AuthError, match="[Dd]ecryption failed"):
            store.get_key("anthropic", "wrong-passphrase")


# ---------------------------------------------------------------
# Corrupt file
# ---------------------------------------------------------------


@skip_no_crypto
class TestCorruptFile:
    def test_truncated_file(self, enc_store_path: Path, passphrase: str):
        from prism.auth.encrypted_store import EncryptedStore

        enc_store_path.parent.mkdir(parents=True, exist_ok=True)
        enc_store_path.write_bytes(b"\x00" * 5)  # less than salt length

        store = EncryptedStore(path=enc_store_path)
        with pytest.raises(AuthError, match="corrupt"):
            store.get_key("anthropic", passphrase)

    def test_garbage_after_salt(self, enc_store_path: Path, passphrase: str):
        from prism.auth.encrypted_store import EncryptedStore

        enc_store_path.parent.mkdir(parents=True, exist_ok=True)
        enc_store_path.write_bytes(b"\x00" * 16 + b"not-valid-fernet-data")

        store = EncryptedStore(path=enc_store_path)
        with pytest.raises(AuthError):
            store.get_key("anthropic", passphrase)


# ---------------------------------------------------------------
# File creation
# ---------------------------------------------------------------


@skip_no_crypto
class TestFileCreation:
    def test_creates_parent_dirs(self, tmp_path: Path, passphrase: str, fake_keys):
        from prism.auth.encrypted_store import EncryptedStore

        deep_path = tmp_path / "a" / "b" / "c" / "credentials.enc"
        store = EncryptedStore(path=deep_path)
        store.set_key("openai", fake_keys["openai"], passphrase)

        assert deep_path.is_file()
        assert store.get_key("openai", passphrase) == fake_keys["openai"]

    def test_file_contains_salt_plus_ciphertext(
        self, enc_store_path: Path, passphrase: str, fake_keys
    ):
        from prism.auth.encrypted_store import EncryptedStore

        store = EncryptedStore(path=enc_store_path)
        store.set_key("anthropic", fake_keys["anthropic"], passphrase)

        raw = enc_store_path.read_bytes()
        # Salt is 16 bytes, then Fernet token follows.
        assert len(raw) > 16
        # The plaintext key must NOT appear in the raw file.
        assert fake_keys["anthropic"].encode() not in raw


# ---------------------------------------------------------------
# Masking
# ---------------------------------------------------------------


class TestMaskKey:
    def test_short_key_fully_masked(self):
        from prism.auth.encrypted_store import _mask_key

        assert _mask_key("abc") == "****"

    def test_normal_key_shows_last_four(self):
        from prism.auth.encrypted_store import _mask_key

        assert _mask_key("super-secret-key-XY99") == "****XY99"

"""Shared fixtures for auth tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------
# Sample API keys (fake — never real)
# ---------------------------------------------------------------

FAKE_KEYS: dict[str, str] = {
    "anthropic": "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA-test1234",
    "openai": "sk-proj-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB-test5678",
    "google": "AIzaSyDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD",
    "deepseek": "sk-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee-test9012",
    "groq": "gsk_ffffffffffffffffffffffffffffffffffffffff",
    "mistral": "MistralKeyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
}


@pytest.fixture()
def fake_keys() -> dict[str, str]:
    """Return a dict of provider -> fake API key."""
    return dict(FAKE_KEYS)


# ---------------------------------------------------------------
# Keyring mocking helpers
# ---------------------------------------------------------------


@pytest.fixture()
def mock_keyring_available():
    """Mock keyring as available and functional.

    Yields a dict acting as the backing store so tests can inspect it.
    """
    store: dict[str, str] = {}

    def _get_password(service: str, username: str) -> str | None:
        return store.get(f"{service}::{username}")

    def _set_password(service: str, username: str, password: str) -> None:
        store[f"{service}::{username}"] = password

    def _delete_password(service: str, username: str) -> None:
        key = f"{service}::{username}"
        if key not in store:
            import keyring.errors
            raise keyring.errors.PasswordDeleteError("not found")
        del store[key]

    mock_kr = MagicMock()
    mock_kr.get_password = MagicMock(side_effect=_get_password)
    mock_kr.set_password = MagicMock(side_effect=_set_password)
    mock_kr.delete_password = MagicMock(side_effect=_delete_password)
    mock_kr.get_keyring.return_value = MagicMock()  # not FailKeyring

    mock_errors = MagicMock()
    mock_errors.KeyringError = type("KeyringError", (Exception,), {})
    mock_errors.PasswordDeleteError = type(
        "PasswordDeleteError", (mock_errors.KeyringError,), {}
    )
    mock_kr.errors = mock_errors

    with (
        patch.dict("sys.modules", {
            "keyring": mock_kr,
            "keyring.errors": mock_errors,
            "keyring.backends": MagicMock(),
            "keyring.backends.fail": MagicMock(),
        }),
    ):
        yield store


@pytest.fixture()
def mock_keyring_unavailable():
    """Mock keyring as unavailable (ImportError)."""
    import sys

    modules_to_remove = [
        "keyring",
        "keyring.errors",
        "keyring.backends",
        "keyring.backends.fail",
    ]
    saved = {m: sys.modules.pop(m, None) for m in modules_to_remove}

    with patch.dict("sys.modules", {m: None for m in modules_to_remove}):
        yield

    # Restore
    for m, mod in saved.items():
        if mod is not None:
            sys.modules[m] = mod
        else:
            sys.modules.pop(m, None)


# ---------------------------------------------------------------
# Encrypted store helpers
# ---------------------------------------------------------------


@pytest.fixture()
def enc_store_path(tmp_path: Path) -> Path:
    """Return a temporary path for the encrypted credentials file."""
    return tmp_path / "credentials.enc"


@pytest.fixture()
def passphrase() -> str:
    """Return a test passphrase."""
    return "test-passphrase-not-real"

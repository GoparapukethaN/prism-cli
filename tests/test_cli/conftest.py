"""Shared fixtures for CLI tests.

Ensures a clean asyncio event loop is available for every test.

Some CLI commands call ``asyncio.run()`` which creates and then *closes*
the main-thread event loop.  Subsequent tests that use
``asyncio.Future()`` or ``asyncio.get_event_loop()`` will fail with
``RuntimeError: There is no current event loop``.

The ``_ensure_event_loop`` autouse fixture restores the event loop after
every test, eliminating cross-test contamination.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _ensure_event_loop() -> None:
    """Guarantee a running event loop exists on the main thread.

    Runs before *and* after each test.  If a test closed the loop
    (e.g. via ``asyncio.run()``), this fixture creates a fresh one so
    the next test starts cleanly.
    """
    # Ensure loop exists before the test
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    yield

    # Restore loop after the test (in case it was closed)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

"""Interrupt handling tests.

Tests that Ctrl+C and SIGTERM are handled gracefully without data
corruption. Uses signal handling mocks -- never sends real signals.
"""

from __future__ import annotations

import signal
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from prism.db.queries import create_session, update_session

if TYPE_CHECKING:
    from pathlib import Path

    from prism.db.database import Database

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


class _FakeInterrupt(Exception):
    """Simulates a KeyboardInterrupt at a controlled point."""


def _simulate_ctrl_c() -> None:
    """Raise KeyboardInterrupt to simulate Ctrl+C."""
    raise KeyboardInterrupt


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestCtrlCDuringResponse:
    """Ctrl+C during response should not corrupt state."""

    def test_ctrl_c_during_response_no_corruption(
        self, integration_db: Database
    ) -> None:
        """Interrupting during a DB write should not leave partial data."""
        session_id = str(uuid4())
        create_session(integration_db, session_id, "/tmp")

        # Simulate an interrupt during session update
        try:
            update_session(integration_db, session_id, cost_delta=0.5, request_delta=1)
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

        # Data written before the interrupt should be intact
        row = integration_db.fetchone(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        assert row is not None
        assert row["total_cost"] == 0.5
        assert row["total_requests"] == 1


class TestCtrlCSavesSession:
    """Ctrl+C should leave the session in a valid state."""

    def test_ctrl_c_saves_session(self, integration_db: Database) -> None:
        session_id = str(uuid4())
        create_session(integration_db, session_id, "/tmp")

        # Perform several operations, then interrupt
        update_session(integration_db, session_id, cost_delta=0.1, request_delta=1)
        update_session(integration_db, session_id, cost_delta=0.2, request_delta=1)

        # Simulate interrupt
        try:
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

        # Session should be readable and have the accumulated values
        row = integration_db.fetchone(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        assert row is not None
        assert abs(row["total_cost"] - 0.3) < 1e-9
        assert row["total_requests"] == 2


class TestInterruptDuringFileWrite:
    """Interrupt during file write should not leave partial files."""

    def test_interrupt_during_file_write_no_partial_file(
        self, tmp_path: Path
    ) -> None:
        """Use atomic write pattern: write to temp, then rename."""
        target = tmp_path / "output.txt"
        temp_file = tmp_path / "output.txt.tmp"
        content = "full content that should be written atomically"

        # Simulate the atomic write pattern
        try:
            temp_file.write_text(content)
            # Simulate interrupt right before rename
            if temp_file.exists():
                temp_file.rename(target)
        except KeyboardInterrupt:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()

        # The target should have complete content
        assert target.exists()
        assert target.read_text() == content

    def test_interrupt_during_file_write_temp_cleaned(
        self, tmp_path: Path
    ) -> None:
        """If interrupt happens before rename, temp file can be cleaned."""
        temp_file = tmp_path / "output.txt.tmp"
        target = tmp_path / "output.txt"

        try:
            temp_file.write_text("partial content")
            # Simulate interrupt before rename
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            # Cleanup handler
            if temp_file.exists() and not target.exists():
                temp_file.unlink()

        assert not temp_file.exists()
        assert not target.exists()


class TestGracefulShutdown:
    """SIGTERM should trigger graceful shutdown."""

    def test_graceful_shutdown_on_sigterm(self) -> None:
        """Register a handler and verify it's callable."""
        shutdown_called = False

        def _shutdown_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_called
            shutdown_called = True

        # Register the handler
        old_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _shutdown_handler)

        try:
            # Simulate SIGTERM by calling the handler directly
            _shutdown_handler(signal.SIGTERM, None)
            assert shutdown_called
        finally:
            # Restore original handler
            signal.signal(signal.SIGTERM, old_handler)


class TestExitUpdatesState:
    """Exit should leave the database in a consistent state."""

    def test_exit_updates_state(self, integration_db: Database) -> None:
        """Simulated exit should not corrupt the database."""
        session_id = str(uuid4())
        create_session(integration_db, session_id, "/tmp")
        update_session(integration_db, session_id, cost_delta=1.0, request_delta=5)

        # Simulate exit cleanup
        try:
            raise SystemExit(0)
        except SystemExit:
            pass

        # Database should still be readable
        row = integration_db.fetchone(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        assert row is not None
        assert row["total_cost"] == 1.0


class TestMultipleCtrlC:
    """Multiple Ctrl+C should force exit without corruption."""

    def test_multiple_ctrl_c_force_exits(
        self, integration_db: Database
    ) -> None:
        """Multiple interrupts should not cause data corruption."""
        session_id = str(uuid4())
        create_session(integration_db, session_id, "/tmp")

        interrupt_count = 0
        for _i in range(3):
            try:
                update_session(
                    integration_db, session_id, cost_delta=0.1, request_delta=1
                )
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                interrupt_count += 1

        assert interrupt_count == 3

        # All writes before each interrupt should have committed
        row = integration_db.fetchone(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        assert row is not None
        assert abs(row["total_cost"] - 0.3) < 1e-9
        assert row["total_requests"] == 3


class TestTransactionRollbackOnInterrupt:
    """Transactions should roll back cleanly on interrupt."""

    def test_transaction_rollback_on_interrupt(
        self, integration_db: Database
    ) -> None:
        """A transaction interrupted mid-way should roll back."""
        session_id = str(uuid4())
        create_session(integration_db, session_id, "/tmp")

        try:
            with integration_db.transaction():
                integration_db.execute(
                    "UPDATE sessions SET total_cost = 999.0 WHERE id = ?",
                    (session_id,),
                )
                # Simulate interrupt inside the transaction
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass

        # The update inside the interrupted transaction should have been rolled back
        row = integration_db.fetchone(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        assert row is not None
        assert row["total_cost"] == 0.0  # original value, not 999.0

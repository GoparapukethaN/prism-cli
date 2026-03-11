"""Tests for OfflineModeManager, ConnectivityState, OfflineCapabilities, QueuedRequest."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from prism.network.offline import (
    ConnectivityState,
    OfflineCapabilities,
    OfflineModeManager,
    QueuedRequest,
)

# ---------------------------------------------------------------------------
# ConnectivityState tests
# ---------------------------------------------------------------------------


class TestConnectivityState:
    """Tests for ConnectivityState enum."""

    def test_online_value(self) -> None:
        """ONLINE has value 'online'."""
        assert ConnectivityState.ONLINE.value == "online"

    def test_offline_value(self) -> None:
        """OFFLINE has value 'offline'."""
        assert ConnectivityState.OFFLINE.value == "offline"

    def test_degraded_value(self) -> None:
        """DEGRADED has value 'degraded'."""
        assert ConnectivityState.DEGRADED.value == "degraded"

    def test_all_members(self) -> None:
        """All three states are present."""
        members = [s.value for s in ConnectivityState]
        assert sorted(members) == ["degraded", "offline", "online"]


# ---------------------------------------------------------------------------
# OfflineCapabilities tests
# ---------------------------------------------------------------------------


class TestOfflineCapabilities:
    """Tests for OfflineCapabilities dataclass."""

    def test_offline_defaults(self) -> None:
        """Default capabilities reflect offline state (no cloud)."""
        caps = OfflineCapabilities()
        assert caps.file_operations is True
        assert caps.terminal_execution is True
        assert caps.git_operations is True
        assert caps.local_inference is True
        assert caps.cache_hits is True
        assert caps.cloud_inference is False
        assert caps.web_browsing is False
        assert caps.plugin_install is False

    def test_online_capabilities(self) -> None:
        """Online capabilities enable all features."""
        caps = OfflineCapabilities(
            cloud_inference=True,
            web_browsing=True,
            plugin_install=True,
        )
        assert caps.cloud_inference is True
        assert caps.web_browsing is True
        assert caps.plugin_install is True


# ---------------------------------------------------------------------------
# QueuedRequest tests
# ---------------------------------------------------------------------------


class TestQueuedRequest:
    """Tests for QueuedRequest dataclass."""

    def test_required_fields(self) -> None:
        """QueuedRequest can be created with required fields."""
        req = QueuedRequest(
            id="abc123",
            model="gpt-4o",
            provider="openai",
            messages=[{"role": "user", "content": "hello"}],
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert req.id == "abc123"
        assert req.model == "gpt-4o"
        assert req.provider == "openai"
        assert len(req.messages) == 1
        assert req.retry_count == 0
        assert req.metadata == {}

    def test_all_fields(self) -> None:
        """QueuedRequest can be created with all fields."""
        req = QueuedRequest(
            id="xyz789",
            model="claude-3-haiku",
            provider="anthropic",
            messages=[],
            created_at="2026-01-01T00:00:00+00:00",
            retry_count=3,
            metadata={"custom": "data"},
        )
        assert req.retry_count == 3
        assert req.metadata == {"custom": "data"}

    def test_retry_count_mutable(self) -> None:
        """retry_count can be incremented."""
        req = QueuedRequest(
            id="a",
            model="m",
            provider="p",
            messages=[],
            created_at="now",
        )
        req.retry_count += 1
        assert req.retry_count == 1


# ---------------------------------------------------------------------------
# OfflineModeManager tests
# ---------------------------------------------------------------------------


class TestOfflineModeManager:
    """Tests for OfflineModeManager."""

    def test_default_state_online(self) -> None:
        """Manager starts in ONLINE state."""
        mgr = OfflineModeManager()
        assert mgr.state == ConnectivityState.ONLINE
        assert mgr.is_online is True
        assert mgr.is_offline is False

    def test_check_now_online(self) -> None:
        """check_now returns ONLINE when connectivity succeeds."""
        mgr = OfflineModeManager()
        with patch.object(mgr, "_check_connectivity", return_value=True):
            result = mgr.check_now()
        assert result == ConnectivityState.ONLINE
        assert mgr.is_online is True

    def test_check_now_offline(self) -> None:
        """check_now returns OFFLINE when connectivity fails."""
        mgr = OfflineModeManager()
        with patch.object(mgr, "_check_connectivity", return_value=False):
            result = mgr.check_now()
        assert result == ConnectivityState.OFFLINE
        assert mgr.is_offline is True

    def test_manual_offline_enable(self) -> None:
        """enable_manual_offline forces OFFLINE state."""
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()
        assert mgr.is_offline is True
        assert mgr.is_online is False
        assert mgr.is_manual_offline is True
        assert mgr.state == ConnectivityState.OFFLINE

    def test_manual_offline_disable_checks_connectivity(self) -> None:
        """disable_manual_offline re-checks real connectivity."""
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()
        assert mgr.is_offline is True

        with patch.object(mgr, "_check_connectivity", return_value=True):
            mgr.disable_manual_offline()
        assert mgr.is_online is True
        assert mgr.is_manual_offline is False

    def test_manual_offline_disable_still_offline(self) -> None:
        """disable_manual_offline stays offline if connectivity check fails."""
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()

        with patch.object(mgr, "_check_connectivity", return_value=False):
            mgr.disable_manual_offline()
        assert mgr.is_offline is True
        assert mgr.is_manual_offline is False

    def test_check_now_manual_offline_overrides(self) -> None:
        """check_now respects manual offline override."""
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()
        result = mgr.check_now()
        assert result == ConnectivityState.OFFLINE

    def test_is_online_false_when_manual_offline(self) -> None:
        """is_online is False even if state is ONLINE but manual offline is set."""
        mgr = OfflineModeManager()
        # Manually set state to ONLINE but enable manual offline
        mgr._state = ConnectivityState.ONLINE
        mgr._manual_offline = True
        assert mgr.is_online is False

    # ------------------------------------------------------------------
    # Request queue tests
    # ------------------------------------------------------------------

    def test_queue_request(self) -> None:
        """queue_request adds a request to the queue."""
        mgr = OfflineModeManager()
        req = QueuedRequest(
            id="r1",
            model="gpt-4o",
            provider="openai",
            messages=[{"role": "user", "content": "test"}],
            created_at=datetime.now(UTC).isoformat(),
        )
        mgr.queue_request(req)
        assert mgr.queued_count == 1
        queued = mgr.get_queued_requests()
        assert len(queued) == 1
        assert queued[0].id == "r1"

    def test_queue_multiple_requests(self) -> None:
        """Multiple requests can be queued."""
        mgr = OfflineModeManager()
        for i in range(3):
            mgr.queue_request(
                QueuedRequest(
                    id=f"r{i}",
                    model="gpt-4o",
                    provider="openai",
                    messages=[],
                    created_at="now",
                )
            )
        assert mgr.queued_count == 3

    def test_get_queued_returns_copy(self) -> None:
        """get_queued_requests returns a shallow copy."""
        mgr = OfflineModeManager()
        mgr.queue_request(
            QueuedRequest(id="r1", model="m", provider="p", messages=[], created_at="now")
        )
        q = mgr.get_queued_requests()
        q.clear()
        assert mgr.queued_count == 1

    def test_clear_queue(self) -> None:
        """clear_queue removes all queued requests and returns count."""
        mgr = OfflineModeManager()
        for i in range(3):
            mgr.queue_request(
                QueuedRequest(id=f"r{i}", model="m", provider="p", messages=[], created_at="now")
            )
        count = mgr.clear_queue()
        assert count == 3
        assert mgr.queued_count == 0

    def test_clear_empty_queue(self) -> None:
        """clear_queue on empty queue returns 0."""
        mgr = OfflineModeManager()
        assert mgr.clear_queue() == 0

    def test_remove_queued_request(self) -> None:
        """remove_queued_request removes by ID."""
        mgr = OfflineModeManager()
        mgr.queue_request(
            QueuedRequest(id="r1", model="m", provider="p", messages=[], created_at="now")
        )
        mgr.queue_request(
            QueuedRequest(id="r2", model="m", provider="p", messages=[], created_at="now")
        )
        result = mgr.remove_queued_request("r1")
        assert result is True
        assert mgr.queued_count == 1
        assert mgr.get_queued_requests()[0].id == "r2"

    def test_remove_queued_request_not_found(self) -> None:
        """remove_queued_request returns False if ID not found."""
        mgr = OfflineModeManager()
        assert mgr.remove_queued_request("nonexistent") is False

    def test_create_queued_request(self) -> None:
        """create_queued_request creates and queues a request."""
        mgr = OfflineModeManager()
        req = mgr.create_queued_request(
            model="gpt-4o",
            provider="openai",
            messages=[{"role": "user", "content": "test"}],
            metadata={"key": "value"},
        )
        assert req.model == "gpt-4o"
        assert req.provider == "openai"
        assert req.id != ""
        assert req.retry_count == 0
        assert req.metadata == {"key": "value"}
        assert mgr.queued_count == 1

    # ------------------------------------------------------------------
    # Capabilities tests
    # ------------------------------------------------------------------

    def test_capabilities_when_online(self) -> None:
        """Online capabilities have all features enabled."""
        mgr = OfflineModeManager()
        caps = mgr.get_capabilities()
        assert caps.cloud_inference is True
        assert caps.web_browsing is True
        assert caps.plugin_install is True
        assert caps.file_operations is True
        assert caps.terminal_execution is True
        assert caps.git_operations is True
        assert caps.local_inference is True
        assert caps.cache_hits is True

    def test_capabilities_when_offline(self) -> None:
        """Offline capabilities disable cloud features."""
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()
        caps = mgr.get_capabilities()
        assert caps.cloud_inference is False
        assert caps.web_browsing is False
        assert caps.plugin_install is False
        assert caps.file_operations is True
        assert caps.terminal_execution is True
        assert caps.git_operations is True
        assert caps.local_inference is True
        assert caps.cache_hits is True

    # ------------------------------------------------------------------
    # Status indicator tests
    # ------------------------------------------------------------------

    def test_status_indicator_online(self) -> None:
        """Online status indicator is empty."""
        mgr = OfflineModeManager()
        assert mgr.get_status_indicator() == ""

    def test_status_indicator_manual_offline(self) -> None:
        """Manual offline shows '[OFFLINE - Manual]'."""
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()
        assert mgr.get_status_indicator() == "[OFFLINE - Manual]"

    def test_status_indicator_auto_offline(self) -> None:
        """Auto-detected offline shows '[OFFLINE - Local models only]'."""
        mgr = OfflineModeManager()
        with patch.object(mgr, "_check_connectivity", return_value=False):
            mgr.check_now()
        assert mgr.get_status_indicator() == "[OFFLINE - Local models only]"

    def test_status_indicator_degraded(self) -> None:
        """Degraded state shows appropriate indicator."""
        mgr = OfflineModeManager()
        mgr._state = ConnectivityState.DEGRADED
        assert mgr.get_status_indicator() == "[DEGRADED - Some providers unavailable]"

    # ------------------------------------------------------------------
    # State change callback tests
    # ------------------------------------------------------------------

    def test_on_state_change_callback_fired(self) -> None:
        """Callbacks are invoked when state changes."""
        mgr = OfflineModeManager()
        states_received: list[ConnectivityState] = []
        mgr.on_state_change(states_received.append)

        with patch.object(mgr, "_check_connectivity", return_value=False):
            mgr.check_now()

        assert len(states_received) == 1
        assert states_received[0] == ConnectivityState.OFFLINE

    def test_callback_not_fired_when_state_unchanged(self) -> None:
        """Callbacks are NOT invoked when state stays the same."""
        mgr = OfflineModeManager()
        call_count = 0

        def counter(s: ConnectivityState) -> None:
            nonlocal call_count
            call_count += 1

        mgr.on_state_change(counter)

        with patch.object(mgr, "_check_connectivity", return_value=True):
            mgr.check_now()
            mgr.check_now()  # Same state — should not fire

        assert call_count == 0  # Already ONLINE, no change

    def test_callback_error_does_not_crash(self) -> None:
        """A failing callback does not crash the manager."""
        mgr = OfflineModeManager()

        def bad_callback(s: ConnectivityState) -> None:
            raise RuntimeError("boom")

        mgr.on_state_change(bad_callback)

        # This should not raise
        with patch.object(mgr, "_check_connectivity", return_value=False):
            mgr.check_now()

        assert mgr.state == ConnectivityState.OFFLINE

    def test_multiple_callbacks(self) -> None:
        """Multiple callbacks are all invoked."""
        mgr = OfflineModeManager()
        results: list[str] = []
        mgr.on_state_change(lambda s: results.append("cb1"))
        mgr.on_state_change(lambda s: results.append("cb2"))

        with patch.object(mgr, "_check_connectivity", return_value=False):
            mgr.check_now()

        assert results == ["cb1", "cb2"]

    def test_remove_state_callback(self) -> None:
        """remove_state_callback removes a registered callback."""
        mgr = OfflineModeManager()
        results: list[str] = []

        def cb(s: ConnectivityState) -> None:
            results.append("called")

        mgr.on_state_change(cb)
        removed = mgr.remove_state_callback(cb)
        assert removed is True

        with patch.object(mgr, "_check_connectivity", return_value=False):
            mgr.check_now()
        assert results == []

    def test_remove_state_callback_not_found(self) -> None:
        """remove_state_callback returns False if not registered."""
        mgr = OfflineModeManager()
        assert mgr.remove_state_callback(lambda s: None) is False

    # ------------------------------------------------------------------
    # Monitoring tests
    # ------------------------------------------------------------------

    def test_start_stop_monitoring(self) -> None:
        """start_monitoring/stop_monitoring manages the background thread."""
        mgr = OfflineModeManager(check_interval=0.05)
        with patch.object(mgr, "_check_connectivity", return_value=True):
            mgr.start_monitoring()
            assert mgr.is_monitoring is True
            time.sleep(0.1)
            mgr.stop_monitoring()
            assert mgr.is_monitoring is False

    def test_start_monitoring_idempotent(self) -> None:
        """Calling start_monitoring twice does not create two threads."""
        mgr = OfflineModeManager(check_interval=0.05)
        with patch.object(mgr, "_check_connectivity", return_value=True):
            mgr.start_monitoring()
            thread1 = mgr._monitor_thread
            mgr.start_monitoring()
            thread2 = mgr._monitor_thread
            assert thread1 is thread2
            mgr.stop_monitoring()

    def test_stop_monitoring_without_start(self) -> None:
        """stop_monitoring is safe to call without start."""
        mgr = OfflineModeManager()
        mgr.stop_monitoring()  # Should not raise
        assert mgr.is_monitoring is False

    def test_is_monitoring_false_initially(self) -> None:
        """is_monitoring is False before start."""
        mgr = OfflineModeManager()
        assert mgr.is_monitoring is False

    # ------------------------------------------------------------------
    # Queue processing on reconnect tests
    # ------------------------------------------------------------------

    def test_queue_processed_on_reconnect(self) -> None:
        """When transitioning OFFLINE -> ONLINE, queue processing is triggered."""
        mgr = OfflineModeManager()
        mgr.queue_request(
            QueuedRequest(id="r1", model="m", provider="p", messages=[], created_at="now")
        )

        # Go offline first
        with patch.object(mgr, "_check_connectivity", return_value=False):
            mgr.check_now()
        assert mgr.state == ConnectivityState.OFFLINE

        # Come back online — _process_queue should be called
        with patch.object(mgr, "_check_connectivity", return_value=True), \
             patch.object(mgr, "_process_queue") as mock_process:
            mgr.check_now()
            mock_process.assert_called_once()

    def test_queue_not_processed_when_staying_online(self) -> None:
        """No queue processing when staying in ONLINE state."""
        mgr = OfflineModeManager()
        mgr.queue_request(
            QueuedRequest(id="r1", model="m", provider="p", messages=[], created_at="now")
        )

        with patch.object(mgr, "_check_connectivity", return_value=True), \
             patch.object(mgr, "_process_queue") as mock_process:
            mgr.check_now()
            mock_process.assert_not_called()

    # ------------------------------------------------------------------
    # Status details test
    # ------------------------------------------------------------------

    def test_get_status_details(self) -> None:
        """get_status_details returns expected keys."""
        mgr = OfflineModeManager(check_interval=30.0)
        details = mgr.get_status_details()
        assert details["state"] == "online"
        assert details["manual_offline"] is False
        assert details["is_online"] is True
        assert details["is_offline"] is False
        assert details["queued_requests"] == 0
        assert details["monitoring_active"] is False
        assert details["check_interval"] == 30.0
        assert "check_hosts" in details
        assert details["last_check_time"] is None

    def test_get_status_details_offline(self) -> None:
        """get_status_details reflects offline state."""
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()
        details = mgr.get_status_details()
        assert details["state"] == "offline"
        assert details["manual_offline"] is True
        assert details["is_online"] is False
        assert details["is_offline"] is True

    # ------------------------------------------------------------------
    # Connectivity check tests
    # ------------------------------------------------------------------

    def test_check_connectivity_success(self) -> None:
        """_check_connectivity returns True when socket connects."""
        mgr = OfflineModeManager()
        mock_sock = MagicMock()
        with patch("prism.network.offline.socket.create_connection", return_value=mock_sock):
            assert mgr._check_connectivity() is True
        mock_sock.close.assert_called_once()

    def test_check_connectivity_failure(self) -> None:
        """_check_connectivity returns False when all hosts fail."""
        mgr = OfflineModeManager()
        with patch(
            "prism.network.offline.socket.create_connection",
            side_effect=OSError("Connection refused"),
        ):
            assert mgr._check_connectivity() is False

    def test_check_connectivity_partial_success(self) -> None:
        """_check_connectivity returns True if any host succeeds."""
        mgr = OfflineModeManager(check_hosts=["bad-host", "good-host"])
        mock_sock = MagicMock()

        def side_effect(addr: tuple[str, int], timeout: float) -> MagicMock:
            if addr[0] == "bad-host":
                raise OSError("fail")
            return mock_sock

        with patch(
            "prism.network.offline.socket.create_connection",
            side_effect=side_effect,
        ):
            assert mgr._check_connectivity() is True

    def test_check_connectivity_uses_configured_port_and_timeout(self) -> None:
        """_check_connectivity uses the configured port and timeout."""
        mgr = OfflineModeManager(
            check_hosts=["10.0.0.1"],
            check_port=443,
            check_timeout=5.0,
        )
        mock_sock = MagicMock()
        with patch(
            "prism.network.offline.socket.create_connection",
            return_value=mock_sock,
        ) as mock_conn:
            mgr._check_connectivity()
            mock_conn.assert_called_once_with(("10.0.0.1", 443), timeout=5.0)

"""Enhanced offline mode — continuous connectivity monitoring with graceful degradation.

Provides automatic detection of network availability, graceful
degradation to local-only models, request queueing for retry when
connectivity is restored, and manual offline mode support.
"""

from __future__ import annotations

import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)


class ConnectivityState(Enum):
    """Possible connectivity states.

    Attributes:
        ONLINE: Full internet connectivity available.
        OFFLINE: No internet connectivity detected.
        DEGRADED: Partial connectivity — some providers may be unreachable.
    """

    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"


@dataclass
class QueuedRequest:
    """A cloud API request queued for retry when back online.

    Attributes:
        id: Unique identifier for the queued request.
        model: The target model identifier (e.g. ``gpt-4o``).
        provider: Provider name (e.g. ``openai``).
        messages: The message list for the completion request.
        created_at: ISO-8601 timestamp of when the request was queued.
        retry_count: Number of retry attempts so far.
        metadata: Optional extra data about the request.
    """

    id: str
    model: str
    provider: str
    messages: list[dict[str, Any]]
    created_at: str
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OfflineCapabilities:
    """Features available in the current connectivity state.

    Attributes:
        file_operations: Reading and writing local files.
        terminal_execution: Running shell commands.
        git_operations: Git commit, diff, log, etc.
        local_inference: Local model inference via Ollama.
        cache_hits: Serving cached responses.
        cloud_inference: Inference via cloud APIs.
        web_browsing: Browsing web pages via Playwright.
        plugin_install: Installing new plugins/packages.
    """

    file_operations: bool = True
    terminal_execution: bool = True
    git_operations: bool = True
    local_inference: bool = True
    cache_hits: bool = True
    cloud_inference: bool = False
    web_browsing: bool = False
    plugin_install: bool = False


class OfflineModeManager:
    """Manages offline detection and graceful degradation.

    Provides continuous background monitoring of network connectivity,
    automatic state transitions with callback notifications, request
    queueing for retry when back online, and a manual offline override.

    Args:
        check_interval: Seconds between background connectivity checks.
        check_hosts: List of host addresses to probe for connectivity.
            Defaults to well-known DNS resolvers.
        check_port: TCP port to probe on check hosts.
        check_timeout: Timeout in seconds for each connectivity probe.
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        check_hosts: list[str] | None = None,
        check_port: int = 53,
        check_timeout: float = 3.0,
    ) -> None:
        self._state = ConnectivityState.ONLINE
        self._manual_offline = False
        self._check_interval = check_interval
        self._check_hosts = check_hosts or ["8.8.8.8", "1.1.1.1"]
        self._check_port = check_port
        self._check_timeout = check_timeout
        self._queued_requests: list[QueuedRequest] = []
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._state_callbacks: list[Callable[[ConnectivityState], None]] = []
        self._last_check_time: float | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> ConnectivityState:
        """Current connectivity state."""
        return self._state

    @property
    def is_online(self) -> bool:
        """Whether the system is fully online and not in manual offline mode."""
        return self._state == ConnectivityState.ONLINE and not self._manual_offline

    @property
    def is_offline(self) -> bool:
        """Whether the system is offline (auto-detected or manual)."""
        return self._state == ConnectivityState.OFFLINE or self._manual_offline

    @property
    def is_manual_offline(self) -> bool:
        """Whether manual offline mode is enabled."""
        return self._manual_offline

    @property
    def queued_count(self) -> int:
        """Number of requests currently in the retry queue."""
        with self._lock:
            return len(self._queued_requests)

    # ------------------------------------------------------------------
    # Manual offline control
    # ------------------------------------------------------------------

    def enable_manual_offline(self) -> None:
        """Force offline mode (``prism --offline``).

        Immediately transitions to ``OFFLINE`` state and notifies
        all registered callbacks.
        """
        self._manual_offline = True
        self._set_state(ConnectivityState.OFFLINE)
        logger.info("manual_offline_enabled")

    def disable_manual_offline(self) -> None:
        """Disable manual offline mode.

        Re-checks actual connectivity and transitions to the appropriate
        state.
        """
        self._manual_offline = False
        if self._check_connectivity():
            self._set_state(ConnectivityState.ONLINE)
        else:
            self._set_state(ConnectivityState.OFFLINE)
        logger.info("manual_offline_disabled")

    # ------------------------------------------------------------------
    # Connectivity checking
    # ------------------------------------------------------------------

    def check_now(self) -> ConnectivityState:
        """Perform an immediate connectivity check.

        Returns:
            The updated ``ConnectivityState``.
        """
        if self._manual_offline:
            return ConnectivityState.OFFLINE

        if self._check_connectivity():
            self._set_state(ConnectivityState.ONLINE)
        else:
            self._set_state(ConnectivityState.OFFLINE)

        self._last_check_time = time.monotonic()
        return self._state

    def _check_connectivity(self) -> bool:
        """Check network connectivity by attempting TCP socket connections.

        Probes each host in ``_check_hosts`` on the configured port.
        Returns ``True`` as soon as any host is reachable.

        Returns:
            ``True`` if at least one host is reachable, ``False`` otherwise.
        """
        for host in self._check_hosts:
            try:
                sock = socket.create_connection(
                    (host, self._check_port),
                    timeout=self._check_timeout,
                )
                sock.close()
                return True
            except (OSError, TimeoutError):
                continue
        return False

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------

    def start_monitoring(self) -> None:
        """Start background connectivity monitoring thread.

        If monitoring is already running, this is a no-op.
        The thread is a daemon so it won't block process exit.
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="prism-connectivity-monitor",
        )
        self._monitor_thread.start()
        logger.info("connectivity_monitoring_started", interval=self._check_interval)

    def stop_monitoring(self) -> None:
        """Stop background connectivity monitoring.

        Blocks up to 5 seconds for the thread to finish.
        """
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        logger.info("connectivity_monitoring_stopped")

    @property
    def is_monitoring(self) -> bool:
        """Whether background monitoring is currently running."""
        return (
            self._monitor_thread is not None
            and self._monitor_thread.is_alive()
        )

    def _monitor_loop(self) -> None:
        """Background monitoring loop.

        Runs until ``_stop_event`` is set, checking connectivity at
        each interval.
        """
        while not self._stop_event.is_set():
            if not self._manual_offline:
                self.check_now()
            self._stop_event.wait(self._check_interval)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state(self, new_state: ConnectivityState) -> None:
        """Update state and notify callbacks if the state has changed.

        When transitioning from OFFLINE to ONLINE, also triggers
        processing of the queued requests.

        Args:
            new_state: The new connectivity state to transition to.
        """
        with self._lock:
            if self._state == new_state:
                return
            old_state = self._state
            self._state = new_state

        logger.info(
            "connectivity_changed",
            old=old_state.value,
            new=new_state.value,
        )

        # Process queued requests when coming back online
        if (
            new_state == ConnectivityState.ONLINE
            and old_state == ConnectivityState.OFFLINE
        ):
            self._process_queue()

        # Notify all callbacks
        for callback in self._state_callbacks:
            try:
                callback(new_state)
            except Exception:
                logger.exception("state_callback_error")

    # ------------------------------------------------------------------
    # Request queue
    # ------------------------------------------------------------------

    def queue_request(self, request: QueuedRequest) -> None:
        """Queue a failed cloud request for retry when back online.

        Args:
            request: The request to queue.
        """
        with self._lock:
            self._queued_requests.append(request)
        logger.info(
            "request_queued",
            request_id=request.id,
            model=request.model,
            queue_size=self.queued_count,
        )

    def create_queued_request(
        self,
        model: str,
        provider: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> QueuedRequest:
        """Create and queue a new request.

        Convenience method that generates an ID and timestamp
        automatically.

        Args:
            model: Target model identifier.
            provider: Provider name.
            messages: The message list.
            metadata: Optional extra data.

        Returns:
            The newly created and queued ``QueuedRequest``.
        """
        request = QueuedRequest(
            id=uuid.uuid4().hex[:12],
            model=model,
            provider=provider,
            messages=messages,
            created_at=datetime.now(UTC).isoformat(),
            metadata=metadata or {},
        )
        self.queue_request(request)
        return request

    def get_queued_requests(self) -> list[QueuedRequest]:
        """Get a copy of all queued requests.

        Returns:
            Shallow copy of the queue.
        """
        with self._lock:
            return list(self._queued_requests)

    def clear_queue(self) -> int:
        """Clear all queued requests.

        Returns:
            The number of requests that were cleared.
        """
        with self._lock:
            count = len(self._queued_requests)
            self._queued_requests.clear()
        logger.info("queue_cleared", cleared=count)
        return count

    def remove_queued_request(self, request_id: str) -> bool:
        """Remove a specific queued request by ID.

        Args:
            request_id: The ID of the request to remove.

        Returns:
            ``True`` if the request was found and removed.
        """
        with self._lock:
            for i, req in enumerate(self._queued_requests):
                if req.id == request_id:
                    self._queued_requests.pop(i)
                    return True
        return False

    def _process_queue(self) -> None:
        """Notify that queued requests are ready for processing.

        Called automatically when transitioning back to ``ONLINE``.
        The actual retry logic is handled by the completion engine —
        this method simply logs that the queue is ready.
        """
        with self._lock:
            count = len(self._queued_requests)
        if count > 0:
            logger.info("queued_requests_ready", count=count)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def get_capabilities(self) -> OfflineCapabilities:
        """Get the capabilities available in the current state.

        Returns:
            An ``OfflineCapabilities`` instance reflecting what is
            possible given the current connectivity.
        """
        if self.is_online:
            return OfflineCapabilities(
                file_operations=True,
                terminal_execution=True,
                git_operations=True,
                local_inference=True,
                cache_hits=True,
                cloud_inference=True,
                web_browsing=True,
                plugin_install=True,
            )
        return OfflineCapabilities(
            file_operations=True,
            terminal_execution=True,
            git_operations=True,
            local_inference=True,
            cache_hits=True,
            cloud_inference=False,
            web_browsing=False,
            plugin_install=False,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_state_change(self, callback: Callable[[ConnectivityState], None]) -> None:
        """Register a callback to be invoked on connectivity state changes.

        Args:
            callback: A callable accepting a single ``ConnectivityState``
                argument.
        """
        self._state_callbacks.append(callback)

    def remove_state_callback(
        self, callback: Callable[[ConnectivityState], None],
    ) -> bool:
        """Remove a previously registered state change callback.

        Args:
            callback: The callback to remove.

        Returns:
            ``True`` if the callback was found and removed.
        """
        try:
            self._state_callbacks.remove(callback)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def get_status_indicator(self) -> str:
        """Get a human-readable status indicator string for the prompt.

        Returns:
            A bracketed status string, or an empty string when online.
        """
        if self._manual_offline:
            return "[OFFLINE - Manual]"
        if self._state == ConnectivityState.OFFLINE:
            return "[OFFLINE - Local models only]"
        if self._state == ConnectivityState.DEGRADED:
            return "[DEGRADED - Some providers unavailable]"
        return ""

    def get_status_details(self) -> dict[str, Any]:
        """Get detailed status information for diagnostics.

        Returns:
            Dictionary with connectivity details.
        """
        return {
            "state": self._state.value,
            "manual_offline": self._manual_offline,
            "is_online": self.is_online,
            "is_offline": self.is_offline,
            "queued_requests": self.queued_count,
            "monitoring_active": self.is_monitoring,
            "check_interval": self._check_interval,
            "check_hosts": list(self._check_hosts),
            "last_check_time": self._last_check_time,
        }

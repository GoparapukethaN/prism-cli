"""Network utilities — connectivity, offline routing, privacy mode, and proxy support."""

from __future__ import annotations

from prism.network.connectivity import ConnectivityChecker, OfflineRouter
from prism.network.offline import (
    ConnectivityState,
    OfflineCapabilities,
    OfflineModeManager,
    QueuedRequest,
)
from prism.network.privacy import PrivacyManager, PrivacyViolationError
from prism.network.proxy import ProxyConfig, ProxyManager

__all__ = [
    "ConnectivityChecker",
    "ConnectivityState",
    "OfflineCapabilities",
    "OfflineModeManager",
    "OfflineRouter",
    "PrivacyManager",
    "PrivacyViolationError",
    "ProxyConfig",
    "ProxyManager",
    "QueuedRequest",
]

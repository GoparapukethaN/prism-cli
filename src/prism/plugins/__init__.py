"""Plugin system — discovery, installation, and sandboxed execution.

Provides a plugin framework for extending Prism with community-contributed
tools, commands, and integrations.  Plugins are installed to
``~/.prism/plugins/`` and declared via a ``plugin.yaml`` manifest.
"""

from __future__ import annotations

from prism.plugins.manager import (
    PluginError,
    PluginInfo,
    PluginManager,
    PluginManifest,
    PluginNotFoundError,
    PluginValidationError,
)

__all__ = [
    "PluginError",
    "PluginInfo",
    "PluginManager",
    "PluginManifest",
    "PluginNotFoundError",
    "PluginValidationError",
]

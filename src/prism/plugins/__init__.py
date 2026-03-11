"""Plugin system — discovery, installation, and sandboxed execution.

Provides a plugin framework for extending Prism with community-contributed
tools, commands, and integrations.  Plugins are installed to
``~/.prism/plugins/`` and declared via a ``plugin.yaml`` manifest.
"""

from __future__ import annotations

from prism.plugins.api import (
    get_cost_summary,
    get_registered_commands,
    get_registered_tools,
    get_repo_map,
    log,
    register_command,
    register_tool,
)
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
    "get_cost_summary",
    "get_registered_commands",
    "get_registered_tools",
    "get_repo_map",
    "log",
    "register_command",
    "register_tool",
]

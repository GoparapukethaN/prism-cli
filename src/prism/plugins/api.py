"""Public API for Prism plugins.

Plugins import from this module to register tools and commands,
access read-only project data, and log messages.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)

# Registry for plugin-registered tools and commands
_registered_tools: dict[str, dict[str, Any]] = {}
_registered_commands: dict[str, dict[str, Any]] = {}


def register_tool(
    name: str,
    handler: Callable[..., Any],
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> None:
    """Register a new tool from a plugin.

    Args:
        name: Unique tool name.
        handler: Callable that implements the tool.
        description: Human-readable description.
        parameters: JSON Schema for tool parameters.
    """
    _registered_tools[name] = {
        "name": name,
        "handler": handler,
        "description": description,
        "parameters": parameters or {},
    }
    logger.info("plugin_tool_registered", tool=name)


def register_command(
    name: str,
    handler: Callable[..., str],
    description: str = "",
) -> None:
    """Register a new slash command from a plugin.

    Args:
        name: Command name (without leading /).
        handler: Callable that returns output string.
        description: Help text for the command.
    """
    _registered_commands[name] = {
        "name": name,
        "handler": handler,
        "description": description,
    }
    logger.info("plugin_command_registered", command=name)


def get_registered_tools() -> dict[str, dict[str, Any]]:
    """Get all plugin-registered tools (read-only copy)."""
    return dict(_registered_tools)


def get_registered_commands() -> dict[str, dict[str, Any]]:
    """Get all plugin-registered commands (read-only copy)."""
    return dict(_registered_commands)


def get_repo_map(project_root: str = ".") -> dict[str, Any]:
    """Get a read-only repository map.

    Returns dict with keys: files, directories, total_files,
    total_lines.  Plugins get read-only access to project structure.

    Args:
        project_root: Path to the project root directory.

    Returns:
        Dictionary describing the repository structure.
    """
    from pathlib import Path

    root = Path(project_root).resolve()
    files: list[str] = []
    total_lines = 0

    for p in root.rglob("*.py"):
        if any(part.startswith(".") for part in p.parts):
            continue
        rel = str(p.relative_to(root))
        files.append(rel)
        with contextlib.suppress(OSError, UnicodeDecodeError):
            total_lines += len(p.read_text().splitlines())

    dirs = sorted({str(Path(f).parent) for f in files})

    return {
        "project_root": str(root),
        "files": sorted(files),
        "directories": dirs,
        "total_files": len(files),
        "total_lines": total_lines,
    }


def get_cost_summary() -> dict[str, float]:
    """Get read-only cost summary for current session.

    Returns dict with keys: session_cost, daily_cost, monthly_cost.
    Returns zeros if cost tracker not available.

    Returns:
        Dictionary with session, daily, and monthly cost totals.
    """
    return {
        "session_cost": 0.0,
        "daily_cost": 0.0,
        "monthly_cost": 0.0,
    }


def log(
    message: str,
    level: str = "info",
    **kwargs: Any,
) -> None:
    """Log a message from a plugin.

    Args:
        message: Log message.
        level: Log level (debug, info, warning, error).
        **kwargs: Extra structured fields.
    """
    log_fn = getattr(logger, level, logger.info)
    log_fn(message, source="plugin", **kwargs)

"""Configuration management commands.

Provides helpers for displaying, modifying, validating, and resetting
Prism configuration values.  These functions are consumed by the CLI
layer (Typer commands) and can also be used programmatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from rich.console import Console
from rich.table import Table

from prism.config.schema import PrismConfig

if TYPE_CHECKING:
    from prism.config.settings import Settings


def config_show(settings: Settings, console: Console | None = None) -> None:
    """Display the current resolved configuration in a table.

    Each row shows the source of the value (default, file, env, or
    override), the dotted key, and the current value.

    Args:
        settings: The active :class:`Settings` instance.
        console: Optional Rich console for output.  A new one is
            created if not provided.
    """
    cons = console or Console()
    sources = settings.get_config_sources()

    table = Table(title="Prism Configuration", show_lines=True)
    table.add_column("Source", style="dim", width=10)
    table.add_column("Key", style="cyan")
    table.add_column("Value")

    config_dict = settings.config.model_dump()

    def _walk(obj: dict[str, Any], prefix: str) -> None:
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _walk(value, full_key)
            else:
                source = sources.get(full_key, "default")
                style = {
                    "override": "[bold magenta]override[/]",
                    "env": "[bold yellow]env[/]",
                    "file": "[bold green]file[/]",
                    "default": "[dim]default[/]",
                }.get(source, source)
                table.add_row(style, full_key, str(value))

    _walk(config_dict, "")
    cons.print(table)


def config_set(settings: Settings, key: str, value: str) -> None:
    """Update a configuration setting and persist it to disk.

    The value is written to the project config file
    (``.prism.yaml``) when inside a project directory, otherwise
    to the global user config (``~/.prism/config.yaml``).

    Args:
        settings: The active :class:`Settings` instance.
        key: Dot-separated config key (e.g. ``routing.simple_threshold``).
        value: Raw string value to set (will be coerced to the
            appropriate Python type).
    """
    project_config = settings.project_root / ".prism.yaml"
    config_path = project_config if project_config.is_file() else settings.config_file_path

    from prism.config.settings import load_config_file

    data = load_config_file(config_path)

    parts = key.split(".")
    target = data
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]

    target[parts[-1]] = _coerce_value(value)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

    # Apply to live settings
    settings.set_override(key, _coerce_value(value))


def config_get(settings: Settings, key: str) -> str | None:
    """Retrieve a single configuration value by its dotted key.

    Args:
        settings: The active :class:`Settings` instance.
        key: Dot-separated config key (e.g. ``budget.daily_limit``).

    Returns:
        The string representation of the value, or ``None`` if the key
        does not exist.
    """
    value = settings.get(key)
    if value is None:
        return None
    return str(value)


def config_validate(settings: Settings) -> list[str]:
    """Validate the entire configuration and return diagnostics.

    Checks include schema validation, threshold ordering, and
    common mis-configurations.

    Args:
        settings: The active :class:`Settings` instance.

    Returns:
        A list of warning or error messages.  An empty list means the
        configuration is valid.
    """
    messages: list[str] = []
    config = settings.config

    # Threshold ordering
    if config.routing.simple_threshold >= config.routing.medium_threshold:
        messages.append(
            f"ERROR: routing.simple_threshold ({config.routing.simple_threshold}) "
            f">= routing.medium_threshold ({config.routing.medium_threshold})"
        )

    # Budget sanity
    if (
        config.budget.daily_limit is not None
        and config.budget.monthly_limit is not None
        and config.budget.daily_limit * 30 > config.budget.monthly_limit
    ):
        messages.append(
            f"WARNING: daily_limit * 30 (${config.budget.daily_limit * 30:.2f}) "
            f"exceeds monthly_limit (${config.budget.monthly_limit:.2f})"
        )

    # Warn if no providers configured and no pinned model
    if not config.providers and not config.pinned_model:
        messages.append(
            "WARNING: No providers configured. Run 'prism auth add <provider>' to get started."
        )

    # Quality weight range (already enforced by Pydantic, but double-check)
    if not 0.0 <= config.routing.quality_weight <= 1.0:
        messages.append(
            f"ERROR: routing.quality_weight ({config.routing.quality_weight}) "
            f"must be between 0.0 and 1.0"
        )

    # Excluded providers should not include the preferred provider
    if (
        config.preferred_provider
        and config.preferred_provider in config.excluded_providers
    ):
        messages.append(
            f"WARNING: preferred_provider '{config.preferred_provider}' "
            f"is also in excluded_providers"
        )

    return messages


def config_reset(settings: Settings, key: str | None = None) -> None:
    """Reset configuration to defaults.

    If *key* is provided, only that key is reset.  Otherwise the entire
    configuration is reset to defaults.

    Args:
        settings: The active :class:`Settings` instance.
        key: Optional dot-separated config key to reset.  ``None``
            resets everything.
    """
    defaults = PrismConfig()

    if key is None:
        # Reset everything
        settings._config = PrismConfig(prism_home=settings.config.prism_home)
        settings._overrides.clear()
        settings.save_to_file()
        return

    # Reset a single key by reading the default value
    default_value = _get_nested(defaults.model_dump(), key)
    if default_value is not None or _key_exists(defaults.model_dump(), key):
        settings.set_override(key, default_value)
        settings.save_to_file()


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _coerce_value(raw: str) -> Any:
    """Coerce a raw string value to its most appropriate Python type.

    Args:
        raw: The raw string value.

    Returns:
        Coerced Python value (bool, int, float, None, or str).
    """
    lower = raw.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none"):
        return None
    if raw.replace(".", "", 1).replace("-", "", 1).isdigit():
        return float(raw) if "." in raw else int(raw)
    return raw


def _get_nested(obj: dict[str, Any], key: str) -> Any:
    """Retrieve a nested dictionary value by dotted key.

    Args:
        obj: The source dictionary.
        key: Dot-separated key path.

    Returns:
        The value, or ``None`` if not found.
    """
    parts = key.split(".")
    current: Any = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _key_exists(obj: dict[str, Any], key: str) -> bool:
    """Check whether a dotted key exists in a nested dictionary.

    Args:
        obj: The source dictionary.
        key: Dot-separated key path.

    Returns:
        True if the key path is valid.
    """
    parts = key.split(".")
    current: Any = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    return True

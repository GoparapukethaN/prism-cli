"""Configuration migration — auto-upgrade config format between versions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — used at runtime in constructor and methods

import structlog
import yaml

logger = structlog.get_logger(__name__)

CONFIG_VERSION = 2


@dataclass
class ConfigMigration:
    """A config migration step."""

    from_version: int
    to_version: int
    description: str


MIGRATIONS: list[ConfigMigration] = [
    ConfigMigration(0, 1, "Add budget and routing sections"),
    ConfigMigration(1, 2, "Add cache, privacy, and plugin sections"),
]


class ConfigMigrator:
    """Manages config file version migrations.

    Reads a YAML config file, determines its version, and applies any
    pending migrations to bring it up to the latest ``CONFIG_VERSION``.
    A backup of the pre-migration file is created before any changes
    are written.

    Args:
        config_path: Absolute path to the YAML configuration file.
    """

    def __init__(self, config_path: Path) -> None:
        self._path = config_path

    def get_version(self) -> int:
        """Get the current config file version.

        Returns:
            Integer version number. ``0`` if the file does not exist,
            is not valid YAML, or lacks a ``config_version`` key.
        """
        if not self._path.is_file():
            return 0
        try:
            data = yaml.safe_load(self._path.read_text())
            if not isinstance(data, dict):
                return 0
            return int(data.get("config_version", 0))
        except (yaml.YAMLError, OSError, ValueError, TypeError):
            return 0

    def needs_migration(self) -> bool:
        """Check if config needs migration.

        Returns:
            ``True`` if the on-disk version is lower than ``CONFIG_VERSION``.
        """
        return self.get_version() < CONFIG_VERSION

    def migrate(self) -> int:
        """Run all pending migrations.

        Creates a timestamped backup of the config file before applying
        any changes.  Migrations are applied sequentially from the
        current version up to ``CONFIG_VERSION``.

        Returns:
            The final config version after migration.
        """
        if not self._path.is_file():
            return CONFIG_VERSION

        try:
            raw_text = self._path.read_text()
            data: dict = yaml.safe_load(raw_text) or {}  # type: ignore[assignment]
        except (yaml.YAMLError, OSError):
            data = {}

        if not isinstance(data, dict):
            data = {}

        current = int(data.get("config_version", 0))

        if current >= CONFIG_VERSION:
            return current

        # Backup before migration
        backup_path = self._path.with_suffix(f".v{current}.bak")
        if self._path.is_file():
            backup_path.write_text(self._path.read_text())
            logger.info(
                "config_backup_created",
                backup=str(backup_path),
                version=current,
            )

        # Apply migrations
        for migration in MIGRATIONS:
            if current < migration.to_version:
                data = self._apply_migration(data, migration)
                current = migration.to_version
                logger.info(
                    "config_migrated",
                    from_v=migration.from_version,
                    to_v=migration.to_version,
                    description=migration.description,
                )

        data["config_version"] = current

        # Write migrated config
        with self._path.open("w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        return current

    def _apply_migration(self, data: dict, migration: ConfigMigration) -> dict:
        """Apply a single migration to the config data.

        Args:
            data: Current config dictionary.
            migration: The migration step to apply.

        Returns:
            Updated config dictionary.
        """
        if migration.to_version == 1:
            data.setdefault(
                "budget",
                {"daily_limit": 10.0, "monthly_limit": 50.0, "warn_at_percent": 70},
            )
            data.setdefault(
                "routing",
                {"prefer_cheap": True, "fallback_enabled": True},
            )
            data.setdefault("log_level", "INFO")

        elif migration.to_version == 2:
            data.setdefault("cache", {"enabled": True, "ttl_seconds": 3600})
            data.setdefault("privacy", {"mode": "normal"})
            data.setdefault("plugins", {"enabled": True, "auto_update": False})
            data.setdefault("offline", {"auto_detect": True, "check_interval": 30})

        return data

    def generate_default_config(self) -> str:
        """Generate a fully documented default configuration string.

        Returns:
            YAML-formatted default config with inline comments.
        """
        return """# Prism CLI Configuration
# Documentation: https://github.com/GoparapukethaN/prism-cli

# Config version (do not modify)
config_version: 2

# Budget limits (USD)
budget:
  daily_limit: 10.0      # Max spend per day
  monthly_limit: 50.0    # Max spend per month
  warn_at_percent: 70    # Warn when this % of budget used

# Routing preferences
routing:
  prefer_cheap: true     # Prefer cheaper models for simple tasks
  fallback_enabled: true # Enable fallback chain on failure

# Response cache
cache:
  enabled: true          # Enable response caching
  ttl_seconds: 3600      # Default cache TTL (1 hour)

# Privacy mode
privacy:
  mode: normal           # "normal" or "private" (Ollama only)

# Plugins
plugins:
  enabled: true          # Enable plugin system
  auto_update: false     # Auto-update plugins

# Offline mode
offline:
  auto_detect: true      # Auto-detect network availability
  check_interval: 30     # Connectivity check interval (seconds)

# Logging
log_level: INFO          # DEBUG, INFO, WARNING, ERROR
"""

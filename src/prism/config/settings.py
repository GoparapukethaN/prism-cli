"""Settings management — loading, merging, and accessing configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import ValidationError

from prism.config.schema import PrismConfig

logger = structlog.get_logger(__name__)


class Settings:
    """Central settings manager for Prism.

    Merges configuration from multiple sources in priority order:
    1. Explicit overrides (CLI flags, method arguments)
    2. Environment variables (PRISM_*)
    3. Project config (.prism.yaml in project root)
    4. User config (~/.prism/config.yaml)
    5. Default values
    """

    def __init__(
        self,
        config: PrismConfig | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._config = config or PrismConfig()
        self._project_root = project_root or Path.cwd()
        self._overrides: dict[str, Any] = {}

    @property
    def config(self) -> PrismConfig:
        """Get the merged configuration."""
        return self._config

    @property
    def project_root(self) -> Path:
        """Get the resolved project root directory."""
        return self._project_root.resolve()

    @property
    def prism_home(self) -> Path:
        """Get the Prism data directory (~/.prism)."""
        env_home = os.environ.get("PRISM_HOME")
        if env_home:
            return Path(env_home).expanduser().resolve()
        return self._config.prism_home.expanduser().resolve()

    @property
    def db_path(self) -> Path:
        """Get the SQLite database path."""
        return self.prism_home / "prism.db"

    @property
    def audit_log_path(self) -> Path:
        """Get the audit log file path."""
        return self.prism_home / "audit.log"

    @property
    def sessions_dir(self) -> Path:
        """Get the sessions directory."""
        return self.prism_home / "sessions"

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory."""
        return self.prism_home / "cache"

    @property
    def config_file_path(self) -> Path:
        """Get the user config file path."""
        return self.prism_home / "config.yaml"

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [
            self.prism_home,
            self.sessions_dir,
            self.cache_dir,
            self.cache_dir / "repo_maps",
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def set_override(self, key: str, value: Any) -> None:
        """Set a runtime override for a configuration value.

        Args:
            key: Dot-separated config key (e.g., 'routing.simple_threshold').
            value: The override value.
        """
        self._overrides[key] = value
        self._apply_overrides()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated key.

        Args:
            key: Dot-separated config key (e.g., 'budget.daily_limit').
            default: Default value if key not found.

        Returns:
            The configuration value.
        """
        if key in self._overrides:
            return self._overrides[key]

        parts = key.split(".")
        obj: Any = self._config
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part, default)
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default
        return obj

    def save_to_file(self, path: Path | None = None) -> Path:
        """Save the current configuration to a YAML file.

        Args:
            path: Target file path. Defaults to the user config file
                (``~/.prism/config.yaml``).

        Returns:
            The path the configuration was written to.
        """
        target = path or self.config_file_path
        target.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self._config.model_dump(mode="json")
        # Convert Path objects that survived serialisation to strings
        _stringify_paths(config_dict)

        with target.open("w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

        return target

    def get_config_sources(self) -> dict[str, str]:
        """Determine the source of each top-level configuration value.

        For each configuration key, reports whether the current value came
        from the defaults, an environment variable, a config file, or an
        explicit runtime override.

        Returns:
            Mapping of dot-separated config key to source label (one of
            ``"default"``, ``"env"``, ``"file"``, ``"override"``).
        """
        sources: dict[str, str] = {}
        defaults = PrismConfig()
        defaults_dict = defaults.model_dump()
        current_dict = self._config.model_dump()

        env_overrides = self._collect_env_overrides()

        def _walk(
            default_obj: dict[str, Any],
            current_obj: dict[str, Any],
            prefix: str,
        ) -> None:
            for key, curr_val in current_obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                def_val = default_obj.get(key)

                if full_key in self._overrides:
                    sources[full_key] = "override"
                elif full_key in env_overrides:
                    sources[full_key] = "env"
                elif isinstance(curr_val, dict) and isinstance(def_val, dict):
                    _walk(def_val, curr_val, full_key)
                    continue
                elif curr_val != def_val:
                    sources[full_key] = "file"
                else:
                    sources[full_key] = "default"

        _walk(defaults_dict, current_dict, "")
        return sources

    def _apply_overrides(self) -> None:
        """Apply environment variable and explicit overrides to config."""
        env_overrides = self._collect_env_overrides()
        merged = {**env_overrides, **self._overrides}

        if not merged:
            return

        config_dict = self._config.model_dump()
        for key, value in merged.items():
            parts = key.split(".")
            target = config_dict
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value

        try:
            self._config = PrismConfig(**config_dict)
        except ValidationError as e:
            logger.warning("config_override_failed", error=str(e))

    @staticmethod
    def _collect_env_overrides() -> dict[str, Any]:
        """Collect configuration overrides from PRISM_* environment variables."""
        overrides: dict[str, Any] = {}

        env_mapping: dict[str, tuple[str, type]] = {
            "PRISM_BUDGET_DAILY": ("budget.daily_limit", float),
            "PRISM_BUDGET_MONTHLY": ("budget.monthly_limit", float),
            "PRISM_MODEL": ("pinned_model", str),
            "PRISM_LOG_LEVEL": ("log_level", str),
        }

        for env_var, (config_key, value_type) in env_mapping.items():
            raw_value = os.environ.get(env_var)
            if raw_value is not None:
                try:
                    overrides[config_key] = value_type(raw_value)
                except (ValueError, TypeError):
                    logger.warning(
                        "invalid_env_override",
                        var=env_var,
                        value=raw_value,
                    )
        return overrides


def load_config_file(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary. Empty dict if file doesn't exist.
    """
    if not path.is_file():
        return {}

    try:
        with path.open("r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.warning("config_file_not_dict", path=str(path))
            return {}
        return data
    except yaml.YAMLError as e:
        logger.error("config_file_parse_error", path=str(path), error=str(e))
        return {}
    except OSError as e:
        logger.error("config_file_read_error", path=str(path), error=str(e))
        return {}


def load_settings(
    project_root: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load settings from all sources and merge them.

    Priority (highest first):
    1. config_overrides parameter
    2. Environment variables (PRISM_*)
    3. Project config (.prism.yaml in project root)
    4. User config (~/.prism/config.yaml)
    5. Default values

    Args:
        project_root: Project root directory. Defaults to cwd.
        config_overrides: Explicit overrides from CLI flags.

    Returns:
        Merged Settings instance.
    """
    resolved_root = (project_root or Path.cwd()).resolve()

    # Determine prism_home from env or default
    prism_home_str = os.environ.get("PRISM_HOME")
    prism_home = Path(prism_home_str).expanduser().resolve() if prism_home_str else Path.home() / ".prism"

    # Load user config
    user_config_path = prism_home / "config.yaml"
    user_config_data = load_config_file(user_config_path)

    # Load project config
    project_config_path = resolved_root / ".prism.yaml"
    project_config_data = load_config_file(project_config_path)

    # Merge: defaults ← user config ← project config
    merged_data: dict[str, Any] = {}
    _deep_merge(merged_data, user_config_data)
    _deep_merge(merged_data, project_config_data)

    # Parse into PrismConfig
    try:
        config = PrismConfig(**merged_data)
    except ValidationError as e:
        logger.warning("config_validation_error", error=str(e))
        config = PrismConfig()

    settings = Settings(config=config, project_root=resolved_root)

    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            settings.set_override(key, value)

    settings._apply_overrides()
    return settings


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Deep merge override dict into base dict (mutates base).

    Args:
        base: Base dictionary to merge into.
        override: Override dictionary whose values take precedence.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _stringify_paths(obj: dict[str, Any]) -> None:
    """Recursively convert :class:`Path` values to strings for YAML output.

    Args:
        obj: Dictionary to mutate in-place.
    """
    for key, value in obj.items():
        if isinstance(value, Path):
            obj[key] = str(value)
        elif isinstance(value, dict):
            _stringify_paths(value)

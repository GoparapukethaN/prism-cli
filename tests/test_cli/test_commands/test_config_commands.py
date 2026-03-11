"""Tests for config_commands — show, get, set, validate, reset, save, sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml
from rich.console import Console

from prism.cli.commands.config_commands import (
    config_get,
    config_reset,
    config_set,
    config_show,
    config_validate,
)
from prism.config.schema import PrismConfig
from prism.config.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Create a Settings instance with isolated paths."""
    prism_home = tmp_path / ".prism"
    prism_home.mkdir(parents=True)
    config = PrismConfig(prism_home=prism_home)
    return Settings(config=config, project_root=tmp_path)


# ------------------------------------------------------------------
# config_show()
# ------------------------------------------------------------------


class TestConfigShow:
    """Tests for config_show()."""

    def test_config_show(self, settings: Settings) -> None:
        """config_show runs without error and produces output."""
        console = Console(file=None, force_terminal=False, width=120)
        # Should not raise
        config_show(settings, console=console)

    def test_config_show_default_console(self, settings: Settings) -> None:
        """config_show works without an explicit console argument."""
        # Should not raise (creates its own Console internally)
        config_show(settings)


# ------------------------------------------------------------------
# config_get()
# ------------------------------------------------------------------


class TestConfigGet:
    """Tests for config_get()."""

    def test_config_get_existing(self, settings: Settings) -> None:
        """Retrieving an existing key returns its string value."""
        result = config_get(settings, "routing.simple_threshold")
        assert result == "0.3"

    def test_config_get_missing(self, settings: Settings) -> None:
        """Retrieving a nonexistent key returns None."""
        result = config_get(settings, "nonexistent.key")
        assert result is None

    def test_config_get_nested(self, settings: Settings) -> None:
        """Deeply nested keys resolve correctly."""
        result = config_get(settings, "budget.warn_at_percent")
        assert result == "80.0"

    def test_config_get_boolean(self, settings: Settings) -> None:
        """Boolean config values are returned as strings."""
        result = config_get(settings, "tools.web_enabled")
        assert result == "False"


# ------------------------------------------------------------------
# config_set()
# ------------------------------------------------------------------


class TestConfigSet:
    """Tests for config_set()."""

    def test_config_set(self, settings: Settings, tmp_path: Path) -> None:
        """Setting a key updates the live settings and writes to disk."""
        # Create a project config so config_set uses it
        project_config = tmp_path / ".prism.yaml"
        project_config.write_text(yaml.dump({"routing": {"simple_threshold": 0.3}}))

        config_set(settings, "routing.simple_threshold", "0.2")

        # Live override should be applied
        assert settings.get("routing.simple_threshold") == 0.2

        # File should be updated
        with project_config.open() as f:
            data = yaml.safe_load(f)
        assert data["routing"]["simple_threshold"] == 0.2

    def test_config_set_creates_new_key(self, settings: Settings, tmp_path: Path) -> None:
        """Setting a new nested key creates intermediate dicts."""
        project_config = tmp_path / ".prism.yaml"
        project_config.write_text(yaml.dump({}))

        config_set(settings, "budget.daily_limit", "15.0")
        assert settings.get("budget.daily_limit") == 15.0

    def test_config_set_boolean(self, settings: Settings, tmp_path: Path) -> None:
        """Boolean values are coerced correctly."""
        project_config = tmp_path / ".prism.yaml"
        project_config.write_text(yaml.dump({}))

        config_set(settings, "tools.web_enabled", "true")
        assert settings.get("tools.web_enabled") is True

    def test_config_set_falls_back_to_global(self, settings: Settings) -> None:
        """Without a project .prism.yaml, writes to the global config."""
        config_set(settings, "routing.simple_threshold", "0.15")
        assert settings.get("routing.simple_threshold") == 0.15


# ------------------------------------------------------------------
# config_validate()
# ------------------------------------------------------------------


class TestConfigValidate:
    """Tests for config_validate()."""

    def test_config_validate_clean(self, settings: Settings) -> None:
        """Default config should produce only the 'no providers' warning."""
        messages = config_validate(settings)
        # Defaults are valid but no providers are configured
        error_messages = [m for m in messages if m.startswith("ERROR")]
        assert len(error_messages) == 0

    def test_config_validate_warnings(self, tmp_path: Path) -> None:
        """Budget mismatch produces a warning."""
        config = PrismConfig(
            prism_home=tmp_path / ".prism",
            budget={"daily_limit": 100.0, "monthly_limit": 50.0},
        )
        s = Settings(config=config, project_root=tmp_path)
        messages = config_validate(s)
        warning_msgs = [m for m in messages if "daily_limit" in m]
        assert len(warning_msgs) >= 1

    def test_config_validate_excluded_preferred(self, tmp_path: Path) -> None:
        """Preferred provider in excluded list triggers a warning."""
        config = PrismConfig(
            prism_home=tmp_path / ".prism",
            preferred_provider="openai",
            excluded_providers=["openai"],
        )
        s = Settings(config=config, project_root=tmp_path)
        messages = config_validate(s)
        assert any("preferred_provider" in m for m in messages)


# ------------------------------------------------------------------
# config_reset()
# ------------------------------------------------------------------


class TestConfigReset:
    """Tests for config_reset()."""

    def test_config_reset_key(self, settings: Settings, tmp_path: Path) -> None:
        """Resetting a single key restores its default value."""
        settings.set_override("routing.simple_threshold", 0.1)
        assert settings.config.routing.simple_threshold == 0.1

        config_reset(settings, key="routing.simple_threshold")
        assert settings.get("routing.simple_threshold") == 0.3

    def test_config_reset_all(self, settings: Settings) -> None:
        """Resetting all keys restores all defaults."""
        settings.set_override("routing.simple_threshold", 0.1)
        settings.set_override("budget.daily_limit", 99.0)

        config_reset(settings, key=None)
        assert settings.config.routing.simple_threshold == 0.3
        assert settings.config.budget.daily_limit is None


# ------------------------------------------------------------------
# get_config_sources()
# ------------------------------------------------------------------


class TestConfigSources:
    """Tests for Settings.get_config_sources()."""

    def test_config_sources(self, settings: Settings) -> None:
        """Default config reports all sources as 'default'."""
        sources = settings.get_config_sources()
        assert "routing.simple_threshold" in sources
        assert sources["routing.simple_threshold"] == "default"

    def test_config_sources_with_override(self, settings: Settings) -> None:
        """Overridden keys report source as 'override'."""
        settings.set_override("routing.simple_threshold", 0.1)
        sources = settings.get_config_sources()
        assert sources["routing.simple_threshold"] == "override"


# ------------------------------------------------------------------
# save_to_file()
# ------------------------------------------------------------------


class TestSaveToFile:
    """Tests for Settings.save_to_file()."""

    def test_save_to_file(self, settings: Settings, tmp_path: Path) -> None:
        """Saving config creates a valid YAML file."""
        target = tmp_path / "output_config.yaml"
        result_path = settings.save_to_file(path=target)

        assert result_path == target
        assert target.is_file()

        with target.open() as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict)
        assert "routing" in data
        assert data["routing"]["simple_threshold"] == 0.3

    def test_save_to_file_default_path(self, settings: Settings) -> None:
        """Saving without a path writes to the default config location."""
        result_path = settings.save_to_file()
        assert result_path == settings.config_file_path
        assert result_path.is_file()

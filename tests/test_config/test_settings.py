"""Tests for configuration settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from prism.config.schema import BudgetConfig, PrismConfig, RoutingConfig
from prism.config.settings import Settings, _deep_merge, load_config_file, load_settings

if TYPE_CHECKING:
    from pathlib import Path


class TestPrismConfig:
    def test_default_config_is_valid(self) -> None:
        config = PrismConfig()
        assert config.routing.simple_threshold == 0.3
        assert config.routing.medium_threshold == 0.55
        assert config.budget.daily_limit is None
        assert config.tools.web_enabled is False

    def test_routing_threshold_validation(self) -> None:
        with pytest.raises(ValueError, match="simple_threshold"):
            RoutingConfig(simple_threshold=0.8, medium_threshold=0.7)

    def test_routing_equal_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError):
            RoutingConfig(simple_threshold=0.5, medium_threshold=0.5)

    def test_tool_use_minimum_tier_default(self) -> None:
        config = RoutingConfig()
        assert config.tool_use_minimum_tier == "medium"

    def test_escalate_on_tool_use_default(self) -> None:
        config = RoutingConfig()
        assert config.escalate_on_tool_use is True

    def test_tool_use_minimum_tier_valid_values(self) -> None:
        for tier in ("simple", "medium", "complex"):
            config = RoutingConfig(tool_use_minimum_tier=tier)
            assert config.tool_use_minimum_tier == tier

    def test_tool_use_minimum_tier_invalid_rejected(self) -> None:
        with pytest.raises(ValueError, match="tool_use_minimum_tier"):
            RoutingConfig(tool_use_minimum_tier="ultra")

    def test_escalate_on_tool_use_false(self) -> None:
        config = RoutingConfig(escalate_on_tool_use=False)
        assert config.escalate_on_tool_use is False

    def test_budget_limits_accept_none(self) -> None:
        budget = BudgetConfig(daily_limit=None, monthly_limit=None)
        assert budget.daily_limit is None

    def test_budget_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            BudgetConfig(daily_limit=-1.0)


class TestSettings:
    def test_create_with_defaults(self, tmp_path: Path) -> None:
        config = PrismConfig(prism_home=tmp_path / ".prism")
        settings = Settings(config=config, project_root=tmp_path)
        assert settings.project_root == tmp_path.resolve()

    def test_prism_home_default(self, tmp_path: Path) -> None:
        config = PrismConfig(prism_home=tmp_path / ".prism")
        settings = Settings(config=config, project_root=tmp_path)
        assert settings.prism_home == (tmp_path / ".prism").resolve()

    def test_db_path(self, tmp_path: Path) -> None:
        config = PrismConfig(prism_home=tmp_path / ".prism")
        settings = Settings(config=config, project_root=tmp_path)
        assert settings.db_path == (tmp_path / ".prism" / "prism.db").resolve()

    def test_ensure_directories(self, tmp_path: Path) -> None:
        config = PrismConfig(prism_home=tmp_path / ".prism")
        settings = Settings(config=config, project_root=tmp_path)
        settings.ensure_directories()
        assert settings.prism_home.is_dir()
        assert settings.sessions_dir.is_dir()
        assert settings.cache_dir.is_dir()

    def test_get_nested_value(self, tmp_path: Path) -> None:
        config = PrismConfig(prism_home=tmp_path / ".prism")
        settings = Settings(config=config, project_root=tmp_path)
        assert settings.get("routing.simple_threshold") == 0.3
        assert settings.get("budget.daily_limit") is None

    def test_get_missing_key_returns_default(self, tmp_path: Path) -> None:
        config = PrismConfig(prism_home=tmp_path / ".prism")
        settings = Settings(config=config, project_root=tmp_path)
        assert settings.get("nonexistent.key", "fallback") == "fallback"

    def test_set_override(self, tmp_path: Path) -> None:
        config = PrismConfig(prism_home=tmp_path / ".prism")
        settings = Settings(config=config, project_root=tmp_path)
        settings.set_override("routing.simple_threshold", 0.2)
        assert settings.config.routing.simple_threshold == 0.2

    def test_prism_home_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        custom_home = tmp_path / "custom_prism"
        monkeypatch.setenv("PRISM_HOME", str(custom_home))
        config = PrismConfig()
        settings = Settings(config=config, project_root=tmp_path)
        assert settings.prism_home == custom_home.resolve()


class TestLoadConfigFile:
    def test_nonexistent_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_config_file(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_valid_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"routing": {"simple_threshold": 0.2}, "budget": {"daily_limit": 5.0}})
        )
        result = load_config_file(config_file)
        assert result["routing"]["simple_threshold"] == 0.2
        assert result["budget"]["daily_limit"] == 5.0

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml::")
        result = load_config_file(config_file)
        assert result == {}

    def test_non_dict_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n")
        result = load_config_file(config_file)
        assert result == {}


class TestDeepMerge:
    def test_simple_merge(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        _deep_merge(base, override)
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"routing": {"threshold": 0.3, "mode": "auto"}}
        override = {"routing": {"threshold": 0.5}}
        _deep_merge(base, override)
        assert base == {"routing": {"threshold": 0.5, "mode": "auto"}}

    def test_override_replaces_non_dict(self) -> None:
        base = {"key": "old"}
        override = {"key": {"nested": "value"}}
        _deep_merge(base, override)
        assert base == {"key": {"nested": "value"}}


class TestLoadSettings:
    def test_load_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PRISM_HOME", str(tmp_path / ".prism"))
        settings = load_settings(project_root=tmp_path)
        assert settings.config.routing.simple_threshold == 0.3

    def test_load_with_user_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prism_home = tmp_path / ".prism"
        prism_home.mkdir(parents=True)
        config_file = prism_home / "config.yaml"
        config_file.write_text(
            yaml.dump({"routing": {"simple_threshold": 0.15}})
        )
        monkeypatch.setenv("PRISM_HOME", str(prism_home))
        settings = load_settings(project_root=tmp_path)
        assert settings.config.routing.simple_threshold == 0.15

    def test_load_with_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRISM_HOME", str(tmp_path / ".prism"))
        settings = load_settings(
            project_root=tmp_path,
            config_overrides={"routing.simple_threshold": 0.1},
        )
        assert settings.config.routing.simple_threshold == 0.1

    def test_env_budget_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRISM_HOME", str(tmp_path / ".prism"))
        monkeypatch.setenv("PRISM_BUDGET_DAILY", "10.0")
        settings = load_settings(project_root=tmp_path)
        assert settings.config.budget.daily_limit == 10.0

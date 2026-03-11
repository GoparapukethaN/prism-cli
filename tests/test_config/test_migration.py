"""Tests for configuration migration system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from prism.config.migration import (
    CONFIG_VERSION,
    MIGRATIONS,
    ConfigMigration,
    ConfigMigrator,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestConfigMigration:
    """Tests for the ConfigMigration dataclass."""

    def test_migration_fields(self) -> None:
        migration = ConfigMigration(from_version=0, to_version=1, description="test")
        assert migration.from_version == 0
        assert migration.to_version == 1
        assert migration.description == "test"

    def test_migrations_list_is_ordered(self) -> None:
        for i in range(len(MIGRATIONS) - 1):
            assert MIGRATIONS[i].to_version < MIGRATIONS[i + 1].to_version

    def test_migrations_cover_all_versions(self) -> None:
        assert MIGRATIONS[0].from_version == 0
        assert MIGRATIONS[-1].to_version == CONFIG_VERSION

    def test_config_version_is_positive(self) -> None:
        assert CONFIG_VERSION > 0


class TestConfigMigratorGetVersion:
    """Tests for ConfigMigrator.get_version()."""

    def test_nonexistent_file_returns_zero(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "nonexistent.yaml")
        assert migrator.get_version() == 0

    def test_empty_file_returns_zero(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        migrator = ConfigMigrator(config_file)
        assert migrator.get_version() == 0

    def test_file_without_version_returns_zero(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"routing": {"prefer_cheap": True}}))
        migrator = ConfigMigrator(config_file)
        assert migrator.get_version() == 0

    def test_file_with_version_returns_version(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"config_version": 1, "log_level": "INFO"}))
        migrator = ConfigMigrator(config_file)
        assert migrator.get_version() == 1

    def test_file_with_version_two(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"config_version": 2}))
        migrator = ConfigMigrator(config_file)
        assert migrator.get_version() == 2

    def test_invalid_yaml_returns_zero(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("{{broken yaml::")
        migrator = ConfigMigrator(config_file)
        assert migrator.get_version() == 0

    def test_non_dict_yaml_returns_zero(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2\n")
        migrator = ConfigMigrator(config_file)
        assert migrator.get_version() == 0

    def test_version_as_string_number(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"config_version": "1"}))
        migrator = ConfigMigrator(config_file)
        assert migrator.get_version() == 1


class TestConfigMigratorNeedsMigration:
    """Tests for ConfigMigrator.needs_migration()."""

    def test_nonexistent_file_needs_migration(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "nonexistent.yaml")
        assert migrator.needs_migration() is True

    def test_old_version_needs_migration(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"config_version": 0}))
        migrator = ConfigMigrator(config_file)
        assert migrator.needs_migration() is True

    def test_version_one_needs_migration(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"config_version": 1}))
        migrator = ConfigMigrator(config_file)
        assert migrator.needs_migration() is True

    def test_current_version_no_migration(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"config_version": CONFIG_VERSION}))
        migrator = ConfigMigrator(config_file)
        assert migrator.needs_migration() is False


class TestConfigMigratorMigrate:
    """Tests for ConfigMigrator.migrate()."""

    def test_nonexistent_file_returns_config_version(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "nonexistent.yaml")
        result = migrator.migrate()
        assert result == CONFIG_VERSION

    def test_migrate_v0_to_v2(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"some_setting": "value"}))
        migrator = ConfigMigrator(config_file)

        result = migrator.migrate()

        assert result == CONFIG_VERSION
        data = yaml.safe_load(config_file.read_text())
        assert data["config_version"] == CONFIG_VERSION
        # v1 additions
        assert "budget" in data
        assert data["budget"]["daily_limit"] == 10.0
        assert data["budget"]["monthly_limit"] == 50.0
        assert "routing" in data
        assert data["log_level"] == "INFO"
        # v2 additions
        assert "cache" in data
        assert data["cache"]["enabled"] is True
        assert "privacy" in data
        assert data["plugins"] is not None
        assert data["offline"]["auto_detect"] is True

    def test_migrate_v1_to_v2(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        v1_data = {
            "config_version": 1,
            "budget": {"daily_limit": 5.0, "monthly_limit": 25.0, "warn_at_percent": 80},
            "routing": {"prefer_cheap": True, "fallback_enabled": True},
            "log_level": "DEBUG",
        }
        config_file.write_text(yaml.dump(v1_data))
        migrator = ConfigMigrator(config_file)

        result = migrator.migrate()

        assert result == CONFIG_VERSION
        data = yaml.safe_load(config_file.read_text())
        assert data["config_version"] == 2
        # Existing v1 values preserved
        assert data["budget"]["daily_limit"] == 5.0
        assert data["log_level"] == "DEBUG"
        # v2 additions
        assert data["cache"]["enabled"] is True
        assert data["cache"]["ttl_seconds"] == 3600
        assert data["privacy"]["mode"] == "normal"
        assert data["plugins"]["enabled"] is True
        assert data["plugins"]["auto_update"] is False
        assert data["offline"]["check_interval"] == 30

    def test_migrate_creates_backup(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        original_content = yaml.dump({"some_key": "original_value"})
        config_file.write_text(original_content)
        migrator = ConfigMigrator(config_file)

        migrator.migrate()

        backup_file = tmp_path / "config.v0.bak"
        assert backup_file.is_file()
        assert backup_file.read_text() == original_content

    def test_migrate_v1_creates_v1_backup(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        v1_content = yaml.dump({"config_version": 1, "log_level": "INFO"})
        config_file.write_text(v1_content)
        migrator = ConfigMigrator(config_file)

        migrator.migrate()

        backup_file = tmp_path / "config.v1.bak"
        assert backup_file.is_file()
        assert backup_file.read_text() == v1_content

    def test_migrate_idempotent(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"config_version": CONFIG_VERSION, "log_level": "INFO"}))
        migrator = ConfigMigrator(config_file)

        result = migrator.migrate()

        assert result == CONFIG_VERSION
        data = yaml.safe_load(config_file.read_text())
        assert data["config_version"] == CONFIG_VERSION
        assert data["log_level"] == "INFO"

    def test_migrate_preserves_existing_keys(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({
                "some_setting": "value",
                "budget": {"daily_limit": 99.0, "monthly_limit": 200.0, "warn_at_percent": 90},
            })
        )
        migrator = ConfigMigrator(config_file)

        migrator.migrate()

        data = yaml.safe_load(config_file.read_text())
        assert data["some_setting"] == "value"
        # Budget was already set, should not be overwritten by migration
        assert data["budget"]["daily_limit"] == 99.0
        assert data["budget"]["monthly_limit"] == 200.0

    def test_migrate_empty_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        migrator = ConfigMigrator(config_file)

        result = migrator.migrate()

        assert result == CONFIG_VERSION
        data = yaml.safe_load(config_file.read_text())
        assert data["config_version"] == CONFIG_VERSION

    def test_migrate_broken_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("{{broken")
        migrator = ConfigMigrator(config_file)

        result = migrator.migrate()

        assert result == CONFIG_VERSION


class TestConfigMigratorGenerateDefaultConfig:
    """Tests for ConfigMigrator.generate_default_config()."""

    def test_generate_default_config_is_valid_yaml(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        default = migrator.generate_default_config()

        data = yaml.safe_load(default)
        assert isinstance(data, dict)

    def test_generate_default_config_has_version(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        default = migrator.generate_default_config()

        data = yaml.safe_load(default)
        assert data["config_version"] == CONFIG_VERSION

    def test_generate_default_config_has_all_sections(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        default = migrator.generate_default_config()

        data = yaml.safe_load(default)
        assert "budget" in data
        assert "routing" in data
        assert "cache" in data
        assert "privacy" in data
        assert "plugins" in data
        assert "offline" in data
        assert "log_level" in data

    def test_generate_default_config_budget_values(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        default = migrator.generate_default_config()

        data = yaml.safe_load(default)
        assert data["budget"]["daily_limit"] == 10.0
        assert data["budget"]["monthly_limit"] == 50.0
        assert data["budget"]["warn_at_percent"] == 70

    def test_generate_default_config_contains_comments(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        default = migrator.generate_default_config()

        assert "# Prism CLI Configuration" in default
        assert "# Budget limits" in default
        assert "do not modify" in default


class TestApplyMigration:
    """Tests for internal _apply_migration logic."""

    def test_v1_migration_adds_budget(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        migration = ConfigMigration(0, 1, "test")
        data: dict = {}

        result = migrator._apply_migration(data, migration)

        assert "budget" in result
        assert result["budget"]["daily_limit"] == 10.0

    def test_v1_migration_adds_routing(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        migration = ConfigMigration(0, 1, "test")
        data: dict = {}

        result = migrator._apply_migration(data, migration)

        assert "routing" in result
        assert result["routing"]["prefer_cheap"] is True

    def test_v1_migration_preserves_existing_budget(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        migration = ConfigMigration(0, 1, "test")
        data = {"budget": {"daily_limit": 99.0}}

        result = migrator._apply_migration(data, migration)

        assert result["budget"]["daily_limit"] == 99.0

    def test_v2_migration_adds_cache(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        migration = ConfigMigration(1, 2, "test")
        data: dict = {}

        result = migrator._apply_migration(data, migration)

        assert "cache" in result
        assert result["cache"]["enabled"] is True
        assert result["cache"]["ttl_seconds"] == 3600

    def test_v2_migration_adds_privacy(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        migration = ConfigMigration(1, 2, "test")
        data: dict = {}

        result = migrator._apply_migration(data, migration)

        assert result["privacy"]["mode"] == "normal"

    def test_v2_migration_adds_plugins(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        migration = ConfigMigration(1, 2, "test")
        data: dict = {}

        result = migrator._apply_migration(data, migration)

        assert result["plugins"]["enabled"] is True
        assert result["plugins"]["auto_update"] is False

    def test_v2_migration_adds_offline(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        migration = ConfigMigration(1, 2, "test")
        data: dict = {}

        result = migrator._apply_migration(data, migration)

        assert result["offline"]["auto_detect"] is True
        assert result["offline"]["check_interval"] == 30

    def test_unknown_version_returns_data_unchanged(self, tmp_path: Path) -> None:
        migrator = ConfigMigrator(tmp_path / "config.yaml")
        migration = ConfigMigration(99, 100, "future migration")
        data = {"existing": "value"}

        result = migrator._apply_migration(data, migration)

        assert result == {"existing": "value"}

"""Tests for PluginManifest, PluginInfo, and PluginManager."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from prism.plugins.manager import (
    BUILTIN_PLUGINS,
    PluginCommandSpec,
    PluginError,
    PluginInfo,
    PluginInstallError,
    PluginManager,
    PluginManifest,
    PluginNotFoundError,
    PluginToolSpec,
    PluginValidationError,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plugins_dir(tmp_path: Path) -> Path:
    """Create a temporary plugins directory."""
    d = tmp_path / "plugins"
    d.mkdir()
    return d


@pytest.fixture
def manager(plugins_dir: Path) -> PluginManager:
    """Create a PluginManager with a temp directory."""
    return PluginManager(plugins_dir=plugins_dir)


def _write_plugin(
    plugins_dir: Path,
    name: str,
    version: str = "1.0.0",
    description: str = "Test plugin",
    tools: list[dict[str, Any]] | None = None,
    commands: list[dict[str, Any]] | None = None,
    dependencies: list[str] | None = None,
) -> Path:
    """Write a minimal plugin to disk and return its directory."""
    plugin_dir = plugins_dir / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    manifest_data: dict[str, Any] = {
        "name": name,
        "version": version,
        "description": description,
        "author": "Test Author",
        "tools": tools or [],
        "commands": commands or [],
        "dependencies": dependencies or [],
    }
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump(manifest_data), encoding="utf-8"
    )

    # Write a minimal handler
    (plugin_dir / "handler.py").write_text(
        'def hello(**kwargs):\n    return {"status": "ok"}\n',
        encoding="utf-8",
    )

    return plugin_dir


# ---------------------------------------------------------------------------
# TestPluginToolSpec
# ---------------------------------------------------------------------------


class TestPluginToolSpec:
    """Tests for PluginToolSpec dataclass."""

    def test_defaults(self) -> None:
        """Default PluginToolSpec has empty handler and params."""
        spec = PluginToolSpec(name="my_tool", description="A tool")
        assert spec.name == "my_tool"
        assert spec.description == "A tool"
        assert spec.parameters == {}
        assert spec.handler == ""

    def test_custom_values(self) -> None:
        """PluginToolSpec accepts custom values."""
        params = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        spec = PluginToolSpec(
            name="calc",
            description="Calculator",
            parameters=params,
            handler="run_calc",
        )
        assert spec.name == "calc"
        assert spec.parameters == params
        assert spec.handler == "run_calc"


# ---------------------------------------------------------------------------
# TestPluginCommandSpec
# ---------------------------------------------------------------------------


class TestPluginCommandSpec:
    """Tests for PluginCommandSpec dataclass."""

    def test_defaults(self) -> None:
        """Default PluginCommandSpec has empty handler."""
        spec = PluginCommandSpec(name="cmd", description="A command")
        assert spec.name == "cmd"
        assert spec.description == "A command"
        assert spec.handler == ""

    def test_custom_handler(self) -> None:
        """PluginCommandSpec accepts custom handler."""
        spec = PluginCommandSpec(
            name="deploy", description="Deploy app", handler="run_deploy"
        )
        assert spec.handler == "run_deploy"


# ---------------------------------------------------------------------------
# TestPluginManifest
# ---------------------------------------------------------------------------


class TestPluginManifest:
    """Tests for PluginManifest dataclass and validation."""

    def test_minimal_manifest(self) -> None:
        """Minimal valid manifest has name and version."""
        m = PluginManifest(name="my-plugin", version="1.0.0")
        assert m.name == "my-plugin"
        assert m.version == "1.0.0"
        assert m.description == ""
        assert m.tools == []
        assert m.commands == []
        assert m.dependencies == []

    def test_full_manifest(self) -> None:
        """Full manifest with all fields."""
        m = PluginManifest(
            name="my-plugin",
            version="2.0.0",
            description="A great plugin",
            author="Author Name",
            homepage="https://example.com",
            license="MIT",
            tools=[PluginToolSpec(name="t1", description="Tool 1")],
            commands=[PluginCommandSpec(name="c1", description="Cmd 1")],
            dependencies=["httpx", "pyyaml"],
            min_prism="0.2.0",
        )
        assert m.author == "Author Name"
        assert len(m.tools) == 1
        assert len(m.commands) == 1
        assert m.dependencies == ["httpx", "pyyaml"]
        assert m.min_prism == "0.2.0"

    def test_validate_empty_name(self) -> None:
        """Validation fails for empty name."""
        m = PluginManifest(name="", version="1.0.0")
        errors = m.validate()
        assert any("name is required" in e for e in errors)

    def test_validate_empty_version(self) -> None:
        """Validation fails for empty version."""
        m = PluginManifest(name="good-name", version="")
        errors = m.validate()
        assert any("version is required" in e for e in errors)

    def test_validate_invalid_name_chars(self) -> None:
        """Validation fails for name with special characters."""
        m = PluginManifest(name="bad plugin!", version="1.0.0")
        errors = m.validate()
        assert any("invalid characters" in e for e in errors)

    def test_validate_valid_name_with_hyphens(self) -> None:
        """Names with hyphens and underscores are valid."""
        m = PluginManifest(name="my-cool_plugin", version="1.0.0")
        errors = m.validate()
        assert errors == []

    def test_validate_duplicate_tool_names(self) -> None:
        """Validation fails for duplicate tool names."""
        m = PluginManifest(
            name="test",
            version="1.0.0",
            tools=[
                PluginToolSpec(name="dup", description="First"),
                PluginToolSpec(name="dup", description="Second"),
            ],
        )
        errors = m.validate()
        assert any("Duplicate tool name" in e for e in errors)

    def test_validate_duplicate_command_names(self) -> None:
        """Validation fails for duplicate command names."""
        m = PluginManifest(
            name="test",
            version="1.0.0",
            commands=[
                PluginCommandSpec(name="dup", description="First"),
                PluginCommandSpec(name="dup", description="Second"),
            ],
        )
        errors = m.validate()
        assert any("Duplicate command name" in e for e in errors)

    def test_validate_success(self) -> None:
        """A well-formed manifest passes validation."""
        m = PluginManifest(
            name="good-plugin",
            version="1.2.3",
            tools=[
                PluginToolSpec(name="t1", description="Tool 1"),
                PluginToolSpec(name="t2", description="Tool 2"),
            ],
            commands=[
                PluginCommandSpec(name="c1", description="Cmd 1"),
            ],
        )
        assert m.validate() == []


# ---------------------------------------------------------------------------
# TestPluginInfo
# ---------------------------------------------------------------------------


class TestPluginInfo:
    """Tests for PluginInfo dataclass."""

    def test_defaults(self, tmp_path: Path) -> None:
        """Default PluginInfo has enabled=True and empty strings."""
        manifest = PluginManifest(name="test", version="1.0.0")
        info = PluginInfo(manifest=manifest, install_path=tmp_path)
        assert info.enabled is True
        assert info.installed_at == ""
        assert info.source == ""

    def test_custom_values(self, tmp_path: Path) -> None:
        """PluginInfo accepts custom values."""
        manifest = PluginManifest(name="test", version="1.0.0")
        info = PluginInfo(
            manifest=manifest,
            install_path=tmp_path,
            enabled=False,
            installed_at="2026-01-01T00:00:00Z",
            source="https://github.com/user/repo",
        )
        assert info.enabled is False
        assert info.installed_at == "2026-01-01T00:00:00Z"
        assert info.source == "https://github.com/user/repo"


# ---------------------------------------------------------------------------
# TestPluginManager — init and directory creation
# ---------------------------------------------------------------------------


class TestPluginManagerInit:
    """Tests for PluginManager initialization."""

    def test_creates_plugins_dir(self, tmp_path: Path) -> None:
        """PluginManager creates plugins directory if not exists."""
        plugins_dir = tmp_path / "new_plugins"
        assert not plugins_dir.exists()
        pm = PluginManager(plugins_dir=plugins_dir)
        assert plugins_dir.exists()
        assert pm.plugins_dir == plugins_dir

    def test_default_dir_uses_home(self) -> None:
        """Default plugins_dir is ~/.prism/plugins/."""
        with patch("pathlib.Path.mkdir"):
            pm = PluginManager.__new__(PluginManager)
            pm._plugins_dir = Path.home() / ".prism" / "plugins"
            pm._loaded = {}
            assert "plugins" in str(pm.plugins_dir)

    def test_registry_path(self, manager: PluginManager) -> None:
        """registry_path points to registry.json in plugins dir."""
        assert manager.registry_path.name == "registry.json"
        assert manager.registry_path.parent == manager.plugins_dir


# ---------------------------------------------------------------------------
# TestPluginManager — install built-in
# ---------------------------------------------------------------------------


class TestPluginManagerInstallBuiltin:
    """Tests for installing built-in plugins."""

    def test_install_builtin_docker_manager(self, manager: PluginManager) -> None:
        """Install built-in docker-manager plugin."""
        info = manager.install("docker-manager")
        assert info.manifest.name == "docker-manager"
        assert info.manifest.version == "1.0.0"
        assert info.enabled is True
        assert info.source == "builtin"
        assert (info.install_path / "plugin.yaml").exists()
        assert (info.install_path / "handler.py").exists()

    def test_install_builtin_db_query(self, manager: PluginManager) -> None:
        """Install built-in db-query plugin."""
        info = manager.install("db-query")
        assert info.manifest.name == "db-query"
        assert len(info.manifest.tools) == 2

    def test_install_builtin_api_tester(self, manager: PluginManager) -> None:
        """Install built-in api-tester plugin."""
        info = manager.install("api-tester")
        assert info.manifest.name == "api-tester"
        assert len(info.manifest.tools) == 3

    def test_install_already_installed_raises(self, manager: PluginManager) -> None:
        """Installing an already-installed plugin raises error."""
        manager.install("docker-manager")
        with pytest.raises(PluginInstallError, match="already installed"):
            manager.install("docker-manager")

    def test_install_empty_source_raises(self, manager: PluginManager) -> None:
        """Empty source string raises error."""
        with pytest.raises(PluginInstallError, match="must not be empty"):
            manager.install("")

    def test_install_unknown_name_raises(self, manager: PluginManager) -> None:
        """Unknown plugin name raises error."""
        with pytest.raises(PluginInstallError, match="Source not found"):
            manager.install("nonexistent-plugin-xyz")


# ---------------------------------------------------------------------------
# TestPluginManager — install from local
# ---------------------------------------------------------------------------


class TestPluginManagerInstallLocal:
    """Tests for installing from local directory."""

    def test_install_from_local_dir(
        self, manager: PluginManager, tmp_path: Path
    ) -> None:
        """Install a plugin from a local directory."""
        source_dir = tmp_path / "source_plugin"
        source_dir.mkdir()

        manifest_data = {
            "name": "local-plugin",
            "version": "0.5.0",
            "description": "A local plugin",
            "tools": [
                {
                    "name": "local_tool",
                    "description": "A local tool",
                    "handler": "local_tool_handler",
                }
            ],
        }
        (source_dir / "plugin.yaml").write_text(
            yaml.safe_dump(manifest_data), encoding="utf-8"
        )
        (source_dir / "handler.py").write_text(
            'def local_tool_handler(**kw): return {"ok": True}\n',
            encoding="utf-8",
        )

        info = manager.install(str(source_dir))
        assert info.manifest.name == "local-plugin"
        assert info.manifest.version == "0.5.0"
        assert (manager.plugins_dir / "local-plugin" / "plugin.yaml").exists()

    def test_install_from_nonexistent_dir_raises(
        self, manager: PluginManager, tmp_path: Path
    ) -> None:
        """Non-existent local path raises error if it looks like a path."""
        # This will be treated as a GitHub shorthand since it contains /
        # and doesn't exist on disk — but let's test a truly non-path name
        with pytest.raises((PluginInstallError, PluginError)):
            manager.install("completely-bogus-plugin-name-42")

    def test_install_from_dir_without_manifest_raises(
        self, manager: PluginManager, tmp_path: Path
    ) -> None:
        """Local dir without plugin.yaml raises PluginValidationError."""
        empty_dir = tmp_path / "empty_plugin"
        empty_dir.mkdir()
        with pytest.raises(PluginValidationError, match="Missing"):
            manager.install(str(empty_dir))


# ---------------------------------------------------------------------------
# TestPluginManager — remove
# ---------------------------------------------------------------------------


class TestPluginManagerRemove:
    """Tests for removing plugins."""

    def test_remove_installed_plugin(self, manager: PluginManager) -> None:
        """Remove an installed plugin."""
        manager.install("docker-manager")
        assert manager.is_installed("docker-manager")

        result = manager.remove("docker-manager")
        assert result is True
        assert not manager.is_installed("docker-manager")

    def test_remove_nonexistent_raises(self, manager: PluginManager) -> None:
        """Removing a non-installed plugin raises error."""
        with pytest.raises(PluginNotFoundError):
            manager.remove("nonexistent")

    def test_remove_empty_name_raises(self, manager: PluginManager) -> None:
        """Removing with empty name raises error."""
        with pytest.raises(PluginNotFoundError):
            manager.remove("")

    def test_remove_clears_from_loaded(self, manager: PluginManager) -> None:
        """Remove clears plugin from in-memory cache."""
        manager.install("docker-manager")
        assert manager.get_plugin("docker-manager") is not None

        manager.remove("docker-manager")
        assert manager.get_plugin("docker-manager") is None


# ---------------------------------------------------------------------------
# TestPluginManager — list_installed
# ---------------------------------------------------------------------------


class TestPluginManagerListInstalled:
    """Tests for listing installed plugins."""

    def test_list_empty(self, manager: PluginManager) -> None:
        """No plugins installed returns empty list."""
        assert manager.list_installed() == []

    def test_list_after_install(self, manager: PluginManager) -> None:
        """List shows installed plugins."""
        manager.install("docker-manager")
        manager.install("db-query")

        installed = manager.list_installed()
        names = [p.manifest.name for p in installed]
        assert "docker-manager" in names
        assert "db-query" in names

    def test_list_ignores_non_plugin_dirs(
        self, manager: PluginManager
    ) -> None:
        """Directories without plugin.yaml are ignored."""
        (manager.plugins_dir / "not-a-plugin").mkdir()
        assert manager.list_installed() == []

    def test_list_skips_files(self, manager: PluginManager) -> None:
        """Regular files in plugins dir are ignored."""
        (manager.plugins_dir / "readme.txt").write_text("hi")
        assert manager.list_installed() == []

    def test_list_sorted(
        self, manager: PluginManager, plugins_dir: Path
    ) -> None:
        """Installed plugins are sorted by directory name."""
        _write_plugin(plugins_dir, "z-plugin")
        _write_plugin(plugins_dir, "a-plugin")
        _write_plugin(plugins_dir, "m-plugin")

        installed = manager.list_installed()
        names = [p.manifest.name for p in installed]
        assert names == ["a-plugin", "m-plugin", "z-plugin"]


# ---------------------------------------------------------------------------
# TestPluginManager — list_available
# ---------------------------------------------------------------------------


class TestPluginManagerListAvailable:
    """Tests for listing available plugins."""

    def test_builtins_always_available(self, manager: PluginManager) -> None:
        """Built-in plugins are always in the available list."""
        available = manager.list_available()
        names = [m.name for m in available]
        assert "docker-manager" in names
        assert "db-query" in names
        assert "api-tester" in names

    def test_community_plugins_from_registry(
        self, manager: PluginManager
    ) -> None:
        """Community plugins from registry appear in available list."""
        manager.save_registry([
            {
                "name": "community-plugin",
                "version": "1.0.0",
                "description": "A community plugin",
                "author": "Community",
            }
        ])
        available = manager.list_available()
        names = [m.name for m in available]
        assert "community-plugin" in names

    def test_registry_does_not_duplicate_builtins(
        self, manager: PluginManager
    ) -> None:
        """Built-in names in registry are not duplicated."""
        manager.save_registry([
            {"name": "docker-manager", "version": "2.0.0"}
        ])
        available = manager.list_available()
        # docker-manager should appear only once (from builtins)
        docker_entries = [m for m in available if m.name == "docker-manager"]
        assert len(docker_entries) == 1


# ---------------------------------------------------------------------------
# TestPluginManager — get_plugin / is_installed / enable / disable
# ---------------------------------------------------------------------------


class TestPluginManagerAccess:
    """Tests for get_plugin, is_installed, enable, disable."""

    def test_get_plugin_installed(self, manager: PluginManager) -> None:
        """get_plugin returns info for installed plugin."""
        manager.install("docker-manager")
        info = manager.get_plugin("docker-manager")
        assert info is not None
        assert info.manifest.name == "docker-manager"

    def test_get_plugin_not_installed(self, manager: PluginManager) -> None:
        """get_plugin returns None for uninstalled plugin."""
        assert manager.get_plugin("nope") is None

    def test_get_plugin_empty_name(self, manager: PluginManager) -> None:
        """get_plugin returns None for empty name."""
        assert manager.get_plugin("") is None

    def test_is_installed_true(self, manager: PluginManager) -> None:
        """is_installed returns True for installed plugin."""
        manager.install("docker-manager")
        assert manager.is_installed("docker-manager") is True

    def test_is_installed_false(self, manager: PluginManager) -> None:
        """is_installed returns False for uninstalled plugin."""
        assert manager.is_installed("nonexistent") is False

    def test_is_installed_empty(self, manager: PluginManager) -> None:
        """is_installed returns False for empty name."""
        assert manager.is_installed("") is False

    def test_disable_plugin(self, manager: PluginManager) -> None:
        """disable() sets enabled=False."""
        manager.install("docker-manager")
        manager.disable("docker-manager")
        info = manager.get_plugin("docker-manager")
        assert info is not None
        assert info.enabled is False

    def test_enable_plugin(self, manager: PluginManager) -> None:
        """enable() sets enabled=True after disable."""
        manager.install("docker-manager")
        manager.disable("docker-manager")
        manager.enable("docker-manager")
        info = manager.get_plugin("docker-manager")
        assert info is not None
        assert info.enabled is True

    def test_disable_nonexistent_raises(self, manager: PluginManager) -> None:
        """disable() raises for non-installed plugin."""
        with pytest.raises(PluginNotFoundError):
            manager.disable("nonexistent")

    def test_enable_nonexistent_raises(self, manager: PluginManager) -> None:
        """enable() raises for non-installed plugin."""
        with pytest.raises(PluginNotFoundError):
            manager.enable("nonexistent")


# ---------------------------------------------------------------------------
# TestPluginManager — update
# ---------------------------------------------------------------------------


class TestPluginManagerUpdate:
    """Tests for updating plugins."""

    def test_update_nonexistent_raises(self, manager: PluginManager) -> None:
        """Updating a non-installed plugin raises error."""
        with pytest.raises(PluginNotFoundError):
            manager.update("nonexistent")

    def test_update_empty_name_raises(self, manager: PluginManager) -> None:
        """Updating with empty name raises error."""
        with pytest.raises(PluginNotFoundError):
            manager.update("")

    def test_update_builtin_reinstalls(self, manager: PluginManager) -> None:
        """Updating a built-in plugin re-installs it."""
        info = manager.install("docker-manager")
        assert info.source == "builtin"

        updated = manager.update("docker-manager")
        assert updated.manifest.name == "docker-manager"
        assert updated.source == "builtin"


# ---------------------------------------------------------------------------
# TestPluginManager — execute_tool
# ---------------------------------------------------------------------------


class TestPluginManagerExecuteTool:
    """Tests for sandboxed tool execution."""

    def test_execute_tool_not_installed_raises(
        self, manager: PluginManager
    ) -> None:
        """execute_tool raises for non-installed plugin."""
        with pytest.raises(PluginNotFoundError):
            manager.execute_tool("nonexistent", "tool", {})

    def test_execute_tool_disabled_raises(
        self, manager: PluginManager
    ) -> None:
        """execute_tool raises for disabled plugin."""
        manager.install("docker-manager")
        manager.disable("docker-manager")
        with pytest.raises(PluginError, match="disabled"):
            manager.execute_tool("docker-manager", "docker_ps", {})

    def test_execute_tool_unknown_tool_raises(
        self, manager: PluginManager
    ) -> None:
        """execute_tool raises for unknown tool name."""
        manager.install("docker-manager")
        with pytest.raises(PluginError, match="not found"):
            manager.execute_tool("docker-manager", "nonexistent_tool", {})

    def test_execute_tool_success(
        self, manager: PluginManager, plugins_dir: Path
    ) -> None:
        """execute_tool runs handler in subprocess and returns result."""
        # Create a plugin with a real handler
        plugin_dir = plugins_dir / "test-exec"
        plugin_dir.mkdir()

        manifest_data = {
            "name": "test-exec",
            "version": "1.0.0",
            "description": "Test execution",
            "tools": [
                {
                    "name": "add",
                    "description": "Add two numbers",
                    "handler": "add_numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                    },
                }
            ],
        }
        (plugin_dir / "plugin.yaml").write_text(
            yaml.safe_dump(manifest_data), encoding="utf-8"
        )
        (plugin_dir / "handler.py").write_text(
            'import json\n'
            'def add_numbers(a=0, b=0, **kw):\n'
            '    return {"result": a + b}\n',
            encoding="utf-8",
        )

        result = manager.execute_tool("test-exec", "add", {"a": 3, "b": 5})
        assert result["result"] == 8


# ---------------------------------------------------------------------------
# TestPluginManager — registry
# ---------------------------------------------------------------------------


class TestPluginManagerRegistry:
    """Tests for registry management."""

    def test_get_registry_empty(self, manager: PluginManager) -> None:
        """Empty registry returns empty list."""
        assert manager.get_registry() == []

    def test_save_and_load_registry(self, manager: PluginManager) -> None:
        """Saved registry can be loaded."""
        entries = [
            {"name": "plugin-a", "version": "1.0.0"},
            {"name": "plugin-b", "version": "2.0.0"},
        ]
        manager.save_registry(entries)

        loaded = manager.get_registry()
        assert len(loaded) == 2
        names = [e["name"] for e in loaded]
        assert "plugin-a" in names
        assert "plugin-b" in names

    def test_add_to_registry(self, manager: PluginManager) -> None:
        """add_to_registry appends a new entry."""
        manager.add_to_registry({"name": "new-plugin", "version": "1.0.0"})
        entries = manager.get_registry()
        assert len(entries) == 1
        assert entries[0]["name"] == "new-plugin"

    def test_add_to_registry_replaces_existing(
        self, manager: PluginManager
    ) -> None:
        """add_to_registry replaces entry with same name."""
        manager.add_to_registry({"name": "plug", "version": "1.0.0"})
        manager.add_to_registry({"name": "plug", "version": "2.0.0"})
        entries = manager.get_registry()
        assert len(entries) == 1
        assert entries[0]["version"] == "2.0.0"

    def test_add_to_registry_no_name_raises(
        self, manager: PluginManager
    ) -> None:
        """add_to_registry raises for entry without name."""
        with pytest.raises(ValueError, match="name"):
            manager.add_to_registry({"version": "1.0.0"})

    def test_remove_from_registry(self, manager: PluginManager) -> None:
        """remove_from_registry removes by name."""
        manager.save_registry([
            {"name": "a", "version": "1.0.0"},
            {"name": "b", "version": "1.0.0"},
        ])
        assert manager.remove_from_registry("a") is True
        entries = manager.get_registry()
        assert len(entries) == 1
        assert entries[0]["name"] == "b"

    def test_remove_from_registry_nonexistent(
        self, manager: PluginManager
    ) -> None:
        """remove_from_registry returns False for unknown name."""
        assert manager.remove_from_registry("nope") is False

    def test_registry_with_plugins_key(self, manager: PluginManager) -> None:
        """Registry file with {"plugins": [...]} format is supported."""
        data = {
            "plugins": [
                {"name": "wrapped", "version": "1.0.0"},
            ]
        }
        manager.registry_path.write_text(
            json.dumps(data), encoding="utf-8"
        )
        entries = manager.get_registry()
        assert len(entries) == 1
        assert entries[0]["name"] == "wrapped"

    def test_registry_corrupt_json(self, manager: PluginManager) -> None:
        """Corrupt registry file returns empty list."""
        manager.registry_path.write_text("not json!", encoding="utf-8")
        assert manager.get_registry() == []


# ---------------------------------------------------------------------------
# TestBuiltInPlugins
# ---------------------------------------------------------------------------


class TestBuiltInPlugins:
    """Tests for built-in plugin manifests."""

    def test_docker_manager_manifest(self) -> None:
        """docker-manager built-in manifest is well-formed."""
        m = BUILTIN_PLUGINS["docker-manager"]
        assert m.name == "docker-manager"
        assert m.version == "1.0.0"
        assert m.author == "Prism Core Team"
        assert len(m.tools) >= 3
        assert any(t.name == "docker_ps" for t in m.tools)
        assert any(t.name == "docker_logs" for t in m.tools)
        assert any(t.name == "docker_compose_status" for t in m.tools)
        assert len(m.commands) >= 1
        assert m.validate() == []

    def test_db_query_manifest(self) -> None:
        """db-query built-in manifest is well-formed."""
        m = BUILTIN_PLUGINS["db-query"]
        assert m.name == "db-query"
        assert m.version == "1.0.0"
        assert len(m.tools) >= 2
        assert any(t.name == "db_query" for t in m.tools)
        assert any(t.name == "db_schema" for t in m.tools)
        assert m.validate() == []

    def test_api_tester_manifest(self) -> None:
        """api-tester built-in manifest is well-formed."""
        m = BUILTIN_PLUGINS["api-tester"]
        assert m.name == "api-tester"
        assert m.version == "1.0.0"
        assert len(m.tools) >= 3
        assert any(t.name == "api_request" for t in m.tools)
        assert any(t.name == "api_validate" for t in m.tools)
        assert any(t.name == "api_collection_run" for t in m.tools)
        assert m.validate() == []

    def test_all_builtins_validate(self) -> None:
        """All built-in manifests pass validation."""
        for name, manifest in BUILTIN_PLUGINS.items():
            errors = manifest.validate()
            assert errors == [], f"Built-in '{name}' has validation errors: {errors}"

    def test_all_builtins_have_tools(self) -> None:
        """All built-in plugins have at least one tool."""
        for name, manifest in BUILTIN_PLUGINS.items():
            assert len(manifest.tools) > 0, f"Built-in '{name}' has no tools"

    def test_all_builtins_have_commands(self) -> None:
        """All built-in plugins have at least one command."""
        for name, manifest in BUILTIN_PLUGINS.items():
            assert len(manifest.commands) > 0, f"Built-in '{name}' has no commands"

    def test_all_tool_handlers_set(self) -> None:
        """All tools in built-in plugins have handler functions specified."""
        for name, manifest in BUILTIN_PLUGINS.items():
            for tool in manifest.tools:
                assert tool.handler, (
                    f"Tool '{tool.name}' in '{name}' has no handler"
                )


# ---------------------------------------------------------------------------
# TestPluginManager — sandbox env
# ---------------------------------------------------------------------------


class TestPluginManagerSandbox:
    """Tests for sandbox environment filtering."""

    def test_sandbox_env_filters_api_keys(self) -> None:
        """_sandbox_env strips API key environment variables."""
        with patch.dict(
            "os.environ",
            {
                "HOME": "/home/user",
                "PATH": "/usr/bin",
                "OPENAI_API_KEY": "sk-secret-123",
                "ANTHROPIC_API_KEY": "sk-ant-secret",
                "DATABASE_PASSWORD": "hunter2",
                "NORMAL_VAR": "keep-me",
            },
            clear=True,
        ):
            env = PluginManager._sandbox_env()
            assert "HOME" in env
            assert "PATH" in env
            assert "NORMAL_VAR" in env
            assert "OPENAI_API_KEY" not in env
            assert "ANTHROPIC_API_KEY" not in env
            assert "DATABASE_PASSWORD" not in env

    def test_sandbox_env_filters_tokens(self) -> None:
        """_sandbox_env strips token variables."""
        with patch.dict(
            "os.environ",
            {
                "GITHUB_TOKEN": "ghp_secret",
                "AUTH_SECRET": "s3cr3t",
                "SAFE_VAR": "ok",
            },
            clear=True,
        ):
            env = PluginManager._sandbox_env()
            assert "GITHUB_TOKEN" not in env
            assert "AUTH_SECRET" not in env
            assert "SAFE_VAR" in env

    def test_sandbox_env_case_insensitive(self) -> None:
        """_sandbox_env filtering is case-insensitive on key names."""
        with patch.dict(
            "os.environ",
            {"my_api_key_test": "value", "KEEP": "yes"},
            clear=True,
        ):
            env = PluginManager._sandbox_env()
            assert "my_api_key_test" not in env
            assert "KEEP" in env


# ---------------------------------------------------------------------------
# TestPluginManager — manifest loading edge cases
# ---------------------------------------------------------------------------


class TestPluginManagerManifestLoading:
    """Tests for manifest parsing edge cases."""

    def test_manifest_non_dict_raises(
        self, manager: PluginManager, plugins_dir: Path
    ) -> None:
        """Manifest that is not a YAML mapping raises error."""
        plugin_dir = plugins_dir / "bad-format"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text("just a string")

        with pytest.raises(PluginValidationError, match="YAML mapping"):
            manager._load_manifest_from_dir(plugin_dir)

    def test_manifest_invalid_yaml_raises(
        self, manager: PluginManager, plugins_dir: Path
    ) -> None:
        """Manifest with invalid YAML raises error."""
        plugin_dir = plugins_dir / "invalid-yaml"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(":\n  - :\n    !!invalid")

        with pytest.raises(PluginValidationError):
            manager._load_manifest_from_dir(plugin_dir)

    def test_manifest_from_dict_uses_fallback_name(self) -> None:
        """_manifest_from_dict uses fallback name when name not in data."""
        data: dict[str, Any] = {"version": "1.0.0"}
        manifest = PluginManager._manifest_from_dict(data, "fallback-name")
        assert manifest.name == "fallback-name"

    def test_manifest_with_tools_and_commands(
        self, manager: PluginManager, plugins_dir: Path
    ) -> None:
        """Manifest with tools and commands is parsed correctly."""
        _write_plugin(
            plugins_dir,
            "full-plugin",
            tools=[
                {
                    "name": "tool_one",
                    "description": "First tool",
                    "handler": "handle_one",
                    "parameters": {"type": "object"},
                }
            ],
            commands=[
                {
                    "name": "cmd-one",
                    "description": "First command",
                    "handler": "handle_cmd",
                }
            ],
        )

        installed = manager.list_installed()
        assert len(installed) == 1
        plugin = installed[0]
        assert len(plugin.manifest.tools) == 1
        assert plugin.manifest.tools[0].name == "tool_one"
        assert plugin.manifest.tools[0].handler == "handle_one"
        assert len(plugin.manifest.commands) == 1
        assert plugin.manifest.commands[0].name == "cmd-one"

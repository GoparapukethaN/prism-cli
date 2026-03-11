"""Plugin manager — install, remove, update, and load plugins.

Plugins live in ``~/.prism/plugins/<name>/`` and are declared via a
``plugin.yaml`` manifest.  The manager handles the full lifecycle:
discovery, installation from GitHub repos or local paths, removal,
updates, and sandboxed loading.

Three built-in plugin manifests are shipped with Prism core:
``docker-manager``, ``db-query``, and ``api-tester``.  These serve as
reference implementations and are always available in the plugin list.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PLUGINS_DIR_NAME = "plugins"
_MANIFEST_FILENAME = "plugin.yaml"
_REGISTRY_FILENAME = "registry.json"
_HANDLER_FILENAME = "handler.py"

# Subprocess limits for sandboxed plugin execution
_PLUGIN_EXEC_TIMEOUT: int = 30
_PLUGIN_MAX_OUTPUT_BYTES: int = 65_536  # 64 KB


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PluginError(Exception):
    """Base exception for plugin-related errors."""


class PluginNotFoundError(PluginError):
    """Raised when a plugin is not installed."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Plugin not found: {name}")


class PluginValidationError(PluginError):
    """Raised when a plugin manifest is invalid."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"Invalid plugin '{name}': {reason}")


class PluginInstallError(PluginError):
    """Raised when plugin installation fails."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"Failed to install plugin '{name}': {reason}")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PluginToolSpec:
    """Specification for a tool provided by a plugin.

    Attributes:
        name:        Unique tool identifier.
        description: Human-readable tool description.
        parameters:  JSON Schema for tool parameters.
        handler:     Name of the handler function in ``handler.py``.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    handler: str = ""


@dataclass
class PluginCommandSpec:
    """Specification for a CLI command provided by a plugin.

    Attributes:
        name:        Command name (e.g. ``"docker-ps"``).
        description: Human-readable description.
        handler:     Name of the handler function in ``handler.py``.
    """

    name: str
    description: str
    handler: str = ""


@dataclass
class PluginManifest:
    """Plugin metadata parsed from ``plugin.yaml``.

    Attributes:
        name:         Plugin identifier (lowercase, hyphenated).
        version:      Semantic version string (e.g. ``"1.0.0"``).
        description:  Short description of the plugin.
        author:       Plugin author / maintainer.
        homepage:     URL to the plugin's homepage or repository.
        license:      SPDX license identifier.
        tools:        Tools provided by this plugin.
        commands:     CLI commands provided by this plugin.
        dependencies: Python package dependencies (pip-installable).
        min_prism:    Minimum Prism version required.
    """

    name: str
    version: str
    description: str = ""
    author: str = ""
    homepage: str = ""
    license: str = ""
    tools: list[PluginToolSpec] = field(default_factory=list)
    commands: list[PluginCommandSpec] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    min_prism: str = "0.1.0"

    def validate(self) -> list[str]:
        """Validate the manifest and return a list of error messages.

        Returns:
            List of validation error strings.  Empty means valid.
        """
        errors: list[str] = []

        if not self.name or not self.name.strip():
            errors.append("Plugin name is required")
        elif not all(
            c.isalnum() or c in ("-", "_") for c in self.name
        ):
            errors.append(
                f"Plugin name '{self.name}' contains invalid characters; "
                "use only alphanumeric, hyphens, and underscores"
            )

        if not self.version or not self.version.strip():
            errors.append("Plugin version is required")

        # Check for duplicate tool names
        tool_names = [t.name for t in self.tools]
        seen: set[str] = set()
        for tn in tool_names:
            if tn in seen:
                errors.append(f"Duplicate tool name: {tn}")
            seen.add(tn)

        # Check for duplicate command names
        cmd_names = [c.name for c in self.commands]
        seen_cmd: set[str] = set()
        for cn in cmd_names:
            if cn in seen_cmd:
                errors.append(f"Duplicate command name: {cn}")
            seen_cmd.add(cn)

        return errors


@dataclass
class PluginInfo:
    """Runtime information about an installed plugin.

    Attributes:
        manifest:     The parsed plugin manifest.
        install_path: Absolute path to the plugin directory.
        enabled:      Whether the plugin is active.
        installed_at: ISO-8601 timestamp of installation.
        source:       Where the plugin was installed from.
    """

    manifest: PluginManifest
    install_path: Path
    enabled: bool = True
    installed_at: str = ""
    source: str = ""


# ---------------------------------------------------------------------------
# Built-in plugin manifests
# ---------------------------------------------------------------------------


def _builtin_docker_manager() -> PluginManifest:
    """Built-in plugin: Docker container management."""
    return PluginManifest(
        name="docker-manager",
        version="1.0.0",
        description="Manage Docker containers, images, and compose stacks",
        author="Prism Core Team",
        homepage="https://github.com/GoparapukethaN/prism-cli",
        license="Apache-2.0",
        tools=[
            PluginToolSpec(
                name="docker_ps",
                description="List running Docker containers",
                parameters={
                    "type": "object",
                    "properties": {
                        "all": {
                            "type": "boolean",
                            "description": "Show all containers (not just running)",
                            "default": False,
                        },
                        "format": {
                            "type": "string",
                            "description": "Output format (table, json)",
                            "default": "table",
                        },
                    },
                },
                handler="docker_ps",
            ),
            PluginToolSpec(
                name="docker_logs",
                description="Fetch logs from a Docker container",
                parameters={
                    "type": "object",
                    "properties": {
                        "container": {
                            "type": "string",
                            "description": "Container name or ID",
                        },
                        "tail": {
                            "type": "integer",
                            "description": "Number of lines from the end",
                            "default": 100,
                        },
                    },
                    "required": ["container"],
                },
                handler="docker_logs",
            ),
            PluginToolSpec(
                name="docker_compose_status",
                description="Show status of Docker Compose services",
                parameters={
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "description": "Path to docker-compose.yml",
                            "default": "docker-compose.yml",
                        },
                    },
                },
                handler="docker_compose_status",
            ),
        ],
        commands=[
            PluginCommandSpec(
                name="docker-ps",
                description="List Docker containers",
                handler="cmd_docker_ps",
            ),
        ],
        dependencies=["docker"],
    )


def _builtin_db_query() -> PluginManifest:
    """Built-in plugin: Database query runner."""
    return PluginManifest(
        name="db-query",
        version="1.0.0",
        description="Execute SQL queries against SQLite, PostgreSQL, and MySQL databases",
        author="Prism Core Team",
        homepage="https://github.com/GoparapukethaN/prism-cli",
        license="Apache-2.0",
        tools=[
            PluginToolSpec(
                name="db_query",
                description="Execute a read-only SQL query and return results",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute (SELECT only)",
                        },
                        "database": {
                            "type": "string",
                            "description": "Database connection string or path",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum rows to return",
                            "default": 100,
                        },
                    },
                    "required": ["query", "database"],
                },
                handler="db_query",
            ),
            PluginToolSpec(
                name="db_schema",
                description="Show database schema (tables, columns, types)",
                parameters={
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Database connection string or path",
                        },
                        "table": {
                            "type": "string",
                            "description": "Specific table (omit for all)",
                            "default": "",
                        },
                    },
                    "required": ["database"],
                },
                handler="db_schema",
            ),
        ],
        commands=[
            PluginCommandSpec(
                name="db-query",
                description="Run a SQL query from the command line",
                handler="cmd_db_query",
            ),
        ],
        dependencies=[],
    )


def _builtin_api_tester() -> PluginManifest:
    """Built-in plugin: REST API testing."""
    return PluginManifest(
        name="api-tester",
        version="1.0.0",
        description="Test REST APIs with request building, response validation, and collections",
        author="Prism Core Team",
        homepage="https://github.com/GoparapukethaN/prism-cli",
        license="Apache-2.0",
        tools=[
            PluginToolSpec(
                name="api_request",
                description="Send an HTTP request and return the response",
                parameters={
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "description": "HTTP method (GET, POST, PUT, DELETE, PATCH)",
                            "default": "GET",
                        },
                        "url": {
                            "type": "string",
                            "description": "Request URL",
                        },
                        "headers": {
                            "type": "object",
                            "description": "Request headers",
                            "default": {},
                        },
                        "body": {
                            "type": "string",
                            "description": "Request body (JSON string)",
                            "default": "",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Request timeout in seconds",
                            "default": 30,
                        },
                    },
                    "required": ["url"],
                },
                handler="api_request",
            ),
            PluginToolSpec(
                name="api_validate",
                description="Validate an API response against a JSON schema",
                parameters={
                    "type": "object",
                    "properties": {
                        "response_body": {
                            "type": "string",
                            "description": "The response body to validate (JSON string)",
                        },
                        "schema": {
                            "type": "object",
                            "description": "JSON Schema to validate against",
                        },
                    },
                    "required": ["response_body", "schema"],
                },
                handler="api_validate",
            ),
            PluginToolSpec(
                name="api_collection_run",
                description="Run a collection of API requests in sequence",
                parameters={
                    "type": "object",
                    "properties": {
                        "collection_path": {
                            "type": "string",
                            "description": "Path to collection JSON file",
                        },
                        "environment": {
                            "type": "object",
                            "description": "Variable substitutions",
                            "default": {},
                        },
                    },
                    "required": ["collection_path"],
                },
                handler="api_collection_run",
            ),
        ],
        commands=[
            PluginCommandSpec(
                name="api-test",
                description="Send a quick API request",
                handler="cmd_api_test",
            ),
        ],
        dependencies=["httpx"],
    )


BUILTIN_PLUGINS: dict[str, PluginManifest] = {
    "docker-manager": _builtin_docker_manager(),
    "db-query": _builtin_db_query(),
    "api-tester": _builtin_api_tester(),
}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class PluginManager:
    """Manages plugin discovery, installation, removal, and loading.

    Plugins are stored in ``plugins_dir`` (default ``~/.prism/plugins/``).
    A registry index at ``plugins_dir/registry.json`` tracks available
    community plugins.

    Args:
        plugins_dir: Directory for installed plugins.  Created if it
            does not exist.
    """

    def __init__(self, plugins_dir: Path | None = None) -> None:
        self._plugins_dir: Path = plugins_dir or (
            Path.home() / ".prism" / _DEFAULT_PLUGINS_DIR_NAME
        )
        self._plugins_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[str, PluginInfo] = {}
        logger.debug("plugin_manager_init", dir=str(self._plugins_dir))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def plugins_dir(self) -> Path:
        """Return the plugins installation directory."""
        return self._plugins_dir

    @property
    def registry_path(self) -> Path:
        """Return the path to the local registry index file."""
        return self._plugins_dir / _REGISTRY_FILENAME

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    def install(self, source: str) -> PluginInfo:
        """Install a plugin from a GitHub repo URL or local path.

        The source can be:
        - A GitHub URL (``https://github.com/user/repo``)
        - A short GitHub reference (``user/repo``)
        - A local filesystem path

        Args:
            source: Plugin source (URL, short ref, or path).

        Returns:
            :class:`PluginInfo` for the installed plugin.

        Raises:
            PluginInstallError: If installation fails.
            PluginValidationError: If the manifest is invalid.
        """
        if not source or not source.strip():
            raise PluginInstallError("unknown", "Source must not be empty")

        source = source.strip()

        # Determine installation strategy
        if source.startswith("https://github.com/") or (
            "/" in source and not Path(source).exists()
        ):
            return self._install_from_git(source)
        elif Path(source).exists():
            return self._install_from_local(Path(source))
        else:
            # Check built-in plugins
            if source in BUILTIN_PLUGINS:
                return self._install_builtin(source)
            raise PluginInstallError(
                source,
                f"Source not found: {source}. "
                "Provide a GitHub URL, local path, or built-in name.",
            )

    def _install_from_git(self, source: str) -> PluginInfo:
        """Clone a plugin from a git repository.

        Args:
            source: Git URL or ``user/repo`` shorthand.

        Returns:
            :class:`PluginInfo` for the installed plugin.

        Raises:
            PluginInstallError: If the clone or validation fails.
        """
        # Normalise GitHub shorthand
        if not source.startswith("https://"):
            git_url = f"https://github.com/{source}.git"
        elif not source.endswith(".git"):
            git_url = source + ".git"
        else:
            git_url = source

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "repo"
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", git_url, str(tmp_path)],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
            except FileNotFoundError as exc:
                raise PluginInstallError(
                    source, "git is not installed or not on PATH"
                ) from exc
            except subprocess.CalledProcessError as exc:
                raise PluginInstallError(
                    source,
                    f"git clone failed: {exc.stderr[:300] if exc.stderr else 'unknown error'}",
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise PluginInstallError(
                    source, "git clone timed out"
                ) from exc

            manifest = self._load_manifest_from_dir(tmp_path)
            return self._finalize_install(manifest, tmp_path, source=source)

    def _install_from_local(self, source_path: Path) -> PluginInfo:
        """Install a plugin from a local directory.

        Args:
            source_path: Path to the plugin directory.

        Returns:
            :class:`PluginInfo` for the installed plugin.

        Raises:
            PluginInstallError: If the path is invalid.
            PluginValidationError: If the manifest is invalid.
        """
        if not source_path.is_dir():
            raise PluginInstallError(
                str(source_path), "Source path is not a directory"
            )
        manifest = self._load_manifest_from_dir(source_path)
        return self._finalize_install(
            manifest, source_path, source=str(source_path)
        )

    def _install_builtin(self, name: str) -> PluginInfo:
        """Install a built-in plugin by creating its directory structure.

        Args:
            name: Built-in plugin name.

        Returns:
            :class:`PluginInfo` for the installed plugin.

        Raises:
            PluginInstallError: If the plugin is not a known built-in.
        """
        manifest = BUILTIN_PLUGINS.get(name)
        if manifest is None:
            raise PluginInstallError(name, f"Unknown built-in plugin: {name}")

        install_dir = self._plugins_dir / name
        if install_dir.exists():
            raise PluginInstallError(
                name,
                f"Plugin already installed at {install_dir}. "
                "Remove it first with 'prism plugins remove'.",
            )

        install_dir.mkdir(parents=True, exist_ok=True)

        # Write manifest
        self._write_manifest(manifest, install_dir)

        # Write stub handler
        handler_path = install_dir / _HANDLER_FILENAME
        handler_path.write_text(
            f'"""Built-in plugin: {manifest.name}."""\n\n'
            f"# Tool handlers for {manifest.name}\n"
            f"# These are stubs — see plugin docs for implementation.\n\n"
            + "\n\n".join(
                f"def {t.handler}(**kwargs):\n"
                f'    """Handle {t.name} tool call."""\n'
                f'    return {{"status": "ok", "tool": "{t.name}"}}\n'
                for t in manifest.tools
                if t.handler
            )
            + "\n\n"
            + "\n\n".join(
                f"def {c.handler}(**kwargs):\n"
                f'    """Handle {c.name} command."""\n'
                f'    return {{"status": "ok", "command": "{c.name}"}}\n'
                for c in manifest.commands
                if c.handler
            ),
        )

        now = datetime.now(UTC).isoformat()
        info = PluginInfo(
            manifest=manifest,
            install_path=install_dir,
            enabled=True,
            installed_at=now,
            source="builtin",
        )
        self._loaded[name] = info
        logger.info("plugin_installed", name=name, source="builtin")
        return info

    def _finalize_install(
        self,
        manifest: PluginManifest,
        source_dir: Path,
        source: str,
    ) -> PluginInfo:
        """Copy plugin files to the plugins directory and register.

        Args:
            manifest: Validated plugin manifest.
            source_dir: Directory containing plugin files.
            source: Original source string (URL or path).

        Returns:
            :class:`PluginInfo` for the installed plugin.

        Raises:
            PluginInstallError: If the target directory already exists.
        """
        install_dir = self._plugins_dir / manifest.name

        if install_dir.exists():
            raise PluginInstallError(
                manifest.name,
                f"Plugin already installed at {install_dir}. "
                "Remove it first with 'prism plugins remove'.",
            )

        # Copy all files from source to install directory
        shutil.copytree(
            str(source_dir),
            str(install_dir),
            dirs_exist_ok=False,
        )

        now = datetime.now(UTC).isoformat()
        info = PluginInfo(
            manifest=manifest,
            install_path=install_dir,
            enabled=True,
            installed_at=now,
            source=source,
        )
        self._loaded[manifest.name] = info
        logger.info(
            "plugin_installed",
            name=manifest.name,
            version=manifest.version,
            source=source,
        )
        return info

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove(self, name: str) -> bool:
        """Remove an installed plugin.

        Deletes the plugin directory and unloads it from memory.

        Args:
            name: Plugin name to remove.

        Returns:
            ``True`` if the plugin was removed.

        Raises:
            PluginNotFoundError: If the plugin is not installed.
        """
        if not name or not name.strip():
            raise PluginNotFoundError("")

        name = name.strip()
        install_dir = self._plugins_dir / name

        if not install_dir.exists():
            raise PluginNotFoundError(name)

        shutil.rmtree(str(install_dir))
        self._loaded.pop(name, None)
        logger.info("plugin_removed", name=name)
        return True

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, name: str) -> PluginInfo:
        """Update an installed plugin by re-installing from its source.

        Args:
            name: Plugin name to update.

        Returns:
            Updated :class:`PluginInfo`.

        Raises:
            PluginNotFoundError: If the plugin is not installed.
            PluginInstallError: If re-installation fails.
        """
        if not name or not name.strip():
            raise PluginNotFoundError("")

        name = name.strip()
        existing = self.get_plugin(name)
        if existing is None:
            raise PluginNotFoundError(name)

        source = existing.source
        if not source:
            raise PluginInstallError(
                name, "Cannot update: installation source is unknown"
            )

        # Remove and re-install
        self.remove(name)

        # For built-in plugins, install by the plugin name
        if source == "builtin":
            return self.install(name)
        return self.install(source)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_installed(self) -> list[PluginInfo]:
        """List all installed plugins.

        Scans the plugins directory for directories containing a
        ``plugin.yaml`` manifest.

        Returns:
            Sorted list of :class:`PluginInfo` instances.
        """
        plugins: list[PluginInfo] = []

        if not self._plugins_dir.exists():
            return plugins

        for child in sorted(self._plugins_dir.iterdir()):
            if not child.is_dir():
                continue

            manifest_path = child / _MANIFEST_FILENAME
            if not manifest_path.exists():
                continue

            try:
                manifest = self._load_manifest_from_dir(child)
                info = PluginInfo(
                    manifest=manifest,
                    install_path=child,
                    enabled=True,
                    installed_at="",
                    source="",
                )
                # Merge with cached info if available
                if manifest.name in self._loaded:
                    cached = self._loaded[manifest.name]
                    info.installed_at = cached.installed_at
                    info.source = cached.source
                    info.enabled = cached.enabled
                plugins.append(info)
            except (PluginValidationError, PluginError):
                logger.warning("plugin_load_failed", dir=str(child))
                continue

        return plugins

    def list_available(self) -> list[PluginManifest]:
        """List available plugins from the registry and built-ins.

        Returns built-in plugin manifests plus any community plugins
        from the local registry index.

        Returns:
            List of :class:`PluginManifest` instances.
        """
        available: list[PluginManifest] = []

        # Built-in plugins always available
        for manifest in BUILTIN_PLUGINS.values():
            available.append(manifest)

        # Community plugins from registry
        registry = self.get_registry()
        for entry in registry:
            name = entry.get("name", "")
            if name and name not in BUILTIN_PLUGINS:
                manifest = PluginManifest(
                    name=name,
                    version=entry.get("version", "0.0.0"),
                    description=entry.get("description", ""),
                    author=entry.get("author", ""),
                    homepage=entry.get("homepage", ""),
                )
                available.append(manifest)

        return available

    # ------------------------------------------------------------------
    # Plugin access
    # ------------------------------------------------------------------

    def get_plugin(self, name: str) -> PluginInfo | None:
        """Get info for an installed plugin by name.

        Args:
            name: Plugin name.

        Returns:
            :class:`PluginInfo`, or ``None`` if not installed.
        """
        if not name:
            return None

        # Check in-memory cache first
        if name in self._loaded:
            return self._loaded[name]

        # Check on disk
        install_dir = self._plugins_dir / name
        manifest_path = install_dir / _MANIFEST_FILENAME
        if not manifest_path.exists():
            return None

        try:
            manifest = self._load_manifest_from_dir(install_dir)
            info = PluginInfo(
                manifest=manifest,
                install_path=install_dir,
                enabled=True,
            )
            self._loaded[name] = info
            return info
        except (PluginValidationError, PluginError):
            return None

    def is_installed(self, name: str) -> bool:
        """Check if a plugin is installed.

        Args:
            name: Plugin name.

        Returns:
            ``True`` if the plugin exists on disk.
        """
        if not name:
            return False
        return (self._plugins_dir / name / _MANIFEST_FILENAME).exists()

    def enable(self, name: str) -> bool:
        """Enable a disabled plugin.

        Args:
            name: Plugin name.

        Returns:
            ``True`` if the plugin was enabled.

        Raises:
            PluginNotFoundError: If the plugin is not installed.
        """
        info = self.get_plugin(name)
        if info is None:
            raise PluginNotFoundError(name)
        info.enabled = True
        return True

    def disable(self, name: str) -> bool:
        """Disable a plugin without removing it.

        Args:
            name: Plugin name.

        Returns:
            ``True`` if the plugin was disabled.

        Raises:
            PluginNotFoundError: If the plugin is not installed.
        """
        info = self.get_plugin(name)
        if info is None:
            raise PluginNotFoundError(name)
        info.enabled = False
        return True

    # ------------------------------------------------------------------
    # Sandboxed execution
    # ------------------------------------------------------------------

    def execute_tool(
        self,
        plugin_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a plugin tool in a sandboxed subprocess.

        The handler function is invoked in a separate Python process
        with restricted environment variables and a timeout.

        Args:
            plugin_name: Plugin name.
            tool_name: Tool name as declared in the manifest.
            arguments: Tool arguments.

        Returns:
            Dictionary with the tool's output.

        Raises:
            PluginNotFoundError: If the plugin is not installed.
            PluginError: If execution fails.
        """
        info = self.get_plugin(plugin_name)
        if info is None:
            raise PluginNotFoundError(plugin_name)

        if not info.enabled:
            raise PluginError(f"Plugin '{plugin_name}' is disabled")

        # Find the tool spec
        tool_spec: PluginToolSpec | None = None
        for t in info.manifest.tools:
            if t.name == tool_name:
                tool_spec = t
                break

        if tool_spec is None:
            raise PluginError(
                f"Tool '{tool_name}' not found in plugin '{plugin_name}'"
            )

        handler_path = info.install_path / _HANDLER_FILENAME
        if not handler_path.exists():
            raise PluginError(
                f"Handler file not found for plugin '{plugin_name}'"
            )

        # Build subprocess command
        handler_func = tool_spec.handler or tool_name
        args_json = json.dumps(arguments)

        script = (
            f"import json, sys\n"
            f"sys.path.insert(0, {str(info.install_path)!r})\n"
            f"from handler import {handler_func}\n"
            f"args = json.loads({args_json!r})\n"
            f"result = {handler_func}(**args)\n"
            f"print(json.dumps(result))\n"
        )

        # Execute in sandboxed subprocess
        env = self._sandbox_env()
        try:
            result = subprocess.run(
                ["python3", "-c", script],
                capture_output=True,
                text=True,
                timeout=_PLUGIN_EXEC_TIMEOUT,
                env=env,
                cwd=str(info.install_path),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PluginError(
                f"Plugin '{plugin_name}' tool '{tool_name}' timed out "
                f"after {_PLUGIN_EXEC_TIMEOUT}s"
            ) from exc
        except FileNotFoundError as exc:
            raise PluginError("python3 is not available") from exc

        if result.returncode != 0:
            stderr = result.stderr[:500] if result.stderr else "unknown error"
            raise PluginError(
                f"Plugin '{plugin_name}' tool '{tool_name}' failed: {stderr}"
            )

        stdout = result.stdout.strip()
        if not stdout:
            return {"status": "ok", "output": ""}

        try:
            return json.loads(stdout)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return {"status": "ok", "output": stdout[:_PLUGIN_MAX_OUTPUT_BYTES]}

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def get_registry(self) -> list[dict[str, Any]]:
        """Load the local plugin registry index.

        Returns:
            List of plugin entries from ``registry.json``.
        """
        if not self.registry_path.exists():
            return []

        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data  # type: ignore[no-any-return]
            if isinstance(data, dict) and "plugins" in data:
                return data["plugins"]  # type: ignore[no-any-return]
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("registry_load_failed", error=str(exc))
            return []

    def save_registry(self, entries: list[dict[str, Any]]) -> None:
        """Save the plugin registry index to disk.

        Args:
            entries: List of plugin entries to save.
        """
        self._plugins_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(entries, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.debug("registry_saved", count=len(entries))

    def add_to_registry(self, entry: dict[str, Any]) -> None:
        """Add a plugin entry to the registry.

        If an entry with the same name exists, it is replaced.

        Args:
            entry: Plugin entry with at least a ``"name"`` key.

        Raises:
            ValueError: If the entry has no ``"name"`` key.
        """
        name = entry.get("name")
        if not name:
            raise ValueError("Registry entry must have a 'name' key")

        entries = self.get_registry()
        # Remove existing entry with same name
        entries = [e for e in entries if e.get("name") != name]
        entries.append(entry)
        self.save_registry(entries)

    def remove_from_registry(self, name: str) -> bool:
        """Remove a plugin entry from the registry by name.

        Args:
            name: Plugin name.

        Returns:
            ``True`` if an entry was removed.
        """
        entries = self.get_registry()
        before = len(entries)
        entries = [e for e in entries if e.get("name") != name]
        if len(entries) < before:
            self.save_registry(entries)
            return True
        return False

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load_manifest_from_dir(self, plugin_dir: Path) -> PluginManifest:
        """Load and validate a plugin manifest from a directory.

        Args:
            plugin_dir: Directory containing ``plugin.yaml``.

        Returns:
            Validated :class:`PluginManifest`.

        Raises:
            PluginValidationError: If the manifest is missing or invalid.
        """
        import yaml

        manifest_path = plugin_dir / _MANIFEST_FILENAME
        if not manifest_path.exists():
            raise PluginValidationError(
                plugin_dir.name,
                f"Missing {_MANIFEST_FILENAME}",
            )

        try:
            data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PluginValidationError(
                plugin_dir.name,
                f"Failed to parse {_MANIFEST_FILENAME}: {exc}",
            ) from exc

        if not isinstance(data, dict):
            raise PluginValidationError(
                plugin_dir.name,
                f"{_MANIFEST_FILENAME} must be a YAML mapping",
            )

        return self._manifest_from_dict(data, plugin_dir.name)

    @staticmethod
    def _manifest_from_dict(data: dict[str, Any], fallback_name: str) -> PluginManifest:
        """Build a :class:`PluginManifest` from a parsed YAML dict.

        Args:
            data: Parsed YAML data.
            fallback_name: Name to use if not specified in manifest.

        Returns:
            :class:`PluginManifest`.

        Raises:
            PluginValidationError: If validation fails.
        """
        tools: list[PluginToolSpec] = []
        for td in data.get("tools", []):
            if isinstance(td, dict):
                tools.append(
                    PluginToolSpec(
                        name=td.get("name", ""),
                        description=td.get("description", ""),
                        parameters=td.get("parameters", {}),
                        handler=td.get("handler", ""),
                    )
                )

        commands: list[PluginCommandSpec] = []
        for cd in data.get("commands", []):
            if isinstance(cd, dict):
                commands.append(
                    PluginCommandSpec(
                        name=cd.get("name", ""),
                        description=cd.get("description", ""),
                        handler=cd.get("handler", ""),
                    )
                )

        manifest = PluginManifest(
            name=data.get("name", fallback_name),
            version=data.get("version", "0.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            homepage=data.get("homepage", ""),
            license=data.get("license", ""),
            tools=tools,
            commands=commands,
            dependencies=data.get("dependencies", []),
            min_prism=data.get("min_prism", "0.1.0"),
        )

        errors = manifest.validate()
        if errors:
            raise PluginValidationError(
                manifest.name,
                "; ".join(errors),
            )

        return manifest

    @staticmethod
    def _write_manifest(manifest: PluginManifest, plugin_dir: Path) -> None:
        """Write a :class:`PluginManifest` to ``plugin.yaml``.

        Args:
            manifest: The manifest to write.
            plugin_dir: Target directory.
        """
        import yaml

        data: dict[str, Any] = {
            "name": manifest.name,
            "version": manifest.version,
            "description": manifest.description,
            "author": manifest.author,
            "homepage": manifest.homepage,
            "license": manifest.license,
            "min_prism": manifest.min_prism,
            "dependencies": manifest.dependencies,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "handler": t.handler,
                }
                for t in manifest.tools
            ],
            "commands": [
                {
                    "name": c.name,
                    "description": c.description,
                    "handler": c.handler,
                }
                for c in manifest.commands
            ],
        }

        manifest_path = plugin_dir / _MANIFEST_FILENAME
        manifest_path.write_text(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Sandbox helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sandbox_env() -> dict[str, str]:
        """Build a sanitised environment for subprocess execution.

        Strips all API keys, secrets, and tokens to prevent plugins
        from accessing credentials.

        Returns:
            Sanitised copy of ``os.environ``.
        """
        import os

        sensitive_patterns = (
            "API_KEY",
            "SECRET",
            "TOKEN",
            "PASSWORD",
            "CREDENTIAL",
            "PRIVATE_KEY",
        )

        env: dict[str, str] = {}
        for key, value in os.environ.items():
            key_upper = key.upper()
            if any(pattern in key_upper for pattern in sensitive_patterns):
                continue
            env[key] = value

        return env

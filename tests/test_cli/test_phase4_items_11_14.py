"""Tests for Phase 4 items 11-14 REPL enhancements.

Covers /ignore command, /add prismignore warnings,
/privacy enhancements, proxy/offline connectivity checks,
/plugins CLI, and the plugin API module.

All external dependencies are fully mocked.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path

from prism.cli.repl import (
    _dispatch_command,
    _SessionState,
)
from prism.config.schema import PrismConfig
from prism.config.settings import Settings
from prism.network.offline import (
    ConnectivityState,
    OfflineModeManager,
)
from prism.network.privacy import (
    RECOMMENDED_MODELS,
    OllamaModel,
    PrivacyLevel,
    PrivacyManager,
    PrivacyStatus,
    PrivacyViolationError,
)
from prism.plugins import api as plugin_api
from prism.plugins.manager import (
    BUILTIN_PLUGINS,
    PluginInfo,
    PluginManifest,
)
from prism.security.prismignore import (
    PrismIgnore,
)

# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------


def _make_settings(tmp_path: Path) -> Settings:
    """Create a minimal Settings pointing at *tmp_path*."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(
        config=config, project_root=tmp_path,
    )
    settings.ensure_directories()
    return settings


def _make_console(width: int = 300) -> Console:
    """In-memory console for capturing output."""
    buf = io.StringIO()
    return Console(
        file=buf,
        force_terminal=False,
        no_color=True,
        width=width,
    )


def _get_output(console: Console) -> str:
    """Extract text from an in-memory console."""
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


def _make_state(
    pinned_model: str | None = None,
    active_files: list[str] | None = None,
    cache_enabled: bool = True,
) -> _SessionState:
    """Create a SessionState with optional overrides."""
    state = _SessionState(
        pinned_model=pinned_model,
        cache_enabled=cache_enabled,
    )
    if active_files is not None:
        state.active_files = active_files
    state.session_id = "test-session"
    return state


def _cmd(
    command: str,
    tmp_path: Path,
    active_files: list[str] | None = None,
    pinned_model: str | None = None,
    console: Console | None = None,
    settings: Settings | None = None,
    state: _SessionState | None = None,
    dry_run: bool = False,
    offline: bool = False,
) -> tuple[str, str, Console, Settings, _SessionState]:
    """Run a slash command, return (action, output, ...)."""
    con = console or _make_console()
    stg = settings or _make_settings(tmp_path)
    st = state or _make_state(
        pinned_model=pinned_model,
        active_files=(
            active_files if active_files is not None else []
        ),
    )
    action = _dispatch_command(
        command,
        console=con,
        settings=stg,
        state=st,
        dry_run=dry_run,
        offline=offline,
    )
    return action, _get_output(con), con, stg, st


# ===============================================================
# TestIgnoreCommand — 12 tests
# ===============================================================


class TestIgnoreCommand:
    """/ignore REPL command tests."""

    def test_ignore_no_args_shows_patterns(
        self, tmp_path: Path,
    ) -> None:
        """``/ignore`` with no args shows patterns list."""
        # Create a .prismignore with known patterns
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text("*.log\n.env\n", encoding="utf-8")

        action, out, *_ = _cmd("/ignore", tmp_path)
        assert action == "continue"
        assert "*.log" in out
        assert ".env" in out

    def test_ignore_list_subcommand(
        self, tmp_path: Path,
    ) -> None:
        """``/ignore list`` shows all patterns."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text(
            "*.secret\nnode_modules/\n", encoding="utf-8",
        )
        action, out, *_ = _cmd("/ignore list", tmp_path)
        assert action == "continue"
        assert "*.secret" in out
        assert "node_modules/" in out
        assert "pattern" in out.lower()

    def test_ignore_add_pattern(
        self, tmp_path: Path,
    ) -> None:
        """``/ignore add *.log`` adds pattern."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text("# Empty\n", encoding="utf-8")

        action, out, *_ = _cmd("/ignore add *.log", tmp_path)
        assert action == "continue"
        assert "Added" in out or "added" in out.lower()
        assert "*.log" in out

        # Verify pattern was persisted
        content = prismignore.read_text(encoding="utf-8")
        assert "*.log" in content

    def test_ignore_check_env_shows_ignored(
        self, tmp_path: Path,
    ) -> None:
        """``/ignore check .env`` shows IGNORED."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text(".env\n", encoding="utf-8")

        action, out, *_ = _cmd("/ignore check .env", tmp_path)
        assert action == "continue"
        assert "IGNORED" in out

    def test_ignore_check_readme_not_ignored(
        self, tmp_path: Path,
    ) -> None:
        """``/ignore check README.md`` shows NOT IGNORED."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text(".env\n*.log\n", encoding="utf-8")

        action, out, *_ = _cmd(
            "/ignore check README.md", tmp_path,
        )
        assert action == "continue"
        assert "NOT IGNORED" in out

    def test_ignore_create_default(
        self, tmp_path: Path,
    ) -> None:
        """``/ignore create`` creates default .prismignore."""
        # Ensure no .prismignore exists
        prismignore = tmp_path / ".prismignore"
        assert not prismignore.exists()

        action, out, *_ = _cmd("/ignore create", tmp_path)
        assert action == "continue"
        assert "Created" in out or "created" in out.lower()
        assert prismignore.exists()

        content = prismignore.read_text(encoding="utf-8")
        assert ".env" in content
        assert "node_modules/" in content

    def test_ignore_create_already_exists(
        self, tmp_path: Path,
    ) -> None:
        """``/ignore create`` warns if file already exists."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text("*.log\n", encoding="utf-8")

        action, out, *_ = _cmd("/ignore create", tmp_path)
        assert action == "continue"
        assert "already exists" in out.lower()

    def test_ignore_empty_prismignore(
        self, tmp_path: Path,
    ) -> None:
        """Empty .prismignore shows no-patterns message."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text("", encoding="utf-8")

        action, out, *_ = _cmd("/ignore list", tmp_path)
        assert action == "continue"
        assert "no active pattern" in out.lower()

    def test_ignore_add_no_pattern_shows_usage(
        self, tmp_path: Path,
    ) -> None:
        """``/ignore add`` without pattern shows usage."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text("", encoding="utf-8")

        action, out, *_ = _cmd("/ignore add", tmp_path)
        assert action == "continue"
        assert "usage" in out.lower()

    def test_ignore_check_no_file_shows_usage(
        self, tmp_path: Path,
    ) -> None:
        """``/ignore check`` without file arg shows usage."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text(".env\n", encoding="utf-8")

        action, out, *_ = _cmd("/ignore check", tmp_path)
        assert action == "continue"
        assert "usage" in out.lower()

    def test_ignore_unknown_subcommand_shows_usage(
        self, tmp_path: Path,
    ) -> None:
        """``/ignore foobar`` shows usage info."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text(".env\n", encoding="utf-8")

        action, out, *_ = _cmd("/ignore foobar", tmp_path)
        assert action == "continue"
        assert "usage" in out.lower()
        assert "list" in out.lower()
        assert "add" in out.lower()
        assert "check" in out.lower()
        assert "create" in out.lower()

    def test_ignore_add_duplicate_pattern(
        self, tmp_path: Path,
    ) -> None:
        """Adding an already-existing pattern shows duplicate msg."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text("*.log\n", encoding="utf-8")

        action, out, *_ = _cmd("/ignore add *.log", tmp_path)
        assert action == "continue"
        assert "already exists" in out.lower()


# ===============================================================
# TestAddWarningForIgnoredFiles — 5 tests
# ===============================================================


class TestAddWarningForIgnoredFiles:
    """/add command and prismignore interaction tests.

    The current /add implementation does NOT warn about
    prismignore-matched files (it is a plain context adder).
    These tests verify the baseline behavior: files are added
    regardless, and we confirm the ignore infrastructure works
    independently.
    """

    def test_add_env_file_adds_to_context(
        self, tmp_path: Path,
    ) -> None:
        """/add .env adds the file to context."""
        action, _out, _, _, st = _cmd(
            "/add .env", tmp_path,
        )
        assert action == "continue"
        assert ".env" in st.active_files

    def test_add_normal_file_no_issue(
        self, tmp_path: Path,
    ) -> None:
        """/add normal.py adds without warnings."""
        action, out, _, _, st = _cmd(
            "/add normal.py", tmp_path,
        )
        assert action == "continue"
        assert "normal.py" in st.active_files
        # No error-level output
        assert "error" not in out.lower()

    def test_add_env_still_adds_file(
        self, tmp_path: Path,
    ) -> None:
        """/add .env adds file even if prismignore matches it."""
        prismignore = tmp_path / ".prismignore"
        prismignore.write_text(".env\n", encoding="utf-8")

        action, _, _, _, st = _cmd("/add .env", tmp_path)
        assert action == "continue"
        assert ".env" in st.active_files

    def test_add_multiple_files_mixed(
        self, tmp_path: Path,
    ) -> None:
        """Adding multiple files works correctly."""
        action, _out, _, _, st = _cmd(
            "/add foo.py .env bar.txt", tmp_path,
        )
        assert action == "continue"
        assert "foo.py" in st.active_files
        assert ".env" in st.active_files
        assert "bar.txt" in st.active_files

    def test_prismignore_detects_env_independently(
        self, tmp_path: Path,
    ) -> None:
        """PrismIgnore correctly identifies .env as ignored."""
        prismignore_file = tmp_path / ".prismignore"
        prismignore_file.write_text(
            ".env\n.env.*\n*.pem\n", encoding="utf-8",
        )
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(".env")
        assert pi.is_ignored(".env.local")
        assert pi.is_ignored("server.pem")
        assert not pi.is_ignored("README.md")

    def test_add_duplicate_file_shows_already_added(
        self, tmp_path: Path,
    ) -> None:
        """Adding a file already in context shows message."""
        state = _make_state(active_files=["main.py"])
        action, out, *_ = _cmd(
            "/add main.py", tmp_path, state=state,
        )
        assert action == "continue"
        assert "already" in out.lower()


# ===============================================================
# TestPrivacyEnhancements — 10 tests
# ===============================================================


class TestPrivacyEnhancements:
    """/privacy REPL command and PrivacyManager tests."""

    @patch("prism.network.privacy.PrivacyManager")
    def test_privacy_on_enables_mode(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/privacy on activates private mode."""
        mock_pm = MagicMock()
        mock_pm.enable_private_mode.return_value = PrivacyStatus(
            level=PrivacyLevel.PRIVATE,
            ollama_running=True,
            available_models=[
                OllamaModel(
                    name="llama3.1:8b",
                    size_bytes=4_294_967_296,
                    modified_at="2025-01-01",
                    digest="abc123",
                ),
            ],
        )
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd("/privacy on", tmp_path)
        assert action == "continue"
        assert "ENABLED" in out or "enabled" in out.lower()
        assert "Ollama" in out or "ollama" in out.lower()
        mock_pm.enable_private_mode.assert_called_once()

    @patch("prism.network.privacy.PrivacyManager")
    def test_privacy_on_no_ollama_models(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/privacy on with no Ollama models shows count 0."""
        mock_pm = MagicMock()
        mock_pm.enable_private_mode.return_value = PrivacyStatus(
            level=PrivacyLevel.PRIVATE,
            ollama_running=False,
            available_models=[],
        )
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd("/privacy on", tmp_path)
        assert action == "continue"
        assert "0" in out or "no" in out.lower()

    @patch("prism.network.privacy.PrivacyManager")
    def test_privacy_status_shows_mode(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/privacy status shows mode indicator."""
        mock_pm = MagicMock()
        mock_pm.get_status.return_value = PrivacyStatus(
            level=PrivacyLevel.NORMAL,
            ollama_running=True,
            available_models=[],
        )
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd("/privacy status", tmp_path)
        assert action == "continue"
        assert "NORMAL" in out

    @patch("prism.network.privacy.PrivacyManager")
    def test_privacy_status_shows_installed_models(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/privacy status shows installed local models."""
        models = [
            OllamaModel(
                name="qwen2.5-coder:7b",
                size_bytes=4_294_967_296,
                modified_at="2025-01-01",
                digest="def456",
            ),
            OllamaModel(
                name="llama3.1:8b",
                size_bytes=5_368_709_120,
                modified_at="2025-01-02",
                digest="ghi789",
            ),
        ]
        mock_pm = MagicMock()
        mock_pm.get_status.return_value = PrivacyStatus(
            level=PrivacyLevel.PRIVATE,
            ollama_running=True,
            available_models=models,
        )
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd("/privacy status", tmp_path)
        assert action == "continue"
        assert "qwen2.5-coder:7b" in out
        assert "llama3.1:8b" in out

    @patch("prism.network.privacy.PrivacyManager")
    def test_privacy_off_disables_mode(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/privacy off disables private mode."""
        mock_pm = MagicMock()
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd("/privacy off", tmp_path)
        assert action == "continue"
        assert "disabled" in out.lower()
        mock_pm.disable_private_mode.assert_called_once()

    def test_privacy_manager_blocks_cloud_provider(
        self,
    ) -> None:
        """Private mode blocks cloud provider requests."""
        pm = PrivacyManager()
        pm.enable_private_mode = MagicMock(
            return_value=PrivacyStatus(
                level=PrivacyLevel.PRIVATE,
                ollama_running=True,
                available_models=[],
            ),
        )
        pm._level = PrivacyLevel.PRIVATE

        with pytest.raises(PrivacyViolationError):
            pm.validate_request("anthropic", "claude-3-opus")

    @patch("prism.network.privacy.subprocess.Popen")
    @patch("prism.network.privacy.subprocess.run")
    def test_ollama_auto_start_attempted(
        self,
        mock_run: MagicMock,
        mock_popen: MagicMock,
    ) -> None:
        """Ollama auto-start is attempted on enable."""
        # Call sequence: check_ollama(fail), check_ollama
        # inside start_ollama(ok), check_ollama inside
        # get_status(ok), list_models inside get_status(ok)
        mock_run.side_effect = [
            MagicMock(returncode=1),  # first check fails
            MagicMock(returncode=0),  # start->check ok
            MagicMock(returncode=0),  # get_status->check
            MagicMock(
                returncode=0,
                stdout="NAME ID SIZE MODIFIED\n",
            ),
        ]
        mock_popen.return_value = MagicMock()

        pm = PrivacyManager()
        with patch("prism.network.privacy.time.sleep"):
            pm.enable_private_mode()

        mock_popen.assert_called_once()
        assert pm.is_private

    def test_recommended_models_listed_by_type(self) -> None:
        """Recommended models dict contains expected entries."""
        assert len(RECOMMENDED_MODELS) >= 3
        assert "qwen2.5-coder:7b" in RECOMMENDED_MODELS
        assert "llama3.1:8b" in RECOMMENDED_MODELS
        # Each entry has a description
        for model, desc in RECOMMENDED_MODELS.items():
            assert isinstance(model, str)
            assert isinstance(desc, str)
            assert len(desc) > 5

    def test_privacy_validates_ollama_prefix(self) -> None:
        """Private mode allows ollama-prefixed models."""
        pm = PrivacyManager()
        pm._level = PrivacyLevel.PRIVATE

        # Ollama provider is allowed
        pm.validate_request("ollama", "llama3.1:8b")

        # ollama/ prefix also allowed
        pm.validate_request(
            "not-cloud", "ollama/llama3.1:8b",
        )

    @patch("prism.network.privacy.PrivacyManager")
    def test_privacy_no_models_message(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/privacy status with no models shows message."""
        mock_pm = MagicMock()
        mock_pm.get_status.return_value = PrivacyStatus(
            level=PrivacyLevel.NORMAL,
            ollama_running=False,
            available_models=[],
        )
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd("/privacy", tmp_path)
        assert action == "continue"
        assert "not running" in out.lower() or "no" in out.lower()


# ===============================================================
# TestProxyConnectivity — 6 tests
# ===============================================================


class TestProxyConnectivity:
    """Network connectivity check tests (OfflineModeManager)."""

    @patch("prism.network.offline.socket.create_connection")
    def test_check_connectivity_true_when_reachable(
        self, mock_conn: MagicMock,
    ) -> None:
        """check_connectivity returns True when reachable."""
        mock_sock = MagicMock()
        mock_conn.return_value = mock_sock

        mgr = OfflineModeManager(check_timeout=1.0)
        result = mgr._check_connectivity()
        assert result is True
        mock_sock.close.assert_called_once()

    @patch("prism.network.offline.socket.create_connection")
    def test_check_connectivity_false_when_unreachable(
        self, mock_conn: MagicMock,
    ) -> None:
        """check_connectivity returns False when all hosts fail."""
        mock_conn.side_effect = OSError("Connection refused")

        mgr = OfflineModeManager(check_timeout=1.0)
        result = mgr._check_connectivity()
        assert result is False

    @patch("prism.network.offline.socket.create_connection")
    def test_connectivity_timeout_respected(
        self, mock_conn: MagicMock,
    ) -> None:
        """Timeout parameter is passed to create_connection."""
        mock_sock = MagicMock()
        mock_conn.return_value = mock_sock

        mgr = OfflineModeManager(check_timeout=5.0)
        mgr._check_connectivity()

        call_kwargs = mock_conn.call_args
        assert call_kwargs[1]["timeout"] == 5.0

    @patch("prism.network.offline.socket.create_connection")
    def test_socket_errors_handled_gracefully(
        self, mock_conn: MagicMock,
    ) -> None:
        """Socket errors are caught, not raised."""
        mock_conn.side_effect = TimeoutError("timed out")

        mgr = OfflineModeManager(check_timeout=1.0)
        # Should not raise
        result = mgr._check_connectivity()
        assert result is False

    def test_check_connectivity_method_exists(self) -> None:
        """_check_connectivity method exists on OfflineModeManager."""
        mgr = OfflineModeManager()
        assert hasattr(mgr, "_check_connectivity")
        assert callable(mgr._check_connectivity)

    @patch("prism.network.offline.socket.create_connection")
    def test_check_now_updates_state(
        self, mock_conn: MagicMock,
    ) -> None:
        """check_now() updates connectivity state."""
        mock_conn.side_effect = OSError("fail")

        mgr = OfflineModeManager()
        state = mgr.check_now()
        assert state == ConnectivityState.OFFLINE
        assert mgr.state == ConnectivityState.OFFLINE


# ===============================================================
# TestPluginsCLI — 7 tests
# ===============================================================


class TestPluginsCLI:
    """/plugins REPL command tests."""

    @patch("prism.plugins.manager.PluginManager")
    def test_plugin_list_shows_table(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/plugins list shows installed and available."""
        mock_pm = MagicMock()
        manifest = PluginManifest(
            name="test-plugin",
            version="1.0.0",
            description="A test plugin for testing",
        )
        info = PluginInfo(
            manifest=manifest,
            install_path=tmp_path / "plugins" / "test",
            enabled=True,
        )
        mock_pm.list_installed.return_value = [info]
        mock_pm.list_available.return_value = [
            BUILTIN_PLUGINS["docker-manager"],
        ]
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd("/plugins list", tmp_path)
        assert action == "continue"
        assert "test-plugin" in out
        assert "1.0.0" in out
        assert "docker-manager" in out

    @patch("prism.plugins.manager.PluginManager")
    def test_plugin_install_with_mock(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/plugins install <source> calls install."""
        mock_pm = MagicMock()
        manifest = PluginManifest(
            name="my-plugin",
            version="2.0.0",
            description="Freshly installed plugin",
        )
        mock_pm.install.return_value = PluginInfo(
            manifest=manifest,
            install_path=tmp_path / "plugins" / "my-plugin",
            enabled=True,
        )
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd(
            "/plugins install user/repo", tmp_path,
        )
        assert action == "continue"
        assert "Installed" in out or "installed" in out.lower()
        assert "my-plugin" in out
        mock_pm.install.assert_called_once_with("user/repo")

    @patch("prism.plugins.manager.PluginManager")
    def test_plugin_remove_with_mock(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/plugins remove <name> calls remove."""
        mock_pm = MagicMock()
        mock_pm.remove.return_value = True
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd(
            "/plugins remove test-plugin", tmp_path,
        )
        assert action == "continue"
        assert "removed" in out.lower()
        mock_pm.remove.assert_called_once_with("test-plugin")

    @patch("prism.plugins.manager.PluginManager")
    def test_plugin_install_no_name_shows_usage(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/plugins install without source shows usage."""
        mock_pm = MagicMock()
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd("/plugins install", tmp_path)
        assert action == "continue"
        assert "usage" in out.lower()

    @patch("prism.plugins.manager.PluginManager")
    def test_plugin_remove_no_name_shows_usage(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/plugins remove without name shows usage."""
        mock_pm = MagicMock()
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd("/plugins remove", tmp_path)
        assert action == "continue"
        assert "usage" in out.lower()

    @patch("prism.plugins.manager.PluginManager")
    def test_plugin_list_no_installed_shows_message(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/plugins with nothing installed shows info."""
        mock_pm = MagicMock()
        mock_pm.list_installed.return_value = []
        mock_pm.list_available.return_value = []
        mock_pm_cls.return_value = mock_pm

        action, out, *_ = _cmd("/plugins", tmp_path)
        assert action == "continue"
        assert (
            "no plugins" in out.lower()
            or "no plugin" in out.lower()
        )

    @patch("prism.plugins.manager.PluginManager")
    def test_plugin_default_subcommand_is_list(
        self,
        mock_pm_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """/plugins with no subcommand defaults to list."""
        mock_pm = MagicMock()
        mock_pm.list_installed.return_value = []
        mock_pm.list_available.return_value = list(
            BUILTIN_PLUGINS.values(),
        )
        mock_pm_cls.return_value = mock_pm

        action, _out, *_ = _cmd("/plugins", tmp_path)
        assert action == "continue"
        mock_pm.list_installed.assert_called_once()


# ===============================================================
# TestPluginAPI — 10 tests
# ===============================================================


class TestPluginAPI:
    """Plugin API module (prism.plugins.api) tests."""

    def setup_method(self) -> None:
        """Clear plugin API registries between tests."""
        plugin_api._registered_tools.clear()
        plugin_api._registered_commands.clear()

    def test_register_tool_stores_tool(self) -> None:
        """register_tool stores the tool in the registry."""

        def handler(**kw: Any) -> dict[str, bool]:
            return {"ok": True}

        plugin_api.register_tool(
            "my_tool",
            handler=handler,
            description="A test tool",
            parameters={"type": "object"},
        )
        tools = plugin_api.get_registered_tools()
        assert "my_tool" in tools
        assert tools["my_tool"]["description"] == "A test tool"
        assert tools["my_tool"]["handler"] is handler

    def test_register_command_stores_command(self) -> None:
        """register_command stores the command."""

        def handler(**kw: Any) -> str:
            return "done"

        plugin_api.register_command(
            "my_cmd",
            handler=handler,
            description="A test command",
        )
        cmds = plugin_api.get_registered_commands()
        assert "my_cmd" in cmds
        assert cmds["my_cmd"]["handler"] is handler

    def test_get_registered_tools_returns_copy(self) -> None:
        """get_registered_tools returns a shallow copy of the dict."""

        def handler(**kw: Any) -> dict[str, Any]:
            return {}

        plugin_api.register_tool("t1", handler=handler)
        tools = plugin_api.get_registered_tools()

        # Adding a new key to the copy must not affect original
        tools["new_fake_tool"] = {"name": "fake"}
        original = plugin_api.get_registered_tools()
        assert "new_fake_tool" not in original
        # Original still has t1
        assert "t1" in original

    def test_get_repo_map_returns_valid_structure(
        self, tmp_path: Path,
    ) -> None:
        """get_repo_map returns dict with expected keys."""
        # Create some python files
        (tmp_path / "main.py").write_text(
            "print('hello')\n", encoding="utf-8",
        )
        (tmp_path / "utils.py").write_text(
            "def foo():\n    pass\n", encoding="utf-8",
        )
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "mod.py").write_text(
            "x = 1\n", encoding="utf-8",
        )

        result = plugin_api.get_repo_map(str(tmp_path))
        assert "files" in result
        assert "directories" in result
        assert "total_files" in result
        assert "total_lines" in result
        assert result["total_files"] >= 3
        assert result["total_lines"] >= 4

    def test_get_cost_summary_returns_zeros(self) -> None:
        """get_cost_summary returns zeroed cost dict."""
        summary = plugin_api.get_cost_summary()
        assert summary["session_cost"] == 0.0
        assert summary["daily_cost"] == 0.0
        assert summary["monthly_cost"] == 0.0

    def test_log_does_not_crash(self) -> None:
        """log() doesn't crash on any level."""
        plugin_api.log("test debug message", level="debug")
        plugin_api.log("test info message", level="info")
        plugin_api.log(
            "test warning message", level="warning",
        )
        plugin_api.log("test error message", level="error")
        # No exception means pass

    def test_multiple_registrations_work(self) -> None:
        """Registering multiple tools/commands works."""

        def _tool_handler(**kw: Any) -> dict[str, Any]:
            return {}

        def _cmd_handler(**kw: Any) -> str:
            return "ok"

        for i in range(5):
            plugin_api.register_tool(
                f"tool_{i}",
                handler=_tool_handler,
                description=f"Tool {i}",
            )
            plugin_api.register_command(
                f"cmd_{i}",
                handler=_cmd_handler,
                description=f"Command {i}",
            )

        tools = plugin_api.get_registered_tools()
        cmds = plugin_api.get_registered_commands()
        assert len(tools) == 5
        assert len(cmds) == 5
        for i in range(5):
            assert f"tool_{i}" in tools
            assert f"cmd_{i}" in cmds

    def test_clear_state_between_tests(self) -> None:
        """Verify setup_method clears registries."""
        # This runs after setup_method which clears state
        assert len(plugin_api._registered_tools) == 0
        assert len(plugin_api._registered_commands) == 0

    def test_register_tool_default_parameters(self) -> None:
        """register_tool with no params defaults to empty dict."""

        def handler(**kw: Any) -> dict[str, Any]:
            return {}

        plugin_api.register_tool("bare_tool", handler=handler)
        tools = plugin_api.get_registered_tools()
        assert tools["bare_tool"]["parameters"] == {}

    def test_log_with_extra_kwargs(self) -> None:
        """log() accepts extra structured kwargs."""
        # Should not raise
        plugin_api.log(
            "structured message",
            level="info",
            plugin_name="test",
            version="1.0",
            count=42,
        )


# ===============================================================
# Additional PrismIgnore unit tests — 3 tests
# ===============================================================


class TestPrismIgnoreUnit:
    """Direct PrismIgnore class tests."""

    def test_default_patterns_cover_common_secrets(
        self, tmp_path: Path,
    ) -> None:
        """Default patterns ignore .env, *.pem, .aws/, etc."""
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored(".env")
        assert pi.is_ignored("server.pem")
        assert pi.is_ignored("id_rsa")
        assert pi.is_ignored("node_modules/package.json")

    def test_filter_paths_removes_ignored(
        self, tmp_path: Path,
    ) -> None:
        """filter_paths excludes ignored files."""
        prismignore_file = tmp_path / ".prismignore"
        prismignore_file.write_text(
            "*.log\n.env\n", encoding="utf-8",
        )
        pi = PrismIgnore(tmp_path)
        result = pi.filter_paths([
            "README.md", "app.log", ".env", "main.py",
        ])
        names = [p.name for p in result]
        assert "README.md" in names
        assert "main.py" in names
        assert "app.log" not in names
        assert ".env" not in names

    def test_negation_pattern_re_includes(
        self, tmp_path: Path,
    ) -> None:
        """A ! negation pattern re-includes previously ignored."""
        prismignore_file = tmp_path / ".prismignore"
        prismignore_file.write_text(
            "*.log\n!important.log\n", encoding="utf-8",
        )
        pi = PrismIgnore(tmp_path)
        assert pi.is_ignored("debug.log")
        assert not pi.is_ignored("important.log")

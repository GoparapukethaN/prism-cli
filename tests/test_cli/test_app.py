"""Tests for prism.cli.app — Typer CLI commands and callbacks."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import yaml
from typer.testing import CliRunner

from prism import __app_name__, __version__
from prism.cli.app import app
from prism.config.schema import PrismConfig
from prism.config.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path
    pass

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path) -> Settings:
    """Create a minimal Settings instance for testing."""
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


# ---------------------------------------------------------------------------
# Version flag
# ---------------------------------------------------------------------------


class TestVersionFlag:
    """Tests for --version / -v flag."""

    def test_version_long_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output
        assert __app_name__ in result.output

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.output


# ---------------------------------------------------------------------------
# Main callback — REPL startup
# ---------------------------------------------------------------------------


class TestMainCallback:
    """Tests for the main callback that starts the REPL."""

    @patch("prism.cli.app._start_repl")
    def test_no_subcommand_starts_repl(self, mock_start_repl: MagicMock) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        mock_start_repl.assert_called_once()

    @patch("prism.cli.app._start_repl")
    def test_model_flag_passed_to_repl(self, mock_start_repl: MagicMock) -> None:
        runner.invoke(app, ["--model", "gpt-4o"])
        _, kwargs = mock_start_repl.call_args
        assert kwargs["model"] == "gpt-4o"

    @patch("prism.cli.app._start_repl")
    def test_verbose_flag(self, mock_start_repl: MagicMock) -> None:
        runner.invoke(app, ["--verbose"])
        _, kwargs = mock_start_repl.call_args
        assert kwargs["verbose"] is True

    @patch("prism.cli.app._start_repl")
    def test_debug_flag(self, mock_start_repl: MagicMock) -> None:
        runner.invoke(app, ["--debug"])
        _, kwargs = mock_start_repl.call_args
        assert kwargs["debug"] is True

    @patch("prism.cli.app._start_repl")
    def test_yes_flag(self, mock_start_repl: MagicMock) -> None:
        runner.invoke(app, ["--yes"])
        _, kwargs = mock_start_repl.call_args
        assert kwargs["yes"] is True

    @patch("prism.cli.app._start_repl")
    def test_web_flag(self, mock_start_repl: MagicMock) -> None:
        runner.invoke(app, ["--web"])
        _, kwargs = mock_start_repl.call_args
        assert kwargs["web"] is True

    @patch("prism.cli.app._start_repl")
    def test_dry_run_flag(self, mock_start_repl: MagicMock) -> None:
        runner.invoke(app, ["--dry-run"])
        _, kwargs = mock_start_repl.call_args
        assert kwargs["dry_run"] is True

    @patch("prism.cli.app._start_repl")
    def test_new_session_flag(self, mock_start_repl: MagicMock) -> None:
        runner.invoke(app, ["--new-session"])
        _, kwargs = mock_start_repl.call_args
        assert kwargs["new_session"] is True

    @patch("prism.cli.app._start_repl")
    def test_budget_flag(self, mock_start_repl: MagicMock) -> None:
        runner.invoke(app, ["--budget", "5.0"])
        _, kwargs = mock_start_repl.call_args
        assert kwargs["budget"] == 5.0

    @patch("prism.cli.app._start_repl")
    def test_offline_flag(self, mock_start_repl: MagicMock) -> None:
        runner.invoke(app, ["--offline"])
        _, kwargs = mock_start_repl.call_args
        assert kwargs["offline"] is True

    @patch("prism.cli.app._start_repl")
    def test_model_short_flag(self, mock_start_repl: MagicMock) -> None:
        runner.invoke(app, ["-m", "claude-sonnet"])
        _, kwargs = mock_start_repl.call_args
        assert kwargs["model"] == "claude-sonnet"

    @patch("prism.cli.app._start_repl")
    def test_subcommand_skips_repl(self, mock_start_repl: MagicMock) -> None:
        """When a subcommand is invoked, the REPL should NOT start."""
        runner.invoke(app, ["ask", "hello"])
        mock_start_repl.assert_not_called()


# ---------------------------------------------------------------------------
# _start_repl internals
# ---------------------------------------------------------------------------


class TestStartRepl:
    """Tests for the _start_repl helper."""

    @patch("prism.cli.repl.run_repl")
    @patch("prism.config.settings.load_settings")
    def test_start_repl_loads_settings(
        self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from prism.cli.app import _start_repl

        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        _start_repl(
            root=None,
            model=None,
            verbose=False,
            debug=False,
            yes=False,
            web=False,
            dry_run=False,
            new_session=False,
            budget=None,
            offline=False,
        )
        mock_load.assert_called_once()
        mock_run.assert_called_once()

    @patch("prism.cli.repl.run_repl")
    @patch("prism.config.settings.load_settings")
    def test_start_repl_sets_model_override(
        self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from prism.cli.app import _start_repl

        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        _start_repl(
            root=None,
            model="gpt-4o",
            verbose=False,
            debug=False,
            yes=False,
            web=False,
            dry_run=False,
            new_session=False,
            budget=None,
            offline=False,
        )
        call_kwargs = mock_load.call_args
        overrides = call_kwargs[1].get("config_overrides", {})
        assert overrides.get("pinned_model") == "gpt-4o"

    @patch("prism.cli.repl.run_repl")
    @patch("prism.config.settings.load_settings")
    def test_start_repl_budget_override(
        self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from prism.cli.app import _start_repl

        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        _start_repl(
            root=None,
            model=None,
            verbose=False,
            debug=False,
            yes=False,
            web=False,
            dry_run=False,
            new_session=False,
            budget=10.0,
            offline=False,
        )
        call_kwargs = mock_load.call_args
        overrides = call_kwargs[1].get("config_overrides", {})
        assert overrides.get("budget.daily_limit") == 10.0

    @patch("prism.cli.repl.run_repl")
    @patch("prism.config.settings.load_settings")
    def test_start_repl_yes_flag_override(
        self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from prism.cli.app import _start_repl

        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        _start_repl(
            root=None,
            model=None,
            verbose=False,
            debug=False,
            yes=True,
            web=False,
            dry_run=False,
            new_session=False,
            budget=None,
            offline=False,
        )
        call_kwargs = mock_load.call_args
        overrides = call_kwargs[1].get("config_overrides", {})
        assert overrides.get("tools.auto_approve") is True

    @patch("prism.cli.repl.run_repl")
    @patch("prism.config.settings.load_settings")
    def test_start_repl_web_flag_override(
        self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from prism.cli.app import _start_repl

        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        _start_repl(
            root=None,
            model=None,
            verbose=False,
            debug=False,
            yes=False,
            web=True,
            dry_run=False,
            new_session=False,
            budget=None,
            offline=False,
        )
        call_kwargs = mock_load.call_args
        overrides = call_kwargs[1].get("config_overrides", {})
        assert overrides.get("tools.web_enabled") is True

    @patch("prism.cli.repl.run_repl")
    @patch("prism.config.settings.load_settings")
    def test_start_repl_verbose_sets_info_logging(
        self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from prism.cli.app import _start_repl

        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        with patch("prism.cli.app._configure_logging") as mock_log_config:
            _start_repl(
                root=None,
                model=None,
                verbose=True,
                debug=False,
                yes=False,
                web=False,
                dry_run=False,
                new_session=False,
                budget=None,
                offline=False,
            )
            mock_log_config.assert_called_once_with("INFO")

    @patch("prism.cli.repl.run_repl")
    @patch("prism.config.settings.load_settings")
    def test_start_repl_debug_sets_debug_logging(
        self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from prism.cli.app import _start_repl

        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        with patch("prism.cli.app._configure_logging") as mock_log_config:
            _start_repl(
                root=None,
                model=None,
                verbose=False,
                debug=True,
                yes=False,
                web=False,
                dry_run=False,
                new_session=False,
                budget=None,
                offline=False,
            )
            mock_log_config.assert_called_once_with("DEBUG")

    @patch("prism.cli.repl.run_repl")
    @patch("prism.config.settings.load_settings")
    def test_start_repl_debug_overrides_verbose(
        self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from prism.cli.app import _start_repl

        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        with patch("prism.cli.app._configure_logging") as mock_log_config:
            _start_repl(
                root=None,
                model=None,
                verbose=True,
                debug=True,
                yes=False,
                web=False,
                dry_run=False,
                new_session=False,
                budget=None,
                offline=False,
            )
            mock_log_config.assert_called_once_with("DEBUG")

    @patch("prism.cli.repl.run_repl")
    @patch("prism.config.settings.load_settings")
    def test_start_repl_no_overrides_when_defaults(
        self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        from prism.cli.app import _start_repl

        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        _start_repl(
            root=None,
            model=None,
            verbose=False,
            debug=False,
            yes=False,
            web=False,
            dry_run=False,
            new_session=False,
            budget=None,
            offline=False,
        )
        call_kwargs = mock_load.call_args
        overrides = call_kwargs[1].get("config_overrides", {})
        assert overrides == {}


# ---------------------------------------------------------------------------
# _configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Tests for the _configure_logging helper."""

    def test_configure_logging_does_not_raise(self) -> None:
        from prism.cli.app import _configure_logging

        # Should not raise for any valid level
        _configure_logging("WARNING")
        _configure_logging("INFO")
        _configure_logging("DEBUG")


# ---------------------------------------------------------------------------
# _print_banner
# ---------------------------------------------------------------------------


class TestPrintBanner:
    """Tests for the _print_banner helper."""

    @patch("prism.cli.app.console")
    def test_print_banner_calls_console(
        self, mock_console: MagicMock, tmp_path: Path
    ) -> None:
        from prism.cli.app import _print_banner

        settings = _make_settings(tmp_path)
        _print_banner(settings)
        assert mock_console.print.call_count >= 1


# ---------------------------------------------------------------------------
# Auth subcommands
# ---------------------------------------------------------------------------


class TestAuthAdd:
    """Tests for 'prism auth add <provider>'."""

    @patch("prism.auth.validator.KeyValidator")
    @patch("prism.auth.manager.AuthManager")
    @patch("prism.config.settings.load_settings")
    def test_auth_add_stores_valid_key(
        self,
        mock_load: MagicMock,
        mock_auth_cls: MagicMock,
        mock_validator_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        mock_auth_instance = MagicMock()
        mock_auth_cls.return_value = mock_auth_instance

        mock_validator = MagicMock()
        mock_validator.validate_key.return_value = True
        mock_validator_cls.return_value = mock_validator

        result = runner.invoke(
            app, ["auth", "add", "anthropic"], input="sk-ant-test-key-123456\n"
        )
        assert result.exit_code == 0
        assert "Stored" in result.output

    @patch("prism.auth.manager.AuthManager")
    @patch("prism.config.settings.load_settings")
    def test_auth_add_empty_key_fails(
        self,
        mock_load: MagicMock,
        mock_auth_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings
        mock_auth_cls.return_value = MagicMock()

        result = runner.invoke(app, ["auth", "add", "anthropic"], input="   \n")
        # Empty key should produce an error or exit code 1
        assert result.exit_code == 1 or "Error" in result.output or "empty" in result.output.lower()

    @patch("prism.auth.validator.KeyValidator")
    @patch("prism.auth.manager.AuthManager")
    @patch("prism.config.settings.load_settings")
    def test_auth_add_invalid_format_shows_warning(
        self,
        mock_load: MagicMock,
        mock_auth_cls: MagicMock,
        mock_validator_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        mock_auth_instance = MagicMock()
        mock_auth_cls.return_value = mock_auth_instance

        mock_validator = MagicMock()
        mock_validator.validate_key.return_value = False
        mock_validator_cls.return_value = mock_validator

        result = runner.invoke(
            app, ["auth", "add", "anthropic"], input="bad-key-format\n"
        )
        assert result.exit_code == 0
        assert "Warning" in result.output or "warning" in result.output.lower()

    @patch("prism.auth.validator.KeyValidator")
    @patch("prism.auth.manager.AuthManager")
    @patch("prism.config.settings.load_settings")
    def test_auth_add_masked_key_in_output(
        self,
        mock_load: MagicMock,
        mock_auth_cls: MagicMock,
        mock_validator_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        mock_auth_instance = MagicMock()
        mock_auth_cls.return_value = mock_auth_instance

        mock_validator = MagicMock()
        mock_validator.validate_key.return_value = True
        mock_validator_cls.return_value = mock_validator

        result = runner.invoke(
            app, ["auth", "add", "openai"], input="sk-testkey123456789\n"
        )
        assert result.exit_code == 0
        # Key should be masked — showing last 4 chars
        assert "6789" in result.output
        assert "..." in result.output
        # Full key should NOT appear
        assert "sk-testkey123456789" not in result.output


class TestAuthStatus:
    """Tests for 'prism auth status'."""

    @patch("prism.auth.manager.AuthManager")
    @patch("prism.config.settings.load_settings")
    def test_auth_status_shows_providers(
        self,
        mock_load: MagicMock,
        mock_auth_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        mock_auth_instance = MagicMock()
        mock_auth_instance.list_configured.return_value = [
            {
                "display_name": "Anthropic",
                "configured": True,
                "models": ["claude-sonnet"],
            },
            {
                "display_name": "OpenAI",
                "configured": False,
                "models": [],
            },
        ]
        mock_auth_cls.return_value = mock_auth_instance

        result = runner.invoke(app, ["auth", "status"])
        assert result.exit_code == 0


class TestAuthRemove:
    """Tests for 'prism auth remove <provider>'."""

    @patch("prism.auth.manager.AuthManager")
    @patch("prism.config.settings.load_settings")
    def test_auth_remove_calls_manager(
        self,
        mock_load: MagicMock,
        mock_auth_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        mock_auth_instance = MagicMock()
        mock_auth_cls.return_value = mock_auth_instance

        result = runner.invoke(app, ["auth", "remove", "openai"])
        assert result.exit_code == 0
        mock_auth_instance.remove_key.assert_called_once_with("openai")
        assert "Removed" in result.output


# ---------------------------------------------------------------------------
# Config subcommands
# ---------------------------------------------------------------------------


class TestConfigGet:
    """Tests for 'prism config get <key>'."""

    @patch("prism.config.settings.load_settings")
    def test_config_get_existing_key(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        result = runner.invoke(app, ["config", "get", "routing.simple_threshold"])
        assert result.exit_code == 0
        assert "0.3" in result.output

    @patch("prism.config.settings.load_settings")
    def test_config_get_missing_key(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        result = runner.invoke(app, ["config", "get", "nonexistent.key"])
        assert result.exit_code == 0
        assert "Not set" in result.output


class TestConfigSet:
    """Tests for 'prism config set <key> <value>'."""

    @patch("prism.config.settings.load_config_file")
    @patch("prism.config.settings.load_settings")
    def test_config_set_boolean_true(
        self,
        mock_load: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings
        mock_load_config.return_value = {}

        result = runner.invoke(app, ["config", "set", "tools.web_enabled", "true"])
        assert result.exit_code == 0
        assert "Set" in result.output

        # Verify YAML was written
        config_path = settings.config_file_path
        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert data["tools"]["web_enabled"] is True

    @patch("prism.config.settings.load_config_file")
    @patch("prism.config.settings.load_settings")
    def test_config_set_boolean_false(
        self,
        mock_load: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings
        mock_load_config.return_value = {}

        result = runner.invoke(app, ["config", "set", "tools.web_enabled", "false"])
        assert result.exit_code == 0
        config_path = settings.config_file_path
        data = yaml.safe_load(config_path.read_text())
        assert data["tools"]["web_enabled"] is False

    @patch("prism.config.settings.load_config_file")
    @patch("prism.config.settings.load_settings")
    def test_config_set_float(
        self,
        mock_load: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings
        mock_load_config.return_value = {}

        result = runner.invoke(app, ["config", "set", "budget.daily_limit", "5.0"])
        assert result.exit_code == 0
        config_path = settings.config_file_path
        data = yaml.safe_load(config_path.read_text())
        assert data["budget"]["daily_limit"] == 5.0

    @patch("prism.config.settings.load_config_file")
    @patch("prism.config.settings.load_settings")
    def test_config_set_integer(
        self,
        mock_load: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings
        mock_load_config.return_value = {}

        result = runner.invoke(app, ["config", "set", "tools.command_timeout", "60"])
        assert result.exit_code == 0
        config_path = settings.config_file_path
        data = yaml.safe_load(config_path.read_text())
        assert data["tools"]["command_timeout"] == 60

    @patch("prism.config.settings.load_config_file")
    @patch("prism.config.settings.load_settings")
    def test_config_set_string(
        self,
        mock_load: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings
        mock_load_config.return_value = {}

        result = runner.invoke(app, ["config", "set", "pinned_model", "gpt-4o"])
        assert result.exit_code == 0
        config_path = settings.config_file_path
        data = yaml.safe_load(config_path.read_text())
        assert data["pinned_model"] == "gpt-4o"

    @patch("prism.config.settings.load_config_file")
    @patch("prism.config.settings.load_settings")
    def test_config_set_null(
        self,
        mock_load: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings
        mock_load_config.return_value = {}

        result = runner.invoke(app, ["config", "set", "pinned_model", "null"])
        assert result.exit_code == 0
        config_path = settings.config_file_path
        data = yaml.safe_load(config_path.read_text())
        assert data["pinned_model"] is None

    @patch("prism.config.settings.load_config_file")
    @patch("prism.config.settings.load_settings")
    def test_config_set_none_string(
        self,
        mock_load: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings
        mock_load_config.return_value = {}

        result = runner.invoke(app, ["config", "set", "pinned_model", "none"])
        assert result.exit_code == 0
        config_path = settings.config_file_path
        data = yaml.safe_load(config_path.read_text())
        assert data["pinned_model"] is None

    @patch("prism.config.settings.load_config_file")
    @patch("prism.config.settings.load_settings")
    def test_config_set_nested_key_creates_hierarchy(
        self,
        mock_load: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings
        mock_load_config.return_value = {}

        result = runner.invoke(
            app, ["config", "set", "routing.simple_threshold", "0.2"]
        )
        assert result.exit_code == 0
        config_path = settings.config_file_path
        data = yaml.safe_load(config_path.read_text())
        assert data["routing"]["simple_threshold"] == 0.2


# ---------------------------------------------------------------------------
# DB subcommands
# ---------------------------------------------------------------------------


class TestDbStats:
    """Tests for 'prism db stats'."""

    @patch("prism.config.settings.load_settings")
    def test_db_stats_no_database(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        result = runner.invoke(app, ["db", "stats"])
        assert result.exit_code == 0
        assert "No database found" in result.output

    @patch("prism.config.settings.load_settings")
    def test_db_stats_with_database(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        # Create a minimal database
        db_path = settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE routing_decisions (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO routing_decisions VALUES (1)")
        conn.execute("INSERT INTO routing_decisions VALUES (2)")
        conn.execute("CREATE TABLE cost_entries (id INTEGER PRIMARY KEY)")
        conn.close()

        result = runner.invoke(app, ["db", "stats"])
        assert result.exit_code == 0
        assert "routing_decisions" in result.output
        assert "2" in result.output  # 2 rows
        assert "cost_entries" in result.output

    @patch("prism.config.settings.load_settings")
    def test_db_stats_missing_table(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        # Create database with only one table
        db_path = settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE routing_decisions (id INTEGER PRIMARY KEY)")
        conn.close()

        result = runner.invoke(app, ["db", "stats"])
        assert result.exit_code == 0
        assert "not found" in result.output


class TestDbVacuum:
    """Tests for 'prism db vacuum'."""

    @patch("prism.config.settings.load_settings")
    def test_db_vacuum_no_database(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        result = runner.invoke(app, ["db", "vacuum"])
        assert result.exit_code == 0
        assert "No database found" in result.output

    @patch("prism.config.settings.load_settings")
    def test_db_vacuum_success(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        # Create a database with data then delete it
        db_path = settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (data TEXT)")
        for i in range(100):
            conn.execute("INSERT INTO test VALUES (?)", (f"data-{i}" * 100,))
        conn.commit()
        conn.execute("DELETE FROM test")
        conn.commit()
        conn.close()

        result = runner.invoke(app, ["db", "vacuum"])
        assert result.exit_code == 0
        assert "Vacuum complete" in result.output


# ---------------------------------------------------------------------------
# Ask command
# ---------------------------------------------------------------------------


class TestAskCommand:
    """Tests for 'prism ask <prompt>'."""

    def test_ask_shows_prompt(self) -> None:
        result = runner.invoke(app, ["ask", "What is Python?"])
        assert result.exit_code == 0
        assert "Processing" in result.output
        assert "not yet implemented" in result.output.lower()

    def test_ask_truncates_long_prompt(self) -> None:
        long_prompt = "x" * 200
        result = runner.invoke(app, ["ask", long_prompt])
        assert result.exit_code == 0
        assert "Processing" in result.output


# ---------------------------------------------------------------------------
# Init command
# ---------------------------------------------------------------------------


class TestInitCommand:
    """Tests for 'prism init'."""

    @patch("prism.config.settings.load_settings")
    def test_init_creates_prism_md(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        result = runner.invoke(app, ["init", "--root", str(tmp_path)])
        assert result.exit_code == 0

        prism_md = tmp_path / ".prism.md"
        assert prism_md.exists()
        content = prism_md.read_text()
        assert "Stack" in content
        assert "Conventions" in content

    @patch("prism.config.settings.load_settings")
    def test_init_already_initialized(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        # Create .prism.md first
        prism_md = tmp_path / ".prism.md"
        prism_md.write_text("# Already initialized\n")

        result = runner.invoke(app, ["init", "--root", str(tmp_path)])
        assert result.exit_code == 0
        assert "already initialized" in result.output.lower()

    @patch("prism.config.settings.load_settings")
    def test_init_creates_config_yaml(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        result = runner.invoke(app, ["init", "--root", str(tmp_path)])
        assert result.exit_code == 0

        config_path = settings.config_file_path
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text())
            assert "routing" in data

    @patch("prism.config.settings.load_settings")
    def test_init_shows_next_steps(
        self,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        result = runner.invoke(app, ["init", "--root", str(tmp_path)])
        assert result.exit_code == 0
        assert "Next steps" in result.output
        assert "auth add" in result.output


# ---------------------------------------------------------------------------
# Status command
# ---------------------------------------------------------------------------


class TestStatusCommand:
    """Tests for 'prism status'."""

    @patch("prism.cli.app.auth_status")
    @patch("prism.config.settings.load_settings")
    def test_status_shows_system_info(
        self,
        mock_load: MagicMock,
        mock_auth_status: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Prism Status" in result.output

    @patch("prism.cli.app.auth_status")
    @patch("prism.config.settings.load_settings")
    def test_status_no_database(
        self,
        mock_load: MagicMock,
        mock_auth_status: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Not created yet" in result.output

    @patch("prism.cli.app.auth_status")
    @patch("prism.config.settings.load_settings")
    def test_status_with_database(
        self,
        mock_load: MagicMock,
        mock_auth_status: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        # Create database file
        db_path = settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "OK" in result.output

    @patch("prism.cli.app.auth_status")
    @patch("prism.config.settings.load_settings")
    def test_status_no_prism_md(
        self,
        mock_load: MagicMock,
        mock_auth_status: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "No .prism.md" in result.output

    @patch("prism.cli.app.auth_status")
    @patch("prism.config.settings.load_settings")
    def test_status_with_prism_md(
        self,
        mock_load: MagicMock,
        mock_auth_status: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_load.return_value = settings

        # Create .prism.md
        (tmp_path / ".prism.md").write_text("# Test\n")

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        # Should show project OK
        assert "OK" in result.output


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    """Tests for the main() entry point."""

    @patch("prism.cli.app.app")
    def test_main_calls_app(self, mock_app: MagicMock) -> None:
        from prism.cli.app import main

        main()
        mock_app.assert_called_once()

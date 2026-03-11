"""Tests for prism.cli.completion — shell tab-completion support."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from prism.cli.completion import (
    _SHELL_RC_FILES,
    _detect_shell,
    _install_script,
    get_completions,
    install_completion,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    pass


# ---------------------------------------------------------------------------
# _SHELL_RC_FILES
# ---------------------------------------------------------------------------


class TestShellRcFiles:
    """Tests for the _SHELL_RC_FILES constant."""

    def test_is_dict(self) -> None:
        assert isinstance(_SHELL_RC_FILES, dict)

    def test_contains_bash(self) -> None:
        assert "bash" in _SHELL_RC_FILES

    def test_contains_zsh(self) -> None:
        assert "zsh" in _SHELL_RC_FILES

    def test_contains_fish(self) -> None:
        assert "fish" in _SHELL_RC_FILES

    def test_all_values_are_strings(self) -> None:
        for shell, path in _SHELL_RC_FILES.items():
            assert isinstance(path, str), f"Path for {shell} is not a string"


# ---------------------------------------------------------------------------
# _detect_shell
# ---------------------------------------------------------------------------


class TestDetectShell:
    """Tests for the _detect_shell function."""

    def test_detect_bash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/bash")
        assert _detect_shell() == "bash"

    def test_detect_zsh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/zsh")
        assert _detect_shell() == "zsh"

    def test_detect_fish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        assert _detect_shell() == "fish"

    def test_detect_unknown_defaults_to_bash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/csh")
        assert _detect_shell() == "bash"

    def test_detect_empty_shell_defaults_to_bash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "")
        assert _detect_shell() == "bash"

    def test_detect_no_shell_env_defaults_to_bash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SHELL", raising=False)
        assert _detect_shell() == "bash"

    def test_detect_shell_with_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/usr/local/bin/zsh")
        assert _detect_shell() == "zsh"


# ---------------------------------------------------------------------------
# _install_script
# ---------------------------------------------------------------------------


class TestInstallScript:
    """Tests for the _install_script function."""

    def test_install_bash_script(self, tmp_path: Path) -> None:
        rc_file = tmp_path / ".bashrc"

        with patch("prism.cli.completion._SHELL_RC_FILES", {"bash": str(rc_file)}):
            _install_script("bash", "# completion script\neval $(_PRISM_COMPLETE)")

        assert rc_file.exists()
        content = rc_file.read_text()
        assert "# prism shell completion" in content
        assert "eval $(_PRISM_COMPLETE)" in content

    def test_install_zsh_script(self, tmp_path: Path) -> None:
        rc_file = tmp_path / ".zshrc"

        with patch("prism.cli.completion._SHELL_RC_FILES", {"zsh": str(rc_file)}):
            _install_script("zsh", "# zsh completion\nautoload")

        assert rc_file.exists()
        content = rc_file.read_text()
        assert "# prism shell completion" in content
        assert "autoload" in content

    def test_install_fish_script(self, tmp_path: Path) -> None:
        fish_dir = tmp_path / ".config" / "fish" / "completions"
        fish_file = fish_dir / "prism.fish"

        with patch("prism.cli.completion._SHELL_RC_FILES", {"fish": str(fish_file)}):
            _install_script("fish", "# fish completion\ncomplete -c prism")

        assert fish_file.exists()
        content = fish_file.read_text()
        assert "complete -c prism" in content

    def test_install_skips_if_already_present(self, tmp_path: Path) -> None:
        rc_file = tmp_path / ".bashrc"
        rc_file.write_text("existing content\n# prism shell completion\nold script\n")

        with patch("prism.cli.completion._SHELL_RC_FILES", {"bash": str(rc_file)}):
            _install_script("bash", "new completion script")

        content = rc_file.read_text()
        assert "new completion script" not in content
        assert content.count("# prism shell completion") == 1

    def test_install_appends_to_existing_rc(self, tmp_path: Path) -> None:
        rc_file = tmp_path / ".bashrc"
        rc_file.write_text("# existing bashrc content\nalias ll='ls -la'\n")

        with patch("prism.cli.completion._SHELL_RC_FILES", {"bash": str(rc_file)}):
            _install_script("bash", "# my completion")

        content = rc_file.read_text()
        assert "alias ll='ls -la'" in content
        assert "# prism shell completion" in content
        assert "# my completion" in content


# ---------------------------------------------------------------------------
# install_completion
# ---------------------------------------------------------------------------


class TestInstallCompletion:
    """Tests for the install_completion function."""

    @patch("prism.cli.completion.subprocess.run")
    @patch("prism.cli.completion._install_script")
    def test_install_completion_auto_detects_shell(
        self,
        mock_install: MagicMock,
        mock_run: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SHELL", "/bin/zsh")
        mock_run.return_value = MagicMock(returncode=0, stdout="completion script")

        install_completion(shell="auto")

        mock_install.assert_called_once_with("zsh", "completion script")
        # Env var should be cleaned up
        assert "_PRISM_COMPLETE" not in os.environ

    @patch("prism.cli.completion.subprocess.run")
    @patch("prism.cli.completion._install_script")
    def test_install_completion_explicit_shell(
        self,
        mock_install: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="bash script")

        install_completion(shell="bash")

        mock_install.assert_called_once_with("bash", "bash script")

    def test_install_completion_unsupported_shell(self) -> None:
        # Should not raise, just log a warning
        install_completion(shell="powershell")

    @patch("prism.cli.completion.subprocess.run")
    def test_install_completion_generation_failed(
        self,
        mock_run: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        # Should not raise
        install_completion(shell="bash")

        assert "_PRISM_COMPLETE" not in os.environ

    @patch("prism.cli.completion.subprocess.run")
    def test_install_completion_empty_stdout(
        self,
        mock_run: MagicMock,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        # Should not raise, empty stdout means generation failed
        install_completion(shell="bash")

    @patch("prism.cli.completion.subprocess.run")
    def test_install_completion_exception_handled(
        self,
        mock_run: MagicMock,
    ) -> None:
        mock_run.side_effect = OSError("command not found")

        # Should not raise
        install_completion(shell="bash")

        assert "_PRISM_COMPLETE" not in os.environ

    @patch("prism.cli.completion.subprocess.run")
    def test_install_completion_timeout_handled(
        self,
        mock_run: MagicMock,
    ) -> None:
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=10)

        # Should not raise
        install_completion(shell="bash")

        assert "_PRISM_COMPLETE" not in os.environ

    @patch("prism.cli.completion.subprocess.run")
    @patch("prism.cli.completion._install_script")
    def test_install_completion_sets_env_var(
        self,
        mock_install: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        captured_env: dict[str, str] = {}

        def capture_env(*args: object, **kwargs: object) -> MagicMock:
            captured_env["_PRISM_COMPLETE"] = os.environ.get("_PRISM_COMPLETE", "")
            return MagicMock(returncode=0, stdout="script")

        mock_run.side_effect = capture_env

        install_completion(shell="zsh")

        assert captured_env["_PRISM_COMPLETE"] == "zsh_source"
        # Cleaned up after
        assert "_PRISM_COMPLETE" not in os.environ


# ---------------------------------------------------------------------------
# get_completions
# ---------------------------------------------------------------------------


class TestGetCompletions:
    """Tests for the get_completions function."""

    def test_complete_provider_anthropic(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="ant")
        assert "anthropic" in results

    def test_complete_provider_openai(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="ope")
        assert "openai" in results

    def test_complete_provider_google(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="goo")
        assert "google" in results

    def test_complete_provider_deepseek(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="dee")
        assert "deepseek" in results

    def test_complete_provider_groq(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="gro")
        assert "groq" in results

    def test_complete_provider_mistral(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="mis")
        assert "mistral" in results

    def test_complete_provider_ollama(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="oll")
        assert "ollama" in results

    def test_complete_command_help(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="/he")
        assert "/help" in results

    def test_complete_command_model(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="/mo")
        assert "/model" in results

    def test_complete_command_status(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="/st")
        assert "/status" in results

    def test_complete_command_clear(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="/cl")
        assert "/clear" in results

    def test_complete_empty_prefix_returns_all(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="")
        # Should contain all providers and commands
        assert len(results) >= 10  # 7 providers + several commands

    def test_complete_no_match(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="xyz")
        assert results == []

    def test_complete_multiple_matches(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="g")
        assert "google" in results
        assert "groq" in results

    def test_complete_slash_prefix(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="/")
        # Should return all commands
        assert len(results) >= 5

    def test_returns_list(self) -> None:
        results = get_completions(ctx=None, args=[], incomplete="")
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, str)

"""Tests for prism.cli.shell_completion — enhanced shell tab-completion."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from prism.cli.shell_completion import (
    CLI_COMMANDS,
    CLI_FLAGS,
    MODELS,
    REPL_COMMANDS,
    detect_shell,
    generate_bash_completion,
    generate_fish_completion,
    generate_zsh_completion,
    install_completion,
)

# ---------------------------------------------------------------------------
# TestBashCompletion
# ---------------------------------------------------------------------------


class TestBashCompletion:
    """Tests for generate_bash_completion."""

    def test_returns_string(self) -> None:
        script = generate_bash_completion()
        assert isinstance(script, str)
        assert len(script) > 100

    def test_contains_bash_function(self) -> None:
        script = generate_bash_completion()
        assert "_prism_completions()" in script

    def test_contains_complete_command(self) -> None:
        script = generate_bash_completion()
        assert "complete -F _prism_completions prism" in script

    def test_contains_all_cli_commands(self) -> None:
        script = generate_bash_completion()
        for cmd in CLI_COMMANDS:
            assert cmd in script, f"Missing CLI command: {cmd}"

    def test_contains_all_flags(self) -> None:
        script = generate_bash_completion()
        for flag in CLI_FLAGS:
            assert flag in script, f"Missing flag: {flag}"

    def test_contains_all_models(self) -> None:
        script = generate_bash_completion()
        for model in MODELS:
            assert model in script, f"Missing model: {model}"

    def test_contains_all_repl_commands(self) -> None:
        script = generate_bash_completion()
        for cmd in REPL_COMMANDS:
            assert cmd in script, f"Missing REPL command: {cmd}"

    def test_contains_model_case(self) -> None:
        script = generate_bash_completion()
        assert "--model)" in script

    def test_contains_project_case(self) -> None:
        script = generate_bash_completion()
        assert "--project)" in script

    def test_contains_file_completion_for_add(self) -> None:
        script = generate_bash_completion()
        assert "/add" in script
        assert "compgen -f" in script

    def test_slash_command_prefix_detection(self) -> None:
        script = generate_bash_completion()
        assert '/* ]' in script or "/*" in script

    def test_flag_prefix_detection(self) -> None:
        script = generate_bash_completion()
        assert '-*' in script


# ---------------------------------------------------------------------------
# TestZshCompletion
# ---------------------------------------------------------------------------


class TestZshCompletion:
    """Tests for generate_zsh_completion."""

    def test_returns_string(self) -> None:
        script = generate_zsh_completion()
        assert isinstance(script, str)
        assert len(script) > 100

    def test_contains_compdef(self) -> None:
        script = generate_zsh_completion()
        assert "#compdef prism" in script

    def test_contains_prism_function(self) -> None:
        script = generate_zsh_completion()
        assert "_prism()" in script

    def test_contains_arguments(self) -> None:
        script = generate_zsh_completion()
        assert "_arguments" in script

    def test_contains_all_cli_commands(self) -> None:
        script = generate_zsh_completion()
        for cmd in CLI_COMMANDS:
            assert cmd in script, f"Missing CLI command: {cmd}"

    def test_contains_all_flags(self) -> None:
        script = generate_zsh_completion()
        for flag in CLI_FLAGS:
            assert flag in script, f"Missing flag: {flag}"

    def test_contains_model_list(self) -> None:
        script = generate_zsh_completion()
        assert "models=(" in script

    def test_contains_describe(self) -> None:
        script = generate_zsh_completion()
        assert "_describe" in script

    def test_contains_compadd(self) -> None:
        script = generate_zsh_completion()
        assert "compadd -a models" in script

    def test_contains_files_fallback(self) -> None:
        script = generate_zsh_completion()
        assert "_files" in script


# ---------------------------------------------------------------------------
# TestFishCompletion
# ---------------------------------------------------------------------------


class TestFishCompletion:
    """Tests for generate_fish_completion."""

    def test_returns_string(self) -> None:
        script = generate_fish_completion()
        assert isinstance(script, str)
        assert len(script) > 100

    def test_contains_complete_command(self) -> None:
        script = generate_fish_completion()
        assert "complete -c prism" in script

    def test_contains_all_cli_commands(self) -> None:
        script = generate_fish_completion()
        for cmd in CLI_COMMANDS:
            assert f"-a '{cmd}'" in script, f"Missing CLI command: {cmd}"

    def test_contains_all_flags(self) -> None:
        script = generate_fish_completion()
        for flag in CLI_FLAGS:
            flag_long = flag.lstrip("-")
            assert f"-l '{flag_long}'" in script, f"Missing flag: {flag}"

    def test_contains_model_completions(self) -> None:
        script = generate_fish_completion()
        for model in MODELS:
            assert model in script, f"Missing model: {model}"

    def test_contains_fish_use_subcommand(self) -> None:
        script = generate_fish_completion()
        assert "__fish_use_subcommand" in script

    def test_contains_fish_seen_subcommand(self) -> None:
        script = generate_fish_completion()
        assert "__fish_seen_subcommand_from --model" in script

    def test_header_comment(self) -> None:
        script = generate_fish_completion()
        assert "# Prism CLI fish completion" in script


# ---------------------------------------------------------------------------
# TestInstallCompletion
# ---------------------------------------------------------------------------


class TestInstallCompletion:
    """Tests for install_completion."""

    def test_install_bash(self, tmp_path: Path) -> None:
        target = tmp_path / "bash_completion" / "prism"
        path = install_completion("bash", path=target)
        assert path == target
        assert target.exists()
        content = target.read_text()
        assert "_prism_completions()" in content

    def test_install_zsh(self, tmp_path: Path) -> None:
        target = tmp_path / "zsh_completion" / "_prism"
        path = install_completion("zsh", path=target)
        assert path == target
        assert target.exists()
        content = target.read_text()
        assert "#compdef prism" in content

    def test_install_fish(self, tmp_path: Path) -> None:
        target = tmp_path / "fish_completion" / "prism.fish"
        path = install_completion("fish", path=target)
        assert path == target
        assert target.exists()
        content = target.read_text()
        assert "complete -c prism" in content

    def test_unsupported_shell_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported shell"):
            install_completion("powershell")

    def test_unsupported_shell_message(self) -> None:
        with pytest.raises(ValueError, match="Use bash, zsh, or fish"):
            install_completion("csh")

    def test_custom_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "my_completions" / "prism_comp"
        path = install_completion("bash", path=custom)
        assert path == custom
        assert custom.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "dir" / "prism"
        install_completion("bash", path=target)
        assert target.exists()
        assert target.parent.is_dir()

    def test_install_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "prism_completion"
        target.write_text("old content")
        install_completion("bash", path=target)
        content = target.read_text()
        assert "old content" not in content
        assert "_prism_completions()" in content

    def test_install_bash_default_path_check(self, tmp_path: Path) -> None:
        """Verify the default path structure for bash."""
        # Just test with explicit path to avoid writing to home
        target = tmp_path / ".bash_completion.d" / "prism"
        install_completion("bash", path=target)
        assert target.exists()

    def test_install_zsh_default_path_check(self, tmp_path: Path) -> None:
        """Verify the default path structure for zsh."""
        target = tmp_path / ".zsh" / "completions" / "_prism"
        install_completion("zsh", path=target)
        assert target.exists()

    def test_install_fish_default_path_check(self, tmp_path: Path) -> None:
        """Verify the default path structure for fish."""
        target = tmp_path / ".config" / "fish" / "completions" / "prism.fish"
        install_completion("fish", path=target)
        assert target.exists()


# ---------------------------------------------------------------------------
# TestDetectShell
# ---------------------------------------------------------------------------


class TestDetectShell:
    """Tests for detect_shell."""

    def test_detect_zsh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/zsh")
        assert detect_shell() == "zsh"

    def test_detect_zsh_usr_local(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/usr/local/bin/zsh")
        assert detect_shell() == "zsh"

    def test_detect_fish(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/usr/bin/fish")
        assert detect_shell() == "fish"

    def test_detect_fish_homebrew(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/opt/homebrew/bin/fish")
        assert detect_shell() == "fish"

    def test_detect_bash_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SHELL", "/bin/bash")
        assert detect_shell() == "bash"

    def test_detect_unknown_defaults_to_bash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SHELL", "/bin/csh")
        assert detect_shell() == "bash"

    def test_detect_empty_defaults_to_bash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SHELL", "")
        assert detect_shell() == "bash"

    def test_detect_no_shell_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SHELL", raising=False)
        assert detect_shell() == "bash"


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constant lists."""

    def test_repl_commands_not_empty(self) -> None:
        assert len(REPL_COMMANDS) >= 30

    def test_repl_commands_all_start_with_slash(self) -> None:
        for cmd in REPL_COMMANDS:
            assert cmd.startswith("/"), f"REPL command missing /: {cmd}"

    def test_repl_commands_no_duplicates(self) -> None:
        assert len(REPL_COMMANDS) == len(set(REPL_COMMANDS))

    def test_cli_commands_not_empty(self) -> None:
        assert len(CLI_COMMANDS) >= 10

    def test_cli_commands_no_duplicates(self) -> None:
        assert len(CLI_COMMANDS) == len(set(CLI_COMMANDS))

    def test_cli_commands_contains_core(self) -> None:
        assert "ask" in CLI_COMMANDS
        assert "edit" in CLI_COMMANDS
        assert "run" in CLI_COMMANDS
        assert "init" in CLI_COMMANDS
        assert "auth" in CLI_COMMANDS

    def test_cli_flags_not_empty(self) -> None:
        assert len(CLI_FLAGS) >= 8

    def test_cli_flags_all_start_with_dash(self) -> None:
        for flag in CLI_FLAGS:
            assert flag.startswith("--"), f"Flag missing --: {flag}"

    def test_cli_flags_no_duplicates(self) -> None:
        assert len(CLI_FLAGS) == len(set(CLI_FLAGS))

    def test_cli_flags_contains_core(self) -> None:
        assert "--help" in CLI_FLAGS
        assert "--version" in CLI_FLAGS
        assert "--model" in CLI_FLAGS

    def test_models_not_empty(self) -> None:
        assert len(MODELS) >= 10

    def test_models_no_duplicates(self) -> None:
        assert len(MODELS) == len(set(MODELS))

    def test_models_contain_major_providers(self) -> None:
        model_str = " ".join(MODELS)
        assert "claude" in model_str
        assert "gpt-4" in model_str
        assert "gemini" in model_str
        assert "deepseek" in model_str
        assert "groq" in model_str
        assert "mistral" in model_str

    def test_models_all_strings(self) -> None:
        for model in MODELS:
            assert isinstance(model, str)
            assert len(model) > 3

    def test_repl_commands_all_strings(self) -> None:
        for cmd in REPL_COMMANDS:
            assert isinstance(cmd, str)

    def test_cli_commands_all_strings(self) -> None:
        for cmd in CLI_COMMANDS:
            assert isinstance(cmd, str)

    def test_cli_flags_all_strings(self) -> None:
        for flag in CLI_FLAGS:
            assert isinstance(flag, str)

"""Shell tab completion — bash, zsh, and fish completion for Prism CLI.

Generates shell-specific completion scripts that enable tab-completion for
Prism commands, flags, REPL slash-commands, and model names.  Also provides
helpers to detect the current shell and install the completion script to
the appropriate location.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Completion data — kept in module-level constants for easy testing
# ---------------------------------------------------------------------------

REPL_COMMANDS: list[str] = [
    "/help",
    "/cost",
    "/model",
    "/add",
    "/drop",
    "/compact",
    "/undo",
    "/web",
    "/status",
    "/clear",
    "/quit",
    "/cache",
    "/image",
    "/compare",
    "/history",
    "/restore",
    "/branch",
    "/branches",
    "/switch",
    "/merge",
    "/sandbox",
    "/queue",
    "/ignore",
    "/private",
    "/forecast",
    "/context",
    "/architecture",
    "/debate",
    "/aei",
    "/impact",
    "/test-gaps",
    "/deps",
    "/why",
    "/memory",
]

CLI_COMMANDS: list[str] = [
    "ask",
    "edit",
    "run",
    "init",
    "auth",
    "config",
    "projects",
    "plugins",
    "blame",
    "impact",
    "test-gaps",
    "deps",
    "version",
    "update",
]

CLI_FLAGS: list[str] = [
    "--help",
    "--version",
    "--model",
    "--private",
    "--offline",
    "--no-cache",
    "--debug",
    "--project",
    "--verbose",
    "--quiet",
]

MODELS: list[str] = [
    "claude-sonnet-4-20250514",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gemini/gemini-1.5-pro",
    "gemini/gemini-1.5-flash",
    "deepseek/deepseek-chat",
    "groq/llama-3.1-70b-versatile",
    "mistral/mistral-large-latest",
]


# ---------------------------------------------------------------------------
# Script generators
# ---------------------------------------------------------------------------


def generate_bash_completion() -> str:
    """Generate bash completion script.

    Returns:
        A complete bash completion script as a string.
    """
    commands = " ".join(CLI_COMMANDS)
    flags = " ".join(CLI_FLAGS)
    models = " ".join(MODELS)
    repl_cmds = " ".join(REPL_COMMANDS)

    return f'''# Prism CLI bash completion
# Add to ~/.bashrc: eval "$(prism completion bash)"

_prism_completions() {{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"

    case "${{prev}}" in
        --model)
            COMPREPLY=( $(compgen -W "{models}" -- "${{cur}}") )
            return 0
            ;;
        --project)
            # Complete with project names from workspace
            local projects=$(prism projects list --names-only 2>/dev/null)
            COMPREPLY=( $(compgen -W "${{projects}}" -- "${{cur}}") )
            return 0
            ;;
        /add|/image|/drop)
            # File path completion
            COMPREPLY=( $(compgen -f -- "${{cur}}") )
            return 0
            ;;
    esac

    if [[ "${{cur}}" == /* ]]; then
        COMPREPLY=( $(compgen -W "{repl_cmds}" -- "${{cur}}") )
        return 0
    fi

    if [[ "${{cur}}" == -* ]]; then
        COMPREPLY=( $(compgen -W "{flags}" -- "${{cur}}") )
        return 0
    fi

    COMPREPLY=( $(compgen -W "{commands} {flags}" -- "${{cur}}") )
    return 0
}}

complete -F _prism_completions prism
'''


def generate_zsh_completion() -> str:
    """Generate zsh completion script.

    Returns:
        A complete zsh completion script as a string.
    """
    commands_list = "\n    ".join(
        f"'{cmd}:{cmd} command'" for cmd in CLI_COMMANDS
    )
    flags_list = "\n    ".join(f"'{flag}[{flag}]'" for flag in CLI_FLAGS)
    models_list = " ".join(MODELS)

    return f'''#compdef prism
# Prism CLI zsh completion
# Add to ~/.zshrc: eval "$(prism completion zsh)"

_prism() {{
    local -a commands flags models

    commands=(
    {commands_list}
    )

    flags=(
    {flags_list}
    )

    models=({models_list})

    _arguments \\
        '1:command:->command' \\
        '*:options:->options'

    case $state in
        command)
            _describe 'command' commands
            ;;
        options)
            case $words[2] in
                --model)
                    compadd -a models
                    ;;
                *)
                    _files
                    ;;
            esac
            ;;
    esac
}}

_prism "$@"
'''


def generate_fish_completion() -> str:
    """Generate fish completion script.

    Returns:
        A complete fish shell completion script as a string.
    """
    lines: list[str] = [
        "# Prism CLI fish completion",
        "# Add to ~/.config/fish/completions/prism.fish",
        "",
    ]

    for cmd in CLI_COMMANDS:
        lines.append(
            f"complete -c prism -n '__fish_use_subcommand' -a '{cmd}' -d '{cmd}'"
        )

    for flag in CLI_FLAGS:
        flag_long = flag.lstrip("-")
        lines.append(
            f"complete -c prism -l '{flag_long}' -d '{flag}'"
        )

    lines.append("")
    lines.append("# Model completion for --model flag")
    for model in MODELS:
        lines.append(
            f"complete -c prism -n '__fish_seen_subcommand_from --model' -a '{model}'"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def install_completion(shell: str, path: Path | None = None) -> Path:
    """Install completion script for the specified shell.

    Args:
        shell: One of ``"bash"``, ``"zsh"``, or ``"fish"``.
        path: Override the default installation path.

    Returns:
        The path where the completion script was written.

    Raises:
        ValueError: If *shell* is not one of the supported shells.
    """
    if shell == "bash":
        script = generate_bash_completion()
        default_path = Path.home() / ".bash_completion.d" / "prism"
    elif shell == "zsh":
        script = generate_zsh_completion()
        default_path = Path.home() / ".zsh" / "completions" / "_prism"
    elif shell == "fish":
        script = generate_fish_completion()
        default_path = (
            Path.home() / ".config" / "fish" / "completions" / "prism.fish"
        )
    else:
        raise ValueError(
            f"Unsupported shell: {shell}. Use bash, zsh, or fish."
        )

    target = path or default_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8")

    logger.info("completion_installed", shell=shell, path=str(target))
    return target


# ---------------------------------------------------------------------------
# Shell detection
# ---------------------------------------------------------------------------


def detect_shell() -> str:
    """Detect the current shell from the ``SHELL`` environment variable.

    Returns:
        ``"zsh"``, ``"fish"``, or ``"bash"`` (the default fallback).
    """
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return "zsh"
    if "fish" in shell:
        return "fish"
    return "bash"

"""Shell tab-completion support for the Prism CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Known shell names and their completion script locations.
_SHELL_RC_FILES: dict[str, str] = {
    "bash": "~/.bashrc",
    "zsh": "~/.zshrc",
    "fish": "~/.config/fish/completions/prism.fish",
}


def install_completion(shell: str = "auto") -> None:
    """Install shell tab completion.

    Supports bash, zsh, and fish via Typer's built-in completion
    infrastructure.

    Args:
        shell: Shell name (``"bash"``, ``"zsh"``, ``"fish"``) or
            ``"auto"`` to detect from the ``SHELL`` environment variable.
    """
    if shell == "auto":
        shell = _detect_shell()

    if shell not in _SHELL_RC_FILES:
        logger.warning("unsupported_shell", shell=shell)
        return

    try:
        # Typer uses an environment variable for shell completion install
        os.environ["_PRISM_COMPLETE"] = f"{shell}_source"
        # Generate completion script
        result = subprocess.run(
            [sys.executable, "-m", "prism"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout:
            _install_script(shell, result.stdout)
            logger.info("completion_installed", shell=shell)
        else:
            logger.warning("completion_generation_failed", shell=shell)
    except Exception as exc:
        logger.error("completion_install_error", error=str(exc))
    finally:
        os.environ.pop("_PRISM_COMPLETE", None)


def _detect_shell() -> str:
    """Detect the current shell from the SHELL environment variable."""
    shell_path = os.environ.get("SHELL", "")
    shell_name = Path(shell_path).name if shell_path else ""
    if shell_name in ("bash", "zsh", "fish"):
        return shell_name
    return "bash"  # Default to bash


def _install_script(shell: str, script: str) -> None:
    """Write a completion script to the appropriate location.

    Args:
        shell: Shell name.
        script: The completion script content.
    """
    rc_path = Path(_SHELL_RC_FILES.get(shell, "~/.bashrc")).expanduser()

    if shell == "fish":
        # Fish uses individual completion files
        rc_path.parent.mkdir(parents=True, exist_ok=True)
        rc_path.write_text(script)
    else:
        # For bash/zsh, append to rc file if not already present
        marker = "# prism shell completion"
        if rc_path.exists():
            content = rc_path.read_text()
            if marker in content:
                return  # Already installed
        with rc_path.open("a") as f:
            f.write(f"\n{marker}\n{script}\n")


def get_completions(
    ctx: object,
    args: list[str],
    incomplete: str,
) -> list[str]:
    """Dynamic completions for model names, file paths, providers.

    Args:
        ctx: Click/Typer context (unused in basic implementation).
        args: Current argument list.
        incomplete: The partial text being completed.

    Returns:
        List of completion strings.
    """
    completions: list[str] = []

    # Complete provider names
    providers = ["anthropic", "openai", "google", "deepseek", "groq", "mistral", "ollama"]
    completions.extend(p for p in providers if p.startswith(incomplete))

    # Complete common commands
    commands = ["/help", "/model", "/budget", "/compact", "/status", "/clear"]
    completions.extend(c for c in commands if c.startswith(incomplete))

    return completions

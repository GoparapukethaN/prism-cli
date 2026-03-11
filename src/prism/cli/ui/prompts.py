"""User prompt and confirmation utilities for the Prism CLI.

IMPORTANT: This module NEVER logs, prints, or stores the actual value
of API keys.  Only masked representations are ever shown.
"""

from __future__ import annotations

from rich.prompt import Confirm, Prompt

from prism.cli.ui.themes import get_console

# ---------------------------------------------------------------------------
# Action confirmation
# ---------------------------------------------------------------------------


def confirm_action(action: str, details: str = "", console: object | None = None) -> bool:
    """Prompt the user for a yes/no confirmation.

    Args:
        action: Short description of the action requiring approval.
        details: Optional additional context shown before the prompt.
        console: Unused — kept for API consistency; the themed console
                 is always used for output.

    Returns:
        ``True`` if the user confirmed, ``False`` otherwise.
    """
    con = get_console()
    con.print()
    if details:
        con.print(f"[info]{details}[/info]")
    return Confirm.ask(f"[prompt]{action}[/prompt]", default=False, console=con)


# ---------------------------------------------------------------------------
# API key input
# ---------------------------------------------------------------------------


def prompt_api_key(provider: str, console: object | None = None) -> str:
    """Securely prompt the user for an API key.

    The key value is **never** echoed, logged, or printed.

    Args:
        provider: Name of the provider (e.g. ``"anthropic"``).
        console: Unused — kept for API consistency.

    Returns:
        The API key string entered by the user.
    """
    con = get_console()
    con.print(f"\n[prompt]Enter API key for [model]{provider}[/model]:[/prompt]")
    key: str = Prompt.ask("API key", password=True, console=con)
    masked = _mask_key(key)
    con.print(f"[success]Key received[/success] ({masked})")
    return key


def _mask_key(key: str) -> str:
    """Return a masked representation of an API key.

    Only the last 4 characters are shown.

    Args:
        key: The raw API key.

    Returns:
        A string like ``"...abcd"`` or ``"****"`` for short keys.
    """
    if len(key) > 4:
        return "..." + key[-4:]
    return "****"


# ---------------------------------------------------------------------------
# Model choice selection
# ---------------------------------------------------------------------------


def prompt_model_choice(models: list[str], console: object | None = None) -> str:
    """Present a numbered list of models and let the user pick one.

    Args:
        models: Available model identifiers.
        console: Unused — kept for API consistency.

    Returns:
        The selected model identifier string.

    Raises:
        ValueError: If *models* is empty.
    """
    if not models:
        raise ValueError("No models available for selection.")

    con = get_console()
    con.print("\n[prompt]Available models:[/prompt]")
    for idx, model in enumerate(models, start=1):
        con.print(f"  [model]{idx}[/model]. {model}")

    while True:
        choice_str: str = Prompt.ask(
            "[prompt]Select model number[/prompt]",
            console=con,
        )
        try:
            choice = int(choice_str)
            if 1 <= choice <= len(models):
                selected = models[choice - 1]
                con.print(f"[success]Selected:[/success] {selected}")
                return selected
        except ValueError:
            pass
        con.print(f"[warning]Please enter a number between 1 and {len(models)}.[/warning]")


# ---------------------------------------------------------------------------
# Budget limit prompt
# ---------------------------------------------------------------------------


def prompt_budget_limit(console: object | None = None) -> float | None:
    """Ask the user for a daily budget limit.

    Args:
        console: Unused — kept for API consistency.

    Returns:
        The budget as a float, or ``None`` if the user skips.
    """
    con = get_console()
    con.print()
    raw: str = Prompt.ask(
        "[prompt]Daily budget limit in USD (leave blank for unlimited)[/prompt]",
        default="",
        console=con,
    )
    raw = raw.strip()
    if not raw:
        con.print("[info]No budget limit set.[/info]")
        return None

    try:
        value = float(raw)
        if value < 0:
            con.print("[warning]Budget cannot be negative. No limit set.[/warning]")
            return None
        con.print(f"[success]Budget set:[/success] ${value:.2f}/day")
        return value
    except ValueError:
        con.print("[warning]Invalid number. No limit set.[/warning]")
        return None

"""Rich-based display utilities for the Prism CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from prism import __version__
from prism.cli.ui.themes import TIER_COLORS, TIER_ICONS, get_console

if TYPE_CHECKING:
    from rich.console import Console

    from prism.config.settings import Settings
    from prism.providers.base import ComplexityTier
    from prism.router.classifier import ClassificationResult

# Maximum characters shown for tool results before truncation.
_MAX_TOOL_RESULT_LENGTH: int = 1000


def _get_console(console: Console | None = None) -> Console:
    """Return the provided console or the default themed console."""
    if console is not None:
        return console
    return get_console()


# ---------------------------------------------------------------------------
# Welcome banner
# ---------------------------------------------------------------------------


def display_welcome(console: Console | None = None) -> None:
    """Print the Prism welcome banner with version information.

    Args:
        console: Optional Rich console; creates a themed one if omitted.
    """
    con = _get_console(console)
    con.print()
    con.print(
        Panel(
            Text.from_markup(
                f"[bold cyan]Prism[/bold cyan] v{__version__} -- "
                "Multi-API Intelligent Router\n"
                "[dim]Type your request, or use /help for commands.[/dim]"
            ),
            border_style="cyan",
        )
    )
    con.print()


# ---------------------------------------------------------------------------
# Classification display
# ---------------------------------------------------------------------------


def display_classification(
    result: ClassificationResult,
    console: Console | None = None,
) -> None:
    """Display the task classification result with a colored badge.

    Args:
        result: The classification result from the task classifier.
        console: Optional Rich console.
    """
    con = _get_console(console)
    tier: ComplexityTier = result.tier
    color = TIER_COLORS.get(tier, "white")
    icon = TIER_ICONS.get(tier, "")

    badge = Text()
    badge.append(f" {icon} {tier.value.upper()} ", style=f"bold {color}")

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style="bold")
    table.add_column("Value")

    table.add_row("Tier:", badge)
    table.add_row("Score:", Text(f"{result.score:.2f}", style="cost"))
    table.add_row("Reasoning:", Text(result.reasoning, style="info"))

    con.print(Panel(table, title="[bold]Classification[/bold]", border_style=color))


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def display_model_selection(
    model: str,
    provider: str,
    estimated_cost: float,
    console: Console | None = None,
) -> None:
    """Display the selected model and estimated cost.

    Args:
        model: Model identifier.
        provider: Provider name.
        estimated_cost: Estimated cost in USD for this request.
        console: Optional Rich console.
    """
    con = _get_console(console)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style="bold")
    table.add_column("Value")

    table.add_row("Model:", Text(model, style="model"))
    table.add_row("Provider:", Text(provider, style="model"))
    table.add_row("Est. cost:", Text(f"${estimated_cost:.4f}", style="cost"))

    con.print(Panel(table, title="[bold]Model Selection[/bold]", border_style="blue"))


# ---------------------------------------------------------------------------
# Tool call / result
# ---------------------------------------------------------------------------


def display_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    console: Console | None = None,
) -> None:
    """Display a formatted tool invocation.

    Args:
        tool_name: Name of the tool being called.
        arguments: Keyword arguments passed to the tool.
        console: Optional Rich console.
    """
    con = _get_console(console)

    args_text = Text()
    for key, value in arguments.items():
        args_text.append(f"  {key}: ", style="bold")
        args_text.append(f"{value}\n")

    con.print(
        Panel(
            args_text or Text("(no arguments)"),
            title=f"[tool_name]{tool_name}[/tool_name]",
            border_style="magenta",
        )
    )


def display_tool_result(
    result: Any,
    console: Console | None = None,
) -> None:
    """Display a formatted tool output, truncated if long.

    Args:
        result: The tool's return value (will be converted to string).
        console: Optional Rich console.
    """
    con = _get_console(console)
    text = str(result)

    if len(text) > _MAX_TOOL_RESULT_LENGTH:
        truncated = text[:_MAX_TOOL_RESULT_LENGTH]
        display_text = Text()
        display_text.append(truncated)
        display_text.append(
            f"\n... (truncated, {len(text)} chars total)", style="info"
        )
    else:
        display_text = Text(text)

    con.print(
        Panel(
            display_text,
            title="[bold]Tool Result[/bold]",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------


def display_error(
    error: str,
    hint: str | None = None,
    console: Console | None = None,
) -> None:
    """Display a styled error message with an optional hint.

    Args:
        error: The error description.
        hint: An optional suggestion for how to resolve the error.
        console: Optional Rich console.
    """
    con = _get_console(console)

    error_text = Text()
    error_text.append("Error: ", style="error")
    error_text.append(error)

    if hint:
        error_text.append("\n")
        error_text.append("Hint: ", style="warning")
        error_text.append(hint)

    con.print(Panel(error_text, border_style="red"))


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------


def display_status(
    settings: Settings,
    console: Console | None = None,
) -> None:
    """Display current configuration status.

    Args:
        settings: The application settings object.
        console: Optional Rich console.
    """
    con = _get_console(console)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Project root:", Text(str(settings.project_root)))
    table.add_row("Prism home:", Text(str(settings.prism_home)))
    table.add_row("Database:", Text(str(settings.db_path)))

    # Routing thresholds
    simple_thresh = settings.get("routing.simple_threshold", "?")
    medium_thresh = settings.get("routing.medium_threshold", "?")
    table.add_row("Simple threshold:", Text(str(simple_thresh)))
    table.add_row("Medium threshold:", Text(str(medium_thresh)))

    # Budget
    daily = settings.get("budget.daily_limit")
    monthly = settings.get("budget.monthly_limit")
    table.add_row(
        "Budget (daily):",
        Text(f"${daily:.2f}" if daily is not None else "unlimited"),
    )
    table.add_row(
        "Budget (monthly):",
        Text(f"${monthly:.2f}" if monthly is not None else "unlimited"),
    )

    # Pinned model
    pinned = settings.get("pinned_model")
    table.add_row(
        "Pinned model:",
        Text(str(pinned) if pinned else "auto (routing)"),
    )

    con.print(
        Panel(table, title="[bold]Prism Status[/bold]", border_style="cyan")
    )


# ---------------------------------------------------------------------------
# Streaming token output
# ---------------------------------------------------------------------------


def display_streaming_token(
    token: str,
    console: Console | None = None,
) -> None:
    """Print a single token without a trailing newline for streaming output.

    Args:
        token: The token string to display.
        console: Optional Rich console.
    """
    con = _get_console(console)
    con.print(token, end="", highlight=False)


# ---------------------------------------------------------------------------
# Diff preview
# ---------------------------------------------------------------------------


def display_diff(
    diff_text: str,
    file_path: str = "",
    console: Console | None = None,
) -> None:
    """Display a colourised diff inside a Rich panel.

    Uses Rich :class:`Syntax` with ``language="diff"`` to provide
    red/green colouring for removed/added lines.

    Args:
        diff_text: The unified-diff (or simple ``+``/``-`` prefixed) text.
        file_path: Optional path shown as the panel title.
        console: Optional Rich console; creates a themed one if omitted.
    """
    con = _get_console(console)

    if not diff_text or not diff_text.strip():
        con.print("  [dim](no changes)[/dim]")
        return

    title = f"[bold]Diff: {file_path}[/bold]" if file_path else "[bold]Diff Preview[/bold]"

    syntax = Syntax(
        diff_text,
        lexer="diff",
        theme="ansi_dark",
        word_wrap=True,
    )

    con.print(Panel(syntax, title=title, border_style="yellow"))

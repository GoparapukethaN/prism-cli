"""Color theme definitions for the Prism CLI."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

from prism.providers.base import ComplexityTier

# Named style definitions for the Prism CLI theme.
PRISM_THEME: dict[str, str] = {
    "tier.simple": "green",
    "tier.medium": "yellow",
    "tier.complex": "red",
    "cost": "cyan",
    "model": "blue",
    "error": "bold red",
    "warning": "yellow",
    "success": "green",
    "info": "dim",
    "prompt": "bold",
    "tool_name": "magenta",
    "file_path": "underline blue",
}

# Mapping from ComplexityTier enum members to Rich color strings.
TIER_COLORS: dict[ComplexityTier, str] = {
    ComplexityTier.SIMPLE: "green",
    ComplexityTier.MEDIUM: "yellow",
    ComplexityTier.COMPLEX: "red",
}

# Text-based tier icons (no emojis).
TIER_ICONS: dict[ComplexityTier, str] = {
    ComplexityTier.SIMPLE: "[S]",
    ComplexityTier.MEDIUM: "[M]",
    ComplexityTier.COMPLEX: "[C]",
}


def get_console() -> Console:
    """Create and return a Rich Console with the Prism theme applied.

    Returns:
        A Console instance configured with ``PRISM_THEME``.
    """
    theme = Theme(PRISM_THEME)
    return Console(theme=theme)

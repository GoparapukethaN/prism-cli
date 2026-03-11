"""Prism CLI UI rendering — display utilities, prompts, and themes."""

from __future__ import annotations

from prism.cli.ui.display import (
    display_classification,
    display_error,
    display_model_selection,
    display_status,
    display_streaming_token,
    display_tool_call,
    display_tool_result,
    display_welcome,
)
from prism.cli.ui.prompts import (
    confirm_action,
    prompt_api_key,
    prompt_budget_limit,
    prompt_model_choice,
)
from prism.cli.ui.themes import (
    PRISM_THEME,
    TIER_COLORS,
    TIER_ICONS,
    get_console,
)

__all__ = [
    # themes
    "PRISM_THEME",
    "TIER_COLORS",
    "TIER_ICONS",
    # prompts
    "confirm_action",
    # display
    "display_classification",
    "display_error",
    "display_model_selection",
    "display_status",
    "display_streaming_token",
    "display_tool_call",
    "display_tool_result",
    "display_welcome",
    "get_console",
    "prompt_api_key",
    "prompt_budget_limit",
    "prompt_model_choice",
]

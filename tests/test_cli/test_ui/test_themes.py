"""Tests for prism.cli.ui.themes — color themes and console factory."""

from __future__ import annotations

from rich.console import Console

from prism.cli.ui.themes import (
    PRISM_THEME,
    TIER_COLORS,
    TIER_ICONS,
    get_console,
)
from prism.providers.base import ComplexityTier

# ---------------------------------------------------------------------------
# PRISM_THEME
# ---------------------------------------------------------------------------


class TestPrismTheme:
    """Tests for the PRISM_THEME style dictionary."""

    def test_contains_tier_simple(self) -> None:
        assert "tier.simple" in PRISM_THEME
        assert PRISM_THEME["tier.simple"] == "green"

    def test_contains_tier_medium(self) -> None:
        assert "tier.medium" in PRISM_THEME
        assert PRISM_THEME["tier.medium"] == "yellow"

    def test_contains_tier_complex(self) -> None:
        assert "tier.complex" in PRISM_THEME
        assert PRISM_THEME["tier.complex"] == "red"

    def test_contains_cost(self) -> None:
        assert "cost" in PRISM_THEME
        assert PRISM_THEME["cost"] == "cyan"

    def test_contains_model(self) -> None:
        assert "model" in PRISM_THEME
        assert PRISM_THEME["model"] == "blue"

    def test_contains_error(self) -> None:
        assert "error" in PRISM_THEME
        assert PRISM_THEME["error"] == "bold red"

    def test_contains_warning(self) -> None:
        assert "warning" in PRISM_THEME
        assert PRISM_THEME["warning"] == "yellow"

    def test_contains_success(self) -> None:
        assert "success" in PRISM_THEME
        assert PRISM_THEME["success"] == "green"

    def test_contains_info(self) -> None:
        assert "info" in PRISM_THEME
        assert PRISM_THEME["info"] == "dim"

    def test_contains_prompt(self) -> None:
        assert "prompt" in PRISM_THEME
        assert PRISM_THEME["prompt"] == "bold"

    def test_contains_tool_name(self) -> None:
        assert "tool_name" in PRISM_THEME
        assert PRISM_THEME["tool_name"] == "magenta"

    def test_contains_file_path(self) -> None:
        assert "file_path" in PRISM_THEME
        assert PRISM_THEME["file_path"] == "underline blue"

    def test_all_required_styles_present(self) -> None:
        required = {
            "tier.simple",
            "tier.medium",
            "tier.complex",
            "cost",
            "model",
            "error",
            "warning",
            "success",
            "info",
            "prompt",
            "tool_name",
            "file_path",
        }
        assert required.issubset(set(PRISM_THEME.keys()))


# ---------------------------------------------------------------------------
# TIER_COLORS
# ---------------------------------------------------------------------------


class TestTierColors:
    """Tests for the tier-to-color mapping."""

    def test_simple_is_green(self) -> None:
        assert TIER_COLORS[ComplexityTier.SIMPLE] == "green"

    def test_medium_is_yellow(self) -> None:
        assert TIER_COLORS[ComplexityTier.MEDIUM] == "yellow"

    def test_complex_is_red(self) -> None:
        assert TIER_COLORS[ComplexityTier.COMPLEX] == "red"

    def test_covers_all_tiers(self) -> None:
        for tier in ComplexityTier:
            assert tier in TIER_COLORS


# ---------------------------------------------------------------------------
# TIER_ICONS
# ---------------------------------------------------------------------------


class TestTierIcons:
    """Tests for the tier-to-icon mapping."""

    def test_simple_icon(self) -> None:
        assert TIER_ICONS[ComplexityTier.SIMPLE] == "[S]"

    def test_medium_icon(self) -> None:
        assert TIER_ICONS[ComplexityTier.MEDIUM] == "[M]"

    def test_complex_icon(self) -> None:
        assert TIER_ICONS[ComplexityTier.COMPLEX] == "[C]"

    def test_covers_all_tiers(self) -> None:
        for tier in ComplexityTier:
            assert tier in TIER_ICONS

    def test_icons_are_text_not_emoji(self) -> None:
        for icon in TIER_ICONS.values():
            # All characters should be ASCII printable
            assert icon.isascii()


# ---------------------------------------------------------------------------
# get_console
# ---------------------------------------------------------------------------


class TestGetConsole:
    """Tests for the console factory function."""

    def test_returns_console_instance(self) -> None:
        con = get_console()
        assert isinstance(con, Console)

    def test_console_has_custom_theme(self) -> None:
        con = get_console()
        # Rich stores custom styles in the console's internal _theme object.
        # We verify the theme was applied by checking that a custom style resolves.
        style = con.get_style("tier.simple")
        assert style is not None

    def test_multiple_calls_return_fresh_instances(self) -> None:
        c1 = get_console()
        c2 = get_console()
        assert c1 is not c2

"""Tests for prism.cli.ui.prompts — user input utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prism.cli.ui.prompts import (
    _mask_key,
    confirm_action,
    prompt_api_key,
    prompt_budget_limit,
    prompt_model_choice,
)

# ---------------------------------------------------------------------------
# confirm_action
# ---------------------------------------------------------------------------


class TestConfirmAction:
    """Tests for the yes/no confirmation prompt."""

    @patch("prism.cli.ui.prompts.Confirm.ask", return_value=True)
    def test_returns_true_on_yes(self, mock_ask: MagicMock) -> None:
        result = confirm_action("Delete file?")
        assert result is True
        mock_ask.assert_called_once()

    @patch("prism.cli.ui.prompts.Confirm.ask", return_value=False)
    def test_returns_false_on_no(self, mock_ask: MagicMock) -> None:
        result = confirm_action("Delete file?")
        assert result is False

    @patch("prism.cli.ui.prompts.Confirm.ask", return_value=True)
    def test_passes_action_text(self, mock_ask: MagicMock) -> None:
        confirm_action("Run dangerous command?", details="This will modify files")
        call_args = mock_ask.call_args
        # The first positional arg should contain the action text.
        assert "Run dangerous command?" in str(call_args)

    @patch("prism.cli.ui.prompts.Confirm.ask", return_value=False)
    def test_with_empty_details(self, mock_ask: MagicMock) -> None:
        result = confirm_action("Proceed?", details="")
        assert result is False


# ---------------------------------------------------------------------------
# prompt_api_key
# ---------------------------------------------------------------------------


class TestPromptApiKey:
    """Tests for secure API key input."""

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="sk-ant-test-key-abcdef1234")
    def test_returns_key_value(self, mock_ask: MagicMock) -> None:
        key = prompt_api_key("anthropic")
        assert key == "sk-ant-test-key-abcdef1234"

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="sk-ant-test-key-abcdef1234")
    def test_ask_uses_password_mode(self, mock_ask: MagicMock) -> None:
        prompt_api_key("openai")
        # Verify password=True was passed
        call_kwargs = mock_ask.call_args
        assert call_kwargs.kwargs.get("password") is True or (
            len(call_kwargs.args) >= 1 and call_kwargs[1].get("password") is True
        )

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="short")
    def test_short_key_masked_as_stars(self, mock_ask: MagicMock) -> None:
        # Should not raise; the mask for short keys is "****"
        key = prompt_api_key("groq")
        assert key == "short"


# ---------------------------------------------------------------------------
# _mask_key
# ---------------------------------------------------------------------------


class TestMaskKey:
    """Tests for the key masking helper."""

    def test_long_key_shows_last_four(self) -> None:
        assert _mask_key("sk-ant-test-key-abcdef1234") == "...1234"

    def test_exact_four_chars(self) -> None:
        assert _mask_key("abcd") == "****"

    def test_short_key(self) -> None:
        assert _mask_key("ab") == "****"

    def test_empty_key(self) -> None:
        assert _mask_key("") == "****"

    def test_five_char_key(self) -> None:
        assert _mask_key("12345") == "...2345"


# ---------------------------------------------------------------------------
# prompt_model_choice
# ---------------------------------------------------------------------------


class TestPromptModelChoice:
    """Tests for model selection from a list."""

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="1")
    def test_selects_first_model(self, mock_ask: MagicMock) -> None:
        models = ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.5-pro"]
        result = prompt_model_choice(models)
        assert result == "gpt-4o"

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="3")
    def test_selects_last_model(self, mock_ask: MagicMock) -> None:
        models = ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.5-pro"]
        result = prompt_model_choice(models)
        assert result == "gemini-2.5-pro"

    def test_empty_list_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="No models available"):
            prompt_model_choice([])

    @patch("prism.cli.ui.prompts.Prompt.ask", side_effect=["abc", "0", "2"])
    def test_retries_on_invalid_input(self, mock_ask: MagicMock) -> None:
        models = ["gpt-4o", "gpt-4o-mini"]
        result = prompt_model_choice(models)
        # First two inputs are invalid; third ("2") is valid.
        assert result == "gpt-4o-mini"
        assert mock_ask.call_count == 3

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="1")
    def test_single_model_list(self, mock_ask: MagicMock) -> None:
        result = prompt_model_choice(["only-model"])
        assert result == "only-model"


# ---------------------------------------------------------------------------
# prompt_budget_limit
# ---------------------------------------------------------------------------


class TestPromptBudgetLimit:
    """Tests for budget limit prompt."""

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="5.50")
    def test_valid_budget(self, mock_ask: MagicMock) -> None:
        result = prompt_budget_limit()
        assert result == 5.50

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="")
    def test_empty_returns_none(self, mock_ask: MagicMock) -> None:
        result = prompt_budget_limit()
        assert result is None

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="not-a-number")
    def test_invalid_returns_none(self, mock_ask: MagicMock) -> None:
        result = prompt_budget_limit()
        assert result is None

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="-10")
    def test_negative_returns_none(self, mock_ask: MagicMock) -> None:
        result = prompt_budget_limit()
        assert result is None

    @patch("prism.cli.ui.prompts.Prompt.ask", return_value="0")
    def test_zero_budget(self, mock_ask: MagicMock) -> None:
        result = prompt_budget_limit()
        assert result == 0.0

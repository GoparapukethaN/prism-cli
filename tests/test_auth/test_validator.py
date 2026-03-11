"""Tests for prism.auth.validator."""

from __future__ import annotations

from prism.auth.validator import KeyValidator


class TestValidateKey:
    """Format-based key validation (no network calls)."""

    # ---- Anthropic ---------------------------------------------------

    def test_anthropic_valid(self):
        assert KeyValidator.validate_key("anthropic", "sk-ant-api03-abc123") is True

    def test_anthropic_wrong_prefix(self):
        assert KeyValidator.validate_key("anthropic", "sk-proj-abc123") is False

    def test_anthropic_empty(self):
        assert KeyValidator.validate_key("anthropic", "") is False

    # ---- OpenAI ------------------------------------------------------

    def test_openai_valid(self):
        assert KeyValidator.validate_key("openai", "sk-proj-abc123") is True

    def test_openai_valid_plain(self):
        assert KeyValidator.validate_key("openai", "sk-abc123") is True

    def test_openai_wrong_prefix(self):
        assert KeyValidator.validate_key("openai", "oai-abc123") is False

    def test_openai_empty(self):
        assert KeyValidator.validate_key("openai", "") is False

    # ---- Google ------------------------------------------------------

    def test_google_valid(self):
        assert KeyValidator.validate_key("google", "AIzaSyDabc123") is True

    def test_google_any_nonempty(self):
        assert KeyValidator.validate_key("google", "anything-goes") is True

    def test_google_empty(self):
        assert KeyValidator.validate_key("google", "") is False

    # ---- DeepSeek ----------------------------------------------------

    def test_deepseek_valid(self):
        assert KeyValidator.validate_key("deepseek", "sk-abc123") is True

    def test_deepseek_wrong_prefix(self):
        assert KeyValidator.validate_key("deepseek", "ds-abc123") is False

    def test_deepseek_empty(self):
        assert KeyValidator.validate_key("deepseek", "") is False

    # ---- Groq --------------------------------------------------------

    def test_groq_valid(self):
        assert KeyValidator.validate_key("groq", "gsk_abc123") is True

    def test_groq_wrong_prefix(self):
        assert KeyValidator.validate_key("groq", "sk-abc123") is False

    def test_groq_empty(self):
        assert KeyValidator.validate_key("groq", "") is False

    # ---- Mistral -----------------------------------------------------

    def test_mistral_valid(self):
        assert KeyValidator.validate_key("mistral", "any-non-empty-key") is True

    def test_mistral_empty(self):
        assert KeyValidator.validate_key("mistral", "") is False

    # ---- Unknown provider --------------------------------------------

    def test_unknown_provider_accepts_nonempty(self):
        assert KeyValidator.validate_key("some_new_provider", "key123") is True

    def test_unknown_provider_rejects_empty(self):
        assert KeyValidator.validate_key("some_new_provider", "") is False


class TestKnownProviders:
    def test_returns_sorted_list(self):
        providers = KeyValidator.known_providers()
        assert providers == sorted(providers)

    def test_includes_all_six(self):
        providers = KeyValidator.known_providers()
        expected = {"anthropic", "deepseek", "google", "groq", "mistral", "openai"}
        assert set(providers) == expected

    def test_returns_list_type(self):
        assert isinstance(KeyValidator.known_providers(), list)

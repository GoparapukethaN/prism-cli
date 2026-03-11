"""Fixtures for CLI UI tests."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest
from rich.console import Console
from rich.theme import Theme

from prism.cli.ui.themes import PRISM_THEME
from prism.config.schema import PrismConfig
from prism.config.settings import Settings
from prism.providers.base import ComplexityTier
from prism.router.classifier import ClassificationResult

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def capture_console() -> Console:
    """Return a Rich Console that writes to an in-memory buffer.

    Use ``capture_console.file.getvalue()`` to read the rendered text.
    """
    buf = io.StringIO()
    theme = Theme(PRISM_THEME)
    return Console(file=buf, theme=theme, force_terminal=True, width=120)


@pytest.fixture()
def plain_console() -> Console:
    """Return a non-themed console writing to a StringIO buffer.

    Useful for tests that only need to verify output presence without
    worrying about ANSI escape codes.
    """
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, no_color=True, width=300)


@pytest.fixture()
def simple_result() -> ClassificationResult:
    """A ClassificationResult with tier=SIMPLE."""
    return ClassificationResult(
        tier=ComplexityTier.SIMPLE,
        score=0.15,
        features={
            "prompt_token_count": 5.0,
            "files_referenced": 0.0,
            "estimated_output_tokens": 500.0,
            "complexity_keywords": 0.2,
            "requires_reasoning": 0.0,
            "scope": 0.1,
        },
        reasoning="Classified as SIMPLE (score: 0.15). simple keywords detected. narrow scope (single file).",
    )


@pytest.fixture()
def medium_result() -> ClassificationResult:
    """A ClassificationResult with tier=MEDIUM."""
    return ClassificationResult(
        tier=ComplexityTier.MEDIUM,
        score=0.50,
        features={
            "prompt_token_count": 30.0,
            "files_referenced": 2.0,
            "estimated_output_tokens": 2000.0,
            "complexity_keywords": 0.55,
            "requires_reasoning": 0.0,
            "scope": 0.3,
        },
        reasoning="Classified as MEDIUM (score: 0.50). moderate scope (multi-file). 2 file(s) in context.",
    )


@pytest.fixture()
def complex_result() -> ClassificationResult:
    """A ClassificationResult with tier=COMPLEX."""
    return ClassificationResult(
        tier=ComplexityTier.COMPLEX,
        score=0.85,
        features={
            "prompt_token_count": 150.0,
            "files_referenced": 8.0,
            "estimated_output_tokens": 4000.0,
            "complexity_keywords": 0.9,
            "requires_reasoning": 1.0,
            "scope": 0.9,
        },
        reasoning=(
            "Classified as COMPLEX (score: 0.85). complex keywords detected. "
            "multi-step reasoning needed. broad scope (architecture/system-level). 8 file(s) in context."
        ),
    )


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    """Minimal Settings instance suitable for display tests."""
    config = PrismConfig(
        prism_home=tmp_path / ".prism",
    )
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings

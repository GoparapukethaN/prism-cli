"""Tests for Phase 5 enhanced multi-model debate — Item 26.

Covers:
- DebateConfig defaults and customization
- DebateRound and DebateResult dataclass creation
- debate() function with mock LLM caller (3 rounds, quick mode)
- _parse_synthesis() parsing logic
- generate_report_text() output format
- save_debate() and list_debates() persistence
- Edge cases: empty question, single model, error handling
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prism.intelligence.debate import (
    DebateConfig,
    DebateResult,
    DebateRound,
    _parse_synthesis,
    _stub_llm_caller,
    debate,
    generate_report_text,
    list_debates,
    save_debate,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def tmp_debates_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for debate storage."""
    d = tmp_path / "debates"
    d.mkdir()
    return d


@pytest.fixture
def mock_llm_caller() -> object:
    """Return a mock LLM caller that returns structured synthesis."""

    def caller(model: str, messages: list[dict[str, str]]) -> str:
        # Check if this is a synthesis call
        for msg in messages:
            content = msg.get("content", "")
            if "Synthesize" in content or "CONSENSUS" in content:
                return (
                    "CONSENSUS: All models agree on X\n"
                    "DISAGREEMENTS: Model A prefers Y, Model B prefers Z\n"
                    "TRADEOFFS: Speed vs accuracy\n"
                    "RECOMMENDATION: Use approach X\n"
                    "CONFIDENCE: 0.85\n"
                    "BLIND_SPOTS:\n"
                    "claude-sonnet-4-5: missed edge case\n"
                    "gpt-4o: missed performance concern\n"
                )
        return f"Position from {model}: I believe the answer is 42."

    return caller


@pytest.fixture
def sample_result() -> DebateResult:
    """Create a sample DebateResult for testing."""
    round1 = DebateRound(
        round_number=1,
        round_type="position",
        positions={
            "model-a": "Position A response",
            "model-b": "Position B response",
        },
    )
    round3 = DebateRound(
        round_number=3,
        round_type="synthesis",
        positions={"model-a": "Synthesis text here"},
    )
    return DebateResult(
        question="Should we use microservices?",
        rounds=[round1, round3],
        synthesis="Full synthesis text",
        consensus="All agree on scalability",
        disagreements="Disagree on complexity",
        tradeoffs="Speed vs maintainability",
        recommendation="Use microservices for large teams",
        confidence=0.85,
        blind_spots={"model-a": "missed cost", "model-b": "missed ops"},
        total_cost=0.05,
        created_at="2025-06-15T10:00:00+00:00",
    )


# ======================================================================
# TestDebateConfig
# ======================================================================


class TestDebateConfig:
    """Tests for DebateConfig dataclass defaults and customization."""

    def test_default_models(self) -> None:
        """Default round1_models includes 3 models."""
        cfg = DebateConfig()
        assert len(cfg.round1_models) == 3
        assert "claude-sonnet-4-5" in cfg.round1_models

    def test_default_synthesis_model(self) -> None:
        """Default synthesis_model is claude-sonnet-4-5."""
        cfg = DebateConfig()
        assert cfg.synthesis_model == "claude-sonnet-4-5"

    def test_default_quick_mode_false(self) -> None:
        """Quick mode defaults to False."""
        cfg = DebateConfig()
        assert cfg.quick_mode is False

    def test_default_save_dir(self) -> None:
        """Default save_dir is ~/.prism/debates/."""
        cfg = DebateConfig()
        assert cfg.save_dir == Path.home() / ".prism" / "debates"

    def test_custom_models(self) -> None:
        """Custom models can be provided."""
        cfg = DebateConfig(round1_models=["a", "b"])
        assert cfg.round1_models == ["a", "b"]

    def test_custom_synthesis_model(self) -> None:
        """Custom synthesis model can be set."""
        cfg = DebateConfig(synthesis_model="gpt-4o")
        assert cfg.synthesis_model == "gpt-4o"

    def test_quick_mode_true(self) -> None:
        """Quick mode can be enabled."""
        cfg = DebateConfig(quick_mode=True)
        assert cfg.quick_mode is True

    def test_custom_save_dir(self, tmp_debates_dir: Path) -> None:
        """Custom save directory can be specified."""
        cfg = DebateConfig(save_dir=tmp_debates_dir)
        assert cfg.save_dir == tmp_debates_dir


# ======================================================================
# TestDebateRound
# ======================================================================


class TestDebateRound:
    """Tests for DebateRound dataclass."""

    def test_default_positions_empty(self) -> None:
        """Positions default to empty dict."""
        rnd = DebateRound(round_number=1)
        assert rnd.positions == {}

    def test_default_round_type(self) -> None:
        """Default round_type is 'position'."""
        rnd = DebateRound(round_number=1)
        assert rnd.round_type == "position"

    def test_round_with_positions(self) -> None:
        """Round can be created with positions."""
        rnd = DebateRound(
            round_number=2,
            round_type="critique",
            positions={"model-a": "critique text"},
        )
        assert rnd.round_number == 2
        assert rnd.round_type == "critique"
        assert "model-a" in rnd.positions

    def test_synthesis_round(self) -> None:
        """Synthesis round has round_type='synthesis'."""
        rnd = DebateRound(round_number=3, round_type="synthesis")
        assert rnd.round_type == "synthesis"


# ======================================================================
# TestDebateResult
# ======================================================================


class TestDebateResult:
    """Tests for DebateResult dataclass."""

    def test_default_values(self) -> None:
        """Default values are empty/zero."""
        result = DebateResult(question="test?")
        assert result.question == "test?"
        assert result.rounds == []
        assert result.synthesis == ""
        assert result.consensus == ""
        assert result.disagreements == ""
        assert result.tradeoffs == ""
        assert result.recommendation == ""
        assert result.confidence == 0.0
        assert result.blind_spots == {}
        assert result.total_cost == 0.0
        assert result.created_at == ""

    def test_full_result(self, sample_result: DebateResult) -> None:
        """Full result has all fields populated."""
        assert sample_result.question == "Should we use microservices?"
        assert len(sample_result.rounds) == 2
        assert sample_result.confidence == 0.85
        assert len(sample_result.blind_spots) == 2

    def test_result_rounds_access(self, sample_result: DebateResult) -> None:
        """Rounds are accessible by index."""
        assert sample_result.rounds[0].round_type == "position"
        assert sample_result.rounds[1].round_type == "synthesis"


# ======================================================================
# TestDebateFunction
# ======================================================================


class TestDebateFunction:
    """Tests for the debate() function with mock LLM caller."""

    def test_full_debate_three_rounds(
        self, mock_llm_caller: object, tmp_debates_dir: Path,
    ) -> None:
        """Full debate produces 3 rounds (position, critique, synthesis)."""
        cfg = DebateConfig(
            round1_models=["model-a", "model-b"],
            synthesis_model="model-a",
            quick_mode=False,
            save_dir=tmp_debates_dir,
        )
        result = debate(
            question="Is Python better than Rust?",
            config=cfg,
            llm_caller=mock_llm_caller,
        )
        assert len(result.rounds) == 3
        assert result.rounds[0].round_type == "position"
        assert result.rounds[1].round_type == "critique"
        assert result.rounds[2].round_type == "synthesis"

    def test_quick_mode_skips_critique(
        self, mock_llm_caller: object, tmp_debates_dir: Path,
    ) -> None:
        """Quick mode produces only 2 rounds (position + synthesis)."""
        cfg = DebateConfig(
            round1_models=["model-a", "model-b"],
            synthesis_model="model-a",
            quick_mode=True,
            save_dir=tmp_debates_dir,
        )
        result = debate(
            question="Is Python better than Rust?",
            config=cfg,
            llm_caller=mock_llm_caller,
        )
        assert len(result.rounds) == 2
        assert result.rounds[0].round_type == "position"
        assert result.rounds[1].round_type == "synthesis"

    def test_round1_positions_populated(
        self, mock_llm_caller: object, tmp_debates_dir: Path,
    ) -> None:
        """Round 1 has a position for each model."""
        cfg = DebateConfig(
            round1_models=["model-a", "model-b", "model-c"],
            save_dir=tmp_debates_dir,
        )
        result = debate(
            question="What framework?",
            config=cfg,
            llm_caller=mock_llm_caller,
        )
        r1 = result.rounds[0]
        assert len(r1.positions) == 3
        assert "model-a" in r1.positions
        assert "model-b" in r1.positions
        assert "model-c" in r1.positions

    def test_synthesis_parsed(
        self, mock_llm_caller: object, tmp_debates_dir: Path,
    ) -> None:
        """Synthesis is parsed into structured fields."""
        cfg = DebateConfig(
            round1_models=["model-a", "model-b"],
            save_dir=tmp_debates_dir,
        )
        result = debate(
            question="Test parsing",
            config=cfg,
            llm_caller=mock_llm_caller,
        )
        # The mock returns structured synthesis for synthesis calls
        assert result.synthesis != ""
        assert result.created_at != ""

    def test_result_has_created_at(
        self, mock_llm_caller: object, tmp_debates_dir: Path,
    ) -> None:
        """Result has a valid created_at timestamp."""
        cfg = DebateConfig(
            round1_models=["model-a", "model-b"],
            save_dir=tmp_debates_dir,
        )
        result = debate(
            question="Timestamp test",
            config=cfg,
            llm_caller=mock_llm_caller,
        )
        assert result.created_at != ""
        assert "T" in result.created_at  # ISO format

    def test_empty_question_raises(self) -> None:
        """Empty question raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            debate(question="")

    def test_whitespace_only_question_raises(self) -> None:
        """Whitespace-only question raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            debate(question="   ")

    def test_default_stub_caller(self, tmp_debates_dir: Path) -> None:
        """Default stub caller returns placeholder text."""
        cfg = DebateConfig(
            round1_models=["model-a", "model-b"],
            save_dir=tmp_debates_dir,
        )
        result = debate(question="Test stub", config=cfg)
        assert "Placeholder" in result.rounds[0].positions["model-a"]

    def test_single_model_skips_critique(self, tmp_debates_dir: Path) -> None:
        """Single model results in position + synthesis only (no critique)."""
        cfg = DebateConfig(
            round1_models=["model-only"],
            quick_mode=False,
            save_dir=tmp_debates_dir,
        )
        result = debate(question="Solo model?", config=cfg)
        # critique is skipped because fewer than 2 positions
        round_types = [r.round_type for r in result.rounds]
        assert "critique" not in round_types

    def test_llm_caller_error_handled(self, tmp_debates_dir: Path) -> None:
        """LLM caller errors are caught and included as error text."""
        call_count = 0

        def failing_caller(model: str, messages: list[dict[str, str]]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API down")
            return f"Response from {model}"

        cfg = DebateConfig(
            round1_models=["model-a", "model-b"],
            save_dir=tmp_debates_dir,
        )
        result = debate(question="Error test", config=cfg, llm_caller=failing_caller)
        # The first call fails, but the debate still completes
        assert len(result.rounds) >= 2
        # First model's response should contain error
        assert "Error" in result.rounds[0].positions["model-a"]


# ======================================================================
# TestParseSynthesis
# ======================================================================


class TestParseSynthesis:
    """Tests for _parse_synthesis() parsing logic."""

    def test_parse_all_sections(self) -> None:
        """All sections are correctly parsed."""
        raw = (
            "CONSENSUS: Everyone agrees\n"
            "DISAGREEMENTS: Some disagree on X\n"
            "TRADEOFFS: Speed vs quality\n"
            "RECOMMENDATION: Go with A\n"
            "CONFIDENCE: 0.9\n"
        )
        parsed = _parse_synthesis(raw, [])
        assert parsed["consensus"] == "Everyone agrees"
        assert parsed["disagreements"] == "Some disagree on X"
        assert parsed["tradeoffs"] == "Speed vs quality"
        assert parsed["recommendation"] == "Go with A"
        assert parsed["confidence"] == 0.9

    def test_parse_confidence_clamped(self) -> None:
        """Confidence values > 1.0 are clamped to 1.0."""
        raw = "CONFIDENCE: 1.5\nRECOMMENDATION: test\n"
        parsed = _parse_synthesis(raw, [])
        assert parsed["confidence"] == 1.0

    def test_parse_confidence_floor(self) -> None:
        """Negative confidence values are clamped to 0.0."""
        raw = "CONFIDENCE: -0.5\nRECOMMENDATION: test\n"
        parsed = _parse_synthesis(raw, [])
        assert parsed["confidence"] == 0.0

    def test_parse_invalid_confidence_defaults(self) -> None:
        """Invalid confidence value defaults to 0.7."""
        raw = "CONFIDENCE: not_a_number\nRECOMMENDATION: test\n"
        parsed = _parse_synthesis(raw, [])
        assert parsed["confidence"] == 0.7

    def test_parse_no_sections_fallback(self) -> None:
        """No sections found: full text used as recommendation."""
        raw = "Just a plain text response with no structure."
        parsed = _parse_synthesis(raw, [])
        assert parsed["recommendation"] == raw.strip()

    def test_parse_blind_spots(self) -> None:
        """Blind spots for listed models are parsed."""
        raw = (
            "CONSENSUS: agree\n"
            "BLIND_SPOTS:\n"
            "gpt-4o: missed performance\n"
            "deepseek-chat: missed security\n"
        )
        models = ["gpt-4o", "deepseek-chat"]
        parsed = _parse_synthesis(raw, models)
        assert "gpt-4o" in parsed["blind_spots"]
        assert parsed["blind_spots"]["gpt-4o"] == "missed performance"


# ======================================================================
# TestGenerateReportText
# ======================================================================


class TestGenerateReportText:
    """Tests for generate_report_text() output format."""

    def test_report_contains_header(self, sample_result: DebateResult) -> None:
        """Report contains MULTI-MODEL DEBATE REPORT header."""
        text = generate_report_text(sample_result)
        assert "MULTI-MODEL DEBATE REPORT" in text

    def test_report_contains_question(self, sample_result: DebateResult) -> None:
        """Report contains the original question."""
        text = generate_report_text(sample_result)
        assert "Should we use microservices?" in text

    def test_report_contains_consensus(self, sample_result: DebateResult) -> None:
        """Report contains consensus section."""
        text = generate_report_text(sample_result)
        assert "Consensus:" in text
        assert "All agree on scalability" in text

    def test_report_contains_recommendation(self, sample_result: DebateResult) -> None:
        """Report contains recommendation section."""
        text = generate_report_text(sample_result)
        assert "Recommendation:" in text

    def test_report_contains_confidence(self, sample_result: DebateResult) -> None:
        """Report contains confidence score."""
        text = generate_report_text(sample_result)
        assert "Confidence:" in text
        assert "85%" in text

    def test_report_contains_blind_spots(self, sample_result: DebateResult) -> None:
        """Report contains blind spots section."""
        text = generate_report_text(sample_result)
        assert "Blind Spots:" in text
        assert "model-a" in text
        assert "missed cost" in text

    def test_report_contains_round_labels(self, sample_result: DebateResult) -> None:
        """Report contains round labels."""
        text = generate_report_text(sample_result)
        assert "Round 1" in text

    def test_empty_result_report(self) -> None:
        """Report handles empty result gracefully."""
        result = DebateResult(question="empty test")
        text = generate_report_text(result)
        assert "MULTI-MODEL DEBATE REPORT" in text
        assert "empty test" in text


# ======================================================================
# TestSaveAndListDebates
# ======================================================================


class TestSaveAndListDebates:
    """Tests for save_debate() and list_debates() persistence."""

    def test_save_debate_creates_file(
        self, sample_result: DebateResult, tmp_debates_dir: Path,
    ) -> None:
        """Saving a debate creates a .md file."""
        path = save_debate(sample_result, tmp_debates_dir)
        assert path.exists()
        assert path.suffix == ".md"

    def test_save_debate_content(
        self, sample_result: DebateResult, tmp_debates_dir: Path,
    ) -> None:
        """Saved file contains the report text."""
        path = save_debate(sample_result, tmp_debates_dir)
        content = path.read_text()
        assert "Should we use microservices?" in content

    def test_save_debate_filename_contains_slug(
        self, sample_result: DebateResult, tmp_debates_dir: Path,
    ) -> None:
        """Filename contains a slug derived from the question."""
        path = save_debate(sample_result, tmp_debates_dir)
        assert "microservices" in path.name.lower()

    def test_list_debates_returns_saved(
        self, sample_result: DebateResult, tmp_debates_dir: Path,
    ) -> None:
        """list_debates returns saved files."""
        save_debate(sample_result, tmp_debates_dir)
        debates = list_debates(tmp_debates_dir)
        assert len(debates) >= 1

    def test_list_debates_empty_dir(self, tmp_path: Path) -> None:
        """list_debates returns empty list for empty directory."""
        empty_dir = tmp_path / "empty_debates"
        empty_dir.mkdir()
        debates = list_debates(empty_dir)
        assert debates == []

    def test_list_debates_nonexistent_dir(self, tmp_path: Path) -> None:
        """list_debates returns empty list for nonexistent directory."""
        debates = list_debates(tmp_path / "nonexistent")
        assert debates == []

    def test_save_debate_creates_dir(
        self, sample_result: DebateResult, tmp_path: Path,
    ) -> None:
        """save_debate creates directory if it does not exist."""
        target = tmp_path / "new" / "nested" / "dir"
        path = save_debate(sample_result, target)
        assert path.exists()
        assert target.exists()

    def test_multiple_saves_listed(
        self, tmp_debates_dir: Path,
    ) -> None:
        """Multiple saves appear in list_debates."""
        for i in range(3):
            result = DebateResult(
                question=f"Question {i}",
                created_at=f"2025-06-{15 + i:02d}T10:00:00+00:00",
            )
            save_debate(result, tmp_debates_dir)
        debates = list_debates(tmp_debates_dir)
        assert len(debates) >= 3


# ======================================================================
# TestStubLLMCaller
# ======================================================================


class TestStubLLMCaller:
    """Tests for the _stub_llm_caller default."""

    def test_stub_returns_string(self) -> None:
        """Stub returns a non-empty string."""
        result = _stub_llm_caller("model-x", [{"role": "user", "content": "hi"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_stub_includes_model_name(self) -> None:
        """Stub response includes the model name."""
        result = _stub_llm_caller("my-model", [])
        assert "my-model" in result

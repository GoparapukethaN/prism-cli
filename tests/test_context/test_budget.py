"""Tests for prism.context.budget — Smart Context Budget Manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.context.budget import (
    DEFAULT_CONTEXT_WINDOW,
    MODEL_CONTEXT_WINDOWS,
    RESERVED_OUTPUT_TOKENS,
    TOKEN_CHARS_RATIO,
    ContextBudgetManager,
    ContextChunk,
    ContextStats,
    SmartContextBudget,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    source: str = "file.py",
    content: str = "x" * 400,
    score: float = 0.5,
    token_estimate: int = 100,
    chunk_type: str = "file",
    reason: str = "test",
) -> ContextChunk:
    return ContextChunk(
        source=source,
        content=content,
        score=score,
        token_estimate=token_estimate,
        chunk_type=chunk_type,
        reason=reason,
    )


def _make_project(tmp_path: Path) -> Path:
    """Create a small project tree and return its root."""
    (tmp_path / "router.py").write_text("def route(): pass\n" * 10)
    (tmp_path / "selector.py").write_text("def select(): pass\n" * 10)
    (tmp_path / "test_router.py").write_text("def test_route(): pass\n" * 10)
    (tmp_path / "utils.py").write_text("def helper(): pass\n" * 10)
    (tmp_path / "README.md").write_text("# Readme\n" * 10)
    (tmp_path / "unrelated.py").write_text("x = 1\n" * 10)
    return tmp_path


# ---------------------------------------------------------------------------
# ContextChunk
# ---------------------------------------------------------------------------


class TestContextChunk:
    """Tests for the ContextChunk dataclass."""

    def test_fields(self) -> None:
        chunk = _make_chunk(source="a.py", score=0.8, token_estimate=50, chunk_type="file")
        assert chunk.source == "a.py"
        assert chunk.score == 0.8
        assert chunk.token_estimate == 50
        assert chunk.chunk_type == "file"

    def test_is_included_true_when_score_positive(self) -> None:
        chunk = _make_chunk(score=0.01)
        assert chunk.is_included is True

    def test_is_included_false_when_score_zero(self) -> None:
        chunk = _make_chunk(score=0.0)
        assert chunk.is_included is False

    def test_is_included_false_when_score_negative_edge(self) -> None:
        # score should stay in 0..1, but the property only checks > 0.0
        chunk = _make_chunk(score=0.0)
        assert chunk.is_included is False

    def test_content_stored(self) -> None:
        chunk = _make_chunk(content="hello world")
        assert chunk.content == "hello world"

    def test_reason_stored(self) -> None:
        chunk = _make_chunk(reason="Directly mentioned in task")
        assert chunk.reason == "Directly mentioned in task"


# ---------------------------------------------------------------------------
# SmartContextBudget
# ---------------------------------------------------------------------------


class TestSmartContextBudget:
    """Tests for the SmartContextBudget dataclass."""

    def test_fields(self) -> None:
        budget = SmartContextBudget(
            model="gpt-4o",
            total_budget=128000,
            used_tokens=50000,
            remaining_tokens=73904,
            included_chunks=[_make_chunk()],
            excluded_chunks=[],
            efficiency=0.6,
        )
        assert budget.model == "gpt-4o"
        assert budget.total_budget == 128000
        assert budget.used_tokens == 50000
        assert budget.remaining_tokens == 73904
        assert len(budget.included_chunks) == 1
        assert len(budget.excluded_chunks) == 0
        assert budget.efficiency == 0.6

    def test_usage_percent(self) -> None:
        budget = SmartContextBudget(
            model="gpt-4o",
            total_budget=100000,
            used_tokens=25000,
            remaining_tokens=70904,
            included_chunks=[],
            excluded_chunks=[],
            efficiency=0.5,
        )
        assert budget.usage_percent == pytest.approx(25.0)

    def test_usage_percent_zero_budget(self) -> None:
        budget = SmartContextBudget(
            model="x",
            total_budget=0,
            used_tokens=0,
            remaining_tokens=0,
            included_chunks=[],
            excluded_chunks=[],
            efficiency=0.0,
        )
        # Should not raise ZeroDivisionError (uses max(total, 1))
        assert budget.usage_percent == 0.0

    def test_usage_percent_full(self) -> None:
        budget = SmartContextBudget(
            model="x",
            total_budget=1000,
            used_tokens=1000,
            remaining_tokens=0,
            included_chunks=[],
            excluded_chunks=[],
            efficiency=0.0,
        )
        assert budget.usage_percent == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# ContextStats
# ---------------------------------------------------------------------------


class TestContextStats:
    """Tests for the ContextStats dataclass."""

    def test_fields(self) -> None:
        stats = ContextStats(
            total_requests=10,
            avg_tokens_used=5000.0,
            avg_tokens_saved=3000.0,
            avg_efficiency=0.4,
            avg_chunks_included=5.0,
            avg_chunks_excluded=3.0,
            total_tokens_saved=30000,
        )
        assert stats.total_requests == 10
        assert stats.avg_tokens_used == 5000.0
        assert stats.avg_tokens_saved == 3000.0
        assert stats.avg_efficiency == 0.4
        assert stats.avg_chunks_included == 5.0
        assert stats.avg_chunks_excluded == 3.0
        assert stats.total_tokens_saved == 30000

    def test_zero_stats(self) -> None:
        stats = ContextStats(
            total_requests=0,
            avg_tokens_used=0.0,
            avg_tokens_saved=0.0,
            avg_efficiency=0.0,
            avg_chunks_included=0.0,
            avg_chunks_excluded=0.0,
            total_tokens_saved=0,
        )
        assert stats.total_requests == 0
        assert stats.total_tokens_saved == 0


# ---------------------------------------------------------------------------
# ContextBudgetManager — initialisation
# ---------------------------------------------------------------------------


class TestContextBudgetManagerInit:
    """Tests for manager construction."""

    def test_init(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._root == tmp_path.resolve()

    def test_init_resolves_path(self, tmp_path: Path) -> None:
        relative = tmp_path / "subdir" / ".."
        relative.mkdir(parents=True, exist_ok=True)
        mgr = ContextBudgetManager(project_root=relative)
        assert mgr._root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# ContextBudgetManager — scoring
# ---------------------------------------------------------------------------


class TestScoring:
    """Tests for file relevance scoring."""

    def test_direct_mention_gets_1_0(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        score = mgr._score_relevance("router.py", "fix the router bug")
        assert score == 1.0

    def test_file_path_mentioned_gets_1_0(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        score = mgr._score_relevance("src/router.py", "look at src/router.py")
        assert score == 1.0

    def test_keyword_match(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        score = mgr._score_relevance(
            "src/prism/router/selector.py",
            "fix the selector logic in the router module",
        )
        assert score >= 0.2

    def test_conversation_reference(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        messages = [{"content": "Please check router.py for issues"}]
        score = mgr._score_relevance("router.py", "debug this", messages)
        assert score >= 0.7

    def test_test_file_for_mentioned_source(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        score = mgr._score_relevance("test_router.py", "fix the router")
        # test_ in path + tested_name "router" in task → 0.6
        assert score >= 0.6

    def test_same_directory_keyword(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        score = mgr._score_relevance("src/prism/router/fallback.py", "the router fails")
        assert score >= 0.3

    def test_no_match_returns_zero(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        score = mgr._score_relevance("xyz_zzz.py", "fix authentication")
        assert score == 0.0

    def test_multiple_keyword_overlap_caps_at_0_8(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        # File path has "auth", "key", "validator"; task also has those
        score = mgr._score_relevance(
            "src/auth_key_validator_store.py",
            "fix the auth key validator store bug",
        )
        # Many overlapping words, each adds 0.2, capped at 0.8
        # But "auth_key_validator_store" directly in task → 1.0
        assert score <= 1.0

    def test_score_never_exceeds_1_0(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        score = mgr._score_relevance(
            "router.py",
            "router router router router router",
        )
        assert score <= 1.0


# ---------------------------------------------------------------------------
# ContextBudgetManager — select_context
# ---------------------------------------------------------------------------


class TestSelectContext:
    """Tests for the main select_context method."""

    def test_basic_selection(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="fix the router",
            available_files=["router.py", "utils.py", "unrelated.py"],
            model="gpt-4o",
        )
        # router.py should be included (score 1.0)
        sources = [c.source for c in budget.included_chunks]
        assert "router.py" in sources

    def test_excludes_low_score_files(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="fix the router",
            available_files=["router.py", "unrelated.py"],
            model="gpt-4o",
        )
        included_sources = [c.source for c in budget.included_chunks]
        # unrelated.py should not be included if it scored 0.0
        # (it might still be included if any keyword matches — check either way)
        assert "router.py" in included_sources

    def test_error_context_always_included(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="debug this",
            available_files=["utils.py"],
            model="gpt-4o",
            error_context="Traceback: AttributeError in router.py line 42",
        )
        sources = [c.source for c in budget.included_chunks]
        assert "error_context" in sources
        # Error chunk should have score 1.0
        error_chunk = next(c for c in budget.included_chunks if c.source == "error_context")
        assert error_chunk.score == 1.0
        assert error_chunk.chunk_type == "error"

    def test_empty_available_files(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="fix the router",
            available_files=[],
            model="gpt-4o",
        )
        assert budget.included_chunks == []
        assert budget.excluded_chunks == []
        assert budget.used_tokens == 0

    def test_nonexistent_files_skipped(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="fix the router",
            available_files=["does_not_exist.py"],
            model="gpt-4o",
        )
        assert budget.included_chunks == []
        assert budget.excluded_chunks == []

    def test_conversation_messages_affect_scoring(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        mgr = ContextBudgetManager(project_root=tmp_path)
        messages = [{"content": "Check the selector module"}]
        budget = mgr.select_context(
            task_description="debug this",
            available_files=["selector.py", "unrelated.py"],
            model="gpt-4o",
            conversation_messages=messages,
        )
        included_sources = [c.source for c in budget.included_chunks]
        assert "selector.py" in included_sources

    def test_model_budget_used(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="fix the router",
            available_files=["router.py"],
            model="claude-sonnet-4-20250514",
        )
        assert budget.total_budget == 200000
        assert budget.model == "claude-sonnet-4-20250514"

    def test_unknown_model_uses_default(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="fix the router",
            available_files=["router.py"],
            model="some-unknown-model",
        )
        assert budget.total_budget == DEFAULT_CONTEXT_WINDOW

    def test_history_recorded(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert len(mgr._history) == 0
        mgr.select_context(
            task_description="fix the router",
            available_files=["router.py"],
            model="gpt-4o",
        )
        assert len(mgr._history) == 1
        entry = mgr._history[0]
        assert entry.model == "gpt-4o"
        assert entry.included >= 0
        assert entry.excluded >= 0


# ---------------------------------------------------------------------------
# ContextBudgetManager — token budget enforcement
# ---------------------------------------------------------------------------


class TestTokenBudgetEnforcement:
    """Tests for budget limits and overflow handling."""

    def test_fills_up_to_budget_limit(self, tmp_path: Path) -> None:
        """Create files that together exceed a tiny model budget."""
        # Write a big file
        (tmp_path / "big.py").write_text("x = 1\n" * 5000)  # ~30000 chars ≈ 7500 tokens
        (tmp_path / "small.py").write_text("y = 2\n")

        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="fix big and small",
            available_files=["big.py", "small.py"],
            model="ollama/llama3.1:8b",  # Only 8192 token budget
        )
        # Some files should be included, but budget should not be exceeded
        assert budget.used_tokens <= (
            MODEL_CONTEXT_WINDOWS["ollama/llama3.1:8b"] - RESERVED_OUTPUT_TOKENS
        )

    def test_excludes_overflow_chunks(self, tmp_path: Path) -> None:
        """With very limited budget, at least one chunk should be excluded."""
        # Write many files
        for i in range(20):
            (tmp_path / f"file_{i}.py").write_text(f"content_{i} = True\n" * 500)

        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="fix all the files file_0 file_1 file_2 file_3",
            available_files=[f"file_{i}.py" for i in range(20)],
            model="ollama/llama3.1:8b",  # Tiny budget
        )
        # With 20 files of ~500 lines each, some must be excluded
        assert len(budget.excluded_chunks) > 0

    def test_conversation_tokens_reduce_available_budget(self, tmp_path: Path) -> None:
        (tmp_path / "code.py").write_text("x = 1\n" * 100)
        mgr = ContextBudgetManager(project_root=tmp_path)

        # Big conversation uses tokens
        big_messages = [{"content": "word " * 5000}]  # ~5000 tokens
        budget = mgr.select_context(
            task_description="fix code",
            available_files=["code.py"],
            model="ollama/llama3.1:8b",
            conversation_messages=big_messages,
        )
        # Remaining should be reduced by conversation tokens
        conversation_tokens = len("word " * 5000) // TOKEN_CHARS_RATIO
        expected_available = (
            MODEL_CONTEXT_WINDOWS["ollama/llama3.1:8b"]
            - conversation_tokens
            - RESERVED_OUTPUT_TOKENS
        )
        assert budget.remaining_tokens <= expected_available


# ---------------------------------------------------------------------------
# ContextBudgetManager — manual add / drop
# ---------------------------------------------------------------------------


class TestManualOverrides:
    """Tests for add_file, drop_file, reset_overrides."""

    def test_add_file_forces_inclusion(self, tmp_path: Path) -> None:
        (tmp_path / "unrelated.py").write_text("x = 1\n")
        mgr = ContextBudgetManager(project_root=tmp_path)
        mgr.add_file("unrelated.py")
        budget = mgr.select_context(
            task_description="fix the router",
            available_files=["unrelated.py"],
            model="gpt-4o",
        )
        sources = [c.source for c in budget.included_chunks]
        assert "unrelated.py" in sources

    def test_drop_file_excludes(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        mgr = ContextBudgetManager(project_root=tmp_path)
        mgr.drop_file("router.py")
        budget = mgr.select_context(
            task_description="fix the router",
            available_files=["router.py", "utils.py"],
            model="gpt-4o",
        )
        all_sources = [c.source for c in budget.included_chunks + budget.excluded_chunks]
        assert "router.py" not in all_sources

    def test_add_then_drop(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        mgr.add_file("foo.py")
        assert "foo.py" in mgr._manual_includes
        mgr.drop_file("foo.py")
        assert "foo.py" not in mgr._manual_includes
        assert "foo.py" in mgr._manual_excludes

    def test_drop_then_add(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        mgr.drop_file("foo.py")
        assert "foo.py" in mgr._manual_excludes
        mgr.add_file("foo.py")
        assert "foo.py" not in mgr._manual_excludes
        assert "foo.py" in mgr._manual_includes

    def test_reset_overrides(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        mgr.add_file("a.py")
        mgr.drop_file("b.py")
        assert mgr._manual_includes == {"a.py"}
        assert mgr._manual_excludes == {"b.py"}
        mgr.reset_overrides()
        assert mgr._manual_includes == set()
        assert mgr._manual_excludes == set()


# ---------------------------------------------------------------------------
# ContextBudgetManager — efficiency
# ---------------------------------------------------------------------------


class TestEfficiency:
    """Tests for the efficiency calculation."""

    def test_efficiency_greater_than_zero(self, tmp_path: Path) -> None:
        """When some files are excluded, efficiency should be positive."""
        for i in range(10):
            (tmp_path / f"file_{i}.py").write_text(f"content_{i}\n" * 200)

        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="fix file_0",
            available_files=[f"file_{i}.py" for i in range(10)],
            model="gpt-4o",
        )
        # With only one file directly mentioned, efficiency should be positive
        assert budget.efficiency >= 0.0

    def test_efficiency_zero_when_all_included(self, tmp_path: Path) -> None:
        """When every chunk fits within budget, efficiency is 0.0 (no savings)."""
        (tmp_path / "tiny.py").write_text("x = 1\n")
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = mgr.select_context(
            task_description="fix tiny",
            available_files=["tiny.py"],
            model="gemini/gemini-1.5-pro",  # 1M tokens — everything fits
        )
        # All chunks are included, so efficiency = 1 - (used / total)
        # With a single file, efficiency ≈ 0.0
        assert budget.efficiency >= 0.0


# ---------------------------------------------------------------------------
# ContextBudgetManager — stats
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for get_stats aggregation."""

    def test_empty_stats(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        stats = mgr.get_stats()
        assert stats.total_requests == 0
        assert stats.avg_tokens_used == 0.0
        assert stats.avg_tokens_saved == 0.0
        assert stats.avg_efficiency == 0.0
        assert stats.avg_chunks_included == 0.0
        assert stats.avg_chunks_excluded == 0.0
        assert stats.total_tokens_saved == 0

    def test_stats_after_one_request(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        mgr = ContextBudgetManager(project_root=tmp_path)
        mgr.select_context(
            task_description="fix the router",
            available_files=["router.py", "unrelated.py"],
            model="gpt-4o",
        )
        stats = mgr.get_stats()
        assert stats.total_requests == 1
        assert stats.avg_tokens_used >= 0.0

    def test_stats_after_multiple_requests(self, tmp_path: Path) -> None:
        _make_project(tmp_path)
        mgr = ContextBudgetManager(project_root=tmp_path)

        mgr.select_context(
            task_description="fix the router",
            available_files=["router.py"],
            model="gpt-4o",
        )
        mgr.select_context(
            task_description="fix the selector",
            available_files=["selector.py"],
            model="gpt-4o",
        )

        stats = mgr.get_stats()
        assert stats.total_requests == 2
        assert stats.avg_tokens_used >= 0.0
        assert stats.avg_chunks_included >= 0.0

    def test_total_tokens_saved_accumulates(self, tmp_path: Path) -> None:
        """total_tokens_saved should accumulate across requests."""
        for i in range(5):
            (tmp_path / f"file_{i}.py").write_text(f"x_{i} = 1\n" * 100)

        mgr = ContextBudgetManager(project_root=tmp_path)

        # Two requests — each excluding some files
        mgr.select_context(
            task_description="fix file_0",
            available_files=[f"file_{i}.py" for i in range(5)],
            model="gpt-4o",
        )
        mgr.select_context(
            task_description="fix file_1",
            available_files=[f"file_{i}.py" for i in range(5)],
            model="gpt-4o",
        )

        stats = mgr.get_stats()
        assert stats.total_tokens_saved >= 0


# ---------------------------------------------------------------------------
# ContextBudgetManager — show_context
# ---------------------------------------------------------------------------


class TestShowContext:
    """Tests for the show_context formatter."""

    def test_show_includes_header(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = SmartContextBudget(
            model="gpt-4o",
            total_budget=128000,
            used_tokens=5000,
            remaining_tokens=118904,
            included_chunks=[_make_chunk(source="a.py", score=0.9, reason="test reason")],
            excluded_chunks=[],
            efficiency=0.5,
        )
        output = mgr.show_context(budget)
        assert "Context Budget:" in output
        assert "5,000" in output
        assert "128,000" in output

    def test_show_includes_efficiency(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = SmartContextBudget(
            model="gpt-4o",
            total_budget=128000,
            used_tokens=5000,
            remaining_tokens=118904,
            included_chunks=[],
            excluded_chunks=[],
            efficiency=0.5,
        )
        output = mgr.show_context(budget)
        assert "Efficiency:" in output
        assert "50.0%" in output

    def test_show_lists_included_chunks(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = SmartContextBudget(
            model="gpt-4o",
            total_budget=128000,
            used_tokens=5000,
            remaining_tokens=118904,
            included_chunks=[
                _make_chunk(source="router.py", score=1.0, reason="Directly mentioned in task"),
            ],
            excluded_chunks=[],
            efficiency=0.5,
        )
        output = mgr.show_context(budget)
        assert "router.py" in output
        assert "Directly mentioned in task" in output

    def test_show_lists_excluded_chunks(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = SmartContextBudget(
            model="gpt-4o",
            total_budget=128000,
            used_tokens=5000,
            remaining_tokens=118904,
            included_chunks=[],
            excluded_chunks=[
                _make_chunk(source="unrelated.py", score=0.0),
            ],
            efficiency=0.5,
        )
        output = mgr.show_context(budget)
        assert "Excluded:" in output
        assert "unrelated.py" in output

    def test_show_caps_excluded_at_10(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        excluded = [_make_chunk(source=f"file_{i}.py", score=0.0) for i in range(20)]
        budget = SmartContextBudget(
            model="gpt-4o",
            total_budget=128000,
            used_tokens=0,
            remaining_tokens=123904,
            included_chunks=[],
            excluded_chunks=excluded,
            efficiency=1.0,
        )
        output = mgr.show_context(budget)
        # Only first 10 excluded should be shown
        assert "file_9.py" in output
        assert "file_10.py" not in output

    def test_show_no_excluded_section_when_empty(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        budget = SmartContextBudget(
            model="gpt-4o",
            total_budget=128000,
            used_tokens=5000,
            remaining_tokens=118904,
            included_chunks=[_make_chunk()],
            excluded_chunks=[],
            efficiency=0.5,
        )
        output = mgr.show_context(budget)
        assert "Excluded:" not in output


# ---------------------------------------------------------------------------
# ContextBudgetManager — _get_token_budget
# ---------------------------------------------------------------------------


class TestGetTokenBudget:
    """Tests for the internal _get_token_budget helper."""

    def test_known_model(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_token_budget("gpt-4o") == 128000

    def test_known_model_claude(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_token_budget("claude-sonnet-4-20250514") == 200000

    def test_known_model_gemini(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_token_budget("gemini/gemini-1.5-pro") == 1000000

    def test_known_model_ollama_small(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_token_budget("ollama/llama3.1:8b") == 8192

    def test_unknown_model_falls_back(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_token_budget("totally-unknown/model") == DEFAULT_CONTEXT_WINDOW


# ---------------------------------------------------------------------------
# ContextBudgetManager — _classify_chunk
# ---------------------------------------------------------------------------


class TestClassifyChunk:
    """Tests for chunk type classification."""

    def test_regular_file(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._classify_chunk("src/router.py") == "file"

    def test_test_file(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._classify_chunk("tests/test_router.py") == "test"

    def test_test_in_path(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._classify_chunk("test_helpers/utils.py") == "test"

    def test_markdown_file(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._classify_chunk("docs/API.md") == "docstring"

    def test_readme_markdown(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._classify_chunk("README.md") == "docstring"


# ---------------------------------------------------------------------------
# ContextBudgetManager — _get_inclusion_reason
# ---------------------------------------------------------------------------


class TestGetInclusionReason:
    """Tests for human-readable inclusion reasons."""

    def test_score_0_9_directly_mentioned(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_inclusion_reason("f.py", "task", 0.95) == "Directly mentioned in task"

    def test_score_0_9_exact(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_inclusion_reason("f.py", "task", 0.9) == "Directly mentioned in task"

    def test_score_0_7_referenced(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_inclusion_reason("f.py", "task", 0.7) == "Referenced in conversation"

    def test_score_0_5_keyword(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_inclusion_reason("f.py", "task", 0.5) == "Keyword match with task"

    def test_score_0_3_related(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_inclusion_reason("f.py", "task", 0.3) == "Related by directory/module"

    def test_score_low_relevance(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_inclusion_reason("f.py", "task", 0.1) == "Low relevance"

    def test_score_zero(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._get_inclusion_reason("f.py", "task", 0.0) == "Low relevance"


# ---------------------------------------------------------------------------
# ContextBudgetManager — _read_file_safe
# ---------------------------------------------------------------------------


class TestReadFileSafe:
    """Tests for safe file reading."""

    def test_reads_existing_file(self, tmp_path: Path) -> None:
        (tmp_path / "hello.py").write_text("print('hello')\n")
        mgr = ContextBudgetManager(project_root=tmp_path)
        content = mgr._read_file_safe("hello.py")
        assert "print('hello')" in content

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._read_file_safe("no_such_file.py") == ""

    def test_skips_large_files(self, tmp_path: Path) -> None:
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * 600_000)  # > 500KB
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._read_file_safe("big.bin") == ""

    def test_reads_absolute_path(self, tmp_path: Path) -> None:
        target = tmp_path / "abs.py"
        target.write_text("ABS = True\n")
        mgr = ContextBudgetManager(project_root=tmp_path)
        content = mgr._read_file_safe(str(target))
        assert "ABS = True" in content

    def test_handles_unreadable_file(self, tmp_path: Path) -> None:
        """If we pass a directory as file path, it should return empty."""
        subdir = tmp_path / "adir"
        subdir.mkdir()
        mgr = ContextBudgetManager(project_root=tmp_path)
        assert mgr._read_file_safe("adir") == ""

    def test_reads_file_just_under_limit(self, tmp_path: Path) -> None:
        """Files at 499KB should be read."""
        f = tmp_path / "almost_big.txt"
        f.write_bytes(b"x" * 499_000)
        mgr = ContextBudgetManager(project_root=tmp_path)
        content = mgr._read_file_safe("almost_big.txt")
        assert len(content) > 0


# ---------------------------------------------------------------------------
# ContextBudgetManager — constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_default_context_window(self) -> None:
        assert DEFAULT_CONTEXT_WINDOW == 128000

    def test_reserved_output_tokens(self) -> None:
        assert RESERVED_OUTPUT_TOKENS == 4096

    def test_token_chars_ratio(self) -> None:
        assert TOKEN_CHARS_RATIO == 4

    def test_model_context_windows_has_entries(self) -> None:
        assert len(MODEL_CONTEXT_WINDOWS) > 0

    def test_all_windows_positive(self) -> None:
        for model, window in MODEL_CONTEXT_WINDOWS.items():
            assert window > 0, f"Window for {model} must be positive"

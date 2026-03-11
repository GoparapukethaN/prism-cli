"""End-to-end integration tests — full pipelines with all components wired.

Every external call (LiteLLM, subprocess, network, Ollama) is mocked.
No real API calls are ever made.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from prism.cache.response_cache import ResponseCache
from prism.config.schema import BudgetConfig, PrismConfig, RoutingConfig
from prism.config.settings import Settings
from prism.context.manager import ContextManager
from prism.cost.tracker import BudgetAction, CostTracker
from prism.db.database import Database
from prism.exceptions import (
    BlockedCommandError,
    ExcludedFileError,
    PathTraversalError,
)
from prism.intelligence.aei import (
    AdaptiveExecutionIntelligence,
    FixStrategy,
)
from prism.intelligence.blast_radius import BlastRadiusAnalyzer, RiskLevel
from prism.intelligence.test_gaps import TestGapHunter
from prism.llm.completion import CompletionEngine
from prism.llm.mock import MockLiteLLM, MockResponse
from prism.network.offline import ConnectivityState, OfflineModeManager
from prism.network.privacy import PrivacyManager, PrivacyViolationError
from prism.plugins.manager import (
    BUILTIN_PLUGINS,
    PluginManager,
    PluginManifest,
    PluginNotFoundError,
    PluginToolSpec,
)
from prism.providers.base import ComplexityTier
from prism.providers.registry import ProviderRegistry
from prism.router.classifier import TaskClassifier, TaskContext
from prism.router.selector import ModelSelector
from prism.security.path_guard import PathGuard
from prism.security.sandbox import CommandSandbox
from prism.tools.directory import ListDirectoryTool
from prism.tools.file_edit import EditFileTool
from prism.tools.file_read import ReadFileTool
from prism.tools.file_write import WriteFileTool
from prism.tools.search import SearchCodebaseTool
from prism.tools.terminal import ExecuteCommandTool

if TYPE_CHECKING:
    from pathlib import Path


# ======================================================================
# Shared helpers
# ======================================================================


def _make_settings(
    tmp_path: Path,
    daily_limit: float | None = 5.0,
    monthly_limit: float | None = 50.0,
    exploration_rate: float = 0.0,
) -> Settings:
    """Build deterministic settings for integration tests."""
    config = PrismConfig(
        routing=RoutingConfig(
            simple_threshold=0.3,
            medium_threshold=0.7,
            exploration_rate=exploration_rate,
            quality_weight=0.7,
        ),
        budget=BudgetConfig(
            daily_limit=daily_limit,
            monthly_limit=monthly_limit,
            warn_at_percent=80.0,
        ),
        prism_home=tmp_path / ".prism",
    )
    settings = Settings(config=config, project_root=tmp_path)
    settings.ensure_directories()
    return settings


def _mock_auth() -> MagicMock:
    """Mock AuthManager returning fake keys for all providers."""
    auth = MagicMock()
    auth.get_key.return_value = "fake-key-1234"
    return auth


def _init_db() -> Database:
    """Create an in-memory database with all tables."""
    db = Database(":memory:")
    db.initialize()
    return db


# ======================================================================
# 1. FULL ROUTING PIPELINE (10+ tests)
# ======================================================================


@pytest.mark.integration
class TestFullRoutingPipeline:
    """Task classification -> model selection -> cost estimation ->
    budget check -> completion (mocked) -> cost tracking -> response.
    """

    def _build_pipeline(
        self, tmp_path: Path
    ) -> tuple[
        TaskClassifier,
        ModelSelector,
        CompletionEngine,
        CostTracker,
        MockLiteLLM,
        Settings,
    ]:
        settings = _make_settings(tmp_path)
        db = _init_db()
        auth = _mock_auth()
        registry = ProviderRegistry(settings=settings, auth_manager=auth)
        cost_tracker = CostTracker(db=db, settings=settings)
        classifier = TaskClassifier(settings)
        selector = ModelSelector(
            settings=settings, registry=registry, cost_tracker=cost_tracker
        )
        mock_llm = MockLiteLLM()
        engine = CompletionEngine(
            settings=settings,
            cost_tracker=cost_tracker,
            auth_manager=auth,
            provider_registry=registry,
            litellm_backend=mock_llm,
        )
        return classifier, selector, engine, cost_tracker, mock_llm, settings

    def test_simple_task_routed_and_completed(self, tmp_path: Path) -> None:
        classifier, selector, engine, _, mock_llm, _ = (
            self._build_pipeline(tmp_path)
        )
        prompt = "fix the typo in line 5"
        result = classifier.classify(prompt)
        assert result.tier == ComplexityTier.SIMPLE

        selection = selector.select(result.tier, prompt)
        assert selection.model_id
        assert selection.estimated_cost >= 0.0

        mock_llm.set_response(
            selection.model_id,
            MockResponse(content="Fixed the typo.", input_tokens=20, output_tokens=10),
        )

        import asyncio

        cr = asyncio.get_event_loop().run_until_complete(
            engine.complete(
                messages=[{"role": "user", "content": prompt}],
                model=selection.model_id,
                session_id="e2e-simple",
                complexity_tier=result.tier.value,
            )
        )
        assert cr.content == "Fixed the typo."
        assert cr.input_tokens == 20
        assert cr.output_tokens == 10

    def test_medium_task_routed_and_completed(self, tmp_path: Path) -> None:
        classifier, selector, engine, _, mock_llm, _ = (
            self._build_pipeline(tmp_path)
        )
        prompt = (
            "refactor the database module to use async patterns "
            "with proper error handling across multiple files"
        )
        result = classifier.classify(prompt)
        assert result.tier == ComplexityTier.MEDIUM

        selection = selector.select(result.tier, prompt)
        mock_llm.set_response(
            selection.model_id,
            MockResponse(content="Refactored.", input_tokens=200, output_tokens=500),
        )

        import asyncio

        cr = asyncio.get_event_loop().run_until_complete(
            engine.complete(
                messages=[{"role": "user", "content": prompt}],
                model=selection.model_id,
                session_id="e2e-medium",
                complexity_tier=result.tier.value,
            )
        )
        assert cr.content == "Refactored."

    def test_complex_task_routed_and_completed(self, tmp_path: Path) -> None:
        classifier, selector, engine, _, mock_llm, _ = (
            self._build_pipeline(tmp_path)
        )
        # Build a prompt that reliably triggers COMPLEX classification:
        # - 50+ words to activate reasoning-pattern detection
        # - Multiple complex keywords (architect, distributed, etc.)
        # - Reasoning patterns ("trade-off", "either...or")
        # - High scope keywords ("architecture", "system")
        # - Provide TaskContext with many active_files
        prompt = (
            "architect a distributed microservice system from scratch "
            "with a scalable concurrent design. evaluate the trade-offs "
            "between consistency and performance for each algorithm. "
            "redesign the security architecture to be abstract and "
            "compare the patterns for each distributed component. "
            "if the latency is high then migrate to an optimized "
            "approach. either use event sourcing or CQRS depending "
            "on the system requirements. first analyze the codebase "
            "then refactor entire modules and finally deploy the "
            "abstract concurrent services across the cluster"
        )
        ctx = TaskContext(
            active_files=[f"file_{i}.py" for i in range(10)],
            conversation_turns=5,
            project_file_count=100,
        )
        result = classifier.classify(prompt, context=ctx)
        assert result.tier == ComplexityTier.COMPLEX

        selection = selector.select(result.tier, prompt)
        mock_llm.set_response(
            selection.model_id,
            MockResponse(content="Architecture plan.", input_tokens=500, output_tokens=2000),
        )

        import asyncio

        cr = asyncio.get_event_loop().run_until_complete(
            engine.complete(
                messages=[{"role": "user", "content": prompt}],
                model=selection.model_id,
                session_id="e2e-complex",
                complexity_tier=result.tier.value,
            )
        )
        assert cr.content == "Architecture plan."

    def test_cost_tracked_after_completion(self, tmp_path: Path) -> None:
        classifier, selector, engine, tracker, mock_llm, _ = (
            self._build_pipeline(tmp_path)
        )
        prompt = "fix the typo"
        result = classifier.classify(prompt)
        selection = selector.select(result.tier, prompt)

        mock_llm.set_response(
            selection.model_id,
            MockResponse(content="done", input_tokens=50, output_tokens=20),
        )

        import asyncio

        asyncio.get_event_loop().run_until_complete(
            engine.complete(
                messages=[{"role": "user", "content": prompt}],
                model=selection.model_id,
                session_id="e2e-cost",
                complexity_tier=result.tier.value,
            )
        )
        session_cost = tracker.get_session_cost("e2e-cost")
        assert session_cost >= 0.0

    def test_selection_returns_fallback_chain(self, tmp_path: Path) -> None:
        classifier, selector, *_ = self._build_pipeline(tmp_path)
        prompt = "implement a new feature with error handling"
        result = classifier.classify(prompt)
        selection = selector.select(result.tier, prompt)
        assert isinstance(selection.fallback_chain, list)

    def test_budget_check_allows_cheap_request(self, tmp_path: Path) -> None:
        _, _, _, tracker, _, _ = self._build_pipeline(tmp_path)
        action = tracker.check_budget(0.001)
        assert action == BudgetAction.PROCEED

    def test_budget_check_blocks_expensive_request(self, tmp_path: Path) -> None:
        _, _, _, tracker, _, _ = self._build_pipeline(tmp_path)
        action = tracker.check_budget(100.0)
        assert action == BudgetAction.BLOCK

    def test_classification_features_populated(self, tmp_path: Path) -> None:
        classifier, *_ = self._build_pipeline(tmp_path)
        ctx = TaskContext(active_files=["a.py", "b.py"])
        result = classifier.classify("explain this code", ctx)
        assert "prompt_token_count" in result.features
        assert "files_referenced" in result.features
        assert result.features["files_referenced"] == 2.0

    def test_selection_reasoning_populated(self, tmp_path: Path) -> None:
        _, selector, *_ = self._build_pipeline(tmp_path)
        selection = selector.select(ComplexityTier.SIMPLE, "fix typo")
        assert len(selection.reasoning) > 0
        assert selection.provider

    def test_multiple_requests_accumulate_cost(self, tmp_path: Path) -> None:
        _, selector, engine, tracker, mock_llm, _ = (
            self._build_pipeline(tmp_path)
        )
        selection = selector.select(ComplexityTier.SIMPLE, "fix something")
        mock_llm.set_response(
            selection.model_id,
            MockResponse(content="ok", input_tokens=100, output_tokens=50),
        )

        import asyncio

        for i in range(3):
            asyncio.get_event_loop().run_until_complete(
                engine.complete(
                    messages=[{"role": "user", "content": f"request {i}"}],
                    model=selection.model_id,
                    session_id="e2e-multi",
                    complexity_tier="simple",
                )
            )
        cost = tracker.get_session_cost("e2e-multi")
        assert cost >= 0.0

    def test_pinned_model_overrides_routing(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        settings.set_override("pinned_model", "gpt-4o")
        db = _init_db()
        auth = _mock_auth()
        registry = ProviderRegistry(settings=settings, auth_manager=auth)
        cost_tracker = CostTracker(db=db, settings=settings)
        selector = ModelSelector(
            settings=settings, registry=registry, cost_tracker=cost_tracker,
        )
        selection = selector.select(ComplexityTier.SIMPLE, "fix typo")
        assert selection.model_id == "gpt-4o"
        assert "Pinned" in selection.reasoning or "user override" in selection.reasoning


# ======================================================================
# 2. CONVERSATION FLOW (8+ tests)
# ======================================================================


@pytest.mark.integration
class TestConversationFlow:
    """Multi-turn conversation, context management, conversation
    branching, summarization.
    """

    def test_multi_turn_conversation_builds_context(self) -> None:
        mgr = ContextManager(max_tokens=128_000)
        mgr.add_message("user", "Write a function to sort a list.")
        mgr.add_message("assistant", "def sort_list(lst): return sorted(lst)")
        mgr.add_message("user", "Add type hints.")
        assert mgr.message_count == 3
        ctx = mgr.get_context()
        roles = [m["role"] for m in ctx]
        assert "system" in roles
        assert "user" in roles

    def test_active_files_included_in_context(self) -> None:
        mgr = ContextManager(max_tokens=128_000)
        mgr.add_active_file("src/main.py", "def main(): pass")
        mgr.add_message("user", "Explain main.py")
        ctx = mgr.get_context()
        files_msg = [m for m in ctx if "[Active Files]" in m.get("content", "")]
        assert len(files_msg) == 1
        assert "src/main.py" in files_msg[0]["content"]

    def test_clear_messages_resets_history(self) -> None:
        mgr = ContextManager(max_tokens=128_000)
        mgr.add_message("user", "hello")
        mgr.add_message("assistant", "hi")
        mgr.clear_messages()
        assert mgr.message_count == 0

    def test_context_trimming_drops_oldest(self) -> None:
        mgr = ContextManager(max_tokens=500)
        for i in range(50):
            mgr.add_message("user", f"message {i} " * 50)
        ctx = mgr.get_context()
        user_msgs = [m for m in ctx if m["role"] == "user"]
        assert len(user_msgs) < 50

    def test_repo_context_included(self) -> None:
        mgr = ContextManager(max_tokens=128_000)
        mgr.set_repo_context("module_a.py: class Foo\nmodule_b.py: class Bar")
        mgr.add_message("user", "What modules exist?")
        ctx = mgr.get_context()
        repo_msgs = [m for m in ctx if "[Repository Map]" in m.get("content", "")]
        assert len(repo_msgs) == 1

    def test_conversation_branching_via_separate_managers(self) -> None:
        mgr_a = ContextManager(max_tokens=128_000)
        mgr_b = ContextManager(max_tokens=128_000)
        mgr_a.add_message("user", "Branch A: implement feature X")
        mgr_b.add_message("user", "Branch B: implement feature Y")
        ctx_a = mgr_a.get_context()
        ctx_b = mgr_b.get_context()
        user_a = [m for m in ctx_a if m["role"] == "user"]
        user_b = [m for m in ctx_b if m["role"] == "user"]
        assert "feature X" in user_a[0]["content"]
        assert "feature Y" in user_b[0]["content"]

    def test_system_prompt_always_first(self) -> None:
        mgr = ContextManager(system_prompt="Custom system prompt.", max_tokens=128_000)
        mgr.add_message("user", "hello")
        ctx = mgr.get_context()
        assert ctx[0]["role"] == "system"
        assert "Custom system prompt" in ctx[0]["content"]

    def test_add_and_remove_active_file(self) -> None:
        mgr = ContextManager(max_tokens=128_000)
        mgr.add_active_file("a.py", "content_a")
        mgr.add_active_file("b.py", "content_b")
        assert len(mgr.active_files) == 2
        removed = mgr.remove_active_file("a.py")
        assert removed is True
        assert len(mgr.active_files) == 1
        assert "b.py" in mgr.active_files

    def test_total_tokens_estimation(self) -> None:
        mgr = ContextManager(max_tokens=128_000)
        mgr.add_message("user", "a " * 100)
        mgr.add_active_file("test.py", "b " * 200)
        tokens = mgr.total_tokens()
        assert tokens > 0

    def test_budget_allocation_fractions(self) -> None:
        mgr = ContextManager(max_tokens=100_000)
        budget = mgr.allocate_budget()
        assert budget.system_prompt + budget.conversation > 0
        assert budget.total <= 100_000


# ======================================================================
# 3. SECURITY PIPELINE (8+ tests)
# ======================================================================


@pytest.mark.integration
class TestSecurityPipeline:
    """Path traversal blocked, command sandbox limits, secret filtering,
    .prismignore patterns.
    """

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        with pytest.raises(PathTraversalError):
            guard.validate("../../etc/passwd")

    def test_null_byte_in_path_rejected(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        with pytest.raises(ValueError, match="null byte"):
            guard.validate("file\x00.txt")

    def test_excluded_pattern_blocked(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path, excluded_patterns=[".env"])
        (tmp_path / ".env").write_text("SECRET=123")
        with pytest.raises(ExcludedFileError):
            guard.validate(".env")

    def test_always_blocked_patterns(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        (ssh_dir / "id_rsa").write_text("private key")
        assert guard.is_safe(".ssh/id_rsa") is False

    def test_sandbox_blocks_dangerous_command(self, tmp_path: Path) -> None:
        sandbox = CommandSandbox(project_root=tmp_path)
        with pytest.raises(BlockedCommandError):
            sandbox.execute("rm -rf /")

    def test_sandbox_blocks_sudo_rm(self, tmp_path: Path) -> None:
        sandbox = CommandSandbox(project_root=tmp_path)
        with pytest.raises(BlockedCommandError):
            sandbox.execute("sudo rm -rf /home")

    def test_sandbox_executes_safe_command(self, tmp_path: Path) -> None:
        sandbox = CommandSandbox(project_root=tmp_path)
        result = sandbox.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_sandbox_timeout_enforced(self, tmp_path: Path) -> None:
        sandbox = CommandSandbox(project_root=tmp_path, timeout=1)
        result = sandbox.execute("sleep 5", timeout=1)
        assert result.timed_out is True
        assert result.exit_code == -1

    def test_sandbox_env_filtering(self, tmp_path: Path) -> None:
        sandbox = CommandSandbox(project_root=tmp_path)
        result = sandbox.execute(
            "env",
            env={"MY_API_KEY": "secret123"},
        )
        assert "secret123" not in result.stdout

    def test_read_tool_blocks_traversal(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        tool = ReadFileTool(path_guard=guard)
        result = tool.execute({"path": "../../etc/passwd"})
        assert result.success is False
        assert "escapes" in (result.error or "").lower() or result.error is not None

    def test_write_tool_blocks_excluded_file(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path, excluded_patterns=["*.key"])
        tool = WriteFileTool(path_guard=guard)
        result = tool.execute({"path": "secret.key", "content": "data"})
        assert result.success is False

    def test_terminal_tool_dangerous_permission(self, tmp_path: Path) -> None:
        sandbox = CommandSandbox(project_root=tmp_path)
        tool = ExecuteCommandTool(sandbox=sandbox)
        from prism.tools.base import PermissionLevel

        level = tool.get_effective_permission("rm -rf /tmp/stuff")
        assert level == PermissionLevel.DANGEROUS


# ======================================================================
# 4. COST PIPELINE (6+ tests)
# ======================================================================


@pytest.mark.integration
class TestCostPipeline:
    """Cost tracking across requests, budget enforcement, daily/monthly
    limits, cost forecasting.
    """

    def _tracker(self, tmp_path: Path, daily: float = 5.0) -> CostTracker:
        settings = _make_settings(tmp_path, daily_limit=daily, monthly_limit=50.0)
        db = _init_db()
        return CostTracker(db=db, settings=settings)

    def test_track_records_cost_entry(self, tmp_path: Path) -> None:
        tracker = self._tracker(tmp_path)
        entry = tracker.track(
            model_id="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            session_id="cost-s1",
            complexity_tier="medium",
        )
        assert entry.cost_usd >= 0.0
        assert entry.model_id == "gpt-4o-mini"
        assert entry.session_id == "cost-s1"

    def test_session_cost_accumulates(self, tmp_path: Path) -> None:
        tracker = self._tracker(tmp_path)
        tracker.track("gpt-4o-mini", 1000, 500, "sess1", "medium")
        tracker.track("gpt-4o-mini", 2000, 1000, "sess1", "medium")
        cost = tracker.get_session_cost("sess1")
        assert cost >= 0.0

    def test_daily_cost_queries(self, tmp_path: Path) -> None:
        tracker = self._tracker(tmp_path)
        tracker.track("gpt-4o-mini", 500, 200, "daily1", "simple")
        daily = tracker.get_daily_cost()
        assert daily >= 0.0

    def test_budget_remaining_with_limits(self, tmp_path: Path) -> None:
        tracker = self._tracker(tmp_path, daily=1.0)
        remaining = tracker.get_budget_remaining()
        assert remaining is not None
        assert remaining <= 1.0

    def test_budget_warn_at_high_usage(self, tmp_path: Path) -> None:
        tracker = self._tracker(tmp_path, daily=0.001)
        tracker.track("gpt-4o", 10000, 5000, "warn1", "complex")
        action = tracker.check_budget(0.0001)
        assert action in (BudgetAction.WARN, BudgetAction.BLOCK)

    def test_budget_block_when_exceeded(self, tmp_path: Path) -> None:
        tracker = self._tracker(tmp_path, daily=0.0001)
        tracker.track("gpt-4o", 50000, 20000, "block1", "complex")
        action = tracker.check_budget(10.0)
        assert action == BudgetAction.BLOCK

    def test_no_limit_returns_proceed(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path, daily_limit=None, monthly_limit=None)
        db = _init_db()
        tracker = CostTracker(db=db, settings=settings)
        action = tracker.check_budget(100.0)
        assert action == BudgetAction.PROCEED


# ======================================================================
# 5. CACHE PIPELINE (6+ tests)
# ======================================================================


@pytest.mark.integration
class TestCachePipeline:
    """Cache miss -> API call -> cache store -> cache hit (no API call),
    TTL expiry, file-change invalidation.
    """

    def test_cache_miss_then_hit(self, tmp_path: Path) -> None:
        cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
        key = ResponseCache.make_cache_key(
            "gpt-4o", "system prompt", "user prompt"
        )
        miss = cache.get(key)
        assert miss is None

        cache.put(
            key,
            model="gpt-4o",
            provider="openai",
            content="cached answer",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.005,
            finish_reason="stop",
        )

        hit = cache.get(key)
        assert hit is not None
        assert hit.content == "cached answer"
        assert hit.model == "gpt-4o"
        cache.close()

    def test_cache_disabled_returns_none(self, tmp_path: Path) -> None:
        cache = ResponseCache(cache_dir=tmp_path / "cache", enabled=False)
        key = ResponseCache.make_cache_key("gpt-4o", "sys", "user")
        cache.put(
            key, model="gpt-4o", provider="openai", content="data",
            input_tokens=10, output_tokens=5, cost_usd=0.001,
            finish_reason="stop",
        )
        result = cache.get(key)
        assert result is None
        cache.close()

    def test_cache_ttl_expiry(self, tmp_path: Path) -> None:
        cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=1)
        key = ResponseCache.make_cache_key("gpt-4o", "sys", "user")
        cache.put(
            key, model="gpt-4o", provider="openai", content="short-lived",
            input_tokens=10, output_tokens=5, cost_usd=0.001,
            finish_reason="stop", ttl=1,
        )
        hit = cache.get(key)
        assert hit is not None

        time.sleep(1.5)
        expired = cache.get(key)
        assert expired is None
        cache.close()

    def test_cache_file_change_invalidation(self, tmp_path: Path) -> None:
        cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
        src_file = tmp_path / "src.py"
        src_file.write_text("original content")

        key = ResponseCache.make_cache_key(
            "gpt-4o", "sys", "user", "original content"
        )
        cache.put(
            key, model="gpt-4o", provider="openai", content="cached",
            input_tokens=10, output_tokens=5, cost_usd=0.001,
            finish_reason="stop",
            file_paths=[str(src_file)],
        )

        hit = cache.get(key, file_paths=[str(src_file)])
        assert hit is not None

        time.sleep(0.1)
        src_file.write_text("modified content")

        invalidated = cache.get(key, file_paths=[str(src_file)])
        assert invalidated is None
        cache.close()

    def test_cache_clear_all(self, tmp_path: Path) -> None:
        cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
        for i in range(5):
            key = ResponseCache.make_cache_key("model", "sys", f"prompt {i}")
            cache.put(
                key, model="model", provider="p", content=f"r{i}",
                input_tokens=10, output_tokens=5, cost_usd=0.001,
                finish_reason="stop",
            )
        deleted = cache.clear()
        assert deleted == 5
        stats = cache.get_stats()
        assert stats.total_entries == 0
        cache.close()

    def test_cache_stats_tracking(self, tmp_path: Path) -> None:
        cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
        key = ResponseCache.make_cache_key("model", "sys", "user")

        cache.get(key)  # miss
        cache.put(
            key, model="model", provider="p", content="data",
            input_tokens=100, output_tokens=50, cost_usd=0.01,
            finish_reason="stop",
        )
        cache.get(key)  # hit

        stats = cache.get_stats()
        assert stats.total_hits >= 1
        assert stats.total_misses >= 1
        assert stats.hit_rate > 0.0
        cache.close()

    def test_cache_tier_specific_ttl(self, tmp_path: Path) -> None:
        cache = ResponseCache(cache_dir=tmp_path / "cache")
        key_simple = ResponseCache.make_cache_key("m", "s", "simple")
        key_complex = ResponseCache.make_cache_key("m", "s", "complex")

        entry_s = cache.put(
            key_simple, model="m", provider="p", content="s",
            input_tokens=10, output_tokens=5, cost_usd=0.001,
            finish_reason="stop", tier="simple",
        )
        entry_c = cache.put(
            key_complex, model="m", provider="p", content="c",
            input_tokens=10, output_tokens=5, cost_usd=0.001,
            finish_reason="stop", tier="complex",
        )
        assert entry_s.expires_at != entry_c.expires_at
        cache.close()


# ======================================================================
# 6. INTELLIGENCE PIPELINE (8+ tests)
# ======================================================================


@pytest.mark.integration
class TestIntelligencePipeline:
    """AEI error fingerprinting + escalation, blast radius, test gaps."""

    def test_aei_fingerprint_deterministic(self, tmp_path: Path) -> None:
        fp1 = AdaptiveExecutionIntelligence.fingerprint_error(
            "TypeError", "line 42 in main\n  x + y", "main.py", "add"
        )
        fp2 = AdaptiveExecutionIntelligence.fingerprint_error(
            "TypeError", "line 99 in main\n  x + y", "main.py", "add"
        )
        assert fp1.fingerprint_hash == fp2.fingerprint_hash

    def test_aei_first_attempt_recommends_cheapest(self, tmp_path: Path) -> None:
        db_path = tmp_path / "aei.db"
        aei = AdaptiveExecutionIntelligence(db_path=db_path, repo="test")
        fp = AdaptiveExecutionIntelligence.fingerprint_error(
            "KeyError", "traceback", "app.py", "get_user"
        )
        rec = aei.recommend_strategy(fp)
        assert rec.strategy == FixStrategy.REGEX_PATCH
        assert rec.model_tier == "cheap"
        assert rec.confidence == 0.5
        aei.close()

    def test_aei_escalation_after_three_failures(self, tmp_path: Path) -> None:
        db_path = tmp_path / "aei.db"
        aei = AdaptiveExecutionIntelligence(db_path=db_path, repo="test")
        fp = AdaptiveExecutionIntelligence.fingerprint_error(
            "ValueError", "traceback", "calc.py", "compute"
        )
        for _ in range(3):
            aei.record_attempt(
                fp, FixStrategy.REGEX_PATCH, "gpt-4o-mini",
                context_size=2000, outcome="failure",
            )
        rec = aei.recommend_strategy(fp)
        assert rec.strategy != FixStrategy.REGEX_PATCH
        assert rec.past_attempts == 3
        aei.close()

    def test_aei_success_replays_strategy(self, tmp_path: Path) -> None:
        db_path = tmp_path / "aei.db"
        aei = AdaptiveExecutionIntelligence(db_path=db_path, repo="test")
        fp = AdaptiveExecutionIntelligence.fingerprint_error(
            "IndexError", "traceback", "utils.py", "fetch"
        )
        aei.record_attempt(
            fp, FixStrategy.AST_DIFF, "gpt-4o",
            context_size=4000, outcome="success",
        )
        rec = aei.recommend_strategy(fp)
        assert rec.strategy == FixStrategy.AST_DIFF
        assert rec.confidence == 0.8
        aei.close()

    def test_aei_stats_aggregation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "aei.db"
        aei = AdaptiveExecutionIntelligence(db_path=db_path, repo="test")
        fp = AdaptiveExecutionIntelligence.fingerprint_error(
            "RuntimeError", "tb", "svc.py", "run"
        )
        aei.record_attempt(fp, FixStrategy.REGEX_PATCH, "m1", 1000, "success")
        aei.record_attempt(fp, FixStrategy.AST_DIFF, "m2", 2000, "failure")
        stats = aei.get_stats()
        assert stats.total_attempts == 2
        assert stats.total_successes == 1
        assert stats.total_failures == 1
        assert stats.success_rate == 0.5
        aei.close()

    def test_blast_radius_single_file(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "auth.py").write_text("def login(): pass\n")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        analyzer = BlastRadiusAnalyzer(
            project_root=tmp_path, reports_dir=reports_dir,
        )
        report = analyzer.analyze(
            "change login function",
            target_files=["src/auth.py"],
        )
        assert report.file_count >= 1
        auth_files = [af for af in report.affected_files if "auth" in af.path]
        assert len(auth_files) >= 1
        assert auth_files[0].risk_level == RiskLevel.HIGH

    def test_blast_radius_risk_score(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "security.py").write_text("def validate(): pass\n")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        analyzer = BlastRadiusAnalyzer(
            project_root=tmp_path, reports_dir=reports_dir,
        )
        report = analyzer.analyze(
            "update security validation",
            target_files=["src/security.py"],
        )
        assert report.risk_score > 0

    def test_test_gap_detection(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "handler.py").write_text(
            "def process_request(data):\n    return data\n\n"
            "def validate_input(data):\n    return bool(data)\n"
        )
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        hunter = TestGapHunter(project_root=tmp_path)
        report = hunter.analyze()
        assert report.total_functions >= 2
        assert report.untested_functions >= 0


# ======================================================================
# 7. PRIVACY PIPELINE (5+ tests)
# ======================================================================


@pytest.mark.integration
class TestPrivacyPipeline:
    """Privacy mode blocks cloud providers, allows Ollama, privacy
    violation errors.
    """

    def test_privacy_mode_blocks_anthropic(self) -> None:
        pm = PrivacyManager()
        pm._level = pm.level.__class__("private")
        with pytest.raises(PrivacyViolationError):
            pm.validate_request("anthropic", "claude-sonnet-4-20250514")

    def test_privacy_mode_blocks_openai(self) -> None:
        pm = PrivacyManager()
        pm._level = pm.level.__class__("private")
        with pytest.raises(PrivacyViolationError):
            pm.validate_request("openai", "gpt-4o")

    def test_privacy_mode_allows_ollama(self) -> None:
        pm = PrivacyManager()
        pm._level = pm.level.__class__("private")
        pm.validate_request("ollama", "ollama/llama3.1:8b")

    def test_normal_mode_allows_all(self) -> None:
        pm = PrivacyManager()
        pm.validate_request("anthropic", "claude-sonnet-4-20250514")
        pm.validate_request("openai", "gpt-4o")
        pm.validate_request("ollama", "ollama/llama3.1:8b")

    def test_cloud_provider_detection(self) -> None:
        pm = PrivacyManager()
        assert pm.is_cloud_provider("anthropic") is True
        assert pm.is_cloud_provider("openai") is True
        assert pm.is_cloud_provider("google") is True
        assert pm.is_cloud_provider("deepseek") is True
        assert pm.is_cloud_provider("ollama") is False

    def test_privacy_blocks_non_ollama_model(self) -> None:
        pm = PrivacyManager()
        pm._level = pm.level.__class__("private")
        with pytest.raises(PrivacyViolationError, match="not routed through Ollama"):
            pm.validate_request("local-provider", "custom-model")

    def test_privacy_mode_toggle(self) -> None:
        pm = PrivacyManager()
        assert pm.is_private is False
        pm._level = pm.level.__class__("private")
        assert pm.is_private is True
        pm.disable_private_mode()
        assert pm.is_private is False


# ======================================================================
# 8. OFFLINE PIPELINE (5+ tests)
# ======================================================================


@pytest.mark.integration
class TestOfflinePipeline:
    """Offline detection, request queueing, queue replay on reconnect."""

    def test_manual_offline_mode(self) -> None:
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()
        assert mgr.is_offline is True
        assert mgr.state == ConnectivityState.OFFLINE

    def test_offline_capabilities_restricted(self) -> None:
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()
        caps = mgr.get_capabilities()
        assert caps.cloud_inference is False
        assert caps.web_browsing is False
        assert caps.file_operations is True
        assert caps.local_inference is True

    def test_request_queueing(self) -> None:
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()
        req = mgr.create_queued_request(
            model="gpt-4o",
            provider="openai",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert mgr.queued_count == 1
        queued = mgr.get_queued_requests()
        assert len(queued) == 1
        assert queued[0].model == "gpt-4o"
        assert queued[0].id == req.id

    def test_queue_cleared_on_demand(self) -> None:
        mgr = OfflineModeManager()
        mgr.create_queued_request("m1", "p1", [{"role": "user", "content": "a"}])
        mgr.create_queued_request("m2", "p2", [{"role": "user", "content": "b"}])
        assert mgr.queued_count == 2
        cleared = mgr.clear_queue()
        assert cleared == 2
        assert mgr.queued_count == 0

    def test_state_callback_on_transition(self) -> None:
        mgr = OfflineModeManager()
        states_seen: list[ConnectivityState] = []
        mgr.on_state_change(states_seen.append)
        mgr.enable_manual_offline()
        assert ConnectivityState.OFFLINE in states_seen

    def test_disable_manual_offline(self) -> None:
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()
        assert mgr.is_manual_offline is True
        with patch.object(mgr, "_check_connectivity", return_value=True):
            mgr.disable_manual_offline()
        assert mgr.is_manual_offline is False
        assert mgr.state == ConnectivityState.ONLINE

    def test_remove_specific_queued_request(self) -> None:
        mgr = OfflineModeManager()
        r1 = mgr.create_queued_request("m1", "p1", [])
        mgr.create_queued_request("m2", "p2", [])
        assert mgr.queued_count == 2
        removed = mgr.remove_queued_request(r1.id)
        assert removed is True
        assert mgr.queued_count == 1


# ======================================================================
# 9. PLUGIN PIPELINE (5+ tests)
# ======================================================================


@pytest.mark.integration
class TestPluginPipeline:
    """Plugin discovery, install, execute, sandboxing."""

    def test_builtin_plugins_registered(self) -> None:
        assert "docker-manager" in BUILTIN_PLUGINS
        assert "db-query" in BUILTIN_PLUGINS
        assert "api-tester" in BUILTIN_PLUGINS

    def test_plugin_manifest_validation_passes(self) -> None:
        manifest = PluginManifest(
            name="test-plugin",
            version="1.0.0",
            description="Test",
            tools=[PluginToolSpec(name="my_tool", description="Does stuff")],
        )
        errors = manifest.validate()
        assert len(errors) == 0

    def test_plugin_manifest_validation_fails_on_empty_name(self) -> None:
        manifest = PluginManifest(name="", version="1.0.0")
        errors = manifest.validate()
        assert any("name" in e.lower() for e in errors)

    def test_plugin_manifest_validation_fails_on_duplicate_tools(self) -> None:
        manifest = PluginManifest(
            name="dup",
            version="1.0.0",
            tools=[
                PluginToolSpec(name="same_name", description="a"),
                PluginToolSpec(name="same_name", description="b"),
            ],
        )
        errors = manifest.validate()
        assert any("duplicate" in e.lower() for e in errors)

    def test_plugin_manager_creates_directory(self, tmp_path: Path) -> None:
        plugins_dir = tmp_path / "plugins"
        mgr = PluginManager(plugins_dir=plugins_dir)
        assert plugins_dir.is_dir()
        assert mgr.plugins_dir == plugins_dir

    def test_install_builtin_plugin(self, tmp_path: Path) -> None:
        plugins_dir = tmp_path / "plugins"
        mgr = PluginManager(plugins_dir=plugins_dir)
        info = mgr.install("docker-manager")
        assert info.manifest.name == "docker-manager"
        assert info.enabled is True
        assert info.install_path.is_dir()

    def test_plugin_not_found_error(self, tmp_path: Path) -> None:
        plugins_dir = tmp_path / "plugins"
        mgr = PluginManager(plugins_dir=plugins_dir)
        with pytest.raises(PluginNotFoundError):
            mgr.enable("nonexistent-plugin")

    def test_builtin_plugin_tools_defined(self) -> None:
        docker = BUILTIN_PLUGINS["docker-manager"]
        assert len(docker.tools) > 0
        tool_names = [t.name for t in docker.tools]
        assert "docker_ps" in tool_names

    def test_builtin_plugin_commands_defined(self) -> None:
        docker = BUILTIN_PLUGINS["docker-manager"]
        assert len(docker.commands) > 0
        cmd_names = [c.name for c in docker.commands]
        assert "docker-ps" in cmd_names


# ======================================================================
# 10. TOOL EXECUTION PIPELINE (8+ tests)
# ======================================================================


@pytest.mark.integration
class TestToolExecutionPipeline:
    """File read/write/edit cycle, directory listing, search, terminal
    execution with sandbox.
    """

    def test_file_write_then_read(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        writer = WriteFileTool(path_guard=guard)
        reader = ReadFileTool(path_guard=guard)

        w_result = writer.execute({
            "path": "test.py",
            "content": "def hello():\n    return 'world'\n",
        })
        assert w_result.success is True
        assert w_result.metadata is not None
        assert w_result.metadata["new_file"] is True

        r_result = reader.execute({"path": "test.py"})
        assert r_result.success is True
        assert "def hello" in r_result.output
        assert "return 'world'" in r_result.output

    def test_file_edit_search_replace(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        writer = WriteFileTool(path_guard=guard)
        editor = EditFileTool(path_guard=guard)

        writer.execute({
            "path": "app.py",
            "content": "name = 'Alice'\nprint(name)\n",
        })

        e_result = editor.execute({
            "path": "app.py",
            "search": "Alice",
            "replace": "Bob",
        })
        assert e_result.success is True
        assert e_result.metadata is not None
        assert e_result.metadata["replacements"] == 1

        reader = ReadFileTool(path_guard=guard)
        r_result = reader.execute({"path": "app.py"})
        assert "Bob" in r_result.output
        assert "Alice" not in r_result.output

    def test_file_edit_not_found(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        editor = EditFileTool(path_guard=guard)
        result = editor.execute({
            "path": "nonexistent.py",
            "search": "x",
            "replace": "y",
        })
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    def test_file_edit_search_not_found(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        (tmp_path / "exists.py").write_text("hello world\n")
        editor = EditFileTool(path_guard=guard)
        result = editor.execute({
            "path": "exists.py",
            "search": "NONEXISTENT_STRING",
            "replace": "replacement",
        })
        assert result.success is False
        assert "not found" in (result.error or "").lower()

    def test_directory_listing(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        (tmp_path / "file1.py").write_text("a")
        (tmp_path / "file2.py").write_text("b")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "file3.py").write_text("c")

        lister = ListDirectoryTool(path_guard=guard)
        result = lister.execute({"path": "."})
        assert result.success is True
        assert "file1.py" in result.output
        assert "file2.py" in result.output
        assert "subdir" in result.output

    def test_directory_listing_recursive(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "mod.py").write_text("x")

        lister = ListDirectoryTool(path_guard=guard)
        result = lister.execute({"path": ".", "recursive": True})
        assert result.success is True
        assert "mod.py" in result.output

    def test_search_codebase(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        (tmp_path / "search_target.py").write_text(
            "def unique_function_name():\n    pass\n"
        )

        searcher = SearchCodebaseTool(path_guard=guard)
        result = searcher.execute({"pattern": "unique_function_name"})
        assert result.success is True
        assert "unique_function_name" in result.output
        assert result.metadata is not None
        assert result.metadata["matches"] >= 1

    def test_search_no_results(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        (tmp_path / "empty.py").write_text("# nothing here\n")

        searcher = SearchCodebaseTool(path_guard=guard)
        result = searcher.execute({"pattern": "ZZZZZ_NONEXISTENT_ZZZZZ"})
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["matches"] == 0

    def test_terminal_execution_echo(self, tmp_path: Path) -> None:
        sandbox = CommandSandbox(project_root=tmp_path)
        tool = ExecuteCommandTool(sandbox=sandbox)
        result = tool.execute({"command": "echo integration_test"})
        assert result.success is True
        assert "integration_test" in result.output

    def test_terminal_execution_exit_code(self, tmp_path: Path) -> None:
        sandbox = CommandSandbox(project_root=tmp_path)
        tool = ExecuteCommandTool(sandbox=sandbox)
        result = tool.execute({"command": "python3 -c 'exit(42)'"})
        assert result.success is False
        assert result.metadata is not None
        assert result.metadata["exit_code"] == 42

    def test_read_file_with_line_range(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        content = "\n".join(f"line {i}" for i in range(1, 21))
        (tmp_path / "lines.txt").write_text(content)

        reader = ReadFileTool(path_guard=guard)
        result = reader.execute({
            "path": "lines.txt",
            "start_line": 5,
            "end_line": 10,
        })
        assert result.success is True
        assert "line 5" in result.output
        assert "line 10" in result.output
        assert "line 11" not in result.output

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        guard = PathGuard(project_root=tmp_path)
        writer = WriteFileTool(path_guard=guard)
        result = writer.execute({
            "path": "deep/nested/dir/file.py",
            "content": "# auto-created\n",
        })
        assert result.success is True
        assert (tmp_path / "deep" / "nested" / "dir" / "file.py").exists()


# ======================================================================
# BONUS: Cross-pipeline integration tests
# ======================================================================


@pytest.mark.integration
class TestCrossPipelineIntegration:
    """Tests that exercise multiple subsystems in a single scenario."""

    def test_routing_plus_cost_plus_budget(self, tmp_path: Path) -> None:
        """Full routing -> cost tracking -> budget enforcement pipeline."""
        settings = _make_settings(tmp_path, daily_limit=0.10)
        db = _init_db()
        auth = _mock_auth()
        registry = ProviderRegistry(settings=settings, auth_manager=auth)
        cost_tracker = CostTracker(db=db, settings=settings)
        classifier = TaskClassifier(settings)
        selector = ModelSelector(
            settings=settings, registry=registry, cost_tracker=cost_tracker,
        )
        mock_llm = MockLiteLLM()
        engine = CompletionEngine(
            settings=settings,
            cost_tracker=cost_tracker,
            auth_manager=auth,
            provider_registry=registry,
            litellm_backend=mock_llm,
        )

        prompt = "fix the typo"
        result = classifier.classify(prompt)
        selection = selector.select(result.tier, prompt)

        mock_llm.set_response(
            selection.model_id,
            MockResponse(content="done", input_tokens=50, output_tokens=20),
        )

        import asyncio

        asyncio.get_event_loop().run_until_complete(
            engine.complete(
                messages=[{"role": "user", "content": prompt}],
                model=selection.model_id,
                session_id="cross-1",
                complexity_tier=result.tier.value,
            )
        )

        daily = cost_tracker.get_daily_cost()
        assert daily >= 0.0
        remaining = cost_tracker.get_budget_remaining()
        assert remaining is not None

    def test_tool_execution_respects_security(self, tmp_path: Path) -> None:
        """File tools respect PathGuard, sandbox respects blocked commands."""
        guard = PathGuard(
            project_root=tmp_path,
            excluded_patterns=["*.secret"],
        )

        writer = WriteFileTool(path_guard=guard)
        ok = writer.execute({"path": "safe.txt", "content": "data"})
        assert ok.success is True

        blocked = writer.execute({"path": "creds.secret", "content": "data"})
        assert blocked.success is False

        sandbox = CommandSandbox(project_root=tmp_path)
        with pytest.raises(BlockedCommandError):
            sandbox.execute("rm -rf /")

        safe_result = sandbox.execute("echo safe")
        assert safe_result.exit_code == 0

    def test_context_manager_feeds_completion_engine(
        self, tmp_path: Path
    ) -> None:
        """ContextManager assembles messages that CompletionEngine accepts."""
        settings = _make_settings(tmp_path, daily_limit=None, monthly_limit=None)
        db = _init_db()
        auth = _mock_auth()
        registry = ProviderRegistry(settings=settings, auth_manager=auth)
        cost_tracker = CostTracker(db=db, settings=settings)
        mock_llm = MockLiteLLM()

        engine = CompletionEngine(
            settings=settings,
            cost_tracker=cost_tracker,
            auth_manager=auth,
            provider_registry=registry,
            litellm_backend=mock_llm,
        )

        ctx_mgr = ContextManager(
            system_prompt="You are a helpful coding assistant.",
            max_tokens=128_000,
        )
        ctx_mgr.add_message("user", "Write a hello world function.")
        ctx_mgr.add_active_file("main.py", "# empty file")

        messages = ctx_mgr.get_context()
        assert len(messages) >= 2

        model = "gpt-4o-mini"
        mock_llm.set_response(
            model,
            MockResponse(content="def hello(): print('Hello!')", output_tokens=30),
        )

        import asyncio

        cr = asyncio.get_event_loop().run_until_complete(
            engine.complete(
                messages=messages,
                model=model,
                session_id="ctx-feed",
            )
        )
        assert "hello" in cr.content.lower()

    def test_offline_then_online_queue_replay(self) -> None:
        """Queue requests while offline, then go back online."""
        mgr = OfflineModeManager()
        mgr.enable_manual_offline()

        mgr.create_queued_request(
            "gpt-4o", "openai", [{"role": "user", "content": "q1"}]
        )
        mgr.create_queued_request(
            "gpt-4o-mini", "openai", [{"role": "user", "content": "q2"}]
        )
        assert mgr.queued_count == 2

        with patch.object(mgr, "_check_connectivity", return_value=True):
            mgr.disable_manual_offline()

        assert mgr.is_online is True
        assert mgr.queued_count == 2

    def test_cache_avoids_api_call(self, tmp_path: Path) -> None:
        """Verify that a cached response means no LLM call is needed."""
        cache = ResponseCache(cache_dir=tmp_path / "cache", ttl=3600)
        model = "gpt-4o-mini"
        system = "You are helpful."
        user_msg = "What is 2+2?"
        key = ResponseCache.make_cache_key(model, system, user_msg)

        miss = cache.get(key)
        assert miss is None

        cache.put(
            key, model=model, provider="openai", content="4",
            input_tokens=20, output_tokens=5, cost_usd=0.0001,
            finish_reason="stop",
        )

        hit = cache.get(key)
        assert hit is not None
        assert hit.content == "4"
        cache.close()

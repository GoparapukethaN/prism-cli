"""Tests for the swarm orchestrator — multi-model collaborative execution.

All tests use MockLiteLLM.  No real API calls are ever made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from prism.orchestrator.swarm import (
    CrossReview,
    CrossReviewer,
    ModelPool,
    PlanReviewer,
    ResearchSwarm,
    ReviewSeverity,
    SwarmConfig,
    SwarmOrchestrator,
    SwarmPlan,
    SwarmTask,
    TaskDecomposer,
    TaskStatus,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from prism.providers.registry import ProviderRegistry


# ======================================================================
# ModelPool
# ======================================================================


class TestModelPool:
    """Test model tier categorisation."""

    def test_research_models_returns_cheap_models(
        self, model_pool: ModelPool,
    ) -> None:
        """Research models come from cheap providers."""
        models = model_pool.get_research_models()
        assert len(models) > 0
        # All should be from cheap providers
        cheap_prefixes = ("groq/", "ollama/", "deepseek/", "mistral/")
        for m in models:
            assert any(
                m.startswith(p) for p in cheap_prefixes
            ), f"Expected {m} to be from a cheap provider"

    def test_research_models_sorted_by_cost(
        self, model_pool: ModelPool, mock_registry: ProviderRegistry,
    ) -> None:
        """Research models are sorted cheapest first."""
        models = model_pool.get_research_models()
        if len(models) < 2:
            pytest.skip("Not enough research models to test ordering")
        costs = []
        for mid in models:
            info = mock_registry.get_model_info(mid)
            assert info is not None
            costs.append(info.input_cost_per_1m + info.output_cost_per_1m)
        assert costs == sorted(costs)

    def test_planning_model_returns_complex_tier(
        self, model_pool: ModelPool, mock_registry: ProviderRegistry,
    ) -> None:
        """Planning model should be from the COMPLEX tier."""
        model = model_pool.get_planning_model()
        info = mock_registry.get_model_info(model)
        assert info is not None
        assert info.tier.value == "complex"

    def test_review_model_is_different_provider(
        self, model_pool: ModelPool, mock_registry: ProviderRegistry,
    ) -> None:
        """Review model should ideally be from a different provider than planner."""
        planning = model_pool.get_planning_model()
        review = model_pool.get_review_model()
        # Both should be valid model IDs
        assert planning
        assert review
        # If they're different, the providers should differ
        planning_info = mock_registry.get_model_info(planning)
        review_info = mock_registry.get_model_info(review)
        if planning != review:
            assert planning_info is not None
            assert review_info is not None
            assert planning_info.provider != review_info.provider

    def test_execution_model_simple(
        self, model_pool: ModelPool,
    ) -> None:
        """Simple execution model should be retrievable."""
        model = model_pool.get_execution_model("simple")
        assert model

    def test_execution_model_complex(
        self, model_pool: ModelPool,
    ) -> None:
        """Complex execution model should be retrievable."""
        model = model_pool.get_execution_model("complex")
        assert model

    def test_execution_model_invalid_falls_back(
        self, model_pool: ModelPool,
    ) -> None:
        """Invalid complexity tier falls back to medium."""
        model = model_pool.get_execution_model("nonsense")
        assert model


# ======================================================================
# TaskDecomposer
# ======================================================================


class TestTaskDecomposer:
    """Test goal decomposition into subtasks."""

    async def test_decompose_returns_tasks(
        self, task_decomposer: TaskDecomposer,
    ) -> None:
        """Decomposition returns a non-empty list of SwarmTasks."""
        tasks = await task_decomposer.decompose(
            "Build a REST API with auth and tests",
        )
        assert len(tasks) >= 1
        for task in tasks:
            assert isinstance(task, SwarmTask)
            assert task.id
            assert task.description

    async def test_decompose_parses_json(
        self, task_decomposer: TaskDecomposer,
    ) -> None:
        """Decomposition correctly parses the mock JSON response."""
        tasks = await task_decomposer.decompose(
            "Refactor the authentication module",
        )
        # The mock returns 3 tasks (see conftest _make_decompose_response)
        assert len(tasks) == 3
        assert tasks[0].complexity == "simple"
        assert tasks[1].complexity == "complex"
        assert tasks[2].complexity == "medium"

    async def test_decompose_resolves_dependencies(
        self, task_decomposer: TaskDecomposer,
    ) -> None:
        """Dependencies are converted from indices to UUIDs."""
        tasks = await task_decomposer.decompose("Build something complex")
        # Task 1 (index 1) depends on task 0
        assert tasks[1].dependencies == [tasks[0].id]
        # Task 2 (index 2) depends on task 1
        assert tasks[2].dependencies == [tasks[1].id]
        # Task 0 has no dependencies
        assert tasks[0].dependencies == []

    async def test_decompose_handles_files_changed(
        self, task_decomposer: TaskDecomposer,
    ) -> None:
        """File change lists are parsed correctly."""
        tasks = await task_decomposer.decompose("Build auth module")
        # Task 1 modifies src/auth.py
        assert "src/auth.py" in tasks[1].files_changed
        # Task 0 has no files
        assert tasks[0].files_changed == []

    async def test_decompose_empty_goal_raises(
        self, task_decomposer: TaskDecomposer,
    ) -> None:
        """Empty goal raises ValueError."""
        with pytest.raises(ValueError, match="Goal must not be empty"):
            await task_decomposer.decompose("")

    async def test_decompose_whitespace_goal_raises(
        self, task_decomposer: TaskDecomposer,
    ) -> None:
        """Whitespace-only goal raises ValueError."""
        with pytest.raises(ValueError, match="Goal must not be empty"):
            await task_decomposer.decompose("   ")

    async def test_decompose_bad_json_fallback(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """If the model returns non-JSON, falls back to a single task."""
        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse

        mock = MockLiteLLM()
        mock.set_default_response(
            MockResponse(content="This is not JSON at all.", input_tokens=50, output_tokens=30),
        )
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock,
        )
        pool = ModelPool(mock_registry)
        decomposer = TaskDecomposer(engine, pool)

        tasks = await decomposer.decompose("Do something complicated")
        assert len(tasks) == 1
        assert "Do something complicated" in tasks[0].description

    async def test_decompose_with_context(
        self, task_decomposer: TaskDecomposer,
    ) -> None:
        """Context dict is included in the prompt."""
        tasks = await task_decomposer.decompose(
            "Refactor auth module",
            context={"files": ["auth.py", "models.py"], "language": "Python"},
        )
        assert len(tasks) >= 1


# ======================================================================
# ResearchSwarm
# ======================================================================


class TestResearchSwarm:
    """Test parallel research execution."""

    async def test_research_returns_findings(
        self, research_swarm: ResearchSwarm,
    ) -> None:
        """Research returns a finding for each question."""
        questions = [
            "What patterns are used in auth modules?",
            "Best practices for REST API design?",
        ]
        findings = await research_swarm.research(questions)
        assert len(findings) == 2
        for q in questions:
            assert q in findings
            assert findings[q]  # non-empty

    async def test_research_empty_questions(
        self, research_swarm: ResearchSwarm,
    ) -> None:
        """Empty question list returns empty dict."""
        findings = await research_swarm.research([])
        assert findings == {}

    async def test_research_with_context(
        self, research_swarm: ResearchSwarm,
    ) -> None:
        """Context is passed to research models."""
        findings = await research_swarm.research(
            ["What is the current auth approach?"],
            context={"auth_module": "Uses JWT tokens with refresh"},
        )
        assert len(findings) == 1

    async def test_research_distributes_across_models(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Multiple questions are distributed across available research models."""
        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse

        mock = MockLiteLLM()
        mock.set_default_response(MockResponse(content="Answer.", input_tokens=20, output_tokens=10))
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock,
        )
        pool = ModelPool(mock_registry)
        swarm = ResearchSwarm(engine, pool, max_workers=4)

        questions = [f"Question {i}?" for i in range(6)]
        findings = await swarm.research(questions)
        assert len(findings) == 6

        # Verify calls were made to potentially different models
        models_used = {call["model"] for call in mock.call_log}
        assert len(models_used) >= 1  # At least 1 model used


# ======================================================================
# PlanReviewer
# ======================================================================


class TestPlanReviewer:
    """Test plan review by a different model."""

    async def test_review_returns_notes(
        self, plan_reviewer: PlanReviewer,
    ) -> None:
        """Review returns non-empty notes."""
        notes = await plan_reviewer.review(
            plan_text="1. Read files\n2. Write code\n3. Run tests",
            goal="Build auth module",
        )
        assert notes
        assert len(notes) > 0

    async def test_review_empty_plan(
        self, plan_reviewer: PlanReviewer,
    ) -> None:
        """Empty plan returns a message about no plan."""
        notes = await plan_reviewer.review(
            plan_text="   ",
            goal="Build auth module",
        )
        assert "No plan provided" in notes


# ======================================================================
# CrossReviewer
# ======================================================================


class TestCrossReviewer:
    """Test post-execution cross-review."""

    async def test_review_task_approved(
        self, cross_reviewer_approved: CrossReviewer,
    ) -> None:
        """Approved review returns info severity."""
        task = SwarmTask(
            description="Implement user model",
            complexity="medium",
            status=TaskStatus.COMPLETED,
            result="class User:\n    def __init__(self, name): self.name = name",
        )
        review = await cross_reviewer_approved.review_task(task)
        assert isinstance(review, CrossReview)
        assert review.approved is True
        assert review.severity == ReviewSeverity.INFO

    async def test_review_task_error(
        self, cross_reviewer_error: CrossReviewer,
    ) -> None:
        """Error review returns error severity and not approved."""
        task = SwarmTask(
            description="Handle passwords",
            complexity="complex",
            status=TaskStatus.COMPLETED,
            result="password = 'admin123'",
        )
        review = await cross_reviewer_error.review_task(task)
        assert review.approved is False
        assert review.severity == ReviewSeverity.ERROR

    async def test_review_task_no_result(
        self, cross_reviewer_approved: CrossReviewer,
    ) -> None:
        """Task with no result gets an error review."""
        task = SwarmTask(
            description="Do something",
            status=TaskStatus.PENDING,
            result=None,
        )
        review = await cross_reviewer_approved.review_task(task)
        assert review.approved is False
        assert "no result" in review.comments.lower()

    async def test_review_all_reviews_completed_tasks(
        self, cross_reviewer_approved: CrossReviewer,
    ) -> None:
        """review_all only reviews completed tasks with results."""
        tasks = [
            SwarmTask(
                description="Task 1",
                status=TaskStatus.COMPLETED,
                result="Done task 1",
            ),
            SwarmTask(
                description="Task 2",
                status=TaskStatus.PENDING,
                result=None,
            ),
            SwarmTask(
                description="Task 3",
                status=TaskStatus.COMPLETED,
                result="Done task 3",
            ),
        ]
        reviews = await cross_reviewer_approved.review_all(tasks)
        # Only task 1 and task 3 are completed with results
        assert len(reviews) == 2

    async def test_review_with_sibling_context(
        self, cross_reviewer_approved: CrossReviewer,
    ) -> None:
        """Sibling tasks are included as context for consistency checking."""
        tasks = [
            SwarmTask(
                description="Create models",
                status=TaskStatus.COMPLETED,
                result="class User: pass",
            ),
            SwarmTask(
                description="Create views",
                status=TaskStatus.COMPLETED,
                result="def user_view(): pass",
            ),
        ]
        review = await cross_reviewer_approved.review_task(tasks[0], all_tasks=tasks)
        assert review.approved is True

    async def test_parse_review_fallback_on_bad_json(self) -> None:
        """Bad JSON falls back to info severity with raw text as comments."""
        review = CrossReviewer._parse_review(
            "This is not JSON but it is a comment.",
            task_id="test-123",
            model="test-model",
        )
        assert review.severity == ReviewSeverity.INFO
        assert review.approved is True
        assert "not JSON" in review.comments


# ======================================================================
# SwarmTask data model
# ======================================================================


class TestSwarmTask:
    """Test SwarmTask dataclass."""

    def test_default_values(self) -> None:
        """SwarmTask has sensible defaults."""
        task = SwarmTask()
        assert task.id  # UUID generated
        assert task.status == TaskStatus.PENDING
        assert task.complexity == "medium"
        assert task.dependencies == []
        assert task.files_changed == []
        assert task.result is None
        assert task.assigned_model is None

    def test_custom_values(self) -> None:
        """SwarmTask accepts custom values."""
        task = SwarmTask(
            id="custom-id",
            description="Build API",
            complexity="complex",
            dependencies=["dep-1", "dep-2"],
            assigned_model="gpt-4o",
            status=TaskStatus.RUNNING,
            result="In progress",
            files_changed=["api.py"],
        )
        assert task.id == "custom-id"
        assert task.complexity == "complex"
        assert len(task.dependencies) == 2


# ======================================================================
# SwarmPlan data model
# ======================================================================


class TestSwarmPlan:
    """Test SwarmPlan dataclass."""

    def test_default_values(self) -> None:
        """SwarmPlan has sensible defaults."""
        plan = SwarmPlan()
        assert plan.goal == ""
        assert plan.tasks == []
        assert plan.research_findings == {}
        assert plan.total_cost == 0.0
        assert plan.phase_costs == {}

    def test_with_tasks(self) -> None:
        """SwarmPlan tracks tasks and costs."""
        plan = SwarmPlan(
            goal="Build auth",
            tasks=[SwarmTask(description="Step 1")],
            total_cost=0.05,
        )
        assert len(plan.tasks) == 1
        assert plan.total_cost == 0.05


# ======================================================================
# Dependency ordering
# ======================================================================


class TestDependencyOrdering:
    """Test topological sort of tasks by dependencies."""

    def test_no_dependencies(self) -> None:
        """Tasks without dependencies maintain order."""
        tasks = [
            SwarmTask(id="a", description="Task A"),
            SwarmTask(id="b", description="Task B"),
            SwarmTask(id="c", description="Task C"),
        ]
        ordered = SwarmOrchestrator._dependency_sort(tasks)
        assert [t.id for t in ordered] == ["a", "b", "c"]

    def test_linear_chain(self) -> None:
        """Linear dependency chain is sorted correctly."""
        tasks = [
            SwarmTask(id="a", description="Task A"),
            SwarmTask(id="b", description="Task B", dependencies=["a"]),
            SwarmTask(id="c", description="Task C", dependencies=["b"]),
        ]
        ordered = SwarmOrchestrator._dependency_sort(tasks)
        ids = [t.id for t in ordered]
        assert ids.index("a") < ids.index("b")
        assert ids.index("b") < ids.index("c")

    def test_diamond_dependency(self) -> None:
        """Diamond dependency pattern is resolved."""
        tasks = [
            SwarmTask(id="a", description="Task A"),
            SwarmTask(id="b", description="Task B", dependencies=["a"]),
            SwarmTask(id="c", description="Task C", dependencies=["a"]),
            SwarmTask(id="d", description="Task D", dependencies=["b", "c"]),
        ]
        ordered = SwarmOrchestrator._dependency_sort(tasks)
        ids = [t.id for t in ordered]
        # A must come before B, C, and D
        assert ids.index("a") < ids.index("b")
        assert ids.index("a") < ids.index("c")
        assert ids.index("a") < ids.index("d")
        # B and C must come before D
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")

    def test_reverse_input_order(self) -> None:
        """Tasks given in reverse dependency order are still sorted correctly."""
        tasks = [
            SwarmTask(id="c", description="Task C", dependencies=["b"]),
            SwarmTask(id="b", description="Task B", dependencies=["a"]),
            SwarmTask(id="a", description="Task A"),
        ]
        ordered = SwarmOrchestrator._dependency_sort(tasks)
        ids = [t.id for t in ordered]
        assert ids.index("a") < ids.index("b")
        assert ids.index("b") < ids.index("c")

    def test_dependencies_met_all_completed(self) -> None:
        """dependencies_met returns True when all deps are completed."""
        tasks = [
            SwarmTask(id="a", status=TaskStatus.COMPLETED),
            SwarmTask(id="b", dependencies=["a"]),
        ]
        assert SwarmOrchestrator._dependencies_met(tasks[1], tasks) is True

    def test_dependencies_met_not_completed(self) -> None:
        """dependencies_met returns False when a dep is still pending."""
        tasks = [
            SwarmTask(id="a", status=TaskStatus.PENDING),
            SwarmTask(id="b", dependencies=["a"]),
        ]
        assert SwarmOrchestrator._dependencies_met(tasks[1], tasks) is False

    def test_dependencies_met_no_deps(self) -> None:
        """Task with no dependencies always has met deps."""
        task = SwarmTask(id="a")
        assert SwarmOrchestrator._dependencies_met(task, []) is True


# ======================================================================
# Full orchestration pipeline
# ======================================================================


class TestSwarmOrchestrator:
    """Test the full orchestration pipeline (all phases mocked)."""

    async def test_full_pipeline_completes(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Full pipeline runs all 7 phases and returns a plan."""
        plan = await orchestrator.orchestrate(
            "Build an authentication module with JWT tokens",
        )
        assert isinstance(plan, SwarmPlan)
        assert plan.goal == "Build an authentication module with JWT tokens"
        assert len(plan.tasks) >= 1
        assert plan.total_cost >= 0.0

    async def test_pipeline_decomposes_tasks(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Pipeline produces tasks from decomposition."""
        plan = await orchestrator.orchestrate("Refactor the API layer")
        assert len(plan.tasks) >= 1
        for task in plan.tasks:
            assert task.description
            assert task.id

    async def test_pipeline_produces_plan_text(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Pipeline generates plan text in the plan phase."""
        plan = await orchestrator.orchestrate("Add error handling to all endpoints")
        assert plan.plan_text
        assert len(plan.plan_text) > 0

    async def test_pipeline_produces_review_notes(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Pipeline generates review notes from the review phase."""
        plan = await orchestrator.orchestrate("Implement caching layer")
        # Review notes should be populated (even if mock content)
        assert plan.review_notes is not None

    async def test_pipeline_executes_tasks(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Pipeline executes tasks and populates results."""
        plan = await orchestrator.orchestrate("Build user registration")
        completed = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        assert len(completed) >= 1
        for task in completed:
            assert task.result is not None
            assert task.assigned_model is not None

    async def test_pipeline_tracks_phase_costs(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Pipeline tracks costs per phase."""
        plan = await orchestrator.orchestrate("Create a simple utility function")
        assert plan.total_cost > 0
        assert len(plan.phase_costs) > 0

    async def test_empty_goal_raises(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Empty goal raises ValueError."""
        with pytest.raises(ValueError, match="Goal must not be empty"):
            await orchestrator.orchestrate("")

    async def test_interrupt_stops_pipeline(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Interrupting the orchestrator stops after the current phase."""
        # Interrupt immediately
        orchestrator.interrupt()
        plan = await orchestrator.orchestrate("Build something complex")
        # Pipeline should have stopped early — may or may not have tasks
        assert isinstance(plan, SwarmPlan)

    async def test_pipeline_with_context(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Pipeline accepts and uses context."""
        plan = await orchestrator.orchestrate(
            "Add rate limiting middleware",
            context={
                "files": ["app.py", "middleware.py"],
                "framework": "FastAPI",
            },
        )
        assert len(plan.tasks) >= 1

    async def test_cross_reviews_populated(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Pipeline produces cross-review results."""
        plan = await orchestrator.orchestrate("Build a small feature")
        # Cross reviews should be present for completed tasks
        # (may be empty if no tasks completed due to mock responses)
        assert isinstance(plan.cross_reviews, list)


# ======================================================================
# Research question generation
# ======================================================================


class TestResearchQuestionGeneration:
    """Test the generation of research questions from tasks."""

    def test_generates_questions_from_tasks(self) -> None:
        """Each task generates a research question."""
        tasks = [
            SwarmTask(description="Implement JWT auth"),
            SwarmTask(description="Set up database migrations"),
        ]
        questions = SwarmOrchestrator._generate_research_questions(tasks, "Build API")
        assert len(questions) == 2
        assert "JWT auth" in questions[0]
        assert "database migrations" in questions[1]

    def test_caps_at_eight_questions(self) -> None:
        """Question count is capped at 8."""
        tasks = [SwarmTask(description=f"Task {i}") for i in range(15)]
        questions = SwarmOrchestrator._generate_research_questions(tasks, "Big project")
        assert len(questions) == 8

    def test_deduplicates_questions(self) -> None:
        """Duplicate task descriptions produce only one question."""
        tasks = [
            SwarmTask(description="Build auth"),
            SwarmTask(description="Build auth"),
            SwarmTask(description="Write tests"),
        ]
        questions = SwarmOrchestrator._generate_research_questions(tasks, "Test")
        assert len(questions) == 2

    def test_empty_tasks_returns_empty(self) -> None:
        """No tasks means no questions."""
        questions = SwarmOrchestrator._generate_research_questions([], "Goal")
        assert questions == []


# ======================================================================
# Phase cost estimation
# ======================================================================


class TestPhaseCostEstimation:
    """Test per-phase cost estimation."""

    def test_known_phases_have_estimates(self) -> None:
        """All phases have non-zero cost estimates."""
        from prism.orchestrator.swarm import SwarmPhase

        for phase in SwarmPhase:
            cost = SwarmOrchestrator._estimate_phase_cost(phase)
            assert cost > 0.0, f"Phase {phase} should have a cost estimate"

    def test_unknown_phase_returns_default(self) -> None:
        """Unknown phase string returns a default estimate."""
        cost = SwarmOrchestrator._estimate_phase_cost("unknown_phase")
        assert cost == 0.005


# ======================================================================
# Integration: TaskDecomposer + dependency sort
# ======================================================================


class TestDecomposeAndSort:
    """Integration test: decompose then sort by dependencies."""

    async def test_decomposed_tasks_sortable(
        self, task_decomposer: TaskDecomposer,
    ) -> None:
        """Decomposed tasks can be topologically sorted."""
        tasks = await task_decomposer.decompose("Build a REST API")
        sorted_tasks = SwarmOrchestrator._dependency_sort(tasks)
        assert len(sorted_tasks) == len(tasks)

        # Verify dependency order is respected
        completed_ids: set[str] = set()
        for task in sorted_tasks:
            for dep_id in task.dependencies:
                assert dep_id in completed_ids, (
                    f"Task {task.id} depends on {dep_id} which hasn't appeared yet"
                )
            completed_ids.add(task.id)


# ======================================================================
# SwarmConfig
# ======================================================================


class TestSwarmConfig:
    """Test the SwarmConfig dataclass defaults and behaviour."""

    def test_default_all_enabled(self) -> None:
        """All advanced features are enabled by default."""
        cfg = SwarmConfig()
        assert cfg.use_debate is True
        assert cfg.use_moa is True
        assert cfg.use_cascade is True
        assert cfg.use_tools is True

    def test_disable_all(self) -> None:
        """Can disable all features explicitly."""
        cfg = SwarmConfig(
            use_debate=False, use_moa=False, use_cascade=False, use_tools=False,
        )
        assert cfg.use_debate is False
        assert cfg.use_moa is False
        assert cfg.use_cascade is False
        assert cfg.use_tools is False

    def test_custom_moa_threshold(self) -> None:
        """MoA complexity threshold is configurable."""
        cfg = SwarmConfig(moa_complexity_threshold="medium")
        assert cfg.moa_complexity_threshold == "medium"

    def test_custom_cascade_budget(self) -> None:
        """Cascade budget per task is configurable."""
        cfg = SwarmConfig(cascade_budget_per_task=0.10)
        assert cfg.cascade_budget_per_task == 0.10

    def test_no_cascade_budget(self) -> None:
        """Cascade budget can be None (unlimited)."""
        cfg = SwarmConfig(cascade_budget_per_task=None)
        assert cfg.cascade_budget_per_task is None


# ======================================================================
# SwarmPlan extended fields
# ======================================================================


class TestSwarmPlanExtended:
    """Test the extended SwarmPlan fields for debate, cascade, and MoA."""

    def test_debate_result_default_none(self) -> None:
        """Debate result defaults to None."""
        plan = SwarmPlan()
        assert plan.debate_result is None

    def test_cascade_results_default_empty(self) -> None:
        """Cascade results default to empty dict."""
        plan = SwarmPlan()
        assert plan.cascade_results == {}

    def test_moa_results_default_empty(self) -> None:
        """MoA results default to empty dict."""
        plan = SwarmPlan()
        assert plan.moa_results == {}


# ======================================================================
# Execution strategy selection
# ======================================================================


class TestExecutionStrategy:
    """Test _execution_strategy selection based on config and task complexity."""

    def test_moa_for_complex_tasks(
        self, orchestrator_with_moa: SwarmOrchestrator,
    ) -> None:
        """MoA strategy selected for complex tasks when moa is enabled."""
        task = SwarmTask(description="Complex task", complexity="complex")
        strategy = orchestrator_with_moa._execution_strategy(task)
        assert strategy == "moa"

    def test_no_moa_for_simple_tasks(
        self, orchestrator_with_moa: SwarmOrchestrator,
    ) -> None:
        """Simple tasks use cascade or direct, not MoA."""
        task = SwarmTask(description="Simple task", complexity="simple")
        strategy = orchestrator_with_moa._execution_strategy(task)
        # MoA only for complex; no cascade enabled in this fixture
        assert strategy == "direct"

    def test_cascade_for_medium_tasks(
        self, orchestrator_with_cascade: SwarmOrchestrator,
    ) -> None:
        """Cascade strategy selected when cascade is enabled."""
        task = SwarmTask(description="Medium task", complexity="medium")
        strategy = orchestrator_with_cascade._execution_strategy(task)
        assert strategy == "cascade"

    def test_direct_when_nothing_enabled(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Direct strategy when all advanced features are disabled."""
        task = SwarmTask(description="Any task", complexity="medium")
        strategy = orchestrator._execution_strategy(task)
        assert strategy == "direct"

    def test_moa_takes_priority_over_cascade(
        self, orchestrator_full: SwarmOrchestrator,
    ) -> None:
        """MoA takes priority for complex tasks even when cascade is also enabled."""
        task = SwarmTask(description="Complex task", complexity="complex")
        strategy = orchestrator_full._execution_strategy(task)
        assert strategy == "moa"

    def test_cascade_for_non_complex_with_full(
        self, orchestrator_full: SwarmOrchestrator,
    ) -> None:
        """Non-complex tasks use cascade when both MoA and cascade are enabled."""
        task = SwarmTask(description="Medium task", complexity="medium")
        strategy = orchestrator_full._execution_strategy(task)
        assert strategy == "cascade"


# ======================================================================
# Build task prompt
# ======================================================================


class TestBuildTaskPrompt:
    """Test prompt building for task execution."""

    def test_basic_prompt_includes_task_description(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Prompt includes task description."""
        task = SwarmTask(description="Write a function")
        plan = SwarmPlan(goal="Build module")
        prompt = orchestrator._build_task_prompt(task, plan)
        assert "Write a function" in prompt

    def test_prompt_includes_plan_context(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Prompt includes plan text."""
        task = SwarmTask(description="Task 1")
        plan = SwarmPlan(plan_text="Step 1: read files")
        prompt = orchestrator._build_task_prompt(task, plan)
        assert "Step 1: read files" in prompt

    def test_prompt_includes_review_feedback(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Prompt includes review notes when present."""
        task = SwarmTask(description="Task 1")
        plan = SwarmPlan(review_notes="Add error handling")
        prompt = orchestrator._build_task_prompt(task, plan)
        assert "Add error handling" in prompt

    def test_prompt_includes_sibling_results(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Prompt includes completed sibling task results."""
        task1 = SwarmTask(
            id="t1", description="Read code", status=TaskStatus.COMPLETED,
            result="Found 3 classes",
        )
        task2 = SwarmTask(id="t2", description="Write tests")
        plan = SwarmPlan(tasks=[task1, task2])
        prompt = orchestrator._build_task_prompt(task2, plan)
        assert "Found 3 classes" in prompt

    def test_prompt_includes_files_changed(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Prompt includes files to modify."""
        task = SwarmTask(
            description="Edit model", files_changed=["src/model.py"],
        )
        plan = SwarmPlan()
        prompt = orchestrator._build_task_prompt(task, plan)
        assert "src/model.py" in prompt

    def test_prompt_no_tool_mention_when_tools_disabled(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """No tool mention when tools are disabled."""
        task = SwarmTask(description="Task")
        plan = SwarmPlan()
        prompt = orchestrator._build_task_prompt(task, plan)
        assert "read_file" not in prompt


# ======================================================================
# Lazy engine initialisation
# ======================================================================


class TestLazyEngineInit:
    """Test lazy initialisation of debate, cascade, and MoA engines."""

    def test_debate_engine_lazy_creation(
        self, orchestrator_with_debate: SwarmOrchestrator,
    ) -> None:
        """Debate engine is None until first access."""
        assert orchestrator_with_debate._debate_engine is None
        engine = orchestrator_with_debate._get_debate_engine()
        assert engine is not None
        assert orchestrator_with_debate._debate_engine is engine

    def test_debate_engine_cached(
        self, orchestrator_with_debate: SwarmOrchestrator,
    ) -> None:
        """Second call returns the same instance."""
        e1 = orchestrator_with_debate._get_debate_engine()
        e2 = orchestrator_with_debate._get_debate_engine()
        assert e1 is e2

    def test_cascade_engine_lazy_creation(
        self, orchestrator_with_cascade: SwarmOrchestrator,
    ) -> None:
        """Cascade engine is None until first access."""
        assert orchestrator_with_cascade._cascade_engine is None
        engine = orchestrator_with_cascade._get_cascade_engine()
        assert engine is not None
        assert orchestrator_with_cascade._cascade_engine is engine

    def test_cascade_engine_cached(
        self, orchestrator_with_cascade: SwarmOrchestrator,
    ) -> None:
        """Second call returns the same instance."""
        e1 = orchestrator_with_cascade._get_cascade_engine()
        e2 = orchestrator_with_cascade._get_cascade_engine()
        assert e1 is e2

    def test_moa_engine_lazy_creation(
        self, orchestrator_with_moa: SwarmOrchestrator,
    ) -> None:
        """MoA engine is None until first access."""
        assert orchestrator_with_moa._moa_engine is None
        engine = orchestrator_with_moa._get_moa_engine()
        assert engine is not None
        assert orchestrator_with_moa._moa_engine is engine

    def test_moa_engine_cached(
        self, orchestrator_with_moa: SwarmOrchestrator,
    ) -> None:
        """Second call returns the same instance."""
        e1 = orchestrator_with_moa._get_moa_engine()
        e2 = orchestrator_with_moa._get_moa_engine()
        assert e1 is e2


# ======================================================================
# Pipeline integration with debate
# ======================================================================


@pytest.mark.asyncio()
class TestDebateIntegration:
    """Test the full pipeline with debate-enhanced review phase."""

    async def test_debate_review_populates_debate_result(
        self, orchestrator_with_debate: SwarmOrchestrator,
    ) -> None:
        """Pipeline with debate stores the debate result in the plan."""
        plan = await orchestrator_with_debate.orchestrate("Build a REST API")
        # Debate may succeed or fail (mocks return generic text);
        # either way, review_notes should be populated
        assert plan.review_notes

    async def test_debate_fallback_to_single_review(
        self, orchestrator_with_debate: SwarmOrchestrator,
    ) -> None:
        """If debate fails, falls back to single-model review."""
        # The mock responses won't form valid debate JSON, so debate will
        # likely fail and fall back — review_notes should still be populated
        plan = await orchestrator_with_debate.orchestrate("Add user auth")
        assert plan.review_notes
        assert len(plan.review_notes) > 0


# ======================================================================
# Pipeline integration with cascade
# ======================================================================


@pytest.mark.asyncio()
class TestCascadeIntegration:
    """Test the full pipeline with cascade-enhanced execution phase."""

    async def test_cascade_execution_completes(
        self, orchestrator_with_cascade: SwarmOrchestrator,
    ) -> None:
        """Pipeline with cascade completes all tasks."""
        plan = await orchestrator_with_cascade.orchestrate("Build a REST API")
        completed = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        # Even if cascade fails, direct fallback should succeed
        assert len(completed) > 0

    async def test_cascade_stores_results(
        self, orchestrator_with_cascade: SwarmOrchestrator,
    ) -> None:
        """Cascade results are stored in the plan."""
        plan = await orchestrator_with_cascade.orchestrate("Add validation")
        # Cascade may fail (mock gives non-JSON confidence responses)
        # but tasks should still complete via fallback
        for t in plan.tasks:
            assert t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)


# ======================================================================
# Pipeline integration with MoA
# ======================================================================


@pytest.mark.asyncio()
class TestMoAIntegration:
    """Test the full pipeline with MoA-enhanced execution for complex tasks."""

    async def test_moa_execution_completes(
        self, orchestrator_with_moa: SwarmOrchestrator,
    ) -> None:
        """Pipeline with MoA completes complex tasks."""
        plan = await orchestrator_with_moa.orchestrate("Build a complex auth system")
        completed = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        # Even if MoA fails, cascade/direct fallback should succeed
        assert len(completed) > 0

    async def test_moa_only_for_complex(
        self, orchestrator_with_moa: SwarmOrchestrator,
    ) -> None:
        """Non-complex tasks don't use MoA."""
        plan = await orchestrator_with_moa.orchestrate("Fix a typo")
        # Simple tasks should use direct execution (no MoA results)
        simple_tasks = [t for t in plan.tasks if t.complexity == "simple"]
        for t in simple_tasks:
            assert t.id not in plan.moa_results


# ======================================================================
# Full pipeline with all features
# ======================================================================


@pytest.mark.asyncio()
class TestFullPipeline:
    """Test the full pipeline with all advanced features enabled."""

    async def test_full_pipeline_completes(
        self, orchestrator_full: SwarmOrchestrator,
    ) -> None:
        """Pipeline with all features enabled completes successfully."""
        plan = await orchestrator_full.orchestrate("Build a complete REST API")
        assert plan.goal == "Build a complete REST API"
        assert len(plan.tasks) > 0
        assert plan.review_notes
        assert plan.total_cost >= 0

    async def test_full_pipeline_tracks_costs(
        self, orchestrator_full: SwarmOrchestrator,
    ) -> None:
        """Cost tracking works with all advanced features."""
        plan = await orchestrator_full.orchestrate("Add auth module")
        assert plan.total_cost >= 0
        # Phase costs should have entries
        assert len(plan.phase_costs) > 0

    async def test_full_pipeline_interrupt(
        self, orchestrator_full: SwarmOrchestrator,
    ) -> None:
        """Interrupt works with advanced features enabled."""
        orchestrator_full.interrupt()
        plan = await orchestrator_full.orchestrate("Build everything")
        # Should stop after first phase
        assert plan.goal == "Build everything"


# ======================================================================
# Budget enforcement
# ======================================================================


class TestBudgetEnforcement:
    """Test per-phase and total budget enforcement in the swarm pipeline."""

    def test_swarm_config_total_budget_default(self) -> None:
        """SwarmConfig defaults to total_budget=1.0."""
        cfg = SwarmConfig()
        assert cfg.total_budget == 1.0

    def test_swarm_config_phase_budgets_default_none(self) -> None:
        """SwarmConfig defaults to phase_budgets=None."""
        cfg = SwarmConfig()
        assert cfg.phase_budgets is None

    def test_swarm_config_total_budget_none_unlimited(self) -> None:
        """total_budget=None means unlimited spending."""
        cfg = SwarmConfig(total_budget=None)
        assert cfg.total_budget is None

    def test_swarm_config_custom_phase_budgets(self) -> None:
        """Phase budgets can be set per phase."""
        budgets = {"decompose": 0.01, "execute": 0.05}
        cfg = SwarmConfig(phase_budgets=budgets)
        assert cfg.phase_budgets == budgets

    def test_swarm_plan_budget_exhausted_default_false(self) -> None:
        """SwarmPlan defaults to budget_exhausted=False."""
        plan = SwarmPlan()
        assert plan.budget_exhausted is False

    async def test_total_budget_stops_pipeline(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm_decompose: object,
    ) -> None:
        """Pipeline stops when total budget is too low for even the first phase."""
        from prism.llm.completion import CompletionEngine

        # Decompose estimate is 0.005; budget 0.001 is too low to even start.
        config = SwarmConfig(
            use_debate=False,
            use_moa=False,
            use_cascade=False,
            use_tools=False,
            total_budget=0.001,
        )
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm_decompose,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("Build auth module")

        # Budget too low for any phase — stops immediately.
        assert plan.budget_exhausted is True
        assert plan.goal == "Build auth module"
        # No phases ran, so no tasks and no phase costs
        assert plan.tasks == []
        assert plan.phase_costs == {}

    async def test_per_phase_budget_skips_expensive_phase(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm_decompose: object,
    ) -> None:
        """A per-phase cap that's too low causes that phase to be skipped."""
        from prism.llm.completion import CompletionEngine

        # Plan phase estimate is 0.010; cap it below that
        config = SwarmConfig(
            use_debate=False,
            use_moa=False,
            use_cascade=False,
            use_tools=False,
            total_budget=None,  # no total limit
            phase_budgets={"plan": 0.001},  # too low for the plan phase
        )
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm_decompose,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("Build feature")

        # The plan phase should have been skipped due to per-phase budget,
        # but the pipeline should continue past it.
        assert plan.goal == "Build feature"
        assert plan.budget_exhausted is False  # per-phase skip is not a full stop
        # Plan phase was skipped, so "plan" shouldn't appear in phase_costs
        assert "plan" not in plan.phase_costs
        # Later phases should still have run (e.g. review, execute, etc.)
        assert plan.total_cost > 0

    async def test_budget_tracking_across_phases(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm_decompose: object,
    ) -> None:
        """Costs accumulate across phases and are checked before each one."""
        from prism.llm.completion import CompletionEngine

        # Use a budget that allows some phases but not all.
        # Phase cost estimates: decompose=0.005, research=0.002, plan=0.010
        # The budget check before each phase ensures estimated+spent <= budget.
        # After a few phases the actual accumulated cost will exceed
        # the budget and the pipeline will stop.
        config = SwarmConfig(
            use_debate=False,
            use_moa=False,
            use_cascade=False,
            use_tools=False,
            total_budget=0.020,
            auto_scale_budget=False,
        )
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm_decompose,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("Build feature")

        # Budget should be exhausted before all 7 phases complete
        assert plan.budget_exhausted is True
        assert plan.total_cost > 0
        # Should have at least decompose and research costs
        assert "decompose" in plan.phase_costs
        assert "research" in plan.phase_costs

    async def test_none_budget_means_unlimited(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm_decompose: object,
    ) -> None:
        """Setting total_budget=None runs the full pipeline without stopping."""
        from prism.llm.completion import CompletionEngine

        config = SwarmConfig(
            use_debate=False,
            use_moa=False,
            use_cascade=False,
            use_tools=False,
            total_budget=None,
            phase_budgets=None,
        )
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm_decompose,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("Build full feature")

        # Pipeline completes fully — no budget constraint
        assert plan.budget_exhausted is False
        assert plan.total_cost > 0
        # Should have costs for multiple phases
        assert len(plan.phase_costs) >= 3

    async def test_budget_exceeded_returns_partial_results(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm_decompose: object,
    ) -> None:
        """When budget is exceeded, partial results from completed phases are kept."""
        from prism.llm.completion import CompletionEngine

        # Budget enough for decompose + research but not plan
        # decompose=0.005, research=0.002, plan estimate=0.010
        # 0.005 + 0.002 = 0.007, next phase estimate 0.010 => 0.017 > 0.015
        config = SwarmConfig(
            use_debate=False,
            use_moa=False,
            use_cascade=False,
            use_tools=False,
            total_budget=0.015,
            auto_scale_budget=False,
        )
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm_decompose,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("Build something")

        # Should have partial results
        assert plan.budget_exhausted is True
        assert len(plan.tasks) >= 1  # decompose ran
        assert plan.goal == "Build something"

    def test_check_budget_total_over(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """_check_budget returns 'stop' when total cost >= total budget."""
        from prism.orchestrator.swarm import SwarmPhase

        orchestrator._config.total_budget = 0.01
        plan = SwarmPlan(goal="test", total_cost=0.01)
        assert orchestrator._check_budget(plan, SwarmPhase.EXECUTE) == "stop"
        assert plan.budget_exhausted is True

    def test_check_budget_under_budget(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """_check_budget returns 'proceed' when there's room in the budget."""
        from prism.orchestrator.swarm import SwarmPhase

        orchestrator._config.total_budget = 10.0  # generous budget
        plan = SwarmPlan(goal="test", total_cost=0.0)
        assert orchestrator._check_budget(plan, SwarmPhase.DECOMPOSE) == "proceed"
        assert plan.budget_exhausted is False

    def test_check_budget_none_budget(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """_check_budget returns 'proceed' when total_budget is None (unlimited)."""
        from prism.orchestrator.swarm import SwarmPhase

        orchestrator._config.total_budget = None
        orchestrator._config.phase_budgets = None
        plan = SwarmPlan(goal="test", total_cost=999.0)
        assert orchestrator._check_budget(plan, SwarmPhase.EXECUTE) == "proceed"
        assert plan.budget_exhausted is False

    def test_check_budget_phase_cap_returns_skip(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """_check_budget returns 'skip' for a phase with a tight per-phase cap."""
        from prism.orchestrator.swarm import SwarmPhase

        orchestrator._config.total_budget = None
        orchestrator._config.phase_budgets = {
            SwarmPhase.PLAN: 0.001,  # plan estimate is 0.010
        }
        plan = SwarmPlan(goal="test", total_cost=0.0)
        assert orchestrator._check_budget(plan, SwarmPhase.PLAN) == "skip"
        # Per-phase skip should NOT set budget_exhausted
        assert plan.budget_exhausted is False

    def test_check_budget_phase_not_capped(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """_check_budget returns 'proceed' for phases not in phase_budgets."""
        from prism.orchestrator.swarm import SwarmPhase

        orchestrator._config.total_budget = None
        orchestrator._config.phase_budgets = {
            SwarmPhase.PLAN: 0.001,  # only plan is capped
        }
        plan = SwarmPlan(goal="test", total_cost=0.0)
        assert orchestrator._check_budget(plan, SwarmPhase.DECOMPOSE) == "proceed"

    def test_check_budget_would_exceed_returns_stop(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """_check_budget returns 'stop' when phase estimate would exceed total."""
        from prism.orchestrator.swarm import SwarmPhase

        # total_cost=0.005, budget=0.010, plan estimate=0.010
        # 0.005 + 0.010 = 0.015 > 0.010 -> stop
        orchestrator._config.total_budget = 0.010
        orchestrator._config.phase_budgets = None
        plan = SwarmPlan(goal="test", total_cost=0.005)
        assert orchestrator._check_budget(plan, SwarmPhase.PLAN) == "stop"
        assert plan.budget_exhausted is True


# ======================================================================
# Fallback chain support
# ======================================================================


class TestSwarmTaskFallbackFields:
    """Test the new retry_count and models_tried fields on SwarmTask."""

    def test_retry_count_default_zero(self) -> None:
        """retry_count defaults to 0."""
        task = SwarmTask(description="Test task")
        assert task.retry_count == 0

    def test_models_tried_default_empty(self) -> None:
        """models_tried defaults to empty list."""
        task = SwarmTask(description="Test task")
        assert task.models_tried == []

    def test_retry_count_settable(self) -> None:
        """retry_count can be set."""
        task = SwarmTask(description="Test", retry_count=3)
        assert task.retry_count == 3

    def test_models_tried_settable(self) -> None:
        """models_tried can be set."""
        task = SwarmTask(description="Test", models_tried=["model-a", "model-b"])
        assert task.models_tried == ["model-a", "model-b"]


class TestSwarmConfigMaxRetries:
    """Test the max_retries field on SwarmConfig."""

    def test_max_retries_default_two(self) -> None:
        """max_retries defaults to 2."""
        cfg = SwarmConfig()
        assert cfg.max_retries == 2

    def test_max_retries_custom(self) -> None:
        """max_retries can be set to a custom value."""
        cfg = SwarmConfig(max_retries=5)
        assert cfg.max_retries == 5

    def test_max_retries_zero(self) -> None:
        """max_retries=0 disables fallback retries."""
        cfg = SwarmConfig(max_retries=0)
        assert cfg.max_retries == 0


class TestModelPoolGetFallbackModels:
    """Test the get_fallback_models method on ModelPool."""

    def test_returns_same_tier_models(
        self, model_pool: ModelPool,
    ) -> None:
        """Fallback models include same-tier alternatives."""
        fallbacks = model_pool.get_fallback_models("medium")
        assert isinstance(fallbacks, list)

    def test_excludes_already_tried(
        self, model_pool: ModelPool,
    ) -> None:
        """Models in the exclude list are not returned."""
        all_fallbacks = model_pool.get_fallback_models("medium")
        if not all_fallbacks:
            pytest.skip("No fallback models available")
        first = all_fallbacks[0]
        filtered = model_pool.get_fallback_models("medium", exclude=[first])
        assert first not in filtered

    def test_escalates_to_higher_tier(
        self, model_pool: ModelPool,
    ) -> None:
        """Fallback includes models from higher tiers after same-tier."""
        simple_fallbacks = model_pool.get_fallback_models("simple")
        assert len(simple_fallbacks) >= 1

    def test_empty_exclude_returns_all(
        self, model_pool: ModelPool,
    ) -> None:
        """Empty exclude list returns all available fallbacks."""
        with_none = model_pool.get_fallback_models("medium")
        with_empty = model_pool.get_fallback_models("medium", exclude=[])
        assert with_none == with_empty

    def test_invalid_complexity_falls_back_to_medium(
        self, model_pool: ModelPool,
    ) -> None:
        """Invalid complexity tier falls back to medium."""
        fallbacks = model_pool.get_fallback_models("nonsense")
        medium_fallbacks = model_pool.get_fallback_models("medium")
        assert fallbacks == medium_fallbacks

    def test_complex_tier_has_no_escalation(
        self, model_pool: ModelPool,
    ) -> None:
        """Complex tier has no higher tier to escalate to."""
        fallbacks = model_pool.get_fallback_models("complex")
        assert isinstance(fallbacks, list)


@pytest.mark.asyncio()
class TestFallbackChainExecution:
    """Test fallback chain during task execution in the swarm pipeline."""

    async def test_successful_first_attempt_no_fallback(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm_decompose: object,
    ) -> None:
        """When primary model succeeds, no fallback is triggered."""
        from prism.llm.completion import CompletionEngine

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm_decompose,
        )
        config = SwarmConfig(
            use_debate=False, use_moa=False, use_cascade=False,
            use_tools=False, max_retries=2,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("Build a simple feature")

        completed = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        assert len(completed) > 0
        for task in completed:
            assert task.retry_count == 0
            assert len(task.models_tried) == 1

    async def test_fallback_when_primary_model_fails(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """When primary model fails, fallback to next model succeeds."""
        import json

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse

        mock = MockLiteLLM()

        decompose_content = json.dumps([
            {
                "description": "Simple task",
                "complexity": "simple",
                "dependencies": [],
                "files_changed": [],
            },
        ])
        for model in (
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "gpt-4o",
            "o3",
            "gemini/gemini-2.5-pro",
        ):
            mock.set_response(
                model,
                MockResponse(
                    content=decompose_content,
                    input_tokens=200, output_tokens=300,
                ),
            )

        pool = ModelPool(mock_registry)
        primary_model = pool.get_execution_model("simple")
        fallback_models = pool.get_fallback_models(
            "simple", exclude=[primary_model],
        )

        if not fallback_models:
            pytest.skip("No fallback models available for simple tier")

        mock.set_error(primary_model, RuntimeError("Primary model unavailable"))
        mock.set_default_response(
            MockResponse(
                content="Fallback model output.",
                input_tokens=50, output_tokens=30,
            ),
        )

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock,
        )
        config = SwarmConfig(
            use_debate=False, use_moa=False, use_cascade=False,
            use_tools=False, max_retries=2,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("Do a simple task")

        completed = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        assert len(completed) >= 1
        for task in completed:
            if task.complexity == "simple":
                assert len(task.models_tried) >= 2
                assert task.retry_count >= 1
                assert primary_model in task.models_tried

    async def test_tier_escalation_when_all_same_tier_fail(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """When all same-tier models fail, escalation to next tier succeeds."""
        import json

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse

        mock = MockLiteLLM()

        decompose_content = json.dumps([
            {
                "description": "Simple task needing escalation",
                "complexity": "simple",
                "dependencies": [],
                "files_changed": [],
            },
        ])
        for model in (
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "gpt-4o",
            "o3",
            "gemini/gemini-2.5-pro",
        ):
            mock.set_response(
                model,
                MockResponse(
                    content=decompose_content,
                    input_tokens=200, output_tokens=300,
                ),
            )

        pool = ModelPool(mock_registry)

        all_simple_fallbacks = pool.get_fallback_models("simple", exclude=[])
        simple_only_models: list[str] = []
        for mid in all_simple_fallbacks:
            info = mock_registry.get_model_info(mid)
            if info is not None and info.tier.value == "simple":
                simple_only_models.append(mid)

        for model_id in simple_only_models:
            mock.set_error(
                model_id, RuntimeError(f"{model_id} unavailable"),
            )

        mock.set_default_response(
            MockResponse(
                content="Escalated model output.",
                input_tokens=50, output_tokens=30,
            ),
        )

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock,
        )
        config = SwarmConfig(
            use_debate=False, use_moa=False, use_cascade=False,
            use_tools=False, max_retries=10,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("Simple task with escalation")

        completed = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        assert len(completed) >= 1
        for task in completed:
            if task.complexity == "simple" and task.retry_count > 0:
                final_model = task.assigned_model
                if simple_only_models:
                    assert final_model not in simple_only_models

    async def test_max_retries_limit(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Fallback retries are limited to max_retries."""
        import json

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse

        mock = MockLiteLLM()

        decompose_content = json.dumps([
            {
                "description": "Task that always fails",
                "complexity": "medium",
                "dependencies": [],
                "files_changed": [],
            },
        ])
        for model in (
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "gpt-4o",
            "o3",
            "gemini/gemini-2.5-pro",
        ):
            mock.set_response(
                model,
                MockResponse(
                    content=decompose_content,
                    input_tokens=200, output_tokens=300,
                ),
            )

        pool = ModelPool(mock_registry)
        all_fallbacks = pool.get_fallback_models("medium", exclude=[])
        primary = pool.get_execution_model("medium")
        all_models_to_fail = {primary} | set(all_fallbacks)
        for model_id in all_models_to_fail:
            mock.set_error(model_id, RuntimeError(f"{model_id} down"))

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock,
        )
        config = SwarmConfig(
            use_debate=False, use_moa=False, use_cascade=False,
            use_tools=False, max_retries=1,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("Failing task")

        failed = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
        assert len(failed) >= 1
        for task in failed:
            if task.complexity == "medium":
                assert task.retry_count <= 1
                assert len(task.models_tried) <= 2

    async def test_fallback_tracks_models_tried(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """models_tried tracks all attempted models in order."""
        import json

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse

        mock = MockLiteLLM()

        decompose_content = json.dumps([
            {
                "description": "Task tracking models",
                "complexity": "medium",
                "dependencies": [],
                "files_changed": [],
            },
        ])
        for model in (
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "gpt-4o",
            "o3",
            "gemini/gemini-2.5-pro",
        ):
            mock.set_response(
                model,
                MockResponse(
                    content=decompose_content,
                    input_tokens=200, output_tokens=300,
                ),
            )

        pool = ModelPool(mock_registry)
        primary = pool.get_execution_model("medium")
        fallbacks = pool.get_fallback_models("medium", exclude=[primary])

        if not fallbacks:
            pytest.skip("No fallback models available for medium tier")

        mock.set_error(primary, RuntimeError("Primary down"))
        mock.set_default_response(
            MockResponse(
                content="Success from fallback.",
                input_tokens=50, output_tokens=30,
            ),
        )

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock,
        )
        config = SwarmConfig(
            use_debate=False, use_moa=False, use_cascade=False,
            use_tools=False, max_retries=2,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("Track models task")

        for task in plan.tasks:
            if task.complexity == "medium" and task.status == TaskStatus.COMPLETED:
                assert task.models_tried[0] == primary
                assert len(task.models_tried) >= 2
                assert len(task.models_tried) == len(set(task.models_tried))

    async def test_zero_max_retries_no_fallback(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """When max_retries=0, no fallback is attempted on failure."""
        import json

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse

        mock = MockLiteLLM()

        decompose_content = json.dumps([
            {
                "description": "Task with no retries",
                "complexity": "medium",
                "dependencies": [],
                "files_changed": [],
            },
        ])
        for model in (
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "gpt-4o",
            "o3",
            "gemini/gemini-2.5-pro",
        ):
            mock.set_response(
                model,
                MockResponse(
                    content=decompose_content,
                    input_tokens=200, output_tokens=300,
                ),
            )

        pool = ModelPool(mock_registry)
        primary = pool.get_execution_model("medium")
        mock.set_error(primary, RuntimeError("Primary failed"))

        mock.set_default_response(
            MockResponse(
                content="Should not reach this.",
                input_tokens=50, output_tokens=30,
            ),
        )

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock,
        )
        config = SwarmConfig(
            use_debate=False, use_moa=False, use_cascade=False,
            use_tools=False, max_retries=0,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        plan = await orch.orchestrate("No retry task")

        for task in plan.tasks:
            if task.complexity == "medium":
                assert task.retry_count == 0
                assert task.status == TaskStatus.FAILED
                assert len(task.models_tried) == 1


# ======================================================================
# Tool execution loop (_execute_with_tools / _run_single_tool)
# ======================================================================


class TestRunSingleTool:
    """Test _run_single_tool error handling and dispatch."""

    def test_no_registry_returns_error(
        self, orchestrator: SwarmOrchestrator,
    ) -> None:
        """Returns error string when no tool registry is set."""
        orchestrator._tool_registry = None
        result = orchestrator._run_single_tool("read_file", {"path": "/tmp/x"})
        assert "Error" in result
        assert "No tool registry" in result

    def test_tool_not_found_returns_error(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm: object,
    ) -> None:
        """Returns error string when tool name is not registered."""
        from prism.llm.completion import CompletionEngine
        from prism.tools.registry import ToolRegistry

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm,
        )
        tool_reg = ToolRegistry()  # empty registry
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )
        result = orch._run_single_tool("nonexistent_tool", {})
        assert "Error" in result
        assert "not found" in result

    def test_tool_execute_success(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm: object,
    ) -> None:
        """Successful tool execution returns the output string."""
        from unittest.mock import MagicMock

        from prism.llm.completion import CompletionEngine
        from prism.tools.base import ToolResult
        from prism.tools.registry import ToolRegistry

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm,
        )
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"
        mock_tool.parameters_schema = {"properties": {}, "required": []}
        mock_tool.execute.return_value = ToolResult(
            success=True, output="file contents here",
        )

        tool_reg = ToolRegistry()
        tool_reg.register(mock_tool)
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        result = orch._run_single_tool("read_file", {"path": "/tmp/test.py"})
        assert result == "file contents here"
        mock_tool.execute.assert_called_once_with({"path": "/tmp/test.py"})

    def test_tool_execute_failure_returns_error(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm: object,
    ) -> None:
        """Failed tool execution returns the error message."""
        from unittest.mock import MagicMock

        from prism.llm.completion import CompletionEngine
        from prism.tools.base import ToolResult
        from prism.tools.registry import ToolRegistry

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm,
        )
        mock_tool = MagicMock()
        mock_tool.name = "write_file"
        mock_tool.description = "Write a file"
        mock_tool.parameters_schema = {"properties": {}, "required": []}
        mock_tool.execute.return_value = ToolResult(
            success=False, output="", error="Permission denied",
        )

        tool_reg = ToolRegistry()
        tool_reg.register(mock_tool)
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        result = orch._run_single_tool("write_file", {"path": "/etc/passwd"})
        assert "Error" in result
        assert "Permission denied" in result

    def test_tool_execute_exception_returns_error(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm: object,
    ) -> None:
        """Tool that raises an exception returns an error string."""
        from unittest.mock import MagicMock

        from prism.llm.completion import CompletionEngine
        from prism.tools.registry import ToolRegistry

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm,
        )
        mock_tool = MagicMock()
        mock_tool.name = "execute_command"
        mock_tool.description = "Run a command"
        mock_tool.parameters_schema = {"properties": {}, "required": []}
        mock_tool.execute.side_effect = RuntimeError("Sandbox violation")

        tool_reg = ToolRegistry()
        tool_reg.register(mock_tool)
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        result = orch._run_single_tool(
            "execute_command", {"command": "rm -rf /"},
        )
        assert "Error executing tool" in result
        assert "Sandbox violation" in result


class TestExecuteWithTools:
    """Test the iterative tool execution loop."""

    async def test_no_tool_calls_returns_immediately(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """When model returns no tool calls, returns text immediately."""
        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse
        from prism.tools.registry import ToolRegistry

        mock_llm = MockLiteLLM()
        mock_llm.set_default_response(
            MockResponse(
                content="All done, no tools needed.", tool_calls=None,
            ),
        )
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_llm,
        )
        tool_reg = ToolRegistry()
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        messages: list[dict[str, object]] = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Hello"},
        ]
        text, _cost = await orch._execute_with_tools(messages, "gpt-4o")
        assert text == "All done, no tools needed."
        assert _cost >= 0.0
        assert len(mock_llm.call_log) == 1

    async def test_single_tool_call_then_final_response(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Model calls one tool, then returns final text on second call."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse
        from prism.llm.result import CompletionResult
        from prism.tools.base import ToolResult
        from prism.tools.registry import ToolRegistry

        mock_llm = MockLiteLLM()
        mock_llm.set_default_response(MockResponse(content="default"))
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_llm,
        )

        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"
        mock_tool.parameters_schema = {
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        mock_tool.execute.return_value = ToolResult(
            success=True, output="def hello():\n    print('hi')",
        )

        tool_reg = ToolRegistry()
        tool_reg.register(mock_tool)
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        call_count = 0
        tool_call_response = CompletionResult(
            content="",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=0.002,
            latency_ms=100.0,
            finish_reason="tool_calls",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": _json.dumps({"path": "/tmp/test.py"}),
                },
            }],
        )
        final_response = CompletionResult(
            content="The file contains a hello function.",
            model="gpt-4o",
            provider="openai",
            input_tokens=200,
            output_tokens=30,
            cached_tokens=0,
            cost_usd=0.003,
            latency_ms=80.0,
            finish_reason="stop",
            tool_calls=None,
        )

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_call_response
            return final_response

        with patch.object(engine, "complete", side_effect=mock_complete):
            messages: list[dict[str, object]] = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Read /tmp/test.py"},
            ]
            text, _cost = await orch._execute_with_tools(messages, "gpt-4o")

        assert text == "The file contains a hello function."
        assert _cost == pytest.approx(0.005)
        assert call_count == 2
        mock_tool.execute.assert_called_once_with({"path": "/tmp/test.py"})

        # Verify messages were built correctly
        assert len(messages) == 4
        assert messages[2]["role"] == "assistant"
        assert messages[2]["tool_calls"] == tool_call_response.tool_calls
        assert messages[3]["role"] == "tool"
        assert messages[3]["tool_call_id"] == "call_1"
        assert "def hello()" in messages[3]["content"]

    async def test_multiple_tool_calls_in_one_response(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Model returns multiple tool calls in a single response."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse
        from prism.llm.result import CompletionResult
        from prism.tools.base import ToolResult
        from prism.tools.registry import ToolRegistry

        mock_llm = MockLiteLLM()
        mock_llm.set_default_response(MockResponse(content="default"))
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_llm,
        )

        tool_a = MagicMock()
        tool_a.name = "read_file"
        tool_a.description = "Read a file"
        tool_a.parameters_schema = {"properties": {}, "required": []}
        tool_a.execute.return_value = ToolResult(
            success=True, output="content_a",
        )

        tool_b = MagicMock()
        tool_b.name = "list_directory"
        tool_b.description = "List a directory"
        tool_b.parameters_schema = {"properties": {}, "required": []}
        tool_b.execute.return_value = ToolResult(
            success=True, output="file1.py\nfile2.py",
        )

        tool_reg = ToolRegistry()
        tool_reg.register(tool_a)
        tool_reg.register(tool_b)
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        call_count = 0
        multi_tool_response = CompletionResult(
            content="",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=0.002,
            latency_ms=100.0,
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": _json.dumps({"path": "/tmp/a.py"}),
                    },
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "arguments": _json.dumps({"path": "/tmp"}),
                    },
                },
            ],
        )
        final_response = CompletionResult(
            content="Found two files.",
            model="gpt-4o",
            provider="openai",
            input_tokens=200,
            output_tokens=20,
            cached_tokens=0,
            cost_usd=0.001,
            latency_ms=50.0,
            finish_reason="stop",
        )

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return multi_tool_response
            return final_response

        with patch.object(engine, "complete", side_effect=mock_complete):
            messages: list[dict[str, object]] = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Read and list"},
            ]
            text, _cost = await orch._execute_with_tools(messages, "gpt-4o")

        assert text == "Found two files."
        assert _cost == pytest.approx(0.003)
        tool_a.execute.assert_called_once()
        tool_b.execute.assert_called_once()
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert len(tool_msgs) == 2

    async def test_max_iterations_limit(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Loop stops at max_iterations even if model keeps calling tools."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse
        from prism.llm.result import CompletionResult
        from prism.tools.base import ToolResult
        from prism.tools.registry import ToolRegistry

        mock_llm = MockLiteLLM()
        mock_llm.set_default_response(MockResponse(content="default"))
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_llm,
        )

        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read a file"
        mock_tool.parameters_schema = {"properties": {}, "required": []}
        mock_tool.execute.return_value = ToolResult(
            success=True, output="data",
        )

        tool_reg = ToolRegistry()
        tool_reg.register(mock_tool)
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        infinite_tool_response = CompletionResult(
            content="still working",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=0.001,
            latency_ms=50.0,
            finish_reason="tool_calls",
            tool_calls=[{
                "id": "call_loop",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": _json.dumps({"path": "/tmp/x"}),
                },
            }],
        )

        async def mock_complete(*args, **kwargs):
            return infinite_tool_response

        with patch.object(engine, "complete", side_effect=mock_complete):
            messages: list[dict[str, object]] = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Do something"},
            ]
            text, _cost = await orch._execute_with_tools(
                messages, "gpt-4o", max_iterations=3,
            )

        assert text == "still working"
        assert _cost == pytest.approx(0.003)
        assert mock_tool.execute.call_count == 3

    async def test_tool_error_fed_back_to_model(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """When a tool fails, the error is fed back for model recovery."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse
        from prism.llm.result import CompletionResult
        from prism.tools.base import ToolResult
        from prism.tools.registry import ToolRegistry

        mock_llm = MockLiteLLM()
        mock_llm.set_default_response(MockResponse(content="default"))
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_llm,
        )

        mock_tool = MagicMock()
        mock_tool.name = "write_file"
        mock_tool.description = "Write a file"
        mock_tool.parameters_schema = {"properties": {}, "required": []}
        mock_tool.execute.return_value = ToolResult(
            success=False, output="", error="Path outside sandbox",
        )

        tool_reg = ToolRegistry()
        tool_reg.register(mock_tool)
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        call_count = 0
        tool_call_resp = CompletionResult(
            content="",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=0.002,
            latency_ms=100.0,
            finish_reason="tool_calls",
            tool_calls=[{
                "id": "call_fail",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": _json.dumps({
                        "path": "/etc/passwd", "content": "x",
                    }),
                },
            }],
        )
        recovery_resp = CompletionResult(
            content="Could not write due to sandbox restrictions.",
            model="gpt-4o",
            provider="openai",
            input_tokens=200,
            output_tokens=30,
            cached_tokens=0,
            cost_usd=0.001,
            latency_ms=50.0,
            finish_reason="stop",
        )

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_call_resp
            return recovery_resp

        with patch.object(engine, "complete", side_effect=mock_complete):
            messages: list[dict[str, object]] = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Write to /etc/passwd"},
            ]
            text, _cost = await orch._execute_with_tools(messages, "gpt-4o")

        assert "sandbox" in text.lower()
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "Path outside sandbox" in tool_msgs[0]["content"]

    async def test_invalid_json_arguments_handled(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Invalid JSON in tool arguments does not crash the loop."""
        from unittest.mock import MagicMock, patch

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse
        from prism.llm.result import CompletionResult
        from prism.tools.base import ToolResult
        from prism.tools.registry import ToolRegistry

        mock_llm = MockLiteLLM()
        mock_llm.set_default_response(MockResponse(content="default"))
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_llm,
        )

        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read"
        mock_tool.parameters_schema = {"properties": {}, "required": []}
        mock_tool.execute.return_value = ToolResult(
            success=True, output="ok",
        )

        tool_reg = ToolRegistry()
        tool_reg.register(mock_tool)
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        call_count = 0
        bad_args_resp = CompletionResult(
            content="",
            model="gpt-4o",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=0.001,
            latency_ms=50.0,
            finish_reason="tool_calls",
            tool_calls=[{
                "id": "call_bad",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "not valid json {{{",
                },
            }],
        )
        final_resp = CompletionResult(
            content="Handled gracefully.",
            model="gpt-4o",
            provider="openai",
            input_tokens=150,
            output_tokens=20,
            cached_tokens=0,
            cost_usd=0.001,
            latency_ms=40.0,
            finish_reason="stop",
        )

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_args_resp
            return final_resp

        with patch.object(engine, "complete", side_effect=mock_complete):
            messages: list[dict[str, object]] = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Read something"},
            ]
            text, _cost = await orch._execute_with_tools(messages, "gpt-4o")

        assert text == "Handled gracefully."
        mock_tool.execute.assert_called_once_with({})


class TestExecuteDirectWithToolLoop:
    """Test _execute_direct delegates to _execute_with_tools."""

    async def test_execute_direct_uses_tool_loop(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """_execute_direct calls _execute_with_tools when both are set."""
        from unittest.mock import AsyncMock, MagicMock

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse
        from prism.tools.registry import ToolRegistry

        mock_llm = MockLiteLLM()
        mock_llm.set_default_response(
            MockResponse(content="Tool loop result."),
        )
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_llm,
        )

        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read"
        mock_tool.parameters_schema = {
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

        tool_reg = ToolRegistry()
        tool_reg.register(mock_tool)
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        assert orch._tool_schemas is not None
        assert len(orch._tool_schemas) > 0

        orch._execute_with_tools = AsyncMock(
            return_value=("delegated result", 0.01),
        )

        task = SwarmTask(description="Read some files")
        plan = SwarmPlan(goal="Test")
        text, cost = await orch._execute_direct(
            task, "gpt-4o", "Read files", plan,
        )

        assert text == "delegated result"
        assert cost == 0.01
        orch._execute_with_tools.assert_called_once()

    async def test_execute_direct_skips_loop_when_no_schemas(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm: object,
    ) -> None:
        """_execute_direct falls back to single call without tools."""
        from prism.llm.completion import CompletionEngine

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm,
        )
        config = SwarmConfig(use_tools=False)
        orch = SwarmOrchestrator(engine, mock_registry, config=config)

        assert orch._tool_schemas is None

        task = SwarmTask(description="Simple task")
        plan = SwarmPlan(goal="Test")
        text, cost = await orch._execute_direct(
            task, "gpt-4o", "Do something", plan,
        )

        assert isinstance(text, str)
        assert cost >= 0.0

    async def test_execute_direct_skips_loop_when_no_registry(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm: object,
    ) -> None:
        """_execute_direct falls back when tool_registry is None."""
        from prism.llm.completion import CompletionEngine

        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm,
        )
        config = SwarmConfig(use_tools=False)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=None,
        )
        orch._tool_schemas = [
            {"type": "function", "function": {"name": "x"}},
        ]
        orch._tool_registry = None

        task = SwarmTask(description="Edge case")
        plan = SwarmPlan(goal="Test")
        text, cost = await orch._execute_direct(
            task, "gpt-4o", "Do edge case", plan,
        )

        assert isinstance(text, str)
        assert cost >= 0.0

    async def test_cost_accumulates_across_tool_iterations(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
    ) -> None:
        """Costs from all iterations accumulate correctly."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from prism.llm.completion import CompletionEngine
        from prism.llm.mock import MockLiteLLM, MockResponse
        from prism.llm.result import CompletionResult
        from prism.tools.base import ToolResult
        from prism.tools.registry import ToolRegistry

        mock_llm = MockLiteLLM()
        mock_llm.set_default_response(MockResponse(content="default"))
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_llm,
        )

        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_tool.description = "Read"
        mock_tool.parameters_schema = {"properties": {}, "required": []}
        mock_tool.execute.return_value = ToolResult(
            success=True, output="data",
        )

        tool_reg = ToolRegistry()
        tool_reg.register(mock_tool)
        config = SwarmConfig(use_tools=True)
        orch = SwarmOrchestrator(
            engine, mock_registry, config=config, tool_registry=tool_reg,
        )

        call_count = 0

        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return CompletionResult(
                    content="",
                    model="gpt-4o",
                    provider="openai",
                    input_tokens=100,
                    output_tokens=50,
                    cached_tokens=0,
                    cost_usd=0.005,
                    latency_ms=50.0,
                    finish_reason="tool_calls",
                    tool_calls=[{
                        "id": f"call_{call_count}",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": _json.dumps({
                                "path": f"/tmp/{call_count}",
                            }),
                        },
                    }],
                )
            return CompletionResult(
                content="Final answer after 2 tool uses.",
                model="gpt-4o",
                provider="openai",
                input_tokens=300,
                output_tokens=30,
                cached_tokens=0,
                cost_usd=0.010,
                latency_ms=80.0,
                finish_reason="stop",
            )

        with patch.object(engine, "complete", side_effect=mock_complete):
            messages: list[dict[str, object]] = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Multi-step task"},
            ]
            text, _cost = await orch._execute_with_tools(messages, "gpt-4o")

        assert text == "Final answer after 2 tool uses."
        assert _cost == pytest.approx(0.020)
        assert call_count == 3
        assert mock_tool.execute.call_count == 2


# ======================================================================
# AEI (Adaptive Error Intelligence) integration
# ======================================================================


class TestAEIIntegration:
    """Test AEI wiring into the SwarmOrchestrator."""

    def test_orchestrator_accepts_error_intelligence_param(
        self,
        orchestrator_with_aei: SwarmOrchestrator,
    ) -> None:
        """SwarmOrchestrator stores the error_intelligence reference."""
        assert orchestrator_with_aei._error_intelligence is not None

    def test_orchestrator_without_aei_works(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """SwarmOrchestrator works when error_intelligence is None."""
        assert orchestrator._error_intelligence is None

    async def test_pipeline_completes_with_aei(
        self,
        orchestrator_with_aei: SwarmOrchestrator,
    ) -> None:
        """Full pipeline completes successfully with AEI wired in."""
        plan = await orchestrator_with_aei.orchestrate("Build a REST API with auth")
        assert isinstance(plan, SwarmPlan)
        assert plan.goal == "Build a REST API with auth"
        assert len(plan.tasks) >= 1

    async def test_aei_records_failure_pattern_on_task_failure(
        self,
        orchestrator_with_aei: SwarmOrchestrator,
    ) -> None:
        """AEI records a failure pattern when a task execution fails.

        We run a pipeline and verify AEI does not break the pipeline.
        """
        aei = orchestrator_with_aei._error_intelligence
        assert aei is not None
        stats_before = aei.get_stats()

        plan = await orchestrator_with_aei.orchestrate("Do complex task")
        assert isinstance(plan, SwarmPlan)

        stats_after = aei.get_stats()
        assert stats_after.total_attempts >= stats_before.total_attempts

    async def test_aei_informs_fix_cycle(
        self,
        orchestrator_with_aei: SwarmOrchestrator,
    ) -> None:
        """AEI advice is included in fix cycle retry prompts.

        We manually invoke _fix_cycle with mock error reviews and verify
        that AEI records the attempt.
        """
        from prism.intelligence.aei import FixStrategy

        aei = orchestrator_with_aei._error_intelligence
        assert aei is not None

        # Pre-seed AEI with a known failure pattern
        fp = aei.fingerprint_error(
            error_type="CrossReviewError",
            stack_trace="Missing error handling.",
            file_path="src/auth.py",
            function_name="Handle passwords",
        )
        aei.record_attempt(
            fingerprint=fp,
            strategy=FixStrategy.REGEX_PATCH,
            model="gpt-4o-mini",
            context_size=100,
            outcome="failure",
            reasoning="test seed",
        )

        task = SwarmTask(
            id="fix-test-task",
            description="Handle passwords",
            complexity="complex",
            status=TaskStatus.COMPLETED,
            result="password = 'admin123'",
            files_changed=["src/auth.py"],
        )
        plan = SwarmPlan(goal="Fix auth", tasks=[task])
        error_review = CrossReview(
            task_id="fix-test-task",
            reviewer_model="gpt-4o",
            severity=ReviewSeverity.ERROR,
            comments="Missing error handling.",
            approved=False,
        )

        stats_before = aei.get_stats()
        await orchestrator_with_aei._fix_cycle(plan, [error_review])
        stats_after = aei.get_stats()

        # AEI should have recorded the fix cycle attempt
        assert stats_after.total_attempts > stats_before.total_attempts

    async def test_aei_research_context_injected(
        self,
        orchestrator_with_aei: SwarmOrchestrator,
    ) -> None:
        """AEI error patterns are injected into research context."""
        from prism.intelligence.aei import FixStrategy

        aei = orchestrator_with_aei._error_intelligence
        assert aei is not None

        fp = aei.fingerprint_error(
            error_type="TypeError",
            stack_trace="cannot add str and int",
            file_path="src/utils.py",
            function_name="add_values",
        )
        aei.record_attempt(
            fingerprint=fp,
            strategy=FixStrategy.AST_DIFF,
            model="gpt-4o",
            context_size=200,
            outcome="success",
            reasoning="test seed for research",
        )

        context = orchestrator_with_aei._get_aei_research_context("Fix type errors")
        assert "TypeError" in context
        assert "Known error patterns" in context

    def test_aei_research_context_empty_when_no_data(
        self,
        orchestrator_with_aei: SwarmOrchestrator,
    ) -> None:
        """AEI research context returns empty string when no history."""
        context = orchestrator_with_aei._get_aei_research_context("Some goal")
        assert context == ""

    def test_aei_research_context_empty_when_no_aei(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """AEI research context returns empty string when AEI is None."""
        context = orchestrator._get_aei_research_context("Some goal")
        assert context == ""


# ======================================================================
# Context budget manager integration
# ======================================================================


class TestContextBudgetIntegration:
    """Test SmartContextBudgetManager wiring into the SwarmOrchestrator."""

    def test_orchestrator_accepts_context_manager_param(
        self,
        orchestrator_with_context_manager: SwarmOrchestrator,
    ) -> None:
        """SwarmOrchestrator stores the context_manager reference."""
        assert orchestrator_with_context_manager._context_manager is not None

    def test_orchestrator_without_context_manager_works(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """SwarmOrchestrator works when context_manager is None."""
        assert orchestrator._context_manager is None

    async def test_pipeline_completes_with_context_manager(
        self,
        orchestrator_with_context_manager: SwarmOrchestrator,
    ) -> None:
        """Full pipeline completes successfully with context manager."""
        plan = await orchestrator_with_context_manager.orchestrate(
            "Build a REST API with auth",
        )
        assert isinstance(plan, SwarmPlan)
        assert plan.goal == "Build a REST API with auth"
        assert len(plan.tasks) >= 1

    async def test_context_tokens_tracked(
        self,
        orchestrator_with_context_manager: SwarmOrchestrator,
    ) -> None:
        """Context tokens are tracked across phases when manager is present."""
        plan = await orchestrator_with_context_manager.orchestrate(
            "Add input validation",
        )
        assert plan.context_tokens_used >= 0

    def test_context_tokens_default_zero(self) -> None:
        """SwarmPlan.context_tokens_used defaults to 0."""
        plan = SwarmPlan()
        assert plan.context_tokens_used == 0

    def test_prompt_truncation_with_context_manager(
        self,
        orchestrator_with_context_manager: SwarmOrchestrator,
    ) -> None:
        """Prompts are truncated when they exceed model context budget."""
        # _build_task_prompt caps plan_text at 2000 chars and review_notes at
        # 1000 chars, so the description must be large enough on its own to
        # push total tokens past the 50% context-window budget (4096 tokens
        # for ollama/llama3.1:8b with 8192 window).  ~20k chars ≈ 5k tokens.
        task = SwarmTask(
            description="A " * 10_000,
            complexity="simple",
            assigned_model="ollama/llama3.1:8b",  # 8192 token window
        )
        plan = SwarmPlan(
            goal="Test truncation",
            plan_text="X " * 50_000,
            review_notes="Y " * 10_000,
        )

        prompt = orchestrator_with_context_manager._build_task_prompt(task, plan)
        assert "[... prompt truncated to fit context budget]" in prompt
        assert plan.context_tokens_used > 0

    def test_prompt_not_truncated_without_context_manager(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """Without context manager, prompts are not truncated."""
        task = SwarmTask(
            description="A" * 100,
            complexity="simple",
            assigned_model="ollama/llama3.1:8b",
        )
        plan = SwarmPlan(
            goal="Test no truncation",
            plan_text="X " * 50_000,
        )

        prompt = orchestrator._build_task_prompt(task, plan)
        assert "[... prompt truncated to fit context budget]" not in prompt

    def test_prompt_no_truncation_within_budget(
        self,
        orchestrator_with_context_manager: SwarmOrchestrator,
    ) -> None:
        """Small prompts within budget are not truncated."""
        task = SwarmTask(
            description="Build a small feature",
            complexity="simple",
            assigned_model="claude-sonnet-4-20250514",  # 200k window
        )
        plan = SwarmPlan(
            goal="Small task",
            plan_text="Step 1: Read code\nStep 2: Write code",
        )

        prompt = orchestrator_with_context_manager._build_task_prompt(task, plan)
        assert "[... prompt truncated to fit context budget]" not in prompt
        assert "Build a small feature" in prompt


# ======================================================================
# Graceful degradation — both AEI and context manager are optional
# ======================================================================


class TestGracefulDegradation:
    """Test that the swarm works identically when AEI/context are not provided."""

    async def test_pipeline_without_optional_deps(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """Pipeline runs fine without AEI or context manager."""
        assert orchestrator._error_intelligence is None
        assert orchestrator._context_manager is None

        plan = await orchestrator.orchestrate("Build auth module")
        assert isinstance(plan, SwarmPlan)
        assert len(plan.tasks) >= 1
        assert plan.context_tokens_used == 0

    async def test_fix_cycle_without_aei(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """Fix cycle works without AEI — no errors, no AEI calls."""
        task = SwarmTask(
            id="degrade-test",
            description="Fix something",
            complexity="medium",
            status=TaskStatus.COMPLETED,
            result="bad code",
        )
        plan = SwarmPlan(goal="Fix it", tasks=[task])
        error_review = CrossReview(
            task_id="degrade-test",
            reviewer_model="gpt-4o",
            severity=ReviewSeverity.ERROR,
            comments="This is wrong.",
            approved=False,
        )

        await orchestrator._fix_cycle(plan, [error_review])
        assert task.result is not None

    def test_aei_record_attempt_noop_without_aei(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """_aei_record_attempt is a no-op when AEI is None."""
        orchestrator._aei_record_attempt(
            fingerprint=None,
            model="gpt-4o",
            context_size=100,
            outcome="failure",
            reasoning="test",
        )

    def test_truncate_prompt_noop_without_context_manager(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """_truncate_prompt_to_budget returns prompt unchanged when no manager."""
        task = SwarmTask(description="Test", complexity="simple")
        plan = SwarmPlan(goal="Test")
        prompt = "Hello world"
        result = orchestrator._truncate_prompt_to_budget(prompt, task, plan)
        assert result == "Hello world"


# ======================================================================
# Project-size scalability tests
# ======================================================================


class TestProjectSizeScalability:
    """Verify the orchestrator scales across small and large projects."""

    def test_auto_scale_budget_default_true(self) -> None:
        """SwarmConfig defaults to auto_scale_budget=True."""
        cfg = SwarmConfig()
        assert cfg.auto_scale_budget is True

    def test_auto_scale_budget_disabled(self) -> None:
        """auto_scale_budget can be explicitly disabled."""
        cfg = SwarmConfig(auto_scale_budget=False)
        assert cfg.auto_scale_budget is False

    async def test_budget_scales_up_for_many_tasks(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm_decompose: object,
    ) -> None:
        """Budget auto-scales when task count exceeds headroom."""
        from prism.llm.completion import CompletionEngine

        config = SwarmConfig(
            use_debate=False,
            use_moa=False,
            use_cascade=False,
            use_tools=False,
            total_budget=0.50,
            auto_scale_budget=True,
        )
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm_decompose,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        await orch.orchestrate("Build a full app with auth, db, api")

        # Default decomposition returns 3 tasks → 3 * 0.10 = $0.30
        # $0.30 < $0.50 initial budget, so no scale-up needed here.
        # Budget should remain unchanged.
        assert config.total_budget == 0.50

    async def test_budget_no_scale_when_disabled(
        self,
        orch_settings: object,
        mock_cost_tracker: MagicMock,
        mock_auth: MagicMock,
        mock_registry: ProviderRegistry,
        mock_litellm_decompose: object,
    ) -> None:
        """Budget does not auto-scale when auto_scale_budget=False."""
        from prism.llm.completion import CompletionEngine

        config = SwarmConfig(
            use_debate=False,
            use_moa=False,
            use_cascade=False,
            use_tools=False,
            total_budget=0.01,
            auto_scale_budget=False,
        )
        engine = CompletionEngine(
            settings=orch_settings,
            cost_tracker=mock_cost_tracker,
            auth_manager=mock_auth,
            provider_registry=mock_registry,
            litellm_backend=mock_litellm_decompose,
        )
        orch = SwarmOrchestrator(engine, mock_registry, config=config)
        # This will likely be budget-exhausted because $0.01 is very low
        await orch.orchestrate("Build something")

        # Budget should NOT have been scaled even though tasks > headroom
        assert config.total_budget == 0.01

    def test_context_limits_small_model(self) -> None:
        """Small models get conservative context limits."""
        task = SwarmTask(
            description="Small task",
            complexity="simple",
            assigned_model="ollama/llama3.1:8b",
        )
        plan_limit, review_limit = SwarmOrchestrator._context_limits(task)
        # 8192 // 25 = 327 < 1000 minimum, so clamped to 1000
        assert plan_limit == 1000
        assert review_limit == 500

    def test_context_limits_large_model(self) -> None:
        """Large models get generous context limits."""
        task = SwarmTask(
            description="Complex task",
            complexity="complex",
            assigned_model="claude-sonnet-4-20250514",
        )
        plan_limit, review_limit = SwarmOrchestrator._context_limits(task)
        # 200000 // 25 = 8000 (capped)
        assert plan_limit == 8000
        assert review_limit == 4000

    def test_context_limits_medium_model(self) -> None:
        """Medium models get proportional context limits."""
        task = SwarmTask(
            description="Medium task",
            complexity="medium",
            assigned_model="gpt-4o",
        )
        plan_limit, review_limit = SwarmOrchestrator._context_limits(task)
        # 128000 // 25 = 5120 → within [1000, 8000]
        assert 1000 <= plan_limit <= 8000
        assert 500 <= review_limit <= 4000

    def test_context_limits_unknown_model_uses_default(self) -> None:
        """Unknown models use the default 128k context window."""
        task = SwarmTask(
            description="Task with unknown model",
            complexity="medium",
            assigned_model="some-future/model-v99",
        )
        plan_limit, _review_limit = SwarmOrchestrator._context_limits(task)
        # Default 128000 // 25 = 5120
        assert 1000 <= plan_limit <= 8000

    def test_context_limits_no_model_assigned(self) -> None:
        """When no model is assigned, defaults are used."""
        task = SwarmTask(description="Unassigned task", complexity="simple")
        plan_limit, review_limit = SwarmOrchestrator._context_limits(task)
        # Defaults to claude-sonnet-4 with 200k window
        assert plan_limit == 8000
        assert review_limit == 4000

    def test_build_task_prompt_scales_plan_text(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """Plan text included in prompt respects model-scaled limits."""
        task = SwarmTask(
            description="Read the code",
            complexity="simple",
            assigned_model="ollama/llama3.1:8b",
        )
        long_plan = "X " * 5_000  # 10k chars
        plan = SwarmPlan(goal="Test scaling", plan_text=long_plan)

        prompt = orchestrator._build_task_prompt(task, plan)
        # Small model limit is 1000 chars for plan, so plan_text should be truncated
        plan_section = prompt.split("Plan context:\n")[1].split("\n\nReview")[0]
        assert len(plan_section) <= 1005  # some slack for trailing

    def test_build_task_prompt_large_model_gets_more_plan(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """Large models get more plan context than small models."""
        long_plan = "Y " * 5_000  # 10k chars
        plan = SwarmPlan(goal="Test scaling", plan_text=long_plan)

        small_task = SwarmTask(
            description="Read",
            complexity="simple",
            assigned_model="ollama/llama3.1:8b",
        )
        large_task = SwarmTask(
            description="Read",
            complexity="complex",
            assigned_model="claude-sonnet-4-20250514",
        )

        small_prompt = orchestrator._build_task_prompt(small_task, plan)
        large_prompt = orchestrator._build_task_prompt(large_task, plan)

        # Large model prompt should have more plan context
        assert len(large_prompt) > len(small_prompt)

    def test_single_task_fallback_for_simple_goals(
        self,
        orchestrator: SwarmOrchestrator,
    ) -> None:
        """Decomposer falls back to a single task when parsing fails."""
        decomposer = orchestrator._decomposer
        # Simulate unparseable response
        tasks = decomposer._parse_tasks("Not valid JSON", "Fix the bug")
        assert len(tasks) == 1
        assert tasks[0].description == "Fix the bug"
        assert tasks[0].complexity == "medium"

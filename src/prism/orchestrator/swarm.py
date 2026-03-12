"""Multi-model collaborative task execution (Swarm Orchestration).

The SwarmOrchestrator decomposes complex tasks and distributes them across
multiple models from different providers.  Cheap models handle research and
simple subtasks; premium models handle planning and complex execution; a
*different* model always reviews another's output, enforcing cross-provider
quality through adversarial collaboration.

Phases:
    1. DECOMPOSE  -- Break goal into subtasks with dependency ordering.
    2. RESEARCH   -- Cheap models gather context in parallel.
    3. PLAN       -- Smart model synthesises research into a plan.
    4. REVIEW     -- Different model critiques the plan.
    5. EXECUTE    -- Subtasks assigned to models by complexity.
    6. CROSS_REVIEW -- Each model's output reviewed by a different model.
    7. INTEGRATE  -- Merge all outputs, resolve conflicts.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

from prism.providers.base import ComplexityTier

if TYPE_CHECKING:
    from prism.intelligence.aei import AdaptiveExecutionIntelligence
    from prism.intelligence.context_budget import SmartContextBudgetManager
    from prism.llm.completion import CompletionEngine
    from prism.orchestrator.cascade import CascadeResult, ConfidenceCascade
    from prism.orchestrator.debate import DebateEngine, DebateResult
    from prism.orchestrator.moa import MixtureOfAgents, MoAResult
    from prism.providers.registry import ProviderRegistry
    from prism.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Enumerations
# ------------------------------------------------------------------


class TaskStatus(StrEnum):
    """Lifecycle status of a single swarm task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SwarmPhase(StrEnum):
    """Phases of the swarm orchestration pipeline."""

    DECOMPOSE = "decompose"
    RESEARCH = "research"
    PLAN = "plan"
    REVIEW = "review"
    EXECUTE = "execute"
    CROSS_REVIEW = "cross_review"
    INTEGRATE = "integrate"


class ReviewSeverity(StrEnum):
    """Severity levels for cross-review findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass
class SwarmTask:
    """A single executable subtask within a swarm plan.

    Attributes:
        id: Unique identifier (UUID).
        description: Human-readable description of the subtask.
        complexity: Complexity tier (``simple``, ``medium``, ``complex``).
        dependencies: IDs of tasks that must complete before this one.
        assigned_model: LiteLLM model ID assigned for execution.
        status: Current lifecycle status.
        result: Output text after execution, or ``None``.
        files_changed: List of file paths modified by this task.
        retry_count: Number of fallback retries attempted for this task.
        models_tried: List of model IDs that were attempted for this task.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    complexity: str = "medium"
    dependencies: list[str] = field(default_factory=list)
    assigned_model: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    files_changed: list[str] = field(default_factory=list)
    retry_count: int = 0
    models_tried: list[str] = field(default_factory=list)


@dataclass
class CrossReview:
    """Result of a cross-review on a completed subtask.

    Attributes:
        task_id: ID of the task that was reviewed.
        reviewer_model: Model that performed the review.
        severity: Highest severity finding (info/warning/error).
        comments: Human-readable review comments.
        approved: Whether the review passed without errors.
    """

    task_id: str
    reviewer_model: str
    severity: ReviewSeverity
    comments: str
    approved: bool


@dataclass
class SwarmConfig:
    """Configuration for advanced swarm orchestration features.

    Controls which research-backed techniques are active and how they
    interact with the base seven-phase pipeline.

    Attributes:
        use_debate: Enable multi-round debate (Du et al. ICML 2024) for the
            REVIEW phase.  Multiple models debate the plan before execution.
        use_moa: Enable Mixture-of-Agents (Wang et al. 2024) for complex
            tasks in the EXECUTE phase.  Parallel generation + fusion.
        use_cascade: Enable FrugalGPT confidence cascading (Chen et al. 2023)
            for the EXECUTE phase.  Try cheap models first, escalate when
            confidence is low.
        use_tools: Enable tool schemas so execution models can call file/terminal
            tools during task execution.
        moa_complexity_threshold: Only use MoA for tasks at or above this
            complexity level (``"complex"`` by default).
        cascade_budget_per_task: Max USD spend per task in the confidence cascade.
        total_budget: Maximum total USD for the entire swarm pipeline.
            ``None`` means unlimited.
        phase_budgets: Optional per-phase budget limits in USD.  Keys are
            ``SwarmPhase`` values (e.g. ``"decompose"``, ``"execute"``).
            ``None`` means no per-phase limits.
        max_retries: Maximum number of fallback retry attempts per task during
            execution.  When the primary model fails, alternative models from
            the same tier (and then escalated tiers) are tried up to this limit.
        auto_scale_budget: When True, automatically scale the total budget
            upward after decomposition if the number of tasks would exceed
            the per-task headroom ($0.10 per task).  This ensures large
            projects with many subtasks don't hit budget caps prematurely
            while small projects stay within the original budget.
    """

    use_debate: bool = True
    use_moa: bool = True
    use_cascade: bool = True
    use_tools: bool = True
    moa_complexity_threshold: str = "complex"
    cascade_budget_per_task: float | None = 0.05
    total_budget: float | None = 1.0
    phase_budgets: dict[str, float] | None = None
    max_retries: int = 2
    auto_scale_budget: bool = True


@dataclass
class SwarmPlan:
    """Full orchestration plan produced by the swarm pipeline.

    Attributes:
        goal: Original user goal.
        tasks: Ordered list of subtasks.
        research_findings: Mapping of research question to answer.
        plan_text: Full planning model output.
        review_notes: Critique from the review model.
        cross_reviews: Post-execution cross-review results.
        total_cost: Accumulated cost in USD across all phases.
        phase_costs: Per-phase cost breakdown.
        debate_result: Debate result from the REVIEW phase, if debate was used.
        cascade_results: Per-task cascade results from the EXECUTE phase.
        moa_results: Per-task MoA results for complex tasks.
        budget_exhausted: True if the pipeline was stopped due to budget limits.
        context_tokens_used: Total context tokens consumed across all phases.
    """

    goal: str = ""
    tasks: list[SwarmTask] = field(default_factory=list)
    research_findings: dict[str, str] = field(default_factory=dict)
    plan_text: str = ""
    review_notes: str = ""
    cross_reviews: list[CrossReview] = field(default_factory=list)
    total_cost: float = 0.0
    phase_costs: dict[str, float] = field(default_factory=dict)
    debate_result: DebateResult | None = None
    cascade_results: dict[str, CascadeResult] = field(default_factory=dict)
    moa_results: dict[str, MoAResult] = field(default_factory=dict)
    budget_exhausted: bool = False
    context_tokens_used: int = 0


# ------------------------------------------------------------------
# ModelPool — categorise available models by capability tier
# ------------------------------------------------------------------

# Provider ordering preferences (lower = preferred for research)
_CHEAP_PROVIDERS = ("groq", "ollama", "deepseek", "mistral")
_SMART_PROVIDERS = ("anthropic", "openai", "google")


class ModelPool:
    """Categorises available models into capability tiers.

    Uses the ``ProviderRegistry`` to discover which models are currently
    available, then exposes helpers to pick the right model for each
    orchestration phase.

    Key principle: the review model is always from a **different provider**
    than the execution/planning model to ensure genuine cross-pollination.
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        """Initialise the model pool from a provider registry.

        Args:
            registry: Provider registry with model metadata and availability.
        """
        self._registry = registry
        self._research_models: list[str] = []
        self._planning_model: str | None = None
        self._review_model: str | None = None
        self._execution_models: dict[str, list[str]] = {
            "simple": [],
            "medium": [],
            "complex": [],
        }
        self._categorise()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_research_models(self) -> list[str]:
        """Return the cheapest available models for research tasks.

        Prefers Groq, Ollama, DeepSeek, and Mistral.  Falls back to any
        available model if no cheap providers are configured.

        Returns:
            List of LiteLLM model identifiers, cheapest first.
        """
        if self._research_models:
            return list(self._research_models)
        # Fallback: any available model
        all_models = self._registry.get_available_models()
        return [m.id for m in all_models[:3]] if all_models else []

    def get_planning_model(self) -> str:
        """Return the smartest available model for planning.

        Prefers Claude Sonnet/Opus, GPT-4o, Gemini Pro.

        Returns:
            LiteLLM model identifier.

        Raises:
            RuntimeError: If no models are available at all.
        """
        if self._planning_model:
            return self._planning_model
        all_models = self._registry.get_available_models()
        if not all_models:
            msg = "No models available for planning"
            raise RuntimeError(msg)
        # Use the most expensive available model (proxy for smartest)
        all_models.sort(
            key=lambda m: m.input_cost_per_1m + m.output_cost_per_1m,
            reverse=True,
        )
        return all_models[0].id

    def get_review_model(self) -> str:
        """Return a smart model from a *different* provider than the planner.

        This ensures genuine cross-pollination: the planner and reviewer
        never share the same provider biases.

        Returns:
            LiteLLM model identifier.

        Raises:
            RuntimeError: If no models are available.
        """
        if self._review_model:
            return self._review_model
        return self.get_planning_model()

    def get_execution_model(self, complexity: str) -> str:
        """Return a model matched to the task difficulty.

        Args:
            complexity: One of ``"simple"``, ``"medium"``, ``"complex"``.

        Returns:
            LiteLLM model identifier.

        Raises:
            RuntimeError: If no models are available.
        """
        tier = complexity if complexity in self._execution_models else "medium"
        candidates = self._execution_models.get(tier, [])
        if candidates:
            return candidates[0]
        # Fall back through tiers
        for fallback_tier in ("medium", "simple", "complex"):
            candidates = self._execution_models.get(fallback_tier, [])
            if candidates:
                return candidates[0]
        return self.get_planning_model()

    def get_fallback_models(
        self,
        complexity: str,
        exclude: list[str] | None = None,
    ) -> list[str]:
        """Return alternative models for a given complexity tier.

        Builds a fallback list by first returning remaining same-tier models
        (excluding any already tried), then escalating to the next higher
        tier.  The escalation order is simple -> medium -> complex.

        Args:
            complexity: One of ``"simple"``, ``"medium"``, ``"complex"``.
            exclude: Model IDs to exclude (already tried).

        Returns:
            Ordered list of fallback model IDs (may be empty).
        """
        excluded: set[str] = set(exclude or [])
        fallbacks: list[str] = []

        # Same-tier alternatives first
        tier = complexity if complexity in self._execution_models else "medium"
        for model_id in self._execution_models.get(tier, []):
            if model_id not in excluded and model_id not in fallbacks:
                fallbacks.append(model_id)

        # Escalate to higher tiers: simple -> medium -> complex
        escalation_order = ["simple", "medium", "complex"]
        try:
            current_idx = escalation_order.index(tier)
        except ValueError:
            current_idx = 1  # default to medium

        for next_tier in escalation_order[current_idx + 1:]:
            for model_id in self._execution_models.get(next_tier, []):
                if model_id not in excluded and model_id not in fallbacks:
                    fallbacks.append(model_id)

        return fallbacks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _categorise(self) -> None:
        """Populate model pools from the registry."""
        # Research models: cheapest from preferred providers
        for provider_name in _CHEAP_PROVIDERS:
            if not self._registry.is_provider_available(provider_name):
                continue
            provider = self._registry.get_provider(provider_name)
            if provider is None:
                continue
            for model in provider.models:
                self._research_models.append(model.id)

        # Sort research models by cost (cheapest first)
        self._research_models.sort(key=self._model_cost)

        # Planning model: smartest available from premium providers
        planning_candidates: list[tuple[float, str, str]] = []
        for provider_name in _SMART_PROVIDERS:
            if not self._registry.is_provider_available(provider_name):
                continue
            provider = self._registry.get_provider(provider_name)
            if provider is None:
                continue
            for model in provider.models:
                if model.tier == ComplexityTier.COMPLEX:
                    cost = model.input_cost_per_1m + model.output_cost_per_1m
                    planning_candidates.append((cost, model.id, provider_name))

        if planning_candidates:
            # Highest cost = smartest (heuristic)
            planning_candidates.sort(reverse=True)
            self._planning_model = planning_candidates[0][1]
            planning_provider = planning_candidates[0][2]

            # Review model: smart model from a DIFFERENT provider
            for _cost, model_id, prov in planning_candidates:
                if prov != planning_provider:
                    self._review_model = model_id
                    break

            # If no different provider found, use the second candidate from same
            if self._review_model is None and len(planning_candidates) > 1:
                self._review_model = planning_candidates[1][1]

        # Execution models per tier
        tier_map = {
            ComplexityTier.SIMPLE: "simple",
            ComplexityTier.MEDIUM: "medium",
            ComplexityTier.COMPLEX: "complex",
        }
        for tier_enum, tier_key in tier_map.items():
            models = self._registry.get_available_models(tier_enum)
            self._execution_models[tier_key] = [m.id for m in models]

        logger.info(
            "model_pool_categorised",
            research=len(self._research_models),
            planning=self._planning_model,
            review=self._review_model,
            execution_simple=len(self._execution_models["simple"]),
            execution_medium=len(self._execution_models["medium"]),
            execution_complex=len(self._execution_models["complex"]),
        )

    def _model_cost(self, model_id: str) -> float:
        """Return combined input+output cost for a model, or infinity."""
        info = self._registry.get_model_info(model_id)
        if info is None:
            return float("inf")
        return info.input_cost_per_1m + info.output_cost_per_1m


# ------------------------------------------------------------------
# TaskDecomposer — break goals into subtasks
# ------------------------------------------------------------------

_DECOMPOSE_SYSTEM_PROMPT = """\
You are a task decomposition engine.  Given a goal and repository context,
break it into discrete subtasks with dependency ordering.

Respond ONLY with a JSON array of objects, each having:
- "description": string describing the subtask
- "complexity": "simple" | "medium" | "complex"
- "dependencies": list of 0-indexed positions of tasks this depends on
- "files_changed": list of file paths likely to be modified

Example:
[
  {"description": "Read existing models.py to understand schema", "complexity": "simple", "dependencies": [], "files_changed": []},
  {"description": "Create new User model with validation", "complexity": "medium", "dependencies": [0], "files_changed": ["src/models.py"]},
  {"description": "Write unit tests for User model", "complexity": "medium", "dependencies": [1], "files_changed": ["tests/test_models.py"]}
]
"""


class TaskDecomposer:
    """Decomposes a goal string into ordered ``SwarmTask`` objects.

    Uses a smart model (via ``CompletionEngine``) to break a goal into
    subtasks, detect dependencies, and estimate complexity.

    Args:
        engine: Completion engine for LLM calls (must have a mock backend
            injected for testing).
        model_pool: Model pool for selecting the decomposition model.
    """

    def __init__(
        self,
        engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        self._engine = engine
        self._model_pool = model_pool

    async def decompose(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> list[SwarmTask]:
        """Decompose a goal into an ordered list of subtasks.

        Args:
            goal: Natural-language description of the overall objective.
            context: Optional context dict (e.g. file list, repo info).

        Returns:
            List of ``SwarmTask`` objects in dependency order.

        Raises:
            ValueError: If *goal* is empty.
        """
        if not goal or not goal.strip():
            raise ValueError("Goal must not be empty")

        context = context or {}
        context_str = self._format_context(context)

        user_content = f"Goal: {goal}"
        if context_str:
            user_content += f"\n\nRepository context:\n{context_str}"

        model = self._model_pool.get_planning_model()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        result = await self._engine.complete(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=2048,
        )

        tasks = self._parse_tasks(result.content, goal)

        logger.info(
            "tasks_decomposed",
            goal_preview=goal[:80],
            task_count=len(tasks),
            model=model,
        )
        return tasks

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_tasks(self, raw: str, goal: str) -> list[SwarmTask]:
        """Parse the LLM's JSON response into SwarmTask objects.

        Falls back to a single task wrapping the entire goal if parsing
        fails.

        Args:
            raw: Raw model output (expected to be JSON array).
            goal: Original goal text for fallback.

        Returns:
            List of SwarmTask objects.
        """
        # Try to extract a JSON array from the response
        stripped = raw.strip()
        json_start = stripped.find("[")
        json_end = stripped.rfind("]")

        if json_start >= 0 and json_end > json_start:
            json_str = stripped[json_start : json_end + 1]
            try:
                items = json.loads(json_str)
                if isinstance(items, list) and items:
                    return self._items_to_tasks(items)
            except (json.JSONDecodeError, TypeError, KeyError):
                logger.warning("task_decompose_parse_failed", raw_preview=raw[:200])

        # Fallback: single task for the whole goal
        return [
            SwarmTask(
                description=goal.strip(),
                complexity="medium",
            ),
        ]

    def _items_to_tasks(self, items: list[dict[str, Any]]) -> list[SwarmTask]:
        """Convert parsed JSON items into SwarmTask objects with resolved deps.

        Args:
            items: List of dicts from the model's JSON output.

        Returns:
            List of SwarmTask objects with UUID-based dependencies.
        """
        tasks: list[SwarmTask] = []
        index_to_id: dict[int, str] = {}

        # First pass: create tasks and map indices to UUIDs
        for idx, item in enumerate(items):
            task_id = str(uuid.uuid4())
            index_to_id[idx] = task_id

            description = str(item.get("description", f"Subtask {idx + 1}"))
            complexity = str(item.get("complexity", "medium"))
            if complexity not in ("simple", "medium", "complex"):
                complexity = "medium"

            files = item.get("files_changed", [])
            if not isinstance(files, list):
                files = []
            files = [str(f) for f in files]

            tasks.append(
                SwarmTask(
                    id=task_id,
                    description=description,
                    complexity=complexity,
                    files_changed=files,
                ),
            )

        # Second pass: resolve dependency indices to UUIDs
        for idx, item in enumerate(items):
            raw_deps = item.get("dependencies", [])
            if not isinstance(raw_deps, list):
                raw_deps = []
            resolved: list[str] = []
            for dep in raw_deps:
                if isinstance(dep, int) and dep in index_to_id:
                    resolved.append(index_to_id[dep])
            tasks[idx].dependencies = resolved

        return tasks

    @staticmethod
    def _format_context(context: dict[str, Any]) -> str:
        """Format context dict into a string for the prompt.

        Args:
            context: Arbitrary context dict.

        Returns:
            Formatted string, or empty string if context is empty.
        """
        if not context:
            return ""
        parts: list[str] = []
        for key, value in context.items():
            if isinstance(value, list):
                parts.append(f"{key}: {', '.join(str(v) for v in value[:20])}")
            else:
                parts.append(f"{key}: {value}")
        return "\n".join(parts)


# ------------------------------------------------------------------
# ResearchSwarm — parallel cheap-model research
# ------------------------------------------------------------------

_RESEARCH_SYSTEM_PROMPT = """\
You are a focused research assistant.  Answer the question concisely based
on the provided context.  If context is insufficient, say so clearly.
Keep your answer under 500 words.
"""


class ResearchSwarm:
    """Runs research questions on cheap models in parallel.

    Uses ``concurrent.futures.ThreadPoolExecutor`` to fan out multiple
    research queries simultaneously, each using the cheapest available model.

    Args:
        engine: Completion engine (mock-safe).
        model_pool: For selecting research models.
        max_workers: Maximum parallel research threads.
    """

    def __init__(
        self,
        engine: CompletionEngine,
        model_pool: ModelPool,
        max_workers: int = 4,
    ) -> None:
        self._engine = engine
        self._model_pool = model_pool
        self._max_workers = max_workers

    async def research(
        self,
        questions: list[str],
        context: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Run research questions in parallel on cheap models.

        Args:
            questions: List of research questions.
            context: Optional context snippets (key=topic, value=content).

        Returns:
            Dict mapping each question to its answer.
        """
        if not questions:
            return {}

        context = context or {}
        research_models = self._model_pool.get_research_models()
        if not research_models:
            logger.warning("no_research_models_available")
            return {q: "No research models available" for q in questions}

        findings: dict[str, str] = {}

        tasks = []
        for idx, question in enumerate(questions):
            model = research_models[idx % len(research_models)]
            tasks.append(self._research_one(question, model, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for question, result in zip(questions, results, strict=False):
            if isinstance(result, Exception):
                logger.warning(
                    "research_question_failed",
                    question=question[:80],
                    error=str(result),
                )
                findings[question] = f"Research failed: {result}"
            else:
                findings[question] = result

        logger.info(
            "research_complete",
            questions=len(questions),
            successful=sum(
                1 for v in findings.values() if not v.startswith("Research failed")
            ),
        )
        return findings

    async def _research_one(
        self,
        question: str,
        model: str,
        context: dict[str, str],
    ) -> str:
        """Execute a single research question.

        Args:
            question: The research question.
            model: Model to use.
            context: Shared context dict.

        Returns:
            The model's answer text.
        """
        context_str = ""
        if context:
            context_str = "\n\nRelevant context:\n" + "\n".join(
                f"- {k}: {v}" for k, v in context.items()
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _RESEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}{context_str}"},
        ]

        result = await self._engine.complete(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )
        return result.content


# ------------------------------------------------------------------
# PlanReviewer — critique a plan with a different model
# ------------------------------------------------------------------

_PLAN_REVIEW_SYSTEM_PROMPT = """\
You are a senior software architect reviewing an execution plan.
Critique the plan for:
- Missing edge cases or error handling
- Security vulnerabilities
- Wrong approach or architecture
- Missing dependencies between tasks
- Potential performance issues

Be specific and actionable.  For each issue, state:
1. What is wrong
2. Why it matters
3. How to fix it

If the plan is solid, say so briefly and suggest minor improvements.
"""


class PlanReviewer:
    """Reviews a plan using a model different from the one that created it.

    The reviewer critiques for missing edge cases, security issues,
    wrong approaches, and returns structured feedback.  The planner
    can then refine the plan based on this review.

    Args:
        engine: Completion engine (mock-safe).
        model_pool: For selecting the review model.
    """

    def __init__(
        self,
        engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        self._engine = engine
        self._model_pool = model_pool

    async def review(self, plan_text: str, goal: str) -> str:
        """Review a plan and return critique notes.

        Args:
            plan_text: The full plan text to review.
            goal: Original goal for context.

        Returns:
            Review notes as a string.
        """
        if not plan_text.strip():
            return "No plan provided to review."

        model = self._model_pool.get_review_model()

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _PLAN_REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original goal: {goal}\n\n"
                    f"Plan to review:\n{plan_text}"
                ),
            },
        ]

        result = await self._engine.complete(
            messages=messages,
            model=model,
            temperature=0.4,
            max_tokens=2048,
        )

        logger.info(
            "plan_reviewed",
            model=model,
            review_length=len(result.content),
        )
        return result.content


# ------------------------------------------------------------------
# CrossReviewer — post-execution review by a different model
# ------------------------------------------------------------------

_CROSS_REVIEW_SYSTEM_PROMPT = """\
You are a code reviewer evaluating the output of another AI model.
For each piece of work, check:
- Correctness: Does it actually accomplish the task description?
- Consistency: Is it consistent with the other tasks' outputs?
- Style: Does it follow project conventions?
- Security: Are there any security concerns?

Respond with a JSON object:
{
  "severity": "info" | "warning" | "error",
  "approved": true | false,
  "comments": "your detailed review"
}

Use "error" severity only for genuine bugs or security issues.
Use "warning" for style or approach concerns.
Use "info" for minor suggestions.
"""


class CrossReviewer:
    """Reviews completed subtask outputs using a different model.

    After execution, each subtask's output is sent to a model from a
    *different* provider.  If the review severity is ``error``, a fix
    cycle can be triggered.

    Args:
        engine: Completion engine (mock-safe).
        model_pool: For selecting review models.
    """

    def __init__(
        self,
        engine: CompletionEngine,
        model_pool: ModelPool,
    ) -> None:
        self._engine = engine
        self._model_pool = model_pool

    async def review_task(
        self,
        task: SwarmTask,
        all_tasks: list[SwarmTask] | None = None,
    ) -> CrossReview:
        """Review a single completed task.

        Args:
            task: The completed task with ``result`` populated.
            all_tasks: All tasks for consistency checking context.

        Returns:
            A ``CrossReview`` with severity, approval, and comments.
        """
        if task.result is None:
            return CrossReview(
                task_id=task.id,
                reviewer_model="none",
                severity=ReviewSeverity.ERROR,
                comments="Task has no result to review.",
                approved=False,
            )

        model = self._model_pool.get_review_model()

        # Build context from sibling tasks
        sibling_context = ""
        if all_tasks:
            completed = [
                t for t in all_tasks
                if t.id != task.id and t.status == TaskStatus.COMPLETED and t.result
            ]
            if completed:
                sibling_summaries = [
                    f"- {t.description}: {(t.result or '')[:200]}"
                    for t in completed[:5]
                ]
                sibling_context = (
                    "\n\nOther completed tasks for consistency reference:\n"
                    + "\n".join(sibling_summaries)
                )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _CROSS_REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {task.description}\n"
                    f"Complexity: {task.complexity}\n"
                    f"Files changed: {', '.join(task.files_changed) or 'none'}\n\n"
                    f"Output to review:\n{task.result}"
                    f"{sibling_context}"
                ),
            },
        ]

        result = await self._engine.complete(
            messages=messages,
            model=model,
            temperature=0.2,
            max_tokens=1024,
        )

        review = self._parse_review(result.content, task.id, model)

        logger.info(
            "cross_review_complete",
            task_id=task.id,
            severity=review.severity,
            approved=review.approved,
            model=model,
        )
        return review

    async def review_all(
        self,
        tasks: list[SwarmTask],
    ) -> list[CrossReview]:
        """Review all completed tasks.

        Args:
            tasks: All tasks (only completed ones are reviewed).

        Returns:
            List of CrossReview results.
        """
        completed = [t for t in tasks if t.status == TaskStatus.COMPLETED and t.result]
        if not completed:
            return []

        review_coros = [self.review_task(task, tasks) for task in completed]
        reviews = await asyncio.gather(*review_coros, return_exceptions=True)

        results: list[CrossReview] = []
        for task, review in zip(completed, reviews, strict=False):
            if isinstance(review, Exception):
                logger.warning(
                    "cross_review_failed",
                    task_id=task.id,
                    error=str(review),
                )
                results.append(
                    CrossReview(
                        task_id=task.id,
                        reviewer_model="error",
                        severity=ReviewSeverity.WARNING,
                        comments=f"Review failed: {review}",
                        approved=True,  # Don't block on review failure
                    ),
                )
            else:
                results.append(review)

        return results

    @staticmethod
    def _parse_review(raw: str, task_id: str, model: str) -> CrossReview:
        """Parse the review model's JSON response into a CrossReview.

        Falls back to a permissive review if parsing fails.

        Args:
            raw: Raw model output (expected JSON).
            task_id: ID of the task being reviewed.
            model: Model that performed the review.

        Returns:
            Parsed CrossReview.
        """
        stripped = raw.strip()
        json_start = stripped.find("{")
        json_end = stripped.rfind("}")

        if json_start >= 0 and json_end > json_start:
            json_str = stripped[json_start : json_end + 1]
            try:
                data = json.loads(json_str)
                severity_str = str(data.get("severity", "info")).lower()
                if severity_str not in ("info", "warning", "error"):
                    severity_str = "info"
                severity = ReviewSeverity(severity_str)
                approved = bool(data.get("approved", True))
                comments = str(data.get("comments", "No comments."))

                return CrossReview(
                    task_id=task_id,
                    reviewer_model=model,
                    severity=severity,
                    comments=comments,
                    approved=approved,
                )
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                pass

        # Fallback: treat raw text as comments, assume info severity
        return CrossReview(
            task_id=task_id,
            reviewer_model=model,
            severity=ReviewSeverity.INFO,
            comments=raw[:500] if raw else "No review output.",
            approved=True,
        )


# ------------------------------------------------------------------
# SwarmOrchestrator — main orchestration pipeline
# ------------------------------------------------------------------

_PLAN_SYSTEM_PROMPT = """\
You are a planning engine for a multi-model AI system.  Given:
1. The user's goal
2. Research findings gathered by other models

Create a detailed execution plan.  For each subtask, specify:
- What needs to be done (description)
- Why it matters (rationale)
- What files will be affected
- What the acceptance criteria are

Structure your plan clearly with numbered steps.
"""

_EXECUTE_SYSTEM_PROMPT = """\
You are an execution engine.  Complete the following task precisely.
Follow the plan exactly.  If you encounter an issue, describe it clearly
rather than silently working around it.
"""

_INTEGRATE_SYSTEM_PROMPT = """\
You are an integration engine.  Given the outputs from multiple subtasks
and any cross-review feedback, produce a unified summary:
1. What was accomplished
2. Any issues found during review
3. Recommended next steps

Be concise but thorough.
"""


class SwarmOrchestrator:
    """Multi-model collaborative task execution.

    Runs a seven-phase pipeline:

    1. **DECOMPOSE** -- Break task into subtasks with dependencies.
    2. **RESEARCH** -- Cheap models gather context in parallel.
    3. **PLAN** -- Smart model synthesises research into a plan.
    4. **REVIEW** -- Different model critiques the plan (or multi-round debate).
    5. **EXECUTE** -- Subtasks assigned to models by complexity, run with tool-use.
       Uses confidence cascading (FrugalGPT) and/or Mixture-of-Agents for
       complex tasks.
    6. **CROSS_REVIEW** -- Each model's output reviewed by a different model.
    7. **INTEGRATE** -- Merge all outputs, resolve conflicts.

    Advanced features (controlled via ``SwarmConfig``):
        - **Multi-round debate** (Du et al. ICML 2024): Models debate the plan
          before execution, improving reasoning by 5-10%.
        - **Mixture-of-Agents** (Wang et al. 2024): Parallel generation + fusion
          for complex tasks — ensembles beat single models.
        - **Confidence cascading** (Chen et al. 2023): Try cheap models first,
          escalate only when uncertain — 50-80% cost savings.
        - **Tool-use**: Execution models can call file/terminal tools.

    Each phase logs progress and tracks cost independently.  The pipeline
    can be interrupted at any phase boundary.

    Args:
        engine: Completion engine for all LLM calls (must be mock-safe).
        registry: Provider registry for model discovery.
        config: Optional swarm configuration controlling advanced features.
        tool_registry: Optional tool registry for execution-phase tool-use.
        error_intelligence: Optional AEI engine for learning from failures and
            recommending fix strategies during fix cycles and research.
        context_manager: Optional smart context budget manager for estimating
            token counts and truncating prompts that exceed model windows.
    """

    def __init__(
        self,
        engine: CompletionEngine,
        registry: ProviderRegistry,
        config: SwarmConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        error_intelligence: AdaptiveExecutionIntelligence | None = None,
        context_manager: SmartContextBudgetManager | None = None,
    ) -> None:
        self._engine = engine
        self._registry = registry
        self._config = config or SwarmConfig()
        self._tool_registry = tool_registry
        self._error_intelligence = error_intelligence
        self._context_manager = context_manager
        self._model_pool = ModelPool(registry)
        self._decomposer = TaskDecomposer(engine, self._model_pool)
        self._researcher = ResearchSwarm(engine, self._model_pool)
        self._plan_reviewer = PlanReviewer(engine, self._model_pool)
        self._cross_reviewer = CrossReviewer(engine, self._model_pool)
        self._interrupted = False

        # Lazily initialised advanced engines (only created when first needed)
        self._debate_engine: DebateEngine | None = None
        self._cascade_engine: ConfidenceCascade | None = None
        self._moa_engine: MixtureOfAgents | None = None

        # Build tool schemas once if tool registry is provided
        self._tool_schemas: list[dict[str, Any]] | None = None
        if self._tool_registry and self._config.use_tools:
            try:
                raw = self._tool_registry.all_schemas()
                self._tool_schemas = [
                    {"type": "function", "function": s} for s in raw
                ]
            except Exception:
                logger.warning("swarm_tool_schema_build_failed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def orchestrate(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> SwarmPlan:
        """Run the full seven-phase orchestration pipeline.

        Budget enforcement: before each phase the orchestrator checks whether
        the remaining total budget and/or per-phase budget allow the phase to
        proceed.  If not, the pipeline stops gracefully and returns partial
        results with ``budget_exhausted=True``.

        Args:
            goal: Natural-language description of the objective.
            context: Optional repo/project context dict.

        Returns:
            Completed ``SwarmPlan`` with all phases' outputs.

        Raises:
            ValueError: If *goal* is empty.
        """
        if not goal or not goal.strip():
            raise ValueError("Goal must not be empty")

        context = context or {}
        plan = SwarmPlan(goal=goal)
        self._interrupted = False

        logger.info("swarm_started", goal_preview=goal[:100])

        # Phase 1: DECOMPOSE
        budget = self._check_budget(plan, SwarmPhase.DECOMPOSE)
        if budget == "stop":
            return plan
        if budget == "proceed":
            plan = await self._phase_decompose(plan, goal, context)
        if self._interrupted:
            return plan

        # Auto-scale budget based on task count (larger projects need more)
        if self._config.auto_scale_budget and self._config.total_budget is not None:
            task_count = len(plan.tasks)
            min_budget_for_tasks = task_count * 0.10  # $0.10 per task
            if min_budget_for_tasks > self._config.total_budget:
                old_budget = self._config.total_budget
                self._config.total_budget = min_budget_for_tasks
                logger.info(
                    "budget_auto_scaled",
                    old_budget=old_budget,
                    new_budget=min_budget_for_tasks,
                    task_count=task_count,
                )

        # Phase 2: RESEARCH
        budget = self._check_budget(plan, SwarmPhase.RESEARCH)
        if budget == "stop":
            return plan
        if budget == "proceed":
            plan = await self._phase_research(plan, goal, context)
        if self._interrupted:
            return plan

        # Phase 3: PLAN
        budget = self._check_budget(plan, SwarmPhase.PLAN)
        if budget == "stop":
            return plan
        if budget == "proceed":
            plan = await self._phase_plan(plan, goal)
        if self._interrupted:
            return plan

        # Phase 4: REVIEW
        budget = self._check_budget(plan, SwarmPhase.REVIEW)
        if budget == "stop":
            return plan
        if budget == "proceed":
            plan = await self._phase_review(plan, goal)
        if self._interrupted:
            return plan

        # Phase 5: EXECUTE
        budget = self._check_budget(plan, SwarmPhase.EXECUTE)
        if budget == "stop":
            return plan
        if budget == "proceed":
            plan = await self._phase_execute(plan)
        if self._interrupted:
            return plan

        # Phase 6: CROSS_REVIEW
        budget = self._check_budget(plan, SwarmPhase.CROSS_REVIEW)
        if budget == "stop":
            return plan
        if budget == "proceed":
            plan = await self._phase_cross_review(plan)
        if self._interrupted:
            return plan

        # Phase 7: INTEGRATE
        budget = self._check_budget(plan, SwarmPhase.INTEGRATE)
        if budget == "stop":
            return plan
        if budget == "proceed":
            plan = await self._phase_integrate(plan, goal)

        logger.info(
            "swarm_completed",
            total_cost=plan.total_cost,
            tasks_completed=sum(
                1 for t in plan.tasks if t.status == TaskStatus.COMPLETED
            ),
            tasks_total=len(plan.tasks),
        )
        return plan

    def interrupt(self) -> None:
        """Signal the orchestrator to stop after the current phase."""
        self._interrupted = True
        logger.info("swarm_interrupted")

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def _check_budget(self, plan: SwarmPlan, phase: SwarmPhase) -> str:
        """Check whether the budget allows ``phase`` to proceed.

        Three possible outcomes:

        * ``"proceed"`` -- budget is fine, run the phase.
        * ``"skip"`` -- per-phase cap exceeded; skip this phase but continue
          the pipeline with subsequent phases.
        * ``"stop"`` -- total pipeline budget exhausted; abort the pipeline
          and set ``plan.budget_exhausted = True``.

        Args:
            plan: Current plan with accumulated cost.
            phase: The phase about to run.

        Returns:
            One of ``"proceed"``, ``"skip"``, or ``"stop"``.
        """
        # --- Total budget check (already spent >= cap) ---
        total_budget = self._config.total_budget
        if total_budget is not None and plan.total_cost >= total_budget:
            plan.budget_exhausted = True
            logger.warning(
                "budget_exhausted_total",
                total_cost=plan.total_cost,
                budget=total_budget,
                skipped_phase=phase,
            )
            return "stop"

        # --- Total budget would be exceeded by this phase ---
        if total_budget is not None:
            estimated = self._estimate_phase_cost(phase)
            if plan.total_cost + estimated > total_budget:
                plan.budget_exhausted = True
                logger.warning(
                    "budget_would_exceed_total",
                    total_cost=plan.total_cost,
                    estimated_phase_cost=estimated,
                    budget=total_budget,
                    skipped_phase=phase,
                )
                return "stop"

        # --- Per-phase budget check ---
        phase_budgets = self._config.phase_budgets
        if phase_budgets is not None and phase in phase_budgets:
            estimated = self._estimate_phase_cost(phase)
            phase_cap = phase_budgets[phase]
            if estimated > phase_cap:
                logger.warning(
                    "budget_exceeded_phase",
                    phase=phase,
                    estimated_cost=estimated,
                    phase_budget=phase_cap,
                )
                return "skip"

        return "proceed"

    # ------------------------------------------------------------------
    # Lazy engine initialisation
    # ------------------------------------------------------------------

    def _get_debate_engine(self) -> DebateEngine:
        """Lazily create and return the debate engine."""
        if self._debate_engine is None:
            from prism.orchestrator.debate import DebateConfig, DebateEngine

            self._debate_engine = DebateEngine(
                engine=self._engine,
                model_pool=self._model_pool,
                config=DebateConfig(max_rounds=2, min_participants=2, max_participants=3),
            )
        return self._debate_engine

    def _get_cascade_engine(self) -> ConfidenceCascade:
        """Lazily create and return the confidence cascade engine."""
        if self._cascade_engine is None:
            from prism.orchestrator.cascade import CascadeConfig, ConfidenceCascade

            self._cascade_engine = ConfidenceCascade(
                engine=self._engine,
                model_pool=self._model_pool,
                config=CascadeConfig(
                    min_confidence=0.7,
                    max_escalations=3,
                    use_external_judge=True,
                    budget_limit=self._config.cascade_budget_per_task,
                ),
            )
        return self._cascade_engine

    def _get_moa_engine(self) -> MixtureOfAgents:
        """Lazily create and return the Mixture-of-Agents engine."""
        if self._moa_engine is None:
            from prism.orchestrator.moa import MixtureOfAgents, MoAConfig

            self._moa_engine = MixtureOfAgents(
                engine=self._engine,
                model_pool=self._model_pool,
                config=MoAConfig(num_proposers=3, num_layers=2, use_ranking=True),
            )
        return self._moa_engine

    # ------------------------------------------------------------------
    # Pipeline phases
    # ------------------------------------------------------------------

    async def _phase_decompose(
        self,
        plan: SwarmPlan,
        goal: str,
        context: dict[str, Any],
    ) -> SwarmPlan:
        """Phase 1: Decompose goal into subtasks."""
        logger.info("phase_start", phase=SwarmPhase.DECOMPOSE)

        try:
            tasks = await self._decomposer.decompose(goal, context)
            plan.tasks = tasks
            cost = self._estimate_phase_cost(SwarmPhase.DECOMPOSE)
            plan.phase_costs[SwarmPhase.DECOMPOSE] = cost
            plan.total_cost += cost
        except Exception as exc:
            logger.error("phase_failed", phase=SwarmPhase.DECOMPOSE, error=str(exc))
            # Fallback: single task
            plan.tasks = [SwarmTask(description=goal.strip(), complexity="medium")]

        logger.info(
            "phase_complete",
            phase=SwarmPhase.DECOMPOSE,
            task_count=len(plan.tasks),
        )
        return plan

    async def _phase_research(
        self,
        plan: SwarmPlan,
        goal: str,
        context: dict[str, Any],
    ) -> SwarmPlan:
        """Phase 2: Parallel research using cheap models."""
        logger.info("phase_start", phase=SwarmPhase.RESEARCH)

        # Generate research questions from task descriptions
        questions = self._generate_research_questions(plan.tasks, goal)
        if not questions:
            logger.info("phase_skipped", phase=SwarmPhase.RESEARCH, reason="no_questions")
            return plan

        # Convert context to string context for research
        str_context = {
            k: str(v)[:500] for k, v in context.items()
        } if context else None

        # Query AEI for relevant past error patterns to inform research
        if self._error_intelligence is not None:
            aei_context = self._get_aei_research_context(goal)
            if aei_context and str_context is not None:
                str_context["aei_error_patterns"] = aei_context
            elif aei_context:
                str_context = {"aei_error_patterns": aei_context}

        try:
            findings = await self._researcher.research(questions, str_context)
            plan.research_findings = findings
            cost = self._estimate_phase_cost(SwarmPhase.RESEARCH)
            plan.phase_costs[SwarmPhase.RESEARCH] = cost
            plan.total_cost += cost
        except Exception as exc:
            logger.error("phase_failed", phase=SwarmPhase.RESEARCH, error=str(exc))

        logger.info(
            "phase_complete",
            phase=SwarmPhase.RESEARCH,
            findings=len(plan.research_findings),
        )
        return plan

    async def _phase_plan(self, plan: SwarmPlan, goal: str) -> SwarmPlan:
        """Phase 3: Smart model synthesises a plan from research."""
        logger.info("phase_start", phase=SwarmPhase.PLAN)

        model = self._model_pool.get_planning_model()

        # Build context from research findings
        research_summary = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in plan.research_findings.items()
        )

        task_summary = "\n".join(
            f"- [{t.complexity}] {t.description}" for t in plan.tasks
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Subtasks identified:\n{task_summary}\n\n"
                    f"Research findings:\n{research_summary or 'None'}"
                ),
            },
        ]

        try:
            result = await self._engine.complete(
                messages=messages,
                model=model,
                temperature=0.4,
                max_tokens=4096,
            )
            plan.plan_text = result.content
            cost = result.cost_usd
            plan.phase_costs[SwarmPhase.PLAN] = cost
            plan.total_cost += cost
        except Exception as exc:
            logger.error("phase_failed", phase=SwarmPhase.PLAN, error=str(exc))
            plan.plan_text = f"Planning failed: {exc}"

        logger.info("phase_complete", phase=SwarmPhase.PLAN, model=model)
        return plan

    async def _phase_review(self, plan: SwarmPlan, goal: str) -> SwarmPlan:
        """Phase 4: Different model critiques the plan.

        When ``use_debate`` is enabled, uses multi-round debate (Du et al.)
        where multiple models discuss the plan, converging toward a consensus
        critique.  Falls back to single-model review on debate failure.
        """
        logger.info("phase_start", phase=SwarmPhase.REVIEW)

        review_cost = 0.0

        # Try multi-round debate if enabled
        if self._config.use_debate:
            try:
                debate_engine = self._get_debate_engine()
                debate_topic = (
                    f"Review this execution plan for the goal: {goal}\n\n"
                    f"Plan:\n{plan.plan_text[:3000]}\n\n"
                    "Debate whether this plan is correct, complete, and optimal. "
                    "Identify issues, missing edge cases, and improvements."
                )
                debate_result = await debate_engine.debate(
                    topic=debate_topic,
                    context="\n".join(
                        f"- {t.description} [{t.complexity}]" for t in plan.tasks
                    ),
                )
                plan.debate_result = debate_result
                plan.review_notes = debate_result.final_synthesis
                review_cost = debate_result.total_cost
                logger.info(
                    "debate_review_complete",
                    rounds=len(debate_result.rounds),
                    consensus=debate_result.consensus_score,
                    participants=len(debate_result.participating_models),
                )
            except Exception as exc:
                logger.warning(
                    "debate_review_failed_fallback_to_single",
                    error=str(exc),
                )
                # Fall through to single-model review below
                plan.debate_result = None

        # Single-model review (either as primary or as fallback)
        if not plan.review_notes:
            try:
                review_notes = await self._plan_reviewer.review(plan.plan_text, goal)
                plan.review_notes = review_notes
                review_cost = self._estimate_phase_cost(SwarmPhase.REVIEW)
            except Exception as exc:
                logger.error("phase_failed", phase=SwarmPhase.REVIEW, error=str(exc))
                plan.review_notes = f"Review failed: {exc}"

        plan.phase_costs[SwarmPhase.REVIEW] = review_cost
        plan.total_cost += review_cost
        logger.info("phase_complete", phase=SwarmPhase.REVIEW)
        return plan

    async def _phase_execute(self, plan: SwarmPlan) -> SwarmPlan:
        """Phase 5: Execute subtasks assigned to models by complexity.

        Advanced execution strategies (activated via ``SwarmConfig``):

        - **Confidence cascading** (FrugalGPT): Try cheap model first, escalate
          when confidence is low.  Saves 50-80% cost on average.
        - **Mixture-of-Agents**: For complex tasks, N models generate in parallel,
          outputs are ranked and fused into a superior result.
        - **Tool-use**: Models receive tool schemas (read_file, write_file,
          execute_command, etc.) so they can interact with the codebase.
        - **Fallback chains**: When a model fails, alternative models from the
          same tier and then escalated tiers are tried up to ``max_retries``.
        """
        logger.info("phase_start", phase=SwarmPhase.EXECUTE)

        # Sort tasks in dependency order
        ordered = self._dependency_sort(plan.tasks)

        for task in ordered:
            if self._interrupted:
                break

            # Skip if dependencies haven't completed
            if not self._dependencies_met(task, plan.tasks):
                task.status = TaskStatus.FAILED
                task.result = "Dependencies not met"
                continue

            model = self._model_pool.get_execution_model(task.complexity)
            task.assigned_model = model
            task.status = TaskStatus.RUNNING
            task.models_tried = [model]
            task.retry_count = 0

            logger.info(
                "task_executing",
                task_id=task.id,
                description=task.description[:60],
                model=model,
                strategy=self._execution_strategy(task),
            )

            try:
                result, cost = await self._execute_task_advanced(task, model, plan)
                task.result = result
                task.status = TaskStatus.COMPLETED
                plan.total_cost += cost
            except Exception as exc:
                logger.warning(
                    "task_execution_failed_trying_fallback",
                    task_id=task.id,
                    model=model,
                    error=str(exc),
                )
                # Attempt fallback retries
                succeeded = await self._attempt_fallback(task, plan)
                if not succeeded:
                    task.status = TaskStatus.FAILED
                    task.result = (
                        f"Execution failed after {task.retry_count + 1} "
                        f"attempt(s): {exc}"
                    )
                    logger.error(
                        "task_execution_failed",
                        task_id=task.id,
                        error=str(exc),
                        models_tried=task.models_tried,
                    )

                    # Record failure pattern in AEI
                    if self._error_intelligence is not None:
                        try:
                            fp = self._error_intelligence.fingerprint_error(
                                error_type=type(exc).__name__,
                                stack_trace=str(exc)[:500],
                                file_path=", ".join(task.files_changed) or "unknown",
                                function_name=task.description[:100],
                            )
                            self._aei_record_attempt(
                                fp, model, 0,
                                "failure",
                                f"Task execution failed: {exc}",
                            )
                        except Exception:
                            logger.debug("aei_record_failed")

        completed = sum(1 for t in plan.tasks if t.status == TaskStatus.COMPLETED)
        logger.info(
            "phase_complete",
            phase=SwarmPhase.EXECUTE,
            completed=completed,
            total=len(plan.tasks),
        )
        return plan

    async def _attempt_fallback(
        self,
        task: SwarmTask,
        plan: SwarmPlan,
    ) -> bool:
        """Attempt fallback execution with alternative models.

        Tries alternative models from the same tier first, then escalates
        to higher tiers.  Limited to ``self._config.max_retries`` attempts.

        Args:
            task: The task that failed primary execution.
            plan: The parent plan for context.

        Returns:
            True if a fallback model succeeded, False otherwise.
        """
        max_retries = self._config.max_retries
        fallback_models = self._model_pool.get_fallback_models(
            task.complexity,
            exclude=task.models_tried,
        )

        for fallback_model in fallback_models:
            if task.retry_count >= max_retries:
                logger.info(
                    "fallback_max_retries_reached",
                    task_id=task.id,
                    retry_count=task.retry_count,
                    max_retries=max_retries,
                )
                break

            task.retry_count += 1
            task.models_tried.append(fallback_model)
            task.assigned_model = fallback_model

            logger.info(
                "fallback_attempt",
                task_id=task.id,
                retry=task.retry_count,
                model=fallback_model,
                models_tried=task.models_tried,
            )

            try:
                result, cost = await self._execute_task_advanced(
                    task, fallback_model, plan,
                )
                task.result = result
                task.status = TaskStatus.COMPLETED
                plan.total_cost += cost
                logger.info(
                    "fallback_succeeded",
                    task_id=task.id,
                    model=fallback_model,
                    retry=task.retry_count,
                )
                return True
            except Exception as fallback_exc:
                logger.warning(
                    "fallback_attempt_failed",
                    task_id=task.id,
                    model=fallback_model,
                    retry=task.retry_count,
                    error=str(fallback_exc),
                )
                continue

        return False

    async def _phase_cross_review(self, plan: SwarmPlan) -> SwarmPlan:
        """Phase 6: Cross-review each task's output."""
        logger.info("phase_start", phase=SwarmPhase.CROSS_REVIEW)

        try:
            reviews = await self._cross_reviewer.review_all(plan.tasks)
            plan.cross_reviews = reviews

            # Track errors that need fix cycles
            errors = [r for r in reviews if r.severity == ReviewSeverity.ERROR]
            if errors:
                logger.warning(
                    "cross_review_errors",
                    error_count=len(errors),
                    task_ids=[r.task_id for r in errors],
                )
                # Trigger fix cycle for errored tasks
                await self._fix_cycle(plan, errors)

            cost = self._estimate_phase_cost(SwarmPhase.CROSS_REVIEW)
            plan.phase_costs[SwarmPhase.CROSS_REVIEW] = cost
            plan.total_cost += cost
        except Exception as exc:
            logger.error("phase_failed", phase=SwarmPhase.CROSS_REVIEW, error=str(exc))

        logger.info("phase_complete", phase=SwarmPhase.CROSS_REVIEW)
        return plan

    async def _phase_integrate(self, plan: SwarmPlan, goal: str) -> SwarmPlan:
        """Phase 7: Merge all outputs and resolve conflicts."""
        logger.info("phase_start", phase=SwarmPhase.INTEGRATE)

        model = self._model_pool.get_planning_model()

        # Build summary of all task outputs
        task_outputs = "\n\n".join(
            f"Task: {t.description}\nStatus: {t.status}\nOutput: {t.result or 'N/A'}"
            for t in plan.tasks
        )

        review_summary = ""
        if plan.cross_reviews:
            review_lines = [
                f"- [{r.severity}] Task {r.task_id[:8]}: {r.comments[:200]}"
                for r in plan.cross_reviews
            ]
            review_summary = "\n\nCross-review findings:\n" + "\n".join(review_lines)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _INTEGRATE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Task outputs:\n{task_outputs}"
                    f"{review_summary}"
                ),
            },
        ]

        try:
            result = await self._engine.complete(
                messages=messages,
                model=model,
                temperature=0.3,
                max_tokens=2048,
            )
            # Append integration summary to plan_text
            plan.plan_text += f"\n\n--- Integration Summary ---\n{result.content}"
            cost = result.cost_usd
            plan.phase_costs[SwarmPhase.INTEGRATE] = cost
            plan.total_cost += cost
        except Exception as exc:
            logger.error("phase_failed", phase=SwarmPhase.INTEGRATE, error=str(exc))

        logger.info("phase_complete", phase=SwarmPhase.INTEGRATE)
        return plan

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _execution_strategy(self, task: SwarmTask) -> str:
        """Determine which execution strategy to use for a task.

        Returns one of ``"moa"``, ``"cascade"``, or ``"direct"``.
        """
        if (
            self._config.use_moa
            and task.complexity == self._config.moa_complexity_threshold
        ):
            return "moa"
        if self._config.use_cascade:
            return "cascade"
        return "direct"

    async def _execute_task_advanced(
        self,
        task: SwarmTask,
        model: str,
        plan: SwarmPlan,
    ) -> tuple[str, float]:
        """Execute a subtask using the best available strategy.

        Selects between Mixture-of-Agents, confidence cascading, or direct
        execution based on task complexity and ``SwarmConfig``.

        Args:
            task: The task to execute.
            model: Default LiteLLM model ID.
            plan: The parent plan for context.

        Returns:
            Tuple of (output_text, cost_usd).
        """
        prompt = self._build_task_prompt(task, plan)
        context = plan.plan_text[:2000]
        strategy = self._execution_strategy(task)

        if strategy == "moa":
            return await self._execute_with_moa(task, prompt, context, plan)

        if strategy == "cascade":
            return await self._execute_with_cascade(task, prompt, context, plan)

        return await self._execute_direct(task, model, prompt, plan)

    async def _execute_with_moa(
        self,
        task: SwarmTask,
        prompt: str,
        context: str,
        plan: SwarmPlan,
    ) -> tuple[str, float]:
        """Execute a complex task using Mixture-of-Agents parallel generation.

        Multiple models generate independently, outputs are ranked via
        pairwise comparison (LLM-Blender), then fused into a single superior
        response.  Falls back to direct execution on failure.

        Args:
            task: The task being executed.
            prompt: The formatted task prompt.
            context: Plan context string.
            plan: Parent plan for metadata storage.

        Returns:
            Tuple of (output_text, cost_usd).
        """
        try:
            moa = self._get_moa_engine()
            moa_result = await moa.generate(prompt=prompt, context=context)
            plan.moa_results[task.id] = moa_result
            logger.info(
                "task_moa_complete",
                task_id=task.id,
                layers=len(moa_result.layers),
                models=len(moa_result.participating_models),
                cost=moa_result.total_cost,
            )
            return moa_result.final_output, moa_result.total_cost
        except Exception as exc:
            logger.warning(
                "task_moa_failed_fallback_to_cascade",
                task_id=task.id,
                error=str(exc),
            )
            # Fallback: try cascade, then direct
            if self._config.use_cascade:
                return await self._execute_with_cascade(
                    task, prompt, context, plan,
                )
            model = self._model_pool.get_execution_model(task.complexity)
            return await self._execute_direct(task, model, prompt, plan)

    async def _execute_with_cascade(
        self,
        task: SwarmTask,
        prompt: str,
        context: str,
        plan: SwarmPlan,
    ) -> tuple[str, float]:
        """Execute a task using FrugalGPT confidence cascading.

        Tries the cheapest model first.  If confidence is below threshold,
        escalates to progressively more expensive models until confidence is
        acceptable or budget is exhausted.

        Args:
            task: The task being executed.
            prompt: The formatted task prompt.
            context: Plan context string.
            plan: Parent plan for metadata storage.

        Returns:
            Tuple of (output_text, cost_usd).
        """
        try:
            cascade = self._get_cascade_engine()
            cascade_result = await cascade.execute(prompt=prompt, context=context)
            plan.cascade_results[task.id] = cascade_result
            logger.info(
                "task_cascade_complete",
                task_id=task.id,
                accepted_level=cascade_result.accepted_at_level,
                attempts=len(cascade_result.attempts),
                confidence=cascade_result.confidence.score,
                cost=cascade_result.total_cost,
                saved=cascade_result.cost_saved_vs_premium,
            )
            return cascade_result.output, cascade_result.total_cost
        except Exception as exc:
            logger.warning(
                "task_cascade_failed_fallback_to_direct",
                task_id=task.id,
                error=str(exc),
            )
            model = self._model_pool.get_execution_model(task.complexity)
            return await self._execute_direct(task, model, prompt, plan)

    async def _execute_with_tools(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_iterations: int = 10,
    ) -> tuple[str, float]:
        """Run a completion loop that executes tool calls iteratively.

        When the model returns ``tool_calls``, each tool is looked up in
        ``self._tool_registry``, executed, and the results fed back as
        ``role=tool`` messages.  The loop continues until the model emits a
        final text response (no more tool calls) or ``max_iterations`` is
        reached.

        Args:
            messages: Initial chat messages (system + user at minimum).
            model: LiteLLM model identifier.
            max_iterations: Maximum number of LLM round-trips (default 10).

        Returns:
            Tuple of (final_text_content, accumulated_cost_usd).
        """
        total_cost = 0.0
        last_content = ""

        for _iteration in range(max_iterations):
            result = await self._engine.complete(
                messages=messages,
                model=model,
                temperature=0.3,
                max_tokens=4096,
                tools=self._tool_schemas,
            )
            total_cost += result.cost_usd
            last_content = result.content or ""

            # If no tool calls, we have the final response
            if not result.tool_calls:
                return last_content, total_cost

            # Append assistant message with tool_calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": result.content or None,
                "tool_calls": result.tool_calls,
            }
            messages.append(assistant_msg)

            # Execute each tool and feed results back
            for tc in result.tool_calls:
                tc_id = tc.get("id", "")
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}

                tool_output = self._run_single_tool(tool_name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_output,
                })

        # Exhausted iterations — return whatever content we have
        logger.warning(
            "tool_loop_max_iterations",
            iterations=max_iterations,
            model=model,
        )
        return last_content, total_cost

    def _run_single_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Execute a single tool by name and return its output as a string.

        If the tool is not found or execution fails, an error string is
        returned rather than raising -- the model can recover from tool
        failures gracefully.

        Args:
            tool_name: Registered tool name (e.g. ``'read_file'``).
            arguments: Tool arguments dict.

        Returns:
            Tool output string (success or error message).
        """
        if self._tool_registry is None:
            return f"Error: No tool registry available to execute '{tool_name}'"

        try:
            tool = self._tool_registry.get_tool(tool_name)
        except Exception:
            logger.warning("swarm_tool_not_found", tool=tool_name)
            return f"Error: Tool '{tool_name}' not found"

        try:
            tool_result = tool.execute(arguments)
            if tool_result.success:
                return tool_result.output
            return f"Error: {tool_result.error or 'Tool execution failed'}"
        except Exception as exc:
            logger.warning(
                "swarm_tool_execution_error",
                tool=tool_name,
                error=str(exc),
            )
            return f"Error executing tool '{tool_name}': {exc}"

    async def _execute_direct(
        self,
        task: SwarmTask,
        model: str,
        prompt: str,
        plan: SwarmPlan,
    ) -> tuple[str, float]:
        """Execute a task directly with a single model.

        When tools are available and a tool registry is configured, delegates
        to :meth:`_execute_with_tools` which runs an iterative tool-use loop.
        Otherwise falls back to a single completion call.

        Args:
            task: The task being executed.
            model: LiteLLM model ID.
            prompt: The formatted task prompt.
            plan: Parent plan (unused, kept for interface consistency).

        Returns:
            Tuple of (output_text, cost_usd).
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _EXECUTE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Use the iterative tool loop when tools + registry are both available
        if self._tool_schemas and self._tool_registry:
            return await self._execute_with_tools(messages, model)

        result = await self._engine.complete(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=4096,
            tools=self._tool_schemas,
        )
        return result.content, result.cost_usd

    def _build_task_prompt(self, task: SwarmTask, plan: SwarmPlan) -> str:
        """Build the full prompt for a task execution.

        Includes task description, complexity, files to modify, plan context,
        review feedback, and completed sibling results.  When a context
        manager is available, estimates token count and truncates the prompt
        if it would exceed the model's context window.

        Args:
            task: The task to build a prompt for.
            plan: The parent plan for context.

        Returns:
            Formatted prompt string.
        """
        parts: list[str] = [
            f"Task: {task.description}",
            f"Complexity: {task.complexity}",
            f"Files to modify: {', '.join(task.files_changed) or 'none'}",
        ]

        # Scale context limits based on model's context window.
        # Bigger models (200k+ tokens) get proportionally more plan context
        # so large projects benefit from full plan detail, while small models
        # still get the essentials without exceeding their window.
        plan_limit, review_limit = self._context_limits(task)

        # Include plan context
        if plan.plan_text:
            parts.append(f"\nPlan context:\n{plan.plan_text[:plan_limit]}")

        # Include review feedback so execution benefits from debate insights
        if plan.review_notes:
            parts.append(f"\nReview feedback:\n{plan.review_notes[:review_limit]}")

        # Include completed sibling task results
        completed_siblings = [
            t for t in plan.tasks
            if t.id != task.id and t.status == TaskStatus.COMPLETED and t.result
        ]
        if completed_siblings:
            sibling_text = "\n".join(
                f"- {t.description}: {(t.result or '')[:300]}"
                for t in completed_siblings[:5]
            )
            parts.append(f"\nPrior task results:\n{sibling_text}")

        # If tools are available, mention it
        if self._tool_schemas:
            parts.append(
                "\nYou have access to tools (read_file, write_file, edit_file, "
                "execute_command, etc.).  Use them to implement changes directly."
            )

        prompt = "\n".join(parts)

        # Use context manager to estimate tokens and truncate if needed
        if self._context_manager is not None:
            prompt = self._truncate_prompt_to_budget(prompt, task, plan)

        return prompt

    @staticmethod
    def _context_limits(task: SwarmTask) -> tuple[int, int]:
        """Return (plan_char_limit, review_char_limit) scaled to the model.

        Models with large context windows (200k+) get up to 8000 chars of plan
        context, while small local models (8k window) get 1000 chars.  This
        ensures the orchestrator works well for both large multi-file projects
        (where plan detail is valuable) and small single-file tasks (where
        we must stay within tight context budgets).

        Args:
            task: The task (used to determine the assigned model).

        Returns:
            Tuple of (plan_text_char_limit, review_notes_char_limit).
        """
        try:
            from prism.intelligence.context_budget import (
                DEFAULT_CONTEXT_WINDOW,
                MODEL_CONTEXT_WINDOWS,
            )

            model = task.assigned_model or "claude-sonnet-4-20250514"
            window = MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)
        except Exception:
            window = 128_000

        # Scale linearly: 8k window → 1000/500, 128k → 4000/2000, 200k → 8000/4000
        plan_limit = max(1000, min(8000, window // 25))
        review_limit = max(500, min(4000, window // 50))
        return plan_limit, review_limit

    def _truncate_prompt_to_budget(
        self,
        prompt: str,
        task: SwarmTask,
        plan: SwarmPlan,
    ) -> str:
        """Truncate *prompt* if it exceeds the model's context budget.

        Uses the :class:`SmartContextBudgetManager` to estimate tokens
        and enforces a 50% ceiling (leaving room for the response and
        system prompt).  Tracks total tokens in ``plan.context_tokens_used``.

        Args:
            prompt: The raw prompt text.
            task: The task being executed (for model lookup).
            plan: The parent plan (for token accounting).

        Returns:
            The (possibly truncated) prompt.
        """
        if self._context_manager is None:
            return prompt

        try:
            from prism.intelligence.context_budget import (
                MODEL_CONTEXT_WINDOWS,
                estimate_tokens,
            )

            model = task.assigned_model or "claude-sonnet-4-20250514"
            context_window = MODEL_CONTEXT_WINDOWS.get(model, 128_000)
            # Reserve 50% for response + system prompt
            max_prompt_tokens = int(context_window * 0.50)

            token_count = estimate_tokens(prompt)
            plan.context_tokens_used += token_count

            if token_count > max_prompt_tokens:
                # Truncate to fit budget — rough char estimate
                ratio = max_prompt_tokens / max(token_count, 1)
                max_chars = int(len(prompt) * ratio)
                prompt = prompt[:max_chars] + "\n\n[... prompt truncated to fit context budget]"
                new_tokens = estimate_tokens(prompt)
                # Adjust the tracked total
                plan.context_tokens_used += new_tokens - token_count
                logger.info(
                    "prompt_truncated_to_budget",
                    task_id=task.id,
                    original_tokens=token_count,
                    truncated_tokens=new_tokens,
                    max_tokens=max_prompt_tokens,
                )
        except Exception as exc:
            logger.warning("context_budget_estimation_failed", error=str(exc))

        return prompt

    async def _fix_cycle(
        self,
        plan: SwarmPlan,
        error_reviews: list[CrossReview],
    ) -> None:
        """Re-execute tasks that received error-level cross-reviews.

        Sends the original task + the review feedback back to the executor
        for a single retry attempt.  When AEI is available, consults it for
        known error patterns and records the outcome of each fix attempt.

        Args:
            plan: The parent plan.
            error_reviews: Reviews with severity=error.
        """
        task_map = {t.id: t for t in plan.tasks}

        for review in error_reviews:
            task = task_map.get(review.task_id)
            if task is None:
                continue

            model = self._model_pool.get_execution_model(task.complexity)

            # Consult AEI for known error patterns if available
            aei_advice = ""
            aei_fingerprint = None
            if self._error_intelligence is not None:
                try:
                    aei_fingerprint = self._error_intelligence.fingerprint_error(
                        error_type="CrossReviewError",
                        stack_trace=review.comments[:500],
                        file_path=", ".join(task.files_changed) or "unknown",
                        function_name=task.description[:100],
                    )
                    recommendation = self._error_intelligence.recommend_strategy(
                        aei_fingerprint,
                    )
                    aei_advice = (
                        f"\n\nAEI recommendation (confidence "
                        f"{recommendation.confidence:.0%}): "
                        f"Use strategy '{recommendation.strategy.value}'. "
                        f"Reasoning: {recommendation.reasoning}"
                    )
                    logger.info(
                        "fix_cycle_aei_consulted",
                        task_id=task.id,
                        strategy=recommendation.strategy.value,
                        confidence=recommendation.confidence,
                    )
                except Exception as aei_exc:
                    logger.warning(
                        "fix_cycle_aei_consult_failed",
                        task_id=task.id,
                        error=str(aei_exc),
                    )

            messages: list[dict[str, str]] = [
                {"role": "system", "content": _EXECUTE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"RETRY — your previous output had issues.\n\n"
                        f"Task: {task.description}\n"
                        f"Previous output:\n{task.result}\n\n"
                        f"Review feedback:\n{review.comments}\n\n"
                        f"Please fix the issues and provide corrected output."
                        f"{aei_advice}"
                    ),
                },
            ]

            try:
                result = await self._engine.complete(
                    messages=messages,
                    model=model,
                    temperature=0.2,
                    max_tokens=4096,
                )
                task.result = result.content
                logger.info("fix_cycle_success", task_id=task.id)

                # Record success in AEI
                self._aei_record_attempt(
                    aei_fingerprint, model, len(review.comments),
                    "success", "fix_cycle retry succeeded",
                )
            except Exception as exc:
                logger.warning(
                    "fix_cycle_failed",
                    task_id=task.id,
                    error=str(exc),
                )

                # Record failure in AEI
                self._aei_record_attempt(
                    aei_fingerprint, model, len(review.comments),
                    "failure", f"fix_cycle retry failed: {exc}",
                )

    def _aei_record_attempt(
        self,
        fingerprint: Any,
        model: str,
        context_size: int,
        outcome: str,
        reasoning: str,
    ) -> None:
        """Record a fix attempt in AEI if available.  Best-effort; never raises.

        Args:
            fingerprint: ErrorFingerprint from AEI, or None.
            model: LiteLLM model identifier.
            context_size: Approximate token count of context supplied.
            outcome: ``"success"`` or ``"failure"``.
            reasoning: Human-readable rationale.
        """
        if self._error_intelligence is None or fingerprint is None:
            return
        try:
            from prism.intelligence.aei import FixStrategy

            self._error_intelligence.record_attempt(
                fingerprint=fingerprint,
                strategy=FixStrategy.FULL_REWRITE,
                model=model,
                context_size=context_size,
                outcome=outcome,
                reasoning=reasoning,
            )
        except Exception:
            logger.debug("aei_record_failed")

    def _get_aei_research_context(self, goal: str) -> str:
        """Query AEI for relevant past error patterns to inform research.

        Builds a short summary of the most common error types and their
        successful fix strategies from the AEI database.

        Args:
            goal: The user's goal (used for logging context only).

        Returns:
            A string summarising known error patterns, or empty string.
        """
        if self._error_intelligence is None:
            return ""
        try:
            stats = self._error_intelligence.get_stats()
            if stats.total_attempts == 0:
                return ""

            parts: list[str] = [
                f"Known error patterns ({stats.total_attempts} past attempts, "
                f"{stats.success_rate:.0%} success rate):",
            ]
            for error_type, count in stats.top_error_types[:5]:
                parts.append(f"  - {error_type}: {count} occurrences")

            if stats.strategies_used:
                best_strategy = max(
                    stats.strategies_used, key=stats.strategies_used.get,  # type: ignore[arg-type]
                )
                parts.append(f"  Most used strategy: {best_strategy}")

            logger.debug(
                "aei_research_context_built",
                goal_preview=goal[:60],
                patterns=len(stats.top_error_types),
            )
            return "\n".join(parts)
        except Exception as exc:
            logger.warning("aei_research_context_failed", error=str(exc))
            return ""

    @staticmethod
    def _dependency_sort(tasks: list[SwarmTask]) -> list[SwarmTask]:
        """Topological sort of tasks by their dependencies.

        Tasks with no dependencies come first.  Cycles are broken
        arbitrarily by falling back to input order.

        Args:
            tasks: Unsorted task list.

        Returns:
            Tasks in dependency-satisfying order.
        """
        task_map = {t.id: t for t in tasks}
        visited: set[str] = set()
        result: list[SwarmTask] = []

        def _visit(task_id: str) -> None:
            if task_id in visited:
                return
            visited.add(task_id)
            task = task_map.get(task_id)
            if task is None:
                return
            for dep_id in task.dependencies:
                if dep_id not in visited:
                    _visit(dep_id)
            result.append(task)

        for task in tasks:
            _visit(task.id)

        return result

    @staticmethod
    def _dependencies_met(task: SwarmTask, all_tasks: list[SwarmTask]) -> bool:
        """Check whether all dependencies of a task have completed.

        Args:
            task: The task to check.
            all_tasks: All tasks in the plan.

        Returns:
            True if all dependencies are completed (or task has none).
        """
        if not task.dependencies:
            return True
        task_map = {t.id: t for t in all_tasks}
        for dep_id in task.dependencies:
            dep = task_map.get(dep_id)
            if dep is None or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    @staticmethod
    def _generate_research_questions(
        tasks: list[SwarmTask],
        goal: str,
    ) -> list[str]:
        """Generate research questions from subtask descriptions.

        Each unique task generates a contextualised research question.
        Limits to 8 questions maximum.

        Args:
            tasks: Decomposed subtasks.
            goal: Original goal for context.

        Returns:
            List of research question strings.
        """
        questions: list[str] = []
        seen: set[str] = set()

        for task in tasks:
            q = f"What are the best practices for: {task.description}?"
            if q not in seen:
                seen.add(q)
                questions.append(q)

        # Cap at 8 to limit cost
        return questions[:8]

    @staticmethod
    def _estimate_phase_cost(phase: SwarmPhase) -> float:
        """Estimate cost for a phase (rough heuristic).

        Real cost tracking happens via CompletionEngine's cost tracker.
        This provides a conservative estimate for budget planning.

        Args:
            phase: The pipeline phase.

        Returns:
            Estimated cost in USD.
        """
        # Conservative per-phase estimates based on typical token usage
        estimates: dict[str, float] = {
            SwarmPhase.DECOMPOSE: 0.005,
            SwarmPhase.RESEARCH: 0.002,
            SwarmPhase.PLAN: 0.010,
            SwarmPhase.REVIEW: 0.008,
            SwarmPhase.EXECUTE: 0.003,
            SwarmPhase.CROSS_REVIEW: 0.004,
            SwarmPhase.INTEGRATE: 0.005,
        }
        return estimates.get(phase, 0.005)

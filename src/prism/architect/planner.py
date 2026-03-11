"""Plan generation for Architect Mode.

Decomposes a user request into discrete, ordered execution steps using
rule-based heuristics.  In production, the planning step would call a
premium model; the current implementation uses keyword-based decomposition
so the module is fully functional without any API calls.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog

from prism.cost.pricing import MODEL_PRICING

if TYPE_CHECKING:
    from prism.config.settings import Settings
    from prism.cost.tracker import CostTracker

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


class StepStatus(StrEnum):
    """Lifecycle status of a single plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single executable step inside a :class:`Plan`."""

    id: str
    order: int
    description: str
    tool_calls: list[dict[str, object]]
    estimated_tokens: int
    status: StepStatus
    result: str | None = None
    error: str | None = None


@dataclass
class Plan:
    """An execution plan created by :class:`ArchitectPlanner`."""

    id: str
    created_at: str
    description: str
    steps: list[PlanStep]
    planning_model: str
    execution_model: str
    estimated_total_cost: float
    status: str  # draft, approved, in_progress, completed, failed, rolled_back
    git_checkpoint: str | None = None


# ------------------------------------------------------------------
# Keyword-based decomposition helpers
# ------------------------------------------------------------------

# Patterns for detecting file-oriented work
_FILE_PATTERN = re.compile(
    r"""
    (?:create|add|write|update|modify|edit|delete|remove|rename)\s+
    (?:a\s+)?(?:file|module|class|function|test|script)\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MULTI_FILE_KEYWORDS: list[str] = [
    "and then",
    "after that",
    "next,",
    "finally,",
    "also ",
    "additionally",
    "then ",
    "step ",
    "1.",
    "2.",
    "3.",
]

_ANALYSIS_KEYWORDS: list[str] = [
    "analyze",
    "review",
    "audit",
    "check",
    "inspect",
    "examine",
    "evaluate",
]

_REFACTOR_KEYWORDS: list[str] = [
    "refactor",
    "restructure",
    "reorganize",
    "rewrite",
    "migrate",
    "move",
]

_TEST_KEYWORDS: list[str] = [
    "test",
    "tests",
    "spec",
    "unittest",
    "pytest",
]

_VALID_PLAN_STATUSES = frozenset({
    "draft",
    "approved",
    "in_progress",
    "completed",
    "failed",
    "rolled_back",
})

# Rough per-step token budget
_DEFAULT_STEP_TOKENS = 500


class ArchitectPlanner:
    """Creates execution plans from user requests.

    Uses a premium model identifier for planning metadata and a cheap
    model identifier for execution metadata.  The actual plan
    decomposition is rule-based (keyword matching and sentence splitting)
    so no API call is required.
    """

    DEFAULT_PLANNING_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_EXECUTION_MODEL = "deepseek/deepseek-chat"

    def __init__(
        self,
        settings: Settings,
        cost_tracker: CostTracker,
        *,
        planning_model: str | None = None,
        execution_model: str | None = None,
    ) -> None:
        """Initialise the planner.

        Args:
            settings: Application settings.
            cost_tracker: Cost tracker for budget awareness.
            planning_model: Override for the premium planning model.
            execution_model: Override for the cheap execution model.
        """
        self._settings = settings
        self._cost_tracker = cost_tracker
        self.planning_model = planning_model or self.DEFAULT_PLANNING_MODEL
        self.execution_model = execution_model or self.DEFAULT_EXECUTION_MODEL

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_plan(self, request: str, context: dict[str, object] | None = None) -> Plan:
        """Generate a step-by-step plan from a user request.

        The plan is created in *draft* status and must be approved before
        execution.

        Args:
            request: The user's natural-language request.
            context: Optional context dict (e.g. active files, repo info).

        Returns:
            A fully populated :class:`Plan` in ``draft`` status.

        Raises:
            ValueError: If *request* is empty or blank.
        """
        if not request or not request.strip():
            raise ValueError("Request must not be empty")

        context = context or {}
        steps = self._decompose(request, context)
        total_cost = self.estimate_cost_from_steps(steps)

        plan = Plan(
            id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            description=request.strip(),
            steps=steps,
            planning_model=self.planning_model,
            execution_model=self.execution_model,
            estimated_total_cost=total_cost,
            status="draft",
            git_checkpoint=None,
        )

        logger.info(
            "plan_created",
            plan_id=plan.id,
            step_count=len(steps),
            estimated_cost=f"${total_cost:.6f}",
        )
        return plan

    def estimate_cost(self, plan: Plan) -> float:
        """Estimate the total cost for executing a plan.

        Args:
            plan: The plan to estimate.

        Returns:
            Estimated cost in USD.
        """
        return self.estimate_cost_from_steps(plan.steps)

    def estimate_cost_from_steps(self, steps: list[PlanStep]) -> float:
        """Estimate cost from a list of steps.

        Uses the execution model pricing and the estimated token counts
        from each step.

        Args:
            steps: Steps to estimate.

        Returns:
            Estimated cost in USD.
        """
        pricing = MODEL_PRICING.get(self.execution_model)
        if pricing is None:
            return 0.0

        total_tokens = sum(s.estimated_tokens for s in steps)
        # Assume a roughly 1:1 input/output ratio for estimates
        input_tokens = total_tokens
        output_tokens = total_tokens
        input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m
        return input_cost + output_cost

    def format_plan_for_review(self, plan: Plan) -> str:
        """Format a plan as human-readable text for review.

        Args:
            plan: The plan to format.

        Returns:
            Multi-line string suitable for terminal display.
        """
        lines: list[str] = [
            f"Plan: {plan.description}",
            f"ID: {plan.id}",
            f"Status: {plan.status}",
            f"Planning model: {plan.planning_model}",
            f"Execution model: {plan.execution_model}",
            f"Estimated cost: ${plan.estimated_total_cost:.4f}",
            "",
            f"Steps ({len(plan.steps)}):",
            "-" * 50,
        ]

        for step in sorted(plan.steps, key=lambda s: s.order):
            status_tag = step.status.value.upper()
            desc = step.description
            if len(desc) > 80:
                desc = desc[:77] + "..."
            lines.append(
                f"  {step.order}. [{status_tag}] {desc}"
            )
            if step.tool_calls:
                for tc in step.tool_calls:
                    tool_name = tc.get("tool", "unknown")
                    lines.append(f"     -> {tool_name}")

        lines.append("-" * 50)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decompose(
        self, request: str, context: dict[str, object]
    ) -> list[PlanStep]:
        """Decompose a request into ordered plan steps.

        Uses keyword heuristics to identify sub-tasks.

        Args:
            request: User's request text.
            context: Additional context.

        Returns:
            Ordered list of :class:`PlanStep`.
        """
        request_lower = request.lower()
        steps: list[PlanStep] = []

        # Check if multi-step indicators are present
        is_multi = any(kw in request_lower for kw in _MULTI_FILE_KEYWORDS)

        if is_multi:
            steps = self._decompose_multi_step(request)
        else:
            steps = self._decompose_single(request, request_lower)

        # Guarantee at least one step
        if not steps:
            steps.append(
                self._make_step(1, f"Execute: {request.strip()}", [])
            )

        return steps

    def _decompose_multi_step(self, request: str) -> list[PlanStep]:
        """Split a multi-step request into individual steps.

        Splits on sentence boundaries and numbered-list markers.
        """
        # Try numbered list first
        numbered = re.split(r"\d+\.\s+", request)
        parts = [p.strip() for p in numbered if p.strip()]

        # Fall back to sentence splitting on conjunction keywords
        if len(parts) <= 1:
            split_pattern = r"(?:,?\s*(?:and then|after that|next,|finally,|then)\s+)"
            parts = re.split(split_pattern, request, flags=re.IGNORECASE)
            parts = [p.strip() for p in parts if p.strip()]

        steps: list[PlanStep] = []
        for idx, part in enumerate(parts, start=1):
            tool_calls = self._infer_tool_calls(part)
            steps.append(self._make_step(idx, part, tool_calls))

        return steps

    def _decompose_single(
        self, request: str, request_lower: str
    ) -> list[PlanStep]:
        """Decompose a single-step request, potentially adding analysis/test steps."""
        steps: list[PlanStep] = []
        order = 1

        # Prepend an analysis step for complex-sounding tasks
        needs_analysis = any(kw in request_lower for kw in _ANALYSIS_KEYWORDS)
        needs_refactor = any(kw in request_lower for kw in _REFACTOR_KEYWORDS)

        if needs_analysis or needs_refactor:
            steps.append(
                self._make_step(
                    order,
                    f"Analyze codebase for: {request.strip()}",
                    [{"tool": "read_file", "args": {}}],
                )
            )
            order += 1

        # Main execution step
        tool_calls = self._infer_tool_calls(request)
        steps.append(
            self._make_step(order, f"Execute: {request.strip()}", tool_calls)
        )
        order += 1

        # Append test step if relevant
        has_test_keyword = any(kw in request_lower for kw in _TEST_KEYWORDS)
        if needs_refactor and not has_test_keyword:
            steps.append(
                self._make_step(
                    order,
                    "Run tests to verify changes",
                    [{"tool": "run_command", "args": {"command": "pytest"}}],
                )
            )

        return steps

    def _infer_tool_calls(self, text: str) -> list[dict[str, object]]:
        """Infer likely tool calls from step text."""
        text_lower = text.lower()
        calls: list[dict[str, object]] = []

        if _FILE_PATTERN.search(text):
            calls.append({"tool": "write_file", "args": {}})
        elif any(kw in text_lower for kw in ("read", "look at", "show", "open")):
            calls.append({"tool": "read_file", "args": {}})

        if any(kw in text_lower for kw in ("run", "execute", "command", "shell")):
            calls.append({"tool": "run_command", "args": {}})

        return calls

    def _make_step(
        self,
        order: int,
        description: str,
        tool_calls: list[dict[str, object]],
    ) -> PlanStep:
        """Create a new PlanStep with a unique ID."""
        return PlanStep(
            id=str(uuid.uuid4()),
            order=order,
            description=description,
            tool_calls=tool_calls,
            estimated_tokens=_DEFAULT_STEP_TOKENS,
            status=StepStatus.PENDING,
            result=None,
            error=None,
        )

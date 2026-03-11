"""Plan generation for Architect Mode.

Decomposes a user request into discrete, ordered execution steps using
rule-based heuristics.  In production, the planning step would call a
premium model; the current implementation uses keyword-based decomposition
so the module is fully functional without any API calls.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
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


class PlanStatus(StrEnum):
    """Lifecycle status of a plan."""

    DRAFT = "draft"
    APPROVED = "approved"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# Risk level constants
RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"
_VALID_RISK_LEVELS = frozenset({RISK_LOW, RISK_MEDIUM, RISK_HIGH})

# Keywords that indicate high-risk areas
_HIGH_RISK_KEYWORDS: list[str] = [
    "security", "auth", "authentication", "authorization",
    "database", "db", "migration", "schema", "credential",
    "password", "secret", "token", "encrypt", "decrypt",
    "permission", "privilege", "admin", "root", "sudo",
]

# Keywords that indicate medium-risk areas
_MEDIUM_RISK_KEYWORDS: list[str] = [
    "test", "config", "configuration", "setting",
    "dependency", "package", "import", "deploy",
    "ci", "pipeline", "build", "lint", "format",
]


@dataclass
class PlanStep:
    """A single executable step inside a :class:`Plan`."""

    id: str
    order: int
    description: str
    tool_calls: list[dict[str, object]] = field(
        default_factory=list,
    )
    estimated_tokens: int = 0
    status: StepStatus = StepStatus.PENDING
    result: str | None = None
    error: str | None = None
    files_to_modify: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    risk_level: str = RISK_LOW
    validation: str = ""
    rollback: str = ""


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
    status: str  # draft, approved, running, paused, completed, failed, rolled_back
    git_checkpoint: str | None = None
    goal_summary: str = ""
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    risk_assessment: str = ""
    estimated_time_minutes: float = 0.0
    git_start_hash: str = ""


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
    "running",
    "paused",
    "in_progress",
    "completed",
    "failed",
    "rolled_back",
})

# Rough per-step token budget
_DEFAULT_STEP_TOKENS = 500

# Approximate minutes per step for time estimation
_MINUTES_PER_STEP = 2.0

# Pattern to detect file paths in step descriptions
_FILE_PATH_PATTERN = re.compile(
    r"""
    (?:                          # Match common path patterns:
      \b[\w./]+\.py\b            # Python files (*.py)
    | \b[\w./]+\.ts\b            # TypeScript files
    | \b[\w./]+\.js\b            # JavaScript files
    | \b[\w./]+\.json\b          # JSON files
    | \b[\w./]+\.yaml\b          # YAML files
    | \b[\w./]+\.yml\b           # YML files
    | \b[\w./]+\.toml\b          # TOML files
    | \b[\w./]+\.cfg\b           # Config files
    | \b[\w./]+\.ini\b           # INI files
    | \b[\w./]+\.sql\b           # SQL files
    | \b[\w./]+\.md\b            # Markdown files
    | \b[\w./]+\.txt\b           # Text files
    | \bsrc/[\w./]+\b            # src/ paths
    | \btests/[\w./]+\b          # tests/ paths
    )
    """,
    re.VERBOSE,
)


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

    def create_plan(
        self,
        request: str,
        context: dict[str, object] | None = None,
    ) -> Plan:
        """Generate a step-by-step plan from a user request.

        The plan is created in *draft* status and must be approved
        before execution.  Each step is enriched with:
        - ``files_to_modify`` auto-detected from the description
        - ``risk_level`` (HIGH/MEDIUM/LOW) based on keywords
        - ``validation`` string describing how to verify success
        - ``rollback`` string describing how to undo the step
        - ``estimated_cost`` per-step cost in USD

        The plan itself is enriched with:
        - ``goal_summary`` from the request text
        - ``preconditions`` / ``postconditions``
        - ``risk_assessment`` aggregated from step risks
        - ``estimated_time_minutes``

        Args:
            request: The user's natural-language request.
            context: Optional context dict (e.g. active files,
                repo info).

        Returns:
            A fully populated :class:`Plan` in ``draft`` status.

        Raises:
            ValueError: If *request* is empty or blank.
        """
        if not request or not request.strip():
            raise ValueError("Request must not be empty")

        context = context or {}
        steps = self._decompose(request, context)

        # Enrich each step with metadata
        for step in steps:
            step.files_to_modify = self._detect_files(
                step.description,
            )
            step.risk_level = self._assess_step_risk(
                step.description,
            )
            step.validation = self._generate_validation(
                step.description, step.files_to_modify,
            )
            step.rollback = self._generate_rollback(
                step.files_to_modify,
            )
            step.estimated_cost = self._estimate_step_cost(step)

        total_cost = self.estimate_cost_from_steps(steps)
        risk_assessment = self._aggregate_risk_assessment(steps)

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
            goal_summary=self._build_goal_summary(request),
            preconditions=[
                "Git working directory clean",
                "All tests passing",
            ],
            postconditions=[
                "All new tests passing",
                "No ruff errors",
            ],
            risk_assessment=risk_assessment,
            estimated_time_minutes=len(steps) * _MINUTES_PER_STEP,
            git_start_hash="",
        )

        logger.info(
            "plan_created",
            plan_id=plan.id,
            step_count=len(steps),
            estimated_cost=f"${total_cost:.6f}",
            risk_assessment=risk_assessment,
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
        ]

        if plan.goal_summary:
            lines.append(f"Goal: {plan.goal_summary}")
        if plan.risk_assessment:
            lines.append(f"Risk: {plan.risk_assessment}")
        if plan.estimated_time_minutes > 0:
            lines.append(
                f"Est. time: {plan.estimated_time_minutes:.0f} min"
            )
        if plan.preconditions:
            lines.append(
                "Preconditions: "
                + "; ".join(plan.preconditions)
            )
        if plan.postconditions:
            lines.append(
                "Postconditions: "
                + "; ".join(plan.postconditions)
            )

        lines.append("")
        lines.append(f"Steps ({len(plan.steps)}):")
        lines.append("-" * 50)

        for step in sorted(plan.steps, key=lambda s: s.order):
            status_tag = step.status.value.upper()
            desc = step.description
            if len(desc) > 80:
                desc = desc[:77] + "..."
            risk_tag = f" [{step.risk_level}]" if step.risk_level else ""
            lines.append(
                f"  {step.order}. [{status_tag}]{risk_tag} {desc}"
            )
            if step.tool_calls:
                for tc in step.tool_calls:
                    tool_name = tc.get("tool", "unknown")
                    lines.append(f"     -> {tool_name}")
            if step.files_to_modify:
                files_str = ", ".join(step.files_to_modify[:5])
                if len(step.files_to_modify) > 5:
                    files_str += f" (+{len(step.files_to_modify) - 5})"
                lines.append(f"     files: {files_str}")
            if step.validation:
                lines.append(
                    f"     validate: {step.validation}"
                )

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

    # ------------------------------------------------------------------
    # Enrichment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_files(description: str) -> list[str]:
        """Auto-detect file paths from a step description.

        Looks for patterns like ``*.py``, ``src/...``,
        ``tests/...``, etc.

        Args:
            description: Step description text.

        Returns:
            Deduplicated list of detected file paths.
        """
        matches = _FILE_PATH_PATTERN.findall(description)
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for m in matches:
            normalised = m.strip()
            if normalised and normalised not in seen:
                seen.add(normalised)
                result.append(normalised)
        return result

    @staticmethod
    def _assess_step_risk(description: str) -> str:
        """Determine risk level from step description keywords.

        Returns:
            One of ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
        """
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in _HIGH_RISK_KEYWORDS):
            return RISK_HIGH
        if any(kw in desc_lower for kw in _MEDIUM_RISK_KEYWORDS):
            return RISK_MEDIUM
        return RISK_LOW

    @staticmethod
    def _generate_validation(
        description: str,
        files: list[str],
    ) -> str:
        """Generate a validation string for a step.

        Heuristically determines the best way to verify the step
        succeeded based on its description and files.

        Args:
            description: Step description text.
            files: Detected files to modify.

        Returns:
            Human-readable validation instruction.
        """
        desc_lower = description.lower()

        # Test-related steps
        if any(kw in desc_lower for kw in _TEST_KEYWORDS):
            # Try to extract a test path from files
            test_files = [
                f for f in files
                if "test" in f.lower()
            ]
            if test_files:
                return f"Run pytest {test_files[0]}"
            return "Run pytest and verify all tests pass"

        # File creation/modification steps
        if files:
            file_checks = [
                f"Verify {f} exists and is valid"
                for f in files[:3]
            ]
            return "; ".join(file_checks)

        # Analysis steps
        if any(kw in desc_lower for kw in _ANALYSIS_KEYWORDS):
            return "Review analysis output for completeness"

        # Default
        return "Verify step output is correct"

    @staticmethod
    def _generate_rollback(files: list[str]) -> str:
        """Generate a rollback string for a step.

        Uses ``git checkout`` to revert modified files.

        Args:
            files: Files that would be modified by this step.

        Returns:
            Git command to undo the step, or a generic message.
        """
        if not files:
            return "git checkout -- ."
        escaped = " ".join(files[:10])
        return f"git checkout -- {escaped}"

    def _estimate_step_cost(self, step: PlanStep) -> float:
        """Estimate cost in USD for a single step.

        Args:
            step: The step to estimate.

        Returns:
            Estimated cost in USD.
        """
        pricing = MODEL_PRICING.get(self.execution_model)
        if pricing is None:
            return 0.0
        tokens = step.estimated_tokens
        input_cost = (tokens / 1_000_000) * pricing.input_cost_per_1m
        output_cost = (
            (tokens / 1_000_000) * pricing.output_cost_per_1m
        )
        return input_cost + output_cost

    @staticmethod
    def _build_goal_summary(request: str) -> str:
        """Build a concise goal summary from the request text.

        Truncates to the first sentence or 120 characters,
        whichever is shorter.

        Args:
            request: The user's full request text.

        Returns:
            A concise summary string.
        """
        text = request.strip()
        # Take first sentence
        for delimiter in (".", "!", "?"):
            idx = text.find(delimiter)
            if 0 < idx < 120:
                return text[: idx + 1]
        # Truncate if too long
        if len(text) > 120:
            return text[:117] + "..."
        return text

    @staticmethod
    def _aggregate_risk_assessment(
        steps: list[PlanStep],
    ) -> str:
        """Build an overall risk assessment from step risks.

        Args:
            steps: All steps in the plan.

        Returns:
            Human-readable risk assessment string.
        """
        high = sum(
            1 for s in steps if s.risk_level == RISK_HIGH
        )
        medium = sum(
            1 for s in steps if s.risk_level == RISK_MEDIUM
        )
        low = sum(
            1 for s in steps if s.risk_level == RISK_LOW
        )
        total = len(steps)

        if high > 0:
            overall = "HIGH"
        elif medium > 0:
            overall = "MEDIUM"
        else:
            overall = "LOW"

        return (
            f"{overall} — {high} high, {medium} medium, "
            f"{low} low risk steps out of {total} total"
        )

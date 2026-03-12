"""FrugalGPT-style confidence cascading for cost-efficient model routing.

Based on Chen, Zaharia & Zou (Stanford, 2023): cascade where cheap models
handle easy queries and expensive models are invoked only when confidence is
low.  Up to 98% cost reduction on classification, 50-70% on complex workloads.

Also incorporates Huang et al. (Google DeepMind, 2023) insight that LLMs
cannot reliably self-correct without external feedback — hence the cross-model
confidence assessment ("external judge").

Algorithm:
    1. Start with the cheapest available models.
    2. Generate a response and assess confidence (self + external judge).
    3. If confidence >= threshold for this tier, accept the result.
    4. Otherwise, escalate to the next tier (medium, then premium).
    5. Include previous attempts in context so higher-tier models learn from
       failures.
    6. Stop when confidence is met, budget is exhausted, or max escalations
       reached.

Key insight: 70-80% of tasks are handled by the cheapest tier.  Only
genuinely hard tasks escalate to premium models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from prism.cost.pricing import calculate_cost

if TYPE_CHECKING:
    from prism.llm.completion import CompletionEngine
    from prism.orchestrator.swarm import ModelPool

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------


@dataclass
class ConfidenceScore:
    """Model confidence assessment for a generated output.

    Attributes:
        score: Confidence level between 0.0 (no confidence) and 1.0 (certain).
        reasoning: Explanation of why this confidence level was assigned.
        uncertainty_areas: Specific areas where the model is unsure.
        alternative_approaches: Other possible approaches the model considered.
        model: Which model produced this confidence assessment.
    """

    score: float
    reasoning: str
    uncertainty_areas: list[str] = field(default_factory=list)
    alternative_approaches: list[str] = field(default_factory=list)
    model: str = ""

    def __post_init__(self) -> None:
        """Clamp score to [0.0, 1.0] and validate types."""
        self.score = max(0.0, min(1.0, float(self.score)))
        if not isinstance(self.reasoning, str):
            self.reasoning = str(self.reasoning)
        if not isinstance(self.uncertainty_areas, list):
            self.uncertainty_areas = []
        if not isinstance(self.alternative_approaches, list):
            self.alternative_approaches = []
        if not isinstance(self.model, str):
            self.model = str(self.model)


@dataclass
class CascadeLevel:
    """Definition of a single tier in the confidence cascade.

    Attributes:
        tier: Human-readable tier name (e.g. ``"cheap"``, ``"medium"``).
        models: LiteLLM model identifiers available at this tier.
        confidence_threshold: Minimum confidence to accept a result here.
        cost_multiplier: Relative cost compared to baseline (1.0).
    """

    tier: str
    models: list[str]
    confidence_threshold: float
    cost_multiplier: float

    def __post_init__(self) -> None:
        """Validate thresholds and cost multiplier."""
        self.confidence_threshold = max(0.0, min(1.0, float(self.confidence_threshold)))
        self.cost_multiplier = max(0.0, float(self.cost_multiplier))
        if not isinstance(self.models, list):
            self.models = list(self.models)
        if not self.tier:
            self.tier = "unknown"


@dataclass
class CascadeConfig:
    """Configuration for the confidence cascade.

    Attributes:
        levels: Custom cascade levels.  If ``None``, defaults to a 3-tier
            cascade (cheap/medium/premium).
        min_confidence: Absolute minimum confidence to accept any result,
            regardless of which level produced it.
        max_escalations: Maximum number of times to escalate to a higher tier.
        use_external_judge: Whether to use a different model to assess
            confidence (catches self-overconfidence).
        budget_limit: Maximum total cost (USD) before forcing acceptance of
            the best available result.  ``None`` means no limit.
    """

    levels: list[CascadeLevel] | None = None
    min_confidence: float = 0.7
    max_escalations: int = 3
    use_external_judge: bool = True
    budget_limit: float | None = None

    def __post_init__(self) -> None:
        """Validate config bounds."""
        self.min_confidence = max(0.0, min(1.0, float(self.min_confidence)))
        self.max_escalations = max(1, int(self.max_escalations))
        if self.budget_limit is not None:
            self.budget_limit = max(0.0, float(self.budget_limit))


@dataclass
class CascadeAttempt:
    """Record of a single attempt within the cascade.

    Attributes:
        level: Which cascade level this attempt was at (0-indexed).
        model: Model that generated the output.
        output: Raw text output from the model.
        self_confidence: The model's self-assessed confidence.
        judge_confidence: External judge's confidence assessment, if used.
        accepted: Whether this attempt was accepted as the final result.
        cost: Cost in USD for this attempt (generation + assessment).
    """

    level: int
    model: str
    output: str
    self_confidence: ConfidenceScore
    judge_confidence: ConfidenceScore | None = None
    accepted: bool = False
    cost: float = 0.0


@dataclass
class CascadeResult:
    """Final result of a confidence cascade execution.

    Attributes:
        output: The accepted output text.
        confidence: Final confidence assessment for the accepted output.
        attempts: All attempts made during the cascade.
        accepted_at_level: Which cascade level produced the accepted output.
        total_cost: Aggregate cost in USD across all attempts.
        cost_saved_vs_premium: Estimated savings compared to always using
            premium models.
    """

    output: str
    confidence: ConfidenceScore
    attempts: list[CascadeAttempt]
    accepted_at_level: int
    total_cost: float
    cost_saved_vs_premium: float


@dataclass
class TaskResult:
    """Structured result for a swarm task, enriched with cascade metadata.

    Attributes:
        output: The actual content or code produced.
        confidence: Confidence assessment for this output.
        model: Which model produced the final output.
        cost: Total cost in USD.
        tokens_used: Total tokens consumed (input + output).
        execution_time: Wall-clock time in seconds.
        files_changed: List of file paths affected.
        cascade_attempts: How many models were tried before acceptance.
        metadata: Extensible key-value metadata.
    """

    output: str
    confidence: ConfidenceScore
    model: str
    cost: float
    tokens_used: int
    execution_time: float
    files_changed: list[str] = field(default_factory=list)
    cascade_attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Confidence assessment prompts
# ------------------------------------------------------------------

_SELF_ASSESS_SYSTEM_PROMPT = """\
You are a confidence assessment engine.  Given an AI-generated output and the
original prompt, assess how confident you are that the output is correct,
complete, and high quality.

Respond ONLY with a JSON object:
{
  "score": 0.0 to 1.0,
  "reasoning": "brief explanation of confidence level",
  "uncertainty_areas": ["area 1", "area 2"],
  "alternative_approaches": ["approach 1", "approach 2"]
}

Scoring guide:
- 0.9-1.0: Highly confident, factual/simple, no ambiguity
- 0.7-0.89: Good confidence, minor concerns
- 0.5-0.69: Moderate confidence, significant unknowns
- 0.3-0.49: Low confidence, major uncertainties
- 0.0-0.29: Very low confidence, likely wrong or incomplete

Be HONEST.  Do NOT inflate confidence.  If you are uncertain, say so.
"""

_JUDGE_ASSESS_SYSTEM_PROMPT = """\
You are an EXTERNAL confidence judge.  You are assessing the quality of an
output produced by a DIFFERENT AI model.  Your job is to provide an objective
confidence assessment.

The generating model may have overestimated its own confidence.  Be critical
and look for:
- Factual errors or hallucinations
- Incomplete or missing handling of edge cases
- Logical inconsistencies
- Security vulnerabilities or bad practices
- Whether the output actually addresses the prompt

Respond ONLY with a JSON object:
{
  "score": 0.0 to 1.0,
  "reasoning": "brief explanation of your assessment",
  "uncertainty_areas": ["area 1", "area 2"],
  "alternative_approaches": ["approach 1", "approach 2"]
}

Be stricter than the generating model.  If something seems off, lower the score.
"""


# ------------------------------------------------------------------
# ConfidenceCascade
# ------------------------------------------------------------------


class ConfidenceCascade:
    """FrugalGPT-style confidence cascading.

    Try cheap model first.  Assess confidence (self + external judge).
    If confidence < threshold, escalate to next tier.
    Repeat until confidence is acceptable or budget exhausted.

    Key insight: most tasks (70-80%) are handled by the cheapest tier.
    Only genuinely hard tasks escalate to premium models.

    Args:
        engine: Completion engine for LLM calls (must have a mock backend
            injected for testing).
        model_pool: Model pool providing tiered model selection.
        config: Optional cascade configuration.  If ``None``, sensible
            defaults are used (3-tier cascade, external judge enabled).
    """

    def __init__(
        self,
        engine: CompletionEngine,
        model_pool: ModelPool,
        config: CascadeConfig | None = None,
    ) -> None:
        self._engine = engine
        self._model_pool = model_pool
        self._config = config or CascadeConfig()
        self._levels = (
            self._config.levels
            if self._config.levels is not None
            else self._build_levels()
        )

        logger.info(
            "cascade_initialised",
            levels=len(self._levels),
            min_confidence=self._config.min_confidence,
            max_escalations=self._config.max_escalations,
            use_judge=self._config.use_external_judge,
            budget_limit=self._config.budget_limit,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(self, prompt: str, context: str = "") -> CascadeResult:
        """Execute a prompt through the confidence cascade.

        Starts at the cheapest level and escalates until confidence is met,
        budget is exhausted, or max escalations reached.

        Args:
            prompt: The user prompt or task description.
            context: Additional context (e.g. file contents, previous outputs).

        Returns:
            A :class:`CascadeResult` with the accepted output, confidence
            scores, all attempts, and cost breakdown.

        Raises:
            ValueError: If *prompt* is empty.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty")

        attempts: list[CascadeAttempt] = []
        total_cost = 0.0
        best_attempt: CascadeAttempt | None = None

        for level_idx in range(min(len(self._levels), self._config.max_escalations)):
            # Budget check before attempting
            if self._config.budget_limit is not None and total_cost >= self._config.budget_limit:
                logger.info(
                    "cascade_budget_exhausted",
                    total_cost=total_cost,
                    budget_limit=self._config.budget_limit,
                    level=level_idx,
                )
                break

            attempt = await self._attempt_at_level(
                level=level_idx,
                prompt=prompt,
                context=context,
                previous_attempts=attempts,
            )
            total_cost += attempt.cost
            attempts.append(attempt)

            # Track best attempt (highest effective confidence)
            effective_confidence = self._effective_confidence(attempt)
            if best_attempt is None:
                best_attempt = attempt
            else:
                best_effective = self._effective_confidence(best_attempt)
                if effective_confidence > best_effective:
                    best_attempt = attempt

            if self._should_accept(attempt):
                attempt.accepted = True
                logger.info(
                    "cascade_accepted",
                    level=level_idx,
                    tier=self._levels[level_idx].tier,
                    model=attempt.model,
                    confidence=effective_confidence,
                )
                return self._build_result(
                    accepted=attempt,
                    attempts=attempts,
                    level=level_idx,
                    total_cost=total_cost,
                )

            if not self._should_escalate(attempt, level_idx):
                logger.info(
                    "cascade_no_escalation",
                    level=level_idx,
                    reason="max_level_reached_or_no_improvement",
                )
                break

            logger.info(
                "cascade_escalating",
                from_level=level_idx,
                to_level=level_idx + 1,
                confidence=effective_confidence,
                threshold=self._levels[level_idx].confidence_threshold,
            )

        # No attempt met the threshold — accept the best one
        if best_attempt is None:
            # Shouldn't happen if levels is non-empty, but guard defensively
            return CascadeResult(
                output="",
                confidence=ConfidenceScore(
                    score=0.0,
                    reasoning="No attempts were made",
                    model="none",
                ),
                attempts=attempts,
                accepted_at_level=0,
                total_cost=total_cost,
                cost_saved_vs_premium=0.0,
            )

        best_attempt.accepted = True
        accepted_level = best_attempt.level

        logger.info(
            "cascade_forced_accept",
            level=accepted_level,
            confidence=self._effective_confidence(best_attempt),
        )

        return self._build_result(
            accepted=best_attempt,
            attempts=attempts,
            level=accepted_level,
            total_cost=total_cost,
        )

    # ------------------------------------------------------------------
    # Internal: attempt execution
    # ------------------------------------------------------------------

    async def _attempt_at_level(
        self,
        level: int,
        prompt: str,
        context: str,
        previous_attempts: list[CascadeAttempt],
    ) -> CascadeAttempt:
        """Execute a single attempt at a given cascade level.

        Selects a model from the level's pool, generates a response,
        assesses confidence (self + optional judge), and returns the
        attempt record.

        Args:
            level: 0-indexed cascade level.
            prompt: The user prompt.
            context: Additional context.
            previous_attempts: Previous failed attempts for learning.

        Returns:
            A :class:`CascadeAttempt` with output and confidence scores.
        """
        cascade_level = self._levels[level]
        model = cascade_level.models[0] if cascade_level.models else ""
        if not model:
            return CascadeAttempt(
                level=level,
                model="none",
                output="",
                self_confidence=ConfidenceScore(
                    score=0.0,
                    reasoning="No models available at this level",
                    model="none",
                ),
                accepted=False,
                cost=0.0,
            )

        attempt_cost = 0.0

        # Build messages with previous attempt context
        messages = self._build_generation_messages(
            prompt=prompt,
            context=context,
            previous_attempts=previous_attempts,
        )

        # Generate the response
        result = await self._engine.complete(
            messages=messages,
            model=model,
            temperature=0.4,
            max_tokens=4096,
        )
        attempt_cost += result.cost_usd

        # Self-assessment
        self_confidence = await self._assess_confidence(
            output=result.content,
            prompt=prompt,
            model=model,
        )

        # The self-assessment itself costs tokens
        assess_cost = self._estimate_assessment_cost(model)
        attempt_cost += assess_cost

        # External judge assessment
        judge_confidence: ConfidenceScore | None = None
        if self._config.use_external_judge:
            judge_model = self._select_judge_model(level)
            if judge_model and judge_model != model:
                judge_confidence = await self._judge_confidence(
                    output=result.content,
                    prompt=prompt,
                    judge_model=judge_model,
                )
                judge_cost = self._estimate_assessment_cost(judge_model)
                attempt_cost += judge_cost

        return CascadeAttempt(
            level=level,
            model=model,
            output=result.content,
            self_confidence=self_confidence,
            judge_confidence=judge_confidence,
            accepted=False,
            cost=attempt_cost,
        )

    # ------------------------------------------------------------------
    # Internal: confidence assessment
    # ------------------------------------------------------------------

    async def _assess_confidence(
        self,
        output: str,
        prompt: str,
        model: str,
    ) -> ConfidenceScore:
        """Ask the generating model to self-assess its confidence.

        Args:
            output: The generated output to assess.
            prompt: The original prompt that produced the output.
            model: Model to use for self-assessment.

        Returns:
            A :class:`ConfidenceScore` from self-assessment.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SELF_ASSESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original prompt:\n{prompt}\n\n"
                    f"Generated output:\n{output}\n\n"
                    "Assess the confidence of this output."
                ),
            },
        ]

        result = await self._engine.complete(
            messages=messages,
            model=model,
            temperature=0.1,
            max_tokens=512,
        )

        return self._parse_confidence(result.content, model)

    async def _judge_confidence(
        self,
        output: str,
        prompt: str,
        judge_model: str,
    ) -> ConfidenceScore:
        """Ask an external judge model to assess the output's confidence.

        Uses a DIFFERENT model from the generator to avoid self-confirmation
        bias.  Per Huang et al. (2023), LLMs cannot reliably self-correct
        without external feedback.

        Args:
            output: The generated output to judge.
            prompt: The original prompt.
            judge_model: Model to use as the external judge.

        Returns:
            A :class:`ConfidenceScore` from the external judge.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _JUDGE_ASSESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original prompt:\n{prompt}\n\n"
                    f"Output from another AI model:\n{output}\n\n"
                    "Assess the quality and correctness of this output."
                ),
            },
        ]

        result = await self._engine.complete(
            messages=messages,
            model=judge_model,
            temperature=0.1,
            max_tokens=512,
        )

        return self._parse_confidence(result.content, judge_model)

    # ------------------------------------------------------------------
    # Internal: level building
    # ------------------------------------------------------------------

    def _build_levels(self) -> list[CascadeLevel]:
        """Build the default 3-tier cascade from the ModelPool.

        Returns:
            A list of :class:`CascadeLevel` objects: cheap, medium, premium.
        """
        cheap_models = self._model_pool.get_research_models()
        medium_model = self._model_pool.get_execution_model("medium")
        premium_model = self._model_pool.get_planning_model()

        levels: list[CascadeLevel] = []

        # Tier 1: Cheap (high threshold — need high confidence to accept)
        if cheap_models:
            levels.append(
                CascadeLevel(
                    tier="cheap",
                    models=cheap_models[:3],
                    confidence_threshold=0.85,
                    cost_multiplier=1.0,
                ),
            )

        # Tier 2: Medium (moderate threshold)
        medium_models = [medium_model] if medium_model else []
        if medium_models:
            # Avoid duplication: exclude models already in cheap tier
            cheap_set = set(cheap_models[:3]) if cheap_models else set()
            medium_models = [m for m in medium_models if m not in cheap_set]
            if not medium_models:
                medium_models = [medium_model] if medium_model else []
            levels.append(
                CascadeLevel(
                    tier="medium",
                    models=medium_models,
                    confidence_threshold=0.75,
                    cost_multiplier=5.0,
                ),
            )

        # Tier 3: Premium (lower threshold — expensive, so accept more liberally)
        premium_models = [premium_model] if premium_model else []
        if premium_models:
            levels.append(
                CascadeLevel(
                    tier="premium",
                    models=premium_models,
                    confidence_threshold=0.6,
                    cost_multiplier=20.0,
                ),
            )

        if not levels:
            logger.warning("cascade_no_levels_built")

        logger.info(
            "cascade_levels_built",
            level_count=len(levels),
            tiers=[lev.tier for lev in levels],
        )
        return levels

    # ------------------------------------------------------------------
    # Internal: decision logic
    # ------------------------------------------------------------------

    def _should_accept(self, attempt: CascadeAttempt) -> bool:
        """Determine whether an attempt should be accepted.

        An attempt is accepted when its effective confidence (combining
        self-assessment and judge assessment) meets or exceeds the threshold
        for its cascade level, AND the confidence is at least the absolute
        minimum.

        Args:
            attempt: The cascade attempt to evaluate.

        Returns:
            ``True`` if the attempt should be accepted.
        """
        effective = self._effective_confidence(attempt)

        # Must meet absolute minimum
        if effective < self._config.min_confidence:
            return False

        # Must meet level-specific threshold
        level_threshold = self._levels[attempt.level].confidence_threshold
        return effective >= level_threshold

    def _should_escalate(self, attempt: CascadeAttempt, level: int) -> bool:
        """Determine whether to escalate to the next cascade level.

        Escalation is warranted when:
        - There is a next level available.
        - Budget is not exhausted (checked in the main loop).
        - The current attempt's confidence is below the level threshold.

        Args:
            attempt: The current (rejected) attempt.
            level: Current 0-indexed level.

        Returns:
            ``True`` if escalation should occur.
        """
        # Cannot escalate beyond available levels
        if level + 1 >= len(self._levels):
            return False

        # Cannot escalate beyond max escalations
        return not level + 1 >= self._config.max_escalations

    @staticmethod
    def _effective_confidence(attempt: CascadeAttempt) -> float:
        """Calculate the effective confidence for an attempt.

        If an external judge assessed the output, the effective confidence
        is the minimum of self-assessment and judge assessment (conservative
        approach).  If no judge was used, the self-assessment is returned.

        Args:
            attempt: The cascade attempt.

        Returns:
            Effective confidence score between 0.0 and 1.0.
        """
        self_score = attempt.self_confidence.score
        if attempt.judge_confidence is not None:
            judge_score = attempt.judge_confidence.score
            # Conservative: take the minimum (catches self-overconfidence)
            return min(self_score, judge_score)
        return self_score

    def _calculate_savings(self, result: CascadeResult) -> float:
        """Calculate estimated savings vs always using the premium tier.

        Estimates what the cost would have been if the premium tier model
        was used directly, then subtracts the actual cascade cost.

        Args:
            result: The completed cascade result.

        Returns:
            Estimated savings in USD.  Negative means the cascade cost more.
        """
        if not self._levels:
            return 0.0

        # Estimate premium cost: use the highest-tier cost multiplier
        premium_level = self._levels[-1]
        premium_multiplier = premium_level.cost_multiplier
        cheap_multiplier = self._levels[0].cost_multiplier if self._levels else 1.0

        if cheap_multiplier <= 0:
            return 0.0

        # The premium hypothetical cost: what the accepted attempt would cost
        # at premium rates, adjusted by cost multiplier ratio
        if result.attempts:
            accepted = next((a for a in result.attempts if a.accepted), result.attempts[0])
            # Rough estimate: scale the accepted attempt's generation cost
            # by the ratio of premium-to-cheap multipliers
            ratio = premium_multiplier / cheap_multiplier
            hypothetical_premium_cost = accepted.cost * ratio
            return max(0.0, hypothetical_premium_cost - result.total_cost)

        return 0.0

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _build_generation_messages(
        self,
        prompt: str,
        context: str,
        previous_attempts: list[CascadeAttempt],
    ) -> list[dict[str, str]]:
        """Build messages for the generation call, including failure context.

        If there are previous (rejected) attempts, their outputs and
        confidence assessments are included so the current model can learn
        from and improve upon them.

        Args:
            prompt: The user prompt.
            context: Additional context.
            previous_attempts: Previous cascade attempts.

        Returns:
            List of messages in OpenAI chat format.
        """
        system_content = (
            "You are a helpful AI assistant.  Provide high-quality, "
            "complete, and accurate responses."
        )

        user_parts: list[str] = []

        if context:
            user_parts.append(f"Context:\n{context}\n")

        if previous_attempts:
            user_parts.append("Previous attempts by other models (for reference):\n")
            for idx, attempt in enumerate(previous_attempts):
                user_parts.append(
                    f"--- Attempt {idx + 1} (model: {attempt.model}, "
                    f"confidence: {self._effective_confidence(attempt):.2f}) ---\n"
                    f"{attempt.output[:2000]}\n"
                )
                if attempt.self_confidence.uncertainty_areas:
                    user_parts.append(
                        "Uncertainty areas: "
                        + ", ".join(attempt.self_confidence.uncertainty_areas)
                        + "\n",
                    )
            user_parts.append(
                "\nImprove upon the previous attempts.  Address the "
                "identified uncertainty areas.\n",
            )

        user_parts.append(prompt)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "\n".join(user_parts)},
        ]

    def _select_judge_model(self, current_level: int) -> str | None:
        """Select a judge model that is different from the current level's model.

        The judge should ideally be cheaper than the next escalation level
        but from a different provider than the current model.

        Args:
            current_level: The current cascade level index.

        Returns:
            A model identifier, or ``None`` if no suitable judge is found.
        """
        current_models = set(self._levels[current_level].models) if self._levels else set()

        # Try to use a model from a different level
        for level_idx, level in enumerate(self._levels):
            if level_idx == current_level:
                continue
            for model in level.models:
                if model not in current_models:
                    return model

        # Fallback: use the review model from the pool
        try:
            review_model = self._model_pool.get_review_model()
            if review_model not in current_models:
                return review_model
        except RuntimeError:
            pass

        return None

    @staticmethod
    def _parse_confidence(raw: str, model: str) -> ConfidenceScore:
        """Parse a confidence assessment from raw model output.

        Attempts JSON parsing first; falls back to a heuristic score
        based on the raw text.

        Args:
            raw: Raw model output (expected JSON).
            model: Model that produced this assessment.

        Returns:
            A :class:`ConfidenceScore` populated from the response.
        """
        stripped = raw.strip()

        # Try to find JSON in the response
        json_start = stripped.find("{")
        json_end = stripped.rfind("}")

        if json_start >= 0 and json_end > json_start:
            json_str = stripped[json_start : json_end + 1]
            try:
                data = json.loads(json_str)
                if isinstance(data, dict):
                    score = float(data.get("score", 0.5))
                    reasoning = str(data.get("reasoning", "No reasoning provided"))
                    uncertainty = data.get("uncertainty_areas", [])
                    alternatives = data.get("alternative_approaches", [])

                    if not isinstance(uncertainty, list):
                        uncertainty = []
                    if not isinstance(alternatives, list):
                        alternatives = []

                    return ConfidenceScore(
                        score=score,
                        reasoning=reasoning,
                        uncertainty_areas=[str(u) for u in uncertainty],
                        alternative_approaches=[str(a) for a in alternatives],
                        model=model,
                    )
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.warning(
                    "confidence_parse_json_failed",
                    raw_preview=raw[:200],
                    model=model,
                )

        # Fallback: heuristic based on raw text
        return ConfidenceScore(
            score=0.5,
            reasoning=f"Could not parse structured confidence; raw: {raw[:200]}",
            model=model,
        )

    @staticmethod
    def _estimate_assessment_cost(model: str) -> float:
        """Estimate the cost of a confidence assessment call.

        Assessment calls use ~200 input tokens and ~100 output tokens.

        Args:
            model: The model used for assessment.

        Returns:
            Estimated cost in USD.
        """
        try:
            return calculate_cost(model, input_tokens=200, output_tokens=100)
        except ValueError:
            # Unknown model — use a conservative default estimate
            return 0.001

    def _build_result(
        self,
        accepted: CascadeAttempt,
        attempts: list[CascadeAttempt],
        level: int,
        total_cost: float,
    ) -> CascadeResult:
        """Build the final CascadeResult from the accepted attempt.

        Args:
            accepted: The accepted attempt.
            attempts: All attempts made.
            level: The level at which the result was accepted.
            total_cost: Total cost across all attempts.

        Returns:
            A complete :class:`CascadeResult`.
        """
        # Determine effective confidence
        if accepted.judge_confidence is not None:
            # Use judge confidence as the final confidence (external validation)
            final_confidence = ConfidenceScore(
                score=self._effective_confidence(accepted),
                reasoning=(
                    f"Self: {accepted.self_confidence.reasoning}; "
                    f"Judge: {accepted.judge_confidence.reasoning}"
                ),
                uncertainty_areas=(
                    accepted.self_confidence.uncertainty_areas
                    + accepted.judge_confidence.uncertainty_areas
                ),
                alternative_approaches=(
                    accepted.self_confidence.alternative_approaches
                    + accepted.judge_confidence.alternative_approaches
                ),
                model=accepted.judge_confidence.model,
            )
        else:
            final_confidence = accepted.self_confidence

        result = CascadeResult(
            output=accepted.output,
            confidence=final_confidence,
            attempts=attempts,
            accepted_at_level=level,
            total_cost=total_cost,
            cost_saved_vs_premium=0.0,
        )

        # Calculate savings
        result.cost_saved_vs_premium = self._calculate_savings(result)
        return result

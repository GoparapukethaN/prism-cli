"""Task complexity classification.

Analyzes user prompts and classifies them into complexity tiers
(SIMPLE, MEDIUM, COMPLEX) using a weighted feature vector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from prism.providers.base import ComplexityTier

if TYPE_CHECKING:
    from prism.config.settings import Settings

logger = structlog.get_logger(__name__)

# --- Keyword Scoring Tables ---

SIMPLE_KEYWORDS: dict[str, float] = {
    "fix typo": -0.25,
    "typo": -0.25,
    "spelling": -0.25,
    "whitespace": -0.25,
    "indent": -0.25,
    "format": -0.20,
    "rename": -0.20,
    "syntax": -0.20,
    "import": -0.20,
    "add a line": -0.20,
    "print": -0.20,
    "log": -0.15,
    "fix": -0.15,
    "explain": -0.15,
    "what does": -0.15,
    "comment": -0.15,
    "delete": -0.15,
    "remove line": -0.15,
    "change": -0.10,
}

MEDIUM_KEYWORDS: dict[str, float] = {
    "refactor": 0.10,
    "implement": 0.10,
    "feature": 0.10,
    "module": 0.10,
    "api": 0.10,
    "endpoint": 0.10,
    "database": 0.10,
    "error handling": 0.10,
    "async": 0.10,
    "debug": 0.10,
    "test": 0.05,
    "class": 0.05,
    "query": 0.05,
    "validate": 0.05,
    "function": 0.00,
}

COMPLEX_KEYWORDS: dict[str, float] = {
    "architect": 0.30,
    "redesign": 0.30,
    "from scratch": 0.25,
    "design": 0.25,
    "microservice": 0.25,
    "distributed": 0.25,
    "refactor entire": 0.25,
    "optimize": 0.20,
    "security audit": 0.25,
    "security": 0.20,
    "scalable": 0.20,
    "concurrent": 0.20,
    "algorithm": 0.20,
    "migrate": 0.20,
    "trade-off": 0.20,
    "performance": 0.15,
    "system": 0.15,
    "abstract": 0.15,
    "pattern": 0.15,
    "evaluate": 0.15,
    "compare": 0.10,
    "why": 0.10,
}

# Reasoning detection patterns
REASONING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bif\b.*\bthen\b", re.IGNORECASE),
    re.compile(r"\bdepends\b", re.IGNORECASE),
    re.compile(r"\beither\b.*\bor\b", re.IGNORECASE),
    re.compile(r"\bfirst\b.*\bthen\b.*\bfinally\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bbetter\b.*\bvs\b", re.IGNORECASE),
    re.compile(r"\bwhich approach\b", re.IGNORECASE),
    re.compile(r"\bpros and cons\b", re.IGNORECASE),
    re.compile(r"\btrade.?off\b", re.IGNORECASE),
]

# Scope detection patterns
SCOPE_PATTERNS: dict[str, float] = {
    "architecture": 0.9,
    "system": 0.9,
    "codebase": 0.7,
    "project": 0.7,
    "module": 0.5,
    "package": 0.5,
}

# Feature weights for scoring
FEATURE_WEIGHTS: dict[str, float] = {
    "prompt_token_count": 0.15,
    "files_referenced": 0.15,
    "estimated_output_tokens": 0.10,
    "complexity_keywords": 0.25,
    "requires_reasoning": 0.20,
    "scope": 0.15,
}


@dataclass
class TaskContext:
    """Context information for task classification."""

    active_files: list[str] = field(default_factory=list)
    conversation_turns: int = 0
    project_file_count: int = 0


@dataclass(frozen=True)
class ClassificationResult:
    """Result of task complexity classification."""

    tier: ComplexityTier
    score: float  # 0.0 to 1.0
    features: dict[str, float]
    reasoning: str  # Human-readable explanation


class TaskClassifier:
    """Classifies user prompts into complexity tiers.

    Uses a weighted feature vector extracted from the prompt
    and task context to compute a complexity score. The score
    is then mapped to a tier using configurable thresholds.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the classifier.

        Args:
            settings: Application settings with routing thresholds.
        """
        self._simple_threshold = settings.get("routing.simple_threshold", 0.3)
        self._medium_threshold = settings.get("routing.medium_threshold", 0.7)

    def classify(self, prompt: str, context: TaskContext | None = None) -> ClassificationResult:
        """Classify a prompt into a complexity tier.

        Args:
            prompt: The user's input text.
            context: Optional task context (active files, etc.).

        Returns:
            ClassificationResult with tier, score, features, and reasoning.
        """
        if context is None:
            context = TaskContext()

        features = self.extract_features(prompt, context)
        score = self._compute_score(features)
        tier = self._score_to_tier(score)

        reasoning = self._explain(tier, score, features)

        logger.info(
            "task_classified",
            tier=tier.value,
            score=round(score, 3),
            features={k: round(v, 3) for k, v in features.items()},
        )

        return ClassificationResult(
            tier=tier,
            score=score,
            features=features,
            reasoning=reasoning,
        )

    def extract_features(self, prompt: str, context: TaskContext) -> dict[str, float]:
        """Extract the feature vector from a prompt.

        Args:
            prompt: User's input text.
            context: Task context.

        Returns:
            Dictionary of feature names to values.
        """
        return {
            "prompt_token_count": self._estimate_tokens(prompt),
            "files_referenced": float(len(context.active_files)),
            "estimated_output_tokens": self._estimate_output(prompt),
            "complexity_keywords": self._keyword_score(prompt),
            "requires_reasoning": self._detect_reasoning(prompt, context),
            "scope": self._assess_scope(prompt, context),
        }

    def get_score(self, prompt: str, context: TaskContext | None = None) -> float:
        """Get the raw complexity score for a prompt.

        Args:
            prompt: User's input text.
            context: Optional task context.

        Returns:
            Score between 0.0 and 1.0.
        """
        if context is None:
            context = TaskContext()
        features = self.extract_features(prompt, context)
        return self._compute_score(features)

    def _compute_score(self, features: dict[str, float]) -> float:
        """Compute weighted complexity score from features.

        Args:
            features: Extracted feature dictionary.

        Returns:
            Score clamped to [0.0, 1.0].
        """
        normalized: dict[str, float] = {
            "prompt_token_count": min(features.get("prompt_token_count", 0) / 2000, 1.0),
            "files_referenced": min(features.get("files_referenced", 0) / 10, 1.0),
            "estimated_output_tokens": min(
                features.get("estimated_output_tokens", 0) / 5000, 1.0
            ),
            "complexity_keywords": max(0.0, min(features.get("complexity_keywords", 0.5), 1.0)),
            "requires_reasoning": features.get("requires_reasoning", 0.0),
            "scope": features.get("scope", 0.3),
        }

        score = sum(normalized[k] * FEATURE_WEIGHTS[k] for k in FEATURE_WEIGHTS)
        return max(0.0, min(1.0, score))

    def _score_to_tier(self, score: float) -> ComplexityTier:
        """Map a score to a complexity tier.

        Args:
            score: Complexity score (0-1).

        Returns:
            The corresponding ComplexityTier.
        """
        if score < self._simple_threshold:
            return ComplexityTier.SIMPLE
        if score < self._medium_threshold:
            return ComplexityTier.MEDIUM
        return ComplexityTier.COMPLEX

    @staticmethod
    def _estimate_tokens(text: str) -> float:
        """Estimate token count for text."""
        if not text:
            return 0.0
        word_count = len(text.split())
        return float(max(1, int(word_count * 0.75)))

    @staticmethod
    def _estimate_output(prompt: str) -> float:
        """Estimate expected output tokens based on prompt content."""
        if not prompt:
            return 1000.0
        prompt_lower = prompt.lower()
        from prism.cost.pricing import DEFAULT_OUTPUT_ESTIMATE, OUTPUT_TOKEN_ESTIMATES

        for keyword, estimate in OUTPUT_TOKEN_ESTIMATES.items():
            if keyword in prompt_lower:
                return float(estimate)
        return float(DEFAULT_OUTPUT_ESTIMATE)

    @staticmethod
    def _keyword_score(prompt: str) -> float:
        """Compute keyword-based complexity score.

        Scans the prompt for keywords associated with different complexity levels
        and computes a weighted score.

        Args:
            prompt: User's input text.

        Returns:
            Score between 0.0 and 1.0 (0.5 is neutral).
        """
        if not prompt:
            return 0.5

        prompt_lower = prompt.lower()
        score = 0.5  # Start neutral

        # Check simple keywords (reduce score)
        for keyword, weight in SIMPLE_KEYWORDS.items():
            if keyword in prompt_lower:
                score += weight

        # Check medium keywords (moderate increase)
        for keyword, weight in MEDIUM_KEYWORDS.items():
            if keyword in prompt_lower:
                score += weight

        # Check complex keywords (significant increase)
        for keyword, weight in COMPLEX_KEYWORDS.items():
            if keyword in prompt_lower:
                score += weight

        return max(0.0, min(1.0, score))

    @staticmethod
    def _detect_reasoning(prompt: str, context: TaskContext) -> float:
        """Detect if the task requires multi-step reasoning.

        Args:
            prompt: User's input text.
            context: Task context.

        Returns:
            1.0 if reasoning is needed, 0.0 otherwise.
        """
        if not prompt:
            return 0.0

        # Long prompts with conditionals suggest reasoning
        word_count = len(prompt.split())
        if word_count > 50:
            for pattern in REASONING_PATTERNS:
                if pattern.search(prompt):
                    return 1.0

        # Many files referenced suggests cross-file reasoning
        if len(context.active_files) > 3:
            return 0.8

        # Check for explicit reasoning requests
        prompt_lower = prompt.lower()
        reasoning_phrases = ["think through", "step by step", "analyze", "consider", "evaluate"]
        for phrase in reasoning_phrases:
            if phrase in prompt_lower:
                return 0.7

        return 0.0

    @staticmethod
    def _assess_scope(prompt: str, context: TaskContext) -> float:
        """Assess the scope of the task.

        Args:
            prompt: User's input text.
            context: Task context.

        Returns:
            Scope score from 0.0 (single line) to 1.0 (architecture-level).
        """
        if not prompt:
            return 0.3

        prompt_lower = prompt.lower()

        # Check scope keywords
        for keyword, scope_score in SCOPE_PATTERNS.items():
            if keyword in prompt_lower:
                return scope_score

        # Infer from file count
        file_count = len(context.active_files)
        if file_count == 0:
            # No files — could be a vague request
            if len(prompt.split()) > 30:
                return 0.6  # Long prompt without files = potentially broad
            return 0.3
        if file_count == 1:
            return 0.1
        if file_count <= 3:
            return 0.3
        if file_count <= 6:
            return 0.5
        return 0.7

    def _explain(
        self, tier: ComplexityTier, score: float, features: dict[str, float]
    ) -> str:
        """Generate a human-readable explanation of the classification.

        Args:
            tier: The classified tier.
            score: The complexity score.
            features: The extracted features.

        Returns:
            Explanation string.
        """
        parts: list[str] = [f"Classified as {tier.value.upper()} (score: {score:.2f})"]

        keyword_score = features.get("complexity_keywords", 0.5)
        if keyword_score < 0.3:
            parts.append("simple keywords detected")
        elif keyword_score > 0.7:
            parts.append("complex keywords detected")

        if features.get("requires_reasoning", 0) > 0.5:
            parts.append("multi-step reasoning needed")

        scope = features.get("scope", 0.3)
        if scope > 0.7:
            parts.append("broad scope (architecture/system-level)")
        elif scope > 0.4:
            parts.append("moderate scope (multi-file)")
        else:
            parts.append("narrow scope (single file)")

        files = int(features.get("files_referenced", 0))
        if files > 0:
            parts.append(f"{files} file(s) in context")

        return ". ".join(parts) + "."

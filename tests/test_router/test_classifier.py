"""Tests for the task complexity classifier."""

from __future__ import annotations

import pytest

from prism.config.schema import PrismConfig
from prism.config.settings import Settings
from prism.providers.base import ComplexityTier
from prism.router.classifier import TaskClassifier, TaskContext


@pytest.fixture
def classifier(tmp_path) -> TaskClassifier:
    config = PrismConfig(prism_home=tmp_path / ".prism")
    settings = Settings(config=config, project_root=tmp_path)
    return TaskClassifier(settings)


@pytest.fixture
def empty_context() -> TaskContext:
    return TaskContext()


@pytest.fixture
def multi_file_context() -> TaskContext:
    return TaskContext(
        active_files=["src/auth.py", "src/models.py", "src/routes.py", "src/middleware.py"]
    )


class TestSimpleClassification:
    """Tasks that should be classified as SIMPLE."""

    def test_fix_typo(self, classifier: TaskClassifier, empty_context: TaskContext) -> None:
        result = classifier.classify("fix the typo in line 5", empty_context)
        assert result.tier == ComplexityTier.SIMPLE

    def test_rename_variable(self, classifier: TaskClassifier, empty_context: TaskContext) -> None:
        result = classifier.classify("rename the variable x to user_count", empty_context)
        assert result.tier == ComplexityTier.SIMPLE

    def test_explain_function(self, classifier: TaskClassifier, empty_context: TaskContext) -> None:
        result = classifier.classify("explain what does this function do", empty_context)
        assert result.tier == ComplexityTier.SIMPLE

    def test_format_code(self, classifier: TaskClassifier, empty_context: TaskContext) -> None:
        result = classifier.classify("format this code properly", empty_context)
        assert result.tier == ComplexityTier.SIMPLE

    def test_fix_indentation(self, classifier: TaskClassifier, empty_context: TaskContext) -> None:
        result = classifier.classify("fix the indentation in this file", empty_context)
        assert result.tier == ComplexityTier.SIMPLE

    def test_add_import(self, classifier: TaskClassifier, empty_context: TaskContext) -> None:
        result = classifier.classify("add the missing import for os", empty_context)
        assert result.tier == ComplexityTier.SIMPLE

    def test_fix_spelling(self, classifier: TaskClassifier, empty_context: TaskContext) -> None:
        result = classifier.classify("fix the spelling error", empty_context)
        assert result.tier == ComplexityTier.SIMPLE


class TestMediumClassification:
    """Tasks that should be classified as MEDIUM."""

    def test_refactor_function(
        self, classifier: TaskClassifier, empty_context: TaskContext
    ) -> None:
        result = classifier.classify(
            "refactor the authenticate module to use async await patterns and improve error handling across multiple files", empty_context
        )
        assert result.tier == ComplexityTier.MEDIUM

    def test_implement_feature(
        self, classifier: TaskClassifier, empty_context: TaskContext
    ) -> None:
        result = classifier.classify(
            "implement a feature to validate email addresses in the api endpoint", empty_context
        )
        assert result.tier == ComplexityTier.MEDIUM

    def test_debug_with_context(
        self, classifier: TaskClassifier, multi_file_context: TaskContext
    ) -> None:
        result = classifier.classify("debug why the test is failing", multi_file_context)
        assert result.tier == ComplexityTier.MEDIUM

    def test_write_tests(self, classifier: TaskClassifier, multi_file_context: TaskContext) -> None:
        result = classifier.classify(
            "write comprehensive test cases for the database module including edge cases, error paths, and integration tests for all query functions",
            multi_file_context,
        )
        assert result.tier == ComplexityTier.MEDIUM

    def test_add_error_handling(
        self, classifier: TaskClassifier, multi_file_context: TaskContext
    ) -> None:
        result = classifier.classify(
            "add proper error handling to the api endpoint with retry logic, validation, and custom exception types across the service layer",
            multi_file_context,
        )
        assert result.tier == ComplexityTier.MEDIUM


class TestComplexClassification:
    """Tasks that should be classified as COMPLEX."""

    def test_design_architecture(
        self, classifier: TaskClassifier, empty_context: TaskContext
    ) -> None:
        result = classifier.classify(
            "design a microservices architecture for the payment system with "
            "event sourcing and CQRS pattern. Consider scalability and trade-offs.",
            empty_context,
        )
        assert result.tier == ComplexityTier.COMPLEX

    def test_architect_from_scratch(
        self, classifier: TaskClassifier, multi_file_context: TaskContext
    ) -> None:
        result = classifier.classify(
            "architect a distributed real-time notification system from scratch. "
            "Consider the trade-offs between WebSocket and SSE approaches. "
            "If we need to handle 100k concurrent connections, evaluate "
            "whether a message queue like RabbitMQ would be more scalable.",
            multi_file_context,
        )
        assert result.tier == ComplexityTier.COMPLEX

    def test_security_audit(
        self, classifier: TaskClassifier, multi_file_context: TaskContext
    ) -> None:
        result = classifier.classify(
            "perform a comprehensive security audit of the authentication system. "
            "Evaluate potential vulnerabilities including SQL injection, XSS, CSRF. "
            "Consider whether the current approach handles edge cases correctly "
            "and suggest architecture improvements for defense-in-depth.",
            multi_file_context,
        )
        assert result.tier == ComplexityTier.COMPLEX

    def test_optimize_system(
        self, classifier: TaskClassifier, multi_file_context: TaskContext
    ) -> None:
        result = classifier.classify(
            "optimize the entire database query layer for performance. "
            "The system needs to be scalable to handle concurrent requests. "
            "Analyze the trade-offs between caching strategies and consider "
            "whether connection pooling would improve throughput. "
            "If the indexes are insufficient, redesign the schema.",
            multi_file_context,
        )
        assert result.tier == ComplexityTier.COMPLEX


class TestFeatureExtraction:
    """Test individual feature extraction methods."""

    def test_extract_features_returns_all_keys(
        self, classifier: TaskClassifier, empty_context: TaskContext
    ) -> None:
        features = classifier.extract_features("hello world", empty_context)
        expected_keys = {
            "prompt_token_count",
            "files_referenced",
            "estimated_output_tokens",
            "complexity_keywords",
            "requires_reasoning",
            "scope",
        }
        assert set(features.keys()) == expected_keys

    def test_files_referenced_from_context(
        self, classifier: TaskClassifier, multi_file_context: TaskContext
    ) -> None:
        features = classifier.extract_features("do something", multi_file_context)
        assert features["files_referenced"] == 4.0

    def test_empty_prompt(self, classifier: TaskClassifier, empty_context: TaskContext) -> None:
        features = classifier.extract_features("", empty_context)
        assert features["prompt_token_count"] == 0.0
        assert features["complexity_keywords"] == 0.5  # Neutral

    def test_reasoning_detection_with_conditionals(
        self, classifier: TaskClassifier, empty_context: TaskContext
    ) -> None:
        long_prompt = (
            "If the user is authenticated then redirect to dashboard, "
            "but if they have 2FA enabled then show the verification page first. "
            "Consider the case when the session has expired. " * 3
        )
        features = classifier.extract_features(long_prompt, empty_context)
        assert features["requires_reasoning"] > 0.0

    def test_scope_single_file(self, classifier: TaskClassifier) -> None:
        context = TaskContext(active_files=["main.py"])
        features = classifier.extract_features("fix this", context)
        assert features["scope"] == 0.1

    def test_scope_architecture_keyword(
        self, classifier: TaskClassifier, empty_context: TaskContext
    ) -> None:
        features = classifier.extract_features(
            "redesign the system architecture", empty_context
        )
        assert features["scope"] >= 0.9


class TestScoring:
    """Test the scoring and threshold logic."""

    def test_score_is_between_0_and_1(
        self, classifier: TaskClassifier, empty_context: TaskContext
    ) -> None:
        for prompt in ["fix typo", "refactor module", "design architecture from scratch"]:
            score = classifier.get_score(prompt, empty_context)
            assert 0.0 <= score <= 1.0

    def test_simple_scores_below_threshold(
        self, classifier: TaskClassifier, empty_context: TaskContext
    ) -> None:
        score = classifier.get_score("fix the typo", empty_context)
        assert score < 0.3

    def test_complex_scores_above_threshold(
        self, classifier: TaskClassifier, multi_file_context: TaskContext
    ) -> None:
        score = classifier.get_score(
            "architect a distributed scalable microservice system from scratch. "
            "Consider the trade-offs between consistency and availability. "
            "If we need strong consistency, evaluate whether event sourcing is appropriate.",
            multi_file_context,
        )
        assert score > 0.55


class TestClassificationResult:
    """Test the classification result object."""

    def test_result_has_all_fields(
        self, classifier: TaskClassifier, empty_context: TaskContext
    ) -> None:
        result = classifier.classify("fix typo", empty_context)
        assert result.tier in ComplexityTier
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.features, dict)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_result_reasoning_mentions_tier(
        self, classifier: TaskClassifier, empty_context: TaskContext
    ) -> None:
        result = classifier.classify("fix typo", empty_context)
        assert result.tier.value.upper() in result.reasoning


class TestCustomThresholds:
    """Test with custom classification thresholds."""

    def test_lowered_thresholds(self, tmp_path) -> None:
        config = PrismConfig(prism_home=tmp_path / ".prism")
        config.routing.simple_threshold = 0.1
        config.routing.medium_threshold = 0.4
        settings = Settings(config=config, project_root=tmp_path)
        classifier = TaskClassifier(settings)

        # With lower thresholds, more things classify as complex
        result = classifier.classify("implement a feature with error handling", TaskContext())
        # Just verify it runs without error and returns a valid tier
        assert result.tier in ComplexityTier

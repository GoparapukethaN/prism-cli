# CONVENTIONS.md — Prism Coding Conventions

## Python Version
- Minimum: Python 3.11
- Target: Python 3.12 (for improved error messages, performance)
- Use modern typing features: `str | None` instead of `Optional[str]`, `list[str]` instead of `List[str]`

## File Organization

### Module Structure
Every module directory follows this pattern:
```
module/
├── __init__.py        # Public API exports only
├── base.py            # Abstract interfaces (if applicable)
├── implementation.py  # Core implementation
└── exceptions.py      # Module-specific exceptions (if needed)
```

### `__init__.py` Rules
- Export only public API symbols
- No logic in `__init__.py` — imports only
- Use `__all__` to declare public API explicitly

```python
# src/prism/router/__init__.py
from prism.router.classifier import TaskClassifier, ComplexityTier
from prism.router.selector import ModelSelector, ModelSelection
from prism.router.cost_estimator import CostEstimator

__all__ = [
    "TaskClassifier",
    "ComplexityTier",
    "ModelSelector",
    "ModelSelection",
    "CostEstimator",
]
```

## Naming Conventions

### Files
- All Python files: `snake_case.py`
- Test files: `test_<module>.py` (e.g., `test_classifier.py`)
- Config files: lowercase with dots/hyphens (e.g., `pyproject.toml`)
- Documentation: `UPPER_CASE.md`

### Python
| Entity | Convention | Example |
|--------|-----------|---------|
| Module | snake_case | `cost_estimator.py` |
| Class | PascalCase | `TaskClassifier` |
| Function/Method | snake_case | `classify_task()` |
| Variable | snake_case | `token_count` |
| Constant | UPPER_SNAKE | `SIMPLE_THRESHOLD` |
| Enum member | UPPER_SNAKE | `ComplexityTier.SIMPLE` |
| Private | leading underscore | `_compute_score()` |
| Type variable | PascalCase + T | `ProviderT` |
| Protocol | PascalCase + suffix | `ToolProtocol` |

### Database
- Table names: snake_case, plural (`routing_decisions`, `cost_entries`)
- Column names: snake_case (`model_id`, `input_tokens`, `created_at`)
- Index names: `idx_<table>_<column>` (e.g., `idx_routing_decisions_created_at`)

## Type Hints

### Required Everywhere
```python
# Functions — parameters and return types
def classify(self, prompt: str, context: TaskContext) -> ComplexityTier:
    ...

# Variables where type is non-obvious
candidates: list[ModelCandidate] = []
cost_map: dict[str, float] = {}

# Class attributes
class RouterConfig:
    simple_threshold: float = 0.3
    medium_threshold: float = 0.7
    daily_budget: float | None = None
```

### Avoid `Any`
Use `Any` only when interfacing with untyped third-party libraries. Always add a comment:
```python
# LiteLLM response type is untyped
response: Any = await litellm.acompletion(**params)  # type: ignore[no-untyped-call]
```

## Error Handling

### Exception Hierarchy
```python
class PrismError(Exception):
    """Base exception for all Prism errors."""

class ConfigError(PrismError):
    """Configuration-related errors."""

class AuthError(PrismError):
    """Authentication/credential errors."""

class ProviderError(PrismError):
    """Provider communication errors."""

class RoutingError(PrismError):
    """Routing decision errors."""

class ToolError(PrismError):
    """Tool execution errors."""

class SecurityError(PrismError):
    """Security violation errors."""

class BudgetExceededError(RoutingError):
    """Budget limit exceeded."""
```

### Rules
1. Always catch specific exceptions, never bare `except:`
2. Always re-raise or handle — never silently swallow
3. Include context in exception messages
4. Use `from` for exception chaining: `raise ProviderError(...) from e`
5. Log at appropriate level before raising

## Logging

### Format
```python
import structlog

logger = structlog.get_logger(__name__)

# Structured logging with context
logger.info("model_selected", model="deepseek-v3", tier="medium", cost=0.0023)
logger.error("provider_failed", provider="anthropic", error=str(e), fallback="deepseek-v3")
```

### Levels
| Level | Use Case |
|-------|----------|
| DEBUG | Internal state, feature vectors, scoring details |
| INFO | Routing decisions, model selections, cost tracking |
| WARNING | Rate limits hit, fallback activated, budget approaching limit |
| ERROR | Provider failures, tool execution errors, security violations |
| CRITICAL | Data corruption, unrecoverable state, security breach |

## Async Patterns

### Default to Async for I/O
```python
# API calls — always async
async def complete(self, prompt: str, model: str) -> CompletionResult:
    response = await litellm.acompletion(model=model, messages=messages)
    ...

# File I/O — sync is fine for local filesystem
def read_file(self, path: Path) -> str:
    return path.read_text()

# HTTP requests — always async
async def fetch_page(self, url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30)
        ...
```

### Concurrency
- Use `asyncio.gather()` for parallel independent operations
- Use `asyncio.Semaphore` for rate limiting concurrent requests
- Never mix sync and async without `asyncio.to_thread()` for blocking calls

## Configuration

### Hierarchy (highest priority first)
1. Command-line flags (`--model`, `--budget`)
2. Environment variables (`PRISM_MODEL`, `PRISM_BUDGET`)
3. Project config (`.prism.md` or `.prism.yaml` in project root)
4. User config (`~/.prism/config.yaml`)
5. Default values (hardcoded in `src/prism/config/defaults.py`)

### Config Format
```yaml
# ~/.prism/config.yaml
routing:
  simple_threshold: 0.3
  medium_threshold: 0.7
  exploration_rate: 0.1

budget:
  daily_limit: 5.00
  monthly_limit: 50.00
  warn_at_percent: 80

providers:
  anthropic:
    enabled: true
    preferred_models: ["claude-sonnet-4-20250514"]
  ollama:
    enabled: true
    models: ["qwen2.5-coder:7b", "llama3.2:3b"]

tools:
  web_browsing: false
  allowed_commands:
    - "npm test"
    - "python -m pytest"

permissions:
  auto_approve_reads: true
  auto_approve_writes: false  # --yes overrides
  always_confirm_deletes: true
```

## Testing Conventions

### File Structure
```
tests/
├── conftest.py              # Shared fixtures
├── test_router/
│   ├── conftest.py          # Router-specific fixtures
│   ├── test_classifier.py
│   ├── test_selector.py
│   └── test_cost_estimator.py
```

### Test Naming
```python
def test_<function>_<scenario>_<expected_result>():
    """Tests should read like specifications."""

# Examples:
def test_classify_simple_prompt_returns_simple_tier():
def test_classify_multi_file_refactor_returns_medium_tier():
def test_selector_falls_back_on_rate_limit():
def test_path_guard_rejects_traversal_attempt():
def test_budget_blocks_when_daily_limit_exceeded():
```

### Fixture Patterns
```python
@pytest.fixture
def classifier() -> TaskClassifier:
    return TaskClassifier(config=default_config())

@pytest.fixture
def mock_litellm(mocker) -> MagicMock:
    return mocker.patch("prism.router.selector.litellm")

@pytest.fixture
def temp_project(tmp_path) -> Path:
    """Creates a temporary project directory with basic structure."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    return tmp_path
```

### Markers
```python
@pytest.mark.slow          # Tests that take > 1s
@pytest.mark.integration   # Tests that need external services
@pytest.mark.security      # Security-focused tests
```

## Docstring Format (Google Style)

```python
def select_model(
    self,
    tier: ComplexityTier,
    context: TaskContext,
    budget_remaining: float,
) -> ModelSelection:
    """Select the optimal model for a classified task.

    Evaluates candidate models for the given complexity tier,
    estimates costs, checks budget constraints, and returns the
    best model with a fallback chain.

    Args:
        tier: The classified complexity tier for the task.
        context: Current task context including active files.
        budget_remaining: Remaining budget in USD for the current period.

    Returns:
        A ModelSelection containing the chosen model and fallback chain.

    Raises:
        BudgetExceededError: If no model fits within the remaining budget.
        RoutingError: If no candidate models are available for the tier.
    """
```

## Import Order

```python
# 1. Standard library
from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# 2. Third-party
import litellm
import typer
from rich.console import Console
from pydantic import BaseModel, Field

# 3. Local (absolute imports only)
from prism.config.settings import Settings
from prism.router.classifier import ComplexityTier, TaskClassifier
from prism.security.path_guard import PathGuard
```

## Git Workflow

### Branch Naming
- `feat/<module>-<description>` — new feature
- `fix/<module>-<description>` — bug fix
- `refactor/<module>-<description>` — code restructuring
- `test/<module>-<description>` — test additions
- `docs/<description>` — documentation

### Commit Messages
```
type(scope): concise description

feat(router): add adaptive learning from outcome tracking
fix(tools): prevent path traversal via symlink resolution
test(auth): add keyring storage integration tests
refactor(cli): extract REPL slash command handlers
perf(context): cache tree-sitter repo maps between requests
security(tools): filter API keys from subprocess environment
```

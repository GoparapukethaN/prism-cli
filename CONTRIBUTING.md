# Contributing to Prism

Thank you for your interest in contributing to Prism! This document explains how to get started, the development workflow, and how to submit high-quality contributions.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Add a Provider](#how-to-add-a-provider)
- [How to Add a Tool](#how-to-add-a-tool)
- [How to Add a Plugin](#how-to-add-a-plugin)
- [Development Workflow](#development-workflow)
- [Pull Request Checklist](#pull-request-checklist)
- [Testing Guidelines](#testing-guidelines)
- [Code Style](#code-style)
- [Reporting Issues](#reporting-issues)
- [Code of Conduct](#code-of-conduct)

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- (Optional) Ollama for local model testing
- (Optional) `pipx` for isolated CLI installation

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/prism-cli.git
cd prism-cli

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# .venv\Scripts\activate   # Windows

# 3. Install with all optional dependencies and dev tools
pip install -e ".[all,dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify everything works
ruff check src/ tests/
pytest tests/ --cov=src/prism --cov-fail-under=90
bandit -r src/prism/ -c pyproject.toml
```

If all checks pass, you are ready to contribute.

## How to Add a Provider

Prism uses LiteLLM under the hood, so adding a new provider is straightforward:

1. **Create the provider config** at `src/prism/providers/<provider_name>.py`:
   - Define a `ProviderConfig` dataclass with the provider's models, pricing, and tier assignments.
   - Each model needs: `id`, `display_name`, `tier` (simple/medium/complex), `input_cost_per_1m`, `output_cost_per_1m`, `context_window`.

2. **Register the provider** in `src/prism/providers/registry.py`:
   - Add the provider to the `PROVIDER_REGISTRY` dict.
   - Set the LiteLLM prefix (e.g., `"anthropic/"`, `"openai/"`).

3. **Add tests** at `tests/test_providers/test_<provider_name>.py`:
   - Test model listing, pricing lookups, and tier assignments.
   - All provider interactions must be mocked (no real API calls).

4. **Update documentation** if needed.

## How to Add a Tool

Tools extend Prism's capabilities (file ops, search, terminal, web):

1. **Implement the Tool interface** at `src/prism/tools/<tool_name>.py`:
   ```python
   from prism.tools.base import Tool, ToolResult

   class MyTool(Tool):
       @property
       def name(self) -> str:
           return "my_tool"

       @property
       def description(self) -> str:
           return "Description of what this tool does"

       @property
       def parameters_schema(self) -> dict:
           return {
               "type": "object",
               "properties": {
                   "param1": {"type": "string", "description": "..."},
               },
               "required": ["param1"],
           }

       @property
       def permission_level(self) -> str:
           return "read"  # or "write", "execute"

       async def execute(self, **kwargs) -> ToolResult:
           # Implementation here
           ...
   ```

2. **Register** in `src/prism/tools/registry.py`.

3. **Add tests** at `tests/test_tools/test_<tool_name>.py`.

4. **Security**: Ensure all inputs are validated and paths are confined to the project root.

## How to Add a Plugin

Plugins are optional extensions that can be enabled/disabled:

1. Create the plugin module under `src/prism/plugins/<plugin_name>/`.
2. Implement a `setup()` function that registers the plugin's tools and hooks.
3. Add an entry point in `pyproject.toml` if the plugin should be discoverable.
4. Add tests at `tests/test_plugins/test_<plugin_name>.py`.

## Development Workflow

### 1. Create a branch

```bash
git checkout main
git pull origin main
git checkout -b feat/your-feature-name
```

### 2. Make changes

- Follow the coding standards in [Code Style](#code-style).
- Write tests for all new code.
- Run the review suite frequently:
  ```bash
  ruff check src/ tests/
  pytest tests/ --cov=src/prism
  bandit -r src/prism/ -c pyproject.toml
  ```

### 3. Commit

Follow the conventional commit format:

```
type(scope): concise description

Types: feat, fix, refactor, test, docs, chore, perf, security
Scope: cli, router, providers, tools, context, auth, db, cost, git, security, config
```

Examples:
- `feat(router): add context-window-aware model selection`
- `fix(tools): prevent path traversal in file_read`
- `test(auth): add keyring fallback coverage`

### 4. Push and create a PR

```bash
git push -u origin feat/your-feature-name
gh pr create --title "feat(router): add context-window-aware model selection"
```

## Pull Request Checklist

Before submitting your PR, ensure:

- [ ] All new code has type hints on every function signature and return type
- [ ] All public functions and classes have Google-style docstrings
- [ ] No hardcoded secrets, API keys, or tokens anywhere
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `ruff format --check src/ tests/` passes
- [ ] `bandit -r src/prism/ -c pyproject.toml` reports no issues
- [ ] `pytest tests/ --cov=src/prism --cov-fail-under=90` passes
- [ ] New code has test coverage of at least 90%
- [ ] Security modules have test coverage of at least 95%
- [ ] No real API calls in tests (all mocked)
- [ ] PR title follows conventional commit format
- [ ] PR description explains the "why", not just the "what"

## Testing Guidelines

### Critical Rules

1. **No real API calls.** All tests must run completely offline. Use `pytest-mock`, `respx`, and hardcoded mock responses.
2. **No real API keys.** Use obviously fake keys like `"sk-test-key-1234"`.
3. **Coverage minimum**: 90% overall, 95% for security modules.
4. **Test file naming**: `test_<module_name>.py` in the corresponding `tests/test_<package>/` directory.

### Test Structure

```python
"""Tests for the thing being tested."""

from __future__ import annotations

import pytest

from prism.module import ThingBeingTested


class TestThingBeingTested:
    """Group related tests in a class."""

    def test_happy_path(self) -> None:
        """Test the normal case."""
        result = ThingBeingTested().do_something("input")
        assert result == "expected"

    def test_edge_case_empty_input(self) -> None:
        """Test with empty input."""
        result = ThingBeingTested().do_something("")
        assert result == ""

    def test_error_case(self) -> None:
        """Test that errors are handled correctly."""
        with pytest.raises(ValueError, match="specific message"):
            ThingBeingTested().do_something(None)
```

## Code Style

- **Python 3.11+** minimum, use modern syntax (`X | Y` unions, `match/case`, etc.)
- **Type hints everywhere**: every function parameter, return type, and non-obvious variable
- **Docstrings**: Google style on every public function and class
- **Formatter/linter**: `ruff` (replaces black + isort + flake8)
- **Line length**: 100 characters maximum
- **Imports**: stdlib, then third-party, then local (enforced by ruff isort)
- **Paths**: Use `pathlib.Path` instead of `os.path`
- **Data structures**: `dataclasses` or `pydantic` models
- **I/O**: Async by default (`httpx`, `litellm` async calls)
- **Resources**: Context managers for DB connections, file handles, etc.

## Reporting Issues

### Bug Reports

Open an issue using the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- Prism version (`prism --version`)
- Python version
- Operating system
- Steps to reproduce
- Expected vs. actual behavior
- Relevant logs (with secrets redacted)

### Feature Requests

Open an issue using the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- Use case description
- Proposed solution
- Alternatives considered

### Security Vulnerabilities

Do NOT open a public issue. See [SECURITY.md](.github/SECURITY.md) for responsible disclosure instructions.

## Code of Conduct

- Be respectful and constructive in all interactions.
- Focus on the code, not the person.
- Welcome newcomers and help them get started.
- Assume good intentions.

## License

By contributing to Prism, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).

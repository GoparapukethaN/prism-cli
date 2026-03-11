# CLAUDE.md — Prism Project Instructions

## Project Overview

Prism is an open-source CLI tool that provides Claude Code-level capabilities (file system access, terminal execution, web browsing, code generation, natural language conversation) while intelligently routing tasks across every major AI provider at a fraction of the cost. Users bring their own API keys and Prism's routing engine automatically dispatches each task to the cheapest model that can handle it well.

## Golden Rules — NEVER Violate These

1. **Never truncate code.** Always write complete implementations. Every function, every class, every module — complete.
2. **Never use placeholders.** No "add logic here", "implement this later", "TODO", or "pass" in production code. If it exists, it must work.
3. **Always production-ready.** Full error handling, input validation, edge case coverage on every piece of code.
4. **Always write tests.** Every module gets a corresponding test file. No exceptions.
5. **Security audit after every module.** Run `bandit`, check for OWASP top 10, validate no secrets leak.
6. **Code review after every module.** Check for code quality, DRY violations, performance issues, type safety.
7. **Never hardcode secrets.** API keys, passwords, tokens — always from keyring, env vars, or encrypted config. Never in source.
8. **Always validate inputs.** Every function that accepts external data must sanitize and validate.
9. **Always handle edge cases.** Empty strings, None values, negative numbers, concurrent access, network failures.
10. **Update MEMORY.md and PROGRESS.md after every task.** Non-negotiable tracking.
11. **Write HANDOFF.md at end of every session.** Future context must be preserved.

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| CLI framework | Typer | 0.9+ |
| Terminal UI | Rich | 13+ |
| Interactive input | Prompt Toolkit | 3.0+ |
| Unified AI API | LiteLLM | 1.x |
| Code analysis | tree-sitter (py-tree-sitter) | latest |
| Web browsing | Playwright | latest |
| Lightweight HTTP | httpx | latest |
| HTML parsing | BeautifulSoup4 | latest |
| Vector search | ChromaDB | optional |
| Credential storage | keyring | latest |
| Database | sqlite3 (stdlib) | stdlib |
| Diff generation | difflib (stdlib) + unidiff | latest |
| File watching | watchdog | latest |
| Testing | pytest + pytest-cov + pytest-asyncio | latest |
| Linting | ruff | latest |
| Type checking | mypy | latest |
| Security scanning | bandit | latest |
| Packaging | PyPI + pipx | latest |

## Project Structure

```
prism/
├── src/
│   └── prism/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── app.py              # Typer app, command definitions
│       │   ├── repl.py             # Interactive REPL loop
│       │   ├── commands/
│       │   │   ├── __init__.py
│       │   │   ├── auth.py         # prism auth add/remove/status
│       │   │   ├── init.py         # prism init setup wizard
│       │   │   ├── ask.py          # prism ask (single-shot)
│       │   │   ├── edit.py         # prism edit (single-shot edit)
│       │   │   ├── run.py          # prism run (execute + fix)
│       │   │   └── config.py       # prism config get/set
│       │   └── ui/
│       │       ├── __init__.py
│       │       ├── display.py      # Rich rendering (diffs, tables, markdown)
│       │       ├── prompts.py      # Prompt Toolkit input handling
│       │       └── themes.py       # Color themes and styling
│       ├── router/
│       │   ├── __init__.py
│       │   ├── classifier.py       # Task complexity classification
│       │   ├── selector.py         # Model selection and fallback chains
│       │   ├── cost_estimator.py   # Token count estimation, cost calculation
│       │   ├── budget.py           # Budget enforcement (daily/monthly caps)
│       │   ├── fallback.py         # Fallback chain management
│       │   ├── learning.py         # Adaptive learning from outcomes
│       │   └── rate_limiter.py     # Per-provider rate limiting
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── registry.py         # Provider registration and discovery
│       │   ├── base.py             # Abstract provider interface
│       │   ├── anthropic.py        # Anthropic-specific config
│       │   ├── openai.py           # OpenAI-specific config
│       │   ├── google.py           # Google AI Studio config
│       │   ├── deepseek.py         # DeepSeek config
│       │   ├── groq.py             # Groq config
│       │   ├── mistral.py          # Mistral config
│       │   ├── ollama.py           # Local Ollama config
│       │   └── custom.py           # Custom OpenAI-compatible endpoints
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py             # Abstract Tool interface
│       │   ├── registry.py         # Tool registration
│       │   ├── file_read.py        # read_file tool
│       │   ├── file_write.py       # write_file tool
│       │   ├── file_edit.py        # edit_file (search/replace)
│       │   ├── directory.py        # list_directory tool
│       │   ├── search.py           # search_codebase (ripgrep)
│       │   ├── terminal.py         # execute_command (sandboxed)
│       │   ├── browser.py          # browse_web (Playwright)
│       │   ├── screenshot.py       # screenshot tool
│       │   └── permissions.py      # Permission manager for tools
│       ├── context/
│       │   ├── __init__.py
│       │   ├── manager.py          # Context window management
│       │   ├── repo_map.py         # tree-sitter repository map
│       │   ├── summarizer.py       # Rolling conversation summarization
│       │   ├── session.py          # Session persistence and resume
│       │   ├── memory.py           # Project memory (.prism.md)
│       │   └── rag.py              # ChromaDB RAG (optional)
│       ├── auth/
│       │   ├── __init__.py
│       │   ├── keyring_store.py    # OS keyring integration
│       │   ├── env_store.py        # Environment variable fallback
│       │   ├── encrypted_store.py  # AES-256 encrypted config
│       │   └── validator.py        # API key validation
│       ├── db/
│       │   ├── __init__.py
│       │   ├── database.py         # SQLite connection management
│       │   ├── models.py           # Data models / schema
│       │   ├── migrations.py       # Schema migrations
│       │   └── queries.py          # Query functions
│       ├── cost/
│       │   ├── __init__.py
│       │   ├── tracker.py          # Cost tracking per request
│       │   ├── dashboard.py        # /cost command display
│       │   ├── pricing.py          # Model pricing data
│       │   └── budget.py           # Budget enforcement logic
│       ├── git/
│       │   ├── __init__.py
│       │   ├── operations.py       # Git operations (commit, diff, undo)
│       │   └── auto_commit.py      # Automatic commit on file edits
│       ├── security/
│       │   ├── __init__.py
│       │   ├── sandbox.py          # Command execution sandbox
│       │   ├── path_guard.py       # Path traversal prevention
│       │   ├── secret_filter.py    # Filter secrets from subprocess env
│       │   └── audit.py            # Audit logging
│       └── config/
│           ├── __init__.py
│           ├── settings.py         # Global settings management
│           ├── defaults.py         # Default configuration values
│           └── schema.py           # Configuration schema validation
├── tests/
│   ├── conftest.py
│   ├── test_cli/
│   ├── test_router/
│   ├── test_providers/
│   ├── test_tools/
│   ├── test_context/
│   ├── test_auth/
│   ├── test_db/
│   ├── test_cost/
│   ├── test_git/
│   └── test_security/
├── pyproject.toml
├── .gitignore
├── .env.example
├── LICENSE                          # Apache 2.0
└── README.md
```

## Coding Standards

- **Python 3.11+** minimum
- **Type hints everywhere** — every function signature, every return type, every variable where non-obvious
- **Docstrings** on every public function and class (Google style)
- **ruff** for linting and formatting (replaces black + isort + flake8)
- **mypy --strict** for type checking
- **pytest** for testing with minimum 90% coverage
- All imports sorted: stdlib → third-party → local
- Max line length: 100 characters
- Use `pathlib.Path` over `os.path` everywhere
- Use `dataclasses` or `pydantic` for data structures
- Async by default for I/O operations (httpx, LiteLLM calls)
- Context managers for resource management (DB connections, file handles)

## Commit Convention

```
type(scope): description

Types: feat, fix, refactor, test, docs, chore, perf, security
Scope: cli, router, providers, tools, context, auth, db, cost, git, security, config
```

## Security Requirements

- All file operations confined to project root via realpath resolution
- API keys never logged, never in error messages, never in git
- Command execution sandboxed with timeout, output limits, env filtering
- Sensitive file patterns excluded from file operations by default
- Audit log for every tool execution at ~/.prism/audit.log
- Path traversal prevention on every file operation
- Input sanitization on every user-facing function

## Testing Requirements

- Every module has a corresponding test file
- Unit tests for all business logic
- Integration tests for provider interactions (mocked)
- End-to-end tests for CLI commands
- Security-focused tests for path traversal, injection, etc.
- Performance benchmarks for routing decisions
- Minimum 90% code coverage enforced in CI

## After Every Module Completion

1. Run `pytest tests/test_<module>/` — all tests must pass
2. Run `ruff check src/prism/<module>/` — no lint errors
3. Run `mypy src/prism/<module>/` — no type errors
4. Run `bandit -r src/prism/<module>/` — no security issues
5. Update PROGRESS.md with completion status
6. Update MEMORY.md with any new patterns or decisions
7. Perform code review checklist (see CODE_REVIEW.md)

## File Naming

- Snake_case for all Python files
- Test files: `test_<module_name>.py`
- Config files: lowercase with dots (pyproject.toml, .gitignore)
- Documentation: UPPER_CASE.md for project docs

## Import Order

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import typer
from rich.console import Console
from litellm import completion

# Local
from prism.router.classifier import TaskClassifier
from prism.tools.base import Tool
```

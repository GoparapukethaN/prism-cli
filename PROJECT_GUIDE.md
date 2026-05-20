# PROJECT_GUIDE.md - Prism Project Notes

## Project Overview

Prism is an experimental open-source CLI for exploring model routing, cost tracking,
project memory, guarded tool execution, and developer workflow automation. Users bring
their own provider keys; local development currently emphasizes deterministic tests,
mocked provider behavior, and clear boundaries around features that still need live
provider validation.

## Working Standards

1. Keep implementations complete for the scoped behavior being changed.
2. Avoid placeholder production paths unless the limitation is explicit and tracked.
3. Prefer clear boundaries around external input, file access, subprocesses,
   provider credentials, and network calls.
4. Add or update tests when behavior changes.
5. Run local verification before treating a change as ready.
6. Keep secrets out of source; use keyring, environment variables, or encrypted config.
7. Update progress and known-issue docs when verification evidence changes.

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

These are target standards for release hardening. The current local gate is documented in
[SHOWCASE_STATUS.md](SHOWCASE_STATUS.md), and known non-gating gaps are tracked in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

- **Python 3.11+** minimum
- **Type hints on touched code paths** first; full strict typing is a hardening target
- **Docstrings** on public functions and classes where behavior is not obvious
- **ruff** for linting now; format enforcement after a dedicated formatting pass
- **mypy --strict** as a target release gate, not a current clean gate
- **pytest** for runtime behavior, with coverage enforcement as a release target
- Imports sorted: stdlib -> third-party -> local
- Max line length: 100 characters
- Use `pathlib.Path` over `os.path` where practical
- Use `dataclasses` or `pydantic` for data structures
- Async by default for I/O operations (httpx, LiteLLM calls)
- Context managers for resource management (DB connections, file handles)

## Commit Convention

```
type(scope): description

Types: feat, fix, refactor, test, docs, chore, perf, security
Scope: cli, router, providers, tools, context, auth, db, cost, git, security, config
```

## Security Targets

- All file operations confined to project root via realpath resolution
- Secret handling is designed to keep API keys out of logs, error messages, and git
- Command execution sandboxed with timeout, output limits, env filtering
- Sensitive file patterns excluded from file operations by default
- Audit log for every tool execution at ~/.prism/audit.log
- Path traversal coverage on file-operation boundaries
- Input validation on user-facing command boundaries

## Testing Targets

- Every module has a corresponding test file
- Unit tests for all business logic
- Integration tests for provider interactions (mocked)
- End-to-end tests for CLI commands
- Security-focused tests for path traversal, injection, etc.
- Performance benchmarks for routing decisions
- Minimum 90% code coverage before it is enforced as a release gate

## After Every Module Completion

1. Run `pytest tests/test_<module>/` — all tests must pass
2. Run `ruff check src/prism/<module>/` — no lint errors
3. Run `mypy src/prism/<module>/` where the module is already type-clean, or record the
   remaining typing debt in the hardening backlog
4. Run `bandit -r src/prism/<module>/` for security-sensitive modules, then record any
   accepted findings before treating the gate as clean
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

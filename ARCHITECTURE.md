# ARCHITECTURE.md — Prism System Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CLI Interface Layer                            │
│                 Typer + Rich + Prompt Toolkit                         │
│          Interactive REPL │ Single-shot mode │ Pipe mode              │
└─────────────────┬──────────────────────────────────┬────────────────┘
                  │                                  │
     ┌────────────▼─────────────┐      ┌─────────────▼───────────────┐
     │     Task Classifier       │      │     Context Manager          │
     │   Rule-based → ML model   │      │   Repo map (tree-sitter)    │
     │   Feature extraction      │      │   Rolling summarization      │
     │   Complexity scoring      │      │   Prompt caching             │
     └────────────┬─────────────┘      │   RAG (ChromaDB, optional)   │
                  │                     └─────────────┬───────────────┘
     ┌────────────▼───────────────────────────────────▼───────────────┐
     │                      Model Router                               │
     │                LiteLLM unified interface                        │
     │    Cost estimator │ Fallback chains │ Budget enforcement         │
     │    Quality tracker │ Adaptive learning │ Rate limiter            │
     └────────────┬───────────────────────────────────┬───────────────┘
                  │                                   │
     ┌────────────▼───────────┐       ┌───────────────▼──────────────┐
     │    Cloud Providers      │       │    Local Providers            │
     │  Anthropic │ OpenAI     │       │  Ollama (localhost:11434)     │
     │  Google │ DeepSeek      │       │  llama.cpp                    │
     │  Mistral │ Groq         │       │  Any local server             │
     │  Custom endpoints       │       │                               │
     └────────────┬───────────┘       └───────────────┬──────────────┘
                  │                                   │
     ┌────────────▼───────────────────────────────────▼───────────────┐
     │                  Tool Execution Layer                            │
     │  read_file │ write_file │ edit_file │ search_codebase           │
     │  execute_command │ browse_web │ screenshot                      │
     │  Permission manager │ Sandboxing │ Audit logger                 │
     └────────────┬──────────────────────────────────────────────────┘
                  │
     ┌────────────▼──────────────────────────────────────────────────┐
     │                  Persistence Layer                              │
     │  SQLite: routing history, cost tracking, learning data         │
     │  Keyring: API keys (OS credential manager)                     │
     │  Filesystem: config (~/.prism/), session history,              │
     │              project memory (.prism.md)                        │
     └───────────────────────────────────────────────────────────────┘
```

## Layer Descriptions

### Layer 1: CLI Interface (`src/prism/cli/`)
**Responsibility**: User interaction — parsing commands, rendering output, managing the REPL loop.

**Components**:
- `app.py` — Typer application with command registration. Entry point for all CLI commands.
- `repl.py` — Interactive REPL using Prompt Toolkit. Handles multi-line input, history, autocomplete, slash commands (`/add`, `/drop`, `/cost`, `/model`, `/undo`, `/compact`, `/web`).
- `commands/` — Individual command implementations (auth, init, ask, edit, run, config).
- `ui/` — Rich-based rendering: syntax-highlighted diffs, cost tables, Markdown output, progress spinners.

**Dependencies**: Router, Context Manager, Tools, Config.
**Does NOT depend on**: Providers (accesses them only through Router).

### Layer 2: Task Classifier (`src/prism/router/classifier.py`)
**Responsibility**: Analyze each user prompt and classify its complexity tier (Simple, Medium, Complex).

**Feature Extraction**:
- Token count estimation (prompt length)
- File reference count (from active context)
- Output token estimation (based on task type)
- Keyword scoring (complexity-associated keywords)
- Reasoning detection (does task require multi-step reasoning?)
- Scope assessment (single file vs. multi-file vs. architecture-level)

**Classification Logic**:
- Phase 1: Rule-based scoring using weighted feature vector
- Phase 2+: Logistic regression trained on local outcome data (after 100+ interactions)
- Configurable thresholds: SIMPLE < 0.3, MEDIUM < 0.7, COMPLEX ≥ 0.7

**Dependencies**: Context Manager (for file/context info).
**Does NOT depend on**: Providers, Tools.

### Layer 3: Model Router (`src/prism/router/`)
**Responsibility**: Select the optimal model for each classified task, manage fallbacks, enforce budgets.

**Components**:
- `selector.py` — Core routing logic: get candidates for tier → estimate costs → sort by quality/cost ratio → select best.
- `cost_estimator.py` — Token estimation and cost calculation per model.
- `budget.py` — Daily/monthly spending limits enforcement.
- `fallback.py` — Fallback chain management: primary → same-tier alt → cheaper tier → Ollama.
- `learning.py` — Adaptive learning: outcome tracking, feature extraction, model retraining.
- `rate_limiter.py` — Per-provider rate limit tracking and backoff.

**Dependencies**: Providers (for availability/pricing), DB (for history/learning data), Cost module.
**Does NOT depend on**: CLI, Tools.

### Layer 4: Provider Registry (`src/prism/providers/`)
**Responsibility**: Manage provider configurations, model metadata, API key availability, health checks.

**Components**:
- `registry.py` — Central registry of all configured providers and their models.
- `base.py` — Abstract `Provider` class defining the interface.
- Per-provider modules — Provider-specific configuration, model lists, pricing, rate limits.
- `custom.py` — Support for any OpenAI-compatible endpoint.

**All API calls go through LiteLLM** — providers do NOT make direct HTTP calls. They configure LiteLLM parameters.

**Dependencies**: Auth (for API keys), Config (for user preferences).
**Does NOT depend on**: Router, CLI, Tools.

### Layer 5: Tool Execution (`src/prism/tools/`)
**Responsibility**: Execute agentic actions (file ops, terminal, web browsing) with security controls.

**Components**:
- `base.py` — Abstract `Tool` class: `name`, `description`, `parameters_schema`, `execute()`, `permission_level`.
- `registry.py` — Tool registration and discovery. Models see tool schemas, router dispatches tool calls.
- File tools — `file_read.py`, `file_write.py`, `file_edit.py`, `directory.py`, `search.py`.
- `terminal.py` — Sandboxed command execution.
- `browser.py` — Playwright-based web browsing.
- `screenshot.py` — Page/element screenshots for multimodal models.
- `permissions.py` — Permission enforcement per tool and operation type.

**Dependencies**: Security (for sandboxing, path guards), Git (for auto-commit), Config.
**Does NOT depend on**: Router, Providers (tools are model-agnostic).

### Layer 6: Context Manager (`src/prism/context/`)
**Responsibility**: Manage what the model sees — repo structure, conversation history, active files.

**Components**:
- `manager.py` — Orchestrates context assembly for each model call.
- `repo_map.py` — tree-sitter AST parsing → compressed codebase view (classes, functions, signatures).
- `summarizer.py` — Rolling summarization for long conversations (`/compact`).
- `session.py` — Session persistence to `~/.prism/sessions/` and resume.
- `memory.py` — Project memory from `.prism.md` file.
- `rag.py` — Optional ChromaDB vector search for large codebases.

**Dependencies**: DB (for session data), Config.
**Does NOT depend on**: Router, Providers, Tools.

### Layer 7: Auth (`src/prism/auth/`)
**Responsibility**: Secure storage and retrieval of API keys.

**Components**:
- `keyring_store.py` — OS keyring integration (preferred).
- `env_store.py` — Environment variable detection.
- `encrypted_store.py` — AES-256-GCM encrypted file fallback.
- `validator.py` — Validates keys with minimal API calls.

**Dependencies**: Config.
**Does NOT depend on**: Anything else.

### Layer 8: Persistence (`src/prism/db/`)
**Responsibility**: SQLite database for all persistent structured data.

**Components**:
- `database.py` — Connection management, WAL mode, thread safety.
- `models.py` — Schema definitions (see DATA_MODELS.md).
- `migrations.py` — Forward-only schema migrations.
- `queries.py` — Query functions (no raw SQL outside this module).

**Dependencies**: Config (for DB path).
**Does NOT depend on**: Anything else.

### Layer 9: Security (`src/prism/security/`)
**Responsibility**: Cross-cutting security controls.

**Components**:
- `sandbox.py` — Subprocess execution sandbox.
- `path_guard.py` — Path resolution and traversal prevention.
- `secret_filter.py` — Filters sensitive vars from environments.
- `audit.py` — Audit log writer.

**Dependencies**: Config (for project root, log paths).
**Does NOT depend on**: Anything else.

## Dependency Flow (acyclic)

```
CLI → Router → Providers → Auth → Config
 │      │         │                  ↑
 │      │         └──────────────────┘
 │      │
 │      ├→ Cost → DB → Config
 │      │
 │      └→ Context → DB → Config
 │
 ├→ Tools → Security → Config
 │    │
 │    └→ Git → Config
 │
 └→ Config (leaf dependency)
```

**Rule**: Dependencies flow downward and rightward only. No circular dependencies. Config is the leaf — everything can depend on it, it depends on nothing.

## Data Flow for a Typical Request

```
1. User types: "Refactor the auth module to use async"
2. CLI (repl.py) captures input, passes to orchestrator
3. Context Manager assembles: repo map + active files + conversation history + project memory
4. Task Classifier extracts features → scores 0.55 → MEDIUM tier
5. Router gets MEDIUM candidates: [DeepSeek-V3, GPT-4o-mini, Groq-Llama, Gemini-Flash]
6. Cost Estimator: est. 500 input tokens, 2000 output tokens → costs per model
7. Budget check: $3.42 remaining of $5/day budget → all candidates OK
8. Quality tracker: DeepSeek-V3 has 87% success on medium tasks → best quality/cost
9. Router selects DeepSeek-V3, fallback: [GPT-4o-mini, Groq-Llama, Ollama-qwen]
10. LiteLLM completion() call with assembled prompt + tool schemas
11. Model responds with tool calls: edit_file(path="src/auth.py", ...)
12. Tool executor: validates path, shows diff, gets user confirmation
13. File written, git auto-commit created
14. Cost tracker logs: model=deepseek-v3, tokens_in=487, tokens_out=1834, cost=$0.0023
15. Outcome tracker waits: did user accept? Re-run? Edit manually?
16. Next prompt...
```

## Key Design Principles

1. **Local-first**: Everything runs on the user's machine. No cloud services required except AI APIs.
2. **Zero vendor lock-in**: Any provider can be added/removed. No Prism-specific API dependencies.
3. **Security by default**: Restrictive permissions, explicit opt-in for dangerous operations.
4. **Cost-aware**: Every API call has a cost estimate before execution and actual cost tracked after.
5. **Fail gracefully**: Network errors, rate limits, API outages handled with automatic fallback.
6. **Modular**: Each component has a clean interface. Swap implementations without changing callers.
7. **Observable**: Audit logs, cost dashboard, routing explanations — users see what Prism does and why.

# CONTEXT_MANAGEMENT.md — Prism Context Window Management

## Overview

Context management determines what the model sees for each request. The goal is to maximize relevant information while staying within token limits and minimizing cost.

## Context Assembly Pipeline

```
User prompt
    ↓
┌─────────────────────────────────┐
│ 1. System Prompt + Tool Schemas │  ~2,000 tokens (cached)
├─────────────────────────────────┤
│ 2. Repository Map               │  ~1,000-5,000 tokens (cached)
├─────────────────────────────────┤
│ 3. Project Memory (.prism.md)   │  ~500-1,000 tokens (cached)
├─────────────────────────────────┤
│ 4. Active File Contents         │  ~2,000-10,000 tokens (variable)
├─────────────────────────────────┤
│ 5. Conversation History         │  ~remaining budget
├─────────────────────────────────┤
│ 6. User Prompt                  │  actual input
├─────────────────────────────────┤
│ 7. Reserved for Output          │  ~4,000 tokens
└─────────────────────────────────┘
```

## Token Budget Allocation

### Per Model Context Windows
| Model | Context Window | Usable (minus output reserve) |
|-------|---------------|-------------------------------|
| Claude Sonnet/Opus | 200K | 196K |
| GPT-4o | 128K | 112K (16K output) |
| Gemini 2.0 Flash | 1M | 992K |
| Gemini 2.5 Pro | 1M | 992K |
| DeepSeek V3 | 64K | 56K |
| Groq Llama 3.3 70B | 128K | 120K |
| Ollama (7B models) | 32K | 28K |

### Budget Allocation Strategy
```python
def allocate_context_budget(model_context_window: int, max_output: int) -> ContextBudget:
    usable = model_context_window - max_output

    return ContextBudget(
        system_prompt=2000,                    # Fixed
        tool_schemas=1000,                     # Fixed
        repo_map=min(5000, usable * 0.05),     # 5% of usable, max 5K
        project_memory=min(1000, usable * 0.02), # 2% of usable, max 1K
        active_files=min(usable * 0.30, 30000),  # 30% of usable, max 30K
        conversation=usable * 0.50,              # 50% of usable
        user_prompt=usable * 0.10,               # 10% of usable
    )
```

## Repository Map (tree-sitter)

### What It Does
Parses every file in the project using tree-sitter AST parsing and generates a compressed view:

```
src/prism/router/classifier.py:
  class TaskClassifier:
    def __init__(self, config: Settings) -> None
    def classify(self, prompt: str, context: TaskContext) -> ComplexityTier
    def extract_features(self, prompt: str, context: TaskContext) -> dict
    def get_score(self, prompt: str, context: TaskContext) -> float
    def _keyword_score(self, prompt: str) -> float
    def _detect_reasoning_need(self, prompt: str, context: TaskContext) -> bool

src/prism/router/selector.py:
  class ModelSelector:
    def __init__(self, config, registry, estimator, db) -> None
    async def select(self, tier, context, budget) -> ModelSelection
    def _rank_candidates(self, candidates: list) -> list
    def _build_fallback_chain(self, ranked: list) -> list[str]
```

### Generation Process
1. Walk project directory (respect .gitignore)
2. For each supported file: parse with tree-sitter
3. Extract: classes, functions, method signatures (not bodies)
4. Sort by relevance (recently modified files first)
5. Truncate to token budget
6. Cache result keyed by (file paths + mtimes)

### Supported Languages
tree-sitter grammars for: Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, Ruby, PHP, Swift, Kotlin, Scala, C#, and more.

For unsupported file types: fall back to showing just file paths.

### Cache Invalidation
```python
def _should_regenerate(self) -> bool:
    """Check if any files changed since last generation."""
    current_state = self._get_file_states()  # {path: mtime}
    return current_state != self._cached_state
```

## Active Files

### Adding Files to Context
```
/add src/auth.py src/routes/login.ts    # Add specific files
/add src/middleware/                      # Add all files in directory
/drop src/auth.py                        # Remove from context
```

### Automatic File Discovery
When a model references a file in its response (tool call to read_file), that file is automatically added to active context for subsequent turns.

### File Truncation
If a file exceeds its token budget share:
1. Show first 100 lines + last 50 lines
2. Insert `... [X lines omitted] ...` in the middle
3. Full file still available via read_file tool

## Conversation History Management

### Rolling Window
Keep full conversation for as long as it fits. When approaching limit:

```
Turns:     [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]
           ↓                                          ↓
Full:      [Summary of turns 1-7] [8] [9] [10]
```

### Summarization (`/compact`)
When triggered (manually or automatically at 80% context):

1. Take all conversation turns except last 3
2. Send to cheapest available model with prompt:
   ```
   Summarize this conversation. Preserve:
   1. What the user is working on
   2. What changes have been made (file names, line numbers)
   3. Decisions made and rationale
   4. Outstanding issues or next steps
   Be concise — this replaces the full conversation.
   ```
3. Replace old turns with summary
4. Notify user: "Context compressed. Summary saved."

### Auto-Compact Trigger
```python
def should_compact(self, total_tokens: int, context_window: int) -> bool:
    usage_ratio = total_tokens / context_window
    return usage_ratio > 0.80  # 80% threshold
```

## Project Memory (`.prism.md`)

### Location
Project root / `.prism.md`

### Created By
`prism init` or automatically on first session in a project.

### Contents (user-editable)
```markdown
# Project: MyApp

## Stack
Python 3.12, FastAPI, PostgreSQL, Redis

## Conventions
- snake_case for functions, PascalCase for classes
- pytest for testing, ruff for linting
- Always handle errors with try/except, never bare except

## Architecture
- src/api/ — FastAPI route handlers
- src/models/ — SQLAlchemy models
- src/services/ — Business logic layer
- src/tasks/ — Celery background tasks

## Key Decisions
- JWT tokens in httpOnly cookies (not localStorage)
- Database migrations via Alembic
- Redis for session cache and rate limiting

## Notes for AI
- Always run `pytest` after making changes
- Don't modify alembic migration files directly
- The CI pipeline is GitHub Actions (.github/workflows/)
```

### Loading
Read at session start, included in every prompt. Cached — only re-read on file change.

## RAG (ChromaDB — Optional)

### When to Use RAG
- Project has > 500 files
- Context window insufficient for full repo map
- User enables with `prism config set rag.enabled true`

### Implementation
```python
class RAGManager:
    def __init__(self, project_root: Path, db_path: Path):
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection("codebase")

    def index_project(self) -> None:
        """Index all project files into ChromaDB."""
        for file in walk_project(self.project_root):
            content = file.read_text()
            chunks = self._chunk_file(content, chunk_size=500)
            for i, chunk in enumerate(chunks):
                self.collection.add(
                    documents=[chunk],
                    metadatas=[{"file": str(file), "chunk": i}],
                    ids=[f"{file}:{i}"],
                )

    def query(self, prompt: str, n_results: int = 5) -> list[CodeChunk]:
        """Find most relevant code chunks for a prompt."""
        results = self.collection.query(
            query_texts=[prompt],
            n_results=n_results,
        )
        return [
            CodeChunk(file=m["file"], content=d)
            for m, d in zip(results["metadatas"][0], results["documents"][0])
        ]
```

### ChromaDB Storage
- Location: `~/.prism/cache/chromadb/<project_hash>/`
- Re-indexed when file mtimes change
- Optional — core Prism works without it

## Context-Aware Routing

### Model Selection by Context Size
When the assembled context is very large:
- If total context > 100K tokens → prefer Gemini (1M window)
- If total context > 50K tokens → avoid Ollama (32K window)
- If total context < 10K tokens → any model works, prefer cheapest

```python
def adjust_for_context_size(
    candidates: list[ModelCandidate],
    estimated_context_tokens: int,
) -> list[ModelCandidate]:
    """Filter out models whose context window can't handle the input."""
    return [
        c for c in candidates
        if c.model_info.context_window > estimated_context_tokens * 1.3  # 30% safety margin
    ]
```

## Session Persistence

### Session Files
Location: `~/.prism/sessions/<session_id>.md`

Format:
```markdown
# Session: 2026-03-10 14:23:01
# Project: /Users/dev/myapp
# Cost: $0.42 (23 requests)

## Turn 1
**User**: Fix the authentication bug in src/auth.py
**Model**: claude-sonnet-4 | Tier: complex | Cost: $0.05
**Response**: [assistant response]

## Turn 2
**User**: /add src/routes/login.ts
**System**: Added src/routes/login.ts to context

## Turn 3
...
```

### Session Resume
```bash
prism                    # Resumes last session for current project
prism --new-session      # Force new session
prism --session <id>     # Resume specific session
```

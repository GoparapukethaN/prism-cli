# PERFORMANCE.md — Prism Performance Requirements and Optimization

## Performance Targets

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| CLI startup | < 500ms | Import optimization, lazy loading |
| Task classification | < 5ms | Pure computation, no I/O |
| Model selection (routing) | < 10ms | DB lookup + computation |
| Cost estimation | < 2ms | In-memory pricing lookup |
| Repo map generation (first) | < 10s | tree-sitter parsing, cached after |
| Repo map generation (cached) | < 100ms | Mtime-based cache invalidation |
| File read (< 1MB) | < 50ms | Direct filesystem read |
| File search (ripgrep) | < 2s | For projects up to 100K files |
| Session save | < 100ms | Async, non-blocking |
| SQLite query (routing history) | < 5ms | Indexed queries |
| `/compact` summarization | < 5s | Uses cheapest available model |
| REPL input latency | < 50ms | Prompt Toolkit responsiveness |

## Optimization Strategies

### 1. Lazy Loading
The CLI must start fast. Import expensive libraries only when needed:

```python
# DO NOT import at module level:
# import litellm          # Heavy — imports 50+ sub-modules
# import playwright       # Very heavy — browser engine
# import chromadb         # Heavy — ML dependencies
# import tree_sitter      # Moderate

# DO import lazily:
def get_litellm():
    import litellm
    return litellm

def get_playwright():
    from playwright.sync_api import sync_playwright
    return sync_playwright
```

**Import budget**: Core CLI must import only `typer`, `rich`, `prompt_toolkit`, `sqlite3`, `pathlib`, and `prism` internal modules at startup. Everything else lazy.

### 2. Caching Strategy

#### tree-sitter Repo Map Cache
- Cache key: sorted list of (filepath, mtime) tuples → SHA-256 hash
- Cache location: `~/.prism/cache/repo_maps/<project_hash>.json`
- Invalidation: any file mtime change → regenerate
- Partial invalidation: only re-parse changed files, merge with cached

#### Prompt Cache (Anthropic)
- System prompt + tool schemas → marked with `cache_control: ephemeral`
- Repo map → marked with `cache_control: ephemeral`
- Saves ~90% of system prompt tokens on subsequent turns

#### Provider Availability Cache
- Cache health check results for 60 seconds
- Rate limit state cached per-provider with expiry timestamps
- Ollama model list cached for 300 seconds

#### Pricing Cache
- Model pricing loaded at startup from `src/prism/cost/pricing.py`
- Updated monthly with new releases
- User can override via config

### 3. Database Optimization

#### SQLite Configuration
```sql
PRAGMA journal_mode = WAL;           -- Write-Ahead Logging for concurrent reads
PRAGMA synchronous = NORMAL;         -- Faster writes, safe with WAL
PRAGMA cache_size = -64000;          -- 64MB page cache
PRAGMA foreign_keys = ON;            -- Referential integrity
PRAGMA temp_store = MEMORY;          -- Temp tables in memory
PRAGMA mmap_size = 268435456;        -- 256MB memory-mapped I/O
```

#### Indexes
```sql
-- Routing decisions: lookup by date for dashboard
CREATE INDEX idx_routing_decisions_created_at ON routing_decisions(created_at);

-- Routing decisions: lookup by model for learning
CREATE INDEX idx_routing_decisions_model ON routing_decisions(model_id);

-- Cost entries: lookup by date for budget checks
CREATE INDEX idx_cost_entries_created_at ON cost_entries(created_at);
CREATE INDEX idx_cost_entries_date ON cost_entries(date(created_at));

-- Learning data: lookup by task features
CREATE INDEX idx_learning_data_tier ON learning_data(complexity_tier);
```

#### Batch Operations
- Cost tracking: batch insert every 10 entries or 5 seconds (whichever first)
- Routing history: write-behind with async flush
- Vacuum: monthly automatic maintenance

### 4. Streaming

#### Response Streaming
All model responses stream token-by-token to the terminal:
```python
async for chunk in response:
    if chunk.choices[0].delta.content:
        console.print(chunk.choices[0].delta.content, end="")
```

**Benefits**:
- User sees output immediately (perceived latency = time-to-first-token)
- Can cancel early if output is wrong (saves tokens/money)
- Running cost display updates during generation

#### Cost Streaming
Display running cost estimate during generation:
```
Generating... $0.0012 (deepseek-v3, 847 tokens)
```
Updated every 500ms based on streamed token count.

### 5. Async I/O

#### Concurrent Operations
```python
# Read multiple files in parallel
files = await asyncio.gather(
    read_file("src/auth.py"),
    read_file("src/models.py"),
    read_file("src/routes.py"),
)

# Health check all providers in parallel
statuses = await asyncio.gather(
    check_anthropic(),
    check_openai(),
    check_google(),
    check_ollama(),
    return_exceptions=True,
)
```

#### Non-blocking Operations
- Session saving: async write, don't block REPL
- Audit logging: async append, don't block tool execution
- Cost tracking: async DB write, don't block response display

### 6. Memory Management

#### Context Window Budgeting
- Track token usage per conversation turn
- Trigger `/compact` warning at 60% capacity
- Auto-compact at 80% capacity
- Never exceed 90% — reserve 10% for output

#### Large File Handling
- Files > 100KB: show first/last 50 lines + summary
- Binary files: skip with warning
- Repo map: limit to 5,000 entries (configurable)
- Search results: limit to 50 matches by default

#### Session Memory
- Session files: cap at 1MB per session
- Auto-archive sessions older than 30 days
- In-memory conversation: only keep last N turns + summary of older turns

### 7. Startup Optimization

**Target: < 500ms from command to REPL prompt.**

Startup sequence (measured):
```
1. Python interpreter startup: ~100ms (unavoidable)
2. Typer + Rich import: ~80ms
3. Config loading: ~20ms (single YAML file)
4. SQLite connection: ~10ms
5. Provider status check: ASYNC (non-blocking, happens in background)
6. Repo map: LAZY (generated on first model call, not startup)
7. Prompt Toolkit init: ~50ms
8. REPL ready: ~260ms total
```

**Key**: Provider health checks and repo map generation happen AFTER the REPL is responsive, not during startup.

## Benchmarking

### Micro-benchmarks (run with `pytest --benchmark`)
```python
def test_benchmark_classification(benchmark):
    classifier = TaskClassifier(config=default_config())
    context = TaskContext(active_files=["main.py"])

    benchmark(classifier.classify, "fix the typo in main.py", context)

def test_benchmark_cost_estimation(benchmark):
    estimator = CostEstimator(pricing=default_pricing())

    benchmark(estimator.estimate, "deepseek-v3", input_tokens=500, output_tokens=2000)

def test_benchmark_routing_decision(benchmark, populated_db):
    router = Router(db=populated_db, config=default_config())
    context = TaskContext(active_files=["main.py"])

    benchmark(router.select_model, ComplexityTier.MEDIUM, context, budget_remaining=5.0)
```

### Load Testing
- Simulate 1000 routing decisions: verify < 10s total
- Simulate 10,000 cost entries: verify dashboard query < 100ms
- Simulate 100-file project: verify repo map < 5s
- Simulate 1000-file project: verify repo map < 30s

## Monitoring

### Runtime Metrics (logged at INFO level)
- `routing_latency_ms`: Time from prompt to model selection
- `api_latency_ms`: Time from LiteLLM call to first token
- `tool_execution_ms`: Time per tool call
- `total_turn_ms`: Total time from user input to complete response
- `token_count_estimated` vs `token_count_actual`: Estimation accuracy
- `cache_hit_rate`: Repo map cache, prompt cache, provider cache

### Health Indicators
- If `routing_latency_ms` > 50ms: investigate DB performance
- If `api_latency_ms` > 30s: likely provider issue, fallback should trigger
- If `cache_hit_rate` < 50%: review cache invalidation strategy
- If token estimation error > 30%: recalibrate estimator

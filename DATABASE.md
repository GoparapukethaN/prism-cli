# DATABASE.md — Prism Database Design

## Technology
- **SQLite** via Python stdlib `sqlite3`
- No ORM — direct SQL with parameterized queries
- WAL mode for concurrent read/write
- All queries in `src/prism/db/queries.py` — no raw SQL elsewhere

## Database Location
- Default: `~/.prism/prism.db`
- Configurable via `PRISM_HOME` env var or `prism_home` setting
- Test: in-memory (`:memory:`) or tmp_path

## Connection Management

```python
class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._local = threading.local()  # Thread-local connections

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.path),
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._configure(self._local.conn)
        return self._local.conn

    def _configure(self, conn: sqlite3.Connection) -> None:
        """Apply performance and safety pragmas."""
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -64000")  # 64MB
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 268435456")  # 256MB

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
```

## Query Functions

All database access goes through named functions in `queries.py`:

### Routing Decisions
```python
def save_routing_decision(db: Database, decision: RoutingDecision) -> None:
    """Insert a routing decision record."""

def update_routing_outcome(db: Database, decision_id: str, outcome: Outcome, actual_cost: float | None) -> None:
    """Update outcome and actual cost after completion."""

def get_routing_history(db: Database, limit: int = 100, session_id: str | None = None) -> list[RoutingDecision]:
    """Get recent routing decisions, optionally filtered by session."""

def get_model_success_rate(db: Database, model_id: str, tier: ComplexityTier, min_entries: int = 10) -> float:
    """Calculate historical success rate for a model on a tier. Returns 0.5 if insufficient data."""

def get_learning_data(db: Database, min_entries: int = 100) -> list[RoutingDecision]:
    """Get all routing decisions with known outcomes for adaptive learning."""
```

### Cost Entries
```python
def save_cost_entry(db: Database, entry: CostEntry) -> None:
    """Insert a cost tracking record."""

def get_session_cost(db: Database, session_id: str) -> float:
    """Total cost for the current session."""

def get_daily_cost(db: Database, date: str | None = None) -> float:
    """Total cost for a given date (default: today)."""

def get_monthly_cost(db: Database, year: int | None = None, month: int | None = None) -> float:
    """Total cost for a given month (default: current month)."""

def get_cost_breakdown(db: Database, period: str) -> list[ModelCostBreakdown]:
    """Cost breakdown by model for a period ('session', 'day', 'month')."""

def get_budget_remaining(db: Database, daily_limit: float | None, monthly_limit: float | None) -> float | None:
    """Calculate remaining budget. Returns None if no limits set."""
```

### Sessions
```python
def create_session(db: Database, session_id: str, project_root: str) -> None:
    """Create a new session record."""

def update_session(db: Database, session_id: str, cost_delta: float, request_delta: int) -> None:
    """Increment session cost and request count."""

def save_session_summary(db: Database, session_id: str, summary: str) -> None:
    """Save compressed conversation summary."""

def get_active_session(db: Database, project_root: str) -> str | None:
    """Get the most recent active session for a project."""
```

### Provider Status
```python
def update_provider_status(db: Database, provider: str, available: bool, error: str | None = None) -> None:
    """Update provider availability status."""

def set_rate_limited(db: Database, provider: str, until: datetime) -> None:
    """Mark a provider as rate-limited until a specific time."""

def is_rate_limited(db: Database, provider: str) -> bool:
    """Check if a provider is currently rate-limited."""

def increment_free_tier_usage(db: Database, provider: str) -> int:
    """Increment and return the free tier request count for today."""

def get_free_tier_remaining(db: Database, provider: str, daily_limit: int) -> int:
    """Get remaining free tier requests for today."""
```

## SQL Safety Rules

1. **Always use parameterized queries** — never string interpolation
   ```python
   # CORRECT
   cursor.execute("SELECT * FROM cost_entries WHERE model_id = ?", (model_id,))

   # NEVER
   cursor.execute(f"SELECT * FROM cost_entries WHERE model_id = '{model_id}'")
   ```

2. **Always use transactions for multi-statement operations**
   ```python
   with db.transaction():
       save_routing_decision(db, decision)
       save_cost_entry(db, cost_entry)
       update_session(db, session_id, cost, 1)
   ```

3. **Always handle IntegrityError for duplicate inserts**
4. **Always use LIMIT on queries that could return large result sets**
5. **Never store raw prompts** — only SHA-256 hashes
6. **Never store API keys** in the database

## Database Maintenance

### Automatic
- WAL checkpoint: every 1000 transactions or on session end
- Free tier counter reset: daily at midnight UTC

### Manual (via `prism db` commands)
```bash
prism db vacuum     # Reclaim unused space
prism db stats      # Show table sizes, row counts
prism db export     # Export all data to JSON
prism db reset      # Drop and recreate all tables (with confirmation)
prism db backup     # Copy database to ~/.prism/backups/
```

## Data Retention

| Data | Retention | Reason |
|------|-----------|--------|
| Routing decisions | 90 days | Learning data, cost reporting |
| Cost entries | 1 year | Monthly reporting |
| Sessions | 30 days (metadata) | Session resume |
| Provider status | Current only | Real-time tracking |
| Tool executions | 30 days | Audit trail |

Cleanup runs automatically on session start:
```sql
DELETE FROM routing_decisions WHERE created_at < datetime('now', '-90 days');
DELETE FROM tool_executions WHERE created_at < datetime('now', '-30 days');
```

## Testing

- All DB tests use `tmp_path` SQLite files or in-memory databases
- Each test gets a fresh database with `initialize()` called
- No shared state between tests
- Test fixtures provide pre-populated databases for query testing
- Schema tests verify all migrations apply cleanly
- Performance tests verify query latency on tables with 10K+ rows

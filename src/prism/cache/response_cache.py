"""Response caching — SQLite-backed LLM response cache with TTL and file-change invalidation.

Wraps all LiteLLM calls with a transparent cache layer.  Each response is
keyed on ``(model + system_prompt + user_prompt + files_context)`` via a SHA-256
hash.  Entries are persisted in ``~/.prism/cache/response_cache.db`` using a
thread-safe SQLite wrapper.

Features
--------
* Configurable TTL per model tier (simple / medium / complex).
* Automatic invalidation when referenced source files change (mtime tracking).
* In-memory hit / miss counters flushed periodically to the DB for persistence.
* Cache bypass (``enabled`` toggle) and selective clearing by age.
* Never caches tool execution results — only pure LLM completions.

Usage::

    cache = ResponseCache(Path("~/.prism/cache"), ttl=3600, enabled=True)

    key = ResponseCache.make_cache_key(model, system, user, files_ctx)
    hit = cache.get(key, file_paths=[Path("src/main.py")])
    if hit is None:
        # call LiteLLM, then store
        cache.put(key, model=..., provider=..., content=..., ...)

    cache.close()
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# TTL defaults per model tier (seconds)
# ---------------------------------------------------------------------------

DEFAULT_TTL: int = 3600  # 1 hour

TIER_TTLS: dict[str, int] = {
    "simple": 7200,   # 2 hours — simple tasks change less often
    "medium": 3600,   # 1 hour  — standard
    "complex": 1800,  # 30 min  — complex tasks need fresher results
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheEntry:
    """A cached LLM response.

    Attributes:
        cache_key: SHA-256 hex digest identifying this request.
        model: LiteLLM model identifier used for the completion.
        provider: Provider name (e.g. ``anthropic``, ``openai``).
        content: The text content of the completion response.
        input_tokens: Number of input tokens in the original request.
        output_tokens: Number of output tokens in the response.
        cached_tokens: Tokens served from provider-side caching.
        cost_usd: Cost in USD of the original API call.
        finish_reason: Completion finish reason (``stop``, ``length``, etc.).
        tool_calls_json: JSON-serialized list of tool calls, or ``None``.
        created_at: ISO 8601 timestamp when the entry was cached.
        expires_at: ISO 8601 timestamp when the entry expires.
        file_hashes: JSON-serialized ``{path: mtime}`` dict for invalidation.
    """

    cache_key: str
    model: str
    provider: str
    content: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    finish_reason: str
    tool_calls_json: str | None
    created_at: str
    expires_at: str
    file_hashes: str


@dataclass
class CacheStats:
    """Cache statistics for the ``/cache stats`` command.

    Attributes:
        total_entries: Number of entries currently in the cache.
        total_hits: Lifetime cache hits (persisted + in-memory).
        total_misses: Lifetime cache misses.
        hit_rate: ``total_hits / (total_hits + total_misses)`` or 0.
        tokens_saved: Total input + output tokens avoided via cache hits.
        cost_saved: Total USD saved via cache hits.
        cache_size_bytes: Approximate SQLite database size in bytes.
        oldest_entry: ISO 8601 of the oldest cached entry, or ``None``.
        newest_entry: ISO 8601 of the newest cached entry, or ``None``.
    """

    total_entries: int
    total_hits: int
    total_misses: int
    hit_rate: float
    tokens_saved: int
    cost_saved: float
    cache_size_bytes: int
    oldest_entry: str | None
    newest_entry: str | None


# ---------------------------------------------------------------------------
# Main cache class
# ---------------------------------------------------------------------------


class ResponseCache:
    """SQLite-backed LLM response cache with TTL and file-change invalidation.

    Caches completion results keyed on
    ``(model + system_prompt + user_prompt + files_context)``.
    Automatically invalidates when referenced files change (mtime tracking).
    Configurable TTL per model tier.  Never caches tool execution results.

    Thread-safe: each thread gets its own ``sqlite3.Connection`` via
    ``threading.local()``.

    Args:
        cache_dir: Directory to store the SQLite database.
        ttl: Default TTL in seconds.  Overridden by tier-specific TTLs.
        enabled: Whether caching is active.  Can be toggled at runtime.
    """

    def __init__(
        self,
        cache_dir: Path,
        ttl: int = DEFAULT_TTL,
        enabled: bool = True,
    ) -> None:
        self._cache_dir = cache_dir
        self._default_ttl = ttl
        self._enabled = enabled
        self._db_path = cache_dir / "response_cache.db"
        self._local = threading.local()
        self._lock = threading.Lock()

        # In-memory counters (flushed to DB periodically via flush_stats)
        self._stats_hits: int = 0
        self._stats_misses: int = 0
        self._stats_tokens_saved: int = 0
        self._stats_cost_saved: float = 0.0

        cache_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether cache look-ups and stores are active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def db_path(self) -> Path:
        """Path to the underlying SQLite database."""
        return self._db_path

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return the thread-local connection, creating one if needed."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _initialize_db(self) -> None:
        """Create tables and indices if they don't exist."""
        conn = self._get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS response_cache (
                cache_key       TEXT PRIMARY KEY,
                model           TEXT NOT NULL,
                provider        TEXT NOT NULL,
                content         TEXT NOT NULL,
                input_tokens    INTEGER NOT NULL,
                output_tokens   INTEGER NOT NULL,
                cached_tokens   INTEGER DEFAULT 0,
                cost_usd        REAL NOT NULL,
                finish_reason   TEXT NOT NULL,
                tool_calls_json TEXT,
                created_at      TEXT NOT NULL,
                expires_at      TEXT NOT NULL,
                file_hashes     TEXT DEFAULT '{}',
                hit_count       INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_cache_expires
                ON response_cache(expires_at);
            CREATE INDEX IF NOT EXISTS idx_cache_model
                ON response_cache(model);

            CREATE TABLE IF NOT EXISTS cache_stats (
                id           INTEGER PRIMARY KEY CHECK (id = 1),
                total_hits   INTEGER DEFAULT 0,
                total_misses INTEGER DEFAULT 0,
                tokens_saved INTEGER DEFAULT 0,
                cost_saved   REAL    DEFAULT 0.0
            );

            INSERT OR IGNORE INTO cache_stats
                (id, total_hits, total_misses, tokens_saved, cost_saved)
            VALUES
                (1, 0, 0, 0, 0.0);
            """
        )

    # ------------------------------------------------------------------
    # Cache key generation
    # ------------------------------------------------------------------

    @staticmethod
    def make_cache_key(
        model: str,
        system_prompt: str,
        user_prompt: str,
        files_context: str = "",
    ) -> str:
        """Generate a deterministic cache key from request parameters.

        The key is a SHA-256 hex digest of a canonicalised JSON payload
        containing the four inputs.  Identical inputs always produce the
        same key regardless of dict ordering.

        Args:
            model: LiteLLM model identifier.
            system_prompt: System-level prompt text.
            user_prompt: User message text.
            files_context: Concatenated file contents injected into context.

        Returns:
            64-character hex string.
        """
        raw = json.dumps(
            {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "files_context": files_context,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # File hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_file_hashes(file_paths: list[str | Path]) -> dict[str, float]:
        """Compute mtime hashes for a list of files for invalidation tracking.

        Args:
            file_paths: List of file paths (strings or ``Path`` objects).

        Returns:
            Dict mapping resolved absolute path strings to their ``st_mtime``
            values.  Files that don't exist or can't be stat'd are skipped.
        """
        hashes: dict[str, float] = {}
        for fp in file_paths:
            p = Path(fp)
            if p.is_file():
                with contextlib.suppress(OSError):
                    hashes[str(p.resolve())] = p.stat().st_mtime
        return hashes

    # ------------------------------------------------------------------
    # Get (look-up)
    # ------------------------------------------------------------------

    def get(
        self,
        cache_key: str,
        file_paths: list[str | Path] | None = None,
    ) -> CacheEntry | None:
        """Look up a cached response.

        Returns ``None`` on a miss, expired entry, disabled cache, or when
        referenced files have changed since the entry was stored.

        Args:
            cache_key: The SHA-256 key returned by :meth:`make_cache_key`.
            file_paths: Optional list of file paths for invalidation checking.

        Returns:
            A :class:`CacheEntry` on hit, or ``None``.
        """
        if not self._enabled:
            return None

        conn = self._get_connection()
        now = datetime.now(UTC).isoformat()

        row = conn.execute(
            "SELECT * FROM response_cache WHERE cache_key = ? AND expires_at > ?",
            (cache_key, now),
        ).fetchone()

        if row is None:
            self._record_miss()
            return None

        # File-change invalidation
        if file_paths:
            stored_hashes_raw = row["file_hashes"] if row["file_hashes"] else "{}"
            stored_hashes: dict[str, float] = json.loads(stored_hashes_raw)
            current_hashes = self.compute_file_hashes(file_paths)
            if stored_hashes != current_hashes:
                conn.execute(
                    "DELETE FROM response_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                conn.commit()
                self._record_miss()
                logger.info("cache_invalidated_file_change", key=cache_key[:12])
                return None

        # Record hit
        conn.execute(
            "UPDATE response_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (cache_key,),
        )
        conn.commit()

        entry = CacheEntry(
            cache_key=row["cache_key"],
            model=row["model"],
            provider=row["provider"],
            content=row["content"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cached_tokens=row["cached_tokens"],
            cost_usd=row["cost_usd"],
            finish_reason=row["finish_reason"],
            tool_calls_json=row["tool_calls_json"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            file_hashes=row["file_hashes"],
        )

        self._record_hit(entry.input_tokens + entry.output_tokens, entry.cost_usd)
        logger.info("cache_hit", key=cache_key[:12], model=entry.model)
        return entry

    # ------------------------------------------------------------------
    # Put (store)
    # ------------------------------------------------------------------

    def put(
        self,
        cache_key: str,
        model: str,
        provider: str,
        content: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        finish_reason: str,
        cached_tokens: int = 0,
        tool_calls: list[dict[str, object]] | None = None,
        file_paths: list[str | Path] | None = None,
        ttl: int | None = None,
        tier: str | None = None,
    ) -> CacheEntry:
        """Store a response in the cache.

        If an entry with the same key exists it is replaced.

        Args:
            cache_key: The SHA-256 key from :meth:`make_cache_key`.
            model: LiteLLM model identifier.
            provider: Provider name.
            content: Response text content.
            input_tokens: Input token count.
            output_tokens: Output token count.
            cost_usd: Cost in USD.
            finish_reason: Completion finish reason.
            cached_tokens: Tokens from provider-side cache.
            tool_calls: Optional list of tool call dicts.
            file_paths: Files to track for invalidation.
            ttl: Override TTL in seconds.  Falls back to tier default.
            tier: Complexity tier for tier-specific TTL look-up.

        Returns:
            The stored :class:`CacheEntry`.
        """
        effective_ttl = ttl if ttl is not None else TIER_TTLS.get(tier or "", self._default_ttl)
        now = datetime.now(UTC)
        expires = datetime.fromtimestamp(now.timestamp() + effective_ttl, tz=UTC)

        file_hashes = self.compute_file_hashes(file_paths or [])
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None

        entry = CacheEntry(
            cache_key=cache_key,
            model=model,
            provider=provider,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            finish_reason=finish_reason,
            tool_calls_json=tool_calls_json,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            file_hashes=json.dumps(file_hashes),
        )

        conn = self._get_connection()
        conn.execute(
            """INSERT OR REPLACE INTO response_cache
               (cache_key, model, provider, content, input_tokens, output_tokens,
                cached_tokens, cost_usd, finish_reason, tool_calls_json,
                created_at, expires_at, file_hashes, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                entry.cache_key,
                entry.model,
                entry.provider,
                entry.content,
                entry.input_tokens,
                entry.output_tokens,
                entry.cached_tokens,
                entry.cost_usd,
                entry.finish_reason,
                entry.tool_calls_json,
                entry.created_at,
                entry.expires_at,
                entry.file_hashes,
            ),
        )
        conn.commit()
        logger.info("cache_put", key=cache_key[:12], model=model, ttl=effective_ttl)
        return entry

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear(self, max_age_hours: int | None = None) -> int:
        """Clear cache entries.

        Args:
            max_age_hours: If provided, only entries older than this many
                hours are removed.  If ``None``, all entries are cleared
                and aggregate stats are reset.

        Returns:
            Number of entries deleted.
        """
        conn = self._get_connection()
        if max_age_hours is None:
            cursor = conn.execute("DELETE FROM response_cache")
            conn.execute(
                "UPDATE cache_stats "
                "SET total_hits=0, total_misses=0, tokens_saved=0, cost_saved=0.0 "
                "WHERE id=1"
            )
            # Also reset in-memory counters
            with self._lock:
                self._stats_hits = 0
                self._stats_misses = 0
                self._stats_tokens_saved = 0
                self._stats_cost_saved = 0.0
        else:
            cutoff = datetime.fromtimestamp(
                datetime.now(UTC).timestamp() - (max_age_hours * 3600),
                tz=UTC,
            ).isoformat()
            cursor = conn.execute(
                "DELETE FROM response_cache WHERE created_at < ?", (cutoff,)
            )
        conn.commit()
        deleted = cursor.rowcount
        logger.info("cache_cleared", deleted=deleted, max_age_hours=max_age_hours)
        return deleted

    def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries removed.
        """
        conn = self._get_connection()
        now = datetime.now(UTC).isoformat()
        cursor = conn.execute(
            "DELETE FROM response_cache WHERE expires_at <= ?", (now,)
        )
        conn.commit()
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> CacheStats:
        """Get cache statistics combining persisted and in-memory counters.

        Returns:
            A :class:`CacheStats` instance.
        """
        conn = self._get_connection()

        stats_row = conn.execute(
            "SELECT * FROM cache_stats WHERE id = 1"
        ).fetchone()

        db_hits = stats_row["total_hits"] if stats_row else 0
        db_misses = stats_row["total_misses"] if stats_row else 0
        db_tokens = stats_row["tokens_saved"] if stats_row else 0
        db_cost = stats_row["cost_saved"] if stats_row else 0.0

        with self._lock:
            total_hits = db_hits + self._stats_hits
            total_misses = db_misses + self._stats_misses
            tokens_saved = db_tokens + self._stats_tokens_saved
            cost_saved = db_cost + self._stats_cost_saved

        total = total_hits + total_misses
        hit_rate = total_hits / total if total > 0 else 0.0

        count_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM response_cache"
        ).fetchone()
        total_entries = count_row["cnt"] if count_row else 0

        oldest_row = conn.execute(
            "SELECT MIN(created_at) AS oldest FROM response_cache"
        ).fetchone()
        newest_row = conn.execute(
            "SELECT MAX(created_at) AS newest FROM response_cache"
        ).fetchone()

        # Approximate database size from page count
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        cache_size_bytes = page_count * page_size

        return CacheStats(
            total_entries=total_entries,
            total_hits=total_hits,
            total_misses=total_misses,
            hit_rate=hit_rate,
            tokens_saved=tokens_saved,
            cost_saved=cost_saved,
            cache_size_bytes=cache_size_bytes,
            oldest_entry=oldest_row["oldest"] if oldest_row else None,
            newest_entry=newest_row["newest"] if newest_row else None,
        )

    def flush_stats(self) -> None:
        """Flush in-memory stats counters to the database.

        Safe to call multiple times — only writes if there are pending
        counter values.  Called automatically by :meth:`close`.
        """
        with self._lock:
            if (
                self._stats_hits == 0
                and self._stats_misses == 0
                and self._stats_tokens_saved == 0
                and self._stats_cost_saved == 0.0
            ):
                return
            hits = self._stats_hits
            misses = self._stats_misses
            tokens = self._stats_tokens_saved
            cost = self._stats_cost_saved
            self._stats_hits = 0
            self._stats_misses = 0
            self._stats_tokens_saved = 0
            self._stats_cost_saved = 0.0

        conn = self._get_connection()
        conn.execute(
            """UPDATE cache_stats SET
               total_hits   = total_hits   + ?,
               total_misses = total_misses + ?,
               tokens_saved = tokens_saved + ?,
               cost_saved   = cost_saved   + ?
               WHERE id = 1""",
            (hits, misses, tokens, cost),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush stats and close the thread-local database connection."""
        try:
            self.flush_stats()
        except Exception:
            logger.exception("cache_flush_stats_error")
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error as exc:
                logger.warning("cache_close_error", error=str(exc))
            finally:
                self._local.conn = None

    def __enter__(self) -> ResponseCache:
        """Support use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Close on context manager exit."""
        self.close()

    # ------------------------------------------------------------------
    # Internal stat helpers
    # ------------------------------------------------------------------

    def _record_hit(self, tokens: int, cost: float) -> None:
        """Increment in-memory hit counters (thread-safe)."""
        with self._lock:
            self._stats_hits += 1
            self._stats_tokens_saved += tokens
            self._stats_cost_saved += cost

    def _record_miss(self) -> None:
        """Increment in-memory miss counter (thread-safe)."""
        with self._lock:
            self._stats_misses += 1

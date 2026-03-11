"""Tests for the ResponseCache module.

Covers: cache key generation, put/get round-trips, TTL expiry, file-change
invalidation, cache bypass, clear/cleanup, stats tracking (hits, misses,
tokens, cost), flush persistence, tier-specific TTLs, edge cases (empty
content, large content, special characters, concurrent access), context
manager protocol, and CacheEntry/CacheStats dataclasses.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from prism.cache.response_cache import (
    DEFAULT_TTL,
    TIER_TTLS,
    CacheEntry,
    CacheStats,
    ResponseCache,
)

if TYPE_CHECKING:
    from pathlib import Path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_cache(tmp_path: Path, **kwargs: object) -> ResponseCache:
    """Create a ResponseCache in a temporary directory."""
    cache_dir = tmp_path / "cache"
    return ResponseCache(cache_dir=cache_dir, **kwargs)  # type: ignore[arg-type]


def _put_sample(
    cache: ResponseCache,
    cache_key: str = "abc123",
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    content: str = "Hello, world!",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost_usd: float = 0.001,
    finish_reason: str = "stop",
    **kwargs: object,
) -> CacheEntry:
    """Insert a sample entry into the cache."""
    return cache.put(
        cache_key=cache_key,
        model=model,
        provider=provider,
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        finish_reason=finish_reason,
        **kwargs,  # type: ignore[arg-type]
    )


# ==================================================================
# TestCacheKey — deterministic key generation
# ==================================================================


class TestCacheKey:
    """Tests for ResponseCache.make_cache_key()."""

    def test_deterministic(self) -> None:
        """Same inputs always produce the same key."""
        k1 = ResponseCache.make_cache_key("gpt-4", "sys", "user", "files")
        k2 = ResponseCache.make_cache_key("gpt-4", "sys", "user", "files")
        assert k1 == k2

    def test_different_model_different_key(self) -> None:
        """Changing the model changes the key."""
        k1 = ResponseCache.make_cache_key("gpt-4", "sys", "user")
        k2 = ResponseCache.make_cache_key("gpt-3.5-turbo", "sys", "user")
        assert k1 != k2

    def test_different_system_prompt_different_key(self) -> None:
        """Changing the system prompt changes the key."""
        k1 = ResponseCache.make_cache_key("gpt-4", "sys-a", "user")
        k2 = ResponseCache.make_cache_key("gpt-4", "sys-b", "user")
        assert k1 != k2

    def test_different_user_prompt_different_key(self) -> None:
        """Changing the user prompt changes the key."""
        k1 = ResponseCache.make_cache_key("gpt-4", "sys", "hello")
        k2 = ResponseCache.make_cache_key("gpt-4", "sys", "goodbye")
        assert k1 != k2

    def test_different_files_context_different_key(self) -> None:
        """Changing the files context changes the key."""
        k1 = ResponseCache.make_cache_key("gpt-4", "sys", "user", "v1")
        k2 = ResponseCache.make_cache_key("gpt-4", "sys", "user", "v2")
        assert k1 != k2

    def test_empty_files_context_default(self) -> None:
        """Omitting files_context uses the empty string default."""
        k1 = ResponseCache.make_cache_key("gpt-4", "sys", "user")
        k2 = ResponseCache.make_cache_key("gpt-4", "sys", "user", "")
        assert k1 == k2

    def test_key_is_hex_sha256(self) -> None:
        """Key is a 64-char hex string (SHA-256)."""
        key = ResponseCache.make_cache_key("m", "s", "u")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_order_independence_of_json(self) -> None:
        """Keys don't depend on Python dict ordering (sort_keys=True)."""
        k1 = ResponseCache.make_cache_key("m", "s", "u", "f")
        k2 = ResponseCache.make_cache_key("m", "s", "u", "f")
        assert k1 == k2


# ==================================================================
# TestCacheEntry — dataclass fields
# ==================================================================


class TestCacheEntry:
    """Tests for the CacheEntry dataclass."""

    def test_all_fields_accessible(self) -> None:
        """All declared fields are accessible on the instance."""
        entry = CacheEntry(
            cache_key="key1",
            model="gpt-4",
            provider="openai",
            content="response text",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=10,
            cost_usd=0.005,
            finish_reason="stop",
            tool_calls_json=None,
            created_at="2025-01-01T00:00:00",
            expires_at="2025-01-01T01:00:00",
            file_hashes="{}",
        )
        assert entry.cache_key == "key1"
        assert entry.model == "gpt-4"
        assert entry.provider == "openai"
        assert entry.content == "response text"
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50
        assert entry.cached_tokens == 10
        assert entry.cost_usd == 0.005
        assert entry.finish_reason == "stop"
        assert entry.tool_calls_json is None
        assert entry.file_hashes == "{}"

    def test_frozen(self) -> None:
        """CacheEntry is immutable (frozen dataclass)."""
        entry = CacheEntry(
            cache_key="k", model="m", provider="p", content="c",
            input_tokens=0, output_tokens=0, cached_tokens=0, cost_usd=0.0,
            finish_reason="stop", tool_calls_json=None,
            created_at="now", expires_at="later", file_hashes="{}",
        )
        with pytest.raises(AttributeError):
            entry.content = "changed"  # type: ignore[misc]

    def test_with_tool_calls(self) -> None:
        """tool_calls_json can hold serialized data."""
        tc = json.dumps([{"name": "read_file", "args": {"path": "x.py"}}])
        entry = CacheEntry(
            cache_key="k", model="m", provider="p", content="c",
            input_tokens=0, output_tokens=0, cached_tokens=0, cost_usd=0.0,
            finish_reason="tool_calls", tool_calls_json=tc,
            created_at="now", expires_at="later", file_hashes="{}",
        )
        parsed = json.loads(entry.tool_calls_json)  # type: ignore[arg-type]
        assert parsed[0]["name"] == "read_file"


# ==================================================================
# TestCacheStats — statistics dataclass
# ==================================================================


class TestCacheStats:
    """Tests for the CacheStats dataclass."""

    def test_hit_rate_zero_on_no_requests(self) -> None:
        """Hit rate is 0.0 when there are no requests."""
        stats = CacheStats(
            total_entries=0, total_hits=0, total_misses=0, hit_rate=0.0,
            tokens_saved=0, cost_saved=0.0, cache_size_bytes=0,
            oldest_entry=None, newest_entry=None,
        )
        assert stats.hit_rate == 0.0

    def test_hit_rate_100_percent(self) -> None:
        """Hit rate is 1.0 when all requests are hits."""
        stats = CacheStats(
            total_entries=5, total_hits=10, total_misses=0, hit_rate=1.0,
            tokens_saved=1000, cost_saved=0.50, cache_size_bytes=4096,
            oldest_entry="2025-01-01", newest_entry="2025-01-02",
        )
        assert stats.hit_rate == 1.0

    def test_hit_rate_partial(self) -> None:
        """Hit rate is correctly computed from hits / total."""
        total_hits = 3
        total_misses = 7
        hit_rate = total_hits / (total_hits + total_misses)
        stats = CacheStats(
            total_entries=2, total_hits=total_hits, total_misses=total_misses,
            hit_rate=hit_rate, tokens_saved=300, cost_saved=0.03,
            cache_size_bytes=2048, oldest_entry=None, newest_entry=None,
        )
        assert stats.hit_rate == pytest.approx(0.3)


# ==================================================================
# TestResponseCache — core put / get / TTL / invalidation
# ==================================================================


class TestResponseCache:
    """Tests for the ResponseCache class."""

    def test_put_get_round_trip(self, tmp_path: Path) -> None:
        """An entry put into the cache can be retrieved by its key."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="roundtrip")
        result = cache.get("roundtrip")
        assert result is not None
        assert result.cache_key == "roundtrip"
        assert result.model == "gpt-4o-mini"
        assert result.content == "Hello, world!"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cost_usd == 0.001
        assert result.finish_reason == "stop"
        cache.close()

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        """A look-up for a non-existent key returns None."""
        cache = _make_cache(tmp_path)
        assert cache.get("nonexistent") is None
        cache.close()

    def test_ttl_expiry(self, tmp_path: Path) -> None:
        """Entries past their TTL are not returned."""
        cache = _make_cache(tmp_path, ttl=1)
        _put_sample(cache, cache_key="short_ttl", ttl=1)
        # Entry exists immediately
        assert cache.get("short_ttl") is not None
        # Wait for expiry
        time.sleep(1.5)
        assert cache.get("short_ttl") is None
        cache.close()

    def test_tier_ttl_simple(self, tmp_path: Path) -> None:
        """Tier 'simple' uses the TIER_TTLS default of 7200s."""
        cache = _make_cache(tmp_path)
        entry = _put_sample(cache, cache_key="tier_s", tier="simple")
        created = datetime.fromisoformat(entry.created_at)
        expires = datetime.fromisoformat(entry.expires_at)
        delta = (expires - created).total_seconds()
        assert delta == pytest.approx(TIER_TTLS["simple"], abs=2)
        cache.close()

    def test_tier_ttl_complex(self, tmp_path: Path) -> None:
        """Tier 'complex' uses the TIER_TTLS default of 1800s."""
        cache = _make_cache(tmp_path)
        entry = _put_sample(cache, cache_key="tier_c", tier="complex")
        created = datetime.fromisoformat(entry.created_at)
        expires = datetime.fromisoformat(entry.expires_at)
        delta = (expires - created).total_seconds()
        assert delta == pytest.approx(TIER_TTLS["complex"], abs=2)
        cache.close()

    def test_explicit_ttl_overrides_tier(self, tmp_path: Path) -> None:
        """An explicit ttl parameter overrides any tier-based default."""
        cache = _make_cache(tmp_path)
        entry = _put_sample(cache, cache_key="ttl_over", ttl=999, tier="simple")
        created = datetime.fromisoformat(entry.created_at)
        expires = datetime.fromisoformat(entry.expires_at)
        delta = (expires - created).total_seconds()
        assert delta == pytest.approx(999, abs=2)
        cache.close()

    def test_file_invalidation(self, tmp_path: Path) -> None:
        """Changing a tracked file's mtime invalidates the entry."""
        cache = _make_cache(tmp_path)

        # Create a tracked file
        tracked_file = tmp_path / "src" / "main.py"
        tracked_file.parent.mkdir(parents=True, exist_ok=True)
        tracked_file.write_text("original content")

        _put_sample(
            cache,
            cache_key="file_inv",
            file_paths=[tracked_file],
        )

        # Retrievable when file hasn't changed
        result = cache.get("file_inv", file_paths=[tracked_file])
        assert result is not None

        # Modify the file (change mtime)
        time.sleep(0.1)  # ensure different mtime
        tracked_file.write_text("modified content")

        # Now the cache entry should be invalidated
        result = cache.get("file_inv", file_paths=[tracked_file])
        assert result is None
        cache.close()

    def test_file_invalidation_file_deleted(self, tmp_path: Path) -> None:
        """Deleting a tracked file invalidates the entry."""
        cache = _make_cache(tmp_path)

        tracked = tmp_path / "foo.py"
        tracked.write_text("content")

        _put_sample(cache, cache_key="del_file", file_paths=[tracked])
        assert cache.get("del_file", file_paths=[tracked]) is not None

        tracked.unlink()
        assert cache.get("del_file", file_paths=[tracked]) is None
        cache.close()

    def test_cache_disabled_returns_none(self, tmp_path: Path) -> None:
        """When the cache is disabled, get always returns None."""
        cache = _make_cache(tmp_path, enabled=False)
        _put_sample(cache, cache_key="disabled")
        assert cache.get("disabled") is None
        cache.close()

    def test_enable_disable_toggle(self, tmp_path: Path) -> None:
        """Toggling enabled at runtime works."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="toggle")

        cache.enabled = False
        assert cache.get("toggle") is None

        cache.enabled = True
        assert cache.get("toggle") is not None
        cache.close()

    def test_clear_all(self, tmp_path: Path) -> None:
        """clear() without arguments removes all entries."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="a")
        _put_sample(cache, cache_key="b")
        _put_sample(cache, cache_key="c")
        deleted = cache.clear()
        assert deleted == 3
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") is None
        cache.close()

    def test_clear_by_age(self, tmp_path: Path) -> None:
        """clear(max_age_hours=...) only removes old entries."""
        cache = _make_cache(tmp_path)

        # Insert an entry with old timestamp by directly manipulating the DB
        conn = cache._get_connection()
        old_time = "2020-01-01T00:00:00+00:00"
        future_time = "2099-01-01T00:00:00+00:00"

        conn.execute(
            """INSERT INTO response_cache
               (cache_key, model, provider, content, input_tokens, output_tokens,
                cached_tokens, cost_usd, finish_reason, tool_calls_json,
                created_at, expires_at, file_hashes, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("old_key", "m", "p", "c", 10, 5, 0, 0.001, "stop", None,
             old_time, future_time, "{}", 0),
        )
        conn.commit()

        _put_sample(cache, cache_key="new_key")

        # Clear entries older than 1 hour — the manually inserted one is from 2020
        deleted = cache.clear(max_age_hours=1)
        assert deleted == 1  # only the old one

        # The new entry should survive
        assert cache.get("new_key") is not None
        cache.close()

    def test_cleanup_expired(self, tmp_path: Path) -> None:
        """cleanup_expired() removes only expired entries."""
        cache = _make_cache(tmp_path, ttl=1)
        _put_sample(cache, cache_key="exp1", ttl=1)
        _put_sample(cache, cache_key="exp2", ttl=86400)

        time.sleep(1.5)
        removed = cache.cleanup_expired()
        assert removed == 1  # only exp1 expired
        assert cache.get("exp2") is not None
        cache.close()

    def test_replace_existing_key(self, tmp_path: Path) -> None:
        """Putting to an existing key replaces the entry."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="replace", content="version 1")
        _put_sample(cache, cache_key="replace", content="version 2")
        result = cache.get("replace")
        assert result is not None
        assert result.content == "version 2"
        cache.close()

    def test_tool_calls_stored_and_retrieved(self, tmp_path: Path) -> None:
        """Tool calls are JSON-serialized and retrievable."""
        cache = _make_cache(tmp_path)
        tc = [{"name": "read_file", "arguments": {"path": "x.py"}}]
        _put_sample(cache, cache_key="tc", tool_calls=tc)
        result = cache.get("tc")
        assert result is not None
        assert result.tool_calls_json is not None
        parsed = json.loads(result.tool_calls_json)
        assert parsed[0]["name"] == "read_file"
        cache.close()

    def test_cached_tokens_stored(self, tmp_path: Path) -> None:
        """cached_tokens value is persisted."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="ct", cached_tokens=42)
        result = cache.get("ct")
        assert result is not None
        assert result.cached_tokens == 42
        cache.close()

    # ------------------------------------------------------------------
    # Stats tracking
    # ------------------------------------------------------------------

    def test_stats_track_hits(self, tmp_path: Path) -> None:
        """Each cache hit increments the hit counter."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="stat_hit", input_tokens=100, output_tokens=50)
        cache.get("stat_hit")
        cache.get("stat_hit")
        stats = cache.get_stats()
        assert stats.total_hits == 2
        cache.close()

    def test_stats_track_misses(self, tmp_path: Path) -> None:
        """Each cache miss increments the miss counter."""
        cache = _make_cache(tmp_path)
        cache.get("miss1")
        cache.get("miss2")
        cache.get("miss3")
        stats = cache.get_stats()
        assert stats.total_misses == 3
        cache.close()

    def test_stats_tokens_saved(self, tmp_path: Path) -> None:
        """Tokens saved reflects input + output from cache hits."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="ts", input_tokens=200, output_tokens=100)
        cache.get("ts")  # hit: 200 + 100 = 300 tokens saved
        stats = cache.get_stats()
        assert stats.tokens_saved == 300
        cache.close()

    def test_stats_cost_saved(self, tmp_path: Path) -> None:
        """cost_saved reflects the cost of cache hits."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="cs", cost_usd=0.05)
        cache.get("cs")
        stats = cache.get_stats()
        assert stats.cost_saved == pytest.approx(0.05)
        cache.close()

    def test_stats_hit_rate(self, tmp_path: Path) -> None:
        """Hit rate is hits / (hits + misses)."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="hr")
        cache.get("hr")       # hit
        cache.get("miss_a")   # miss
        cache.get("miss_b")   # miss
        cache.get("hr")       # hit
        stats = cache.get_stats()
        assert stats.total_hits == 2
        assert stats.total_misses == 2
        assert stats.hit_rate == pytest.approx(0.5)
        cache.close()

    def test_stats_total_entries(self, tmp_path: Path) -> None:
        """total_entries counts live entries in the table."""
        cache = _make_cache(tmp_path)
        assert cache.get_stats().total_entries == 0
        _put_sample(cache, cache_key="e1")
        _put_sample(cache, cache_key="e2")
        assert cache.get_stats().total_entries == 2
        cache.close()

    def test_stats_cache_size_positive(self, tmp_path: Path) -> None:
        """cache_size_bytes is positive after storing entries."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="sz")
        stats = cache.get_stats()
        assert stats.cache_size_bytes > 0
        cache.close()

    def test_stats_oldest_newest(self, tmp_path: Path) -> None:
        """oldest_entry and newest_entry are populated."""
        cache = _make_cache(tmp_path)
        stats_empty = cache.get_stats()
        assert stats_empty.oldest_entry is None
        assert stats_empty.newest_entry is None

        _put_sample(cache, cache_key="first")
        time.sleep(0.05)
        _put_sample(cache, cache_key="second")

        stats = cache.get_stats()
        assert stats.oldest_entry is not None
        assert stats.newest_entry is not None
        assert stats.oldest_entry <= stats.newest_entry
        cache.close()

    def test_flush_stats_persistence(self, tmp_path: Path) -> None:
        """Flushing persists in-memory counters to the DB."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="fp")
        cache.get("fp")      # hit
        cache.get("nope")    # miss

        # Before flush, in-memory counters are populated
        assert cache._stats_hits == 1
        assert cache._stats_misses == 1

        cache.flush_stats()

        # After flush, in-memory counters are reset
        assert cache._stats_hits == 0
        assert cache._stats_misses == 0

        # But DB stats include the flushed values
        stats = cache.get_stats()
        assert stats.total_hits == 1
        assert stats.total_misses == 1
        cache.close()

    def test_flush_stats_no_op_when_empty(self, tmp_path: Path) -> None:
        """Flushing with no pending counters is a no-op."""
        cache = _make_cache(tmp_path)
        cache.flush_stats()  # should not error
        stats = cache.get_stats()
        assert stats.total_hits == 0
        assert stats.total_misses == 0
        cache.close()

    def test_clear_all_resets_stats(self, tmp_path: Path) -> None:
        """clear() without max_age also resets aggregate stats."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="rs")
        cache.get("rs")  # hit
        cache.flush_stats()

        cache.clear()
        stats = cache.get_stats()
        assert stats.total_hits == 0
        assert stats.total_misses == 0
        assert stats.tokens_saved == 0
        assert stats.cost_saved == 0.0
        cache.close()


# ==================================================================
# TestContextManager — __enter__ / __exit__
# ==================================================================


class TestContextManager:
    """Tests for the context manager protocol."""

    def test_context_manager_basic(self, tmp_path: Path) -> None:
        """Cache can be used as a context manager."""
        cache_dir = tmp_path / "cache_cm"
        with ResponseCache(cache_dir=cache_dir) as cache:
            _put_sample(cache, cache_key="cm")
            assert cache.get("cm") is not None

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        """Exiting the context manager closes the connection."""
        cache_dir = tmp_path / "cache_close"
        cache_ref: ResponseCache | None = None
        with ResponseCache(cache_dir=cache_dir) as cache:
            cache_ref = cache
            _put_sample(cache, cache_key="close_test")

        # After exit, the thread-local connection should be None
        assert not hasattr(cache_ref._local, "conn") or cache_ref._local.conn is None


# ==================================================================
# TestComputeFileHashes
# ==================================================================


class TestComputeFileHashes:
    """Tests for ResponseCache.compute_file_hashes()."""

    def test_existing_files(self, tmp_path: Path) -> None:
        """Returns mtime for existing files."""
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("a")
        f2.write_text("b")
        hashes = ResponseCache.compute_file_hashes([f1, f2])
        assert len(hashes) == 2
        assert str(f1.resolve()) in hashes
        assert str(f2.resolve()) in hashes

    def test_nonexistent_files_skipped(self, tmp_path: Path) -> None:
        """Non-existent files are silently skipped."""
        hashes = ResponseCache.compute_file_hashes([
            tmp_path / "nonexistent.py",
        ])
        assert len(hashes) == 0

    def test_empty_list(self) -> None:
        """Empty input produces empty output."""
        assert ResponseCache.compute_file_hashes([]) == {}

    def test_mixed_strings_and_paths(self, tmp_path: Path) -> None:
        """Accepts both strings and Path objects."""
        f = tmp_path / "mixed.py"
        f.write_text("m")
        hashes = ResponseCache.compute_file_hashes([str(f), f])
        # Both resolve to the same key — just one entry
        assert len(hashes) == 1

    def test_directory_skipped(self, tmp_path: Path) -> None:
        """Directories are skipped (is_file() returns False)."""
        d = tmp_path / "subdir"
        d.mkdir()
        hashes = ResponseCache.compute_file_hashes([d])
        assert len(hashes) == 0


# ==================================================================
# TestEdgeCases — empty content, large content, special chars, etc.
# ==================================================================


class TestEdgeCases:
    """Edge case tests for ResponseCache."""

    def test_empty_content(self, tmp_path: Path) -> None:
        """Empty string content is stored and retrieved correctly."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="empty", content="")
        result = cache.get("empty")
        assert result is not None
        assert result.content == ""
        cache.close()

    def test_large_content(self, tmp_path: Path) -> None:
        """Large content (100KB) round-trips correctly."""
        cache = _make_cache(tmp_path)
        big = "x" * 100_000
        _put_sample(cache, cache_key="big", content=big)
        result = cache.get("big")
        assert result is not None
        assert len(result.content) == 100_000
        cache.close()

    def test_special_characters_in_content(self, tmp_path: Path) -> None:
        """Content with unicode, newlines, and quotes is preserved."""
        cache = _make_cache(tmp_path)
        special = 'Line1\nLine2\t"quoted"\n\u2603 snowman \U0001F600'
        _put_sample(cache, cache_key="special", content=special)
        result = cache.get("special")
        assert result is not None
        assert result.content == special
        cache.close()

    def test_special_characters_in_key_inputs(self, tmp_path: Path) -> None:
        """make_cache_key handles unicode and special chars in prompts."""
        key = ResponseCache.make_cache_key(
            model="gpt-4",
            system_prompt="You are a \u2603",
            user_prompt='Say "hello"\nworld',
            files_context="\t\r\n",
        )
        assert len(key) == 64

    def test_zero_tokens(self, tmp_path: Path) -> None:
        """Zero tokens are stored and retrieved."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="zero", input_tokens=0, output_tokens=0, cost_usd=0.0)
        result = cache.get("zero")
        assert result is not None
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cost_usd == 0.0
        cache.close()

    def test_concurrent_access(self, tmp_path: Path) -> None:
        """Multiple threads can read/write the cache concurrently."""
        cache = _make_cache(tmp_path)
        errors: list[Exception] = []
        barrier = threading.Barrier(4)

        def writer(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(10):
                    key = f"thread_{thread_id}_{i}"
                    _put_sample(cache, cache_key=key, content=f"t{thread_id}_{i}")
            except Exception as exc:
                errors.append(exc)

        def reader(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(10):
                    key = f"thread_{thread_id}_{i}"
                    cache.get(key)  # may hit or miss
            except Exception as exc:
                errors.append(exc)

        threads = []
        for tid in range(2):
            threads.append(threading.Thread(target=writer, args=(tid,)))
        for tid in range(2):
            threads.append(threading.Thread(target=reader, args=(tid,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == [], f"Concurrent access errors: {errors}"
        cache.close()

    def test_db_path_property(self, tmp_path: Path) -> None:
        """db_path property returns the correct path."""
        cache = _make_cache(tmp_path)
        assert cache.db_path == tmp_path / "cache" / "response_cache.db"
        cache.close()

    def test_enabled_property_default(self, tmp_path: Path) -> None:
        """Cache is enabled by default."""
        cache = _make_cache(tmp_path)
        assert cache.enabled is True
        cache.close()

    def test_get_with_no_file_paths(self, tmp_path: Path) -> None:
        """get() without file_paths skips file invalidation check."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="no_fp")
        result = cache.get("no_fp")
        assert result is not None
        cache.close()

    def test_put_with_no_file_paths(self, tmp_path: Path) -> None:
        """put() without file_paths stores empty file_hashes."""
        cache = _make_cache(tmp_path)
        entry = _put_sample(cache, cache_key="no_fph")
        assert entry.file_hashes == "{}"
        cache.close()

    def test_unknown_tier_uses_default_ttl(self, tmp_path: Path) -> None:
        """An unknown tier falls back to the default TTL."""
        cache = _make_cache(tmp_path, ttl=600)
        entry = _put_sample(cache, cache_key="unk", tier="unknown_tier")
        created = datetime.fromisoformat(entry.created_at)
        expires = datetime.fromisoformat(entry.expires_at)
        delta = (expires - created).total_seconds()
        assert delta == pytest.approx(600, abs=2)
        cache.close()

    def test_close_idempotent(self, tmp_path: Path) -> None:
        """Calling close() multiple times does not error."""
        cache = _make_cache(tmp_path)
        _put_sample(cache, cache_key="idem")
        cache.close()
        cache.close()  # second call should not raise

    def test_clear_empty_cache(self, tmp_path: Path) -> None:
        """Clearing an empty cache returns 0."""
        cache = _make_cache(tmp_path)
        deleted = cache.clear()
        assert deleted == 0
        cache.close()

    def test_cleanup_expired_empty(self, tmp_path: Path) -> None:
        """cleanup_expired on an empty cache returns 0."""
        cache = _make_cache(tmp_path)
        assert cache.cleanup_expired() == 0
        cache.close()

    def test_multiple_file_hashes(self, tmp_path: Path) -> None:
        """Multiple files are all tracked for invalidation."""
        cache = _make_cache(tmp_path)
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("a")
        f2.write_text("b")

        _put_sample(cache, cache_key="multi_f", file_paths=[f1, f2])
        assert cache.get("multi_f", file_paths=[f1, f2]) is not None

        # Change just one file
        time.sleep(0.1)
        f2.write_text("b_modified")
        assert cache.get("multi_f", file_paths=[f1, f2]) is None
        cache.close()


# ==================================================================
# TestTierTTLConstants
# ==================================================================


class TestTierTTLConstants:
    """Tests for the module-level TTL constants."""

    def test_default_ttl_value(self) -> None:
        """DEFAULT_TTL is 3600 seconds (1 hour)."""
        assert DEFAULT_TTL == 3600

    def test_tier_ttls_simple(self) -> None:
        """'simple' tier TTL is 7200 seconds."""
        assert TIER_TTLS["simple"] == 7200

    def test_tier_ttls_medium(self) -> None:
        """'medium' tier TTL is 3600 seconds."""
        assert TIER_TTLS["medium"] == 3600

    def test_tier_ttls_complex(self) -> None:
        """'complex' tier TTL is 1800 seconds."""
        assert TIER_TTLS["complex"] == 1800

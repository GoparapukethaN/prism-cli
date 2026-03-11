"""Tests for prism.router.rate_limiter — RateLimiter."""

from __future__ import annotations

import time

from prism.router.rate_limiter import RateLimiter

# ------------------------------------------------------------------
# Basic recording and limiting
# ------------------------------------------------------------------


class TestRecordAndLimit:
    """Tests for recording requests and checking limits."""

    def test_not_limited_initially(self) -> None:
        rl = RateLimiter(provider_limits={"test": 5})
        assert rl.is_rate_limited("test") is False

    def test_limited_after_max_requests(self) -> None:
        rl = RateLimiter(provider_limits={"test": 3}, window_seconds=60.0)
        for _ in range(3):
            rl.record_request("test")
        assert rl.is_rate_limited("test") is True

    def test_not_limited_below_max(self) -> None:
        rl = RateLimiter(provider_limits={"test": 5}, window_seconds=60.0)
        for _ in range(4):
            rl.record_request("test")
        assert rl.is_rate_limited("test") is False

    def test_different_providers_independent(self) -> None:
        rl = RateLimiter(provider_limits={"a": 2, "b": 2}, window_seconds=60.0)
        rl.record_request("a")
        rl.record_request("a")
        assert rl.is_rate_limited("a") is True
        assert rl.is_rate_limited("b") is False

    def test_unknown_provider_uses_default(self) -> None:
        rl = RateLimiter()  # default 60 RPM
        # Should not be limited after a few requests
        for _ in range(5):
            rl.record_request("unknown")
        assert rl.is_rate_limited("unknown") is False


# ------------------------------------------------------------------
# Sliding window behaviour
# ------------------------------------------------------------------


class TestSlidingWindow:
    """Tests for the time-based sliding window."""

    def test_requests_expire_after_window(self) -> None:
        rl = RateLimiter(provider_limits={"test": 2}, window_seconds=0.1)
        rl.record_request("test")
        rl.record_request("test")
        assert rl.is_rate_limited("test") is True

        # Wait for window to expire
        time.sleep(0.15)
        assert rl.is_rate_limited("test") is False

    def test_only_recent_requests_count(self) -> None:
        rl = RateLimiter(provider_limits={"test": 2}, window_seconds=0.1)
        rl.record_request("test")
        time.sleep(0.15)
        # First request should be expired now
        rl.record_request("test")
        assert rl.is_rate_limited("test") is False


# ------------------------------------------------------------------
# get_wait_time()
# ------------------------------------------------------------------


class TestGetWaitTime:
    """Tests for the wait time calculation."""

    def test_zero_when_not_limited(self) -> None:
        rl = RateLimiter(provider_limits={"test": 5})
        assert rl.get_wait_time("test") == 0.0

    def test_positive_when_limited(self) -> None:
        rl = RateLimiter(provider_limits={"test": 1}, window_seconds=10.0)
        rl.record_request("test")
        wait = rl.get_wait_time("test")
        assert wait > 0.0
        assert wait <= 10.0

    def test_wait_decreases_over_time(self) -> None:
        rl = RateLimiter(provider_limits={"test": 1}, window_seconds=1.0)
        rl.record_request("test")
        wait1 = rl.get_wait_time("test")
        time.sleep(0.1)
        wait2 = rl.get_wait_time("test")
        assert wait2 < wait1


# ------------------------------------------------------------------
# get_remaining_requests()
# ------------------------------------------------------------------


class TestGetRemainingRequests:
    """Tests for the remaining requests counter."""

    def test_full_capacity_initially(self) -> None:
        rl = RateLimiter(provider_limits={"test": 10})
        assert rl.get_remaining_requests("test") == 10

    def test_decreases_after_recording(self) -> None:
        rl = RateLimiter(provider_limits={"test": 5})
        rl.record_request("test")
        rl.record_request("test")
        assert rl.get_remaining_requests("test") == 3

    def test_never_negative(self) -> None:
        rl = RateLimiter(provider_limits={"test": 2})
        for _ in range(5):
            rl.record_request("test")
        assert rl.get_remaining_requests("test") == 0


# ------------------------------------------------------------------
# reset()
# ------------------------------------------------------------------


class TestReset:
    """Tests for clearing a provider's window."""

    def test_reset_clears_limit(self) -> None:
        rl = RateLimiter(provider_limits={"test": 2})
        rl.record_request("test")
        rl.record_request("test")
        assert rl.is_rate_limited("test") is True
        rl.reset("test")
        assert rl.is_rate_limited("test") is False

    def test_reset_restores_capacity(self) -> None:
        rl = RateLimiter(provider_limits={"test": 5})
        for _ in range(5):
            rl.record_request("test")
        rl.reset("test")
        assert rl.get_remaining_requests("test") == 5

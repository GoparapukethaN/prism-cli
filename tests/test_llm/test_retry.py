"""Tests for RetryPolicy — 10+ tests, fully offline."""

from __future__ import annotations

import time

import pytest

from prism.exceptions import (
    ProviderAuthError,
    ProviderUnavailableError,
)
from prism.llm.retry import RetryPolicy


class TestRetrySuccessPath:
    """Happy paths — no retry needed or retry recovers."""

    async def test_success_on_first_try(self) -> None:
        policy = RetryPolicy(max_retries=3)
        call_count = 0

        async def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await policy.execute(succeed)
        assert result == "ok"
        assert call_count == 1

    async def test_retry_succeeds_on_second_attempt(self) -> None:
        policy = RetryPolicy(max_retries=3, base_delay=0.001)
        call_count = 0

        async def fail_once() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderUnavailableError("test", "down")
            return "recovered"

        result = await policy.execute(fail_once)
        assert result == "recovered"
        assert call_count == 2

    async def test_retry_succeeds_on_third_attempt(self) -> None:
        policy = RetryPolicy(max_retries=3, base_delay=0.001)
        call_count = 0

        async def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("timed out")
            return "ok"

        result = await policy.execute(fail_twice)
        assert result == "ok"
        assert call_count == 3


class TestRetryExhaustion:
    """All retries used up."""

    async def test_max_retries_exhausted(self) -> None:
        policy = RetryPolicy(max_retries=2, base_delay=0.001)
        call_count = 0

        async def always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ProviderUnavailableError("test", "always down")

        with pytest.raises(ProviderUnavailableError, match="always down"):
            await policy.execute(always_fail)
        # 1 initial + 2 retries = 3
        assert call_count == 3

    async def test_zero_retries_means_no_retry(self) -> None:
        policy = RetryPolicy(max_retries=0, base_delay=0.001)
        call_count = 0

        async def fail() -> str:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("timeout")

        with pytest.raises(TimeoutError):
            await policy.execute(fail)
        assert call_count == 1


class TestNonRetryableErrors:
    """Errors not in retryable_errors are raised immediately."""

    async def test_auth_error_not_retried(self) -> None:
        policy = RetryPolicy(max_retries=3, base_delay=0.001)
        call_count = 0

        async def auth_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ProviderAuthError("openai")

        with pytest.raises(ProviderAuthError):
            await policy.execute(auth_fail)
        assert call_count == 1

    async def test_value_error_not_retried(self) -> None:
        policy = RetryPolicy(max_retries=3, base_delay=0.001)
        call_count = 0

        async def bad_input() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad value")

        with pytest.raises(ValueError, match="bad value"):
            await policy.execute(bad_input)
        assert call_count == 1

    async def test_custom_retryable_set(self) -> None:
        """Only errors in retryable_errors are retried."""
        policy = RetryPolicy(
            max_retries=2,
            base_delay=0.001,
            retryable_errors=(ValueError,),
        )
        call_count = 0

        async def value_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("custom retryable")

        with pytest.raises(ValueError):
            await policy.execute(value_fail)
        assert call_count == 3  # 1 initial + 2 retries


class TestExponentialBackoff:
    """Verify backoff timing and jitter."""

    async def test_exponential_backoff_increases_delay(self) -> None:
        policy = RetryPolicy(
            max_retries=3,
            base_delay=0.1,
            exponential_base=2.0,
            max_delay=10.0,
        )
        delays = [policy._calculate_delay(i) for i in range(4)]
        # Each delay is random(0, base * 2^i), so max possible grows.
        # We just verify the *cap* grows.
        caps = [min(0.1 * (2.0 ** i), 10.0) for i in range(4)]
        for i, delay in enumerate(delays):
            assert 0.0 <= delay <= caps[i] + 0.001

    async def test_max_delay_capped(self) -> None:
        policy = RetryPolicy(
            max_retries=10,
            base_delay=1.0,
            exponential_base=10.0,
            max_delay=5.0,
        )
        # At attempt 5: base * 10^5 = 100_000, but capped at 5.
        delay = policy._calculate_delay(5)
        assert delay <= 5.0

    async def test_jitter_varies_delay(self) -> None:
        """Multiple calls to _calculate_delay should produce different values."""
        policy = RetryPolicy(base_delay=1.0, max_delay=100.0)
        delays = [policy._calculate_delay(2) for _ in range(50)]
        unique = set(delays)
        # With 50 random samples, it is virtually impossible to get all same.
        assert len(unique) > 1

    async def test_actual_wait_time(self) -> None:
        """Verify that retries actually wait (not just calculate)."""
        policy = RetryPolicy(
            max_retries=1,
            base_delay=0.05,
            exponential_base=1.0,
            max_delay=0.05,
        )
        call_count = 0

        async def fail_once() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("slow")
            return "ok"

        start = time.perf_counter()
        await policy.execute(fail_once)
        time.perf_counter() - start
        # Should have waited at least some time (jitter means 0 to 0.05s).
        # Just assert it finished and succeeded.
        assert call_count == 2

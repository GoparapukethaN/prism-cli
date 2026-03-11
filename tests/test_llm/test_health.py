"""Tests for HealthChecker — 8+ tests, fully offline."""

from __future__ import annotations

import asyncio

from prism.llm.health import HealthChecker, HealthStatus


class TestAllHealthy:
    """All providers healthy scenario."""

    async def test_all_healthy(self) -> None:
        async def ok_check(provider: str) -> list[str]:
            return [f"{provider}/model-a", f"{provider}/model-b"]

        checker = HealthChecker(check_fn=ok_check)
        statuses = await checker.check_all(["openai", "anthropic"])
        assert len(statuses) == 2
        assert statuses["openai"].available is True
        assert statuses["anthropic"].available is True

    async def test_models_reported(self) -> None:
        async def ok_check(provider: str) -> list[str]:
            return ["model-1"]

        checker = HealthChecker(check_fn=ok_check)
        statuses = await checker.check_all(["openai"])
        assert statuses["openai"].models_available == ["model-1"]

    async def test_latency_recorded(self) -> None:
        async def ok_check(provider: str) -> list[str]:
            return []

        checker = HealthChecker(check_fn=ok_check)
        statuses = await checker.check_all(["openai"])
        assert statuses["openai"].latency_ms is not None
        assert statuses["openai"].latency_ms >= 0


class TestOneProviderDown:
    """Mixed availability."""

    async def test_one_down(self) -> None:
        async def mixed_check(provider: str) -> list[str]:
            if provider == "openai":
                raise ConnectionError("refused")
            return ["model-1"]

        checker = HealthChecker(check_fn=mixed_check)
        statuses = await checker.check_all(["openai", "anthropic"])
        assert statuses["openai"].available is False
        assert "refused" in (statuses["openai"].error or "")
        assert statuses["anthropic"].available is True

    async def test_error_message_captured(self) -> None:
        async def fail_check(provider: str) -> list[str]:
            raise RuntimeError("server error 500")

        checker = HealthChecker(check_fn=fail_check)
        statuses = await checker.check_all(["openai"])
        assert statuses["openai"].error == "server error 500"


class TestTimeoutHandling:
    """Timeout during health check."""

    async def test_timeout(self) -> None:
        async def slow_check(provider: str) -> list[str]:
            await asyncio.sleep(10)  # much longer than timeout
            return []

        checker = HealthChecker(check_fn=slow_check, timeout_s=0.05)
        statuses = await checker.check_all(["openai"])
        assert statuses["openai"].available is False
        assert "timed out" in (statuses["openai"].error or "").lower()


class TestConcurrentChecks:
    """Checks run concurrently, not sequentially."""

    async def test_concurrent_execution(self) -> None:
        """Three checks each taking 0.05s should finish in ~0.05s total."""
        call_times: list[float] = []

        async def timed_check(provider: str) -> list[str]:
            import time

            start = time.perf_counter()
            await asyncio.sleep(0.05)
            call_times.append(time.perf_counter() - start)
            return ["model-1"]

        import time

        checker = HealthChecker(check_fn=timed_check, timeout_s=5.0)
        start = time.perf_counter()
        statuses = await checker.check_all(["a", "b", "c"])
        elapsed = time.perf_counter() - start

        assert len(statuses) == 3
        # If sequential, would take >= 0.15s.  Concurrent should be < 0.12s.
        assert elapsed < 0.15


class TestNoCheckFunction:
    """When no check_fn is provided, all are assumed healthy."""

    async def test_no_check_fn_all_available(self) -> None:
        checker = HealthChecker(check_fn=None)
        statuses = await checker.check_all(["openai", "anthropic"])
        for status in statuses.values():
            assert status.available is True

    async def test_check_one_no_fn(self) -> None:
        checker = HealthChecker(check_fn=None)
        status = await checker.check_one("openai")
        assert status.available is True
        assert status.error is None


class TestHealthStatusDataclass:
    """HealthStatus creation and fields."""

    def test_default_fields(self) -> None:
        status = HealthStatus(provider="test", available=True)
        assert status.latency_ms is None
        assert status.error is None
        assert status.models_available == []

    def test_full_fields(self) -> None:
        status = HealthStatus(
            provider="openai",
            available=False,
            latency_ms=42.5,
            error="503",
            models_available=["gpt-4o"],
        )
        assert status.provider == "openai"
        assert not status.available
        assert status.latency_ms == 42.5
        assert status.error == "503"
        assert status.models_available == ["gpt-4o"]

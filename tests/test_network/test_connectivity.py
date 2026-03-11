"""Tests for ConnectivityChecker and OfflineRouter."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from prism.network.connectivity import ConnectivityChecker, OfflineRouter

# ---------------------------------------------------------------------------
# ConnectivityChecker tests
# ---------------------------------------------------------------------------


class TestConnectivityChecker:
    """Tests for ConnectivityChecker."""

    def test_online_by_default(self, checker: ConnectivityChecker) -> None:
        """Default state is online (cached value is True)."""
        with patch.object(checker, "_do_check", return_value=True):
            assert checker.is_online() is True

    def test_force_offline(self, checker: ConnectivityChecker) -> None:
        """force_offline() makes is_online() return False."""
        checker.force_offline()
        # Should not call _do_check at all
        assert checker.is_online() is False

    def test_force_online(self, checker: ConnectivityChecker) -> None:
        """force_online() makes is_online() return True regardless."""
        checker.force_offline()
        assert checker.is_online() is False

        checker.force_online()
        assert checker.is_online() is True

    def test_check_caching(self, checker: ConnectivityChecker) -> None:
        """is_online() does not re-check within check_interval."""
        call_count = 0

        def _counting_check() -> bool:
            nonlocal call_count
            call_count += 1
            return True

        with patch.object(checker, "_do_check", side_effect=_counting_check):
            checker.is_online()
            checker.is_online()
            checker.is_online()

        # Should only have called _do_check once (result is cached)
        assert call_count == 1

    def test_check_re_evaluates_after_interval(self, checker: ConnectivityChecker) -> None:
        """After check_interval elapses, _do_check is called again."""
        checker._check_interval = 0.05  # 50ms

        call_count = 0

        def _counting_check() -> bool:
            nonlocal call_count
            call_count += 1
            return True

        with patch.object(checker, "_do_check", side_effect=_counting_check):
            checker.is_online()
            assert call_count == 1

            time.sleep(0.06)
            checker.is_online()
            assert call_count == 2

    def test_check_failure_sets_offline(self, checker: ConnectivityChecker) -> None:
        """When _do_check returns False, is_online() returns False."""
        with patch.object(checker, "_do_check", return_value=False):
            assert checker.is_online() is False

    def test_reset_clears_forced_state(self, checker: ConnectivityChecker) -> None:
        """reset() goes back to auto-detection mode."""
        checker.force_offline()
        assert checker.is_online() is False

        checker.reset()
        with patch.object(checker, "_do_check", return_value=True):
            assert checker.is_online() is True

    def test_do_check_not_called_when_forced(self, checker: ConnectivityChecker) -> None:
        """_do_check is never called when in forced mode."""
        checker.force_offline()

        with patch.object(checker, "_do_check") as mock_check:
            checker.is_online()
            mock_check.assert_not_called()


# ---------------------------------------------------------------------------
# OfflineRouter tests
# ---------------------------------------------------------------------------


class TestOfflineRouter:
    """Tests for OfflineRouter."""

    def test_available_local_model(self, offline_router: OfflineRouter) -> None:
        """get_available_model returns an Ollama model ID."""
        model = offline_router.get_available_model()
        assert model == "ollama/qwen2.5-coder:7b"

    def test_available_local_model_no_ollama(
        self,
        checker: ConnectivityChecker,
    ) -> None:
        """get_available_model returns None when no Ollama is registered."""
        registry = MagicMock()
        registry.get_provider.return_value = None

        router = OfflineRouter(
            provider_registry=registry,
            connectivity_checker=checker,
        )
        assert router.get_available_model() is None

    def test_should_route_locally_when_offline(
        self,
        offline_router: OfflineRouter,
        checker: ConnectivityChecker,
    ) -> None:
        """should_route_locally returns True when offline."""
        checker.force_offline()
        assert offline_router.should_route_locally() is True

    def test_should_not_route_locally_when_online(
        self,
        offline_router: OfflineRouter,
        checker: ConnectivityChecker,
    ) -> None:
        """should_route_locally returns False when online."""
        checker.force_online()
        assert offline_router.should_route_locally() is False

    def test_queue_request(self, offline_router: OfflineRouter) -> None:
        """queue_for_retry adds requests to the queue."""
        request = {"model": "gpt-4o", "prompt": "hello"}
        offline_router.queue_for_retry(request)

        queued = offline_router.get_queued_requests()
        assert len(queued) == 1
        assert queued[0] == request

    def test_queue_multiple_requests(self, offline_router: OfflineRouter) -> None:
        """Multiple requests can be queued."""
        offline_router.queue_for_retry({"id": 1})
        offline_router.queue_for_retry({"id": 2})
        offline_router.queue_for_retry({"id": 3})

        assert len(offline_router.get_queued_requests()) == 3

    def test_clear_queue(self, offline_router: OfflineRouter) -> None:
        """clear_queue empties the retry queue."""
        offline_router.queue_for_retry({"id": 1})
        offline_router.queue_for_retry({"id": 2})
        assert len(offline_router.get_queued_requests()) == 2

        offline_router.clear_queue()
        assert len(offline_router.get_queued_requests()) == 0

    def test_get_queued_returns_copy(self, offline_router: OfflineRouter) -> None:
        """get_queued_requests returns a copy, not a reference."""
        offline_router.queue_for_retry({"id": 1})
        q1 = offline_router.get_queued_requests()
        q1.clear()
        # Original queue should still have the item
        assert len(offline_router.get_queued_requests()) == 1

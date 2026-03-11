"""Comprehensive tests for Phase 3 web browsing enhancements.

Covers: BrowseWebTool caching, user agent rotation, concurrency limiting,
enhanced content extraction, popup dismissal, ScreenshotTool enhancements,
SearchWebTool follow_first mode, and FetchDocsTool markdown conversion.

All HTTP calls and Playwright interactions are fully mocked -- no real
network requests are made.
"""

from __future__ import annotations

import re
import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import httpx

from prism.tools.base import ToolResult
from prism.tools.browser import (
    _CACHE_TTL,
    _SEMAPHORE,
    _USER_AGENTS,
    BrowseWebTool,
)
from prism.tools.fetch_docs import FetchDocsTool
from prism.tools.screenshot import (
    ScreenshotTool,
    _compress_screenshot,
    _save_screenshot,
)
from prism.tools.search_web import SearchWebTool

if TYPE_CHECKING:
    from pathlib import Path


# ======================================================================
# Helpers
# ======================================================================


def _make_httpx_mock(
    *,
    text: str = "<html><body><p>Content</p></body></html>",
    status_code: int = 200,
    content_type: str = "text/html",
    url: str = "https://example.com",
) -> tuple[MagicMock, MagicMock]:
    """Build a mock httpx.Client context manager.

    Returns:
        (mock_httpx_module, mock_client) for patching.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {"content-type": content_type}
    mock_response.text = text
    mock_response.url = url
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_response
    mock_client.post.return_value = mock_response

    mock_httpx = MagicMock()
    mock_httpx.Client.return_value = mock_client
    mock_httpx.TimeoutException = httpx.TimeoutException
    mock_httpx.HTTPStatusError = httpx.HTTPStatusError

    return mock_httpx, mock_client


def _make_playwright_mock(
    *,
    screenshot_bytes: bytes | None = None,
    page_content: str = "<html><body><p>Rendered</p></body></html>",
    element_found: bool = True,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Build a complete mock Playwright stack.

    Returns:
        (sync_playwright_func, mock_page, mock_browser)
    """
    if screenshot_bytes is None:
        screenshot_bytes = b"\x89PNG\r\n\x1a\nfake_image_data"

    mock_element = MagicMock()
    mock_element.screenshot.return_value = screenshot_bytes

    mock_page = MagicMock()
    mock_page.screenshot.return_value = screenshot_bytes
    mock_page.content.return_value = page_content
    mock_page.goto = MagicMock()
    mock_page.evaluate = MagicMock()
    if element_found:
        mock_page.query_selector.return_value = mock_element
    else:
        mock_page.query_selector.return_value = None

    mock_browser = MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_browser.close = MagicMock()

    mock_chromium = MagicMock()
    mock_chromium.launch.return_value = mock_browser

    mock_pw_instance = MagicMock()
    mock_pw_instance.chromium = mock_chromium

    mock_context = MagicMock()
    mock_context.__enter__ = MagicMock(return_value=mock_pw_instance)
    mock_context.__exit__ = MagicMock(return_value=False)

    sync_playwright_func = MagicMock(return_value=mock_context)

    return sync_playwright_func, mock_page, mock_browser


# ======================================================================
# TestBrowseWebCache
# ======================================================================


class TestBrowseWebCache:
    """Tests for BrowseWebTool instance-level URL caching."""

    def test_cache_miss_triggers_fetch(self) -> None:
        """First request to a URL should actually fetch it."""
        tool = BrowseWebTool()
        mock_httpx, mock_client = _make_httpx_mock()

        with (
            patch("prism.tools.browser.httpx", mock_httpx),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            result = tool.execute({"url": "https://example.com"})

        assert result.success is True
        mock_client.get.assert_called_once()

    def test_cache_hit_returns_cached_without_fetch(self) -> None:
        """Second identical request should return cached result."""
        tool = BrowseWebTool()
        mock_httpx, mock_client = _make_httpx_mock()

        with (
            patch("prism.tools.browser.httpx", mock_httpx),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            result1 = tool.execute({"url": "https://example.com"})
            result2 = tool.execute({"url": "https://example.com"})

        assert result1.success is True
        assert result2.success is True
        assert result1.output == result2.output
        # httpx.Client should only be called once (second time is cached)
        assert mock_client.get.call_count == 1

    def test_cache_expires_after_ttl(self) -> None:
        """Cache entries expire after the TTL (10 minutes)."""
        tool = BrowseWebTool()
        mock_httpx, mock_client = _make_httpx_mock()

        with (
            patch("prism.tools.browser.httpx", mock_httpx),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            result1 = tool.execute({"url": "https://example.com"})
            assert result1.success is True

            # Manually expire the cache entry by backdating the timestamp
            for url_key in list(tool._cache.keys()):
                ts, res = tool._cache[url_key]
                tool._cache[url_key] = (
                    ts - _CACHE_TTL - 1,
                    res,
                )

            result2 = tool.execute({"url": "https://example.com"})
            assert result2.success is True

        # Should have fetched twice (cache expired)
        assert mock_client.get.call_count == 2

    def test_different_urls_have_separate_cache_entries(self) -> None:
        """Each URL gets its own cache entry."""
        tool = BrowseWebTool()
        mock_httpx, mock_client = _make_httpx_mock()

        with (
            patch("prism.tools.browser.httpx", mock_httpx),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            tool.execute({"url": "https://example.com/page1"})
            tool.execute({"url": "https://example.com/page2"})

        # Both URLs should trigger separate fetches
        assert mock_client.get.call_count == 2

    def test_cache_stores_successful_results_only(self) -> None:
        """Failed fetches should not be cached."""
        tool = BrowseWebTool()

        mock_httpx_fail = MagicMock()
        mock_client_fail = MagicMock()
        mock_client_fail.__enter__ = MagicMock(
            return_value=mock_client_fail
        )
        mock_client_fail.__exit__ = MagicMock(return_value=False)
        mock_client_fail.get.side_effect = httpx.TimeoutException(
            "timeout"
        )
        mock_httpx_fail.Client.return_value = mock_client_fail
        mock_httpx_fail.TimeoutException = httpx.TimeoutException
        mock_httpx_fail.HTTPStatusError = httpx.HTTPStatusError

        with (
            patch("prism.tools.browser.httpx", mock_httpx_fail),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            result = tool.execute({"url": "https://example.com"})

        assert result.success is False
        # Cache should be empty -- failed results not stored
        assert len(tool._cache) == 0

    def test_cache_cleared_between_tool_instances(self) -> None:
        """Different BrowseWebTool instances have independent caches."""
        tool1 = BrowseWebTool()
        tool2 = BrowseWebTool()
        mock_httpx, _mock_client = _make_httpx_mock()

        with (
            patch("prism.tools.browser.httpx", mock_httpx),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            tool1.execute({"url": "https://example.com"})

        # tool2 should have an empty cache
        assert len(tool2._cache) == 0
        assert len(tool1._cache) == 1

    def test_cache_key_is_url_string(self) -> None:
        """Cache keys are plain URL strings."""
        tool = BrowseWebTool()
        mock_httpx, _ = _make_httpx_mock()

        with (
            patch("prism.tools.browser.httpx", mock_httpx),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            tool.execute({"url": "https://example.com/test"})

        assert "https://example.com/test" in tool._cache
        key = next(iter(tool._cache.keys()))
        assert isinstance(key, str)

    def test_concurrent_requests_to_same_url_use_cache(self) -> None:
        """When a URL is already cached, concurrent callers get the cache."""
        tool = BrowseWebTool()
        mock_httpx, mock_client = _make_httpx_mock()

        with (
            patch("prism.tools.browser.httpx", mock_httpx),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            # Prime the cache
            tool.execute({"url": "https://example.com"})

            # Simulate concurrent access: both should get cached result
            results = []
            for _ in range(5):
                r = tool.execute({"url": "https://example.com"})
                results.append(r)

        # Only 1 actual HTTP call (first one), rest cached
        assert mock_client.get.call_count == 1
        assert all(r.success for r in results)


# ======================================================================
# TestUserAgents
# ======================================================================


class TestUserAgents:
    """Tests for the user agent rotation list."""

    def test_at_least_10_user_agents_available(self) -> None:
        """Module should define at least 10 user agents."""
        assert len(_USER_AGENTS) >= 10

    def test_all_agents_are_non_empty_strings(self) -> None:
        """Every user agent must be a non-empty string."""
        for agent in _USER_AGENTS:
            assert isinstance(agent, str)
            assert len(agent) > 0

    def test_all_agents_contain_mozilla(self) -> None:
        """All user agents should include Mozilla identifier."""
        for agent in _USER_AGENTS:
            assert "Mozilla" in agent

    def test_rotation_cycles_through_agents(self) -> None:
        """random.choice should be able to select various agents."""
        import random

        random.seed(42)
        chosen = {random.choice(_USER_AGENTS) for _ in range(100)}  # noqa: S311
        # Should pick more than 1 different agent out of 100 draws
        assert len(chosen) > 1

    def test_different_domains_may_get_different_agents(self) -> None:
        """Agent selection is random, not domain-based, so
        different calls may receive different agents."""
        import random

        random.seed(12345)
        agents_seen: set[str] = set()
        for _ in range(50):
            agent = random.choice(_USER_AGENTS)  # noqa: S311
            agents_seen.add(agent)
        # With 10 agents and 50 draws, we expect multiple unique agents
        assert len(agents_seen) >= 2


# ======================================================================
# TestConcurrentLimiting
# ======================================================================


class TestConcurrentLimiting:
    """Tests for the threading semaphore concurrency limiter."""

    def test_semaphore_limits_concurrent_requests_to_10(self) -> None:
        """The global semaphore should allow at most 10 concurrent."""
        # _SEMAPHORE is initialized with _MAX_CONCURRENT = 10
        # Acquire 10 slots and verify the 11th would block
        acquired: list[bool] = []
        for _ in range(10):
            got = _SEMAPHORE.acquire(blocking=False)
            acquired.append(got)

        assert all(acquired), "Should acquire 10 slots"

        # 11th should fail (non-blocking)
        got_11 = _SEMAPHORE.acquire(blocking=False)
        assert got_11 is False, "11th acquisition should fail"

        # Release all
        for _ in range(10):
            _SEMAPHORE.release()

    def test_semaphore_releases_after_request_completes(self) -> None:
        """After a successful fetch, the semaphore slot is released."""
        tool = BrowseWebTool()
        mock_httpx, _ = _make_httpx_mock()

        with (
            patch("prism.tools.browser.httpx", mock_httpx),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            tool.execute({"url": "https://example.com"})

        # Semaphore should be fully available: try acquiring 10
        slots = []
        for _ in range(10):
            got = _SEMAPHORE.acquire(blocking=False)
            slots.append(got)
        assert all(slots)
        # Release them back
        for _ in range(10):
            _SEMAPHORE.release()

    def test_semaphore_releases_on_error(self) -> None:
        """If the fetch raises, the semaphore slot is still released."""
        tool = BrowseWebTool()

        mock_httpx_err = MagicMock()
        mock_client_err = MagicMock()
        mock_client_err.__enter__ = MagicMock(
            return_value=mock_client_err
        )
        mock_client_err.__exit__ = MagicMock(return_value=False)
        mock_client_err.get.side_effect = httpx.TimeoutException(
            "boom"
        )
        mock_httpx_err.Client.return_value = mock_client_err
        mock_httpx_err.TimeoutException = httpx.TimeoutException
        mock_httpx_err.HTTPStatusError = httpx.HTTPStatusError

        with (
            patch("prism.tools.browser.httpx", mock_httpx_err),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            result = tool.execute({"url": "https://example.com"})

        assert result.success is False

        # Semaphore should still be fully available
        slots = []
        for _ in range(10):
            got = _SEMAPHORE.acquire(blocking=False)
            slots.append(got)
        assert all(slots)
        for _ in range(10):
            _SEMAPHORE.release()

    def test_multiple_requests_can_proceed_concurrently(self) -> None:
        """Multiple threads can proceed concurrently up to the limit."""
        active_count = 0
        max_active = 0
        lock = threading.Lock()
        results: list[bool] = []

        def worker() -> None:
            nonlocal active_count, max_active
            tool = BrowseWebTool()

            fake_result = ToolResult(
                success=True, output="content"
            )

            def tracking_fetch(
                url: str,
                *,
                timeout: int,
                extract_text: bool = True,
            ) -> ToolResult:
                nonlocal active_count, max_active
                with lock:
                    active_count += 1
                    max_active = max(max_active, active_count)
                time.sleep(0.05)
                with lock:
                    active_count -= 1
                return fake_result

            tool._fetch_with_httpx = tracking_fetch  # type: ignore[assignment]
            with patch("prism.tools.browser._RATE_LIMITER"):
                res = tool.execute({"url": "https://example.com"})
            results.append(res.success)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 5
        assert all(results)
        # At least 2 threads were active concurrently
        assert max_active >= 2


# ======================================================================
# TestEnhancedContentExtraction
# ======================================================================


class TestEnhancedContentExtraction:
    """Tests for enhanced content extraction in BrowseWebTool."""

    def test_cookie_banner_elements_removed(self) -> None:
        """Elements with 'cookie' in class/id are removed."""
        tool = BrowseWebTool()
        html_input = (
            '<div class="cookie-banner">Accept cookies</div>'
            "<p>Real content here</p>"
        )
        result = tool._extract_content(html_input)
        assert "Accept cookies" not in result
        assert "Real content" in result

    def test_consent_gdpr_elements_removed(self) -> None:
        """Elements with 'consent' or 'gdpr' in class are removed."""
        tool = BrowseWebTool()
        html_input = (
            '<div class="consent-dialog">We use cookies</div>'
            '<div class="gdpr-notice">GDPR notice</div>'
            "<p>Article text</p>"
        )
        result = tool._extract_content(html_input)
        assert "We use cookies" not in result
        assert "GDPR notice" not in result
        assert "Article text" in result

    def test_hidden_elements_removed(self) -> None:
        """Elements with display:none are removed."""
        tool = BrowseWebTool()
        html_input = (
            '<div style="display:none">Hidden content</div>'
            "<p>Visible content</p>"
        )
        result = tool._extract_content(html_input)
        assert "Hidden content" not in result
        assert "Visible content" in result

    def test_code_blocks_preserved_with_language_detection(
        self,
    ) -> None:
        """Pre/code blocks with language class are preserved."""
        tool = BrowseWebTool()
        html_input = (
            '<pre><code class="language-python">'
            "def hello():\n    print(&quot;hi&quot;)"
            "</code></pre>"
            "<p>Surrounding text</p>"
        )
        result = tool._extract_content(html_input)
        assert "```python" in result
        assert "def hello():" in result
        assert 'print("hi")' in result

    def test_inline_code_preserved_as_backticks(self) -> None:
        """Inline <code> tags become backtick-wrapped text."""
        tool = BrowseWebTool()
        html_input = "<p>Use the <code>print()</code> function</p>"
        result = tool._extract_content(html_input)
        assert "`print()`" in result

    def test_multiple_code_blocks_all_preserved(self) -> None:
        """Multiple code blocks on the same page are all preserved."""
        tool = BrowseWebTool()
        html_input = (
            "<pre><code class=\"language-python\">x = 1</code></pre>"
            "<p>Text between</p>"
            '<pre><code class="language-javascript">'
            "const y = 2</code></pre>"
        )
        result = tool._extract_content(html_input)
        assert "```python" in result
        assert "x = 1" in result
        assert "```javascript" in result
        assert "const y = 2" in result

    def test_clean_content_has_no_scripts_or_styles(self) -> None:
        """Script and style tags are fully removed from output."""
        tool = BrowseWebTool()
        html_input = (
            "<script>alert('xss')</script>"
            "<style>.x{color:red}</style>"
            "<p>Clean text</p>"
        )
        result = tool._extract_content(html_input)
        assert "alert" not in result
        assert "color:red" not in result
        assert "Clean text" in result

    def test_empty_code_blocks_handled_gracefully(self) -> None:
        """Empty code blocks don't cause errors."""
        tool = BrowseWebTool()
        html_input = "<pre><code></code></pre><p>After</p>"
        result = tool._extract_content(html_input)
        assert "After" in result
        # Should have backticks from the code block but not crash
        assert "```" in result


# ======================================================================
# TestPopupDismissal
# ======================================================================


class TestPopupDismissal:
    """Tests for Playwright popup/modal/overlay dismissal."""

    def test_modal_elements_removed_from_page(self) -> None:
        """Playwright evaluate should remove modal elements."""
        tool = BrowseWebTool()
        sync_pw, mock_page, _ = _make_playwright_mock(
            page_content="<p>Page content after modal removed</p>"
        )

        with (
            patch(
                "prism.tools.browser._get_sync_playwright",
                return_value=sync_pw,
            ),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            result = tool.execute({
                "url": "https://example.com",
                "use_browser": True,
            })

        assert result.success is True
        # Verify evaluate was called (popup dismissal)
        mock_page.evaluate.assert_called_once()
        call_arg = mock_page.evaluate.call_args[0][0]
        assert "modal" in call_arg

    def test_popup_elements_removed_from_page(self) -> None:
        """The evaluate script targets popup class selectors."""
        tool = BrowseWebTool()
        sync_pw, mock_page, _ = _make_playwright_mock()

        with (
            patch(
                "prism.tools.browser._get_sync_playwright",
                return_value=sync_pw,
            ),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            tool.execute({
                "url": "https://example.com",
                "use_browser": True,
            })

        call_arg = mock_page.evaluate.call_args[0][0]
        assert "popup" in call_arg

    def test_overlay_elements_removed_from_page(self) -> None:
        """The evaluate script targets overlay class selectors."""
        tool = BrowseWebTool()
        sync_pw, mock_page, _ = _make_playwright_mock()

        with (
            patch(
                "prism.tools.browser._get_sync_playwright",
                return_value=sync_pw,
            ),
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            tool.execute({
                "url": "https://example.com",
                "use_browser": True,
            })

        call_arg = mock_page.evaluate.call_args[0][0]
        assert "overlay" in call_arg


# ======================================================================
# TestScreenshotEnhancements
# ======================================================================


class TestScreenshotEnhancements:
    """Tests for ScreenshotTool Phase 3 enhancements."""

    def test_element_screenshot_via_css_selector(self) -> None:
        """When selector is provided, element.screenshot() is called."""
        tool = ScreenshotTool()
        fake_png = b"\x89PNG\r\n\x1a\nsmall_element"
        sync_pw, mock_page, _ = _make_playwright_mock(
            screenshot_bytes=fake_png, element_found=True
        )

        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            return_value=sync_pw,
        ):
            result = tool.execute({
                "url": "https://example.com",
                "selector": "#main-content",
            })

        assert result.success is True
        mock_page.query_selector.assert_called_once_with(
            "#main-content"
        )
        assert result.metadata is not None
        assert result.metadata["selector"] == "#main-content"

    def test_element_not_found_returns_error(self) -> None:
        """When selector matches nothing, return an error."""
        tool = ScreenshotTool()
        sync_pw, _mock_page, _ = _make_playwright_mock(
            element_found=False
        )

        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            return_value=sync_pw,
        ):
            result = tool.execute({
                "url": "https://example.com",
                "selector": "#nonexistent",
            })

        assert result.success is False
        assert "not found" in (result.error or "").lower()

    def test_viewport_only_mode_sets_full_page_false(self) -> None:
        """viewport_only=True overrides full_page to False."""
        tool = ScreenshotTool()
        sync_pw, mock_page, _ = _make_playwright_mock()

        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            return_value=sync_pw,
        ):
            result = tool.execute({
                "url": "https://example.com",
                "viewport_only": True,
            })

        assert result.success is True
        mock_page.screenshot.assert_called_once_with(
            full_page=False
        )
        assert result.metadata is not None
        assert result.metadata.get("viewport_only") is True

    def test_default_is_full_page(self) -> None:
        """Default screenshot mode is full_page=True."""
        tool = ScreenshotTool()
        sync_pw, mock_page, _ = _make_playwright_mock()

        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            return_value=sync_pw,
        ):
            result = tool.execute({"url": "https://example.com"})

        assert result.success is True
        mock_page.screenshot.assert_called_once_with(full_page=True)
        assert result.metadata is not None
        assert result.metadata["full_page"] is True

    def test_auto_compress_triggered_when_over_1mb(self) -> None:
        """Screenshots larger than 1MB trigger compression."""
        large_data = b"\x89PNG" + b"\x00" * (1_048_577)
        compressed_data = b"\x89PNG" + b"\x00" * 500

        with patch(
            "prism.tools.screenshot._compress_screenshot",
            return_value=compressed_data,
        ) as mock_compress:
            tool = ScreenshotTool()
            sync_pw, _mock_page, _ = _make_playwright_mock(
                screenshot_bytes=large_data
            )

            with patch(
                "prism.tools.screenshot._get_sync_playwright",
                return_value=sync_pw,
            ):
                result = tool.execute(
                    {"url": "https://example.com"}
                )

            assert result.success is True
            mock_compress.assert_called_once_with(large_data)

    def test_small_screenshots_not_compressed(self) -> None:
        """Screenshots under 1MB are returned as-is."""
        small_data = b"\x89PNG" + b"\x00" * 100
        compressed = _compress_screenshot(small_data)
        assert compressed == small_data

    def test_screenshots_saved_to_prism_directory(
        self, tmp_path: Path
    ) -> None:
        """Screenshots are saved to ~/.prism/screenshots/."""
        screenshot_dir = tmp_path / ".prism" / "screenshots"

        with patch(
            "prism.tools.screenshot._SCREENSHOT_DIR",
            screenshot_dir,
        ):
            saved = _save_screenshot(
                b"\x89PNGfakedata",
                "https://docs.python.org/3/tutorial",
            )

        assert saved is not None
        assert saved.exists()
        assert saved.parent == screenshot_dir
        assert saved.suffix == ".png"

    def test_saved_filename_includes_domain_and_timestamp(
        self, tmp_path: Path
    ) -> None:
        """Saved filename contains the domain and a timestamp."""
        screenshot_dir = tmp_path / ".prism" / "screenshots"

        with patch(
            "prism.tools.screenshot._SCREENSHOT_DIR",
            screenshot_dir,
        ):
            saved = _save_screenshot(
                b"\x89PNGdata",
                "https://example.com/page",
            )

        assert saved is not None
        name = saved.name
        assert "example_com" in name
        # Timestamp pattern: YYYYMMDD_HHMMSS
        assert re.search(r"\d{8}_\d{6}", name) is not None

    def test_screenshot_metadata_includes_all_fields(self) -> None:
        """Metadata should include url, width, height, full_page,
        format, and size_bytes."""
        tool = ScreenshotTool()
        fake_png = b"\x89PNG" + b"\x00" * 50
        sync_pw, _, _ = _make_playwright_mock(
            screenshot_bytes=fake_png
        )

        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            return_value=sync_pw,
        ):
            result = tool.execute({
                "url": "https://example.com",
                "width": 1024,
                "height": 768,
            })

        assert result.success is True
        meta = result.metadata
        assert meta is not None
        assert meta["url"] == "https://example.com"
        assert meta["width"] == 1024
        assert meta["height"] == 768
        assert meta["full_page"] is True
        assert meta["format"] == "png"
        assert "size_bytes" in meta
        assert isinstance(meta["size_bytes"], int)

    def test_selector_and_viewport_only_together(self) -> None:
        """Both selector and viewport_only can be used together;
        selector takes precedence for the screenshot target."""
        tool = ScreenshotTool()
        fake_png = b"\x89PNGfake"
        sync_pw, mock_page, _ = _make_playwright_mock(
            screenshot_bytes=fake_png, element_found=True
        )

        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            return_value=sync_pw,
        ):
            result = tool.execute({
                "url": "https://example.com",
                "selector": ".hero",
                "viewport_only": True,
            })

        assert result.success is True
        # When selector is present, element.screenshot() is used
        mock_page.query_selector.assert_called_once_with(".hero")
        meta = result.metadata
        assert meta is not None
        assert meta.get("selector") == ".hero"
        assert meta.get("viewport_only") is True


# ======================================================================
# TestSearchWebFollowFirst
# ======================================================================


class TestSearchWebFollowFirst:
    """Tests for SearchWebTool follow_first parameter."""

    @staticmethod
    def _make_ddg_html(
        results: list[dict[str, str]],
    ) -> str:
        """Build fake DuckDuckGo HTML from result dicts."""
        parts = []
        for r in results:
            parts.append(
                f'<a class="result__a" href="{r["url"]}">'
                f'{r["title"]}</a>'
            )
            parts.append(
                f'<td class="result__snippet">'
                f'{r.get("snippet", "")}</td>'
            )
        return "\n".join(parts)

    def test_follow_first_false_returns_only_search_results(
        self,
    ) -> None:
        """Default (follow_first=False) returns search results only."""
        tool = SearchWebTool()
        ddg_html = self._make_ddg_html([
            {
                "title": "Result 1",
                "url": "https://example.com",
                "snippet": "Snippet 1",
            },
        ])

        mock_response = MagicMock()
        mock_response.text = ddg_html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with (
            patch("httpx.Client") as mock_cls,
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.post.return_value = mock_response
            mock_cls.return_value = mc

            result = tool.execute({"query": "python testing"})

        assert result.success is True
        assert "Content from top result" not in result.output

    def test_follow_first_true_fetches_first_result_content(
        self,
    ) -> None:
        """follow_first=True appends content from the top result."""
        tool = SearchWebTool()
        ddg_html = self._make_ddg_html([
            {
                "title": "Python Docs",
                "url": "https://docs.python.org",
                "snippet": "Official Python documentation",
            },
        ])

        mock_search_resp = MagicMock()
        mock_search_resp.text = ddg_html
        mock_search_resp.status_code = 200
        mock_search_resp.raise_for_status = MagicMock()

        mock_follow_resp = MagicMock()
        mock_follow_resp.text = (
            "<html><body><p>Welcome to Python docs</p></body></html>"
        )
        mock_follow_resp.status_code = 200
        mock_follow_resp.headers = {
            "content-type": "text/html"
        }
        mock_follow_resp.raise_for_status = MagicMock()

        with (
            patch("httpx.Client") as mock_cls,
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.post.return_value = mock_search_resp
            mc.get.return_value = mock_follow_resp
            mock_cls.return_value = mc

            result = tool.execute({
                "query": "python docs",
                "follow_first": True,
            })

        assert result.success is True
        assert "Content from top result" in result.output

    def test_follow_first_with_empty_results_no_crash(self) -> None:
        """follow_first=True with no results should not crash."""
        tool = SearchWebTool()

        mock_response = MagicMock()
        mock_response.text = "<html><body>No results</body></html>"
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with (
            patch("httpx.Client") as mock_cls,
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.post.return_value = mock_response
            mock_cls.return_value = mc

            result = tool.execute({
                "query": "obscure nothing found",
                "follow_first": True,
            })

        assert result.success is True
        assert "No results found" in result.output

    def test_fetched_content_appended_to_output(self) -> None:
        """Fetched first-result content appears after search results."""
        tool = SearchWebTool()
        ddg_html = self._make_ddg_html([
            {
                "title": "Example",
                "url": "https://example.com",
                "snippet": "Example site",
            },
        ])

        mock_search_resp = MagicMock()
        mock_search_resp.text = ddg_html
        mock_search_resp.status_code = 200
        mock_search_resp.raise_for_status = MagicMock()

        mock_follow_resp = MagicMock()
        mock_follow_resp.text = (
            "<html><body><p>Detailed example content</p>"
            "</body></html>"
        )
        mock_follow_resp.status_code = 200
        mock_follow_resp.headers = {
            "content-type": "text/html"
        }
        mock_follow_resp.raise_for_status = MagicMock()

        with (
            patch("httpx.Client") as mock_cls,
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.post.return_value = mock_search_resp
            mc.get.return_value = mock_follow_resp
            mock_cls.return_value = mc

            result = tool.execute({
                "query": "example",
                "follow_first": True,
            })

        assert result.success is True
        output = result.output
        # Search results appear first
        search_pos = output.find("Example")
        # Fetched content appears after
        content_pos = output.find("Content from top result")
        assert search_pos < content_pos

    def test_fetch_failure_still_returns_search_results(
        self,
    ) -> None:
        """If fetching the first result fails, search results still
        returned."""
        tool = SearchWebTool()
        ddg_html = self._make_ddg_html([
            {
                "title": "Working Result",
                "url": "https://example.com",
                "snippet": "Works fine",
            },
        ])

        mock_search_resp = MagicMock()
        mock_search_resp.text = ddg_html
        mock_search_resp.status_code = 200
        mock_search_resp.raise_for_status = MagicMock()

        with (
            patch("httpx.Client") as mock_cls,
            patch("prism.tools.browser._RATE_LIMITER"),
        ):
            mc = MagicMock()
            mc.__enter__ = MagicMock(return_value=mc)
            mc.__exit__ = MagicMock(return_value=False)
            mc.post.return_value = mock_search_resp
            # GET for follow_first raises an error
            mc.get.side_effect = httpx.TimeoutException("fail")
            mock_cls.return_value = mc

            result = tool.execute({
                "query": "test",
                "follow_first": True,
            })

        assert result.success is True
        assert "Working Result" in result.output


# ======================================================================
# TestFetchDocsMarkdown
# ======================================================================


class TestFetchDocsMarkdown:
    """Tests for FetchDocsTool HTML-to-markdown conversion."""

    def test_h1_converted_to_hash_heading(self) -> None:
        """<h1> becomes # heading."""
        result = FetchDocsTool._extract_doc_content(
            "<main><h1>Main Title</h1></main>"
        )
        assert "# Main Title" in result

    def test_h2_converted_to_double_hash_heading(self) -> None:
        """<h2> becomes ## heading."""
        result = FetchDocsTool._extract_doc_content(
            "<main><h2>Section</h2></main>"
        )
        assert "## Section" in result

    def test_h3_through_h6_converted_to_headings(self) -> None:
        """<h3>-<h6> become ###-###### headings."""
        for level in range(3, 7):
            tag = f"h{level}"
            html_input = (
                f"<main><{tag}>Heading {level}</{tag}></main>"
            )
            result = FetchDocsTool._extract_doc_content(
                html_input
            )
            prefix = "#" * level
            assert f"{prefix} Heading {level}" in result

    def test_code_blocks_with_language_class_preserved(
        self,
    ) -> None:
        """<pre><code class="language-python"> becomes ```python."""
        html_input = (
            "<main><pre><code class=\"language-python\">"
            "def foo(): pass</code></pre></main>"
        )
        result = FetchDocsTool._extract_doc_content(html_input)
        assert "```python" in result
        assert "def foo(): pass" in result

    def test_inline_code_wrapped_in_backticks(self) -> None:
        """<code>x</code> becomes `x`."""
        html_input = (
            "<main><p>Use <code>pip install</code> to install"
            "</p></main>"
        )
        result = FetchDocsTool._extract_doc_content(html_input)
        assert "`pip install`" in result

    def test_links_converted_to_markdown_format(self) -> None:
        """<a href="url">text</a> becomes [text](url)."""
        html_input = (
            '<main><a href="https://python.org">Python</a>'
            "</main>"
        )
        result = FetchDocsTool._extract_doc_content(html_input)
        assert "[Python](https://python.org)" in result

    def test_bold_tags_converted_to_double_asterisks(self) -> None:
        """<strong> and <b> become **text**."""
        html_input = (
            "<main><strong>important</strong> and "
            "<b>bold</b></main>"
        )
        result = FetchDocsTool._extract_doc_content(html_input)
        assert "**important**" in result
        assert "**bold**" in result

    def test_italic_tags_converted_to_single_asterisks(
        self,
    ) -> None:
        """<em> and <i> become *text*."""
        html_input = (
            "<main><em>emphasis</em> and <i>italic</i></main>"
        )
        result = FetchDocsTool._extract_doc_content(html_input)
        assert "*emphasis*" in result
        assert "*italic*" in result

    def test_ordered_lists_converted_to_numbered_items(
        self,
    ) -> None:
        """<ol><li> becomes 1. item, 2. item, etc."""
        html_input = (
            "<main><ol>"
            "<li>First item</li>"
            "<li>Second item</li>"
            "<li>Third item</li>"
            "</ol></main>"
        )
        result = FetchDocsTool._extract_doc_content(html_input)
        assert "1. First item" in result
        assert "2. Second item" in result
        assert "3. Third item" in result

    def test_unordered_lists_converted_to_bullet_items(
        self,
    ) -> None:
        """<ul><li> becomes - item."""
        html_input = (
            "<main><ul>"
            "<li>Apple</li>"
            "<li>Banana</li>"
            "</ul></main>"
        )
        result = FetchDocsTool._extract_doc_content(html_input)
        assert "- Apple" in result
        assert "- Banana" in result

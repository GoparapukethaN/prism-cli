"""Web browsing tool -- fetches and extracts content from URLs.

Supports two modes:

* **httpx** (default) -- lightweight, no JavaScript rendering.
  Good for documentation pages, REST APIs, and static HTML.
* **Playwright** (optional) -- full browser engine, lazy-loaded.
  Needed for JavaScript-heavy SPAs.

Security:
  ``file://`` URLs, localhost (except Ollama port 11434), and
  private/internal IPs are rejected by default.
"""

from __future__ import annotations

import base64
import contextlib
import html
import ipaddress
import random
import re
import threading
import time as _time
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

logger = structlog.get_logger(__name__)

# Tags whose content is not useful for text extraction.
_STRIP_TAGS_RE = re.compile(
    r"<(script|style|nav|footer|header|aside|noscript|iframe|svg)"
    r"[\s>].*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

# Cookie banners, consent dialogs, GDPR overlays.
_COOKIE_BANNER_RE = re.compile(
    r"<[^>]+(?:class|id)=[\"'][^\"']*"
    r"(?:cookie|consent|gdpr)[^\"']*[\"'][^>]*>.*?</(?:div|section|aside)>",
    re.DOTALL | re.IGNORECASE,
)

# Hidden elements (display:none or visibility:hidden).
_HIDDEN_ELEMENT_RE = re.compile(
    r"<[^>]+style=[\"'][^\"']*"
    r"(?:display\s*:\s*none|visibility\s*:\s*hidden)"
    r"[^\"']*[\"'][^>]*>.*?</[^>]+>",
    re.DOTALL | re.IGNORECASE,
)

# Code block patterns for preservation.
_CODE_BLOCK_RE = re.compile(
    r"<pre[^>]*>\s*<code(?:\s+class=[\"']"
    r"(?:language-)?(\w+)[\"'])?[^>]*>(.*?)</code>\s*</pre>",
    re.DOTALL | re.IGNORECASE,
)
_INLINE_CODE_RE = re.compile(
    r"<code[^>]*>(.*?)</code>",
    re.DOTALL | re.IGNORECASE,
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\n{3,}")

# ------------------------------------------------------------------
# URL result cache (10-minute TTL)
# ------------------------------------------------------------------
_URL_CACHE: dict[str, tuple[float, ToolResult]] = {}
_CACHE_TTL = 600  # 10 minutes


def _get_cached(url: str) -> ToolResult | None:
    """Return cached result for *url* if still valid.

    Args:
        url: The URL to look up.

    Returns:
        Cached ToolResult or None if not found / expired.
    """
    if url in _URL_CACHE:
        ts, result = _URL_CACHE[url]
        if _time.time() - ts < _CACHE_TTL:
            return result
        del _URL_CACHE[url]
    return None


def _set_cached(url: str, result: ToolResult) -> None:
    """Store a successful fetch result in the cache.

    Args:
        url: The URL that was fetched.
        result: The ToolResult to cache.
    """
    _URL_CACHE[url] = (_time.time(), result)


def _get_sync_playwright() -> Any:
    """Import and return ``sync_playwright`` from Playwright.

    Raises:
        ImportError: If Playwright is not installed.
    """
    from playwright.sync_api import sync_playwright

    return sync_playwright


# Ports that are allowed on localhost.
_LOCALHOST_ALLOWED_PORTS: frozenset[int] = frozenset({11434})  # Ollama

# Private IP ranges.
_PRIVATE_PREFIXES = (
    "10.", "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
    "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
    "172.30.", "172.31.", "192.168.", "169.254.",
)

_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) "
    "Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) "
    "Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) "
    "Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Edge/122.0.0.0 Safari/537.36",
]

# ------------------------------------------------------------------
# Concurrency limiter (threading-based for sync execute())
# ------------------------------------------------------------------
_MAX_CONCURRENT = 10
_SEMAPHORE = threading.Semaphore(_MAX_CONCURRENT)


class DomainRateLimiter:
    """Per-domain rate limiter to avoid getting blocked.

    Enforces a minimum delay between requests to the same domain.
    Thread-safe via simple dict-based tracking.
    """

    def __init__(self, min_delay: float = 2.0) -> None:
        """Initialize the rate limiter.

        Args:
            min_delay: Minimum seconds between requests to same domain.
        """
        self._min_delay = min_delay
        self._last_request: dict[str, float] = {}

    def wait_if_needed(self, url: str) -> None:
        """Block until it's safe to request the URL's domain.

        Args:
            url: The target URL.
        """
        domain = self._extract_domain(url)
        now = _time.monotonic()
        last = self._last_request.get(domain, 0.0)
        elapsed = now - last

        if elapsed < self._min_delay:
            wait_time = self._min_delay - elapsed
            _time.sleep(wait_time)

        self._last_request[domain] = _time.monotonic()

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract the domain from a URL.

        Args:
            url: Full URL string.

        Returns:
            Domain string (lowercase).
        """
        try:
            parsed = urlparse(url)
            return (parsed.hostname or "").lower()
        except Exception:
            return "unknown"


# Module-level rate limiter instance (shared across tool calls)
_RATE_LIMITER = DomainRateLimiter(min_delay=2.0)


class BrowseWebTool(Tool):
    """Browse a URL and extract content.

    Uses httpx for simple fetches, Playwright for JS-heavy pages.
    Playwright is lazy-loaded (only imported when actually needed).
    Each instance maintains its own URL result cache with a
    10-minute TTL.
    """

    def __init__(self) -> None:
        """Initialize the tool with an empty URL cache."""
        self._cache: dict[str, tuple[float, ToolResult]] = {}

    @property
    def name(self) -> str:
        return "browse_web"

    @property
    def description(self) -> str:
        return (
            "Fetch a web page and return its text content. "
            "Set use_browser=true for JavaScript-heavy pages."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "extract_text": {
                    "type": "boolean",
                    "description": (
                        "Extract readable text from HTML "
                        "(default true)."
                    ),
                    "default": True,
                },
                "screenshot": {
                    "type": "boolean",
                    "description": (
                        "Capture a screenshot "
                        "(requires Playwright)."
                    ),
                    "default": False,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds.",
                    "default": 30,
                },
                "use_browser": {
                    "type": "boolean",
                    "description": (
                        "Use Playwright instead of httpx."
                    ),
                    "default": False,
                },
            },
            "required": ["url"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Fetch a URL and return extracted content.

        Args:
            arguments: Must contain ``url``; optional
                ``extract_text``, ``screenshot``, ``timeout``,
                ``use_browser``.
        """
        validated = self.validate_arguments(arguments)

        url: str = validated["url"]
        extract_text: bool = validated.get("extract_text", True)
        screenshot: bool = validated.get("screenshot", False)
        timeout: int = validated.get("timeout", 30)
        use_browser: bool = validated.get("use_browser", False)

        if not self._is_safe_url(url):
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"URL rejected by security policy: {url}"
                ),
            )

        # Check instance cache (skip for screenshot requests)
        if not screenshot:
            cached = self._get_cached(url)
            if cached is not None:
                logger.debug("cache_hit", url=url)
                return cached

        with _SEMAPHORE:
            if use_browser or screenshot:
                result = self._fetch_with_playwright(
                    url,
                    timeout=timeout,
                    screenshot=screenshot,
                )
            else:
                result = self._fetch_with_httpx(
                    url,
                    timeout=timeout,
                    extract_text=extract_text,
                )

        # Cache successful results (skip screenshots)
        if result.success and not screenshot:
            self._set_cached(url, result)

        return result

    # ------------------------------------------------------------------
    # Instance-level URL cache
    # ------------------------------------------------------------------

    def _get_cached(self, url: str) -> ToolResult | None:
        """Return cached result for *url* if still valid.

        Args:
            url: The URL to look up.

        Returns:
            Cached ToolResult or None if not found / expired.
        """
        if url in self._cache:
            ts, result = self._cache[url]
            if _time.time() - ts < _CACHE_TTL:
                return result
            del self._cache[url]
        return None

    def _set_cached(
        self, url: str, result: ToolResult
    ) -> None:
        """Store a successful result in instance cache.

        Args:
            url: The URL that was fetched.
            result: The ToolResult to cache.
        """
        self._cache[url] = (_time.time(), result)

    # ------------------------------------------------------------------
    # Fetchers
    # ------------------------------------------------------------------

    def _fetch_with_httpx(
        self,
        url: str,
        *,
        timeout: int,
        extract_text: bool = True,
    ) -> ToolResult:
        """Lightweight fetch using httpx."""
        try:
            _RATE_LIMITER.wait_if_needed(url)
            user_agent = random.choice(_USER_AGENTS)  # noqa: S311
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
            ) as client:
                response = client.get(url)
                response.raise_for_status()

            content_type = response.headers.get(
                "content-type", ""
            )
            raw_text = response.text

            if extract_text and "html" in content_type.lower():
                extracted = self._extract_content(raw_text)
            else:
                extracted = raw_text

            return ToolResult(
                success=True,
                output=extracted,
                metadata={
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "url": str(response.url),
                },
            )
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Request timed out after {timeout}s: {url}"
                ),
            )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"HTTP {exc.response.status_code}: {url}"
                ),
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Fetch failed: {exc}",
            )

    def _fetch_with_playwright(
        self, url: str, *, timeout: int, screenshot: bool
    ) -> ToolResult:
        """Full browser fetch. Lazy loads Playwright."""
        try:
            sync_playwright = _get_sync_playwright()
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "Playwright is not installed. "
                    "Install with: pip install playwright "
                    "&& playwright install chromium"
                ),
            )

        try:
            _RATE_LIMITER.wait_if_needed(url)
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=timeout * 1000)

                # Auto-dismiss popups / modals / overlays
                with contextlib.suppress(Exception):
                    page.evaluate(
                        "document.querySelectorAll("
                        "'[class*=modal], [class*=popup], "
                        "[class*=overlay]'"
                        ").forEach(e => e.remove())"
                    )

                content = page.content()
                extracted = self._extract_content(content)

                result_meta: dict[str, Any] = {"url": url}
                output = extracted

                if screenshot:
                    screenshot_bytes = page.screenshot(
                        full_page=True
                    )
                    result_meta["screenshot_base64"] = (
                        base64.b64encode(
                            screenshot_bytes
                        ).decode("utf-8")
                    )

                browser.close()

            return ToolResult(
                success=True,
                output=output,
                metadata=result_meta,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Browser fetch failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def _extract_content(self, raw_html: str) -> str:
        """Extract readable text from HTML.

        Removes script, style, nav, footer, header, aside,
        noscript, iframe, SVG elements, cookie banners, consent
        dialogs, and hidden elements. Preserves code blocks with
        language detection, then strips remaining tags and
        collapses excessive whitespace into clean markdown.
        """
        text = _STRIP_TAGS_RE.sub("", raw_html)

        # Remove cookie banners / consent / GDPR overlays
        text = _COOKIE_BANNER_RE.sub("", text)

        # Remove hidden elements
        text = _HIDDEN_ELEMENT_RE.sub("", text)

        # Preserve code blocks with language detection
        def _replace_code_block(match: re.Match[str]) -> str:
            lang = match.group(1) or ""
            code = match.group(2)
            code = re.sub(r"<[^>]+>", "", code)
            code = html.unescape(code)
            return f"\n```{lang}\n{code}\n```\n"

        text = _CODE_BLOCK_RE.sub(_replace_code_block, text)

        # Preserve inline code
        def _replace_inline_code(
            match: re.Match[str],
        ) -> str:
            code = match.group(1)
            code = re.sub(r"<[^>]+>", "", code)
            code = html.unescape(code)
            return f"`{code}`"

        text = _INLINE_CODE_RE.sub(
            _replace_inline_code, text
        )

        # Strip remaining HTML tags
        text = _HTML_TAG_RE.sub(" ", text)
        text = html.unescape(text)
        text = _WHITESPACE_RE.sub("\n\n", text)
        text = text.strip()
        text = self._truncate_content(text)
        return text

    @staticmethod
    def _truncate_content(
        text: str, max_tokens: int = 8000
    ) -> str:
        """Intelligently truncate content to fit token budget.

        Preserves the beginning and end of content, removing the
        middle if truncation is needed. Uses approximate token
        estimation.

        Args:
            text: The text to truncate.
            max_tokens: Maximum approximate token count.

        Returns:
            Truncated text with indicator if truncated.
        """
        if not text:
            return text

        # Rough estimate: 1 token ~ 4 characters
        max_chars = max_tokens * 4

        if len(text) <= max_chars:
            return text

        # Keep first 70% and last 30% of the budget
        head_budget = int(max_chars * 0.7)
        tail_budget = max_chars - head_budget - 50

        head = text[:head_budget]
        tail = text[-tail_budget:] if tail_budget > 0 else ""

        return (
            f"{head}\n\n"
            f"[... content truncated "
            f"({len(text)} chars total) ...]\n\n"
            f"{tail}"
        )

    # ------------------------------------------------------------------
    # URL safety
    # ------------------------------------------------------------------

    def _is_safe_url(self, url: str) -> bool:
        """Reject file://, localhost (except Ollama), private IPs."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False

        # Scheme check
        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname or ""

        # Localhost check
        if hostname in (
            "localhost", "127.0.0.1", "::1", "0.0.0.0",
        ):
            port = parsed.port
            return bool(
                port is not None
                and port in _LOCALHOST_ALLOWED_PORTS
            )

        # Private IP check
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_reserved or ip.is_loopback:
                return False
        except ValueError:
            # Not an IP -- check hostname heuristics
            for prefix in _PRIVATE_PREFIXES:
                if hostname.startswith(prefix):
                    return False

        return True

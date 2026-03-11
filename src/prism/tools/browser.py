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
import html
import ipaddress
import re
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
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\n{3,}")


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
_PRIVATE_PREFIXES = ("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                     "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                     "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                     "172.30.", "172.31.", "192.168.", "169.254.")


class BrowseWebTool(Tool):
    """Browse a URL and extract content.

    Uses httpx for simple fetches, Playwright for JavaScript-heavy pages.
    Playwright is lazy-loaded (only imported when actually needed).
    """

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
                    "description": "Extract readable text from HTML (default true).",
                    "default": True,
                },
                "screenshot": {
                    "type": "boolean",
                    "description": "Capture a screenshot (requires Playwright).",
                    "default": False,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Request timeout in seconds.",
                    "default": 30,
                },
                "use_browser": {
                    "type": "boolean",
                    "description": "Use Playwright instead of httpx.",
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
            arguments: Must contain ``url``; optional ``extract_text``,
                       ``screenshot``, ``timeout``, ``use_browser``.
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
                error=f"URL rejected by security policy: {url}",
            )

        if use_browser or screenshot:
            return self._fetch_with_playwright(
                url, timeout=timeout, screenshot=screenshot
            )
        return self._fetch_with_httpx(
            url, timeout=timeout, extract_text=extract_text
        )

    # ------------------------------------------------------------------
    # Fetchers
    # ------------------------------------------------------------------

    def _fetch_with_httpx(
        self, url: str, *, timeout: int, extract_text: bool = True
    ) -> ToolResult:
        """Lightweight fetch using httpx."""
        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "Prism-CLI/1.0"},
            ) as client:
                response = client.get(url)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
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
                error=f"Request timed out after {timeout}s: {url}",
            )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"HTTP {exc.response.status_code}: {url}",
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
                    "Install with: pip install playwright && playwright install chromium"
                ),
            )

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=timeout * 1000)

                content = page.content()
                extracted = self._extract_content(content)

                result_meta: dict[str, Any] = {"url": url}
                output = extracted

                if screenshot:
                    screenshot_bytes = page.screenshot(full_page=True)
                    result_meta["screenshot_base64"] = base64.b64encode(
                        screenshot_bytes
                    ).decode("utf-8")

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

        Removes script, style, nav, footer, header, aside, noscript,
        iframe, and SVG elements, then strips remaining tags and collapses
        excessive whitespace.
        """
        text = _STRIP_TAGS_RE.sub("", raw_html)
        text = _HTML_TAG_RE.sub(" ", text)
        text = html.unescape(text)
        text = _WHITESPACE_RE.sub("\n\n", text)
        return text.strip()

    # ------------------------------------------------------------------
    # URL safety
    # ------------------------------------------------------------------

    def _is_safe_url(self, url: str) -> bool:
        """Reject ``file://``, localhost (except Ollama), and private IPs."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False

        # Scheme check
        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname or ""

        # Localhost check
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            port = parsed.port
            return bool(port is not None and port in _LOCALHOST_ALLOWED_PORTS)

        # Private IP check
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_reserved or ip.is_loopback:
                return False
        except ValueError:
            # Not an IP address -- check hostname heuristics
            for prefix in _PRIVATE_PREFIXES:
                if hostname.startswith(prefix):
                    return False

        return True

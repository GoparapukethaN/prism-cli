"""Lightweight documentation fetcher -- fast, no browser needed."""

from __future__ import annotations

import html
import re
from typing import Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

logger = structlog.get_logger(__name__)


class FetchDocsTool(Tool):
    """Lightweight fast documentation fetch using httpx.

    Optimized for fetching documentation pages, API references,
    and technical content. Uses httpx (no browser) for speed.
    Extracts main content and removes navigation, ads, and
    boilerplate. Converts HTML to clean markdown format.
    """

    @property
    def name(self) -> str:
        return "fetch_docs"

    @property
    def description(self) -> str:
        return (
            "Fetch documentation or technical content from "
            "a URL. Fast, lightweight, no browser needed."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "The documentation URL to fetch"
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Request timeout in seconds"
                    ),
                    "default": 15,
                },
                "max_length": {
                    "type": "integer",
                    "description": (
                        "Maximum content length in "
                        "characters"
                    ),
                    "default": 32000,
                },
            },
            "required": ["url"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Fetch and extract documentation content.

        Args:
            arguments: Must contain 'url' string.

        Returns:
            ToolResult with extracted documentation text.
        """
        url = arguments.get("url", "").strip()
        if not url:
            return ToolResult(
                success=False,
                output="",
                error="URL is required",
            )

        timeout = arguments.get("timeout", 15)
        max_length = arguments.get("max_length", 32000)

        # URL safety check
        from prism.tools.browser import BrowseWebTool

        checker = BrowseWebTool()
        if not checker._is_safe_url(url):
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"URL rejected by security policy: {url}"
                ),
            )

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="httpx is required for fetch_docs",
            )

        try:
            import random

            from prism.tools.browser import (
                _RATE_LIMITER,
                _USER_AGENTS,
            )

            _RATE_LIMITER.wait_if_needed(url)
            user_agent = random.choice(  # noqa: S311
                _USER_AGENTS
            )
        except ImportError:
            user_agent = (
                "Mozilla/5.0 (compatible; Prism/1.0)"
            )

        try:
            with httpx.Client(
                timeout=float(timeout),
                follow_redirects=True,
                headers={"User-Agent": user_agent},
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()

            content_type = resp.headers.get(
                "content-type", ""
            )

            if "html" in content_type.lower():
                text = self._extract_doc_content(resp.text)
            else:
                text = resp.text

            # Truncate if needed
            if len(text) > max_length:
                text = (
                    text[:max_length]
                    + f"\n\n[... truncated at "
                    f"{max_length} chars ...]"
                )

            return ToolResult(
                success=True,
                output=text,
                metadata={
                    "url": str(resp.url),
                    "status_code": resp.status_code,
                    "content_type": content_type,
                    "content_length": len(text),
                },
            )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Request timed out after "
                    f"{timeout}s: {url}"
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
            logger.warning(
                "fetch_docs_failed",
                url=url,
                error=str(exc),
            )
            return ToolResult(
                success=False,
                output="",
                error=f"Fetch failed: {exc}",
            )

    @staticmethod
    def _extract_doc_content(raw_html: str) -> str:
        """Extract documentation content from HTML.

        Converts HTML to clean markdown format. Targets
        documentation structures (main, article, .content,
        .documentation, #content, etc.).

        Args:
            raw_html: Raw HTML string.

        Returns:
            Clean markdown-formatted text content.
        """
        # Try to find main documentation content area
        main_patterns = [
            re.compile(
                r"<main[^>]*>(.*?)</main>",
                re.DOTALL | re.IGNORECASE,
            ),
            re.compile(
                r"<article[^>]*>(.*?)</article>",
                re.DOTALL | re.IGNORECASE,
            ),
            re.compile(
                r'<div[^>]*(?:class|id)=["\'][^"\']*'
                r"(?:content|docs|documentation|"
                r"main-content|article)"
                r'[^"\']*["\'][^>]*>(.*?)</div>',
                re.DOTALL | re.IGNORECASE,
            ),
        ]

        content = raw_html
        for pattern in main_patterns:
            match = pattern.search(raw_html)
            if match:
                content = match.group(1)
                break

        # Remove unwanted elements
        removals = [
            r"<script[^>]*>.*?</script>",
            r"<style[^>]*>.*?</style>",
            r"<nav[^>]*>.*?</nav>",
            r"<footer[^>]*>.*?</footer>",
            r"<header[^>]*>.*?</header>",
            r"<aside[^>]*>.*?</aside>",
            r"<noscript[^>]*>.*?</noscript>",
            r"<iframe[^>]*>.*?</iframe>",
            r"<svg[^>]*>.*?</svg>",
            r"<!--.*?-->",
        ]

        for pat in removals:
            content = re.sub(
                pat, " ", content,
                flags=re.DOTALL | re.IGNORECASE,
            )

        # Convert code blocks with language detection
        # <pre><code class="language-python">...</code></pre>
        def _replace_code_block(
            match: re.Match[str],
        ) -> str:
            lang_attr = match.group(1) or ""
            # Extract language from class attribute
            lang = ""
            lang_match = re.search(
                r"(?:language-|lang-)(\w+)", lang_attr
            )
            if lang_match:
                lang = lang_match.group(1)
            code = match.group(2)
            code = re.sub(r"<[^>]+>", "", code)
            code = html.unescape(code)
            return f"\n```{lang}\n{code}\n```\n"

        content = re.sub(
            r"<pre[^>]*>\s*<code([^>]*)>(.*?)"
            r"</code>\s*</pre>",
            _replace_code_block,
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        content = re.sub(
            r"<pre[^>]*>(.*?)</pre>",
            r"\n```\n\1\n```\n",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert inline code: <code>...</code> -> `...`
        def _replace_inline_code(
            match: re.Match[str],
        ) -> str:
            code = match.group(1)
            code = re.sub(r"<[^>]+>", "", code)
            code = html.unescape(code)
            return f"`{code}`"

        content = re.sub(
            r"<code[^>]*>(.*?)</code>",
            _replace_inline_code,
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert headers to markdown: <h1> -> #, etc.
        for level in range(1, 7):
            content = re.sub(
                rf"<h{level}[^>]*>(.*?)</h{level}>",
                rf"\n{'#' * level} \1\n",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )

        # Convert links: <a href="...">text</a> -> [text](href)
        def _replace_link(match: re.Match[str]) -> str:
            href = match.group(1)
            text = match.group(2)
            text = re.sub(r"<[^>]+>", "", text).strip()
            text = html.unescape(text)
            if not text:
                return ""
            return f"[{text}]({href})"

        content = re.sub(
            r'<a[^>]+href=["\']([^"\']*)["\'][^>]*>'
            r"(.*?)</a>",
            _replace_link,
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert bold: <strong>/<b> -> **...**
        content = re.sub(
            r"<(?:strong|b)[^>]*>(.*?)</(?:strong|b)>",
            r"**\1**",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert italic: <em>/<i> -> *...*
        content = re.sub(
            r"<(?:em|i)[^>]*>(.*?)</(?:em|i)>",
            r"*\1*",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert ordered lists: <ol><li> -> 1. item
        # First handle <ol> blocks to number items
        def _replace_ol(match: re.Match[str]) -> str:
            ol_content = match.group(1)
            items = re.findall(
                r"<li[^>]*>(.*?)</li>",
                ol_content,
                re.DOTALL | re.IGNORECASE,
            )
            lines = []
            for idx, item in enumerate(items, 1):
                item_text = re.sub(
                    r"<[^>]+>", "", item
                ).strip()
                item_text = html.unescape(item_text)
                lines.append(f"{idx}. {item_text}")
            return "\n" + "\n".join(lines) + "\n"

        content = re.sub(
            r"<ol[^>]*>(.*?)</ol>",
            _replace_ol,
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert unordered lists: <ul><li> -> - item
        def _replace_ul(match: re.Match[str]) -> str:
            ul_content = match.group(1)
            items = re.findall(
                r"<li[^>]*>(.*?)</li>",
                ul_content,
                re.DOTALL | re.IGNORECASE,
            )
            lines = []
            for item in items:
                item_text = re.sub(
                    r"<[^>]+>", "", item
                ).strip()
                item_text = html.unescape(item_text)
                lines.append(f"- {item_text}")
            return "\n" + "\n".join(lines) + "\n"

        content = re.sub(
            r"<ul[^>]*>(.*?)</ul>",
            _replace_ul,
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert remaining <li> not inside <ol>/<ul>
        content = re.sub(
            r"<li[^>]*>(.*?)</li>",
            r"\n- \1",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Strip remaining tags
        content = re.sub(r"<[^>]+>", " ", content)
        content = html.unescape(content)

        # Clean whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r" {2,}", " ", content)

        return content.strip()

"""DuckDuckGo web search tool -- returns top results without browser."""

from __future__ import annotations

import html
import re
from typing import Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

logger = structlog.get_logger(__name__)

# DuckDuckGo HTML search URL (no API key needed)
_DDG_URL = "https://html.duckduckgo.com/html/"


class SearchWebTool(Tool):
    """Search the web using DuckDuckGo and return top results.

    Uses the DuckDuckGo HTML endpoint which requires no API key.
    Returns up to 5 results with title, URL, and snippet.
    Optionally fetches the content of the top result.
    """

    @property
    def name(self) -> str:
        return "search_web"

    @property
    def description(self) -> str:
        return (
            "Search the web using DuckDuckGo. Returns top 5 "
            "results with title, URL, and snippet."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        "Maximum number of results (1-10)"
                    ),
                    "default": 5,
                },
                "follow_first": {
                    "type": "boolean",
                    "description": (
                        "Fetch content of top result"
                    ),
                    "default": False,
                },
            },
            "required": ["query"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.CONFIRM

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute a web search.

        Args:
            arguments: Must contain 'query' string. Optional
                'max_results' (int) and 'follow_first' (bool).

        Returns:
            ToolResult with search results formatted as text.
        """
        query = arguments.get("query", "").strip()
        if not query:
            return ToolResult(
                success=False,
                output="",
                error="Search query is required",
            )

        max_results = min(
            max(1, arguments.get("max_results", 5)), 10
        )
        follow_first: bool = arguments.get(
            "follow_first", False
        )

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                output="",
                error="httpx is required for web search",
            )

        try:
            import random

            from prism.tools.browser import (
                _RATE_LIMITER,
                _USER_AGENTS,
            )

            _RATE_LIMITER.wait_if_needed(_DDG_URL)
            user_agent = random.choice(  # noqa: S311
                _USER_AGENTS
            )
        except ImportError:
            user_agent = (
                "Mozilla/5.0 (compatible; Prism/1.0)"
            )

        try:
            with httpx.Client(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
            ) as client:
                resp = client.post(
                    _DDG_URL,
                    data={"q": query, "b": ""},
                )
                resp.raise_for_status()

            results = self._parse_ddg_html(
                resp.text, max_results
            )

            if not results:
                return ToolResult(
                    success=True,
                    output=f"No results found for: {query}",
                    metadata={
                        "query": query,
                        "result_count": 0,
                    },
                )

            formatted = self._format_results(query, results)

            # Optionally follow the first result URL
            if follow_first and results:
                first_url = results[0].get("url", "")
                if first_url:
                    fetched = self._fetch_first_result(
                        first_url, user_agent
                    )
                    if fetched:
                        formatted += (
                            "\n--- Content from top "
                            "result ---\n"
                            f"{fetched}"
                        )

            return ToolResult(
                success=True,
                output=formatted,
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "followed_first": follow_first,
                },
            )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="",
                error=f"Search timed out for: {query}",
            )
        except Exception as exc:
            logger.warning(
                "search_failed",
                query=query,
                error=str(exc),
            )
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {exc}",
            )

    @staticmethod
    def _fetch_first_result(
        url: str, user_agent: str
    ) -> str:
        """Fetch content from the first search result URL.

        Args:
            url: The URL to fetch.
            user_agent: User-Agent header value.

        Returns:
            Extracted text content, or empty string on
            failure.
        """
        try:
            import httpx as _httpx

            from prism.tools.browser import BrowseWebTool

            checker = BrowseWebTool()
            if not checker._is_safe_url(url):
                return ""

            with _httpx.Client(
                timeout=10.0,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()

            content_type = resp.headers.get(
                "content-type", ""
            )
            if "html" in content_type.lower():
                extracted = checker._extract_content(
                    resp.text
                )
            else:
                extracted = resp.text

            # Truncate to 4000 chars for first-result preview
            if len(extracted) > 4000:
                extracted = (
                    extracted[:4000]
                    + "\n\n[... truncated ...]"
                )
            return extracted
        except Exception:
            return ""

    @staticmethod
    def _parse_ddg_html(
        html_content: str, max_results: int
    ) -> list[dict[str, str]]:
        """Parse DuckDuckGo HTML search results.

        Args:
            html_content: Raw HTML from DuckDuckGo.
            max_results: Maximum results to extract.

        Returns:
            List of dicts with 'title', 'url', 'snippet'.
        """
        results: list[dict[str, str]] = []

        # DuckDuckGo uses class="result__a" for links
        link_pattern = re.compile(
            r'class="result__a"[^>]*href="([^"]*)"'
            r"[^>]*>(.*?)</a>",
            re.DOTALL | re.IGNORECASE,
        )
        snippet_pattern = re.compile(
            r'class="result__snippet"[^>]*>'
            r"(.*?)</(?:a|td|div|span)",
            re.DOTALL | re.IGNORECASE,
        )

        links = link_pattern.findall(html_content)
        snippets = snippet_pattern.findall(html_content)

        for i, (url, title_html) in enumerate(
            links[:max_results]
        ):
            title = re.sub(
                r"<[^>]+>", "", title_html
            ).strip()
            title = html.unescape(title)

            # Clean URL (DuckDuckGo wraps URLs)
            clean_url = url
            if "uddg=" in url:
                import urllib.parse

                parsed = urllib.parse.parse_qs(
                    urllib.parse.urlparse(url).query
                )
                if "uddg" in parsed:
                    clean_url = parsed["uddg"][0]

            snippet = ""
            if i < len(snippets):
                snippet = re.sub(
                    r"<[^>]+>", "", snippets[i]
                ).strip()
                snippet = html.unescape(snippet)

            if title and clean_url:
                results.append({
                    "title": title,
                    "url": clean_url,
                    "snippet": snippet,
                })

        return results

    @staticmethod
    def _format_results(
        query: str, results: list[dict[str, str]]
    ) -> str:
        """Format search results as readable text.

        Args:
            query: Original search query.
            results: Parsed search results.

        Returns:
            Formatted string with numbered results.
        """
        lines = [f'Search results for: "{query}"\n']

        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['url']}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet']}")
            lines.append("")

        return "\n".join(lines)

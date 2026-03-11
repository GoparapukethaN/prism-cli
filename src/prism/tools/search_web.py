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
    """

    @property
    def name(self) -> str:
        return "search_web"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo. Returns top 5 results with title, URL, and snippet."

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
                    "description": "Maximum number of results (1-10)",
                    "default": 5,
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
            arguments: Must contain 'query' string.

        Returns:
            ToolResult with search results formatted as text.
        """
        query = arguments.get("query", "").strip()
        if not query:
            return ToolResult(success=False, output="", error="Search query is required")

        max_results = min(max(1, arguments.get("max_results", 5)), 10)

        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, output="", error="httpx is required for web search")

        try:
            import random

            from prism.tools.browser import _RATE_LIMITER, _USER_AGENTS

            _RATE_LIMITER.wait_if_needed(_DDG_URL)
            user_agent = random.choice(_USER_AGENTS)  # noqa: S311
        except ImportError:
            user_agent = "Mozilla/5.0 (compatible; Prism/1.0)"

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

            results = self._parse_ddg_html(resp.text, max_results)

            if not results:
                return ToolResult(
                    success=True,
                    output=f"No results found for: {query}",
                    metadata={"query": query, "result_count": 0},
                )

            formatted = self._format_results(query, results)
            return ToolResult(
                success=True,
                output=formatted,
                metadata={
                    "query": query,
                    "result_count": len(results),
                },
            )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                output="",
                error=f"Search timed out for: {query}",
            )
        except Exception as exc:
            logger.warning("search_failed", query=query, error=str(exc))
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {exc}",
            )

    @staticmethod
    def _parse_ddg_html(html_content: str, max_results: int) -> list[dict[str, str]]:
        """Parse DuckDuckGo HTML search results.

        Args:
            html_content: Raw HTML from DuckDuckGo.
            max_results: Maximum results to extract.

        Returns:
            List of dicts with 'title', 'url', 'snippet' keys.
        """
        results: list[dict[str, str]] = []

        # Find result blocks -- DuckDuckGo uses class="result__a" for links
        link_pattern = re.compile(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        snippet_pattern = re.compile(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div|span)',
            re.DOTALL | re.IGNORECASE,
        )

        links = link_pattern.findall(html_content)
        snippets = snippet_pattern.findall(html_content)

        for i, (url, title_html) in enumerate(links[:max_results]):
            title = re.sub(r"<[^>]+>", "", title_html).strip()
            title = html.unescape(title)

            # Clean URL (DuckDuckGo wraps URLs)
            clean_url = url
            if "uddg=" in url:
                import urllib.parse

                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                if "uddg" in parsed:
                    clean_url = parsed["uddg"][0]

            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                snippet = html.unescape(snippet)

            if title and clean_url:
                results.append({
                    "title": title,
                    "url": clean_url,
                    "snippet": snippet,
                })

        return results

    @staticmethod
    def _format_results(query: str, results: list[dict[str, str]]) -> str:
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

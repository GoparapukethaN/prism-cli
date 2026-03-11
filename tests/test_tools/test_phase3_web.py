"""Tests for Phase 3 web browsing enhancements.

Tests cover: DomainRateLimiter, rotating user agents, content truncation,
SearchWebTool, and FetchDocsTool. All tests use mocks -- no real HTTP requests.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from prism.tools.browser import (
    _RATE_LIMITER,
    _USER_AGENTS,
    BrowseWebTool,
    DomainRateLimiter,
)

# ======================================================================
# Domain Rate Limiter
# ======================================================================


class TestDomainRateLimiter:
    """Test per-domain rate limiting."""

    def test_first_request_no_delay(self) -> None:
        limiter = DomainRateLimiter(min_delay=1.0)
        start = time.monotonic()
        limiter.wait_if_needed("https://example.com/page1")
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # No delay for first request

    def test_second_request_delayed(self) -> None:
        limiter = DomainRateLimiter(min_delay=0.2)
        limiter.wait_if_needed("https://example.com/page1")
        start = time.monotonic()
        limiter.wait_if_needed("https://example.com/page2")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15  # Should wait ~0.2s

    def test_different_domains_no_delay(self) -> None:
        limiter = DomainRateLimiter(min_delay=1.0)
        limiter.wait_if_needed("https://example.com/page1")
        start = time.monotonic()
        limiter.wait_if_needed("https://other.com/page1")
        elapsed = time.monotonic() - start
        assert elapsed < 0.1  # Different domain, no delay

    def test_extract_domain(self) -> None:
        assert DomainRateLimiter._extract_domain("https://example.com/path") == "example.com"
        assert (
            DomainRateLimiter._extract_domain("https://Sub.Example.COM/") == "sub.example.com"
        )
        # Invalid URL returns "unknown" per implementation
        assert DomainRateLimiter._extract_domain("invalid") == ""

    def test_module_level_limiter_exists(self) -> None:
        assert _RATE_LIMITER is not None
        assert isinstance(_RATE_LIMITER, DomainRateLimiter)


# ======================================================================
# Rotating User Agents
# ======================================================================


class TestUserAgents:
    """Test rotating user agent list."""

    def test_multiple_agents_available(self) -> None:
        assert len(_USER_AGENTS) >= 4

    def test_all_agents_are_strings(self) -> None:
        for agent in _USER_AGENTS:
            assert isinstance(agent, str)
            assert len(agent) > 20

    def test_agents_contain_mozilla(self) -> None:
        for agent in _USER_AGENTS:
            assert "Mozilla" in agent


# ======================================================================
# Content Truncation
# ======================================================================


class TestContentTruncation:
    """Test intelligent content truncation."""

    def test_short_content_unchanged(self) -> None:
        text = "Short content"
        result = BrowseWebTool._truncate_content(text, max_tokens=8000)
        assert result == text

    def test_long_content_truncated(self) -> None:
        text = "A" * 100_000
        result = BrowseWebTool._truncate_content(text, max_tokens=1000)
        assert len(result) < 100_000
        assert "truncated" in result

    def test_truncated_preserves_head_and_tail(self) -> None:
        text = "HEAD_MARKER " + "X" * 50_000 + " TAIL_MARKER"
        result = BrowseWebTool._truncate_content(text, max_tokens=500)
        assert "HEAD_MARKER" in result
        assert "TAIL_MARKER" in result

    def test_empty_content(self) -> None:
        assert BrowseWebTool._truncate_content("") == ""
        assert BrowseWebTool._truncate_content("", max_tokens=100) == ""


# ======================================================================
# SearchWebTool
# ======================================================================


class TestSearchWebTool:
    """Test DuckDuckGo search tool."""

    def test_tool_properties(self) -> None:
        from prism.tools.search_web import SearchWebTool

        tool = SearchWebTool()
        assert tool.name == "search_web"
        assert "query" in tool.parameters_schema["properties"]

    def test_empty_query_error(self) -> None:
        from prism.tools.search_web import SearchWebTool

        tool = SearchWebTool()
        result = tool.execute({"query": ""})
        assert not result.success
        assert "required" in result.error.lower()

    def test_search_with_mock_response(self) -> None:
        from prism.tools.search_web import SearchWebTool

        tool = SearchWebTool()

        mock_html = """
        <div class="result">
            <a class="result__a" href="https://example.com">Example Title</a>
            <td class="result__snippet">This is a snippet about example.</td>
        </div>
        <div class="result">
            <a class="result__a" href="https://other.com">Other Title</a>
            <td class="result__snippet">Another snippet.</td>
        </div>
        """

        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            with patch("prism.tools.browser._RATE_LIMITER"):
                result = tool.execute({"query": "python testing"})

        assert result.success
        assert "query" in result.metadata

    def test_timeout_handled(self) -> None:
        import httpx

        from prism.tools.search_web import SearchWebTool

        tool = SearchWebTool()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client

            with patch("prism.tools.browser._RATE_LIMITER"):
                result = tool.execute({"query": "test"})

        assert not result.success
        assert "timed out" in result.error.lower()

    def test_parse_ddg_html(self) -> None:
        from prism.tools.search_web import SearchWebTool

        tool = SearchWebTool()
        html_content = """
        <a class="result__a" href="https://example.com/page">Test Title</a>
        <td class="result__snippet">Test snippet content.</td>
        """
        results = tool._parse_ddg_html(html_content, 5)
        assert len(results) >= 1
        assert results[0]["title"] == "Test Title"
        assert results[0]["url"] == "https://example.com/page"

    def test_format_results(self) -> None:
        from prism.tools.search_web import SearchWebTool

        tool = SearchWebTool()
        results = [
            {"title": "Title 1", "url": "https://example.com", "snippet": "Snippet 1"},
            {"title": "Title 2", "url": "https://other.com", "snippet": ""},
        ]
        formatted = tool._format_results("test query", results)
        assert "test query" in formatted
        assert "Title 1" in formatted
        assert "https://example.com" in formatted


# ======================================================================
# FetchDocsTool
# ======================================================================


class TestFetchDocsTool:
    """Test lightweight documentation fetcher."""

    def test_tool_properties(self) -> None:
        from prism.tools.fetch_docs import FetchDocsTool

        tool = FetchDocsTool()
        assert tool.name == "fetch_docs"
        assert "url" in tool.parameters_schema["properties"]

    def test_empty_url_error(self) -> None:
        from prism.tools.fetch_docs import FetchDocsTool

        tool = FetchDocsTool()
        result = tool.execute({"url": ""})
        assert not result.success
        assert "required" in result.error.lower()

    def test_unsafe_url_rejected(self) -> None:
        from prism.tools.fetch_docs import FetchDocsTool

        tool = FetchDocsTool()
        result = tool.execute({"url": "file:///etc/passwd"})
        assert not result.success
        assert "rejected" in result.error.lower()

    def test_fetch_with_mock_html(self) -> None:
        from prism.tools.fetch_docs import FetchDocsTool

        tool = FetchDocsTool()

        mock_html = """
        <html>
        <head><title>Docs</title></head>
        <body>
        <nav>Navigation</nav>
        <main>
            <h1>API Reference</h1>
            <p>This is the documentation content.</p>
            <pre><code>print("hello")</code></pre>
        </main>
        <footer>Footer</footer>
        </body>
        </html>
        """

        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.status_code = 200
        mock_response.url = "https://docs.example.com/api"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            with patch("prism.tools.browser._RATE_LIMITER"):
                result = tool.execute({"url": "https://docs.example.com/api"})

        assert result.success
        assert "API Reference" in result.output or "documentation" in result.output.lower()

    def test_content_truncated_at_max_length(self) -> None:
        from prism.tools.fetch_docs import FetchDocsTool

        tool = FetchDocsTool()

        mock_response = MagicMock()
        mock_response.text = "X" * 100_000  # Plain text, not HTML
        mock_response.status_code = 200
        mock_response.url = "https://example.com/big"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            with patch("prism.tools.browser._RATE_LIMITER"):
                result = tool.execute(
                    {"url": "https://example.com/big", "max_length": 1000}
                )

        assert result.success
        # Content is truncated to max_length + truncation suffix
        assert len(result.output) <= 1100

    def test_extract_doc_content_main_tag(self) -> None:
        from prism.tools.fetch_docs import FetchDocsTool

        tool = FetchDocsTool()
        html_content = """
        <nav>Nav stuff</nav>
        <main><h1>Title</h1><p>Doc content here.</p></main>
        <footer>Footer</footer>
        """
        result = tool._extract_doc_content(html_content)
        assert "Title" in result or "Doc content" in result
        assert "Nav stuff" not in result

    def test_extract_preserves_code_blocks(self) -> None:
        from prism.tools.fetch_docs import FetchDocsTool

        tool = FetchDocsTool()
        html_content = '<main><pre><code>def hello():\n    pass</code></pre></main>'
        result = tool._extract_doc_content(html_content)
        assert "def hello" in result
        assert "```" in result  # Should be converted to markdown code block

    def test_timeout_handled(self) -> None:
        import httpx

        from prism.tools.fetch_docs import FetchDocsTool

        tool = FetchDocsTool()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.TimeoutException("timed out")
            mock_client_cls.return_value = mock_client

            with patch("prism.tools.browser._RATE_LIMITER"):
                result = tool.execute({"url": "https://example.com"})

        assert not result.success
        assert "timed out" in result.error.lower()

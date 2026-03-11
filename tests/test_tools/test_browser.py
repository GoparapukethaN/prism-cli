"""Tests for BrowseWebTool -- all network calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from prism.tools.base import PermissionLevel
from prism.tools.browser import BrowseWebTool


@pytest.fixture
def tool() -> BrowseWebTool:
    """Create a BrowseWebTool instance."""
    return BrowseWebTool()


# =====================================================================
# Tool properties
# =====================================================================


class TestToolProperties:
    def test_tool_name(self, tool: BrowseWebTool) -> None:
        assert tool.name == "browse_web"

    def test_tool_permission(self, tool: BrowseWebTool) -> None:
        assert tool.permission_level == PermissionLevel.CONFIRM

    def test_tool_description(self, tool: BrowseWebTool) -> None:
        assert "web page" in tool.description.lower()

    def test_tool_parameters_schema(self, tool: BrowseWebTool) -> None:
        schema = tool.parameters_schema
        assert "url" in schema["properties"]
        assert "url" in schema["required"]


# =====================================================================
# URL safety
# =====================================================================


class TestUrlSafety:
    def test_safe_https_url(self, tool: BrowseWebTool) -> None:
        assert tool._is_safe_url("https://example.com") is True

    def test_safe_http_url(self, tool: BrowseWebTool) -> None:
        assert tool._is_safe_url("http://example.com") is True

    def test_unsafe_file_url(self, tool: BrowseWebTool) -> None:
        assert tool._is_safe_url("file:///etc/passwd") is False

    def test_unsafe_localhost(self, tool: BrowseWebTool) -> None:
        assert tool._is_safe_url("http://localhost:8080") is False

    def test_safe_localhost_ollama(self, tool: BrowseWebTool) -> None:
        assert tool._is_safe_url("http://localhost:11434") is True

    def test_unsafe_private_ip(self, tool: BrowseWebTool) -> None:
        assert tool._is_safe_url("http://192.168.1.1") is False

    def test_unsafe_loopback(self, tool: BrowseWebTool) -> None:
        assert tool._is_safe_url("http://127.0.0.1") is False

    def test_unsafe_ftp_scheme(self, tool: BrowseWebTool) -> None:
        assert tool._is_safe_url("ftp://example.com/file") is False

    def test_unsafe_10_prefix(self, tool: BrowseWebTool) -> None:
        assert tool._is_safe_url("http://10.0.0.1") is False

    def test_rejected_url_returns_error(self, tool: BrowseWebTool) -> None:
        result = tool.execute({"url": "file:///etc/passwd"})
        assert result.success is False
        assert "rejected" in (result.error or "").lower()


# =====================================================================
# httpx fetch (default mode)
# =====================================================================


class TestHttpxFetch:
    def test_httpx_fetch_success(self, tool: BrowseWebTool) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.text = "<html><body><p>Hello World</p></body></html>"
        mock_response.url = "https://example.com"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("prism.tools.browser.httpx") as mock_httpx:
            mock_httpx.Client.return_value = mock_client
            mock_httpx.TimeoutException = httpx.TimeoutException
            mock_httpx.HTTPStatusError = httpx.HTTPStatusError

            result = tool.execute({"url": "https://example.com"})

        assert result.success is True
        assert "Hello World" in result.output

    def test_httpx_fetch_timeout(self, tool: BrowseWebTool) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.TimeoutException("timeout")

        with patch("prism.tools.browser.httpx") as mock_httpx:
            mock_httpx.Client.return_value = mock_client
            mock_httpx.TimeoutException = httpx.TimeoutException
            mock_httpx.HTTPStatusError = httpx.HTTPStatusError

            result = tool.execute({"url": "https://example.com"})

        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    def test_default_uses_httpx_not_playwright(
        self, tool: BrowseWebTool
    ) -> None:
        """When use_browser is False (default), httpx should be used."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.text = "plain text content"
        mock_response.url = "https://example.com"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("prism.tools.browser.httpx") as mock_httpx:
            mock_httpx.Client.return_value = mock_client
            mock_httpx.TimeoutException = httpx.TimeoutException
            mock_httpx.HTTPStatusError = httpx.HTTPStatusError

            result = tool.execute({"url": "https://example.com"})

        assert result.success is True
        # httpx was used
        mock_httpx.Client.assert_called_once()


# =====================================================================
# Content extraction
# =====================================================================


class TestContentExtraction:
    def test_content_extraction_strips_scripts(
        self, tool: BrowseWebTool
    ) -> None:
        raw_html = (
            "<html><head><script>alert('x')</script></head>"
            "<body><p>Real content</p></body></html>"
        )
        extracted = tool._extract_content(raw_html)
        assert "Real content" in extracted
        assert "alert" not in extracted

    def test_content_extraction_strips_style(
        self, tool: BrowseWebTool
    ) -> None:
        raw_html = (
            "<html><head><style>.x{color:red}</style></head>"
            "<body><p>Visible text</p></body></html>"
        )
        extracted = tool._extract_content(raw_html)
        assert "Visible text" in extracted
        assert "color:red" not in extracted

    def test_content_extraction_strips_nav(
        self, tool: BrowseWebTool
    ) -> None:
        raw_html = (
            "<html><body>"
            "<nav>Menu Link1 Link2</nav>"
            "<main><p>Main content</p></main>"
            "</body></html>"
        )
        extracted = tool._extract_content(raw_html)
        assert "Main content" in extracted
        assert "Menu Link1" not in extracted

    def test_content_extraction_handles_entities(
        self, tool: BrowseWebTool
    ) -> None:
        raw_html = "<p>Hello &amp; goodbye &lt;world&gt;</p>"
        extracted = tool._extract_content(raw_html)
        assert "Hello & goodbye <world>" in extracted


# =====================================================================
# Validation
# =====================================================================


class TestValidation:
    def test_missing_url_raises(self, tool: BrowseWebTool) -> None:
        with pytest.raises(ValueError, match="Missing required"):
            tool.execute({})

    def test_screenshot_not_available_without_playwright(
        self, tool: BrowseWebTool
    ) -> None:
        """When screenshot=True and Playwright is unavailable, return error."""
        with patch(
            "prism.tools.browser.BrowseWebTool._fetch_with_playwright"
        ) as mock_pw:
            mock_pw.return_value = MagicMock(
                success=False,
                output="",
                error="Playwright is not installed.",
            )
            result = tool.execute({
                "url": "https://example.com",
                "screenshot": True,
            })
            assert result.success is False
            assert "playwright" in (result.error or "").lower()

"""Tests for ScreenshotTool -- all Playwright calls are mocked."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from prism.tools.base import PermissionLevel
from prism.tools.screenshot import ScreenshotTool


@pytest.fixture
def tool() -> ScreenshotTool:
    """Create a ScreenshotTool instance."""
    return ScreenshotTool()


# =====================================================================
# Tool properties
# =====================================================================


class TestToolProperties:
    def test_tool_name(self, tool: ScreenshotTool) -> None:
        assert tool.name == "screenshot"

    def test_tool_permission(self, tool: ScreenshotTool) -> None:
        assert tool.permission_level == PermissionLevel.CONFIRM

    def test_tool_description(self, tool: ScreenshotTool) -> None:
        assert "screenshot" in tool.description.lower()

    def test_tool_parameters_schema(self, tool: ScreenshotTool) -> None:
        schema = tool.parameters_schema
        assert "url" in schema["properties"]
        assert "url" in schema["required"]
        assert "full_page" in schema["properties"]
        assert "width" in schema["properties"]
        assert "height" in schema["properties"]


# =====================================================================
# Playwright not installed
# =====================================================================


class TestPlaywrightNotInstalled:
    def test_playwright_not_installed(self, tool: ScreenshotTool) -> None:
        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            side_effect=ImportError("No module named 'playwright'"),
        ):
            result = tool.execute({"url": "https://example.com"})
            assert result.success is False
            assert "playwright" in (result.error or "").lower()


# =====================================================================
# Successful screenshot (mocked Playwright)
# =====================================================================


def _build_mock_playwright(
    *,
    screenshot_bytes: bytes | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Build a complete mock playwright context manager.

    Returns:
        (sync_playwright_func, mock_page, mock_browser)
    """
    if screenshot_bytes is None:
        screenshot_bytes = b"\x89PNG\r\n\x1a\nfake_image_data"

    mock_page = MagicMock()
    mock_page.screenshot.return_value = screenshot_bytes
    mock_page.goto = MagicMock()

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

    # The sync_playwright() function returns a context manager
    sync_playwright_func = MagicMock(return_value=mock_context)

    return sync_playwright_func, mock_page, mock_browser


class TestScreenshotCapture:
    def test_screenshot_returns_base64(self, tool: ScreenshotTool) -> None:
        fake_png = b"\x89PNG\r\n\x1a\nfake_screenshot_data"
        sync_pw_func, _mock_page, _mock_browser = _build_mock_playwright(
            screenshot_bytes=fake_png
        )

        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            return_value=sync_pw_func,
        ):
            result = tool.execute({"url": "https://example.com"})

        assert result.success is True
        # Output should be valid base64
        decoded = base64.b64decode(result.output)
        assert decoded == fake_png
        assert result.metadata is not None
        assert result.metadata["format"] == "png"
        assert result.metadata["url"] == "https://example.com"

    def test_full_page_option(self, tool: ScreenshotTool) -> None:
        sync_pw_func, mock_page, _ = _build_mock_playwright()

        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            return_value=sync_pw_func,
        ):
            result = tool.execute({
                "url": "https://example.com",
                "full_page": False,
            })

        assert result.success is True
        mock_page.screenshot.assert_called_once_with(full_page=False)

    def test_custom_dimensions(self, tool: ScreenshotTool) -> None:
        sync_pw_func, _mock_page, mock_browser = _build_mock_playwright()

        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            return_value=sync_pw_func,
        ):
            result = tool.execute({
                "url": "https://example.com",
                "width": 800,
                "height": 600,
            })

        assert result.success is True
        mock_browser.new_page.assert_called_once_with(
            viewport={"width": 800, "height": 600}
        )
        assert result.metadata is not None
        assert result.metadata["width"] == 800
        assert result.metadata["height"] == 600

    def test_default_dimensions(self, tool: ScreenshotTool) -> None:
        sync_pw_func, _mock_page, mock_browser = _build_mock_playwright()

        with patch(
            "prism.tools.screenshot._get_sync_playwright",
            return_value=sync_pw_func,
        ):
            result = tool.execute({"url": "https://example.com"})

        assert result.success is True
        mock_browser.new_page.assert_called_once_with(
            viewport={"width": 1280, "height": 720}
        )
        assert result.metadata is not None
        assert result.metadata["width"] == 1280
        assert result.metadata["height"] == 720


# =====================================================================
# URL safety
# =====================================================================


class TestUrlSafety:
    def test_unsafe_url_rejected(self, tool: ScreenshotTool) -> None:
        result = tool.execute({"url": "file:///etc/passwd"})
        assert result.success is False
        assert "rejected" in (result.error or "").lower()

    def test_private_ip_rejected(self, tool: ScreenshotTool) -> None:
        result = tool.execute({"url": "http://192.168.1.1"})
        assert result.success is False
        assert "rejected" in (result.error or "").lower()

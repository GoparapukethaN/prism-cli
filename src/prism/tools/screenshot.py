"""Screenshot capture tool -- renders a URL and returns a base64 PNG.

Requires Playwright (optional dependency).  The browser is lazy-loaded
so that importing this module never triggers a heavy import.

Intended for multimodal models that can interpret images.
"""

from __future__ import annotations

import base64
from typing import Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult
from prism.tools.browser import BrowseWebTool

logger = structlog.get_logger(__name__)

# Re-use the URL safety check from BrowseWebTool.
_URL_CHECKER = BrowseWebTool()


def _get_sync_playwright() -> Any:
    """Import and return ``sync_playwright`` from Playwright.

    Raises:
        ImportError: If Playwright is not installed.
    """
    from playwright.sync_api import sync_playwright

    return sync_playwright


class ScreenshotTool(Tool):
    """Capture a screenshot of a URL as base64-encoded PNG.

    Requires Playwright (optional dependency).  The tool will return
    a clear error message if Playwright is not installed.
    """

    @property
    def name(self) -> str:
        return "screenshot"

    @property
    def description(self) -> str:
        return (
            "Capture a screenshot of a web page. "
            "Returns a base64-encoded PNG image."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to screenshot.",
                },
                "full_page": {
                    "type": "boolean",
                    "description": "Capture the full scrollable page.",
                    "default": True,
                },
                "width": {
                    "type": "integer",
                    "description": "Viewport width in pixels.",
                    "default": 1280,
                },
                "height": {
                    "type": "integer",
                    "description": "Viewport height in pixels.",
                    "default": 720,
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
        """Capture a screenshot of the given URL.

        Args:
            arguments: Must contain ``url``; optional ``full_page``,
                       ``width``, ``height``.

        Returns:
            ToolResult with base64 PNG in ``output`` on success.
        """
        validated = self.validate_arguments(arguments)

        url: str = validated["url"]
        full_page: bool = validated.get("full_page", True)
        width: int = validated.get("width", 1280)
        height: int = validated.get("height", 720)

        # Safety check
        if not _URL_CHECKER._is_safe_url(url):
            return ToolResult(
                success=False,
                output="",
                error=f"URL rejected by security policy: {url}",
            )

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
                page = browser.new_page(
                    viewport={"width": width, "height": height}
                )
                page.goto(url, timeout=30_000)

                screenshot_bytes = page.screenshot(full_page=full_page)
                browser.close()

            encoded = base64.b64encode(screenshot_bytes).decode("utf-8")
            return ToolResult(
                success=True,
                output=encoded,
                metadata={
                    "url": url,
                    "width": width,
                    "height": height,
                    "full_page": full_page,
                    "format": "png",
                    "size_bytes": len(screenshot_bytes),
                },
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Screenshot failed: {exc}",
            )

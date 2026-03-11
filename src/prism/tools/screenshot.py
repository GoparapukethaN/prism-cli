"""Screenshot capture tool -- renders a URL and returns a base64 PNG.

Requires Playwright (optional dependency).  The browser is lazy-loaded
so that importing this module never triggers a heavy import.

Intended for multimodal models that can interpret images.
"""

from __future__ import annotations

import base64
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult
from prism.tools.browser import BrowseWebTool

logger = structlog.get_logger(__name__)

# Re-use the URL safety check from BrowseWebTool.
_URL_CHECKER = BrowseWebTool()

# Screenshot storage directory.
_SCREENSHOT_DIR = Path.home() / ".prism" / "screenshots"


def _get_sync_playwright() -> Any:
    """Import and return ``sync_playwright`` from Playwright.

    Raises:
        ImportError: If Playwright is not installed.
    """
    from playwright.sync_api import sync_playwright

    return sync_playwright


def _compress_screenshot(
    data: bytes, max_width: int = 1280
) -> bytes:
    """Re-encode screenshot at lower quality if too large.

    If *data* exceeds 1 MB, attempt to compress using ``sips``
    on macOS or return the original bytes on other platforms.

    Args:
        data: Raw PNG screenshot bytes.
        max_width: Maximum width for the resized image.

    Returns:
        Compressed image bytes, or original if compression
        is unavailable or unnecessary.
    """
    if len(data) <= 1_048_576:  # 1 MB
        return data

    if platform.system() != "Darwin":
        return data

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as src:
            src.write(data)
            src_path = Path(src.name)

        dst_path = src_path.with_suffix(".compressed.png")

        subprocess.run(
            [
                "sips",
                "--resampleWidth",
                str(max_width),
                str(src_path),
                "--out",
                str(dst_path),
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )

        if dst_path.exists() and dst_path.stat().st_size > 0:
            compressed = dst_path.read_bytes()
            dst_path.unlink(missing_ok=True)
            src_path.unlink(missing_ok=True)
            return compressed

        src_path.unlink(missing_ok=True)
        dst_path.unlink(missing_ok=True)
        return data
    except Exception:
        return data


def _save_screenshot(data: bytes, url: str) -> Path | None:
    """Save screenshot to ``~/.prism/screenshots/``.

    Args:
        data: PNG image bytes.
        url: The URL that was screenshotted (used in filename).

    Returns:
        Path to the saved file, or None on failure.
    """
    try:
        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

        import time
        from urllib.parse import urlparse

        parsed = urlparse(url)
        domain = (parsed.hostname or "unknown").replace(
            ".", "_"
        )
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{domain}_{ts}.png"
        filepath = _SCREENSHOT_DIR / filename

        filepath.write_bytes(data)
        logger.debug(
            "screenshot_saved", path=str(filepath)
        )
        return filepath
    except Exception as exc:
        logger.warning(
            "screenshot_save_failed", error=str(exc)
        )
        return None


class ScreenshotTool(Tool):
    """Capture a screenshot of a URL as base64-encoded PNG.

    Requires Playwright (optional dependency).  The tool will
    return a clear error message if Playwright is not installed.
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
                    "description": (
                        "Capture the full scrollable page."
                    ),
                    "default": True,
                },
                "width": {
                    "type": "integer",
                    "description": "Viewport width in pixels.",
                    "default": 1280,
                },
                "height": {
                    "type": "integer",
                    "description": (
                        "Viewport height in pixels."
                    ),
                    "default": 720,
                },
                "selector": {
                    "type": "string",
                    "description": (
                        "CSS selector to screenshot a "
                        "specific element."
                    ),
                },
                "viewport_only": {
                    "type": "boolean",
                    "description": (
                        "Capture only visible viewport."
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
        """Capture a screenshot of the given URL.

        Args:
            arguments: Must contain ``url``; optional
                ``full_page``, ``width``, ``height``,
                ``selector``, ``viewport_only``.

        Returns:
            ToolResult with base64 PNG in ``output`` on success.
        """
        validated = self.validate_arguments(arguments)

        url: str = validated["url"]
        full_page: bool = validated.get("full_page", True)
        width: int = validated.get("width", 1280)
        height: int = validated.get("height", 720)
        selector: str | None = validated.get("selector")
        viewport_only: bool = validated.get(
            "viewport_only", False
        )

        # viewport_only overrides full_page
        if viewport_only:
            full_page = False

        # Safety check
        if not _URL_CHECKER._is_safe_url(url):
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"URL rejected by security policy: {url}"
                ),
            )

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
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    viewport={
                        "width": width,
                        "height": height,
                    }
                )
                page.goto(url, timeout=30_000)

                if selector:
                    element = page.query_selector(selector)
                    if element is None:
                        browser.close()
                        return ToolResult(
                            success=False,
                            output="",
                            error=(
                                f"Element not found: "
                                f"{selector}"
                            ),
                        )
                    screenshot_bytes = element.screenshot()
                else:
                    screenshot_bytes = page.screenshot(
                        full_page=full_page
                    )

                browser.close()

            # Auto-compress if > 1MB
            screenshot_bytes = _compress_screenshot(
                screenshot_bytes
            )

            # Save to ~/.prism/screenshots/
            saved_path = _save_screenshot(
                screenshot_bytes, url
            )

            encoded = base64.b64encode(
                screenshot_bytes
            ).decode("utf-8")

            metadata: dict[str, Any] = {
                "url": url,
                "width": width,
                "height": height,
                "full_page": full_page,
                "format": "png",
                "size_bytes": len(screenshot_bytes),
            }
            if selector:
                metadata["selector"] = selector
            if viewport_only:
                metadata["viewport_only"] = True
            if saved_path is not None:
                metadata["saved_path"] = str(saved_path)

            return ToolResult(
                success=True,
                output=encoded,
                metadata=metadata,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Screenshot failed: {exc}",
            )

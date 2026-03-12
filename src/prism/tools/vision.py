"""Vision and multimodal input -- image processing and attachment for LLM conversations.

Provides image loading, validation, compression, base64 encoding, terminal
preview, message formatting, and a ``VisionTool`` for the tool registry.

No heavy dependencies required -- uses raw byte parsing for image dimensions
and ``sips`` (macOS) for compression.  Falls back gracefully on other platforms.
"""

from __future__ import annotations

import base64
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from prism.tools.base import PermissionLevel, Tool, ToolResult

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Vision model registry
# ---------------------------------------------------------------------------

VISION_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
    ],
    "google": [
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-pro",
        "gemini/gemini-2.0-flash",
    ],
}

ALL_VISION_MODELS: list[str] = [m for models in VISION_MODELS.values() for m in models]

# Patterns that indicate a model supports vision (case-insensitive).
_VISION_PATTERN = re.compile(
    r"vision|gpt-4o|claude-3|gemini",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_IMAGE_SIZE_BYTES: int = 5_242_880  # 5 MB (matches resize_if_needed default)
SUPPORTED_FORMATS: set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif",
}
TARGET_FORMAT: str = "JPEG"  # Default compression target

# Image magic-byte signatures for validation.
_MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # WebP starts with RIFF...WEBP
]


# ---------------------------------------------------------------------------
# ImageAttachment
# ---------------------------------------------------------------------------


class ImageAttachment:
    """Represents a processed image ready for LLM consumption.

    Attributes:
        original_path: Resolved path to the source image file.
        media_type: MIME type string (e.g. ``image/jpeg``).
        base64_data: Base64-encoded image bytes.
        width: Image width in pixels (0 if unknown).
        height: Image height in pixels (0 if unknown).
        size_bytes: Size of the (possibly compressed) payload.
        was_compressed: Whether the image was compressed to fit limits.
    """

    def __init__(
        self,
        original_path: Path,
        media_type: str,
        base64_data: str,
        width: int,
        height: int,
        size_bytes: int,
        was_compressed: bool,
    ) -> None:
        self.original_path = original_path
        self.media_type = media_type
        self.base64_data = base64_data
        self.width = width
        self.height = height
        self.size_bytes = size_bytes
        self.was_compressed = was_compressed

    def to_message_content(self, provider: str = "anthropic") -> dict:
        """Convert to the provider-specific message content dict.

        Args:
            provider: One of ``"anthropic"``, ``"openai"``, ``"google"``, etc.

        Returns:
            A dict suitable for inclusion in the ``content`` list of an
            LLM chat message.
        """
        if provider == "anthropic":
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self.media_type,
                    "data": self.base64_data,
                },
            }
        # OpenAI / Google / default format
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self.media_type};base64,{self.base64_data}",
            },
        }


# ---------------------------------------------------------------------------
# ImageProcessor
# ---------------------------------------------------------------------------


class ImageProcessor:
    """Loads, validates, and resizes images from various sources.

    Supports local file paths, remote URLs, and raw bytes.  All operations
    are synchronous; URL fetching uses ``httpx`` with a short timeout.
    """

    # MIME mapping (extension -> media type)
    _MIME_MAP: dict[str, str] = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }

    def load_from_path(self, path: str) -> tuple[str, str]:
        """Read a local image file and return its base64-encoded data.

        Args:
            path: Absolute or relative file system path.

        Returns:
            ``(base64_data, mime_type)`` tuple.

        Raises:
            ValueError: If the path does not exist, escapes the working
                directory via traversal, or is not a valid image.
        """
        resolved = Path(path).resolve()

        # Path traversal check: reject ``..`` components in the raw string.
        if ".." in Path(path).parts:
            raise ValueError(f"Path traversal detected: {path}")

        if not resolved.is_file():
            raise ValueError(f"Image file not found: {path}")

        raw_data = resolved.read_bytes()

        if not self.validate_image(raw_data):
            raise ValueError(f"File is not a valid image: {path}")

        mime_type = self._mime_from_magic(raw_data)
        if mime_type is None:
            suffix = resolved.suffix.lower()
            mime_type = self._MIME_MAP.get(suffix, "image/jpeg")

        b64 = base64.b64encode(raw_data).decode("ascii")
        return b64, mime_type

    def load_from_url(self, url: str) -> tuple[str, str]:
        """Fetch an image from a URL and return its base64-encoded data.

        Args:
            url: HTTP or HTTPS URL pointing to an image.

        Returns:
            ``(base64_data, mime_type)`` tuple.

        Raises:
            ValueError: If the URL scheme is not http(s), the fetch fails,
                or the response is not a valid image.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        try:
            with httpx.Client(timeout=30, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
        except (httpx.HTTPError, OSError) as exc:
            raise ValueError(f"Failed to fetch image from URL: {exc}") from exc

        raw_data = response.content

        if not self.validate_image(raw_data):
            raise ValueError(f"URL did not return a valid image: {url}")

        content_type = response.headers.get("content-type", "")
        mime_type = self._mime_from_content_type(content_type)
        if mime_type is None:
            mime_type = self._mime_from_magic(raw_data) or "image/jpeg"

        b64 = base64.b64encode(raw_data).decode("ascii")
        return b64, mime_type

    @staticmethod
    def validate_image(data: bytes) -> bool:
        """Check whether *data* starts with a known image magic signature.

        Args:
            data: Raw bytes to inspect.

        Returns:
            ``True`` if the first bytes match PNG, JPEG, GIF, or WebP.
        """
        if len(data) < 4:
            return False

        # PNG
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return True

        # JPEG
        if data[:3] == b"\xff\xd8\xff":
            return True

        # GIF
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return True

        # WebP: RIFF....WEBP
        return data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP"

    @staticmethod
    def resize_if_needed(
        data: bytes, max_size: int = MAX_IMAGE_SIZE_BYTES
    ) -> bytes:
        """Return *data* unchanged if within *max_size*, else truncate.

        For production use this would re-encode at a lower quality.
        This implementation returns a truncated copy as a size-reduction
        placeholder; real compression requires PIL or ``sips``.

        Args:
            data: Raw image bytes.
            max_size: Maximum allowed size in bytes (default 5 MB).

        Returns:
            The original bytes if small enough, otherwise the first
            *max_size* bytes (best-effort; caller should re-validate).
        """
        if len(data) <= max_size:
            return data
        logger.warning(
            "image_resize_truncated",
            original=len(data),
            limit=max_size,
        )
        return data[:max_size]

    # ---- private helpers ------------------------------------------------

    @staticmethod
    def _mime_from_magic(data: bytes) -> str | None:
        """Infer MIME type from magic bytes.

        Args:
            data: Raw file bytes.

        Returns:
            MIME type string or ``None``.
        """
        for sig, mime in _MAGIC_SIGNATURES:
            if data[: len(sig)] == sig:
                # WebP needs extra check
                if mime == "image/webp":
                    if len(data) >= 12 and data[8:12] == b"WEBP":
                        return mime
                    continue
                return mime
        return None

    @staticmethod
    def _mime_from_content_type(content_type: str) -> str | None:
        """Extract a recognised image MIME from a Content-Type header.

        Args:
            content_type: HTTP Content-Type value.

        Returns:
            MIME type string or ``None``.
        """
        ct_lower = content_type.lower()
        for mime in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            if mime in ct_lower:
                return mime
        return None


# ---------------------------------------------------------------------------
# build_vision_message (OpenAI-format)
# ---------------------------------------------------------------------------


def build_vision_message(
    image_data: str,
    mime_type: str,
    prompt: str,
) -> dict[str, Any]:
    """Build an OpenAI-format multimodal user message.

    Args:
        image_data: Base64-encoded image bytes.
        mime_type: MIME type (e.g. ``"image/png"``).
        prompt: Text prompt to accompany the image.

    Returns:
        A dict with ``role`` and ``content`` keys suitable for
        direct inclusion in a chat completion messages list.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}",
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    }


# ---------------------------------------------------------------------------
# VisionTool
# ---------------------------------------------------------------------------


class VisionTool(Tool):
    """Analyse an image from a local file, URL, or base64 data.

    The tool loads the image, validates it, and returns a multimodal
    message structure that the conversation engine can insert into
    the LLM chat context.

    Accepted ``source`` formats:
      - A local file path (absolute or relative).
      - An ``http://`` or ``https://`` URL.
      - A ``base64:`` prefixed string with raw base64 data.
    """

    def __init__(self) -> None:
        self._processor = ImageProcessor()

    @property
    def name(self) -> str:
        return "analyze_image"

    @property
    def description(self) -> str:
        return (
            "Analyse an image from a local file path, URL, or base64 data. "
            "Returns a multimodal message for the LLM conversation."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": (
                        "Image source: a local file path, an http(s) URL, "
                        "or 'base64:<data>'."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": "What to analyse about the image.",
                    "default": "Describe this image in detail.",
                },
            },
            "required": ["source"],
        }

    @property
    def permission_level(self) -> PermissionLevel:
        return PermissionLevel.AUTO

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Load the image and return a multimodal message structure.

        Args:
            arguments: Must contain ``source``; optional ``prompt``.

        Returns:
            A ``ToolResult`` whose ``output`` is a human-readable summary
            and whose ``metadata["vision_message"]`` contains the dict
            ready for LLM injection.
        """
        validated = self.validate_arguments(arguments)
        source: str = validated["source"]
        prompt: str = validated.get("prompt", "Describe this image in detail.")

        try:
            b64_data, mime_type = self._resolve_source(source)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        vision_msg = build_vision_message(b64_data, mime_type, prompt)

        return ToolResult(
            success=True,
            output=(
                f"Image loaded ({mime_type}, "
                f"{len(b64_data)} base64 chars). "
                f"Prompt: {prompt}"
            ),
            metadata={
                "vision_message": vision_msg,
                "mime_type": mime_type,
                "base64_length": len(b64_data),
            },
        )

    # ---- private helpers ------------------------------------------------

    def _resolve_source(self, source: str) -> tuple[str, str]:
        """Dispatch to the appropriate loader based on source format.

        Args:
            source: Raw source string from the user.

        Returns:
            ``(base64_data, mime_type)`` tuple.

        Raises:
            ValueError: On any load/validation failure.
        """
        # Already base64-encoded data
        if source.startswith("base64:"):
            raw_b64 = source[7:]
            if not raw_b64:
                raise ValueError("Empty base64 data provided.")
            # Validate it decodes
            try:
                decoded = base64.b64decode(raw_b64, validate=True)
            except Exception as exc:
                raise ValueError(f"Invalid base64 data: {exc}") from exc
            if not self._processor.validate_image(decoded):
                raise ValueError("Base64 data is not a valid image.")
            mime = ImageProcessor._mime_from_magic(decoded) or "image/jpeg"
            return raw_b64, mime

        # URL
        if source.startswith(("http://", "https://")):
            return self._processor.load_from_url(source)

        # Local file path
        return self._processor.load_from_path(source)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_vision_model(model_id: str) -> bool:
    """Check whether *model_id* supports vision / image input.

    Uses both the explicit ``ALL_VISION_MODELS`` list (substring match)
    and a pattern-based heuristic for model families known to support
    vision (``"vision"``, ``"gpt-4o"``, ``"claude-3"``, ``"gemini"``).

    The comparison is case-insensitive.

    Args:
        model_id: The model identifier to test.

    Returns:
        ``True`` if the model is known to support vision.
    """
    model_lower = model_id.lower()

    # Explicit match against the known model list
    if any(
        m.lower() in model_lower or model_lower in m.lower()
        for m in ALL_VISION_MODELS
    ):
        return True

    # Pattern-based heuristic
    return bool(_VISION_PATTERN.search(model_lower))


def get_vision_models_for_provider(provider: str) -> list[str]:
    """Return vision-capable model ids for *provider*.

    Args:
        provider: Provider key (e.g. ``"anthropic"``, ``"openai"``).

    Returns:
        List of model id strings, possibly empty.
    """
    return VISION_MODELS.get(provider, [])


def detect_terminal_image_support() -> str | None:
    """Detect the terminal image display protocol (if any).

    Inspects the ``TERM_PROGRAM`` and ``TERM`` environment variables.

    Returns:
        ``"iterm2"``, ``"kitty"``, or ``None``.
    """
    term_program = os.environ.get("TERM_PROGRAM", "")
    if "iTerm" in term_program:
        return "iterm2"

    term = os.environ.get("TERM", "")
    if "kitty" in term.lower():
        return "kitty"

    return None


def display_image_preview(path: Path, protocol: str | None = None) -> bool:
    """Display an inline image preview in the terminal.

    Uses the iTerm2 or Kitty graphics protocol.  Does nothing (and
    returns ``False``) if the terminal does not support inline images.

    Args:
        path: Path to the image file to preview.
        protocol: Override auto-detection (``"iterm2"`` or ``"kitty"``).

    Returns:
        ``True`` if a preview was successfully written to stdout.
    """
    if protocol is None:
        protocol = detect_terminal_image_support()

    if protocol is None:
        return False

    try:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")

        if protocol == "iterm2":
            osc = (
                f"\033]1337;File=inline=1;width=40;"
                f"preserveAspectRatio=1:{b64}\a"
            )
            sys.stdout.write(osc)
            sys.stdout.flush()
            return True

        if protocol == "kitty":
            chunk_size = 4096
            chunks = [b64[i : i + chunk_size] for i in range(0, len(b64), chunk_size)]
            for i, chunk in enumerate(chunks):
                m = 1 if i < len(chunks) - 1 else 0
                cmd = f"\033_Ga=T,f=100,m={m};{chunk}\033\\"
                sys.stdout.write(cmd)
            sys.stdout.flush()
            return True
    except OSError:
        logger.debug("image_preview_failed", path=str(path))

    return False


def process_image(path: str | Path) -> ImageAttachment:
    """Load, validate, and optionally compress an image for LLM use.

    The function resolves the path, checks for a supported extension,
    reads the bytes, determines dimensions, and compresses if the raw
    size exceeds ``MAX_IMAGE_SIZE_BYTES``.

    Args:
        path: File system path to the image.

    Returns:
        An ``ImageAttachment`` ready to be serialised into an API message.

    Raises:
        ValueError: If the file is missing or the format is unsupported.
    """
    image_path = Path(path).resolve()

    if not image_path.is_file():
        raise ValueError(f"Image file not found: {image_path}")

    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    raw_data = image_path.read_bytes()
    original_size = len(raw_data)

    width, height = _get_image_dimensions(raw_data)
    media_type = _get_media_type(suffix)
    was_compressed = False

    if original_size > MAX_IMAGE_SIZE_BYTES:
        raw_data, media_type, width, height = _compress_image(raw_data, suffix)
        was_compressed = True

    b64_data = base64.b64encode(raw_data).decode("ascii")

    return ImageAttachment(
        original_path=image_path,
        media_type=media_type,
        base64_data=b64_data,
        width=width,
        height=height,
        size_bytes=len(raw_data),
        was_compressed=was_compressed,
    )


def build_multimodal_messages(
    text_prompt: str,
    images: list[ImageAttachment],
    provider: str = "anthropic",
) -> list[dict]:
    """Build a ``content`` list containing images and text.

    Images appear first so the model "sees" them before reading the
    text instruction.

    Args:
        text_prompt: The text portion of the user message.
        images: Processed image attachments.
        provider: Target provider for format selection.

    Returns:
        A list of content dicts suitable for the ``content`` field of
        a chat message.
    """
    content: list[dict] = []

    for img in images:
        content.append(img.to_message_content(provider))

    content.append({"type": "text", "text": text_prompt})

    return content


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_media_type(suffix: str) -> str:
    """Map a file extension to its MIME type.

    Args:
        suffix: Lowercase file extension including the dot.

    Returns:
        MIME type string; defaults to ``image/jpeg`` for unknown extensions.
    """
    mapping: dict[str, str] = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return mapping.get(suffix, "image/jpeg")


def _get_image_dimensions(data: bytes) -> tuple[int, int]:
    """Extract width and height from raw image bytes.

    Supports PNG (IHDR chunk) and JPEG (SOF markers) without requiring
    PIL / Pillow.  Returns ``(0, 0)`` for unrecognised formats.

    Args:
        data: Raw image file bytes.

    Returns:
        ``(width, height)`` tuple.
    """
    # PNG magic: 89 50 4E 47 0D 0A 1A 0A
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w = int.from_bytes(data[16:20], "big")
        h = int.from_bytes(data[20:24], "big")
        return w, h

    # JPEG magic: FF D8
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            # SOF0, SOF1, SOF2 markers contain dimensions
            if marker in (0xC0, 0xC1, 0xC2):
                h = int.from_bytes(data[i + 5 : i + 7], "big")
                w = int.from_bytes(data[i + 7 : i + 9], "big")
                return w, h
            # SOS or EOI -- stop scanning
            if marker in (0xD9, 0xDA):
                break
            length = int.from_bytes(data[i + 2 : i + 4], "big")
            i += 2 + length

    return 0, 0


def _compress_image(
    data: bytes,
    suffix: str,
) -> tuple[bytes, str, int, int]:
    """Compress an image to fit within ``MAX_IMAGE_SIZE_BYTES``.

    On macOS, uses ``sips`` (always available).  On other platforms
    the image is returned as-is with a warning logged.

    Args:
        data: Raw image bytes.
        suffix: Original file extension (e.g. ``".png"``).

    Returns:
        ``(compressed_bytes, media_type, width, height)``
    """
    width, height = _get_image_dimensions(data)

    if platform.system() == "Darwin":
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as src_file:
                src_file.write(data)
                src_path = src_file.name

            out_path = src_path + ".jpg"

            # Calculate a conservative scale factor
            scale = (MAX_IMAGE_SIZE_BYTES / len(data)) ** 0.5
            new_w = max(int(width * scale), 100) if width > 0 else 800
            new_h = max(int(height * scale), 100) if height > 0 else 600

            subprocess.run(
                [
                    "sips",
                    "-s", "format", "jpeg",
                    "-s", "formatOptions", "70",
                    "-z", str(new_h), str(new_w),
                    src_path,
                    "--out", out_path,
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )

            out = Path(out_path)
            if out.is_file() and out.stat().st_size <= MAX_IMAGE_SIZE_BYTES:
                compressed = out.read_bytes()
                w, h = _get_image_dimensions(compressed)
                Path(src_path).unlink(missing_ok=True)
                out.unlink(missing_ok=True)
                return compressed, "image/jpeg", w, h

            Path(src_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)
        except (subprocess.SubprocessError, OSError):
            pass

    # Fallback: return original data with a warning
    logger.warning("image_compression_failed", size=len(data))
    return data, _get_media_type(suffix), width, height

"""Vision and multimodal input -- image processing and attachment for LLM conversations.

Provides image loading, validation, compression, base64 encoding, terminal
preview, and message formatting for vision-capable LLM models.

No heavy dependencies required -- uses raw byte parsing for image dimensions
and ``sips`` (macOS) for compression.  Falls back gracefully on other platforms.
"""

from __future__ import annotations

import base64
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import structlog

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_IMAGE_SIZE_BYTES: int = 1_048_576  # 1 MB
SUPPORTED_FORMATS: set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif",
}
TARGET_FORMAT: str = "JPEG"  # Default compression target


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
# Public helpers
# ---------------------------------------------------------------------------


def is_vision_model(model_id: str) -> bool:
    """Check whether *model_id* supports vision / image input.

    The comparison is case-insensitive and checks for substring matches
    so that prefixed model ids (e.g. ``anthropic/claude-3-opus-20240229``)
    are recognised.

    Args:
        model_id: The model identifier to test.

    Returns:
        ``True`` if the model is known to support vision.
    """
    model_lower = model_id.lower()
    return any(
        m.lower() in model_lower or model_lower in m.lower()
        for m in ALL_VISION_MODELS
    )


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

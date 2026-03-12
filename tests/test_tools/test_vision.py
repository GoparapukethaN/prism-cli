"""Tests for vision and multimodal input -- image processing for LLM conversations.

All filesystem access uses ``tmp_path``; URL fetches are mocked via httpx;
subprocess calls (``sips``) are mocked; environment variables for terminal
detection are patched.  No real file I/O outside of ``tmp_path``.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prism.tools.base import PermissionLevel
from prism.tools.vision import (
    ALL_VISION_MODELS,
    MAX_IMAGE_SIZE_BYTES,
    SUPPORTED_FORMATS,
    VISION_MODELS,
    ImageAttachment,
    ImageProcessor,
    VisionTool,
    _get_image_dimensions,
    _get_media_type,
    build_multimodal_messages,
    build_vision_message,
    detect_terminal_image_support,
    display_image_preview,
    get_vision_models_for_provider,
    is_vision_model,
    process_image,
)

# =====================================================================
# Helpers -- minimal valid image byte sequences
# =====================================================================


def _make_png_bytes(width: int = 100, height: int = 50) -> bytes:
    """Build a minimal PNG file with the given dimensions.

    Only the signature and IHDR chunk are constructed; this is enough
    for dimension parsing but is not a renderable image.
    """
    signature = b"\x89PNG\r\n\x1a\n"
    # IHDR: width (4B) + height (4B) + bit-depth (1) + colour (1)
    #        + compression (1) + filter (1) + interlace (1) = 13 bytes
    ihdr_data = (
        struct.pack(">II", width, height)
        + b"\x08"  # bit depth
        + b"\x02"  # colour type (RGB)
        + b"\x00"  # compression
        + b"\x00"  # filter
        + b"\x00"  # interlace
    )
    ihdr_length = struct.pack(">I", len(ihdr_data))
    ihdr_type = b"IHDR"
    # CRC is not validated by our parser, so a dummy is fine.
    ihdr_crc = b"\x00\x00\x00\x00"
    return signature + ihdr_length + ihdr_type + ihdr_data + ihdr_crc


def _make_jpeg_bytes(width: int = 640, height: int = 480) -> bytes:
    """Build a minimal JPEG byte sequence with a SOF0 marker.

    Not a renderable image, but sufficient for dimension extraction.
    """
    # SOI marker
    soi = b"\xff\xd8"
    # APP0 (JFIF) -- smallest valid segment
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    # SOF0 marker: length=2+6+3*3=17, precision=8, 3 components
    sof0_length = struct.pack(">H", 17)
    sof0 = (
        b"\xff\xc0"
        + sof0_length
        + b"\x08"  # precision
        + struct.pack(">HH", height, width)
        + b"\x03"  # number of components
        + b"\x01\x11\x00"  # component 1
        + b"\x02\x11\x01"  # component 2
        + b"\x03\x11\x01"  # component 3
    )
    # EOI
    eoi = b"\xff\xd9"
    return soi + app0 + sof0 + eoi


def _make_gif_bytes() -> bytes:
    """Build minimal GIF89a header bytes."""
    return b"GIF89a" + b"\x00" * 20


def _make_webp_bytes() -> bytes:
    """Build minimal WebP header bytes (RIFF...WEBP)."""
    # RIFF + 4-byte size + WEBP + padding
    return b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 20


def _make_dummy_attachment(
    path: Path | None = None,
    media_type: str = "image/png",
    width: int = 100,
    height: int = 50,
) -> ImageAttachment:
    """Create a dummy ``ImageAttachment`` for tests that don't need I/O."""
    return ImageAttachment(
        original_path=path or Path("/tmp/test.png"),
        media_type=media_type,
        base64_data=base64.b64encode(b"fake").decode(),
        width=width,
        height=height,
        size_bytes=4,
        was_compressed=False,
    )


# =====================================================================
# TestImageFormats
# =====================================================================


class TestImageFormats:
    """Verify that SUPPORTED_FORMATS contains expected extensions."""

    def test_jpg_supported(self) -> None:
        assert ".jpg" in SUPPORTED_FORMATS

    def test_jpeg_supported(self) -> None:
        assert ".jpeg" in SUPPORTED_FORMATS

    def test_png_supported(self) -> None:
        assert ".png" in SUPPORTED_FORMATS

    def test_gif_supported(self) -> None:
        assert ".gif" in SUPPORTED_FORMATS

    def test_webp_supported(self) -> None:
        assert ".webp" in SUPPORTED_FORMATS

    def test_bmp_supported(self) -> None:
        assert ".bmp" in SUPPORTED_FORMATS

    def test_tiff_supported(self) -> None:
        assert ".tiff" in SUPPORTED_FORMATS

    def test_tif_supported(self) -> None:
        assert ".tif" in SUPPORTED_FORMATS

    def test_svg_not_supported(self) -> None:
        assert ".svg" not in SUPPORTED_FORMATS

    def test_pdf_not_supported(self) -> None:
        assert ".pdf" not in SUPPORTED_FORMATS

    def test_txt_not_supported(self) -> None:
        assert ".txt" not in SUPPORTED_FORMATS


# =====================================================================
# TestIsVisionModel
# =====================================================================


class TestIsVisionModel:
    """Verify ``is_vision_model`` recognises known vision models."""

    def test_claude_sonnet_4(self) -> None:
        assert is_vision_model("claude-sonnet-4-20250514") is True

    def test_claude_35_sonnet(self) -> None:
        assert is_vision_model("claude-3-5-sonnet-20241022") is True

    def test_claude_3_opus(self) -> None:
        assert is_vision_model("claude-3-opus-20240229") is True

    def test_claude_3_haiku(self) -> None:
        assert is_vision_model("claude-3-haiku-20240307") is True

    def test_gpt4o(self) -> None:
        assert is_vision_model("gpt-4o") is True

    def test_gpt4o_mini(self) -> None:
        assert is_vision_model("gpt-4o-mini") is True

    def test_gpt4_turbo(self) -> None:
        assert is_vision_model("gpt-4-turbo") is True

    def test_gemini_flash(self) -> None:
        assert is_vision_model("gemini/gemini-1.5-flash") is True

    def test_gemini_pro(self) -> None:
        assert is_vision_model("gemini/gemini-1.5-pro") is True

    def test_gemini_20_flash(self) -> None:
        assert is_vision_model("gemini/gemini-2.0-flash") is True

    def test_non_vision_model_gpt35(self) -> None:
        assert is_vision_model("gpt-3.5-turbo") is False

    def test_non_vision_model_deepseek(self) -> None:
        assert is_vision_model("deepseek-chat") is False

    def test_non_vision_model_random(self) -> None:
        assert is_vision_model("some-unknown-model-v1") is False

    def test_case_insensitive(self) -> None:
        assert is_vision_model("GPT-4O") is True

    def test_prefixed_model_id(self) -> None:
        assert is_vision_model("anthropic/claude-3-opus-20240229") is True

    def test_vision_keyword_in_name(self) -> None:
        assert is_vision_model("my-custom-vision-model") is True

    def test_gemini_keyword_match(self) -> None:
        assert is_vision_model("gemini-2.5-pro") is True

    def test_claude3_pattern_match(self) -> None:
        assert is_vision_model("claude-3-unknown-variant") is True


# =====================================================================
# TestGetVisionModels
# =====================================================================


class TestGetVisionModels:
    """Verify per-provider model listing."""

    def test_anthropic_models(self) -> None:
        models = get_vision_models_for_provider("anthropic")
        assert len(models) >= 4
        assert "claude-sonnet-4-20250514" in models

    def test_openai_models(self) -> None:
        models = get_vision_models_for_provider("openai")
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models

    def test_google_models(self) -> None:
        models = get_vision_models_for_provider("google")
        assert "gemini/gemini-2.0-flash" in models

    def test_unknown_provider_returns_empty(self) -> None:
        assert get_vision_models_for_provider("unknown_provider") == []

    def test_all_vision_models_populated(self) -> None:
        total = sum(len(v) for v in VISION_MODELS.values())
        assert len(ALL_VISION_MODELS) == total


# =====================================================================
# TestMediaType
# =====================================================================


class TestMediaType:
    """Extension-to-MIME mapping."""

    def test_jpg(self) -> None:
        assert _get_media_type(".jpg") == "image/jpeg"

    def test_jpeg(self) -> None:
        assert _get_media_type(".jpeg") == "image/jpeg"

    def test_png(self) -> None:
        assert _get_media_type(".png") == "image/png"

    def test_gif(self) -> None:
        assert _get_media_type(".gif") == "image/gif"

    def test_webp(self) -> None:
        assert _get_media_type(".webp") == "image/webp"

    def test_bmp(self) -> None:
        assert _get_media_type(".bmp") == "image/bmp"

    def test_tiff(self) -> None:
        assert _get_media_type(".tiff") == "image/tiff"

    def test_tif(self) -> None:
        assert _get_media_type(".tif") == "image/tiff"

    def test_unknown_defaults_jpeg(self) -> None:
        assert _get_media_type(".xyz") == "image/jpeg"


# =====================================================================
# TestImageDimensions
# =====================================================================


class TestImageDimensions:
    """PNG and JPEG header dimension parsing."""

    def test_png_dimensions(self) -> None:
        data = _make_png_bytes(width=1920, height=1080)
        assert _get_image_dimensions(data) == (1920, 1080)

    def test_png_small(self) -> None:
        data = _make_png_bytes(width=1, height=1)
        assert _get_image_dimensions(data) == (1, 1)

    def test_png_large(self) -> None:
        data = _make_png_bytes(width=7680, height=4320)
        assert _get_image_dimensions(data) == (7680, 4320)

    def test_jpeg_dimensions(self) -> None:
        data = _make_jpeg_bytes(width=800, height=600)
        assert _get_image_dimensions(data) == (800, 600)

    def test_jpeg_small(self) -> None:
        data = _make_jpeg_bytes(width=16, height=16)
        assert _get_image_dimensions(data) == (16, 16)

    def test_unknown_format_returns_zero(self) -> None:
        assert _get_image_dimensions(b"NOT_AN_IMAGE") == (0, 0)

    def test_empty_data_returns_zero(self) -> None:
        assert _get_image_dimensions(b"") == (0, 0)

    def test_truncated_png_returns_zero(self) -> None:
        # PNG signature but not enough data for IHDR
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        assert _get_image_dimensions(data) == (0, 0)

    def test_jpeg_without_sof_returns_zero(self) -> None:
        # SOI + EOI with no SOF marker
        data = b"\xff\xd8\xff\xd9"
        assert _get_image_dimensions(data) == (0, 0)


# =====================================================================
# TestProcessImage
# =====================================================================


class TestProcessImage:
    """Image loading, validation, and compression via ``process_image``."""

    def test_valid_png(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        png_data = _make_png_bytes(200, 150)
        img.write_bytes(png_data)

        att = process_image(img)

        assert att.original_path == img.resolve()
        assert att.media_type == "image/png"
        assert att.width == 200
        assert att.height == 150
        assert att.was_compressed is False
        # base64 round-trip
        assert base64.b64decode(att.base64_data) == png_data

    def test_valid_jpeg(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.jpg"
        jpeg_data = _make_jpeg_bytes(640, 480)
        img.write_bytes(jpeg_data)

        att = process_image(img)

        assert att.media_type == "image/jpeg"
        assert att.width == 640
        assert att.height == 480
        assert att.was_compressed is False

    def test_missing_file_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Image file not found"):
            process_image("/nonexistent/path/to/image.png")

    def test_unsupported_format_raises_value_error(self, tmp_path: Path) -> None:
        svg = tmp_path / "icon.svg"
        svg.write_text("<svg></svg>")

        with pytest.raises(ValueError, match="Unsupported image format"):
            process_image(svg)

    def test_auto_compression_with_sips(self, tmp_path: Path) -> None:
        """When image exceeds MAX_IMAGE_SIZE_BYTES, _compress_image is called."""
        img = tmp_path / "large.png"
        # Create an oversized file (just padding; dimensions don't matter)
        large_data = _make_png_bytes(4000, 3000) + b"\x00" * (MAX_IMAGE_SIZE_BYTES + 100)
        img.write_bytes(large_data)

        # Mock _compress_image to avoid real sips call
        compressed = _make_jpeg_bytes(2000, 1500)
        with patch(
            "prism.tools.vision._compress_image",
            return_value=(compressed, "image/jpeg", 2000, 1500),
        ):
            att = process_image(img)

        assert att.was_compressed is True
        assert att.media_type == "image/jpeg"
        assert att.width == 2000
        assert att.height == 1500

    def test_small_image_not_compressed(self, tmp_path: Path) -> None:
        img = tmp_path / "small.png"
        img.write_bytes(_make_png_bytes(10, 10))

        att = process_image(img)

        assert att.was_compressed is False

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(_make_png_bytes())

        att = process_image(str(img))
        assert att.original_path == img.resolve()

    def test_gif_format(self, tmp_path: Path) -> None:
        img = tmp_path / "anim.gif"
        img.write_bytes(_make_gif_bytes())

        att = process_image(img)

        assert att.media_type == "image/gif"
        assert att.width == 0  # our parser doesn't handle GIF
        assert att.height == 0

    def test_webp_format(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.webp"
        img.write_bytes(_make_webp_bytes())

        att = process_image(img)

        assert att.media_type == "image/webp"


# =====================================================================
# TestImageAttachment
# =====================================================================


class TestImageAttachment:
    """Message content formatting for different providers."""

    def test_anthropic_format(self) -> None:
        att = _make_dummy_attachment(media_type="image/png")
        content = att.to_message_content(provider="anthropic")

        assert content["type"] == "image"
        assert content["source"]["type"] == "base64"
        assert content["source"]["media_type"] == "image/png"
        assert content["source"]["data"] == att.base64_data

    def test_openai_format(self) -> None:
        att = _make_dummy_attachment(media_type="image/jpeg")
        content = att.to_message_content(provider="openai")

        assert content["type"] == "image_url"
        url = content["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        assert att.base64_data in url

    def test_google_format_uses_openai_style(self) -> None:
        att = _make_dummy_attachment()
        content = att.to_message_content(provider="google")

        assert content["type"] == "image_url"

    def test_default_provider_is_anthropic(self) -> None:
        att = _make_dummy_attachment()
        content = att.to_message_content()

        assert content["type"] == "image"

    def test_attributes_stored(self) -> None:
        att = ImageAttachment(
            original_path=Path("/img.png"),
            media_type="image/png",
            base64_data="AAAA",
            width=320,
            height=240,
            size_bytes=1024,
            was_compressed=True,
        )
        assert att.original_path == Path("/img.png")
        assert att.width == 320
        assert att.height == 240
        assert att.size_bytes == 1024
        assert att.was_compressed is True


# =====================================================================
# TestBuildMultimodalMessages
# =====================================================================


class TestBuildMultimodalMessages:
    """Build content arrays with mixed image + text."""

    def test_single_image_and_text(self) -> None:
        att = _make_dummy_attachment()
        content = build_multimodal_messages("Describe this image", [att])

        assert len(content) == 2
        assert content[0]["type"] == "image"
        assert content[1] == {"type": "text", "text": "Describe this image"}

    def test_multiple_images_and_text(self) -> None:
        att1 = _make_dummy_attachment()
        att2 = _make_dummy_attachment(media_type="image/jpeg")
        content = build_multimodal_messages(
            "Compare these", [att1, att2], provider="openai"
        )

        assert len(content) == 3
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"
        assert content[2]["text"] == "Compare these"

    def test_no_images(self) -> None:
        content = build_multimodal_messages("Hello", [])

        assert len(content) == 1
        assert content[0] == {"type": "text", "text": "Hello"}

    def test_provider_passed_through(self) -> None:
        att = _make_dummy_attachment()
        content_a = build_multimodal_messages("x", [att], provider="anthropic")
        content_o = build_multimodal_messages("x", [att], provider="openai")

        assert content_a[0]["type"] == "image"
        assert content_o[0]["type"] == "image_url"


# =====================================================================
# TestBuildVisionMessage (OpenAI-format standalone function)
# =====================================================================


class TestBuildVisionMessage:
    """Verify ``build_vision_message`` produces the correct structure."""

    def test_basic_structure(self) -> None:
        msg = build_vision_message("AAAA", "image/png", "What is this?")

        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "image_url"
        assert msg["content"][1]["type"] == "text"
        assert msg["content"][1]["text"] == "What is this?"

    def test_data_url_format(self) -> None:
        msg = build_vision_message("BASE64DATA", "image/jpeg", "Describe")

        url = msg["content"][0]["image_url"]["url"]
        assert url == "data:image/jpeg;base64,BASE64DATA"

    def test_different_mime_types(self) -> None:
        for mime in ("image/png", "image/gif", "image/webp"):
            msg = build_vision_message("x", mime, "test")
            url = msg["content"][0]["image_url"]["url"]
            assert url.startswith(f"data:{mime};base64,")


# =====================================================================
# TestImageProcessor
# =====================================================================


class TestImageProcessor:
    """Tests for the ImageProcessor class."""

    def setup_method(self) -> None:
        self.processor = ImageProcessor()

    # --- validate_image ---

    def test_validate_png(self) -> None:
        assert self.processor.validate_image(_make_png_bytes()) is True

    def test_validate_jpeg(self) -> None:
        assert self.processor.validate_image(_make_jpeg_bytes()) is True

    def test_validate_gif(self) -> None:
        assert self.processor.validate_image(_make_gif_bytes()) is True

    def test_validate_webp(self) -> None:
        assert self.processor.validate_image(_make_webp_bytes()) is True

    def test_validate_invalid_data(self) -> None:
        assert self.processor.validate_image(b"NOT AN IMAGE") is False

    def test_validate_empty_data(self) -> None:
        assert self.processor.validate_image(b"") is False

    def test_validate_short_data(self) -> None:
        assert self.processor.validate_image(b"\x89P") is False

    def test_validate_riff_without_webp(self) -> None:
        # RIFF header but no WEBP tag
        data = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 20
        assert self.processor.validate_image(data) is False

    # --- load_from_path ---

    def test_load_from_path_png(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        png_data = _make_png_bytes()
        img.write_bytes(png_data)

        b64, mime = self.processor.load_from_path(str(img))

        assert mime == "image/png"
        assert base64.b64decode(b64) == png_data

    def test_load_from_path_jpeg(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        jpeg_data = _make_jpeg_bytes()
        img.write_bytes(jpeg_data)

        _b64, mime = self.processor.load_from_path(str(img))

        assert mime == "image/jpeg"

    def test_load_from_path_missing_file(self) -> None:
        with pytest.raises(ValueError, match="Image file not found"):
            self.processor.load_from_path("/nonexistent/image.png")

    def test_load_from_path_not_image(self, tmp_path: Path) -> None:
        txt = tmp_path / "data.png"
        txt.write_text("This is not an image")

        with pytest.raises(ValueError, match="not a valid image"):
            self.processor.load_from_path(str(txt))

    def test_load_from_path_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Path traversal"):
            self.processor.load_from_path(str(tmp_path / ".." / "etc" / "passwd"))

    # --- load_from_url ---

    def test_load_from_url_success(self) -> None:
        png_data = _make_png_bytes()
        mock_response = MagicMock()
        mock_response.content = png_data
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("prism.tools.vision.httpx.Client", return_value=mock_client):
            b64, mime = self.processor.load_from_url("https://example.com/image.png")

        assert mime == "image/png"
        assert base64.b64decode(b64) == png_data

    def test_load_from_url_invalid_scheme(self) -> None:
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            self.processor.load_from_url("ftp://example.com/image.png")

    def test_load_from_url_fetch_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = OSError("Connection failed")

        with patch("prism.tools.vision.httpx.Client", return_value=mock_client):
            with pytest.raises(ValueError, match="Failed to fetch"):
                self.processor.load_from_url("https://example.com/image.png")

    def test_load_from_url_not_image(self) -> None:
        mock_response = MagicMock()
        mock_response.content = b"This is not an image at all"
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("prism.tools.vision.httpx.Client", return_value=mock_client):
            with pytest.raises(ValueError, match="valid image"):
                self.processor.load_from_url("https://example.com/page.html")

    def test_load_from_url_infers_mime_from_magic(self) -> None:
        """If content-type is missing, MIME is inferred from magic bytes."""
        jpeg_data = _make_jpeg_bytes()
        mock_response = MagicMock()
        mock_response.content = jpeg_data
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("prism.tools.vision.httpx.Client", return_value=mock_client):
            _, mime = self.processor.load_from_url("https://example.com/photo")

        assert mime == "image/jpeg"

    # --- resize_if_needed ---

    def test_resize_small_data_unchanged(self) -> None:
        data = b"\x00" * 100
        assert self.processor.resize_if_needed(data) is data

    def test_resize_at_limit_unchanged(self) -> None:
        data = b"\x00" * MAX_IMAGE_SIZE_BYTES
        assert self.processor.resize_if_needed(data) is data

    def test_resize_over_limit_truncated(self) -> None:
        data = b"\x00" * (MAX_IMAGE_SIZE_BYTES + 1000)
        result = self.processor.resize_if_needed(data)
        assert len(result) == MAX_IMAGE_SIZE_BYTES

    def test_resize_custom_max_size(self) -> None:
        data = b"\x00" * 200
        result = self.processor.resize_if_needed(data, max_size=100)
        assert len(result) == 100

    def test_resize_custom_size_under_limit(self) -> None:
        data = b"\x00" * 50
        result = self.processor.resize_if_needed(data, max_size=100)
        assert result is data


# =====================================================================
# TestVisionTool
# =====================================================================


class TestVisionTool:
    """Tests for the VisionTool class."""

    def setup_method(self) -> None:
        self.tool = VisionTool()

    # --- properties ---

    def test_name(self) -> None:
        assert self.tool.name == "analyze_image"

    def test_description(self) -> None:
        assert "image" in self.tool.description.lower()

    def test_permission_level(self) -> None:
        assert self.tool.permission_level == PermissionLevel.AUTO

    def test_parameters_schema_has_source(self) -> None:
        schema = self.tool.parameters_schema
        assert "source" in schema["properties"]
        assert "source" in schema["required"]

    def test_parameters_schema_has_prompt(self) -> None:
        schema = self.tool.parameters_schema
        assert "prompt" in schema["properties"]

    def test_prompt_not_required(self) -> None:
        schema = self.tool.parameters_schema
        assert "prompt" not in schema["required"]

    # --- execute with local file ---

    def test_execute_local_file(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(_make_png_bytes())

        result = self.tool.execute({
            "source": str(img),
            "prompt": "What is in this image?",
        })

        assert result.success is True
        assert "image/png" in result.output
        assert result.metadata is not None
        assert "vision_message" in result.metadata
        msg = result.metadata["vision_message"]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2

    def test_execute_local_file_default_prompt(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(_make_jpeg_bytes())

        result = self.tool.execute({"source": str(img)})

        assert result.success is True
        assert "Describe this image" in result.output

    def test_execute_missing_file(self) -> None:
        result = self.tool.execute({"source": "/no/such/file.png"})

        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_execute_invalid_file(self, tmp_path: Path) -> None:
        txt = tmp_path / "fake.png"
        txt.write_text("not an image")

        result = self.tool.execute({"source": str(txt)})

        assert result.success is False
        assert "not a valid image" in (result.error or "").lower()

    # --- execute with URL ---

    def test_execute_url(self) -> None:
        png_data = _make_png_bytes()
        mock_response = MagicMock()
        mock_response.content = png_data
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_response

        with patch("prism.tools.vision.httpx.Client", return_value=mock_client):
            result = self.tool.execute({
                "source": "https://example.com/photo.png",
                "prompt": "Analyse this",
            })

        assert result.success is True
        assert "image/png" in result.output

    def test_execute_url_failure(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = OSError("timeout")

        with patch("prism.tools.vision.httpx.Client", return_value=mock_client):
            result = self.tool.execute({
                "source": "https://example.com/photo.png",
            })

        assert result.success is False
        assert "fetch" in (result.error or "").lower()

    # --- execute with base64 ---

    def test_execute_base64_png(self) -> None:
        png_data = _make_png_bytes()
        b64 = base64.b64encode(png_data).decode()

        result = self.tool.execute({
            "source": f"base64:{b64}",
            "prompt": "What is this?",
        })

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["mime_type"] == "image/png"

    def test_execute_base64_jpeg(self) -> None:
        jpeg_data = _make_jpeg_bytes()
        b64 = base64.b64encode(jpeg_data).decode()

        result = self.tool.execute({"source": f"base64:{b64}"})

        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["mime_type"] == "image/jpeg"

    def test_execute_base64_empty(self) -> None:
        result = self.tool.execute({"source": "base64:"})

        assert result.success is False
        assert "empty" in (result.error or "").lower()

    def test_execute_base64_invalid(self) -> None:
        result = self.tool.execute({"source": "base64:NOT_VALID_BASE64!!!"})

        assert result.success is False
        assert "invalid" in (result.error or "").lower() or "base64" in (result.error or "").lower()

    def test_execute_base64_not_image(self) -> None:
        b64 = base64.b64encode(b"this is plain text").decode()
        result = self.tool.execute({"source": f"base64:{b64}"})

        assert result.success is False
        assert "not a valid image" in (result.error or "").lower()

    # --- argument validation ---

    def test_missing_source_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing required"):
            self.tool.execute({})

    def test_vision_message_structure(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(_make_png_bytes())

        result = self.tool.execute({
            "source": str(img),
            "prompt": "Count the objects",
        })

        msg = result.metadata["vision_message"]
        assert msg["content"][0]["type"] == "image_url"
        assert "data:image/png;base64," in msg["content"][0]["image_url"]["url"]
        assert msg["content"][1]["text"] == "Count the objects"


# =====================================================================
# TestTerminalDetection
# =====================================================================


class TestTerminalDetection:
    """Environment variable-based terminal detection."""

    def test_iterm2_detected(self) -> None:
        with patch.dict("os.environ", {"TERM_PROGRAM": "iTerm.app"}, clear=False):
            assert detect_terminal_image_support() == "iterm2"

    def test_kitty_detected(self) -> None:
        with patch.dict(
            "os.environ",
            {"TERM_PROGRAM": "", "TERM": "xterm-kitty"},
            clear=False,
        ):
            assert detect_terminal_image_support() == "kitty"

    def test_unsupported_terminal(self) -> None:
        with patch.dict(
            "os.environ",
            {"TERM_PROGRAM": "Apple_Terminal", "TERM": "xterm-256color"},
            clear=False,
        ):
            assert detect_terminal_image_support() is None

    def test_empty_env_vars(self) -> None:
        with patch.dict(
            "os.environ", {"TERM_PROGRAM": "", "TERM": ""}, clear=False
        ):
            assert detect_terminal_image_support() is None

    def test_iterm2_takes_precedence_over_kitty_term(self) -> None:
        """If TERM_PROGRAM says iTerm but TERM says kitty, iTerm wins."""
        with patch.dict(
            "os.environ",
            {"TERM_PROGRAM": "iTerm.app", "TERM": "xterm-kitty"},
            clear=False,
        ):
            assert detect_terminal_image_support() == "iterm2"


# =====================================================================
# TestDisplayPreview
# =====================================================================


class TestDisplayPreview:
    """Terminal image preview (stdout mocked)."""

    def test_iterm2_protocol(self, tmp_path: Path) -> None:
        img = tmp_path / "preview.png"
        img.write_bytes(_make_png_bytes())

        mock_stdout = MagicMock()
        with patch("sys.stdout", mock_stdout):
            result = display_image_preview(img, protocol="iterm2")

        assert result is True
        written = mock_stdout.write.call_args[0][0]
        assert "\033]1337;File=" in written
        mock_stdout.flush.assert_called_once()

    def test_kitty_protocol(self, tmp_path: Path) -> None:
        img = tmp_path / "preview.png"
        img.write_bytes(_make_png_bytes())

        mock_stdout = MagicMock()
        with patch("sys.stdout", mock_stdout):
            result = display_image_preview(img, protocol="kitty")

        assert result is True
        # Kitty writes escape sequences
        first_write = mock_stdout.write.call_args_list[0][0][0]
        assert "\033_Ga=T" in first_write
        mock_stdout.flush.assert_called_once()

    def test_no_protocol_returns_false(self, tmp_path: Path) -> None:
        img = tmp_path / "preview.png"
        img.write_bytes(_make_png_bytes())

        with patch(
            "prism.tools.vision.detect_terminal_image_support",
            return_value=None,
        ):
            assert display_image_preview(img) is False

    def test_oserror_returns_false(self, tmp_path: Path) -> None:
        # File does not exist, so read_bytes raises
        missing = tmp_path / "missing.png"
        result = display_image_preview(missing, protocol="iterm2")
        assert result is False

    def test_auto_detection_used(self, tmp_path: Path) -> None:
        img = tmp_path / "preview.png"
        img.write_bytes(_make_png_bytes())

        with (
            patch(
                "prism.tools.vision.detect_terminal_image_support",
                return_value="iterm2",
            ),
            patch("sys.stdout", MagicMock()),
        ):
            assert display_image_preview(img) is True


# =====================================================================
# TestCompressImage
# =====================================================================


class TestCompressImage:
    """Compression via sips (mocked)."""

    def test_sips_called_on_darwin(self, tmp_path: Path) -> None:
        """On macOS, sips should be invoked for compression."""
        large_png = _make_png_bytes(4000, 3000)
        large_data = large_png + b"\x00" * (MAX_IMAGE_SIZE_BYTES + 500)

        compressed_jpeg = _make_jpeg_bytes(2000, 1500)

        def fake_sips_run(cmd: list[str], *, capture_output: bool, timeout: int, check: bool) -> MagicMock:
            # Write compressed output to the --out path
            out_path = cmd[-1]
            Path(out_path).write_bytes(compressed_jpeg)
            return MagicMock(returncode=0)

        img = tmp_path / "big.png"
        img.write_bytes(large_data)

        with (
            patch("prism.tools.vision.platform.system", return_value="Darwin"),
            patch("prism.tools.vision.subprocess.run", side_effect=fake_sips_run),
        ):
            att = process_image(img)

        assert att.was_compressed is True
        assert att.media_type == "image/jpeg"

    def test_sips_failure_returns_original(self, tmp_path: Path) -> None:
        """If sips fails, original data is returned with a warning."""
        from prism.tools.vision import _compress_image

        large_data = _make_png_bytes(2000, 1000) + b"\x00" * (MAX_IMAGE_SIZE_BYTES + 100)

        with (
            patch("prism.tools.vision.platform.system", return_value="Darwin"),
            patch(
                "prism.tools.vision.subprocess.run",
                side_effect=OSError("sips not found"),
            ),
        ):
            result_data, media_type, _w, _h = _compress_image(large_data, ".png")

        assert result_data == large_data
        assert media_type == "image/png"

    def test_non_darwin_returns_original(self) -> None:
        """On non-macOS, compression is a no-op."""
        from prism.tools.vision import _compress_image

        data = b"\x00" * 100

        with patch("prism.tools.vision.platform.system", return_value="Linux"):
            result_data, media_type, _w, _h = _compress_image(data, ".png")

        assert result_data == data
        assert media_type == "image/png"

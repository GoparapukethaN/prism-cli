"""Response validation — sanitises and checks LLM outputs before returning."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

import structlog

from prism.llm.result import CompletionResult

logger = structlog.get_logger(__name__)

# Patterns that look like leaked secrets.  We deliberately keep these broad
# to catch accidental echoing of credentials in model output.
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI / generic
    re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"),  # Anthropic
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),  # Groq
    re.compile(r"AIzaSy[A-Za-z0-9\-_]{30,}"),  # Google
    re.compile(r"xai-[A-Za-z0-9]{20,}"),  # xAI
    re.compile(r"AKIA[A-Z0-9]{16}"),  # AWS access key
]

_REDACTED = "[REDACTED]"


@dataclass(frozen=True)
class ValidationWarning:
    """A non-fatal issue detected during validation."""

    code: str
    message: str


class ResponseValidator:
    """Validates and sanitises LLM responses before surfacing to the user.

    Checks performed:

    * Empty content detection.
    * ``finish_reason`` analysis (``length`` means truncated output).
    * Secret / API-key leak scrubbing.
    """

    def validate(self, result: CompletionResult) -> CompletionResult:
        """Validate and optionally sanitise *result*.

        Returns a *new* ``CompletionResult`` if the content was modified
        (e.g. secrets redacted), otherwise returns the same object.

        Raises:
            ValueError: If the response is fundamentally invalid (empty
                content with ``finish_reason='stop'`` and no tool calls).
        """
        warnings_list: list[ValidationWarning] = []

        # --- 1. Empty content ---
        has_tool_calls = bool(result.tool_calls)
        if not result.content and not has_tool_calls:
            msg = (
                f"Empty response from {result.model} "
                f"(finish_reason={result.finish_reason})"
            )
            raise ValueError(msg)

        # --- 2. Finish reason ---
        if result.finish_reason == "length":
            warnings_list.append(
                ValidationWarning(
                    code="truncated",
                    message=(
                        f"Response from {result.model} was truncated "
                        "(finish_reason=length). Output may be incomplete."
                    ),
                )
            )
            logger.warning(
                "response_truncated",
                model=result.model,
                output_tokens=result.output_tokens,
            )

        if result.finish_reason == "content_filter":
            warnings_list.append(
                ValidationWarning(
                    code="content_filter",
                    message=(
                        f"Response from {result.model} was blocked by "
                        "content filter."
                    ),
                )
            )
            logger.warning("response_content_filtered", model=result.model)

        # --- 3. Secret-leak scrubbing ---
        sanitised_content = self._redact_secrets(result.content)

        # Emit Python warnings so callers can capture them if needed.
        for w in warnings_list:
            warnings.warn(w.message, UserWarning, stacklevel=2)

        if sanitised_content != result.content:
            logger.warning("response_secrets_redacted", model=result.model)
            return CompletionResult(
                content=sanitised_content,
                model=result.model,
                provider=result.provider,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cached_tokens=result.cached_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
                finish_reason=result.finish_reason,
                tool_calls=result.tool_calls,
                raw_response=result.raw_response,
            )

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _redact_secrets(text: str) -> str:
        """Replace anything that looks like a secret with ``[REDACTED]``."""
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(_REDACTED, text)
        return text

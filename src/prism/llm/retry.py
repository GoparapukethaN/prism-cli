"""Retry logic with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import structlog

from prism.exceptions import (
    ProviderRateLimitError,
    ProviderUnavailableError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)

T = TypeVar("T")

# Default set of errors worth retrying — auth and quota are NOT retried.
_DEFAULT_RETRYABLE: tuple[type[Exception], ...] = (
    ProviderRateLimitError,
    ProviderUnavailableError,
    TimeoutError,
    ConnectionError,
    OSError,
)


@dataclass
class RetryPolicy:
    """Configurable retry policy with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry *attempts* (0 = no retries).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Cap for the computed delay.
        exponential_base: Multiplier applied each attempt (2 = doubling).
        retryable_errors: Exception types that trigger a retry.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_errors: tuple[type[Exception], ...] = field(
        default_factory=lambda: _DEFAULT_RETRYABLE,
    )

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *func* with retries on retryable errors.

        Args:
            func: An async callable.
            *args: Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            The return value of *func*.

        Raises:
            Exception: The last error encountered if all retries are exhausted,
                or any non-retryable error immediately.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except self.retryable_errors as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    logger.warning(
                        "retry_exhausted",
                        func=getattr(func, "__name__", str(func)),
                        attempts=attempt + 1,
                        error=str(exc),
                    )
                    raise

                delay = self._calculate_delay(attempt)
                logger.info(
                    "retry_scheduled",
                    func=getattr(func, "__name__", str(func)),
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    delay_s=round(delay, 3),
                    error=str(exc),
                )
                await asyncio.sleep(delay)
            except Exception:
                # Non-retryable — bubble up immediately.
                raise

        # Should not be reachable, but satisfy type-checkers.
        if last_error is not None:  # pragma: no cover
            raise last_error
        msg = "execute() finished without returning or raising"  # pragma: no cover
        raise RuntimeError(msg)  # pragma: no cover

    def _calculate_delay(self, attempt: int) -> float:
        """Compute the delay for the given *attempt* index.

        Uses exponential back-off with full jitter::

            delay = random(0, min(max_delay, base_delay * exponential_base^attempt))

        Args:
            attempt: Zero-based attempt index.

        Returns:
            Delay in seconds.
        """
        raw = self.base_delay * (self.exponential_base ** attempt)
        capped = min(raw, self.max_delay)
        return random.uniform(0, capped)  # noqa: S311

"""Privacy mode — Ollama-only routing with zero cloud data transmission.

When activated via ``--private`` at startup, **every** LLM request is forced
through a local Ollama instance.  Cloud providers are hard-blocked and
``PrivacyViolationError`` is raised if any code path attempts a cloud call.

The module also manages Ollama lifecycle: checking whether the server is
running, starting it automatically, listing locally installed models, and
pulling recommended models on demand.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

RECOMMENDED_MODELS: dict[str, str] = {
    "qwen2.5-coder:7b": "Fast coding (7B params, good quality)",
    "llama3.1:8b": "General tasks (8B params, well-rounded)",
    "deepseek-coder-v2:16b": "Complex coding (16B params, needs 12GB+ RAM)",
    "codellama:7b": "Code generation (7B params, Meta)",
    "mistral:7b": "General purpose (7B params, fast)",
}

CLOUD_PROVIDERS: frozenset[str] = frozenset({
    "anthropic",
    "openai",
    "google",
    "deepseek",
    "mistral",
    "groq",
    "cohere",
    "together_ai",
    "fireworks_ai",
    "kimi",
    "perplexity",
    "qwen",
})

# How long to wait after starting the Ollama server before re-checking
_OLLAMA_START_WAIT: float = 2.0

# Subprocess timeouts (seconds)
_OLLAMA_LIST_TIMEOUT: int = 15
_OLLAMA_PULL_TIMEOUT: int = 600
_OLLAMA_CHECK_TIMEOUT: int = 10


# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------


class PrivacyLevel(Enum):
    """Privacy level for the current Prism session."""

    NORMAL = "normal"
    PRIVATE = "private"


@dataclass
class OllamaModel:
    """A locally installed Ollama model."""

    name: str
    size_bytes: int
    modified_at: str
    digest: str


@dataclass
class PrivacyStatus:
    """Snapshot of the current privacy mode state."""

    level: PrivacyLevel
    ollama_running: bool
    available_models: list[OllamaModel] = field(default_factory=list)
    active_model: str = ""


# -------------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------------


class PrivacyViolationError(Exception):
    """Raised when a cloud API call is attempted while private mode is active.

    This is a hard block — the request is **never** sent.
    """


# -------------------------------------------------------------------------
# Manager
# -------------------------------------------------------------------------


class PrivacyManager:
    """Manages privacy mode — Ollama-only routing with zero cloud data leakage.

    Usage::

        pm = PrivacyManager()
        status = pm.enable_private_mode()
        # All subsequent calls to ``validate_request`` block cloud providers.
        pm.validate_request("anthropic", "claude-3-opus")  # raises PrivacyViolationError
    """

    def __init__(self) -> None:
        """Initialise the privacy manager in ``NORMAL`` mode."""
        self._level: PrivacyLevel = PrivacyLevel.NORMAL
        self._ollama_running: bool | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def level(self) -> PrivacyLevel:
        """Return the current privacy level."""
        return self._level

    @property
    def is_private(self) -> bool:
        """Return ``True`` when private mode is active."""
        return self._level == PrivacyLevel.PRIVATE

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def enable_private_mode(self) -> PrivacyStatus:
        """Activate private mode.

        * Sets level to ``PRIVATE``.
        * Checks if Ollama is running; if not, attempts to start it.
        * Returns a ``PrivacyStatus`` snapshot.
        """
        self._level = PrivacyLevel.PRIVATE

        if not self.check_ollama():
            started = self.start_ollama()
            if not started:
                logger.warning("ollama_not_available")

        status = self.get_status()
        logger.info(
            "privacy_mode_enabled",
            ollama_running=status.ollama_running,
            models=len(status.available_models),
        )
        return status

    def disable_private_mode(self) -> None:
        """Deactivate private mode, restoring normal multi-provider routing."""
        self._level = PrivacyLevel.NORMAL
        logger.info("privacy_mode_disabled")

    # ------------------------------------------------------------------
    # Ollama lifecycle
    # ------------------------------------------------------------------

    def check_ollama(self) -> bool:
        """Return ``True`` if the Ollama CLI is present and responsive."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=_OLLAMA_CHECK_TIMEOUT,
                check=False,
            )
            self._ollama_running = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._ollama_running = False
        return self._ollama_running or False

    def start_ollama(self) -> bool:
        """Attempt to start the Ollama server in the background.

        Returns:
            ``True`` if Ollama is running after the attempt, ``False``
            otherwise.
        """
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Give the server a moment to initialise
            time.sleep(_OLLAMA_START_WAIT)
            return self.check_ollama()
        except (FileNotFoundError, OSError) as exc:
            logger.warning("ollama_start_failed", error=str(exc))
            return False

    def list_models(self) -> list[OllamaModel]:
        """List locally installed Ollama models.

        Returns:
            A list of ``OllamaModel`` instances.  Returns an empty list
            if Ollama is not running or the command fails.
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=_OLLAMA_LIST_TIMEOUT,
                check=False,
            )
            if result.returncode != 0:
                return []

            models: list[OllamaModel] = []
            lines = result.stdout.strip().split("\n")
            # Skip the header line ("NAME  ID  SIZE  MODIFIED")
            for line in lines[1:]:
                parts = line.split()
                if len(parts) < 4:
                    continue

                name = parts[0]
                digest = parts[1]
                size_str = parts[2]
                unit = parts[3] if len(parts) > 3 else ""
                modified = " ".join(parts[4:]) if len(parts) > 4 else ""

                size_bytes = self._parse_size(size_str, unit)

                models.append(
                    OllamaModel(
                        name=name,
                        size_bytes=size_bytes,
                        modified_at=modified,
                        digest=digest,
                    )
                )
            return models
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull (download) an Ollama model.

        Args:
            model_name: The model identifier, e.g. ``"llama3.1:8b"``.

        Returns:
            ``True`` if the model was successfully pulled, ``False``
            otherwise.
        """
        if not model_name or not model_name.strip():
            return False

        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True,
                timeout=_OLLAMA_PULL_TIMEOUT,
                check=False,
            )
            success = result.returncode == 0
            if success:
                logger.info("ollama_model_pulled", model=model_name)
            else:
                logger.warning(
                    "ollama_model_pull_failed",
                    model=model_name,
                    stderr=result.stderr[:200] if result.stderr else "",
                )
            return success
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("ollama_pull_error", model=model_name, error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Request validation
    # ------------------------------------------------------------------

    def is_cloud_provider(self, provider: str) -> bool:
        """Return ``True`` if *provider* is a cloud service (blocked in private mode).

        Args:
            provider: Provider name (case-insensitive).
        """
        return provider.lower().strip() in CLOUD_PROVIDERS

    def validate_request(self, provider: str, model: str) -> None:
        """Validate an outgoing LLM request against the current privacy mode.

        In ``NORMAL`` mode this is a no-op.  In ``PRIVATE`` mode, any
        request targeting a cloud provider raises ``PrivacyViolationError``.

        Args:
            provider: The provider name (e.g. ``"anthropic"``).
            model: The model identifier (e.g. ``"claude-3-opus"``).

        Raises:
            PrivacyViolationError: If a cloud provider is used in private mode.
        """
        if not self.is_private:
            return

        if self.is_cloud_provider(provider):
            raise PrivacyViolationError(
                f"Cloud provider '{provider}' blocked in private mode. "
                f"Only local Ollama models are allowed. "
                f"Use /private off to disable private mode."
            )

        # Also block models that are not prefixed with "ollama/" unless the
        # provider is explicitly set to "ollama".
        if provider.lower().strip() != "ollama" and not model.startswith("ollama/"):
            raise PrivacyViolationError(
                f"Model '{model}' is not routed through Ollama. "
                f"In private mode, only local models are allowed."
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> PrivacyStatus:
        """Build a ``PrivacyStatus`` snapshot of the current state.

        This always re-checks Ollama and refreshes the model list.
        """
        running = self.check_ollama()
        models = self.list_models() if running else []
        return PrivacyStatus(
            level=self._level,
            ollama_running=running,
            available_models=models,
        )

    def get_recommended_model(self) -> str:
        """Return the best locally installed model from the recommended list.

        Falls back to ``"llama3.1:8b"`` if none of the recommended models
        are installed.
        """
        installed = {m.name for m in self.list_models()}
        for model in RECOMMENDED_MODELS:
            if model in installed:
                return model
        # Default fallback
        return "llama3.1:8b"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_size(size_str: str, unit: str) -> int:
        """Parse a human-readable size string (e.g. ``"4.1"`` ``"GB"``) to bytes.

        Args:
            size_str: Numeric part of the size.
            unit: Unit string (``GB``, ``MB``, ``KB``, or empty).

        Returns:
            Size in bytes, or ``0`` if parsing fails.
        """
        try:
            value = float(size_str)
            unit_upper = unit.upper()
            if "GB" in unit_upper:
                return int(value * 1_073_741_824)
            if "MB" in unit_upper:
                return int(value * 1_048_576)
            if "KB" in unit_upper:
                return int(value * 1024)
            return int(value)
        except (ValueError, TypeError):
            return 0

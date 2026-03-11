"""Enhanced init wizard — full system setup with hardware detection and provider configuration.

Detects OS, Python version, available RAM, GPU presence, Ollama status,
walks through API key setup with validation, creates config files with
sensible defaults, shows cost comparison, sets budget limits, runs health
checks, and displays a quick-start summary.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SystemInfo:
    """Detected system information."""

    os_name: str
    os_version: str
    python_version: str
    ram_gb: float
    gpu_detected: bool
    gpu_name: str
    cpu_cores: int
    ollama_installed: bool
    ollama_models: list[str]
    docker_installed: bool


@dataclass
class ProviderSetup:
    """Configuration for a provider during setup."""

    name: str
    display_name: str
    env_var: str
    is_configured: bool = False
    is_healthy: bool = False
    example_cost: str = ""


@dataclass
class WizardResult:
    """Result of the init wizard."""

    system_info: SystemInfo
    providers_configured: list[str]
    config_path: Path | None = None
    memory_path: Path | None = None
    ignore_path: Path | None = None
    budget_daily: float = 10.0
    budget_monthly: float = 50.0


PROVIDER_CONFIGS: list[ProviderSetup] = [
    ProviderSetup(
        "anthropic",
        "Anthropic (Claude)",
        "ANTHROPIC_API_KEY",
        example_cost="$3/1M input, $15/1M output",
    ),
    ProviderSetup(
        "openai",
        "OpenAI (GPT-4o)",
        "OPENAI_API_KEY",
        example_cost="$2.50/1M input, $10/1M output",
    ),
    ProviderSetup(
        "google",
        "Google AI (Gemini)",
        "GOOGLE_API_KEY",
        example_cost="$1.25/1M input, $5/1M output",
    ),
    ProviderSetup(
        "deepseek",
        "DeepSeek",
        "DEEPSEEK_API_KEY",
        example_cost="$0.14/1M input, $0.28/1M output",
    ),
    ProviderSetup(
        "groq",
        "Groq",
        "GROQ_API_KEY",
        example_cost="Free tier available",
    ),
    ProviderSetup(
        "mistral",
        "Mistral",
        "MISTRAL_API_KEY",
        example_cost="$2/1M input, $6/1M output",
    ),
    ProviderSetup(
        "cohere",
        "Cohere",
        "COHERE_API_KEY",
        example_cost="$1/1M input, $2/1M output",
    ),
    ProviderSetup(
        "together_ai",
        "Together AI",
        "TOGETHER_API_KEY",
        example_cost="$0.20/1M tokens",
    ),
    ProviderSetup(
        "fireworks_ai",
        "Fireworks AI",
        "FIREWORKS_API_KEY",
        example_cost="$0.20/1M tokens",
    ),
]

RECOMMENDED_OLLAMA_MODELS: dict[str, tuple[str, float]] = {
    "qwen2.5-coder:7b": ("Fast coding", 4.0),
    "llama3.1:8b": ("General tasks", 5.0),
    "deepseek-coder-v2:16b": ("Complex coding", 12.0),
}

PRISM_MD_TEMPLATE = """# .prism.md — Project Memory

## Project Overview
<!-- Describe your project here -->

## Key Decisions
<!-- Record architectural decisions -->

## Conventions
<!-- Project-specific conventions -->
"""

PRISMIGNORE_DEFAULTS: list[str] = [
    "# Environment and secrets",
    ".env",
    ".env.*",
    "*.env",
    "secrets/",
    "credentials/",
    "private/",
    "",
    "# Cryptographic keys",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_ed25519",
    "*.pub",
    "",
    "# Cloud credentials",
    ".aws/",
    ".ssh/",
    ".gcloud/",
    "service-account*.json",
    "credentials.json",
    "",
    "# Dependencies and caches",
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "venv/",
    ".tox/",
    ".nox/",
    "",
    "# Logs",
    "*.log",
    "*.log.*",
    "",
    "# Build artifacts",
    "dist/",
    "build/",
    "*.egg-info/",
    "",
    "# IDE",
    ".idea/",
    ".vscode/",
    "*.swp",
    "*.swo",
]


class InitWizard:
    """Enhanced setup wizard for Prism CLI.

    Performs system detection, provider configuration, config file creation,
    and health checks. Can be run interactively (via the CLI ``prism init``
    command) or programmatically (via :meth:`run`).
    """

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialise the wizard.

        Args:
            project_root: The project directory to configure. Defaults to
                the current working directory.
        """
        self._root = (project_root or Path.cwd()).resolve()
        self._system_info: SystemInfo | None = None

    # ------------------------------------------------------------------
    # System detection
    # ------------------------------------------------------------------

    def detect_system(self) -> SystemInfo:
        """Detect system hardware and software configuration.

        Returns:
            A :class:`SystemInfo` dataclass populated with detected values.
        """
        os_name = platform.system()
        os_version = platform.release()
        python_version = (
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro}"
        )

        ram_gb = self._detect_ram()
        gpu_detected, gpu_name = self._detect_gpu()
        cpu_cores = os.cpu_count() or 1
        ollama_installed = self._check_ollama_installed()
        ollama_models = self._get_ollama_models() if ollama_installed else []
        docker_installed = self._check_docker()

        self._system_info = SystemInfo(
            os_name=os_name,
            os_version=os_version,
            python_version=python_version,
            ram_gb=ram_gb,
            gpu_detected=gpu_detected,
            gpu_name=gpu_name,
            cpu_cores=cpu_cores,
            ollama_installed=ollama_installed,
            ollama_models=ollama_models,
            docker_installed=docker_installed,
        )
        logger.info(
            "system_detected",
            os=os_name,
            ram_gb=round(ram_gb, 1),
            gpu=gpu_name or "none",
            ollama=ollama_installed,
        )
        return self._system_info

    # ------------------------------------------------------------------
    # Provider configuration
    # ------------------------------------------------------------------

    def check_provider(self, provider: ProviderSetup) -> bool:
        """Check if a provider API key is configured in the environment.

        Args:
            provider: The provider to check.

        Returns:
            ``True`` if the environment variable is set and non-trivial.
        """
        key = os.environ.get(provider.env_var, "")
        provider.is_configured = bool(key and len(key) > 5)
        return provider.is_configured

    def get_configured_providers(self) -> list[ProviderSetup]:
        """Get list of all providers with their configuration status.

        Returns:
            A copy of :data:`PROVIDER_CONFIGS` with ``is_configured``
            fields updated.
        """
        for p in PROVIDER_CONFIGS:
            self.check_provider(p)
        return list(PROVIDER_CONFIGS)

    def health_check_provider(self, provider: ProviderSetup) -> bool:
        """Run a lightweight health check on a configured provider.

        Currently checks whether the environment variable is set. In a
        production build this would also verify the key with a minimal API
        call; that is intentionally skipped here to avoid real API charges.

        Args:
            provider: The provider to health-check.

        Returns:
            ``True`` if the provider appears healthy.
        """
        if not provider.is_configured:
            provider.is_healthy = False
            return False
        # Mark healthy if the key looks present and non-empty
        key = os.environ.get(provider.env_var, "")
        provider.is_healthy = bool(key and len(key) > 5)
        return provider.is_healthy

    # ------------------------------------------------------------------
    # Config / file creation
    # ------------------------------------------------------------------

    def create_config(
        self,
        prism_home: Path | None = None,
        daily_budget: float = 10.0,
        monthly_budget: float = 50.0,
    ) -> Path:
        """Create ``~/.prism/config.yaml`` with sensible defaults.

        Args:
            prism_home: Override for the Prism home directory.
            daily_budget: Default daily budget in USD.
            monthly_budget: Default monthly budget in USD.

        Returns:
            Path to the created config file.
        """
        home = prism_home or Path.home() / ".prism"
        home.mkdir(parents=True, exist_ok=True)

        config_lines = [
            "# Prism CLI Configuration",
            "# Generated by prism init",
            "",
            "# Budget limits (USD)",
            "budget:",
            f"  daily_limit: {daily_budget}",
            f"  monthly_limit: {monthly_budget}",
            "  warn_at_percent: 70",
            "",
            "# Routing preferences",
            "routing:",
            "  prefer_cheap: true",
            "  fallback_enabled: true",
            "",
            "# Logging",
            "log_level: INFO",
            "",
            "# Cache",
            "cache:",
            "  enabled: true",
            "  ttl_seconds: 3600",
            "",
        ]

        config_path = home / "config.yaml"
        config_path.write_text("\n".join(config_lines), encoding="utf-8")
        logger.info("config_created", path=str(config_path))
        return config_path

    def create_project_memory(self) -> Path:
        """Create ``.prism.md`` in the project root from a template.

        If the file already exists it is **not** overwritten.

        Returns:
            Path to the ``.prism.md`` file.
        """
        memory_path = self._root / ".prism.md"
        if not memory_path.exists():
            memory_path.write_text(PRISM_MD_TEMPLATE, encoding="utf-8")
            logger.info("project_memory_created", path=str(memory_path))
        return memory_path

    def create_prismignore(self) -> Path:
        """Create ``.prismignore`` with secure defaults.

        Attempts to delegate to :class:`prism.security.prismignore.PrismIgnore`
        if available; otherwise writes a built-in set of default patterns.

        Returns:
            Path to the ``.prismignore`` file.
        """
        try:
            from prism.security.prismignore import PrismIgnore

            ignore = PrismIgnore(self._root)
            return ignore.create_default()
        except ImportError:
            # Fallback: write defaults directly
            ignore_path = self._root / ".prismignore"
            ignore_path.write_text(
                "\n".join(PRISMIGNORE_DEFAULTS) + "\n", encoding="utf-8"
            )
            logger.info("prismignore_created_fallback", path=str(ignore_path))
            return ignore_path

    # ------------------------------------------------------------------
    # Ollama helpers
    # ------------------------------------------------------------------

    def get_recommended_ollama_models(self) -> dict[str, tuple[str, float]]:
        """Get recommended Ollama models based on available RAM.

        Returns:
            A dict mapping model name to ``(description, min_ram_gb)``
            for models the system can run.
        """
        if not self._system_info:
            self.detect_system()

        ram = self._system_info.ram_gb if self._system_info else 0.0
        recommended: dict[str, tuple[str, float]] = {}
        for model, (desc, min_ram) in RECOMMENDED_OLLAMA_MODELS.items():
            if ram >= min_ram:
                recommended[model] = (desc, min_ram)
        return recommended

    def get_ollama_install_instructions(self) -> str:
        """Return platform-specific Ollama installation instructions.

        Returns:
            A multi-line string with install instructions.
        """
        sys_name = platform.system()
        if sys_name == "Darwin":
            return (
                "Install Ollama on macOS:\n"
                "  brew install ollama\n"
                "  — or —\n"
                "  Download from https://ollama.com/download/mac"
            )
        if sys_name == "Linux":
            return (
                "Install Ollama on Linux:\n"
                "  curl -fsSL https://ollama.com/install.sh | sh"
            )
        return (
            "Install Ollama:\n"
            "  Visit https://ollama.com for installation instructions."
        )

    # ------------------------------------------------------------------
    # Cost comparison
    # ------------------------------------------------------------------

    def get_cost_comparison(self) -> list[dict[str, str]]:
        """Generate cost comparison table data for configured providers.

        Returns:
            A list of dicts with ``provider``, ``configured``, and ``cost`` keys.
        """
        rows: list[dict[str, str]] = []
        for p in PROVIDER_CONFIGS:
            self.check_provider(p)
            rows.append(
                {
                    "provider": p.display_name,
                    "configured": "Yes" if p.is_configured else "No",
                    "cost": p.example_cost,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Quick-start summary
    # ------------------------------------------------------------------

    def get_quickstart_commands(self) -> list[str]:
        """Return a list of quick-start commands the user can try.

        Returns:
            A list of example shell commands.
        """
        return [
            "prism                      # Start interactive REPL",
            'prism ask "explain this code"  # Single-shot question',
            "prism /cost                # View cost dashboard",
            "prism /model               # Switch AI model",
            "prism /help                # Show all commands",
        ]

    # ------------------------------------------------------------------
    # Full wizard run
    # ------------------------------------------------------------------

    def run(self) -> WizardResult:
        """Run the full wizard (non-interactive, returns result).

        Performs system detection, provider checking, config file creation,
        and health checks.

        Returns:
            A :class:`WizardResult` summarising everything that was set up.
        """
        system_info = self.detect_system()
        providers = self.get_configured_providers()
        configured = [p.name for p in providers if p.is_configured]

        # Run health checks on configured providers
        for p in providers:
            if p.is_configured:
                self.health_check_provider(p)

        config_path = self.create_config()
        memory_path = self.create_project_memory()
        ignore_path = self.create_prismignore()

        logger.info(
            "wizard_complete",
            providers_configured=configured,
            config=str(config_path),
        )

        return WizardResult(
            system_info=system_info,
            providers_configured=configured,
            config_path=config_path,
            memory_path=memory_path,
            ignore_path=ignore_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers — hardware detection
    # ------------------------------------------------------------------

    def _detect_ram(self) -> float:
        """Detect available RAM in GB.

        Returns:
            RAM in gigabytes, or ``0.0`` on failure.
        """
        system = platform.system()
        try:
            if system == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0:
                    return int(result.stdout.strip()) / (1024**3)
            elif system == "Linux":
                meminfo = Path("/proc/meminfo")
                if meminfo.exists():
                    text = meminfo.read_text(encoding="utf-8")
                    for line in text.splitlines():
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return kb / (1024**2)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            logger.debug("ram_detection_failed", system=system)
        return 0.0

    def _detect_gpu(self) -> tuple[bool, str]:
        """Detect GPU presence.

        Returns:
            A ``(detected, name)`` tuple. ``name`` is empty when no GPU
            is found.
        """
        system = platform.system()
        try:
            if system == "Darwin":
                result = subprocess.run(
                    [
                        "system_profiler",
                        "SPDisplaysDataType",
                        "-detailLevel",
                        "mini",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode == 0 and (
                    "Chipset" in result.stdout or "Chip" in result.stdout
                ):
                    for line in result.stdout.split("\n"):
                        if "Chipset" in line or "Chip" in line:
                            return True, line.split(":")[-1].strip()

            # Try nvidia-smi for NVIDIA GPUs on any platform
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return True, result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("gpu_detection_failed", system=system)
        return False, ""

    @staticmethod
    def _check_ollama_installed() -> bool:
        """Check if Ollama is installed.

        Returns:
            ``True`` if the ``ollama`` binary is available.
        """
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    def _get_ollama_models() -> list[str]:
        """Get installed Ollama models.

        Returns:
            A list of model names, or an empty list on failure.
        """
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                models: list[str] = []
                for line in lines[1:]:
                    stripped = line.strip()
                    if stripped:
                        parts = stripped.split()
                        if parts:
                            models.append(parts[0])
                return models
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return []

    @staticmethod
    def _check_docker() -> bool:
        """Check if Docker is available.

        Returns:
            ``True`` if ``docker info`` succeeds.
        """
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

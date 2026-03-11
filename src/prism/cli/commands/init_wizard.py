"""Interactive setup wizard for first-time Prism users."""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from prism.auth.validator import KeyValidator
from prism.cost.pricing import MODEL_PRICING
from prism.providers.base import BUILTIN_PROVIDERS

if TYPE_CHECKING:
    from pathlib import Path

    from prism.auth.manager import AuthManager
    from prism.config.settings import Settings
    from prism.providers.registry import ProviderRegistry

logger = structlog.get_logger(__name__)


@dataclass
class EnvInfo:
    """Detected environment information."""

    os_name: str
    os_version: str
    python_version: str
    git_available: bool
    ollama_available: bool


@dataclass
class OllamaInfo:
    """Ollama installation status."""

    installed: bool
    running: bool
    models: list[str] = field(default_factory=list)


@dataclass
class InitResult:
    """Result of running the init wizard."""

    configured_providers: list[str]
    config_path: Path
    project_config_path: Path | None
    ollama_available: bool
    suggested_budget: float | None


class InitWizard:
    """Interactive setup wizard for first-time users.

    Steps:
    1. Detect OS and Python version
    2. Walk through API key setup for each provider
    3. Validate each key (offline format check only -- no real API calls)
    4. Detect Ollama installation
    5. Create ~/.prism/config.yaml with defaults
    6. Create .prism.yaml in current project if project root detected
    7. Show cost comparison of configured providers
    8. Suggest default budget limits
    """

    # Providers that require API keys (excludes ollama)
    _KEY_PROVIDERS: list[str] = [
        "anthropic",
        "openai",
        "google",
        "deepseek",
        "groq",
        "mistral",
    ]

    # Display names for providers
    _DISPLAY_NAMES: dict[str, str] = {
        "anthropic": "Anthropic (Claude)",
        "openai": "OpenAI (GPT-4o, o3)",
        "google": "Google AI Studio (Gemini)",
        "deepseek": "DeepSeek",
        "groq": "Groq (Llama, Mixtral)",
        "mistral": "Mistral",
    }

    def __init__(
        self,
        auth_manager: AuthManager,
        provider_registry: ProviderRegistry | None,
        settings: Settings,
        console: Console | None = None,
    ) -> None:
        self._auth = auth_manager
        self._registry = provider_registry
        self._settings = settings
        self._console = console or Console()
        self._validator = KeyValidator()

    def run(self, project_root: Path | None = None) -> InitResult:
        """Run the full wizard. Returns summary of what was set up."""
        self._console.print()
        self._console.print(
            Panel(
                "[bold cyan]Prism Setup Wizard[/bold cyan]\n"
                "[dim]Let's get you configured in a few simple steps.[/dim]",
                border_style="cyan",
            )
        )
        self._console.print()

        # Step 1: Detect environment
        env_info = self._detect_environment()
        self._display_env_info(env_info)

        # Step 2: Setup providers
        configured = self._setup_providers()

        # Step 3: Detect Ollama
        ollama_info = self._detect_ollama()
        if ollama_info.installed:
            self._console.print(
                "[green]Found Ollama[/green] installed on this system."
            )
            if ollama_info.running:
                self._console.print(
                    f"  Models available: {', '.join(ollama_info.models) or 'none detected'}"
                )
            else:
                self._console.print(
                    "  [yellow]Ollama is installed but not running.[/] "
                    "Start it with: [cyan]ollama serve[/]"
                )
        else:
            self._console.print(
                "[dim]Ollama not found.[/] Install it from https://ollama.com "
                "for free local models."
            )
        self._console.print()

        # Step 4: Create config files
        effective_root = project_root or self._settings.project_root
        config_path = self._create_config(configured)

        project_config_path: Path | None = None
        if effective_root and effective_root.is_dir():
            project_config_path = self._create_project_config(effective_root)

        # Step 5: Show cost comparison
        if configured:
            self._show_cost_comparison(configured)

        # Step 6: Suggest budget
        suggested_budget = self._suggest_budget(configured)

        self._console.print()
        self._console.print(
            Panel(
                "[bold green]Setup complete![/bold green]\n"
                f"Configured providers: {', '.join(configured) or 'none'}\n"
                f"Config: {config_path}\n"
                "[dim]Run [cyan]prism[/cyan] to start using Prism.[/dim]",
                border_style="green",
            )
        )

        return InitResult(
            configured_providers=configured,
            config_path=config_path,
            project_config_path=project_config_path,
            ollama_available=ollama_info.installed,
            suggested_budget=suggested_budget,
        )

    def _detect_environment(self) -> EnvInfo:
        """Detect OS, Python version, available tools."""
        os_name = platform.system()
        os_version = platform.version()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        git_available = shutil.which("git") is not None
        ollama_available = shutil.which("ollama") is not None

        return EnvInfo(
            os_name=os_name,
            os_version=os_version,
            python_version=python_version,
            git_available=git_available,
            ollama_available=ollama_available,
        )

    def _display_env_info(self, env: EnvInfo) -> None:
        """Display detected environment information."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("OS:", f"{env.os_name} ({env.os_version[:40]})")
        table.add_row("Python:", env.python_version)
        table.add_row(
            "Git:",
            "[green]available[/]" if env.git_available else "[yellow]not found[/]",
        )
        table.add_row(
            "Ollama:",
            "[green]installed[/]" if env.ollama_available else "[dim]not found[/]",
        )

        self._console.print(
            Panel(table, title="[bold]Environment[/bold]", border_style="blue")
        )
        self._console.print()

    def _setup_providers(self) -> list[str]:
        """Walk through each provider, ask for API key, validate format.

        Returns:
            List of configured provider names.
        """
        configured: list[str] = []
        self._console.print("[bold]API Key Setup[/bold]")
        self._console.print(
            "[dim]Configure the providers you want to use. "
            "You can skip any and add them later with 'prism auth add'.[/dim]\n"
        )

        for provider in self._KEY_PROVIDERS:
            display = self._DISPLAY_NAMES.get(provider, provider)

            want = Confirm.ask(
                f"  Configure [bold]{display}[/bold]?",
                default=False,
                console=self._console,
            )

            if not want:
                continue

            # Prompt for API key (masked)
            key: str = Prompt.ask(
                f"  Enter API key for {display}",
                password=True,
                console=self._console,
            )

            if not key.strip():
                self._console.print("  [yellow]Skipped (empty key).[/]")
                continue

            key = key.strip()

            # Validate format (offline only)
            if not self._validator.validate_key(provider, key):
                self._console.print(
                    f"  [yellow]Warning:[/] Key format doesn't match expected pattern for {provider}."
                )
                store_anyway = Confirm.ask(
                    "  Store anyway?",
                    default=False,
                    console=self._console,
                )
                if not store_anyway:
                    self._console.print("  [dim]Skipped.[/]")
                    continue

            # Store via auth_manager
            try:
                self._auth.store_key(provider, key, validate=False)
                masked = "..." + key[-4:] if len(key) > 4 else "****"
                self._console.print(
                    f"  [green]Stored[/] key for {display} ({masked})"
                )
                configured.append(provider)
            except Exception as exc:
                self._console.print(
                    f"  [red]Failed to store key:[/] {exc}"
                )
                logger.debug("init_store_key_failed", provider=provider, error=str(exc))

        self._console.print()
        return configured

    def _detect_ollama(self) -> OllamaInfo:
        """Check if Ollama is installed and running.

        This method checks for the ``ollama`` binary.  In production it would
        also probe the local API; in tests this is always mocked.
        """
        installed = shutil.which("ollama") is not None
        if not installed:
            return OllamaInfo(installed=False, running=False, models=[])

        # Try to detect running status by probing localhost:11434
        running = self._probe_ollama()
        models: list[str] = []
        if running:
            models = self._list_ollama_models()

        return OllamaInfo(installed=installed, running=running, models=models)

    def _probe_ollama(self) -> bool:
        """Probe whether Ollama is running on localhost.

        Returns ``False`` on any error.  MUST be mocked in tests.
        """
        try:
            import urllib.request

            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
                return resp.status == 200  # type: ignore[no-any-return]
        except Exception:
            return False

    def _list_ollama_models(self) -> list[str]:
        """List available Ollama models.  MUST be mocked in tests."""
        try:
            import json
            import urllib.request

            req = urllib.request.Request(
                "http://localhost:11434/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:  # noqa: S310
                data = json.loads(resp.read().decode())
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def _create_config(self, configured_providers: list[str]) -> Path:
        """Create ~/.prism/config.yaml with sensible defaults.

        If a config already exists it will NOT be overwritten.
        """
        config_path = self._settings.config_file_path
        if config_path.exists():
            self._console.print(
                f"[dim]Config already exists at {config_path}, not overwriting.[/]"
            )
            return config_path

        config_path.parent.mkdir(parents=True, exist_ok=True)

        default_config: dict[str, object] = {
            "routing": {
                "simple_threshold": 0.3,
                "medium_threshold": 0.7,
            },
            "budget": {
                "daily_limit": None,
                "monthly_limit": None,
            },
            "tools": {
                "web_enabled": False,
                "allowed_commands": ["python -m pytest"],
            },
        }

        # If user configured any premium provider, prefer it
        if configured_providers:
            default_config["preferred_provider"] = configured_providers[0]

        with config_path.open("w") as f:
            yaml.safe_dump(default_config, f, default_flow_style=False)

        self._console.print(f"[green]Created[/] {config_path}")
        return config_path

    def _create_project_config(self, project_root: Path) -> Path:
        """Create .prism.yaml in project root with project-specific settings.

        If the file already exists it will NOT be overwritten.
        """
        project_config = project_root / ".prism.yaml"
        if project_config.exists():
            self._console.print(
                f"[dim]Project config already exists at {project_config}, not overwriting.[/]"
            )
            return project_config

        project_name = project_root.name

        config: dict[str, object] = {
            "project": project_name,
            "routing": {
                "simple_threshold": 0.3,
                "medium_threshold": 0.7,
            },
            "tools": {
                "allowed_commands": [
                    "python -m pytest",
                    "make lint",
                ],
            },
        }

        with project_config.open("w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        self._console.print(f"[green]Created[/] {project_config}")
        return project_config

    def _show_cost_comparison(self, configured_providers: list[str]) -> None:
        """Display a cost comparison table for configured providers."""
        table = Table(title="Cost Comparison (per 1M tokens)")
        table.add_column("Provider", style="bold")
        table.add_column("Model", style="cyan")
        table.add_column("Input", justify="right", style="green")
        table.add_column("Output", justify="right", style="yellow")

        for provider_config in BUILTIN_PROVIDERS:
            if provider_config.name not in configured_providers and provider_config.name != "ollama":
                    continue
            for model in provider_config.models:
                table.add_row(
                    provider_config.display_name,
                    model.display_name,
                    f"${model.input_cost_per_1m:.2f}",
                    f"${model.output_cost_per_1m:.2f}",
                )

        self._console.print()
        self._console.print(table)

    def _suggest_budget(self, configured_providers: list[str]) -> float | None:
        """Suggest a daily budget based on configured providers.

        Returns the suggested budget or None if user declines.
        """
        if not configured_providers:
            return None

        # Find cheapest model cost among configured providers for baseline
        min_cost = float("inf")
        for _model_id, pricing in MODEL_PRICING.items():
            if pricing.provider in configured_providers:
                total = pricing.input_cost_per_1m + pricing.output_cost_per_1m
                min_cost = min(min_cost, total)

        # Suggest a daily budget: ~100 requests with cheapest model
        # assuming 1k input + 1k output tokens per request
        if min_cost == float("inf") or min_cost == 0:
            suggested = 5.00
        else:
            per_request_cost = min_cost * 2 / 1_000_000 * 1000  # 1k tokens
            suggested = round(max(1.00, per_request_cost * 100), 2)

        self._console.print(
            f"\n[bold]Suggested daily budget:[/bold] ${suggested:.2f}"
        )
        self._console.print(
            "[dim]This covers ~100 requests with your cheapest provider.[/dim]"
        )

        return suggested

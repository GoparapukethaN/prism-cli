"""Model comparison mode — run same prompt across multiple models and compare.

Provides the /compare command which sends an identical prompt to 2-5 models
in parallel, displays results side-by-side in a rich terminal layout, shows
token counts and costs, and lets the user pick the best response.  The winning
choice is logged as training data for the router's adaptive learning.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import structlog
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

logger = structlog.get_logger(__name__)

# ── Default comparison set ─────────────────────────────────────────────────
DEFAULT_COMPARISON_MODELS: list[str] = [
    "claude-sonnet-4-20250514",
    "gpt-4o",
    "deepseek/deepseek-chat",
]

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "gpt-4o": "GPT-4o",
    "deepseek/deepseek-chat": "DeepSeek V3",
    "gemini/gemini-1.5-pro": "Gemini 1.5 Pro",
    "gemini/gemini-1.5-flash": "Gemini Flash",
    "claude-haiku-3-20240307": "Claude Haiku 3",
    "gpt-4o-mini": "GPT-4o Mini",
    "mistral/mistral-large-latest": "Mistral Large",
    "groq/llama-3.1-70b-versatile": "Llama 3.1 70B (Groq)",
}

# Panel border colours cycled for visual distinction
_PANEL_COLORS: list[str] = ["blue", "green", "yellow", "magenta", "cyan"]

# Minimum / maximum models allowed in a comparison set
_MIN_MODELS = 2
_MAX_MODELS = 5

# Terminal width threshold for side-by-side layout
_SIDE_BY_SIDE_MIN_WIDTH = 120
_SIDE_BY_SIDE_MAX_PANELS = 3


# ── Completion engine protocol ─────────────────────────────────────────────
class CompletionEngineProtocol(Protocol):
    """Minimal interface a completion engine must satisfy for comparisons."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> Any:  # Returns CompletionResult or compatible object
        ...  # pragma: no cover


# ── Data classes ───────────────────────────────────────────────────────────
@dataclass
class ComparisonResult:
    """Result from a single model in a comparison run."""

    model: str
    display_name: str
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return ``True`` if the model produced a response without error."""
        return self.error is None

    @property
    def total_tokens(self) -> int:
        """Combined input + output token count."""
        return self.input_tokens + self.output_tokens


@dataclass
class ComparisonSession:
    """A complete comparison run across multiple models."""

    prompt: str
    results: list[ComparisonResult] = field(default_factory=list)
    winner_index: int | None = None
    created_at: str = ""
    system_prompt: str = ""

    @property
    def has_winner(self) -> bool:
        """Return ``True`` if the user has picked a winner."""
        return self.winner_index is not None

    @property
    def winner(self) -> ComparisonResult | None:
        """Return the winning result, or ``None`` if not yet chosen."""
        if self.winner_index is not None and 0 <= self.winner_index < len(self.results):
            return self.results[self.winner_index]
        return None

    @property
    def total_cost(self) -> float:
        """Sum of costs across all models in this comparison."""
        return sum(r.cost_usd for r in self.results)

    @property
    def successful_count(self) -> int:
        """Number of models that succeeded."""
        return sum(1 for r in self.results if r.succeeded)


# ── Main comparator ───────────────────────────────────────────────────────
class ModelComparator:
    """Runs prompts across multiple models and displays side-by-side results.

    The comparator is stateful — it keeps a history of all comparison sessions
    so that aggregate statistics can be shown later.

    Args:
        completion_engine: Any object satisfying ``CompletionEngineProtocol``.
        models: Initial comparison model set.  Defaults to
            ``DEFAULT_COMPARISON_MODELS``.
        console: Rich console instance for display.
    """

    def __init__(
        self,
        completion_engine: CompletionEngineProtocol,
        models: list[str] | None = None,
        console: Console | None = None,
    ) -> None:
        self._engine = completion_engine
        self._models = list(models) if models is not None else list(DEFAULT_COMPARISON_MODELS)
        self._console = console or Console()
        self._history: list[ComparisonSession] = []

        # Validate at construction time
        if len(self._models) < _MIN_MODELS:
            msg = f"Need at least {_MIN_MODELS} models for comparison"
            raise ValueError(msg)
        if len(self._models) > _MAX_MODELS:
            msg = f"Maximum {_MAX_MODELS} models for comparison"
            raise ValueError(msg)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def models(self) -> list[str]:
        """Return a copy of the current comparison model set."""
        return list(self._models)

    @models.setter
    def models(self, value: list[str]) -> None:
        """Set the comparison model set with validation.

        Args:
            value: New list of model identifiers.

        Raises:
            ValueError: If fewer than 2 or more than 5 models supplied.
        """
        if not isinstance(value, list):
            msg = "models must be a list of strings"
            raise TypeError(msg)
        if len(value) < _MIN_MODELS:
            msg = f"Need at least {_MIN_MODELS} models for comparison"
            raise ValueError(msg)
        if len(value) > _MAX_MODELS:
            msg = f"Maximum {_MAX_MODELS} models for comparison"
            raise ValueError(msg)
        self._models = list(value)

    @property
    def history(self) -> list[ComparisonSession]:
        """Return the full history of comparison sessions (read-only copy)."""
        return list(self._history)

    @property
    def console(self) -> Console:
        """Return the Rich console instance."""
        return self._console

    # ── Core comparison ────────────────────────────────────────────────────

    async def compare(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> ComparisonSession:
        """Run *prompt* on all configured models in parallel.

        Args:
            prompt: The user's question or task.
            system_prompt: Optional system-level instruction.

        Returns:
            A ``ComparisonSession`` containing one ``ComparisonResult``
            per model.
        """
        if not prompt or not prompt.strip():
            msg = "Prompt must not be empty"
            raise ValueError(msg)

        session = ComparisonSession(
            prompt=prompt,
            system_prompt=system_prompt,
            created_at=datetime.now(UTC).isoformat(),
        )

        tasks = [
            self._run_single(model, prompt, system_prompt)
            for model in self._models
        ]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, raw in enumerate(raw_results):
            if isinstance(raw, BaseException):
                session.results.append(ComparisonResult(
                    model=self._models[i],
                    display_name=MODEL_DISPLAY_NAMES.get(
                        self._models[i], self._models[i],
                    ),
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    error=str(raw),
                ))
            else:
                session.results.append(raw)

        self._history.append(session)

        logger.info(
            "comparison_complete",
            models=[r.model for r in session.results],
            succeeded=session.successful_count,
            total_cost=session.total_cost,
        )

        return session

    async def _run_single(
        self,
        model: str,
        prompt: str,
        system_prompt: str,
    ) -> ComparisonResult:
        """Execute a single model completion and wrap the outcome.

        Args:
            model: LiteLLM model identifier.
            prompt: User prompt text.
            system_prompt: System-level instruction.

        Returns:
            ``ComparisonResult`` with either content or error populated.
        """
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        start = time.monotonic()

        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            result = await self._engine.complete(messages=messages, model=model)

            latency = (time.monotonic() - start) * 1000

            # Support both CompletionResult-like objects and raw dicts
            content = getattr(result, "content", "")
            input_tokens = getattr(result, "input_tokens", 0)
            output_tokens = getattr(result, "output_tokens", 0)
            cost_usd = getattr(result, "cost_usd", 0.0)

            return ComparisonResult(
                model=model,
                display_name=display_name,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.warning(
                "comparison_model_failed",
                model=model,
                error=str(exc),
                latency_ms=latency,
            )
            return ComparisonResult(
                model=model,
                display_name=display_name,
                content="",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=latency,
                error=str(exc),
            )

    # ── Display helpers ────────────────────────────────────────────────────

    def display_results(self, session: ComparisonSession) -> None:
        """Render comparison results as rich panels in the terminal.

        Uses side-by-side ``Columns`` when the terminal is wide enough
        (>= 120 cols) and there are at most 3 models; otherwise stacks
        panels vertically.

        Args:
            session: The comparison session to display.
        """
        if not session.results:
            self._console.print("[yellow]No results to display.[/yellow]")
            return

        panels: list[Panel] = []
        for i, result in enumerate(session.results):
            if result.error:
                body: Text | Markdown = Text(f"Error: {result.error}", style="red")
            else:
                body = Markdown(result.content) if result.content else Text(
                    "(empty response)", style="dim",
                )

            subtitle = (
                f"Tokens: {result.input_tokens}\u2192{result.output_tokens} | "
                f"Cost: ${result.cost_usd:.4f} | "
                f"Latency: {result.latency_ms:.0f}ms"
            )

            color = _PANEL_COLORS[i % len(_PANEL_COLORS)]
            panel = Panel(
                body,
                title=f"[{i + 1}] {result.display_name}",
                subtitle=subtitle,
                border_style=color,
                expand=True,
            )
            panels.append(panel)

        width = self._console.size.width
        if width >= _SIDE_BY_SIDE_MIN_WIDTH and len(panels) <= _SIDE_BY_SIDE_MAX_PANELS:
            self._console.print(Columns(panels, equal=True, expand=True))
        else:
            for panel in panels:
                self._console.print(panel)

    def display_cost_table(self, session: ComparisonSession) -> None:
        """Render a cost / performance comparison table.

        Args:
            session: The comparison session to display.
        """
        if not session.results:
            self._console.print("[yellow]No results to display.[/yellow]")
            return

        table = Table(
            title="Cost Comparison",
            show_header=True,
            header_style="bold",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Model", min_width=20)
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Total Tokens", justify="right")
        table.add_column("Cost", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Status", justify="center")

        for i, result in enumerate(session.results):
            status = "[green]OK[/green]" if result.succeeded else "[red]FAIL[/red]"
            table.add_row(
                str(i + 1),
                result.display_name,
                str(result.input_tokens),
                str(result.output_tokens),
                str(result.total_tokens),
                f"${result.cost_usd:.4f}",
                f"{result.latency_ms:.0f}ms",
                status,
            )

        # Total row
        total_input = sum(r.input_tokens for r in session.results)
        total_output = sum(r.output_tokens for r in session.results)
        total_all = total_input + total_output
        total_cost = session.total_cost
        table.add_row(
            "",
            "[bold]Total[/bold]",
            str(total_input),
            str(total_output),
            str(total_all),
            f"[bold]${total_cost:.4f}[/bold]",
            "",
            "",
            style="dim",
        )

        self._console.print(table)

    def display_prompt_hint(self, session: ComparisonSession) -> None:
        """Show a prompt asking the user to pick the best response.

        Args:
            session: The comparison session (used to know how many models).
        """
        count = len(session.results)
        if count == 0:
            return
        keys = "/".join(str(i + 1) for i in range(count))
        self._console.print(
            f"\n[bold]Pick the best response ({keys})[/bold] "
            "[dim]or press Enter to skip:[/dim]"
        )

    # ── Winner recording ───────────────────────────────────────────────────

    def record_winner(self, session: ComparisonSession, choice: int) -> None:
        """Record the user's preferred model as training data.

        Args:
            session: The comparison session to annotate.
            choice: 1-based index of the chosen model.

        Raises:
            ValueError: If *choice* is out of range.
        """
        if not session.results:
            msg = "Cannot record winner for a session with no results"
            raise ValueError(msg)
        if choice < 1 or choice > len(session.results):
            msg = (
                f"Invalid choice: {choice}. "
                f"Must be 1-{len(session.results)}"
            )
            raise ValueError(msg)

        session.winner_index = choice - 1
        winner = session.results[session.winner_index]

        logger.info(
            "comparison_winner",
            prompt_preview=session.prompt[:80],
            winner_model=winner.model,
            winner_display=winner.display_name,
            winner_cost=winner.cost_usd,
            winner_latency=winner.latency_ms,
            models=[r.model for r in session.results],
        )

        self._console.print(
            f"\n[green]Winner recorded:[/green] [bold]{winner.display_name}[/bold] "
            f"(${winner.cost_usd:.4f}, {winner.latency_ms:.0f}ms)"
        )

    # ── Configuration helpers ──────────────────────────────────────────────

    def get_config_summary(self) -> str:
        """Return a human-readable summary of the current comparison config.

        Returns:
            Multi-line string listing each model with its display name.
        """
        lines: list[str] = ["Comparison models:"]
        for i, model in enumerate(self._models):
            name = MODEL_DISPLAY_NAMES.get(model, model)
            lines.append(f"  {i + 1}. {name} ({model})")
        return "\n".join(lines)

    def display_config(self) -> None:
        """Print the current comparison configuration to the console."""
        self._console.print(f"\n[bold]{self.get_config_summary()}[/bold]\n")

"""Abstract base class for all Prism tools and shared data types."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class PermissionLevel(enum.Enum):
    """Permission level required to execute a tool.

    AUTO:      No user confirmation needed (read-only operations).
    CONFIRM:   User must approve before execution (writes, commands).
    DANGEROUS: Always requires explicit confirmation, even in ``--yes`` mode.
    """

    AUTO = "auto"
    CONFIRM = "confirm"
    DANGEROUS = "dangerous"


@dataclass
class ToolResult:
    """Result returned by every tool execution.

    Attributes:
        success:  Whether the tool executed successfully.
        output:   The tool output (file contents, command output, etc.).
        error:    Error message when ``success`` is ``False``.
        metadata: Extra structured data (bytes written, exit code, etc.).
    """

    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)


class Tool(ABC):
    """Abstract base class for all Prism tools.

    Subclasses must define the four abstract properties and implement
    :meth:`execute`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (e.g., ``'read_file'``)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the model."""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema describing the tool's parameters."""

    @property
    @abstractmethod
    def permission_level(self) -> PermissionLevel:
        """Permission level required to run this tool."""

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given *arguments*.

        Args:
            arguments: A dictionary of tool arguments, conforming to
                       :attr:`parameters_schema`.

        Returns:
            A :class:`ToolResult` describing the outcome.
        """

    # ------------------------------------------------------------------
    # Argument validation
    # ------------------------------------------------------------------

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate *arguments* against :attr:`parameters_schema`.

        Checks:
        - All ``required`` keys are present.
        - Supplied values match their declared ``type``.
        - Unknown keys are rejected.

        Returns:
            The validated (and possibly defaulted) arguments dict.

        Raises:
            ValueError: When validation fails.
        """
        schema = self.parameters_schema
        properties: dict[str, Any] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        # Check for missing required keys
        for key in required:
            if key not in arguments:
                raise ValueError(
                    f"Missing required argument '{key}' for tool '{self.name}'"
                )

        # Check for unknown keys
        if properties:
            unknown = set(arguments.keys()) - set(properties.keys())
            if unknown:
                raise ValueError(
                    f"Unknown arguments for tool '{self.name}': {sorted(unknown)}"
                )

        # Basic type validation
        json_type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "integer": (int,),
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        validated: dict[str, Any] = {}
        for key, prop_schema in properties.items():
            if key not in arguments:
                # Apply default if available
                if "default" in prop_schema:
                    validated[key] = prop_schema["default"]
                continue

            value = arguments[key]
            expected_type_name = prop_schema.get("type")
            if expected_type_name and value is not None:
                expected_types = json_type_map.get(expected_type_name)
                if expected_types and not isinstance(value, expected_types):
                    raise ValueError(
                        f"Argument '{key}' for tool '{self.name}' "
                        f"must be {expected_type_name}, got {type(value).__name__}"
                    )
            validated[key] = value

        return validated

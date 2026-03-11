"""Prism Intelligence — adaptive execution and causal analysis modules.

Provides:
- :class:`AdaptiveExecutionIntelligence` — learns fix strategies from past attempts
- :class:`CausalBlameTracer` — automated git bisect with causal analysis
- :class:`CodeArchaeologist` — temporal code archaeology and evolution tracing
- :class:`DependencyMonitor` — autonomous dependency health monitoring
- :class:`MultiModelDebate` — structured multi-model deliberation
- :class:`ArchitectureMapper` — living architecture map with drift detection
- :class:`DebugMemory` — cross-session debugging memory
"""

from __future__ import annotations

from prism.intelligence.aei import (
    AdaptiveExecutionIntelligence,
    AEIStats,
    AttemptRecord,
    ErrorFingerprint,
    FixStrategy,
    StrategyRecommendation,
)
from prism.intelligence.archaeologist import (
    AuthorContribution,
    CodeArchaeologist,
    CodeEvolution,
    CommitEvent,
)
from prism.intelligence.architecture import (
    ArchitectureMapper,
    ArchitectureState,
    DependencyEdge,
    DriftViolation,
    ModuleInfo,
)
from prism.intelligence.blame import (
    BisectResult,
    BlameReport,
    CausalBlameTracer,
)
from prism.intelligence.debate import (
    DebateCritique,
    DebatePosition,
    DebateSession,
    DebateSynthesis,
    MultiModelDebate,
)
from prism.intelligence.debug_memory import (
    BugFingerprint,
    DebugMemory,
    FixRecord,
    FixSuggestion,
)
from prism.intelligence.deps import (
    DependencyInfo,
    DependencyMonitor,
    DepsReport,
    MigrationComplexity,
    Vulnerability,
    VulnerabilitySeverity,
)

__all__ = [
    "AEIStats",
    "AdaptiveExecutionIntelligence",
    "ArchitectureMapper",
    "ArchitectureState",
    "AttemptRecord",
    "AuthorContribution",
    "BisectResult",
    "BlameReport",
    "BugFingerprint",
    "CausalBlameTracer",
    "CodeArchaeologist",
    "CodeEvolution",
    "CommitEvent",
    "DebateCritique",
    "DebatePosition",
    "DebateSession",
    "DebateSynthesis",
    "DebugMemory",
    "DependencyEdge",
    "DependencyInfo",
    "DependencyMonitor",
    "DepsReport",
    "DriftViolation",
    "ErrorFingerprint",
    "FixRecord",
    "FixStrategy",
    "FixSuggestion",
    "MigrationComplexity",
    "ModuleInfo",
    "MultiModelDebate",
    "StrategyRecommendation",
    "Vulnerability",
    "VulnerabilitySeverity",
]

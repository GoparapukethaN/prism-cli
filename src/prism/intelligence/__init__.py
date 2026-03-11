"""Prism Intelligence — adaptive execution and causal analysis modules.

Provides:
- :class:`AdaptiveExecutionIntelligence` — learns fix strategies from past attempts
- :class:`CausalBlameTracer` — automated git bisect with causal analysis
- :class:`CodeArchaeologist` — temporal code archaeology and evolution tracing
- :class:`DependencyMonitor` — autonomous dependency health monitoring
- :class:`MultiModelDebate` — structured multi-model deliberation
- :class:`ArchitectureMapper` — living architecture map with drift detection
- :class:`DebugMemory` — cross-session debugging memory
- :class:`SmartContextBudgetManager` — smart context budget allocation
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
    ArchaeologyReport,
    AuthorContribution,
    CodeArchaeologist,
    CodeEvolution,
    CommitEvent,
    CommitInfo,
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
from prism.intelligence.context_budget import (
    BudgetAllocation,
    ContextEfficiencyRecord,
    ContextItem,
    EfficiencyStats,
    RelevanceLevel,
    SmartContextBudgetManager,
)
from prism.intelligence.debate import (
    DebateConfig,
    DebateCritique,
    DebatePosition,
    DebateResult,
    DebateRound,
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
    DependencyStatusReport,
    DepsReport,
    MigrationComplexity,
    Vulnerability,
    VulnerabilityReport,
    VulnerabilitySeverity,
    assess_migration_by_version,
)

__all__ = [
    "AEIStats",
    "AdaptiveExecutionIntelligence",
    "ArchaeologyReport",
    "ArchitectureMapper",
    "ArchitectureState",
    "AttemptRecord",
    "AuthorContribution",
    "BisectResult",
    "BlameReport",
    "BudgetAllocation",
    "BugFingerprint",
    "CausalBlameTracer",
    "CodeArchaeologist",
    "CodeEvolution",
    "CommitEvent",
    "CommitInfo",
    "ContextEfficiencyRecord",
    "ContextItem",
    "DebateConfig",
    "DebateCritique",
    "DebatePosition",
    "DebateResult",
    "DebateRound",
    "DebateSession",
    "DebateSynthesis",
    "DebugMemory",
    "DependencyEdge",
    "DependencyInfo",
    "DependencyMonitor",
    "DependencyStatusReport",
    "DepsReport",
    "DriftViolation",
    "EfficiencyStats",
    "ErrorFingerprint",
    "FixRecord",
    "FixStrategy",
    "FixSuggestion",
    "MigrationComplexity",
    "ModuleInfo",
    "MultiModelDebate",
    "RelevanceLevel",
    "SmartContextBudgetManager",
    "StrategyRecommendation",
    "Vulnerability",
    "VulnerabilityReport",
    "VulnerabilitySeverity",
    "assess_migration_by_version",
]

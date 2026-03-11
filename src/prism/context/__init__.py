"""Prism context management.

Public API:

- :class:`ContextManager` — assembles context for LLM calls.
- :func:`generate_repo_map` — generates a compressed repo map.
- :func:`summarize` — summarizes old conversation history.
- :class:`SessionManager` — session persistence to disk.
- :class:`ProjectMemory` — key-value project memory.
- :class:`BranchManager` — conversation branching.
"""

from __future__ import annotations

from prism.context.branching import BranchManager, BranchMetadata, ConversationBranch
from prism.context.manager import ContextBudget, ContextManager, Message, estimate_tokens
from prism.context.memory import ProjectMemory
from prism.context.repo_map import generate_repo_map, invalidate_cache
from prism.context.session import SessionManager
from prism.context.summarizer import summarize

__all__ = [
    "BranchManager",
    "BranchMetadata",
    "ContextBudget",
    "ContextManager",
    "ConversationBranch",
    "Message",
    "ProjectMemory",
    "SessionManager",
    "estimate_tokens",
    "generate_repo_map",
    "invalidate_cache",
    "summarize",
]

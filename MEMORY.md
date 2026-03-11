# Prism Project Memory

## Project Identity
- **Name**: Prism — Multi-API Intelligent Router CLI
- **Working directory**: `/Users/kethangoparapu/Downloads/Multi-API Intelligent Router CLI/`
- **License**: Apache 2.0
- **Python**: 3.11+ required
- **Package name**: prism-cli (PyPI), prism (import)

## Architecture Decisions
- LiteLLM as unified provider abstraction — single `completion()` interface for 100+ models
- tree-sitter for repository maps — AST parsing across 40+ languages for compressed codebase context
- SQLite for all persistent data — routing history, cost tracking, learning data, audit logs
- keyring for credential storage — OS-native encryption (macOS Keychain, Windows Credential Manager)
- Typer + Rich + Prompt Toolkit for CLI — type-safe args, beautiful output, multi-line editing
- ChromaDB optional for RAG on large codebases
- Search/replace block format for code edits (proven by Aider)
- Architect mode: premium model plans, cheap model executes (60-80% cost reduction)

## Routing Tiers
- **Simple (score < 0.3)**: Ollama local, Gemini Flash free, Groq free → $0.00
- **Medium (score 0.3-0.7)**: DeepSeek V3, GPT-4o-mini, Groq paid, Mistral → pennies
- **Complex (score > 0.7)**: Claude Sonnet/Opus, GPT-4o, o3, Gemini Pro → dollars

## Key Patterns
- Every file operation resolves through realpath() before execution
- Every tool execution logged to audit.log with timestamps
- Every API call passes through cost tracker before and after
- Fallback chain: primary → same-tier alt → cheaper tier → Ollama local
- Budget enforcement checked before every API call
- 10% exploration rounds for adaptive learning after 100+ data points

## Current State
- **Phase**: Project setup — instruction files created, no code yet
- **Next**: Begin Phase 1 implementation (core CLI + basic routing)

## Detailed Reference Files
- `ARCHITECTURE.md` — full system architecture and component diagram
- `ROUTING_LOGIC.md` — complete routing algorithm and classification details
- `PROVIDER_SPECS.md` — all provider configs, pricing, rate limits
- `COST_LOGIC.md` — cost estimation, tracking, budget enforcement
- `TOOL_SPECS.md` — all tool interfaces and implementations
- `DATA_MODELS.md` — database schema and data structures
- `SECURITY.md` — security model, threat mitigations
- `API_CONTRACTS.md` — internal API contracts between modules
- `PROGRESS.md` — task completion tracking

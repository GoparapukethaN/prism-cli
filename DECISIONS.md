# DECISIONS.md — Prism Architecture Decision Records

## Format
Each decision follows this template:
- **Context**: What is the situation?
- **Decision**: What did we decide?
- **Rationale**: Why?
- **Consequences**: What are the trade-offs?

---

## ADR-001: Python as Primary Language

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Need to choose implementation language for an AI coding CLI. Options considered: Python, TypeScript, Rust, Go.

**Decision**: Python 3.11+

**Rationale**:
- LiteLLM is Python-native — no FFI or subprocess overhead
- Richest AI library ecosystem (tree-sitter, ChromaDB, etc.)
- Aider proved Python works excellently for AI coding CLIs at scale (25K+ stars)
- Rapid development velocity — important for early iteration
- Async support mature enough for I/O-bound workloads

**Consequences**:
- Startup time slower than Rust/Go (mitigated with lazy imports)
- No compile-time type safety (mitigated with mypy --strict)
- GIL limits true parallelism (acceptable — workload is I/O bound)

---

## ADR-002: LiteLLM for Provider Abstraction

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Need a unified interface to call 100+ AI models across different providers.

**Decision**: Use LiteLLM as the provider abstraction layer.

**Rationale**:
- Single `completion()` interface for all providers
- Handles provider-specific auth, request formatting, response parsing
- Built-in cost tracking and token counting
- Active development, wide community adoption
- Supports all target providers (Anthropic, OpenAI, Google, DeepSeek, Groq, Mistral, Ollama)

**Consequences**:
- Heavy dependency (~50+ sub-imports)
- Model ID format tied to LiteLLM conventions
- Breaking changes in LiteLLM could break Prism
- Mitigated by pinning version and isolating behind our own interface

---

## ADR-003: SQLite for All Persistent Data

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Need persistent storage for routing history, cost tracking, learning data, sessions, and audit logs.

**Decision**: SQLite via Python stdlib `sqlite3`. No ORM.

**Rationale**:
- Zero external dependencies (stdlib)
- Single file — easy backup, migration, deletion
- WAL mode supports concurrent reads with single writer
- More than fast enough for our workload (thousands of entries, not millions)
- Direct SQL gives full control over queries and schema
- No server process to manage

**Consequences**:
- No concurrent writers (fine — single CLI process)
- Must manage schema migrations manually
- No built-in query builder (raw SQL in queries.py)
- Large databases (100K+ entries) may need periodic VACUUM

---

## ADR-004: Keyring for Credential Storage (with fallbacks)

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Need to securely store API keys for multiple providers.

**Decision**: Three-tier credential storage: OS keyring → environment variables → AES-256 encrypted file.

**Rationale**:
- OS keyring is the most secure option (hardware-backed on macOS, encrypted on Windows)
- Environment variables are the industry standard for CI/CD and containers
- Encrypted file fallback covers headless Linux and Docker
- No single approach works on all platforms

**Consequences**:
- `keyring` library can be flaky on some platforms (headless Linux)
- Three code paths to maintain and test
- User must understand which storage is in use

---

## ADR-005: Rule-Based Classification First, ML Later

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Need to classify task complexity to route to appropriate models.

**Decision**: Start with rule-based keyword/heuristic classification. Add ML (logistic regression) after 100+ data points.

**Rationale**:
- Rule-based works immediately without training data
- Transparent — users can understand why a task was classified a certain way
- ML layer adds value only after sufficient outcome data
- Logistic regression is simple, fast, interpretable, and trains locally
- Avoids cold-start problem

**Consequences**:
- Early routing may be suboptimal
- Keyword scoring requires tuning
- Need to collect outcome data before ML can improve decisions

---

## ADR-006: Search/Replace for Code Edits

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Need a reliable format for models to specify code edits.

**Decision**: Search/replace blocks where the model specifies exact text to find and replacement text.

**Rationale**:
- Proven by Aider (25K+ stars, years of production use)
- More reliable than whole-file rewrites (less chance of corruption)
- Works across all models (simple format, not model-dependent)
- Easy to show colored diffs for user review
- Exact matching prevents ambiguity

**Consequences**:
- Models must produce exact matches (whitespace-sensitive)
- Smaller models may fail more often at exact matching
- Need fuzzy matching fallback for robustness

---

## ADR-007: No Telemetry by Default

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Collecting usage data would improve routing decisions. But users are privacy-conscious.

**Decision**: Zero telemetry, zero phone-home, zero crash reporting by default. All data stays local. Optional opt-in for community routing intelligence in Phase 4.

**Rationale**:
- Trust is non-negotiable for a tool with filesystem access
- Open source community expects local-first tools
- Regulatory compliance (GDPR, etc.) simplified
- Local-only learning still works well for individual users

**Consequences**:
- Cannot aggregate routing intelligence across users (until opt-in Phase 4)
- Cannot measure adoption without external analytics
- Must provide excellent local experience without server support

---

## ADR-008: Apache 2.0 License

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Need an open-source license.

**Decision**: Apache 2.0

**Rationale**:
- Permissive enough for enterprise adoption
- Patent protection (unlike MIT)
- Matches Aider, Cline, and Continue.dev
- Ecosystem compatibility

**Consequences**:
- Anyone can fork and create commercial derivatives
- Must include license and notice in distributions

---

## ADR-009: Architect Mode for Complex Tasks

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Complex tasks are expensive when using premium models end-to-end.

**Decision**: Split complex tasks into planning (premium model) and execution (cheap model) phases.

**Rationale**:
- Planning requires intelligence, execution requires following instructions
- 60-80% cost reduction on complex tasks
- Proven pattern (Aider's architect mode)
- Premium model's plan provides structure that cheap models can follow reliably

**Consequences**:
- More complex orchestration code
- Potential quality loss if cheap model misinterprets plan
- Need good plan format that cheap models understand consistently
- Added latency (two API calls instead of one)

---

## ADR-010: All Tests Offline — No Real API Calls

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Tests need to verify provider interactions without using real API keys or making real requests.

**Decision**: All tests run completely offline. Every provider interaction is mocked. No real API keys in test environment.

**Rationale**:
- Tests must be reproducible and fast
- API keys are secrets that should never appear in test code
- Real API calls are slow, flaky, and cost money
- CI/CD must run without any API access
- User explicitly requested this

**Consequences**:
- Must maintain comprehensive mock responses
- Integration tests don't catch real API behavior changes
- Manual testing with real keys needed before releases (done by user only)

---

## ADR-011: GitHub Account GoparapukethaN

**Date**: 2026-03-10
**Status**: Accepted

**Context**: Need a GitHub account for the repository.

**Decision**: Use GoparapukethaN as the GitHub account owner.

**Rationale**: User's account.

**Consequences**: Repository URL will be github.com/GoparapukethaN/prism-cli.

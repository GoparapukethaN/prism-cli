# CHANGELOG.md — Prism Release History

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project instruction files (30 files covering architecture, security, testing, etc.)
- Product plan documentation

### Infrastructure
- Git repository initialized
- Project structure defined in CLAUDE.md
- All coding conventions established in CONVENTIONS.md
- CI/CD pipeline designed in CI_CD.md
- Security model documented in SECURITY.md

---

## Version History Format

Each release entry follows this format:

### [X.Y.Z] - YYYY-MM-DD

#### Added
- New features

#### Changed
- Changes to existing functionality

#### Deprecated
- Features that will be removed in future versions

#### Removed
- Features removed in this version

#### Fixed
- Bug fixes

#### Security
- Security-related changes

---

## Planned Releases

### [0.1.0] - Target: End of Phase 1 Month 3
- Interactive REPL with multi-model routing
- File read/write/edit tools with permission controls
- Task complexity classification (rule-based)
- Cost-based model selection with fallback chains
- Budget enforcement (daily/monthly limits)
- Cost tracking dashboard (`/cost`)
- tree-sitter repository maps
- Git integration (auto-commit, `/undo`)
- `prism init` setup wizard
- `prism auth add/status` commands
- Support for: Anthropic, OpenAI, Google, DeepSeek, Groq, Mistral, Ollama
- PyPI release as `prism-cli`

### [0.2.0] - Target: End of Phase 2
- Terminal execution (sandboxed `execute_command`)
- Web browsing (Playwright integration)
- Screenshot tool (multimodal)
- Architect mode (premium plans, cheap executes)
- Rolling conversation summarization (`/compact`)
- Prompt caching (Anthropic)
- Session persistence and resume
- Optional ChromaDB RAG

### [0.3.0] - Target: End of Phase 3
- Adaptive routing (ML-based classifier)
- Outcome tracking and learning
- Exploration rounds
- Parallel tool execution
- Model-specific prompt optimization
- Context-window-aware routing

### [1.0.0] - Target: End of Phase 4
- Plugin system with Tool interface
- Plugin registry and installer
- Community routing intelligence (opt-in)
- Web dashboard for analytics
- Enterprise features (teams, SSO)
- Full documentation and stable API

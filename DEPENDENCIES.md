# DEPENDENCIES.md — Prism Dependency Management

## Core Dependencies

### Runtime Dependencies
| Package | Version | Purpose | Required? |
|---------|---------|---------|-----------|
| typer | ≥0.9.0 | CLI framework | Yes |
| rich | ≥13.0.0 | Terminal UI rendering | Yes |
| prompt-toolkit | ≥3.0.0 | Interactive REPL input | Yes |
| litellm | ≥1.0.0 | Unified AI API interface | Yes |
| keyring | ≥25.0.0 | OS-native credential storage | Yes |
| httpx | ≥0.27.0 | Async HTTP client | Yes |
| pydantic | ≥2.0.0 | Data validation and settings | Yes |
| structlog | ≥24.0.0 | Structured logging | Yes |
| click | ≥8.0.0 | Required by Typer | Yes (transitive) |

### Optional Runtime Dependencies
| Package | Version | Purpose | Install Extra |
|---------|---------|---------|---------------|
| tree-sitter | ≥0.22.0 | AST parsing for repo maps | `.[analysis]` |
| tree-sitter-languages | ≥1.10.0 | Language grammars | `.[analysis]` |
| playwright | ≥1.40.0 | Web browsing and screenshots | `.[web]` |
| beautifulsoup4 | ≥4.12.0 | HTML content extraction | `.[web]` |
| chromadb | ≥0.5.0 | Vector search RAG | `.[rag]` |
| cryptography | ≥42.0.0 | Encrypted credential storage | `.[crypto]` |
| unidiff | ≥0.7.0 | Unified diff parsing | Yes |
| watchdog | ≥4.0.0 | File change detection | Yes |

### Development Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| pytest | ≥8.0.0 | Test runner |
| pytest-cov | ≥5.0.0 | Coverage reporting |
| pytest-asyncio | ≥0.23.0 | Async test support |
| pytest-mock | ≥3.12.0 | Mocking via mocker fixture |
| pytest-xdist | ≥3.5.0 | Parallel test execution |
| hypothesis | ≥6.100.0 | Property-based testing |
| respx | ≥0.21.0 | httpx request mocking |
| ruff | ≥0.3.0 | Linting and formatting |
| mypy | ≥1.8.0 | Static type checking |
| bandit | ≥1.7.0 | Security scanning |
| pip-audit | ≥2.7.0 | Dependency vulnerability scanning |
| pre-commit | ≥3.6.0 | Git hook management |

## pyproject.toml Dependency Spec

```toml
[project]
name = "prism-cli"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.9.0,<1.0.0",
    "rich>=13.0.0,<14.0.0",
    "prompt-toolkit>=3.0.0,<4.0.0",
    "litellm>=1.0.0,<2.0.0",
    "keyring>=25.0.0",
    "httpx>=0.27.0,<1.0.0",
    "pydantic>=2.0.0,<3.0.0",
    "structlog>=24.0.0",
    "unidiff>=0.7.0",
    "watchdog>=4.0.0",
]

[project.optional-dependencies]
analysis = [
    "tree-sitter>=0.22.0",
    "tree-sitter-languages>=1.10.0",
]
web = [
    "playwright>=1.40.0",
    "beautifulsoup4>=4.12.0",
]
rag = [
    "chromadb>=0.5.0",
]
crypto = [
    "cryptography>=42.0.0",
]
all = [
    "prism-routing-cli[analysis,web,rag,crypto]",
]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.12.0",
    "pytest-xdist>=3.5.0",
    "hypothesis>=6.100.0",
    "respx>=0.21.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "bandit>=1.7.0",
    "pip-audit>=2.7.0",
    "pre-commit>=3.6.0",
]
```

## Version Pinning Strategy

- **Runtime deps**: Use compatible release ranges (`>=X.Y.0,<X+1.0.0`) to allow patch updates
- **Dev deps**: Use minimum version pins (`>=X.Y.0`) — less critical for reproducibility
- **Lock file**: Generate `requirements.lock` for exact reproducibility in CI
- **Renovate/Dependabot**: Enable automated dependency update PRs

## Dependency Rules

1. **Minimize dependencies**: Prefer stdlib over third-party when functionality is equivalent
2. **No native extensions** except tree-sitter (audited, widely used)
3. **Optional heavy deps**: Playwright, ChromaDB, tree-sitter are optional — core works without them
4. **Security scanning**: `pip-audit` runs in CI on every PR
5. **License compatibility**: All deps must be Apache 2.0, MIT, BSD, or PSF compatible
6. **No abandoned packages**: All deps must have commits within the last 12 months

## Stdlib Alternatives (prefer these)

| Need | Stdlib | Third-party (avoid unless needed) |
|------|--------|----------------------------------|
| JSON | `json` | `orjson` (only if benchmarks show need) |
| Path ops | `pathlib` | — |
| Regex | `re` | — |
| SQLite | `sqlite3` | `sqlalchemy` (avoid — overkill) |
| Diffs | `difflib` | — |
| Hashing | `hashlib` | — |
| UUID | `uuid` | — |
| Dates | `datetime` | — |
| Dataclasses | `dataclasses` | `attrs` (avoid — Pydantic covers validation) |
| Async | `asyncio` | `trio` (avoid — asyncio is sufficient) |
| Subprocess | `subprocess` | — |
| Temp files | `tempfile` | — |
| Config files | — | `pydantic` (for validation), `pyyaml` (if needed) |

## Security Audit Process

```bash
# Check for known vulnerabilities in installed packages
pip-audit

# Check for known vulnerabilities in requirements
pip-audit -r requirements.lock

# Scan source code for security issues
bandit -r src/prism/ -ll

# Full dependency tree audit
pip-audit --desc --fix --dry-run
```

## Update Process

1. Dependabot/Renovate creates PR for dependency update
2. CI runs: tests, type check, lint, security scan
3. If all pass → auto-merge for patch updates
4. Manual review for minor/major version bumps
5. Test locally before merging major version updates

# ENVIRONMENT.md — Prism Environment Configuration

## Required Environment

### Python
- **Minimum**: Python 3.11
- **Recommended**: Python 3.12
- Verify: `python3 --version`

### Operating Systems
| OS | Status | Notes |
|----|--------|-------|
| macOS 12+ | Primary | Full support, keychain integration |
| Ubuntu 22.04+ | Primary | Full support, secret-service keyring |
| Windows 10/11 | Supported | Credential Manager integration |
| WSL2 | Supported | Uses Linux keyring or env vars |

### Optional: Ollama (for free local models)
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b
ollama pull llama3.2:3b

# Verify
ollama list
curl http://localhost:11434/api/tags
```

### Optional: Playwright (for web browsing)
```bash
pip install prism-cli[web]
playwright install chromium
```

## Environment Variables

### API Keys (auto-detected)
| Variable | Provider | Required? |
|----------|----------|-----------|
| `ANTHROPIC_API_KEY` | Anthropic (Claude) | No |
| `OPENAI_API_KEY` | OpenAI (GPT-4o) | No |
| `GOOGLE_API_KEY` | Google AI Studio (Gemini) | No |
| `DEEPSEEK_API_KEY` | DeepSeek | No |
| `GROQ_API_KEY` | Groq | No |
| `MISTRAL_API_KEY` | Mistral | No |

**None are required** — Prism works with any combination of providers. At minimum, Ollama (free, local) is recommended.

### Prism Configuration Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `PRISM_HOME` | `~/.prism` | Prism data directory |
| `PRISM_CONFIG` | `~/.prism/config.yaml` | Config file path |
| `PRISM_LOG_LEVEL` | `WARNING` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `PRISM_LOG_FILE` | None | Optional file path for log output |
| `PRISM_BUDGET_DAILY` | None | Daily budget override (USD) |
| `PRISM_BUDGET_MONTHLY` | None | Monthly budget override (USD) |
| `PRISM_MODEL` | None | Force specific model for all requests |
| `PRISM_NO_COLOR` | None | Disable colored output (set to any value) |
| `PRISM_OFFLINE` | None | Disable all cloud providers (Ollama only) |

### Custom Provider Variables
```bash
# For any OpenAI-compatible endpoint
CUSTOM_PROVIDER_API_KEY=your-key
CUSTOM_PROVIDER_API_BASE=https://api.example.com/v1
```

## Directory Structure

### User Data (`~/.prism/`)
```
~/.prism/
├── config.yaml              # User configuration
├── credentials.enc          # Encrypted credentials (fallback)
├── prism.db                 # SQLite database (routing, costs, learning)
├── audit.log                # Tool execution audit log
├── cache/
│   └── repo_maps/           # Cached tree-sitter repo maps
├── sessions/
│   ├── 2026-03-10_abc123.md # Session history files
│   └── ...
└── plugins/                 # User-installed plugins
```

### Project Data (`.prism.md` in project root)
```markdown
# Project: MyApp

## Stack
- Python 3.12, FastAPI, PostgreSQL

## Conventions
- Use snake_case for all function names
- pytest for testing
- ruff for linting

## Architecture
- src/api/ — API routes
- src/models/ — Database models
- src/services/ — Business logic

## Notes
- Auth uses JWT tokens stored in httpOnly cookies
- Database migrations use Alembic
```

## Development Environment Setup

### First Time Setup
```bash
# Clone repository
git clone https://github.com/GoparapukethaN/prism-cli.git
cd prism-cli

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with all dev dependencies
pip install -e ".[all,dev]"

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest
ruff check src/
mypy src/prism/
```

### IDE Configuration

#### VS Code (`.vscode/settings.json`)
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.analysis.typeCheckingMode": "strict",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "editor.rulers": [100],
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    }
}
```

#### PyCharm
- Mark `src/` as Sources Root
- Set Python interpreter to `.venv/bin/python`
- Enable pytest as default test runner
- Set line length to 100

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic, typer]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.7
    hooks:
      - id: bandit
        args: ["-r", "src/prism/", "-ll"]
```

## Testing Environment

### CRITICAL RULE: No Real API Calls
- **ALL tests run completely offline**
- **ALL provider interactions are mocked**
- **NEVER use real API keys in tests**
- **Mock classes exist for every provider**
- Use `pytest-mock` and `respx` for mocking
- Use hardcoded mock responses for routing logic tests
- Only the user tests with real API keys manually

### Test Database
- Tests use in-memory SQLite or tmp_path SQLite
- Each test gets a fresh database instance
- No shared state between tests

### Environment Variables for Tests
```bash
# Set in conftest.py or pytest.ini, NOT real keys
PRISM_HOME=/tmp/prism-test
PRISM_LOG_LEVEL=DEBUG
# Provider keys are NEVER set in test environment
```

## Production Environment Checklist

Before running Prism for real use:
- [ ] At least one API key configured (or Ollama running)
- [ ] `prism auth status` shows at least one provider connected
- [ ] `prism status` shows all green
- [ ] Project root is a git repository (for undo support)
- [ ] Sufficient disk space for SQLite database and session files
- [ ] For Ollama: sufficient RAM for chosen model (7B needs 4-8GB)

# Prism CLI

> Experimental model-routing CLI and developer-tooling playground.

Prism explores a CLI shape for routing requests across LLM providers, tracking cost, and
experimenting with developer-tool workflows such as context management, project memory,
and tool execution.

## Status

This repository is an experimental alpha. It is useful as a design and systems-learning
project, but it should not be read as a finished replacement for production coding
assistants. The public README intentionally describes the intended architecture and the
areas under test; features should be verified locally before being treated as complete.

## Quick Start (Under 5 Minutes)

### Install

```bash
pip install prism-cli
# or
pipx install prism-cli
```

For development:

```bash
git clone https://github.com/GoparapukethaN/prism-cli.git
cd prism-cli
pip install -e ".[dev,all]"
```

### Setup

Interactive wizard (recommended for first-time users):

```bash
prism init
```

Or add providers manually:

```bash
prism auth add openai
prism auth add anthropic
prism auth add google
```

### Use

```bash
# One-shot question
prism ask "explain this codebase"

# Start interactive REPL
prism

# Check system health
prism status
```

## Feature Areas

- **Model routing experiments** across multiple provider adapters
- **Cost tracking concepts** for budget-aware routing
- **File/tool operation experiments** with path and command guardrails
- **Git workflow experiments** around checkpoints and history
- **Architect mode experiments** for separating planning and execution
- **Session and project memory experiments** using local configuration files

## REPL Commands

Inside the interactive REPL, use slash commands:

| Command      | Description                                      |
| ------------ | ------------------------------------------------ |
| `/help`      | Show all available commands                      |
| `/cost`      | Display spending breakdown for the current session |
| `/model`     | Switch the active model                          |
| `/undo`      | Revert the last file change or commit            |
| `/compact`   | Summarize conversation to free context window    |
| `/add`       | Add a file to the conversation context           |
| `/drop`      | Remove a file from the conversation context      |
| `/web`       | Fetch a URL and include its content              |
| `/status`    | Show provider health and configuration           |
| `/budget`    | View remaining budget and spending history       |
| `/memory`    | View or edit project memory (`.prism.md`)        |
| `/feedback`  | Rate the last response (improves routing)        |
| `/providers` | List all configured providers and their status   |
| `/clear`     | Clear conversation history                       |
| `/save`      | Save the current session to disk                 |
| `/load`      | Load a previously saved session                  |
| `/exit`      | Exit the REPL                                    |

## Provider Setup

Prism supports the following providers. Add API keys with `prism auth add <name>`:

| Provider   | Key Format         | Free Tier | Notes                          |
| ---------- | ------------------ | --------- | ------------------------------ |
| OpenAI     | `sk-...`           | No        | GPT-4o, GPT-4o-mini, o1       |
| Anthropic  | `sk-ant-...`       | No        | Claude 4, Claude 3.5 Sonnet   |
| Google     | `AI...`            | Yes       | Gemini 2.0, Gemini 1.5        |
| DeepSeek   | `sk-...`           | Yes       | DeepSeek-V3, DeepSeek-R1      |
| Groq       | `gsk_...`          | Yes       | Llama 3, Mixtral (ultra-fast)  |
| Mistral    | (varies)           | Yes       | Mistral Large, Codestral      |
| Ollama     | (none)             | Local     | Any GGUF model, fully offline  |
| Together   | (varies)           | Yes       | Open-source model hosting      |
| Fireworks  | (varies)           | Yes       | Fast open-source inference     |
| OpenRouter | `sk-or-...`        | No        | Meta-router for 100+ models    |
| Cohere     | (varies)           | Yes       | Command R+                     |
| Perplexity | `pplx-...`         | No        | Search-augmented models        |
| AWS Bedrock| (IAM credentials)  | No        | Claude, Titan, Llama on AWS    |
| Azure OpenAI| (varies)          | No        | GPT-4o on Azure infrastructure |

## Architecture

```
prism ask "fix the bug"
        |
        v
+------------------+     +------------------+     +------------------+
|   CLI / REPL     | --> |  Task Classifier | --> | Routing Engine   |
| (Typer + Rich)   |     | (complexity tier) |    | (cost + quality) |
+------------------+     +------------------+     +------------------+
        |                                                  |
        v                                                  v
+------------------+     +------------------+     +------------------+
|   Tool Engine    |     |  Context Manager |     | Provider Layer   |
| (file, exec, git)|     | (window, memory) |     | (LiteLLM unified)|
+------------------+     +------------------+     +------------------+
        |                                                  |
        v                                                  v
+------------------+     +------------------+     +------------------+
|  Security Layer  |     |   Cost Tracker   |     | Fallback Chain   |
| (sandbox, guard) |     |  (budget, audit) |     | (retry, degrade) |
+------------------+     +------------------+     +------------------+
```

## Configuration

Prism uses a layered configuration system. Higher layers override lower ones:

1. **CLI flags** (`--model`, `--budget`, `--yes`, etc.)
2. **Environment variables** (`PRISM_MODEL`, `PRISM_BUDGET_DAILY`, etc.)
3. **Project config** (`.prism.yaml` in your project root)
4. **User config** (`~/.prism/config.yaml`)
5. **Built-in defaults**

### Example `~/.prism/config.yaml`

```yaml
routing:
  simple_threshold: 0.3
  medium_threshold: 0.55
  quality_weight: 0.7
  architect_mode: true

budget:
  daily_limit: 5.0
  monthly_limit: 50.0
  warn_at_percent: 80.0

tools:
  web_enabled: false
  auto_approve: false
  command_timeout: 30
  allowed_commands:
    - python -m pytest
    - ruff check
```

### Configuration Commands

```bash
prism config get routing.simple_threshold
prism config set budget.daily_limit 10.0
```

### Environment Variables

| Variable             | Config Key           | Description              |
| -------------------- | -------------------- | ------------------------ |
| `PRISM_MODEL`        | `pinned_model`       | Force a specific model   |
| `PRISM_BUDGET_DAILY` | `budget.daily_limit` | Daily spend limit (USD)  |
| `PRISM_BUDGET_MONTHLY`| `budget.monthly_limit`| Monthly spend limit     |
| `PRISM_LOG_LEVEL`    | `log_level`          | Logging verbosity        |
| `PRISM_HOME`         | `prism_home`         | Data directory override  |

## Cost Routing Concept

Prism classifies each request into a complexity tier:

| Tier     | Examples                          | Models Used              |
| -------- | --------------------------------- | ------------------------ |
| SIMPLE   | "what does this function do?"     | GPT-4o-mini, Gemini Flash|
| MEDIUM   | "add error handling to this file" | GPT-4o, Claude Sonnet    |
| COMPLEX  | "refactor this module completely" | Claude Opus, GPT-4o, o1  |

The routing engine is intended to score available models on quality and cost, using the
`quality_weight` setting to balance the tradeoff. Models that exceed the remaining budget
are excluded from routing candidates.

**Architect mode** explores splitting complex tasks into planning and execution phases so
different model choices can be evaluated for each phase.

## Security

Prism includes multiple security layers:

- **Path traversal prevention** -- all file operations are restricted to the project root
- **Command sandboxing** -- dangerous commands (rm -rf, sudo, etc.) are blocked by default
- **Secret redaction** -- API keys and credentials are never logged or displayed
- **Excluded file patterns** -- `.env`, `*.pem`, `credentials.json`, and similar files are blocked from tool operations
- **API key storage** -- keys are stored via OS keyring when available, falling back to environment variables or encrypted file storage

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev,all]"`
4. Run tests: `pytest`
5. Run linters: `ruff check src/ tests/ && mypy src/`
6. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.

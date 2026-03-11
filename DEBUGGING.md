# DEBUGGING.md — Prism Debugging Guide

## Diagnostic Commands

### Built-in Diagnostics
```bash
# Check all provider connections
prism status

# Output:
# Provider Status:
# ✓ Anthropic     Connected (Claude Sonnet 4, Claude Haiku 3.5)
# ✓ OpenAI        Connected (GPT-4o, GPT-4o-mini)
# ✓ Google        Connected - FREE TIER (Gemini 2.0 Flash, 1423/1500 requests remaining)
# ✓ Ollama        Connected (qwen2.5-coder:7b running, llama3.2:3b available)
# ✗ DeepSeek      Not configured
# ✗ Groq          Rate limited (resets in 42s)
#
# Database: OK (1,247 routing entries, 45.2KB)
# Config: ~/.prism/config.yaml (valid)
# Project: /Users/dev/myapp (.prism.md found)

# Verbose mode for any command
prism --verbose ask "explain this function"

# Debug mode (maximum logging)
prism --debug ask "explain this function"

# Check routing decision without executing
prism --dry-run ask "refactor the auth module"
# Output:
# Task: "refactor the auth module"
# Classified: MEDIUM (score: 0.54)
# Features: {tokens: 8, files: 0, keywords: ["refactor"], reasoning: false}
# Selected: deepseek-v3 (est. cost: $0.0018)
# Fallback: [gpt-4o-mini, groq/llama-3.3-70b, ollama/qwen2.5-coder:7b]
# Budget: $3.42 remaining of $5.00/day
```

## Common Issues and Solutions

### 1. Provider Connection Failures

**Symptom**: `ProviderError: Failed to connect to <provider>`

**Diagnosis**:
```bash
# Check if API key is configured
prism auth status

# Test specific provider
prism auth test anthropic

# Check network connectivity
curl -s https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"
```

**Common Causes**:
- API key expired or revoked → re-add with `prism auth add <provider>`
- Network/firewall blocking API endpoint → check proxy settings
- Provider outage → check status pages (status.anthropic.com, status.openai.com)
- Ollama not running → `ollama serve`

### 2. Routing Issues

**Symptom**: Tasks routed to wrong tier (expensive model for simple task, or weak model for complex task)

**Diagnosis**:
```bash
# Check recent routing decisions
prism --debug
> /route-history 10
# Shows last 10 routing decisions with features, scores, models

# Check classifier thresholds
prism config get routing.simple_threshold
prism config get routing.medium_threshold
```

**Common Causes**:
- Thresholds not calibrated for user's workflow → adjust in config
- Keyword detection too aggressive/lenient → check feature extraction
- After ML model trained: overfitting to early data → reset learning data with `prism reset-learning`

### 3. File Tool Errors

**Symptom**: `SecurityError: Path escapes project root`

**Diagnosis**:
```python
# Check resolved path
import os
print(os.path.realpath("/path/to/file"))
# Compare with project root
```

**Common Causes**:
- Symlink pointing outside project → file legitimately outside root, add to allowed paths
- Relative path resolution incorrect → check working directory
- Model hallucinated a file path → check if file exists

### 4. Cost Tracking Discrepancies

**Symptom**: `/cost` shows different amount than expected

**Diagnosis**:
```bash
# Check raw cost entries
prism --debug
> /cost --detailed
# Shows every API call with exact token counts and costs

# Verify pricing data
prism config get providers.<provider>.pricing
```

**Common Causes**:
- Pricing data outdated → update with `prism update-pricing`
- Token count estimation differs from actual → this is normal, tracked after response
- Cached responses not counted correctly → check cache hit tracking

### 5. REPL Issues

**Symptom**: REPL hangs, crashes, or loses history

**Diagnosis**:
```bash
# Check session files
ls -la ~/.prism/sessions/

# Check for corrupted session
python -c "import json; json.load(open('~/.prism/sessions/<id>.json'))"

# Start fresh session
prism --new-session
```

**Common Causes**:
- Session file corrupted → delete and restart
- Prompt Toolkit conflict with terminal → try `TERM=xterm-256color prism`
- Context window overflow → use `/compact` more frequently

### 6. Ollama Issues

**Symptom**: Ollama models not responding or slow

**Diagnosis**:
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check model is pulled
ollama list

# Check GPU usage
ollama ps

# Test model directly
ollama run qwen2.5-coder:7b "say hello"
```

**Common Causes**:
- Ollama not running → `ollama serve`
- Model not pulled → `ollama pull qwen2.5-coder:7b`
- Insufficient RAM → check with `ollama ps`, use smaller model
- GPU not detected → check CUDA/Metal drivers

### 7. Database Issues

**Symptom**: `DatabaseError` or slow queries

**Diagnosis**:
```bash
# Check database integrity
sqlite3 ~/.prism/prism.db "PRAGMA integrity_check;"

# Check database size
ls -lh ~/.prism/prism.db

# Check WAL mode
sqlite3 ~/.prism/prism.db "PRAGMA journal_mode;"

# Vacuum if needed
sqlite3 ~/.prism/prism.db "VACUUM;"
```

**Common Causes**:
- Database corrupted → restore from WAL or recreate
- Database too large → run vacuum, archive old data
- WAL file huge → checkpoint with `PRAGMA wal_checkpoint(TRUNCATE)`
- Concurrent access issues → ensure WAL mode is enabled

### 8. Git Integration Issues

**Symptom**: Auto-commit fails or undo doesn't work

**Diagnosis**:
```bash
# Check git status in project
cd /project && git status

# Check Prism's git log
git log --oneline -20 | grep "prism:"

# Verify git config
git config user.name
git config user.email
```

**Common Causes**:
- Not a git repository → `git init`
- Uncommitted changes before Prism edit → Prism won't auto-commit over dirty state
- Git hooks blocking commits → check `.git/hooks/`
- No git user configured → `git config user.name "..."`

## Logging Configuration

### Log Levels
```bash
# Default (warnings and errors only)
prism ask "..."

# Verbose (info + warnings + errors)
prism --verbose ask "..."

# Debug (everything including internal state)
prism --debug ask "..."

# Log to file
PRISM_LOG_FILE=~/.prism/debug.log prism --debug ask "..."
```

### Log Locations
| Log | Location | Content |
|-----|----------|---------|
| Audit log | `~/.prism/audit.log` | Every tool execution with timestamps |
| Debug log | `~/.prism/debug.log` | Full debug output (when enabled) |
| Session log | `~/.prism/sessions/<id>.md` | Conversation history |
| Cost log | `~/.prism/prism.db` (cost_entries table) | Every API call cost |

### Structured Log Format
```
[2026-03-10T14:23:01Z] level=INFO module=router.selector msg="model_selected" model="deepseek-v3" tier="medium" score=0.54 cost_est=0.0018
[2026-03-10T14:23:02Z] level=DEBUG module=router.classifier msg="features_extracted" tokens=8 files=0 keywords=["refactor"] reasoning=false scope="single_module"
[2026-03-10T14:23:03Z] level=WARNING module=router.fallback msg="provider_rate_limited" provider="groq" retry_after=42
[2026-03-10T14:23:03Z] level=INFO module=router.fallback msg="fallback_activated" from="groq/llama-3.3-70b" to="deepseek-v3"
```

## Debug Checklist

When something goes wrong, check in this order:
1. `prism status` — are all providers connected?
2. `prism auth status` — are API keys valid?
3. `prism --verbose` — what does the verbose output show?
4. `~/.prism/audit.log` — what tool executions happened?
5. `~/.prism/prism.db` — are cost/routing entries being written?
6. `prism --debug` — full debug trace
7. `ruff check src/` — any lint errors in source?
8. `pytest tests/` — are tests passing?

## Profiling

### Performance Profiling
```bash
# Profile a command
python -m cProfile -o prism.prof -m prism ask "hello"
python -m pstats prism.prof

# Memory profiling
pip install memray
memray run -o prism.bin -m prism ask "hello"
memray flamegraph prism.bin
```

### Database Query Profiling
```python
# Enable SQLite query logging
import sqlite3
conn = sqlite3.connect("~/.prism/prism.db")
conn.set_trace_callback(print)  # Prints every SQL query
```

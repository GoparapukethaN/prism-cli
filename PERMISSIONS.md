# PERMISSIONS.md — Prism Permission Model

## Overview

Prism operates on the user's machine with access to their filesystem, terminal, and network. The permission model ensures users maintain control over all actions while allowing smooth workflow.

## Permission Levels

### AUTO — No Confirmation Needed
Actions that only read data and cannot cause harm:
- `read_file` — read any file in project root
- `list_directory` — list any directory in project root
- `search_codebase` — search files in project root
- Cost queries — check spending, budget
- Provider status checks

### CONFIRM — User Approval Required
Actions that modify state:
- `write_file` — create or overwrite files (show diff first)
- `edit_file` — modify files (show diff first)
- `execute_command` — run shell commands (show command first)
- `browse_web` — fetch web pages (show URL first)
- `screenshot` — capture web screenshots (show URL first)

### DANGEROUS — Always Explicit Confirmation
Actions that are destructive or irreversible:
- File deletion
- Git operations that modify history (reset, force push)
- Running commands that match dangerous patterns

## Confirmation UI

### File Write/Edit Confirmation
```
╭── write_file: src/auth.py ────────────────────────╮
│                                                     │
│  - def authenticate(user, password):                │
│  + def authenticate(user: str, password: str):      │
│      """Authenticate a user."""                      │
│  -   if check_password(password):                   │
│  +   if verify_password(user, password):            │
│          return create_token(user)                   │
│                                                     │
╰── Apply this change? [y/n/e(dit)] ────────────────╯
```

Options:
- `y` / Enter — apply the change
- `n` — reject the change
- `e` — open the diff in $EDITOR for manual editing

### Command Execution Confirmation
```
╭── execute_command ──────────────────────────────────╮
│ $ npm test                                           │
│                                                      │
│ Working directory: /Users/dev/myapp                   │
│ Timeout: 30s                                         │
╰── Run this command? [y/n] ──────────────────────────╯
```

### Web Browsing Confirmation
```
╭── browse_web ───────────────────────────────────────╮
│ URL: https://docs.python.org/3/library/asyncio.html  │
│ Mode: text extraction                                │
╰── Fetch this URL? [y/n] ────────────────────────────╯
```

## Auto-Approve Mode (`--yes`)

When user starts Prism with `--yes` or `/auto on`:
- All CONFIRM-level actions execute without asking
- DANGEROUS-level actions STILL require confirmation
- A warning is shown at session start:
  ```
  ⚠ Auto-approve mode enabled. File writes and commands will execute without confirmation.
  ```

## Command Allowlist

Users pre-approve specific commands that skip confirmation:

```yaml
# ~/.prism/config.yaml
tools:
  allowed_commands:
    - "npm test"
    - "npm run build"
    - "npm run lint"
    - "python -m pytest"
    - "python -m pytest -v"
    - "cargo build"
    - "cargo test"
    - "make"
    - "make test"
    - "go test ./..."
    - "ruff check ."
    - "mypy src/"
```

### Matching Rules
- Exact match required (including arguments)
- No glob patterns (for security)
- Prefix matching optional: `"npm test"` matches `"npm test src/"` only if configured with `prefix: true`

## Blocked Actions (hardcoded, cannot be overridden)

### Blocked Commands
```python
BLOCKED_COMMANDS = [
    r"^rm\s+-rf\s+/",           # rm -rf /
    r"^rm\s+-rf\s+~",           # rm -rf ~
    r"^rm\s+-rf\s+/\*",         # rm -rf /*
    r"^:\(\)\{.*\}",            # Fork bomb
    r"^mkfs\.",                   # Filesystem format
    r"^dd\s+if=/dev/zero",       # Disk zeroing
    r"^chmod\s+-R\s+777",        # Dangerous permissions
    r"^sudo\s+rm",               # Sudo destructive
    r"\|\s*sh\s*$",              # Pipe to shell (from model output)
    r"\|\s*bash\s*$",            # Pipe to bash
]
```

### Blocked File Patterns
These files can never be read or written by tools:
```python
ALWAYS_BLOCKED = [
    "~/.ssh/*",
    "~/.aws/credentials",
    "~/.azure/credentials",
    "~/.gcloud/credentials",
    "**/id_rsa",
    "**/id_ed25519",
    "**/*.pem",
]
```

## Provider-Level Permissions

### Data Routing Exclusions
Users can exclude specific providers from receiving their code:

```yaml
# ~/.prism/config.yaml
providers:
  excluded:
    - deepseek    # Never send code to DeepSeek
    - groq        # Never send code to Groq
```

When excluded:
- Provider is completely removed from candidate models
- No API calls made to that provider
- User's code never sent to excluded providers

### Per-Task Overrides
```
> /model claude-sonnet     # Force Claude for this session
> /exclude deepseek        # Exclude DeepSeek for this session
```

## Audit Trail

Every permission-requiring action is logged:

```
[2026-03-10T14:23:01Z] PERMISSION tool=write_file path=src/auth.py action=APPROVED
[2026-03-10T14:23:05Z] PERMISSION tool=execute_command cmd="npm test" action=APPROVED
[2026-03-10T14:23:18Z] PERMISSION tool=write_file path=src/db.py action=DENIED
[2026-03-10T14:23:20Z] PERMISSION tool=execute_command cmd="rm -rf /" action=BLOCKED reason=dangerous_pattern
```

## First-Run Permissions

On first run (`prism init`), users are walked through:

1. **API key setup** — which providers to configure
2. **Permission level** — conservative (confirm everything) or relaxed (auto-approve reads + familiar commands)
3. **Budget limits** — daily and monthly spending caps
4. **Web browsing** — enable or disable
5. **Command allowlist** — pre-approve common commands

Default: conservative mode (confirm all writes and commands).

## Permission Escalation

If the model requests an action that was denied:
1. The model is informed the action was denied
2. The model can suggest an alternative approach
3. The model CANNOT re-request the same denied action in the same turn
4. The user can manually re-approve via `/approve`

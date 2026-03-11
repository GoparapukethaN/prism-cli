# SECURITY.md — Prism Security Model

## Threat Model

Prism has a uniquely sensitive threat surface: it has read/write access to user codebases, executes shell commands, makes HTTP requests, and handles API keys for multiple providers. A vulnerability in Prism could leak source code, API keys, or allow arbitrary code execution.

### Assets to Protect
1. **API keys** — stored in keyring, env vars, or encrypted config
2. **Source code** — user's codebase accessible via file tools
3. **Shell environment** — accessible via terminal tool
4. **Network** — outbound requests to AI providers and web browsing
5. **Conversation history** — may contain sensitive code snippets
6. **Routing/cost data** — usage patterns in SQLite

### Threat Actors
- **Malicious AI output** — model generates commands to exfiltrate data
- **Prompt injection** — crafted file contents or web pages manipulate model behavior
- **Supply chain** — compromised dependencies
- **Local attacker** — another user/process on the same machine

## Security Controls

### 1. API Key Management

#### Storage Hierarchy (in order of preference)
1. **OS Keyring** (macOS Keychain, Windows Credential Manager, Linux Secret Service)
   - Encrypted at rest by the operating system
   - Accessed via `keyring` library
   - Requires user authentication (biometrics, password) on some systems
2. **Environment Variables** (standard `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` patterns)
   - Auto-detected on startup
   - Never written to disk by Prism
3. **Encrypted Config File** (AES-256-GCM via `cryptography` library)
   - Stored at `~/.prism/credentials.enc`
   - Passphrase required to decrypt
   - PBKDF2-HMAC-SHA256 with 600,000 iterations for key derivation
   - Random 16-byte salt, 12-byte nonce per encryption

#### Key Handling Rules
- Keys NEVER appear in log files, error messages, or stack traces
- Keys NEVER written to SQLite database
- Keys NEVER included in git commits (`.gitignore` enforced)
- Keys NEVER passed to subprocess environments (filtered in `secret_filter.py`)
- Keys NEVER sent to AI models as part of conversation context
- Key validation: minimal API call on storage to confirm validity
- Key display: always masked except last 4 characters (`sk-ant-...x7Qm`)

### 2. File System Security

#### Path Traversal Prevention
```
Every file operation MUST:
1. Resolve the path with os.path.realpath() to eliminate symlinks
2. Verify the resolved path starts with the project root
3. Reject if path escapes project root (raise SecurityError)
4. Check against the excluded patterns list
```

#### Excluded File Patterns (default, configurable)
```
.env, .env.*, *.env
.git/credentials, .git/config (if contains credentials)
**/node_modules/**
**/.ssh/**
**/*.pem, **/*.key, **/*.p12, **/*.pfx
**/credentials.json, **/service-account*.json
**/.aws/credentials, **/.azure/credentials
**/secrets.yaml, **/secrets.yml
**/*.sqlite, **/*.db (unless explicitly added)
~/.prism/credentials.enc
~/.prism/audit.log (read-only from tools)
```

#### Permission Tiers for File Operations
| Operation | Permission | User Action |
|-----------|-----------|-------------|
| read_file | Auto-approved | None |
| list_directory | Auto-approved | None |
| search_codebase | Auto-approved | None |
| write_file | Confirmation required | Shown diff, approve/deny |
| edit_file | Confirmation required | Shown diff, approve/deny |
| delete_file | Always requires explicit confirmation | Must type "yes" |
| write_file (--yes mode) | Auto-approved | None (user opted in) |

### 3. Command Execution Sandbox

#### Isolation Controls
- **Working directory**: Locked to project root, cannot cd outside
- **Timeout**: Default 30 seconds, configurable per command, max 300 seconds
- **Output limits**: 100KB stdout, 10KB stderr (truncated with warning)
- **Environment filtering**: All `*_API_KEY`, `*_SECRET`, `*_TOKEN`, `*_PASSWORD` vars stripped
- **No shell expansion by default**: Commands run via `subprocess.run(shell=False)` where possible
- **Process group**: Subprocess runs in its own process group for clean termination

#### Command Allowlist
Users configure pre-approved commands that skip confirmation:
```yaml
allowed_commands:
  - "npm test"
  - "npm run build"
  - "python -m pytest"
  - "cargo build"
  - "cargo test"
  - "make"
  - "go test ./..."
```

#### Blocked Commands (hardcoded, cannot be overridden)
```
rm -rf /
rm -rf ~
rm -rf /*
:(){ :|:& };:          # Fork bomb
> /dev/sda             # Disk destruction
mkfs.*                 # Filesystem format
dd if=/dev/zero        # Disk zeroing
curl | sh              # Pipe to shell (when from model output)
wget | sh
chmod -R 777           # Dangerous permissions
sudo rm                # Sudo destructive ops
```

### 4. Web Browsing Security

- **Disabled by default** — enabled with `--web` flag or `/web on`
- **Domain allowlist** (optional): restrict browsing to specified domains
- **No JavaScript execution** by default in lightweight httpx mode
- **Playwright sandboxing**: headless Chromium with `--no-sandbox` disabled
- **Content size limits**: 1MB max page content extraction
- **No cookie persistence** between sessions
- **No file downloads** — browsing is read-only
- **URL validation**: reject file://, javascript:, data: schemes
- **Prompt injection defense**: web content wrapped in clear delimiters, model instructed to treat as untrusted data

### 5. Audit Logging

Every tool execution logged to `~/.prism/audit.log`:
```
[2026-03-10T14:23:01Z] TOOL=read_file PATH=/project/src/main.py RESULT=success SIZE=4823
[2026-03-10T14:23:05Z] TOOL=execute_command CMD="npm test" RESULT=success EXIT=0 DURATION=12.3s
[2026-03-10T14:23:18Z] TOOL=write_file PATH=/project/src/main.py RESULT=success DIFF_LINES=+5/-3
[2026-03-10T14:23:20Z] TOOL=browse_web URL=https://docs.python.org RESULT=success SIZE=45KB
[2026-03-10T14:24:01Z] ROUTING MODEL=deepseek-v3 TIER=medium COST=$0.0012 TOKENS_IN=450 TOKENS_OUT=1200
```

Log rotation: 10MB max file size, 5 rotated files kept.

### 6. Data Privacy

- **All data stays local** — no telemetry, no phone-home, no crash reporting unless user opts in
- **SQLite database** at `~/.prism/prism.db` — user owns and controls all data
- **Session history** at `~/.prism/sessions/` — Markdown files, user can read/delete
- **No data sent to Prism servers** — there are no Prism servers (open source)
- **Model interactions**: only sent to the user's configured providers via their own API keys
- **Prompt hashing**: only SHA-256 hashes stored for deduplication, not raw prompts

### 7. Dependency Security

- All dependencies pinned to exact versions in `pyproject.toml`
- `pip-audit` run in CI to check for known vulnerabilities
- Minimal dependency tree — prefer stdlib over third-party where possible
- No native extensions except tree-sitter (audited, widely used)
- Playwright installed separately (`playwright install chromium`) to avoid bloating default install

## Security Testing Checklist

Run after every module:
- [ ] `bandit -r src/prism/<module>/ -ll` — no medium+ severity findings
- [ ] Path traversal tests: symlinks, `../`, absolute paths outside root
- [ ] Input injection tests: shell metacharacters in filenames, SQL injection in queries
- [ ] API key leak tests: keys never in logs, errors, git, subprocess env
- [ ] Timeout tests: verify commands are killed after timeout
- [ ] Output limit tests: verify truncation at size limits
- [ ] Permission tests: verify write operations require confirmation

## Incident Response

If a security vulnerability is found:
1. Do not disclose publicly until fix is ready
2. Create a security advisory on GitHub
3. Fix, test, release patch version
4. Notify users via GitHub release notes and security advisory
5. Add regression test to prevent reoccurrence

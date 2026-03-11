# KNOWN_ISSUES.md — Prism Known Issues and Limitations

## v0.2.0-beta — Current Issues

### 1. No Live API Testing Yet
**Severity**: High
**Description**: All LiteLLM calls use MockLiteLLM during development. No real API calls have been made to any provider. The CompletionEngine, streaming, retry logic, and provider configs need live validation.
**Impact**: Provider integrations may have model ID mismatches, unexpected response formats, or rate limit behavior differences.
**Next Step**: Manual testing with real API keys.

### 2. Two CostEntry Classes
**Severity**: Low
**Description**: `prism.cost.tracker.CostEntry` (dataclass) and `prism.db.models.CostEntry` (Pydantic) coexist with slightly different field names. `save_cost_entry()` uses `hasattr` to handle both.
**Impact**: Confusing for developers. May cause subtle bugs if one class changes.
**Mitigation**: Unify into a single CostEntry in a future refactor.

### 3. Token Estimation Heuristic
**Severity**: Medium
**Description**: Token estimates use `len/4` heuristic. Actual counts from tiktoken/Anthropic tokenizers can differ by 10-30%. Context budget manager falls back to this when tiktoken is unavailable.
**Impact**: Cost estimates and context window enforcement may be inaccurate.
**Mitigation**: Use actual token counts from API responses for cost tracking. tiktoken used when available.

### 4. Git Credential Conflict (Development Machine)
**Severity**: Low (local)
**Description**: `git push` uses cached credential (`kxgst228`) instead of `GoparapukethaN`. The `gh` CLI auth works correctly.
**Workaround**: Use `GH_TOKEN=$(gh auth token) git push origin main`.

### 5. Keyring Platform Variability
**Severity**: Medium
**Description**: `keyring` has inconsistent behavior across platforms. Headless Linux, Docker, and WSL may lack backends.
**Impact**: Credential storage fails on some platforms.
**Mitigation**: 3-tier fallback (keyring → env vars → encrypted file). Clear error messages suggest alternatives.

### 6. LiteLLM Model ID Drift
**Severity**: Medium
**Description**: LiteLLM frequently updates model identifiers. IDs in provider configs may become stale.
**Impact**: Provider configs may break on LiteLLM updates.
**Mitigation**: Pin LiteLLM version, verify IDs during live testing.

### 7. Playwright Install Size
**Severity**: Low
**Description**: Playwright downloads ~400MB of Chromium. Not included by default.
**Impact**: Long install for `[web]` extra.
**Mitigation**: By design — `pip install prism-cli[web]` is optional. httpx path available for basic browsing.

### 8. OSV.dev API Requires Network
**Severity**: Medium
**Description**: `prism deps audit` queries the OSV.dev API for vulnerability data. Requires active internet connection.
**Impact**: Dependency security scanning unavailable offline.
**Mitigation**: Offline mode gracefully handles network failures. Results are cached when available.

### 9. Debate Mode Uses Stub LLM Caller
**Severity**: Medium
**Description**: The `debate()` function accepts an `llm_caller` callback but defaults to a stub that returns placeholder text. Real multi-model debate requires live API keys.
**Impact**: `/debate` command returns synthetic responses until connected to real providers.
**Next Step**: Wire up to CompletionEngine when live API testing begins.

### 10. Archaeologist Requires Git Repository
**Severity**: Low
**Description**: `prism why` requires the project to be a git repository with history. Fails gracefully on non-git directories.
**Impact**: Feature unavailable for non-git projects.
**Mitigation**: Clear error message suggesting `git init`.

### 11. Bandit Findings (All Reviewed, Acceptable)
**Severity**: Low
**Description**: Bandit reports 1 HIGH (sandbox.py `shell=True` — intentional for command execution sandbox), 10 MEDIUM (hardcoded SQL table names, XML parsing, urllib, shell completion logging). All are false positives or intentional design.
**Impact**: None. All findings reviewed and documented.

### 12. Classifier Threshold Sensitivity
**Severity**: Medium
**Description**: Task complexity thresholds may need tuning with real usage data.
**Impact**: Tasks may be misclassified, routing to wrong model tiers.
**Mitigation**: Adaptive learning module adjusts over time after 100+ interactions.

## Resolved Issues

| Issue | Resolution | Version |
|-------|-----------|---------|
| Transaction rollback on Ctrl+C | Changed `except Exception` to `except BaseException` in `database.py` | v0.1.0-alpha |
| SecretFilter missed API keys in values | Added `scrub_value()` with regex patterns for value-level scrubbing | v0.1.0-alpha |
| Sensitive env patterns too narrow | Broadened from `*_API_KEY` to `*API_KEY*` | v0.1.0-alpha |
| CI/CD Pipeline missing | Added GitHub Actions workflows, issue templates, PR template | v0.2.0-beta |
| AEI escalation logic bug | Fixed condition to check failure counts directly instead of current strategy | v0.2.0-beta |
| Blast radius report empty-list defaults | Fixed `[] or default` truthy-defaulting in test helpers | v0.2.0-beta |
| setup.py extras bracket parsing | Fixed regex to use bracket-depth matching instead of non-greedy `.*?` | v0.2.0-beta |
| Debate confidence negative values | Fixed regex to handle `-0.5` with `-?` pattern | v0.2.0-beta |

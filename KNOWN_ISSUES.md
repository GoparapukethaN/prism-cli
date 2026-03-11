# KNOWN_ISSUES.md — Prism Known Issues and Limitations

## v0.1.0-alpha — Current Issues

### 1. No Live API Testing Yet
**Severity**: High
**Description**: All LiteLLM calls use MockLiteLLM during development. No real API calls have been made to any provider. The CompletionEngine, streaming, retry logic, and provider configs need live validation.
**Impact**: Provider integrations may have model ID mismatches, unexpected response formats, or rate limit behavior differences.
**Next Step**: Manual testing with real API keys (next phase).

### 2. Two CostEntry Classes
**Severity**: Low
**Description**: `prism.cost.tracker.CostEntry` (dataclass) and `prism.db.models.CostEntry` (Pydantic) coexist with slightly different field names. `save_cost_entry()` uses `hasattr` to handle both.
**Impact**: Confusing for developers. May cause subtle bugs if one class changes.
**Mitigation**: Unify into a single CostEntry in a future refactor.

### 3. Token Estimation Heuristic
**Severity**: Medium
**Description**: Token estimates use `words × 0.75` heuristic. Actual counts from tiktoken/Anthropic tokenizers can differ by 10-30%.
**Impact**: Cost estimates and context window enforcement may be inaccurate.
**Mitigation**: Use actual token counts from API responses for cost tracking. Consider adding tiktoken as optional dependency.

### 4. Git Credential Conflict (Development Machine)
**Severity**: Low (local)
**Description**: `git push` uses cached credential (`kxgst228`) instead of `GoparapukethaN`. The `gh` CLI auth works correctly. Subsequent pushes need credential manager update or SSH.
**Workaround**: Use `gh` CLI for pushes, or update git credential keychain.

### 5. Keyring Platform Variability
**Severity**: Medium
**Description**: `keyring` has inconsistent behavior across platforms. Headless Linux, Docker, and WSL may lack backends.
**Impact**: Credential storage fails on some platforms.
**Mitigation**: Already implemented: 3-tier fallback (keyring → env vars → encrypted file). Clear error messages suggest alternatives.

### 6. LiteLLM Model ID Drift
**Severity**: Medium
**Description**: LiteLLM frequently updates model identifiers. IDs in `provider_config.py` and `base.py` may become stale.
**Impact**: Provider configs may break on LiteLLM updates.
**Mitigation**: Pin LiteLLM version, verify IDs during live testing.

### 7. Playwright Install Size
**Severity**: Low
**Description**: Playwright downloads ~400MB of Chromium. Not included by default.
**Impact**: Long install for `[web]` extra.
**Mitigation**: By design — `pip install prism-cli[web]` is optional. httpx path available for basic browsing.

### 8. encrypted_store.py Low Coverage (30%)
**Severity**: Low
**Description**: `cryptography` is an optional dependency. Most encrypted store paths are untested because the dep isn't installed in the test environment.
**Impact**: Encrypted credential storage may have untested edge cases.
**Mitigation**: Add `cryptography` to dev dependencies and write tests in a future pass.

### 9. Classifier Threshold Sensitivity
**Severity**: Medium
**Description**: `medium_threshold` was lowered from 0.7 to 0.55 because the original was unreachable for text-only prompts without file context. Thresholds may need further tuning with real usage data.
**Impact**: Tasks may be misclassified, routing to wrong model tiers.
**Mitigation**: Adaptive learning module adjusts over time after 100+ interactions.

### 10. No CI/CD Pipeline
**Severity**: Medium
**Description**: No GitHub Actions, no automated testing on push.
**Impact**: Regressions may be merged without detection.
**Next Step**: Set up CI with pytest + ruff + bandit in a future phase.

## Resolved Issues

| Issue | Resolution | Version |
|-------|-----------|---------|
| Transaction rollback on Ctrl+C | Changed `except Exception` to `except BaseException` in `database.py` | v0.1.0-alpha |
| SecretFilter missed API keys in values | Added `scrub_value()` with regex patterns for value-level scrubbing | v0.1.0-alpha |
| Sensitive env patterns too narrow | Broadened from `*_API_KEY` to `*API_KEY*` | v0.1.0-alpha |

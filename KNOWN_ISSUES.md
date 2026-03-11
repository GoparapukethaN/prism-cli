# KNOWN_ISSUES.md — Prism Known Issues and Limitations

## Current Issues
None yet — project is in pre-development phase.

---

## Anticipated Issues (from product plan analysis)

### 1. Token Estimation Accuracy
**Severity**: Medium
**Description**: Token estimates use heuristics (words × 0.75, chars × 0.4) that vary significantly across models and content types. Actual token counts from tiktoken (OpenAI), Anthropic's tokenizer, and other model-specific tokenizers can differ by 10-30%.
**Impact**: Cost estimates may be inaccurate, leading to budget overruns or unnecessary downgrades.
**Mitigation**: Track estimation error over time, calibrate per-model. Use actual token counts from API responses for cost tracking (not estimates).
**Status**: To be addressed in Phase 1

### 2. LiteLLM Model ID Drift
**Severity**: Medium
**Description**: LiteLLM frequently updates model identifiers and provider support. Model IDs like `deepseek/deepseek-chat` may change between LiteLLM versions.
**Impact**: Provider configs may break on LiteLLM updates.
**Mitigation**: Pin LiteLLM version, test model IDs on every update. Add integration tests that verify model ID validity.
**Status**: To be monitored

### 3. Ollama Availability Detection
**Severity**: Low
**Description**: Ollama may be installed but not running, or running but without the required models pulled. Need to distinguish between "Ollama not installed", "Ollama not running", and "model not available".
**Impact**: False fallback failures when Ollama should be available.
**Mitigation**: Health check sequence: (1) check if ollama binary exists, (2) check if server responds on :11434, (3) check if required model is available.
**Status**: To be addressed in Phase 1

### 4. Free Tier Rate Limit Tracking
**Severity**: Medium
**Description**: Tracking free tier usage (Google 1,500 req/day, Groq 14,400 req/day, 30 RPM) requires persistent counters that reset at provider-specific times (which may not be midnight UTC).
**Impact**: May exceed free tier limits and get unexpected 429 errors.
**Mitigation**: Track usage in SQLite, add safety margin (stop routing to free tier at 90% of limit), handle 429s gracefully with fallback.
**Status**: To be addressed in Phase 1 Month 2

### 5. Search/Replace Edit Reliability
**Severity**: Medium
**Description**: The search/replace edit format requires models to produce exact string matches for the search portion. Smaller models (Ollama 7B) may produce approximate matches that fail.
**Impact**: Edit operations fail, requiring manual intervention or retry with a better model.
**Mitigation**: Implement fuzzy matching fallback: if exact match fails, try normalized whitespace matching and suggest the closest match to the user.
**Status**: To be addressed in Phase 1 Month 3

### 6. tree-sitter Language Coverage
**Severity**: Low
**Description**: Not all programming languages have mature tree-sitter grammars. Some files (configs, data formats) won't be parseable.
**Impact**: Repo map may be incomplete for projects using niche languages.
**Mitigation**: Graceful fallback: show file path and first few lines for unparseable files.
**Status**: To be addressed in Phase 1 Month 3

### 7. Keyring Compatibility
**Severity**: Medium
**Description**: `keyring` library has inconsistent behavior across platforms. Headless Linux (no GUI) may not have a keyring backend. Docker containers lack keyring. WSL may have limited support.
**Impact**: Credential storage fails on some platforms.
**Mitigation**: Three-tier fallback: keyring → env vars → encrypted file. Clear error messages suggesting alternatives when keyring fails.
**Status**: To be addressed in Phase 1

### 8. Context Window Limits with Multiple Active Files
**Severity**: Medium
**Description**: When users `/add` many large files, the context may exceed the model's window, especially for smaller models (Ollama 32K, DeepSeek 64K).
**Impact**: API calls fail or responses are truncated.
**Mitigation**: Context budget enforcement — measure tokens before sending, auto-truncate or auto-compact, route to larger-context models when needed.
**Status**: To be addressed in Phase 2

### 9. Concurrent Session Conflicts
**Severity**: Low
**Description**: Running multiple Prism sessions in the same project could cause SQLite WAL conflicts or git commit race conditions.
**Impact**: Database errors or conflicting git commits.
**Mitigation**: File-based lock per project root. Warning if another session detected.
**Status**: To be addressed in Phase 2

### 10. Playwright Installation Size
**Severity**: Low
**Description**: Playwright downloads ~400MB of Chromium binaries. This significantly increases install size for the `[web]` extra.
**Impact**: Long install times, disk space usage.
**Mitigation**: Web browsing is an optional extra (`pip install prism-cli[web]`). Not included by default. Fast httpx path available for most documentation lookups.
**Status**: By design — optional dependency

## Issue Tracking

When new issues are discovered during development:
1. Add to this file with severity, description, impact, mitigation, and status
2. If critical: create a GitHub issue
3. Update status as work progresses
4. Remove from this file when resolved (move to CHANGELOG.md)

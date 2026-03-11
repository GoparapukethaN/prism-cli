# CODE_REVIEW.md — Prism Code Review Standards

## Automated Review Checklist (Run After Every Module)

### 1. Correctness
- [ ] All functions return the correct types as declared in signatures
- [ ] All error paths are handled — no bare `except:`, no silenced exceptions
- [ ] Edge cases covered: empty inputs, None values, zero-length collections, boundary values
- [ ] No off-by-one errors in loops, slicing, or pagination
- [ ] Async functions properly awaited — no fire-and-forget coroutines
- [ ] Resource cleanup: all files, DB connections, subprocess handles properly closed
- [ ] No race conditions in concurrent operations

### 2. Security
- [ ] No hardcoded secrets, API keys, or credentials anywhere
- [ ] All file paths resolved via `realpath()` and checked against project root
- [ ] All user inputs validated and sanitized before use
- [ ] No shell injection vectors (use `shell=False` in subprocess)
- [ ] No SQL injection (use parameterized queries exclusively)
- [ ] No path traversal possible (symlinks resolved, `..` checked)
- [ ] API keys filtered from subprocess environments
- [ ] Sensitive data never logged (keys, passwords, tokens)
- [ ] `bandit` scan passes with no medium+ findings

### 3. Type Safety
- [ ] All function parameters have type hints
- [ ] All return types declared
- [ ] No `Any` types unless absolutely necessary (document why)
- [ ] `mypy --strict` passes with no errors
- [ ] Pydantic models used for external data validation
- [ ] Enums used for fixed sets of values (tiers, statuses, permissions)

### 4. Testing
- [ ] Every public function has at least one test
- [ ] Every error path has a test
- [ ] Edge cases have dedicated tests
- [ ] Mocks used for external dependencies (API calls, file system, DB)
- [ ] No test interdependencies — each test is independent
- [ ] Test names describe the scenario: `test_classify_simple_prompt_returns_simple_tier`
- [ ] Coverage ≥ 90% for the module
- [ ] Security-focused tests included (path traversal, injection, etc.)

### 5. Code Quality
- [ ] No code duplication — shared logic extracted to utilities
- [ ] Functions are < 50 lines (extract if longer)
- [ ] Classes have single responsibility
- [ ] No circular imports
- [ ] No global mutable state
- [ ] Constants defined, not magic numbers/strings
- [ ] `ruff check` passes with no warnings
- [ ] Docstrings on all public functions and classes (Google style)

### 6. Performance
- [ ] No N+1 query patterns in database operations
- [ ] No unnecessary file reads (cache when appropriate)
- [ ] No blocking I/O in async contexts
- [ ] Large collections use generators/iterators where possible
- [ ] Database queries use appropriate indexes
- [ ] No unbounded memory growth (streaming for large data)

### 7. Architecture Compliance
- [ ] Module follows the dependency direction defined in ARCHITECTURE.md
- [ ] No direct provider API calls outside the providers module
- [ ] All tool implementations follow the Tool interface contract
- [ ] All database access goes through the db module
- [ ] Configuration accessed via the config module, not directly
- [ ] Logging uses the structured logging format defined in LOGGING.md

## Review Process

### Self-Review (Automated)
After writing any module, run:
```bash
# Lint
ruff check src/prism/<module>/

# Type check
mypy src/prism/<module>/

# Security scan
bandit -r src/prism/<module>/ -ll

# Tests
pytest tests/test_<module>/ -v --cov=src/prism/<module>/ --cov-fail-under=90

# All together
make review MODULE=<module>
```

### Code Review Questions
For each change, ask:
1. **Does this do what it claims?** Read the code, not just the description.
2. **What happens when it fails?** Trace every error path.
3. **Is there a simpler way?** If yes, use it.
4. **Does this introduce a security risk?** Even a small one?
5. **Will this scale?** What happens with 10K routing history entries? 1M?
6. **Is this tested?** Not just happy path — error paths and edge cases too.

## Common Anti-Patterns to Reject

### Never Accept
```python
# Bare except
try:
    do_something()
except:  # NO — catches KeyboardInterrupt, SystemExit
    pass

# Hardcoded credentials
api_key = "sk-ant-abc123"  # NEVER

# Shell injection
subprocess.run(f"grep {user_input} file.txt", shell=True)  # NO

# Unvalidated paths
with open(user_provided_path) as f:  # NO — must validate against root

# SQL injection
cursor.execute(f"SELECT * FROM routes WHERE model = '{model}'")  # NO

# Silenced errors
result = do_something()  # type: ignore  # NO — fix the type error

# Magic numbers
if score < 0.3:  # NO — use named constant SIMPLE_THRESHOLD = 0.3
```

### Always Require
```python
# Specific exception handling
try:
    response = await litellm.acompletion(**params)
except litellm.RateLimitError:
    return await self._fallback(params)
except litellm.AuthenticationError:
    raise ProviderAuthError(provider=params["model"])

# Parameterized queries
cursor.execute("SELECT * FROM routes WHERE model = ?", (model,))

# Path validation
resolved = path.resolve()
if not resolved.is_relative_to(self.project_root):
    raise SecurityError(f"Path escapes project root: {path}")

# Named constants
SIMPLE_THRESHOLD = 0.3
MEDIUM_THRESHOLD = 0.7

# Explicit typing
def classify(self, prompt: str, context: TaskContext) -> ComplexityTier:
```

## Severity Levels

| Level | Action | Examples |
|-------|--------|---------|
| **CRITICAL** | Block — must fix before merge | Security vulnerability, data loss risk, credential leak |
| **HIGH** | Block — must fix | Missing error handling, type errors, missing tests |
| **MEDIUM** | Fix recommended | Code duplication, performance issue, missing docstring |
| **LOW** | Optional | Style preference, naming suggestion, minor optimization |

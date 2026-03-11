# TESTING.md — Prism Testing Strategy

## Testing Framework
- **pytest** — test runner
- **pytest-cov** — coverage reporting
- **pytest-asyncio** — async test support
- **pytest-mock** — mocking via `mocker` fixture
- **pytest-xdist** — parallel test execution (optional)
- **hypothesis** — property-based testing for edge cases
- **responses** or **respx** — HTTP request mocking

## Directory Structure
```
tests/
├── conftest.py                  # Global fixtures
├── test_cli/
│   ├── conftest.py
│   ├── test_app.py              # CLI command parsing
│   ├── test_repl.py             # REPL loop behavior
│   ├── test_commands/
│   │   ├── test_auth_cmd.py
│   │   ├── test_init_cmd.py
│   │   ├── test_ask_cmd.py
│   │   └── test_config_cmd.py
│   └── test_ui/
│       ├── test_display.py
│       └── test_themes.py
├── test_router/
│   ├── conftest.py
│   ├── test_classifier.py       # Task classification
│   ├── test_selector.py         # Model selection
│   ├── test_cost_estimator.py   # Cost estimation
│   ├── test_budget.py           # Budget enforcement
│   ├── test_fallback.py         # Fallback chains
│   ├── test_learning.py         # Adaptive learning
│   └── test_rate_limiter.py     # Rate limiting
├── test_providers/
│   ├── conftest.py
│   ├── test_registry.py         # Provider registration
│   ├── test_anthropic.py
│   ├── test_openai.py
│   ├── test_google.py
│   ├── test_deepseek.py
│   ├── test_groq.py
│   ├── test_mistral.py
│   ├── test_ollama.py
│   └── test_custom.py
├── test_tools/
│   ├── conftest.py
│   ├── test_file_read.py
│   ├── test_file_write.py
│   ├── test_file_edit.py
│   ├── test_directory.py
│   ├── test_search.py
│   ├── test_terminal.py
│   ├── test_browser.py
│   ├── test_screenshot.py
│   └── test_permissions.py
├── test_context/
│   ├── conftest.py
│   ├── test_manager.py
│   ├── test_repo_map.py
│   ├── test_summarizer.py
│   ├── test_session.py
│   └── test_memory.py
├── test_auth/
│   ├── conftest.py
│   ├── test_keyring_store.py
│   ├── test_env_store.py
│   ├── test_encrypted_store.py
│   └── test_validator.py
├── test_db/
│   ├── conftest.py
│   ├── test_database.py
│   ├── test_migrations.py
│   └── test_queries.py
├── test_cost/
│   ├── conftest.py
│   ├── test_tracker.py
│   ├── test_dashboard.py
│   ├── test_pricing.py
│   └── test_budget.py
├── test_git/
│   ├── conftest.py
│   ├── test_operations.py
│   └── test_auto_commit.py
├── test_security/
│   ├── conftest.py
│   ├── test_sandbox.py
│   ├── test_path_guard.py
│   ├── test_secret_filter.py
│   └── test_audit.py
└── test_integration/
    ├── conftest.py
    ├── test_routing_flow.py     # Full routing: classify → select → call → track
    ├── test_edit_flow.py        # Full edit: prompt → tool call → write → commit
    ├── test_fallback_flow.py    # Rate limit → fallback → success
    └── test_budget_flow.py      # Budget enforcement end-to-end
```

## Test Categories

### Unit Tests (fast, isolated, no I/O)
- Mock all external dependencies
- Test single functions/methods
- Run in < 100ms each
- Target: 90%+ coverage

```python
def test_classify_simple_rename_prompt():
    classifier = TaskClassifier(config=default_config())
    context = TaskContext(active_files=["main.py"])

    result = classifier.classify("rename the variable x to user_count", context)

    assert result == ComplexityTier.SIMPLE

def test_classify_architecture_prompt():
    classifier = TaskClassifier(config=default_config())
    context = TaskContext(active_files=["main.py", "db.py", "api.py", "models.py"])

    result = classifier.classify(
        "Design a microservices architecture for the payment system with event sourcing",
        context,
    )

    assert result == ComplexityTier.COMPLEX
```

### Integration Tests (slower, real I/O, mocked external APIs)
- Test component interactions
- Use real SQLite (in-memory), real filesystem (tmp_path)
- Mock only external APIs (LiteLLM, network)
- Run in < 5s each

```python
@pytest.mark.integration
async def test_full_routing_flow(mock_litellm, temp_db):
    """Classify prompt → select model → make API call → track cost."""
    router = Router(db=temp_db, config=default_config())

    mock_litellm.acompletion.return_value = mock_completion_response(
        content="Here's the fix...",
        usage={"prompt_tokens": 500, "completion_tokens": 200},
    )

    result = await router.route("fix the typo in line 5")

    assert result.model == "ollama/qwen2.5-coder:7b"
    assert result.tier == ComplexityTier.SIMPLE
    assert result.cost == 0.0
```

### Security Tests (critical, run on every PR)
```python
@pytest.mark.security
class TestPathGuard:
    def test_rejects_parent_directory_traversal(self, path_guard):
        with pytest.raises(SecurityError):
            path_guard.validate("/project/../../../etc/passwd")

    def test_rejects_symlink_escape(self, path_guard, tmp_path):
        symlink = tmp_path / "link"
        symlink.symlink_to("/etc/passwd")
        with pytest.raises(SecurityError):
            path_guard.validate(str(symlink))

    def test_rejects_null_byte_injection(self, path_guard):
        with pytest.raises(SecurityError):
            path_guard.validate("/project/file.py\x00.txt")

    def test_allows_valid_project_path(self, path_guard):
        result = path_guard.validate("/project/src/main.py")
        assert result == Path("/project/src/main.py")

@pytest.mark.security
class TestSecretFilter:
    def test_strips_api_keys_from_env(self, secret_filter):
        env = {
            "PATH": "/usr/bin",
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "OPENAI_API_KEY": "sk-secret",
            "HOME": "/Users/dev",
            "MY_SECRET_TOKEN": "token123",
        }
        filtered = secret_filter.filter_env(env)

        assert "PATH" in filtered
        assert "HOME" in filtered
        assert "ANTHROPIC_API_KEY" not in filtered
        assert "OPENAI_API_KEY" not in filtered
        assert "MY_SECRET_TOKEN" not in filtered

@pytest.mark.security
class TestSandbox:
    def test_command_timeout_kills_process(self, sandbox):
        result = sandbox.execute("sleep 60", timeout=1)
        assert result.timed_out is True

    def test_blocks_dangerous_commands(self, sandbox):
        with pytest.raises(SecurityError):
            sandbox.execute("rm -rf /")

    def test_output_truncated_at_limit(self, sandbox):
        result = sandbox.execute("yes", timeout=2)
        assert len(result.stdout) <= 100 * 1024  # 100KB limit
```

### Performance Tests (benchmarks, run periodically)
```python
@pytest.mark.slow
def test_classifier_performance():
    """Classification should complete in < 5ms."""
    classifier = TaskClassifier(config=default_config())
    context = TaskContext(active_files=["main.py"])

    start = time.perf_counter()
    for _ in range(1000):
        classifier.classify("fix the typo in main.py", context)
    elapsed = time.perf_counter() - start

    assert elapsed / 1000 < 0.005  # < 5ms per classification

@pytest.mark.slow
def test_repo_map_performance(large_project):
    """Repo map generation for 1000-file project should complete in < 10s."""
    mapper = RepoMapper(project_root=large_project)

    start = time.perf_counter()
    result = mapper.generate()
    elapsed = time.perf_counter() - start

    assert elapsed < 10.0
    assert len(result.entries) > 0
```

## Fixtures (conftest.py)

### Global Fixtures
```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture
def temp_project(tmp_path: Path) -> Path:
    """Create a minimal project directory."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text('def main():\n    print("hello")\n')
    (src / "__init__.py").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".prism.md").write_text("# Test Project\n")
    return tmp_path

@pytest.fixture
def default_config() -> Settings:
    """Default settings for testing."""
    return Settings(
        simple_threshold=0.3,
        medium_threshold=0.7,
        daily_budget=5.0,
        monthly_budget=50.0,
    )

@pytest.fixture
def temp_db(tmp_path: Path) -> Database:
    """In-memory SQLite database with schema applied."""
    db = Database(path=tmp_path / "test.db")
    db.initialize()
    return db
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/prism --cov-report=term-missing --cov-fail-under=90

# Run specific module tests
pytest tests/test_router/ -v

# Run security tests only
pytest -m security -v

# Run fast tests only (skip slow/integration)
pytest -m "not slow and not integration"

# Run with parallel execution
pytest -n auto

# Run with verbose output for debugging
pytest tests/test_router/test_classifier.py -v -s
```

## Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| router/ | 95% |
| security/ | 95% |
| auth/ | 95% |
| tools/ | 90% |
| db/ | 90% |
| cost/ | 90% |
| cli/ | 85% |
| context/ | 85% |
| config/ | 90% |
| git/ | 85% |
| **Overall** | **90%** |

## Mocking Guidelines

### What to Mock
- LiteLLM API calls (always)
- Network requests (always)
- OS keyring (in unit tests)
- File system (in unit tests, use tmp_path in integration tests)
- Time (when testing timeouts, rate limits)
- Subprocess (in unit tests, real in integration tests with sandbox)

### What NOT to Mock
- SQLite (use in-memory DB)
- Path resolution logic
- Configuration parsing
- Data validation
- Serialization/deserialization

### Mock Patterns
```python
# Mock LiteLLM completion
@pytest.fixture
def mock_litellm(mocker):
    mock = mocker.patch("prism.router.selector.litellm")
    mock.acompletion.return_value = AsyncMock(return_value=MockResponse(
        choices=[MockChoice(message=MockMessage(content="result"))],
        usage=MockUsage(prompt_tokens=100, completion_tokens=50),
    ))
    return mock

# Mock keyring
@pytest.fixture
def mock_keyring(mocker):
    store = {}
    mocker.patch("keyring.get_password", side_effect=lambda svc, usr: store.get(f"{svc}:{usr}"))
    mocker.patch("keyring.set_password", side_effect=lambda svc, usr, pwd: store.update({f"{svc}:{usr}": pwd}))
    return store
```

## CI Test Pipeline

```yaml
# Run on every PR
test:
  steps:
    - name: Lint
      run: ruff check src/ tests/

    - name: Type check
      run: mypy src/prism/

    - name: Security scan
      run: bandit -r src/prism/ -ll

    - name: Unit tests
      run: pytest -m "not slow and not integration" --cov=src/prism --cov-fail-under=90

    - name: Integration tests
      run: pytest -m integration --timeout=30

    - name: Security tests
      run: pytest -m security -v
```

# CONTRIBUTING.md — Contributing to Prism

## Welcome

Prism is an open-source project under Apache 2.0 license. Contributions are welcome from everyone.

## Getting Started

### Prerequisites
- Python 3.11+
- Git
- (Optional) Ollama for local model testing

### Setup
```bash
# Fork and clone
git clone https://github.com/GoparapukethaN/prism-cli.git
cd prism-cli

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[all,dev]"

# Install pre-commit hooks
pre-commit install

# Verify everything works
make review
```

## Contribution Types

### 1. New Provider
Add a LiteLLM-compatible model configuration:
1. Create `src/prism/providers/<provider>.py`
2. Add model configs with pricing and tier assignments
3. Register in `src/prism/providers/registry.py`
4. Add tests in `tests/test_providers/test_<provider>.py`
5. Update PROVIDER_SPECS.md

### 2. New Tool
Implement the `Tool` interface:
1. Create `src/prism/tools/<tool_name>.py`
2. Implement `name`, `description`, `parameters_schema`, `permission_level`, `execute()`
3. Register in `src/prism/tools/registry.py`
4. Add tests in `tests/test_tools/test_<tool_name>.py`
5. Update TOOL_SPECS.md

### 3. Routing Improvements
Improve classification accuracy or cost optimization:
1. Modify `src/prism/router/classifier.py` or `selector.py`
2. Add tests showing improvement over baseline
3. Include benchmark results in PR description

### 4. Bug Fixes
1. Create a test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Submit PR with "Fixes #<issue>" in description

### 5. Documentation
1. Update relevant .md files
2. Ensure code examples are correct and runnable

## Development Workflow

### Branch from main
```bash
git checkout main
git pull origin main
git checkout -b feat/your-feature
```

### Make changes
- Follow coding conventions in CONVENTIONS.md
- Write tests for all new code
- Run the full review suite:
  ```bash
  make review
  ```

### Commit
```bash
git add <specific files>
git commit -m "feat(module): concise description"
```

Follow commit conventions:
- `feat(scope)`: New feature
- `fix(scope)`: Bug fix
- `test(scope)`: Test additions
- `docs(scope)`: Documentation
- `refactor(scope)`: Code restructuring
- `perf(scope)`: Performance improvement
- `security(scope)`: Security fix

### Push and create PR
```bash
git push -u origin feat/your-feature
# Create PR via GitHub UI or gh cli
```

## Pull Request Guidelines

### PR Title
- Under 70 characters
- Follow commit message format: `feat(router): add context-window-aware routing`

### PR Body Template
```markdown
## Summary
- Brief description of what this PR does
- Why this change is needed

## Changes
- List of specific changes

## Test Plan
- [ ] Unit tests added/updated
- [ ] Integration tests (if applicable)
- [ ] Security tests (if applicable)
- [ ] All existing tests pass

## Checklist
- [ ] Code follows CONVENTIONS.md
- [ ] No hardcoded secrets
- [ ] All functions have type hints
- [ ] Tests have ≥90% coverage for changed code
- [ ] `ruff check` passes
- [ ] `mypy` passes
- [ ] `bandit` passes
- [ ] Documentation updated (if applicable)
```

## Testing Rules

### CRITICAL: No Real API Calls
- All tests must run completely offline
- All provider interactions must be mocked
- Never use real API keys in tests
- Use `pytest-mock` and `respx` for mocking
- Use hardcoded mock responses

### Coverage Requirements
- Overall: ≥ 90%
- Security modules: ≥ 95%
- New code in PR: ≥ 90%

## Code Review Process

1. Automated checks must pass (CI)
2. At least one maintainer review
3. All review comments addressed
4. Squash merge to main

## Reporting Issues

### Bug Reports
Include:
- Prism version (`prism --version`)
- Python version
- OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs (with secrets redacted)

### Feature Requests
Include:
- Use case description
- Proposed solution (if any)
- Alternatives considered

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Welcome newcomers
- Help each other learn

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

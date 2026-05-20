# CI_CD.md - Proposed Prism CI/CD Pipeline

This document is a target CI/CD design for Prism, not the current enforced release
process. The current public status is tracked in [SHOWCASE_STATUS.md](SHOWCASE_STATUS.md)
and [KNOWN_ISSUES.md](KNOWN_ISSUES.md): runtime tests and Ruff lint pass locally, while
strict mypy, Ruff format, Bandit, pip-audit, and live-provider validation are not release
gates yet.

## GitHub Repository
- **Owner**: GoparapukethaN
- **Repo**: prism-cli
- **Default branch**: main
- **Target protected branch rule**: main requires PR review and clean release gates once
  the non-gating checks are hardened

## GitHub Actions Workflows

### 1. Proposed CI Pipeline (on every PR and push to main)

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Ruff lint
        run: ruff check src/ tests/
      - name: Ruff format check
        run: ruff format --check src/ tests/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install -e ".[all,dev]"
      - name: Mypy
        run: mypy src/prism/

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Bandit security scan
        run: bandit -r src/prism/ -ll -f json -o bandit-report.json || true
      - name: Check for critical findings
        run: bandit -r src/prism/ -ll
      - name: Dependency audit
        run: pip-audit

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -e ".[all,dev]"
      - name: Run unit tests
        run: pytest -m "not slow and not integration" --cov=src/prism --cov-report=xml --cov-fail-under=90
      - name: Run integration tests
        run: pytest -m integration --timeout=60
      - name: Run security tests
        run: pytest -m security -v
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  test-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run core tests
        run: pytest -m "not slow and not integration" --timeout=60
```

### 2. Release Pipeline (on version tags)

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install build tools
        run: pip install build twine
      - name: Build package
        run: python -m build
      - name: Check package
        run: twine check dist/*
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*

  github-release:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          generate_release_notes: true
```

### 3. Dependency Updates (weekly)

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
```

## Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: no-commit-to-branch
        args: [--branch, main]
      - id: detect-private-key
```

## Makefile Targets

```makefile
.PHONY: install lint typecheck security test test-all review clean

install:
	pip install -e ".[all,dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/prism/

security:
	bandit -r src/prism/ -ll
	pip-audit

test:
	pytest -m "not slow and not integration" --cov=src/prism --cov-fail-under=90

test-all:
	pytest --cov=src/prism --cov-report=term-missing

test-security:
	pytest -m security -v

review: lint typecheck security test
	@echo "All checks passed!"

# Per-module review
review-module:
	@test -n "$(MODULE)" || (echo "Usage: make review-module MODULE=router" && exit 1)
	ruff check src/prism/$(MODULE)/
	mypy src/prism/$(MODULE)/
	bandit -r src/prism/$(MODULE)/ -ll
	pytest tests/test_$(MODULE)/ -v --cov=src/prism/$(MODULE)/ --cov-fail-under=90

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
```

## Proposed Release Process

### Version Bumping
1. Update version in `pyproject.toml`
2. Update CHANGELOG.md with release notes
3. Create PR with version bump
4. After merge, tag: `git tag v0.1.0 && git push --tags`
5. When release automation is enabled, GitHub Actions can build and publish to PyPI

### Versioning Scheme
- **SemVer**: MAJOR.MINOR.PATCH
- Pre-1.0: breaking changes increment MINOR
- Post-1.0: breaking changes increment MAJOR

## Target Quality Gates

- [ ] Ruff lint and runtime tests stay clean
- [ ] Strict mypy is reduced to zero errors before it becomes blocking
- [ ] Ruff format is applied in a dedicated mechanical pass before it becomes blocking
- [ ] Bandit and pip-audit findings are cleaned or documented before they become blocking
- [ ] Live-provider smoke coverage exists before provider behavior is advertised as
      validated
- [ ] Code coverage target is tracked before it becomes a merge requirement
- [ ] No bandit findings of medium+ severity
- [ ] No pip-audit vulnerabilities
- [ ] Tests pass on Python 3.11 and 3.12 before a packaged release

## Secrets Management (GitHub)

Only needed if release automation and coverage upload are enabled.

| Secret | Purpose |
|--------|---------|
| `PYPI_TOKEN` | PyPI publishing |
| `CODECOV_TOKEN` | Coverage reporting |

**NO API keys** stored in GitHub — all tests use mocks.

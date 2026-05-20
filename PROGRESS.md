# Prism Development Snapshot

This file is a development ledger, not a production-readiness claim. Prism is an
experimental model-routing CLI with broad local test coverage and several unfinished
validation areas. For current caveats, see `KNOWN_ISSUES.md`.

## Latest Local Audit

Audit date: 2026-05-20

- Local verification gate: `PYTHON=.venv/bin/python make verify` passes with Ruff
  clean, full pytest `5817 passed`, and CLI smoke checks.
- Focused routing/config/auth/security/cost test slice: `658 passed, 15 skipped`.
- Full mypy: not clean yet (`169 errors in 38 files`).
- Live provider calls: not validated yet; provider behavior is mostly covered through
  mocks and deterministic tests.
- Security review: no real API keys found in the repository during the latest local
  cleanup pass; known security limitations are tracked separately.

## Implemented Areas

These areas have code and local tests, but they should still be treated as experimental
until live-provider validation and type-checking debt are handled.

| Area | Current Evidence | Caveat |
|---|---|---|
| Configuration and auth | Settings, defaults, key handling, and auth tests | Live provider setup needs more real-world validation |
| SQLite state and migrations | Database models, queries, migrations, and tests | Long-running upgrade paths need more fixtures |
| Routing and fallback | Classifier, selector, fallback chains, budgets, and rate limits | Quality/cost choices are not benchmarked against production workloads |
| Cost tracking | Pricing tables, tracker, dashboard, and forecast tests | Pricing needs periodic provider review |
| CLI and REPL | Typer commands, prompt-toolkit REPL, slash-command coverage | Some commands are experimental workflows |
| Tool and file operations | Path guard, secret filtering, audit logging, tool registry | User-facing permission UX needs more manual testing |
| Context and project memory | Repo map, sessions, summaries, and memory modules | Large-repo behavior needs more profiling |
| Git workflow helpers | Checkpoints, undo, rollback history, and timeline tests | Needs more manual validation across real repos |
| Web and vision tools | HTTP/Playwright paths, screenshot/vision helpers, tests | Optional dependencies and provider-specific behavior vary |
| Orchestration experiments | Debate, mixture-of-agents, cascade, and swarm modules | Research-inspired experiments, not production-agent guarantees |

## Current Release Posture

- Public status: experimental/supporting repo.
- Recommended install path: source install from this repository.
- PyPI status: not published from this repo under the `prism-cli` name.
- Profile role: keep as a supporting systems project, not a flagship claim.

## Next Work

1. Reduce the strict mypy error count and document the remaining typed boundaries.
2. Run live-provider validation with OpenAI, Anthropic, Google, and local/Ollama paths.
3. Turn important provider checks into repeatable smoke tests.
4. Review command/file permissions through real interactive sessions.
5. Keep README claims tied to locally verified behavior.

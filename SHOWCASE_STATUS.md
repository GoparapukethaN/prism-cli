# Prism Showcase Status

This project is a supporting systems repo for experimenting with model routing, local
tool execution, cost tracking, and CLI workflow design. I keep it public because the
architecture and tests are useful to inspect, but I do not treat it as a production-ready
coding assistant.

## Verified Locally

Latest local verification: 2026-05-20.

```bash
PYTHON=.venv/bin/python make verify
```

That gate currently passes:

- Ruff lint for `src` and `tests`
- Full pytest suite: `5817 passed`, `8 warnings`
- CLI smoke checks for `python -m prism --help` and `python -m prism status`
- Default provider setup documented for the runtime registry:
  OpenAI, Anthropic, Google, DeepSeek, Groq, Mistral, OpenRouter, and Ollama

The smoke checks do not require provider API keys.

## Not Yet A Release Gate

These checks are intentionally tracked, but they are not clean enough to be part of the
local gate yet:

| Check | Current Status | Why It Matters |
|---|---|---|
| Ruff format | `209 files would be reformatted` | Needs one mechanical formatting pass before it can be enforced |
| Strict mypy | `162 errors in 35 files` | Typed boundaries need to be cleaned module by module |
| Bandit | `4` low findings, `1` medium XML parser finding | The dependency parser path needs a safer XML handling pass |
| Live providers | Not validated with real API keys | Mocked provider behavior does not prove provider-specific edge cases |
| Extended provider stubs | Kimi, Perplexity, Qwen, Cohere, Together AI, Fireworks AI, and custom endpoint configs are source-level experiments | They should stay out of the primary setup path until a live smoke proves them |

## How I Present It

- Experimental model-routing CLI, not a finished assistant.
- Broad local runtime tests, but type/security/format debt is still visible.
- Useful as evidence of CLI architecture, routing experiments, local persistence,
  cost tracking, and security-aware tool design.
- Not the flagship AI reliability project; the flagship remains the reliability/eval
  platform work.

## Next Focus

1. Fix Bandit XML parsing and cleanup ignored low-severity findings.
2. Run Ruff format in a dedicated mechanical commit.
3. Reduce mypy debt starting with shared result/config/cost models.
4. Add a small live-provider smoke script for one paid provider and one local provider.

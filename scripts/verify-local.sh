#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SMOKE_HOME="$(mktemp -d "${TMPDIR:-/tmp}/prism-local-verify.XXXXXX")"

"$PYTHON" -m ruff check src tests
env -u PRISM_HOME "$PYTHON" -m pytest -q
PRISM_HOME="$SMOKE_HOME" "$PYTHON" -m prism --help >/dev/null
PRISM_HOME="$SMOKE_HOME" "$PYTHON" -m prism status >/dev/null

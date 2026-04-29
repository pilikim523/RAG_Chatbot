#!/usr/bin/env bash
set -euo pipefail

# Keep this hook lightweight. It should not run slow integration tests.
if [ -f "pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
  uv run ruff format . >/dev/null 2>&1 || true
  uv run ruff check . --fix >/dev/null 2>&1 || true
fi

exit 0

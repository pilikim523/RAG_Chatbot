#!/usr/bin/env bash
set -euo pipefail

# SessionStart advisory hook. It does not block work.
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Notice: this project is tuned for macOS Apple Silicon development."
  exit 0
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
  echo "Notice: Apple Silicon arm64 was expected, got: $ARCH"
  exit 0
fi

if command -v system_profiler >/dev/null 2>&1; then
  system_profiler SPHardwareDataType | grep -E "Chip|Memory" || true
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Notice: ollama is not installed. Install with: brew install ollama"
fi

exit 0

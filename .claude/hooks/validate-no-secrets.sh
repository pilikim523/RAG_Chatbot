#!/usr/bin/env bash
set -euo pipefail

# Fast fallback secret scan. Replace with gitleaks/trufflehog in CI.
PATTERN='(OPENAI_API_KEY|ANTHROPIC_API_KEY|AWS_SECRET_ACCESS_KEY|GITHUB_TOKEN|BEGIN RSA PRIVATE KEY|BEGIN OPENSSH PRIVATE KEY)'

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git grep -n -E "$PATTERN" -- ':!*.example' ':!*.md' ':!uv.lock' >/tmp/claude_secret_scan.txt 2>/dev/null; then
    echo "Potential secret found:" >&2
    cat /tmp/claude_secret_scan.txt >&2
    exit 2
  fi
fi

exit 0

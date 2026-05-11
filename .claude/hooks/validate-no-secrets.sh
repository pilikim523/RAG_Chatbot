#!/usr/bin/env bash
set -euo pipefail

# Fast fallback secret scan. Replace with gitleaks/trufflehog in CI.
# Matches literal secret values assigned in code, not env var name references.
# Patterns:
#   sk-[a-zA-Z0-9]{20,}          OpenAI API key value
#   sk-ant-[a-zA-Z0-9]{20,}      Anthropic API key value
#   AKIA[A-Z0-9]{16}             AWS Access Key ID
#   ghp_[a-zA-Z0-9]{36}         GitHub personal access token
#   tvly-[a-zA-Z0-9-]{20,}      Tavily API key value
#   BEGIN (RSA|OPENSSH) PRIVATE KEY  PEM private keys
PATTERN='(sk-[a-zA-Z0-9]{20,}|sk-ant-[a-zA-Z0-9-]{20,}|AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|tvly-[a-zA-Z0-9-]{20,}|BEGIN RSA PRIVATE KEY|BEGIN OPENSSH PRIVATE KEY)'

if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git grep -n -E "$PATTERN" -- ':!*.example' ':!*.md' ':!uv.lock' ':!.claude/hooks/' >/tmp/claude_secret_scan.txt 2>/dev/null; then
    echo "Potential secret found:" >&2
    cat /tmp/claude_secret_scan.txt >&2
    exit 2
  fi
fi

exit 0

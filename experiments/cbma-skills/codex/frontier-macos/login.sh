#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_API_BASE
exec "$root/bin/codex" login "$@"

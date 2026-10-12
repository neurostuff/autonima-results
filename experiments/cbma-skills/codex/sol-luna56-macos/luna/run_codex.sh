#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="${1:?usage: run_codex.sh PROJECT [codex args]}"; shift
[[ "$project" =~ ^[a-z0-9_]+$ && -d "$root/projects/$project" ]] || { echo "unknown project" >&2; exit 2; }
"$root/../bin/python" "$root/../helpers/copy_articles.py" --check
if [[ $# == 0 ]]; then set -- "$(cat "$root/START_PROMPT.txt")"; fi
exec "$root/codex_session.sh" tui gpt-6.1-sol medium -C "$root/projects/$project" \
  --sandbox workspace-write -c 'approval_policy="never"' "$@"

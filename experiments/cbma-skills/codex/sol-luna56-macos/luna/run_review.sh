#!/usr/bin/env bash
# Headless end-to-end review for unattended execution. No stdin or stage approvals.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="${1:?usage: run_review.sh PROJECT [codex exec args]}"; shift
[[ "$project" =~ ^[a-z0-9_]+$ && -d "$root/projects/$project" ]] || { echo "unknown project" >&2; exit 2; }
"$root/../bin/python" "$root/../helpers/copy_articles.py" --check
if [[ $# == 0 ]]; then set -- "$(cat "$root/START_PROMPT.txt")"; fi
exec "$root/codex_session.sh" exec gpt-6.1-sol medium -C "$root/projects/$project" \
  --skip-git-repo-check --sandbox workspace-write -c 'approval_policy="never"' "$@" < /dev/null

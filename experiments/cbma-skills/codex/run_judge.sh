#!/usr/bin/env bash
# One tool-free response; shared Python transport saves raw ledger output.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# == 3 && "$1" =~ ^[a-z0-9_]+$ ]] || { echo "usage: run_judge.sh PROJECT STAGE BATCH" >&2; exit 2; }
exec python3 "$here/projects/$1/.agents/skills/cbma-review/scripts/single_response.py" "$here" "$@"

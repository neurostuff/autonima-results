#!/usr/bin/env bash
# Install in an arm root. Per-project orchestrator defaults live in
# projects/PROJECT/.codex/orchestrator.json; absent config preserves Astra/high.
# ORCH_MODEL and ORCH_EFFORT remain explicit overrides.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="${1:?usage: run_codex.sh PROJECT [codex args]}"; shift
[[ "$project" =~ ^[a-z0-9_]+$ ]] || { echo "invalid project name" >&2; exit 2; }
ws="$here/projects/$project"
[[ -d "$ws" ]] || { echo "no workspace $ws" >&2; exit 2; }
defaults="$(python3 - "$ws/.codex/orchestrator.json" <<'PY'
import json, pathlib, re, sys
path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text()) if path.exists() else {}
model = data.get('model', 'gpt-6-astra')
effort = data.get('effort', 'high')
if not isinstance(model, str) or not re.fullmatch(r'[a-z0-9.-]+', model):
    sys.exit('invalid orchestrator model in project config')
if effort not in {'low', 'medium', 'high', 'xhigh', 'max', 'ultra'}:
    sys.exit('invalid orchestrator effort in project config')
print(model, effort)
PY
)"
read -r default_model default_effort <<< "$defaults"
exec "$here/../portkey_codex.sh" tui "${ORCH_MODEL:-$default_model}" "${ORCH_EFFORT:-$default_effort}" \
  -C "$ws" "$@"

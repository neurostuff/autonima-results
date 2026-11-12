#!/usr/bin/env bash
# Run NeuroStore feature arms live, a few projects at a time, each followed by one retry pass that
# re-decides only what failed (failures are never cached).
#
#   scripts/run_neurostore_feature_arms.sh <rec|rec-comb|rec-comb-noabs|rec-global> <project>...
#       [LANES=3] [PASSES="live live-retry1"]
#
# Needs ~/.keys/portkey.key (OPENAI_API_KEY, OPENAI_API_GATEWAY), the feat/neurostore-features
# autonima venv (AUTONIMA_PY, default ~/repos/autonima-nsfeat/.venv/bin/python), and NeuroStore's
# analysis query with base_study_ids deployed. Logs: reports/neurostore_feature_arms/logs/.
# Run it detached (setsid nohup ... &) so it outlives the shell that started it, and check each
# log for 429s: autonima exits 0 on quota errors.
set -euo pipefail
arm="${1:?usage: run_neurostore_feature_arms.sh arm project...}"; shift
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${AUTONIMA_PY:-$HOME/repos/autonima-nsfeat/.venv/bin/python}"
logs="$repo/reports/neurostore_feature_arms/logs"; mkdir -p "$logs"
set -a; . "$HOME/.keys/portkey.key"; set +a
export PYTHONPATH="$HOME/repos/ACE" REPO="$repo" PY="$py" ARM="$arm" LOGS="$logs"
export PASSES="${PASSES:-live live-retry1}"

one() {
  project="$1"
  cd "$REPO"
  for pass in $PASSES; do
    log="$LOGS/$project-$ARM-$pass.log"
    echo "$(date +%H:%M) start $project $ARM $pass"
    if ! "$PY" -W ignore scripts/neurostore_feature_arms.py "$project" "$ARM" --live --workers 8 > "$log" 2>&1; then
      echo "$(date +%H:%M) FAILED $project $ARM $pass (see $log)"; return 0
    fi
  done
  echo "$(date +%H:%M) done $project $ARM"
}
export -f one
printf '%s\n' "$@" | xargs -P "${LANES:-3}" -I{} bash -c 'one "$@"' _ {}
echo "$(date +%H:%M) all done"

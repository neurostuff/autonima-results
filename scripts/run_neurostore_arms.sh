#!/usr/bin/env bash
# Run NeuroStore arms live, a few projects at a time, each followed by one retry pass that
# re-annotates only the studies whose LLM call failed (failures are never cached).
#
#   scripts/run_neurostore_arms.sh <coords|search> <project>...   [LANES=3] [PASSES="live live-retry1"]
#
# PASSES names the runs to make, in order; any pass after the first only re-annotates what is
# still failed. PASSES=live-retry2 adds one more retry pass to finished arms.
#
# Needs ~/.keys/portkey.key (OPENAI_API_KEY, OPENAI_API_GATEWAY) and the feat/neurostore
# autonima venv. Logs: reports/neurostore_arms/logs/<project>-<kind>-live[-retry1].log.
# Run it detached (setsid nohup ... &) so it outlives the shell that started it.
set -euo pipefail
kind="${1:?usage: run_neurostore_arms.sh coords|search project...}"; shift
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${AUTONIMA_PY:-$HOME/repos/autonima-neurostore/.venv/bin/python}"
logs="$repo/reports/neurostore_arms/logs"; mkdir -p "$logs"
set -a; . "$HOME/.keys/portkey.key"; set +a
export PYTHONPATH="$HOME/repos/ACE" REPO="$repo" PY="$py" KIND="$kind" LOGS="$logs"
export PASSES="${PASSES:-live live-retry1}"

one() {
  project="$1"
  cd "$REPO"
  for pass in $PASSES; do
    log="$LOGS/$project-$KIND-$pass.log"
    echo "$(date +%H:%M) start $project $KIND $pass"
    if ! "$PY" -W ignore scripts/neurostore_arms.py "$project" "$KIND" --live --workers 8 > "$log" 2>&1; then
      echo "$(date +%H:%M) FAILED $project $KIND $pass (see $log)"; return 0
    fi
  done
  echo "$(date +%H:%M) done $project $KIND"
}
export -f one
printf '%s\n' "$@" | xargs -P "${LANES:-3}" -I{} bash -c 'one "$@"' _ {}
echo "$(date +%H:%M) all done"

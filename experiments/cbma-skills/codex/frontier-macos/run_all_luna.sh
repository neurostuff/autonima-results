#!/usr/bin/env bash
# Four independent, sequential, headless Luna runs. Requires login first.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
failures=0
for project in decision_making dementia executive_function social; do
  logdir="$root/luna/projects/$project/$project/results"
  mkdir -p "$logdir"
  logfile="$logdir/orchestrator_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "Starting Luna $project; log: $logfile"
  if "$root/luna/run_review.sh" "$project" > "$logfile" 2>&1; then
    echo "Luna $project session finished; final coverage/status is in its RUN_NOTES.md."
  else
    echo "Luna $project session failed; inspect $logfile" >&2
    failures=$((failures + 1))
  fi
done
[[ "$failures" == 0 ]]

#!/usr/bin/env bash
# Separate ChatGPT login profile. No global config or API keys are copied.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:?usage: codex_session.sh tui|exec MODEL EFFORT [args]}"
model="${2:?missing model}"; effort="${3:?missing effort}"; shift 3
case "$mode" in tui|exec) ;; *) echo "invalid mode" >&2; exit 2 ;; esac
case "$effort" in low|medium|high|xhigh|max) ;; *) echo "invalid effort" >&2; exit 2 ;; esac
export PATH="$root/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export NILEARN_DATA="$root/runtime/data/nilearn"
export MPLCONFIGDIR="$root/.run-state/matplotlib"
export CBMA_FRONTIER_ROOT="$root"
mkdir -p "$root/.run-state" "$root/runtime/data"
unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_API_BASE OPENAI_API_GATEWAY AUTONIMA_MODEL_PREFIX
subcommand=()
if [[ "$mode" == exec ]]; then subcommand=(exec); fi
exec "$root/bin/codex" "${subcommand[@]}" -m "$model" \
  -c 'model_provider="openai"' -c "model_reasoning_effort=\"$effort\"" \
  -c 'features.multi_agent=false' -c 'features.multi_agent_v2=false' \
  --add-dir "$root/.run-state" --add-dir "$root/runtime/data" "$@"

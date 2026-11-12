#!/usr/bin/env bash
# Shared transport for interactive orchestrators and fresh headless judges.
# Usage: portkey_codex.sh tui|exec MODEL EFFORT [codex arguments...]
set -euo pipefail
mode="${1:?usage: portkey_codex.sh tui|exec MODEL EFFORT [codex args]}"
bare_model="${2:?missing model}"
effort="${3:?missing effort}"
shift 3
case "$mode" in tui|exec) ;; *) echo "unsupported mode: $mode" >&2; exit 2 ;; esac
case "$effort" in low|medium|high|xhigh|max|ultra) ;; *) echo "unsupported effort: $effort" >&2; exit 2 ;; esac
# Never log credentials, including when invoked through bash -x.
set +x
set -a
. "${PORTKEY_KEY_FILE:-$HOME/.keys/portkey.key}"
set +a
: "${OPENAI_API_GATEWAY:?missing gateway in key file}"
: "${OPENAI_API_KEY:?missing API key in key file}"
: "${AUTONIMA_MODEL_PREFIX:?missing model prefix in key file}"
prefix="${AUTONIMA_MODEL_PREFIX%/}"
case "$bare_model" in "$prefix"/*) model="$bare_model" ;; @*|*/*) echo "model belongs to an unexpected route" >&2; exit 2 ;; *) model="$prefix/$bare_model" ;; esac
case "$OPENAI_API_GATEWAY" in https://*) ;; *) echo "gateway must use HTTPS" >&2; exit 2 ;; esac
# JSON strings are valid TOML strings; quote the URL without shell interpolation.
provider_config="$(python3 - <<'PY'
import json, os
print('{ name = "Portkey", base_url = ' + json.dumps(os.environ['OPENAI_API_GATEWAY']) +
      ', env_key = "OPENAI_API_KEY", wire_api = "responses", requires_openai_auth = false, supports_websockets = false }')
PY
)"
subcommand=()
if [[ "$mode" == exec ]]; then subcommand=(exec); fi
exec codex "${subcommand[@]}" \
  -m "$model" \
  -c 'model_provider="portkey"' \
  -c "model_providers.portkey=$provider_config" \
  -c "model_reasoning_effort=\"$effort\"" \
  -c 'features.multi_agent=false' \
  -c 'features.multi_agent_v2=false' \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/rerun_autonima_meta_batch.sh

PROJECTS=(
"projects/vbm_of_ptsd/v1.yaml"
"projects/vbm_of_ptsd/v1-annotation-only.yaml"
"projects/vbm_of_ptsd/v1-recent.yaml"
"projects/vbm_of_substance_use/v1.yaml"
"projects/vbm_of_substance_use/v2-recent.yaml"
"projects/vbm_of_substance_use/v2-annotation-only.yaml"
"projects/vbm_of_substance_use/v2.yaml"
"projects/decision_making/v1.yaml"
"projects/decision_making/v2-annotation-only.yaml"
"projects/decision_making/v2.yaml"
"projects/cue_reactivity/v1.yaml"
"projects/cue_reactivity/v2.yaml"
"projects/cue_reactivity/v3.yaml"
"projects/cue_reactivity/v4.yaml"
"projects/cue_reactivity/v5-recent.yaml"
"projects/cue_reactivity/v5-annotation-only.yaml"
"projects/cue_reactivity/v5.yaml"
"projects/cue_reactivity/v5-recent.yaml"
"projects/social/v3-all_pmids-multi_analysis-ft-gpt52.yaml"
"projects/social/v3-all_pmids-multi_analysis-ft.yaml"
"projects/social/v3-all_pmids-multi_analysis.yaml"
"projects/social/v3-all_pmids.yaml"
"projects/social/v3-annotation-only.yaml"
"projects/social/v3-search-all_pmids-multi_analysis-ft.yaml"
"projects/social/v3-search-all_pmids-multi_analysis.yaml"
"projects/social/v3-search-all_pmids.yaml"
"projects/social/v2-all_pmids.yaml"
"projects/social/v2.yaml"
"projects/executive_function/v1.yaml"
)


resolve_project_yaml() {
  local p="$1"
  if [[ -f "$p" ]]; then
    printf "%s" "$p"
    return 0
  fi
  if [[ "$p" != *.yaml && -f "${p}.yaml" ]]; then
    printf "%s" "${p}.yaml"
    return 0
  fi
  return 1
}

for project in "${PROJECTS[@]}"; do
  if resolved="$(resolve_project_yaml "$project")"; then
    project_name="${resolved%.yaml}"
    echo "==> autonima meta ${project_name} --estimator-args '{\"n_cores\": 9}'"
    autonima meta "${project_name}" --estimator-args '{"n_cores": 9}'
  else
    echo "!! Skipping missing project config: ${project}" >&2
  fi
done

echo "Batch meta run complete."

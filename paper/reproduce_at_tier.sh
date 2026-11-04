#!/usr/bin/env bash
#
# Recompute the paper's evidence tables and figures at ONE run tier.
#
# paper/reproduce.sh rebuilds the manuscript's display items at the `best` tier -- the curated
# preferred runs, where benchmark-informed tuning is allowed. This does the same at another tier,
# into a separate root, so the two can be compared without either overwriting the other.
#
#     paper/reproduce_at_tier.sh [--tier untuned] [--root manual-tier] [--figures-only]
#
# HOW THE TIER IS ENFORCED, which is the whole difficulty
#
# Half the chain takes --tier. The other half does not: compute_gold_survival's `canonical_run`
# (shared by compute_stage_precision_recall, compute_attainable_recall and
# false_positive_composition) picks the highest-numbered vN directory it can SEE. So the tier is
# enforced twice over:
#
#   * by flag            where the script supports --tier
#   * by visibility      scripts/build_tier_tree.py builds $ROOT/projects containing symlinks to
#                        only this tier's runs, making `max(versions)` correct by construction
#
# Getting this wrong is silent -- the chain would mix a tuned screening table with untuned maps
# and report it as one result -- so both mechanisms are applied, never one.
#
# WHY THE EVIDENCE STAGES RUN IN PLACE BUT THE FIGURES ARE MIRRORED
#
# The evidence scripts resolve the gold standard as REPO_ROOT.parent/neurometabench, so they must
# run from the real repo root and be redirected with flags. The figure scripts have no input-path
# flag -- they read REPO_ROOT/reports/* directly -- so they are copied into $ROOT/paper/, where
# REPO_ROOT resolves to $ROOT and they read this tier's tables. They are copied fresh on every
# run, so they cannot drift from paper/.
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TIER=untuned ROOT=manual-tier FIGURES_ONLY=0 SKIP_SLOW=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --figures-only) FIGURES_ONLY=1; SKIP_SLOW=1; shift ;;
    --skip-slow) SKIP_SLOW=1; shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

# Absolute, because several scripts print their output path via Path.relative_to(REPO_ROOT),
# which raises on a relative path. The work succeeds and then the run dies on the log line.
ROOT="$(cd "$(dirname "$ROOT")" 2>/dev/null && pwd)/$(basename "$ROOT")" || ROOT="$PWD/$ROOT"
mkdir -p "$ROOT"

PY=(pixi run python)
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run()  { echo "   \$ $*"; "$@"; }

PROJECTS=(cue_reactivity decision_making dementia emotion_regulation_2022
          executive_function problem_solving social vbm_of_ptsd vbm_of_substance_use)
RP="$ROOT/projects" RR="$ROOT/reports"

if [ "$FIGURES_ONLY" = 0 ]; then
  step "0/5  shadow tree exposing only the '$TIER' runs"
  run "${PY[@]}" scripts/build_tier_tree.py --tier "$TIER" --root "$ROOT"

  step "1/5  per-project baseline comparison"
  for p in "${PROJECTS[@]}"; do
    run "${PY[@]}" scripts/compare_baselines_to_benchmark.py \
        --project "$p" --tier "$TIER" --output-dir "$RP/$p/reports" || echo "   !! $p failed"
  done

  step "2/5  per-project map matrices (annotation-only arm)"
  while IFS=$'\t' read -r p r; do
    run "${PY[@]}" scripts/compare_meta_to_benchmark.py \
        --project-dir "projects/$p" --run-dir "$r" \
        --output-dir "$RP/$p/reports/manual_vs_auto_meta_$r" || echo "   !! $p/$r failed"
  done < <("${PY[@]}" - "$TIER" <<'PYEOF'
import sys; sys.path.insert(0, "scripts")
from run_tiers import resolve_tier
tier = sys.argv[1]
for p in ["cue_reactivity","decision_making","dementia","emotion_regulation_2022",
          "executive_function","problem_solving","social","vbm_of_ptsd",
          "vbm_of_substance_use"]:
    r = resolve_tier(p, "annotation_only", tier)
    if r: print(f"{p}\t{r}")
PYEOF
)

  step "3/5  cross-project evidence tables"
  # dementia is excluded from the map-level table only: its reference pools coordinates across
  # studies, so its analysis units are not comparable at map level.
  run "${PY[@]}" scripts/compile_best_baselines.py --projects-root "$RP" \
      --exclude-project dementia --output "$RR/cross_project_best_baseline.csv"
  ALL_BB="$(mktemp -d)"
  run "${PY[@]}" scripts/compile_best_baselines.py --projects-root "$RP" \
      --output "$ALL_BB/best_baseline_all.csv"
  run "${PY[@]}" scripts/compile_analysis_counts.py --tier "$TIER" \
      --baseline-table "$ALL_BB/best_baseline_all.csv" --output "$RR/analysis_counts.csv"
  rm -rf "$ALL_BB"
  run "${PY[@]}" scripts/decompose_selection_gain.py --tier "$TIER" \
      --baseline-table "$RR/cross_project_best_baseline.csv" \
      --output "$RR/selection_decomposition.csv"
  run "${PY[@]}" scripts/run_cross_project_analysis_reports.py --tier "$TIER" \
      --output-dir "$RR/cross_project_analysis"
  # MUST follow run_cross_project_analysis_reports: annotation_value reads each run's
  # reports/annotation_review_reports/annotation_metrics_by_mode.json, which that step writes.
  # Run earlier it silently emits blank annotation_f1 rather than failing.
  run "${PY[@]}" scripts/annotation_value.py --tier "$TIER" \
      --projects-root "$RP" --output "$RR/annotation_value.csv"
  run "${PY[@]}" scripts/run_cross_project_screening_reports.py --tier "$TIER" \
      --output-dir "$RR/cross_project_screening"
  # feeds figure S1; reads the registry directly, so it is tier-independent by nature, but its
  # table must live beside this tier's others for the figure script to find it
  run "${PY[@]}" scripts/compile_tier_progression.py --output "$RR/tier_progression.csv"
  run "${PY[@]}" scripts/compute_gold_survival.py --projects-root "$RP" \
      --output "$RR/gold_survival_by_stage.csv"
  run "${PY[@]}" scripts/compute_stage_precision_recall.py --projects-root "$RP" \
      --output "$RR/stage_precision_recall.csv"
  run "${PY[@]}" scripts/compute_attainable_recall.py --projects-root "$RP" \
      --output "$RR/attainable_recall_by_stage.csv"
  run "${PY[@]}" scripts/false_positive_composition.py --projects-root "$RP" \
      --output "$RR/false_positive_composition.csv"
fi

if [ "$SKIP_SLOW" = 0 ]; then
  step "3.5/5  size-matched null (~45 min: 32 contrasts x 500 MKDA refits)  ->  figure 5"
  # --source, NOT --tier. The run column in the source table wins, so --tier is inert whenever
  # the source names runs -- which the default source always does. Passing this tier's own
  # annotation_value.csv is what actually selects the runs; the script now refuses the
  # combination that silently computed a "untuned" null from the tuned runs.
  run "${PY[@]}" scripts/bootstrap_annotation_null.py \
      --source "$RR/annotation_value.csv" --output "$RR/annotation_null.csv" --jobs "${JOBS:-6}"
fi

step "4/5  mirror the figure scripts (fresh copies, so they cannot drift)"
mkdir -p "$ROOT/paper"
run cp paper/make_nature_methods_figures.py paper/make_brain_map_figure.py \
      paper/make_er_surface_figure.py paper/manuscript_numbers.py "$ROOT/paper/"
# The figure scripts import run_tiers from REPO_ROOT/scripts, which is $ROOT here. Symlink
# rather than copy: run_tiers resolves __file__ through the link to the real scripts/, so it
# keeps reading the shared run_categories.yaml instead of needing a copy of the registry.
run ln -sfn "$PWD/scripts" "$ROOT/scripts"

step "5/5  figures  ->  $RR/nature_methods_figures/"
run "${PY[@]}" "$ROOT/paper/make_nature_methods_figures.py" || echo "   !! figures failed"

printf '\n\033[1mdone\033[0m  tier=%s  root=%s\n' "$TIER" "$ROOT"

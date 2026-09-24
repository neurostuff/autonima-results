#!/usr/bin/env bash
#
# Regenerate every figure and number in the manuscript, from the run artifacts
# tracked under projects/, in dependency order.
#
# WHY THIS EXISTS
#
# The chain from run outputs to figures is real but was never written down: it
# lives in the import graph of ~27 scripts and in whoever last ran them. That is
# how reports/annotation_bootstrap_null.csv came to hold 35 rows after three of
# its columns were excluded, and how figureS4 sat a week stale at a superseded
# column count. A file
# that states the order, and fails loudly when a step does, is the difference
# between "the figures are in git" and "the figures can be rebuilt".
#
# USAGE
#
#   paper/reproduce.sh                  # everything (~1h, most of it stage 4)
#   paper/reproduce.sh --figures-only   # redraw from existing reports/*.csv (~3 min)
#   paper/reproduce.sh --skip-slow      # everything except the size-matched null
#
# TWO INTERPRETERS, ON PURPOSE
#
# Three figure scripts need nilearn >= 0.13 for surface and glass-brain
# rendering; the pixi environment pins nilearn 0.10.1 because NiMARE 0.2.1 does.
# Those three run under the system python3. This is a real constraint, not an
# accident -- see paper/README.md.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY=(pixi run python)
SYSPY=/usr/bin/python3
JOBS="${JOBS:-$(nproc)}"

DO_EVIDENCE=1 DO_SLOW=1
for a in "$@"; do
  case "$a" in
    --figures-only) DO_EVIDENCE=0; DO_SLOW=0 ;;
    --skip-slow)    DO_SLOW=0 ;;
    -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $a" >&2; exit 2 ;;
  esac
done

START=$(date +%s)
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run()  { echo "   \$ $*"; "$@"; }

PROJECTS=(cue_reactivity decision_making dementia emotion_regulation_2022
          executive_function problem_solving social vbm_of_ptsd vbm_of_substance_use)

if [ "$DO_EVIDENCE" = 1 ]; then

  step "1/6  per-project baseline comparison  ->  projects/*/reports/baseline_vs_autonima.csv"
  # Feeds Figure 4, Figure 6 and everything derived from the best-baseline table.
  for p in "${PROJECTS[@]}"; do
    run "${PY[@]}" scripts/compare_baselines_to_benchmark.py --project "$p" --tier best
  done

  step "2/6  per-project map matrices (annotation-only arm)  ->  manual_vs_auto_meta_<run>/tables/"
  # annotation_value.py reads these matrices rather than recomputing from maps,
  # so they must be refreshed before it runs or its pearson columns go stale.
  while IFS=$'\t' read -r p r; do
    run "${PY[@]}" scripts/compare_meta_to_benchmark.py \
        --project-dir "projects/$p" --run-dir "$r" \
        --output-dir "projects/$p/reports/manual_vs_auto_meta_$r"
  done < <("${PY[@]}" - <<'PYEOF'
import sys; sys.path.insert(0, "scripts")
from run_tiers import resolve_tier
for p in ["cue_reactivity","decision_making","dementia","emotion_regulation_2022",
          "executive_function","problem_solving","social","vbm_of_ptsd",
          "vbm_of_substance_use"]:
    r = resolve_tier(p, "annotation_only", "best")
    if r: print(f"{p}\t{r}")
PYEOF
)

  step "3/6  cross-project evidence tables  ->  reports/*.csv"
  # compile_best_baselines must precede everything that reads its output.
  # --exclude-project dementia: its reference pools coordinates across studies,
  # so its analysis units are not comparable at map level (28 columns, 8 projects).
  run "${PY[@]}" scripts/compile_best_baselines.py --exclude-project dementia
  run "${PY[@]}" scripts/compile_analysis_counts.py
  run "${PY[@]}" scripts/decompose_selection_gain.py
  run "${PY[@]}" scripts/annotation_value.py
  run "${PY[@]}" scripts/compile_tier_progression.py

  run "${PY[@]}" scripts/run_cross_project_analysis_reports.py --tier best
  run "${PY[@]}" scripts/run_cross_project_screening_reports.py --tier best

  run "${PY[@]}" scripts/compute_gold_survival.py
  run "${PY[@]}" scripts/compute_stage_precision_recall.py
  run "${PY[@]}" scripts/compute_stage_precision_recall.py \
      --suffix -allstudies --output reports/stage_precision_recall_allstudies.csv
  run "${PY[@]}" scripts/compute_attainable_recall.py
  # manuscript_numbers.py reads this one as data, not as an import -- easy to miss
  # when tracing the chain through the import graph alone.
  run "${PY[@]}" scripts/false_positive_composition.py
fi

if [ "$DO_SLOW" = 1 ]; then
  step "4/6  size-matched null (~45 min: 32 contrasts x 500 MKDA refits)"
  run "${PY[@]}" scripts/bootstrap_annotation_null.py --jobs "$JOBS"
fi

step "5/6  figures  ->  reports/nature_methods_figures/"
run "${PY[@]}" paper/make_nature_methods_figures.py          # all but S4
run "$SYSPY" paper/make_brain_map_figure.py --mode all       # figureS4
run "$SYSPY" paper/make_brain_map_figure.py --mode contrast
run "$SYSPY" paper/make_er_surface_figure.py                 # paper Figure 5

step "6/6  manuscript numbers"
run "${PY[@]}" paper/manuscript_numbers.py

step "verifying every expected display item was written"
FIGS=(figure2_precision_and_attainable_recall figure3_recover_and_select_analyses
      figure4_pipeline_vs_baseline figure5_selection_decomposition
      figureS1_tier_progression figureS2_pool_mismatch figureS3_size_matched_null
      figureS4_brain_maps_all figureS5_measured_cost
      figure_er_surface_contrasts figure_brain_maps_contrast
      figure_annotation_precision_recall figure_gold_retention_raw)
missing=0 stale=0
for f in "${FIGS[@]}"; do
  for ext in pdf png; do
    p="reports/nature_methods_figures/$f.$ext"
    if [ ! -f "$p" ]; then echo "   MISSING $p"; missing=$((missing+1))
    elif [ "$(stat -c %Y "$p")" -lt "$START" ]; then echo "   STALE   $p"; stale=$((stale+1))
    fi
  done
done
n=${#FIGS[@]}
echo "   $((n - missing / 2 - stale / 2)) / $n display items freshly written"
[ "$missing" -gt 0 ] && { echo "FAILED: $missing missing"; exit 1; }
[ "$stale" -gt 0 ] && [ "$DO_EVIDENCE" = 1 ] && echo "   NOTE: $stale not rewritten this run"
printf '\ndone in %d min\n' $(( ($(date +%s) - START) / 60 ))

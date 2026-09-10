#!/usr/bin/env python3
"""Split the pipeline's advantage into paper selection and analysis selection.

THE CLAIM THIS TESTS

Result 5 says the gain comes from selecting *analyses*, not from selecting *papers*. That is
argued from the annotation-only arm, which holds the study pool fixed. This tests it from the
other direction, on the end-to-end arm, using a third map that sits between the baseline and the
full pipeline:

    baseline        a search, coordinates pooled. No LLM anywhere.
    screening only  the canonical run's `all_analyses` column -- every parsed analysis from the
                    studies that SURVIVED SCREENING, with no annotation. Papers chosen, analyses
                    not.
    full pipeline   the run's column for that contrast. Papers chosen and analyses chosen.

So `screening only - baseline` is what paper selection buys, and `pipeline - screening only` is
what analysis selection buys, on the same runs and the same maps the headline uses.

READ THE FIRST NUMBER CAREFULLY

`all_analyses` is ONE map per project, scored against every contrast that project has -- which is
the honest representation of "no analysis selection": without it you have a single map and no way
to answer a specific contrast. But 31 of the 32 baselines are TARGETED searches, built per
contrast. So the baseline is not an unselected corpus either; it selects papers by query where the
pipeline selects them by LLM.

The right conclusion is therefore NOT "paper selection does nothing". It is that **two different
ways of selecting papers come out even**, and everything the pipeline gains comes from the step
the baseline cannot perform at all. Stating it the first way would be overclaiming.

Metrics follow the metric/map rule: r-squared on the unthresholded z maps.

Writes reports/selection_decomposition.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tiers import resolve_tier  # noqa: E402
from benchmark_exclusions import filter_rows  # noqa: E402

MANUAL_BASE = Path("/home/zorro/repos/neurometabench/analysis")
UNCORRECTED_MAP = "z.nii.gz"
SCREENING_ONLY_COLUMN = "all_analyses"


def auto_column(project: str, key: str) -> str:
    f = REPO_ROOT / "projects" / project / "nmb_mappings.json"
    if f.exists():
        try:
            return (json.loads(f.read_text()).get("annotation_mappings") or {}).get(key, key)
        except ValueError:
            pass
    return key


def gold_dir(project: str, key: str) -> Path | None:
    d = MANUAL_BASE / project / key
    if d.is_dir():
        return d
    for c in (MANUAL_BASE / project).glob("*"):
        if c.is_dir() and c.name.lower().replace("-", "_") == key.lower().replace("-", "_"):
            return c
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-table", type=Path,
                    default=REPO_ROOT / "reports" / "cross_project_best_baseline.csv")
    ap.add_argument("--tier", default="best")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" / "selection_decomposition.csv")
    args = ap.parse_args(argv)

    import nibabel as nib

    cache: dict[Path, np.ndarray] = {}

    def load(path: Path) -> np.ndarray:
        if path not in cache:
            cache[path] = nib.load(str(path)).get_fdata()
        return cache[path]

    def r2(a: np.ndarray, b: np.ndarray) -> float:
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m].ravel(), b[m].ravel())[0, 1] ** 2)

    with open(args.baseline_table) as f:
        wanted = filter_rows(list(csv.DictReader(f)), label="selection decomposition")

    rows, skipped = [], []
    for r in wanted:
        project, key = r["project"], r["manual_annotation"]
        run = resolve_tier(project, "canonical", args.tier)
        if not run:
            skipped.append(f"{project}/{key}: no canonical run"); continue
        meta = REPO_ROOT / "projects" / project / run / "outputs" / "meta_analysis_results"
        gd = gold_dir(project, key)
        screen = meta / SCREENING_ONLY_COLUMN / UNCORRECTED_MAP
        if not (gd and (gd / UNCORRECTED_MAP).exists() and screen.exists()):
            skipped.append(f"{project}/{key}: missing gold or screening-only map"); continue
        gold = load(gd / UNCORRECTED_MAP)
        if load(screen).shape != gold.shape:
            skipped.append(f"{project}/{key}: shape mismatch"); continue

        base, pipe = float(r["best_available"]), float(r["autonima"])
        scr = r2(load(screen), gold)
        rows.append({
            "project": project, "manual_annotation": key, "run": run,
            "baseline_source": r.get("best_available_source", ""),
            "r2_baseline": round(base, 6),
            "r2_screening_only": round(scr, 6),
            "r2_pipeline": round(pipe, 6),
            "gain_paper_selection": round(scr - base, 6),
            "gain_analysis_selection": round(pipe - scr, 6),
            "gain_total": round(pipe - base, 6),
        })

    if not rows:
        print("nothing to decompose")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    paper = [x["gain_paper_selection"] for x in rows]
    analysis = [x["gain_analysis_selection"] for x in rows]
    total = [x["gain_total"] for x in rows]
    n = len(rows)
    print(f"{n} columns\n")
    for label, v in (("paper selection   (baseline -> screening only)", paper),
                     ("analysis selection (screening only -> pipeline)", analysis),
                     ("total              (baseline -> pipeline)", total)):
        print(f"  {label:48} mean {st.mean(v):+.4f}  median {st.median(v):+.4f}  "
              f"positive {sum(1 for x in v if x > 0):>2}/{n}")
    if st.mean(total):
        print(f"\n  share of the total attributable to analysis selection: "
              f"{100 * st.mean(analysis) / st.mean(total):.0f}%")
    src = {s: [x["gain_paper_selection"] for x in rows if x["baseline_source"] == s]
           for s in {x["baseline_source"] for x in rows}}
    print("\n  paper-selection gain by what the baseline is:")
    for s, v in sorted(src.items()):
        if v:
            print(f"    {s or '?':10} n={len(v):>3}  mean {st.mean(v):+.4f}")
    for m in skipped:
        print(f"  skipped: {m}")
    print(f"  wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

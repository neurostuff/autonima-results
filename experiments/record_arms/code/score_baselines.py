#!/usr/bin/env python3
"""Score each project's per-column baseline map against its manual map.

`reports/cross_project_best_baseline.csv` already holds baseline numbers, but they cannot be
used against the arms: they were produced on another host with an unrecorded estimator, and
on shared columns the committed pipeline values differ from ours by up to 0.66 (cannabis
0.021 there against 0.686 here). Subtracting those baselines from our maps would measure the
estimator change, not the pipeline.

So the baselines are re-estimated here from their own studysets with exactly the settings the
arms used -- MKDA density, NiMARE default kernel, FDR independent -- and scored with the same
masks as `meta_r2.py`. Both sides then come from one chain.

Each baseline run is named for the manual column it targets and carries a single
`all_analyses` annotation key, so the mapping from directory to column is the directory name.
"""
from __future__ import annotations

import csv, sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_r2 import DICE_T, R, brain_mask, dice, manual_base, rr  # noqa: E402

PROJECTS = ["vbm_of_ptsd", "cue_reactivity", "dementia", "vbm_of_substance_use"]

rows = []
for project in PROJECTS:
    for d in sorted((R / "projects" / project / "baselines").glob("*/")):
        column = d.name
        apath = d / "outputs/meta_analysis_results/all_analyses/z.nii.gz"
        mpath = manual_base(project) / project / column / "z.nii.gz"
        if not apath.is_file():
            print(f"  {project}/{column}: no baseline map", file=sys.stderr)
            continue
        if not mpath.is_file():
            print(f"  {project}/{column}: no manual map", file=sys.stderr)
            continue
        mimg = nib.load(str(mpath))
        man, auto = mimg.get_fdata(), nib.load(str(apath)).get_fdata()
        if man.shape != auto.shape:
            print(f"  {project}/{column}: shape {auto.shape} vs {man.shape}", file=sys.stderr)
            continue
        finite = np.isfinite(man) & np.isfinite(auto)
        brain = finite & brain_mask(mimg)
        a_all, b_all = man[finite], auto[finite]
        a_br, b_br = man[brain], auto[brain]
        r_all, r2_all = rr(a_all, b_all)
        r_br, r2_br = rr(a_br, b_br)
        rows.append({"project": project, "manual_analysis": column,
                     "dice": round(dice(a_all, b_all), 4),
                     "pearson_r": r_br, "r2": r2_br,
                     "r_allfinite": r_all, "r2_allfinite": r2_all})
        print(f"  {project:22} {column:30} dice={rows[-1]['dice']:<7} "
              f"r2_brain={r2_br:<8} r2_all={r2_all}")

if not rows:
    sys.exit("no baseline rows produced")
out = R / "experiments/record_arms/data/baseline_meta_metrics.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"\nwrote {out}  ({len(rows)} rows)")

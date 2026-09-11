#!/usr/bin/env python3
"""Dice, Pearson r and r^2 for each arm's map against its manual benchmark map.

Metrics are `compare_meta_to_benchmark.py`'s, reused rather than reinvented: dice over
`z > 1.96`, Pearson over the voxels finite in both maps. That script is not called directly
only because it imports seaborn at module scope for figures and neither venv on this host
has it; nothing about the numbers differs.

r^2 is r squared, which is how `annotation_value.csv` feeds Figure 5 -- its own comment
records that the identity was verified against baseline_vs_autonima.csv, 104/104 rows.
"""
from __future__ import annotations

import csv, json, sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import pearsonr

R = Path("/data/james/pondie-vs-fulltext/repos/autonima-results")
def manual_base(project: str) -> Path:
    """Each project keeps its own manual maps; there is no shared root on this host."""
    return (R / "projects" / project / "reports/manual_vs_auto_meta_fair"
            / "fair_manual_meta/manual_analysis")
DICE_T = 1.96

ARMS = {
    "vbm_of_ptsd": ("v1-A1-mini", "v1-record-with-evidence", "v1-record-no-evidence"),
    "cue_reactivity": ("v5-gpt-A1-mini", "v5-gpt-record-with-evidence", "v5-gpt-record-no-evidence"),
    "dementia": ("v3", "v3-record-with-evidence", "v3-record-no-evidence"),
    "vbm_of_substance_use": ("v2", "v2-record-with-evidence", "v2-record-no-evidence"),
}
LABEL = {0: "full text", 1: "record + evidence", 2: "record, no evidence"}


def dice(a, b, t=DICE_T):
    ba, bb = a > t, b > t
    s = ba.sum() + bb.sum()
    return float(2 * (ba & bb).sum() / s) if s else 0.0


def pearson(a, b):
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(pearsonr(a, b)[0])


rows = []
for project, runs in ARMS.items():
    mapping = json.loads((R / "projects" / project / "nmb_mappings.json").read_text())
    for manual_name, auto_name in (mapping.get("annotation_mappings") or {}).items():
        mpath = manual_base(project) / project / manual_name / "z.nii.gz"
        if not mpath.is_file():
            print(f"  {project}/{manual_name}: no manual map", file=sys.stderr)
            continue
        man = nib.load(str(mpath)).get_fdata()
        for i, run in enumerate(runs):
            apath = R / "projects" / project / run / "outputs/meta_analysis_results" / auto_name / "z.nii.gz"
            if not apath.is_file():
                continue
            auto = nib.load(str(apath)).get_fdata()
            if auto.shape != man.shape:
                print(f"  {project}/{run}/{auto_name}: shape {auto.shape} vs {man.shape}",
                      file=sys.stderr)
                continue
            mask = np.isfinite(man) & np.isfinite(auto)
            a, b = man[mask].ravel(), auto[mask].ravel()
            r = pearson(a, b)
            rows.append({"project": project, "arm": LABEL[i], "run": run,
                         "manual_analysis": manual_name, "auto_analysis": auto_name,
                         "dice": round(dice(a, b), 4),
                         "pearson_r": round(r, 4) if r == r else "",
                         "r2": round(r * r, 4) if r == r else ""})

out = R / "reports/record_arms_meta_metrics.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}  ({len(rows)} rows)\n")
print(f"{'project':22} {'analysis':30} {'arm':20} {'dice':>7} {'r':>7} {'r2':>7}")
for r_ in rows:
    print(f"{r_['project'][:22]:22} {r_['manual_analysis'][:30]:30} {r_['arm']:20} "
          f"{r_['dice']:>7} {r_['pearson_r']:>7} {r_['r2']:>7}")

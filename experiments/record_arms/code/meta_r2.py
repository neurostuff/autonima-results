#!/usr/bin/env python3
"""Dice and map-vs-manual correlation for each arm, under four masks.

WHY FOUR MASKS

`compare_meta_to_benchmark.py` correlates over `np.isfinite(manual) & np.isfinite(auto)`,
which for these maps is the entire 91x109x91 volume. That is not a brain mask: the maps are
whole-volume z images and roughly 90-96% of their voxels are exactly zero in BOTH maps, so
the correlation is dominated by agreement about empty space. Measured on VBM PTSD full text:
95.8% of voxels are zero in both, the whole-volume r is 0.690, and restricting to voxels
where either map found something drops it to -0.198.

So the whole-volume number is kept for continuity with the rest of the repo, but it is not
the one to quote. The columns are:

    r_allfinite   every finite voxel -- the repo's existing convention, inflated
    r_brain       MNI152 brain mask resampled onto the map grid   <- PRIMARY, `r2` mirrors it
    r_nonzero     voxels nonzero in either map, within the brain
    r_sig         voxels above z > 1.96 in either map, within the brain

`r_brain` is primary because it is the only one of the three restricted masks that does not
condition on the data. `r_nonzero` and `r_sig` both select voxels using the values being
correlated, which biases the estimate; they are reported as sensitivity, not as the result.

NOTE ON THE `r2` COLUMN. It now holds the brain-masked r^2, not the whole-volume one it held
before 2026-09-11. `r2_allfinite` carries the old number. Anything reading `r2` from an older
copy of this file is reading a different quantity.

Dice is unaffected by masking: it counts only voxels above z > 1.96, and no such voxel lies
outside the brain.
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

_BRAIN: dict[tuple, np.ndarray] = {}


def brain_mask(ref_img) -> np.ndarray:
    """MNI152 brain mask on the reference image's grid.

    nilearn ships it at 99x117x95; these maps are the FSL 91x109x91 grid, so it has to be
    resampled rather than used as-is. Nearest-neighbour, because it is a binary mask.
    """
    key = (ref_img.shape, ref_img.affine.tobytes())
    if key not in _BRAIN:
        from nilearn.datasets import load_mni152_brain_mask
        from nilearn.image import resample_to_img
        m = resample_to_img(load_mni152_brain_mask(resolution=2), ref_img,
                            interpolation="nearest", force_resample=True, copy_header=True)
        _BRAIN[key] = np.asarray(m.dataobj) > 0.5
    return _BRAIN[key]


def dice(a, b, t=DICE_T):
    ba, bb = a > t, b > t
    s = ba.sum() + bb.sum()
    return float(2 * (ba & bb).sum() / s) if s else 0.0


def pearson(a, b):
    if a.size < 3 or not np.any(a != a[0]) or not np.any(b != b[0]):
        return float("nan")
    return float(pearsonr(a, b)[0])


def rr(a, b):
    """(r, r^2) with r^2 blank when r is undefined."""
    r = pearson(a, b)
    if r != r:
        return "", ""
    return round(r, 4), round(r * r, 4)


rows = []
for project, runs in ARMS.items():
    mapping = json.loads((R / "projects" / project / "nmb_mappings.json").read_text())
    for manual_name, auto_name in (mapping.get("annotation_mappings") or {}).items():
        mpath = manual_base(project) / project / manual_name / "z.nii.gz"
        if not mpath.is_file():
            print(f"  {project}/{manual_name}: no manual map", file=sys.stderr)
            continue
        mimg = nib.load(str(mpath))
        man = mimg.get_fdata()
        for i, run in enumerate(runs):
            apath = (R / "projects" / project / run / "outputs/meta_analysis_results"
                     / auto_name / "z.nii.gz")
            if not apath.is_file():
                continue
            auto = nib.load(str(apath)).get_fdata()
            if auto.shape != man.shape:
                print(f"  {project}/{run}/{auto_name}: shape {auto.shape} vs {man.shape}",
                      file=sys.stderr)
                continue
            finite = np.isfinite(man) & np.isfinite(auto)
            brain = finite & brain_mask(mimg)
            a_all, b_all = man[finite], auto[finite]
            a_br, b_br = man[brain], auto[brain]
            nz = (a_br != 0) | (b_br != 0)
            sg = (a_br > DICE_T) | (b_br > DICE_T)

            r_all, r2_all = rr(a_all, b_all)
            r_br, r2_br = rr(a_br, b_br)
            r_nz, r2_nz = rr(a_br[nz], b_br[nz])
            r_sg, r2_sg = rr(a_br[sg], b_br[sg])
            rows.append({
                "project": project, "arm": LABEL[i], "run": run,
                "manual_analysis": manual_name, "auto_analysis": auto_name,
                "dice": round(dice(a_all, b_all), 4),
                "n_brain": int(brain.sum()),
                "n_nonzero": int(nz.sum()), "n_sig": int(sg.sum()),
                "pearson_r": r_br, "r2": r2_br,                 # primary: brain-masked
                "r_allfinite": r_all, "r2_allfinite": r2_all,
                "r_nonzero": r_nz, "r2_nonzero": r2_nz,
                "r_sig": r_sg, "r2_sig": r2_sg,
            })

if not rows:
    sys.exit("no rows produced")

for out in (R / "reports/record_arms_meta_metrics.csv",
            R / "experiments/record_arms/data/record_arms_meta_metrics.csv"):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")

print(f"\n{'project':22} {'analysis':26} {'arm':20} {'dice':>6} "
      f"{'r2_brain':>9} {'r2_all':>8} {'r2_nonz':>8} {'r2_sig':>8}")
for r_ in rows:
    print(f"{r_['project'][:22]:22} {r_['manual_analysis'][:26]:26} {r_['arm']:20} "
          f"{r_['dice']:>6} {str(r_['r2']):>9} {str(r_['r2_allfinite']):>8} "
          f"{str(r_['r2_nonzero']):>8} {str(r_['r2_sig']):>8}")

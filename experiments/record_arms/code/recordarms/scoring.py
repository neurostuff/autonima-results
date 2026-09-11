"""Step 8: screening against the pinned gold, and maps against the manual benchmark.

Screening is reported under two denominators because they differ enough to mislead. The
paired one is what compare_arms.py reports -- papers all arms screened in common -- and on
vbm_of_ptsd it reads recall 0.941 where the end-to-end figure over the full gold is 0.727.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import pearsonr

from . import paths
from .checks import bench_commit, gold
from .spec import Spec

DICE_T = 1.96
_BRAIN: dict = {}


def brain_mask(ref_img):
    """MNI152 brain mask on the map grid. nilearn ships 99x117x95; the maps are the FSL
    91x109x91 grid, so it is resampled nearest-neighbour rather than used as-is."""
    key = (ref_img.shape, ref_img.affine.tobytes())
    if key not in _BRAIN:
        from nilearn.datasets import load_mni152_brain_mask
        from nilearn.image import resample_to_img
        m = resample_to_img(load_mni152_brain_mask(resolution=2), ref_img,
                            interpolation="nearest", force_resample=True, copy_header=True)
        _BRAIN[key] = np.asarray(m.dataobj) > 0.5
    return _BRAIN[key]


def dice(a, b, t=DICE_T) -> float:
    ba, bb = a > t, b > t
    s = ba.sum() + bb.sum()
    return float(2 * (ba & bb).sum() / s) if s else 0.0


def rr(a, b):
    if a.size < 3 or not np.any(a != a[0]) or not np.any(b != b[0]):
        return "", ""
    r = float(pearsonr(a, b)[0])
    return round(r, 4), round(r * r, 4)


def _prf(inc: set[str], g: set[str]) -> dict:
    tp, fp, fn = len(inc & g), len(inc - g), len(g - inc)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(pr, 4), "recall": round(rc, 4),
            "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0}


def screening(spec: Spec) -> list[dict]:
    g = gold(spec.meta_pmid)
    decided, included = {}, {}
    for label, run in spec.arms.items():
        f = paths.outputs(spec.project, run) / "fulltext_screening_results.json"
        if not f.is_file():
            continue
        rows = json.loads(f.read_text())["screening_results"]
        decided[label] = {str(r["study_id"]) for r in rows}
        included[label] = {str(r["study_id"]) for r in rows
                           if r["decision"] == "included_fulltext"}
    if not included:
        return []
    common = set.intersection(*decided.values())
    out = []
    for label in included:
        row = {"project": spec.project, "arm": label, "run": spec.arms[label],
               "gold": len(g), "n_common": len(common),
               "benchmark_commit": bench_commit()}
        row |= {f"endtoend_{k}": v for k, v in _prf(included[label], g).items()}
        row |= {f"paired_{k}": v for k, v in
                _prf(included[label] & common, g & common).items()}
        out.append(row)
    return out


def _pair(man_path: Path, auto_path: Path) -> dict | None:
    if not (man_path.is_file() and auto_path.is_file()):
        return None
    mimg = nib.load(str(man_path))
    man, auto = mimg.get_fdata(), nib.load(str(auto_path)).get_fdata()
    if man.shape != auto.shape:
        return None
    finite = np.isfinite(man) & np.isfinite(auto)
    brain = finite & brain_mask(mimg)
    a_all, b_all = man[finite], auto[finite]
    a_br, b_br = man[brain], auto[brain]
    nz = (a_br != 0) | (b_br != 0)
    r_all, r2_all = rr(a_all, b_all)
    r_br, r2_br = rr(a_br, b_br)
    _, r2_nz = rr(a_br[nz], b_br[nz])
    return {"dice": round(dice(a_all, b_all), 4), "pearson_r": r_br, "r2": r2_br,
            "r_allfinite": r_all, "r2_allfinite": r2_all, "r2_nonzero": r2_nz,
            "n_brain": int(brain.sum())}


def maps(spec: Spec) -> list[dict]:
    out = []
    for manual_key, auto_key in spec.mapping.items():
        man = paths.manual_map(spec.project, manual_key)
        for label, run in spec.arms.items():
            m = _pair(man, paths.auto_map(spec.project, run, auto_key))
            if m:
                out.append({"project": spec.project, "arm": label, "run": run,
                            "manual_analysis": manual_key, "auto_analysis": auto_key, **m})
        b = _pair(man, paths.baseline_map(spec.project, manual_key))
        if b:
            out.append({"project": spec.project, "arm": "baseline",
                        "run": f"baselines/{manual_key}", "manual_analysis": manual_key,
                        "auto_analysis": "all_analyses", **b})
    return out


def write(spec: Spec) -> None:
    paths.DATA.mkdir(parents=True, exist_ok=True)
    for name, rows in (("screening", screening(spec)), ("maps", maps(spec))):
        if not rows:
            print(f"  {name}: nothing to write")
            continue
        dest = paths.DATA / f"{spec.project}_{name}.csv"
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
        with dest.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) + [k for k in keys
                                                               if k not in rows[0]])
            w.writeheader(); w.writerows(rows)
        print(f"  wrote {dest} ({len(rows)} rows)")
    # a combined marker the status check looks for
    (paths.DATA / f"{spec.project}_metrics.csv").write_text(
        f"# see {spec.project}_screening.csv and {spec.project}_maps.csv\n")

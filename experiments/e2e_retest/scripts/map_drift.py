#!/usr/bin/env python3
"""Per-annotation-group agreement between a retest's meta-analysis maps and the original's.

This is the measurement the whole retest exists for: decision-level flip rates only matter insofar
as they reach the brain map, and this expresses drift in the units the paper already reports.

Two metrics, matching `scripts/compare_meta_to_benchmark.py` so the numbers are directly
comparable to `reports/baseline_vs_autonima.csv`:

- **r2** from Pearson correlation on the **uncorrected** `z.nii.gz`, over voxels finite in both
  maps. This is the primary measure here: it uses the whole continuous pattern and does not depend
  on a threshold.
- **dice** on the **FDR-corrected** map, which is threshold-dependent and discards magnitude, but
  is what a reader pictures when they look at a published figure.

They can disagree, and the disagreement is informative: high r2 with low dice means the pattern is
stable but the significance boundary moved, which is a much milder problem than the pattern itself
being unstable.

Groups are reported in three classes, because averaging them together misleads:

- **manual-matched** -- has a counterpart in the published meta-analysis, so it is what §7 scores
  and what a reader would care about. Read from `reports/baseline_vs_autonima.csv`.
- **unmatched** -- a real annotation group that simply has no manual counterpart. Subject to the
  same drift as matched groups; just not scored in the paper.
- **bypass arm** (`all_*`) -- skips analysis selection and mostly skips full-text gating, so it
  inherits far less of the drift by construction. Folding these into a mean flatters the result.

Usage:
    python experiments/e2e_retest/scripts/map_drift.py \\
        --original projects/emotion_regulation_2022/v4 \\
        --retest   experiments/e2e_retest/emotion_regulation_2022/v4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import pearsonr


def _manual_matched(project_dir: Path) -> set[str]:
    """Annotation columns scored against the manual benchmark, per the project's report."""
    report = project_dir / "reports" / "baseline_vs_autonima.csv"
    if not report.exists():
        return set()
    with report.open(encoding="utf-8") as handle:
        return {
            (row.get("auto_column") or "").strip()
            for row in csv.DictReader(handle)
            if (row.get("auto_column") or "").strip()
        }


def _classify(group: str, matched: set[str]) -> str:
    if group.startswith("all_"):
        return "bypass"
    if matched:
        return "matched" if group in matched else "unmatched"
    return "matched"          # no report to consult; treat every non-all_ group as of interest


def _r2(a: np.ndarray, b: np.ndarray) -> float:
    """r-squared over voxels finite in both maps, as compare_meta_to_benchmark does."""
    mask = np.isfinite(a) & np.isfinite(b)
    va, vb = a[mask].ravel(), b[mask].ravel()
    if va.size < 2 or np.all(va == va[0]) or np.all(vb == vb[0]):
        return float("nan")
    return float(pearsonr(va, vb)[0] ** 2)


def _dice(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    ba, bb = np.nan_to_num(a) > threshold, np.nan_to_num(b) > threshold
    denom = int(ba.sum()) + int(bb.sum())
    return float(2.0 * np.logical_and(ba, bb).sum() / denom) if denom else float("nan")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--retest", type=Path, required=True)
    parser.add_argument("--map-filename", default="z.nii.gz",
                        help="uncorrected map, used for r2")
    parser.add_argument("--corrected-map-filename", default="z_corr-FDR_method-indep.nii.gz",
                        help="FDR-corrected map, used for dice")
    parser.add_argument("--dice-threshold", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    a_root = args.original / "outputs" / "meta_analysis_results"
    b_root = args.retest / "outputs" / "meta_analysis_results"
    if not (a_root.is_dir() and b_root.is_dir()):
        print(f"missing meta_analysis_results under {args.original} or {args.retest}")
        return 1

    groups = sorted({p.name for p in a_root.iterdir() if p.is_dir()} &
                    {p.name for p in b_root.iterdir() if p.is_dir()})
    if not groups:
        print("no annotation groups in common")
        return 1

    matched = _manual_matched(args.original.parent)
    print(f"r2 on {args.map_filename} (finite-in-both voxels); "
          f"dice on {args.corrected_map_filename} at z > {args.dice_threshold}")
    print(f"manual-matched columns: {sorted(matched) or '(none recorded)'}\n")
    print(f"{'annotation group':<26}{'class':<11}{'r2':>8}{'dice':>8}{'vox orig':>10}{'delta vox':>11}")
    print("-" * 74)

    rows = []
    for group in groups:
        pa_u, pb_u = a_root / group / args.map_filename, b_root / group / args.map_filename
        pa_c, pb_c = (a_root / group / args.corrected_map_filename,
                      b_root / group / args.corrected_map_filename)
        if not all(p.exists() for p in (pa_u, pb_u, pa_c, pb_c)):
            print(f"{group:<26}{'(map missing in one run)':>48}")
            continue

        r2 = _r2(nib.load(str(pa_u)).get_fdata(), nib.load(str(pb_u)).get_fdata())
        ca, cb = nib.load(str(pa_c)).get_fdata(), nib.load(str(pb_c)).get_fdata()
        dice = _dice(ca, cb, args.dice_threshold)
        va = int((np.nan_to_num(ca) > args.dice_threshold).sum())
        vb = int((np.nan_to_num(cb) > args.dice_threshold).sum())
        cls = _classify(group, matched)
        rows.append({"group": group, "class": cls, "r2": r2, "dice": dice,
                     "voxels_original": va, "voxels_retest": vb})
        print(f"{group:<26}{cls:<11}{r2:>8.4f}{dice:>8.4f}{va:>10,}{vb - va:>+11,}")

    print("-" * 74)
    for cls, label in (("matched", "manual-matched"), ("unmatched", "unmatched groups"),
                       ("bypass", "bypass arms (all_*)")):
        subset = [r for r in rows if r["class"] == cls]
        vals = [(r["r2"], r["dice"]) for r in subset if not np.isnan(r["r2"])]
        if not vals:
            continue
        r2s, dices = [v[0] for v in vals], [v[1] for v in vals]
        worst = min(subset, key=lambda r: r["r2"])
        print(f"  {label:<22}n={len(vals):<3} mean r2 {np.mean(r2s):.4f}   "
              f"mean dice {np.mean(dices):.4f}   worst r2 {np.min(r2s):.4f} ({worst['group']})")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"map_filename": args.map_filename,
             "corrected_map_filename": args.corrected_map_filename,
             "dice_threshold": args.dice_threshold,
             "manual_matched": sorted(matched), "groups": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare, per sub-annotation, the full autonima pipeline against two baselines.

THE QUESTION
------------
How much does LLM screening (study- and analysis-level) buy over a coarse, search-only
meta-analysis? For each manual sub-meta-analysis we line up three arms against the same
manual map:

  autonima       the full pipeline's map for that sub-annotation (search -> screen ->
                 parse -> annotate)
  baseline_broad the project's `all_studies` column: every study the project's ONE broad
                 search returned that we could parse coordinates from, no screening
  baseline_sub   the same idea but over a search targeted at THIS sub-annotation
                 (see scripts/run_baseline_searches.py), column `all_analyses`

`baseline_sub` exists because `baseline_broad` is an unfairly weak opponent whenever the
benchmark ran one broad search and split the hits into sub-topics: nobody targeting only the
alcohol sub-meta-analysis would search the whole substance-use literature. Beating
`baseline_broad` while losing to `baseline_sub` would mean the apparent win came from giving
the baseline a bad search, not from screening.

POOLS ARE NOT EQUALISED
-----------------------
Each arm keeps whatever its own search plus our retrieval could actually obtain. This is a
real-world end-to-end comparison: given comparable effort, whose map lands closer to the
manual result. Equalising study pools is not attempted -- for sub-topic searches it is not
even well defined, since some gold studies are multi-substance papers no sub-topic query can
reach. Pool sizes (studies, coordinate points) are reported next to every metric so the
asymmetry stays visible instead of being silently corrected away.

METRICS
-------
Matches scripts/compare_meta_to_benchmark.py so numbers are comparable to existing reports:
FDR-corrected maps, voxels finite across all compared maps, Dice at z > 1.96, Pearson r.
R^2 is reported alongside Dice because Dice depends on a threshold and so moves with how
many studies entered a map, which is exactly what differs between these arms.

LAYOUT
------
Baseline runs are read from projects/<project>/baselines/<key>/, falling back to the
legacy flat projects/<project>/baseline-<key>/ when that is what exists.

USAGE
-----
    python scripts/compare_baselines_to_benchmark.py --project vbm_of_substance_use
    python scripts/compare_baselines_to_benchmark.py --project vbm_of_substance_use \
        --autonima-run v2 --output-dir projects/vbm_of_substance_use/reports/baseline_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANUAL_ANALYSIS_BASE = Path("/home/zorro/repos/neurometabench/analysis")
DEFAULT_MANUAL_NIMADS_BASE = Path("/home/zorro/repos/neurometabench/data/nimads")
CORRECTED_MAP = "z_corr-FDR_method-indep.nii.gz"
DICE_THRESHOLD = 1.96
BASELINE_PREFIX = "baseline-"   # legacy flat layout
BASELINES_DIR = "baselines"     # current layout: <project>/baselines/<key>/
BROAD_COLUMN = "all_studies"       # project-wide, screening-free pool from the broad search
SUB_COLUMN = "all_analyses"        # the baseline run skips screening, so this is its pool


def compute_dice(a: np.ndarray, b: np.ndarray, threshold: float = DICE_THRESHOLD) -> float:
    ba, bb = a > threshold, b > threshold
    denom = int(ba.sum()) + int(bb.sum())
    if denom == 0:
        return 0.0
    return float(2.0 * int((ba & bb).sum()) / denom)


def compute_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(pearsonr(a, b)[0])


def column_pool(run_dir: Path, column: str) -> dict[str, Any]:
    """Studies and coordinate points feeding one annotation column of a run."""
    out: dict[str, Any] = {"studies": None, "points": None}
    ann_path = run_dir / "outputs" / "annotation_results.json"
    ss_path = run_dir / "outputs" / "nimads_studyset.json"
    if not ann_path.exists() or not ss_path.exists():
        return out
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    included_analyses = {
        str(x.get("analysis_id"))
        for x in ann
        if x.get("annotation_name") == column and x.get("include")
    }
    if not included_analyses:
        return out
    studyset = json.loads(ss_path.read_text(encoding="utf-8"))
    studies = studyset.get("studies", studyset)
    n_studies = 0
    n_points = 0
    for study in studies:
        hit = 0
        for analysis in study.get("analyses") or []:
            if str(analysis.get("id")) in included_analyses:
                hit += 1
                n_points += len(analysis.get("points") or [])
        if hit:
            n_studies += 1
    out["studies"] = n_studies
    out["points"] = n_points
    return out


def baseline_run_dir(project_dir: Path, key: str) -> Path:
    """Run dir for a baseline, preferring the nested layout over the legacy flat one."""
    nested = project_dir / BASELINES_DIR / key
    if (nested / "outputs").is_dir():
        return nested
    legacy = project_dir / f"{BASELINE_PREFIX}{key}"
    if (legacy / "outputs").is_dir():
        return legacy
    return nested   # report the expected path when neither exists


def pick_autonima_run(project_dir: Path, explicit: str | None) -> str:
    if explicit:
        if not (project_dir / explicit / "outputs" / "meta_analysis_results").is_dir():
            raise SystemExit(f"no meta results under {project_dir / explicit}")
        return explicit
    candidates = [
        p.name
        for p in sorted(project_dir.iterdir())
        if p.is_dir()
        and not p.name.startswith(BASELINE_PREFIX)
        and p.name != BASELINES_DIR
        and (p / "outputs" / "meta_analysis_results").is_dir()
    ]
    if not candidates:
        raise SystemExit(f"no run with meta results in {project_dir}")
    plain = [c for c in candidates if re.fullmatch(r"v\d+", c)]
    return (plain or candidates)[-1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--project", required=True)
    ap.add_argument("--autonima-run", default=None,
                    help="run dir supplying the pipeline and broad-baseline maps (default: highest vN)")
    ap.add_argument("--manual-analysis-base", type=Path, default=DEFAULT_MANUAL_ANALYSIS_BASE)
    ap.add_argument("--map-filename", default=CORRECTED_MAP)
    ap.add_argument("--dice-threshold", type=float, default=DICE_THRESHOLD)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    project_dir = REPO_ROOT / "projects" / args.project
    if not project_dir.is_dir():
        raise SystemExit(f"no such project: {project_dir}")

    spec_path = project_dir / "baselines.yaml"
    if not spec_path.exists():
        raise SystemExit(f"no baselines.yaml in {project_dir}; run run_baseline_searches.py first")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    keys = [e["manual_annotation"] for e in spec.get("baselines") or []]

    mapping_path = project_dir / "nmb_mappings.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
    auto_col_for = (mapping.get("annotation_mappings") or {})

    auto_run = pick_autonima_run(project_dir, args.autonima_run)
    auto_meta = project_dir / auto_run / "outputs" / "meta_analysis_results"

    # Resolve the broad arm. Prefer a baseline entry flagged `broad_control: true`,
    # because it was run in the same retrieval vintage as the other baselines. The
    # project run's `all_studies` column is the fallback, but it is only comparable if
    # that run happens to share the baselines' vintage -- typically it predates them,
    # which silently understates the broad baseline (its pool is smaller, and a smaller
    # screening-free pool scores HIGHER, so the bias favours the baseline).
    broad_key = next(
        (e["manual_annotation"] for e in (spec.get("baselines") or []) if e.get("broad_control")),
        None,
    )
    broad_path: Path | None = None
    broad_label = f"{auto_run}:{BROAD_COLUMN}"
    if broad_key:
        candidate = (baseline_run_dir(project_dir, broad_key) / "outputs"
                     / "meta_analysis_results" / SUB_COLUMN / args.map_filename)
        if candidate.exists():
            broad_path = candidate
            broad_label = f"{baseline_run_dir(project_dir, broad_key).name}:{SUB_COLUMN}"
        else:
            print(f"NOTE: broad_control {broad_key!r} has no map yet; "
                  f"falling back to {auto_run}:{BROAD_COLUMN} (vintage may differ)", file=sys.stderr)
    print(f"broad arm      : {broad_label}")
    print(f"project        : {args.project}")
    print(f"autonima run   : {auto_run}")
    print(f"map            : {args.map_filename}   dice z > {args.dice_threshold}")
    print(f"pools are NOT equalised; sizes reported per arm\n")

    rows: list[dict[str, Any]] = []
    for key in keys:
        manual_path = args.manual_analysis_base / args.project / key / args.map_filename
        auto_col = auto_col_for.get(key, key)
        arms = {
            "autonima": auto_meta / auto_col / args.map_filename,
            "baseline_broad": broad_path or (auto_meta / BROAD_COLUMN / args.map_filename),
            "baseline_sub": baseline_run_dir(project_dir, key) / "outputs"
                            / "meta_analysis_results" / SUB_COLUMN / args.map_filename,
        }
        broad_pool = (column_pool(baseline_run_dir(project_dir, broad_key), SUB_COLUMN)
                      if broad_path else column_pool(project_dir / auto_run, BROAD_COLUMN))
        pools = {
            "autonima": column_pool(project_dir / auto_run, auto_col),
            "baseline_broad": broad_pool,
            "baseline_sub": column_pool(baseline_run_dir(project_dir, key), SUB_COLUMN),
        }

        print(f"=== {key}  (autonima column: {auto_col}) ===")
        if not manual_path.exists():
            print(f"  SKIP: no manual map at {manual_path}\n")
            continue
        present = {name: p for name, p in arms.items() if p.exists()}
        for name in arms:
            if name not in present:
                print(f"  {name:<15} MISSING ({arms[name]})")
        if not present:
            print()
            continue

        # Mask on voxels finite across the manual map and every arm actually available, so
        # each sub-annotation is compared on its own common support.
        loaded = {"manual": nib.load(str(manual_path)).get_fdata()}
        for name, path in present.items():
            loaded[name] = nib.load(str(path)).get_fdata()
        shapes = {arr.shape for arr in loaded.values()}
        if len(shapes) != 1:
            print(f"  SKIP: mismatched map shapes {shapes}\n")
            continue
        mask = np.ones(next(iter(shapes)), dtype=bool)
        for arr in loaded.values():
            mask &= np.isfinite(arr)
        vecs = {name: arr[mask].ravel() for name, arr in loaded.items()}
        print(f"  voxels compared: {int(mask.sum())}")

        for name in ("autonima", "baseline_sub", "baseline_broad"):
            if name not in vecs:
                continue
            dice = compute_dice(vecs["manual"], vecs[name], args.dice_threshold)
            r = compute_pearson(vecs["manual"], vecs[name])
            r2 = r * r if r == r else float("nan")
            pool = pools.get(name) or {}
            print(f"  {name:<15} dice={dice:.3f}  r={r:.3f}  R2={r2:.3f}"
                  f"   studies={pool.get('studies')} points={pool.get('points')}")
            rows.append({
                "project": args.project, "manual_annotation": key, "arm": name,
                "autonima_run": auto_run, "auto_column": auto_col, "broad_arm": broad_label,
                "dice": round(dice, 4), "pearson_r": round(r, 4) if r == r else "",
                "r2": round(r2, 4) if r2 == r2 else "",
                "studies": pool.get("studies"), "points": pool.get("points"),
                "voxels": int(mask.sum()),
            })
        # Headline deltas: does the pipeline beat each baseline on this sub-annotation?
        have = {rw["arm"]: rw for rw in rows if rw["manual_annotation"] == key}
        if "autonima" in have:
            for base in ("baseline_sub", "baseline_broad"):
                if base in have and have[base]["r2"] != "" and have["autonima"]["r2"] != "":
                    d = have["autonima"]["r2"] - have[base]["r2"]
                    print(f"    autonima - {base:<15} dR2={d:+.3f}"
                          f"   {'BEATS' if d > 0 else 'LOSES'}")
        print()

    if not rows:
        print("nothing comparable found", file=sys.stderr)
        return 1

    out_dir = args.output_dir or (project_dir / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "baseline_vs_autonima.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv.relative_to(REPO_ROOT)}  ({len(rows)} rows)")

    # Summary across sub-annotations, on R^2 (threshold-free).
    by_arm: dict[str, list[float]] = {}
    for rw in rows:
        if rw["r2"] != "":
            by_arm.setdefault(rw["arm"], []).append(float(rw["r2"]))
    if by_arm:
        print("\nmean R^2 across sub-annotations:")
        for arm in ("autonima", "baseline_sub", "baseline_broad"):
            if arm in by_arm:
                vals = by_arm[arm]
                print(f"  {arm:<15} {sum(vals)/len(vals):.3f}  (n={len(vals)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""A size-matched null for analysis selection: is annotation better than picking at random?

WHY THIS EXISTS

Figure 5 compares the annotated map against `all_analyses` -- every parsed analysis from the same
studies. That isolates annotation from study selection, but it confounds two things: annotation
*chooses* analyses, and in choosing it also *shrinks* the set. A smaller CBMA is not simply a
worse one; changing N changes the density map's scale and sparsity. So a gain over `all_analyses`
could in principle be a gain from using fewer analyses, whatever they were.

This removes that confound. For each column it draws many random subsets of the same size as the
annotated set, from the same pool, runs the same MKDA + FDR, and scores each against the same
expert map. The result is a null distribution of "what a size-matched arbitrary selection would
have scored", against which the annotated map's position is the effect of selecting *well* rather
than of selecting *fewer*.

WHY IT RECOMPUTES RATHER THAN READS

The observed value is recomputed here through the identical code path, not read from
annotation_value.csv. That table derives from per-project similarity matrices that predate the
current maps -- the emotion regulation matrix was written 15 hours before the maps it describes --
and 26 of 35 columns disagree with a fresh computation, several by more than 0.2. Comparing a
stale observed value against a freshly computed null would manufacture or erase an effect.

The MKDA + FDR path used here reproduces the stored maps exactly (max|difference| 1.4e-07,
r = 1.000000 on emotion regulation `increase`), so the null and the observed value are on the
same footing as the pipeline's own outputs.

BOTH METRICS, EACH ON ITS OWN MAP

R-squared is computed on the raw z maps and dice on the FDR-corrected z maps, from the same fit.
The pairing is not interchangeable: dice on an uncorrected map thresholds at a level with no error
control behind it, and r-squared on a corrected map correlates an image whose sub-threshold
structure -- most of what the correlation was measuring -- has been zeroed. Reporting both from
one run also means the paper can settle the primary metric without another 35-column re-run.

Writes reports/annotation_bootstrap_null.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/zorro/repos/autonima")
from run_tiers import resolve_tier  # noqa: E402
from benchmark_exclusions import filter_rows  # noqa: E402

MANUAL_BASE = Path("/home/zorro/repos/neurometabench/analysis")
# THE METRIC/MAP RULE. R-squared compares *unthresholded* maps, so it reads the raw z. Dice
# compares *thresholded* maps, so it reads the FDR-corrected z at the corrected q <= 0.05
# boundary, which on these maps is exactly z > 1.96 (verified voxel-identical against the
# corrected p map). Pairing either metric with the other map is what this file exists not to do:
# dice on an uncorrected map thresholds at an arbitrary level with no error control, and
# r-squared on a corrected map correlates a mostly-zeroed image whose sub-threshold structure has
# been discarded.
UNCORRECTED_MAP = "z.nii.gz"
CORRECTED_MAP = "z_corr-FDR_method-indep.nii.gz"
DICE_THRESHOLD = 1.96
POOL_COLUMN = "all_analyses"


def auto_column(project: str, key: str) -> str:
    path = REPO_ROOT / "projects" / project / "nmb_mappings.json"
    if path.exists():
        try:
            mapping = json.loads(path.read_text()).get("annotation_mappings") or {}
        except ValueError:
            mapping = {}
        return mapping.get(key, key)
    return key


def gold_dir(project: str, key: str) -> Path | None:
    d = MANUAL_BASE / project / key
    if d.is_dir():
        return d
    for c in (MANUAL_BASE / project).glob("*"):
        if c.is_dir() and c.name.lower().replace("-", "_") == key.lower().replace("-", "_"):
            return c
    return None


def _fit(studyset, analysis_ids):
    """MKDA + FDR over one set of analyses.

    Returns BOTH maps from the single fit -- raw z for r-squared and FDR-corrected z for dice --
    because the two metrics require different maps and refitting to get the second would double
    the bootstrap's cost for nothing.
    """
    from nimare.meta.cbma import MKDADensity
    from nimare.correct import FDRCorrector

    sub = studyset.slice(analyses=list(analysis_ids))
    for s in sub.studies:
        s.name = s.id
        for a in s.analyses:
            a.name = a.id
    ds = sub.to_dataset()
    res = MKDADensity().fit(ds)
    corrected = FDRCorrector(method="indep").transform(res)
    return (np.asarray(res.get_map("z").dataobj),
            np.asarray(corrected.get_map("z_corr-FDR_method-indep").dataobj),
            len(ds.ids))


def _r2(a: np.ndarray, gold: np.ndarray) -> float:
    """Unthresholded agreement. Both arguments must be raw z maps."""
    m = np.isfinite(a) & np.isfinite(gold)
    if m.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[m].ravel(), gold[m].ravel())[0, 1] ** 2)


def _dice(a: np.ndarray, gold: np.ndarray) -> float:
    """Thresholded overlap. Both arguments must be FDR-corrected z maps."""
    ba, bb = a > DICE_THRESHOLD, gold > DICE_THRESHOLD
    total = ba.sum() + bb.sum()
    return 0.0 if total == 0 else float(2.0 * (ba & bb).sum() / total)


def run_column(task: dict) -> dict:
    """One column: recompute the observed value, then the size-matched null."""
    warnings.filterwarnings("ignore")
    import nibabel as nib
    from nimare.nimads import Studyset
    from autonima.meta import _analysis_ids_for_column, _analysis_has_valid_coordinates

    project, key, run, n_boot, seed = (
        task["project"], task["key"], task["run"], task["n_boot"], task["seed"])
    out = REPO_ROOT / "projects" / project / run / "outputs"
    row = {"project": project, "manual_column": key, "auto_column": auto_column(project, key),
           "run": run, "n_boot": 0, "status": ""}

    gdir = gold_dir(project, key)
    if gdir is None or not (gdir / UNCORRECTED_MAP).exists() \
            or not (gdir / CORRECTED_MAP).exists():
        row["status"] = "expert maps incomplete (need both raw and corrected z)"
        return row
    gold_raw = nib.load(str(gdir / UNCORRECTED_MAP)).get_fdata()
    gold_cor = nib.load(str(gdir / CORRECTED_MAP)).get_fdata()

    studyset = Studyset(json.loads((out / "nimads_studyset.json").read_text()))
    ann = json.loads((out / "nimads_annotation.json").read_text())
    if isinstance(ann, list):
        ann = ann[0] if ann else {}

    col = auto_column(project, key)
    if col not in ann.get("note_keys", {}) or POOL_COLUMN not in ann.get("note_keys", {}):
        row["status"] = f"column {col!r} or pool absent from annotation"
        return row

    def valid_ids(column):
        ids = _analysis_ids_for_column(ann, column)
        sub = studyset.slice(analyses=ids)
        return [a.id for s in sub.studies for a in s.analyses
                if _analysis_has_valid_coordinates(a)]

    selected, pool = valid_ids(col), valid_ids(POOL_COLUMN)
    # The annotated set should be a subset of the pool; where it is not (an analysis annotated
    # into a construct but absent from all_analyses) the union keeps the null's support honest.
    pool = sorted(set(pool) | set(selected))
    k, n_pool = len(selected), len(pool)
    row.update(k_analyses=k, pool_analyses=n_pool)
    if k == 0:
        row["status"] = "no valid analyses in column"
        return row
    if k >= n_pool:
        row["status"] = f"column is the whole pool ({k}/{n_pool}); null undefined"
        return row

    obs_raw, obs_cor, n_used = _fit(studyset, selected)
    if obs_raw.shape != gold_raw.shape:
        row["status"] = f"shape mismatch {obs_raw.shape} vs {gold_raw.shape}"
        return row
    observed = _r2(obs_raw, gold_raw)
    observed_dice = _dice(obs_cor, gold_cor)

    rng = np.random.default_rng(seed)
    null, null_d = [], []
    for _ in range(n_boot):
        draw = rng.choice(n_pool, size=k, replace=False)
        raw, cor, _n = _fit(studyset, [pool[i] for i in draw])
        null.append(_r2(raw, gold_raw))
        null_d.append(_dice(cor, gold_cor))
    keep = [i for i, v in enumerate(null) if np.isfinite(v)]
    null = np.asarray([null[i] for i in keep])
    null_d = np.asarray([null_d[i] for i in keep])

    row.update(
        n_boot=int(null.size), observed_r2=round(observed, 6),
        null_mean=round(float(null.mean()), 6), null_sd=round(float(null.std(ddof=1)), 6),
        null_p05=round(float(np.percentile(null, 5)), 6),
        null_p50=round(float(np.percentile(null, 50)), 6),
        null_p95=round(float(np.percentile(null, 95)), 6),
        null_max=round(float(null.max()), 6),
        delta_vs_null_mean=round(float(observed - null.mean()), 6),
        # One-sided: how often an arbitrary same-size selection matches or beats annotation.
        # (r + 1) / (B + 1) is the standard non-zero floor for a Monte Carlo p.
        n_null_ge_observed=int((null >= observed).sum()),
        p_value=round(float((int((null >= observed).sum()) + 1) / (null.size + 1)), 6),
        z_vs_null=round(float((observed - null.mean()) / null.std(ddof=1)), 4)
        if null.std(ddof=1) > 0 else "",
        # Dice on the corrected maps, same draws. Reported alongside rather than instead: r2 is
        # the primary metric per PAPER_OUTLINE section 8d, and dice is degenerate at small N.
        observed_dice=round(observed_dice, 6),
        dice_null_mean=round(float(null_d.mean()), 6),
        dice_null_sd=round(float(null_d.std(ddof=1)), 6),
        dice_null_p95=round(float(np.percentile(null_d, 95)), 6),
        dice_delta_vs_null_mean=round(float(observed_dice - null_d.mean()), 6),
        dice_p_value=round(float((int((null_d >= observed_dice).sum()) + 1)
                                 / (null_d.size + 1)), 6),
        status="ok",
    )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstraps", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tier", default="best")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--only-project", nargs="*", default=None)
    ap.add_argument("--source", type=Path,
                    default=REPO_ROOT / "reports" / "annotation_value.csv",
                    help="supplies the project/column list only; its numbers are not used")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" / "annotation_bootstrap_null.csv")
    args = ap.parse_args(argv)

    with open(args.source) as f:
        src = filter_rows(list(csv.DictReader(f)), label="bootstrap null")
    wanted = [(r["project"], r["manual_column"], r["run"]) for r in src]
    if args.only_project:
        wanted = [w for w in wanted if w[0] in args.only_project]

    tasks = []
    for i, (project, key, run) in enumerate(wanted):
        run = run or resolve_tier(project, "annotation_only", args.tier)
        if not run:
            continue
        tasks.append({"project": project, "key": key, "run": run,
                      "n_boot": args.bootstraps, "seed": args.seed + i * 7919})

    print(f"{len(tasks)} columns x {args.bootstraps} bootstraps on {args.jobs} workers")
    import multiprocessing as mp
    rows = []
    with mp.Pool(args.jobs) as pool:
        for row in pool.imap_unordered(run_column, tasks):
            rows.append(row)
            if row.get("status") == "ok":
                print(f"  {row['project']:22}{row['manual_column']:34} "
                      f"k={row['k_analyses']:>5}/{row['pool_analyses']:<5} "
                      f"obs={row['observed_r2']:.3f} null={row['null_mean']:.3f}"
                      f"+-{row['null_sd']:.3f}  p={row['p_value']:.4f}", flush=True)
            else:
                print(f"  {row['project']:22}{row['manual_column']:34} {row['status']}", flush=True)

    order = ["project", "manual_column", "auto_column", "run", "k_analyses", "pool_analyses",
             "observed_r2", "n_boot", "null_mean", "null_sd", "null_p05", "null_p50",
             "null_p95", "null_max", "delta_vs_null_mean", "n_null_ge_observed", "p_value",
             "z_vs_null",
             "observed_dice", "dice_null_mean", "dice_null_sd", "dice_null_p95",
             "dice_delta_vs_null_mean", "dice_p_value", "status"]
    rows.sort(key=lambda r: (r["project"], r["manual_column"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=order, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "ok"]
    beat = [r for r in ok if r["p_value"] < 0.05]
    print(f"\n{len(ok)}/{len(rows)} columns produced a null")
    print(f"  annotated map beats the size-matched null at p<0.05: {len(beat)}/{len(ok)}")
    # relative_to raises for an --output outside the repo, which is the normal case for a scratch
    # run; fall back to the absolute path rather than crashing after the work is done.
    try:
        shown = args.output.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.output
    print(f"  wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

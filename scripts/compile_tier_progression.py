#!/usr/bin/env python3
"""How much does performance rise as gold-standard information reaches the criteria?

WHY THIS EXISTS

`run_categories.yaml` separates runs by how much benchmark information shaped their criteria:

    verbatim   transcribed from the source paper before any results were seen -- held out by
               construction
    manual     the project author revised the criteria by hand, lightly, after seeing reports
    best       the version chosen on performance; in practice agent-written from the error
               reports in full

That ordering is a gradient of leakage, and the paper leans on it for the overfitting argument
without ever having measured it. This measures it: the same metric as Figure 4, computed for each
tier's run, so the rise from `verbatim` to `best` is an estimate of how much of the headline could
be attributable to criteria having seen the answers.

WHAT THE GAPS MEAN

Not every project has every tier, and the absences are informative rather than missing data. Only
four projects were ever hand-revised, so `manual` has a smaller n by construction. `vbm_of_ptsd`
registers the same run at all four tiers, so it contributes no progression and is reported as flat
rather than dropped. Two projects have no `verbatim` maps at all. The honest summary is therefore
paired within project, never a cross-tier mean over different project sets.

METRICS

Both, each on its own map, per the metric/map rule: r-squared on the raw z, dice on the
FDR-corrected z at z > 1.96. Excluded benchmark columns are dropped.

Writes reports/tier_progression.csv.
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
from run_tiers import resolve_tier, load_registry  # noqa: E402
from benchmark_exclusions import is_excluded  # noqa: E402
from map_mask import common_mask  # noqa: E402

MANUAL_BASE = Path("/home/zorro/repos/neurometabench/analysis")
UNCORRECTED_MAP = "z.nii.gz"
CORRECTED_MAP = "z_corr-FDR_method-indep.nii.gz"
DICE_THRESHOLD = 1.96
TIERS = ("verbatim", "manual", "best")


def auto_column(project: str, key: str) -> str:
    path = REPO_ROOT / "projects" / project / "nmb_mappings.json"
    if path.exists():
        try:
            return (json.loads(path.read_text()).get("annotation_mappings") or {}).get(key, key)
        except ValueError:
            pass
    return key


def mappings(project: str) -> dict[str, str]:
    path = REPO_ROOT / "projects" / project / "nmb_mappings.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("annotation_mappings") or {}
    except ValueError:
        return {}


def gold_dir(project: str, key: str) -> Path | None:
    d = MANUAL_BASE / project / key
    if d.is_dir():
        return d
    for c in (MANUAL_BASE / project).glob("*"):
        if c.is_dir() and c.name.lower().replace("-", "_") == key.lower().replace("-", "_"):
            return c
    return None


_cache: dict[Path, np.ndarray] = {}


def _load(path: Path) -> np.ndarray:
    if path not in _cache:
        import nibabel as nib
        _cache[path] = nib.load(str(path)).get_fdata()
    return _cache[path]


def score(auto_dir: Path, gold: Path) -> tuple[float, float] | None:
    need = [auto_dir / UNCORRECTED_MAP, auto_dir / CORRECTED_MAP,
            gold / UNCORRECTED_MAP, gold / CORRECTED_MAP]
    if not all(p.exists() for p in need):
        return None
    A, G = _load(need[0]), _load(need[2])
    if A.shape != G.shape:
        return None
    m = common_mask(A, G)
    r2 = float(np.corrcoef(A[m].ravel(), G[m].ravel())[0, 1] ** 2)
    Ac, Gc = _load(need[1]), _load(need[3])
    ba, bb = Ac > DICE_THRESHOLD, Gc > DICE_THRESHOLD
    total = ba.sum() + bb.sum()
    dice = 0.0 if total == 0 else float(2.0 * (ba & bb).sum() / total)
    return r2, dice


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", default="canonical")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" / "tier_progression.csv")
    args = ap.parse_args(argv)

    registry = load_registry()
    rows = []
    for project in sorted(p for p in registry if not p.startswith("_")):
        cols = mappings(project)
        if not cols:
            continue
        runs = {}
        for tier in TIERS:
            try:
                runs[tier] = resolve_tier(project, args.family, tier, registry)
            except Exception:
                runs[tier] = None
        for tier, run in runs.items():
            if not run:
                continue
            meta = REPO_ROOT / "projects" / project / run / "outputs" / "meta_analysis_results"
            if not meta.is_dir():
                continue
            for key in cols:
                if is_excluded(project, key):
                    continue
                gold = gold_dir(project, key)
                if gold is None:
                    continue
                got = score(meta / auto_column(project, key), gold)
                if got is None:
                    continue
                rows.append({"project": project, "family": args.family, "tier": tier,
                             "run": run, "manual_annotation": key,
                             "r2": round(got[0], 6), "dice": round(got[1], 6),
                             # Flat means this tier resolves to the same run as `best`, so it
                             # contributes no progression rather than a null result.
                             "same_run_as_best": "yes" if run == runs.get("best") else "no"})

    if not rows:
        print("no scoreable tier runs found")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        by.setdefault((r["project"], r["tier"]), []).append(r["r2"])
    print(f"{'project':26}{'verbatim':>11}{'manual':>10}{'best':>10}   {'V->B':>8}  n")
    deltas = []
    for project in sorted({r["project"] for r in rows}):
        cells = []
        for tier in TIERS:
            v = by.get((project, tier))
            cells.append(f"{st.mean(v):.3f}" if v else "--")
        vb = by.get((project, "verbatim")), by.get((project, "best"))
        same = {r["same_run_as_best"] for r in rows
                if r["project"] == project and r["tier"] == "verbatim"}
        if vb[0] and vb[1] and "yes" not in same:
            d = st.mean(vb[1]) - st.mean(vb[0])
            deltas.append(d)
            dtxt = f"{d:+.3f}"
        else:
            dtxt = "flat" if "yes" in same else "--"
        n = len(by.get((project, "best")) or [])
        print(f"{project:26}{cells[0]:>11}{cells[1]:>10}{cells[2]:>10}   {dtxt:>8}  {n}")
    if deltas:
        print(f"\n  verbatim -> best, paired within project: n = {len(deltas)}, "
              f"mean {st.mean(deltas):+.4f}, median {st.median(deltas):+.4f}, "
              f"rises in {sum(1 for d in deltas if d > 0)}/{len(deltas)}")
    print(f"  wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

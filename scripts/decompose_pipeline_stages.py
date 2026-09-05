#!/usr/bin/env python3
"""Attribute map quality to search, screening and annotation separately.

Every cross-project number so far confounds at least two stages: a project's headline run does
search AND screening AND annotation, so a margin over a baseline says the pipeline helped without
saying which part did the helping.

A project escapes that only if three of its family members share criteria byte-for-byte and differ
solely in how studies enter the pipeline:

    baseline            broad search, NO screening, NO annotation   -- the floor
    vN-annotation-only  fixed gold pool, no screening, annotation   -- annotation alone
    vN-allstudies       fixed broad pool, screening + annotation    -- adds screening
    vN                  real search,      screening + annotation    -- adds search

The differences between adjacent arms are informative, but they are NOT clean stage effects, and
labelling them "annotation", "screening", "search" overclaims. Each arm changes the pool at the
same time as the stage:

    annotation_only - baseline   bundles annotation WITH being handed the gold pool. It cannot be
                                 read as annotation's contribution: of course a run restricted to
                                 gold studies matches the gold. Treat it as an upper bound.
    allstudies - annotation_only the useful one. Both apply the same annotation, so this asks how
                                 close SCREENING A BROAD POOL gets to being handed a curated one.
                                 Near zero means screening substitutes for hand curation.
    full - allstudies            what a real search adds over a fixed, hand-assembled pool.

So read the table as three questions rather than a variance decomposition:

  1. full - baseline        what the whole pipeline is worth over a search-only meta-analysis
  2. allstudies vs annot    can screening replace a curated pool?
  3. full vs allstudies     does running your own search cost you anything?

These are properties of THIS corpus's arms, not causal stage effects.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def diag(project: Path, run: str) -> dict[str, float]:
    f = project / "reports" / f"manual_vs_auto_meta_{run}" / "tables" / "diagonal_metrics.csv"
    if not f.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        try:
            out[r["manual_name"]] = float(r["dice"])
        except (ValueError, TypeError, KeyError):
            continue
    return out


def baseline_dice(project: Path) -> dict[str, float]:
    """Best available baseline per column, from the baseline evaluation."""
    f = project / "reports" / "baseline_vs_autonima.csv"
    if not f.exists():
        return {}
    by: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(open(f)):
        try:
            by.setdefault(r["manual_annotation"], {})[r["arm"]] = float(r["dice"])
        except (ValueError, TypeError, KeyError):
            continue
    out = {}
    for col, arms in by.items():
        # targeted arm where one exists, else broad -- same rule as compile_best_baselines
        v = arms.get("baseline_sub", arms.get("baseline_broad"))
        if v is not None:
            out[col] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--version", required=True, help="e.g. v4 -- expects vN, vN-allstudies, vN-annotation-only")
    ap.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    ap.add_argument("--append-csv", type=Path, default=None,
                    help="append per-column rows to this CSV. Without it this analysis exists "
                         "only as console output, so nothing downstream can cite it.")
    args = ap.parse_args()

    p = args.projects_root / args.project
    v = args.version
    arms = {"baseline": baseline_dice(p), "annotation_only": diag(p, f"{v}-annotation-only"),
            "allstudies": diag(p, f"{v}-allstudies"), "full": diag(p, v)}
    missing = [k for k, d in arms.items() if not d]
    if missing:
        print(f"  cannot decompose {args.project}/{v}: no data for {missing}")
        return 1

    cols = sorted(set(arms["full"]) & set(arms["allstudies"]) & set(arms["annotation_only"]) & set(arms["baseline"]))
    if not cols:
        print(f"  no columns present in all four arms for {args.project}/{v}")
        return 1

    print(f"  === {args.project} / {v} : dice per column, by arm ===")
    print(f"  {'column':<26}{'baseline':>9}{'annot':>8}{'allstud':>9}{'full':>8}"
          f"{'  |':>3}{'annot-base':>11}{'all-annot':>10}{'full-all':>9}")
    inc = {"annotation": [], "screening": [], "search": []}
    for c in cols:
        b, a, s, f = arms["baseline"][c], arms["annotation_only"][c], arms["allstudies"][c], arms["full"][c]
        inc["annotation"].append(a - b); inc["screening"].append(s - a); inc["search"].append(f - s)
        print(f"  {c[:25]:<26}{b:>9.3f}{a:>8.3f}{s:>9.3f}{f:>8.3f}{'  |':>3}"
              f"{a-b:>+11.3f}{s-a:>+10.3f}{f-s:>+9.3f}")
    print(f"\n  {'mean':<26}" + "".join(f"{st.mean(arms[k][c] for c in cols):>{w}.3f}"
          for k, w in (("baseline", 9), ("annotation_only", 8), ("allstudies", 9), ("full", 8)))
          + f"{'  |':>3}" + f"{st.mean(inc['annotation']):>+11.3f}{st.mean(inc['screening']):>+10.3f}"
          f"{st.mean(inc['search']):>+9.3f}")
    print(f"\n  n columns: {len(cols)}")
    tot = st.mean(arms['full'][c] - arms['baseline'][c] for c in cols)
    scr, sea = st.mean(inc['screening']), st.mean(inc['search'])
    print(f"  1. whole pipeline vs search-only baseline : {tot:+.3f}")
    print(f"  2. screening a broad pool vs a curated one: {scr:+.3f}"
          f"   ({'screening MATCHES hand curation' if abs(scr) < 0.03 else ('screening BEATS it' if scr > 0 else 'curated pool still ahead')})")
    print(f"  3. own search vs fixed hand-assembled pool: {sea:+.3f}")

    if args.append_csv:
        args.append_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["project", "version", "column", "baseline", "annotation_only",
                  "allstudies", "full"]
        exists = args.append_csv.exists()
        with open(args.append_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if not exists:
                w.writeheader()
            for c in cols:
                w.writerow({
                    "project": args.project, "version": v, "column": c,
                    "baseline": round(arms["baseline"][c], 4),
                    "annotation_only": round(arms["annotation_only"][c], 4),
                    "allstudies": round(arms["allstudies"][c], 4),
                    "full": round(arms["full"][c], 4),
                })
        print(f"  appended {len(cols)} rows to {args.append_csv.name}")
    print(f"\n  NOT reported as an annotation share: annotation_only is restricted to gold studies,")
    print(f"  so its margin over the baseline bundles annotation with a perfect pool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

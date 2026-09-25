#!/usr/bin/env python3
"""Re-gate a finished Jev run at other thresholds. No API calls.

WHY THIS CAN EXIST

A chat model returns a verdict, so changing the operating point means re-running the project.
Jev returns a calibrated probability per criterion, and autonima persists the whole vector in
`criterion_probabilities`, so the gate is pure arithmetic over stored numbers. The screening
precision/recall trade-off becomes a curve instead of one point, for free.

WHAT IT FOUND ON emotion_regulation_2022

`maintain` selected ZERO analyses at every threshold down to 0.3 -- not a threshold problem.
Per-criterion means located it immediately: three of the criteria are not propositions.

    GLOBAL_I4  "Judge this analysis from its own name, description and table caption..."
    GLOBAL_I5  "Assign every label that applies. A regulation contrast carries TWO labels..."
    MAINTAIN_I2 "Expect to find this INSIDE regulation studies. Almost every study in this
                 benchmark is a reappraisal experiment..."

Those are instructions to a reader and commentary on the benchmark. A chat model reads them as
guidance; a proposition evaluator is asked "is this true of the analysis?", which is not a
question these can answer, so they score low and the conjunction dies. Dropping the three takes
`maintain` from 0 to 43 and the corpus total from 202 to 326 at tau=0.5.

Usage:
    python scripts/jev_threshold_sweep.py projects/emotion_regulation_2022/v4-jev
    python scripts/jev_threshold_sweep.py <run> --drop GLOBAL_I4 GLOBAL_I5 --per-criterion
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# autonima.backends.jev does not exist in the pinned v0.1.0. Run this under
# `pixi run -e dev`, whose editable ../autonima checkout provides it.
from autonima.backends.jev import apply_gate  # noqa: E402

SYSTEM_ANNOTATIONS = {"all_studies", "all_abstract", "all_analyses"}


def load(run: Path) -> list[dict]:
    d = json.load(open(run / "outputs" / "annotation_results.json"))
    rows = d if isinstance(d, list) else (d.get("decisions") or d.get("results") or [])
    return [r for r in rows
            if r.get("criterion_probabilities")
            and r.get("annotation_name") not in SYSTEM_ANNOTATIONS]


def mapping_from(row: dict, drop: frozenset[str]) -> dict[str, dict[str, str]]:
    """Criterion IDs are scope-prefixed (GLOBAL_I1, MAINTAIN_E1); classify on the tail.

    Getting this wrong yields an EMPTY mapping, which `apply_gate` now refuses rather than
    reporting a vacuous 100% selection at every threshold -- which is what it did here first.
    """
    out: dict[str, dict[str, str]] = {"inclusion": {}, "exclusion": {}}
    for cid in row["criterion_probabilities"]:
        if cid in drop:
            continue
        tail = cid.rsplit("_", 1)[-1]
        out["inclusion" if tail.startswith("I") else "exclusion"][cid] = cid
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a run directory containing outputs/")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--drop", nargs="*", default=[],
                    help="criterion IDs to exclude, e.g. instruction-shaped ones")
    ap.add_argument("--per-criterion", action="store_true",
                    help="mean/max probability per criterion, to find what blocks a target")
    args = ap.parse_args(argv)

    rows = load(args.run)
    if not rows:
        print("no decisions with stored probabilities -- was this run made with backend: jev?")
        return 1
    drop = frozenset(args.drop)
    targets = sorted({r["annotation_name"] for r in rows})
    print(f"{len(rows):,} decisions over {len(targets)} targets"
          + (f", dropping {sorted(drop)}" if drop else ""))

    print(f"\n  {'tau':<6}" + "".join(f"{t[:12]:>13}" for t in targets) + f"{'total':>8}")
    for tau in args.taus:
        counts = dict.fromkeys(targets, 0)
        for r in rows:
            answers = {k: {"type": "noul", "noul": v}
                       for k, v in r["criterion_probabilities"].items() if k not in drop}
            if not answers:
                continue
            if apply_gate(answers, mapping_from(r, drop), tau, tau).include:
                counts[r["annotation_name"]] += 1
        print(f"  {tau:<6.1f}" + "".join(f"{counts[t]:>13}" for t in targets)
              + f"{sum(counts.values()):>8}")

    if args.per_criterion:
        for target in targets:
            rs = [r for r in rows if r["annotation_name"] == target]
            agg = collections.defaultdict(list)
            for r in rs:
                for k, v in r["criterion_probabilities"].items():
                    agg[k].append(v)
            print(f"\n=== {target} (n={len(rs)}) — lowest-scoring criteria block the gate ===")
            print(f"  {'criterion':<24}{'mean':>7}{'max':>7}{'>=0.5':>8}")
            for k in sorted(agg, key=lambda k: st.mean(agg[k])):
                v = agg[k]
                print(f"  {k:<24}{st.mean(v):>7.3f}{max(v):>7.2f}"
                      f"{sum(x >= 0.5 for x in v):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

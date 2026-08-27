#!/usr/bin/env python3
"""Pool every sub-analysis into one autonima-vs-best-baseline table.

WHY THIS EXISTS

Per-project baseline tables are not directly comparable, because the baselines are not all the
same kind of thing. Most sub-analyses have a targeted arm -- a narrowed search a real competitor
would plausibly run -- but some have none, because the sub-analysis is not a separable search
topic. emotion_regulation_2022's decrease/increase/maintain are contrast DIRECTIONS within one
paradigm: there is no query that targets "down-regulation" as opposed to "up-regulation", so the
broad arm is not a fallback there, it is genuinely the best baseline anyone could build.

Averaging a project's targeted margin over columns that have no targeted arm silently mixes the
two. This script instead resolves the best baseline PER COLUMN and pools the columns, so every
row compares autonima against the strongest competitor that could exist for that particular
sub-analysis.

THE RULE: use the targeted arm wherever one was defined, and the broad arm only where targeting
is impossible.

  available  <- PRIMARY. The targeted arm when one exists, else the broad arm. This is the
             honest counterfactual: what would a competent practitioner aiming at this column
             actually have built? If they would have narrowed the search, that narrowed search is
             their baseline, whether or not it happens to score well.
  strongest  max(targeted, broad). Reported as a robustness check only. It is tempting because it
             looks conservative, but it is the wrong counterfactual -- it lets the baseline switch
             arms per column with hindsight, picking whichever scored better after the fact. No
             practitioner gets to do that.

In 8 of 35 columns the broad arm outscores the targeted one, i.e. narrowing the search made the
baseline WORSE. That is a result about when sub-targeting helps, not a reason to substitute the
broad arm. The two definitions differ by 0.004 anyway, so the conclusion does not turn on it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_columns(projects_root: Path) -> dict[tuple[str, str], dict[str, float]]:
    by: dict[tuple[str, str], dict[str, float]] = collections.defaultdict(dict)
    for f in sorted(glob.glob(str(projects_root / "*" / "reports" / "baseline_vs_autonima.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                by[(r["project"], r["manual_annotation"])][r["arm"]] = float(r["r2"])
            except (ValueError, TypeError, KeyError):
                continue
    return by


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "cross_project_best_baseline.csv")
    args = ap.parse_args()

    by = load_columns(args.projects_root)
    rows = []
    for (proj, col), arms in sorted(by.items()):
        auto = arms.get("autonima")
        sub, broad = arms.get("baseline_sub"), arms.get("baseline_broad")
        if auto is None:
            continue
        candidates = {k: v for k, v in (("targeted", sub), ("broad", broad)) if v is not None}
        if not candidates:
            continue
        avail_src = "targeted" if sub is not None else "broad"
        avail = candidates[avail_src]
        strong_src = max(candidates, key=lambda k: candidates[k])
        strong = candidates[strong_src]
        rows.append({
            "project": proj, "manual_annotation": col, "autonima_r2": round(auto, 4),
            "baseline_sub_r2": round(sub, 4) if sub is not None else "",
            "baseline_broad_r2": round(broad, 4) if broad is not None else "",
            "targeting_possible": "yes" if sub is not None else "no",
            "best_available_r2": round(avail, 4), "best_available_source": avail_src,
            "strongest_r2": round(strong, 4), "strongest_source": strong_src,
            "delta_vs_available": round(auto - avail, 4),
            "delta_vs_strongest": round(auto - strong, 4),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    a = [r["autonima_r2"] for r in rows]
    for lbl, key in (("best AVAILABLE (targeted if any)", "best_available_r2"),
                     ("STRONGEST (max of targeted, broad)", "strongest_r2")):
        b = [r[key] for r in rows]
        d = [x - y for x, y in zip(a, b)]
        print(f"  {lbl:<36} autonima {st.mean(a):.3f}  baseline {st.mean(b):.3f}  "
              f"delta {st.mean(d):+.3f}  ahead {sum(1 for x in d if x > 0)}/{len(d)}")
    no_target = [r for r in rows if r["targeting_possible"] == "no"]
    print(f"\n  columns: {len(rows)}   targetable: {len(rows) - len(no_target)}   "
          f"no targeting possible: {len(no_target)}")
    for r in no_target:
        print(f"    {r['project']}/{r['manual_annotation']}")
    print(f"\n  wrote {args.output.relative_to(REPO_ROOT) if str(args.output).startswith(str(REPO_ROOT)) else args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

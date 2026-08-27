#!/usr/bin/env python3
"""What does annotation buy, once you already have the right studies?

This isolates annotation more tightly than any cross-arm comparison can. Both sides come from the
SAME run: the same fixed gold pool, the same retrieved texts, the same parsed analyses. The only
difference is which analyses go into the map.

    all_analyses   every parsed analysis from those studies, pooled -- what you get if you have
                   perfect study selection and no analysis selection at all
    <column>       only the analyses annotation assigned to that construct

So `annotated - all_analyses` answers: given that you already found the right papers, how much
closer to the manual meta-analysis does analysis-level selection get you? Nothing else varies --
not the pool, not retrieval, not parsing. That makes it the cleanest available estimate of
annotation's contribution, and it avoids the trap in a cross-arm decomposition, where an
annotation-only arm's margin over a baseline bundles annotation with being handed a perfect pool.

Reported alongside is the annotation F1 on matched analyses: of the analyses that matched a gold
analysis, how often did annotation put them in the same construct. The two measure different
things and can disagree -- F1 scores the labelling decision, dice scores the map that results --
and where they diverge is informative rather than contradictory.

Annotation-only runs are used because they have no screening, so study selection cannot contribute
to the difference.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tiers import resolve_tier  # noqa: E402
from nmb_mapping import load_mappings  # noqa: E402


def matrix(project: Path, run: str, kind: str = "dice") -> dict[str, dict[str, float | None]]:
    f = project / "reports" / f"manual_vs_auto_meta_{run}" / "tables" / f"{kind}_matrix_{run}.csv"
    if not f.exists():
        return {}
    rows = list(csv.reader(open(f)))
    if not rows:
        return {}
    hdr = rows[0][1:]
    out = {}
    for r in rows[1:]:
        out[r[0]] = {h: (float(v) if v not in ("", "nan", None) else None) for h, v in zip(hdr, r[1:])}
    return out


def ann_f1(project: Path, run: str) -> dict[str, float]:
    f = project / run / "reports" / "annotation_review_reports" / "annotation_metrics_by_mode.json"
    if not f.exists():
        return {}
    m = json.load(open(f))["metrics_by_mode"]["accepted"]
    return {k: v["analysis_metrics"]["f1"] for k, v in m.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "annotation_value.csv")
    args = ap.parse_args()

    rows = []
    for pdir in sorted(p for p in args.projects_root.iterdir() if p.is_dir() and p.name != "template_"):
        run = resolve_tier(pdir.name, "annotation_only", "latest")
        if not run:
            continue
        M, P, F = matrix(pdir, run, "dice"), matrix(pdir, run, "pearson"), ann_f1(pdir, run)
        if not M or "all_analyses" not in M:
            continue
        # Manual column names and auto annotation names are not always the same string -- social
        # maps `others_merged` -> `perception_others`, for instance -- so the annotated row has to
        # be looked up through the project's mapping rather than assumed to match.
        mp = pdir / "nmb_mappings.json"
        auto_for = load_mappings(mp) if mp.exists() else {}
        for col in M["all_analyses"]:
            auto_row = auto_for.get(col, col)
            ann, alla = M.get(auto_row, {}).get(col), M["all_analyses"].get(col)
            if ann is None or alla is None:
                continue
            rows.append({"project": pdir.name, "run": run, "manual_column": col,
                         "dice_annotated": round(ann, 4), "dice_all_analyses": round(alla, 4),
                         "dice_gain": round(ann - alla, 4),
                         "pearson_annotated": round(P.get(col, {}).get(col), 4) if P.get(col, {}).get(col) is not None else "",
                         "pearson_all_analyses": round(P.get("all_analyses", {}).get(col), 4) if P.get("all_analyses", {}).get(col) is not None else "",
                         "annotation_f1": round(F[auto_row], 4) if auto_row in F else ""})

    if not rows:
        print("  no comparable annotation-only runs found")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    cur = None
    print(f"  {'project / column':<44}{'annotated':>10}{'all_anal':>10}{'gain':>8}{'ann F1':>9}")
    for r in rows:
        if r["project"] != cur:
            cur = r["project"]
            print(f"  -- {cur}  ({r['run']})")
        f1txt = f"{r['annotation_f1']:.3f}" if r["annotation_f1"] != "" else "--"
        print(f"     {r['manual_column'][:38]:<41}{r['dice_annotated']:>10.3f}"
              f"{r['dice_all_analyses']:>10.3f}{r['dice_gain']:>+8.3f}{f1txt:>9}")
    g = [r["dice_gain"] for r in rows]
    print(f"\n  {len(rows)} columns across {len({r['project'] for r in rows})} projects")
    print(f"  mean gain {st.mean(g):+.3f}   median {st.median(g):+.3f}   positive in {sum(1 for x in g if x > 0)}/{len(g)}")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

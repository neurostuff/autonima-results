#!/usr/bin/env python3
"""Which projects had LLM tuning applied, and what it changed -- the table plots should group on.

WHY THIS IS NOT A TIER

`untuned` and `manual` stay separate in run_categories.yaml, because they answer different
questions. But a plot quantifying "what did benchmark tuning buy" needs ONE pre-tuning run per
project, and needs to show how many projects were tuned at all -- four of nine canonical projects
never were, and silently averaging them in would dilute the effect toward zero while hiding that
the denominator is not nine.

COLLAPSING `manual` AND `untuned`

Only one family in the corpus has a hand-tuning step above `untuned`: social/annotation_only,
where `manual` (v5-annotation-only) is a report-informed author edit and `untuned` is
v3-annotation-only. Everywhere else the two coincide, or `manual` sits BELOW `untuned` (social's
canonical v2 vs v3) and so adds nothing.

That means the "untuned -> manual vs untuned -> LLM-tuned" decomposition is not plottable: its
manual arm has n = 1.

For the collapse this emits BOTH candidates rather than picking a winner. The choice is
metric-dependent, and social is precisely the project where the metrics disagree -- the registry
records v1 as best at map level (dice 0.570) while v4 has much better annotation F1. A hard-coded
winner here would be a silent methodological choice. The plotting code applies "best performing of
the two" using whatever metric that plot is about, and `pre_alt_run` is empty everywhere else, so
for 20 of 21 rows the rule is a no-op.

    scripts/compile_tuning_contrast.py [--output reports/tuning_contrast.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tiers import load_registry  # noqa: E402
from compile_tier_specs import availability, cfg, crit, delta, search_delta  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ("canonical", "annotation_only", "allstudies")
VERSION_ORDER_NOTE = "manual is only a step ABOVE untuned when it is neither untuned nor best"


def same_criteria(project: str, a: str, b: str) -> bool:
    """Do two runs carry the same screening AND annotation criteria?

    This, not version order, decides whether a `manual` entry is a genuine alternative to the
    untuned run. Version numbers do not track criteria here: social's canonical v2 and v3 are
    byte-identical in criteria and differ only in search, while its v5-annotation-only is
    chronologically OLDER than v4-annotation-only despite the higher number.
    """
    ca, cb = cfg(project, a), cfg(project, b)
    if ca is None or cb is None:
        return False
    for block in (lambda c: (c.get("screening") or {}).get("abstract"),
                  lambda c: (c.get("screening") or {}).get("fulltext"),
                  lambda c: c.get("annotation")):
        if crit(block(ca)) != crit(block(cb)):
            return False
    return True


def changed_blocks(project: str, pre: str, post: str) -> str:
    a, b = cfg(project, pre), cfg(project, post)
    if a is None or b is None:
        return "config missing"
    out = []
    if search_delta(a, b) != "unchanged":
        out.append("search")
    for key, label in (("abstract", "abstract"), ("fulltext", "fulltext")):
        if delta((a.get("screening") or {}).get(key),
                 (b.get("screening") or {}).get(key)) != "unchanged":
            out.append(label)
    if delta(a.get("annotation"), b.get("annotation")) != "unchanged":
        out.append("annotation")
    return "+".join(out) or "none"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=str(REPO_ROOT / "reports" / "tuning_contrast.csv"))
    args = ap.parse_args(argv)

    reg = load_registry()
    rows = []
    for project in sorted(reg):
        for family in FAMILIES:
            fam = (reg[project] or {}).get(family)
            if not isinstance(fam, dict):
                continue
            # The tuned candidate is `benchmark_tuned`, an explicit curated key as of 2026-11-04.
            # It is NOT `best`: a tuned version can exist and not be promoted -- three of the four
            # added 2026-11-04 made their project worse and were deliberately left out of `best`,
            # and keying on `best` would report them as "never tuned". It is also no longer
            # `latest`, which was only ever a proxy and a wrong one in four rows (cue_reactivity
            # v6, ER v4, dementia v5-allstudies, cue_reactivity v5-annotation-only are `latest`
            # without their headers declaring tuning).
            pre, post = fam.get("untuned"), fam.get("benchmark_tuned")
            promoted = fam.get("best")
            if not pre or not post:
                continue
            tuned = pre != post
            # `manual` was folded into `untuned` on 2026-11-04, so there is no longer a second
            # pre-tuning candidate to consider: it never resolved to different CRITERIA anywhere.
            alt = ""
            rows.append({
                "project": project,
                "family": family,
                "tuning_applied": "yes" if tuned else "no",
                "pre_run": pre,
                "pre_alt_run": alt,
                "pre_alt_tier": "",
                "tuned_run": post if tuned else "",
                "tuned_is_best": ("yes" if tuned and promoted == post
                                  else "no" if tuned else ""),
                "blocks_changed": changed_blocks(project, pre, post) if tuned else "",
                "pre_outputs": availability(project, pre),
                "tuned_outputs": availability(project, post) if tuned else "",
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print(f"wrote {out.relative_to(REPO_ROOT)}  ({len(rows)} rows)\n")
    for family in FAMILIES:
        sub = [r for r in rows if r["family"] == family]
        if not sub:
            continue
        t = [r for r in sub if r["tuning_applied"] == "yes"]
        print(f"  {family:<17} tuned {len(t)}/{len(sub)}   "
              f"({', '.join(r['project'] for r in t) or 'none'})")
    for family in FAMILIES:
        sub=[r for r in rows if r["family"]==family and r["tuning_applied"]=="yes"]
        kept=[r for r in sub if r["tuned_is_best"]=="yes"]
        if sub:
            print(f"  {family:<17} of those tuned, promoted to best: {len(kept)}/{len(sub)}")
    alt = [r for r in rows if r["pre_alt_run"]]
    print(f"\n  rows where collapsing manual+untuned is NOT a no-op: {len(alt)}")
    for r in alt:
        print(f"    {r['project']}/{r['family']}: untuned={r['pre_run']} vs manual={r['pre_alt_run']}")
    pending = [r for r in rows if "no outputs" in (r["pre_outputs"], r["tuned_outputs"])]
    if pending:
        print("\n  rows not yet computable (a run has no outputs):")
        for r in pending:
            print(f"    {r['project']}/{r['family']}: "
                  f"pre={r['pre_run']}({r['pre_outputs']}) tuned={r['tuned_run']}({r['tuned_outputs']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Of the pipeline's false positives, how many did the researchers never even look at?

THE QUESTION

Precision against the expert inclusion list is the paper's weakest headline number, and
Supplementary S3 shows a large part of it is a pool mismatch: our PubMed query returns a corpus
the published review never screened. S3 demonstrates that by swapping the pool. This asks the
same thing directly, at the level of individual false positives:

    considered and rejected   the study IS in the researchers' own candidate pool, so they saw it
                              and said no. Our including it is a genuine disagreement.
    never considered          the study is absent from that pool entirely. It could not have
                              appeared in their inclusion list whatever we decided, so counting it
                              as a false positive is a property of the corpus, not of screening.

WHAT THE FIXED POOL IS, AND WHERE IT IS A GOOD PROXY

The `vN-allstudies` arm runs from a hand-supplied `pmids_file` rather than a query. For dementia
(`Dementia_All_adjudicated.txt`) and social (`validation_ids/all_pmids.txt`) that file contains
100% of the gold studies, so it really is the researchers' adjudicated candidate set and the
"never considered" figure means what it says.

For emotion regulation the file is a PubMed search list holding only 56 of 88 gold studies, so it
is NOT the researchers' pool -- it is missing gold the canonical search found. Its numbers are
reported but should be read as a lower bound on "considered", and the project should not be
pooled with the other two without saying so.

Writes reports/false_positive_composition.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_gold_survival import canonical_run, load_gold, survivors  # noqa: E402
from compute_stage_precision_recall import annotated_pmids  # noqa: E402

STAGES = ["search", "abstract", "fulltext", "annotation"]
FIXED_SUFFIX = "-allstudies"


def pool_pmids(run: Path) -> set[str]:
    path = run / "outputs" / "search_results.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return set()
    return {str(s.get("pmid")).strip() for s in (data.get("studies") or [])
            if isinstance(s, dict) and s.get("pmid")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" / "false_positive_composition.csv")
    args = ap.parse_args(argv)

    gold_sets = load_gold()
    rows = []
    print(f"{'project':24}{'stage':<12}{'FPs':>6}{'considered':>14}{'never considered':>19}")
    print("-" * 78)
    for project, gold in sorted(gold_sets.items()):
        pdir = args.projects_root / project
        canon = canonical_run(pdir, "") if pdir.is_dir() else None
        fixed_run = canonical_run(pdir, FIXED_SUFFIX) if pdir.is_dir() else None
        if not (canon and fixed_run):
            continue
        fixed = pool_pmids(fixed_run)
        if not fixed:
            continue
        gold_in_pool = len(gold & fixed)

        alive = survivors(canon / "outputs")
        alive["annotation"] = annotated_pmids(canon / "outputs")
        carried: set[str] | None = None
        for stage in STAGES:
            here = alive.get(stage)
            if here is not None:
                carried = here if carried is None else (carried & here)
            if carried is None:
                continue
            fps = carried - gold
            if not fps:
                continue
            considered = len(fps & fixed)
            rows.append({
                "project": project, "canonical_run": canon.name, "fixed_run": fixed_run.name,
                "stage": stage, "n_false_positives": len(fps),
                "fp_considered_and_rejected": considered,
                "fp_never_considered": len(fps) - considered,
                "frac_never_considered": round((len(fps) - considered) / len(fps), 6),
                "fixed_pool_n": len(fixed),
                "gold_in_fixed_pool": gold_in_pool, "gold_total": len(gold),
                # A pool missing gold is not the researchers' candidate set, so its
                # "never considered" share understates what they actually saw.
                "pool_is_full_candidate_set": "yes" if gold_in_pool == len(gold) else "no",
            })
            print(f"{project:24}{stage:<12}{len(fps):>6}"
                  f"{considered:>9} ({100 * considered / len(fps):3.0f}%)"
                  f"{len(fps) - considered:>12} ({100 * (len(fps) - considered) / len(fps):3.0f}%)")
        print(f"{'':24}fixed pool n={len(fixed)}, holds {gold_in_pool}/{len(gold)} gold"
              f"{'' if gold_in_pool == len(gold) else '   <- not a full candidate set'}\n")

    if not rows:
        print("no project has both a canonical and a fixed-pool arm")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    trust = [r for r in rows if r["stage"] == "fulltext" and r["pool_is_full_candidate_set"] == "yes"]
    everything = [r for r in rows if r["stage"] == "fulltext"]
    for label, group in (("full candidate sets only", trust), ("all arms", everything)):
        if not group:
            continue
        fp = sum(r["n_false_positives"] for r in group)
        nv = sum(r["fp_never_considered"] for r in group)
        print(f"  at full-text screening, {label} (n={len(group)} projects): "
              f"{nv}/{fp} = {100 * nv / fp:.0f}% never considered")
    try:
        shown = args.output.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = args.output
    print(f"  wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

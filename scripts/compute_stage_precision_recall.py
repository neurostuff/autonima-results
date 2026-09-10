#!/usr/bin/env python3
"""Precision and recall against the gold standard at each pipeline stage, including annotation.

WHY THE ANNOTATION STAGE MATTERS HERE

Figure 2's precision curve stops at full-text screening, and it looks poor -- a mean of 0.32. A
large part of that is not a screening error at all. The screener is asked whether a paper *meets
the criteria*, and it often correctly says yes about a paper that then turns out to contribute no
usable coordinates: no table, no extractable analysis, nothing that could enter a meta-analysis.
The benchmark counts such a paper as not included, because the expert meta-analysis could not use
it either, so it scores as a false positive against a decision that was right on the merits.

Adding an annotation stage separates the two. A paper "survives annotation" here if it passed
full-text screening AND at least one of its analyses was assigned to a construct column. That is
the pipeline's own answer to "did this paper actually yield data for one of the target
contrasts?", which is much closer to what the benchmark's inclusion list means.

    search      the study appeared in the PubMed results
    abstract    survived abstract screening
    fulltext    survived full-text screening
    annotation  survived full-text screening AND has >= 1 analysis in a construct column

Retrieval is deliberately not a stage here: it is a supply event, not a decision about a paper's
eligibility, so it belongs on the recall figure rather than in a precision series.

Aggregate annotation columns (`all_studies`, `all_abstract`, `all_analyses`) are excluded when
deciding whether an analysis was "assigned" -- they mark everything by construction, so counting
them would make the annotation stage identical to the full-text stage.

Writes reports/stage_precision_recall.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_gold_survival import load_gold, canonical_run, survivors  # noqa: E402

STAGES = ["search", "abstract", "fulltext", "annotation"]
AGGREGATE_PREFIX = "all_"


def annotated_pmids(outputs: Path) -> set[str] | None:
    """PMIDs with at least one analysis assigned to a non-aggregate annotation column."""
    ann_path, ss_path = outputs / "nimads_annotation.json", outputs / "nimads_studyset.json"
    if not (ann_path.exists() and ss_path.exists()):
        return None
    try:
        ann = json.loads(ann_path.read_text())
        studyset = json.loads(ss_path.read_text())
    except ValueError:
        return None
    if isinstance(ann, list):
        ann = ann[0] if ann else {}

    constructs = [k for k, v in (ann.get("note_keys") or {}).items()
                  if v == "boolean" and not k.startswith(AGGREGATE_PREFIX)]
    if not constructs:
        return None

    analysis_to_pmid: dict[str, str] = {}
    for study in studyset.get("studies", []):
        pmid = str(study.get("pmid") or "").strip()
        if not pmid:
            continue
        for a in study.get("analyses", []):
            aid = a.get("id") if isinstance(a, dict) else a
            if aid:
                analysis_to_pmid[aid] = pmid

    out: set[str] = set()
    for note in ann.get("notes", []):
        if not isinstance(note, dict):
            continue
        values = note.get("note") or {}
        if any(values.get(c) for c in constructs):
            pmid = analysis_to_pmid.get(note.get("analysis"))
            if pmid:
                out.add(pmid)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" / "stage_precision_recall.csv")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args(argv)

    gold_sets = load_gold()
    rows = []
    print(f"{'project':24}{'stage':<12}{'included':>10}{'TP':>7}{'precision':>11}{'recall':>9}")
    print("-" * 74)
    for project, gold in sorted(gold_sets.items()):
        pdir = args.projects_root / project
        run = canonical_run(pdir, args.suffix) if pdir.is_dir() else None
        if run is None:
            print(f"{project:24}(no canonical run)")
            continue
        alive = survivors(run / "outputs")
        alive["annotation"] = annotated_pmids(run / "outputs")

        carried: set[str] | None = None
        for stage in STAGES:
            here = alive.get(stage)
            if here is not None:
                carried = here if carried is None else (carried & here)
            if carried is None:
                continue
            tp = len(gold & carried)
            n = len(carried)
            rows.append({
                "project": project, "run": run.name, "stage": stage,
                "n_included": n, "n_gold": len(gold), "tp": tp,
                "precision": round(tp / n, 6) if n else "",
                "recall": round(tp / len(gold), 6) if gold else "",
            })
            print(f"{project:24}{stage:<12}{n:>10}{tp:>7}"
                  f"{(tp / n if n else 0):>11.3f}{(tp / len(gold) if gold else 0):>9.3f}")
        print()

    if not rows:
        print("nothing computed")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import statistics as st
    for stage in STAGES:
        v = [float(r["precision"]) for r in rows if r["stage"] == stage and r["precision"] != ""]
        rc = [float(r["recall"]) for r in rows if r["stage"] == stage and r["recall"] != ""]
        if v:
            print(f"  mean {stage:<11} precision {st.mean(v):.3f}   recall {st.mean(rc):.3f}"
                  f"   (n={len(v)})")
    # relative_to raises when --output is relative or outside the repo, which is the normal case
    # for a scratch run; fall back to the path as given rather than crashing after the work.
    try:
        shown = args.output.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = args.output
    print(f"  wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

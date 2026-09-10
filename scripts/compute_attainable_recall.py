#!/usr/bin/env python3
"""Recall at each stage against an ATTAINABLE denominator, not the whole gold list.

WHAT THIS FIXES

`compute_stage_precision_recall.py` divides every stage by the full gold list, so a stage is
charged for gold studies it never had a chance to keep. Three different failures get billed to
screening judgement:

    never returned by the search      the query missed it; no screener ever saw it
    full text not obtainable          a supply event, not a decision
    full text obtained but INCOMPLETE the screener got title/abstract only and could not
                                      verify the criteria; a late-detected retrieval failure
    no coordinates parsed             the paper is in, but there is nothing to annotate

Those are availability, not judgement. The denominator here removes each one at the stage where
it happens and never removes anything else, so what is left is the question a reader actually
wants answered: of the gold studies this stage COULD have kept, how many did it keep?

    stage        denominator                                    removes
    search       gold the search returned                       gold the query never returned
    abstract     same as search (no new supply loss)            --
    fulltext     ... minus gold with no usable full text        retrieval failures AND
                                                                `fulltext_incomplete`
    annotation   ... minus gold that parsed to zero analyses    nothing to annotate

WHY `fulltext_incomplete` COUNTS AS SUPPLY

The retrieval stage flags `fulltext_available`, and that flag OVERSTATES what the screener
received: 43 gold studies across the corpus were marked available and then screened with only
title, abstract and metadata (or after an API error), so the screener recorded
`fulltext_incomplete` and could not verify the inclusion criteria. Those are retrieval failures
detected one stage late, not judgements, and charging them to full-text screening is exactly the
mistake this script exists to avoid. It is not a rounding detail: 43 of the 68 gold studies lost
at full-text screening are of this kind, and in executive_function 24 of 25 are -- that project's
full-text recall reads 0.626 with them charged and 0.807 with them removed.

Genuine `excluded_fulltext` decisions stay charged. Corpus-wide there are 25 of those.

Search is therefore 1.000 by construction: it is a pure supply stage. Abstract shares its
denominator because nothing becomes unavailable in between.

WHAT IT DELIBERATELY STILL PENALISES

Only supply losses leave the denominator. A gold study rejected at abstract or full-text
screening, or parsed and then not assigned to any construct column, stays in it -- those are
judgements, and forgiving them is the whole failure mode this metric exists to avoid. In
particular the annotation denominator is NOT "gold studies with >= 1 parsed analysis": that set
excludes everything full-text screening threw away, which would silently forgive every
full-text rejection and push annotation recall towards 1.

NOT THE SAME AS "ADJUSTED GOLD"

projects/dementia/REPORT.md uses "adjusted gold" for the opposite operation -- it ADDS the
studies excluded only for `Data Not Reported` (74 -> 162) to make *precision* interpretable, and
that report is explicit that using it as a recall denominator asks a harder question rather than
a fairer one. It is also defined for one project. This script is unrelated to it.

Writes reports/attainable_recall_by_stage.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_gold_survival import load_gold, canonical_run, survivors  # noqa: E402
from compute_stage_precision_recall import annotated_pmids  # noqa: E402

STAGES = ["search", "abstract", "fulltext", "annotation"]


def parsed_pmids(outputs: Path) -> set[str] | None:
    """PMIDs with at least one parsed analysis in the studyset.

    This is "we had something to annotate", independent of whether annotation then assigned it
    to a construct. A study present in the studyset with an empty analyses list counts as
    nothing parsed, which is exactly the supply failure the annotation denominator drops.
    """
    path = outputs / "nimads_studyset.json"
    if not path.exists():
        return None
    try:
        studyset = json.loads(path.read_text())
    except ValueError:
        return None
    out: set[str] = set()
    for study in studyset.get("studies", []):
        pmid = str(study.get("pmid") or "").strip()
        if pmid and study.get("analyses"):
            out.add(pmid)
    return out or None


def incomplete_fulltext_pmids(outputs: Path) -> set[str] | None:
    """PMIDs the full-text screener saw without usable full text.

    `fulltext_incomplete` means the retriever reported the text as available but delivered
    title/abstract/metadata only, or errored, so the screener could not check the inclusion
    criteria and recorded that rather than a judgement. Treating these as rejections blames
    screening for a retrieval failure noticed one stage late.
    """
    path = outputs / "fulltext_screening_results.json"
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text()).get("screening_results")
    except ValueError:
        return None
    if not rows:
        return None
    return {str(r.get("study_id")).strip() for r in rows
            if isinstance(r, dict) and str(r.get("decision")) == "fulltext_incomplete"}


def stage_rows(project: str, run: Path, gold: set[str]) -> list[dict]:
    """Attainable-denominator recall for one project, or [] if an artifact is missing."""
    outputs = run / "outputs"
    alive = survivors(outputs)
    search, abstract = alive.get("search"), alive.get("abstract")
    retrieval, fulltext = alive.get("retrieval"), alive.get("fulltext")
    parsed, annotated = parsed_pmids(outputs), annotated_pmids(outputs)
    incomplete = incomplete_fulltext_pmids(outputs)
    if any(s is None for s in (search, abstract, retrieval, fulltext, parsed, annotated,
                               incomplete)):
        return []

    # Cumulative gold survivors, one stage at a time.
    g_search = gold & search
    g_abs = g_search & abstract
    g_retr = g_abs & retrieval
    g_full = g_retr & fulltext
    g_parsed = g_full & parsed
    g_annot = g_full & annotated

    lost_retrieval = len(g_abs) - len(g_retr)          # supply: no full text at all
    lost_incomplete = len(g_retr & incomplete)         # supply: text too thin to screen
    lost_parsing = len(g_full) - len(g_parsed)         # supply: nothing to annotate

    d_search = d_abs = len(g_search)
    d_full = d_abs - lost_retrieval - lost_incomplete
    d_annot = d_full - lost_parsing

    per_stage = [
        ("search", len(g_search), d_search, 0),
        ("abstract", len(g_abs), d_abs, 0),
        ("fulltext", len(g_full), d_full, lost_retrieval + lost_incomplete),
        ("annotation", len(g_annot), d_annot, lost_parsing),
    ]
    rows = []
    for stage, surviving, attainable, supply_lost in per_stage:
        rows.append({
            "project": project, "run": run.name, "stage": stage,
            "gold_total": len(gold),
            "n_surviving": surviving,
            "n_attainable": attainable,
            "supply_lost_here": supply_lost,
            "recall_attainable": round(surviving / attainable, 6) if attainable else "",
            "recall_raw": round(surviving / len(gold), 6) if gold else "",
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" / "attainable_recall_by_stage.csv")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args(argv)

    rows: list[dict] = []
    print(f"{'project':<24}{'stage':<12}{'surv':>6}{'attain':>8}{'adj':>8}{'raw':>8}")
    print("-" * 66)
    for project, gold in sorted(load_gold().items()):
        pdir = args.projects_root / project
        run = canonical_run(pdir, args.suffix) if pdir.is_dir() else None
        if run is None:
            print(f"{project:<24}(no canonical run)")
            continue
        got = stage_rows(project, run, gold)
        if not got:
            print(f"{project:<24}(missing an artifact; skipped)")
            continue
        rows.extend(got)
        for r in got:
            print(f"{project:<24}{r['stage']:<12}{r['n_surviving']:>6}{r['n_attainable']:>8}"
                  f"{float(r['recall_attainable']):>8.3f}{float(r['recall_raw']):>8.3f}")
        print()

    if not rows:
        print("nothing computed")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for stage in STAGES:
        adj = [float(r["recall_attainable"]) for r in rows
               if r["stage"] == stage and r["recall_attainable"] != ""]
        raw = [float(r["recall_raw"]) for r in rows
               if r["stage"] == stage and r["recall_raw"] != ""]
        if adj:
            print(f"  mean {stage:<11} attainable {st.mean(adj):.3f}   raw {st.mean(raw):.3f}"
                  f"   (n={len(adj)})")
    # relative_to raises for a relative or out-of-repo --output, the normal case for a scratch
    # run; fall back to the path as given rather than crashing after the work.
    try:
        shown = args.output.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = args.output
    print(f"  wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

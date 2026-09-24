#!/usr/bin/env python3
"""Merge a project's per-run missing-full-text lists into one prioritised download list.

Each run writes outputs/missing_fulltexts.csv covering only its own pool, so a project with
several family members (a search-driven run, a fixed-pool run, an annotation-only run) has the
same PMID appearing in some lists and not others, with different reasons. Downloading from any
one list under-covers; downloading from all three duplicates work.

This merges them, dedupes by PMID, and sorts by how much a download would buy:

  1 gold      the PMID is in the project's benchmark gold set. Fetching it can raise the recall
              ceiling directly, so these come first regardless of which run reported them.
  2 included  passed screening in at least one run, so it would contribute analyses, but is not
              itself a gold study. Improves the map without moving recall.
  3 excluded  reported missing only from an excluded/candidate arm. Lowest value.

`type` is preserved and matters for the fix: "missing" means no file at all, while "incomplete"
means a file exists but is partial (e.g. an HTML holding only metadata). An incomplete entry will
not be repaired by the same download step that fixes a missing one -- the stale file has to be
replaced, and move_downloaded_pmid_html.py needs --conflict overwrite for those.

Writes a .txt of PMIDs (one per line, what move_downloaded_pmid_html.py consumes) and a .csv
carrying the provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run directory names to merge")
    ap.add_argument("--gold-file", type=Path, default=None,
                    help="text file of gold PMIDs; defaults to <project>/annotation-only-ids.txt")
    ap.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    ap.add_argument("--out-prefix", default=None, help="default: <project>/reports/missing_fulltexts_joint")
    args = ap.parse_args()

    pdir = args.projects_root / args.project
    gold_path = args.gold_file or (pdir / "annotation-only-ids.txt")
    gold = {l.strip() for l in gold_path.read_text().splitlines() if l.strip().isdigit()} if gold_path.exists() else set()

    merged: dict[str, dict] = {}
    per_run: dict[str, int] = {}
    for run in args.runs:
        f = pdir / run / "outputs" / "missing_fulltexts.csv"
        if not f.exists():
            print(f"  WARNING: no missing_fulltexts.csv for {run}")
            continue
        rows = list(csv.DictReader(open(f)))
        per_run[run] = len(rows)
        for r in rows:
            pmid = (r.get("pmid") or "").strip()
            if not pmid.isdigit():
                continue
            e = merged.setdefault(pmid, {"pmid": pmid, "type": r.get("type", ""),
                                         "runs": [], "in_included_set": False,
                                         "existing_path": r.get("full_text_path", "")})
            e["runs"].append(run)
            if str(r.get("in_included_set", "")).strip().lower() == "true":
                e["in_included_set"] = True
            # "incomplete" is the more actionable label: a stale file needs replacing
            if r.get("type") == "incomplete":
                e["type"] = "incomplete"
                e["existing_path"] = r.get("full_text_path", "") or e["existing_path"]

    for e in merged.values():
        e["is_gold"] = e["pmid"] in gold
        e["priority"] = 1 if e["is_gold"] else (2 if e["in_included_set"] else 3)
        e["reported_by"] = ";".join(sorted(set(e.pop("runs"))))

    rows = sorted(merged.values(), key=lambda e: (e["priority"], e["type"] != "incomplete", e["pmid"]))
    prefix = Path(args.out_prefix) if args.out_prefix else (pdir / "reports" / "missing_fulltexts_joint")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "priority", "is_gold", "in_included_set", "type",
                                          "reported_by", "existing_path"])
        w.writeheader()
        w.writerows(rows)
    Path(f"{prefix}.txt").write_text("\n".join(e["pmid"] for e in rows) + "\n")
    # Tier files, because move_downloaded_pmid_html.py takes one list at a time and the tiers
    # have very different value per download.
    for tier, name in ((1, "gold"), (2, "included")):
        sel = [e["pmid"] for e in rows if e["priority"] == tier]
        Path(f"{prefix}.{name}.txt").write_text("\n".join(sel) + ("\n" if sel else ""))
    incomplete = [e["pmid"] for e in rows if e["type"] == "incomplete"]
    Path(f"{prefix}.incomplete.txt").write_text("\n".join(incomplete) + ("\n" if incomplete else ""))

    counts = defaultdict(lambda: defaultdict(int))
    for e in rows:
        counts[e["priority"]][e["type"] or "missing"] += 1
    print(f"  project: {args.project}   gold set: {len(gold)}")
    print(f"  per-run rows: " + ", ".join(f"{k}={v}" for k, v in per_run.items()))
    print(f"  merged unique PMIDs: {len(rows)}\n")
    labels = {1: "1 gold (raises the recall ceiling)", 2: "2 included (improves maps)", 3: "3 excluded arm"}
    for p in sorted(counts):
        tot = sum(counts[p].values())
        detail = ", ".join(f"{t}={n}" for t, n in sorted(counts[p].items()))
        print(f"    {labels[p]:<38} {tot:>5}   ({detail})")
    print(f"\n  wrote {prefix}.txt (all, priority-sorted) and {prefix}.csv")
    print(f"  tier files: {prefix.name}.gold.txt, .included.txt, .incomplete.txt")
    print(f"  NOTE .incomplete.txt entries already have a stale file on disk -- those need")
    print(f"       move_downloaded_pmid_html.py --conflict overwrite, not the default skip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

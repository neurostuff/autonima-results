#!/usr/bin/env python3
"""PMIDs whose gold analyses were annotated as living in supplemental material.

WHY THIS EXISTS

`supplemental_data` is the largest single disposition in the parser-failure review -- larger than
`parser_missed` -- so "the coordinates were not in the article we fetched" is the biggest known
cause of unmatched gold analyses. That list is only usable if it is extracted from the raw review
exports rather than re-read by hand each time, and the exports need reconciling first.

RECONCILING THE EXPORTS

`reviews/` holds one JSON per reviewer pass, and the passes OVERLAP rather than stack: review 1
labels 17 units and review 2 labels 68, but the union is 69 unique units, so ~16 were annotated
twice. Later passes are preferred per `unit_id` by `updated_at`, which matters because a unit can
in principle be relabelled on a second look.

CAVEAT ON THE RUN DIRECTORIES

The entries carry the `run_dir` they were reviewed against, and some are runs that predate the
tier registry (`v5-annotation-only-gpt`). The PMIDs are still the right PMIDs, but do not assume
the affected analyses map onto the runs the paper currently reports without checking.

Writes reports/supplemental_only_pmids.csv (one row per PMID) and prints a bare PMID list.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEWS = REPO_ROOT / "reports" / "parser_error_annotation" / "reviews"
LABEL = "supplemental_data"
FIELD = "unmatched_gold_disposition"


def load_units(review_dir: Path) -> dict[str, dict]:
    """Latest annotation per review unit, across every reviewer export."""
    latest: dict[str, dict] = {}
    for path in sorted(review_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            continue
        entries = payload.get("entries")
        if not isinstance(entries, list):
            continue          # README.json and other non-export files
        for entry in entries:
            uid = entry.get("unit_id")
            if not uid:
                continue
            prev = latest.get(uid)
            if prev is None or str(entry.get("updated_at", "")) >= str(prev.get("updated_at", "")):
                latest[uid] = {**entry, "_source": path.name,
                               "_reviewer": payload.get("reviewer", "")}
    return latest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" / "supplemental_only_pmids.csv")
    args = ap.parse_args(argv)

    units = load_units(args.reviews)
    hits = [u for u in units.values() if u.get(FIELD) == LABEL]

    by_pmid: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for u in hits:
        by_pmid[(str(u.get("pmid", "")), u.get("project", ""))].append(u)

    rows = []
    for (pmid, project), us in sorted(by_pmid.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows.append({
            "pmid": pmid,
            "project": project,
            "n_analyses_flagged": len(us),
            "reviewer": ";".join(sorted({u["_reviewer"] for u in us if u["_reviewer"]})),
            "review_files": ";".join(sorted({u["_source"] for u in us})),
            "run_dirs": ";".join(sorted({Path(u["run_dir"]).name for u in us if u.get("run_dir")})),
            "unit_ids": ";".join(sorted(u["unit_id"] for u in us)),
            "notes": " | ".join(sorted({u["note"] for u in us if u.get("note")})),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    total_units = len(units)
    print(f"{total_units} unique review units across {args.reviews.name}/ "
          f"({len(list(args.reviews.glob('*.json')))} files)")
    print(f"{len(hits)} annotated {LABEL!r} -> {len(rows)} unique PMIDs\n")
    print(f"{'project':26}{'PMIDs':>6}{'analyses':>10}")
    for project in sorted({r["project"] for r in rows}):
        sub = [r for r in rows if r["project"] == project]
        print(f"  {project:24}{len(sub):>6}{sum(r['n_analyses_flagged'] for r in sub):>10}")
    print(f"\n  {'TOTAL':24}{len(rows):>6}"
          f"{sum(r['n_analyses_flagged'] for r in rows):>10}")
    print("\nPMIDs:")
    print(" ".join(r["pmid"] for r in rows))
    print(f"\nwrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

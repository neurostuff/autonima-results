#!/usr/bin/env python3
"""Keep a tracked, text-free copy of the pubget table coordinates behind the table-only baseline.

WHY THIS EXISTS

Figure 3a and the Results compare LLM analysis parsing against a table-only baseline (each
coordinate table treated as one analysis). scripts/compare_analyses_to_benchmark.py builds that
baseline from whichever source has more coordinates: the run's own coordinate parses regrouped by
table (tracked), or pubget's raw table coordinates in retrieval/pubget_data (untracked, because
retrieval/ also holds the article full texts). For two reported runs -- decision_making
v2-annotation-only and emotion_regulation_2022 v4-annotation-only -- pubget wins, so the
manuscript's baseline mean could not be rebuilt from a checkout without the articles.

This script copies, for every reported run whose baseline came from pubget, only the fields the
baseline reads into <run>/outputs/table_only_baseline/:

    metadata.csv     pmcid, pmid
    tables.csv       pmcid, table_id, table_label          (no captions or footnotes)
    coordinates.csv  pmcid, table_id, table_label, x, y, z

compare_analyses_to_benchmark.py falls back to this folder when retrieval/pubget_data is absent.
Captions and footnotes only label the baseline's units; dropping them leaves every match
unchanged (checked on both runs: 10/25 and 7/42 matched, identical to the full pubget folder).

    tools/snapshot_table_baseline.py        # reads reports/cross_project_analysis/parsing_metrics_by_project.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = REPO_ROOT / "reports" / "cross_project_analysis" / "parsing_metrics_by_project.csv"
FIELDS = {
    "metadata.csv": ["pmcid", "pmid"],
    "tables.csv": ["pmcid", "table_id", "table_label"],
    "coordinates.csv": ["pmcid", "table_id", "table_label", "x", "y", "z"],
}


def repo_path(recorded: str) -> Path:
    """Paths in the metrics table are absolute in whichever checkout wrote them."""
    return REPO_ROOT / "projects" / recorded.split("/projects/", 1)[1]


def main() -> int:
    rows = list(csv.DictReader(METRICS.open(newline="", encoding="utf-8")))
    written = 0
    for row in rows:
        source = row.get("table_only_baseline_source", "")
        if not source.rstrip("/").endswith("retrieval/pubget_data"):
            continue
        src, run = repo_path(source), repo_path(row["selected_run_dir"])
        dst = run / "outputs" / "table_only_baseline"
        dst.mkdir(parents=True, exist_ok=True)
        for name, fields in FIELDS.items():
            with (src / name).open(newline="", encoding="utf-8") as fin, \
                    (dst / name).open("w", newline="", encoding="utf-8") as fout:
                w = csv.DictWriter(fout, fieldnames=fields)
                w.writeheader()
                w.writerows({k: r.get(k, "") for k in fields} for r in csv.DictReader(fin))
        print(f"{row['project_name']}: {src.relative_to(REPO_ROOT)} -> {dst.relative_to(REPO_ROOT)}")
        written += 1
    print(f"{written} run(s) snapshotted")
    return 0


if __name__ == "__main__":
    sys.exit(main())

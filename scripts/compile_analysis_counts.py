#!/usr/bin/env python3
"""How many analyses went into each map, per benchmark column.

WHY THIS EXISTS

Every similarity in Figures 4 and 5 is between two brain maps, and a reader cannot tell from a
scatter whether a column's score rests on eight analyses or eight hundred. The counts are already
recorded -- NiMARE writes them into each meta-analysis's `boilerplate.txt` as "included N foci
from M experiments" -- but nothing collected them, so the figures had no N to show.

"Experiments" is NiMARE's term for what this project calls an analysis: one contrast from one
study, which is the unit the whole paper argues about. So the count is the number of analyses the
arm actually pooled, not the number of studies.

Three arms are emitted per column so a figure can size by whichever is the honest denominator:

    autonima      analyses annotation assigned to that construct
    all_analyses  every parsed analysis from the same studies (annotation off)
    all_studies   the project's fixed-pool arm, one map shared by every column

Columns are keyed by the manual annotation name used in cross_project_best_baseline.csv and
translated through the project's nmb_mappings.json, so the output joins onto that table directly.

Writes reports/analysis_counts.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tiers import resolve_tier  # noqa: E402
from benchmark_exclusions import filter_rows  # noqa: E402

# NiMARE's phrasing; the numbers are foci then experiments.
COUNTS = re.compile(r"included\s+([\d,]+)\s+foci\s+from\s+([\d,]+)\s+experiments")
SHARED = ("all_analyses", "all_studies", "all_abstract")


def auto_column(project: str, key: str) -> str:
    """Translate a manual annotation key into the pipeline's column name."""
    path = REPO_ROOT / "projects" / project / "nmb_mappings.json"
    if path.exists():
        try:
            mapping = json.loads(path.read_text()).get("annotation_mappings") or {}
        except ValueError:
            mapping = {}
        return mapping.get(key, key)
    return key


def counts_for(map_dir: Path) -> tuple[int, int] | None:
    """(analyses, foci) for one meta-analysis directory, or None if it did not run.

    A column with too few analyses to meta-analyse leaves the directory without a boilerplate --
    vbm_of_ptsd/increased_gm is the real case -- so absence is expected, not an error.
    """
    bp = map_dir / "boilerplate.txt"
    if not bp.exists():
        return None
    m = COUNTS.search(bp.read_text())
    if not m:
        return None
    foci, analyses = (int(g.replace(",", "")) for g in m.groups())
    return analyses, foci


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-table", type=Path,
                    default=REPO_ROOT / "reports" / "cross_project_best_baseline.csv")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" / "analysis_counts.csv")
    ap.add_argument("--tier", default="best")
    args = ap.parse_args(argv)

    with open(args.baseline_table) as f:
        table = filter_rows(list(csv.DictReader(f)), label="analysis counts")
    wanted = [(r["project"], r["manual_annotation"]) for r in table]

    rows, missing = [], []
    runs: dict[str, str] = {}
    for project, key in wanted:
        if project not in runs:
            runs[project] = resolve_tier(project, "canonical", args.tier) or ""
        run = runs[project]
        if not run:
            missing.append(f"{project}/{key}: no canonical run at tier {args.tier}")
            continue
        mdir = REPO_ROOT / "projects" / project / run / "outputs" / "meta_analysis_results"
        col = auto_column(project, key)

        row = {"project": project, "manual_annotation": key, "auto_column": col, "run": run}
        for arm, name in (("autonima", col), *((a, a) for a in SHARED)):
            got = counts_for(mdir / name)
            row[f"n_analyses_{arm}"] = "" if got is None else got[0]
            row[f"n_foci_{arm}"] = "" if got is None else got[1]

        # The per-column baseline search, which is what "targeted" means in the baseline table.
        # It is a separate run tree, not a column of the canonical run.
        got = counts_for(REPO_ROOT / "projects" / project / "baselines" / key / "outputs"
                         / "meta_analysis_results" / "all_analyses")
        row["n_analyses_targeted"] = "" if got is None else got[0]
        row["n_foci_targeted"] = "" if got is None else got[1]
        if row["n_analyses_autonima"] == "":
            missing.append(f"{project}/{key}: no boilerplate for column {col!r}")
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    have = [r for r in rows if r["n_analyses_autonima"] != ""]
    ns = sorted(int(r["n_analyses_autonima"]) for r in have)
    print(f"{len(have)}/{len(rows)} columns have an analysis count")
    if ns:
        print(f"  analyses per column: min {ns[0]}, median {ns[len(ns)//2]}, max {ns[-1]}")
    for m in missing:
        print(f"  missing: {m}")
    print(f"  wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

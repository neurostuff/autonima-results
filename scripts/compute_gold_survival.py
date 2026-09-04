#!/usr/bin/env python3
"""Track what fraction of each project's gold standard survives each pipeline stage.

WHY THIS EXISTS

`screening_metrics_top_v_stage_progression.csv` looks like it answers this and does not. It holds
*conditional retention* -- the share of studies entering a stage that survive it -- so its
denominator changes at every stage and its "recall" column rises across the funnel, which is
impossible for cumulative recall. Chaining those retentions does not recover the truth either,
because retrieval loss is absent from that file entirely: executive_function's conditional product
is 0.645 against an actual end-to-end 0.392.

This counts gold PMIDs directly against a fixed denominator -- every gold study for the project --
at each stage in turn, so the curve is monotonically non-increasing by construction and every drop
is attributable to a named stage.

Retrieval is reported as its own stage rather than folded into screening. That distinction is the
point: a study lost because no full text could be obtained is a supply failure, and a study lost
because the screener rejected it is a judgement. Presenting them together would blame screening
for the pipeline's largest loss in several projects.

    search      the gold study appeared in the PubMed results at all
    abstract    survived abstract screening
    retrieval   full text was actually obtained
    fulltext    survived full-text screening

Writes reports/gold_survival_by_stage.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT.parent / "neurometabench" / "data" / "included_studies.csv"

# Benchmark meta-analysis PMID -> project directory name.
META_TO_PROJECT = {
    "34400176": "cue_reactivity", "32078973": "decision_making", "35664889": "dementia",
    "35413444": "emotion_regulation_2022", "22282036": "executive_function",
    "29944961": "problem_solving", "36436737": "social", "36100907": "vbm_of_ptsd",
    "36115222": "vbm_of_substance_use",
}
BARE_VERSION = re.compile(r"^v(\d+)$")
STAGES = ["search", "abstract", "retrieval", "fulltext"]


def load_gold() -> dict[str, set[str]]:
    df = pd.read_csv(BENCHMARK)
    df["study_pmid"] = pd.to_numeric(df["study_pmid"], errors="coerce")
    df = df.dropna(subset=["study_pmid"])
    df["study_pmid"] = df["study_pmid"].astype(int).astype(str)
    df["meta_pmid"] = df["meta_pmid"].astype(str)
    out: dict[str, set[str]] = {}
    for meta, group in df.groupby("meta_pmid"):
        project = META_TO_PROJECT.get(meta)
        if project:
            out[project] = set(group["study_pmid"])
    return out


def canonical_run(project_dir: Path) -> Path | None:
    """Highest bare vN run -- the canonical family, excluding suffixed derivatives."""
    versions = [
        (int(BARE_VERSION.match(p.name).group(1)), p)
        for p in project_dir.iterdir()
        if p.is_dir() and BARE_VERSION.match(p.name) and (p / "outputs").is_dir()
    ]
    return max(versions)[1] if versions else None


def _read(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def survivors(outputs: Path) -> dict[str, set[str] | None]:
    """PMIDs present after each stage. None where the stage produced no artifact."""
    out: dict[str, set[str] | None] = {}

    search = _read(outputs / "search_results.json").get("studies")
    out["search"] = {str(s.get("pmid")).strip() for s in search
                     if isinstance(s, dict) and s.get("pmid")} if search else None

    for stage, fname in (("abstract", "abstract_screening_results.json"),
                         ("fulltext", "fulltext_screening_results.json")):
        rows = _read(outputs / fname).get("screening_results")
        out[stage] = ({str(r.get("study_id")).strip() for r in rows
                       if isinstance(r, dict) and "includ" in str(r.get("decision", "")).lower()}
                      if rows else None)

    rows = _read(outputs / "fulltext_retrieval_results.json").get("studies_with_fulltext")
    out["retrieval"] = ({str(r.get("pmid")).strip() for r in rows
                         if isinstance(r, dict) and r.get("fulltext_available")}
                        if rows else None)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" / "gold_survival_by_stage.csv")
    args = ap.parse_args(argv)

    gold_sets = load_gold()
    rows = []
    print(f"{'project':<24}{'run':<6}{'gold':>5}   " + "".join(f"{s:>11}" for s in STAGES))
    print("-" * 80)
    for project, gold in sorted(gold_sets.items()):
        project_dir = args.projects_root / project
        run = canonical_run(project_dir) if project_dir.is_dir() else None
        if run is None:
            print(f"{project:<24}(no canonical run)")
            continue

        alive = survivors(run / "outputs")
        # Cumulative: a study must have survived every stage up to this one. Intersecting rather
        # than reading each stage independently keeps the curve monotone even if an artifact is
        # regenerated out of order.
        carried: set[str] | None = None
        cells = []
        for stage in STAGES:
            here = alive.get(stage)
            if here is not None:
                carried = here if carried is None else (carried & here)
            kept = len(gold & carried) if carried is not None else None
            cells.append(kept)
            rows.append({
                "project": project, "run": run.name, "stage": stage,
                "gold_total": len(gold),
                "gold_surviving": "" if kept is None else kept,
                "cumulative_recall": "" if kept is None else round(kept / len(gold), 4),
                "artifact_present": "yes" if here is not None else "no",
            })
        pretty = "".join(
            f"{'  n/a' if c is None else f'{c}/{len(gold)}':>11}" for c in cells)
        print(f"{project:<24}{run.name:<6}{len(gold):>5}   {pretty}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

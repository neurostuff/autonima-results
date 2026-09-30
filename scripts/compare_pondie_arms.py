"""Score the pondie document arms against each project's best run.

  python scripts/compare_pondie_arms.py [--gold ../neurometabench/data/included_studies.csv]

For each project, scores best and every <best>[-ft]-pondie-<kind> run with
compare_screening_to_benchmark.py (evaluations go to reports/pondie_arms/, never into the runs),
and reports full-text screening precision/recall, fulltext_incomplete counts, annotation health
(included studies with analyses that got no LLM decision, and decisions whose analysis id is
not autonima's), decision agreement with best, and recorded LLM cost. Writes
reports/pondie_arms/summary.csv.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PROJECTS = ["cue_reactivity", "dementia", "emotion_regulation_2022", "vbm_of_ptsd", "vbm_of_substance_use"]
SYSTEM = {"all_studies", "all_abstract", "all_analyses"}


def score(run: Path, gold: Path, out: Path):
    subprocess.run(
        [sys.executable, "scripts/compare_screening_to_benchmark.py", str(gold), str(run),
         "--output_dir", str(out), "--skip-qualitative-report"],
        cwd=REPO, env={"PYTHONPATH": "scripts", "PATH": "/usr/bin:/bin"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    return json.loads((out / "performance_metrics.json").read_text())["fulltext"]


def decisions(run: Path):
    rows = json.loads((run / "outputs/fulltext_screening_results.json").read_text())["screening_results"]
    return {r["study_id"]: r["decision"] for r in rows}


def annotation_health(run: Path):
    final = json.loads((run / "outputs/final_results.json").read_text())
    with_analyses = {s["pmid"] for s in final["studies"] if s.get("analyses")}
    rows = json.loads((run / "outputs/annotation_results.json").read_text())
    llm = [r for r in rows if r.get("annotation_name") not in SYSTEM]
    annotated = {r["study_id"] for r in llm}
    foreign = sum(not re.fullmatch(r"\d+_analysis_\d+", r["analysis_id"]) for r in llm)
    return len(with_analyses), len(with_analyses - annotated), foreign


def cost(run: Path):
    progress = json.loads((run / "outputs/execution_progress.json").read_text())
    total = progress.get("usage_total") or {}
    return total.get("cost_usd"), total.get("calls")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=REPO.parent / "neurometabench/data/included_studies.csv")
    args = ap.parse_args()
    categories = yaml.safe_load((REPO / "run_categories.yaml").read_text())["projects"]
    rows = []
    for project in PROJECTS:
        best = categories[project]["canonical"]["best"]
        base = REPO / "projects" / project
        runs = [best] + sorted(
            d.name for d in base.iterdir()
            if re.fullmatch(rf"{re.escape(best)}(-ft)?-pondie-(text|records)", d.name)
            and (d / "outputs/final_results.json").is_file()
        )
        best_decisions = decisions(base / best)
        for name in runs:
            run = base / name
            ft = score(run, args.gold, REPO / "reports/pondie_arms" / project / name)
            counts, metrics = ft["counts"], ft["metrics"]
            d = decisions(run)
            common = set(d) & set(best_decisions)
            studies, dropped, foreign = annotation_health(run)
            usd, calls = cost(run) if name != best else (None, None)
            rows.append({
                "project": project, "run": name,
                "tp": counts["true_positives"], "fp": counts["false_positives"],
                "fn": counts["meta_total"] - counts["true_positives"],
                "precision": round(metrics["precision"], 3),
                "recall": round(metrics["recall_all_meta"], 3),
                "incomplete": sum(v == "fulltext_incomplete" for v in d.values()),
                "agree_with_best": f"{sum(d[p] == best_decisions[p] for p in common)}/{len(common)}",
                "annotated_studies": studies, "studies_dropped": dropped, "non_host_ids": foreign,
                "cost_usd": round(usd, 2) if usd is not None else "", "llm_calls": calls or "",
            })
    out = REPO / "reports/pondie_arms/summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in rows[0]}
    print("  ".join(k.ljust(widths[k]) for k in rows[0]))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in r))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

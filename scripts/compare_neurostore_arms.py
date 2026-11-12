"""Score the NeuroStore arms against each project's best run.

  python scripts/compare_neurostore_arms.py [--projects P ...] [--skip-meta]

For each project, best and each <best>-ns-<kind> run present (scripts/neurostore_arms.py):
  1. `autonima meta` on the run, if its maps are missing (MKDA, FDR: the defaults best used);
  2. screening against NeuroMetaBench with compare_screening_to_benchmark.py, plus recall over
     the gold studies NeuroStore has coordinates for (reports/neurostore_coverage/gold_studies.csv);
  3. maps against the expert maps with compare_baselines_to_benchmark.py (brain-masked Pearson r
     and r^2 of unthresholded z, Dice of FDR maps at z > 1.96), the metric of the paper's
     pipeline-versus-baseline comparison;
  4. LLM calls and cost recorded by the run.

Evaluations go to reports/neurostore_arms/<project>/<run>/, never into the runs. Writes
reports/neurostore_arms/summary.csv (one row per run) and maps.csv (one row per contrast).
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "neurostore_arms"
GOLD = REPO.parent / "neurometabench" / "data" / "included_studies.csv"
PROJECTS = ["cue_reactivity", "decision_making", "dementia", "emotion_regulation_2022", "executive_function",
            "problem_solving", "social", "vbm_of_ptsd", "vbm_of_substance_use"]
KINDS = ["coords", "search"]


def best_run(project):
    return yaml.safe_load((REPO / "run_categories.yaml").read_text())["projects"][project]["canonical"]["best"]


def ensure_meta(run: Path):
    if (run / "outputs" / "meta_analysis_results").exists():
        return
    # The autonima beside this interpreter, so the maps come from the same install as the run.
    local = Path(sys.executable).parent / "autonima"
    autonima = str(local) if local.exists() else shutil.which("autonima")
    subprocess.run([autonima, "meta", str(run / "outputs")], check=True, cwd=REPO,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def screening(run: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "scripts/compare_screening_to_benchmark.py", str(GOLD), str(run),
         "--output_dir", str(out), "--skip-qualitative-report"],
        cwd=REPO, env={**os.environ, "PYTHONPATH": "scripts"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    metrics = json.loads((out / "performance_metrics.json").read_text())
    classes = json.loads((out / "study_classifications.json").read_text())
    return metrics, classes


def maps(project: str, run: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "scripts/compare_baselines_to_benchmark.py", "--project", project,
         "--autonima-run", run, "--output-dir", str(out)],
        cwd=REPO, env={**os.environ, "PYTHONPATH": "scripts"},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    rows = []
    for path in out.glob("*.csv"):
        with open(path) as fh:
            rows += [r for r in csv.DictReader(fh) if r.get("arm") == "autonima"]
    return rows


def cost(run: Path):
    progress = json.loads((run / "outputs" / "execution_progress.json").read_text())
    total = progress.get("usage_total") or {}
    return total.get("cost_usd"), total.get("calls")


def neurostore_gold():
    path = REPO / "reports" / "neurostore_coverage" / "gold_studies.csv"
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row["neurostore_source"]:
                out.setdefault(row["project"], set()).add(row["pmid"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", nargs="*", default=PROJECTS)
    ap.add_argument("--skip-meta", action="store_true")
    args = ap.parse_args()
    ns_gold = neurostore_gold()
    summary, contrasts = [], []
    for project in args.projects:
        best = best_run(project)
        for name in [best] + [f"{best}-ns-{k}" for k in KINDS]:
            run = REPO / "projects" / project / name
            if not (run / "outputs" / "final_results.json").exists():
                continue
            if not args.skip_meta:
                ensure_meta(run)
            out = OUT / project / name
            metrics, classes = screening(run, out / "screening")
            final = metrics["fulltext_with_coords"]
            included = set(classes["fulltext_with_coords"]["true_positives"]) if "fulltext_with_coords" in classes else set()
            subset = ns_gold.get(project, set())
            map_rows = maps(project, name, out / "maps")
            for r in map_rows:
                contrasts.append({"project": project, "run": name, "contrast": r["manual_annotation"],
                                  "r": r["pearson_r"], "r2": r["r2"], "dice": r["dice"],
                                  "studies": r["studies"], "points": r["points"]})
            r2 = [float(r["r2"]) for r in map_rows if r.get("r2") not in (None, "")]
            usd, calls = cost(run)
            summary.append({
                "project": project, "run": name, "arm": "best" if name == best else name.rsplit("-", 1)[-1],
                "included": final["counts"]["included_total"],
                "precision": round(final["metrics"]["precision"], 3),
                # Over every gold study, so arms with different searches share a denominator.
                "recall": round(final["counts"]["true_positives"] / final["counts"]["meta_total"], 3),
                "recall_in_search": round(final["metrics"].get("recall_in_search", float("nan")), 3),
                "gold_with_ns_coords": len(subset),
                "recall_ns_subset": round(len(included & subset) / len(subset), 3) if subset else "",
                "contrasts": len(r2), "mean_r2": round(sum(r2) / len(r2), 3) if r2 else "",
                "llm_calls": calls, "cost_usd": round(usd, 2) if usd is not None else "",
            })
            print(summary[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary.csv", summary), ("maps.csv", contrasts)):
        if rows:
            with open(OUT / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(OUT / name)


if __name__ == "__main__":
    main()

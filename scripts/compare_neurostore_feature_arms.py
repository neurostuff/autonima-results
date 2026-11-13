"""Score the NeuroStore feature arms against their baseline, the NeuroStore search arm.

  python scripts/compare_neurostore_feature_arms.py [--projects P ...] [--skip-meta]

Run with the feat/neurostore-features autonima venv (it runs `autonima meta`). For each project,
the baseline (reports/neurostore_arms/baseline.yaml) and each <best>-ns-<arm> run present
(scripts/neurostore_feature_arms.py):

  1. `autonima meta` on the run if its maps are missing (MKDA, FDR, as the baseline);
  2. screening against NeuroMetaBench with compare_screening_to_benchmark.py, where the run has
     full-text decisions (rec-global has none);
  3. selection-level inclusion for every run alike: a study counts as included when any of its
     analyses is selected for any target, read from nimads_annotation.json. It is the only
     study-level measure rec-global has, and it puts all arms on one footing;
  4. maps against the expert maps with compare_baselines_to_benchmark.py (brain-masked r, r^2
     and Dice of FDR maps at z > 1.96);
  5. cost: calls and tokens per stage, summed over every pass of the arm
     (outputs/usage_history/ plus the last pass). The baseline's progress file holds only its
     last pass, so its calls are counted from its decisions instead, and its tokens are not
     reported.

Evaluations go to reports/neurostore_feature_arms/<project>/<run>/, never into the runs. Writes
reports/neurostore_feature_arms/summary.csv and maps.csv.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from compare_neurostore_arms import GOLD, PROJECTS, ensure_meta, maps, screening  # noqa: E402

OUT = REPO / "reports" / "neurostore_feature_arms"
ARMS = ["rec", "rec-comb", "rec-comb-noabs", "rec-global", "rec-comb-ft"]
SYSTEM_NOTES = {"all_studies", "all_abstract", "all_analyses"}


def gold_studies(project: str) -> set:
    meta = str(json.loads((REPO / "projects" / project / "nmb_mappings.json").read_text())["meta_pmid"])
    with open(GOLD) as fh:
        return {row["study_pmid"] for row in csv.DictReader(fh) if row["meta_pmid"] == meta}


def selected_studies(run: Path) -> set:
    """Studies with at least one analysis selected for at least one target."""
    path = run / "outputs" / "nimads_annotation.json"
    if not path.exists():
        return set()
    selected = set()
    for note in json.loads(path.read_text())["notes"]:
        if any(v is True for k, v in note["note"].items() if k not in SYSTEM_NOTES):
            selected.add(note["analysis"].split("_analysis_")[0])
    return selected


def usage(run: Path) -> dict:
    """Calls, tokens and cost per stage, summed over the arm's passes."""
    outputs = run / "outputs"
    files = sorted((outputs / "usage_history").glob("*.json")) if (outputs / "usage_history").exists() else []
    current = outputs / "execution_progress.json"
    seen = {f.read_bytes() for f in files}
    if current.exists() and current.read_bytes() not in seen:
        files.append(current)
    totals: dict = {}
    for path in files:
        progress = json.loads(path.read_text())
        for stage in progress.get("stages") or []:
            for key, value in (stage.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    bucket = totals.setdefault(stage.get("stage") or stage.get("name"), {})
                    bucket[key] = bucket.get(key, 0) + value
    return totals


def baseline_calls(run: Path) -> dict:
    """The baseline's calls by stage, counted from its decisions (its usage covers one pass)."""
    outputs = run / "outputs"
    fulltext = json.loads((outputs / "fulltext_screening_results.json").read_text())["screening_results"]
    annotation = json.loads((outputs / "annotation_results.json").read_text())
    annotated = {d["study_id"] for d in annotation if d.get("annotation_name") not in SYSTEM_NOTES}
    return {"fulltext": {"calls": len(fulltext)}, "annotation": {"calls": len(annotated)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", nargs="*", default=PROJECTS)
    ap.add_argument("--skip-meta", action="store_true")
    args = ap.parse_args()
    baselines = yaml.safe_load((REPO / "reports" / "neurostore_arms" / "baseline.yaml").read_text())["projects"]
    summary, contrasts = [], []
    for project in args.projects:
        entry = baselines[project]
        gold = gold_studies(project)
        for name in [entry["run"]] + [f"{entry['best']}-ns-{arm}" for arm in ARMS]:
            run = REPO / "projects" / project / name
            if not (run / "outputs" / "final_results.json").exists():
                continue
            arm = "baseline" if name == entry["run"] else name.split("-ns-", 1)[1]
            if not args.skip_meta:
                ensure_meta(run)
            out = OUT / project / name
            row = {"project": project, "run": name, "arm": arm, "gold": len(gold)}
            if (run / "outputs" / "fulltext_screening_results.json").exists() and arm != "rec-global":
                metrics, _ = screening(run, out / "screening")
                final = metrics["fulltext_with_coords"]
                row.update({
                    "fulltext_included": final["counts"]["included_total"],
                    "fulltext_precision": round(final["metrics"]["precision"], 3),
                    "fulltext_recall": round(final["counts"]["true_positives"] / final["counts"]["meta_total"], 3),
                })
            selected = selected_studies(run)
            hits = len(selected & gold)
            row.update({
                "selected_studies": len(selected),
                "selected_precision": round(hits / len(selected), 3) if selected else "",
                "selected_recall": round(hits / len(gold), 3) if gold else "",
            })
            map_rows = maps(project, name, out / "maps")
            for r in map_rows:
                contrasts.append({"project": project, "run": name, "arm": arm, "contrast": r["manual_annotation"],
                                  "r": r["pearson_r"], "r2": r["r2"], "dice": r["dice"],
                                  "studies": r["studies"], "points": r["points"]})
            r2 = [float(r["r2"]) for r in map_rows if r.get("r2") not in (None, "")]
            dice = [float(r["dice"]) for r in map_rows if r.get("dice") not in (None, "")]
            row.update({"contrasts": len(r2), "mean_r2": round(sum(r2) / len(r2), 3) if r2 else "",
                        "mean_dice": round(sum(dice) / len(dice), 3) if dice else ""})
            stages = baseline_calls(run) if arm == "baseline" else usage(run)
            for stage in ("abstract", "fulltext", "annotation"):
                row[f"{stage}_calls"] = (stages.get(stage) or {}).get("calls", "")
                if arm != "baseline":
                    row[f"{stage}_input_tokens"] = (stages.get(stage) or {}).get("input_tokens", "")
                    row[f"{stage}_output_tokens"] = (stages.get(stage) or {}).get("output_tokens", "")
            if arm != "baseline":
                row["cost_usd"] = round(sum((s.get("cost_usd") or 0) for s in stages.values()), 3)
            summary.append(row)
            print(row, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary.csv", summary), ("maps.csv", contrasts)):
        if rows:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            with open(OUT / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(OUT / name)


if __name__ == "__main__":
    main()

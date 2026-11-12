"""Diagnostic: NeuroStore coordinates on best's own studies, contrast by contrast.

  python scripts/neurostore_same_studies.py [--projects P ...]

The coords arm's maps hold more studies than best's (NeuroStore has coordinates for studies
whose tables best could not parse) and score slightly lower. This separates the two causes.
For each target, the coords arm's analysis selections are kept only for studies that best
included in that target, so the map differs from best's only in whose coordinates (and
analysis splits) it uses. Written to projects/<project>/<best>-ns-coords-beststudies/ (NiMADS
files only), fitted with `autonima meta`, scored with compare_baselines_to_benchmark.py.
Writes reports/neurostore_arms/same_studies.csv.
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
PROJECTS = ["cue_reactivity", "decision_making", "dementia", "emotion_regulation_2022", "executive_function",
            "problem_solving", "social", "vbm_of_ptsd", "vbm_of_substance_use"]
SYSTEM = {"all_studies", "all_abstract", "all_analyses"}


def best_run(project):
    return yaml.safe_load((REPO / "run_categories.yaml").read_text())["projects"][project]["canonical"]["best"]


def included_by_target(annotation):
    out = {}
    for note in annotation["notes"]:
        study = note["analysis"].split("_analysis_")[0]
        for target, value in note["note"].items():
            if target not in SYSTEM and value is True:
                out.setdefault(target, set()).add(study)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", nargs="*", default=PROJECTS)
    args = ap.parse_args()
    autonima = Path(sys.executable).parent / "autonima"
    rows = []
    for project in args.projects:
        best = best_run(project)
        src = REPO / "projects" / project / f"{best}-ns-coords" / "outputs"
        best_ann = json.loads((REPO / "projects" / project / best / "outputs" / "nimads_annotation.json").read_text())
        keep = included_by_target(best_ann)
        diag = REPO / "projects" / project / f"{best}-ns-coords-beststudies"
        if diag.exists():
            shutil.rmtree(diag)
        (diag / "outputs").mkdir(parents=True)
        shutil.copy2(src / "nimads_studyset.json", diag / "outputs" / "nimads_studyset.json")
        annotation = json.loads((src / "nimads_annotation.json").read_text())
        for note in annotation["notes"]:
            study = note["analysis"].split("_analysis_")[0]
            for target in list(note["note"]):
                if target not in SYSTEM and note["note"][target] is True and study not in keep.get(target, set()):
                    note["note"][target] = False
        (diag / "outputs" / "nimads_annotation.json").write_text(json.dumps(annotation))
        subprocess.run([str(autonima), "meta", str(diag / "outputs")], check=True, cwd=REPO,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = REPO / "reports" / "neurostore_arms" / project / diag.name / "maps"
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "scripts/compare_baselines_to_benchmark.py", "--project", project,
                        "--autonima-run", diag.name, "--output-dir", str(out)],
                       cwd=REPO, env={**os.environ, "PYTHONPATH": "scripts"}, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out / "baseline_vs_autonima.csv") as fh:
            for r in csv.DictReader(fh):
                if r["arm"] == "autonima":
                    rows.append({"project": project, "contrast": r["manual_annotation"], "r2": r["r2"],
                                 "dice": r["dice"], "studies": r["studies"], "points": r["points"]})
        print(project, "done", flush=True)
    with open(REPO / "reports" / "neurostore_arms" / "same_studies.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

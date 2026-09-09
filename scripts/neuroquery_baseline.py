#!/usr/bin/env python3
"""NeuroQuery as a baseline: does curating a studyset beat asking a trained model for the map?

WHY THIS EXISTS

Every baseline in §7 is a *search* baseline -- a PubMed query, coordinates extracted, pooled. That
tests the pipeline against the Neurosynth-style workflow it descends from, but not against the
current generation of automated map generators. Since the paper positions itself as Neurosynth's
successor, the obvious reviewer question is "why not just ask NeuroQuery for a map of this
construct?", and it currently has no answer.

This answers it. NeuroQuery predicts a brain map from free text, with no studyset, no screening
and no coordinate extraction, in milliseconds. It is the strongest available "do nothing" arm.

WHY R-SQUARED ONLY

Per the metric/map rule: dice compares FDR-corrected thresholded maps, and NeuroQuery produces no
FDR-corrected map -- thresholding its output would mean picking a level with no error control
behind it, which is exactly the pairing the rule forbids. So this arm is reported on r-squared
over unthresholded maps, which is the metric that applies. Saying "NeuroQuery scores no dice"
would be an artefact of the comparison, not a result.

THE QUERIES ARE THE METHOD, SO THEY ARE FIXED IN ADVANCE

NeuroQuery is a text model: the query determines the map, so query choice is a real methodological
decision and a tempting place to overfit. The queries below were written once from each column's
construct -- what a neuroimaging researcher would type -- and were NOT revised against the
resulting scores. Several benchmark column names are unusable as queries on their own ("decrease",
"functional", "alcohol"), which is why they are spelled out rather than derived automatically.

That discipline matters here more than usual: Supplementary S2 shows that revising criteria
against error reports is worth +0.031 while fixing mis-specified criteria is worth +0.221. Tuning
these queries against the benchmark would be the same mistake, in our own favour this time.

Writes reports/neuroquery_baseline.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_exclusions import filter_rows  # noqa: E402

_RESAMPLE_PARAMS: set[str] = set()
MANUAL_BASE = Path("/home/zorro/repos/neurometabench/analysis")
UNCORRECTED_MAP = "z.nii.gz"

# (project, manual_annotation) -> query. Written from the construct, not from the scores.
QUERIES: dict[tuple[str, str], str] = {
    ("cue_reactivity", "1_reward_neutral_2020_wbonly"): "cue reactivity",
    ("cue_reactivity", "2_drug_neutral_2020_wbonly"): "drug cue reactivity",
    ("cue_reactivity", "3_natural_neutral_2020_wbonly"): "natural reward cue reactivity",
    ("decision_making", "adm_july2019"): "decision making under ambiguity",
    ("decision_making", "perceptual_dm_july2019"): "perceptual decision making",
    ("decision_making", "rdm_july2019"): "risky decision making",
    ("dementia", "all"): "dementia",
    ("dementia", "decrease"): "decreased brain activity in dementia",
    ("dementia", "functional"): "functional activation in dementia",
    ("dementia", "structural"): "gray matter atrophy in dementia",
    ("emotion_regulation_2022", "decrease"): "decreasing negative emotion",
    ("emotion_regulation_2022", "increase"): "increasing emotional response",
    ("emotion_regulation_2022", "maintain"): "maintaining emotional response",
    ("emotion_regulation_2022", "reappraisal"): "cognitive reappraisal",
    ("executive_function", "all"): "executive function",
    ("executive_function", "flexibility"): "cognitive flexibility task switching",
    ("executive_function", "inhibition"): "response inhibition",
    ("executive_function", "working_memory"): "working memory",
    ("problem_solving", "calculation_mni_final"): "mental arithmetic calculation",
    ("problem_solving", "demand_mni_final"): "logical reasoning",
    ("problem_solving",
     "global_calculationandvisuospatialandword_mni_final"): "problem solving reasoning",
    ("problem_solving", "visuospatialreasoning_mni_final"): "visuospatial reasoning",
    ("problem_solving", "wordproblems_mni_final"): "verbal problem solving",
    ("social", "affiliation_merged"): "attachment and affiliation",
    ("social", "all_merged"): "social cognition",
    ("social", "others_merged"): "person perception of others",
    ("social", "self_merged"): "self referential processing",
    ("social", "soccomm_merged"): "social communication",
    ("vbm_of_ptsd", "nonptsdgtptsd_merged"): "posttraumatic stress disorder gray matter",
    ("vbm_of_substance_use", "alcohol"): "alcohol dependence gray matter",
    ("vbm_of_substance_use", "all_drug_classes"): "substance use disorder gray matter",
    ("vbm_of_substance_use", "nicotine"): "nicotine smoking gray matter",
}


def gold_dir(project: str, key: str) -> Path | None:
    d = MANUAL_BASE / project / key
    if d.is_dir():
        return d
    for c in (MANUAL_BASE / project).glob("*"):
        if c.is_dir() and c.name.lower().replace("-", "_") == key.lower().replace("-", "_"):
            return c
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-table", type=Path,
                    default=REPO_ROOT / "reports" / "cross_project_best_baseline.csv")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" / "neuroquery_baseline.csv")
    ap.add_argument("--map-dir", type=Path,
                    default=REPO_ROOT / "reports" / "neuroquery_maps",
                    help="where the predicted maps are cached, one nifti per column")
    args = ap.parse_args(argv)

    import nibabel as nib
    from nilearn.image import resample_to_img
    from neuroquery import fetch_neuroquery_model, NeuroQueryModel
    import inspect
    global _RESAMPLE_PARAMS
    _RESAMPLE_PARAMS = set(inspect.signature(resample_to_img).parameters)

    with open(args.baseline_table) as f:
        wanted = filter_rows(list(csv.DictReader(f)), label="neuroquery baseline")

    print("loading NeuroQuery model")
    model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model())
    args.map_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for r in wanted:
        project, key = r["project"], r["manual_annotation"]
        query = QUERIES.get((project, key))
        if not query:
            missing.append(f"{project}/{key}: no query defined")
            continue
        gd = gold_dir(project, key)
        if gd is None or not (gd / UNCORRECTED_MAP).exists():
            missing.append(f"{project}/{key}: no expert map")
            continue

        gold_img = nib.load(str(gd / UNCORRECTED_MAP))
        pred = model(query)["brain_map"]
        # NeuroQuery is 4mm isotropic and the benchmark is 2mm, so the prediction is resampled to
        # the expert grid -- the same direction the other comparison scripts use, which leaves the
        # reference map untouched.
        # copy_header is nilearn >= 0.11 only and the pixi env pins 0.10.1, so it is passed
        # only when supported rather than pinned to one nilearn.
        kw = dict(interpolation="continuous", force_resample=True)
        if "copy_header" in _RESAMPLE_PARAMS:
            kw["copy_header"] = True
        pred_r = resample_to_img(pred, gold_img, **kw)
        nib.save(pred_r, args.map_dir / f"{project}__{key}.nii.gz")

        a = np.asarray(pred_r.dataobj)
        g = gold_img.get_fdata()
        m = np.isfinite(a) & np.isfinite(g)
        r2 = float(np.corrcoef(a[m].ravel(), g[m].ravel())[0, 1] ** 2)

        rows.append({
            "project": project, "manual_annotation": key, "query": query,
            "neuroquery_r2": round(r2, 6),
            # For reference, from the committed table -- both already r2 on unthresholded maps.
            "autonima_r2": r["autonima"], "best_baseline_r2": r["best_available"],
            "delta_autonima_vs_neuroquery": round(float(r["autonima"]) - r2, 6),
        })
        print(f"  {project:24}{key[:30]:32}{query[:34]:36}r2={r2:.4f}"
              f"  autonima={float(r['autonima']):.4f}", flush=True)

    if not rows:
        print("no columns scored")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    nq = [x["neuroquery_r2"] for x in rows]
    au = [float(x["autonima_r2"]) for x in rows]
    bb = [float(x["best_baseline_r2"]) for x in rows]
    d = [x["delta_autonima_vs_neuroquery"] for x in rows]
    print(f"\n  {len(rows)} columns")
    print(f"  mean r2   neuroquery {st.mean(nq):.3f}   search baseline {st.mean(bb):.3f}   "
          f"autonima {st.mean(au):.3f}")
    print(f"  autonima - neuroquery: mean {st.mean(d):+.3f}, median {st.median(d):+.3f}, "
          f"ahead in {sum(1 for x in d if x > 0)}/{len(d)}")
    for m_ in missing:
        print(f"  missing: {m_}")
    print(f"  wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

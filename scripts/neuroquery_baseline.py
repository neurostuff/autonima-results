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
CORRECTED_MAP = "z_corr-FDR_method-indep.nii.gz"
DICE_THRESHOLD = 1.96
MIN_TOPK = 200  # below this the gold cluster is too small for a stable ranked comparison

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


def top_k_mask(a: np.ndarray, k: int) -> np.ndarray:
    """The k highest-valued voxels, as a boolean mask."""
    a = np.where(np.isfinite(a), a, -np.inf)
    idx = np.argpartition(a.ravel(), -k)[-k:]
    m = np.zeros(a.size, dtype=bool)
    m[idx] = True
    return m.reshape(a.shape)


def top_k_dice(candidate: np.ndarray, gold_raw: np.ndarray, k: int) -> float:
    """Overlap of the two maps' k strongest voxels.

    WHY THIS IS HERE ALONGSIDE R-SQUARED, AND WHY IT IS THE FAIRER NUMBER

    R-squared over all voxels quietly rewards sharing the expert map's *form*. Every MKDA arm is
    about 94% exact zeros, non-negative, on the same mask and at a similar scale, so two MKDA maps
    correlate substantially before any anatomy agrees. NeuroQuery is dense (100% non-zero), signed
    (down to -5.7) and roughly five times smaller in typical magnitude, so it forgoes all of that
    shared-background correlation regardless of where it puts its peaks.

    Ranking each map and taking the same number of voxels removes sparsity, sign and scale, and
    asks only whether the maps point at the same places. It narrows the NeuroQuery gap by about
    2.4x -- which means the r-squared comparison alone overstates the case, and both belong in the
    table.
    """
    return _dice_of(top_k_mask(gold_raw, k), top_k_mask(candidate, k))


def _dice_of(m1: np.ndarray, m2: np.ndarray) -> float:
    total = m1.sum() + m2.sum()
    return 0.0 if total == 0 else float(2.0 * (m1 & m2).sum() / total)


def auto_column(project: str, key: str) -> str:
    f = REPO_ROOT / "projects" / project / "nmb_mappings.json"
    if f.exists():
        try:
            return (json.loads(f.read_text()).get("annotation_mappings") or {}).get(key, key)
        except ValueError:
            pass
    return key


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
    from run_tiers import resolve_tier
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

        # Comparators on the same footing: the pipeline's own column and the project-wide
        # fixed-pool arm, both raw z, ranked against the same k.
        run = resolve_tier(project, "canonical", "best")
        meta = REPO_ROOT / "projects" / project / run / "outputs" / "meta_analysis_results"
        gcor = gd / CORRECTED_MAP
        k = int((nib.load(str(gcor)).get_fdata() > DICE_THRESHOLD).sum()) if gcor.exists() else 0
        # The r-squared columns compare against the BEST AVAILABLE baseline, which per column is
        # either the targeted search or the project-wide broad arm. The ranked metric has to use
        # the same arm or the two measures are not describing the same comparison.
        if r.get("best_available_source") == "targeted":
            best_path = (REPO_ROOT / "projects" / project / "baselines" / key / "outputs"
                         / "meta_analysis_results" / "all_analyses" / UNCORRECTED_MAP)
        else:
            best_path = meta / "all_studies" / UNCORRECTED_MAP

        topk = {}
        if k >= MIN_TOPK:
            for name, path in (("neuroquery", None),
                               ("pipeline", meta / auto_column(project, key) / UNCORRECTED_MAP),
                               ("best_baseline", best_path),
                               ("broad", meta / "all_studies" / UNCORRECTED_MAP)):
                arr = a if path is None else (
                    nib.load(str(path)).get_fdata() if path.exists() else None)
                if arr is not None and arr.shape == g.shape:
                    topk[name] = round(top_k_dice(arr, g, k), 6)

        rows.append({
            "project": project, "manual_annotation": key, "query": query,
            "neuroquery_r2": round(r2, 6),
            "topk_k": k if k >= MIN_TOPK else "",
            "topk_dice_neuroquery": topk.get("neuroquery", ""),
            "topk_dice_pipeline": topk.get("pipeline", ""),
            "topk_dice_best_baseline": topk.get("best_baseline", ""),
            "topk_dice_broad": topk.get("broad", ""),
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
    tk = {n: [float(x[f"topk_dice_{n}"]) for x in rows if x[f"topk_dice_{n}"] != ""]
          for n in ("neuroquery", "pipeline", "best_baseline", "broad")}
    if tk["neuroquery"]:
        print(f"\n  top-k dice (form-insensitive, n={len(tk['neuroquery'])}): "
              f"neuroquery {st.mean(tk['neuroquery']):.3f}   "
              f"best baseline {st.mean(tk['best_baseline']):.3f}   "
              f"pipeline {st.mean(tk['pipeline']):.3f}")
        rr = st.mean(nq) / st.mean(bb)
        tr = st.mean(tk["neuroquery"]) / st.mean(tk["best_baseline"])
        print(f"  neuroquery as a share of the best baseline: {rr:.1%} on r2, "
              f"{tr:.1%} ranked  ->  r2 overstates the gap {tr / rr:.1f}x")
    for m_ in missing:
        print(f"  missing: {m_}")
    print(f"  wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Coordinate-based meta-analysis of an exported NiMADS studyset, one per target.

    pip install nimare
    python run_meta.py REVIEW_DIR [--targets working_memory ...]
                                  [--estimator mkdadensity|ale|kda] [--corrector fdr|montecarlo|bonferroni]
                                  [--unknown-space mni|exclude] [--output-dir results/meta]

Reads results/nimads/studyset.json and annotation.json (written by
`ledger.py export`) and writes results/meta/<target>/ with NiMARE's maps and a
summary.json per target. Estimator and corrector default to review.yaml `meta:`.

Points whose coordinate space is unknown (null in the export) need a stated policy, recorded in
every summary.json. NiMARE itself labels them "UNKNOWN", leaves them untransformed and relabels
them as the dataset's space (MNI), so before this option every run fitted them as MNI without
saying so:
    mni      fit them as MNI (the default: the same maps as before, now stated)
    exclude  leave every analysis with an unknown-space point out of the fit
review.yaml `meta: unknown_space:` sets the default.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

UNKNOWN_SPACE = ("mni", "exclude")


def apply_unknown_space(studyset_data: dict, ids: list, policy: str):
    """Resolve unknown point spaces for one target's analyses, explicitly.

    Returns (studyset copy, analysis ids to fit, report). With "mni", every unknown-space point
    is labelled MNI; with "exclude", an analysis holding any unknown-space point is dropped.
    NiMARE takes an analysis's space from its first point, so analyses whose points mix spaces
    are counted too: their later points follow the first point's space whatever they say."""
    if policy not in UNKNOWN_SPACE:
        raise ValueError(f"unknown-space policy must be one of {UNKNOWN_SPACE}")
    data = copy.deepcopy(studyset_data)
    wanted = set(ids)
    keep, report = [], {"policy": policy, "analyses_with_unknown_space": 0, "points_with_unknown_space": 0,
                        "studies_with_unknown_space": 0, "analyses_excluded": 0, "analyses_mixed_space": 0,
                        "analysis_ids_with_unknown_space": []}
    studies_hit = set()
    for study in data.get("studies", []):
        for analysis in study.get("analyses", []):
            if analysis["id"] not in wanted:
                continue
            points = analysis.get("points") or []
            unknown = [p for p in points if not p.get("space")]
            if len({p.get("space") or None for p in points}) > 1:
                report["analyses_mixed_space"] += 1
            if unknown:
                report["analyses_with_unknown_space"] += 1
                report["points_with_unknown_space"] += len(unknown)
                report["analysis_ids_with_unknown_space"].append(analysis["id"])
                studies_hit.add(study["id"])
                if policy == "exclude":
                    report["analyses_excluded"] += 1
                    continue
                for point in unknown:
                    point["space"] = "MNI"
            keep.append(analysis["id"])
    report["studies_with_unknown_space"] = len(studies_hit)
    return data, keep, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review_dir", type=Path)
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--estimator")
    ap.add_argument("--corrector")
    ap.add_argument("--min-studies", type=int, default=10,
                    help="skip targets with fewer included studies (CBMA is unstable below ~10-17)")
    ap.add_argument("--unknown-space", choices=UNKNOWN_SPACE,
                    help="unknown-space points: fit as MNI, or exclude their analyses (default: review.yaml "
                         "meta.unknown_space, else mni)")
    ap.add_argument("--output-dir", type=Path, default=Path("results/meta"),
                    help="where maps go, relative to REVIEW_DIR (default results/meta)")
    args = ap.parse_args(argv)

    try:
        from nimare.correct import FDRCorrector, FWECorrector
        from nimare.meta.cbma import ALE, KDA, MKDADensity
        from nimare.nimads import Studyset
        from nimare.workflows import CBMAWorkflow
    except ImportError:
        print("NiMARE is required for this stage: pip install nimare", file=sys.stderr)
        return 1

    spec_meta = {}
    spec_path = args.review_dir / "review.yaml"
    if spec_path.exists():
        import yaml  # type: ignore
        spec_meta = (yaml.safe_load(spec_path.read_text()) or {}).get("meta") or {}
    estimator_name = args.estimator or spec_meta.get("estimator", "mkdadensity")
    corrector_name = args.corrector or spec_meta.get("corrector", "fdr")
    estimator_args = spec_meta.get("estimator_args") or {}
    corrector_args = spec_meta.get("corrector_args") or {}
    unknown_space = args.unknown_space or spec_meta.get("unknown_space", "mni")
    if unknown_space not in UNKNOWN_SPACE:
        print(f"review.yaml meta.unknown_space must be one of {UNKNOWN_SPACE}", file=sys.stderr)
        return 1
    meta_root = args.output_dir if args.output_dir.is_absolute() else args.review_dir / args.output_dir

    nimads = args.review_dir / "results" / "nimads"
    studyset_data = json.loads((nimads / "studyset.json").read_text())
    annotation = json.loads((nimads / "annotation.json").read_text())
    targets = args.targets or list(annotation["note_keys"])

    estimators = {"mkdadensity": MKDADensity, "ale": ALE, "kda": KDA}
    if estimator_name not in estimators:
        print(f"unknown estimator {estimator_name}", file=sys.stderr)
        return 1

    def corrector():
        if corrector_name == "fdr":
            return FDRCorrector(**corrector_args)
        if corrector_name in ("montecarlo", "bonferroni"):
            return FWECorrector(method=corrector_name, **corrector_args)
        raise SystemExit(f"unknown corrector {corrector_name}")

    summary = {}
    for target in targets:
        selected = [n["analysis_id"] for n in annotation["notes"] if n["note"].get(target) is True]
        target_data, ids, space_report = apply_unknown_space(studyset_data, selected, unknown_space)
        study_of = {a["id"]: s["id"] for s in studyset_data.get("studies", []) for a in s.get("analyses", [])}
        n_studies = len({study_of.get(i, i.split("-", 1)[0]) for i in ids})
        out_dir = meta_root / target
        out_dir.mkdir(parents=True, exist_ok=True)
        info = {"target": target, "analyses": len(ids), "studies": n_studies, "selected_analyses": len(selected),
                "estimator": estimator_name, "corrector": corrector_name, "unknown_space": space_report}
        if n_studies < args.min_studies:
            info["skipped"] = f"only {n_studies} studies (< --min-studies {args.min_studies})"
            print(json.dumps(info))
            summary[target] = info
            (out_dir / "summary.json").write_text(json.dumps(info, indent=1))
            continue
        studyset = Studyset(target_data).slice(analyses=ids)
        dataset = studyset.to_dataset()
        workflow = CBMAWorkflow(estimator=estimators[estimator_name](**estimator_args), corrector=corrector(),
                                diagnostics="focuscounter", output_dir=str(out_dir))
        workflow.fit(dataset)
        info["output_dir"] = str(out_dir)
        info["dataset_ids"] = len(dataset.ids)
        (out_dir / "summary.json").write_text(json.dumps(info, indent=1))
        summary[target] = info
        print(json.dumps(info))
    (meta_root / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

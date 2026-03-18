#!/usr/bin/env python3
"""Run analysis matching + annotation review reports in one script.

This combines:
- scripts/run_fuzzy_analysis_matching.py
- scripts/generate_annotation_review_reports.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import generate_annotation_review_reports as annotation_review
import run_fuzzy_analysis_matching as fuzzy_matching


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-output-dir",
        type=Path,
        default=None,
        help=(
            "Path to project run dir containing outputs/ (e.g., projects/cue_reactivity/v1 "
            "or projects/social/coordinates/annotation-only). "
            "If omitted, auto-detects the most recently updated run under projects/."
        ),
    )
    parser.add_argument(
        "--manual-dir",
        type=Path,
        default=None,
        help=(
            "Path to project NiMADS dir or merged dir containing nimads_studyset.json. "
            "If omitted, infers project_name from projects/{project_name}/... and uses "
            "../neurometabench/data/nimads/{project_name}."
        ),
    )
    parser.add_argument(
        "--match-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for fuzzy matching outputs: match_results_overall.json and "
            "analysis_fuzzy_matching_report.html. Defaults to project-output-dir/reports."
        ),
    )
    parser.add_argument(
        "--review-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-annotation HTML reports. Defaults to "
            "project-output-dir/reports/annotation_review_reports."
        ),
    )
    parser.add_argument(
        "--match-input-dir",
        type=Path,
        default=None,
        help=(
            "Directory to read match results for review report generation. "
            "Defaults to --match-output-dir (or project-output-dir/reports)."
        ),
    )
    parser.add_argument(
        "--manual-annotation-path",
        type=Path,
        default=None,
        help=(
            "Optional path to merged nimads_annotation.json used to slice "
            "match_results_overall.json into per-annotation manual truth."
        ),
    )
    parser.add_argument(
        "--annotation-mapping-path",
        type=Path,
        default=None,
        help=(
            "Optional path to project annotation mapping JSON (manual-key -> auto-annotation), "
            "such as projects/<project>/nmb_mappings.json. "
            "If omitted, attempts to load projects/{project_name}/nmb_mappings.json."
        ),
    )
    parser.add_argument(
        "--coord-accept-override-threshold",
        type=float,
        default=0.9,
        help=(
            "Coordinate score threshold above which a matched pair is accepted regardless of "
            "combined score/name (greater-than-or-equal). Default: 0.9."
        ),
    )
    parser.add_argument(
        "--exclude-decimal-manual-coordinates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled (default), manual analyses with non-zero decimal coordinate values are "
            "excluded from matching because they likely represent converted coordinates rather "
            "than raw extracted values. Values ending in .0 are not excluded by this heuristic. "
            "Use --no-exclude-decimal-manual-coordinates to disable."
        ),
    )
    parser.add_argument(
        "--decimal-manual-coordinate-handling",
        choices=("exclude", "convert_to_talairach", "keep"),
        default=None,
        help=(
            "How to handle manual analyses that include non-zero decimal coordinates. "
            "'exclude' removes those analyses from matching. "
            "'convert_to_talairach' applies nimare.utils.mni2tal to those manual analyses and keeps them. "
            "'keep' keeps decimal manual coordinates as-is. "
            "If omitted, behavior follows --exclude-decimal-manual-coordinates for backward compatibility."
        ),
    )
    parser.add_argument(
        "--converted-talairach-exact-axis-tolerance",
        type=float,
        default=1.0,
        help=(
            "Axis-wise tolerance used only for converted decimal manual coordinates when "
            "--decimal-manual-coordinate-handling=convert_to_talairach. If each converted manual "
            "coordinate can be one-to-one matched to an auto coordinate with |dx|,|dy|,|dz| <= tolerance, "
            "the set is treated as exact. Default: 1.0."
        ),
    )
    return parser.parse_args()


def resolve_decimal_coordinate_handling(args: argparse.Namespace) -> str:
    if args.decimal_manual_coordinate_handling:
        return str(args.decimal_manual_coordinate_handling)
    return "exclude" if bool(args.exclude_decimal_manual_coordinates) else "keep"


def run_fuzzy_matching_stage(
    args: argparse.Namespace,
    project_output_dir: Path,
    match_output_dir: Path,
) -> dict[str, Any]:
    if not (0.0 <= args.coord_accept_override_threshold <= 1.0):
        raise ValueError("--coord-accept-override-threshold must be between 0.0 and 1.0 (inclusive).")
    if args.converted_talairach_exact_axis_tolerance < 0:
        raise ValueError("--converted-talairach-exact-axis-tolerance must be >= 0.0.")

    decimal_manual_coordinate_handling = resolve_decimal_coordinate_handling(args)
    if (
        decimal_manual_coordinate_handling == "convert_to_talairach"
        and fuzzy_matching.mni2tal is None
    ):
        raise ImportError(
            "--decimal-manual-coordinate-handling convert_to_talairach requires NiMARE "
            "(nimare.utils.mni2tal). Install nimare or choose a different handling mode."
        )

    manual_dir = fuzzy_matching.resolve_manual_dir(project_output_dir, args.manual_dir)
    coordinate_parsing_results = project_output_dir / "outputs" / "coordinate_parsing_results.json"
    if not coordinate_parsing_results.exists():
        raise FileNotFoundError(f"Missing coordinate parsing results: {coordinate_parsing_results}")

    auto_by_pmid = fuzzy_matching.load_auto_parsed_data(coordinate_parsing_results)
    manual_by_pmid, manual_study_names_by_pmid = fuzzy_matching.load_manual_analyses_overall(manual_dir)
    pubget_by_pmid = fuzzy_matching.build_pubget_index(project_output_dir)
    match_result = fuzzy_matching.build_match_results_overall(
        manual_analyses_by_pmid=manual_by_pmid,
        manual_study_names_by_pmid=manual_study_names_by_pmid,
        auto_parsed_by_pmid=auto_by_pmid,
        coord_accept_override_threshold=float(args.coord_accept_override_threshold),
        decimal_manual_coordinate_handling=decimal_manual_coordinate_handling,
        converted_talairach_exact_axis_tolerance=float(args.converted_talairach_exact_axis_tolerance),
    )
    fuzzy_matching.annotate_match_result_with_pubget(match_result, pubget_by_pmid)
    fuzzy_matching.write_match_artifacts(match_output_dir, match_result, pubget_by_pmid=pubget_by_pmid)

    summary = match_result["summary"]
    print(
        f"overall: accepted={summary['accepted']} "
        f"uncertain={summary['uncertain']} unmatched={summary['unmatched']} "
        f"overlap_pmids={summary['overlap_pmids']} "
        f"manual_pmids_total={summary['manual_pmids_total']} "
        f"excluded_manual_only_pmids={summary['excluded_manual_only_pmids']} "
        f"unavailable_manual_decimal_pmids={summary.get('unavailable_manual_decimal_pmids', 0)} "
        f"converted_manual_decimal_analyses={summary.get('converted_manual_decimal_analyses', 0)} "
        f"decimal_handling={decimal_manual_coordinate_handling} "
        f"converted_talairach_exact_axis_tolerance={float(args.converted_talairach_exact_axis_tolerance):.3f} "
        f"pmids_all_manual_accepted={summary['pmids_all_manual_accepted']} "
        f"pmids_with_pubget={summary.get('pmids_with_pubget', 0)}"
    )
    print(f"Wrote matching artifacts to {match_output_dir}")
    return match_result


def run_annotation_review_stage(
    args: argparse.Namespace,
    project_output_dir: Path,
    match_input_dir: Path,
    review_output_dir: Path,
) -> None:
    annotation_mapping_path = annotation_review.resolve_project_annotation_mapping_path(
        project_output_dir,
        args.annotation_mapping_path,
    )
    annotation_review.configure_active_annotations(annotation_mapping_path)

    annotation_results = project_output_dir / "outputs" / "annotation_results.json"
    coordinate_parsing_results = project_output_dir / "outputs" / "coordinate_parsing_results.json"
    auto_annotation_path = project_output_dir / "outputs" / "nimads_annotation.json"
    criteria_mapping_path = project_output_dir / "outputs" / "criteria_mapping.json"
    retrieval_dir = project_output_dir / "retrieval" / "pubget_data"
    manual_annotation_path = annotation_review.resolve_manual_annotation_path(
        project_output_dir,
        args.manual_annotation_path,
    )
    criteria = annotation_review.load_annotation_criteria(criteria_mapping_path)
    if not criteria_mapping_path.exists():
        print(f"Warning: criteria mapping not found at {criteria_mapping_path}; criteria section may be empty.")

    parsed_analyses = annotation_review.load_auto_parsed_analysis_info(coordinate_parsing_results)
    model_decisions = annotation_review.load_model_decisions(annotation_results)
    match_results_by_annotation, overall_fallback = annotation_review.load_match_results_by_annotation(match_input_dir)
    manual_annotation_membership = annotation_review.load_manual_annotation_membership(manual_annotation_path)
    if overall_fallback and not manual_annotation_membership:
        print(
            "Warning: Using match_results_overall.json without nimads_annotation membership; "
            "manual truth cannot be sliced by annotation and may be over-inclusive."
        )
    manual_truth = annotation_review.build_manual_truth_from_match_results(
        match_results_by_annotation,
        overall_fallback=overall_fallback,
        manual_annotation_membership=manual_annotation_membership,
    )
    study_universe_pmids, auto_study_pmids_by_annotation, manual_study_pmids_by_annotation = (
        annotation_review.load_study_pmid_sets_from_annotations(
            auto_annotation_path=auto_annotation_path,
            manual_annotation_path=manual_annotation_path,
        )
    )
    if not study_universe_pmids:
        study_universe_pmids = set(parsed_analyses.keys())
    pmid_to_fulltext, pmid_to_coord_tables = annotation_review.load_retrieval_context(retrieval_dir)

    review_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_annotation_by_mode: dict[str, dict[str, dict[str, Any]]] = {
        mode_id: {}
        for mode_id in annotation_review.OVERALL_SUMMARY_MODE_ORDER
    }
    for annotation_name in annotation_review.ACTIVE_ANNOTATION_NAMES:
        mode_results: dict[str, dict[str, Any]] = {}
        for mode_id, mode_cfg in annotation_review.EVAL_MODE_CONFIGS.items():
            docs, metrics = annotation_review.classify_documents(
                annotation_name=annotation_name,
                parsed_analyses=parsed_analyses,
                model_decisions=model_decisions,
                manual_truth=manual_truth,
                criteria=criteria,
                pmid_to_fulltext=pmid_to_fulltext,
                pmid_to_coord_tables=pmid_to_coord_tables,
                allowed_match_statuses=set(mode_cfg.get("allowed_statuses", set())),
                study_universe_pmids=study_universe_pmids,
                auto_study_pmids_by_annotation=auto_study_pmids_by_annotation,
                manual_study_pmids_by_annotation=manual_study_pmids_by_annotation,
            )
            mode_results[mode_id] = {"docs": docs, "metrics": metrics}
            if mode_id in metrics_by_annotation_by_mode:
                metrics_by_annotation_by_mode[mode_id][annotation_name] = metrics

        html = annotation_review.render_html(annotation_name, mode_results, criteria=criteria)
        output_path = review_output_dir / f"{annotation_name}.html"
        output_path.write_text(html, encoding="utf-8")

        strict_metrics = mode_results.get("accepted", {}).get("metrics", {})
        combined_metrics = mode_results.get("combined", {}).get("metrics", {})
        print(
            f"Wrote {output_path} | "
            f"strict_doc_f1={float(strict_metrics.get('f1', 0.0)):.3f} "
            f"strict_study_f1={float(strict_metrics.get('study_metrics', {}).get('f1', 0.0)):.3f} "
            f"strict_analysis_f1={float(strict_metrics.get('analysis_metrics', {}).get('f1', 0.0)):.3f} "
            f"combined_doc_f1={float(combined_metrics.get('f1', 0.0)):.3f} "
            f"combined_study_f1={float(combined_metrics.get('study_metrics', {}).get('f1', 0.0)):.3f} "
            f"combined_analysis_f1={float(combined_metrics.get('analysis_metrics', {}).get('f1', 0.0)):.3f} "
            f"missing_manual_pmids={len(strict_metrics.get('missing_manual_pmids', []))}"
        )

    overall_summary_html = annotation_review.render_overall_summary_html(metrics_by_annotation_by_mode)
    overall_summary_path = review_output_dir / "overall_submeta_summary.html"
    overall_summary_path.write_text(overall_summary_html, encoding="utf-8")
    print(f"Wrote {overall_summary_path}")


def main() -> None:
    args = parse_args()
    project_output_dir = fuzzy_matching.infer_project_output_dir(args.project_output_dir)

    match_output_dir = fuzzy_matching.resolve_output_dir(project_output_dir, args.match_output_dir)
    match_output_dir.mkdir(parents=True, exist_ok=True)

    run_fuzzy_matching_stage(args, project_output_dir=project_output_dir, match_output_dir=match_output_dir)

    match_input_dir = (args.match_input_dir or match_output_dir).expanduser().resolve()
    review_output_dir = (
        args.review_output_dir.expanduser().resolve()
        if args.review_output_dir is not None
        else (project_output_dir / "reports" / "annotation_review_reports")
    )
    run_annotation_review_stage(
        args,
        project_output_dir=project_output_dir,
        match_input_dir=match_input_dir,
        review_output_dir=review_output_dir,
    )


if __name__ == "__main__":
    main()

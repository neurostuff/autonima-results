#!/usr/bin/env python3
"""Generate per-annotation HTML review reports from precomputed fuzzy match results."""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


MANUAL_FILE_MAP = {
    "social_processing_all": "ALL-Merged.json",
    "affiliation_attachment": "Affiliation-Merged.json",
    "perception_others": "Others-Merged.json",
    "perception_self": "Self-Merged.json",
    "social_communication": "SocComm-Merged.json",
}

ANNOTATION_TO_NOTE_KEY = {
    "social_processing_all": "all_merged",
    "affiliation_attachment": "affiliation_merged",
    "perception_others": "others_merged",
    "perception_self": "self_merged",
    "social_communication": "soccomm_merged",
}

ANALYSIS_ID_RE = re.compile(r"^(?P<pmid>.+?)_analysis_(?P<index>\d+)$")


@dataclass
class Decision:
    include: bool
    reasoning: str
    analysis_id: str


def clean_text(value: str) -> str:
    return "".join(ch for ch in str(value) if ch >= " " or ch in "\n\t\r")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-output-dir",
        type=Path,
        default=None,
        help=(
            "Path to project output dir (e.g., .../annotation-only). "
            "If omitted, auto-detect prefers annotation-only under projects/social/coordinates."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated HTML reports. Defaults to project-output-dir/reports/annotation_review_reports.",
    )
    parser.add_argument(
        "--match-input-dir",
        type=Path,
        default=None,
        help="Directory containing match results JSON files. Defaults to project-output-dir/reports.",
    )
    parser.add_argument(
        "--manual-annotation-path",
        type=Path,
        default=None,
        help=(
            "Optional path to merged nimads_annotation.json used to slice match_results_overall.json "
            "into per-annotation manual truth."
        ),
    )
    return parser.parse_args()


def infer_project_output_dir(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path

    coordinates_root = Path("../autonima-results/projects/social/coordinates")
    candidates: list[Path] = []
    if coordinates_root.exists():
        for entry in coordinates_root.iterdir():
            if not entry.is_dir():
                continue
            outputs_dir = entry / "outputs"
            if not outputs_dir.exists():
                continue
            if not (outputs_dir / "annotation_results.json").exists():
                continue
            candidates.append(entry)

    pool = candidates
    if not pool:
        raise FileNotFoundError(
            "Could not infer project output dir. Pass --project-output-dir explicitly."
        )

    annotation_only = [c for c in pool if c.name == "annotation-only"]
    if annotation_only:
        selected = max(annotation_only, key=lambda p: (p / "outputs" / "annotation_results.json").stat().st_mtime)
        print(f"Auto-selected project output dir (annotation-only preferred): {selected}")
        return selected

    preferred = [
        c
        for c in pool
        if "search-all_pmids-studyann-ft" in c.name
    ]
    if preferred:
        selected = max(preferred, key=lambda p: (p / "outputs" / "annotation_results.json").stat().st_mtime)
        print(f"Auto-selected project output dir (preferred pattern): {selected}")
        return selected

    latest = max(pool, key=lambda p: (p / "outputs" / "annotation_results.json").stat().st_mtime)
    print(f"Auto-selected project output dir: {latest}")
    return latest


def resolve_dirs(project_output_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = args.output_dir or (project_output_dir / "reports" / "annotation_review_reports")
    match_input_dir = args.match_input_dir or (project_output_dir / "reports")
    return output_dir, match_input_dir


def infer_project_name(project_output_dir: Path) -> str:
    parts = list(project_output_dir.parts)
    if "projects" in parts:
        idx = parts.index("projects")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "social"


def resolve_manual_annotation_path(project_output_dir: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path

    project_name = infer_project_name(project_output_dir)
    candidates = [
        Path(f"../neurometabench/data/nimads/{project_name}/merged/nimads_annotation.json"),
        project_output_dir / "outputs" / "nimads_annotation.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def find_article_xml(retrieval_dir: Path, pmcid: str) -> Path | None:
    matches = list(retrieval_dir.glob(f"articles/**/pmcid_{pmcid}/article.xml"))
    return matches[0] if matches else None


def extract_coord_table_html(article_xml_path: Path, target_table_ids: set[str]) -> dict[str, str]:
    if not target_table_ids or not article_xml_path.exists():
        return {}
    try:
        root = ET.parse(article_xml_path).getroot()
    except ET.ParseError:
        return {}

    html_by_table_id: dict[str, str] = {}
    for element in root.iter():
        if local_name(element.tag) != "table-wrap":
            continue
        table_id = element.attrib.get("id", "")
        if table_id in target_table_ids:
            html_by_table_id[table_id] = clean_text(ET.tostring(element, encoding="unicode"))
    return html_by_table_id


def load_retrieval_context(retrieval_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    metadata_path = retrieval_dir / "metadata.csv"
    text_path = retrieval_dir / "text.csv"
    tables_path = retrieval_dir / "tables.csv"
    coordinates_path = retrieval_dir / "coordinates.csv"

    if not (metadata_path.exists() and text_path.exists() and tables_path.exists() and coordinates_path.exists()):
        return {}, {}

    metadata_rows = load_csv_rows(metadata_path)
    text_rows = load_csv_rows(text_path)
    tables_rows = load_csv_rows(tables_path)
    coordinates_rows = load_csv_rows(coordinates_path)

    pmcid_to_pmid: dict[str, str] = {}
    for row in metadata_rows:
        pmcid = clean_text(row.get("pmcid", "")).strip()
        pmid = clean_text(row.get("pmid", "")).strip()
        if pmcid and pmid:
            pmcid_to_pmid[pmcid] = pmid

    text_by_pmcid = {clean_text(r.get("pmcid", "")).strip(): r for r in text_rows}
    pmid_to_fulltext: dict[str, dict[str, str]] = {}
    for pmcid, row in text_by_pmcid.items():
        pmid = pmcid_to_pmid.get(pmcid)
        if not pmid:
            continue
        title = clean_text(row.get("title", "")).strip()
        abstract = clean_text(row.get("abstract", "")).strip()
        body = clean_text(row.get("body", "")).strip()
        if not (abstract or body):
            continue
        pmid_to_fulltext[pmid] = {
            "pmcid": pmcid,
            "title": title,
            "abstract": abstract,
            "body": body,
        }

    table_meta: dict[tuple[str, str], dict[str, str]] = {}
    for row in tables_rows:
        pmcid = clean_text(row.get("pmcid", "")).strip()
        table_id = clean_text(row.get("table_id", "")).strip()
        if pmcid and table_id:
            table_meta[(pmcid, table_id)] = row

    coord_table_ids_by_pmcid: dict[str, set[str]] = defaultdict(set)
    for row in coordinates_rows:
        pmcid = clean_text(row.get("pmcid", "")).strip()
        table_id = clean_text(row.get("table_id", "")).strip()
        if pmcid and table_id:
            coord_table_ids_by_pmcid[pmcid].add(table_id)

    pmid_to_coord_tables: dict[str, list[dict[str, str]]] = defaultdict(list)
    for pmcid, table_ids in coord_table_ids_by_pmcid.items():
        pmid = pmcid_to_pmid.get(pmcid)
        if not pmid:
            continue
        article_xml = find_article_xml(retrieval_dir, pmcid)
        table_html_by_id = extract_coord_table_html(article_xml, table_ids) if article_xml else {}
        for table_id in sorted(table_ids):
            meta = table_meta.get((pmcid, table_id), {})
            pmid_to_coord_tables[pmid].append(
                {
                    "table_id": table_id,
                    "table_label": clean_text(meta.get("table_label", "")).strip(),
                    "table_caption": clean_text(meta.get("table_caption", "")).strip(),
                    "table_html": table_html_by_id.get(table_id, ""),
                }
            )

    return pmid_to_fulltext, dict(pmid_to_coord_tables)


def load_auto_parsed_names(path: Path) -> dict[str, list[str]]:
    payload = load_json(path)
    studies = payload.get("studies", [])
    parsed: dict[str, list[str]] = {}
    for study in studies:
        pmid = str(study.get("pmid"))
        analyses = study.get("analyses", [])
        parsed[pmid] = [clean_text(a.get("name") or f"analysis_{i}") for i, a in enumerate(analyses)]
    return parsed


def load_model_decisions(path: Path) -> dict[str, dict[str, dict[int, Decision]]]:
    rows = load_json(path)
    decisions: dict[str, dict[str, dict[int, Decision]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        analysis_id = str(row.get("analysis_id", ""))
        match = ANALYSIS_ID_RE.match(analysis_id)
        if not match:
            continue
        pmid = match.group("pmid")
        idx = int(match.group("index"))
        annotation = str(row.get("annotation_name"))
        decisions[annotation][pmid][idx] = Decision(
            include=bool(row.get("include", False)),
            reasoning=clean_text(row.get("reasoning") or ""),
            analysis_id=analysis_id,
        )
    return decisions


def load_manual_annotation_membership(path: Path | None) -> dict[str, dict[str, bool]]:
    if path is None or not path.exists():
        return {}

    payload = load_json(path)
    notes = payload.get("notes", [])
    membership: dict[str, dict[str, bool]] = {}
    for row in notes:
        analysis_id = clean_text(row.get("analysis", "")).strip()
        if not analysis_id:
            continue
        note = row.get("note", {})
        if isinstance(note, dict):
            membership[analysis_id] = {str(k): bool(v) for k, v in note.items()}
    return membership


def parse_pmid_from_analysis_id(analysis_id: str) -> str | None:
    analysis_text = clean_text(analysis_id).strip()
    if not analysis_text:
        return None

    match = ANALYSIS_ID_RE.match(analysis_text)
    if match:
        return match.group("pmid")

    if "_" in analysis_text:
        return analysis_text.split("_", 1)[0].strip()

    return None


def load_study_pmid_sets_from_annotations(
    auto_annotation_path: Path | None,
    manual_annotation_path: Path | None,
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    auto_grouped: dict[str, set[str]] = {annotation: set() for annotation in MANUAL_FILE_MAP}
    manual_grouped: dict[str, set[str]] = {annotation: set() for annotation in MANUAL_FILE_MAP}
    unique_pmids_in_auto: set[str] = set()

    if auto_annotation_path is not None and auto_annotation_path.exists():
        payload = load_json(auto_annotation_path)
        for note in payload.get("notes", []):
            analysis_id = clean_text(note.get("analysis", "")).strip()
            match = ANALYSIS_ID_RE.match(analysis_id)
            if not match:
                continue
            pmid = match.group("pmid")
            unique_pmids_in_auto.add(pmid)
            note_obj = note.get("note", {})
            if not isinstance(note_obj, dict):
                continue
            for annotation in MANUAL_FILE_MAP:
                if bool(note_obj.get(annotation, False)):
                    auto_grouped[annotation].add(pmid)

    if manual_annotation_path is not None and manual_annotation_path.exists():
        payload = load_json(manual_annotation_path)
        for note in payload.get("notes", []):
            pmid = parse_pmid_from_analysis_id(str(note.get("analysis", "")))
            if not pmid:
                continue
            if unique_pmids_in_auto and pmid not in unique_pmids_in_auto:
                continue
            note_obj = note.get("note", {})
            if not isinstance(note_obj, dict):
                continue
            for annotation in MANUAL_FILE_MAP:
                manual_key = ANNOTATION_TO_NOTE_KEY.get(annotation, annotation)
                included = bool(note_obj.get(manual_key, False))
                if not included and manual_key != annotation:
                    included = bool(note_obj.get(annotation, False))
                if included:
                    manual_grouped[annotation].add(pmid)

    return unique_pmids_in_auto, auto_grouped, manual_grouped


def load_match_results_by_annotation(match_input_dir: Path) -> tuple[dict[str, Any], bool]:
    results: dict[str, Any] = {}
    missing_per_annotation: list[str] = []
    for annotation_name in MANUAL_FILE_MAP:
        path = match_input_dir / f"match_results_{annotation_name}.json"
        if not path.exists():
            missing_per_annotation.append(annotation_name)
            continue
        results[annotation_name] = load_json(path)
    if not missing_per_annotation:
        return results, False

    overall_path = match_input_dir / "match_results_overall.json"
    if not overall_path.exists():
        alt_overall_path = match_input_dir.parent / "match_results_overall.json"
        if alt_overall_path.exists():
            overall_path = alt_overall_path
    if overall_path.exists():
        overall = load_json(overall_path)
        for annotation_name in MANUAL_FILE_MAP:
            results[annotation_name] = overall
        print(
            "Using match_results_overall.json fallback for per-annotation reports. "
            "Per-annotation truth will be sliced using nimads_annotation notes when available."
        )
        if overall_path.parent != match_input_dir:
            print(f"Loaded overall match file from alternate path: {overall_path}")
        return results, True

    missing_list = ", ".join(missing_per_annotation)
    raise FileNotFoundError(
        f"Missing match result files ({missing_list}) under {match_input_dir}. "
        "Expected either match_results_<annotation>.json files or match_results_overall.json. "
        "Run run_fuzzy_analysis_matching.py first."
    )


def build_manual_truth_from_match_results(
    match_results_by_annotation: dict[str, Any],
    overall_fallback: bool,
    manual_annotation_membership: dict[str, dict[str, bool]],
) -> dict[str, dict[str, dict[str, Any]]]:
    manual_truth: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for annotation_name, match_results in match_results_by_annotation.items():
        target_note_key = ANNOTATION_TO_NOTE_KEY.get(annotation_name)
        for pmid, pmid_result in match_results.get("pmids", {}).items():
            manual_analyses = list(pmid_result.get("manual_analyses", []))
            if overall_fallback and manual_annotation_membership:
                filtered_analyses: list[dict[str, Any]] = []
                for entry in manual_analyses:
                    analysis_id = clean_text(entry.get("manual_analysis_id", "")).strip()
                    auto_analysis_id = clean_text(entry.get("best_auto_analysis_id", "")).strip()
                    candidate_keys = [annotation_name]
                    if target_note_key and target_note_key != annotation_name:
                        candidate_keys.insert(0, target_note_key)

                    include_for_annotation = False
                    note_manual = manual_annotation_membership.get(analysis_id, {})
                    if note_manual:
                        include_for_annotation = any(bool(note_manual.get(k, False)) for k in candidate_keys)
                    if not include_for_annotation and auto_analysis_id:
                        note_auto = manual_annotation_membership.get(auto_analysis_id, {})
                        if note_auto:
                            include_for_annotation = any(bool(note_auto.get(k, False)) for k in candidate_keys)

                    if include_for_annotation:
                        filtered_analyses.append(entry)
                manual_analyses = filtered_analyses

            accepted_indices = {
                int(entry["best_auto_index"])
                for entry in manual_analyses
                if entry.get("best_auto_index") is not None and entry.get("match_status") == "accepted"
            }

            status_counts = {
                "accepted": sum(1 for entry in manual_analyses if entry.get("match_status") == "accepted"),
                "uncertain": sum(1 for entry in manual_analyses if entry.get("match_status") == "uncertain"),
                "unmatched": sum(1 for entry in manual_analyses if entry.get("match_status") == "unmatched"),
            }
            if manual_analyses:
                status_counts["mean_combined_score"] = (
                    sum(float(entry.get("combined_score", 0.0)) for entry in manual_analyses)
                    / len(manual_analyses)
                )
            else:
                status_counts["mean_combined_score"] = 0.0

            manual_truth[annotation_name][pmid] = {
                "true_indices": accepted_indices,
                "manual_names": [entry.get("manual_name", "") for entry in manual_analyses],
                "unmatched_manual_names": [
                    entry.get("manual_name", "")
                    for entry in manual_analyses
                    if entry.get("match_status") == "unmatched"
                ],
                "match_diagnostics": manual_analyses,
                "status_counts": {
                    "accepted": int(status_counts.get("accepted", 0)),
                    "uncertain": int(status_counts.get("uncertain", 0)),
                    "unmatched": int(status_counts.get("unmatched", 0)),
                    "mean_combined_score": float(status_counts.get("mean_combined_score", 0.0)),
                },
                "manual_missing_in_auto": bool(pmid_result.get("manual_missing_in_auto", False))
                if not overall_fallback
                else False,
            }

    return manual_truth


def make_document_row(
    pmid: str,
    annotation_name: str,
    parsed_names: list[str],
    decisions_by_idx: dict[int, Decision],
    true_indices: set[int],
    manual_names: list[str],
    unmatched_manual_names: list[str],
    bucket: str,
    fulltext_entry: dict[str, str] | None,
    coord_tables: list[dict[str, str]],
    match_diagnostics: list[dict[str, Any]],
    status_counts: dict[str, Any],
    manual_missing_in_auto: bool,
) -> dict[str, Any]:
    pred_indices = {idx for idx, decision in decisions_by_idx.items() if decision.include}
    correct_indices = pred_indices & true_indices
    matched_auto_indices = {
        int(entry["best_auto_index"])
        for entry in match_diagnostics
        if entry.get("best_auto_index") is not None
    }
    match_status_by_auto_idx = {
        int(entry["best_auto_index"]): str(entry.get("match_status", ""))
        for entry in match_diagnostics
        if entry.get("best_auto_index") is not None
    }

    max_idx = len(parsed_names) - 1
    if decisions_by_idx:
        max_idx = max(max_idx, max(decisions_by_idx.keys()))

    analysis_rows: list[dict[str, Any]] = []
    for idx in range(max_idx + 1):
        name = parsed_names[idx] if idx < len(parsed_names) else f"analysis_{idx}"
        decision = decisions_by_idx.get(idx)
        model_include = None if decision is None else decision.include
        matched_for_review = idx in matched_auto_indices
        match_status_for_idx = match_status_by_auto_idx.get(idx, "")
        manual_include = idx in true_indices

        if matched_for_review and match_status_for_idx == "unmatched":
            confusion_label = "UNMATCHED"
            confusion_class = "confusion-na"
        elif matched_for_review and model_include is not None:
            if model_include and manual_include:
                confusion_label = "TP"
                confusion_class = "confusion-good"
            elif model_include and not manual_include:
                confusion_label = "FP"
                confusion_class = "confusion-bad"
            elif (not model_include) and manual_include:
                confusion_label = "FN"
                confusion_class = "confusion-bad"
            else:
                confusion_label = "TN"
                confusion_class = "confusion-good"
        else:
            confusion_label = "-"
            confusion_class = "confusion-na"

        if model_include is True:
            decision_icon = "+"
            decision_class = "decision-include"
        elif model_include is False:
            decision_icon = "-"
            decision_class = "decision-exclude"
        else:
            decision_icon = "?"
            decision_class = "decision-none"

        analysis_rows.append(
            {
                "analysis_id": f"{pmid}_analysis_{idx}",
                "parsed_name": name,
                "model_include": model_include,
                "model_decision_icon": decision_icon,
                "model_decision_class": decision_class,
                "confusion_label": confusion_label,
                "confusion_class": confusion_class,
                "matched_for_review": matched_for_review,
                "reasoning": "" if decision is None else decision.reasoning,
                "manual_include": manual_include,
                "correct": idx in correct_indices,
            }
        )

    return {
        "pmid": pmid,
        "annotation_name": annotation_name,
        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "bucket": bucket,
        "pred_indices": sorted(pred_indices),
        "true_indices": sorted(true_indices),
        "correct_indices": sorted(correct_indices),
        "manual_names": manual_names,
        "unmatched_manual_names": unmatched_manual_names,
        "analysis_rows": analysis_rows,
        "fulltext": fulltext_entry,
        "coord_tables": coord_tables,
        "match_diagnostics": match_diagnostics,
        "status_counts": status_counts,
        "manual_missing_in_auto": manual_missing_in_auto,
    }


def compute_prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def classify_documents(
    annotation_name: str,
    parsed_analyses: dict[str, list[str]],
    model_decisions: dict[str, dict[str, dict[int, Decision]]],
    manual_truth: dict[str, dict[str, dict[str, Any]]],
    pmid_to_fulltext: dict[str, dict[str, str]],
    pmid_to_coord_tables: dict[str, list[dict[str, str]]],
    study_universe_pmids: set[str] | None = None,
    auto_study_pmids_by_annotation: dict[str, set[str]] | None = None,
    manual_study_pmids_by_annotation: dict[str, set[str]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    docs = {"Correct": [], "False Positive": [], "False Negative": []}
    ann_decisions = model_decisions.get(annotation_name, {})
    ann_truth = manual_truth.get(annotation_name, {})
    run_pmids = set(parsed_analyses.keys())
    doc_overlap_pmids = run_pmids & set(ann_truth.keys())
    pmids = doc_overlap_pmids

    for pmid in sorted(pmids, key=lambda x: (len(x), x)):
        parsed_names = parsed_analyses.get(pmid, [])
        decisions_by_idx = ann_decisions.get(pmid, {})
        truth_entry = ann_truth.get(
            pmid,
            {
                "true_indices": set(),
                "manual_names": [],
                "unmatched_manual_names": [],
                "match_diagnostics": [],
                "status_counts": {"accepted": 0, "uncertain": 0, "unmatched": 0, "mean_combined_score": 0.0},
                "manual_missing_in_auto": False,
            },
        )

        pred_indices = {idx for idx, decision in decisions_by_idx.items() if decision.include}
        true_indices = set(truth_entry["true_indices"])
        correct_indices = pred_indices & true_indices

        if correct_indices:
            bucket = "Correct"
        elif pred_indices and not true_indices:
            bucket = "False Positive"
        elif true_indices and not correct_indices:
            bucket = "False Negative"
        else:
            continue

        docs[bucket].append(
            make_document_row(
                pmid=pmid,
                annotation_name=annotation_name,
                parsed_names=parsed_names,
                decisions_by_idx=decisions_by_idx,
                true_indices=true_indices,
                manual_names=truth_entry["manual_names"],
                unmatched_manual_names=truth_entry["unmatched_manual_names"],
                bucket=bucket,
                fulltext_entry=pmid_to_fulltext.get(pmid),
                coord_tables=pmid_to_coord_tables.get(pmid, []),
                match_diagnostics=truth_entry.get("match_diagnostics", []),
                status_counts=truth_entry.get("status_counts", {}),
                manual_missing_in_auto=bool(truth_entry.get("manual_missing_in_auto", False)),
            )
        )

    document_metrics = compute_prf(
        tp=len(docs["Correct"]),
        fp=len(docs["False Positive"]),
        fn=len(docs["False Negative"]),
    )

    if (
        study_universe_pmids is not None
        and auto_study_pmids_by_annotation is not None
        and manual_study_pmids_by_annotation is not None
    ):
        study_universe = set(study_universe_pmids)
        predicted_study_set = set(auto_study_pmids_by_annotation.get(annotation_name, set())) & study_universe
        manual_study_set = set(manual_study_pmids_by_annotation.get(annotation_name, set())) & study_universe
    else:
        study_universe = set(doc_overlap_pmids)
        manual_study_set = {
            pmid
            for pmid in study_universe
            if ann_truth.get(pmid, {}).get("manual_names")
        }
        predicted_study_set = {
            pmid
            for pmid in study_universe
            if any(decision.include for decision in ann_decisions.get(pmid, {}).values())
        }

    study_tp = len(predicted_study_set & manual_study_set)
    study_fp = len(predicted_study_set - manual_study_set)
    study_fn = len(manual_study_set - predicted_study_set)
    study_tn = max(0, len(study_universe) - study_tp - study_fp - study_fn)
    study_metrics = compute_prf(tp=study_tp, fp=study_fp, fn=study_fn)
    study_metrics["tn"] = int(study_tn)
    study_metrics["accuracy"] = (
        float((study_tp + study_tn) / len(study_universe))
        if study_universe
        else 0.0
    )
    study_metrics["manual_studies"] = len(manual_study_set)
    study_metrics["predicted_studies"] = len(predicted_study_set)
    study_metrics["run_pmids"] = len(run_pmids)
    study_metrics["overlap_pmids"] = len(study_universe)

    # Analysis-level metrics are restricted to the set of MANUAL analyses that have
    # an assigned auto analysis index (best_auto_index is not None). This avoids
    # counting unmatched auto analyses as FP in the "accepted matches only" row.
    analysis_tp = 0
    analysis_fp = 0
    analysis_fn = 0
    analysis_tn = 0
    matched_manual_universe = 0
    manual_accepted_matched = 0
    predicted_positive_on_matched = 0

    for pmid in doc_overlap_pmids:
        decisions_for_pmid = ann_decisions.get(pmid, {})
        match_rows = ann_truth.get(pmid, {}).get("match_diagnostics", [])
        for entry in match_rows:
            best_auto_index = entry.get("best_auto_index")
            if best_auto_index is None:
                continue

            matched_manual_universe += 1
            idx_int = int(best_auto_index)
            decision = decisions_for_pmid.get(idx_int)
            pred_include = bool(decision.include) if decision is not None else False
            true_include = str(entry.get("match_status", "")) == "accepted"

            if true_include:
                manual_accepted_matched += 1
            if pred_include:
                predicted_positive_on_matched += 1

            if pred_include and true_include:
                analysis_tp += 1
            elif pred_include and not true_include:
                analysis_fp += 1
            elif (not pred_include) and true_include:
                analysis_fn += 1
            else:
                analysis_tn += 1

    analysis_metrics = compute_prf(tp=analysis_tp, fp=analysis_fp, fn=analysis_fn)
    analysis_metrics["tn"] = int(analysis_tn)
    analysis_metrics["accuracy"] = (
        float((analysis_tp + analysis_tn) / matched_manual_universe)
        if matched_manual_universe
        else 0.0
    )
    analysis_metrics["manual_accepted_analyses"] = int(manual_accepted_matched)
    analysis_metrics["predicted_analyses"] = int(predicted_positive_on_matched)
    analysis_metrics["analysis_universe"] = int(matched_manual_universe)

    bucket_match_counts: dict[str, dict[str, int]] = {}
    for bucket, bucket_docs in docs.items():
        counts = defaultdict(int)
        for doc in bucket_docs:
            c = doc.get("status_counts", {})
            counts["accepted"] += int(c.get("accepted", 0))
            counts["uncertain"] += int(c.get("uncertain", 0))
            counts["unmatched"] += int(c.get("unmatched", 0))
        bucket_match_counts[bucket] = {
            "accepted": int(counts["accepted"]),
            "uncertain": int(counts["uncertain"]),
            "unmatched": int(counts["unmatched"]),
        }

    missing_manual_pmids = sorted(
        [pmid for pmid, entry in ann_truth.items() if entry.get("manual_missing_in_auto")],
        key=lambda x: (len(x), x),
    )

    metrics: dict[str, Any] = {
        "tp": int(document_metrics["tp"]),
        "fp": int(document_metrics["fp"]),
        "fn": int(document_metrics["fn"]),
        "precision": float(document_metrics["precision"]),
        "recall": float(document_metrics["recall"]),
        "f1": float(document_metrics["f1"]),
        "document_metrics": document_metrics,
        "study_metrics": study_metrics,
        "analysis_metrics": analysis_metrics,
        "bucket_match_counts": bucket_match_counts,
        "missing_manual_pmids": missing_manual_pmids,
    }
    return docs, metrics


def render_match_diagnostics(match_rows: list[dict[str, Any]]) -> str:
    if not match_rows:
        return "<p>No manual-to-auto match diagnostics for this document.</p>"

    rows_html = []
    for row in match_rows:
        reasons = ", ".join(row.get("reason_codes", []))
        rows_html.append(
            "<tr>"
            f"<td>{escape(str(row.get('manual_analysis_id', '')))}</td>"
            f"<td>{escape(str(row.get('manual_name', '')))}</td>"
            f"<td>{escape(str(row.get('best_auto_analysis_id') or ''))}</td>"
            f"<td>{escape(str(row.get('best_auto_name') or ''))}</td>"
            f"<td>{float(row.get('name_score', 0.0)):.3f}</td>"
            f"<td>{float(row.get('coord_score', 0.0)):.3f}</td>"
            f"<td>{float(row.get('combined_score', 0.0)):.3f}</td>"
            f"<td>{escape(str(row.get('match_status', '')))}</td>"
            f"<td>{escape(reasons)}</td>"
            "</tr>"
        )

    return (
        "<div class=\"table-wrap\">"
        "<table>"
        "<thead><tr>"
        "<th>Manual ID</th>"
        "<th>Manual Name</th>"
        "<th>Matched Auto ID</th>"
        "<th>Matched Auto Name</th>"
        "<th>Name Score</th>"
        "<th>Coord Score</th>"
        "<th>Combined</th>"
        "<th>Status</th>"
        "<th>Reason Codes</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
        "</div>"
    )


def render_doc_card(doc: dict[str, Any]) -> str:
    status_counts = doc.get("status_counts", {})
    status_meta = (
        f"accepted={int(status_counts.get('accepted', 0))}, "
        f"uncertain={int(status_counts.get('uncertain', 0))}, "
        f"unmatched={int(status_counts.get('unmatched', 0))}"
    )
    meta = (
        f"Pred included: {len(doc['pred_indices'])} | "
        f"Manual included (accepted matches only): {len(doc['true_indices'])} | "
        f"Correct overlaps: {len(doc['correct_indices'])} | "
        f"Match statuses: {status_meta}"
    )

    missing_manual_msg = ""
    if doc.get("manual_missing_in_auto"):
        missing_manual_msg = "<p><strong>Manual study exists but PMID is missing from auto parsing outputs.</strong></p>"

    unmatched_html = ""
    if doc["unmatched_manual_names"]:
        joined = ", ".join(escape(x) for x in doc["unmatched_manual_names"])
        unmatched_html = f"<p><strong>Unmatched manual analyses:</strong> {joined}</p>"

    rows_html = []
    for row in doc["analysis_rows"]:
        label_parts = []
        if row["manual_include"]:
            label_parts.append("manual+ (accepted)")
        if row["correct"]:
            label_parts.append("correct")
        label_text = ", ".join(label_parts)
        rows_html.append(
            "<tr>"
            f"<td>{escape(row['analysis_id'])}</td>"
            f"<td>{escape(row['parsed_name'])}</td>"
            f"<td class=\"decision-cell\"><span class=\"decision-pill {escape(row['model_decision_class'])}\">{escape(row['model_decision_icon'])}</span></td>"
            f"<td class=\"confusion-cell\"><span class=\"confusion-pill {escape(row['confusion_class'])}\">{escape(row['confusion_label'])}</span></td>"
            f"<td>{escape(label_text)}</td>"
            f"<td>{escape(row['reasoning'])}</td>"
            "</tr>"
        )

    rows = "\n".join(rows_html)

    fulltext_html = ""
    fulltext = doc.get("fulltext")
    if fulltext:
        title_html = f"<p><strong>Title:</strong> {escape(fulltext.get('title', ''))}</p>" if fulltext.get("title") else ""
        abstract_html = ""
        if fulltext.get("abstract"):
            abstract_html = (
                "<details><summary>Abstract</summary>"
                f"<pre class=\"paper-text\">{escape(fulltext['abstract'])}</pre>"
                "</details>"
            )
        body_html = ""
        if fulltext.get("body"):
            body_html = (
                "<details><summary>Body</summary>"
                f"<pre class=\"paper-text\">{escape(fulltext['body'])}</pre>"
                "</details>"
            )
        fulltext_html = (
            "<details class=\"inner-accordion\">"
            f"<summary>PMC full text available (PMCID {escape(fulltext.get('pmcid', ''))})</summary>"
            f"{title_html}{abstract_html}{body_html}"
            "</details>"
        )

    tables_html = ""
    coord_tables = doc.get("coord_tables", [])
    if coord_tables:
        table_blocks = []
        for t in coord_tables:
            caption = f" - {escape(t['table_caption'])}" if t.get("table_caption") else ""
            label = escape(t.get("table_label") or t["table_id"])
            rendered_table = t.get("table_html") or "<p>Table HTML unavailable.</p>"
            table_blocks.append(
                "<details class=\"inner-accordion\">"
                f"<summary>{label} ({escape(t['table_id'])}){caption}</summary>"
                f"<div class=\"table-html\">{rendered_table}</div>"
                "</details>"
            )
        tables_html = (
            "<details class=\"inner-accordion\">"
            f"<summary>Coordinate-relevant source tables ({len(coord_tables)})</summary>"
            + "".join(table_blocks)
            + "</details>"
        )

    match_diag_html = render_match_diagnostics(doc.get("match_diagnostics", []))

    return f"""
<details class="doc-card">
  <summary><strong>PMID {escape(doc['pmid'])}</strong> | {escape(meta)}</summary>
  <p><a href="{escape(doc['pubmed_url'])}" target="_blank" rel="noopener noreferrer">PubMed full text page</a></p>
  {missing_manual_msg}
  {unmatched_html}
  <details class="inner-accordion" open>
    <summary>Parsed analyses and annotation reasoning</summary>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Analysis ID</th>
            <th>Parsed Analysis Name</th>
            <th>Model Decision</th>
            <th>Matched Outcome</th>
            <th>Tags</th>
            <th>Model Reasoning</th>
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
  </details>
  <details class="inner-accordion" open>
    <summary>Manual-to-Auto Match Diagnostics</summary>
    {match_diag_html}
  </details>
  {fulltext_html}
  {tables_html}
</details>
"""


def render_html(annotation_name: str, docs: dict[str, list[dict[str, Any]]], metrics: dict[str, Any]) -> str:
    bucket_ids = {
        "Correct": "bucket-correct",
        "False Positive": "bucket-false-positive",
        "False Negative": "bucket-false-negative",
    }

    sections = []
    for bucket in ["Correct", "False Positive", "False Negative"]:
        cards = sorted(
            docs[bucket],
            key=lambda d: (
                0 if d.get("fulltext") else 1,
                len(str(d.get("pmid", ""))),
                str(d.get("pmid", "")),
            ),
        )
        if cards:
            body = "\n".join(render_doc_card(card) for card in cards)
        else:
            body = "<p>No documents in this bucket.</p>"

        bm = metrics["bucket_match_counts"].get(bucket, {"accepted": 0, "uncertain": 0, "unmatched": 0})
        bucket_summary = (
            f"<p><strong>Match status totals:</strong> accepted={bm['accepted']} | "
            f"uncertain={bm['uncertain']} | unmatched={bm['unmatched']}</p>"
        )

        open_attr = " open" if bucket != "Correct" else ""
        sections.append(
            "<section id=\"{sid}\">"
            "<details class=\"bucket\"{open_attr}>"
            "<summary><h2>{bucket} ({count})</h2></summary>"
            "{bucket_summary}"
            "{body}"
            "</details>"
            "</section>".format(
                sid=bucket_ids[bucket],
                open_attr=open_attr,
                bucket=bucket,
                count=len(cards),
                bucket_summary=bucket_summary,
                body=body,
            )
        )

    document_metrics = metrics.get("document_metrics", {})
    study_metrics = metrics.get("study_metrics", {})
    analysis_metrics = metrics.get("analysis_metrics", {})

    precision_str = f"{float(document_metrics.get('precision', metrics.get('precision', 0.0))):.3f}"
    recall_str = f"{float(document_metrics.get('recall', metrics.get('recall', 0.0))):.3f}"
    f1_str = f"{float(document_metrics.get('f1', metrics.get('f1', 0.0))):.3f}"

    study_precision_str = f"{float(study_metrics.get('precision', 0.0)):.3f}"
    study_recall_str = f"{float(study_metrics.get('recall', 0.0)):.3f}"
    study_f1_str = f"{float(study_metrics.get('f1', 0.0)):.3f}"

    analysis_precision_str = f"{float(analysis_metrics.get('precision', 0.0)):.3f}"
    analysis_recall_str = f"{float(analysis_metrics.get('recall', 0.0)):.3f}"
    analysis_f1_str = f"{float(analysis_metrics.get('f1', 0.0)):.3f}"

    missing_pmids = metrics.get("missing_manual_pmids", [])
    missing_html = ""
    if missing_pmids:
        missing_items = "".join(
            f"<li><a href=\"https://pubmed.ncbi.nlm.nih.gov/{escape(pmid)}/\" target=\"_blank\" rel=\"noopener noreferrer\">PMID {escape(pmid)}</a></li>"
            for pmid in missing_pmids
        )
        missing_html = (
            "<section id=\"missing-manual\">"
            "<details class=\"bucket\" open>"
            f"<summary><h2>Manual PMIDs Missing In Auto Parsing ({len(missing_pmids)})</h2></summary>"
            "<p>These studies exist in manual NiMADS but were not found in auto parsed outputs for this project.</p>"
            f"<ul>{missing_items}</ul>"
            "</details>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(annotation_name)} review report</title>
  <style>
    :root {{
      --bg: #f7f6f2;
      --panel: #ffffff;
      --ink: #1d2730;
      --line: #d8dde3;
    }}
    body {{ margin: 0; padding: 1.25rem; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    header {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }}
    .top-nav {{ position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; gap: 0.5rem; background: #eef3f2; border: 1px solid var(--line); border-radius: 10px; padding: 0.6rem; margin-bottom: 1rem; }}
    .top-nav a {{ display: inline-block; padding: 0.35rem 0.6rem; border: 1px solid var(--line); border-radius: 999px; background: #fff; text-decoration: none; font-size: 0.9rem; color: #0e4f85; }}
    section {{ margin-bottom: 1rem; }}
    .bucket > summary, .doc-card > summary, .inner-accordion > summary {{ cursor: pointer; }}
    .doc-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 0.85rem; margin-bottom: 0.85rem; }}
    .table-wrap, .table-html {{ overflow-x: auto; }}
    .inner-accordion {{ margin-top: 0.6rem; border-top: 1px dashed var(--line); padding-top: 0.4rem; }}
    .paper-text {{ white-space: pre-wrap; max-height: 26rem; overflow-y: auto; background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; padding: 0.6rem; font-size: 0.88rem; line-height: 1.35; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border: 1px solid var(--line); padding: 0.45rem; vertical-align: top; text-align: left; }}
    th {{ background: #edf2f5; }}
    .decision-cell, .confusion-cell {{ text-align: center; vertical-align: middle; }}
    .decision-pill, .confusion-pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 1.55rem;
      padding: 0.12rem 0.45rem;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.82rem;
      border: 1px solid transparent;
    }}
    .decision-include {{ background: #e9f8ef; color: #1f7a3d; border-color: #b7e4c6; }}
    .decision-exclude {{ background: #fdecec; color: #9b1c1c; border-color: #f6caca; }}
    .decision-none {{ background: #f2f4f7; color: #5b6775; border-color: #dde3ea; }}
    .confusion-good {{ background: #e9f8ef; color: #166534; border-color: #b7e4c6; }}
    .confusion-bad {{ background: #fdecec; color: #991b1b; border-color: #f6caca; }}
    .confusion-na {{ background: #f2f4f7; color: #5b6775; border-color: #dde3ea; }}
    a {{ color: #0e4f85; }}
  </style>
</head>
<body>
  <header>
    <a id="top"></a>
    <h1>{escape(annotation_name)} report</h1>
    <p>Manual benchmark is sliced to the auto PMID universe from <code>outputs/nimads_annotation.json</code>. Analysis-level row is evaluated only on manual analyses with an assigned auto match (truth positives are accepted fuzzy matches only).</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Level</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Manual Positives</th>
            <th>Predicted Positives</th>
            <th>Universe</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Document bucket overlap</td>
            <td>{int(document_metrics.get('tp', metrics.get('tp', 0)))}</td>
            <td>{int(document_metrics.get('fp', metrics.get('fp', 0)))}</td>
            <td>{int(document_metrics.get('fn', metrics.get('fn', 0)))}</td>
            <td>{precision_str}</td>
            <td>{recall_str}</td>
            <td>{f1_str}</td>
            <td>{int(document_metrics.get('tp', metrics.get('tp', 0))) + int(document_metrics.get('fn', metrics.get('fn', 0)))}</td>
            <td>{int(document_metrics.get('tp', metrics.get('tp', 0))) + int(document_metrics.get('fp', metrics.get('fp', 0)))}</td>
            <td>{int(document_metrics.get('tp', metrics.get('tp', 0))) + int(document_metrics.get('fp', metrics.get('fp', 0))) + int(document_metrics.get('fn', metrics.get('fn', 0)))}</td>
          </tr>
          <tr>
            <td>Study inclusion</td>
            <td>{int(study_metrics.get('tp', 0))}</td>
            <td>{int(study_metrics.get('fp', 0))}</td>
            <td>{int(study_metrics.get('fn', 0))}</td>
            <td>{study_precision_str}</td>
            <td>{study_recall_str}</td>
            <td>{study_f1_str}</td>
            <td>{int(study_metrics.get('manual_studies', 0))}</td>
            <td>{int(study_metrics.get('predicted_studies', 0))}</td>
            <td>{int(study_metrics.get('overlap_pmids', 0))}</td>
          </tr>
          <tr>
            <td>Analysis inclusion (matched manual universe; accepted=positive)</td>
            <td>{int(analysis_metrics.get('tp', 0))}</td>
            <td>{int(analysis_metrics.get('fp', 0))}</td>
            <td>{int(analysis_metrics.get('fn', 0))}</td>
            <td>{analysis_precision_str}</td>
            <td>{analysis_recall_str}</td>
            <td>{analysis_f1_str}</td>
            <td>{int(analysis_metrics.get('manual_accepted_analyses', 0))}</td>
            <td>{int(analysis_metrics.get('predicted_analyses', 0))}</td>
            <td>{int(analysis_metrics.get('analysis_universe', 0))}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </header>
  <nav class="top-nav">
    <a href="#bucket-correct">Correct ({len(docs['Correct'])})</a>
    <a href="#bucket-false-positive">False Positive ({len(docs['False Positive'])})</a>
    <a href="#bucket-false-negative">False Negative ({len(docs['False Negative'])})</a>
    <a href="#missing-manual">Missing PMIDs ({len(missing_pmids)})</a>
    <a href="#top">Top</a>
  </nav>
  {''.join(sections)}
  {missing_html}
</body>
</html>
"""


def render_overall_summary_html(metrics_by_annotation: dict[str, dict[str, Any]]) -> str:
    rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    for annotation_name in MANUAL_FILE_MAP:
        metrics = metrics_by_annotation.get(annotation_name, {})
        study = metrics.get("study_metrics", {})
        analysis = metrics.get("analysis_metrics", {})
        tp = int(study.get("tp", 0))
        fp = int(study.get("fp", 0))
        fn = int(study.get("fn", 0))
        tn = int(study.get("tn", 0))
        precision = float(study.get("precision", 0.0))
        recall = float(study.get("recall", 0.0))
        f1 = float(study.get("f1", 0.0))
        accuracy = float(study.get("accuracy", 0.0))
        overlap_pmids = int(study.get("overlap_pmids", 0))
        manual_studies = int(study.get("manual_studies", 0))
        predicted_studies = int(study.get("predicted_studies", 0))

        rows.append(
            {
                "annotation": annotation_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "overlap_pmids": overlap_pmids,
                "manual_studies": manual_studies,
                "predicted_studies": predicted_studies,
            }
        )

        analysis_rows.append(
            {
                "annotation": annotation_name,
                "tp": int(analysis.get("tp", 0)),
                "fp": int(analysis.get("fp", 0)),
                "fn": int(analysis.get("fn", 0)),
                "tn": int(analysis.get("tn", 0)),
                "precision": float(analysis.get("precision", 0.0)),
                "recall": float(analysis.get("recall", 0.0)),
                "f1": float(analysis.get("f1", 0.0)),
                "accuracy": float(analysis.get("accuracy", 0.0)),
                "manual_accepted_analyses": int(analysis.get("manual_accepted_analyses", 0)),
                "predicted_analyses": int(analysis.get("predicted_analyses", 0)),
                "analysis_universe": int(analysis.get("analysis_universe", 0)),
            }
        )

    if rows:
        macro_precision = sum(r["precision"] for r in rows) / len(rows)
        macro_recall = sum(r["recall"] for r in rows) / len(rows)
        macro_f1 = sum(r["f1"] for r in rows) / len(rows)
        macro_accuracy = sum(r["accuracy"] for r in rows) / len(rows)
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        macro_accuracy = 0.0

    micro_tp = sum(r["tp"] for r in rows)
    micro_fp = sum(r["fp"] for r in rows)
    micro_fn = sum(r["fn"] for r in rows)
    micro_tn = sum(r["tn"] for r in rows)
    micro_prf = compute_prf(tp=micro_tp, fp=micro_fp, fn=micro_fn)
    micro_total = micro_tp + micro_fp + micro_fn + micro_tn
    micro_accuracy = (micro_tp + micro_tn) / micro_total if micro_total else 0.0

    analysis_micro_tp = sum(r["tp"] for r in analysis_rows)
    analysis_micro_fp = sum(r["fp"] for r in analysis_rows)
    analysis_micro_fn = sum(r["fn"] for r in analysis_rows)
    analysis_micro_tn = sum(r["tn"] for r in analysis_rows)
    analysis_micro_prf = compute_prf(tp=analysis_micro_tp, fp=analysis_micro_fp, fn=analysis_micro_fn)
    analysis_micro_total = analysis_micro_tp + analysis_micro_fp + analysis_micro_fn + analysis_micro_tn
    analysis_micro_accuracy = (
        (analysis_micro_tp + analysis_micro_tn) / analysis_micro_total
        if analysis_micro_total
        else 0.0
    )

    def render_metric_bars(precision: float, recall: float, f1: float) -> str:
        return (
            "<div class=\"metric-bars\">"
            f"<div class=\"metric-row\"><span class=\"metric-label\">P</span><div class=\"bar\"><div class=\"fill fill-p\" style=\"width:{max(0.0, min(100.0, precision * 100.0)):.1f}%\"></div></div><span class=\"metric-val\">{precision:.3f}</span></div>"
            f"<div class=\"metric-row\"><span class=\"metric-label\">R</span><div class=\"bar\"><div class=\"fill fill-r\" style=\"width:{max(0.0, min(100.0, recall * 100.0)):.1f}%\"></div></div><span class=\"metric-val\">{recall:.3f}</span></div>"
            f"<div class=\"metric-row\"><span class=\"metric-label\">F1</span><div class=\"bar\"><div class=\"fill fill-f1\" style=\"width:{max(0.0, min(100.0, f1 * 100.0)):.1f}%\"></div></div><span class=\"metric-val\">{f1:.3f}</span></div>"
            "</div>"
        )

    def render_confusion_plot(tp: int, fp: int, fn: int, tn: int) -> str:
        total = tp + fp + fn + tn
        if total <= 0:
            return "<div class=\"confusion-plot empty\">No overlap PMIDs</div>"

        tp_w = (tp / total) * 100.0
        fp_w = (fp / total) * 100.0
        fn_w = (fn / total) * 100.0
        tn_w = max(0.0, 100.0 - tp_w - fp_w - fn_w)

        return (
            "<div class=\"confusion-plot\">"
            "<div class=\"stack-bar\">"
            f"<span class=\"seg seg-tp\" style=\"width:{tp_w:.3f}%\" title=\"TP={tp}\"></span>"
            f"<span class=\"seg seg-fp\" style=\"width:{fp_w:.3f}%\" title=\"FP={fp}\"></span>"
            f"<span class=\"seg seg-fn\" style=\"width:{fn_w:.3f}%\" title=\"FN={fn}\"></span>"
            f"<span class=\"seg seg-tn\" style=\"width:{tn_w:.3f}%\" title=\"TN={tn}\"></span>"
            "</div>"
            "<div class=\"legend\">"
            "<span class=\"lg lg-tp\">TP</span>"
            "<span class=\"lg lg-fp\">FP</span>"
            "<span class=\"lg lg-fn\">FN</span>"
            "<span class=\"lg lg-tn\">TN</span>"
            "</div>"
            "</div>"
        )

    row_html: list[str] = []
    for row in rows:
        row_html.append(
            "<tr>"
            f"<td>{escape(row['annotation'])}</td>"
            f"<td>{row['overlap_pmids']}</td>"
            f"<td>{row['manual_studies']}</td>"
            f"<td>{row['predicted_studies']}</td>"
            f"<td>{row['tp']}</td>"
            f"<td>{row['fp']}</td>"
            f"<td>{row['fn']}</td>"
            f"<td>{row['tn']}</td>"
            f"<td>{row['precision']:.3f}</td>"
            f"<td>{row['recall']:.3f}</td>"
            f"<td>{row['f1']:.3f}</td>"
            f"<td>{row['accuracy']:.3f}</td>"
            f"<td>{render_metric_bars(row['precision'], row['recall'], row['f1'])}</td>"
            f"<td>{render_confusion_plot(row['tp'], row['fp'], row['fn'], row['tn'])}</td>"
            "</tr>"
        )

    analysis_row_html: list[str] = []
    for row in analysis_rows:
        analysis_row_html.append(
            "<tr>"
            f"<td>{escape(row['annotation'])}</td>"
            f"<td>{row['analysis_universe']}</td>"
            f"<td>{row['manual_accepted_analyses']}</td>"
            f"<td>{row['predicted_analyses']}</td>"
            f"<td>{row['tp']}</td>"
            f"<td>{row['fp']}</td>"
            f"<td>{row['fn']}</td>"
            f"<td>{row['tn']}</td>"
            f"<td>{row['precision']:.3f}</td>"
            f"<td>{row['recall']:.3f}</td>"
            f"<td>{row['f1']:.3f}</td>"
            f"<td>{row['accuracy']:.3f}</td>"
            f"<td>{render_metric_bars(row['precision'], row['recall'], row['f1'])}</td>"
            f"<td>{render_confusion_plot(row['tp'], row['fp'], row['fn'], row['tn'])}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Overall Sub-Meta-Analysis Summary</title>
  <style>
    :root {{
      --bg: #f7f6f2;
      --panel: #ffffff;
      --ink: #1d2730;
      --line: #d8dde3;
    }}
    body {{ margin: 0; padding: 1.25rem; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    header, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border: 1px solid var(--line); padding: 0.45rem; vertical-align: top; text-align: left; }}
    th {{ background: #edf2f5; }}
    .metric-bars {{ min-width: 250px; }}
    .metric-row {{ display: grid; grid-template-columns: 22px 1fr 46px; gap: 0.35rem; align-items: center; margin-bottom: 0.2rem; }}
    .metric-label {{ font-weight: 600; font-size: 0.82rem; }}
    .metric-val {{ font-size: 0.82rem; text-align: right; }}
    .bar {{ height: 0.55rem; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; background: #fbfcfe; }}
    .fill {{ height: 100%; }}
    .fill-p {{ background: #3b82f6; }}
    .fill-r {{ background: #16a34a; }}
    .fill-f1 {{ background: #f59e0b; }}
    .confusion-plot {{ min-width: 220px; }}
    .stack-bar {{ width: 100%; height: 0.78rem; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; background: #fbfcfe; }}
    .seg {{ display: inline-block; height: 100%; }}
    .seg-tp {{ background: #16a34a; }}
    .seg-fp {{ background: #dc2626; }}
    .seg-fn {{ background: #ea580c; }}
    .seg-tn {{ background: #64748b; }}
    .legend {{ margin-top: 0.25rem; font-size: 0.77rem; color: #435164; display: flex; gap: 0.55rem; }}
    .lg::before {{ content: ""; display: inline-block; width: 0.55rem; height: 0.55rem; margin-right: 0.2rem; border-radius: 50%; vertical-align: -1px; }}
    .lg-tp::before {{ background: #16a34a; }}
    .lg-fp::before {{ background: #dc2626; }}
    .lg-fn::before {{ background: #ea580c; }}
    .lg-tn::before {{ background: #64748b; }}
    .confusion-plot.empty {{ font-size: 0.82rem; color: #5a6878; }}
  </style>
</head>
<body>
  <header>
    <h1>Overall Sub-Meta-Analysis Summary</h1>
    <p>Study-level evaluation across sub-meta-analyses. A study is included if at least one analysis is included. Universe is PMIDs found in auto <code>outputs/nimads_annotation.json</code>, with manual labels sliced to that same PMID universe.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Aggregate</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
            <th>TN</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Accuracy</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Macro (mean over annotations)</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
            <td>{macro_precision:.3f}</td>
            <td>{macro_recall:.3f}</td>
            <td>{macro_f1:.3f}</td>
            <td>{macro_accuracy:.3f}</td>
          </tr>
          <tr>
            <td>Micro (pooled confusion)</td>
            <td>{micro_tp}</td>
            <td>{micro_fp}</td>
            <td>{micro_fn}</td>
            <td>{micro_tn}</td>
            <td>{float(micro_prf.get('precision', 0.0)):.3f}</td>
            <td>{float(micro_prf.get('recall', 0.0)):.3f}</td>
            <td>{float(micro_prf.get('f1', 0.0)):.3f}</td>
            <td>{micro_accuracy:.3f}</td>
          </tr>
          <tr>
            <td>Matched analyses micro (pooled confusion)</td>
            <td>{analysis_micro_tp}</td>
            <td>{analysis_micro_fp}</td>
            <td>{analysis_micro_fn}</td>
            <td>{analysis_micro_tn}</td>
            <td>{float(analysis_micro_prf.get('precision', 0.0)):.3f}</td>
            <td>{float(analysis_micro_prf.get('recall', 0.0)):.3f}</td>
            <td>{float(analysis_micro_prf.get('f1', 0.0)):.3f}</td>
            <td>{analysis_micro_accuracy:.3f}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </header>
  <section>
    <h2>Per-Annotation Study-Level Metrics</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Annotation</th>
            <th>Auto PMID Universe</th>
            <th>Manual Studies+</th>
            <th>Predicted Studies+</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
            <th>TN</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Accuracy</th>
            <th>PRF Plot</th>
            <th>Confusion Plot</th>
          </tr>
        </thead>
        <tbody>
          {''.join(row_html)}
        </tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Per-Annotation Matched-Analysis Metrics</h2>
    <p>Mirrors the per-report analysis row: universe is manual analyses that have an assigned auto match; positives are accepted fuzzy matches.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Annotation</th>
            <th>Matched Manual Universe</th>
            <th>Manual Accepted+</th>
            <th>Predicted+</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
            <th>TN</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Accuracy</th>
            <th>PRF Plot</th>
            <th>Confusion Plot</th>
          </tr>
        </thead>
        <tbody>
          {''.join(analysis_row_html)}
        </tbody>
      </table>
    </div>
  </section>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    project_output_dir = infer_project_output_dir(args.project_output_dir)
    output_dir, match_input_dir = resolve_dirs(project_output_dir, args)

    annotation_results = project_output_dir / "outputs" / "annotation_results.json"
    coordinate_parsing_results = project_output_dir / "outputs" / "coordinate_parsing_results.json"
    auto_annotation_path = project_output_dir / "outputs" / "nimads_annotation.json"
    retrieval_dir = project_output_dir / "retrieval" / "pubget_data"
    manual_annotation_path = resolve_manual_annotation_path(project_output_dir, args.manual_annotation_path)

    parsed_analyses = load_auto_parsed_names(coordinate_parsing_results)
    model_decisions = load_model_decisions(annotation_results)
    match_results_by_annotation, overall_fallback = load_match_results_by_annotation(match_input_dir)
    manual_annotation_membership = load_manual_annotation_membership(manual_annotation_path)
    if overall_fallback and not manual_annotation_membership:
        print(
            "Warning: Using match_results_overall.json without nimads_annotation membership; "
            "manual truth cannot be sliced by annotation and may be over-inclusive."
        )
    manual_truth = build_manual_truth_from_match_results(
        match_results_by_annotation,
        overall_fallback=overall_fallback,
        manual_annotation_membership=manual_annotation_membership,
    )
    study_universe_pmids, auto_study_pmids_by_annotation, manual_study_pmids_by_annotation = (
        load_study_pmid_sets_from_annotations(
            auto_annotation_path=auto_annotation_path,
            manual_annotation_path=manual_annotation_path,
        )
    )
    if not study_universe_pmids:
        study_universe_pmids = set(parsed_analyses.keys())
    pmid_to_fulltext, pmid_to_coord_tables = load_retrieval_context(retrieval_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_annotation: dict[str, dict[str, Any]] = {}
    for annotation_name in MANUAL_FILE_MAP:
        docs, metrics = classify_documents(
            annotation_name=annotation_name,
            parsed_analyses=parsed_analyses,
            model_decisions=model_decisions,
            manual_truth=manual_truth,
            pmid_to_fulltext=pmid_to_fulltext,
            pmid_to_coord_tables=pmid_to_coord_tables,
            study_universe_pmids=study_universe_pmids,
            auto_study_pmids_by_annotation=auto_study_pmids_by_annotation,
            manual_study_pmids_by_annotation=manual_study_pmids_by_annotation,
        )
        metrics_by_annotation[annotation_name] = metrics
        html = render_html(annotation_name, docs, metrics)
        output_path = output_dir / f"{annotation_name}.html"
        output_path.write_text(html, encoding="utf-8")

        print(
            f"Wrote {output_path} | "
            f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
            f"doc_precision={metrics['precision']:.3f} doc_recall={metrics['recall']:.3f} doc_f1={metrics.get('f1', 0.0):.3f} "
            f"study_precision={metrics.get('study_metrics', {}).get('precision', 0.0):.3f} "
            f"study_recall={metrics.get('study_metrics', {}).get('recall', 0.0):.3f} "
            f"study_f1={metrics.get('study_metrics', {}).get('f1', 0.0):.3f} "
            f"analysis_precision={metrics.get('analysis_metrics', {}).get('precision', 0.0):.3f} "
            f"analysis_recall={metrics.get('analysis_metrics', {}).get('recall', 0.0):.3f} "
            f"analysis_f1={metrics.get('analysis_metrics', {}).get('f1', 0.0):.3f} "
            f"missing_manual_pmids={len(metrics.get('missing_manual_pmids', []))}"
        )

    overall_summary_html = render_overall_summary_html(metrics_by_annotation)
    overall_summary_path = output_dir / "overall_submeta_summary.html"
    overall_summary_path.write_text(overall_summary_html, encoding="utf-8")
    print(f"Wrote {overall_summary_path}")


if __name__ == "__main__":
    main()

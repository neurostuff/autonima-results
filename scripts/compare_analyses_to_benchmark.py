#!/usr/bin/env python3
"""Run analysis matching + annotation review reports in one script.

This combines and inlines logic from:
- scripts/run_fuzzy_analysis_matching.py
- scripts/generate_annotation_review_reports.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

try:
    from nimare.utils import mni2tal
except Exception:  # pragma: no cover
    mni2tal = None

OVERALL_RESULT_NAME = "overall"

ACCEPTED_THRESHOLD = 0.75

UNCERTAIN_THRESHOLD = 0.55

NAME_WEIGHT = 0.30

COORD_WEIGHT = 0.70

LOW_NAME_SCORE_HIGHLIGHT_THRESHOLD = UNCERTAIN_THRESHOLD

HUMAN_REVIEW_EXTRACTION_REASONS = [
    ("multiple_analyses_merged_into_one", "Multiple analyses merged into one"),
    ("single_analysis_split_into_multiple", "Single analysis split into multiple"),
    ("section_header_parsed_as_analysis", "Section/header parsed as analysis"),
    ("coordinate_rows_assigned_wrong_analysis", "Coordinate rows assigned to wrong analysis"),
    ("contrast_label_missed_or_truncated", "Contrast label missed or truncated"),
    ("table_structure_misparsed", "Table structure misparsed"),
    ("coordinates_missed_or_incomplete", "Coordinates missed or incomplete"),
    ("other_extraction_issue", "Other extraction issue"),
]

REVIEW_CREDITED_DISPOSITIONS = {
    "matching_error",
    "gold_standard_error",
    "expected_difference",
}

REVIEW_EXCLUDED_DISPOSITIONS = {
    "supplemental_data",
    "source_material_missing",
    "out_of_scope",
    "uncertain",
}

LEGACY_REVIEW_DISPOSITIONS = {
    "missed_unit_confirmed": "parser_missed",
    "missed_unit_supplemental_data": "supplemental_data",
    "missed_unit_out_of_scope": "out_of_scope",
    "gold_standard_wrong": "gold_standard_error",
}

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = SCRIPT_DIR.parent

PROJECTS_ROOT = REPO_ROOT / "projects"

MANUAL_NIMADS_ROOT = REPO_ROOT.parent / "neurometabench" / "data" / "nimads"

REQUIRED_OUTPUT_FILES = ("annotation_results.json", "coordinate_parsing_results.json")

def clean_text(value: str) -> str:
    return "".join(ch for ch in str(value) if ch >= " " or ch in "\n\t\r")

def normalize_text(value: str) -> str:
    text = clean_text(value).lower().strip()
    text = text.replace(">", " > ")
    text = re.sub(r"\s+", " ", text)
    return text

def is_valid_project_output_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    outputs_dir = path / "outputs"
    return outputs_dir.exists() and all((outputs_dir / name).exists() for name in REQUIRED_OUTPUT_FILES)

def annotation_result_mtime(project_output_dir: Path) -> float:
    return (project_output_dir / "outputs" / "annotation_results.json").stat().st_mtime

def find_project_output_dirs_within(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []

    candidates: list[Path] = []
    seen: set[Path] = set()

    def maybe_add(path: Path) -> None:
        if not path.is_dir():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        if is_valid_project_output_dir(path):
            seen.add(resolved)
            candidates.append(path)

    maybe_add(root)

    coordinates_dir = root / "coordinates"
    if coordinates_dir.is_dir():
        for entry in coordinates_dir.iterdir():
            maybe_add(entry)

    for entry in root.iterdir():
        maybe_add(entry)

    return candidates

def infer_project_output_dir(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--project-output-dir does not exist: {explicit_path}")

        direct_candidates = [path.parent] if path.name == "outputs" else [path]
        for candidate in direct_candidates:
            if is_valid_project_output_dir(candidate):
                return candidate

        scoped_candidates = find_project_output_dirs_within(path)
        if not scoped_candidates:
            raise FileNotFoundError(
                "Could not resolve a project output dir from --project-output-dir. "
                "Expected a run directory containing outputs/annotation_results.json and "
                "outputs/coordinate_parsing_results.json."
            )
        selected = max(scoped_candidates, key=annotation_result_mtime)
        print(f"Auto-selected project output dir within {path}: {selected}")
        return selected

    if not PROJECTS_ROOT.exists():
        raise FileNotFoundError(
            f"Could not infer project output dir because projects root was not found: {PROJECTS_ROOT}. "
            "Pass --project-output-dir explicitly."
        )

    all_candidates: list[Path] = []
    seen: set[Path] = set()
    for project_dir in PROJECTS_ROOT.iterdir():
        if not project_dir.is_dir():
            continue
        for candidate in find_project_output_dirs_within(project_dir):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            all_candidates.append(candidate)

    if not all_candidates:
        raise FileNotFoundError(
            "Could not infer project output dir from projects/. Pass --project-output-dir explicitly."
        )

    selected = max(all_candidates, key=annotation_result_mtime)
    print(f"Auto-selected project output dir (most recently updated): {selected}")
    return selected

def infer_project_name(project_output_dir: Path) -> str:
    parts = list(project_output_dir.resolve().parts)
    project_indices = [i for i, part in enumerate(parts) if part == "projects"]
    if project_indices:
        idx = project_indices[-1]
        if idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError(
        "Could not infer project name from project output dir. "
        f"Expected path under projects/{{project_name}}/... but got: {project_output_dir}"
    )

def resolve_output_dir(project_output_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return project_output_dir / "reports"

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def review_disposition(entry: dict[str, Any]) -> str:
    current = clean_text(
        str(entry.get("unmatched_gold_disposition") or "")
    ).strip()
    if current:
        return current
    legacy = clean_text(str(entry.get("missed_unit_disposition") or "")).strip()
    return LEGACY_REVIEW_DISPOSITIONS.get(legacy, legacy)

def load_parser_review_entries(
    review_path: Path,
    *,
    project_name: str,
) -> dict[str, dict[str, Any]]:
    payload = load_json(review_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Parser review must contain a JSON object: {review_path}")
    raw_entries = payload.get("entries", []) or []
    if not isinstance(raw_entries, list):
        raise ValueError(
            f"Parser review 'entries' must contain a list: {review_path}"
        )

    entries: dict[str, dict[str, Any]] = {}
    prefix = f"{project_name}:"
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        unit_id = clean_text(str(raw_entry.get("unit_id") or "")).strip()
        if not unit_id or not unit_id.startswith(prefix):
            continue
        if unit_id in entries:
            raise ValueError(
                f"Parser review contains duplicate unit ID for {project_name}: "
                f"{unit_id}"
            )
        entries[unit_id] = raw_entry
    return entries

def refresh_match_result_summaries(match_result: dict[str, Any]) -> None:
    """Recalculate counts after human-review adjustments."""
    pmid_results = match_result.get("pmids", {}) or {}
    all_entries: list[dict[str, Any]] = []
    status_counts: defaultdict[str, int] = defaultdict(int)
    combined_scores: list[float] = []
    category_counts: defaultdict[str, int] = defaultdict(int)
    perfect_pmids = 0
    evaluable_pmids = 0
    coord_override_accepted = 0
    low_name_coord_override_matches = 0

    for data in pmid_results.values():
        entries = data.get("manual_analyses", []) or []
        all_entries.extend(entries)
        counts: defaultdict[str, int] = defaultdict(int)
        for entry in entries:
            status = str(entry.get("match_status", "unmatched"))
            counts[status] += 1
            status_counts[status] += 1
            combined_scores.append(float(entry.get("combined_score", 0.0)))
            if (
                bool(entry.get("coord_override_accepted", False))
                and float(entry.get("combined_score", 0.0)) < ACCEPTED_THRESHOLD
            ):
                coord_override_accepted += 1
            if bool(entry.get("low_name_with_exact_coords", False)):
                low_name_coord_override_matches += 1

        pmid_summary = data.setdefault("pmid_summary", {})
        pmid_summary["accepted"] = int(counts["accepted"])
        pmid_summary["uncertain"] = int(counts["uncertain"])
        pmid_summary["unmatched"] = int(counts["unmatched"])
        pmid_summary["manual_analysis_count"] = len(entries)
        pmid_summary["all_manual_accepted"] = bool(entries) and (
            int(counts["accepted"]) == len(entries)
        )
        pmid_summary["mean_combined_score"] = round(
            (
                sum(float(entry.get("combined_score", 0.0)) for entry in entries)
                / len(entries)
            )
            if entries
            else 0.0,
            6,
        )
        if entries:
            category = classify_study_match_category(
                accepted=int(counts["accepted"]),
                manual_total=len(entries),
            )
            evaluable_pmids += 1
            category_counts[category] += 1
            if pmid_summary["all_manual_accepted"]:
                perfect_pmids += 1
        else:
            category = "review_excluded"
        pmid_summary["study_category"] = category

    combined_arr = (
        np.array(combined_scores, dtype=float)
        if combined_scores
        else np.array([], dtype=float)
    )
    summary = match_result.setdefault("summary", {})
    summary.update(
        {
            "manual_analyses_total": len(all_entries),
            "accepted": int(status_counts["accepted"]),
            "uncertain": int(status_counts["uncertain"]),
            "unmatched": int(status_counts["unmatched"]),
            "accepted_coord_override": int(coord_override_accepted),
            "low_name_coord_override_matches": int(
                low_name_coord_override_matches
            ),
            "accepted_exact_coord_override": int(coord_override_accepted),
            "low_name_exact_matches": int(low_name_coord_override_matches),
            "review_evaluable_pmids": evaluable_pmids,
            "pmids_all_manual_accepted": int(perfect_pmids),
            "pmids_all_manual_accepted_rate": (
                float(perfect_pmids) / evaluable_pmids
                if evaluable_pmids
                else 0.0
            ),
            "all_correct_pmids": int(category_counts["all_correct"]),
            "mixed_pmids": int(category_counts["mixed"]),
            "all_incorrect_pmids": int(category_counts["all_incorrect"]),
            "review_excluded_pmids": sum(
                1
                for data in pmid_results.values()
                if data.get("pmid_summary", {}).get("study_category")
                == "review_excluded"
            ),
            "mean_combined_score": (
                float(np.mean(combined_arr)) if combined_arr.size else 0.0
            ),
            "median_combined_score": (
                float(np.median(combined_arr)) if combined_arr.size else 0.0
            ),
            "p25_combined_score": (
                float(np.percentile(combined_arr, 25))
                if combined_arr.size
                else 0.0
            ),
            "p75_combined_score": (
                float(np.percentile(combined_arr, 75))
                if combined_arr.size
                else 0.0
            ),
        }
    )

def apply_parser_review_adjustments(
    match_result: dict[str, Any],
    *,
    review_entries: dict[str, dict[str, Any]],
    project_name: str,
    review_path: Path,
) -> dict[str, Any]:
    """Apply reviewed dispositions to parser-scoring outcomes.

    Confirmed parser misses remain penalties. Benchmark/matching/expected
    differences are credited as accepted. Cases whose source was unavailable or
    outside the parser's evaluable scope are removed from the denominator.
    """
    credited = 0
    excluded = 0
    parser_missed = 0
    matched_review_entries = 0
    dispositions: defaultdict[str, int] = defaultdict(int)

    for pmid, data in (match_result.get("pmids", {}) or {}).items():
        retained: list[dict[str, Any]] = []
        excluded_entries: list[dict[str, Any]] = []
        entries = data.get("manual_analyses", []) or []
        pmid_summary = data.setdefault("pmid_summary", {})
        pmid_summary["manual_analysis_count_before_review"] = len(entries)

        for match_entry in entries:
            manual_id = clean_text(
                str(match_entry.get("manual_analysis_id") or "")
            ).strip()
            unit_id = f"{project_name}:{pmid}:{manual_id}"
            review_entry = review_entries.get(unit_id)
            if review_entry is None:
                retained.append(match_entry)
                continue

            matched_review_entries += 1
            disposition = review_disposition(review_entry)
            dispositions[disposition or "(blank)"] += 1
            match_entry["review_unit_id"] = unit_id
            match_entry["review_disposition"] = disposition
            match_entry["review_note"] = clean_text(
                str(review_entry.get("note") or "")
            ).strip()

            if disposition in REVIEW_CREDITED_DISPOSITIONS:
                match_entry["raw_match_status"] = str(
                    match_entry.get("match_status", "unmatched")
                )
                match_entry["match_status"] = "accepted"
                match_entry["review_adjustment"] = "credited_as_accepted"
                credited += 1
                retained.append(match_entry)
            elif disposition in REVIEW_EXCLUDED_DISPOSITIONS:
                match_entry["raw_match_status"] = str(
                    match_entry.get("match_status", "unmatched")
                )
                match_entry["review_adjustment"] = (
                    "excluded_non_parser_evaluable"
                )
                excluded += 1
                excluded_entries.append(match_entry)
            else:
                if disposition == "parser_missed":
                    parser_missed += 1
                    match_entry["review_adjustment"] = (
                        "confirmed_parser_evaluable"
                    )
                retained.append(match_entry)

        data["manual_analyses"] = retained
        data["review_excluded_manual_analyses"] = excluded_entries
        pmid_summary["review_excluded_manual_analysis_count"] = len(
            excluded_entries
        )

    refresh_match_result_summaries(match_result)
    summary = match_result.setdefault("summary", {})
    summary["review_credited_manual_analyses"] = credited
    summary["review_excluded_manual_analyses"] = excluded
    summary["review_confirmed_parser_misses"] = parser_missed
    adjustment = {
        "enabled": True,
        "review_path": str(review_path),
        "project": project_name,
        "review_entries_for_project": len(review_entries),
        "matched_review_entries": matched_review_entries,
        "unmatched_review_entries": len(review_entries) - matched_review_entries,
        "credited_as_accepted": credited,
        "excluded_non_parser_evaluable": excluded,
        "confirmed_parser_misses": parser_missed,
        "credited_dispositions": sorted(REVIEW_CREDITED_DISPOSITIONS),
        "excluded_dispositions": sorted(REVIEW_EXCLUDED_DISPOSITIONS),
        "disposition_counts": dict(sorted(dispositions.items())),
    }
    match_result["parser_review_adjustment"] = adjustment
    match_result.setdefault("matching_policy", {})[
        "human_review_adjustment"
    ] = {
        "confirmed_parser_misses_remain_penalties": True,
        "credited_dispositions": sorted(REVIEW_CREDITED_DISPOSITIONS),
        "excluded_from_denominator": sorted(REVIEW_EXCLUDED_DISPOSITIONS),
    }
    return adjustment

def normalize_pmid(value: Any) -> str:
    text = clean_text(str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(r"^pmid\s*[:#]?\s*", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text

def normalize_pmcid(value: Any) -> str:
    text = clean_text(str(value or "")).strip().upper()
    if not text:
        return ""
    text = re.sub(r"^PMCID\s*[:#]?\s*", "", text, flags=re.IGNORECASE)
    if text.startswith("PMC"):
        text = text[3:]
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text

def build_pubget_index(project_output_dir: Path) -> dict[str, dict[str, Any]]:
    pubget_data_dir = project_output_dir / "retrieval" / "pubget_data"
    metadata_csv = pubget_data_dir / "metadata.csv"
    tables_csv = pubget_data_dir / "tables.csv"
    if not metadata_csv.exists():
        return {}

    article_xml_by_pmcid: dict[str, str] = {}
    for article_xml_path in pubget_data_dir.glob("articles/*/pmcid_*/article.xml"):
        pmcid = normalize_pmcid(article_xml_path.parent.name)
        if pmcid:
            article_xml_by_pmcid[pmcid] = str(article_xml_path.relative_to(pubget_data_dir))

    by_pmid: dict[str, dict[str, Any]] = {}
    pmcid_to_pmids: dict[str, list[str]] = defaultdict(list)
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = normalize_pmid(row.get("pmid"))
            pmcid = normalize_pmcid(row.get("pmcid"))
            if not pmid or not pmcid:
                continue
            if pmid not in by_pmid:
                by_pmid[pmid] = {
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/",
                    "article_xml_file": article_xml_by_pmcid.get(pmcid),
                    "title": clean_text(row.get("title") or ""),
                    "journal": clean_text(row.get("journal") or ""),
                    "publication_year": clean_text(row.get("publication_year") or ""),
                    "tables": [],
                }
            pmcid_to_pmids[pmcid].append(pmid)

    if tables_csv.exists():
        with tables_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pmcid = normalize_pmcid(row.get("pmcid"))
                target_pmids = pmcid_to_pmids.get(pmcid, [])
                if not target_pmids:
                    continue

                table_data_file = clean_text(row.get("table_data_file") or "").strip()
                table_csv_path = pubget_data_dir / table_data_file if table_data_file else None
                n_header_rows_raw = clean_text(row.get("n_header_rows") or "").strip()
                try:
                    n_header_rows = int(n_header_rows_raw) if n_header_rows_raw else 1
                except Exception:
                    n_header_rows = 1

                entry = {
                    "pmcid": pmcid,
                    "table_id": clean_text(row.get("table_id") or ""),
                    "table_label": clean_text(row.get("table_label") or ""),
                    "table_caption": clean_text(row.get("table_caption") or ""),
                    "table_foot": clean_text(row.get("table_foot") or ""),
                    "n_header_rows": max(0, n_header_rows),
                    "table_data_file": table_data_file,
                    "table_csv_path": str(table_csv_path) if table_csv_path else "",
                    "table_csv_exists": bool(table_csv_path and table_csv_path.exists()),
                }
                for pmid in target_pmids:
                    by_pmid[pmid]["tables"].append(entry)

    for item in by_pmid.values():
        item["tables"] = sorted(
            item.get("tables", []),
            key=lambda row: (
                str(row.get("table_label") or ""),
                str(row.get("table_id") or ""),
                str(row.get("table_data_file") or ""),
            ),
        )
    return by_pmid

def annotate_match_result_with_pubget(
    match_result: dict[str, Any],
    pubget_by_pmid: dict[str, dict[str, Any]],
) -> None:
    pmid_results = match_result.get("pmids", {})
    pmids_with_pubget = 0
    tables_total = 0
    for pmid, data in pmid_results.items():
        resource = pubget_by_pmid.get(str(pmid))
        if not resource:
            data["pubget"] = {"available": False}
            continue
        table_count = len(resource.get("tables", []))
        pmids_with_pubget += 1
        tables_total += table_count
        data["pubget"] = {
            "available": True,
            "pmcid": resource.get("pmcid"),
            "pmc_url": resource.get("pmc_url"),
            "article_xml_file": resource.get("article_xml_file"),
            "table_count": table_count,
            "title": resource.get("title"),
            "journal": resource.get("journal"),
            "publication_year": resource.get("publication_year"),
        }

    summary = match_result.setdefault("summary", {})
    summary["pmids_with_pubget"] = int(pmids_with_pubget)
    summary["pubget_tables_total"] = int(tables_total)

def render_csv_table_html(csv_path: Path, n_header_rows: int) -> str:
    if not csv_path.exists() or not csv_path.is_file():
        return (
            "<p class=\"resource-note\">CSV file missing: "
            f"<code>{escape(str(csv_path))}</code></p>"
        )

    rows: list[list[str]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append([clean_text(cell) for cell in row])
    except Exception as exc:
        return (
            "<p class=\"resource-note\">Failed to read CSV: "
            f"<code>{escape(str(csv_path))}</code> ({escape(str(exc))})</p>"
        )

    if not rows:
        return "<p class=\"resource-note\">No table rows extracted.</p>"

    header_count = max(0, min(int(n_header_rows), len(rows)))
    if header_count == 0:
        header_count = 1

    header_rows = rows[:header_count]
    body_rows = rows[header_count:]

    thead_html = "".join(
        "<tr>{}</tr>".format("".join(f"<th>{escape(cell)}</th>" for cell in row))
        for row in header_rows
    )
    tbody_html = "".join(
        "<tr>{}</tr>".format("".join(f"<td>{escape(cell)}</td>" for cell in row))
        for row in body_rows
    )

    return (
        "<div class=\"table-wrap\">"
        "<table class=\"extracted-table\">"
        f"<thead>{thead_html}</thead>"
        f"<tbody>{tbody_html}</tbody>"
        "</table>"
        "</div>"
    )

def parse_points(points: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    parsed: list[tuple[float, float, float]] = []
    for point in points or []:
        coords = point.get("coordinates", [])
        if not isinstance(coords, (list, tuple)) or len(coords) != 3:
            continue
        try:
            parsed.append((float(coords[0]), float(coords[1]), float(coords[2])))
        except Exception:
            continue
    return parsed

def coordinate_value_has_decimal(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        if isinstance(value, str):
            numeric = float(clean_text(value).strip())
        else:
            numeric = float(value)
    except Exception:
        return False
    return not float(numeric).is_integer()

def parse_points_with_decimal_detection(
    points: list[dict[str, Any]],
) -> tuple[list[tuple[float, float, float]], bool]:
    parsed: list[tuple[float, float, float]] = []
    has_decimal_coordinate = False
    for point in points or []:
        coords = point.get("coordinates", [])
        if not isinstance(coords, (list, tuple)) or len(coords) != 3:
            continue
        try:
            x_raw, y_raw, z_raw = coords[0], coords[1], coords[2]
            if (
                coordinate_value_has_decimal(x_raw)
                or coordinate_value_has_decimal(y_raw)
                or coordinate_value_has_decimal(z_raw)
            ):
                has_decimal_coordinate = True
            parsed.append((float(x_raw), float(y_raw), float(z_raw)))
        except Exception:
            continue
    return parsed, has_decimal_coordinate

def convert_coords_mni_to_talairach(
    coords: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    if not coords:
        return []
    if mni2tal is None:
        raise ImportError(
            "decimal manual coordinate handling mode 'convert_to_talairach' requires NiMARE. "
            "Install nimare to enable mni2tal conversion."
        )
    arr = np.array(coords, dtype=float)
    converted = np.array(mni2tal(arr), dtype=float)
    return [(float(x), float(y), float(z)) for x, y, z in converted.tolist()]

def load_auto_parsed_data(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = load_json(path)
    studies = payload.get("studies", [])
    auto_by_pmid: dict[str, list[dict[str, Any]]] = {}

    for study in studies:
        pmid = str(study.get("pmid"))
        analyses = study.get("analyses", [])
        entries: list[dict[str, Any]] = []
        for idx, analysis in enumerate(analyses):
            name = clean_text(analysis.get("name") or f"analysis_{idx}")
            entries.append(
                {
                    "index": idx,
                    "analysis_id": f"{pmid}_analysis_{idx}",
                    "name": name,
                    "points": parse_points(analysis.get("points", [])),
                }
            )
        auto_by_pmid[pmid] = entries

    return auto_by_pmid


def resolve_retrieval_pubget_dir(project_output_dir: Path) -> Path | None:
    def is_valid_retrieval_dir(path: Path) -> bool:
        return (path / "metadata.csv").exists() and (path / "coordinates.csv").exists()

    direct = project_output_dir / "retrieval" / "pubget_data"
    if is_valid_retrieval_dir(direct):
        return direct

    run_name = project_output_dir.name
    project_dir = project_output_dir.parent
    sibling_names: list[str] = []
    annotation_removed = re.sub(r"-annotation-only(?:-.+)?$", "", run_name)
    if annotation_removed and annotation_removed != run_name:
        sibling_names.append(annotation_removed)
    version_match = re.match(r"^(v\d+)", run_name)
    if version_match:
        sibling_names.append(version_match.group(1))
    sibling_names.extend([run_name.replace("-gpt", ""), run_name.replace("-annotation-only-gpt", "")])

    for sibling_name in sibling_names:
        sibling = project_dir / sibling_name / "retrieval" / "pubget_data"
        if is_valid_retrieval_dir(sibling):
            return sibling

    candidates = sorted(project_dir.glob("*/retrieval/pubget_data"))
    valid_candidates = [path for path in candidates if is_valid_retrieval_dir(path)]
    if len(valid_candidates) == 1:
        return valid_candidates[0]
    return None


def load_table_only_auto_data_from_coordinate_parsing(
    coordinate_parsing_results_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not coordinate_parsing_results_path.exists():
        return {}, {
            "available": False,
            "source": "",
            "reason": "Missing outputs/coordinate_parsing_results.json.",
        }

    payload = load_json(coordinate_parsing_results_path)
    studies = payload.get("studies", []) if isinstance(payload, dict) else []
    grouped_by_pmid_table: dict[tuple[str, str], dict[str, Any]] = {}
    for study in studies:
        if not isinstance(study, dict):
            continue
        pmid = str(study.get("pmid") or "").strip()
        if not pmid:
            continue
        for analysis in study.get("analyses", []) or []:
            if not isinstance(analysis, dict):
                continue
            table_id = clean_text(analysis.get("table_id") or "").strip() or "unknown_table"
            key = (pmid, table_id)
            payload_for_table = grouped_by_pmid_table.setdefault(
                key,
                {
                    "pmid": pmid,
                    "table_id": table_id,
                    "names": [],
                    "points": [],
                },
            )
            analysis_name = clean_text(analysis.get("name") or "").strip()
            if analysis_name and analysis_name not in payload_for_table["names"]:
                payload_for_table["names"].append(analysis_name)
            payload_for_table["points"].extend(parse_points(analysis.get("points", [])))

    by_pmid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_pmid, table_id), payload_for_table in grouped_by_pmid_table.items():
        points = payload_for_table.get("points", [])
        if not points:
            continue
        pmid = str(payload_for_table["pmid"])
        names = payload_for_table.get("names", [])
        table_name = f"Table {table_id}"
        if names:
            table_name = f"{table_name}: " + " | ".join(str(name) for name in names[:6])
        by_pmid[pmid].append(
            {
                "index": len(by_pmid[pmid]),
                "analysis_id": f"{pmid}_table_{sanitize_id(table_id)}",
                "name": table_name,
                "points": points,
                "table_id": table_id,
                "table_label": table_id,
                "table_caption": "",
                "source": "table_only_coordinate_parsing_results",
            }
        )

    for entries in by_pmid.values():
        entries.sort(key=lambda item: str(item.get("table_id") or ""))
        for idx, entry in enumerate(entries):
            entry["index"] = idx

    table_count = sum(len(entries) for entries in by_pmid.values())
    coord_count = sum(len(entry.get("points", [])) for entries in by_pmid.values() for entry in entries)
    return dict(by_pmid), {
        "available": bool(by_pmid),
        "source": str(coordinate_parsing_results_path),
        "pmids_with_table_coordinates": len(by_pmid),
        "table_units": table_count,
        "coordinate_rows": coord_count,
        "reason": "" if by_pmid else "No table-coordinate groups found in coordinate_parsing_results.json.",
    }


def load_raw_retrieval_table_only_auto_data(
    project_output_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load a no-analysis-parsing baseline: one automated unit per source table.

    The retrieval exporters normalize coordinate extraction into
    retrieval/pubget_data/{metadata,tables,coordinates}.csv across Pubget-style,
    Elsevier, and ACE-backed runs. This baseline keeps every coordinate extracted
    from a paper, but splits only by the source table ID instead of by parsed
    analysis rows.
    """
    retrieval_dir = resolve_retrieval_pubget_dir(project_output_dir)
    if retrieval_dir is None:
        return {}, {
            "available": False,
            "source": "",
            "reason": "Missing retrieval/pubget_data metadata.csv and coordinates.csv.",
        }
    metadata_path = retrieval_dir / "metadata.csv"
    tables_path = retrieval_dir / "tables.csv"
    coordinates_path = retrieval_dir / "coordinates.csv"

    pmcid_to_pmid: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmcid = normalize_pmcid(row.get("pmcid"))
            pmid = normalize_pmid(row.get("pmid"))
            if pmcid and pmid:
                pmcid_to_pmid[pmcid] = pmid

    table_meta: dict[tuple[str, str], dict[str, str]] = {}
    if tables_path.exists():
        with tables_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pmcid = normalize_pmcid(row.get("pmcid"))
                table_id = clean_text(row.get("table_id") or "").strip()
                if pmcid and table_id:
                    table_meta[(pmcid, table_id)] = row

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    with coordinates_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmcid = normalize_pmcid(row.get("pmcid"))
            pmid = pmcid_to_pmid.get(pmcid)
            if not pmid:
                continue
            table_id = clean_text(row.get("table_id") or "").strip()
            if not table_id:
                table_id = "unknown_table"
            table_label = clean_text(row.get("table_label") or "").strip()
            key = (pmid, pmcid, table_id)
            payload = grouped.setdefault(
                key,
                {
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "table_id": table_id,
                    "table_label": table_label,
                    "points": [],
                },
            )
            if not payload.get("table_label") and table_label:
                payload["table_label"] = table_label
            try:
                payload["points"].append((float(row.get("x")), float(row.get("y")), float(row.get("z"))))
            except Exception:
                continue

    by_pmid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (_pmid, pmcid, table_id), payload in grouped.items():
        points = payload.get("points", [])
        if not points:
            continue
        meta = table_meta.get((pmcid, table_id), {})
        table_label = clean_text(payload.get("table_label") or meta.get("table_label") or "").strip()
        caption = clean_text(meta.get("table_caption") or "").strip()
        foot = clean_text(meta.get("table_foot") or "").strip()
        name_parts = [part for part in [table_label, caption, foot] if part]
        table_name = " | ".join(name_parts) if name_parts else table_id
        by_pmid[str(payload["pmid"])].append(
            {
                "index": len(by_pmid[str(payload["pmid"])]),
                "analysis_id": f"{payload['pmid']}_table_{sanitize_id(table_id)}",
                "name": table_name,
                "points": points,
                "table_id": table_id,
                "table_label": table_label,
                "table_caption": caption,
                "source": "table_only_raw_coordinates",
            }
        )

    for entries in by_pmid.values():
        entries.sort(key=lambda item: (str(item.get("table_label") or ""), str(item.get("table_id") or "")))
        for idx, entry in enumerate(entries):
            entry["index"] = idx
    table_count = sum(len(entries) for entries in by_pmid.values())
    coord_count = sum(len(entry.get("points", [])) for entries in by_pmid.values() for entry in entries)
    return dict(by_pmid), {
        "available": bool(by_pmid),
        "source": str(retrieval_dir),
        "pmids_with_table_coordinates": len(by_pmid),
        "table_units": table_count,
        "coordinate_rows": coord_count,
        "reason": "" if by_pmid else "No table-coordinate groups found in coordinates.csv.",
    }


def load_table_only_auto_data(project_output_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    raw_by_pmid, raw_info = load_raw_retrieval_table_only_auto_data(project_output_dir)
    parsed_by_pmid, parsed_info = load_table_only_auto_data_from_coordinate_parsing(
        project_output_dir / "outputs" / "coordinate_parsing_results.json"
    )
    raw_coord_count = int(raw_info.get("coordinate_rows", 0) or 0)
    parsed_coord_count = int(parsed_info.get("coordinate_rows", 0) or 0)
    raw_pmid_count = int(raw_info.get("pmids_with_table_coordinates", 0) or 0)
    parsed_pmid_count = int(parsed_info.get("pmids_with_table_coordinates", 0) or 0)

    if parsed_by_pmid and (parsed_coord_count > raw_coord_count or parsed_pmid_count > raw_pmid_count):
        parsed_info["preferred_over_raw_retrieval"] = bool(raw_by_pmid)
        parsed_info["raw_retrieval_source"] = raw_info.get("source", "")
        parsed_info["raw_retrieval_coordinate_rows"] = raw_coord_count
        parsed_info["raw_retrieval_pmids_with_table_coordinates"] = raw_pmid_count
        return parsed_by_pmid, parsed_info
    if raw_by_pmid:
        raw_info["preferred_over_coordinate_parsing_results"] = bool(parsed_by_pmid)
        raw_info["coordinate_parsing_results_source"] = parsed_info.get("source", "")
        raw_info["coordinate_parsing_results_coordinate_rows"] = parsed_coord_count
        raw_info["coordinate_parsing_results_pmids_with_table_coordinates"] = parsed_pmid_count
        return raw_by_pmid, raw_info
    return parsed_by_pmid, parsed_info


def sanitize_id(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text or "unknown"

def resolve_manual_merged_studyset_path(manual_dir: Path) -> Path:
    direct_studyset = manual_dir / "nimads_studyset.json"
    if direct_studyset.exists():
        return direct_studyset

    merged_studyset = manual_dir / "merged" / "nimads_studyset.json"
    if merged_studyset.exists():
        return merged_studyset

    raise FileNotFoundError(
        "Could not find merged manual NiMADS studyset. Expected either "
        f"{direct_studyset} or {merged_studyset}."
    )

def resolve_manual_dir(project_output_dir: Path, explicit_manual_dir: Path | None) -> Path:
    if explicit_manual_dir is not None:
        return explicit_manual_dir.expanduser().resolve()

    project_name = infer_project_name(project_output_dir)
    inferred_manual_dir = (MANUAL_NIMADS_ROOT / project_name).resolve()

    if not inferred_manual_dir.exists():
        raise FileNotFoundError(
            "Could not infer manual benchmark directory. "
            f"Expected to find: {inferred_manual_dir}. "
            "Pass --manual-dir explicitly."
        )

    resolve_manual_merged_studyset_path(inferred_manual_dir)
    print(f"Auto-selected manual benchmark dir from project '{project_name}': {inferred_manual_dir}")
    return inferred_manual_dir

def load_manual_analyses_overall(manual_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    studyset_path = resolve_manual_merged_studyset_path(manual_dir)
    studyset_payload = load_json(studyset_path)
    result: dict[str, list[dict[str, Any]]] = {}
    study_names: dict[str, str] = {}
    for study in studyset_payload.get("studies", []):
        pmid = str(study.get("id"))
        study_names[pmid] = clean_text(study.get("name") or pmid)
        analyses: list[dict[str, Any]] = []
        for analysis in study.get("analyses", []):
            analysis_id = clean_text(analysis.get("id") or "").strip()
            if not analysis_id:
                continue
            analysis_name = clean_text(analysis.get("name") or analysis_id)
            points, has_decimal_coordinates = parse_points_with_decimal_detection(analysis.get("points", []))
            analyses.append(
                {
                    "id": analysis_id,
                    "name": analysis_name,
                    "points": points,
                    "has_decimal_coordinates": has_decimal_coordinates,
                }
            )
        result[pmid] = sorted(analyses, key=lambda item: item["id"])
    return result, study_names

def split_name_base(name: str) -> str:
    return normalize_text(name).split(";", 1)[0].strip()

def compute_name_score(manual_name: str, auto_name: str) -> float:
    m_full = normalize_text(manual_name)
    a_full = normalize_text(auto_name)
    m_base = split_name_base(manual_name)
    a_base = split_name_base(auto_name)

    scores = [
        SequenceMatcher(None, m_full, a_full).ratio(),
        SequenceMatcher(None, m_base, a_base).ratio(),
        SequenceMatcher(None, m_full, a_base).ratio(),
        SequenceMatcher(None, m_base, a_full).ratio(),
    ]
    return max(scores)

def rounded_coords(coords: list[tuple[float, float, float]], decimals: int = 1) -> list[tuple[float, float, float]]:
    return sorted((round(x, decimals), round(y, decimals), round(z, decimals)) for x, y, z in coords)

def distance_to_similarity(distance: float) -> float:
    if distance <= 1.0:
        return 1.0
    if distance <= 2.0:
        return 0.9
    if distance <= 4.0:
        return 0.9 - ((distance - 2.0) * (0.3 / 2.0))
    if distance <= 8.0:
        return 0.6 - ((distance - 4.0) * (0.4 / 4.0))
    return 0.0

def assign_pairs(score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if score_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if linear_sum_assignment is not None:
        return linear_sum_assignment(1.0 - score_matrix)

    n_rows, n_cols = score_matrix.shape
    pairs = [(i, j, float(score_matrix[i, j])) for i in range(n_rows) for j in range(n_cols)]
    pairs.sort(key=lambda x: x[2], reverse=True)

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    out_rows: list[int] = []
    out_cols: list[int] = []

    for i, j, _score in pairs:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        out_rows.append(i)
        out_cols.append(j)
        if len(used_rows) == min(n_rows, n_cols):
            break

    return np.array(out_rows, dtype=int), np.array(out_cols, dtype=int)

def has_componentwise_tolerant_coord_set_match(
    manual_coords: list[tuple[float, float, float]],
    auto_coords: list[tuple[float, float, float]],
    axis_tolerance: float,
) -> bool:
    if axis_tolerance < 0:
        return False
    if len(manual_coords) != len(auto_coords):
        return False
    n = len(manual_coords)
    if n == 0:
        return False

    allowed: list[list[bool]] = []
    for mx, my, mz in manual_coords:
        row: list[bool] = []
        for ax, ay, az in auto_coords:
            row.append(
                abs(mx - ax) <= axis_tolerance
                and abs(my - ay) <= axis_tolerance
                and abs(mz - az) <= axis_tolerance
            )
        allowed.append(row)

    match_to_row = [-1] * n

    def dfs(row_idx: int, seen_cols: list[bool]) -> bool:
        for col_idx in range(n):
            if seen_cols[col_idx] or not allowed[row_idx][col_idx]:
                continue
            seen_cols[col_idx] = True
            prev_row = match_to_row[col_idx]
            if prev_row == -1 or dfs(prev_row, seen_cols):
                match_to_row[col_idx] = row_idx
                return True
        return False

    matched = 0
    for row_idx in range(n):
        seen = [False] * n
        if dfs(row_idx, seen):
            matched += 1
    return matched == n

def compute_coord_score(
    manual_coords: list[tuple[float, float, float]],
    auto_coords: list[tuple[float, float, float]],
    exact_match_axis_tolerance: float | None = None,
) -> tuple[float, dict[str, Any], list[str]]:
    reasons: list[str] = []
    if not manual_coords or not auto_coords:
        reasons.append("missing_coords_on_one_side")
        return 0.0, {"exact_coord_set": False, "coverage_penalty": 0.0, "match_quality": 0.0}, reasons

    m = np.array(manual_coords, dtype=float)
    a = np.array(auto_coords, dtype=float)
    dists = np.sqrt(np.sum((m[:, None, :] - a[None, :, :]) ** 2, axis=2))
    sim_matrix = np.vectorize(distance_to_similarity)(dists)

    row_ind, col_ind = assign_pairs(sim_matrix)
    if row_ind.size == 0:
        reasons.append("low_total_score")
        return 0.0, {"exact_coord_set": False, "coverage_penalty": 0.0, "match_quality": 0.0}, reasons

    paired_sims = [float(sim_matrix[r, c]) for r, c in zip(row_ind, col_ind)]
    match_quality = float(np.mean(paired_sims)) if paired_sims else 0.0
    coverage_penalty = min(len(manual_coords), len(auto_coords)) / max(len(manual_coords), len(auto_coords))
    strict_exact_coord_set = (
        len(manual_coords) == len(auto_coords)
        and rounded_coords(manual_coords) == rounded_coords(auto_coords)
    )
    tolerance_exact_coord_set = (
        not strict_exact_coord_set
        and exact_match_axis_tolerance is not None
        and has_componentwise_tolerant_coord_set_match(
            manual_coords,
            auto_coords,
            axis_tolerance=float(exact_match_axis_tolerance),
        )
    )
    exact_coord_set = strict_exact_coord_set or tolerance_exact_coord_set
    exact_bonus = 0.05 if exact_coord_set else 0.0

    score = max(0.0, min(1.0, (match_quality * coverage_penalty) + exact_bonus))

    if exact_coord_set:
        reasons.append("exact_coord_set")
    if tolerance_exact_coord_set:
        reasons.append("exact_coord_set_axis_tolerance")
    if len(manual_coords) != len(auto_coords):
        reasons.append("coord_count_mismatch")
    if score >= 0.75:
        reasons.append("high_coord_match")

    return score, {
        "exact_coord_set": exact_coord_set,
        "strict_exact_coord_set": strict_exact_coord_set,
        "tolerance_exact_coord_set": tolerance_exact_coord_set,
        "exact_match_axis_tolerance": (
            float(exact_match_axis_tolerance) if exact_match_axis_tolerance is not None else None
        ),
        "coverage_penalty": coverage_penalty,
        "match_quality": match_quality,
    }, reasons

def status_from_score(score: float) -> str:
    if score >= ACCEPTED_THRESHOLD:
        return "accepted"
    if score >= UNCERTAIN_THRESHOLD:
        return "uncertain"
    return "unmatched"

def status_from_detail(detail: dict[str, Any]) -> str:
    if bool(detail.get("coord_override_accepted", False)):
        return "accepted"
    return status_from_score(float(detail.get("combined_score", 0.0)))

def score_pair(
    manual_analysis: dict[str, Any],
    auto_analysis: dict[str, Any],
    coord_accept_override_threshold: float,
    converted_talairach_exact_axis_tolerance: float,
) -> dict[str, Any]:
    name_score = compute_name_score(manual_analysis["name"], auto_analysis["name"])
    exact_match_axis_tolerance: float | None = None
    if bool(manual_analysis.get("converted_from_decimal_mni_to_talairach", False)):
        exact_match_axis_tolerance = float(converted_talairach_exact_axis_tolerance)
    coord_score, coord_meta, reasons = compute_coord_score(
        manual_analysis["points"],
        auto_analysis["points"],
        exact_match_axis_tolerance=exact_match_axis_tolerance,
    )
    combined = (COORD_WEIGHT * coord_score) + (NAME_WEIGHT * name_score)
    exact_coord_set = bool(coord_meta.get("exact_coord_set", False))
    coord_override_accepted = coord_score >= coord_accept_override_threshold
    low_name_with_exact_coords = coord_override_accepted and name_score < LOW_NAME_SCORE_HIGHLIGHT_THRESHOLD

    if coord_score < 0.4 and name_score >= 0.75:
        reasons.append("low_coord_high_name")
    if coord_score == 0.0 and name_score >= 0.6:
        reasons.append("name_only_signal")
    if low_name_with_exact_coords:
        reasons.append("low_name_with_coord_override")
    if coord_override_accepted and combined < ACCEPTED_THRESHOLD:
        reasons.append("accepted_coord_override")
    if combined < UNCERTAIN_THRESHOLD:
        reasons.append("low_total_score")

    return {
        "name_score": round(name_score, 6),
        "coord_score": round(coord_score, 6),
        "combined_score": round(combined, 6),
        "exact_coord_set": exact_coord_set,
        "strict_exact_coord_set": bool(coord_meta.get("strict_exact_coord_set", False)),
        "tolerance_exact_coord_set": bool(coord_meta.get("tolerance_exact_coord_set", False)),
        "exact_match_axis_tolerance": coord_meta.get("exact_match_axis_tolerance"),
        "coord_override_accepted": coord_override_accepted,
        "low_name_with_exact_coords": low_name_with_exact_coords,
        "reason_codes": sorted(set(reasons)),
    }

def match_with_hungarian(
    manual_analyses: list[dict[str, Any]],
    auto_analyses: list[dict[str, Any]],
    coord_accept_override_threshold: float,
    converted_talairach_exact_axis_tolerance: float,
) -> tuple[list[dict[str, Any]], list[int]]:
    if not manual_analyses:
        return [], [a["index"] for a in auto_analyses]

    if not auto_analyses:
        out = []
        for m in manual_analyses:
            out.append(
                {
                    "manual_analysis_id": m["id"],
                    "manual_name": m["name"],
                    "manual_coord_count": len(m["points"]),
                    "best_auto_index": None,
                    "best_auto_analysis_id": None,
                    "best_auto_name": None,
                    "name_score": 0.0,
                    "coord_score": 0.0,
                    "combined_score": 0.0,
                    "match_status": "unmatched",
                    "exact_coord_set": False,
                    "strict_exact_coord_set": False,
                    "tolerance_exact_coord_set": False,
                    "exact_match_axis_tolerance": None,
                    "coord_override_accepted": False,
                    "low_name_with_exact_coords": False,
                    "reason_codes": ["no_auto_analyses_for_pmid"],
                    "manual_coordinates": [[float(x), float(y), float(z)] for x, y, z in m.get("points", [])],
                    "best_auto_coordinates": [],
                }
            )
        return out, []

    pair_scores: dict[tuple[int, int], dict[str, Any]] = {}
    matrix = np.zeros((len(manual_analyses), len(auto_analyses)), dtype=float)
    for i, m in enumerate(manual_analyses):
        for j, a in enumerate(auto_analyses):
            detail = score_pair(
                m,
                a,
                coord_accept_override_threshold=coord_accept_override_threshold,
                converted_talairach_exact_axis_tolerance=converted_talairach_exact_axis_tolerance,
            )
            pair_scores[(i, j)] = detail
            matrix[i, j] = detail["combined_score"]

    row_ind, col_ind = assign_pairs(matrix)
    mapping = {int(i): int(j) for i, j in zip(row_ind.tolist(), col_ind.tolist())}

    out: list[dict[str, Any]] = []
    for i, m in enumerate(manual_analyses):
        if i not in mapping:
            out.append(
                {
                    "manual_analysis_id": m["id"],
                    "manual_name": m["name"],
                    "manual_coord_count": len(m["points"]),
                    "best_auto_index": None,
                    "best_auto_analysis_id": None,
                    "best_auto_name": None,
                    "name_score": 0.0,
                    "coord_score": 0.0,
                    "combined_score": 0.0,
                    "match_status": "unmatched",
                    "exact_coord_set": False,
                    "strict_exact_coord_set": False,
                    "tolerance_exact_coord_set": False,
                    "exact_match_axis_tolerance": None,
                    "coord_override_accepted": False,
                    "low_name_with_exact_coords": False,
                    "reason_codes": ["unassigned_by_global_matching", "low_total_score"],
                    "manual_coordinates": [[float(x), float(y), float(z)] for x, y, z in m.get("points", [])],
                    "best_auto_coordinates": [],
                }
            )
            continue

        j = mapping[i]
        a = auto_analyses[j]
        d = pair_scores[(i, j)]
        out.append(
            {
                "manual_analysis_id": m["id"],
                "manual_name": m["name"],
                "manual_coord_count": len(m["points"]),
                "best_auto_index": a["index"],
                "best_auto_analysis_id": a["analysis_id"],
                "best_auto_name": a["name"],
                "name_score": d["name_score"],
                "coord_score": d["coord_score"],
                "combined_score": d["combined_score"],
                "match_status": status_from_detail(d),
                "exact_coord_set": bool(d.get("exact_coord_set", False)),
                "strict_exact_coord_set": bool(d.get("strict_exact_coord_set", False)),
                "tolerance_exact_coord_set": bool(d.get("tolerance_exact_coord_set", False)),
                "exact_match_axis_tolerance": d.get("exact_match_axis_tolerance"),
                "coord_override_accepted": bool(d.get("coord_override_accepted", False)),
                "low_name_with_exact_coords": bool(d.get("low_name_with_exact_coords", False)),
                "reason_codes": d["reason_codes"],
                "manual_coordinates": [[float(x), float(y), float(z)] for x, y, z in m.get("points", [])],
                "best_auto_coordinates": [[float(x), float(y), float(z)] for x, y, z in a.get("points", [])],
            }
        )

    assigned_auto_indices = {e["best_auto_index"] for e in out if e["best_auto_index"] is not None}
    unassigned_auto_indices = [a["index"] for a in auto_analyses if a["index"] not in assigned_auto_indices]
    return out, unassigned_auto_indices

def classify_study_match_category(accepted: int, manual_total: int) -> str:
    if manual_total <= 0:
        return "mixed"
    if accepted == manual_total:
        return "all_correct"
    if accepted == 0:
        return "all_incorrect"
    return "mixed"

def build_match_results_overall(
    manual_analyses_by_pmid: dict[str, list[dict[str, Any]]],
    manual_study_names_by_pmid: dict[str, str],
    auto_parsed_by_pmid: dict[str, list[dict[str, Any]]],
    coord_accept_override_threshold: float,
    decimal_manual_coordinate_handling: str,
    converted_talairach_exact_axis_tolerance: float,
) -> dict[str, Any]:
    valid_decimal_handling = {"exclude", "convert_to_talairach", "keep"}
    if decimal_manual_coordinate_handling not in valid_decimal_handling:
        raise ValueError(
            "decimal_manual_coordinate_handling must be one of: "
            + ", ".join(sorted(valid_decimal_handling))
        )
    if converted_talairach_exact_axis_tolerance < 0:
        raise ValueError("converted_talairach_exact_axis_tolerance must be >= 0.0")

    pmid_results: dict[str, dict[str, Any]] = {}
    unavailable_manual_decimal_pmids: list[str] = []
    converted_decimal_manual_analyses_total = 0

    manual_pmids = set(manual_analyses_by_pmid.keys())
    auto_pmids = set(auto_parsed_by_pmid.keys())
    overlap_pmids_all = sorted(manual_pmids & auto_pmids, key=lambda x: (len(x), x))
    excluded_manual_only_pmids = sorted(manual_pmids - auto_pmids, key=lambda x: (len(x), x))
    auto_only_pmids = sorted(auto_pmids - manual_pmids, key=lambda x: (len(x), x))

    for pmid in overlap_pmids_all:
        manual_analyses_original = manual_analyses_by_pmid.get(pmid, [])
        excluded_decimal_analyses: list[dict[str, Any]] = []
        converted_decimal_analyses: list[dict[str, Any]] = []
        manual_analyses: list[dict[str, Any]] = []
        for analysis in manual_analyses_original:
            has_decimal_coordinates = bool(analysis.get("has_decimal_coordinates", False))
            if not has_decimal_coordinates:
                manual_analyses.append(analysis)
                continue

            if decimal_manual_coordinate_handling == "exclude":
                excluded_decimal_analyses.append(analysis)
                continue

            if decimal_manual_coordinate_handling == "convert_to_talairach":
                converted_analysis = dict(analysis)
                converted_analysis["points"] = convert_coords_mni_to_talairach(analysis.get("points", []))
                converted_analysis["converted_from_decimal_mni_to_talairach"] = True
                converted_decimal_analyses.append(converted_analysis)
                manual_analyses.append(converted_analysis)
                continue

            manual_analyses.append(analysis)

        auto_analyses = auto_parsed_by_pmid.get(pmid, [])
        if (
            decimal_manual_coordinate_handling == "exclude"
            and manual_analyses_original
            and not manual_analyses
        ):
            unavailable_manual_decimal_pmids.append(pmid)
            continue

        matched_entries, unassigned_auto_indices = match_with_hungarian(
            manual_analyses,
            auto_analyses,
            coord_accept_override_threshold=coord_accept_override_threshold,
            converted_talairach_exact_axis_tolerance=converted_talairach_exact_axis_tolerance,
        )
        counts = defaultdict(int)
        for entry in matched_entries:
            counts[entry["match_status"]] += 1

        mean_combined = (
            sum(float(entry["combined_score"]) for entry in matched_entries) / len(matched_entries)
            if matched_entries
            else 0.0
        )

        pmid_results[pmid] = {
            "manual_missing_in_auto": False,
            "manual_analyses": matched_entries,
            "excluded_manual_analyses_decimal": [
                {
                    "id": str(analysis.get("id", "")),
                    "name": str(analysis.get("name", "")),
                    "coord_count": len(analysis.get("points", [])),
                }
                for analysis in excluded_decimal_analyses
            ],
            "converted_manual_analyses_decimal": [
                {
                    "id": str(analysis.get("id", "")),
                    "name": str(analysis.get("name", "")),
                    "coord_count": len(analysis.get("points", [])),
                }
                for analysis in converted_decimal_analyses
            ],
            "auto_analyses": [
                {
                    "index": int(a["index"]),
                    "analysis_id": str(a["analysis_id"]),
                    "name": str(a["name"]),
                    "coord_count": len(a.get("points", [])),
                    "coordinates": [[float(x), float(y), float(z)] for x, y, z in a.get("points", [])],
                }
                for a in auto_analyses
            ],
            "unassigned_auto_indices": unassigned_auto_indices,
            "pmid_summary": {
                "accepted": int(counts["accepted"]),
                "uncertain": int(counts["uncertain"]),
                "unmatched": int(counts["unmatched"]),
                "manual_analysis_count": len(matched_entries),
                "manual_analysis_count_original": len(manual_analyses_original),
                "excluded_manual_decimal_analysis_count": len(excluded_decimal_analyses),
                "converted_manual_decimal_analysis_count": len(converted_decimal_analyses),
                "all_manual_accepted": bool(matched_entries) and int(counts["accepted"]) == len(matched_entries),
                "mean_combined_score": round(mean_combined, 6),
            },
        }
        converted_decimal_manual_analyses_total += len(converted_decimal_analyses)

    all_entries = [entry for data in pmid_results.values() for entry in data["manual_analyses"]]
    status_counts = defaultdict(int)
    combined_scores = []
    perfect_pmids = 0
    category_counts = defaultdict(int)
    coord_override_accepted = 0
    low_name_coord_override_matches = 0
    for entry in all_entries:
        status_counts[entry["match_status"]] += 1
        combined_scores.append(float(entry["combined_score"]))
        if bool(entry.get("coord_override_accepted", False)) and float(entry.get("combined_score", 0.0)) < ACCEPTED_THRESHOLD:
            coord_override_accepted += 1
        if bool(entry.get("low_name_with_exact_coords", False)):
            low_name_coord_override_matches += 1
    for pmid, data in pmid_results.items():
        pmid_summary = data.get("pmid_summary", {})
        manual_count = int(pmid_summary.get("manual_analysis_count", 0))
        accepted_count = int(pmid_summary.get("accepted", 0))
        category = classify_study_match_category(accepted=accepted_count, manual_total=manual_count)
        pmid_summary["study_category"] = category
        data["study_name"] = manual_study_names_by_pmid.get(pmid, pmid)
        category_counts[category] += 1
        if bool(pmid_summary.get("all_manual_accepted", False)):
            perfect_pmids += 1

    combined_arr = np.array(combined_scores, dtype=float) if combined_scores else np.array([], dtype=float)
    summary_stats = {
        "mean_combined_score": float(np.mean(combined_arr)) if combined_arr.size else 0.0,
        "median_combined_score": float(np.median(combined_arr)) if combined_arr.size else 0.0,
        "p25_combined_score": float(np.percentile(combined_arr, 25)) if combined_arr.size else 0.0,
        "p75_combined_score": float(np.percentile(combined_arr, 75)) if combined_arr.size else 0.0,
    }

    return {
        "result_name": OVERALL_RESULT_NAME,
        "matching_policy": {
            "assignment": "one_to_one_hungarian",
            "coordinate_weight": COORD_WEIGHT,
            "name_weight": NAME_WEIGHT,
            "accepted_threshold": ACCEPTED_THRESHOLD,
            "uncertain_threshold": UNCERTAIN_THRESHOLD,
            "coordinate_space_handling": "ignore_space_labels_use_raw_xyz",
            "metric_truth_policy": "accepted_only",
            "pmid_scope_for_scoring": "overlap_only_manual_and_auto",
            "coord_accept_override": True,
            "coord_accept_override_threshold": coord_accept_override_threshold,
            "exact_coord_accept_override": False,
            "low_name_highlight_threshold": LOW_NAME_SCORE_HIGHLIGHT_THRESHOLD,
            "decimal_manual_coordinate_handling": decimal_manual_coordinate_handling,
            "converted_talairach_exact_axis_tolerance": converted_talairach_exact_axis_tolerance,
            "exclude_decimal_manual_coordinates": decimal_manual_coordinate_handling == "exclude",
            "convert_decimal_manual_coordinates_to_talairach": decimal_manual_coordinate_handling == "convert_to_talairach",
        },
        "pmids": pmid_results,
        "unavailable_manual_decimal_pmids": unavailable_manual_decimal_pmids,
        "missing_manual_pmids": [],
        "excluded_manual_only_pmids": excluded_manual_only_pmids,
        "auto_only_pmids": auto_only_pmids,
        "summary": {
            "manual_pmids": len(pmid_results),
            "missing_manual_pmids": 0,
            "manual_pmids_total": len(manual_pmids),
            "auto_pmids_total": len(auto_pmids),
            "overlap_pmids": len(pmid_results),
            "overlap_pmids_before_decimal_filter": len(overlap_pmids_all),
            "unavailable_manual_decimal_pmids": len(unavailable_manual_decimal_pmids),
            "converted_manual_decimal_analyses": int(converted_decimal_manual_analyses_total),
            "excluded_manual_only_pmids": len(excluded_manual_only_pmids),
            "auto_only_pmids": len(auto_only_pmids),
            "manual_analyses_total": len(all_entries),
            "accepted": int(status_counts["accepted"]),
            "uncertain": int(status_counts["uncertain"]),
            "unmatched": int(status_counts["unmatched"]),
            "accepted_coord_override": int(coord_override_accepted),
            "low_name_coord_override_matches": int(low_name_coord_override_matches),
            # Back-compat aliases.
            "accepted_exact_coord_override": int(coord_override_accepted),
            "low_name_exact_matches": int(low_name_coord_override_matches),
            "pmids_all_manual_accepted": int(perfect_pmids),
            "pmids_all_manual_accepted_rate": (float(perfect_pmids) / len(pmid_results)) if pmid_results else 0.0,
            "all_correct_pmids": int(category_counts["all_correct"]),
            "mixed_pmids": int(category_counts["mixed"]),
            "all_incorrect_pmids": int(category_counts["all_incorrect"]),
            **summary_stats,
        },
    }


def build_table_only_baseline_summary(
    *,
    manual_analyses_by_pmid: dict[str, list[dict[str, Any]]],
    manual_study_names_by_pmid: dict[str, str],
    table_auto_by_pmid: dict[str, list[dict[str, Any]]],
    table_source_info: dict[str, Any],
    coord_accept_override_threshold: float,
    decimal_manual_coordinate_handling: str,
    converted_talairach_exact_axis_tolerance: float,
) -> dict[str, Any]:
    if not table_source_info.get("available") or not table_auto_by_pmid:
        return {
            "available": False,
            "reason": table_source_info.get("reason", "No table-level coordinate source available."),
            "source": table_source_info.get("source", ""),
            "manual_analyses_total": 0,
            "accepted": 0,
            "uncertain": 0,
            "unmatched": 0,
            "matched_count": 0,
            "matched_pct": None,
        }

    table_match_result = build_match_results_overall(
        manual_analyses_by_pmid=manual_analyses_by_pmid,
        manual_study_names_by_pmid=manual_study_names_by_pmid,
        auto_parsed_by_pmid=table_auto_by_pmid,
        coord_accept_override_threshold=coord_accept_override_threshold,
        decimal_manual_coordinate_handling=decimal_manual_coordinate_handling,
        converted_talairach_exact_axis_tolerance=converted_talairach_exact_axis_tolerance,
    )
    summary = table_match_result.get("summary", {})
    manual_total = int(summary.get("manual_analyses_total", 0))
    accepted = int(summary.get("accepted", 0))
    uncertain = int(summary.get("uncertain", 0))
    unmatched = int(summary.get("unmatched", 0))
    matched_count = accepted + uncertain
    matched_pct = (float(matched_count) / manual_total) if manual_total else None
    return {
        "available": True,
        "source": table_source_info.get("source", ""),
        "pmids_with_table_coordinates": int(table_source_info.get("pmids_with_table_coordinates", 0)),
        "table_units": int(table_source_info.get("table_units", 0)),
        "coordinate_rows": int(table_source_info.get("coordinate_rows", 0)),
        "manual_analyses_total": manual_total,
        "accepted": accepted,
        "uncertain": uncertain,
        "unmatched": unmatched,
        "matched_count": matched_count,
        "matched_pct": matched_pct,
        "overlap_pmids": int(summary.get("overlap_pmids", 0)),
        "manual_pmids_total": int(summary.get("manual_pmids_total", 0)),
        "auto_pmids_total": int(summary.get("auto_pmids_total", 0)),
        "excluded_manual_only_pmids": int(summary.get("excluded_manual_only_pmids", 0)),
        "auto_only_pmids": int(summary.get("auto_only_pmids", 0)),
        "decimal_manual_coordinate_handling": decimal_manual_coordinate_handling,
        "policy": "Raw extracted coordinates grouped only by source table; fuzzy matched to manual analyses.",
    }

def render_matching_summary_html(match_result: dict[str, Any]) -> str:
    summary = match_result.get("summary", {})
    table_baseline = match_result.get("table_only_baseline", {})
    review_adjustment = match_result.get("parser_review_adjustment", {})
    unavailable_manual_decimal_pmids = int(summary.get("unavailable_manual_decimal_pmids", 0))
    overlap_pmids = int(summary.get("overlap_pmids", 0))
    manual_total = int(summary.get("manual_analyses_total", 0))
    accepted = int(summary.get("accepted", 0))
    uncertain = int(summary.get("uncertain", 0))
    unmatched = int(summary.get("unmatched", 0))
    accepted_rate = (accepted / manual_total) if manual_total else 0.0
    perfect_pmids = int(summary.get("pmids_all_manual_accepted", 0))
    perfect_pmid_rate = float(summary.get("pmids_all_manual_accepted_rate", 0.0))
    excluded_manual_only_pmids = int(summary.get("excluded_manual_only_pmids", 0))
    auto_only_pmids = int(summary.get("auto_only_pmids", 0))
    all_correct_pmids = int(summary.get("all_correct_pmids", 0))
    mixed_pmids = int(summary.get("mixed_pmids", 0))
    all_incorrect_pmids = int(summary.get("all_incorrect_pmids", 0))
    if isinstance(table_baseline, dict) and table_baseline.get("available"):
        table_baseline_html = (
            "<p><strong>Table-only baseline matched %:</strong> "
            f"{float(table_baseline.get('matched_pct', 0.0)):.3f} "
            f"({int(table_baseline.get('matched_count', 0))}/"
            f"{int(table_baseline.get('manual_analyses_total', 0))}) | "
            f"<strong>Table units:</strong> {int(table_baseline.get('table_units', 0))} | "
            f"<strong>Coordinate rows:</strong> {int(table_baseline.get('coordinate_rows', 0))}</p>"
        )
    else:
        table_reason = ""
        if isinstance(table_baseline, dict) and table_baseline.get("reason"):
            table_reason = f" ({escape(str(table_baseline.get('reason', '')))})"
        table_baseline_html = f"<p><strong>Table-only baseline:</strong> unavailable{table_reason}</p>"
    review_adjustment_html = ""
    if isinstance(review_adjustment, dict) and review_adjustment.get("enabled"):
        review_adjustment_html = (
            "<p><strong>Human-review adjustment:</strong> "
            f"{int(review_adjustment.get('credited_as_accepted', 0))} "
            "benchmark/matching/expected-difference analyses credited as accepted; "
            f"{int(review_adjustment.get('excluded_non_parser_evaluable', 0))} "
            "source/scope/uncertain analyses excluded from the parser denominator; "
            f"{int(review_adjustment.get('confirmed_parser_misses', 0))} "
            "confirmed parser misses retained as evaluable outcomes. "
            "Raw statuses remain available in the JSON output.</p>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fuzzy Matching Summary</title>
  <style>
    body {{ font-family: "IBM Plex Sans", "Segoe UI", sans-serif; margin: 1rem; background: #f7f6f2; color: #1d2730; }}
    header, section {{ background: #fff; border: 1px solid #d8dde3; border-radius: 10px; padding: 0.9rem; margin-bottom: 1rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #d8dde3; padding: 0.45rem; text-align: left; vertical-align: top; }}
    th {{ background: #edf2f5; }}
    a {{ color: #0e4f85; }}
  </style>
</head>
<body>
  <header>
    <h1>Overall Fuzzy Matching Summary</h1>
    <p>Coordinate-first matching (70%) + name similarity (30%), one-to-one Hungarian assignment, accepted &gt;= 0.75, uncertain &gt;= 0.55. Metrics include only overlap PMIDs (manual ∩ auto).</p>
    <p><strong>Overlap PMIDs:</strong> {overlap_pmids} |
       <strong>Manual analyses (overlap only):</strong> {manual_total} |
       <strong>Accepted:</strong> {accepted} |
       <strong>Uncertain:</strong> {uncertain} |
       <strong>Unmatched:</strong> {unmatched} |
       <strong>Accepted rate:</strong> {accepted_rate:.3f}</p>
    <p><strong>PMIDs with all manual analyses accepted:</strong> {perfect_pmids} |
       <strong>Perfect PMID rate:</strong> {perfect_pmid_rate:.3f}</p>
    <p><strong>Study categories:</strong> All correct={all_correct_pmids} | Mixed={mixed_pmids} | All incorrect={all_incorrect_pmids}</p>
    <p><strong>Unavailable Manual Studies (decimal coordinates):</strong> {unavailable_manual_decimal_pmids}</p>
    <p><strong>Excluded manual-only PMIDs:</strong> {excluded_manual_only_pmids} |
       <strong>Auto-only PMIDs:</strong> {auto_only_pmids}</p>
    {review_adjustment_html}
    {table_baseline_html}
  </header>

  <section>
    <h2>Score Distribution</h2>
    <table>
      <thead>
        <tr>
          <th>Mean Score</th>
          <th>P25 Score</th>
          <th>Median Score</th>
          <th>P75 Score</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>{float(summary.get('mean_combined_score', 0.0)):.3f}</td>
          <td>{float(summary.get('p25_combined_score', 0.0)):.3f}</td>
          <td>{float(summary.get('median_combined_score', 0.0)):.3f}</td>
          <td>{float(summary.get('p75_combined_score', 0.0)):.3f}</td>
        </tr>
      </tbody>
    </table>
  </section>
</body>
</html>
"""

def render_detailed_study_review_html(match_result: dict[str, Any]) -> str:
    pmids = match_result.get("pmids", {})
    summary = match_result.get("summary", {})
    unavailable_manual_decimal_pmids = sorted(
        [str(pmid) for pmid in match_result.get("unavailable_manual_decimal_pmids", [])],
        key=lambda value: (len(value), value),
    )
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "all_correct": [],
        "mixed": [],
        "all_incorrect": [],
    }

    for pmid in sorted(pmids.keys(), key=lambda value: (len(value), value)):
        data = pmids[pmid]
        category = str(data.get("pmid_summary", {}).get("study_category", "mixed"))
        if category not in grouped:
            category = "mixed"
        grouped[category].append((pmid, data))

    category_labels = {
        "all_correct": "All correct",
        "mixed": "Mixed",
        "all_incorrect": "All incorrect",
    }
    row_classes = {
        "accepted": "st-accepted",
        "uncertain": "st-uncertain",
        "unmatched": "st-unmatched",
    }

    def render_study_card(pmid: str, data: dict[str, Any]) -> str:
        manual_rows = data.get("manual_analyses", [])
        auto_rows = data.get("auto_analyses", [])
        summary_row = data.get("pmid_summary", {})
        study_name = clean_text(data.get("study_name") or pmid)

        auto_match_by_index: dict[int, dict[str, Any]] = {}
        for manual in manual_rows:
            idx = manual.get("best_auto_index")
            if idx is not None:
                auto_match_by_index[int(idx)] = manual

        manual_table_rows: list[str] = []
        for manual in manual_rows:
            status = str(manual.get("match_status", "unmatched"))
            css_class = row_classes.get(status, "st-unmatched")
            reason_codes = ", ".join(str(code) for code in manual.get("reason_codes", []))
            manual_table_rows.append(
                "<tr class=\"{cls}\">"
                "<td>{manual_id}</td>"
                "<td>{manual_name}</td>"
                "<td>{coord_count}</td>"
                "<td>{status}</td>"
                "<td>{auto_id}</td>"
                "<td>{auto_name}</td>"
                "<td>{score:.3f}</td>"
                "<td>{reasons}</td>"
                "</tr>".format(
                    cls=css_class,
                    manual_id=escape(str(manual.get("manual_analysis_id", ""))),
                    manual_name=escape(str(manual.get("manual_name", ""))),
                    coord_count=int(manual.get("manual_coord_count", 0)),
                    status=escape(status),
                    auto_id=escape(str(manual.get("best_auto_analysis_id") or "")),
                    auto_name=escape(str(manual.get("best_auto_name") or "")),
                    score=float(manual.get("combined_score", 0.0)),
                    reasons=escape(reason_codes),
                )
            )

        auto_table_rows: list[str] = []
        for auto in auto_rows:
            idx = int(auto.get("index", -1))
            linked_manual = auto_match_by_index.get(idx)
            if linked_manual is None:
                status = "not_matched"
                css_class = "st-auto-unmatched"
                linked_manual_id = ""
                linked_manual_name = ""
                score = ""
            else:
                status = str(linked_manual.get("match_status", "unmatched"))
                css_class = row_classes.get(status, "st-unmatched")
                linked_manual_id = str(linked_manual.get("manual_analysis_id", ""))
                linked_manual_name = str(linked_manual.get("manual_name", ""))
                score = f"{float(linked_manual.get('combined_score', 0.0)):.3f}"
            auto_table_rows.append(
                "<tr class=\"{cls}\">"
                "<td>{index}</td>"
                "<td>{auto_id}</td>"
                "<td>{auto_name}</td>"
                "<td>{coord_count}</td>"
                "<td>{linked_manual_id}</td>"
                "<td>{linked_manual_name}</td>"
                "<td>{status}</td>"
                "<td>{score}</td>"
                "</tr>".format(
                    cls=css_class,
                    index=idx,
                    auto_id=escape(str(auto.get("analysis_id", ""))),
                    auto_name=escape(str(auto.get("name", ""))),
                    coord_count=int(auto.get("coord_count", 0)),
                    linked_manual_id=escape(linked_manual_id),
                    linked_manual_name=escape(linked_manual_name),
                    status=escape(status),
                    score=escape(score),
                )
            )

        return (
            "<details class=\"study-card\">"
            "<summary><strong>PMID {pmid}</strong> | {study_name} | accepted={accepted} uncertain={uncertain} unmatched={unmatched} manual={manual_total} auto={auto_total}</summary>"
            "<div class=\"split\">"
            "<section>"
            "<h4>Manual Analyses</h4>"
            "<table><thead><tr><th>Manual ID</th><th>Manual Name</th><th>Coords</th><th>Status</th><th>Matched Auto ID</th><th>Matched Auto Name</th><th>Combined</th><th>Reasons</th></tr></thead><tbody>{manual_rows}</tbody></table>"
            "</section>"
            "<section>"
            "<h4>Automated Analyses</h4>"
            "<table><thead><tr><th>Index</th><th>Auto ID</th><th>Auto Name</th><th>Coords</th><th>Linked Manual ID</th><th>Linked Manual Name</th><th>Status</th><th>Combined</th></tr></thead><tbody>{auto_rows}</tbody></table>"
            "</section>"
            "</div>"
            "</details>"
        ).format(
            pmid=escape(pmid),
            study_name=escape(study_name),
            accepted=int(summary_row.get("accepted", 0)),
            uncertain=int(summary_row.get("uncertain", 0)),
            unmatched=int(summary_row.get("unmatched", 0)),
            manual_total=int(summary_row.get("manual_analysis_count", 0)),
            auto_total=len(auto_rows),
            manual_rows="".join(manual_table_rows) if manual_table_rows else "<tr><td colspan=\"8\">No manual analyses.</td></tr>",
            auto_rows="".join(auto_table_rows) if auto_table_rows else "<tr><td colspan=\"8\">No automated analyses.</td></tr>",
        )

    sections: list[str] = []
    for category_key in ("all_correct", "mixed", "all_incorrect"):
        studies = grouped[category_key]
        cards = "".join(render_study_card(pmid, data) for pmid, data in studies)
        sections.append(
            "<section id=\"cat-{cat}\">"
            "<h2>{label} ({count})</h2>"
            "{cards}"
            "</section>".format(
                cat=category_key,
                label=escape(category_labels[category_key]),
                count=len(studies),
                cards=cards or "<p>No studies in this category.</p>",
            )
        )
    unavailable_section = ""
    unavailable_nav_link = ""
    if unavailable_manual_decimal_pmids:
        unavailable_nav_link = (
            f"<a href=\"#cat-unavailable\">Unavailable ({len(unavailable_manual_decimal_pmids)})</a>"
        )
        unavailable_section = (
            "<section id=\"cat-unavailable\">"
            "<h2>Unavailable Manual Studies (decimal coordinates): [{pmids}]</h2>"
            "</section>"
        ).format(pmids=", ".join(escape(pmid) for pmid in unavailable_manual_decimal_pmids))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fuzzy Matching Study Review</title>
  <style>
    body {{ font-family: "IBM Plex Sans", "Segoe UI", sans-serif; margin: 1rem; background: #f7f6f2; color: #1d2730; }}
    header, nav, section, .study-card {{ background: #fff; border: 1px solid #d8dde3; border-radius: 10px; padding: 0.9rem; margin-bottom: 1rem; }}
    nav a {{ margin-right: 0.6rem; color: #0e4f85; text-decoration: none; }}
    .study-card > summary {{ cursor: pointer; }}
    .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.8rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
    th, td {{ border: 1px solid #d8dde3; padding: 0.35rem; text-align: left; vertical-align: top; }}
    th {{ background: #edf2f5; }}
    .st-accepted {{ background: #e6f7eb; }}
    .st-uncertain {{ background: #fff7dd; }}
    .st-unmatched {{ background: #fdecec; }}
    .st-auto-unmatched {{ background: #f1f2f4; }}
    @media (max-width: 1100px) {{ .split {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Fuzzy Matching Study Review</h1>
    <p><strong>Overlap PMIDs:</strong> {int(summary.get("overlap_pmids", 0))} |
       <strong>Manual analyses:</strong> {int(summary.get("manual_analyses_total", 0))} |
       <strong>Accepted:</strong> {int(summary.get("accepted", 0))} |
       <strong>Uncertain:</strong> {int(summary.get("uncertain", 0))} |
       <strong>Unmatched:</strong> {int(summary.get("unmatched", 0))}</p>
    <p>Study categories are based on manual analysis match status: All correct (all manual accepted), Mixed (some accepted), All incorrect (none accepted).</p>
  </header>
  <nav>
    <a href="#cat-all_correct">All correct ({int(summary.get("all_correct_pmids", 0))})</a>
    <a href="#cat-mixed">Mixed ({int(summary.get("mixed_pmids", 0))})</a>
    <a href="#cat-all_incorrect">All incorrect ({int(summary.get("all_incorrect_pmids", 0))})</a>
    {unavailable_nav_link}
  </nav>
  {"".join(sections)}
  {unavailable_section}
</body>
</html>
"""

def extract_body_content(html_doc: str) -> str:
    match = re.search(r"<body[^>]*>(.*)</body>", html_doc, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return html_doc

def render_combined_report_html(
    match_result: dict[str, Any],
    pubget_by_pmid: dict[str, dict[str, Any]] | None = None,
) -> str:
    summary = match_result.get("summary", {})
    policy = match_result.get("matching_policy", {})
    coord_override_threshold = float(policy.get("coord_accept_override_threshold", 0.9))
    decimal_handling_mode = str(policy.get("decimal_manual_coordinate_handling", "exclude"))
    converted_talairach_exact_axis_tolerance = float(
        policy.get("converted_talairach_exact_axis_tolerance", 1.0)
    )
    pmids = match_result.get("pmids", {})
    unavailable_manual_decimal_pmids = sorted(
        [str(pmid) for pmid in match_result.get("unavailable_manual_decimal_pmids", [])],
        key=lambda value: (len(value), value),
    )
    decimal_handling_label = {
        "exclude": "exclude non-zero decimal manual coordinates",
        "convert_to_talairach": "convert decimal manual coordinates MNI→Talairach",
        "keep": "keep decimal manual coordinates as-is",
    }.get(decimal_handling_mode, decimal_handling_mode)
    talairach_tolerance_note = ""
    if decimal_handling_mode == "convert_to_talairach":
        talairach_tolerance_note = (
            f" | <strong>Converted exact-set axis tolerance:</strong> ±{converted_talairach_exact_axis_tolerance:.3f}"
        )
    pubget_by_pmid = pubget_by_pmid or {}
    table_html_cache: dict[str, str] = {}
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "all_correct": [],
        "mixed": [],
        "all_incorrect": [],
    }

    for pmid in sorted(pmids.keys(), key=lambda value: (len(value), value)):
        data = pmids[pmid]
        category = str(data.get("pmid_summary", {}).get("study_category", "mixed"))
        if category not in grouped:
            category = "mixed"
        grouped[category].append((pmid, data))

    all_correct_total = len(grouped["all_correct"])
    all_correct_exact_count = sum(
        1
        for _pmid, data in grouped["all_correct"]
        if int(data.get("pmid_summary", {}).get("manual_analysis_count", 0)) == len(data.get("auto_analyses", []))
    )

    row_classes = {
        "accepted": "st-accepted",
        "uncertain": "st-uncertain",
        "unmatched": "st-unmatched",
    }

    def render_pubget_section(pmid: str) -> str:
        resource = pubget_by_pmid.get(str(pmid))
        if not resource:
            return ""

        pmcid = str(resource.get("pmcid") or "")
        pmc_url = str(resource.get("pmc_url") or "")
        article_xml_file = str(resource.get("article_xml_file") or "")
        tables = resource.get("tables", [])

        table_blocks: list[str] = []
        for table in tables:
            table_label = str(table.get("table_label") or table.get("table_id") or "Table")
            table_id = str(table.get("table_id") or "")
            table_caption = str(table.get("table_caption") or "")
            table_foot = str(table.get("table_foot") or "")
            table_data_file = str(table.get("table_data_file") or "")
            table_csv_path = Path(str(table.get("table_csv_path") or ""))
            n_header_rows = int(table.get("n_header_rows") or 1)
            cache_key = f"{table_csv_path}|{n_header_rows}"
            if cache_key not in table_html_cache:
                table_html_cache[cache_key] = render_csv_table_html(table_csv_path, n_header_rows)

            table_blocks.append(
                "<details class=\"table-accordion\">"
                "<summary>{label}{table_id_suffix}</summary>"
                "{caption}"
                "{foot}"
                "<p class=\"resource-note\"><strong>Source:</strong> <code>{source}</code></p>"
                "{table_html}"
                "</details>".format(
                    label=escape(table_label),
                    table_id_suffix=f" ({escape(table_id)})" if table_id else "",
                    caption=f"<p><strong>Caption:</strong> {escape(table_caption)}</p>" if table_caption else "",
                    foot=f"<p><strong>Footnote:</strong> {escape(table_foot)}</p>" if table_foot else "",
                    source=escape(table_data_file),
                    table_html=table_html_cache[cache_key],
                )
            )

        table_html = "".join(table_blocks) if table_blocks else "<p>No extracted tables found.</p>"
        article_xml_line = (
            f"<p class=\"resource-note\"><strong>Article XML:</strong> <code>{escape(article_xml_file)}</code></p>"
            if article_xml_file
            else ""
        )
        pmc_link = (
            f"<a href=\"{escape(pmc_url)}\" target=\"_blank\" rel=\"noopener noreferrer\">PubMedCentral full text</a>"
            if pmc_url
            else ""
        )
        return (
            "<details class=\"inner-accordion\">"
            "<summary>Pubget full text + extracted tables</summary>"
            "<div class=\"resource-box\">"
            "<p><strong>PMCID:</strong> PMC{pmcid} | {pmc_link}</p>"
            "{article_xml_line}"
            "<p><strong>Extracted tables:</strong> {table_count}</p>"
            "{table_html}"
            "</div>"
            "</details>"
        ).format(
            pmcid=escape(pmcid),
            pmc_link=pmc_link or "N/A",
            article_xml_line=article_xml_line,
            table_count=len(tables),
            table_html=table_html,
        )

    def render_study_card(pmid: str, data: dict[str, Any]) -> str:
        def coords_text(coords: list[Any]) -> str:
            if not coords:
                return "No coordinates extracted."
            lines: list[str] = []
            for item in coords:
                if isinstance(item, (list, tuple)) and len(item) == 3:
                    try:
                        x = float(item[0])
                        y = float(item[1])
                        z = float(item[2])
                        lines.append(f"{x:.1f}, {y:.1f}, {z:.1f}")
                    except Exception:
                        continue
            return "\n".join(lines) if lines else "No coordinates extracted."

        def render_coord_toggle(label: str, coords: list[Any]) -> str:
            return (
                "<details class=\"analysis-coords\">"
                "<summary>{label} ({count})</summary>"
                "<pre class=\"coord-list\">{coord_text}</pre>"
                "</details>".format(
                    label=escape(label),
                    count=len(coords),
                    coord_text=escape(coords_text(coords)),
                )
            )

        manual_rows = data.get("manual_analyses", [])
        auto_rows = data.get("auto_analyses", [])
        pmid_summary = data.get("pmid_summary", {})
        study_category = str(pmid_summary.get("study_category", "mixed"))
        needs_human_review = study_category in {"mixed", "all_incorrect"}
        study_name = clean_text(data.get("study_name") or pmid)
        pubget_resource = pubget_by_pmid.get(str(pmid))
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{escape(str(pmid))}/"
        pmc_link_inline = ""
        if pubget_resource:
            pmc_url_inline = str(pubget_resource.get("pmc_url") or "")
            if pmc_url_inline:
                pmc_link_inline = (
                    f" | <a href=\"{escape(pmc_url_inline)}\" target=\"_blank\" rel=\"noopener noreferrer\">PMC full text</a>"
                )
        manual_total = int(pmid_summary.get("manual_analysis_count", 0))
        auto_total = len(auto_rows)
        delta = auto_total - manual_total
        if delta > 0:
            delta_label = f"auto>manual = {delta}"
        elif delta < 0:
            delta_label = f"manual>auto = {abs(delta)}"
        else:
            delta_label = "auto=manual = 0"

        review_section = ""
        if needs_human_review:
            reason_options_html = "".join(
                (
                    "<label class=\"review-reason-option\">"
                    "<input type=\"checkbox\" data-role=\"extraction-reason\" value=\"{value}\"> {label}"
                    "</label>"
                ).format(value=escape(value), label=escape(label))
                for value, label in HUMAN_REVIEW_EXTRACTION_REASONS
            )
            review_radio_name = f"review-decision-{pmid}"
            review_section = (
                "<details class=\"inner-accordion review-annotation\" "
                "data-pmid=\"{pmid}\" "
                "data-study-name=\"{study_name}\" "
                "data-study-category=\"{study_category}\">"
                "<summary>Human adjudication</summary>"
                "<div class=\"review-box\">"
                "<p class=\"resource-note\">Choose <strong>Annotation Error</strong> or <strong>Extraction Error</strong>, "
                "then optionally add notes. Saved automatically in browser localStorage.</p>"
                "<div class=\"review-decision-group\">"
                "<label><input type=\"radio\" data-role=\"review-decision\" name=\"{radio_name}\" value=\"annotation_error\"> Annotation Error</label>"
                "<label><input type=\"radio\" data-role=\"review-decision\" name=\"{radio_name}\" value=\"extraction_error\"> Extraction Error</label>"
                "<label><input type=\"radio\" data-role=\"review-decision\" name=\"{radio_name}\" value=\"\"> Clear selection</label>"
                "</div>"
                "<div class=\"review-reasons\" data-role=\"extraction-reasons\" hidden>"
                "<p><strong>Extraction error reasons</strong></p>"
                "{reason_options_html}"
                "</div>"
                "<label class=\"review-note-label\" for=\"review-note-{pmid}\"><strong>Note</strong></label>"
                "<textarea id=\"review-note-{pmid}\" data-role=\"review-note\" rows=\"3\" "
                "placeholder=\"Add evidence, rationale, or follow-up notes...\"></textarea>"
                "<p class=\"resource-note\" data-role=\"review-status\">Not reviewed yet.</p>"
                "</div>"
                "</details>"
            ).format(
                pmid=escape(pmid),
                study_name=escape(study_name),
                study_category=escape(study_category),
                radio_name=escape(review_radio_name),
                reason_options_html=reason_options_html,
            )

        auto_match_by_index: dict[int, dict[str, Any]] = {}
        for manual in manual_rows:
            idx = manual.get("best_auto_index")
            if idx is not None:
                auto_match_by_index[int(idx)] = manual

        manual_row_html: list[str] = []
        for manual in manual_rows:
            status = str(manual.get("match_status", "unmatched"))
            css_class = row_classes.get(status, "st-unmatched")
            reasons = ", ".join(str(code) for code in manual.get("reason_codes", []))
            manual_coords = manual.get("manual_coordinates", [])
            name_score_cell_class = "score-discrepancy" if bool(manual.get("low_name_with_exact_coords", False)) else ""
            name_score_title = (
                " title=\"High coordinate score override accepted this match, but name similarity is low.\""
                if bool(manual.get("low_name_with_exact_coords", False))
                else ""
            )
            manual_row_html.append(
                "<tr class=\"{cls}\">"
                "<td>{manual_id}</td>"
                "<td>{manual_name}</td>"
                "<td>{coord_count}</td>"
                "<td>{status}</td>"
                "<td>{auto_id}</td>"
                "<td>{auto_name}</td>"
                "<td class=\"{name_score_cell_class}\"{name_score_title}>{name_score:.3f}</td>"
                "<td>{coord_score:.3f}</td>"
                "<td>{combined:.3f}</td>"
                "<td>{reasons}</td>"
                "</tr>".format(
                    cls=css_class,
                    manual_id=render_coord_toggle(str(manual.get("manual_analysis_id", "")), manual_coords),
                    manual_name=escape(str(manual.get("manual_name", ""))),
                    coord_count=int(manual.get("manual_coord_count", 0)),
                    status=escape(status),
                    auto_id=escape(str(manual.get("best_auto_analysis_id") or "")),
                    auto_name=escape(str(manual.get("best_auto_name") or "")),
                    name_score_cell_class=name_score_cell_class,
                    name_score_title=name_score_title,
                    name_score=float(manual.get("name_score", 0.0)),
                    coord_score=float(manual.get("coord_score", 0.0)),
                    combined=float(manual.get("combined_score", 0.0)),
                    reasons=escape(reasons),
                )
            )

        auto_row_html: list[str] = []
        for auto in auto_rows:
            idx = int(auto.get("index", -1))
            auto_coords = auto.get("coordinates", [])
            linked_manual = auto_match_by_index.get(idx)
            if linked_manual is None:
                status = "not_matched"
                css_class = "st-auto-unmatched"
                linked_id = ""
                linked_name = ""
                name_score = ""
                coord_score = ""
                combined = ""
            else:
                status = str(linked_manual.get("match_status", "unmatched"))
                css_class = row_classes.get(status, "st-unmatched")
                linked_id = str(linked_manual.get("manual_analysis_id", ""))
                linked_name = str(linked_manual.get("manual_name", ""))
                name_score = f"{float(linked_manual.get('name_score', 0.0)):.3f}"
                coord_score = f"{float(linked_manual.get('coord_score', 0.0)):.3f}"
                combined = f"{float(linked_manual.get('combined_score', 0.0)):.3f}"
            name_score_cell_class = (
                "score-discrepancy"
                if (linked_manual is not None and bool(linked_manual.get("low_name_with_exact_coords", False)))
                else ""
            )
            name_score_title = (
                " title=\"High coordinate score override accepted this match, but name similarity is low.\""
                if (linked_manual is not None and bool(linked_manual.get("low_name_with_exact_coords", False)))
                else ""
            )

            auto_row_html.append(
                "<tr class=\"{cls}\">"
                "<td>{index}</td>"
                "<td>{auto_id}</td>"
                "<td>{auto_name}</td>"
                "<td>{coord_count}</td>"
                "<td>{linked_id}</td>"
                "<td>{linked_name}</td>"
                "<td>{status}</td>"
                "<td class=\"{name_score_cell_class}\"{name_score_title}>{name_score}</td>"
                "<td>{coord_score}</td>"
                "<td>{combined}</td>"
                "</tr>".format(
                    cls=css_class,
                    index=idx,
                    auto_id=render_coord_toggle(str(auto.get("analysis_id", "")), auto_coords),
                    auto_name=escape(str(auto.get("name", ""))),
                    coord_count=int(auto.get("coord_count", 0)),
                    linked_id=escape(linked_id),
                    linked_name=escape(linked_name),
                    status=escape(status),
                    name_score_cell_class=name_score_cell_class,
                    name_score_title=name_score_title,
                    name_score=escape(name_score),
                    coord_score=escape(coord_score),
                    combined=escape(combined),
                )
            )

        return (
            "<details class=\"doc-card\" data-pmid=\"{pmid}\" data-study-category=\"{study_category}\">"
            "<summary><strong>PMID {pmid}</strong> | {study_name} | accepted={accepted} uncertain={uncertain} unmatched={unmatched} manual={manual_total} auto={auto_total} | <strong>{delta_label}</strong></summary>"
            "<p class=\"doc-links\"><a href=\"{pubmed_url}\" target=\"_blank\" rel=\"noopener noreferrer\">PubMed</a>{pmc_link_inline}</p>"
            "{review_section}"
            "<details class=\"inner-accordion\" open>"
            "<summary>Manual analyses</summary>"
            "<div class=\"table-wrap\">"
            "<table><thead><tr><th>Manual ID</th><th>Manual Name</th><th>Coords</th><th>Status</th><th>Matched Auto ID</th><th>Matched Auto Name</th><th>Name Score</th><th>Coord Score</th><th>Combined</th><th>Reason Codes</th></tr></thead><tbody>{manual_rows}</tbody></table>"
            "</div>"
            "</details>"
            "<details class=\"inner-accordion\" open>"
            "<summary>Automated analyses</summary>"
            "<div class=\"table-wrap\">"
            "<table><thead><tr><th>Index</th><th>Auto ID</th><th>Auto Name</th><th>Coords</th><th>Linked Manual ID</th><th>Linked Manual Name</th><th>Status</th><th>Name Score</th><th>Coord Score</th><th>Combined</th></tr></thead><tbody>{auto_rows}</tbody></table>"
            "</div>"
            "</details>"
            "{pubget_section}"
            "</details>"
        ).format(
            pmid=escape(pmid),
            study_category=escape(study_category),
            study_name=escape(study_name),
            pubmed_url=pubmed_url,
            pmc_link_inline=pmc_link_inline,
            accepted=int(pmid_summary.get("accepted", 0)),
            uncertain=int(pmid_summary.get("uncertain", 0)),
            unmatched=int(pmid_summary.get("unmatched", 0)),
            manual_total=manual_total,
            auto_total=auto_total,
            delta_label=escape(delta_label),
            review_section=review_section,
            manual_rows="".join(manual_row_html) if manual_row_html else "<tr><td colspan=\"10\">No manual analyses.</td></tr>",
            auto_rows="".join(auto_row_html) if auto_row_html else "<tr><td colspan=\"10\">No automated analyses.</td></tr>",
            pubget_section=render_pubget_section(pmid),
        )

    bucket_specs = [
        ("all_correct", "All Correct", "bucket-all-correct"),
        ("mixed", "Mixed", "bucket-mixed"),
        ("all_incorrect", "All Incorrect", "bucket-all-incorrect"),
    ]
    bucket_html: list[str] = []
    for key, label, sid in bucket_specs:
        cards = "".join(render_study_card(pmid, data) for pmid, data in grouped[key])
        bucket_extra = ""
        if key == "all_correct":
            bucket_extra = (
                "<p><strong>Exact same # of analyses within All Correct:</strong> "
                f"{all_correct_exact_count} / {all_correct_total}</p>"
            )
        open_attr = " open" if key != "all_correct" else ""
        bucket_html.append(
            "<section id=\"{sid}\">"
            "<details class=\"bucket\"{open_attr}>"
            "<summary><h2>{label} ({count})</h2></summary>"
            "{bucket_extra}"
            "{cards}"
            "</details>"
            "</section>".format(
                sid=sid,
                open_attr=open_attr,
                label=escape(label),
                count=len(grouped[key]),
                bucket_extra=bucket_extra,
                cards=cards or "<p>No studies in this category.</p>",
            )
        )

    unavailable_section_html = ""
    unavailable_nav_link = ""
    if unavailable_manual_decimal_pmids:
        unavailable_nav_link = (
            f"<a href=\"#bucket-unavailable\">Unavailable Manual Studies ({len(unavailable_manual_decimal_pmids)})</a>"
        )
        unavailable_pmid_text = ", ".join(escape(pmid) for pmid in unavailable_manual_decimal_pmids)
        unavailable_pmid_chips = "".join(
            (
                "<a class=\"pmid-chip\" href=\"https://pubmed.ncbi.nlm.nih.gov/{pmid}/\" "
                "target=\"_blank\" rel=\"noopener noreferrer\">PMID {pmid}</a>"
            ).format(pmid=escape(pmid))
            for pmid in unavailable_manual_decimal_pmids
        )
        unavailable_section_html = (
            "<section id=\"bucket-unavailable\">"
            "<details class=\"bucket\" open>"
            "<summary><h2>Unavailable Manual Studies (decimal coordinates) ({count})</h2></summary>"
            "<p class=\"resource-note\">These studies were excluded from matching because all manual analyses "
            "used non-zero decimal coordinates and are likely coordinate-converted (not raw extracted values).</p>"
            "<div class=\"pmid-chip-list\">{chips}</div>"
            "<details class=\"unavailable-raw\">"
            "<summary>Raw PMID list</summary>"
            "<code>[{pmids}]</code>"
            "</details>"
            "</details>"
            "</section>"
        ).format(
            count=len(unavailable_manual_decimal_pmids),
            chips=unavailable_pmid_chips,
            pmids=unavailable_pmid_text,
        )

    needs_review_total = len(grouped["mixed"]) + len(grouped["all_incorrect"])
    review_toolbar = ""
    review_script = ""
    if needs_review_total:
        review_toolbar = (
            "<div class=\"review-toolbar\">"
            "<p><strong>Human review workflow:</strong> annotate Mixed/All Incorrect studies as "
            "<code>Annotation Error</code> or <code>Extraction Error</code>, then add notes.</p>"
            "<p id=\"review-progress\"><strong>Review progress:</strong> 0 / {total} completed</p>"
            "<div class=\"review-toolbar-actions\">"
            "<button type=\"button\" class=\"review-btn\" id=\"review-export-json\">Download Review JSON</button>"
            "<button type=\"button\" class=\"review-btn\" id=\"review-export-csv\">Download Review CSV</button>"
            "<button type=\"button\" class=\"review-btn review-btn-muted\" id=\"review-clear\">Clear Saved Review</button>"
            "</div>"
            "</div>"
        ).format(total=needs_review_total)

        review_script = """
<script>
(() => {
  const STORAGE_KEY = "fuzzy_matching_human_review_v1";
  const reviewPanels = Array.from(document.querySelectorAll(".review-annotation"));
  if (!reviewPanels.length) {
    return;
  }

  const REVIEW_TOTAL = __REVIEW_TOTAL__;
  const progressEl = document.getElementById("review-progress");
  const exportJsonBtn = document.getElementById("review-export-json");
  const exportCsvBtn = document.getElementById("review-export-csv");
  const clearBtn = document.getElementById("review-clear");

  function readStore() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return { entries: {} };
      }
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") {
        return { entries: {} };
      }
      if (!parsed.entries || typeof parsed.entries !== "object") {
        parsed.entries = {};
      }
      return parsed;
    } catch (_err) {
      return { entries: {} };
    }
  }

  function writeStore(store) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
    } catch (_err) {
      // best-effort only
    }
  }

  function escapeCsvValue(value) {
    const text = String(value ?? "");
    if (text.includes(",") || text.includes("\\n") || text.includes("\\\"")) {
      return "\\""+text.replace(/\\\"/g, "\\\"\\\"")+"\\"";
    }
    return text;
  }

  function buildCsv(entries) {
    const header = [
      "pmid",
      "study_name",
      "study_category",
      "decision",
      "extraction_reasons",
      "note",
      "updated_at"
    ];
    const lines = [header.join(",")];
    for (const row of entries) {
      lines.push([
        escapeCsvValue(row.pmid),
        escapeCsvValue(row.study_name),
        escapeCsvValue(row.study_category),
        escapeCsvValue(row.decision),
        escapeCsvValue((row.extraction_reasons || []).join("|")),
        escapeCsvValue(row.note || ""),
        escapeCsvValue(row.updated_at || "")
      ].join(","));
    }
    return lines.join("\\n");
  }

  function downloadFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function getDecision(panel) {
    const selected = panel.querySelector('input[data-role="review-decision"]:checked');
    return selected ? selected.value : "";
  }

  function setExtractionVisibility(panel) {
    const reasonsWrap = panel.querySelector('[data-role="extraction-reasons"]');
    if (!reasonsWrap) {
      return;
    }
    const isExtraction = getDecision(panel) === "extraction_error";
    reasonsWrap.hidden = !isExtraction;
    const checkboxes = reasonsWrap.querySelectorAll('input[data-role="extraction-reason"]');
    checkboxes.forEach((checkbox) => {
      checkbox.disabled = !isExtraction;
    });
  }

  function setStatus(panel, decision) {
    const statusEl = panel.querySelector('[data-role="review-status"]');
    if (!statusEl) {
      return;
    }
    if (!decision) {
      statusEl.textContent = "Not reviewed yet.";
    } else if (decision === "annotation_error") {
      statusEl.textContent = "Saved: Annotation Error";
    } else if (decision === "extraction_error") {
      statusEl.textContent = "Saved: Extraction Error";
    } else {
      statusEl.textContent = "Saved";
    }
  }

  function collectEntry(panel) {
    const pmid = panel.getAttribute("data-pmid") || "";
    const studyName = panel.getAttribute("data-study-name") || "";
    const studyCategory = panel.getAttribute("data-study-category") || "";
    const decision = getDecision(panel);
    const extractionReasons = Array.from(
      panel.querySelectorAll('input[data-role="extraction-reason"]:checked')
    ).map((node) => node.value);
    const noteNode = panel.querySelector('textarea[data-role="review-note"]');
    const note = noteNode ? noteNode.value.trim() : "";
    return {
      pmid,
      study_name: studyName,
      study_category: studyCategory,
      decision,
      extraction_reasons: decision === "extraction_error" ? extractionReasons : [],
      note,
      updated_at: new Date().toISOString(),
    };
  }

  function loadPanel(panel, entry) {
    if (!entry) {
      setExtractionVisibility(panel);
      setStatus(panel, "");
      return;
    }
    const decision = entry.decision || "";
    const radios = panel.querySelectorAll('input[data-role="review-decision"]');
    radios.forEach((radio) => {
      radio.checked = radio.value === decision;
    });
    const reasons = new Set(Array.isArray(entry.extraction_reasons) ? entry.extraction_reasons : []);
    const reasonChecks = panel.querySelectorAll('input[data-role="extraction-reason"]');
    reasonChecks.forEach((checkbox) => {
      checkbox.checked = reasons.has(checkbox.value);
    });
    const noteNode = panel.querySelector('textarea[data-role="review-note"]');
    if (noteNode) {
      noteNode.value = entry.note || "";
    }
    setExtractionVisibility(panel);
    setStatus(panel, decision);
  }

  function updateProgress(entries) {
    if (!progressEl) {
      return;
    }
    const completed = Object.values(entries).filter((entry) => {
      return entry && (entry.decision === "annotation_error" || entry.decision === "extraction_error");
    }).length;
    progressEl.innerHTML = "<strong>Review progress:</strong> " + completed + " / " + REVIEW_TOTAL + " completed";
  }

  const store = readStore();
  reviewPanels.forEach((panel) => {
    const pmid = panel.getAttribute("data-pmid") || "";
    loadPanel(panel, store.entries[pmid]);
  });
  updateProgress(store.entries);

  function persistPanel(panel) {
    const entry = collectEntry(panel);
    if (!entry.pmid) {
      return;
    }
    if (!entry.decision && !entry.note && entry.extraction_reasons.length === 0) {
      delete store.entries[entry.pmid];
      writeStore(store);
      setExtractionVisibility(panel);
      setStatus(panel, "");
      updateProgress(store.entries);
      return;
    }
    store.entries[entry.pmid] = entry;
    writeStore(store);
    setExtractionVisibility(panel);
    setStatus(panel, entry.decision);
    updateProgress(store.entries);
  }

  reviewPanels.forEach((panel) => {
    panel.addEventListener("change", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      if (target.matches('input[data-role="review-decision"]')) {
        setExtractionVisibility(panel);
      }
      persistPanel(panel);
    });
    panel.addEventListener("input", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      if (target.matches('textarea[data-role="review-note"]')) {
        persistPanel(panel);
      }
    });
  });

  function getExportEntries() {
    const entries = [];
    for (const panel of reviewPanels) {
      const pmid = panel.getAttribute("data-pmid") || "";
      const stored = store.entries[pmid];
      if (!stored) {
        continue;
      }
      if (stored.decision === "annotation_error" || stored.decision === "extraction_error" || stored.note) {
        entries.push(stored);
      }
    }
    entries.sort((a, b) => {
      const aP = String(a.pmid || "");
      const bP = String(b.pmid || "");
      if (aP.length !== bP.length) {
        return aP.length - bP.length;
      }
      return aP.localeCompare(bP);
    });
    return entries;
  }

  if (exportJsonBtn) {
    exportJsonBtn.addEventListener("click", () => {
      const entries = getExportEntries();
      const payload = {
        generated_at: new Date().toISOString(),
        total_review_studies: REVIEW_TOTAL,
        completed_reviews: entries.filter((entry) => entry.decision).length,
        entries,
      };
      downloadFile("fuzzy_matching_human_review.json", JSON.stringify(payload, null, 2), "application/json");
    });
  }

  if (exportCsvBtn) {
    exportCsvBtn.addEventListener("click", () => {
      const entries = getExportEntries();
      const csv = buildCsv(entries);
      downloadFile("fuzzy_matching_human_review.csv", csv, "text/csv;charset=utf-8");
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      const ok = window.confirm("Clear all saved human review annotations for this report?");
      if (!ok) {
        return;
      }
      store.entries = {};
      writeStore(store);
      reviewPanels.forEach((panel) => {
        loadPanel(panel, null);
      });
      updateProgress(store.entries);
    });
  }
})();
</script>
""".replace("__REVIEW_TOTAL__", str(needs_review_total))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fuzzy Matching Report</title>
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
    .review-toolbar {{ margin-top: 0.85rem; padding: 0.75rem; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; }}
    .review-toolbar p {{ margin: 0 0 0.45rem 0; }}
    .review-toolbar-actions {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
    .review-btn {{ border: 1px solid #1f5f94; color: #1f5f94; background: #fff; border-radius: 999px; padding: 0.28rem 0.65rem; font-size: 0.86rem; cursor: pointer; }}
    .review-btn:hover {{ background: #eef6ff; }}
    .review-btn-muted {{ border-color: #7a8692; color: #48555f; }}
    section {{ margin-bottom: 1rem; }}
    .bucket > summary, .doc-card > summary, .inner-accordion > summary {{ cursor: pointer; }}
    .doc-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 0.85rem; margin-bottom: 0.85rem; }}
    .pmid-chip-list {{ display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.45rem 0 0.2rem 0; }}
    .pmid-chip {{ display: inline-block; border: 1px solid #bed2e5; background: #f3f8ff; color: #0f4978; border-radius: 999px; padding: 0.2rem 0.55rem; font-size: 0.84rem; text-decoration: none; }}
    .pmid-chip:hover {{ background: #e7f1ff; border-color: #98bbdc; }}
    .unavailable-raw {{ margin-top: 0.55rem; }}
    .unavailable-raw > summary {{ cursor: pointer; color: #0e4f85; }}
    .unavailable-raw code {{ display: block; margin-top: 0.35rem; white-space: normal; line-height: 1.45; background: #f7f9fc; border: 1px solid var(--line); border-radius: 7px; padding: 0.45rem; }}
    .doc-links {{ margin: 0.45rem 0 0.25rem 0; font-size: 0.92rem; }}
    .inner-accordion {{ margin-top: 0.6rem; border-top: 1px dashed var(--line); padding-top: 0.4rem; }}
    .resource-box {{ background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; padding: 0.55rem; }}
    .resource-note {{ font-size: 0.85rem; color: #3b4b5a; }}
    .table-accordion {{ margin: 0.45rem 0; border-top: 1px solid #e3e8ed; padding-top: 0.3rem; }}
    .table-accordion > summary {{ cursor: pointer; color: #0e4f85; }}
    .extracted-table {{ margin-top: 0.4rem; }}
    .analysis-coords > summary {{ cursor: pointer; color: #0e4f85; }}
    .coord-list {{ white-space: pre-wrap; margin-top: 0.35rem; background: #fbfcfe; border: 1px solid var(--line); border-radius: 6px; padding: 0.35rem; font-family: "IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, monospace; font-size: 0.82rem; max-height: 10rem; overflow-y: auto; }}
    .review-box {{ margin-top: 0.4rem; background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; padding: 0.6rem; }}
    .review-decision-group {{ display: flex; flex-wrap: wrap; gap: 0.8rem; margin: 0.45rem 0 0.6rem 0; }}
    .review-reasons {{ margin: 0.35rem 0 0.6rem 0; }}
    .review-reason-option {{ display: inline-flex; align-items: center; gap: 0.35rem; margin: 0.2rem 0.7rem 0.2rem 0; font-size: 0.88rem; }}
    .review-note-label {{ display: inline-block; margin-bottom: 0.25rem; }}
    textarea[data-role=\"review-note\"] {{ width: 100%; box-sizing: border-box; border: 1px solid var(--line); border-radius: 6px; padding: 0.45rem; font-family: inherit; font-size: 0.9rem; background: #fff; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border: 1px solid var(--line); padding: 0.45rem; vertical-align: top; text-align: left; }}
    th {{ background: #edf2f5; }}
    td.score-discrepancy {{ background: #ffefc2 !important; border-color: #d69700; font-weight: 700; }}
    a {{ color: #0e4f85; }}
    .st-accepted td {{ background: #e6f7eb; }}
    .st-uncertain td {{ background: #fff7dd; }}
    .st-unmatched td {{ background: #fdecec; }}
    .st-auto-unmatched td {{ background: #f1f2f4; }}
  </style>
</head>
<body>
  <header>
    <a id="top"></a>
    <h1>Fuzzy Matching Report</h1>
    <p>Coordinate-first matching (70%) + name similarity (30%), one-to-one Hungarian assignment, accepted &gt;= 0.75, uncertain &gt;= 0.55, coordinate override when coord score &gt;= {coord_override_threshold:.2f}. Metrics include overlap PMIDs only (manual ∩ auto).</p>
    <p><strong>Decimal manual coordinate handling:</strong> {escape(decimal_handling_label)} |
       <strong>Converted decimal analyses:</strong> {int(summary.get("converted_manual_decimal_analyses", 0))}{talairach_tolerance_note}</p>
    <p><strong>Overlap PMIDs:</strong> {int(summary.get("overlap_pmids", 0))} |
       <strong>Manual analyses:</strong> {int(summary.get("manual_analyses_total", 0))} |
       <strong>Accepted:</strong> {int(summary.get("accepted", 0))} |
       <strong>Uncertain:</strong> {int(summary.get("uncertain", 0))} |
       <strong>Unmatched:</strong> {int(summary.get("unmatched", 0))}</p>
    <p><strong>Study categories:</strong> All correct={int(summary.get("all_correct_pmids", 0))} |
       Mixed={int(summary.get("mixed_pmids", 0))} |
       All incorrect={int(summary.get("all_incorrect_pmids", 0))}</p>
    <p><strong>Unavailable Manual Studies (decimal coordinates):</strong> {len(unavailable_manual_decimal_pmids)}</p>
    <p><strong>Accepted by coordinate-score override:</strong> {int(summary.get("accepted_coord_override", summary.get("accepted_exact_coord_override", 0)))} |
       <strong>Coordinate-override matches with low name score:</strong> {int(summary.get("low_name_coord_override_matches", summary.get("low_name_exact_matches", 0)))}</p>
    <p><strong>PMIDs with Pubget docs:</strong> {int(summary.get("pmids_with_pubget", 0))} |
       <strong>Extracted tables available:</strong> {int(summary.get("pubget_tables_total", 0))}</p>
    <p><strong>All Correct exact same # analyses:</strong> {all_correct_exact_count} / {all_correct_total}</p>
    <p><strong>Score distribution:</strong> mean={float(summary.get("mean_combined_score", 0.0)):.3f} |
       p25={float(summary.get("p25_combined_score", 0.0)):.3f} |
       median={float(summary.get("median_combined_score", 0.0)):.3f} |
       p75={float(summary.get("p75_combined_score", 0.0)):.3f}</p>
    {review_toolbar}
  </header>
  <nav class="top-nav">
    <a href="#bucket-all-correct">All Correct ({int(summary.get("all_correct_pmids", 0))})</a>
    <a href="#bucket-mixed">Mixed ({int(summary.get("mixed_pmids", 0))})</a>
    <a href="#bucket-all-incorrect">All Incorrect ({int(summary.get("all_incorrect_pmids", 0))})</a>
    {unavailable_nav_link}
    <a href="#top">Top</a>
  </nav>
  {"".join(bucket_html)}
  {unavailable_section_html}
  {review_script}
</body>
</html>
"""

def write_match_artifacts(
    output_dir: Path,
    match_result: dict[str, Any],
    pubget_by_pmid: dict[str, dict[str, Any]] | None = None,
) -> None:
    overall_path = output_dir / "match_results_overall.json"
    overall_path.write_text(json.dumps(match_result, indent=2), encoding="utf-8")

    for legacy_name in ("fuzzy_matching_summary.html", "fuzzy_matching_study_review.html"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    combined_html = render_combined_report_html(match_result, pubget_by_pmid=pubget_by_pmid)
    combined_path = output_dir / "analysis_fuzzy_matching_report.html"
    combined_path.write_text(combined_html, encoding="utf-8")


DEFAULT_ANNOTATION_NAMES = [
    "social_processing_all",
    "affiliation_attachment",
    "perception_others",
    "perception_self",
    "social_communication",
]

DEFAULT_ANNOTATION_TO_NOTE_KEYS = {
    "social_processing_all": ["all_merged"],
    "affiliation_attachment": ["affiliation_merged"],
    "perception_others": ["others_merged"],
    "perception_self": ["self_merged"],
    "social_communication": ["soccomm_merged"],
}

ACTIVE_ANNOTATION_NAMES = list(DEFAULT_ANNOTATION_NAMES)

ACTIVE_ANNOTATION_TO_NOTE_KEYS = {
    key: list(values) for key, values in DEFAULT_ANNOTATION_TO_NOTE_KEYS.items()
}

ANALYSIS_ID_RE = re.compile(r"^(?P<pmid>.+?)_analysis_(?P<index>\d+)$")

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = SCRIPT_DIR.parent

PROJECTS_ROOT = REPO_ROOT / "projects"

MANUAL_NIMADS_ROOT = REPO_ROOT.parent / "neurometabench" / "data" / "nimads"

REQUIRED_OUTPUT_FILES = ("annotation_results.json", "coordinate_parsing_results.json")

CRITERIA_CONFUSION_LABELS = {"TP", "TN", "FP", "FN"}

CRITERIA_ERROR_CATEGORY_RULES: dict[str, dict[str, str]] = {
    "fn_inclusion_wrongly_failed": {
        "label": "False Negative: inclusion criterion wrongly failed",
        "confusion": "FN",
        "criterion_type": "inclusion",
    },
    "fn_exclusion_wrongly_applied": {
        "label": "False Negative: exclusion criterion wrongly applied",
        "confusion": "FN",
        "criterion_type": "exclusion",
    },
    "fp_inclusion_wrongly_applied": {
        "label": "False Positive: inclusion criterion wrongly applied",
        "confusion": "FP",
        "criterion_type": "inclusion",
    },
    "fp_exclusion_missed_or_mishandled": {
        "label": "False Positive: exclusion criterion missed/mishandled",
        "confusion": "FP",
        "criterion_type": "exclusion",
    },
}

CRITERIA_ERROR_CATEGORY_ORDER = list(CRITERIA_ERROR_CATEGORY_RULES.keys())

REVIEW_BUCKET_ORDER = ["Correct", "False Positive", "False Negative", "True Negatives"]

REVIEW_BUCKET_IDS = {
    "Correct": "bucket-correct",
    "False Positive": "bucket-false-positive",
    "False Negative": "bucket-false-negative",
    "True Negatives": "bucket-true-negatives",
}

EVAL_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "accepted": {
        "label": "ACCEPTED (strict)",
        "allowed_statuses": {"accepted"},
    },
    "uncertain": {
        "label": "UNCERTAIN (borderline only)",
        "allowed_statuses": {"uncertain"},
    },
    "combined": {
        "label": "COMBINED (accepted + uncertain)",
        "allowed_statuses": {"accepted", "uncertain"},
    },
}

ANNOTATION_SECTION_MODE_ORDER = ["accepted", "uncertain"]

OVERALL_SUMMARY_MODE_ORDER = ["accepted", "combined"]

EXHAUSTED_MANUAL_ASSUMPTION_RULE_ID = "paper_all_manual_accepted_unassigned_auto_as_negatives"
MATCHED_ONLY_RULE_ID = "matched_only_evaluable_auto_indices"

@dataclass
class Decision:
    include: bool
    reasoning: str
    analysis_id: str
    table_caption: str
    inclusion_criteria_applied: list[str]
    exclusion_criteria_applied: list[str]

def clean_text(value: str) -> str:
    return "".join(ch for ch in str(value) if ch >= " " or ch in "\n\t\r")

def dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = clean_text(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out

def note_keys_for_annotation(annotation_name: str) -> list[str]:
    manual_keys = list(ACTIVE_ANNOTATION_TO_NOTE_KEYS.get(annotation_name, []))
    expanded: list[str] = [*manual_keys]
    for key in manual_keys:
        if key.endswith("_wbonly"):
            expanded.append(key[: -len("_wbonly")])
    expanded.append(annotation_name)
    return dedupe_keep_order(expanded)

def infer_project_output_dir(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--project-output-dir does not exist: {explicit_path}")

        direct_candidates = [path.parent] if path.name == "outputs" else [path]
        for candidate in direct_candidates:
            if is_valid_project_output_dir(candidate):
                return candidate

        scoped_candidates = find_project_output_dirs_within(path)
        if not scoped_candidates:
            raise FileNotFoundError(
                "Could not resolve a project output dir from --project-output-dir. "
                "Expected a run directory containing outputs/annotation_results.json and "
                "outputs/coordinate_parsing_results.json."
            )
        selected = max(scoped_candidates, key=annotation_result_mtime)
        print(f"Auto-selected project output dir within {path}: {selected}")
        return selected

    if not PROJECTS_ROOT.exists():
        raise FileNotFoundError(
            f"Could not infer project output dir because projects root was not found: {PROJECTS_ROOT}. "
            "Pass --project-output-dir explicitly."
        )

    all_candidates: list[Path] = []
    seen: set[Path] = set()
    for project_dir in PROJECTS_ROOT.iterdir():
        if not project_dir.is_dir():
            continue
        for candidate in find_project_output_dirs_within(project_dir):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            all_candidates.append(candidate)

    if not all_candidates:
        raise FileNotFoundError(
            "Could not infer project output dir from projects/. Pass --project-output-dir explicitly."
        )

    selected = max(all_candidates, key=annotation_result_mtime)
    print(f"Auto-selected project output dir (most recently updated): {selected}")
    return selected

def is_valid_project_output_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    outputs_dir = path / "outputs"
    return outputs_dir.exists() and all((outputs_dir / name).exists() for name in REQUIRED_OUTPUT_FILES)

def annotation_result_mtime(project_output_dir: Path) -> float:
    return (project_output_dir / "outputs" / "annotation_results.json").stat().st_mtime

def find_project_output_dirs_within(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []

    candidates: list[Path] = []
    seen: set[Path] = set()

    def maybe_add(path: Path) -> None:
        if not path.is_dir():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        if is_valid_project_output_dir(path):
            seen.add(resolved)
            candidates.append(path)

    maybe_add(root)

    coordinates_dir = root / "coordinates"
    if coordinates_dir.is_dir():
        for entry in coordinates_dir.iterdir():
            maybe_add(entry)

    for entry in root.iterdir():
        maybe_add(entry)

    return candidates

def resolve_dirs(project_output_dir: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = args.output_dir or (project_output_dir / "reports" / "annotation_review_reports")
    match_input_dir = args.match_input_dir or (project_output_dir / "reports")
    return output_dir, match_input_dir

def infer_project_name(project_output_dir: Path) -> str:
    parts = list(project_output_dir.resolve().parts)
    project_indices = [i for i, part in enumerate(parts) if part == "projects"]
    if project_indices:
        idx = project_indices[-1]
        if idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError(
        "Could not infer project name from project output dir. "
        f"Expected path under projects/{{project_name}}/... but got: {project_output_dir}"
    )

def resolve_manual_annotation_path(project_output_dir: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()

    project_name = infer_project_name(project_output_dir)
    candidates = [
        (MANUAL_NIMADS_ROOT / project_name / "merged" / "nimads_annotation.json").resolve(),
        project_output_dir / "outputs" / "nimads_annotation.json",
    ]
    for path in candidates:
        if path.exists():
            print(f"Auto-selected manual annotation path: {path}")
            return path
    return None

def resolve_project_annotation_mapping_path(
    project_output_dir: Path,
    explicit_path: Path | None,
) -> Path | None:
    if explicit_path is not None:
        resolved = explicit_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"--annotation-mapping-path does not exist: {explicit_path}")
        return resolved

    project_name = infer_project_name(project_output_dir)
    inferred = (PROJECTS_ROOT / project_name / "nmb_mappings.json").resolve()
    if inferred.exists():
        return inferred
    return None

def configure_active_annotations(mapping_path: Path | None) -> None:
    global ACTIVE_ANNOTATION_NAMES
    global ACTIVE_ANNOTATION_TO_NOTE_KEYS

    if mapping_path is None:
        ACTIVE_ANNOTATION_NAMES = list(DEFAULT_ANNOTATION_NAMES)
        ACTIVE_ANNOTATION_TO_NOTE_KEYS = {
            key: list(values)
            for key, values in DEFAULT_ANNOTATION_TO_NOTE_KEYS.items()
        }
        print(
            "Using built-in social annotation mapping defaults; "
            "no project nmb_mappings.json was found."
        )
        return

    payload = load_json(mapping_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid annotation mapping format at {mapping_path}: expected JSON object")

    raw_mappings: Any
    if "annotation_mappings" in payload:
        raw_mappings = payload.get("annotation_mappings")
        if not isinstance(raw_mappings, dict):
            raise ValueError(
                f"Invalid annotation mapping format at {mapping_path}: "
                "expected 'annotation_mappings' to be a JSON object"
            )
    else:
        raw_mappings = {
            key: value
            for key, value in payload.items()
            if str(key).strip() != "meta_pmid"
        }

    annotation_names: list[str] = []
    note_keys_by_annotation: dict[str, list[str]] = defaultdict(list)
    for manual_key_raw, auto_annotation_raw in raw_mappings.items():
        if isinstance(auto_annotation_raw, (dict, list)):
            continue
        manual_key = clean_text(str(manual_key_raw)).strip()
        auto_annotation = clean_text(str(auto_annotation_raw)).strip()
        if not manual_key or not auto_annotation:
            continue
        if auto_annotation not in annotation_names:
            annotation_names.append(auto_annotation)
        note_keys_by_annotation[auto_annotation].append(manual_key)

    annotation_names = dedupe_keep_order(annotation_names)
    if not annotation_names:
        raise ValueError(f"Annotation mapping at {mapping_path} did not contain any usable entries")

    ACTIVE_ANNOTATION_NAMES = annotation_names
    ACTIVE_ANNOTATION_TO_NOTE_KEYS = {
        annotation: dedupe_keep_order(note_keys_by_annotation.get(annotation, []))
        for annotation in annotation_names
    }
    print(
        f"Loaded project annotation mapping from {mapping_path} "
        f"({len(ACTIVE_ANNOTATION_NAMES)} annotations)."
    )

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def normalize_criteria_items(section: Any) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if isinstance(section, dict):
        for code, text in section.items():
            code_text = clean_text(str(code)).strip()
            item_text = clean_text(str(text)).strip()
            if not item_text:
                continue
            items.append((code_text, item_text))
        return items

    if isinstance(section, list):
        for idx, row in enumerate(section, start=1):
            if isinstance(row, dict):
                code = clean_text(str(row.get("id") or row.get("code") or row.get("key") or "")).strip()
                text = clean_text(str(row.get("text") or row.get("description") or row.get("value") or "")).strip()
                if not text:
                    continue
                items.append((code, text))
            else:
                text = clean_text(str(row)).strip()
                if not text:
                    continue
                items.append((f"C{idx}", text))
    return items

def load_annotation_criteria(criteria_mapping_path: Path | None) -> dict[str, Any]:
    empty = {"global": {"inclusion": [], "exclusion": []}, "annotations": {}}
    if criteria_mapping_path is None or not criteria_mapping_path.exists():
        return empty

    payload = load_json(criteria_mapping_path)
    annotation_block = payload.get("annotation", {}) if isinstance(payload, dict) else {}
    if not isinstance(annotation_block, dict):
        return empty

    global_block = annotation_block.get("global", {})
    global_inclusion = normalize_criteria_items(global_block.get("inclusion", {}) if isinstance(global_block, dict) else {})
    global_exclusion = normalize_criteria_items(global_block.get("exclusion", {}) if isinstance(global_block, dict) else {})

    annotations_out: dict[str, dict[str, list[tuple[str, str]]]] = {}
    annotations_block = annotation_block.get("annotations", {})
    if isinstance(annotations_block, dict):
        for annotation_name, rule_block in annotations_block.items():
            if not isinstance(rule_block, dict):
                continue
            annotations_out[str(annotation_name)] = {
                "inclusion": normalize_criteria_items(rule_block.get("inclusion", {})),
                "exclusion": normalize_criteria_items(rule_block.get("exclusion", {})),
            }

    return {
        "global": {"inclusion": global_inclusion, "exclusion": global_exclusion},
        "annotations": annotations_out,
    }

def build_annotation_criteria_metadata(
    criteria: dict[str, Any] | None,
    annotation_name: str,
) -> dict[str, dict[str, str]]:
    criteria = criteria or {}
    metadata: dict[str, dict[str, str]] = {}

    global_criteria = criteria.get("global", {}) if isinstance(criteria, dict) else {}
    annotation_criteria = criteria.get("annotations", {}) if isinstance(criteria, dict) else {}
    annotation_rules = (
        annotation_criteria.get(annotation_name, {})
        if isinstance(annotation_criteria, dict)
        else {}
    )

    def add_items(scope: str, criterion_type: str, items: Any) -> None:
        for raw_code, _raw_text in items if isinstance(items, list) else []:
            code = clean_text(str(raw_code)).strip()
            if not code:
                continue
            metadata[code] = {
                "scope": scope,
                "criterion_type": criterion_type,
            }

    if isinstance(global_criteria, dict):
        add_items("global", "inclusion", global_criteria.get("inclusion", []))
        add_items("global", "exclusion", global_criteria.get("exclusion", []))
    if isinstance(annotation_rules, dict):
        add_items("annotation", "inclusion", annotation_rules.get("inclusion", []))
        add_items("annotation", "exclusion", annotation_rules.get("exclusion", []))

    return metadata

def compile_criteria_code_pattern(allowed_codes: set[str]) -> re.Pattern[str] | None:
    if not allowed_codes:
        return None
    sorted_codes = sorted(allowed_codes, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(code) for code in sorted_codes) + r")\b")

def extract_criteria_codes_from_reasoning(
    reasoning: str,
    pattern: re.Pattern[str] | None,
) -> list[str]:
    if pattern is None:
        return []
    text = clean_text(reasoning or "")
    if not text:
        return []
    return dedupe_keep_order(list(pattern.findall(text)))

def compute_criteria_error_analysis(
    docs: dict[str, list[dict[str, Any]]],
    annotation_name: str,
    criteria: dict[str, Any] | None,
) -> dict[str, Any]:
    criterion_meta = build_annotation_criteria_metadata(criteria, annotation_name)
    allowed_codes = set(criterion_meta.keys())
    criteria_pattern = compile_criteria_code_pattern(allowed_codes)

    rows_considered = 0
    rows_with_codes = 0
    rows_without_codes = 0
    per_code_stats: dict[str, dict[str, Any]] = {}

    for bucket_docs in docs.values():
        for doc in bucket_docs:
            for row in doc.get("analysis_rows", []):
                confusion_label = clean_text(str(row.get("confusion_label", ""))).strip().upper()
                if confusion_label not in CRITERIA_CONFUSION_LABELS:
                    continue
                rows_considered += 1

                found_codes = extract_criteria_codes_from_reasoning(
                    str(row.get("reasoning") or ""),
                    criteria_pattern,
                )
                if not found_codes:
                    rows_without_codes += 1
                    continue

                rows_with_codes += 1
                for code in found_codes:
                    meta = criterion_meta.get(code)
                    if meta is None:
                        continue

                    entry = per_code_stats.setdefault(
                        code,
                        {
                            "criterion": code,
                            "scope": str(meta.get("scope", "")),
                            "criterion_type": str(meta.get("criterion_type", "")),
                            "correct_mentions": 0,
                            "fp_mentions": 0,
                            "fn_mentions": 0,
                            **{category_id: 0 for category_id in CRITERIA_ERROR_CATEGORY_ORDER},
                        },
                    )

                    if confusion_label in {"TP", "TN"}:
                        entry["correct_mentions"] += 1
                    if confusion_label == "FP":
                        entry["fp_mentions"] += 1
                    if confusion_label == "FN":
                        entry["fn_mentions"] += 1

                    for category_id, category_rule in CRITERIA_ERROR_CATEGORY_RULES.items():
                        if (
                            confusion_label == category_rule["confusion"]
                            and entry["criterion_type"] == category_rule["criterion_type"]
                        ):
                            entry[category_id] += 1

    per_code_out: dict[str, dict[str, Any]] = {}
    for code, entry in per_code_stats.items():
        correct_mentions = int(entry.get("correct_mentions", 0))
        fp_mentions = int(entry.get("fp_mentions", 0))
        fn_mentions = int(entry.get("fn_mentions", 0))
        enriched = {
            "criterion": code,
            "scope": str(entry.get("scope", "")),
            "criterion_type": str(entry.get("criterion_type", "")),
            "correct_mentions": correct_mentions,
            "fp_mentions": fp_mentions,
            "fn_mentions": fn_mentions,
            "error_mentions": int(fp_mentions + fn_mentions),
        }
        for category_id in CRITERIA_ERROR_CATEGORY_ORDER:
            category_error_mentions = int(entry.get(category_id, 0))
            enriched[category_id] = category_error_mentions
            enriched[f"{category_id}_rate_vs_correct"] = float(category_error_mentions / max(1, correct_mentions))
            enriched[f"{category_id}_minus_correct"] = int(category_error_mentions - correct_mentions)
        per_code_out[code] = enriched

    ranked_categories: dict[str, list[dict[str, Any]]] = {}
    for category_id in CRITERIA_ERROR_CATEGORY_ORDER:
        rows: list[dict[str, Any]] = []
        for code, entry in per_code_out.items():
            error_mentions = int(entry.get(category_id, 0))
            if error_mentions <= 0:
                continue
            correct_mentions = int(entry.get("correct_mentions", 0))
            rows.append(
                {
                    "criterion": code,
                    "scope": str(entry.get("scope", "")),
                    "criterion_type": str(entry.get("criterion_type", "")),
                    "category": category_id,
                    "category_label": CRITERIA_ERROR_CATEGORY_RULES[category_id]["label"],
                    "error_mentions": error_mentions,
                    "correct_mentions": correct_mentions,
                    "error_rate_vs_correct": float(error_mentions / max(1, correct_mentions)),
                    "error_minus_correct": int(error_mentions - correct_mentions),
                    "fp_mentions": int(entry.get("fp_mentions", 0)),
                    "fn_mentions": int(entry.get("fn_mentions", 0)),
                }
            )

        rows.sort(
            key=lambda row: (
                -float(row.get("error_rate_vs_correct", 0.0)),
                -int(row.get("error_mentions", 0)),
                str(row.get("criterion", "")),
            )
        )
        ranked_categories[category_id] = rows

    return {
        "coverage": {
            "analysis_rows_considered": int(rows_considered),
            "rows_with_codes": int(rows_with_codes),
            "rows_without_codes": int(rows_without_codes),
            "coverage_rate": float(rows_with_codes / max(1, rows_considered)),
            "allowed_criteria_codes": int(len(allowed_codes)),
        },
        "per_code": per_code_out,
        "ranked_categories": ranked_categories,
    }

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
                    "table_foot": clean_text(meta.get("table_foot", "")).strip(),
                    "table_html": table_html_by_id.get(table_id, ""),
                }
            )

    return pmid_to_fulltext, dict(pmid_to_coord_tables)

def load_auto_parsed_analysis_info(path: Path) -> dict[str, list[dict[str, str]]]:
    payload = load_json(path)
    studies = payload.get("studies", [])
    parsed: dict[str, list[dict[str, str]]] = {}
    for study in studies:
        pmid = str(study.get("pmid"))
        analyses = study.get("analyses", [])
        parsed_rows: list[dict[str, str]] = []
        for i, a in enumerate(analyses):
            parsed_rows.append(
                {
                    "name": clean_text(a.get("name") or f"analysis_{i}"),
                    "description": clean_text(a.get("description") or ""),
                    "table_id": clean_text(a.get("table_id") or "").strip(),
                }
            )
        parsed[pmid] = parsed_rows
    return parsed

def load_model_decisions(path: Path) -> dict[str, dict[str, dict[int, Decision]]]:
    rows = load_json(path)
    decisions: dict[str, dict[str, dict[int, Decision]]] = defaultdict(lambda: defaultdict(dict))

    def parse_applied_codes(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return dedupe_keep_order([str(item) for item in value if clean_text(str(item)).strip()])

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
            table_caption=clean_text(row.get("table_caption") or "").strip(),
            inclusion_criteria_applied=parse_applied_codes(row.get("inclusion_criteria_applied")),
            exclusion_criteria_applied=parse_applied_codes(row.get("exclusion_criteria_applied")),
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
    auto_grouped: dict[str, set[str]] = {annotation: set() for annotation in ACTIVE_ANNOTATION_NAMES}
    manual_grouped: dict[str, set[str]] = {annotation: set() for annotation in ACTIVE_ANNOTATION_NAMES}
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
            for annotation in ACTIVE_ANNOTATION_NAMES:
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
            for annotation in ACTIVE_ANNOTATION_NAMES:
                candidate_keys = note_keys_for_annotation(annotation)
                included = any(bool(note_obj.get(key, False)) for key in candidate_keys)
                if included:
                    manual_grouped[annotation].add(pmid)

    return unique_pmids_in_auto, auto_grouped, manual_grouped

def load_match_results_by_annotation(match_input_dir: Path) -> tuple[dict[str, Any], bool]:
    results: dict[str, Any] = {}
    missing_per_annotation: list[str] = []
    for annotation_name in ACTIVE_ANNOTATION_NAMES:
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
        for annotation_name in ACTIVE_ANNOTATION_NAMES:
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
        target_note_keys = note_keys_for_annotation(annotation_name)
        for pmid, pmid_result in match_results.get("pmids", {}).items():
            manual_analyses = list(pmid_result.get("manual_analyses", []))
            review_match_diagnostics = list(manual_analyses)
            unassigned_auto_indices: list[int] = []
            for value in pmid_result.get("unassigned_auto_indices", []):
                try:
                    unassigned_auto_indices.append(int(value))
                except Exception:
                    continue
            unassigned_auto_indices = sorted(set(unassigned_auto_indices))
            if overall_fallback and manual_annotation_membership:
                filtered_analyses: list[dict[str, Any]] = []
                for entry in manual_analyses:
                    analysis_id = clean_text(entry.get("manual_analysis_id", "")).strip()
                    auto_analysis_id = clean_text(entry.get("best_auto_analysis_id", "")).strip()
                    candidate_keys = target_note_keys

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

            paper_all_manual_accepted = bool(review_match_diagnostics) and all(
                str(entry.get("match_status", "")).strip().lower() == "accepted"
                for entry in review_match_diagnostics
            )

            accepted_indices = {
                int(entry["best_auto_index"])
                for entry in manual_analyses
                if entry.get("best_auto_index") is not None and entry.get("match_status") == "accepted"
            }
            uncertain_indices = {
                int(entry["best_auto_index"])
                for entry in manual_analyses
                if entry.get("best_auto_index") is not None and entry.get("match_status") == "uncertain"
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
                "uncertain_indices": uncertain_indices,
                "manual_names": [entry.get("manual_name", "") for entry in manual_analyses],
                "unmatched_manual_names": [
                    entry.get("manual_name", "")
                    for entry in manual_analyses
                    if entry.get("match_status") == "unmatched"
                ],
                "match_diagnostics": manual_analyses,
                # Preserve overall fuzzy matches for row-level "Matched Outcome" labels and
                # the diagnostics panel when per-annotation truth is sliced from overall fallback.
                "review_match_diagnostics": review_match_diagnostics,
                "status_counts": {
                    "accepted": int(status_counts.get("accepted", 0)),
                    "uncertain": int(status_counts.get("uncertain", 0)),
                    "unmatched": int(status_counts.get("unmatched", 0)),
                    "mean_combined_score": float(status_counts.get("mean_combined_score", 0.0)),
                },
                "paper_all_manual_accepted": bool(paper_all_manual_accepted),
                "unassigned_auto_indices": list(unassigned_auto_indices),
                "manual_missing_in_auto": bool(pmid_result.get("manual_missing_in_auto", False))
                if not overall_fallback
                else False,
            }

    return manual_truth

def make_document_row(
    pmid: str,
    annotation_name: str,
    parsed_analyses: list[dict[str, str]],
    decisions_by_idx: dict[int, Decision],
    true_indices: set[int],
    manual_names: list[str],
    unmatched_manual_names: list[str],
    bucket: str,
    fulltext_entry: dict[str, str] | None,
    coord_tables: list[dict[str, str]],
    match_diagnostics: list[dict[str, Any]],
    review_match_diagnostics: list[dict[str, Any]],
    evaluable_auto_indices: set[int] | None,
    status_counts: dict[str, Any],
    manual_missing_in_auto: bool,
    classification_rule_id: str,
    classification_rule_activated: bool,
    added_assumed_negative_indices: set[int] | None = None,
) -> dict[str, Any]:
    evaluable_indices = None if evaluable_auto_indices is None else set(evaluable_auto_indices)
    assumed_negative_indices = set(added_assumed_negative_indices or set())
    pred_indices = {
        idx
        for idx, decision in decisions_by_idx.items()
        if decision.include and (evaluable_indices is None or idx in evaluable_indices)
    }
    true_indices_eval = set(true_indices)
    if evaluable_indices is not None:
        true_indices_eval &= evaluable_indices
    correct_indices = pred_indices & true_indices_eval
    matched_auto_indices = {
        int(entry["best_auto_index"])
        for entry in review_match_diagnostics
        if entry.get("best_auto_index") is not None
    }
    match_status_by_auto_idx = {
        int(entry["best_auto_index"]): str(entry.get("match_status", ""))
        for entry in review_match_diagnostics
        if entry.get("best_auto_index") is not None
    }

    max_idx = len(parsed_analyses) - 1
    if decisions_by_idx:
        max_idx = max(max_idx, max(decisions_by_idx.keys()))
    if evaluable_indices:
        max_idx = max(max_idx, max(evaluable_indices))

    analysis_rows: list[dict[str, Any]] = []
    for idx in range(max_idx + 1):
        parsed_info = parsed_analyses[idx] if idx < len(parsed_analyses) else {}
        name = clean_text(parsed_info.get("name") or f"analysis_{idx}")
        decision = decisions_by_idx.get(idx)
        model_include = None if decision is None else decision.include
        is_evaluable = evaluable_indices is None or idx in evaluable_indices
        matched_for_review = idx in matched_auto_indices
        match_status_for_idx = match_status_by_auto_idx.get(idx, "")
        manual_include = idx in true_indices_eval

        if not is_evaluable:
            confusion_label = "-"
            confusion_class = "confusion-na"
        elif matched_for_review and match_status_for_idx == "unmatched":
            confusion_label = "*"
            confusion_class = "confusion-na"
        elif model_include is not None:
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
                "parsed_description": clean_text(parsed_info.get("description") or ""),
                "table_id": clean_text(parsed_info.get("table_id") or "").strip(),
                "model_include": model_include,
                "model_decision_icon": decision_icon,
                "model_decision_class": decision_class,
                "confusion_label": confusion_label,
                "confusion_class": confusion_class,
                "matched_for_review": matched_for_review,
                "assumed_negative_for_review": idx in assumed_negative_indices,
                "reasoning": "" if decision is None else decision.reasoning,
                "llm_table_caption": "" if decision is None else clean_text(decision.table_caption).strip(),
                "inclusion_criteria_applied": (
                    []
                    if decision is None
                    else list(decision.inclusion_criteria_applied)
                ),
                "exclusion_criteria_applied": (
                    []
                    if decision is None
                    else list(decision.exclusion_criteria_applied)
                ),
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
        "true_indices": sorted(true_indices_eval),
        "correct_indices": sorted(correct_indices),
        "manual_names": manual_names,
        "unmatched_manual_names": unmatched_manual_names,
        "analysis_rows": analysis_rows,
        "fulltext": fulltext_entry,
        "coord_tables": coord_tables,
        "match_diagnostics": match_diagnostics,
        "review_match_diagnostics": review_match_diagnostics,
        "status_counts": status_counts,
        "manual_missing_in_auto": manual_missing_in_auto,
        "classification_rule_id": clean_text(classification_rule_id).strip(),
        "classification_rule_activated": bool(classification_rule_activated),
        "added_assumed_negative_indices": sorted(assumed_negative_indices),
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

def extract_evaluable_auto_indices(
    review_match_diagnostics: list[dict[str, Any]],
    allowed_statuses: set[str],
) -> set[int]:
    normalized_statuses = {str(status).strip().lower() for status in allowed_statuses}
    return {
        int(entry["best_auto_index"])
        for entry in review_match_diagnostics
        if entry.get("best_auto_index") is not None
        and str(entry.get("match_status", "")).strip().lower() in normalized_statuses
    }


def build_evaluable_auto_indices(
    truth_entry: dict[str, Any],
    allowed_statuses: set[str],
    allow_exhausted_manual_expansion: bool,
) -> tuple[set[int], set[int], set[int], bool]:
    review_match_rows = truth_entry.get(
        "review_match_diagnostics",
        truth_entry.get("match_diagnostics", []),
    )
    matched_auto_indices = extract_evaluable_auto_indices(
        review_match_rows,
        allowed_statuses=allowed_statuses,
    )
    expanded_auto_indices = set(matched_auto_indices)
    paper_all_manual_accepted = bool(truth_entry.get("paper_all_manual_accepted", False))
    added_assumed_negative_indices: set[int] = set()

    if allow_exhausted_manual_expansion and paper_all_manual_accepted:
        unassigned_auto_indices: set[int] = set()
        for value in truth_entry.get("unassigned_auto_indices", []):
            try:
                unassigned_auto_indices.add(int(value))
            except Exception:
                continue
        added_assumed_negative_indices = unassigned_auto_indices - expanded_auto_indices
        expanded_auto_indices |= added_assumed_negative_indices

    return (
        matched_auto_indices,
        expanded_auto_indices,
        added_assumed_negative_indices,
        paper_all_manual_accepted,
    )


def derive_true_indices_for_mode(
    truth_entry: dict[str, Any],
    allowed_statuses: set[str],
    evaluable_auto_indices: set[int],
) -> set[int]:
    normalized_statuses = {str(status).strip().lower() for status in allowed_statuses}

    accepted_indices = {
        int(idx)
        for idx in set(truth_entry.get("true_indices", set()) or set())
    }

    uncertain_indices_raw = truth_entry.get("uncertain_indices")
    if uncertain_indices_raw is None:
        uncertain_indices = {
            int(entry["best_auto_index"])
            for entry in truth_entry.get(
                "review_match_diagnostics",
                truth_entry.get("match_diagnostics", []),
            )
            if entry.get("best_auto_index") is not None
            and str(entry.get("match_status", "")).strip().lower() == "uncertain"
        }
    else:
        uncertain_indices = {int(idx) for idx in set(uncertain_indices_raw)}

    mode_true_indices: set[int] = set()
    if "accepted" in normalized_statuses:
        mode_true_indices |= accepted_indices
    if "uncertain" in normalized_statuses:
        mode_true_indices |= uncertain_indices

    return mode_true_indices & set(evaluable_auto_indices)

def classify_documents(
    annotation_name: str,
    parsed_analyses: dict[str, list[dict[str, str]]],
    model_decisions: dict[str, dict[str, dict[int, Decision]]],
    manual_truth: dict[str, dict[str, dict[str, Any]]],
    criteria: dict[str, Any] | None,
    pmid_to_fulltext: dict[str, dict[str, str]],
    pmid_to_coord_tables: dict[str, list[dict[str, str]]],
    allowed_match_statuses: set[str],
    study_universe_pmids: set[str] | None = None,
    auto_study_pmids_by_annotation: dict[str, set[str]] | None = None,
    manual_study_pmids_by_annotation: dict[str, set[str]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    docs = {bucket: [] for bucket in REVIEW_BUCKET_ORDER}
    ann_decisions = model_decisions.get(annotation_name, {})
    ann_truth = manual_truth.get(annotation_name, {})
    normalized_mode_statuses = {str(status).strip().lower() for status in allowed_match_statuses}
    allow_exhausted_manual_expansion = "accepted" in normalized_mode_statuses
    run_pmids = set(parsed_analyses.keys())
    doc_overlap_pmids = run_pmids & set(ann_truth.keys())
    pmids = doc_overlap_pmids
    evaluable_pmids: set[str] = set()

    for pmid in sorted(pmids, key=lambda x: (len(x), x)):
        parsed_analysis_info = parsed_analyses.get(pmid, [])
        decisions_by_idx = ann_decisions.get(pmid, {})
        truth_entry = ann_truth.get(
            pmid,
            {
                "true_indices": set(),
                "uncertain_indices": set(),
                "manual_names": [],
                "unmatched_manual_names": [],
                "match_diagnostics": [],
                "status_counts": {"accepted": 0, "uncertain": 0, "unmatched": 0, "mean_combined_score": 0.0},
                "paper_all_manual_accepted": False,
                "unassigned_auto_indices": [],
                "manual_missing_in_auto": False,
            },
        )

        review_match_diagnostics = truth_entry.get(
            "review_match_diagnostics",
            truth_entry.get("match_diagnostics", []),
        )
        (
            matched_auto_indices,
            evaluable_auto_indices,
            added_assumed_negative_indices,
            paper_all_manual_accepted,
        ) = build_evaluable_auto_indices(
            truth_entry=truth_entry,
            allowed_statuses=allowed_match_statuses,
            allow_exhausted_manual_expansion=allow_exhausted_manual_expansion,
        )
        if not evaluable_auto_indices:
            continue
        evaluable_pmids.add(pmid)

        pred_indices = {idx for idx, decision in decisions_by_idx.items() if decision.include and idx in evaluable_auto_indices}
        true_indices = derive_true_indices_for_mode(
            truth_entry=truth_entry,
            allowed_statuses=allowed_match_statuses,
            evaluable_auto_indices=evaluable_auto_indices,
        )
        correct_indices = pred_indices & true_indices

        if correct_indices:
            bucket = "Correct"
        elif pred_indices and not true_indices:
            bucket = "False Positive"
        elif true_indices and not correct_indices:
            bucket = "False Negative"
        elif not pred_indices and not true_indices:
            bucket = "True Negatives"
        else:
            continue

        docs[bucket].append(
            make_document_row(
                pmid=pmid,
                annotation_name=annotation_name,
                parsed_analyses=parsed_analysis_info,
                decisions_by_idx=decisions_by_idx,
                true_indices=true_indices,
                manual_names=truth_entry["manual_names"],
                unmatched_manual_names=truth_entry["unmatched_manual_names"],
                bucket=bucket,
                fulltext_entry=pmid_to_fulltext.get(pmid),
                coord_tables=pmid_to_coord_tables.get(pmid, []),
                match_diagnostics=truth_entry.get("match_diagnostics", []),
                review_match_diagnostics=review_match_diagnostics,
                evaluable_auto_indices=evaluable_auto_indices,
                status_counts=truth_entry.get("status_counts", {}),
                manual_missing_in_auto=bool(truth_entry.get("manual_missing_in_auto", False)),
                classification_rule_id=(
                    EXHAUSTED_MANUAL_ASSUMPTION_RULE_ID
                    if allow_exhausted_manual_expansion
                    else MATCHED_ONLY_RULE_ID
                ),
                classification_rule_activated=(
                    bool(allow_exhausted_manual_expansion)
                    and bool(paper_all_manual_accepted)
                    and bool(added_assumed_negative_indices)
                ),
                added_assumed_negative_indices=added_assumed_negative_indices,
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
        study_universe = set(study_universe_pmids) & evaluable_pmids
        predicted_study_set = set(auto_study_pmids_by_annotation.get(annotation_name, set())) & study_universe
        manual_study_set = set(manual_study_pmids_by_annotation.get(annotation_name, set())) & study_universe
    else:
        study_universe = set(doc_overlap_pmids) & evaluable_pmids
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

    # Analysis-level metrics:
    # 1) Baseline matched-only universe (existing behavior).
    # 2) Exhausted-manual assumption universe (matched + unassigned auto), enabled
    #    only when mode includes accepted statuses and all paper-level manual
    #    analyses are accepted for that PMID.

    analysis_tp = 0
    analysis_fp = 0
    analysis_fn = 0
    analysis_tn = 0
    matched_auto_universe = 0
    manual_accepted_matched = 0
    predicted_positive_on_matched = 0

    expanded_analysis_tp = 0
    expanded_analysis_fp = 0
    expanded_analysis_fn = 0
    expanded_analysis_tn = 0
    expanded_auto_universe = 0
    expanded_manual_accepted = 0
    expanded_predicted_positive = 0

    assumption_activated_pmids = 0
    added_assumed_negative_analyses = 0
    pmids_with_paper_all_manual_accepted = 0

    for pmid in doc_overlap_pmids:
        decisions_for_pmid = ann_decisions.get(pmid, {})
        truth_for_pmid = ann_truth.get(pmid, {})
        (
            matched_auto_indices,
            expanded_auto_indices,
            added_indices,
            paper_all_manual_accepted,
        ) = build_evaluable_auto_indices(
            truth_entry=truth_for_pmid,
            allowed_statuses=allowed_match_statuses,
            allow_exhausted_manual_expansion=allow_exhausted_manual_expansion,
        )
        if not matched_auto_indices and not expanded_auto_indices:
            continue
        true_indices_matched = derive_true_indices_for_mode(
            truth_entry=truth_for_pmid,
            allowed_statuses=allowed_match_statuses,
            evaluable_auto_indices=matched_auto_indices,
        )

        for idx_int in matched_auto_indices:
            matched_auto_universe += 1
            decision = decisions_for_pmid.get(idx_int)
            pred_include = bool(decision.include) if decision is not None else False
            true_include = idx_int in true_indices_matched

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

        if paper_all_manual_accepted:
            pmids_with_paper_all_manual_accepted += 1

        if added_indices:
            assumption_activated_pmids += 1
            added_assumed_negative_analyses += len(added_indices)

        true_indices_expanded = derive_true_indices_for_mode(
            truth_entry=truth_for_pmid,
            allowed_statuses=allowed_match_statuses,
            evaluable_auto_indices=expanded_auto_indices,
        )
        for idx_int in expanded_auto_indices:
            expanded_auto_universe += 1
            decision = decisions_for_pmid.get(idx_int)
            pred_include = bool(decision.include) if decision is not None else False
            true_include = idx_int in true_indices_expanded

            if true_include:
                expanded_manual_accepted += 1
            if pred_include:
                expanded_predicted_positive += 1

            if pred_include and true_include:
                expanded_analysis_tp += 1
            elif pred_include and not true_include:
                expanded_analysis_fp += 1
            elif (not pred_include) and true_include:
                expanded_analysis_fn += 1
            else:
                expanded_analysis_tn += 1

    analysis_metrics = compute_prf(tp=analysis_tp, fp=analysis_fp, fn=analysis_fn)
    analysis_metrics["tn"] = int(analysis_tn)
    analysis_metrics["accuracy"] = (
        float((analysis_tp + analysis_tn) / matched_auto_universe)
        if matched_auto_universe
        else 0.0
    )
    analysis_metrics["manual_accepted_analyses"] = int(manual_accepted_matched)
    analysis_metrics["predicted_analyses"] = int(predicted_positive_on_matched)
    analysis_metrics["analysis_universe"] = int(matched_auto_universe)

    analysis_metrics_exhausted_manual_assumption = compute_prf(
        tp=expanded_analysis_tp,
        fp=expanded_analysis_fp,
        fn=expanded_analysis_fn,
    )
    analysis_metrics_exhausted_manual_assumption["tn"] = int(expanded_analysis_tn)
    analysis_metrics_exhausted_manual_assumption["accuracy"] = (
        float((expanded_analysis_tp + expanded_analysis_tn) / expanded_auto_universe)
        if expanded_auto_universe
        else 0.0
    )
    analysis_metrics_exhausted_manual_assumption["manual_accepted_analyses"] = int(expanded_manual_accepted)
    analysis_metrics_exhausted_manual_assumption["predicted_analyses"] = int(expanded_predicted_positive)
    analysis_metrics_exhausted_manual_assumption["analysis_universe"] = int(expanded_auto_universe)

    assumed_negative_expansion = {
        "activation_rule_id": EXHAUSTED_MANUAL_ASSUMPTION_RULE_ID,
        "mode_allowed_match_statuses": sorted(normalized_mode_statuses),
        "mode_supports_expansion": bool(allow_exhausted_manual_expansion),
        "pmids_with_paper_all_manual_accepted": int(pmids_with_paper_all_manual_accepted),
        "activated_pmids": int(assumption_activated_pmids),
        "added_assumed_negative_analyses": int(added_assumed_negative_analyses),
    }

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
    criteria_error_analysis = compute_criteria_error_analysis(
        docs=docs,
        annotation_name=annotation_name,
        criteria=criteria,
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
        "analysis_metrics_exhausted_manual_assumption": analysis_metrics_exhausted_manual_assumption,
        "assumed_negative_expansion": assumed_negative_expansion,
        "bucket_match_counts": bucket_match_counts,
        "missing_manual_pmids": missing_manual_pmids,
        "criteria_error_analysis": criteria_error_analysis,
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

def classify_criterion_status_for_row(
    inclusion_criteria_applied: list[str],
    exclusion_criteria_applied: list[str],
    criterion_meta: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    exclusion_codes = [
        code
        for code, meta in criterion_meta.items()
        if str(meta.get("criterion_type", "")).lower() == "exclusion"
    ]

    # Explicit-only behavior: only exclusion criteria listed as applied are treated as unmet.
    # Do not infer unmet inclusion criteria from omissions.
    explicit_unmet_known = [
        code for code in dedupe_keep_order(exclusion_criteria_applied) if code in exclusion_codes
    ]
    unknown_unmet = [code for code in dedupe_keep_order(exclusion_criteria_applied) if code not in criterion_meta]
    return {
        "explicit_unmet_known": explicit_unmet_known,
        "unknown_unmet": unknown_unmet,
    }

def render_criteria_checks_cell(
    row: dict[str, Any],
    criterion_meta: dict[str, dict[str, str]],
) -> str:
    if not criterion_meta:
        return "<span class=\"muted\">No criteria mapping loaded.</span>"

    status = classify_criterion_status_for_row(
        inclusion_criteria_applied=list(row.get("inclusion_criteria_applied", []) or []),
        exclusion_criteria_applied=list(row.get("exclusion_criteria_applied", []) or []),
        criterion_meta=criterion_meta,
    )

    def render_pills(codes: list[str], pill_class: str) -> str:
        if not codes:
            return ""
        pills = []
        for code in codes:
            meta = criterion_meta.get(code, {})
            title = clean_text(str(meta.get("text", ""))).strip()
            title_attr = escape(f"{code}: {title}") if title else escape(code)
            pills.append(
                f"<span class=\"criteria-pill {escape(pill_class)}\" title=\"{title_attr}\">{escape(code)}</span>"
            )
        return "".join(pills)

    unmet_codes = list(status.get("explicit_unmet_known", []))
    unknown_unmet = list(status.get("unknown_unmet", []))
    if not unmet_codes and not unknown_unmet:
        return "<span class=\"muted\">---</span>"

    unmet_html = render_pills(unmet_codes, "criteria-bad")
    unknown_html = ""
    if unknown_unmet:
        unknown_html = (
            "<div class=\"criteria-status-row\">"
            "<span class=\"criteria-status-label\">Unknown</span>"
            + "".join(
                f"<span class=\"criteria-pill criteria-unknown\">{escape(code)}</span>"
                for code in unknown_unmet
            )
            + "</div>"
        )
    return "<div class=\"criteria-status-wrap\">" + unmet_html + unknown_html + "</div>"

def render_doc_card(
    doc: dict[str, Any],
    criterion_meta: dict[str, dict[str, str]],
) -> str:
    status_counts = doc.get("status_counts", {})
    status_meta = (
        f"accepted={int(status_counts.get('accepted', 0))}, "
        f"uncertain={int(status_counts.get('uncertain', 0))}, "
        f"unmatched={int(status_counts.get('unmatched', 0))}"
    )
    classification_rule_id = clean_text(str(doc.get("classification_rule_id", MATCHED_ONLY_RULE_ID))).strip()
    classification_rule_activated = bool(doc.get("classification_rule_activated", False))
    added_assumed_negative = len(doc.get("added_assumed_negative_indices", []))
    classification_rule_meta = (
        f"{classification_rule_id} (activated={str(classification_rule_activated).lower()}, "
        f"added_assumed_negative={added_assumed_negative})"
    )
    meta = (
        f"Pred included (classification universe): {len(doc['pred_indices'])} | "
        f"Manual included (mode truth positives): {len(doc['true_indices'])} | "
        f"Correct overlaps: {len(doc['correct_indices'])} | "
        f"Rule: {classification_rule_meta} | "
        f"Match statuses: {status_meta}"
    )

    missing_manual_msg = ""
    if doc.get("manual_missing_in_auto"):
        missing_manual_msg = "<p><strong>Manual study exists but PMID is missing from auto parsing outputs.</strong></p>"

    unmatched_html = ""
    if doc["unmatched_manual_names"]:
        joined = ", ".join(escape(x) for x in doc["unmatched_manual_names"])
        unmatched_html = f"<p><strong>Unmatched manual analyses:</strong> {joined}</p>"

    coord_tables = doc.get("coord_tables", [])
    table_meta_by_id = {str(t.get("table_id", "")).strip(): t for t in coord_tables}
    analyses_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in doc["analysis_rows"]:
        analyses_by_table[str(row.get("table_id", "")).strip()].append(row)

    # Preserve source table order when available, then append unknown/unlisted table groups.
    ordered_table_ids: list[str] = []
    for t in coord_tables:
        tid = str(t.get("table_id", "")).strip()
        if tid and tid not in ordered_table_ids:
            ordered_table_ids.append(tid)
    for tid in sorted(k for k in analyses_by_table.keys() if k and k not in ordered_table_ids):
        ordered_table_ids.append(tid)
    if "" in analyses_by_table:
        ordered_table_ids.append("")

    grouped_blocks: list[str] = []
    for group_index, table_id in enumerate(ordered_table_ids, start=1):
        rows_for_table = analyses_by_table.get(table_id, [])
        if not rows_for_table:
            continue

        table_meta = table_meta_by_id.get(table_id, {})
        table_label = str(table_meta.get("table_label", "")).strip()
        llm_table_captions = dedupe_keep_order(
            [
                clean_text(str(row.get("llm_table_caption", ""))).strip()
                for row in rows_for_table
                if clean_text(str(row.get("llm_table_caption", ""))).strip()
            ]
        )
        display_caption = clean_text(str(table_meta.get("table_caption", ""))).strip()
        if not display_caption and llm_table_captions:
            display_caption = llm_table_captions[0]
        table_heading = f"{group_index}) {table_label}" if table_label else f"{group_index}) Table"
        table_meta_lines = []
        if not table_label and table_id:
            table_meta_lines.append(
                f"<li><strong>Table ID:</strong> {escape(table_id)}</li>"
            )
        if display_caption:
            table_meta_lines.append(
                f"<li><strong>Caption:</strong> {escape(display_caption)}</li>"
            )
        if table_meta.get("table_foot"):
            table_meta_lines.append(
                f"<li><strong>Footer:</strong> {escape(str(table_meta.get('table_foot', '')))}</li>"
            )
        if not table_meta_lines and table_id:
            table_meta_lines.append(
                f"<li><strong>Table ID:</strong> {escape(table_id)}</li>"
            )
        if not table_meta_lines and not table_id:
            table_meta_lines.append("<li><strong>Table:</strong> Unspecified / not parsed</li>")

        # Make sparse table groups more compact, e.g. "2) Table ID: 46".
        if (
            len(table_meta_lines) == 1
            and not table_meta.get("table_label")
            and not display_caption
            and not table_meta.get("table_foot")
            and table_id
        ):
            table_heading = f"{group_index}) Table ID: {table_id}"
            table_meta_html = ""
        else:
            table_meta_html = f"<ul class=\"table-meta-list\">{''.join(table_meta_lines)}</ul>"

        rows_html = []
        for row in rows_for_table:
            parsed_name_html = escape(row["parsed_name"])
            parsed_description = str(row.get("parsed_description", "") or "").strip()
            if parsed_description:
                parsed_name_html += f"<br><span class=\"muted\">{escape(parsed_description)}</span>"
            rows_html.append(
                "<tr>"
                f"<td>{escape(row['analysis_id'])}</td>"
                f"<td>{parsed_name_html}</td>"
                f"<td class=\"decision-cell\"><span class=\"decision-pill {escape(row['model_decision_class'])}\">{escape(row['model_decision_icon'])}</span></td>"
                f"<td class=\"confusion-cell\"><span class=\"confusion-pill {escape(row['confusion_class'])}\">{escape(row['confusion_label'])}</span></td>"
                f"<td>{render_criteria_checks_cell(row, criterion_meta)}</td>"
                f"<td>{escape(row['reasoning'])}</td>"
                "</tr>"
            )

        grouped_blocks.append(
            "<div class=\"table-group\">"
            f"<h4>{escape(table_heading)}</h4>"
            f"{table_meta_html}"
            "<div class=\"table-wrap\">"
            "<table class=\"analysis-review-table\">"
            "<colgroup>"
            "<col class=\"col-analysis-id\">"
            "<col class=\"col-parsed-name\">"
            "<col class=\"col-model-decision\">"
            "<col class=\"col-matched-outcome\">"
            "<col class=\"col-criteria-checks\">"
            "<col class=\"col-reasoning\">"
            "</colgroup>"
            "<thead>"
            "<tr>"
            "<th>Analysis ID</th>"
            "<th>Parsed Analysis Name</th>"
            "<th>Model Decision</th>"
            "<th>Matched Outcome</th>"
            "<th>Unmet Criteria</th>"
            "<th>Model Reasoning</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{''.join(rows_html)}</tbody>"
            "</table>"
            "</div>"
            "</div>"
        )

    grouped_analysis_html = "".join(grouped_blocks)

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

    match_diag_html = render_match_diagnostics(
        doc.get("review_match_diagnostics", doc.get("match_diagnostics", []))
    )

    return f"""
<details class="doc-card">
  <summary><strong>PMID {escape(doc['pmid'])}</strong> | {escape(meta)}</summary>
  <p><a href="{escape(doc['pubmed_url'])}" target="_blank" rel="noopener noreferrer">PubMed full text page</a></p>
  {missing_manual_msg}
  {unmatched_html}
  <details class="inner-accordion" open>
    <summary>Parsed analyses and annotation reasoning</summary>
    {grouped_analysis_html}
  </details>
  <details class="inner-accordion" open>
    <summary>Manual-to-Auto Match Diagnostics</summary>
    {match_diag_html}
  </details>
  {fulltext_html}
  {tables_html}
</details>
"""

def render_html(
    annotation_name: str,
    mode_results: dict[str, dict[str, Any]],
    criteria: dict[str, Any] | None = None,
) -> str:
    criteria = criteria or {}
    criterion_meta = build_annotation_criteria_metadata(criteria, annotation_name)

    def get_mode_payload(mode_id: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
        payload = mode_results.get(mode_id, {})
        docs_in = payload.get("docs", {}) if isinstance(payload, dict) else {}
        metrics_in = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        docs_out = {
            bucket: list(docs_in.get(bucket, [])) if isinstance(docs_in, dict) else []
            for bucket in REVIEW_BUCKET_ORDER
        }
        return docs_out, (metrics_in if isinstance(metrics_in, dict) else {})

    accepted_docs, accepted_metrics = get_mode_payload("accepted")
    uncertain_docs, uncertain_metrics = get_mode_payload("uncertain")
    combined_docs, combined_metrics = get_mode_payload("combined")

    def render_mode_bucket_panel(
        mode_id: str,
        mode_title: str,
        docs: dict[str, list[dict[str, Any]]],
        metrics: dict[str, Any],
    ) -> str:
        sections = []
        for bucket in REVIEW_BUCKET_ORDER:
            cards = sorted(
                docs.get(bucket, []),
                key=lambda d: (
                    0 if d.get("fulltext") else 1,
                    len(str(d.get("pmid", ""))),
                    str(d.get("pmid", "")),
                ),
            )
            if cards:
                body = "\n".join(render_doc_card(card, criterion_meta=criterion_meta) for card in cards)
            else:
                body = "<p>No documents in this bucket.</p>"

            bm = metrics.get("bucket_match_counts", {}).get(
                bucket,
                {"accepted": 0, "uncertain": 0, "unmatched": 0},
            )
            bucket_summary = (
                f"<p><strong>Match status totals:</strong> accepted={int(bm.get('accepted', 0))} | "
                f"uncertain={int(bm.get('uncertain', 0))} | unmatched={int(bm.get('unmatched', 0))}</p>"
            )
            open_attr = " open" if bucket != "Correct" else ""
            bucket_anchor = f"{mode_id}-{REVIEW_BUCKET_IDS[bucket]}"
            sections.append(
                "<section id=\"{sid}\">"
                "<details class=\"bucket\"{open_attr}>"
                "<summary><h3>{bucket} ({count})</h3></summary>"
                "{bucket_summary}"
                "{body}"
                "</details>"
                "</section>".format(
                    sid=bucket_anchor,
                    open_attr=open_attr,
                    bucket=bucket,
                    count=len(cards),
                    bucket_summary=bucket_summary,
                    body=body,
                )
            )

        return (
            f"<section id=\"{mode_id}-section\" class=\"mode-panel\">"
            f"<h2>{escape(mode_title)}</h2>"
            f"{''.join(sections)}"
            "</section>"
        )

    def render_metric_rows(mode_label: str, metrics: dict[str, Any]) -> str:
        document_metrics = metrics.get("document_metrics", {})
        study_metrics = metrics.get("study_metrics", {})
        analysis_metrics = metrics.get("analysis_metrics", {})
        analysis_metrics_exhausted = metrics.get(
            "analysis_metrics_exhausted_manual_assumption",
            analysis_metrics,
        )
        assumed_negative_expansion = metrics.get("assumed_negative_expansion", {})
        if not isinstance(assumed_negative_expansion, dict):
            assumed_negative_expansion = {}

        document_tp = int(document_metrics.get("tp", metrics.get("tp", 0)))
        document_fp = int(document_metrics.get("fp", metrics.get("fp", 0)))
        document_fn = int(document_metrics.get("fn", metrics.get("fn", 0)))
        document_precision = float(document_metrics.get("precision", metrics.get("precision", 0.0)))
        document_recall = float(document_metrics.get("recall", metrics.get("recall", 0.0)))
        document_f1 = float(document_metrics.get("f1", metrics.get("f1", 0.0)))
        assumption_rule_id = clean_text(
            str(assumed_negative_expansion.get("activation_rule_id", EXHAUSTED_MANUAL_ASSUMPTION_RULE_ID))
        ).strip()
        assumption_activated_pmids = int(assumed_negative_expansion.get("activated_pmids", 0))
        assumption_added = int(assumed_negative_expansion.get("added_assumed_negative_analyses", 0))
        assumption_label = (
            "Analysis inclusion (exhausted-manual assumption; "
            f"activated_pmids={assumption_activated_pmids}, added_assumed_negative={assumption_added}, "
            f"rule={assumption_rule_id})"
        )

        return (
            "<tr>"
            f"<td>{escape(mode_label)}</td>"
            "<td>Document bucket overlap</td>"
            f"<td>{document_tp}</td>"
            f"<td>{document_fp}</td>"
            f"<td>{document_fn}</td>"
            f"<td>{document_precision:.3f}</td>"
            f"<td>{document_recall:.3f}</td>"
            f"<td>{document_f1:.3f}</td>"
            f"<td>{document_tp + document_fn}</td>"
            f"<td>{document_tp + document_fp}</td>"
            f"<td>{document_tp + document_fp + document_fn}</td>"
            "</tr>"
            "<tr>"
            f"<td>{escape(mode_label)}</td>"
            "<td>Study inclusion</td>"
            f"<td>{int(study_metrics.get('tp', 0))}</td>"
            f"<td>{int(study_metrics.get('fp', 0))}</td>"
            f"<td>{int(study_metrics.get('fn', 0))}</td>"
            f"<td>{float(study_metrics.get('precision', 0.0)):.3f}</td>"
            f"<td>{float(study_metrics.get('recall', 0.0)):.3f}</td>"
            f"<td>{float(study_metrics.get('f1', 0.0)):.3f}</td>"
            f"<td>{int(study_metrics.get('manual_studies', 0))}</td>"
            f"<td>{int(study_metrics.get('predicted_studies', 0))}</td>"
            f"<td>{int(study_metrics.get('overlap_pmids', 0))}</td>"
            "</tr>"
            "<tr>"
            f"<td>{escape(mode_label)}</td>"
            "<td>Analysis inclusion (matched-only baseline)</td>"
            f"<td>{int(analysis_metrics.get('tp', 0))}</td>"
            f"<td>{int(analysis_metrics.get('fp', 0))}</td>"
            f"<td>{int(analysis_metrics.get('fn', 0))}</td>"
            f"<td>{float(analysis_metrics.get('precision', 0.0)):.3f}</td>"
            f"<td>{float(analysis_metrics.get('recall', 0.0)):.3f}</td>"
            f"<td>{float(analysis_metrics.get('f1', 0.0)):.3f}</td>"
            f"<td>{int(analysis_metrics.get('manual_accepted_analyses', 0))}</td>"
            f"<td>{int(analysis_metrics.get('predicted_analyses', 0))}</td>"
            f"<td>{int(analysis_metrics.get('analysis_universe', 0))}</td>"
            "</tr>"
            "<tr>"
            f"<td>{escape(mode_label)}</td>"
            f"<td>{escape(assumption_label)}</td>"
            f"<td>{int(analysis_metrics_exhausted.get('tp', 0))}</td>"
            f"<td>{int(analysis_metrics_exhausted.get('fp', 0))}</td>"
            f"<td>{int(analysis_metrics_exhausted.get('fn', 0))}</td>"
            f"<td>{float(analysis_metrics_exhausted.get('precision', 0.0)):.3f}</td>"
            f"<td>{float(analysis_metrics_exhausted.get('recall', 0.0)):.3f}</td>"
            f"<td>{float(analysis_metrics_exhausted.get('f1', 0.0)):.3f}</td>"
            f"<td>{int(analysis_metrics_exhausted.get('manual_accepted_analyses', 0))}</td>"
            f"<td>{int(analysis_metrics_exhausted.get('predicted_analyses', 0))}</td>"
            f"<td>{int(analysis_metrics_exhausted.get('analysis_universe', 0))}</td>"
            "</tr>"
        )

    missing_pmids: set[str] = set()
    for metrics in [accepted_metrics, uncertain_metrics, combined_metrics]:
        missing_pmids |= {str(pmid) for pmid in metrics.get("missing_manual_pmids", [])}
    missing_pmids_sorted = sorted(missing_pmids, key=lambda x: (len(x), x))
    missing_html = ""
    if missing_pmids_sorted:
        missing_items = "".join(
            f"<li><a href=\"https://pubmed.ncbi.nlm.nih.gov/{escape(pmid)}/\" target=\"_blank\" rel=\"noopener noreferrer\">PMID {escape(pmid)}</a></li>"
            for pmid in missing_pmids_sorted
        )
        missing_html = (
            "<section id=\"missing-manual\">"
            "<details class=\"bucket\" open>"
            f"<summary><h2>Manual PMIDs Missing In Auto Parsing ({len(missing_pmids_sorted)})</h2></summary>"
            "<p>These studies exist in manual NiMADS but were not found in auto parsed outputs for this project.</p>"
            f"<ul>{missing_items}</ul>"
            "</details>"
            "</section>"
        )

    global_criteria = criteria.get("global", {}) if isinstance(criteria, dict) else {}
    per_annotation = criteria.get("annotations", {}) if isinstance(criteria, dict) else {}
    annotation_criteria = per_annotation.get(annotation_name, {}) if isinstance(per_annotation, dict) else {}

    def render_criteria_items(items: list[tuple[str, str]]) -> str:
        if not items:
            return "<p class=\"muted\">None specified.</p>"
        lines = []
        for code, text in items:
            code_text = clean_text(str(code)).strip()
            text_value = clean_text(str(text)).strip()
            if code_text:
                lines.append(f"<li><strong>{escape(code_text)}:</strong> {escape(text_value)}</li>")
            else:
                lines.append(f"<li>{escape(text_value)}</li>")
        return "<ul class=\"criteria-list\">" + "".join(lines) + "</ul>"

    global_inclusion = list(global_criteria.get("inclusion", [])) if isinstance(global_criteria, dict) else []
    global_exclusion = list(global_criteria.get("exclusion", [])) if isinstance(global_criteria, dict) else []
    annotation_inclusion = list(annotation_criteria.get("inclusion", [])) if isinstance(annotation_criteria, dict) else []
    annotation_exclusion = list(annotation_criteria.get("exclusion", [])) if isinstance(annotation_criteria, dict) else []

    criteria_html = (
        "<section id=\"criteria\" class=\"criteria-panel\">"
        "<h2>Inclusion / Exclusion Criteria</h2>"
        "<div class=\"criteria-block\">"
        "<h3>Global Criteria</h3>"
        "<div class=\"criteria-grid\">"
        "<div><h4>Inclusion</h4>{global_inclusion}</div>"
        "<div><h4>Exclusion</h4>{global_exclusion}</div>"
        "</div>"
        "</div>"
        "<div class=\"criteria-block\">"
        "<h3>Annotation-Specific Criteria ({annotation_name})</h3>"
        "<div class=\"criteria-grid\">"
        "<div><h4>Inclusion</h4>{annotation_inclusion}</div>"
        "<div><h4>Exclusion</h4>{annotation_exclusion}</div>"
        "</div>"
        "</div>"
        "</section>"
    ).format(
        global_inclusion=render_criteria_items(global_inclusion),
        global_exclusion=render_criteria_items(global_exclusion),
        annotation_name=escape(annotation_name),
        annotation_inclusion=render_criteria_items(annotation_inclusion),
        annotation_exclusion=render_criteria_items(annotation_exclusion),
    )

    criteria_error_analysis = accepted_metrics.get("criteria_error_analysis", {})
    criteria_error_coverage = (
        criteria_error_analysis.get("coverage", {})
        if isinstance(criteria_error_analysis, dict)
        else {}
    )
    criteria_error_ranked = (
        criteria_error_analysis.get("ranked_categories", {})
        if isinstance(criteria_error_analysis, dict)
        else {}
    )

    def render_criteria_error_table(category_id: str) -> str:
        rows = list(criteria_error_ranked.get(category_id, [])) if isinstance(criteria_error_ranked, dict) else []
        if not rows:
            return "<p class=\"muted\">No criteria IDs from this category were detected in misclassified analyses.</p>"

        body_rows = []
        for row in rows:
            body_rows.append(
                "<tr>"
                f"<td>{escape(str(row.get('criterion', '')))}</td>"
                f"<td>{escape(str(row.get('criterion_type', '')))}</td>"
                f"<td>{int(row.get('error_mentions', 0))}</td>"
                f"<td>{int(row.get('correct_mentions', 0))}</td>"
                f"<td>{float(row.get('error_rate_vs_correct', 0.0)):.3f}</td>"
                "</tr>"
            )
        return (
            "<div class=\"table-wrap\">"
            "<table class=\"criteria-error-table\">"
            "<thead><tr>"
            "<th>Criterion</th>"
            "<th>Type (Inclusion/Exclusion)</th>"
            "<th>Error Mentions</th>"
            "<th>Correct Mentions</th>"
            "<th>Error Rate vs Correct</th>"
            "</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
            "</div>"
        )

    criteria_error_sections = []
    for category_id in CRITERIA_ERROR_CATEGORY_ORDER:
        label = CRITERIA_ERROR_CATEGORY_RULES[category_id]["label"]
        criteria_error_sections.append(
            "<div class=\"criteria-error-block\">"
            f"<h3>{escape(label)}</h3>"
            f"{render_criteria_error_table(category_id)}"
            "</div>"
        )

    criteria_error_html = (
        "<section id=\"criteria-errors\" class=\"criteria-panel\">"
        "<h2>Commonly Misapplied Criteria (from ACCEPTED strict evaluation)</h2>"
        "<p class=\"muted\">"
        "Computed at analysis level from explicit criterion IDs in model reasoning. "
        f"Rows considered={int(criteria_error_coverage.get('analysis_rows_considered', 0))}, "
        f"rows with criteria IDs={int(criteria_error_coverage.get('rows_with_codes', 0))}, "
        f"rows without criteria IDs={int(criteria_error_coverage.get('rows_without_codes', 0))}, "
        f"coverage={float(criteria_error_coverage.get('coverage_rate', 0.0)):.3f}."
        "</p>"
        f"{''.join(criteria_error_sections)}"
        "</section>"
    )

    accepted_section_html = render_mode_bucket_panel(
        mode_id="accepted",
        mode_title="ACCEPTED Matches (Strict)",
        docs=accepted_docs,
        metrics=accepted_metrics,
    )
    uncertain_section_html = render_mode_bucket_panel(
        mode_id="uncertain",
        mode_title="UNCERTAIN Matches (Borderline)",
        docs=uncertain_docs,
        metrics=uncertain_metrics,
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
    .table-group {{ margin: 0.8rem 0 1rem; padding: 0.6rem; border: 1px solid var(--line); border-radius: 8px; background: #fcfcfd; }}
    .table-group h4 {{ margin: 0 0 0.35rem; }}
    .table-meta-list {{ margin: 0 0 0.55rem 0; padding-left: 1.1rem; }}
    .table-meta-list li {{ margin: 0.15rem 0; }}
    .muted {{ color: #5b6775; font-size: 0.84rem; }}
    .analysis-review-table {{ table-layout: fixed; width: 100%; }}
    .analysis-review-table th, .analysis-review-table td {{ overflow-wrap: anywhere; word-break: normal; }}
    .analysis-review-table col.col-analysis-id {{ width: 12%; }}
    .analysis-review-table col.col-parsed-name {{ width: 21%; }}
    .analysis-review-table col.col-model-decision {{ width: 7%; }}
    .analysis-review-table col.col-matched-outcome {{ width: 8%; }}
    .analysis-review-table col.col-criteria-checks {{ width: 27%; }}
    .analysis-review-table col.col-reasoning {{ width: 25%; }}
    .criteria-status-wrap {{ display: grid; gap: 0.3rem; }}
    .criteria-status-row {{ display: flex; flex-wrap: wrap; gap: 0.25rem; align-items: center; }}
    .criteria-status-label {{ min-width: 6.5rem; font-weight: 600; font-size: 0.77rem; color: #334155; }}
    .criteria-pill {{
      display: inline-flex;
      align-items: center;
      padding: 0.08rem 0.45rem;
      border-radius: 999px;
      border: 1px solid transparent;
      font-size: 0.76rem;
      font-weight: 600;
      line-height: 1.2;
    }}
    .criteria-pill.criteria-good {{ background: #e9f8ef; color: #166534; border-color: #b7e4c6; }}
    .criteria-pill.criteria-bad {{ background: #fdecec; color: #991b1b; border-color: #f6caca; }}
    .criteria-pill.criteria-unknown {{ background: #eef2f7; color: #334155; border-color: #cbd5e1; }}
    .criteria-panel {{ margin-top: 1rem; border-top: 1px solid var(--line); padding-top: 0.8rem; }}
    .criteria-panel h2 {{ margin: 0 0 0.7rem 0; }}
    .criteria-block {{ background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; padding: 0.65rem; margin-bottom: 0.7rem; }}
    .criteria-block h3 {{ margin: 0 0 0.45rem 0; font-size: 1rem; }}
    .criteria-block h4 {{ margin: 0 0 0.35rem 0; font-size: 0.92rem; }}
    .criteria-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem; }}
    .criteria-list {{ margin: 0; padding-left: 1.1rem; }}
    .criteria-list li {{ margin: 0.2rem 0; }}
    .criteria-error-block {{ background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; padding: 0.65rem; margin-bottom: 0.7rem; }}
    .criteria-error-block h3 {{ margin: 0 0 0.45rem 0; font-size: 1rem; }}
    @media (max-width: 900px) {{ .criteria-grid {{ grid-template-columns: 1fr; }} }}
    a {{ color: #0e4f85; }}
  </style>
</head>
<body>
  <header>
    <a id="top"></a>
    <h1>{escape(annotation_name)} report</h1>
    <p>Manual benchmark is sliced to the auto PMID universe from <code>outputs/nimads_annotation.json</code>. This report shows ACCEPTED and UNCERTAIN matches in separate sections.</p>
    <p class="muted">Analysis-level metrics below include both matched-only baseline and an exhausted-manual assumption view (adds unmatched auto analyses as negatives only when all manual analyses in a PMID are accepted). Document buckets/cards default to the exhausted-manual assumption rule when the mode supports accepted matches and the PMID-level guard is satisfied.</p>
    <p class="muted">Unmet Criteria is computed from explicit model outputs in <code>outputs/annotation_results.json</code> and shows only <code>exclusion_criteria_applied</code> entries (no inferred unmet inclusion criteria).</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Mode</th>
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
          {render_metric_rows("STRICT (accepted only)", accepted_metrics)}
          {render_metric_rows("COMBINED (accepted + uncertain)", combined_metrics)}
        </tbody>
      </table>
    </div>
  </header>
  <nav class="top-nav">
    <a href="#criteria">Criteria</a>
    <a href="#criteria-errors">Criteria Errors</a>
    <a href="#accepted-section">ACCEPTED Sections</a>
    <a href="#uncertain-section">UNCERTAIN Sections</a>
    <a href="#missing-manual">Missing PMIDs ({len(missing_pmids_sorted)})</a>
    <a href="#top">Top</a>
  </nav>
  {criteria_html}
  {criteria_error_html}
  {accepted_section_html}
  {uncertain_section_html}
  {missing_html}
</body>
</html>
"""

def render_overall_summary_html(
    metrics_by_annotation_by_mode: dict[str, dict[str, dict[str, Any]]],
) -> str:
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

    def render_annotation_link(annotation: str) -> str:
        href = f"{quote(annotation)}.html"
        return f"<a href=\"{escape(href)}\">{escape(annotation)}</a>"

    def render_cross_annotation_criteria_table(table_rows: list[dict[str, Any]]) -> str:
        if not table_rows:
            return "<p class=\"muted\">No criteria misapplication rows found.</p>"
        sorted_rows = sorted(
            table_rows,
            key=lambda row: (
                -float(row.get("error_rate_vs_correct", 0.0)),
                -int(row.get("error_mentions", 0)),
                str(row.get("annotation", "")),
                str(row.get("criterion", "")),
            ),
        )
        html_rows = []
        for row in sorted_rows:
            html_rows.append(
                "<tr>"
                f"<td><a href=\"{escape(str(row.get('annotation_href', '')))}\">{escape(str(row.get('annotation', '')))}</a></td>"
                f"<td>{escape(str(row.get('category', '')))}</td>"
                f"<td>{escape(str(row.get('criterion', '')))}</td>"
                f"<td>{escape(str(row.get('criterion_type', '')))}</td>"
                f"<td>{int(row.get('error_mentions', 0))}</td>"
                f"<td>{int(row.get('correct_mentions', 0))}</td>"
                f"<td>{float(row.get('error_rate_vs_correct', 0.0)):.3f}</td>"
                "</tr>"
            )
        return (
            "<div class=\"table-wrap\">"
            "<table>"
            "<thead><tr>"
            "<th>Annotation</th>"
            "<th>Error Category</th>"
            "<th>Criterion</th>"
            "<th>Type (Inclusion/Exclusion)</th>"
            "<th>Error Mentions</th>"
            "<th>Correct Mentions</th>"
            "<th>Error Rate vs Correct</th>"
            "</tr></thead>"
            f"<tbody>{''.join(html_rows)}</tbody>"
            "</table>"
            "</div>"
        )

    strict_metrics_by_annotation = metrics_by_annotation_by_mode.get("accepted", {})
    criteria_global_rows: list[dict[str, Any]] = []
    criteria_annotation_rows: list[dict[str, Any]] = []
    for annotation_name in ACTIVE_ANNOTATION_NAMES:
        metrics = strict_metrics_by_annotation.get(annotation_name, {})
        criteria_error_analysis = metrics.get("criteria_error_analysis", {})
        ranked_categories = (
            criteria_error_analysis.get("ranked_categories", {})
            if isinstance(criteria_error_analysis, dict)
            else {}
        )
        for category_id in CRITERIA_ERROR_CATEGORY_ORDER:
            category_rows = (
                ranked_categories.get(category_id, [])
                if isinstance(ranked_categories, dict)
                else []
            )
            for item in category_rows:
                row = {
                    "annotation": annotation_name,
                    "annotation_href": f"{quote(annotation_name)}.html",
                    "category": CRITERIA_ERROR_CATEGORY_RULES[category_id]["label"],
                    "criterion": str(item.get("criterion", "")),
                    "criterion_type": str(item.get("criterion_type", "")),
                    "scope": str(item.get("scope", "")),
                    "error_mentions": int(item.get("error_mentions", 0)),
                    "correct_mentions": int(item.get("correct_mentions", 0)),
                    "error_rate_vs_correct": float(item.get("error_rate_vs_correct", 0.0)),
                }
                if row["scope"] == "global":
                    criteria_global_rows.append(row)
                else:
                    criteria_annotation_rows.append(row)

    mode_sections: list[str] = []
    for mode_id in OVERALL_SUMMARY_MODE_ORDER:
        mode_cfg = EVAL_MODE_CONFIGS.get(mode_id, {})
        mode_label = str(mode_cfg.get("label", mode_id.upper()))
        metrics_by_annotation = metrics_by_annotation_by_mode.get(mode_id, {})

        rows: list[dict[str, Any]] = []
        analysis_rows_baseline: list[dict[str, Any]] = []
        analysis_rows_exhausted: list[dict[str, Any]] = []
        for annotation_name in ACTIVE_ANNOTATION_NAMES:
            metrics = metrics_by_annotation.get(annotation_name, {})
            study = metrics.get("study_metrics", {})
            analysis_baseline = metrics.get("analysis_metrics", {})
            analysis_exhausted = metrics.get(
                "analysis_metrics_exhausted_manual_assumption",
                analysis_baseline,
            )
            assumption_meta = metrics.get("assumed_negative_expansion", {})
            if not isinstance(assumption_meta, dict):
                assumption_meta = {}
            rows.append(
                {
                    "annotation": annotation_name,
                    "tp": int(study.get("tp", 0)),
                    "fp": int(study.get("fp", 0)),
                    "fn": int(study.get("fn", 0)),
                    "tn": int(study.get("tn", 0)),
                    "precision": float(study.get("precision", 0.0)),
                    "recall": float(study.get("recall", 0.0)),
                    "f1": float(study.get("f1", 0.0)),
                    "accuracy": float(study.get("accuracy", 0.0)),
                    "overlap_pmids": int(study.get("overlap_pmids", 0)),
                    "manual_studies": int(study.get("manual_studies", 0)),
                    "predicted_studies": int(study.get("predicted_studies", 0)),
                }
            )
            analysis_rows_baseline.append(
                {
                    "annotation": annotation_name,
                    "tp": int(analysis_baseline.get("tp", 0)),
                    "fp": int(analysis_baseline.get("fp", 0)),
                    "fn": int(analysis_baseline.get("fn", 0)),
                    "tn": int(analysis_baseline.get("tn", 0)),
                    "precision": float(analysis_baseline.get("precision", 0.0)),
                    "recall": float(analysis_baseline.get("recall", 0.0)),
                    "f1": float(analysis_baseline.get("f1", 0.0)),
                    "accuracy": float(analysis_baseline.get("accuracy", 0.0)),
                    "manual_accepted_analyses": int(analysis_baseline.get("manual_accepted_analyses", 0)),
                    "predicted_analyses": int(analysis_baseline.get("predicted_analyses", 0)),
                    "analysis_universe": int(analysis_baseline.get("analysis_universe", 0)),
                }
            )
            analysis_rows_exhausted.append(
                {
                    "annotation": annotation_name,
                    "tp": int(analysis_exhausted.get("tp", 0)),
                    "fp": int(analysis_exhausted.get("fp", 0)),
                    "fn": int(analysis_exhausted.get("fn", 0)),
                    "tn": int(analysis_exhausted.get("tn", 0)),
                    "precision": float(analysis_exhausted.get("precision", 0.0)),
                    "recall": float(analysis_exhausted.get("recall", 0.0)),
                    "f1": float(analysis_exhausted.get("f1", 0.0)),
                    "accuracy": float(analysis_exhausted.get("accuracy", 0.0)),
                    "manual_accepted_analyses": int(analysis_exhausted.get("manual_accepted_analyses", 0)),
                    "predicted_analyses": int(analysis_exhausted.get("predicted_analyses", 0)),
                    "analysis_universe": int(analysis_exhausted.get("analysis_universe", 0)),
                    "activated_pmids": int(assumption_meta.get("activated_pmids", 0)),
                    "added_assumed_negative_analyses": int(
                        assumption_meta.get("added_assumed_negative_analyses", 0)
                    ),
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

        analysis_micro_tp = sum(r["tp"] for r in analysis_rows_baseline)
        analysis_micro_fp = sum(r["fp"] for r in analysis_rows_baseline)
        analysis_micro_fn = sum(r["fn"] for r in analysis_rows_baseline)
        analysis_micro_tn = sum(r["tn"] for r in analysis_rows_baseline)
        analysis_micro_prf = compute_prf(tp=analysis_micro_tp, fp=analysis_micro_fp, fn=analysis_micro_fn)
        analysis_micro_total = analysis_micro_tp + analysis_micro_fp + analysis_micro_fn + analysis_micro_tn
        analysis_micro_accuracy = (
            (analysis_micro_tp + analysis_micro_tn) / analysis_micro_total
            if analysis_micro_total
            else 0.0
        )

        analysis_exhausted_micro_tp = sum(r["tp"] for r in analysis_rows_exhausted)
        analysis_exhausted_micro_fp = sum(r["fp"] for r in analysis_rows_exhausted)
        analysis_exhausted_micro_fn = sum(r["fn"] for r in analysis_rows_exhausted)
        analysis_exhausted_micro_tn = sum(r["tn"] for r in analysis_rows_exhausted)
        analysis_exhausted_micro_prf = compute_prf(
            tp=analysis_exhausted_micro_tp,
            fp=analysis_exhausted_micro_fp,
            fn=analysis_exhausted_micro_fn,
        )
        analysis_exhausted_micro_total = (
            analysis_exhausted_micro_tp
            + analysis_exhausted_micro_fp
            + analysis_exhausted_micro_fn
            + analysis_exhausted_micro_tn
        )
        analysis_exhausted_micro_accuracy = (
            (analysis_exhausted_micro_tp + analysis_exhausted_micro_tn) / analysis_exhausted_micro_total
            if analysis_exhausted_micro_total
            else 0.0
        )
        assumption_activated_pmids_total = sum(r["activated_pmids"] for r in analysis_rows_exhausted)
        assumption_added_total = sum(r["added_assumed_negative_analyses"] for r in analysis_rows_exhausted)

        row_html = []
        for row in rows:
            row_html.append(
                "<tr>"
                f"<td>{render_annotation_link(row['annotation'])}</td>"
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

        analysis_baseline_row_html = []
        for row in analysis_rows_baseline:
            analysis_baseline_row_html.append(
                "<tr>"
                f"<td>{render_annotation_link(row['annotation'])}</td>"
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

        analysis_exhausted_row_html = []
        for row in analysis_rows_exhausted:
            analysis_exhausted_row_html.append(
                "<tr>"
                f"<td>{render_annotation_link(row['annotation'])}</td>"
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
                f"<td>{row['activated_pmids']}</td>"
                f"<td>{row['added_assumed_negative_analyses']}</td>"
                f"<td>{render_metric_bars(row['precision'], row['recall'], row['f1'])}</td>"
                f"<td>{render_confusion_plot(row['tp'], row['fp'], row['fn'], row['tn'])}</td>"
                "</tr>"
            )

        mode_sections.append(
            f"""
  <section>
    <h2>{escape(mode_label)} Aggregates</h2>
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
            <td>Matched-only analyses micro (pooled confusion)</td>
            <td>{analysis_micro_tp}</td>
            <td>{analysis_micro_fp}</td>
            <td>{analysis_micro_fn}</td>
            <td>{analysis_micro_tn}</td>
            <td>{float(analysis_micro_prf.get('precision', 0.0)):.3f}</td>
            <td>{float(analysis_micro_prf.get('recall', 0.0)):.3f}</td>
            <td>{float(analysis_micro_prf.get('f1', 0.0)):.3f}</td>
            <td>{analysis_micro_accuracy:.3f}</td>
          </tr>
          <tr>
            <td>Exhausted-manual assumption analyses micro (pooled confusion; activated_pmids={assumption_activated_pmids_total}, added_assumed_negative={assumption_added_total})</td>
            <td>{analysis_exhausted_micro_tp}</td>
            <td>{analysis_exhausted_micro_fp}</td>
            <td>{analysis_exhausted_micro_fn}</td>
            <td>{analysis_exhausted_micro_tn}</td>
            <td>{float(analysis_exhausted_micro_prf.get('precision', 0.0)):.3f}</td>
            <td>{float(analysis_exhausted_micro_prf.get('recall', 0.0)):.3f}</td>
            <td>{float(analysis_exhausted_micro_prf.get('f1', 0.0)):.3f}</td>
            <td>{analysis_exhausted_micro_accuracy:.3f}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>{escape(mode_label)} Per-Annotation Study-Level Metrics</h2>
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
    <h2>{escape(mode_label)} Per-Annotation Matched-Only Analysis Metrics</h2>
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
          {''.join(analysis_baseline_row_html)}
        </tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>{escape(mode_label)} Per-Annotation Exhausted-Manual-Assumption Analysis Metrics</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Annotation</th>
            <th>Expanded Analysis Universe</th>
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
            <th>Activated PMIDs</th>
            <th>Added Assumed Negatives</th>
            <th>PRF Plot</th>
            <th>Confusion Plot</th>
          </tr>
        </thead>
        <tbody>
          {''.join(analysis_exhausted_row_html)}
        </tbody>
      </table>
    </div>
  </section>
"""
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
    <p>Metrics are shown for STRICT (accepted only) and COMBINED (accepted + uncertain) evaluations.</p>
    <p>Analysis-level sections report both matched-only baseline metrics and exhausted-manual-assumption metrics.</p>
  </section>
  {''.join(mode_sections)}
  <section>
    <h2>Cross-Annotation Criteria Misapplication (Global Criteria, Strict)</h2>
    <p>Aggregated from strict (accepted-only) criteria-error analysis. Rows are sorted by error rate vs correctly classified mentions.</p>
    {render_cross_annotation_criteria_table(criteria_global_rows)}
  </section>
  <section>
    <h2>Cross-Annotation Criteria Misapplication (Annotation-Specific Criteria, Strict)</h2>
    <p>Aggregated from strict (accepted-only) criteria-error analysis. Rows are sorted by error rate vs correctly classified mentions.</p>
    {render_cross_annotation_criteria_table(criteria_annotation_rows)}
  </section>
</body>
</html>
"""


def make_json_friendly(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_friendly(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_friendly(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_friendly(item) for item in value]
    if isinstance(value, set):
        return [make_json_friendly(item) for item in sorted(value, key=lambda x: str(x))]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_annotation_metrics_export_payload(
    *,
    project_output_dir: Path,
    metrics_by_annotation_by_mode: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    mode_ids = ("accepted", "combined")
    per_mode: dict[str, dict[str, dict[str, Any]]] = {}

    for mode_id in mode_ids:
        metrics_by_annotation = metrics_by_annotation_by_mode.get(mode_id, {})
        per_mode[mode_id] = {}
        for annotation_name, metrics in metrics_by_annotation.items():
            per_mode[mode_id][annotation_name] = {
                "document_metrics": make_json_friendly(metrics.get("document_metrics", {})),
                "study_metrics": make_json_friendly(metrics.get("study_metrics", {})),
                "analysis_metrics": make_json_friendly(metrics.get("analysis_metrics", {})),
                "analysis_metrics_exhausted_manual_assumption": make_json_friendly(
                    metrics.get(
                        "analysis_metrics_exhausted_manual_assumption",
                        metrics.get("analysis_metrics", {}),
                    )
                ),
                "assumed_negative_expansion": make_json_friendly(
                    metrics.get("assumed_negative_expansion", {})
                ),
            }

    try:
        project_name = infer_project_name(project_output_dir)
    except Exception:
        project_name = project_output_dir.name

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "project_name": project_name,
        "project_output_dir": str(project_output_dir.resolve()),
        "mode_metadata": {
            mode_id: {
                "label": str(EVAL_MODE_CONFIGS.get(mode_id, {}).get("label", mode_id.upper())),
                "allowed_statuses": sorted(
                    str(status).strip().lower()
                    for status in set(EVAL_MODE_CONFIGS.get(mode_id, {}).get("allowed_statuses", set()))
                ),
            }
            for mode_id in mode_ids
        },
        "metrics_by_mode": per_mode,
    }


def write_annotation_metrics_export(
    *,
    project_output_dir: Path,
    review_output_dir: Path,
    metrics_by_annotation_by_mode: dict[str, dict[str, dict[str, Any]]],
) -> Path:
    payload = build_annotation_metrics_export_payload(
        project_output_dir=project_output_dir,
        metrics_by_annotation_by_mode=metrics_by_annotation_by_mode,
    )
    output_path = review_output_dir / "annotation_metrics_by_mode.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path




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
    parser.add_argument(
        "--parser-review",
        type=Path,
        default=None,
        help=(
            "Optional parser-failure review JSON. Confirmed parser misses remain "
            "penalties; matching/gold/expected differences are credited; "
            "non-evaluable source/scope dispositions are excluded from the "
            "parser-scoring denominator."
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
    if decimal_manual_coordinate_handling == "convert_to_talairach" and mni2tal is None:
        raise ImportError(
            "--decimal-manual-coordinate-handling convert_to_talairach requires NiMARE "
            "(nimare.utils.mni2tal). Install nimare or choose a different handling mode."
        )

    manual_dir = resolve_manual_dir(project_output_dir, args.manual_dir)
    coordinate_parsing_results = project_output_dir / "outputs" / "coordinate_parsing_results.json"
    if not coordinate_parsing_results.exists():
        raise FileNotFoundError(f"Missing coordinate parsing results: {coordinate_parsing_results}")

    auto_by_pmid = load_auto_parsed_data(coordinate_parsing_results)
    manual_by_pmid, manual_study_names_by_pmid = load_manual_analyses_overall(manual_dir)
    pubget_by_pmid = build_pubget_index(project_output_dir)
    table_auto_by_pmid, table_source_info = load_table_only_auto_data(project_output_dir)
    match_result = build_match_results_overall(
        manual_analyses_by_pmid=manual_by_pmid,
        manual_study_names_by_pmid=manual_study_names_by_pmid,
        auto_parsed_by_pmid=auto_by_pmid,
        coord_accept_override_threshold=float(args.coord_accept_override_threshold),
        decimal_manual_coordinate_handling=decimal_manual_coordinate_handling,
        converted_talairach_exact_axis_tolerance=float(args.converted_talairach_exact_axis_tolerance),
    )
    if args.parser_review is not None:
        parser_review_path = args.parser_review.expanduser().resolve()
        if not parser_review_path.exists():
            raise FileNotFoundError(
                f"--parser-review does not exist: {parser_review_path}"
            )
        project_name = infer_project_name(project_output_dir)
        review_entries = load_parser_review_entries(
            parser_review_path,
            project_name=project_name,
        )
        adjustment = apply_parser_review_adjustments(
            match_result,
            review_entries=review_entries,
            project_name=project_name,
            review_path=parser_review_path,
        )
        print(
            "parser_review_adjustment: "
            f"project={project_name} "
            f"matched={adjustment['matched_review_entries']} "
            f"credited={adjustment['credited_as_accepted']} "
            f"excluded={adjustment['excluded_non_parser_evaluable']} "
            f"parser_misses={adjustment['confirmed_parser_misses']}"
        )
    match_result["table_only_baseline"] = build_table_only_baseline_summary(
        manual_analyses_by_pmid=manual_by_pmid,
        manual_study_names_by_pmid=manual_study_names_by_pmid,
        table_auto_by_pmid=table_auto_by_pmid,
        table_source_info=table_source_info,
        coord_accept_override_threshold=float(args.coord_accept_override_threshold),
        decimal_manual_coordinate_handling=decimal_manual_coordinate_handling,
        converted_talairach_exact_axis_tolerance=float(args.converted_talairach_exact_axis_tolerance),
    )
    annotate_match_result_with_pubget(match_result, pubget_by_pmid)
    write_match_artifacts(match_output_dir, match_result, pubget_by_pmid=pubget_by_pmid)

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
    table_baseline = match_result.get("table_only_baseline", {})
    if table_baseline.get("available"):
        print(
            "table_only_baseline: "
            f"matched_pct={float(table_baseline.get('matched_pct', 0.0)):.3f} "
            f"accepted={int(table_baseline.get('accepted', 0))} "
            f"uncertain={int(table_baseline.get('uncertain', 0))} "
            f"unmatched={int(table_baseline.get('unmatched', 0))} "
            f"table_units={int(table_baseline.get('table_units', 0))} "
            f"source={table_baseline.get('source', '')}"
        )
    else:
        print(f"table_only_baseline: unavailable ({table_baseline.get('reason', '')})")
    print(f"Wrote matching artifacts to {match_output_dir}")
    return match_result


def run_annotation_review_stage(
    args: argparse.Namespace,
    project_output_dir: Path,
    match_input_dir: Path,
    review_output_dir: Path,
) -> None:
    annotation_mapping_path = resolve_project_annotation_mapping_path(
        project_output_dir,
        args.annotation_mapping_path,
    )
    configure_active_annotations(annotation_mapping_path)

    annotation_results = project_output_dir / "outputs" / "annotation_results.json"
    coordinate_parsing_results = project_output_dir / "outputs" / "coordinate_parsing_results.json"
    auto_annotation_path = project_output_dir / "outputs" / "nimads_annotation.json"
    criteria_mapping_path = project_output_dir / "outputs" / "criteria_mapping.json"
    retrieval_dir = project_output_dir / "retrieval" / "pubget_data"
    manual_annotation_path = resolve_manual_annotation_path(project_output_dir, args.manual_annotation_path)
    criteria = load_annotation_criteria(criteria_mapping_path)
    if not criteria_mapping_path.exists():
        print(f"Warning: criteria mapping not found at {criteria_mapping_path}; criteria section may be empty.")

    parsed_analyses = load_auto_parsed_analysis_info(coordinate_parsing_results)
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

    review_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_annotation_by_mode: dict[str, dict[str, dict[str, Any]]] = {
        mode_id: {}
        for mode_id in OVERALL_SUMMARY_MODE_ORDER
    }
    for annotation_name in ACTIVE_ANNOTATION_NAMES:
        mode_results: dict[str, dict[str, Any]] = {}
        for mode_id, mode_cfg in EVAL_MODE_CONFIGS.items():
            docs, metrics = classify_documents(
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

        html = render_html(annotation_name, mode_results, criteria=criteria)
        output_path = review_output_dir / f"{annotation_name}.html"
        output_path.write_text(html, encoding="utf-8")

        strict_metrics = mode_results.get("accepted", {}).get("metrics", {})
        combined_metrics = mode_results.get("combined", {}).get("metrics", {})
        print(
            f"Wrote {output_path} | "
            f"strict_doc_f1={float(strict_metrics.get('f1', 0.0)):.3f} "
            f"strict_study_f1={float(strict_metrics.get('study_metrics', {}).get('f1', 0.0)):.3f} "
            f"strict_analysis_f1={float(strict_metrics.get('analysis_metrics', {}).get('f1', 0.0)):.3f} "
            f"strict_analysis_f1_assumption={float(strict_metrics.get('analysis_metrics_exhausted_manual_assumption', {}).get('f1', strict_metrics.get('analysis_metrics', {}).get('f1', 0.0))):.3f} "
            f"combined_doc_f1={float(combined_metrics.get('f1', 0.0)):.3f} "
            f"combined_study_f1={float(combined_metrics.get('study_metrics', {}).get('f1', 0.0)):.3f} "
            f"combined_analysis_f1={float(combined_metrics.get('analysis_metrics', {}).get('f1', 0.0)):.3f} "
            f"combined_analysis_f1_assumption={float(combined_metrics.get('analysis_metrics_exhausted_manual_assumption', {}).get('f1', combined_metrics.get('analysis_metrics', {}).get('f1', 0.0))):.3f} "
            f"missing_manual_pmids={len(strict_metrics.get('missing_manual_pmids', []))}"
        )

    overall_summary_html = render_overall_summary_html(metrics_by_annotation_by_mode)
    overall_summary_path = review_output_dir / "overall_submeta_summary.html"
    overall_summary_path.write_text(overall_summary_html, encoding="utf-8")
    print(f"Wrote {overall_summary_path}")
    annotation_metrics_json_path = write_annotation_metrics_export(
        project_output_dir=project_output_dir,
        review_output_dir=review_output_dir,
        metrics_by_annotation_by_mode=metrics_by_annotation_by_mode,
    )
    print(f"Wrote {annotation_metrics_json_path}")


def main() -> None:
    args = parse_args()
    project_output_dir = infer_project_output_dir(args.project_output_dir)

    match_output_dir = resolve_output_dir(project_output_dir, args.match_output_dir)
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

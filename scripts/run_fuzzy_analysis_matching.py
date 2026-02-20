#!/usr/bin/env python3
"""Run coordinate-first fuzzy matching between manual and auto analyses.

Outputs:
- match_results_overall.json
- fuzzy_matching_report.html

Manual analyses are loaded from merged NiMADS artifacts:
- nimads_studyset.json

Scoring/reporting uses only overlap PMIDs (manual ∩ auto).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


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

def clean_text(value: str) -> str:
    return "".join(ch for ch in str(value) if ch >= " " or ch in "\n\t\r")

def normalize_text(value: str) -> str:
    text = clean_text(value).lower().strip()
    text = text.replace(">", " > ")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_args() -> argparse.Namespace:
    default_manual_dir = Path("../neurometabench/data/nimads/social")

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
        "--manual-dir",
        type=Path,
        default=default_manual_dir,
        help="Path to project NiMADS dir or merged dir containing nimads_studyset.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for match JSON + summary HTML. Defaults to sibling reports/.",
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
            if not (
                (outputs_dir / "annotation_results.json").exists()
                and (outputs_dir / "coordinate_parsing_results.json").exists()
            ):
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

    exact = [c for c in pool if c.name == "rev3-search-all_pmids-studyann-ft"]
    if exact:
        selected = max(exact, key=lambda p: (p / "outputs" / "annotation_results.json").stat().st_mtime)
        print(f"Auto-selected project output dir (exact preferred): {selected}")
        return selected

    preferred = [c for c in pool if "search-all_pmids-studyann-ft" in c.name]
    if preferred:
        selected = max(preferred, key=lambda p: (p / "outputs" / "annotation_results.json").stat().st_mtime)
        print(f"Auto-selected project output dir (preferred pattern): {selected}")
        return selected

    selected = max(pool, key=lambda p: (p / "outputs" / "annotation_results.json").stat().st_mtime)
    print(f"Auto-selected project output dir: {selected}")
    return selected


def resolve_output_dir(project_output_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return project_output_dir / "reports"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
            analyses.append(
                {
                    "id": analysis_id,
                    "name": analysis_name,
                    "points": parse_points(analysis.get("points", [])),
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


def compute_coord_score(
    manual_coords: list[tuple[float, float, float]],
    auto_coords: list[tuple[float, float, float]],
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
    exact_coord_set = (
        len(manual_coords) == len(auto_coords)
        and rounded_coords(manual_coords) == rounded_coords(auto_coords)
    )
    exact_bonus = 0.05 if exact_coord_set else 0.0

    score = max(0.0, min(1.0, (match_quality * coverage_penalty) + exact_bonus))

    if exact_coord_set:
        reasons.append("exact_coord_set")
    if len(manual_coords) != len(auto_coords):
        reasons.append("coord_count_mismatch")
    if score >= 0.75:
        reasons.append("high_coord_match")

    return score, {
        "exact_coord_set": exact_coord_set,
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
    if bool(detail.get("exact_coord_set", False)):
        return "accepted"
    return status_from_score(float(detail.get("combined_score", 0.0)))


def score_pair(manual_analysis: dict[str, Any], auto_analysis: dict[str, Any]) -> dict[str, Any]:
    name_score = compute_name_score(manual_analysis["name"], auto_analysis["name"])
    coord_score, coord_meta, reasons = compute_coord_score(manual_analysis["points"], auto_analysis["points"])
    combined = (COORD_WEIGHT * coord_score) + (NAME_WEIGHT * name_score)
    exact_coord_set = bool(coord_meta.get("exact_coord_set", False))
    low_name_with_exact_coords = exact_coord_set and name_score < LOW_NAME_SCORE_HIGHLIGHT_THRESHOLD

    if coord_score < 0.4 and name_score >= 0.75:
        reasons.append("low_coord_high_name")
    if coord_score == 0.0 and name_score >= 0.6:
        reasons.append("name_only_signal")
    if low_name_with_exact_coords:
        reasons.append("low_name_with_exact_coords")
    if exact_coord_set and combined < ACCEPTED_THRESHOLD:
        reasons.append("accepted_exact_coord_override")
    if combined < UNCERTAIN_THRESHOLD:
        reasons.append("low_total_score")

    return {
        "name_score": round(name_score, 6),
        "coord_score": round(coord_score, 6),
        "combined_score": round(combined, 6),
        "exact_coord_set": exact_coord_set,
        "low_name_with_exact_coords": low_name_with_exact_coords,
        "reason_codes": sorted(set(reasons)),
    }


def match_with_hungarian(
    manual_analyses: list[dict[str, Any]],
    auto_analyses: list[dict[str, Any]],
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
            detail = score_pair(m, a)
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
) -> dict[str, Any]:
    pmid_results: dict[str, dict[str, Any]] = {}

    manual_pmids = set(manual_analyses_by_pmid.keys())
    auto_pmids = set(auto_parsed_by_pmid.keys())
    overlap_pmids = sorted(manual_pmids & auto_pmids, key=lambda x: (len(x), x))
    excluded_manual_only_pmids = sorted(manual_pmids - auto_pmids, key=lambda x: (len(x), x))
    auto_only_pmids = sorted(auto_pmids - manual_pmids, key=lambda x: (len(x), x))

    for pmid in overlap_pmids:
        manual_analyses = manual_analyses_by_pmid.get(pmid, [])
        auto_analyses = auto_parsed_by_pmid.get(pmid, [])

        matched_entries, unassigned_auto_indices = match_with_hungarian(manual_analyses, auto_analyses)
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
                "all_manual_accepted": bool(matched_entries) and int(counts["accepted"]) == len(matched_entries),
                "mean_combined_score": round(mean_combined, 6),
            },
        }

    all_entries = [entry for data in pmid_results.values() for entry in data["manual_analyses"]]
    status_counts = defaultdict(int)
    combined_scores = []
    perfect_pmids = 0
    category_counts = defaultdict(int)
    exact_coord_override_accepted = 0
    low_name_exact_matches = 0
    for entry in all_entries:
        status_counts[entry["match_status"]] += 1
        combined_scores.append(float(entry["combined_score"]))
        if bool(entry.get("exact_coord_set", False)) and float(entry.get("combined_score", 0.0)) < ACCEPTED_THRESHOLD:
            exact_coord_override_accepted += 1
        if bool(entry.get("low_name_with_exact_coords", False)):
            low_name_exact_matches += 1
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
            "exact_coord_accept_override": True,
            "low_name_highlight_threshold": LOW_NAME_SCORE_HIGHLIGHT_THRESHOLD,
        },
        "pmids": pmid_results,
        "missing_manual_pmids": [],
        "excluded_manual_only_pmids": excluded_manual_only_pmids,
        "auto_only_pmids": auto_only_pmids,
        "summary": {
            "manual_pmids": len(overlap_pmids),
            "missing_manual_pmids": 0,
            "manual_pmids_total": len(manual_pmids),
            "auto_pmids_total": len(auto_pmids),
            "overlap_pmids": len(overlap_pmids),
            "excluded_manual_only_pmids": len(excluded_manual_only_pmids),
            "auto_only_pmids": len(auto_only_pmids),
            "manual_analyses_total": len(all_entries),
            "accepted": int(status_counts["accepted"]),
            "uncertain": int(status_counts["uncertain"]),
            "unmatched": int(status_counts["unmatched"]),
            "accepted_exact_coord_override": int(exact_coord_override_accepted),
            "low_name_exact_matches": int(low_name_exact_matches),
            "pmids_all_manual_accepted": int(perfect_pmids),
            "pmids_all_manual_accepted_rate": (float(perfect_pmids) / len(overlap_pmids)) if overlap_pmids else 0.0,
            "all_correct_pmids": int(category_counts["all_correct"]),
            "mixed_pmids": int(category_counts["mixed"]),
            "all_incorrect_pmids": int(category_counts["all_incorrect"]),
            **summary_stats,
        },
    }


def render_matching_summary_html(match_result: dict[str, Any]) -> str:
    summary = match_result.get("summary", {})
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
    <p><strong>Excluded manual-only PMIDs:</strong> {excluded_manual_only_pmids} |
       <strong>Auto-only PMIDs:</strong> {auto_only_pmids}</p>
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
  </nav>
  {"".join(sections)}
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
    pmids = match_result.get("pmids", {})
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
                " title=\"Exact coordinate-set match accepted, but name similarity is low.\""
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
                " title=\"Exact coordinate-set match accepted, but name similarity is low.\""
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
    <p>Coordinate-first matching (70%) + name similarity (30%), one-to-one Hungarian assignment, accepted &gt;= 0.75, uncertain &gt;= 0.55. Metrics include overlap PMIDs only (manual ∩ auto).</p>
    <p><strong>Overlap PMIDs:</strong> {int(summary.get("overlap_pmids", 0))} |
       <strong>Manual analyses:</strong> {int(summary.get("manual_analyses_total", 0))} |
       <strong>Accepted:</strong> {int(summary.get("accepted", 0))} |
       <strong>Uncertain:</strong> {int(summary.get("uncertain", 0))} |
       <strong>Unmatched:</strong> {int(summary.get("unmatched", 0))}</p>
    <p><strong>Study categories:</strong> All correct={int(summary.get("all_correct_pmids", 0))} |
       Mixed={int(summary.get("mixed_pmids", 0))} |
       All incorrect={int(summary.get("all_incorrect_pmids", 0))}</p>
    <p><strong>Accepted by exact-coordinate override:</strong> {int(summary.get("accepted_exact_coord_override", 0))} |
       <strong>Exact-coordinate matches with low name score:</strong> {int(summary.get("low_name_exact_matches", 0))}</p>
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
    <a href="#top">Top</a>
  </nav>
  {"".join(bucket_html)}
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


def main() -> None:
    args = parse_args()
    project_output_dir = infer_project_output_dir(args.project_output_dir)
    output_dir = resolve_output_dir(project_output_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coordinate_parsing_results = project_output_dir / "outputs" / "coordinate_parsing_results.json"
    auto_by_pmid = load_auto_parsed_data(coordinate_parsing_results)
    manual_by_pmid, manual_study_names_by_pmid = load_manual_analyses_overall(args.manual_dir)
    pubget_by_pmid = build_pubget_index(project_output_dir)
    match_result = build_match_results_overall(
        manual_analyses_by_pmid=manual_by_pmid,
        manual_study_names_by_pmid=manual_study_names_by_pmid,
        auto_parsed_by_pmid=auto_by_pmid,
    )
    annotate_match_result_with_pubget(match_result, pubget_by_pmid)
    write_match_artifacts(output_dir, match_result, pubget_by_pmid=pubget_by_pmid)

    summary = match_result["summary"]
    print(
        f"overall: accepted={summary['accepted']} "
        f"uncertain={summary['uncertain']} unmatched={summary['unmatched']} "
        f"overlap_pmids={summary['overlap_pmids']} "
        f"manual_pmids_total={summary['manual_pmids_total']} "
        f"excluded_manual_only_pmids={summary['excluded_manual_only_pmids']} "
        f"pmids_all_manual_accepted={summary['pmids_all_manual_accepted']} "
        f"pmids_with_pubget={summary.get('pmids_with_pubget', 0)}"
    )

    print(f"Wrote matching artifacts to {output_dir}")


if __name__ == "__main__":
    main()

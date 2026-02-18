#!/usr/bin/env python3
"""Run coordinate-first fuzzy matching between manual and auto analyses.

Outputs:
- match_results_<annotation>.json
- match_results_all_annotations.json
- fuzzy_matching_summary.html

Manual analyses are loaded from merged NiMADS artifacts:
- nimads_studyset.json
- nimads_annotation.json
"""

from __future__ import annotations

import argparse
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


ANNOTATION_SOURCE_FILE_MAP = {
    "social_processing_all": "ALL-Merged.json",
    "affiliation_attachment": "Affiliation-Merged.json",
    "perception_others": "Others-Merged.json",
    "perception_self": "Self-Merged.json",
    "social_communication": "SocComm-Merged.json",
}

ACCEPTED_THRESHOLD = 0.75
UNCERTAIN_THRESHOLD = 0.55
NAME_WEIGHT = 0.30
COORD_WEIGHT = 0.70

def clean_text(value: str) -> str:
    return "".join(ch for ch in str(value) if ch >= " " or ch in "\n\t\r")

def source_filename_to_note_key(filename: str) -> str:
    stem = Path(filename).stem
    key = clean_text(stem).lower().strip()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


ANNOTATION_NOTE_KEY_MAP = {
    annotation_name: source_filename_to_note_key(filename)
    for annotation_name, filename in ANNOTATION_SOURCE_FILE_MAP.items()
}


def normalize_text(value: str) -> str:
    text = clean_text(value).lower().strip()
    text = text.replace(">", " > ")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_args() -> argparse.Namespace:
    default_project_output_dir = Path("../autonima-results/projects/social/coordinates/annotation-only")
    default_manual_dir = Path("../neurometabench/data/nimads/social")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-output-dir",
        type=Path,
        default=default_project_output_dir,
        help="Path to project output dir (e.g., .../annotation-only).",
    )
    parser.add_argument(
        "--manual-dir",
        type=Path,
        default=default_manual_dir,
        help="Path to project NiMADS dir or merged dir containing nimads_studyset.json + nimads_annotation.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for match JSON + summary HTML. Defaults to sibling reports/annotation_review_reports.",
    )
    return parser.parse_args()


def resolve_output_dir(project_output_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    return project_output_dir / "reports" / "annotation_review_reports"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def resolve_manual_merged_paths(manual_dir: Path) -> tuple[Path, Path]:
    direct_studyset = manual_dir / "nimads_studyset.json"
    direct_annotation = manual_dir / "nimads_annotation.json"
    if direct_studyset.exists() and direct_annotation.exists():
        return direct_studyset, direct_annotation

    merged_studyset = manual_dir / "merged" / "nimads_studyset.json"
    merged_annotation = manual_dir / "merged" / "nimads_annotation.json"
    if merged_studyset.exists() and merged_annotation.exists():
        return merged_studyset, merged_annotation

    raise FileNotFoundError(
        "Could not find merged manual NiMADS files. Expected either "
        f"{direct_studyset} + {direct_annotation} or "
        f"{merged_studyset} + {merged_annotation}."
    )


def load_manual_analyses_by_annotation(manual_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    studyset_path, annotation_path = resolve_manual_merged_paths(manual_dir)
    studyset_payload = load_json(studyset_path)
    annotation_payload = load_json(annotation_path)

    required_note_keys = set(ANNOTATION_NOTE_KEY_MAP.values())
    declared_note_keys = set((annotation_payload.get("note_keys") or {}).keys())
    missing_note_keys = sorted(required_note_keys - declared_note_keys)
    if missing_note_keys:
        raise ValueError(
            f"Missing expected note_keys in {annotation_path}: {missing_note_keys}. "
            f"Expected at least: {sorted(required_note_keys)}"
        )

    analysis_lookup: dict[str, dict[str, Any]] = {}
    for study in studyset_payload.get("studies", []):
        pmid = str(study.get("id"))
        for analysis in study.get("analyses", []):
            analysis_id = clean_text(analysis.get("id") or "").strip()
            if not analysis_id:
                continue
            analysis_name = clean_text(analysis.get("name") or analysis_id)
            analysis_lookup[analysis_id] = {
                "pmid": pmid,
                "analysis": {
                    "id": analysis_id,
                    "name": analysis_name,
                    "points": parse_points(analysis.get("points", [])),
                },
            }

    result: dict[str, dict[str, list[dict[str, Any]]]] = {
        annotation_name: defaultdict(list) for annotation_name in ANNOTATION_SOURCE_FILE_MAP
    }
    seen_analysis_ids: dict[str, set[str]] = {annotation_name: set() for annotation_name in ANNOTATION_SOURCE_FILE_MAP}

    for note_entry in annotation_payload.get("notes", []):
        analysis_id = clean_text(note_entry.get("analysis") or "").strip()
        if not analysis_id:
            continue
        resolved = analysis_lookup.get(analysis_id)
        if not resolved:
            continue

        note_values = note_entry.get("note") or {}
        if not isinstance(note_values, dict):
            continue

        for annotation_name, note_key in ANNOTATION_NOTE_KEY_MAP.items():
            if not bool(note_values.get(note_key)):
                continue
            if analysis_id in seen_analysis_ids[annotation_name]:
                continue
            pmid = resolved["pmid"]
            result[annotation_name][pmid].append(resolved["analysis"])
            seen_analysis_ids[annotation_name].add(analysis_id)

    finalized: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for annotation_name, by_pmid in result.items():
        finalized[annotation_name] = {}
        for pmid, analyses in by_pmid.items():
            finalized[annotation_name][pmid] = sorted(analyses, key=lambda item: item["id"])
    return finalized


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


def score_pair(manual_analysis: dict[str, Any], auto_analysis: dict[str, Any]) -> dict[str, Any]:
    name_score = compute_name_score(manual_analysis["name"], auto_analysis["name"])
    coord_score, _meta, reasons = compute_coord_score(manual_analysis["points"], auto_analysis["points"])
    combined = (COORD_WEIGHT * coord_score) + (NAME_WEIGHT * name_score)

    if coord_score < 0.4 and name_score >= 0.75:
        reasons.append("low_coord_high_name")
    if coord_score == 0.0 and name_score >= 0.6:
        reasons.append("name_only_signal")
    if combined < UNCERTAIN_THRESHOLD:
        reasons.append("low_total_score")

    return {
        "name_score": round(name_score, 6),
        "coord_score": round(coord_score, 6),
        "combined_score": round(combined, 6),
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
                    "reason_codes": ["no_auto_analyses_for_pmid"],
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
                    "reason_codes": ["unassigned_by_global_matching", "low_total_score"],
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
                "match_status": status_from_score(d["combined_score"]),
                "reason_codes": d["reason_codes"],
            }
        )

    assigned_auto_indices = {e["best_auto_index"] for e in out if e["best_auto_index"] is not None}
    unassigned_auto_indices = [a["index"] for a in auto_analyses if a["index"] not in assigned_auto_indices]
    return out, unassigned_auto_indices


def build_match_results_for_annotation(
    annotation_name: str,
    manual_analyses_by_pmid: dict[str, list[dict[str, Any]]],
    auto_parsed_by_pmid: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pmid_results: dict[str, dict[str, Any]] = {}

    for pmid in sorted(manual_analyses_by_pmid.keys(), key=lambda x: (len(x), x)):
        manual_analyses = manual_analyses_by_pmid.get(pmid, [])
        auto_analyses = auto_parsed_by_pmid.get(pmid, [])

        if pmid not in auto_parsed_by_pmid:
            manual_entries = []
            for m in manual_analyses:
                manual_entries.append(
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
                        "reason_codes": ["manual_pmid_missing_in_auto"],
                    }
                )

            pmid_results[pmid] = {
                "manual_missing_in_auto": True,
                "manual_analyses": manual_entries,
                "unassigned_auto_indices": [],
                "pmid_summary": {
                    "accepted": 0,
                    "uncertain": 0,
                    "unmatched": len(manual_entries),
                    "mean_combined_score": 0.0,
                },
            }
            continue

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
            "unassigned_auto_indices": unassigned_auto_indices,
            "pmid_summary": {
                "accepted": int(counts["accepted"]),
                "uncertain": int(counts["uncertain"]),
                "unmatched": int(counts["unmatched"]),
                "mean_combined_score": round(mean_combined, 6),
            },
        }

    missing_manual_pmids = sorted(
        [pmid for pmid, data in pmid_results.items() if data.get("manual_missing_in_auto")],
        key=lambda x: (len(x), x),
    )

    all_entries = [entry for data in pmid_results.values() for entry in data["manual_analyses"]]
    status_counts = defaultdict(int)
    combined_scores = []
    for entry in all_entries:
        status_counts[entry["match_status"]] += 1
        combined_scores.append(float(entry["combined_score"]))

    combined_arr = np.array(combined_scores, dtype=float) if combined_scores else np.array([], dtype=float)
    summary_stats = {
        "mean_combined_score": float(np.mean(combined_arr)) if combined_arr.size else 0.0,
        "median_combined_score": float(np.median(combined_arr)) if combined_arr.size else 0.0,
        "p25_combined_score": float(np.percentile(combined_arr, 25)) if combined_arr.size else 0.0,
        "p75_combined_score": float(np.percentile(combined_arr, 75)) if combined_arr.size else 0.0,
    }

    return {
        "annotation_name": annotation_name,
        "matching_policy": {
            "assignment": "one_to_one_hungarian",
            "coordinate_weight": COORD_WEIGHT,
            "name_weight": NAME_WEIGHT,
            "accepted_threshold": ACCEPTED_THRESHOLD,
            "uncertain_threshold": UNCERTAIN_THRESHOLD,
            "coordinate_space_handling": "ignore_space_labels_use_raw_xyz",
            "metric_truth_policy": "accepted_only",
        },
        "pmids": pmid_results,
        "missing_manual_pmids": missing_manual_pmids,
        "summary": {
            "manual_pmids": len(pmid_results),
            "missing_manual_pmids": len(missing_manual_pmids),
            "manual_analyses_total": len(all_entries),
            "accepted": int(status_counts["accepted"]),
            "uncertain": int(status_counts["uncertain"]),
            "unmatched": int(status_counts["unmatched"]),
            **summary_stats,
        },
    }


def render_matching_summary_html(match_results_by_annotation: dict[str, Any]) -> str:
    rows = []
    total = defaultdict(float)
    total_missing_pmids: list[str] = []

    for annotation_name in ANNOTATION_SOURCE_FILE_MAP:
        data = match_results_by_annotation.get(annotation_name, {})
        summary = data.get("summary", {})
        missing = data.get("missing_manual_pmids", [])

        manual_total = int(summary.get("manual_analyses_total", 0))
        accepted = int(summary.get("accepted", 0))
        uncertain = int(summary.get("uncertain", 0))
        unmatched = int(summary.get("unmatched", 0))
        acc_rate = (accepted / manual_total) if manual_total else 0.0

        total["manual_total"] += manual_total
        total["accepted"] += accepted
        total["uncertain"] += uncertain
        total["unmatched"] += unmatched

        rows.append(
            "<tr>"
            f"<td>{escape(annotation_name)}</td>"
            f"<td>{int(summary.get('manual_pmids', 0))}</td>"
            f"<td>{int(summary.get('missing_manual_pmids', 0))}</td>"
            f"<td>{manual_total}</td>"
            f"<td>{accepted}</td>"
            f"<td>{uncertain}</td>"
            f"<td>{unmatched}</td>"
            f"<td>{acc_rate:.3f}</td>"
            f"<td>{float(summary.get('mean_combined_score', 0.0)):.3f}</td>"
            f"<td>{float(summary.get('median_combined_score', 0.0)):.3f}</td>"
            "</tr>"
        )

        if missing:
            missing_links = "".join(
                f"<li><a href=\"https://pubmed.ncbi.nlm.nih.gov/{escape(pmid)}/\" target=\"_blank\" rel=\"noopener noreferrer\">PMID {escape(pmid)}</a></li>"
                for pmid in missing
            )
            total_missing_pmids.extend(missing)
            rows.append(
                "<tr><td colspan=\"10\">"
                "<details><summary>Missing PMIDs for this annotation</summary>"
                f"<ul>{missing_links}</ul>"
                "</details></td></tr>"
            )

    total_manual = int(total["manual_total"])
    total_acc_rate = (total["accepted"] / total_manual) if total_manual else 0.0

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
    <h1>Fuzzy Matching Summary</h1>
    <p>Coordinate-first matching (70%) + name similarity (30%), one-to-one Hungarian assignment, accepted >= 0.75, uncertain >= 0.55.</p>
    <p><strong>Total manual analyses:</strong> {total_manual} |
       <strong>Accepted:</strong> {int(total['accepted'])} |
       <strong>Uncertain:</strong> {int(total['uncertain'])} |
       <strong>Unmatched:</strong> {int(total['unmatched'])} |
       <strong>Accepted rate:</strong> {total_acc_rate:.3f}</p>
    <p><strong>Total missing manual PMID entries across annotations:</strong> {len(total_missing_pmids)}</p>
  </header>

  <section>
    <h2>Per-Annotation Performance</h2>
    <table>
      <thead>
        <tr>
          <th>Annotation</th>
          <th>Manual PMIDs</th>
          <th>Missing PMIDs</th>
          <th>Manual Analyses</th>
          <th>Accepted</th>
          <th>Uncertain</th>
          <th>Unmatched</th>
          <th>Accepted Rate</th>
          <th>Mean Score</th>
          <th>Median Score</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </section>
</body>
</html>
"""


def write_match_artifacts(output_dir: Path, match_results_by_annotation: dict[str, Any]) -> None:
    aggregate: dict[str, Any] = {}
    for annotation_name, data in match_results_by_annotation.items():
        per_path = output_dir / f"match_results_{annotation_name}.json"
        per_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        aggregate[annotation_name] = data.get("summary", {})

    aggregate_path = output_dir / "match_results_all_annotations.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    summary_html = render_matching_summary_html(match_results_by_annotation)
    summary_path = output_dir / "fuzzy_matching_summary.html"
    summary_path.write_text(summary_html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.project_output_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coordinate_parsing_results = args.project_output_dir / "outputs" / "coordinate_parsing_results.json"
    auto_by_pmid = load_auto_parsed_data(coordinate_parsing_results)
    manual_by_annotation = load_manual_analyses_by_annotation(args.manual_dir)

    match_results_by_annotation: dict[str, Any] = {}
    for annotation_name, manual_by_pmid in manual_by_annotation.items():
        match_results_by_annotation[annotation_name] = build_match_results_for_annotation(
            annotation_name=annotation_name,
            manual_analyses_by_pmid=manual_by_pmid,
            auto_parsed_by_pmid=auto_by_pmid,
        )

    write_match_artifacts(output_dir, match_results_by_annotation)

    for annotation_name in ANNOTATION_SOURCE_FILE_MAP:
        summary = match_results_by_annotation[annotation_name]["summary"]
        print(
            f"{annotation_name}: accepted={summary['accepted']} "
            f"uncertain={summary['uncertain']} unmatched={summary['unmatched']} "
            f"manual_pmids={summary['manual_pmids']} missing_pmids={summary['missing_manual_pmids']}"
        )

    print(f"Wrote matching artifacts to {output_dir}")


if __name__ == "__main__":
    main()

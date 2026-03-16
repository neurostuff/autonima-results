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
from urllib.parse import quote


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


@dataclass
class Decision:
    include: bool
    reasoning: str
    analysis_id: str
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
    return parser.parse_args()


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

    annotation_names: list[str] = []
    note_keys_by_annotation: dict[str, list[str]] = defaultdict(list)
    for manual_key_raw, auto_annotation_raw in payload.items():
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
                # Preserve overall fuzzy matches for row-level "Matched Outcome" labels and
                # the diagnostics panel when per-annotation truth is sliced from overall fallback.
                "review_match_diagnostics": review_match_diagnostics,
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
) -> dict[str, Any]:
    evaluable_indices = None if evaluable_auto_indices is None else set(evaluable_auto_indices)
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

    analysis_rows: list[dict[str, Any]] = []
    for idx in range(max_idx + 1):
        parsed_info = parsed_analyses[idx] if idx < len(parsed_analyses) else {}
        name = clean_text(parsed_info.get("name") or f"analysis_{idx}")
        decision = decisions_by_idx.get(idx)
        model_include = None if decision is None else decision.include
        matched_for_review = idx in matched_auto_indices
        match_status_for_idx = match_status_by_auto_idx.get(idx, "")
        manual_include = idx in true_indices_eval

        if matched_for_review and match_status_for_idx == "unmatched":
            confusion_label = "*"
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
                "parsed_description": clean_text(parsed_info.get("description") or ""),
                "table_id": clean_text(parsed_info.get("table_id") or "").strip(),
                "model_include": model_include,
                "model_decision_icon": decision_icon,
                "model_decision_class": decision_class,
                "confusion_label": confusion_label,
                "confusion_class": confusion_class,
                "matched_for_review": matched_for_review,
                "reasoning": "" if decision is None else decision.reasoning,
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
                "manual_names": [],
                "unmatched_manual_names": [],
                "match_diagnostics": [],
                "status_counts": {"accepted": 0, "uncertain": 0, "unmatched": 0, "mean_combined_score": 0.0},
                "manual_missing_in_auto": False,
            },
        )

        review_match_diagnostics = truth_entry.get(
            "review_match_diagnostics",
            truth_entry.get("match_diagnostics", []),
        )
        evaluable_auto_indices = extract_evaluable_auto_indices(
            review_match_diagnostics,
            allowed_statuses=allowed_match_statuses,
        )
        if not evaluable_auto_indices:
            continue
        evaluable_pmids.add(pmid)

        pred_indices = {idx for idx, decision in decisions_by_idx.items() if decision.include and idx in evaluable_auto_indices}
        true_indices = set(truth_entry["true_indices"]) & evaluable_auto_indices
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

    # Analysis-level metrics are computed over the set of AUTO analyses that have
    # evaluable manual-to-auto matches (accepted/uncertain with best_auto_index).
    # Positives are annotation-sliced manual accepted matches (true_indices).
    analysis_tp = 0
    analysis_fp = 0
    analysis_fn = 0
    analysis_tn = 0
    matched_auto_universe = 0
    manual_accepted_matched = 0
    predicted_positive_on_matched = 0

    for pmid in doc_overlap_pmids:
        decisions_for_pmid = ann_decisions.get(pmid, {})
        truth_for_pmid = ann_truth.get(pmid, {})
        true_indices = set(truth_for_pmid.get("true_indices", set()))
        review_match_rows = truth_for_pmid.get(
            "review_match_diagnostics",
            truth_for_pmid.get("match_diagnostics", []),
        )
        matched_auto_indices = extract_evaluable_auto_indices(
            review_match_rows,
            allowed_statuses=allowed_match_statuses,
        )
        if not matched_auto_indices:
            continue

        for idx_int in matched_auto_indices:
            matched_auto_universe += 1
            decision = decisions_for_pmid.get(idx_int)
            pred_include = bool(decision.include) if decision is not None else False
            true_include = idx_int in true_indices

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
        float((analysis_tp + analysis_tn) / matched_auto_universe)
        if matched_auto_universe
        else 0.0
    )
    analysis_metrics["manual_accepted_analyses"] = int(manual_accepted_matched)
    analysis_metrics["predicted_analyses"] = int(predicted_positive_on_matched)
    analysis_metrics["analysis_universe"] = int(matched_auto_universe)

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
    meta = (
        f"Pred included (matched analyses only): {len(doc['pred_indices'])} | "
        f"Manual included (accepted matched analyses only): {len(doc['true_indices'])} | "
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
        table_heading = f"{group_index}) {table_label}" if table_label else f"{group_index}) Table"
        table_meta_lines = []
        if not table_label and table_id:
            table_meta_lines.append(
                f"<li><strong>Table ID:</strong> {escape(table_id)}</li>"
            )
        if table_meta.get("table_caption"):
            table_meta_lines.append(
                f"<li><strong>Caption:</strong> {escape(str(table_meta.get('table_caption', '')))}</li>"
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
            and not table_meta.get("table_caption")
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

        document_tp = int(document_metrics.get("tp", metrics.get("tp", 0)))
        document_fp = int(document_metrics.get("fp", metrics.get("fp", 0)))
        document_fn = int(document_metrics.get("fn", metrics.get("fn", 0)))
        document_precision = float(document_metrics.get("precision", metrics.get("precision", 0.0)))
        document_recall = float(document_metrics.get("recall", metrics.get("recall", 0.0)))
        document_f1 = float(document_metrics.get("f1", metrics.get("f1", 0.0)))

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
            "<td>Analysis inclusion</td>"
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
        analysis_rows: list[dict[str, Any]] = []
        for annotation_name in ACTIVE_ANNOTATION_NAMES:
            metrics = metrics_by_annotation.get(annotation_name, {})
            study = metrics.get("study_metrics", {})
            analysis = metrics.get("analysis_metrics", {})
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

        analysis_row_html = []
        for row in analysis_rows:
            analysis_row_html.append(
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
    <h2>{escape(mode_label)} Per-Annotation Matched-Analysis Metrics</h2>
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


def main() -> None:
    args = parse_args()
    project_output_dir = infer_project_output_dir(args.project_output_dir)
    annotation_mapping_path = resolve_project_annotation_mapping_path(
        project_output_dir,
        args.annotation_mapping_path,
    )
    configure_active_annotations(annotation_mapping_path)
    output_dir, match_input_dir = resolve_dirs(project_output_dir, args)

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

    output_dir.mkdir(parents=True, exist_ok=True)
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
        output_path = output_dir / f"{annotation_name}.html"
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

    overall_summary_html = render_overall_summary_html(metrics_by_annotation_by_mode)
    overall_summary_path = output_dir / "overall_submeta_summary.html"
    overall_summary_path.write_text(overall_summary_html, encoding="utf-8")
    print(f"Wrote {overall_summary_path}")


if __name__ == "__main__":
    main()

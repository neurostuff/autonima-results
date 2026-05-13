#!/usr/bin/env python3
"""Generate a manual review report for coordinate parsing + annotation decisions.

This report is benchmark-free: it does not label decisions as correct/incorrect.
It is intended for human inspection of:
1) how coordinates were parsed into analyses, and
2) how those analyses were assigned to annotation groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

REQUIRED_OUTPUT_FILES = ("annotation_results.json", "coordinate_parsing_results.json")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROJECTS_ROOT = REPO_ROOT / "projects"


def clean_text(value: Any) -> str:
    return "".join(ch for ch in str(value) if ch >= " " or ch in "\n\t\r")


def normalize_pmid(value: Any) -> str:
    text = clean_text(value).strip()
    text = re.sub(r"^pmid\s*[:#]?\s*", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def infer_config_path(project_output_dir: Path, explicit_config_path: Path | None) -> Path | None:
    if explicit_config_path is not None:
        config_path = explicit_config_path.expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"--config-path does not exist: {explicit_config_path}")
        return config_path

    try:
        project_name = infer_project_name(project_output_dir)
    except Exception:
        project_name = ""

    run_name = project_output_dir.name
    project_dir = PROJECTS_ROOT / project_name if project_name else project_output_dir.parent
    candidates = [
        (project_dir / f"{run_name}.yaml").resolve(),
        (project_dir / f"{run_name}.yml").resolve(),
        (project_dir / "config.yaml").resolve(),
        (project_dir / "config.yml").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_annotation_names_from_config(config_path: Path | None) -> list[str]:
    if config_path is None or not config_path.exists():
        return []

    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore

    if yaml is not None:
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            annotations = payload.get("annotation", {}).get("annotations", [])
            names: list[str] = []
            if isinstance(annotations, list):
                for row in annotations:
                    if not isinstance(row, dict):
                        continue
                    name = clean_text(row.get("name", "")).strip()
                    if name:
                        names.append(name)
            return dedupe_keep_order(names)
        except Exception:
            pass

    text = config_path.read_text(encoding="utf-8")
    names: list[str] = []
    in_annotation_block = False
    in_annotations_list = False
    annotation_indent = 0
    annotations_indent = 0
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if re.fullmatch(r"annotation\s*:", stripped):
            in_annotation_block = True
            in_annotations_list = False
            annotation_indent = indent
            continue

        if in_annotation_block and indent <= annotation_indent and not stripped.startswith("annotation:"):
            in_annotation_block = False
            in_annotations_list = False

        if in_annotation_block and re.fullmatch(r"annotations\s*:", stripped):
            in_annotations_list = True
            annotations_indent = indent
            continue

        if in_annotations_list and indent <= annotations_indent:
            in_annotations_list = False

        if in_annotations_list:
            match = re.match(r'^\s*-\s*name\s*:\s*["\']?([^"\']+)["\']?\s*$', line)
            if match:
                name = clean_text(match.group(1)).strip()
                if name:
                    names.append(name)

    return dedupe_keep_order(names)


def load_annotation_metadata_fields_from_config(config_path: Path | None) -> list[str]:
    if config_path is None or not config_path.exists():
        return []

    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore

    if yaml is not None:
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            fields = payload.get("annotation", {}).get("metadata_fields", [])
            if isinstance(fields, list):
                out = [clean_text(item).strip() for item in fields if clean_text(item).strip()]
                return dedupe_keep_order(out)
        except Exception:
            pass

    text = config_path.read_text(encoding="utf-8")
    fields: list[str] = []
    in_annotation_block = False
    in_fields_list = False
    annotation_indent = 0
    fields_indent = 0
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if re.fullmatch(r"annotation\s*:", stripped):
            in_annotation_block = True
            in_fields_list = False
            annotation_indent = indent
            continue

        if in_annotation_block and indent <= annotation_indent and not stripped.startswith("annotation:"):
            in_annotation_block = False
            in_fields_list = False

        if in_annotation_block and re.fullmatch(r"metadata_fields\s*:", stripped):
            in_fields_list = True
            fields_indent = indent
            continue

        if in_fields_list and indent <= fields_indent:
            in_fields_list = False

        if in_fields_list:
            match = re.match(r'^\s*-\s*["\']?([^"\']+)["\']?\s*$', line)
            if match:
                field = clean_text(match.group(1)).strip()
                if field:
                    fields.append(field)

    return dedupe_keep_order(fields)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_table_id(value: Any) -> str:
    text = clean_text(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def load_retrieval_context(
    project_output_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    retrieval_dir = project_output_dir / "retrieval" / "pubget_data"
    metadata_path = retrieval_dir / "metadata.csv"
    text_path = retrieval_dir / "text.csv"
    tables_path = retrieval_dir / "tables.csv"
    if not (metadata_path.exists() and text_path.exists() and tables_path.exists()):
        return {}, {}

    metadata_rows = load_csv_rows(metadata_path)
    text_rows = load_csv_rows(text_path)
    table_rows = load_csv_rows(tables_path)

    pmcid_to_pmid: dict[str, str] = {}
    study_meta_by_pmid: dict[str, dict[str, str]] = {}

    for row in metadata_rows:
        pmid = normalize_pmid(row.get("pmid", ""))
        pmcid = clean_text(row.get("pmcid", "")).strip()
        if not pmid or not pmcid:
            continue
        pmcid_to_pmid[pmcid] = pmid
        study_meta_by_pmid[pmid] = {
            "pmid": pmid,
            "pmcid": pmcid,
            "title": clean_text(row.get("title", "")).strip(),
            "last_author_et_al": "",
            "journal": clean_text(row.get("journal", "")).strip(),
            "publication_year": clean_text(row.get("publication_year", "")).strip(),
            "abstract": "",
            "pmc_url": f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/",
            "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }

    for row in text_rows:
        pmcid = clean_text(row.get("pmcid", "")).strip()
        pmid = pmcid_to_pmid.get(pmcid, "")
        if not pmid or pmid not in study_meta_by_pmid:
            continue
        title = clean_text(row.get("title", "")).strip()
        abstract = clean_text(row.get("abstract", "")).strip()
        if title and not study_meta_by_pmid[pmid].get("title"):
            study_meta_by_pmid[pmid]["title"] = title
        if abstract:
            study_meta_by_pmid[pmid]["abstract"] = abstract

    tables_by_pmid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in table_rows:
        pmcid = clean_text(row.get("pmcid", "")).strip()
        pmid = pmcid_to_pmid.get(pmcid, "")
        if not pmid:
            continue
        raw_table_id = clean_text(row.get("table_id", "")).strip()
        tables_by_pmid[pmid].append(
            {
                "table_id": raw_table_id,
                "table_id_norm": normalize_table_id(raw_table_id),
                "table_label": clean_text(row.get("table_label", "")).strip(),
                "table_caption": clean_text(row.get("table_caption", "")).strip(),
                "table_foot": clean_text(row.get("table_foot", "")).strip(),
            }
        )

    return study_meta_by_pmid, dict(tables_by_pmid)


def first_author_et_al_from_authors(authors: Any) -> str:
    if not isinstance(authors, list) or not authors:
        return ""
    first_raw = clean_text(authors[0]).strip()
    if not first_raw:
        return ""
    parts = [part for part in re.split(r"\s+", first_raw) if part]
    last_name = parts[-1] if parts else first_raw
    last_name = re.sub(r"[.,;:]+$", "", last_name).strip()
    if not last_name:
        return ""
    return f"{last_name} et al"


def extract_year_from_text(value: Any) -> str:
    text = clean_text(value).strip()
    if not text:
        return ""
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else ""


def load_search_metadata(project_output_dir: Path) -> dict[str, dict[str, str]]:
    search_results_path = project_output_dir / "outputs" / "search_results.json"
    if not search_results_path.exists():
        return {}
    payload = load_json(search_results_path)
    studies = payload.get("studies", []) if isinstance(payload, dict) else []
    out: dict[str, dict[str, str]] = {}
    for row in studies:
        pmid = normalize_pmid(row.get("pmid", ""))
        title = clean_text(row.get("title", "")).strip()
        last_author_et_al = first_author_et_al_from_authors(row.get("authors"))
        publication_year = clean_text(row.get("publication_year", "")).strip()
        if not publication_year:
            publication_year = extract_year_from_text(row.get("publication_date", ""))
        if pmid:
            out[pmid] = {
                "title": title,
                "last_author_et_al": last_author_et_al,
                "publication_year": publication_year,
            }
    return out


def parse_point_coords(points: list[dict[str, Any]]) -> tuple[list[tuple[float, float, float]], set[str], int]:
    coords: list[tuple[float, float, float]] = []
    spaces: set[str] = set()
    invalid_points = 0
    for point in points or []:
        space = clean_text(point.get("space", "")).strip()
        if space:
            spaces.add(space)
        raw = point.get("coordinates", [])
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            invalid_points += 1
            continue
        try:
            x, y, z = float(raw[0]), float(raw[1]), float(raw[2])
            coords.append((x, y, z))
        except Exception:
            invalid_points += 1
    return coords, spaces, invalid_points


def load_parsed_analyses(
    coordinate_parsing_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    payload = load_json(coordinate_parsing_path)
    studies = payload.get("studies", [])

    analyses: list[dict[str, Any]] = []
    by_pmid: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for study in studies:
        pmid = normalize_pmid(study.get("pmid"))
        study_analyses = study.get("analyses", [])
        for idx, analysis in enumerate(study_analyses):
            analysis_id = f"{pmid}_analysis_{idx}"
            coords, spaces, invalid_points = parse_point_coords(analysis.get("points", []))
            row = {
                "pmid": pmid,
                "analysis_index": idx,
                "analysis_id": analysis_id,
                "analysis_name": clean_text(analysis.get("name") or f"analysis_{idx}"),
                "analysis_description": clean_text(analysis.get("description") or ""),
                "table_id": clean_text(analysis.get("table_id") or "").strip(),
                "coordinates": coords,
                "coordinate_count": len(coords),
                "coordinate_spaces": sorted(spaces),
                "invalid_point_count": int(invalid_points),
            }
            analyses.append(row)
            by_pmid[pmid].append(row)

    analyses.sort(key=lambda r: (len(r["pmid"]), r["pmid"], int(r["analysis_index"])))
    return analyses, dict(by_pmid)


def load_annotation_decisions(
    annotation_results_path: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    payload = load_json(annotation_results_path)
    by_analysis: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    seen_annotations: list[str] = []

    for row in payload:
        analysis_id = clean_text(row.get("analysis_id", "")).strip()
        annotation_name = clean_text(row.get("annotation_name", "")).strip()
        if not analysis_id or not annotation_name:
            continue
        if annotation_name not in seen_annotations:
            seen_annotations.append(annotation_name)
        by_analysis[analysis_id][annotation_name] = {
            "include": bool(row.get("include", False)),
            "confidence": row.get("confidence"),
            "reasoning": clean_text(row.get("reasoning", "")),
            "inclusion_criteria_applied": list(row.get("inclusion_criteria_applied") or []),
            "exclusion_criteria_applied": list(row.get("exclusion_criteria_applied") or []),
            "model_used": clean_text(row.get("model_used", "")),
        }

    return dict(by_analysis), seen_annotations


def coord_text(coords: list[tuple[float, float, float]], max_rows: int) -> str:
    show = coords[:max_rows]
    lines = [f"({x:g}, {y:g}, {z:g})" for x, y, z in show]
    hidden = len(coords) - len(show)
    if hidden > 0:
        lines.append(f"... and {hidden} more")
    return "\n".join(lines) if lines else "(none)"


def render_decision_pill(decision: dict[str, Any] | None) -> str:
    if decision is None:
        return '<span class="pill missing">missing</span>'
    if bool(decision.get("include", False)):
        return '<span class="pill include">true</span>'
    return '<span class="pill exclude">false</span>'


def normalize_criteria_items(section: Any) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if isinstance(section, dict):
        for code, text in section.items():
            code_text = clean_text(code).strip()
            item_text = clean_text(text).strip()
            if not code_text:
                continue
            items.append((code_text, item_text))
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
    annotation_rules = annotation_criteria.get(annotation_name, {}) if isinstance(annotation_criteria, dict) else {}

    def add_items(scope: str, criterion_type: str, items: Any) -> None:
        for raw_code, raw_text in items if isinstance(items, list) else []:
            code = clean_text(raw_code).strip()
            text = clean_text(raw_text).strip()
            if not code:
                continue
            metadata[code] = {
                "scope": scope,
                "criterion_type": criterion_type,
                "text": text,
            }

    if isinstance(global_criteria, dict):
        add_items("global", "inclusion", global_criteria.get("inclusion", []))
        add_items("global", "exclusion", global_criteria.get("exclusion", []))
    if isinstance(annotation_rules, dict):
        add_items("annotation", "inclusion", annotation_rules.get("inclusion", []))
        add_items("annotation", "exclusion", annotation_rules.get("exclusion", []))

    return metadata


def render_code_pills(codes: list[str], criterion_meta: dict[str, dict[str, str]], pill_class: str) -> str:
    if not codes:
        return '<span class="muted">none</span>'
    pills = []
    for code in codes:
        meta = criterion_meta.get(code, {})
        title = clean_text(meta.get("text", "")).strip()
        title_attr = escape(f"{code}: {title}") if title else escape(code)
        pills.append(f'<span class="criteria-pill {escape(pill_class)}" title="{title_attr}">{escape(code)}</span>')
    return "".join(pills)


def render_criteria_status(
    decision: dict[str, Any] | None,
    criterion_meta: dict[str, dict[str, str]],
) -> str:
    if decision is None:
        return '<span class="muted">No decision</span>'

    applied_inclusion = dedupe_keep_order([clean_text(x).strip() for x in decision.get("inclusion_criteria_applied", []) if clean_text(x).strip()])
    applied_exclusion = dedupe_keep_order([clean_text(x).strip() for x in decision.get("exclusion_criteria_applied", []) if clean_text(x).strip()])
    known_inclusion = sorted([code for code, meta in criterion_meta.items() if meta.get("criterion_type") == "inclusion"])
    known_exclusion = sorted([code for code, meta in criterion_meta.items() if meta.get("criterion_type") == "exclusion"])

    inclusion_met = [code for code in known_inclusion if code in applied_inclusion]
    inclusion_not_met = [code for code in known_inclusion if code not in applied_inclusion]
    exclusion_met = [code for code in known_exclusion if code in applied_exclusion]

    unknown_inclusion = [code for code in applied_inclusion if code not in criterion_meta]
    unknown_exclusion = [code for code in applied_exclusion if code not in criterion_meta]

    if known_inclusion:
        if len(inclusion_not_met) == 0:
            inclusion_block = (
                '<div class="criteria-row"><span class="criteria-label">Inclusion</span>'
                '<span class="criteria-pill criteria-good">all inclusions met</span>'
                "</div>"
            )
        else:
            inclusion_block = (
                '<div class="criteria-row"><span class="criteria-label">Inclusion not met</span>'
                + render_code_pills(inclusion_not_met, criterion_meta, "criteria-neutral")
                + "</div>"
            )
    else:
        inclusion_block = (
            '<div class="criteria-row"><span class="criteria-label">Inclusion</span>'
            '<span class="muted">none defined</span>'
            "</div>"
        )

    if known_exclusion:
        if len(exclusion_met) == len(known_exclusion):
            exclusion_block = (
                '<div class="criteria-row"><span class="criteria-label">Exclusion</span>'
                '<span class="criteria-pill criteria-bad">all exclusions met</span>'
                "</div>"
            )
        else:
            exclusion_block = (
                '<div class="criteria-row"><span class="criteria-label">Exclusion met</span>'
                + render_code_pills(exclusion_met, criterion_meta, "criteria-bad")
                + "</div>"
            )
    else:
        exclusion_block = (
            '<div class="criteria-row"><span class="criteria-label">Exclusion</span>'
            '<span class="muted">none defined</span>'
            "</div>"
        )

    return (
        '<div class="criteria-status-wrap">'
        + inclusion_block
        + exclusion_block
        + (
            '<div class="criteria-row"><span class="criteria-label">Unknown codes</span>'
            + "".join(f'<span class="criteria-pill criteria-unknown">{escape(code)}</span>' for code in (unknown_inclusion + unknown_exclusion))
            + "</div>"
            if (unknown_inclusion or unknown_exclusion)
            else ""
        )
        + "</div>"
    )


def render_analysis_annotation_rows(
    *,
    analysis: dict[str, Any],
    annotation_names: list[str],
    decisions_by_analysis: dict[str, dict[str, dict[str, Any]]],
    criteria: dict[str, Any],
) -> str:
    decision_map = decisions_by_analysis.get(analysis["analysis_id"], {})
    rows: list[str] = []
    for annotation_name in annotation_names:
        decision = decision_map.get(annotation_name)
        criterion_meta = build_annotation_criteria_metadata(criteria, annotation_name)
        reasoning = clean_text(decision.get("reasoning", "") if decision else "").strip()
        if not reasoning:
            reasoning = "(no reasoning provided)"

        rows.append(
            "<tr>"
            f"<td><strong>{escape(annotation_name)}</strong></td>"
            f"<td>{render_decision_pill(decision)}</td>"
            f"<td>{render_criteria_status(decision, criterion_meta)}</td>"
            f"<td><pre>{escape(reasoning)}</pre></td>"
            "</tr>"
        )

    return (
        '<div class="table-wrap">'
        '<table class="annotation-pairing-table">'
        "<thead><tr><th>Annotation</th><th>Decision</th><th>Criteria status</th><th>Reasoning</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def render_coordinates_block(
    *,
    analysis: dict[str, Any],
    max_coordinate_preview: int,
) -> str:
    coord_preview = coord_text(analysis.get("coordinates", []), max_rows=max_coordinate_preview)
    spaces = ", ".join(analysis.get("coordinate_spaces", [])) if analysis.get("coordinate_spaces") else "-"
    return (
        '<details class="inner-accordion">'
        f"<summary>Show coordinates ({int(analysis.get('coordinate_count', 0))}; spaces={escape(spaces)})</summary>"
        f"<pre>{escape(coord_preview)}</pre>"
        "</details>"
    )


def render_html(
    *,
    project_output_dir: Path,
    config_path: Path | None,
    annotation_names: list[str],
    analyses: list[dict[str, Any]],
    analyses_by_pmid: dict[str, list[dict[str, Any]]],
    decisions_by_analysis: dict[str, dict[str, dict[str, Any]]],
    criteria: dict[str, Any],
    metadata_fields: list[str],
    study_meta_by_pmid: dict[str, dict[str, str]],
    tables_by_pmid: dict[str, list[dict[str, str]]],
    max_coordinate_preview: int,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    study_count = len({row["pmid"] for row in analyses})
    analysis_count = len(analyses)
    with_coordinates = sum(1 for row in analyses if int(row["coordinate_count"]) > 0)
    with_invalid_points = sum(1 for row in analyses if int(row["invalid_point_count"]) > 0)

    annotation_stats: list[dict[str, Any]] = []
    grouped_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for annotation_name in annotation_names:
        grouped_rows[annotation_name] = {"included": [], "excluded": [], "missing": []}
        include_count = 0
        exclude_count = 0
        missing_count = 0
        pmids_with_any_true: set[str] = set()
        for row in analyses:
            decision = decisions_by_analysis.get(row["analysis_id"], {}).get(annotation_name)
            if decision is None:
                missing_count += 1
                grouped_rows[annotation_name]["missing"].append(row)
                continue
            if bool(decision.get("include", False)):
                include_count += 1
                grouped_rows[annotation_name]["included"].append(row)
                pmids_with_any_true.add(str(row["pmid"]))
            else:
                exclude_count += 1
                grouped_rows[annotation_name]["excluded"].append(row)
        annotation_stats.append(
            {
                "annotation_name": annotation_name,
                "included": include_count,
                "pmids_with_true": len(pmids_with_any_true),
                "excluded": exclude_count,
                "missing": missing_count,
            }
        )

    annotation_summary_rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(row['annotation_name']))}</td>"
        f"<td>{int(row['included'])}</td>"
        f"<td>{int(row['pmids_with_true'])}</td>"
        f"<td>{int(row['excluded'])}</td>"
        f"<td>{int(row['missing'])}</td>"
        "</tr>"
        for row in annotation_stats
    )

    def render_criteria_items(items: list[tuple[str, str]]) -> str:
        if not items:
            return '<p class="muted">None specified.</p>'
        lines: list[str] = []
        for code, text in items:
            code_text = clean_text(code).strip()
            text_value = clean_text(text).strip()
            if code_text:
                lines.append(f"<li><strong>{escape(code_text)}:</strong> {escape(text_value)}</li>")
            else:
                lines.append(f"<li>{escape(text_value)}</li>")
        return "<ul class=\"criteria-list\">" + "".join(lines) + "</ul>"

    global_criteria = criteria.get("global", {}) if isinstance(criteria, dict) else {}
    per_annotation_criteria = criteria.get("annotations", {}) if isinstance(criteria, dict) else {}
    global_inclusion = list(global_criteria.get("inclusion", [])) if isinstance(global_criteria, dict) else []
    global_exclusion = list(global_criteria.get("exclusion", [])) if isinstance(global_criteria, dict) else []
    annotation_criteria_blocks: list[str] = []
    for annotation_name in annotation_names:
        block = per_annotation_criteria.get(annotation_name, {}) if isinstance(per_annotation_criteria, dict) else {}
        annotation_inclusion = list(block.get("inclusion", [])) if isinstance(block, dict) else []
        annotation_exclusion = list(block.get("exclusion", [])) if isinstance(block, dict) else []
        annotation_criteria_blocks.append(
            '<div class="criteria-block">'
            f"<h4>{escape(annotation_name)}</h4>"
            '<div class="criteria-grid">'
            f"<div><h5>Inclusion</h5>{render_criteria_items(annotation_inclusion)}</div>"
            f"<div><h5>Exclusion</h5>{render_criteria_items(annotation_exclusion)}</div>"
            "</div>"
            "</div>"
        )

    criteria_panel_html = (
        '<div class="card criteria-panel-top">'
        "<h2>Project Inclusion / Exclusion Reasons</h2>"
        '<div class="criteria-block">'
        "<h3>Global Criteria</h3>"
        '<div class="criteria-grid">'
        f"<div><h5>Inclusion</h5>{render_criteria_items(global_inclusion)}</div>"
        f"<div><h5>Exclusion</h5>{render_criteria_items(global_exclusion)}</div>"
        "</div>"
        "</div>"
        + (
            '<div class="criteria-block"><h3>Annotation-Specific Criteria</h3>'
            + "".join(annotation_criteria_blocks)
            + "</div>"
            if annotation_criteria_blocks
            else '<p class="muted">No annotation-specific criteria found.</p>'
        )
        + "</div>"
    )

    study_cards_no_true: list[str] = []
    study_cards_with_true: list[str] = []
    for pmid in sorted(analyses_by_pmid.keys(), key=lambda x: (len(x), x)):
        study_rows = list(analyses_by_pmid.get(pmid, []))
        study_has_any_true = any(
            bool(decisions_by_analysis.get(str(row.get("analysis_id", "")), {}).get(annotation_name, {}).get("include", False))
            for row in study_rows
            for annotation_name in annotation_names
        )
        study_meta = study_meta_by_pmid.get(
            pmid,
            {
                "pmid": pmid,
                "title": "",
                "last_author_et_al": "",
                "journal": "",
                "publication_year": "",
                "abstract": "",
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "pmc_url": "",
            },
        )
        table_catalog = list(tables_by_pmid.get(pmid, []))
        table_lookup: dict[str, dict[str, str]] = {}
        ordered_table_norm_ids: list[str] = []
        for table in table_catalog:
            norm_id = normalize_table_id(table.get("table_id", ""))
            if not norm_id:
                continue
            if norm_id not in table_lookup:
                table_lookup[norm_id] = table
            if norm_id not in ordered_table_norm_ids:
                ordered_table_norm_ids.append(norm_id)

        analyses_by_table_norm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in study_rows:
            table_norm = normalize_table_id(row.get("table_id", ""))
            analyses_by_table_norm[table_norm].append(row)

        ordered_group_keys: list[str] = []
        for norm_id in ordered_table_norm_ids:
            if norm_id in analyses_by_table_norm:
                ordered_group_keys.append(norm_id)
        for norm_id in sorted(k for k in analyses_by_table_norm.keys() if k not in ordered_group_keys and k):
            ordered_group_keys.append(norm_id)
        if "" in analyses_by_table_norm:
            ordered_group_keys.append("")

        table_groups_html: list[str] = []
        for group_index, table_norm in enumerate(ordered_group_keys, start=1):
            rows_for_table = sorted(
                analyses_by_table_norm.get(table_norm, []),
                key=lambda r: int(r.get("analysis_index", 0)),
            )
            table_meta = table_lookup.get(table_norm, None)
            raw_table_id = clean_text(rows_for_table[0].get("table_id", "")).strip() if rows_for_table else ""
            table_label = clean_text((table_meta or {}).get("table_label", "")).strip()
            table_caption = clean_text((table_meta or {}).get("table_caption", "")).strip()
            header_label = table_label or (f"Table ID: {raw_table_id}" if raw_table_id else "Unspecified table")

            analysis_blocks: list[str] = []
            for analysis in rows_for_table:
                coordinates_html = render_coordinates_block(
                    analysis=analysis,
                    max_coordinate_preview=max_coordinate_preview,
                )
                annotation_rows_html = render_analysis_annotation_rows(
                    analysis=analysis,
                    annotation_names=annotation_names,
                    decisions_by_analysis=decisions_by_analysis,
                    criteria=criteria,
                )
                description_text = clean_text(analysis.get("analysis_description", "")).strip()
                description_html = f'<p class="muted">{escape(description_text)}</p>' if description_text else ""
                analysis_blocks.append(
                    '<div class="analysis-card">'
                    f"<h5><code>{escape(str(analysis.get('analysis_id', '')))}</code> | {escape(str(analysis.get('analysis_name', '')))}</h5>"
                    f"{description_html}"
                    f"{coordinates_html}"
                    '<details class="inner-accordion" open><summary>Annotation rows (decision + criteria + reasoning)</summary>'
                    f"{annotation_rows_html}</details>"
                    "</div>"
                )

            table_groups_html.append(
                '<div class="table-group">'
                f"<h4>{group_index}) {escape(header_label)}</h4>"
                + (f"<p><strong>Caption:</strong> {escape(table_caption)}</p>" if table_caption else "")
                + "".join(analysis_blocks)
                + "</div>"
            )

        title_text = clean_text(study_meta.get("title", "")).strip()
        last_author_et_al = clean_text(study_meta.get("last_author_et_al", "")).strip()
        year_text = clean_text(study_meta.get("publication_year", "")).strip()
        author_year_label = (
            f"{last_author_et_al} {year_text}".strip()
            if last_author_et_al
            else ""
        )
        header_parts = []
        if author_year_label:
            header_parts.append(escape(author_year_label))
        if title_text:
            header_parts.append(escape(title_text))
        header_suffix = f" | {' | '.join(header_parts)}" if header_parts else ""
        journal_text = clean_text(study_meta.get("journal", "")).strip()
        journal_meta = " | ".join(x for x in [journal_text, year_text] if x)
        pmc_link = ""
        if study_meta.get("pmc_url"):
            pmc_link = (
                f' | <a href="{escape(study_meta["pmc_url"])}" target="_blank" rel="noopener noreferrer">PMC</a>'
            )
        byline_parts = []
        if author_year_label:
            byline_parts.append(author_year_label)
        if title_text:
            byline_parts.append(title_text)
        byline_text = " | ".join(byline_parts)
        abstract_text = clean_text(study_meta.get("abstract", "")).strip()
        abstract_html = (
            '<details class="inner-accordion"><summary>Study abstract</summary>'
            f"<pre>{escape(abstract_text)}</pre></details>"
            if abstract_text
            else ""
        )
        study_card_html = (
            '<details class="doc-card">'
            f"<summary><strong>PMID {escape(pmid)}</strong>{header_suffix} | analyses={len(study_rows)}</summary>"
            f'<p><a href="https://pubmed.ncbi.nlm.nih.gov/{escape(pmid)}/" target="_blank" rel="noopener noreferrer">PubMed</a>{pmc_link}'
            + (f' <span class="muted">| {escape(journal_meta)}</span>' if journal_meta else "")
            + "</p>"
            + abstract_html
            + "".join(table_groups_html)
            + "</details>"
        )
        if study_has_any_true:
            study_cards_with_true.append(study_card_html)
        else:
            study_cards_no_true.append(study_card_html)

    config_text = str(config_path) if config_path is not None else "(not found)"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Parsing + Annotation Review Report</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #d1d5db;
      --include-bg: #e7f7ea;
      --include-text: #116329;
      --exclude-bg: #fde8e8;
      --exclude-text: #8a1f1f;
      --missing-bg: #eceef2;
      --missing-text: #4b5563;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 20px;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 14px;
    }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #f9fafb;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .table-wrap {{
      overflow: auto;
      max-height: 72vh;
    }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 600;
    }}
    .pill.include {{ background: var(--include-bg); color: var(--include-text); }}
    .pill.exclude {{ background: var(--exclude-bg); color: var(--exclude-text); }}
    .pill.missing {{ background: var(--missing-bg); color: var(--missing-text); }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f9fafb;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      margin: 6px 0 0;
    }}
    .doc-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 12px;
      margin: 10px 0;
    }}
    .table-group {{
      margin: 0.8rem 0 1rem;
      padding: 0.6rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fcfcfd;
    }}
    .analysis-card {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      margin: 8px 0;
      background: #fff;
    }}
    .analysis-card h5 {{ margin: 0 0 6px; }}
    .inner-accordion {{
      margin-top: 0.5rem;
      border-top: 1px dashed var(--line);
      padding-top: 0.45rem;
    }}
    .metadata-list {{
      margin: 0.3rem 0 0.5rem 0;
      padding-left: 1.1rem;
    }}
    .metadata-list li {{
      margin: 0.22rem 0;
    }}
    .annotation-pairing-table th {{
      position: static;
    }}
    .criteria-status-wrap {{
      display: grid;
      gap: 0.22rem;
    }}
    .criteria-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.25rem;
      align-items: center;
    }}
    .criteria-label {{
      min-width: 7.9rem;
      font-size: 0.77rem;
      color: #334155;
      font-weight: 600;
    }}
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
    .criteria-pill.criteria-neutral {{ background: #eef2f7; color: #334155; border-color: #cbd5e1; }}
    .criteria-pill.criteria-unknown {{ background: #f4f3e9; color: #5b4b07; border-color: #e3d9aa; }}
    .criteria-panel-top h3 {{ margin: 0.2rem 0 0.6rem; }}
    .criteria-panel-top h4 {{ margin: 0.2rem 0 0.4rem; }}
    .criteria-panel-top h5 {{ margin: 0.2rem 0 0.4rem; font-size: 0.95rem; }}
    .criteria-block {{
      background: #fbfcfe;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.7rem;
      margin-bottom: 0.7rem;
    }}
    .criteria-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.8rem;
    }}
    .criteria-list {{
      margin: 0;
      padding-left: 1.1rem;
    }}
    .criteria-list li {{
      margin: 0.2rem 0;
    }}
    details > summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    @media (max-width: 900px) {{
      .criteria-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Parsing + Annotation Review Report</h1>
    <p class="muted">Generated at {escape(generated_at)}</p>
    <p><strong>Project output dir:</strong> <code>{escape(str(project_output_dir.resolve()))}</code></p>
    <p><strong>Config path:</strong> <code>{escape(config_text)}</code></p>
    <p><strong>Summary:</strong> {study_count} studies, {analysis_count} analyses, {with_coordinates} analyses with parsed coordinates, {with_invalid_points} analyses with at least one invalid coordinate row.</p>
  </div>

  {criteria_panel_html}

  <div class="card">
    <h2>Annotation Summary</h2>
    <table>
      <thead>
        <tr><th>Annotation</th><th>Included Analyses (true)</th><th>Unique PMIDs with any true</th><th>Excluded Analyses (false)</th><th>Missing</th></tr>
      </thead>
      <tbody>
        {annotation_summary_rows}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Study-by-Study Review</h2>
    <p class="muted">Grouped by study and source table. Within each analysis, rows are annotation pairings with decision, criteria status (met/not met), and reasoning.</p>
    <h3>Studies With No True Annotations ({len(study_cards_no_true)})</h3>
    {"".join(study_cards_no_true) if study_cards_no_true else '<p class="muted">None.</p>'}
    <h3>All Other Studies ({len(study_cards_with_true)})</h3>
    {"".join(study_cards_with_true) if study_cards_with_true else '<p class="muted">None.</p>'}
  </div>
</body>
</html>
"""


def write_csv(
    *,
    output_csv_path: Path,
    analyses: list[dict[str, Any]],
    annotation_names: list[str],
    decisions_by_analysis: dict[str, dict[str, dict[str, Any]]],
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "pmid",
        "analysis_index",
        "analysis_id",
        "analysis_name",
        "analysis_description",
        "table_id",
        "coordinate_count",
        "coordinate_spaces",
        "coordinates_json",
        "invalid_point_count",
    ]
    for annotation_name in annotation_names:
        header.extend(
            [
                f"{annotation_name}__include",
                f"{annotation_name}__confidence",
                f"{annotation_name}__reasoning",
                f"{annotation_name}__inclusion_criteria_applied",
                f"{annotation_name}__exclusion_criteria_applied",
            ]
        )

    with output_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in analyses:
            out = {
                "pmid": row["pmid"],
                "analysis_index": row["analysis_index"],
                "analysis_id": row["analysis_id"],
                "analysis_name": row["analysis_name"],
                "analysis_description": row["analysis_description"],
                "table_id": row["table_id"],
                "coordinate_count": row["coordinate_count"],
                "coordinate_spaces": "|".join(row["coordinate_spaces"]),
                "coordinates_json": json.dumps(row["coordinates"], ensure_ascii=False),
                "invalid_point_count": row["invalid_point_count"],
            }
            decisions_for_analysis = decisions_by_analysis.get(row["analysis_id"], {})
            for annotation_name in annotation_names:
                decision = decisions_for_analysis.get(annotation_name)
                if decision is None:
                    out[f"{annotation_name}__include"] = ""
                    out[f"{annotation_name}__confidence"] = ""
                    out[f"{annotation_name}__reasoning"] = ""
                    out[f"{annotation_name}__inclusion_criteria_applied"] = ""
                    out[f"{annotation_name}__exclusion_criteria_applied"] = ""
                    continue
                out[f"{annotation_name}__include"] = bool(decision.get("include", False))
                out[f"{annotation_name}__confidence"] = decision.get("confidence")
                out[f"{annotation_name}__reasoning"] = decision.get("reasoning", "")
                out[f"{annotation_name}__inclusion_criteria_applied"] = json.dumps(
                    decision.get("inclusion_criteria_applied", []),
                    ensure_ascii=False,
                )
                out[f"{annotation_name}__exclusion_criteria_applied"] = json.dumps(
                    decision.get("exclusion_criteria_applied", []),
                    ensure_ascii=False,
                )
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-output-dir",
        type=Path,
        default=None,
        help=(
            "Path to project run dir containing outputs/annotation_results.json and "
            "outputs/coordinate_parsing_results.json. If omitted, auto-selects the most "
            "recently updated run under projects/."
        ),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help=(
            "Optional project YAML path used to determine annotation names. "
            "If omitted, attempts to infer from project/run name."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output HTML path. Defaults to "
            "project-output-dir/reports/parsing_annotation_review_report.html"
        ),
    )
    parser.add_argument(
        "--include-extra-annotations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled, include annotation names present in annotation_results.json but not "
            "listed in config. Default: false."
        ),
    )
    parser.add_argument(
        "--max-coordinate-preview",
        type=int,
        default=30,
        help="Maximum number of coordinates shown in HTML preview per analysis. Default: 30.",
    )
    parser.add_argument(
        "--criteria-mapping-path",
        type=Path,
        default=None,
        help=(
            "Optional path to criteria mapping JSON. Defaults to "
            "project-output-dir/outputs/criteria_mapping.json when present."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_coordinate_preview < 1:
        raise ValueError("--max-coordinate-preview must be >= 1")

    project_output_dir = infer_project_output_dir(args.project_output_dir)
    outputs_dir = project_output_dir / "outputs"
    coordinate_parsing_path = outputs_dir / "coordinate_parsing_results.json"
    annotation_results_path = outputs_dir / "annotation_results.json"
    if not coordinate_parsing_path.exists():
        raise FileNotFoundError(f"Missing coordinate parsing file: {coordinate_parsing_path}")
    if not annotation_results_path.exists():
        raise FileNotFoundError(f"Missing annotation results file: {annotation_results_path}")

    config_path = infer_config_path(project_output_dir, args.config_path)
    config_annotation_names = load_annotation_names_from_config(config_path)
    metadata_fields = load_annotation_metadata_fields_from_config(config_path)

    analyses, by_pmid = load_parsed_analyses(coordinate_parsing_path)
    decisions_by_analysis, seen_annotations = load_annotation_decisions(annotation_results_path)
    study_meta_by_pmid, tables_by_pmid = load_retrieval_context(project_output_dir)
    search_meta_by_pmid = load_search_metadata(project_output_dir)
    for pmid, search_meta in search_meta_by_pmid.items():
        title = clean_text(search_meta.get("title", "")).strip()
        last_author_et_al = clean_text(search_meta.get("last_author_et_al", "")).strip()
        publication_year = clean_text(search_meta.get("publication_year", "")).strip()
        if pmid not in study_meta_by_pmid:
            study_meta_by_pmid[pmid] = {
                "pmid": pmid,
                "pmcid": "",
                "title": title,
                "last_author_et_al": last_author_et_al,
                "journal": "",
                "publication_year": publication_year,
                "abstract": "",
                "pmc_url": "",
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
            continue
        if not clean_text(study_meta_by_pmid[pmid].get("title", "")).strip():
            study_meta_by_pmid[pmid]["title"] = title
        if not clean_text(study_meta_by_pmid[pmid].get("last_author_et_al", "")).strip():
            study_meta_by_pmid[pmid]["last_author_et_al"] = last_author_et_al
        if not clean_text(study_meta_by_pmid[pmid].get("publication_year", "")).strip():
            study_meta_by_pmid[pmid]["publication_year"] = publication_year
    criteria_mapping_path = (
        args.criteria_mapping_path.expanduser().resolve()
        if args.criteria_mapping_path is not None
        else (project_output_dir / "outputs" / "criteria_mapping.json")
    )
    criteria = load_annotation_criteria(criteria_mapping_path if criteria_mapping_path.exists() else None)
    if config_annotation_names:
        annotation_names = list(config_annotation_names)
        if args.include_extra_annotations:
            extras = [name for name in seen_annotations if name not in annotation_names]
            annotation_names.extend(extras)
    else:
        annotation_names = list(seen_annotations)

    annotation_names = dedupe_keep_order(annotation_names)
    if not annotation_names:
        raise ValueError(
            "Could not determine annotation names from config or annotation_results.json."
        )

    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else (project_output_dir / "reports" / "parsing_annotation_review_report.html").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = render_html(
        project_output_dir=project_output_dir,
        config_path=config_path,
        annotation_names=annotation_names,
        analyses=analyses,
        analyses_by_pmid=by_pmid,
        decisions_by_analysis=decisions_by_analysis,
        criteria=criteria,
        metadata_fields=metadata_fields,
        study_meta_by_pmid=study_meta_by_pmid,
        tables_by_pmid=tables_by_pmid,
        max_coordinate_preview=int(args.max_coordinate_preview),
    )
    output_path.write_text(html, encoding="utf-8")

    csv_output_path = output_path.with_suffix(".csv")
    write_csv(
        output_csv_path=csv_output_path,
        analyses=analyses,
        annotation_names=annotation_names,
        decisions_by_analysis=decisions_by_analysis,
    )

    print(f"Wrote HTML report: {output_path}")
    print(f"Wrote CSV export: {csv_output_path}")
    print(
        "Summary: "
        f"studies={len({row['pmid'] for row in analyses})} "
        f"analyses={len(analyses)} "
        f"annotations={len(annotation_names)}"
    )


if __name__ == "__main__":
    main()

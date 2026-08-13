#!/usr/bin/env python3
"""Sample gold-anchored and auto-only analysis units across projects and emit a
paper-centric, fillable HTML review report for manual annotation of parser
coordinate-separation errors (over-split, merge, missed unit, spurious unit,
misattribution, partial coordinate error) and the common source-table conditions
that caused them.

The report is organized by PAPER, not by isolated analysis unit: for each paper
with at least one sampled unit, it renders the paper's original source tables
(as close to their original HTML as available), with rows color-highlighted to
show exactly how the parser split each table into predicted analyses, a legend
of those predicted analyses, and a side-by-side panel of gold-standard analyses
that had no reliable auto match. PubMed/PMC links are included when available.
Keyboard shortcuts support rapid review.

Scope: coordinate-separation correctness only -- did the parser carve tables into
the right analysis units with the right coordinates attached. Not expressivity,
not facet extraction.

Does NOT re-run any matching. Purely consumes existing outputs from
compare_analyses_to_benchmark.py (reports/match_results_overall.json) and the
pipeline's own outputs (outputs/coordinate_parsing_results.json,
outputs/fulltext_retrieval_results.json, retrieval/pubget_data/*.csv).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))

import run_cross_project_analysis_reports as rcpar  # noqa: E402
import compare_analyses_to_benchmark as cab  # noqa: E402
import parser_failure_taxonomy as taxonomy  # noqa: E402

_REQUIRED_CAB_ATTRS = (
    "load_retrieval_context",
    "extract_coord_table_html",
    "find_article_xml",
    "resolve_retrieval_pubget_dir",
    "clean_text",
    "HUMAN_REVIEW_EXTRACTION_REASONS",
)
for _name in _REQUIRED_CAB_ATTRS:
    if not hasattr(cab, _name):
        raise ImportError(
            f"compare_analyses_to_benchmark.py no longer exposes {_name!r}; "
            "update generate_parser_failure_annotation_report.py"
        )

_LEGACY_IDS = {reason_id for reason_id, _label in cab.HUMAN_REVIEW_EXTRACTION_REASONS}
if _LEGACY_IDS != set(taxonomy.LEGACY_REASON_TO_FAILURE_MODES.keys()):
    raise ImportError(
        "HUMAN_REVIEW_EXTRACTION_REASONS in compare_analyses_to_benchmark.py no longer "
        "matches parser_failure_taxonomy.LEGACY_REASON_TO_FAILURE_MODES -- update the crosswalk"
    )

MATCH_RESULTS_FILE = "reports/match_results_overall.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "parser_error_annotation"
DEFAULT_SEED = 20260721
DEFAULT_ACCEPTED_SAMPLE_RATE = 0.0
DEFAULT_SPURIOUS_CANDIDATE_SAMPLE_RATE = 0.0

# Keep the review UI concrete and compact. These options use the existing export
# fields so previously written analysis code can continue to consume the report.
PARSING_REASON_OPTIONS = (
    ("failure-mode", "over_split", "One analysis was split into several", "gold_unit", "1"),
    ("failure-mode", "under_split_merge", "Multiple analyses were merged", "gold_unit", "2"),
    ("failure-mode", "misattribution", "Coordinates were assigned to the wrong analysis", "both", "3"),
    (
        "failure-mode",
        "partial_coord_error",
        "Coordinates are missing, extra, or malformed",
        "gold_unit",
        "4",
    ),
    (
        "parsing-reason",
        "section_header_parsed_as_analysis",
        "A section or table header was mistaken for an analysis",
        "auto_only_unit",
        "",
    ),
    (
        "parsing-reason",
        "contrast_label_missed_or_truncated",
        "The analysis or contrast label was missed or truncated",
        "both",
        "",
    ),
    (
        "parsing-reason",
        "table_structure_misparsed",
        "Complex table structure was misread (headers, merged cells, or layout)",
        "both",
        "",
    ),
    (
        "trigger",
        "spans_multiple_tables",
        "The analysis coordinates span multiple tables",
        "both",
        "",
    ),
    (
        "trigger",
        "footnote_carried_context",
        "Key context appears only in a caption or footnote",
        "both",
        "",
    ),
    ("parsing-reason", "other_extraction_issue", "Other parsing issue", "both", ""),
)

# Colors cycle by the analysis's GLOBAL (per-paper) index, so a badge/color pair
# stays stable across every table in the same paper.
ANALYSIS_COLOR_PALETTE = [
    "#2f6fed", "#1f9e6d", "#c9722a", "#a640c9", "#d63d5c",
    "#2aa7a7", "#7a8b1f", "#c94a9a", "#5c6bc0", "#8d6e63",
]

# Unicode minus/dash variants seen in extracted JATS tables; normalized to '-'.
_MINUS_CHARS = "−–‐‑"
# Thin space glued between a minus sign and its digits (e.g. "− 12"); dropped.
_THIN_SPACE = " "

AUTO_MATCH_TOL = 0.6  # tight: auto coordinates come directly from this exact table
GOLD_HINT_TOL = 1.5  # loose: gold coordinates may differ in space/rounding -- hint only


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_existing_review(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Existing review must contain a JSON object: {path}")
    if not isinstance(payload.get("entries", []), list):
        raise ValueError(f"Existing review 'entries' must be a list: {path}")
    return payload


def build_seeded_browser_store(
    review_payload: dict[str, Any] | None,
    sampled_units: list[Unit],
) -> tuple[dict[str, Any], dict[str, int]]:
    sampled_by_id = {unit.unit_id: unit for unit in sampled_units}
    sampled_paper_keys = {f"{unit.project}:{unit.pmid}" for unit in sampled_units}
    source_entries = (review_payload or {}).get("entries", []) or []
    seeded_entries: dict[str, dict[str, Any]] = {}

    for raw_entry in source_entries:
        if not isinstance(raw_entry, dict):
            continue
        unit_id = str(raw_entry.get("unit_id") or "")
        unit = sampled_by_id.get(unit_id)
        if unit is None:
            continue
        entry = dict(raw_entry)
        entry.update(
            {
                "unit_id": unit.unit_id,
                "unit_kind": unit.unit_kind,
                "project": unit.project,
                "run_dir": unit.run_dir,
                "pmid": unit.pmid,
                "match_status": unit.match_status or "",
            }
        )
        seeded_entries[unit_id] = entry

    source_paper_notes = (review_payload or {}).get("paper_notes", []) or []
    if isinstance(source_paper_notes, dict):
        paper_note_items = source_paper_notes.items()
    else:
        paper_note_items = (
            (str(note.get("paper_key") or ""), note)
            for note in source_paper_notes
            if isinstance(note, dict)
        )
    seeded_paper_notes = {
        paper_key: {
            key: value
            for key, value in dict(note).items()
            if key != "paper_key"
        }
        for paper_key, note in paper_note_items
        if paper_key in sampled_paper_keys
    }

    store = {
        "entries": seeded_entries,
        "paper_notes": seeded_paper_notes,
        "reviewer": str((review_payload or {}).get("reviewer") or ""),
    }
    stats = {
        "source_entries": len(source_entries),
        "seeded_entries": len(seeded_entries),
        "dropped_entries": len(source_entries) - len(seeded_entries),
        "source_paper_notes": len(source_paper_notes),
        "seeded_paper_notes": len(seeded_paper_notes),
    }
    return store, stats


@dataclass
class RunContext:
    project_name: str
    run_dir: Path
    match_result: dict[str, Any]
    coordinate_parsing_by_pmid: dict[str, list[dict[str, Any]]]
    fulltext_index: dict[str, str]
    article_links_by_pmid: dict[str, dict[str, str]]
    pmid_to_coord_tables: dict[str, list[dict[str, str]]]
    pmid_to_html_tables: dict[str, list[dict[str, str]]]


@dataclass
class Unit:
    unit_id: str
    unit_kind: str  # "gold_unit" | "auto_only_unit"
    sample_bucket: str  # "accepted" | "failure"
    project: str
    run_dir: str
    pmid: str
    study_name: str
    match_status: str | None
    manual_analysis_id: str | None
    auto_index: int | None
    combined_score: float | None
    table_id: str | None
    crowding: int
    spans_multiple_tables_hint: bool
    pubget_available: bool
    gold_name: str = ""
    gold_coordinates: list[Any] = field(default_factory=list)
    best_auto_name: str = ""
    best_auto_coordinates: list[Any] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    auto_name: str = ""
    auto_coordinates: list[Any] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-root", type=Path, default=rcpar.PROJECTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--accepted-sample-rate", type=float, default=DEFAULT_ACCEPTED_SAMPLE_RATE)
    parser.add_argument("--accepted-sample-count-per-project", type=int, default=None)
    parser.add_argument("--accepted-sample-min-per-project", type=int, default=0)
    parser.add_argument("--failure-sample-rate", type=float, default=1.0)
    parser.add_argument("--failure-sample-count-per-project", type=int, default=None)
    parser.add_argument(
        "--include-uncertain-gold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include uncertain gold matches in the failure sample. By default, only unmatched "
            "gold analyses are sampled."
        ),
    )
    parser.add_argument(
        "--spurious-candidate-sample-rate",
        type=float,
        default=DEFAULT_SPURIOUS_CANDIDATE_SAMPLE_RATE,
        help=(
            "Sample rate for auto-only spurious-candidate units. Kept separate from "
            "--failure-sample-rate so gold failures can remain fully sampled by default."
        ),
    )
    parser.add_argument("--spurious-candidate-sample-count-per-project", type=int, default=None)
    parser.add_argument(
        "--existing-review",
        type=Path,
        default=None,
        help=(
            "Review JSON to preload into browser localStorage. By default, uses "
            "<output-dir>/reviews/parser_failure_review.json when that file exists."
        ),
    )
    parser.add_argument(
        "--load-existing-review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load surviving annotations from the existing review JSON (default: enabled).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--projects", nargs="+", default=None, help="Optional allow-list of project names.")
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_pmcid(value: Any) -> str:
    text = cab.clean_text(str(value or "")).strip().upper()
    if not text:
        return ""
    text = re.sub(r"^PMCID\s*[:#]?\s*", "", text, flags=re.IGNORECASE)
    if text.startswith("PMC"):
        text = text[3:]
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def normalize_pmid(value: Any) -> str:
    text = cab.clean_text(str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(r"^pmid\s*[:#]?\s*", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def normalize_doi(value: Any) -> str:
    text = cab.clean_text(str(value or "")).strip()
    if not text:
        return ""
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^doi\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def load_article_links_by_pmid(
    run_dir: Path, retrieval_dir: Path | None
) -> dict[str, dict[str, str]]:
    links: dict[str, dict[str, str]] = {}

    def add_row(row: dict[str, Any]) -> None:
        pmid = normalize_pmid(row.get("pmid"))
        if not pmid:
            return
        target = links.setdefault(pmid, {})
        pmcid = normalize_pmcid(row.get("pmcid"))
        doi = normalize_doi(row.get("doi"))
        if pmcid and not target.get("pmc_url"):
            target["pmc_url"] = f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"
        if doi and not target.get("publisher_url"):
            encoded_doi = quote(doi, safe="/:;().-_~")
            target["publisher_url"] = f"https://doi.org/{encoded_doi}"

    search_results_path = run_dir / "outputs" / "search_results.json"
    if search_results_path.exists():
        payload = load_json(search_results_path)
        for row in payload.get("studies", []) if isinstance(payload, dict) else []:
            if isinstance(row, dict):
                add_row(row)

    if retrieval_dir is not None:
        for row in load_csv_rows(retrieval_dir / "metadata.csv"):
            add_row(row)

    return links


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return cab.clean_text(" ".join("".join(element.itertext()).split())).strip()


def first_descendant_text(element: ET.Element, name: str) -> str:
    for child in element.iter():
        if child is not element and local_name(child.tag) == name:
            return element_text(child)
    return ""


def extract_all_table_wraps_html(article_xml_path: Path) -> dict[str, dict[str, str]]:
    if not article_xml_path.exists():
        return {}
    try:
        root = ET.parse(article_xml_path).getroot()
    except ET.ParseError:
        return {}

    tables: dict[str, dict[str, str]] = {}
    fallback_index = 1
    for element in root.iter():
        if local_name(element.tag) != "table-wrap":
            continue
        table_id = cab.clean_text(element.attrib.get("id", "")).strip()
        if not table_id:
            table_id = f"table-wrap-{fallback_index}"
            fallback_index += 1
        if table_id in tables:
            continue
        tables[table_id] = {
            "table_id": table_id,
            "table_label": first_descendant_text(element, "label") or table_id,
            "table_caption": first_descendant_text(element, "caption"),
            "table_foot": first_descendant_text(element, "table-wrap-foot"),
            "table_html": cab.clean_text(ET.tostring(element, encoding="unicode")),
        }
    return tables


def load_all_html_tables_by_pmid(retrieval_dir: Path) -> dict[str, list[dict[str, str]]]:
    metadata_rows = load_csv_rows(retrieval_dir / "metadata.csv")
    tables_rows = load_csv_rows(retrieval_dir / "tables.csv")
    if not metadata_rows:
        return {}

    pmcid_to_pmid: dict[str, str] = {}
    for row in metadata_rows:
        pmcid = normalize_pmcid(row.get("pmcid"))
        pmid = normalize_pmid(row.get("pmid"))
        if pmcid and pmid:
            pmcid_to_pmid[pmcid] = pmid

    table_meta: dict[tuple[str, str], dict[str, str]] = {}
    for row in tables_rows:
        pmcid = normalize_pmcid(row.get("pmcid"))
        table_id = cab.clean_text(row.get("table_id") or "").strip()
        if pmcid and table_id:
            table_meta[(pmcid, table_id)] = {
                "table_label": cab.clean_text(row.get("table_label") or "").strip(),
                "table_caption": cab.clean_text(row.get("table_caption") or "").strip(),
                "table_foot": cab.clean_text(row.get("table_foot") or "").strip(),
            }

    pmid_to_tables: dict[str, list[dict[str, str]]] = {}
    for pmcid, pmid in pmcid_to_pmid.items():
        article_xml = cab.find_article_xml(retrieval_dir, pmcid)
        if article_xml is None:
            continue
        table_html_by_id = extract_all_table_wraps_html(article_xml)
        rows: list[dict[str, str]] = []
        for table_id, table in table_html_by_id.items():
            meta = table_meta.get((pmcid, table_id), {})
            rows.append(
                {
                    "table_id": table_id,
                    "table_label": meta.get("table_label") or table.get("table_label") or table_id,
                    "table_caption": meta.get("table_caption") or table.get("table_caption") or "",
                    "table_foot": meta.get("table_foot") or table.get("table_foot") or "",
                    "table_html": table.get("table_html") or "",
                }
            )
        if rows:
            rows.sort(key=lambda row: row["table_id"])
            pmid_to_tables[pmid] = rows
    return pmid_to_tables


def discover_selections(
    projects_root: Path, projects_filter: list[str] | None
) -> list[rcpar.ProjectSelection]:
    selections = rcpar.discover_project_selections(projects_root)
    if projects_filter:
        allowed = set(projects_filter)
        selections = [s for s in selections if s.project_name in allowed]

    filtered: list[rcpar.ProjectSelection] = []
    for sel in selections:
        if sel.status != "selected" or sel.selected_run_dir is None:
            filtered.append(sel)
            continue
        match_path = sel.selected_run_dir / MATCH_RESULTS_FILE
        if not match_path.exists():
            filtered.append(
                rcpar.ProjectSelection(
                    project_name=sel.project_name,
                    status="skipped",
                    reason=(
                        f"{MATCH_RESULTS_FILE} not found under {sel.selected_run_dir} -- "
                        "run compare_analyses_to_benchmark.py first"
                    ),
                    selected_run_dir=sel.selected_run_dir,
                    selected_version=sel.selected_version,
                    matched_candidate_names=sel.matched_candidate_names,
                )
            )
            continue
        filtered.append(sel)
    return filtered


def load_run_context(project_name: str, run_dir: Path) -> RunContext:
    match_result = load_json(run_dir / MATCH_RESULTS_FILE)

    coordinate_parsing_by_pmid: dict[str, list[dict[str, Any]]] = {}
    coordinate_parsing_path = run_dir / "outputs" / "coordinate_parsing_results.json"
    if coordinate_parsing_path.exists():
        payload = load_json(coordinate_parsing_path)
        for study in payload.get("studies", []) if isinstance(payload, dict) else []:
            pmid = str(study.get("pmid"))
            coordinate_parsing_by_pmid[pmid] = study.get("analyses", []) or []

    fulltext_index: dict[str, str] = {}
    fulltext_path = run_dir / "outputs" / "fulltext_retrieval_results.json"
    if fulltext_path.exists():
        payload = load_json(fulltext_path)
        for row in payload.get("studies_with_fulltext", []) if isinstance(payload, dict) else []:
            pmid = str(row.get("pmid") or "")
            path = row.get("full_text_path")
            if pmid and path:
                fulltext_index[pmid] = str(path)

    pmid_to_coord_tables: dict[str, list[dict[str, str]]] = {}
    pmid_to_html_tables: dict[str, list[dict[str, str]]] = {}
    retrieval_dir = cab.resolve_retrieval_pubget_dir(run_dir)
    article_links_by_pmid = load_article_links_by_pmid(run_dir, retrieval_dir)
    if retrieval_dir is not None:
        try:
            _pmid_to_fulltext, pmid_to_coord_tables = cab.load_retrieval_context(retrieval_dir)
        except Exception:
            pmid_to_coord_tables = {}
        try:
            pmid_to_html_tables = load_all_html_tables_by_pmid(retrieval_dir)
        except Exception:
            pmid_to_html_tables = {}

    return RunContext(
        project_name=project_name,
        run_dir=run_dir,
        match_result=match_result,
        coordinate_parsing_by_pmid=coordinate_parsing_by_pmid,
        fulltext_index=fulltext_index,
        article_links_by_pmid=article_links_by_pmid,
        pmid_to_coord_tables=pmid_to_coord_tables,
        pmid_to_html_tables=pmid_to_html_tables,
    )


def compute_table_id_for_auto_index(
    coordinate_parsing_by_pmid: dict[str, list[dict[str, Any]]], pmid: str, auto_index: int | None
) -> str | None:
    if auto_index is None:
        return None
    analyses = coordinate_parsing_by_pmid.get(pmid, [])
    if 0 <= auto_index < len(analyses):
        table_id = analyses[auto_index].get("table_id")
        return str(table_id) if table_id is not None else None
    return None


def compute_spans_multiple_tables_hint(
    coordinate_parsing_by_pmid: dict[str, list[dict[str, Any]]], pmid: str, analysis_name: str
) -> bool:
    if not analysis_name:
        return False
    analyses = coordinate_parsing_by_pmid.get(pmid, [])
    table_ids = {
        str(a.get("table_id"))
        for a in analyses
        if str(a.get("name") or "") == analysis_name and a.get("table_id") is not None
    }
    return len(table_ids) > 1


def enumerate_units(run_contexts: dict[str, RunContext]) -> tuple[list[Unit], list[Unit]]:
    gold_units: list[Unit] = []
    auto_only_units: list[Unit] = []

    for project_name, ctx in run_contexts.items():
        pmids = ctx.match_result.get("pmids", {})
        for pmid, data in pmids.items():
            study_name = str(data.get("study_name") or pmid)
            pubget_available = bool(data.get("pubget", {}).get("available", False))
            manual_rows = data.get("manual_analyses", []) or []
            auto_rows = data.get("auto_analyses", []) or []
            crowding = len(manual_rows) + len(auto_rows)

            for manual in manual_rows:
                status = str(manual.get("match_status", "unmatched"))
                bucket = "accepted" if status == "accepted" else "failure"
                manual_analysis_id = str(manual.get("manual_analysis_id"))
                best_auto_index_raw = manual.get("best_auto_index")
                best_auto_index = int(best_auto_index_raw) if best_auto_index_raw is not None else None
                table_id = compute_table_id_for_auto_index(
                    ctx.coordinate_parsing_by_pmid, pmid, best_auto_index
                )
                spans_hint = compute_spans_multiple_tables_hint(
                    ctx.coordinate_parsing_by_pmid, pmid, str(manual.get("best_auto_name") or "")
                )
                gold_units.append(
                    Unit(
                        unit_id=f"{project_name}:{pmid}:{manual_analysis_id}",
                        unit_kind="gold_unit",
                        sample_bucket=bucket,
                        project=project_name,
                        run_dir=str(ctx.run_dir),
                        pmid=pmid,
                        study_name=study_name,
                        match_status=status,
                        manual_analysis_id=manual_analysis_id,
                        auto_index=best_auto_index,
                        combined_score=float(manual.get("combined_score", 0.0)),
                        table_id=table_id,
                        crowding=crowding,
                        spans_multiple_tables_hint=spans_hint,
                        pubget_available=pubget_available,
                        gold_name=str(manual.get("manual_name") or ""),
                        gold_coordinates=manual.get("manual_coordinates", []) or [],
                        best_auto_name=str(manual.get("best_auto_name") or ""),
                        best_auto_coordinates=manual.get("best_auto_coordinates", []) or [],
                        reason_codes=[str(c) for c in manual.get("reason_codes", []) or []],
                    )
                )

            auto_by_index = {int(a.get("index", -1)): a for a in auto_rows}
            for auto_idx_raw in data.get("unassigned_auto_indices", []) or []:
                auto_idx = int(auto_idx_raw)
                matched_auto = auto_by_index.get(auto_idx)
                if matched_auto is None:
                    continue
                table_id = compute_table_id_for_auto_index(ctx.coordinate_parsing_by_pmid, pmid, auto_idx)
                spans_hint = compute_spans_multiple_tables_hint(
                    ctx.coordinate_parsing_by_pmid, pmid, str(matched_auto.get("name") or "")
                )
                auto_only_units.append(
                    Unit(
                        unit_id=f"{project_name}:{pmid}:auto{auto_idx}",
                        unit_kind="auto_only_unit",
                        sample_bucket="failure",
                        project=project_name,
                        run_dir=str(ctx.run_dir),
                        pmid=pmid,
                        study_name=study_name,
                        match_status=None,
                        manual_analysis_id=None,
                        auto_index=auto_idx,
                        combined_score=None,
                        table_id=table_id,
                        crowding=crowding,
                        spans_multiple_tables_hint=spans_hint,
                        pubget_available=pubget_available,
                        auto_name=str(matched_auto.get("name") or ""),
                        auto_coordinates=matched_auto.get("coordinates", []) or [],
                    )
                )

    return gold_units, auto_only_units


def group_by_project(units: list[Unit]) -> dict[str, list[Unit]]:
    grouped: dict[str, list[Unit]] = {}
    for unit in units:
        grouped.setdefault(unit.project, []).append(unit)
    return grouped


def stratified_sample_by_project(
    units_by_project: dict[str, list[Unit]],
    rate: float,
    count_override: int | None,
    min_per_project: int,
    seed: int,
) -> list[Unit]:
    sampled: list[Unit] = []
    for project in sorted(units_by_project.keys()):
        units_sorted = sorted(units_by_project[project], key=lambda u: u.unit_id)
        rng = random.Random(f"{seed}:{project}")
        rng.shuffle(units_sorted)
        if count_override is not None:
            n = count_override
        else:
            n = max(min_per_project, round(len(units_sorted) * rate)) if units_sorted else 0
        n = min(n, len(units_sorted))
        sampled.extend(units_sorted[:n])
    return sampled


def build_manifest(
    args: argparse.Namespace,
    selections: list[rcpar.ProjectSelection],
    gold_units: list[Unit],
    auto_only_units: list[Unit],
    sampled_units: list[Unit],
) -> dict[str, Any]:
    included = sorted({sel.project_name for sel in selections if sel.status == "selected"})
    skipped = [
        {"project": sel.project_name, "reason": sel.reason}
        for sel in sorted(selections, key=lambda s: s.project_name)
        if sel.status != "selected"
    ]

    by_project: dict[str, dict[str, int]] = {}
    for unit in gold_units:
        row = by_project.setdefault(unit.project, {"accepted": 0, "failure": 0, "auto_only": 0})
        row[unit.sample_bucket] += 1
    for unit in auto_only_units:
        row = by_project.setdefault(unit.project, {"accepted": 0, "failure": 0, "auto_only": 0})
        row["auto_only"] += 1

    denominators = {
        "total_gold_units_all_projects": len(gold_units),
        "total_accepted_all_projects": sum(1 for u in gold_units if u.sample_bucket == "accepted"),
        "total_failure_all_projects": sum(1 for u in gold_units if u.sample_bucket == "failure"),
        "total_uncertain_all_projects": sum(1 for u in gold_units if u.match_status == "uncertain"),
        "total_unmatched_all_projects": sum(1 for u in gold_units if u.match_status == "unmatched"),
        "total_auto_only_all_projects": len(auto_only_units),
        "by_project": by_project,
    }

    unit_rows = []
    for unit in sampled_units:
        unit_rows.append(
            {
                "unit_id": unit.unit_id,
                "unit_kind": unit.unit_kind,
                "sample_bucket": unit.sample_bucket,
                "project": unit.project,
                "run_dir": unit.run_dir,
                "pmid": unit.pmid,
                "manual_analysis_id": unit.manual_analysis_id,
                "match_status": unit.match_status,
                "combined_score": unit.combined_score,
                "table_id": unit.table_id,
                "crowding": unit.crowding,
                "spans_multiple_tables_hint": unit.spans_multiple_tables_hint,
                "pubget_available": unit.pubget_available,
            }
        )

    return {
        "generated_at": iso_now(),
        "seed": args.seed,
        "accepted_sample_rate": args.accepted_sample_rate,
        "failure_sample_rate": args.failure_sample_rate,
        "spurious_candidate_sample_rate": args.spurious_candidate_sample_rate,
        "include_uncertain_gold": args.include_uncertain_gold,
        "accepted_sample_count_per_project": args.accepted_sample_count_per_project,
        "failure_sample_count_per_project": args.failure_sample_count_per_project,
        "spurious_candidate_sample_count_per_project": args.spurious_candidate_sample_count_per_project,
        "projects_included": included,
        "projects_skipped": skipped,
        "units": unit_rows,
        "denominators": denominators,
    }


# ---------------------------------------------------------------------------
# Row <-> analysis coordinate matching (drives the "seamless split" highlight)
# ---------------------------------------------------------------------------


def extract_numbers_from_text(text: str) -> list[float]:
    normalized = text
    for ch in _MINUS_CHARS:
        normalized = normalized.replace(ch, "-")
    normalized = normalized.replace(_THIN_SPACE, "").replace("\xa0", " ")
    numbers: list[float] = []
    for match in re.finditer(r"-?\d+\.?\d*", normalized):
        try:
            numbers.append(float(match.group()))
        except ValueError:
            continue
    return numbers


def cell_text(cell: ET.Element) -> str:
    return "".join(cell.itertext())


def row_numbers(tr: ET.Element) -> list[float]:
    cells = list(tr)
    if not cells:
        return extract_numbers_from_text("".join(tr.itertext()))
    joined = " ".join(cell_text(c) for c in cells)
    return extract_numbers_from_text(joined)


def iter_table_rows(root: ET.Element) -> list[ET.Element]:
    return [element for element in root.iter() if local_name(element.tag) == "tr"]


def point_matches(numbers: list[float], point: Any, tol: float) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) != 3:
        return False
    try:
        target = [float(point[0]), float(point[1]), float(point[2])]
    except (TypeError, ValueError):
        return False
    for i in range(len(numbers) - 2):
        if all(abs(a - b) <= tol for a, b in zip(numbers[i : i + 3], target)):
            return True
    return False


def annotate_table_html(table_html: str, analyses: list[dict[str, Any]]) -> tuple[str, int]:
    """analyses: [{"coordinates": [[x,y,z],...], "color": str, "badge": str}, ...].
    Returns (annotated_html, matched_row_count). Falls back to the unmodified
    html on any parse failure -- highlighting is a bonus, never a hard requirement."""
    if not table_html or not analyses:
        return table_html, 0
    try:
        root = ET.fromstring(table_html)
    except ET.ParseError:
        return table_html, 0

    matched = 0
    try:
        for tr in iter_table_rows(root):
            numbers = row_numbers(tr)
            if not numbers:
                continue
            hit = None
            for analysis in analyses:
                for point in analysis["coordinates"]:
                    if point_matches(numbers, point, AUTO_MATCH_TOL):
                        hit = analysis
                        break
                if hit:
                    break
            if hit is None:
                continue
            matched += 1
            existing_style = (tr.get("style") or "").rstrip("; ")
            tr.set(
                "style",
                f"{existing_style}; border-left: 5px solid {hit['color']}; "
                f"background: {hit['color']}1a;".lstrip("; "),
            )
            cells = list(tr)
            if cells:
                first_cell = cells[0]
                badge = ET.Element("span")
                badge.set(
                    "style",
                    f"display:inline-block;background:{hit['color']};color:#fff;border-radius:3px;"
                    "padding:0 0.32em;font-size:0.72em;font-weight:700;margin-right:0.4em;",
                )
                badge.text = hit["badge"]
                badge.tail = first_cell.text or ""
                first_cell.text = ""
                first_cell.insert(0, badge)
        return ET.tostring(root, encoding="unicode"), matched
    except Exception:
        return table_html, 0


def compute_gold_hint_tables(gold_coordinates: list[Any], tables: list[dict[str, Any]]) -> list[str]:
    hints: list[str] = []
    for table in tables:
        html = table.get("html") or ""
        if not html:
            continue
        try:
            root = ET.fromstring(html)
        except ET.ParseError:
            continue
        found = False
        try:
            for tr in iter_table_rows(root):
                numbers = row_numbers(tr)
                if not numbers:
                    continue
                for gc in gold_coordinates:
                    if point_matches(numbers, gc, GOLD_HINT_TOL):
                        found = True
                        break
                if found:
                    break
        except Exception:
            continue
        if found:
            hints.append(str(table.get("table_id") or ""))
    return hints


# ---------------------------------------------------------------------------
# Paper-centric view assembly
# ---------------------------------------------------------------------------


@dataclass
class PaperView:
    project: str
    run_dir: str
    pmid: str
    study_name: str
    article_url: str
    article_link_label: str
    fallback_full_text_path: str
    tables: list[dict[str, Any]]
    legend_rows: list[dict[str, Any]]
    gold_rows: list[dict[str, Any]]


def build_paper_views(
    run_contexts: dict[str, RunContext],
    gold_units: list[Unit],
    auto_only_units: list[Unit],
    sampled_ids: set[str],
) -> list[PaperView]:
    gold_by_paper: dict[tuple[str, str], list[Unit]] = {}
    for u in gold_units:
        gold_by_paper.setdefault((u.project, u.pmid), []).append(u)
    auto_only_by_paper: dict[tuple[str, str], list[Unit]] = {}
    for u in auto_only_units:
        auto_only_by_paper.setdefault((u.project, u.pmid), []).append(u)

    paper_keys_in_scope: set[tuple[str, str]] = set()
    for u in gold_units + auto_only_units:
        if u.unit_id in sampled_ids:
            paper_keys_in_scope.add((u.project, u.pmid))

    papers: list[PaperView] = []
    for project, pmid in sorted(paper_keys_in_scope):
        ctx = run_contexts[project]
        data = ctx.match_result.get("pmids", {}).get(pmid, {})
        study_name = str(data.get("study_name") or pmid)
        pubget = data.get("pubget", {}) or {}
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        article_links = ctx.article_links_by_pmid.get(pmid, {})
        pmc_url = (
            str(pubget.get("pmc_url") or "") if pubget.get("available") else ""
        ) or article_links.get("pmc_url", "")
        publisher_url = article_links.get("publisher_url", "")
        if pmc_url:
            article_url = pmc_url
            article_link_label = "PMC full text"
        elif publisher_url:
            article_url = publisher_url
            article_link_label = "Publisher full text"
        else:
            article_url = pubmed_url
            article_link_label = "PubMed"

        gold_paper_units = sorted(
            gold_by_paper.get((project, pmid), []), key=lambda u: u.manual_analysis_id or ""
        )
        auto_only_paper_units = sorted(
            auto_only_by_paper.get((project, pmid), []), key=lambda u: u.auto_index or 0
        )
        auto_only_by_index = {u.auto_index: u for u in auto_only_paper_units}
        gold_by_auto_index = {u.auto_index: u for u in gold_paper_units if u.auto_index is not None}

        analyses_all = ctx.coordinate_parsing_by_pmid.get(pmid, [])
        analyses_by_table: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for idx, a in enumerate(analyses_all):
            table_id = str(a.get("table_id") or "")
            analyses_by_table.setdefault(table_id, []).append((idx, a))

        legend_rows: list[dict[str, Any]] = []
        for row in data.get("auto_analyses", []) or []:
            idx = int(row.get("index", -1))
            color = ANALYSIS_COLOR_PALETTE[idx % len(ANALYSIS_COLOR_PALETTE)]
            badge = f"A{idx + 1}"
            coordinates = row.get("coordinates", []) or []
            if not coordinates and 0 <= idx < len(analyses_all):
                coordinates = [
                    p.get("coordinates")
                    for p in analyses_all[idx].get("points", [])
                    if p.get("coordinates")
                ]
            unit: Unit | None
            context_note = ""
            if idx in auto_only_by_index:
                unit = auto_only_by_index[idx]
                status = "spurious_candidate"
            elif idx in gold_by_auto_index:
                candidate = gold_by_auto_index[idx]
                if candidate.match_status in ("accepted", "uncertain"):
                    unit = candidate
                    status = candidate.match_status
                else:
                    unit = None
                    status = "unmatched_low_confidence_guess"
                    context_note = (
                        f"Low-confidence guess against gold analysis "
                        f"'{candidate.gold_name}' (score={candidate.combined_score:.3f}) -- "
                        "see that analysis in the gold panel instead."
                    )
            else:
                unit = None
                status = "context_only"
            legend_rows.append(
                {
                    "auto_index": idx,
                    "name": str(row.get("name") or ""),
                    "table_id": compute_table_id_for_auto_index(ctx.coordinate_parsing_by_pmid, pmid, idx),
                    "badge": badge,
                    "color": color,
                    "coord_count": int(row.get("coord_count", 0)),
                    "coordinates": coordinates,
                    "status": status,
                    "context_note": context_note,
                    "unit": unit,
                    "in_sample": bool(unit and unit.unit_id in sampled_ids),
                }
            )

        tables_rendered: list[dict[str, Any]] = []
        rendered_table_ids: set[str] = set()
        all_html_tables_by_id = {
            str(t.get("table_id") or ""): t
            for t in ctx.pmid_to_html_tables.get(pmid, [])
            if str(t.get("table_id") or "")
        }
        for t in ctx.pmid_to_coord_tables.get(pmid, []):
            table_id = str(t.get("table_id") or "")
            rendered_table_ids.add(table_id)
            html_fallback = all_html_tables_by_id.get(table_id, {})
            table_analyses = []
            for idx, a in analyses_by_table.get(table_id, []):
                coords = [p.get("coordinates") for p in a.get("points", []) if p.get("coordinates")]
                table_analyses.append(
                    {
                        "coordinates": coords,
                        "color": ANALYSIS_COLOR_PALETTE[idx % len(ANALYSIS_COLOR_PALETTE)],
                        "badge": f"A{idx + 1}",
                    }
                )
            source_html = t.get("table_html", "") or html_fallback.get("table_html", "") or ""
            annotated_html, matched_count = annotate_table_html(source_html, table_analyses)
            total_points = sum(len(ta["coordinates"]) for ta in table_analyses)
            tables_rendered.append(
                {
                    "table_id": table_id,
                    "table_label": t.get("table_label") or html_fallback.get("table_label") or table_id,
                    "table_caption": t.get("table_caption", "") or html_fallback.get("table_caption", ""),
                    "table_foot": t.get("table_foot", "") or html_fallback.get("table_foot", ""),
                    "html": annotated_html,
                    "matched_count": matched_count,
                    "total_points": total_points,
                    "source_kind": "coordinate",
                }
            )

        for t in ctx.pmid_to_html_tables.get(pmid, []):
            table_id = str(t.get("table_id") or "")
            if table_id in rendered_table_ids:
                continue
            rendered_table_ids.add(table_id)
            tables_rendered.append(
                {
                    "table_id": table_id,
                    "table_label": t.get("table_label") or table_id,
                    "table_caption": t.get("table_caption", ""),
                    "table_foot": t.get("table_foot", ""),
                    "html": t.get("table_html", "") or "",
                    "matched_count": 0,
                    "total_points": 0,
                    "source_kind": "supplemental_html",
                }
            )

        for table_id, entries in analyses_by_table.items():
            if table_id and table_id not in rendered_table_ids:
                tables_rendered.append(
                    {
                        "table_id": table_id,
                        "table_label": table_id,
                        "table_caption": "",
                        "table_foot": "",
                        "html": "",
                        "matched_count": 0,
                        "total_points": sum(len(a.get("points", []) or []) for _idx, a in entries),
                        "source_kind": "missing_coordinate_html",
                    }
                )
        tables_rendered.sort(key=lambda t: (t.get("source_kind") != "coordinate", t["table_id"]))

        fallback_full_text_path = "" if pubget.get("available") else ctx.fulltext_index.get(pmid, "")

        predicted_choices = [
            {
                "auto_index": row["auto_index"],
                "badge": row["badge"],
                "name": row["name"],
                "coord_count": len(row.get("coordinates", []) or []),
                "table_id": row.get("table_id") or "?",
            }
            for row in legend_rows
        ]
        gold_rows: list[dict[str, Any]] = []
        for unit in gold_paper_units:
            if unit.match_status != "unmatched":
                continue
            gold_rows.append(
                {
                    "unit": unit,
                    "in_sample": unit.unit_id in sampled_ids,
                    "hint_table_ids": compute_gold_hint_tables(unit.gold_coordinates, tables_rendered),
                    "predicted_choices": predicted_choices,
                }
            )

        papers.append(
            PaperView(
                project=project,
                run_dir=str(ctx.run_dir),
                pmid=pmid,
                study_name=study_name,
                article_url=article_url,
                article_link_label=article_link_label,
                fallback_full_text_path=fallback_full_text_path,
                tables=tables_rendered,
                legend_rows=legend_rows,
                gold_rows=gold_rows,
            )
        )
    return papers


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def coords_text(coords: list[Any]) -> str:
    lines: list[str] = []
    for item in coords:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            try:
                x, y, z = float(item[0]), float(item[1]), float(item[2])
                lines.append(f"{x:.1f}, {y:.1f}, {z:.1f}")
            except Exception:
                continue
    return "\n".join(lines) if lines else "No coordinates."


def render_coord_toggle(label: str, coords: list[Any]) -> str:
    return (
        "<details class=\"analysis-coords\">"
        f"<summary>{escape(label)} ({len(coords)})</summary>"
        f"<pre class=\"coord-list\">{escape(coords_text(coords))}</pre>"
        "</details>"
    )


def render_parsing_reason_options(unit: Unit) -> str:
    options: list[str] = []
    for role, reason_id, label, applies_to, shortcut in PARSING_REASON_OPTIONS:
        if applies_to not in ("both", unit.unit_kind):
            continue
        shortcut_html = (
            f'<span class="key-hint">[{escape(shortcut)}]</span> ' if shortcut else ""
        )
        trigger_attr = f' data-trigger-id="{escape(reason_id)}"' if role == "trigger" else ""
        shortcut_attr = f' data-shortcut="{escape(shortcut)}"' if shortcut else ""
        options.append(
            '<label class="review-reason-option">'
            f'<input type="checkbox" data-role="{escape(role)}" value="{escape(reason_id)}"'
            f'{trigger_attr}{shortcut_attr}> '
            f'{shortcut_html}{escape(label)}'
            "</label>"
        )
    return "".join(options)


def render_unit_review_controls(
    unit: Unit, in_sample: bool, predicted_choices: list[dict[str, Any]] | None = None
) -> str:
    if not in_sample:
        return "<p class=\"resource-note context-only-note\">Context only (not in review sample).</p>"

    parts: list[str] = []
    if unit.unit_kind == "gold_unit" and unit.match_status == "unmatched":
        radio_name = f"unmatched-gold-disposition-{unit.unit_id}"
        parts.append(
            "<div class=\"disposition-group\">"
            "<p><strong>Why is this gold analysis unmatched?</strong></p>"
            f"<label><input type=\"radio\" data-role=\"unmatched-gold-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"parser_missed\"> Parser missed it although the source was available "
            "<span class=\"key-hint\">[g]</span></label>"
            f"<label><input type=\"radio\" data-role=\"unmatched-gold-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"source_material_missing\"> Source table or text was missing/unavailable "
            "<span class=\"key-hint\">[t]</span></label>"
            f"<label><input type=\"radio\" data-role=\"unmatched-gold-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"supplemental_data\"> Analysis was only in supplemental data "
            "<span class=\"key-hint\">[s]</span></label>"
            f"<label><input type=\"radio\" data-role=\"unmatched-gold-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"matching_error\"> Matching error: gold and prediction are both correct "
            "<span class=\"key-hint\">[m]</span></label>"
            f"<label><input type=\"radio\" data-role=\"unmatched-gold-disposition\" name=\"{escape(radio_name)}\" "
            f"value=\"{escape(taxonomy.EXPECTED_DIFFERENCE_DISPOSITION)}\"> "
            "Expected source/curation difference (neither is wrong), such as corrected unusual "
            "coordinates or filtered non-significant peaks "
            "<span class=\"key-hint\">[d]</span></label>"
            f"<label><input type=\"radio\" data-role=\"unmatched-gold-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"gold_standard_error\"> Gold standard is wrong "
            "<span class=\"key-hint\">[w]</span></label>"
            f"<label><input type=\"radio\" data-role=\"unmatched-gold-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"out_of_scope\"> Out of scope (real, but not curated in gold) "
            "<span class=\"key-hint\">[o]</span></label>"
            f"<label><input type=\"radio\" data-role=\"unmatched-gold-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"uncertain\"> Unsure <span class=\"key-hint\">[u]</span></label>"
            f"<label><input type=\"radio\" data-role=\"unmatched-gold-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"\"> Clear selection</label>"
            "</div>"
        )
        choices_html = ['<option value="">-- select predicted analysis --</option>']
        for choice in predicted_choices or []:
            suggested = " (closest automatic guess)" if choice["auto_index"] == unit.auto_index else ""
            option_label = (
                f"{choice['badge']} - {choice['name']} "
                f"({choice['coord_count']} coords, table {choice['table_id']}){suggested}"
            )
            choices_html.append(
                f'<option value="{escape(str(choice["auto_index"]))}" '
                f'data-name="{escape(str(choice["name"]))}">{escape(option_label)}</option>'
            )
        suggested_index = str(unit.auto_index) if unit.auto_index is not None else ""
        parts.append(
            '<div class="matching-error-fields" data-role="matching-error-fields" hidden>'
            '<label class="matching-error-field"><strong>Correct predicted analysis</strong>'
            f'<select data-role="matching-predicted-index" '
            f'data-suggested-index="{escape(suggested_index)}">{"".join(choices_html)}</select></label>'
            '<label class="matching-error-field"><strong>Why matching failed</strong>'
            '<select data-role="matching-failure-reason">'
            '<option value="">-- select reason --</option>'
            '<option value="coordinates_out_of_order">Coordinates are the same but ordered differently</option>'
            '<option value="coordinate_tolerance_or_space">Coordinate tolerance or coordinate-space issue</option>'
            '<option value="label_similarity">Analysis-label similarity failed</option>'
            '<option value="other">Other matching issue</option>'
            "</select></label></div>"
        )
    elif unit.unit_kind == "gold_unit":
        radio_name = f"parsing-disposition-{unit.unit_id}"
        parts.append(
            "<div class=\"disposition-group\">"
            "<p><strong>Parsing assessment</strong></p>"
            f"<label><input type=\"radio\" data-role=\"parsing-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"correct\"> Looks correct <span class=\"key-hint\">[c]</span></label>"
            f"<label><input type=\"radio\" data-role=\"parsing-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"error\"> Parsing error <span class=\"key-hint\">[g]</span></label>"
            f"<label><input type=\"radio\" data-role=\"parsing-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"gold_standard_error\"> Gold standard is wrong "
            "<span class=\"key-hint\">[w]</span></label>"
            f"<label><input type=\"radio\" data-role=\"parsing-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"uncertain\"> Unsure <span class=\"key-hint\">[u]</span></label>"
            f"<label><input type=\"radio\" data-role=\"parsing-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"\"> Clear selection</label>"
            "</div>"
        )

    if unit.unit_kind == "auto_only_unit":
        radio_name = f"spurious-disposition-{unit.unit_id}"
        parts.append(
            "<div class=\"disposition-group\">"
            "<p><strong>Spurious or out of scope?</strong> "
            "<span class=\"key-hint\">[g]=fabricated  [o]=out of scope</span></p>"
            f"<label><input type=\"radio\" data-role=\"spurious-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"spurious_fabricated\"> Spurious / fabricated</label>"
            f"<label><input type=\"radio\" data-role=\"spurious-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"out_of_scope_real\"> Out of scope (real, correct analysis)</label>"
            f"<label><input type=\"radio\" data-role=\"spurious-disposition\" name=\"{escape(radio_name)}\" "
            "value=\"\"> Clear selection</label>"
            "</div>"
        )
    parts.append(
        "<div class=\"review-reasons\" data-role=\"parsing-reason-fields\" hidden>"
        "<p><strong>What went wrong?</strong> Select all that apply.</p>"
        f'<div class="parsing-reason-grid">{render_parsing_reason_options(unit)}</div>'
        + "</div>"
    )
    parts.append(
        f"<p class=\"resource-note\">Crowding (analyses in paper): {unit.crowding}</p>"
        "<textarea data-role=\"review-note\" rows=\"2\" placeholder=\"Notes... [e]\"></textarea>"
    )
    return "".join(parts)


def render_unit_review_wrapper(unit: Unit, in_sample: bool, body_html: str) -> str:
    css_class = "unit-review" + (" in-sample" if in_sample else " context-only")
    return (
        f"<div class=\"{css_class}\" data-unit-id=\"{escape(unit.unit_id)}\" "
        f"data-unit-kind=\"{escape(unit.unit_kind)}\" data-project=\"{escape(unit.project)}\" "
        f"data-run-dir=\"{escape(unit.run_dir)}\" data-pmid=\"{escape(unit.pmid)}\" "
        f"data-match-status=\"{escape(unit.match_status or '')}\">"
        f"{body_html}"
        "</div>"
    )


def render_legend_row(row: dict[str, Any]) -> str:
    unit = row["unit"]
    status_label = {
        "accepted": "accepted",
        "uncertain": "uncertain",
        "spurious_candidate": "spurious candidate",
        "unmatched_low_confidence_guess": "unmatched (low-confidence guess)",
        "context_only": "context",
    }.get(row["status"], row["status"])
    status_class = {
        "accepted": "st-accepted",
        "uncertain": "st-uncertain",
        "spurious_candidate": "st-auto-only",
        "unmatched_low_confidence_guess": "st-unmatched",
    }.get(row["status"], "")

    gold_match_text = ""
    if unit is not None and unit.unit_kind == "gold_unit":
        gold_match_text = f"<p class=\"resource-note\"><strong>Matched gold:</strong> {escape(unit.gold_name)}</p>"
    context_note_html = (
        f"<p class=\"resource-note\">{escape(row['context_note'])}</p>" if row.get("context_note") else ""
    )
    coordinates_html = render_coord_toggle("Predicted coordinates", row.get("coordinates", []) or [])

    header = (
        f"<div class=\"legend-header\">"
        f"<span class=\"badge-chip\" style=\"background:{row['color']}\">{escape(row['badge'])}</span> "
        f"<strong>{escape(row['name'])}</strong> "
        f"<span class=\"status-pill\">{escape(status_label)}</span> "
        f"<span class=\"resource-note\">({row['coord_count']} coords, table_id={escape(row['table_id'] or '?')})</span>"
        f"</div>{gold_match_text}{context_note_html}{coordinates_html}"
    )
    if unit is None:
        return f"<div class=\"legend-row {status_class}\">{header}</div>"
    body = render_unit_review_controls(unit, row["in_sample"])
    wrapped = render_unit_review_wrapper(unit, row["in_sample"], body)
    return f"<div class=\"legend-row {status_class}\">{header}{wrapped}</div>"


def render_table_block(table: dict[str, Any]) -> str:
    if table["html"]:
        body = f"<div class=\"table-html\">{table['html']}</div>"
        if table.get("total_points", 0):
            coverage = (
                f"<p class=\"resource-note\">Highlighted {table['matched_count']}/{table['total_points']} "
                "coordinate rows by predicted analysis (colored left border + badge).</p>"
            )
        else:
            coverage = "<p class=\"resource-note\">Supplemental HTML table found in article XML.</p>"
    else:
        body = (
            "<p class=\"resource-note\">Original table HTML not available for this table_id "
            "(not PMC open access, or table not extracted).</p>"
        )
        coverage = ""
    caption = f"<p><strong>Caption:</strong> {escape(table['table_caption'])}</p>" if table["table_caption"] else ""
    foot = f"<p><strong>Footnote:</strong> {escape(table['table_foot'])}</p>" if table["table_foot"] else ""
    return (
        "<div class=\"table-block\">"
        f"<h4>{escape(table['table_label'])} "
        f"<span class=\"resource-note\">(table_id={escape(table['table_id'])})</span></h4>"
        f"{caption}{foot}{coverage}{body}"
        "</div>"
    )


def render_gold_row(row: dict[str, Any]) -> str:
    unit = row["unit"]
    best_guess = ""
    if unit.table_id:
        best_guess = (
            f"<p class=\"resource-note\">Closest automatic guess: table_id={escape(unit.table_id)} "
            f"(predicted as '{escape(unit.best_auto_name)}', score={unit.combined_score:.3f})</p>"
        )
    hint_text = ""
    if row["hint_table_ids"]:
        hint_text = (
            "<p class=\"resource-note\">Possible source table(s) by coordinate proximity "
            f"(approximate, unverified): {escape(', '.join(row['hint_table_ids']))}</p>"
        )
    header = (
        "<div class=\"legend-header\">"
        f"<strong>{escape(unit.gold_name)}</strong> "
        "<span class=\"status-pill\">unmatched</span> "
        f"<span class=\"resource-note\">({len(unit.gold_coordinates)} coords)</span>"
        "</div>"
        f"{render_coord_toggle('Gold coordinates', unit.gold_coordinates)}"
        f"{best_guess}{hint_text}"
    )
    body = render_unit_review_controls(unit, row["in_sample"], row.get("predicted_choices", []))
    wrapped = render_unit_review_wrapper(unit, row["in_sample"], body)
    return f"<div class=\"gold-row st-unmatched\">{header}{wrapped}</div>"


def render_paper_header(paper: PaperView) -> str:
    links = [
        f'<a href="{escape(paper.article_url)}" target="_blank" rel="noopener noreferrer">'
        f'{escape(paper.article_link_label)}</a>'
    ]
    if paper.fallback_full_text_path:
        links.append(f'<a href="file://{escape(paper.fallback_full_text_path)}">Local full text</a>')
    return (
        "<div class=\"paper-header\">"
        f"<strong>PMID {escape(paper.pmid)}</strong> &middot; {escape(paper.study_name)} &middot; "
        f"project={escape(paper.project)} &middot; {' &middot; '.join(links)}"
        "</div>"
    )


def render_paper_card(paper: PaperView) -> str:
    tables_html = "".join(render_table_block(t) for t in paper.tables) or (
        "<p class=\"resource-note\">No source tables available for this paper.</p>"
    )
    legend_html = "".join(render_legend_row(r) for r in paper.legend_rows) or (
        "<p class=\"resource-note\">No predicted analyses for this paper.</p>"
    )
    gold_html = "".join(render_gold_row(r) for r in paper.gold_rows) or (
        "<p class=\"resource-note\">No unmatched gold analyses for this paper.</p>"
    )
    reviewable_count = sum(1 for r in paper.legend_rows if r["in_sample"]) + sum(
        1 for r in paper.gold_rows if r["in_sample"]
    )
    return (
        f"<details class=\"paper-card\" data-paper-key=\"{escape(paper.project)}:{escape(paper.pmid)}\" open>"
        f"<summary>PMID {escape(paper.pmid)} &middot; {escape(paper.study_name)} &middot; "
        f"project={escape(paper.project)} &middot; {reviewable_count} unit(s) to review</summary>"
        "<div class=\"paper-body\">"
        f"{render_paper_header(paper)}"
        "<div class=\"paper-flag-row\">"
        f"<label class=\"paper-flag-label\"><input type=\"checkbox\" class=\"paper-flag-missed-table\" "
        f"data-paper-key=\"{escape(paper.project)}:{escape(paper.pmid)}\"> "
        "<strong>&#9888; Missed table</strong> — one or more coordinate tables were not retrieved/parsed at all</label>"
        "</div>"
        "<div class=\"paper-columns\">"
        "<div class=\"paper-tables-col\">"
        f"<h3>Original tables + predicted split</h3>{tables_html}"
        f"<h3>Predicted analyses (legend)</h3>{legend_html}"
        "</div>"
        f"<div class=\"paper-gold-col\"><h3>Gold-standard: unmatched analyses</h3>{gold_html}</div>"
        "</div>"
        "</div>"
        "</details>"
    )


REVIEW_SCRIPT_TEMPLATE = """
<script>
(() => {
  const STORAGE_KEY = "parser_failure_annotation_v1";
  const SEEDED_STORE = __SEEDED_STORE__;
  const panels = Array.from(document.querySelectorAll(".unit-review"));
  const currentUnitIds = new Set(
    panels
      .filter((panel) => panel.classList.contains("in-sample"))
      .map((panel) => panel.getAttribute("data-unit-id") || "")
      .filter(Boolean)
  );
  const currentPaperKeys = new Set(
    Array.from(document.querySelectorAll(".paper-card"))
      .map((card) => card.getAttribute("data-paper-key") || "")
      .filter(Boolean)
  );
  const TOTAL_UNITS = __TOTAL_UNITS__;
  const progressEl = document.getElementById("review-progress");
  const exportJsonBtn = document.getElementById("review-export-json");
  const exportCsvBtn = document.getElementById("review-export-csv");
  const clearBtn = document.getElementById("review-clear");
  const reviewerInput = document.getElementById("review-reviewer");

  function normalizeStore(value) {
    const parsed = value && typeof value === "object" ? value : {};
    if (!parsed.entries || typeof parsed.entries !== "object") parsed.entries = {};
    if (!parsed.paper_notes || typeof parsed.paper_notes !== "object") parsed.paper_notes = {};
    Object.values(parsed.entries).forEach((entry) => {
      if (!entry || typeof entry !== "object") return;
      if (!Array.isArray(entry.parsing_reasons)) {
        entry.parsing_reasons = Array.isArray(entry.legacy_reasons)
          ? entry.legacy_reasons.slice()
          : [];
      }
      if (!entry.unmatched_gold_disposition && entry.missed_unit_disposition) {
        const oldDispositionMap = {
          missed_unit_confirmed: "parser_missed",
          missed_unit_supplemental_data: "supplemental_data",
          missed_unit_out_of_scope: "out_of_scope",
          gold_standard_wrong: "gold_standard_error",
        };
        entry.unmatched_gold_disposition =
          oldDispositionMap[entry.missed_unit_disposition] || entry.missed_unit_disposition;
      }
      const hasSavedReasons =
        (entry.failure_modes || []).length || entry.parsing_reasons.length ||
        entry.spans_multiple_tables || entry.footnote_carried_context;
      if (
        !entry.parsing_disposition && entry.unit_kind === "gold_unit" &&
        entry.match_status !== "unmatched" && hasSavedReasons
      ) {
        entry.parsing_disposition = "error";
      }
    });
    return parsed;
  }

  function readStore() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return normalizeStore(raw ? JSON.parse(raw) : {});
    } catch (_err) {
      return normalizeStore({});
    }
  }

  function updatedAtMillis(value) {
    const millis = Date.parse((value && value.updated_at) || "");
    return Number.isFinite(millis) ? millis : 0;
  }

  function mergeStores(seededValue, localValue) {
    const seeded = normalizeStore(seededValue);
    const local = normalizeStore(localValue);
    const merged = {
      entries: {},
      paper_notes: {},
      reviewer: local.reviewer || seeded.reviewer || "",
    };

    currentUnitIds.forEach((unitId) => {
      const seededEntry = seeded.entries[unitId];
      const localEntry = local.entries[unitId];
      if (seededEntry && localEntry) {
        merged.entries[unitId] =
          updatedAtMillis(seededEntry) > updatedAtMillis(localEntry)
            ? seededEntry
            : localEntry;
      } else if (localEntry || seededEntry) {
        merged.entries[unitId] = localEntry || seededEntry;
      }
    });

    currentPaperKeys.forEach((paperKey) => {
      const seededNote = seeded.paper_notes[paperKey];
      const localNote = local.paper_notes[paperKey];
      if (seededNote && localNote) {
        merged.paper_notes[paperKey] =
          updatedAtMillis(seededNote) > updatedAtMillis(localNote)
            ? seededNote
            : localNote;
      } else if (localNote || seededNote) {
        merged.paper_notes[paperKey] = localNote || seededNote;
      }
    });
    return normalizeStore(merged);
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
    if (text.includes(",") || text.includes("\\n") || text.includes("\\"")) {
      return "\\""+text.replace(/\\"/g, "\\"\\"")+"\\"";
    }
    return text;
  }

  const CSV_HEADER = [
    "unit_id", "unit_kind", "project", "run_dir", "pmid", "match_status",
    "parsing_disposition", "unmatched_gold_disposition", "spurious_disposition",
    "matching_predicted_index", "matching_predicted_name", "matching_failure_reason",
    "failure_modes", "parsing_reasons",
    "spans_multiple_tables",
    "footnote_carried_context", "note", "updated_at"
  ];

  function buildCsv(entries) {
    const lines = [CSV_HEADER.join(",")];
    for (const row of entries) {
      lines.push(CSV_HEADER.map((key) => {
        const value = row[key];
        if (Array.isArray(value)) return escapeCsvValue(value.join("|"));
        return escapeCsvValue(value);
      }).join(","));
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

  function setReasonVisibility(panel) {
    const reasons = panel.querySelector('[data-role="parsing-reason-fields"]');
    if (!reasons) return;
    const parsing = panel.querySelector('input[data-role="parsing-disposition"]:checked');
    const unmatched = panel.querySelector('input[data-role="unmatched-gold-disposition"]:checked');
    const spurious = panel.querySelector('input[data-role="spurious-disposition"]:checked');
    const hasError =
      (parsing && parsing.value === "error") ||
      (unmatched && unmatched.value === "parser_missed") ||
      (spurious && spurious.value === "spurious_fabricated");
    reasons.hidden = !hasError;

    const matchingFields = panel.querySelector('[data-role="matching-error-fields"]');
    if (matchingFields) {
      const isMatchingError = unmatched && unmatched.value === "matching_error";
      matchingFields.hidden = !isMatchingError;
      if (isMatchingError) {
        const select = matchingFields.querySelector('[data-role="matching-predicted-index"]');
        if (select && !select.value) {
          const suggested = select.getAttribute("data-suggested-index") || "";
          if (suggested && select.querySelector('option[value="' + suggested + '"]')) {
            select.value = suggested;
          }
        }
      }
    }
  }

  function clearReasons(panel) {
    panel.querySelectorAll(
      '[data-role="parsing-reason-fields"] input[type="checkbox"]'
    ).forEach((checkbox) => { checkbox.checked = false; });
  }

  function clearMatchingFields(panel) {
    const predicted = panel.querySelector('[data-role="matching-predicted-index"]');
    const reason = panel.querySelector('[data-role="matching-failure-reason"]');
    if (predicted) predicted.value = "";
    if (reason) reason.value = "";
  }

  function collectEntry(panel) {
    const unitId = panel.getAttribute("data-unit-id") || "";
    const unitKind = panel.getAttribute("data-unit-kind") || "";
    const parsingRadio = panel.querySelector('input[data-role="parsing-disposition"]:checked');
    const unmatchedRadio = panel.querySelector(
      'input[data-role="unmatched-gold-disposition"]:checked'
    );
    const spuriousRadio = panel.querySelector('input[data-role="spurious-disposition"]:checked');
    const matchingPredicted = panel.querySelector('[data-role="matching-predicted-index"]');
    const matchingOption = matchingPredicted
      ? matchingPredicted.options[matchingPredicted.selectedIndex]
      : null;
    const matchingReason = panel.querySelector('[data-role="matching-failure-reason"]');
    const failureModes = Array.from(
      panel.querySelectorAll('input[data-role="failure-mode"]:checked')
    ).map((n) => n.value);
    const parsingReasons = Array.from(
      panel.querySelectorAll('input[data-role="parsing-reason"]:checked')
    ).map((n) => n.value);
    const spansEl = panel.querySelector('input[data-trigger-id="spans_multiple_tables"]');
    const footnoteEl = panel.querySelector('input[data-trigger-id="footnote_carried_context"]');
    const noteEl = panel.querySelector('textarea[data-role="review-note"]');
    return {
      unit_id: unitId,
      unit_kind: unitKind,
      project: panel.getAttribute("data-project") || "",
      run_dir: panel.getAttribute("data-run-dir") || "",
      pmid: panel.getAttribute("data-pmid") || "",
      match_status: panel.getAttribute("data-match-status") || "",
      parsing_disposition: parsingRadio ? parsingRadio.value : "",
      unmatched_gold_disposition: unmatchedRadio ? unmatchedRadio.value : "",
      spurious_disposition: spuriousRadio ? spuriousRadio.value : "",
      matching_predicted_index: matchingPredicted ? matchingPredicted.value : "",
      matching_predicted_name: matchingOption
        ? matchingOption.getAttribute("data-name") || ""
        : "",
      matching_failure_reason: matchingReason ? matchingReason.value : "",
      failure_modes: failureModes,
      parsing_reasons: parsingReasons,
      spans_multiple_tables: spansEl ? spansEl.checked : false,
      footnote_carried_context: footnoteEl ? footnoteEl.checked : false,
      note: noteEl ? noteEl.value.trim() : "",
      updated_at: new Date().toISOString(),
    };
  }

  function isTouched(entry) {
    return Boolean(
      entry.parsing_disposition || entry.unmatched_gold_disposition ||
      entry.missed_unit_disposition || entry.spurious_disposition ||
      entry.matching_predicted_index || entry.matching_failure_reason ||
      (entry.failure_modes || []).length ||
      (entry.parsing_reasons || entry.legacy_reasons || []).length ||
      entry.spans_multiple_tables || entry.footnote_carried_context || entry.note
    );
  }

  function loadPanel(panel, entry) {
    setReasonVisibility(panel);
    if (!entry) return;
    const parsingRadios = panel.querySelectorAll('input[data-role="parsing-disposition"]');
    parsingRadios.forEach((r) => { r.checked = r.value === entry.parsing_disposition; });
    const unmatchedDisposition =
      entry.unmatched_gold_disposition ||
      ({
        missed_unit_confirmed: "parser_missed",
        missed_unit_supplemental_data: "supplemental_data",
        missed_unit_out_of_scope: "out_of_scope",
        gold_standard_wrong: "gold_standard_error",
      })[entry.missed_unit_disposition] ||
      "";
    const unmatchedRadios = panel.querySelectorAll(
      'input[data-role="unmatched-gold-disposition"]'
    );
    unmatchedRadios.forEach((r) => { r.checked = r.value === unmatchedDisposition; });
    const spuriousRadios = panel.querySelectorAll('input[data-role="spurious-disposition"]');
    spuriousRadios.forEach((r) => { r.checked = r.value === entry.spurious_disposition; });
    const failureSet = new Set(entry.failure_modes || []);
    panel.querySelectorAll('input[data-role="failure-mode"]').forEach((c) => {
      c.checked = failureSet.has(c.value);
    });
    const parsingSet = new Set(entry.parsing_reasons || entry.legacy_reasons || []);
    panel.querySelectorAll('input[data-role="parsing-reason"]').forEach((c) => {
      c.checked = parsingSet.has(c.value);
    });
    const spansEl = panel.querySelector('input[data-trigger-id="spans_multiple_tables"]');
    if (spansEl) spansEl.checked = Boolean(entry.spans_multiple_tables);
    const footnoteEl = panel.querySelector('input[data-trigger-id="footnote_carried_context"]');
    if (footnoteEl) footnoteEl.checked = Boolean(entry.footnote_carried_context);
    const matchingPredicted = panel.querySelector('[data-role="matching-predicted-index"]');
    if (matchingPredicted) matchingPredicted.value = entry.matching_predicted_index || "";
    const matchingReason = panel.querySelector('[data-role="matching-failure-reason"]');
    if (matchingReason) matchingReason.value = entry.matching_failure_reason || "";
    const noteEl = panel.querySelector('textarea[data-role="review-note"]');
    if (noteEl) noteEl.value = entry.note || "";
    setReasonVisibility(panel);
  }

  function updateProgress(entries) {
    if (!progressEl) return;
    const completed = Object.values(entries).filter(isTouched).length;
    progressEl.innerHTML = "<strong>Review progress:</strong> " + completed + " / " + TOTAL_UNITS + " completed";
  }

  const store = mergeStores(SEEDED_STORE, readStore());
  if (reviewerInput) {
    reviewerInput.value = store.reviewer || "";
    reviewerInput.addEventListener("input", () => {
      store.reviewer = reviewerInput.value.trim();
      writeStore(store);
    });
  }
  writeStore(store);
  panels.forEach((panel) => {
    const unitId = panel.getAttribute("data-unit-id") || "";
    loadPanel(panel, store.entries[unitId]);
  });
  updateProgress(store.entries);

  function persistPanel(panel) {
    const entry = collectEntry(panel);
    if (!entry.unit_id) return;
    if (!isTouched(entry)) {
      delete store.entries[entry.unit_id];
    } else {
      store.entries[entry.unit_id] = entry;
    }
    writeStore(store);
    setReasonVisibility(panel);
    updateProgress(store.entries);
  }

  panels.forEach((panel) => {
    panel.addEventListener("change", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      if (target.matches(
        'input[data-role="parsing-disposition"], ' +
        'input[data-role="unmatched-gold-disposition"], ' +
        'input[data-role="spurious-disposition"]'
      )) {
        const value = target.getAttribute("value") || "";
        const errorValues = new Set(["error", "parser_missed", "spurious_fabricated"]);
        if (!errorValues.has(value)) clearReasons(panel);
        if (value !== "matching_error") clearMatchingFields(panel);
        setReasonVisibility(panel);
      }
      persistPanel(panel);
    });
    panel.addEventListener("input", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      if (target.matches('textarea[data-role="review-note"]')) {
        persistPanel(panel);
      }
    });
  });

  function getExportEntries() {
    const entries = [];
    for (const panel of panels) {
      const unitId = panel.getAttribute("data-unit-id") || "";
      const stored = store.entries[unitId];
      if (stored) entries.push(stored);
    }
    entries.sort((a, b) => String(a.unit_id).localeCompare(String(b.unit_id)));
    return entries;
  }

  if (exportJsonBtn) {
    exportJsonBtn.addEventListener("click", () => {
      const entries = getExportEntries();
      const paperNotesArr = Object.entries(store.paper_notes || {}).map(([key, val]) => ({
        paper_key: key, ...val,
      }));
      const payload = {
        generated_at: new Date().toISOString(),
        manifest_generated_at: "__MANIFEST_GENERATED_AT__",
        reviewer: reviewerInput ? reviewerInput.value.trim() : store.reviewer || "",
        total_review_units: TOTAL_UNITS,
        completed_reviews: entries.filter(isTouched).length,
        entries,
        paper_notes: paperNotesArr,
      };
      downloadFile("parser_failure_review.json", JSON.stringify(payload, null, 2), "application/json");
    });
  }

  if (exportCsvBtn) {
    exportCsvBtn.addEventListener("click", () => {
      const csv = buildCsv(getExportEntries());
      downloadFile("parser_failure_review.csv", csv, "text/csv;charset=utf-8");
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      if (!window.confirm("Clear all saved parser-failure review annotations for this report?")) return;
      store.entries = {};
      store.paper_notes = {};
      writeStore(store);
      panels.forEach((panel) => loadPanel(panel, null));
      updateProgress(store.entries);
      document.querySelectorAll(".paper-flag-missed-table").forEach((cb) => {
        cb.checked = false;
        const card = cb.closest(".paper-card");
        if (card) card.removeAttribute("data-missed-table");
      });
    });
  }

  // --- Paper-level missed-table flag ---
  if (!store.paper_notes || typeof store.paper_notes !== "object") store.paper_notes = {};

  function persistPaperNote(paperKey, missed) {
    if (missed) {
      store.paper_notes[paperKey] = { missed_table: true, updated_at: new Date().toISOString() };
    } else {
      delete store.paper_notes[paperKey];
    }
    writeStore(store);
  }

  document.querySelectorAll(".paper-flag-missed-table").forEach((cb) => {
    const paperKey = cb.getAttribute("data-paper-key") || "";
    const saved = store.paper_notes[paperKey];
    if (saved && saved.missed_table) {
      cb.checked = true;
      const card = cb.closest(".paper-card");
      if (card) card.setAttribute("data-missed-table", "true");
    }
    cb.addEventListener("change", () => {
      const card = cb.closest(".paper-card");
      if (card) {
        if (cb.checked) card.setAttribute("data-missed-table", "true");
        else card.removeAttribute("data-missed-table");
      }
      persistPaperNote(paperKey, cb.checked);
    });
  });
})();
</script>
"""

KEYBOARD_NAV_SCRIPT = """
<script>
(() => {
  const paperCards = Array.from(document.querySelectorAll(".paper-card"));
  if (!paperCards.length) return;
  let currentPaperIdx = 0;
  let currentUnitIdx = 0;

  function reviewableUnitsFor(card) {
    return Array.from(card.querySelectorAll(".unit-review.in-sample"));
  }

  function clearFocusStyles() {
    document.querySelectorAll(".unit-review.focused").forEach((el) => el.classList.remove("focused"));
  }

  function focusUnit(idx) {
    clearFocusStyles();
    const card = paperCards[currentPaperIdx];
    if (!card) return;
    const units = reviewableUnitsFor(card);
    if (!units.length) return;
    currentUnitIdx = ((idx % units.length) + units.length) % units.length;
    const unit = units[currentUnitIdx];
    unit.classList.add("focused");
    unit.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function focusPaper(idx) {
    currentPaperIdx = ((idx % paperCards.length) + paperCards.length) % paperCards.length;
    paperCards.forEach((card, i) => { card.open = (i === currentPaperIdx); });
    const card = paperCards[currentPaperIdx];
    card.scrollIntoView({ behavior: "smooth", block: "start" });
    currentUnitIdx = 0;
    focusUnit(0);
  }

  function currentFocusedUnit() {
    const card = paperCards[currentPaperIdx];
    if (!card) return null;
    const units = reviewableUnitsFor(card);
    return units[currentUnitIdx] || null;
  }

  function isTypingTarget(el) {
    return el && (el.tagName === "TEXTAREA" || el.tagName === "INPUT" || el.tagName === "SELECT");
  }

  function toggleHelp() {
    const el = document.getElementById("shortcut-help-overlay");
    if (el) el.hidden = !el.hidden;
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "?") {
      toggleHelp();
      return;
    }
    if (isTypingTarget(document.activeElement)) {
      if (event.key === "Escape") document.activeElement.blur();
      return;
    }
    switch (event.key) {
      case "j":
      case "ArrowDown":
        event.preventDefault();
        focusPaper(currentPaperIdx + 1);
        break;
      case "k":
      case "ArrowUp":
        event.preventDefault();
        focusPaper(currentPaperIdx - 1);
        break;
      case "n":
        event.preventDefault();
        focusUnit(currentUnitIdx + 1);
        break;
      case "p":
        event.preventDefault();
        focusUnit(currentUnitIdx - 1);
        break;
      case "1":
      case "2":
      case "3":
      case "4": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const box = unit.querySelector('input[data-shortcut="' + event.key + '"]');
        if (box) {
          const problemRadio = unit.querySelector(
            'input[data-role="parsing-disposition"][value="error"], ' +
            'input[data-role="unmatched-gold-disposition"][value="parser_missed"], ' +
            'input[data-role="spurious-disposition"][value="spurious_fabricated"]'
          );
          if (problemRadio) problemRadio.checked = true;
          box.checked = !box.checked;
          box.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
      case "g": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const radio = unit.querySelector(
          'input[data-role="parsing-disposition"][value="error"], ' +
          'input[data-role="unmatched-gold-disposition"][value="parser_missed"], ' +
          'input[data-role="spurious-disposition"][value="spurious_fabricated"]'
        );
        if (radio) {
          radio.checked = true;
          radio.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
      case "c": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const radio = unit.querySelector('input[data-role="parsing-disposition"][value="correct"]');
        if (radio) {
          radio.checked = true;
          radio.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
      case "u": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const radio = unit.querySelector(
          'input[data-role="parsing-disposition"][value="uncertain"], ' +
          'input[data-role="unmatched-gold-disposition"][value="uncertain"]'
        );
        if (radio) {
          radio.checked = true;
          radio.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
      case "o": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const radio = unit.querySelector(
          'input[data-role="unmatched-gold-disposition"][value="out_of_scope"], ' +
          'input[data-role="spurious-disposition"][value="out_of_scope_real"]'
        );
        if (radio) {
          radio.checked = true;
          radio.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
      case "s": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const radio = unit.querySelector(
          'input[data-role="unmatched-gold-disposition"][value="supplemental_data"]'
        );
        if (radio) {
          radio.checked = true;
          radio.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
      case "w": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const radio = unit.querySelector(
          'input[data-role="parsing-disposition"][value="gold_standard_error"], ' +
          'input[data-role="unmatched-gold-disposition"][value="gold_standard_error"]'
        );
        if (radio) {
          radio.checked = true;
          radio.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
      case "m": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const radio = unit.querySelector(
          'input[data-role="unmatched-gold-disposition"][value="matching_error"]'
        );
        if (radio) {
          radio.checked = true;
          radio.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
      case "d": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const radio = unit.querySelector(
          'input[data-role="unmatched-gold-disposition"][value="expected_difference"]'
        );
        if (radio) {
          radio.checked = true;
          radio.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
      case "t": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const radio = unit.querySelector(
          'input[data-role="unmatched-gold-disposition"][value="source_material_missing"]'
        );
        if (radio) {
          radio.checked = true;
          radio.dispatchEvent(new Event("change", { bubbles: true }));
        }
        break;
      }
      case "x": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const container = unit.closest(".legend-row, .gold-row") || unit;
        const details = container.querySelector("details.analysis-coords");
        if (details) details.open = !details.open;
        break;
      }
      case "e": {
        const unit = currentFocusedUnit();
        if (!unit) break;
        const note = unit.querySelector('textarea[data-role="review-note"]');
        if (note) {
          event.preventDefault();
          note.focus();
        }
        break;
      }
      default:
        break;
    }
  });

  focusPaper(0);
})();
</script>
"""


STYLE_BLOCK = """
  <style>
    :root {
      --bg: #f7f6f2;
      --panel: #ffffff;
      --ink: #1d2730;
      --line: #d8dde3;
    }
    body { margin: 0; padding: 1.25rem; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }
    header { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    .review-toolbar { margin-top: 0.85rem; padding: 0.75rem; border: 1px solid var(--line); border-radius: 8px; background: #fbfcfe; }
    .review-toolbar-actions { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
    .review-btn { border: 1px solid #1f5f94; color: #1f5f94; background: #fff; border-radius: 999px; padding: 0.28rem 0.65rem; font-size: 0.86rem; cursor: pointer; }
    .review-btn:hover { background: #eef6ff; }
    .review-btn-muted { border-color: #7a8692; color: #48555f; }
    .paper-card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 0.8rem; margin-bottom: 0.9rem; }
    .paper-card > summary { cursor: pointer; font-weight: 600; }
    .paper-body { margin-top: 0.6rem; }
    .paper-header { font-size: 0.92rem; margin-bottom: 0.6rem; }
    .paper-header a { margin: 0 0.15rem; }
    .paper-columns { display: grid; grid-template-columns: 2fr 1fr; gap: 1.1rem; align-items: start; }
    .paper-tables-col h3, .paper-gold-col h3 { font-size: 1rem; margin: 0.6rem 0 0.4rem 0; }
    .table-block { background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; padding: 0.6rem; margin-bottom: 0.6rem; }
    .table-html { overflow-x: auto; margin-top: 0.4rem; }
    .table-html table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
    .table-html th, .table-html td { border: 1px solid var(--line); padding: 0.3rem; }
    .legend-row, .gold-row { background: #fbfcfe; border: 1px solid var(--line); border-radius: 8px; padding: 0.5rem 0.6rem; margin-bottom: 0.5rem; }
    .legend-header { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem; }
    .badge-chip { display: inline-block; color: #fff; border-radius: 4px; padding: 0.05rem 0.4rem; font-size: 0.78rem; font-weight: 700; }
    .status-pill { font-size: 0.78rem; padding: 0.05rem 0.4rem; border-radius: 999px; background: #edf2f5; }
    .st-accepted { border-left: 4px solid #2f9e52; }
    .st-uncertain { border-left: 4px solid #d69700; }
    .st-unmatched { border-left: 4px solid #c94a4a; }
    .st-auto-only { border-left: 4px solid #7a5ac9; }
    .unit-review { margin-top: 0.5rem; padding: 0.5rem; border: 1px dashed var(--line); border-radius: 6px; }
    .unit-review.context-only { opacity: 0.6; }
    .unit-review.in-sample { border-style: solid; background: #fffefb; }
    .unit-review.focused { outline: 3px solid #2f6fed; box-shadow: 0 0 0 4px rgba(47,111,237,0.22); }
    .context-only-note { font-style: italic; }
    .paper-flag-row { margin: 0.4rem 0 0.6rem 0; padding: 0.4rem 0.6rem; background: #fff8e6; border: 1px solid #e8d8ad; border-radius: 6px; display: inline-block; }
    .paper-flag-label { font-size: 0.9rem; cursor: pointer; user-select: none; }
    .paper-flag-missed-table:checked ~ * { /* handled via JS class */ }
    .paper-card[data-missed-table="true"] > summary::after { content: " ⚠ missed table"; color: #b25000; font-weight: 700; font-size: 0.82rem; margin-left: 0.4rem; }
    .coord-list { white-space: pre-wrap; margin-top: 0.35rem; background: #fbfcfe; border: 1px solid var(--line); border-radius: 6px; padding: 0.35rem; font-family: "IBM Plex Mono", monospace; font-size: 0.82rem; max-height: 10rem; overflow-y: auto; }
    .disposition-group { margin: 0.5rem 0; padding: 0.5rem; background: #fff7e6; border: 1px solid #e8d8ad; border-radius: 6px; }
    .disposition-group label { display: block; margin: 0.2rem 0; }
    .review-reasons { margin: 0.4rem 0; }
    .matching-error-fields { margin: 0.5rem 0; padding: 0.5rem; border: 1px solid var(--line); border-radius: 6px; background: #f7fbff; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.6rem; }
    .matching-error-field { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.86rem; }
    .matching-error-field select { width: 100%; min-width: 0; }
    .parsing-reason-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.25rem 0.8rem; margin: 0.4rem 0; }
    .review-reason-option { display: flex; align-items: flex-start; gap: 0.35rem; font-size: 0.88rem; line-height: 1.3; }
    .review-reason-option input { flex: 0 0 auto; margin-top: 0.15rem; }
    .resource-note { font-size: 0.85rem; color: #3b4b5a; }
    .key-hint { font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; color: #6b7a89; }
    textarea[data-role="review-note"] { width: 100%; box-sizing: border-box; border: 1px solid var(--line); border-radius: 6px; padding: 0.45rem; font-family: inherit; font-size: 0.9rem; margin-top: 0.4rem; }
    a { color: #0e4f85; }
    #shortcut-help-overlay { position: fixed; top: 1rem; right: 1rem; z-index: 50; background: #1d2730; color: #fff; border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.85rem; box-shadow: 0 4px 18px rgba(0,0,0,0.3); white-space: pre-line; }
    @media (max-width: 720px) { .parsing-reason-grid, .matching-error-fields { grid-template-columns: 1fr; } }
  </style>
"""

SHORTCUT_HELP_TEXT = (
    "Keyboard shortcuts\n"
    "j / ↓ : next paper\n"
    "k / ↑ : previous paper\n"
    "n : next unit in paper\n"
    "p : previous unit in paper\n"
    "1-4 : toggle failure mode N\n"
    "c : parsing looks correct\n"
    "g : confirmed problem (error / missed / fabricated)\n"
    "u : unsure about parsing\n"
    "o : out of scope\n"
    "s : in supplemental data (not parsed)\n"
    "t : source table/text missing\n"
    "m : matching error\n"
    "d : expected source/curation difference\n"
    "w : gold standard is wrong\n"
    "x : toggle predicted coordinates\n"
    "e : edit note\n"
    "Escape : stop editing note\n"
    "? : toggle this help"
)


def render_report_html(
    papers: list[PaperView],
    manifest: dict[str, Any],
    seeded_store: dict[str, Any],
) -> str:
    total_units = 0
    for paper in papers:
        total_units += sum(1 for r in paper.legend_rows if r["in_sample"])
        total_units += sum(1 for r in paper.gold_rows if r["in_sample"])

    review_toolbar = (
        "<div class=\"review-toolbar\">"
        "<p><strong>Parser failure annotation workflow.</strong> Tables are shown in their original "
        "format with rows colored/badged by predicted analysis. Assess parsing for the in-sample "
        "units (solid border), then mark reasons when an error is present. Press <code>?</code> for "
        "keyboard shortcuts. "
        "Saved automatically in browser localStorage.</p>"
        f"<p id=\"review-progress\"><strong>Review progress:</strong> 0 / {total_units} completed</p>"
        "<div class=\"review-toolbar-actions\">"
        "<label>Reviewer: <input type=\"text\" id=\"review-reviewer\" placeholder=\"initials\"></label>"
        "<button type=\"button\" class=\"review-btn\" id=\"review-export-json\">Download Review JSON</button>"
        "<button type=\"button\" class=\"review-btn\" id=\"review-export-csv\">Download Review CSV</button>"
        "<button type=\"button\" class=\"review-btn review-btn-muted\" id=\"review-clear\">Clear Saved Review</button>"
        "</div>"
        "</div>"
    )

    seeded_store_json = json.dumps(seeded_store, ensure_ascii=True).replace("</", "<\\/")
    review_script = (
        REVIEW_SCRIPT_TEMPLATE.replace("__TOTAL_UNITS__", str(total_units))
        .replace("__MANIFEST_GENERATED_AT__", manifest["generated_at"])
        .replace("__SEEDED_STORE__", seeded_store_json)
    )

    papers_html = "".join(render_paper_card(p) for p in papers) or (
        "<p class=\"resource-note\">No papers matched the current sample.</p>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Parser Failure Annotation Report</title>
  {STYLE_BLOCK}
</head>
<body>
  <div id="shortcut-help-overlay" hidden>{escape(SHORTCUT_HELP_TEXT)}</div>
  <header>
    <h1>Parser Failure Annotation Report</h1>
    <p>Coordinate-separation correctness review: did the parser carve tables into the right analysis
    units with the right coordinates? Not expressivity, not facet extraction. Organized by paper --
    expand a paper to see its original source tables with the predicted split highlighted, plus any
    gold-standard analyses the parser missed entirely.</p>
    <p><strong>Seed:</strong> {manifest['seed']} | <strong>Accepted sample rate:</strong>
    {manifest['accepted_sample_rate']:.3f} | <strong>Failure sample rate:</strong> {manifest['failure_sample_rate']:.3f} |
    <strong>Spurious-candidate sample rate:</strong> {manifest['spurious_candidate_sample_rate']:.3f} |
    <strong>Projects included:</strong> {len(manifest['projects_included'])} |
    <strong>Projects skipped:</strong> {len(manifest['projects_skipped'])} |
    <strong>Papers in sample:</strong> {len(papers)} |
    <strong>Prior annotations restored:</strong>
    {manifest.get('existing_review', {}).get('seeded_entries', 0)}</p>
    {review_toolbar}
  </header>
  {papers_html}
  {review_script}
  {KEYBOARD_NAV_SCRIPT}
</body>
</html>
"""


def write_outputs(
    output_dir: Path,
    html: str,
    manifest: dict[str, Any],
    selections: list[rcpar.ProjectSelection],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parser_failure_annotation_report.html").write_text(html, encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    discovery_log = [
        {
            "project_name": sel.project_name,
            "status": sel.status,
            "reason": sel.reason,
            "selected_run_dir": str(sel.selected_run_dir) if sel.selected_run_dir else "",
            "selected_version": sel.selected_version,
        }
        for sel in sorted(selections, key=lambda s: s.project_name)
    ]
    (output_dir / "discovery_log.json").write_text(json.dumps(discovery_log, indent=2), encoding="utf-8")

    reviews_dir = output_dir / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    readme_path = reviews_dir / "README.json"
    if not readme_path.exists():
        readme_path.write_text(
            json.dumps(
                {
                    "usage": (
                        "Drop each reviewer's downloaded parser_failure_review.json export here "
                        "(rename to include reviewer initials if more than one person reviews) before "
                        "running build_parser_failure_contingency.py --reviews reviews/*.json"
                    )
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    projects_root = args.projects_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for option_name in (
        "accepted_sample_rate",
        "failure_sample_rate",
        "spurious_candidate_sample_rate",
    ):
        value = float(getattr(args, option_name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{option_name.replace('_', '-')} must be between 0 and 1")

    selections = discover_selections(projects_root, args.projects)
    selected = [s for s in selections if s.status == "selected" and s.selected_run_dir is not None]
    print(f"Discovered {len(selections)} project directories under {projects_root}")
    for sel in selections:
        if sel.status == "selected":
            print(f"[SELECT] {sel.project_name}: {sel.selected_run_dir}")
        else:
            print(f"[SKIP]   {sel.project_name}: {sel.reason}")

    run_contexts: dict[str, RunContext] = {}
    for sel in selected:
        assert sel.selected_run_dir is not None
        run_contexts[sel.project_name] = load_run_context(sel.project_name, sel.selected_run_dir)

    gold_units, auto_only_units = enumerate_units(run_contexts)
    print(f"Enumerated {len(gold_units)} gold-anchored units and {len(auto_only_units)} auto-only units")

    accepted_gold = [u for u in gold_units if u.sample_bucket == "accepted"]
    sampled_gold_statuses = {"unmatched"}
    if args.include_uncertain_gold:
        sampled_gold_statuses.add("uncertain")
    failure_gold = [u for u in gold_units if u.match_status in sampled_gold_statuses]

    sampled_accepted = stratified_sample_by_project(
        group_by_project(accepted_gold),
        rate=args.accepted_sample_rate,
        count_override=args.accepted_sample_count_per_project,
        min_per_project=args.accepted_sample_min_per_project,
        seed=args.seed,
    )
    sampled_gold_failure = stratified_sample_by_project(
        group_by_project(failure_gold),
        rate=args.failure_sample_rate,
        count_override=args.failure_sample_count_per_project,
        min_per_project=0,
        seed=args.seed,
    )
    sampled_auto_only = stratified_sample_by_project(
        group_by_project(auto_only_units),
        rate=args.spurious_candidate_sample_rate,
        count_override=args.spurious_candidate_sample_count_per_project,
        min_per_project=0,
        seed=args.seed,
    )

    print(
        f"Sampled {len(sampled_accepted)} accepted units, {len(sampled_gold_failure)} failing gold units, "
        f"{len(sampled_auto_only)} spurious-candidate units "
        f"(spurious rate={args.spurious_candidate_sample_rate:.3f})"
    )

    sampled_units = sampled_accepted + sampled_gold_failure + sampled_auto_only
    sampled_ids = {u.unit_id for u in sampled_units}

    manifest = build_manifest(args, selections, gold_units, auto_only_units, sampled_units)
    existing_review_path = (
        args.existing_review.expanduser().resolve()
        if args.existing_review is not None
        else output_dir / "reviews" / "parser_failure_review.json"
    )
    existing_review_payload: dict[str, Any] | None = None
    if args.load_existing_review and existing_review_path.exists():
        existing_review_payload = load_existing_review(existing_review_path)
    seeded_store, existing_review_stats = build_seeded_browser_store(
        existing_review_payload,
        sampled_units,
    )
    manifest["existing_review"] = {
        "enabled": bool(args.load_existing_review),
        "path": str(existing_review_path),
        "found": existing_review_payload is not None,
        **existing_review_stats,
    }
    print(
        "Existing review seed: "
        f"{existing_review_stats['seeded_entries']}/{existing_review_stats['source_entries']} "
        "annotations retained for units in the new sample"
    )

    papers = build_paper_views(run_contexts, gold_units, auto_only_units, sampled_ids)
    print(f"Built {len(papers)} paper views")

    html = render_report_html(papers, manifest, seeded_store)
    write_outputs(output_dir, html, manifest, selections)

    print(f"Wrote {output_dir / 'parser_failure_annotation_report.html'}")
    print(f"Wrote {output_dir / 'manifest.json'}")
    print(f"Wrote {output_dir / 'discovery_log.json'}")


if __name__ == "__main__":
    main()

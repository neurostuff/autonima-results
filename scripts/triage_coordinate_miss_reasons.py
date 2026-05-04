#!/usr/bin/env python3
"""Triage likely reasons coordinates were not extracted from full-text sources.

Builds a cross-project PMID set from study_classifications.json entries under
fulltext_with_coords.false_negatives_missing_analyses_or_coordinates, resolves
source records per (PMID, source), and assigns reasons using source-aware
heuristics.

Supported sources:
- ace_html
- pubget
- elsevier_output
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


DEFAULT_PROJECT_RUNS = [
    "projects/vbm_of_substance_use/v2",
    "projects/vbm_of_ptsd/v1",
    "projects/social/v3-search-all_pmids",
    "projects/decision_making/v2",
    "projects/cue_reactivity/v5",
]

REASON_ORDER = [
    "missed_in_main_text",
    "tables_present_no_coordinate_content",
    "tables_linked_not_fetched",
    "supplement_only_or_referenced",
    "incomplete_html",
    "unknown",
]

SOURCE_ORDER = ["ace_html", "pubget", "elsevier_output"]

SUPPLEMENT_CUE_RE = re.compile(
    r"\b("
    r"supplement(?:ary|al)?|"
    r"appendi(?:x|ces)|"
    r"supporting\s+information|"
    r"online[-\s]+only|"
    r"table\s*s\d+|"
    r"fig(?:ure)?\s*s\d+"
    r")\b",
    re.IGNORECASE,
)

COORD_KEYWORD_RE = re.compile(
    r"\b(mni|talairach|coordinate(?:s)?|cluster(?:s)?|voxel(?:s)?|peak(?:\s+voxel)?)\b",
    re.IGNORECASE,
)

TITLE_INCOMPLETE_RE = re.compile(
    r"(^\s*404\b)|(\bpage\s+not\s+found\b)|(\b404\s+not\s+found\b)",
    re.IGNORECASE,
)

BODY_INCOMPLETE_CUE_PATTERNS = [
    re.compile(r"\b404\s+not\s+found\b", re.IGNORECASE),
    re.compile(r"\bpage\s+not\s+found\b", re.IGNORECASE),
    re.compile(r"\bthe\s+page\s+you\s+requested\s+could\s+not\s+be\s+found\b", re.IGNORECASE),
    re.compile(r"\brequested\s+url\s+was\s+not\s+found\b", re.IGNORECASE),
    re.compile(r"\bwe\s+(could\s+not|can't)\s+find\s+the\s+page\b", re.IGNORECASE),
    re.compile(r"\barticle\s+not\s+found\b", re.IGNORECASE),
]

COORD_TRIPLET_RE = re.compile(
    r"(?<!\d)[+-]?\d{1,3}\s*[,;/|\t ]\s*[+-]?\d{1,3}\s*[,;/|\t ]\s*[+-]?\d{1,3}(?!\d)"
)

XYZ_HEADER_RE = re.compile(
    r"\b[xX]\b\s*[,/| ]?\s*\b[yY]\b\s*[,/| ]?\s*\b[zZ]\b"
)

TABLE_COORD_CONTEXT_RE = re.compile(
    r"\b(brodmann|ba\s*\d+|region|anatomical|talairach|mni|cluster|peak)\b",
    re.IGNORECASE,
)

TABLE_EXPANSION_HREF_RE = re.compile(
    r"(?:^|/)[Tt]\d+[A-Za-z0-9_-]*\.expansion\.html(?:$|[?#])"
)

TABLE_EXTERNAL_HREF_RE = re.compile(
    r"("
    r"/highwire/markup/\d+/expansion\b|"
    r"/article/.+/tables/\d+(?:$|[/?#])|"
    r"/tables/\d+(?:$|[/?#])"
    r")",
    re.IGNORECASE,
)

TABLE_PLACEHOLDER_TEXT_RE = re.compile(
    r"\bview\s+this\s+table\b.*\bin\s+this\s+window\b.*\bin\s+a\s+new\s+window\b",
    re.IGNORECASE,
)

TABLE_LINK_TEXT_RE = re.compile(
    r"\b(full\s*size\s*table|view\s+this\s+table|view\s+popup|view\s+inline)\b",
    re.IGNORECASE,
)

UNICODE_COORD_TRANSLATION = str.maketrans(
    {
        "−": "-",
        "–": "-",
        "—": "-",
        "‒": "-",
        "﹣": "-",
        "－": "-",
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
    }
)


@dataclass
class SourceRecord:
    pmid: str
    source: str
    projects: set[str]
    full_text_paths: set[str]
    pmcids: set[str]


@dataclass
class TriageRow:
    pmid: str
    source: str
    projects_for_source: list[str]
    source_primary_path: str | None
    article_path: str | None
    tables_path_or_glob: str | None
    table_files: list[str]
    source_found: bool
    source_status: str
    reason: str | None
    journal_or_container: str | None
    title: str | None
    table_count: int
    coord_table_count: int
    supplement_cue_count: int
    incomplete_cue_count: int
    html_path_count: int
    incomplete_cues: list[str]
    evidence_signals: list[str]
    evidence_snippet: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/zorro/repos/autonima-results"),
        help="Repo root containing projects/ and articles/.",
    )
    parser.add_argument(
        "--project-run",
        action="append",
        default=[],
        help=(
            "Project run path relative to repo root (repeatable). "
            "Defaults to the five planned runs."
        ),
    )
    parser.add_argument(
        "--html-root",
        type=Path,
        default=None,
        help="Override ACE HTML root. Defaults to <repo-root>/articles/ace_outputs/html.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output dir. Defaults to <repo-root>/reports/coordinate_miss_triage.",
    )
    return parser.parse_args()


def normalize_pmid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_pmcid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if text.startswith("PMC"):
        text = text[3:]
    if text.lower().startswith("pmcid_"):
        text = text[6:]
    if text.endswith(".0"):
        text = text[:-2]
    return text if text else None


def compact_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_coordinate_text(text: str) -> str:
    return text.translate(UNICODE_COORD_TRANSLATION)


def safe_read_text(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def find_keyword_hits(text: str, pattern: re.Pattern[str], max_hits: int = 6) -> list[str]:
    seen: list[str] = []
    seen_lower: set[str] = set()
    for match in pattern.finditer(text):
        token = match.group(0).strip()
        token_lower = token.lower()
        if token_lower not in seen_lower:
            seen.append(token)
            seen_lower.add(token_lower)
        if len(seen) >= max_hits:
            break
    return seen


def find_body_incomplete_hits(text: str, max_hits: int = 4) -> list[str]:
    hits: list[str] = []
    seen_lower: set[str] = set()
    for pattern in BODY_INCOMPLETE_CUE_PATTERNS:
        match = pattern.search(text)
        if match:
            token = compact_whitespace(match.group(0))
            token_lower = token.lower()
            if token_lower not in seen_lower:
                hits.append(token)
                seen_lower.add(token_lower)
        if len(hits) >= max_hits:
            break
    return hits


def snippet_around(text: str, term: str, window: int = 120) -> str:
    lowered = text.lower()
    idx = lowered.find(term.lower())
    if idx == -1:
        return compact_whitespace(text[: 2 * window])
    start = max(0, idx - window)
    end = min(len(text), idx + len(term) + window)
    return compact_whitespace(text[start:end])


def load_target_pmids(repo_root: Path, project_runs: list[str]) -> tuple[dict[str, set[str]], list[str]]:
    pmid_projects: dict[str, set[str]] = {}
    missing_classifications: list[str] = []

    for project_run in project_runs:
        class_path = repo_root / project_run / "evaluation" / "study_classifications.json"
        if not class_path.exists():
            missing_classifications.append(project_run)
            continue

        with class_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        pmids = (
            payload.get("fulltext_with_coords", {})
            .get("false_negatives_missing_analyses_or_coordinates", [])
        )
        for raw_pmid in pmids:
            pmid = normalize_pmid(raw_pmid)
            if pmid is None:
                continue
            pmid_projects.setdefault(pmid, set()).add(project_run)

    return pmid_projects, missing_classifications


def load_pubget_pmcid_map(run_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    metadata_path = run_root / "retrieval" / "pubget_data" / "metadata.csv"
    if not metadata_path.exists():
        return mapping

    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = normalize_pmid(row.get("pmid"))
            pmcid = normalize_pmcid(row.get("pmcid"))
            if pmid and pmcid:
                mapping[pmid] = pmcid
    return mapping


def classify_source(fulltext_available: Any, full_text_path: Any) -> str | None:
    if fulltext_available is True and (full_text_path is None or str(full_text_path).strip() == ""):
        return "pubget"

    if isinstance(full_text_path, str):
        ftp = full_text_path.strip()
        if not ftp:
            return None
        ftp_lower = ftp.lower()
        if "elsevier_output" in ftp_lower:
            return "elsevier_output"
        if ftp_lower.endswith(".html"):
            return "ace_html"

    return None


def build_source_records(
    repo_root: Path,
    project_runs: list[str],
    target_pmids: set[str],
) -> dict[tuple[str, str], SourceRecord]:
    records: dict[tuple[str, str], SourceRecord] = {}

    for project_run in project_runs:
        run_root = repo_root / project_run
        retrieval_path = run_root / "outputs" / "fulltext_retrieval_results.json"
        if not retrieval_path.exists():
            continue

        with retrieval_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        studies = payload.get("studies_with_fulltext", [])

        pubget_pmcid_map = load_pubget_pmcid_map(run_root)

        for study in studies:
            pmid = normalize_pmid(study.get("pmid"))
            if pmid is None or pmid not in target_pmids:
                continue

            source = classify_source(study.get("fulltext_available"), study.get("full_text_path"))
            if source is None:
                continue

            key = (pmid, source)
            if key not in records:
                records[key] = SourceRecord(
                    pmid=pmid,
                    source=source,
                    projects=set(),
                    full_text_paths=set(),
                    pmcids=set(),
                )

            record = records[key]
            record.projects.add(project_run)

            full_text_path = study.get("full_text_path")
            if isinstance(full_text_path, str) and full_text_path.strip():
                record.full_text_paths.add(full_text_path.strip())

            pmcid = normalize_pmcid(study.get("pmcid"))
            if pmcid:
                record.pmcids.add(pmcid)
            pmcid_meta = pubget_pmcid_map.get(pmid)
            if pmcid_meta:
                record.pmcids.add(pmcid_meta)

    return records


def resolve_ace_html_path(record: SourceRecord, html_root: Path) -> tuple[Path | None, int]:
    explicit_htmls = [
        Path(path) for path in record.full_text_paths
        if path.lower().endswith(".html")
    ]

    existing_explicit = [path for path in explicit_htmls if path.exists()]
    if existing_explicit:
        return sorted(existing_explicit)[0], len(existing_explicit)

    matches = sorted(html_root.glob(f"*/{record.pmid}.html"))
    if matches:
        return matches[0], len(matches)

    # If explicit path was supplied but missing, return first explicit as primary hint.
    if explicit_htmls:
        return sorted(explicit_htmls)[0], len(explicit_htmls)

    return None, 0


def resolve_pubget_paths(repo_root: Path, record: SourceRecord) -> tuple[Path | None, Path | None]:
    pmcid_candidates = sorted(record.pmcids)
    for project_run in sorted(record.projects):
        run_root = repo_root / project_run
        articles_base = run_root / "retrieval" / "pubget_data" / "articles"
        if not articles_base.exists():
            continue

        for pmcid in pmcid_candidates:
            normalized = f"pmcid_{pmcid}"
            candidates = sorted(articles_base.glob(f"*/{normalized}/article.xml"))
            if not candidates:
                continue
            article_path = candidates[0]
            tables_path = article_path.parent / "tables" / "tables.xml"
            return article_path, (tables_path if tables_path.exists() else None)

    return None, None


def resolve_elsevier_paths(record: SourceRecord) -> tuple[Path | None, str | None]:
    if not record.full_text_paths:
        return None, None

    full_text_path = Path(sorted(record.full_text_paths)[0])
    base_dir = full_text_path.parent
    article_path = base_dir / "article.xml"
    tables_glob = str(base_dir / "tables" / "*.csv")

    return (article_path if article_path.exists() else None), tables_glob


def analyze_ace_html(record: SourceRecord, html_path: Path | None, html_path_count: int) -> TriageRow:
    if html_path is None or not html_path.exists():
        return TriageRow(
            pmid=record.pmid,
            source="ace_html",
            projects_for_source=sorted(record.projects),
            source_primary_path=str(html_path) if html_path is not None else None,
            article_path=None,
            tables_path_or_glob=None,
            table_files=[],
            source_found=False,
            source_status="missing",
            reason=None,
            journal_or_container=None,
            title=None,
            table_count=0,
            coord_table_count=0,
            supplement_cue_count=0,
            incomplete_cue_count=0,
            html_path_count=html_path_count,
            incomplete_cues=[],
            evidence_signals=["html_missing"],
            evidence_snippet=None,
        )

    html_content = safe_read_text(html_path)
    soup = BeautifulSoup(html_content, "html.parser")

    title = compact_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body_text = compact_whitespace(soup.get_text(" ", strip=True))

    title_and_body = f"{title} {body_text}".strip()
    early_body_text = compact_whitespace(html_content[:8000])

    incomplete_cues: list[str] = []
    title_match = TITLE_INCOMPLETE_RE.search(title)
    if title_match:
        incomplete_cues.append(compact_whitespace(title_match.group(0)))
    incomplete_cues.extend(find_body_incomplete_hits(early_body_text))

    supplement_cues = find_keyword_hits(body_text, SUPPLEMENT_CUE_RE, max_hits=10)

    tables = soup.find_all("table")
    table_count = len(tables)
    coord_table_count = 0
    coord_keyword_table_count = 0
    xyz_header_table_count = 0
    max_triplets_in_table = 0
    coord_table_evidence = ""
    table_expansion_links: list[str] = []
    table_external_links: list[str] = []
    table_placeholder_count = 0

    for table in tables:
        table_text = compact_whitespace(normalize_coordinate_text(table.get_text(" ", strip=True)))
        if not table_text:
            continue

        coord_keyword = bool(COORD_KEYWORD_RE.search(table_text))
        xyz_header = bool(XYZ_HEADER_RE.search(table_text))
        triplets = len(COORD_TRIPLET_RE.findall(table_text))
        context = bool(TABLE_COORD_CONTEXT_RE.search(table_text))

        if coord_keyword:
            coord_keyword_table_count += 1
        if xyz_header:
            xyz_header_table_count += 1
        max_triplets_in_table = max(max_triplets_in_table, triplets)

        is_coord_table = (
            (coord_keyword and triplets >= 1)
            or (xyz_header and triplets >= 2)
            or (triplets >= 5 and context)
        )
        if is_coord_table:
            coord_table_count += 1
            if not coord_table_evidence:
                coord_table_evidence = table_text[:350]

    for link in soup.find_all("a", href=True):
        href = (link.get("href") or "").strip()
        if TABLE_EXPANSION_HREF_RE.search(href):
            if href not in table_expansion_links:
                table_expansion_links.append(href)
        href_lower = href.lower()
        link_text = compact_whitespace(link.get_text(" ", strip=True))
        is_external_table_link = bool(TABLE_EXTERNAL_HREF_RE.search(href_lower))
        if not is_external_table_link and TABLE_LINK_TEXT_RE.search(link_text):
            is_external_table_link = bool(href and not href.startswith("#"))
        if is_external_table_link and href not in table_external_links:
            table_external_links.append(href)

    for container in soup.find_all(["div", "section", "p", "li"]):
        container_text = compact_whitespace(
            normalize_coordinate_text(container.get_text(" ", strip=True))
        )
        if not container_text:
            continue
        if not TABLE_PLACEHOLDER_TEXT_RE.search(container_text):
            continue
        has_expansion_link = any(
            TABLE_EXPANSION_HREF_RE.search((a.get("href") or "").strip())
            for a in container.find_all("a", href=True)
        )
        if has_expansion_link:
            table_placeholder_count += 1

    evidence_signals: list[str] = []

    if coord_table_count > 0:
        reason = "missed_in_main_text"
        evidence_signals.extend(
            [
                f"coord_tables:{coord_table_count}",
                f"coord_keyword_tables:{coord_keyword_table_count}",
                f"xyz_header_tables:{xyz_header_table_count}",
                f"max_triplets_in_table:{max_triplets_in_table}",
            ]
        )
        evidence_snippet = coord_table_evidence
    elif (table_count == 0) and (table_expansion_links or table_external_links or table_placeholder_count > 0):
        reason = "tables_linked_not_fetched"
        evidence_signals.extend(
            [
                f"table_expansion_links:{len(table_expansion_links)}",
                f"table_external_links:{len(table_external_links)}",
                f"table_placeholders:{table_placeholder_count}",
            ]
        )
        if table_expansion_links:
            evidence_signals.append(f"example_link:{table_expansion_links[0]}")
        elif table_external_links:
            evidence_signals.append(f"example_link:{table_external_links[0]}")
        cue_term = "View this table"
        evidence_snippet = snippet_around(body_text, cue_term) if cue_term.lower() in body_text.lower() else compact_whitespace((title + " " + body_text)[:350])
    elif table_count > 0:
        reason = "tables_present_no_coordinate_content"
        evidence_signals.extend([f"table_count:{table_count}", "no_coord_table_detected"])
        evidence_snippet = compact_whitespace((title + " " + body_text)[:350])
    elif supplement_cues:
        reason = "supplement_only_or_referenced"
        evidence_signals.extend([f"supplement_cue:{cue}" for cue in supplement_cues[:5]])
        evidence_snippet = snippet_around(body_text, supplement_cues[0])
    elif incomplete_cues:
        reason = "incomplete_html"
        evidence_signals.extend([f"incomplete_cue:{cue}" for cue in incomplete_cues])
        evidence_snippet = snippet_around(title_and_body, incomplete_cues[0])
    else:
        reason = "unknown"
        evidence_signals.extend(["no_coord_table_detected", "no_supplement_cues_detected"])
        evidence_snippet = compact_whitespace((title + " " + body_text)[:350])

    return TriageRow(
        pmid=record.pmid,
        source="ace_html",
        projects_for_source=sorted(record.projects),
        source_primary_path=str(html_path),
        article_path=str(html_path),
        tables_path_or_glob=None,
        table_files=[],
        source_found=True,
        source_status="found",
        reason=reason,
        journal_or_container=html_path.parent.name,
        title=title,
        table_count=table_count,
        coord_table_count=coord_table_count,
        supplement_cue_count=len(supplement_cues),
        incomplete_cue_count=len(incomplete_cues),
        html_path_count=html_path_count,
        incomplete_cues=incomplete_cues,
        evidence_signals=evidence_signals,
        evidence_snippet=evidence_snippet,
    )


def analyze_pubget(record: SourceRecord, article_path: Path | None, tables_path: Path | None) -> TriageRow:
    article_text = safe_read_text(article_path)
    tables_text = normalize_coordinate_text(safe_read_text(tables_path))

    source_found = bool(article_text or tables_text)
    status = "found" if source_found else "missing"

    title = ""
    if article_text:
        title_match = re.search(r"<article-title[^>]*>(.*?)</article-title>", article_text, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = compact_whitespace(re.sub(r"<[^>]+>", " ", title_match.group(1)))

    supplement_cues = find_keyword_hits(article_text, SUPPLEMENT_CUE_RE, max_hits=10)

    extracted_table_count = len(re.findall(r"<extracted-table\b", tables_text, re.IGNORECASE))
    if extracted_table_count == 0 and tables_text:
        extracted_table_count = len(re.findall(r"<table-wrap\b", tables_text, re.IGNORECASE))

    coord_keyword_count = len(COORD_KEYWORD_RE.findall(tables_text))
    triplet_count = len(COORD_TRIPLET_RE.findall(tables_text))
    xyz_count = len(XYZ_HEADER_RE.findall(tables_text))

    coord_table_count = 0
    table_snippets: list[str] = []
    for block in re.findall(r"<extracted-table\b.*?</extracted-table>", tables_text, re.IGNORECASE | re.DOTALL):
        cleaned = compact_whitespace(normalize_coordinate_text(re.sub(r"<[^>]+>", " ", block)))
        kw = bool(COORD_KEYWORD_RE.search(cleaned))
        xyz = bool(XYZ_HEADER_RE.search(cleaned))
        trip = len(COORD_TRIPLET_RE.findall(cleaned))
        ctx = bool(TABLE_COORD_CONTEXT_RE.search(cleaned))
        is_coord = (kw and trip >= 1) or (xyz and trip >= 2) or (trip >= 5 and ctx)
        if is_coord:
            coord_table_count += 1
            if len(table_snippets) < 1:
                table_snippets.append(cleaned[:350])

    if source_found and coord_table_count > 0:
        reason = "missed_in_main_text"
        evidence_signals = [
            f"table_count:{extracted_table_count}",
            f"coord_tables:{coord_table_count}",
            f"coord_keywords:{coord_keyword_count}",
            f"triplets:{triplet_count}",
            f"xyz_markers:{xyz_count}",
        ]
        evidence_snippet = table_snippets[0] if table_snippets else compact_whitespace(tables_text[:350])
    elif source_found and extracted_table_count > 0:
        reason = "tables_present_no_coordinate_content"
        evidence_signals = [
            f"table_count:{extracted_table_count}",
            f"coord_keywords:{coord_keyword_count}",
            f"triplets:{triplet_count}",
            "no_coord_table_detected",
        ]
        evidence_snippet = compact_whitespace(re.sub(r"<[^>]+>", " ", tables_text[:350]))
    elif source_found and supplement_cues:
        reason = "supplement_only_or_referenced"
        evidence_signals = [f"supplement_cue:{cue}" for cue in supplement_cues[:5]]
        evidence_snippet = snippet_around(article_text, supplement_cues[0])
    elif source_found:
        reason = "unknown"
        evidence_signals = ["no_coord_table_detected", "no_supplement_cues_detected"]
        evidence_snippet = compact_whitespace(re.sub(r"<[^>]+>", " ", article_text[:350]))
    else:
        reason = None
        evidence_signals = ["source_missing"]
        evidence_snippet = None

    return TriageRow(
        pmid=record.pmid,
        source="pubget",
        projects_for_source=sorted(record.projects),
        source_primary_path=str(article_path) if article_path else (str(tables_path) if tables_path else None),
        article_path=str(article_path) if article_path else None,
        tables_path_or_glob=str(tables_path) if tables_path else None,
        table_files=[str(tables_path)] if tables_path else [],
        source_found=source_found,
        source_status=status,
        reason=reason,
        journal_or_container="pubget_data",
        title=title or None,
        table_count=extracted_table_count,
        coord_table_count=coord_table_count,
        supplement_cue_count=len(supplement_cues),
        incomplete_cue_count=0,
        html_path_count=0,
        incomplete_cues=[],
        evidence_signals=evidence_signals,
        evidence_snippet=evidence_snippet,
    )


def analyze_elsevier(record: SourceRecord, article_path: Path | None, tables_glob: str | None) -> TriageRow:
    article_text = safe_read_text(article_path)
    table_files = sorted(Path(path) for path in (glob_paths(tables_glob) if tables_glob else []))

    source_found = bool(article_text or table_files)
    status = "found" if source_found else "missing"

    title = ""
    if article_text:
        title_match = re.search(r"<dc:title>(.*?)</dc:title>", article_text, re.IGNORECASE | re.DOTALL)
        if not title_match:
            title_match = re.search(r"<article-title[^>]*>(.*?)</article-title>", article_text, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = compact_whitespace(re.sub(r"<[^>]+>", " ", title_match.group(1)))

    supplement_cues = find_keyword_hits(article_text, SUPPLEMENT_CUE_RE, max_hits=10)

    table_count = len(table_files)
    coord_table_count = 0
    coord_keyword_count = 0
    triplet_count = 0
    xyz_count = 0
    coord_table_evidence = ""

    for table_file in table_files:
        text = normalize_coordinate_text(safe_read_text(table_file))
        if not text:
            continue
        kw = len(COORD_KEYWORD_RE.findall(text))
        tr = len(COORD_TRIPLET_RE.findall(text))
        xyz = len(XYZ_HEADER_RE.findall(text))
        ctx = bool(TABLE_COORD_CONTEXT_RE.search(text))

        coord_keyword_count += kw
        triplet_count += tr
        xyz_count += xyz

        is_coord = (kw > 0 and tr >= 1) or (xyz > 0 and tr >= 2) or (tr >= 5 and ctx)
        if is_coord:
            coord_table_count += 1
            if not coord_table_evidence:
                coord_table_evidence = compact_whitespace(text[:350])

    if source_found and coord_table_count > 0:
        reason = "missed_in_main_text"
        evidence_signals = [
            f"table_count:{table_count}",
            f"coord_tables:{coord_table_count}",
            f"coord_keywords:{coord_keyword_count}",
            f"triplets:{triplet_count}",
            f"xyz_markers:{xyz_count}",
        ]
        evidence_snippet = coord_table_evidence
    elif source_found and table_count > 0:
        reason = "tables_present_no_coordinate_content"
        evidence_signals = [
            f"table_count:{table_count}",
            f"coord_keywords:{coord_keyword_count}",
            f"triplets:{triplet_count}",
            "no_coord_table_detected",
        ]
        evidence_snippet = compact_whitespace(article_text[:350])
    elif source_found and supplement_cues:
        reason = "supplement_only_or_referenced"
        evidence_signals = [f"supplement_cue:{cue}" for cue in supplement_cues[:5]]
        evidence_snippet = snippet_around(article_text, supplement_cues[0])
    elif source_found:
        reason = "unknown"
        evidence_signals = ["no_coord_table_detected", "no_supplement_cues_detected"]
        evidence_snippet = compact_whitespace(article_text[:350])
    else:
        reason = None
        evidence_signals = ["source_missing"]
        evidence_snippet = None

    return TriageRow(
        pmid=record.pmid,
        source="elsevier_output",
        projects_for_source=sorted(record.projects),
        source_primary_path=str(article_path) if article_path else tables_glob,
        article_path=str(article_path) if article_path else None,
        tables_path_or_glob=tables_glob,
        table_files=[str(path) for path in table_files],
        source_found=source_found,
        source_status=status,
        reason=reason,
        journal_or_container="elsevier_output",
        title=title or None,
        table_count=table_count,
        coord_table_count=coord_table_count,
        supplement_cue_count=len(supplement_cues),
        incomplete_cue_count=0,
        html_path_count=0,
        incomplete_cues=[],
        evidence_signals=evidence_signals,
        evidence_snippet=evidence_snippet,
    )


def glob_paths(pattern: str | None) -> list[str]:
    if not pattern:
        return []
    return sorted(glob.glob(pattern))


def triage_pmids(
    repo_root: Path,
    project_runs: list[str],
    html_root: Path,
) -> dict[str, Any]:
    pmid_projects, missing_classifications = load_target_pmids(repo_root, project_runs)
    target_pmids = set(pmid_projects.keys())

    source_records = build_source_records(repo_root, project_runs, target_pmids)

    rows: list[TriageRow] = []
    html_missing_rows: list[TriageRow] = []

    for key in sorted(source_records.keys()):
        record = source_records[key]

        if record.source == "ace_html":
            html_path, html_path_count = resolve_ace_html_path(record, html_root)
            row = analyze_ace_html(record, html_path, html_path_count)
            rows.append(row)
            if not row.source_found:
                html_missing_rows.append(row)

        elif record.source == "pubget":
            article_path, tables_path = resolve_pubget_paths(repo_root, record)
            rows.append(analyze_pubget(record, article_path, tables_path))

        elif record.source == "elsevier_output":
            article_path, tables_glob = resolve_elsevier_paths(record)
            rows.append(analyze_elsevier(record, article_path, tables_glob))

    found_rows = [row for row in rows if row.source_found]

    reason_counts_global = {
        reason: sum(1 for row in found_rows if row.reason == reason)
        for reason in REASON_ORDER
    }

    source_counts: dict[str, dict[str, int]] = {}
    for source in SOURCE_ORDER:
        source_rows = [row for row in rows if row.source == source]
        source_found_rows = [row for row in source_rows if row.source_found]
        source_counts[source] = {
            "total_rows": len(source_rows),
            "found_rows": len(source_found_rows),
            "missing_rows": len(source_rows) - len(source_found_rows),
        }
        for reason in REASON_ORDER:
            source_counts[source][reason] = sum(1 for row in source_found_rows if row.reason == reason)

    per_source_per_project_reason_counts: dict[str, dict[str, dict[str, int]]] = {}
    for source in SOURCE_ORDER:
        per_source_per_project_reason_counts[source] = {}
        for project in project_runs:
            per_source_per_project_reason_counts[source][project] = {reason: 0 for reason in REASON_ORDER}

    for row in rows:
        if not row.source_found or row.reason is None:
            continue
        for project in row.projects_for_source:
            per_source_per_project_reason_counts[row.source][project][row.reason] += 1

    # ACE-only coverage check requested in plan.
    ace_rows = [row for row in rows if row.source == "ace_html"]
    ace_found_count = sum(1 for row in ace_rows if row.source_found)
    ace_missing_count = len(ace_rows) - ace_found_count

    # Existing aggregate per-project counts (across all sources)
    per_project_reason_counts: dict[str, dict[str, int]] = {}
    for project in project_runs:
        per_project_reason_counts[project] = {reason: 0 for reason in REASON_ORDER}

    for row in rows:
        if not row.source_found or row.reason is None:
            continue
        for project in row.projects_for_source:
            per_project_reason_counts[project][row.reason] += 1

    return {
        "project_runs_requested": project_runs,
        "missing_classification_projects": missing_classifications,
        "total_unique_pmids": len(target_pmids),
        "total_rows": len(rows),
        "found_rows": len(found_rows),
        "reason_counts_global": reason_counts_global,
        "source_counts": source_counts,
        "per_project_reason_counts": per_project_reason_counts,
        "per_source_per_project_reason_counts": per_source_per_project_reason_counts,
        "ace_coverage": {
            "target_pmids": len(target_pmids),
            "ace_rows": len(ace_rows),
            "ace_found": ace_found_count,
            "ace_missing": ace_missing_count,
        },
        "html_missing_pmids": sorted({row.pmid for row in html_missing_rows}),
        "rows": rows,
    }


def write_outputs(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[TriageRow] = results["rows"]
    found_rows = [row for row in rows if row.source_found]

    csv_path = output_dir / "pmid_coordinate_miss_triage.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pmid",
                "source",
                "projects_for_source",
                "source_found",
                "source_status",
                "reason",
                "journal_or_container",
                "source_primary_path",
                "article_path",
                "tables_path_or_glob",
                "table_files",
                "title",
                "table_count",
                "coord_table_count",
                "supplement_cue_count",
                "incomplete_cue_count",
                "html_path_count",
                "incomplete_cues",
                "evidence_signals",
                "evidence_snippet",
            ],
        )
        writer.writeheader()
        for row in rows:
            d = asdict(row)
            d["projects_for_source"] = "|".join(row.projects_for_source)
            d["incomplete_cues"] = "|".join(row.incomplete_cues)
            d["evidence_signals"] = "|".join(row.evidence_signals)
            d["table_files"] = "|".join(row.table_files)
            writer.writerow(d)

    html_missing_path = output_dir / "html_missing_pmids.csv"
    with html_missing_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pmid"])
        writer.writeheader()
        for pmid in results["html_missing_pmids"]:
            writer.writerow({"pmid": pmid})

    summary_path = output_dir / "summary.json"
    summary_payload = {k: v for k, v in results.items() if k != "rows"}
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_html_path = output_dir / "coordinate_miss_triage_report.html"
    with report_html_path.open("w", encoding="utf-8") as f:
        f.write(
            "<!doctype html>\n"
            "<html lang='en'>\n"
            "<head>\n"
            "  <meta charset='utf-8'>\n"
            "  <meta name='viewport' content='width=device-width, initial-scale=1'>\n"
            "  <title>Coordinate-Miss Triage Report</title>\n"
            "  <style>\n"
            "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; line-height: 1.4; }\n"
            "    h1, h2, h3 { margin-top: 1.2em; }\n"
            "    table { border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 20px; }\n"
            "    th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; text-align: left; }\n"
            "    th { background: #f5f5f5; }\n"
            "    code { background: #f6f8fa; padding: 1px 4px; border-radius: 3px; }\n"
            "  </style>\n"
            "</head>\n"
            "<body>\n"
        )

        f.write("<h1>Coordinate-Miss Triage Report</h1>\n")
        f.write("<ul>\n")
        f.write(f"<li>Total unique PMIDs: <strong>{results['total_unique_pmids']}</strong></li>\n")
        f.write(f"<li>Total rows (PMID+source): <strong>{results['total_rows']}</strong></li>\n")
        f.write(f"<li>Found rows: <strong>{results['found_rows']}</strong></li>\n")
        f.write(
            f"<li>HTML missing PMIDs (ACE): <strong>{len(results['html_missing_pmids'])}</strong></li>\n"
        )
        if results["missing_classification_projects"]:
            missing_projects = ", ".join(results["missing_classification_projects"])
            f.write(f"<li>Missing classification project runs: <code>{html.escape(missing_projects)}</code></li>\n")
        f.write("</ul>\n")

        f.write("<h2>Aggregate Reason Counts (Found Rows)</h2>\n<ul>\n")
        for reason in REASON_ORDER:
            f.write(
                f"<li><code>{html.escape(reason)}</code>: <strong>{results['reason_counts_global'][reason]}</strong></li>\n"
            )
        f.write("</ul>\n")

        f.write("<h2>By-Source Stats</h2>\n")
        f.write("<table><thead><tr><th>Source</th><th>Total Rows</th><th>Found</th><th>Missing</th>")
        for reason in REASON_ORDER:
            f.write(f"<th>{html.escape(reason)}</th>")
        f.write("</tr></thead><tbody>\n")
        for source in SOURCE_ORDER:
            counts = results["source_counts"].get(source, {})
            f.write("<tr>")
            f.write(f"<td><code>{html.escape(source)}</code></td>")
            f.write(f"<td>{counts.get('total_rows', 0)}</td>")
            f.write(f"<td>{counts.get('found_rows', 0)}</td>")
            f.write(f"<td>{counts.get('missing_rows', 0)}</td>")
            for reason in REASON_ORDER:
                f.write(f"<td>{counts.get(reason, 0)}</td>")
            f.write("</tr>\n")
        f.write("</tbody></table>\n")

        ace_cov = results.get("ace_coverage", {})
        f.write("<h2>ACE Coverage Check</h2>\n<ul>\n")
        f.write(f"<li>target_pmids: <strong>{ace_cov.get('target_pmids', 0)}</strong></li>\n")
        f.write(f"<li>ace_found: <strong>{ace_cov.get('ace_found', 0)}</strong></li>\n")
        f.write(f"<li>ace_missing: <strong>{ace_cov.get('ace_missing', 0)}</strong></li>\n")
        f.write("</ul>\n")

        f.write("<h2>PMID+Source Results (Split by Source)</h2>\n")
        for source in SOURCE_ORDER:
            source_rows = [row for row in found_rows if row.source == source]
            f.write(f"<h3><code>{html.escape(source)}</code> ({len(source_rows)} found rows)</h3>\n")
            f.write(
                "<table><thead><tr>"
                "<th>PMID</th><th>Projects</th><th>Reason</th><th>Container</th>"
                "<th>Primary Link</th><th>Article</th><th>Tables</th><th>Signals</th><th>Evidence</th>"
                "</tr></thead><tbody>\n"
            )
            for row in sorted(source_rows, key=lambda r: (r.reason or "", r.pmid)):
                projects = "; ".join(row.projects_for_source)
                signals = ", ".join(row.evidence_signals[:8])
                evidence = compact_whitespace(row.evidence_snippet or "")
                evidence = evidence[:220] + ("..." if len(evidence) > 220 else "")

                def mk_link(path_str: str | None, label: str) -> str:
                    if not path_str:
                        return ""
                    try:
                        uri = Path(path_str).resolve().as_uri()
                    except Exception:
                        uri = path_str
                    return f"<a href='{html.escape(uri)}' target='_blank' rel='noopener'>{html.escape(label)}</a>"

                primary_link = mk_link(row.source_primary_path, "open")
                article_link = mk_link(row.article_path, "article")
                tables_link = ""
                if row.table_files:
                    links = [mk_link(path, f"table_{idx + 1}") for idx, path in enumerate(row.table_files[:4])]
                    tables_link = " | ".join(link for link in links if link)
                elif row.tables_path_or_glob:
                    tables_link = mk_link(row.tables_path_or_glob, "tables")

                f.write("<tr>")
                f.write(f"<td>{html.escape(row.pmid)}</td>")
                f.write(f"<td>{html.escape(projects)}</td>")
                f.write(f"<td><code>{html.escape(row.reason or '')}</code></td>")
                f.write(f"<td>{html.escape(row.journal_or_container or '')}</td>")
                f.write(f"<td>{primary_link}</td>")
                f.write(f"<td>{article_link}</td>")
                f.write(f"<td>{tables_link}</td>")
                f.write(f"<td>{html.escape(signals)}</td>")
                f.write(f"<td>{html.escape(evidence)}</td>")
                f.write("</tr>\n")
            f.write("</tbody></table>\n")

        f.write("<h2>HTML Missing PMIDs (ACE)</h2>\n<ul>\n")
        for pmid in results["html_missing_pmids"]:
            f.write(f"<li><code>{html.escape(pmid)}</code></li>\n")
        f.write("</ul>\n")

        f.write("</body>\n</html>\n")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote HTML missing CSV: {html_missing_path}")
    print(f"Wrote summary JSON: {summary_path}")
    print(f"Wrote HTML report: {report_html_path}")


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    project_runs = args.project_run if args.project_run else DEFAULT_PROJECT_RUNS
    html_root = (
        args.html_root.resolve()
        if args.html_root is not None
        else (repo_root / "articles" / "ace_outputs" / "html").resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (repo_root / "reports" / "coordinate_miss_triage").resolve()
    )

    results = triage_pmids(
        repo_root=repo_root,
        project_runs=project_runs,
        html_root=html_root,
    )
    write_outputs(results, output_dir)


if __name__ == "__main__":
    main()

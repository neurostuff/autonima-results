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

ANALYSIS_ID_RE = re.compile(r"^(?P<pmid>.+?)_analysis_(?P<index>\d+)$")


@dataclass
class Decision:
    include: bool
    reasoning: str
    analysis_id: str


def clean_text(value: str) -> str:
    return "".join(ch for ch in str(value) if ch >= " " or ch in "\n\t\r")


def parse_args() -> argparse.Namespace:
    default_project_output_dir = Path("../autonima-results/projects/social/coordinates/annotation-only")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-output-dir",
        type=Path,
        default=default_project_output_dir,
        help="Path to project output dir (e.g., .../annotation-only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated HTML reports. Defaults to sibling analysis/annotation_review_reports.",
    )
    parser.add_argument(
        "--match-input-dir",
        type=Path,
        default=None,
        help="Directory containing match_results_<annotation>.json files. Defaults to output dir.",
    )
    return parser.parse_args()


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = args.output_dir or (args.project_output_dir.parent / "analysis" / "annotation_review_reports")
    match_input_dir = args.match_input_dir or output_dir
    return output_dir, match_input_dir


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


def load_match_results_by_annotation(match_input_dir: Path) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for annotation_name in MANUAL_FILE_MAP:
        path = match_input_dir / f"match_results_{annotation_name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing match result file for {annotation_name}: {path}. "
                "Run run_fuzzy_analysis_matching.py first."
            )
        results[annotation_name] = load_json(path)
    return results


def build_manual_truth_from_match_results(match_results_by_annotation: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    manual_truth: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for annotation_name, match_results in match_results_by_annotation.items():
        for pmid, pmid_result in match_results.get("pmids", {}).items():
            manual_analyses = pmid_result.get("manual_analyses", [])
            accepted_indices = {
                int(entry["best_auto_index"])
                for entry in manual_analyses
                if entry.get("best_auto_index") is not None and entry.get("match_status") == "accepted"
            }

            status_counts = pmid_result.get("pmid_summary", {})
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
                "manual_missing_in_auto": bool(pmid_result.get("manual_missing_in_auto", False)),
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

    max_idx = len(parsed_names) - 1
    if decisions_by_idx:
        max_idx = max(max_idx, max(decisions_by_idx.keys()))

    analysis_rows: list[dict[str, Any]] = []
    for idx in range(max_idx + 1):
        name = parsed_names[idx] if idx < len(parsed_names) else f"analysis_{idx}"
        decision = decisions_by_idx.get(idx)
        analysis_rows.append(
            {
                "analysis_id": f"{pmid}_analysis_{idx}",
                "parsed_name": name,
                "model_include": None if decision is None else decision.include,
                "reasoning": "" if decision is None else decision.reasoning,
                "manual_include": idx in true_indices,
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


def classify_documents(
    annotation_name: str,
    parsed_analyses: dict[str, list[str]],
    model_decisions: dict[str, dict[str, dict[int, Decision]]],
    manual_truth: dict[str, dict[str, dict[str, Any]]],
    pmid_to_fulltext: dict[str, dict[str, str]],
    pmid_to_coord_tables: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    docs = {"Correct": [], "False Positive": [], "False Negative": []}
    ann_decisions = model_decisions.get(annotation_name, {})
    ann_truth = manual_truth.get(annotation_name, {})
    pmids = set(ann_decisions.keys()) | set(ann_truth.keys())

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

    tp = len(docs["Correct"])
    fp = len(docs["False Positive"])
    fn = len(docs["False Negative"])
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0

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
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
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

    manual_names_html = ""
    if doc["manual_names"]:
        joined = ", ".join(escape(x) for x in doc["manual_names"])
        manual_names_html = f"<p><strong>Manual analyses:</strong> {joined}</p>"

    rows_html = []
    for row in doc["analysis_rows"]:
        include_text = "Include" if row["model_include"] else "Exclude"
        if row["model_include"] is None:
            include_text = "No decision"
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
            f"<td>{escape(include_text)}</td>"
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
  {manual_names_html}
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
  <details class="inner-accordion">
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

    precision_str = f"{metrics['precision']:.3f}"
    recall_str = f"{metrics['recall']:.3f}"

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
    a {{ color: #0e4f85; }}
  </style>
</head>
<body>
  <header>
    <a id="top"></a>
    <h1>{escape(annotation_name)} report</h1>
    <p>Document-level buckets based on accepted manual-to-auto matches only. Thresholds come from precomputed matching artifacts.</p>
    <p><strong>TP/Correct:</strong> {metrics['tp']} |
       <strong>FP:</strong> {metrics['fp']} |
       <strong>FN:</strong> {metrics['fn']} |
       <strong>Precision:</strong> {precision_str} |
       <strong>Recall:</strong> {recall_str}</p>
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


def main() -> None:
    args = parse_args()
    output_dir, match_input_dir = resolve_dirs(args)

    annotation_results = args.project_output_dir / "outputs" / "annotation_results.json"
    coordinate_parsing_results = args.project_output_dir / "outputs" / "coordinate_parsing_results.json"
    retrieval_dir = args.project_output_dir / "retrieval" / "pubget_data"

    parsed_analyses = load_auto_parsed_names(coordinate_parsing_results)
    model_decisions = load_model_decisions(annotation_results)
    match_results_by_annotation = load_match_results_by_annotation(match_input_dir)
    manual_truth = build_manual_truth_from_match_results(match_results_by_annotation)
    pmid_to_fulltext, pmid_to_coord_tables = load_retrieval_context(retrieval_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for annotation_name in MANUAL_FILE_MAP:
        docs, metrics = classify_documents(
            annotation_name=annotation_name,
            parsed_analyses=parsed_analyses,
            model_decisions=model_decisions,
            manual_truth=manual_truth,
            pmid_to_fulltext=pmid_to_fulltext,
            pmid_to_coord_tables=pmid_to_coord_tables,
        )
        html = render_html(annotation_name, docs, metrics)
        output_path = output_dir / f"{annotation_name}.html"
        output_path.write_text(html, encoding="utf-8")

        print(
            f"Wrote {output_path} | "
            f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} "
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"missing_manual_pmids={len(metrics.get('missing_manual_pmids', []))}"
        )


if __name__ == "__main__":
    main()

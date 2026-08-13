#!/usr/bin/env python3
"""Reparse papers with confirmed parser failures using the current Autonima prompt.

The script rebuilds activation-table inputs from each run's configured local
sources or PubGet retrieval data, reparses only PMIDs marked ``parser_missed``
in a parser-review export, and atomically updates coordinate_parsing_results.
All studies not in the targeted review set are preserved byte-for-byte at the
JSON-object level.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_AUTONIMA_REPO = REPO_ROOT.parent / "autonima"
DEFAULT_REVIEW = (
    REPO_ROOT
    / "reports"
    / "parser_error_annotation"
    / "reviews"
    / "parser_failure_review_2.json"
)

RUNS_BY_PROJECT = {
    "cue_reactivity": "v5-annotation-only-gpt",
    "decision_making": "v2-annotation-only",
    "dementia": "v3-annotation-only",
    "problem_solving": "v1-annotation-only",
    "social": "v3-annotation-only",
    "vbm_of_ptsd": "v1-annotation-only",
    "vbm_of_substance_use": "v2-annotation-only-gpt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    parser.add_argument("--autonima-repo", type=Path, default=DEFAULT_AUTONIMA_REPO)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional coordinate-model override. Useful when the configured "
            "gateway requires a provider-qualified model name."
        ),
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=None,
        help="Optional project allow-list.",
    )
    parser.add_argument(
        "--pmids",
        nargs="+",
        default=None,
        help="Optional PMID allow-list within the reviewed parser-miss set.",
    )
    parser.add_argument(
        "--include-ingestion-misses",
        action="store_true",
        help=(
            "Also reparse source_material_missing review entries. Use this "
            "after changing table extraction or candidate gating."
        ),
    )
    parser.add_argument(
        "--rebuild-ace",
        action="store_true",
        help=(
            "Ignore configured legacy ACE exports and build a run-managed "
            "export from the source HTML."
        ),
    )
    parser.add_argument(
        "--no-summary-write",
        action="store_true",
        help="Do not replace the aggregate targeted-reparse summary.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def lookup(mapping: dict[Any, Any] | None, identifier: Any) -> Any:
    if not mapping:
        return None
    wanted = str(identifier)
    for key, value in mapping.items():
        if str(key) == wanted:
            return value
    return None


def reviewed_parser_miss_pmids(
    review_path: Path,
    *,
    include_ingestion_misses: bool = False,
) -> dict[str, set[str]]:
    payload = load_json(review_path)
    by_project: dict[str, set[str]] = defaultdict(set)
    dispositions = {"parser_missed"}
    if include_ingestion_misses:
        dispositions.add("source_material_missing")
    for entry in payload.get("entries", []) or []:
        disposition = str(entry.get("unmatched_gold_disposition") or "")
        if disposition not in dispositions:
            continue
        project = str(entry.get("project") or "")
        pmid = str(entry.get("pmid") or "")
        if project and pmid:
            by_project[project].add(pmid)
    return dict(by_project)


def resolve_config_path(project_dir: Path, run_name: str) -> Path:
    for suffix in (".yaml", ".yml"):
        candidate = project_dir / f"{run_name}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find config for {project_dir.name}/{run_name}"
    )


def pubget_pmid_to_pmcid(pubget_dir: Path) -> dict[str, str]:
    metadata_path = pubget_dir / "metadata.csv"
    if not metadata_path.exists():
        return {}
    out: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            pmid = str(row.get("pmid") or "").strip()
            pmcid = str(row.get("pmcid") or "").strip()
            if pmid and pmcid:
                out[pmid] = pmcid
    return out


def build_tables_for_project(
    *,
    config: dict[str, Any],
    run_dir: Path,
    target_pmids: set[str],
    rebuild_ace: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    from autonima.retrieval.utils import (
        _map_pmids_to_text,
        load_activation_table_map,
    )

    remaining = {int(pmid) for pmid in target_pmids if pmid.isdigit()}
    tables_by_pmid: dict[str, list[dict[str, Any]]] = {}
    source_by_pmid: dict[str, str] = {}

    retrieval = config.get("retrieval", {}) or {}
    for source_index, source_config in enumerate(
        retrieval.get("full_text_sources", []) or []
    ):
        if not remaining:
            break
        source_config = dict(source_config)
        for path_key in ("root_path", "processed_data_path"):
            configured = source_config.get(path_key)
            if not configured or Path(str(configured)).exists():
                continue
            marker = "/autonima-results/"
            configured_text = str(configured)
            if marker in configured_text:
                relative = configured_text.split(marker, 1)[1]
                rebased = REPO_ROOT / relative
                if rebased.exists():
                    source_config[path_key] = str(rebased)
                    continue
            if "ace_outputs" in configured_text:
                shared_ace = REPO_ROOT / "articles" / "ace_outputs"
                if path_key == "processed_data_path":
                    shared_ace = shared_ace / "processed"
                if shared_ace.exists():
                    source_config[path_key] = str(shared_ace)
        is_html_source = (
            source_config.get("pmid_source") == "file_name"
            and any(
                str(extension).lower() in {".html", ".htm"}
                for extension in source_config.get(
                    "allowed_extensions",
                    [],
                )
            )
        )
        if is_html_source and rebuild_ace:
            source_config.pop("processed_data_path", None)
        text_paths, _analyses, tables = _map_pmids_to_text(
            **source_config,
            pmids_to_include=remaining,
            generated_processed_data_path=(
                run_dir
                / "retrieval"
                / "ace"
                / f"targeted-source-{source_index + 1}"
            ),
        )
        found_table_pmids: set[int] = set()
        for pmid in sorted(remaining):
            if pmid not in text_paths:
                continue
            source_tables = lookup(tables, pmid) or []
            if not source_tables:
                continue
            tables_by_pmid[str(pmid)] = source_tables
            found_table_pmids.add(pmid)
            source_by_pmid[str(pmid)] = (
                str(source_config.get("source_name") or "")
                or str(source_config.get("root_path") or "")
                or f"full_text_source_{source_index + 1}"
            )
        remaining -= found_table_pmids

    pubget_candidates = [run_dir / "retrieval" / "pubget_data"]
    pubget_candidates.extend(
        sorted(
            (
                path
                for path in run_dir.parent.glob("*/retrieval/pubget_data")
                if path not in pubget_candidates
            ),
            reverse=True,
        )
    )
    for pubget_dir in pubget_candidates:
        if not pubget_dir.exists():
            continue
        pmid_to_pmcid = pubget_pmid_to_pmcid(pubget_dir)
        candidate_pmids = {
            pmid
            for pmid in target_pmids
            if pmid not in tables_by_pmid and pmid in pmid_to_pmcid
        }
        if not candidate_pmids:
            continue
        _analyses, pubget_tables = load_activation_table_map(
            processed_data_path=pubget_dir,
            filter_by_coordinates=True,
            identifier_key="pmcid",
            fallback_candidate_gate=True,
        )
        for pmid in candidate_pmids:
            pmcid = pmid_to_pmcid[pmid]
            source_tables = lookup(pubget_tables, pmcid)
            if source_tables:
                tables_by_pmid[pmid] = source_tables
                source_by_pmid[pmid] = f"pubget:{pubget_dir.parent.parent.name}"

    return tables_by_pmid, source_by_pmid


def deduplicate_tables(
    table_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in table_rows:
        key = (
            str(row.get("table_id") or ""),
            str(row.get("table_raw_path") or ""),
            str(row.get("table_data_path") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def parse_table_job(
    processor: Any,
    table_row: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    from autonima.models.types import ActivationTable

    table = ActivationTable(
        table_id=str(table_row.get("table_id") or ""),
        table_label=str(table_row.get("table_label") or ""),
        table_caption=table_row.get("table_caption"),
        table_foot=table_row.get("table_foot"),
        table_data_path=table_row.get("table_data_path"),
        table_raw_path=table_row.get("table_raw_path"),
        raw_table=table_row.get("raw_table"),
    )
    analyses = processor.process_single_table(table)
    return table.table_id, [
        analysis.model_dump(exclude={"parsed"})
        for analysis in analyses
    ]


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def reparse_project(
    *,
    project: str,
    run_dir: Path,
    config_path: Path,
    target_pmids: set[str],
    num_workers: int,
    model_override: str | None,
    rebuild_ace: bool,
) -> dict[str, Any]:
    from autonima.coordinates.processor import CoordinateProcessor
    from autonima.coordinates.prompts import (
        COORDINATE_PARSING_PROMPT_VERSION,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model = str(
        model_override
        or (config.get("parsing", {}) or {}).get(
            "coordinate_model",
            "gpt-4o-mini",
        )
    )
    tables_by_pmid, source_by_pmid = build_tables_for_project(
        config=config,
        run_dir=run_dir,
        target_pmids=target_pmids,
        rebuild_ace=rebuild_ace,
    )
    processor = CoordinateProcessor(model=model)

    jobs: list[tuple[str, dict[str, Any]]] = []
    for pmid in sorted(target_pmids):
        for row in deduplicate_tables(tables_by_pmid.get(pmid, [])):
            jobs.append((pmid, row))

    parsed_by_pmid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    table_counts: dict[str, int] = defaultdict(int)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, num_workers)
    ) as executor:
        future_map = {
            executor.submit(parse_table_job, processor, row): (pmid, row)
            for pmid, row in jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            pmid, row = future_map[future]
            table_id, analyses = future.result()
            table_counts[pmid] += 1
            parsed_by_pmid[pmid].extend(analyses)
            print(
                f"[TABLE] {project}:{pmid}:{table_id} "
                f"analyses={len(analyses)}"
            )

    output_path = run_dir / "outputs" / "coordinate_parsing_results.json"
    payload = load_json(output_path)
    studies = {
        str(study.get("pmid") or ""): study
        for study in payload.get("studies", []) or []
    }
    replaced_pmids: list[str] = []
    preserved_empty_pmids: list[str] = []
    missing_source_pmids: list[str] = []
    for pmid in sorted(target_pmids):
        if pmid not in tables_by_pmid:
            missing_source_pmids.append(pmid)
            continue
        analyses = parsed_by_pmid.get(pmid, [])
        if not analyses:
            preserved_empty_pmids.append(pmid)
            continue
        if pmid in studies:
            studies[pmid]["analyses"] = analyses
        else:
            new_study = {"pmid": pmid, "analyses": analyses}
            payload.setdefault("studies", []).append(new_study)
            studies[pmid] = new_study
        replaced_pmids.append(pmid)

    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": COORDINATE_PARSING_PROMPT_VERSION,
        "model": model,
        "review_target_pmids": sorted(target_pmids),
        "replaced_pmids": replaced_pmids,
        "preserved_pmids_with_empty_reparse": preserved_empty_pmids,
        "missing_source_pmids": missing_source_pmids,
        "tables_processed": sum(table_counts.values()),
        "table_counts_by_pmid": dict(sorted(table_counts.items())),
        "source_by_pmid": dict(sorted(source_by_pmid.items())),
    }
    previous_record = payload.get("targeted_reparse")
    history = payload.get("targeted_reparse_history", [])
    if not isinstance(history, list):
        history = []
    if isinstance(previous_record, dict):
        history.append(previous_record)
    if history:
        payload["targeted_reparse_history"] = history
    payload["targeted_reparse"] = run_record
    payload["timestamp"] = run_record["timestamp"]
    payload["cache_signature"] = {
        "schema_version": 1,
        "stage": "parsing",
        "prompt_version": COORDINATE_PARSING_PROMPT_VERSION,
    }
    atomic_write_json(output_path, payload)
    print(
        f"[PROJECT] {project}: targets={len(target_pmids)} "
        f"replaced={len(replaced_pmids)} "
        f"tables={sum(table_counts.values())} "
        f"missing_source={len(missing_source_pmids)} "
        f"empty={len(preserved_empty_pmids)}"
    )
    return {"project": project, **run_record}


def main() -> None:
    args = parse_args()
    review_path = args.review.expanduser().resolve()
    projects_root = args.projects_root.expanduser().resolve()
    autonima_repo = args.autonima_repo.expanduser().resolve()
    sys.path.insert(0, str(autonima_repo))

    by_project = reviewed_parser_miss_pmids(
        review_path,
        include_ingestion_misses=args.include_ingestion_misses,
    )
    allowed_projects = set(args.projects or by_project)
    allowed_pmids = set(args.pmids or [])
    summaries: list[dict[str, Any]] = []
    for project in sorted(by_project):
        if project not in allowed_projects:
            continue
        target_pmids = by_project[project]
        if allowed_pmids:
            target_pmids = target_pmids & allowed_pmids
        if not target_pmids:
            continue
        run_name = RUNS_BY_PROJECT.get(project)
        if not run_name:
            print(f"[SKIP] {project}: no configured benchmark run")
            continue
        project_dir = projects_root / project
        run_dir = project_dir / run_name
        config_path = resolve_config_path(project_dir, run_name)
        summaries.append(
            reparse_project(
                project=project,
                run_dir=run_dir,
                config_path=config_path,
                target_pmids=target_pmids,
                num_workers=args.num_workers,
                model_override=args.model,
                rebuild_ace=args.rebuild_ace,
            )
        )

    summary_path = (
        REPO_ROOT
        / "reports"
        / "parser_error_annotation"
        / "targeted_reparse_summary.json"
    )
    if not args.no_summary_write:
        atomic_write_json(
            summary_path,
            {
                "review": str(review_path),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "projects": summaries,
            },
        )
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge cached outputs from a later run into an earlier run.

Behavior:
- Updates `fulltext_screening_results.json` for PMIDs that passed abstract screening
  in the target run.
- Copies full-text decisions from source when target is missing a record, and
  optionally when target has `fulltext_incomplete`.
- For studies newly marked `included_fulltext` via this merge, copies matching
  coordinate parsing entries and annotation rows from source into target.

By default this is a dry run. Pass --write to persist changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FULLTEXT_FILE = "fulltext_screening_results.json"
ABSTRACT_FILE = "abstract_screening_results.json"
COORD_FILE = "coordinate_parsing_results.json"
ANNOT_FILE = "annotation_results.json"

VALID_SOURCE_FULLTEXT_DECISIONS = {"included_fulltext", "excluded_fulltext"}


@dataclass
class MergeStats:
    target_abstract_included: int = 0
    source_fulltext_candidates: int = 0
    fulltext_added: int = 0
    fulltext_replaced_incomplete: int = 0
    fulltext_changed_total: int = 0
    fulltext_changed_included: int = 0
    coord_added: int = 0
    coord_skipped_missing_in_source: int = 0
    annotation_added_rows: int = 0
    annotation_added_studies: int = 0


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_pmid(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("pmid"):
        text = text.split(":", 1)[-1].strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def index_by_study_id(items: list[dict[str, Any]]) -> dict[str, int]:
    by_id: dict[str, int] = {}
    for idx, item in enumerate(items):
        study_id = normalize_pmid(item.get("study_id"))
        if study_id:
            by_id[study_id] = idx
    return by_id


def index_by_pmid(items: list[dict[str, Any]]) -> dict[str, int]:
    by_id: dict[str, int] = {}
    for idx, item in enumerate(items):
        pmid = normalize_pmid(item.get("pmid"))
        if pmid:
            by_id[pmid] = idx
    return by_id


def collect_target_abstract_included(target_abstract: dict[str, Any]) -> set[str]:
    included: set[str] = set()
    for row in target_abstract.get("screening_results", []):
        if row.get("decision") == "included_abstract":
            pmid = normalize_pmid(row.get("study_id"))
            if pmid:
                included.add(pmid)
    return included


def merge_fulltext(
    target_fulltext_rows: list[dict[str, Any]],
    source_fulltext_rows: list[dict[str, Any]],
    target_abstract_included: set[str],
    replace_incomplete: bool,
    stats: MergeStats,
) -> set[str]:
    target_idx = index_by_study_id(target_fulltext_rows)
    source_idx = index_by_study_id(source_fulltext_rows)

    changed_included: set[str] = set()

    for pmid in sorted(target_abstract_included):
        source_row_idx = source_idx.get(pmid)
        if source_row_idx is None:
            continue

        source_row = source_fulltext_rows[source_row_idx]
        source_decision = source_row.get("decision")
        if source_decision not in VALID_SOURCE_FULLTEXT_DECISIONS:
            continue

        stats.source_fulltext_candidates += 1

        target_row_idx = target_idx.get(pmid)
        if target_row_idx is None:
            target_fulltext_rows.append(source_row)
            target_idx[pmid] = len(target_fulltext_rows) - 1
            stats.fulltext_added += 1
            stats.fulltext_changed_total += 1
            if source_decision == "included_fulltext":
                changed_included.add(pmid)
            continue

        target_row = target_fulltext_rows[target_row_idx]
        target_decision = target_row.get("decision")

        if replace_incomplete and target_decision == "fulltext_incomplete":
            target_fulltext_rows[target_row_idx] = source_row
            stats.fulltext_replaced_incomplete += 1
            stats.fulltext_changed_total += 1
            if source_decision == "included_fulltext":
                changed_included.add(pmid)

    stats.fulltext_changed_included = len(changed_included)
    return changed_included


def merge_coordinates(
    target_coord_studies: list[dict[str, Any]],
    source_coord_studies: list[dict[str, Any]],
    changed_included_pmids: set[str],
    stats: MergeStats,
) -> None:
    target_idx = index_by_pmid(target_coord_studies)
    source_idx = index_by_pmid(source_coord_studies)

    for pmid in sorted(changed_included_pmids):
        if pmid in target_idx:
            continue
        source_row_idx = source_idx.get(pmid)
        if source_row_idx is None:
            stats.coord_skipped_missing_in_source += 1
            continue
        target_coord_studies.append(source_coord_studies[source_row_idx])
        target_idx[pmid] = len(target_coord_studies) - 1
        stats.coord_added += 1


def annotation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_pmid(row.get("study_id")),
        str(row.get("analysis_id") or ""),
        str(row.get("annotation_name") or ""),
    )


def merge_annotations(
    target_annotations: list[dict[str, Any]],
    source_annotations: list[dict[str, Any]],
    changed_included_pmids: set[str],
    stats: MergeStats,
) -> None:
    existing = {annotation_key(row) for row in target_annotations}
    added_studies: set[str] = set()

    for row in source_annotations:
        study_id = normalize_pmid(row.get("study_id"))
        if study_id not in changed_included_pmids:
            continue

        key = annotation_key(row)
        if key in existing:
            continue

        target_annotations.append(row)
        existing.add(key)
        stats.annotation_added_rows += 1
        added_studies.add(study_id)

    stats.annotation_added_studies = len(added_studies)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge cached outputs from source run into target run, limited to PMIDs "
            "that passed abstract screening in target."
        )
    )
    parser.add_argument("--target-outputs", type=Path, required=True, help="Path to target outputs dir")
    parser.add_argument("--source-outputs", type=Path, required=True, help="Path to source outputs dir")
    parser.add_argument(
        "--replace-incomplete",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Replace target fulltext rows with decision=fulltext_incomplete when source "
            "has included/excluded fulltext decision (default: true)."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write updates in place. Without this flag, runs as dry-run.",
    )
    return parser.parse_args()


def validate_outputs_dir(path: Path) -> None:
    required = [ABSTRACT_FILE, FULLTEXT_FILE, COORD_FILE, ANNOT_FILE]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Missing required files in {path}: {names}")


def main() -> None:
    args = parse_args()
    target_dir = args.target_outputs.expanduser().resolve()
    source_dir = args.source_outputs.expanduser().resolve()

    validate_outputs_dir(target_dir)
    validate_outputs_dir(source_dir)

    target_abstract = load_json(target_dir / ABSTRACT_FILE)
    target_fulltext = load_json(target_dir / FULLTEXT_FILE)
    target_coords = load_json(target_dir / COORD_FILE)
    target_annotations = load_json(target_dir / ANNOT_FILE)

    source_fulltext = load_json(source_dir / FULLTEXT_FILE)
    source_coords = load_json(source_dir / COORD_FILE)
    source_annotations = load_json(source_dir / ANNOT_FILE)

    target_abstract_included = collect_target_abstract_included(target_abstract)

    stats = MergeStats(target_abstract_included=len(target_abstract_included))

    changed_included_pmids = merge_fulltext(
        target_fulltext_rows=target_fulltext["screening_results"],
        source_fulltext_rows=source_fulltext["screening_results"],
        target_abstract_included=target_abstract_included,
        replace_incomplete=args.replace_incomplete,
        stats=stats,
    )

    merge_coordinates(
        target_coord_studies=target_coords["studies"],
        source_coord_studies=source_coords["studies"],
        changed_included_pmids=changed_included_pmids,
        stats=stats,
    )

    merge_annotations(
        target_annotations=target_annotations,
        source_annotations=source_annotations,
        changed_included_pmids=changed_included_pmids,
        stats=stats,
    )

    print(f"Target abstract-included PMIDs: {stats.target_abstract_included}")
    print(f"Source fulltext candidate PMIDs: {stats.source_fulltext_candidates}")
    print(f"Fulltext rows added: {stats.fulltext_added}")
    print(f"Fulltext incomplete rows replaced: {stats.fulltext_replaced_incomplete}")
    print(f"Fulltext rows changed total: {stats.fulltext_changed_total}")
    print(f"Changed rows with included_fulltext: {stats.fulltext_changed_included}")
    print(f"Coordinate study rows added: {stats.coord_added}")
    print(f"Coordinate rows skipped (missing in source): {stats.coord_skipped_missing_in_source}")
    print(f"Annotation rows added: {stats.annotation_added_rows}")
    print(f"Annotation studies added: {stats.annotation_added_studies}")

    if args.write:
        dump_json(target_dir / FULLTEXT_FILE, target_fulltext)
        dump_json(target_dir / COORD_FILE, target_coords)
        dump_json(target_dir / ANNOT_FILE, target_annotations)
        print("Wrote updates to target output JSON files.")
    else:
        print("Dry run complete. Re-run with --write to persist changes.")


if __name__ == "__main__":
    main()

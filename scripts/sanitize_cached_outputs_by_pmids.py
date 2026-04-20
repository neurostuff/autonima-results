#!/usr/bin/env python3
"""Copy and filter cached output files to only keep selected PMIDs.

Usage:
  sanitize_cached_outputs_by_pmids.py \
    <source_outputs_dir> <pmids_file> <destination_outputs_dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

ABSTRACT_FILE = "abstract_screening_results.json"
ANNOTATION_FILE = "annotation_results.json"
COORDINATE_FILE = "coordinate_parsing_results.json"
FULLTEXT_FILE = "fulltext_screening_results.json"

FILES_TO_COPY = (
    ABSTRACT_FILE,
    ANNOTATION_FILE,
    COORDINATE_FILE,
    FULLTEXT_FILE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected cached outputs to a new directory and keep only records "
            "whose PMID is listed in a text file (one PMID per line)."
        )
    )
    parser.add_argument("source_outputs_dir", type=Path, help="Existing outputs directory")
    parser.add_argument("pmids_file", type=Path, help="Text file with one PMID per line")
    parser.add_argument("destination_outputs_dir", type=Path, help="Destination outputs directory")
    return parser.parse_args()


def normalize_pmid(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("pmid"):
        text = text.split(":", 1)[-1].strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def load_pmids(path: Path) -> set[str]:
    pmids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = normalize_pmid(line)
            if value:
                pmids.add(value)
    if not pmids:
        raise ValueError(f"No PMIDs found in {path}")
    return pmids


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def filter_screening_payload(payload: dict[str, Any], pmids: set[str]) -> tuple[dict[str, Any], int, int]:
    rows = payload.get("screening_results")
    if not isinstance(rows, list):
        raise ValueError("Expected key 'screening_results' with a list value.")
    original_count = len(rows)
    payload["screening_results"] = [
        row for row in rows if normalize_pmid(row.get("study_id")) in pmids
    ]
    return payload, original_count, len(payload["screening_results"])


def filter_annotation_payload(payload: list[dict[str, Any]], pmids: set[str]) -> tuple[list[dict[str, Any]], int, int]:
    original_count = len(payload)
    filtered = [row for row in payload if normalize_pmid(row.get("study_id")) in pmids]
    return filtered, original_count, len(filtered)


def filter_coordinate_payload(payload: dict[str, Any], pmids: set[str]) -> tuple[dict[str, Any], int, int]:
    studies = payload.get("studies")
    if not isinstance(studies, list):
        raise ValueError("Expected key 'studies' with a list value.")
    original_count = len(studies)
    payload["studies"] = [row for row in studies if normalize_pmid(row.get("pmid")) in pmids]
    return payload, original_count, len(payload["studies"])


def sanitize_file(path: Path, pmids: set[str]) -> tuple[int, int]:
    payload = load_json(path)

    if path.name in {ABSTRACT_FILE, FULLTEXT_FILE}:
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name}: expected JSON object payload.")
        filtered_payload, original_count, kept_count = filter_screening_payload(payload, pmids)
    elif path.name == ANNOTATION_FILE:
        if not isinstance(payload, list):
            raise ValueError(f"{path.name}: expected JSON array payload.")
        filtered_payload, original_count, kept_count = filter_annotation_payload(payload, pmids)
    elif path.name == COORDINATE_FILE:
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name}: expected JSON object payload.")
        filtered_payload, original_count, kept_count = filter_coordinate_payload(payload, pmids)
    else:
        raise ValueError(f"Unexpected file type: {path.name}")

    dump_json(path, filtered_payload)
    return original_count, kept_count


def validate_source_dir(path: Path) -> None:
    missing = [name for name in FILES_TO_COPY if not (path / name).exists()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(f"Missing required source file(s) in {path}: {names}")


def main() -> None:
    args = parse_args()

    source_dir = args.source_outputs_dir.expanduser().resolve()
    pmids_file = args.pmids_file.expanduser().resolve()
    destination_dir = args.destination_outputs_dir.expanduser().resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        raise NotADirectoryError(f"Source outputs directory does not exist: {source_dir}")
    if not pmids_file.exists() or not pmids_file.is_file():
        raise FileNotFoundError(f"PMIDs file does not exist: {pmids_file}")

    validate_source_dir(source_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    pmids = load_pmids(pmids_file)
    print(f"Loaded {len(pmids)} unique PMIDs from {pmids_file}")

    for file_name in FILES_TO_COPY:
        source_path = source_dir / file_name
        destination_path = destination_dir / file_name

        shutil.copy2(source_path, destination_path)
        before, after = sanitize_file(destination_path, pmids)
        print(f"{file_name}: kept {after}/{before} rows")

    print(f"Sanitized cache written to: {destination_dir}")


if __name__ == "__main__":
    main()

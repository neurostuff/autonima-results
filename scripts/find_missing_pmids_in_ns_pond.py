#!/usr/bin/env python3
"""Find PMID HTML paths that exist in a json-based hash-folder source.

The source layout is expected to be:
  <root_path>/<hash_id>/source/ace/<pmid>.html
  <root_path>/<hash_id>/identifiers.json

A folder only counts as an available hit if BOTH files exist and identifiers.json
contains a PMID at the configured key (default: "pmid").
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find paths for PMIDs from --pmids-file that are present in a "
            "json-based hash-folder source."
        )
    )
    parser.add_argument(
        "--pmids-file",
        type=Path,
        required=True,
        help="Text file containing one PMID per line.",
    )
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/data/alejandro/projects/ns-pond/data"),
        help="Root folder containing hash-id article directories.",
    )
    parser.add_argument(
        "--html-path-template",
        default="source/ace/{pmid}.html",
        help=(
            "Relative path to required HTML inside each hash folder. "
            "Use {pmid} placeholder for PMID-specific filenames."
        ),
    )
    parser.add_argument(
        "--json-filename",
        default="identifiers.json",
        help="JSON filename inside each hash folder.",
    )
    parser.add_argument(
        "--json-pmid-key",
        default="pmid",
        help="Key name to extract PMID from JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--archive-output",
        type=Path,
        default=None,
        help=(
            "Optional .tar.gz output path. When provided, all matched HTML files "
            "are copied into a single archive using their original filenames."
        ),
    )
    parser.add_argument(
        "--allow-nondigit-pmids",
        action="store_true",
        help="Keep non-numeric PMID strings instead of dropping them.",
    )
    return parser.parse_args()


def normalize_pmid(value: object, allow_nondigit: bool) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return text
    return text if allow_nondigit else None


def iter_key_values(data: object, key: str) -> Iterator[object]:
    if isinstance(data, dict):
        if key in data:
            yield data[key]
        for value in data.values():
            yield from iter_key_values(value, key)
    elif isinstance(data, list):
        for value in data:
            yield from iter_key_values(value, key)


def read_input_pmids(path: Path, allow_nondigit: bool) -> list[str]:
    pmids: list[str] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            pmid = normalize_pmid(raw, allow_nondigit=allow_nondigit)
            if pmid is None or pmid in seen:
                continue
            seen.add(pmid)
            pmids.append(pmid)

    return pmids


def resolve_html_path(folder: Path, html_path_template: str, pmid: str) -> Path:
    rel_path = html_path_template.format(pmid=pmid)
    return folder / rel_path


def build_found_pmid_path_map(
    root_path: Path,
    html_path_template: str,
    json_filename: str,
    json_pmid_key: str,
    target_pmids: set[str],
    allow_nondigit: bool,
) -> tuple[dict[str, str], dict[str, int]]:
    found_paths_by_pmid: dict[str, str] = {}
    stats = {
        "hash_dirs": 0,
        "with_required_html": 0,
        "with_json": 0,
        "json_read_errors": 0,
        "folders_with_pmid": 0,
    }

    for child in root_path.iterdir():
        if not child.is_dir():
            continue
        stats["hash_dirs"] += 1

        id_json = child / json_filename
        if not id_json.is_file():
            continue
        stats["with_json"] += 1

        try:
            data = json.loads(id_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stats["json_read_errors"] += 1
            continue

        found_any = False
        for value in iter_key_values(data, json_pmid_key):
            pmid = normalize_pmid(value, allow_nondigit=allow_nondigit)
            if pmid is None:
                continue
            found_any = True
            if pmid in target_pmids and pmid not in found_paths_by_pmid:
                html_path = resolve_html_path(child, html_path_template=html_path_template, pmid=pmid)
                if html_path.is_file():
                    stats["with_required_html"] += 1
                    found_paths_by_pmid[pmid] = str(html_path)

        if found_any:
            stats["folders_with_pmid"] += 1

    return found_paths_by_pmid, stats


def write_output(lines: list[str], output_path: Path | None) -> None:
    content = "\n".join(lines)
    if lines:
        content += "\n"

    if output_path is None:
        print(content, end="")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def write_archive(html_paths: list[str], archive_output: Path) -> None:
    archive_output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_output, mode="w:gz") as tar:
        for path_str in html_paths:
            src = Path(path_str)
            tar.add(src, arcname=src.name, recursive=False)


def main() -> None:
    args = parse_args()

    pmids_file = args.pmids_file.expanduser().resolve()
    root_path = args.root_path.expanduser().resolve()

    if not pmids_file.is_file():
        raise FileNotFoundError(f"PMID file not found: {pmids_file}")
    if not root_path.is_dir():
        raise FileNotFoundError(f"Root path not found or not a directory: {root_path}")

    input_pmids = read_input_pmids(pmids_file, allow_nondigit=args.allow_nondigit_pmids)
    input_pmid_set = set(input_pmids)

    found_paths_by_pmid, stats = build_found_pmid_path_map(
        root_path=root_path,
        html_path_template=args.html_path_template,
        json_filename=args.json_filename,
        json_pmid_key=args.json_pmid_key,
        target_pmids=input_pmid_set,
        allow_nondigit=args.allow_nondigit_pmids,
    )

    found_pmids = [pmid for pmid in input_pmids if pmid in found_paths_by_pmid]
    found_paths = [found_paths_by_pmid[pmid] for pmid in found_pmids]

    write_output(found_paths, args.output)
    if args.archive_output is not None:
        write_archive(found_paths, args.archive_output.expanduser().resolve())

    print(
        (
            "\n# summary\n"
            f"input_pmids={len(input_pmids)}\n"
            f"found_pmids={len(found_pmids)}\n"
            f"hash_dirs={stats['hash_dirs']}\n"
            f"with_required_html={stats['with_required_html']}\n"
            f"with_json={stats['with_json']}\n"
            f"json_read_errors={stats['json_read_errors']}\n"
            f"folders_with_pmid={stats['folders_with_pmid']}\n"
            f"unique_found_pmids_in_input={len(found_paths_by_pmid)}\n"
            f"archive_output={args.archive_output if args.archive_output is not None else ''}"
        ),
        file=__import__("sys").stderr,
    )


if __name__ == "__main__":
    main()

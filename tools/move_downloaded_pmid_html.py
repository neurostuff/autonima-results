#!/usr/bin/env python3
"""Move downloaded PMID HTML files into a target folder."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


PMID_RE = re.compile(r"^\d+$")
DOWNLOAD_NAME_RE = re.compile(r"^(?P<pmid>\d+)(?: \((?P<copy>\d+)\))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move HTML files named like PMID.html (or PMID (1).html) from a source "
            "download directory into a destination folder, filtered by PMID list."
        )
    )
    parser.add_argument("pmid_file", type=Path, help="Text file with one PMID per line.")
    parser.add_argument("destination_dir", type=Path, help="Destination folder for moved files.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("~/Downloads"),
        help="Download folder to scan (default: ~/Downloads).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without changing files.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    parser.add_argument(
        "--conflict",
        choices=["skip", "overwrite", "keep-both"],
        default="skip",
        help=(
            "Behavior when destination PMID.html exists: skip (default), overwrite, "
            "or keep-both (writes PMID__2.html, PMID__3.html, ...)."
        ),
    )
    return parser.parse_args()


def load_pmids(path: Path) -> set[str]:
    pmids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not PMID_RE.fullmatch(line):
                print(f"Skipping invalid PMID on line {line_number}: {line}")
                continue
            pmids.add(line)
    return pmids


def pmid_from_download_name(file_path: Path) -> str | None:
    suffix = file_path.suffix.lower()
    if suffix not in {".html", ".htm"}:
        return None
    match = DOWNLOAD_NAME_RE.fullmatch(file_path.stem)
    if not match:
        return None
    return match.group("pmid")


def next_keep_both_path(destination_dir: Path, pmid: str) -> Path:
    candidate = destination_dir / f"{pmid}.html"
    if not candidate.exists():
        return candidate
    n = 2
    while True:
        candidate = destination_dir / f"{pmid}__{n}.html"
        if not candidate.exists():
            return candidate
        n += 1


def resolve_destination_path(destination_dir: Path, pmid: str, conflict: str) -> tuple[Path, str]:
    canonical = destination_dir / f"{pmid}.html"
    if not canonical.exists():
        return canonical, "write"
    if conflict == "skip":
        return canonical, "skip_existing"
    if conflict == "overwrite":
        return canonical, "overwrite"
    return next_keep_both_path(destination_dir, pmid), "write"


def main() -> None:
    args = parse_args()
    pmid_file = args.pmid_file.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    destination_dir = args.destination_dir.expanduser().resolve()

    if not pmid_file.exists():
        raise FileNotFoundError(f"PMID file not found: {pmid_file}")
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    pmids = load_pmids(pmid_file)
    if not pmids:
        print("No valid PMIDs found.")
        return

    if not args.dry_run:
        destination_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted([p for p in source_dir.iterdir() if p.is_file()], key=lambda p: p.name)

    scanned = 0
    matched = 0
    moved_or_copied = 0
    skipped_not_in_list = 0
    skipped_conflict = 0

    operation = "COPY" if args.copy else "MOVE"
    print(f"Loaded {len(pmids)} PMIDs from {pmid_file}")
    print(f"Scanning source directory: {source_dir}")
    print(f"Destination directory: {destination_dir}")
    print(f"Mode: {operation}, conflict={args.conflict}, dry_run={args.dry_run}")

    for src in candidates:
        scanned += 1
        pmid = pmid_from_download_name(src)
        if pmid is None:
            continue

        matched += 1
        if pmid not in pmids:
            skipped_not_in_list += 1
            continue

        dest, action = resolve_destination_path(destination_dir, pmid=pmid, conflict=args.conflict)
        if action == "skip_existing":
            skipped_conflict += 1
            print(f"SKIP  {src} -> {dest} (already exists)")
            continue

        print(f"{operation}  {src} -> {dest}")
        moved_or_copied += 1
        if args.dry_run:
            continue

        if action == "overwrite" and dest.exists():
            dest.unlink()

        if args.copy:
            shutil.copy2(src, dest)
        else:
            shutil.move(str(src), str(dest))

    print("\nSummary:")
    print(f"- scanned files: {scanned}")
    print(f"- pmid-like html files seen: {matched}")
    print(f"- moved/copied: {moved_or_copied}")
    print(f"- skipped not in PMID list: {skipped_not_in_list}")
    print(f"- skipped due to existing destination: {skipped_conflict}")


if __name__ == "__main__":
    main()

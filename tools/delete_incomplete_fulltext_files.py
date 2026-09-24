#!/usr/bin/env python3
"""Delete files listed as incomplete in missing_fulltexts.csv.

Defaults to dry-run. Pass --write to actually delete files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete full_text_path entries where type=incomplete in missing_fulltexts.csv"
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to missing_fulltexts.csv")
    parser.add_argument(
        "--type",
        dest="row_type",
        default="incomplete",
        help="Row type to delete (default: incomplete)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually delete files. Without this flag, only print what would be deleted.",
    )
    parser.add_argument(
        "--prune-empty-dirs",
        action="store_true",
        help="After deleting files, remove empty parent directories up to filesystem root.",
    )
    return parser.parse_args()


def prune_empty_dirs(start: Path) -> int:
    removed = 0
    current = start
    while current.exists() and current.is_dir():
        try:
            current.rmdir()
        except OSError:
            break
        removed += 1
        if current.parent == current:
            break
        current = current.parent
    return removed


def main() -> None:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    candidates: list[Path] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV appears empty or missing header")

        required = {"type", "full_text_path"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

        for row in reader:
            if (row.get("type") or "").strip() != args.row_type:
                continue

            path_text = (row.get("full_text_path") or "").strip().strip('"')
            if not path_text:
                continue

            candidates.append(Path(path_text))

    total_candidates = len(candidates)
    unique_candidates = list(dict.fromkeys(candidates))

    existing = [p for p in unique_candidates if p.exists()]
    missing_paths = [p for p in unique_candidates if not p.exists()]

    print(f"CSV: {csv_path}")
    print(f"Row type filter: {args.row_type}")
    print(f"Candidate paths: {total_candidates} (unique: {len(unique_candidates)})")
    print(f"Existing paths: {len(existing)}")
    print(f"Already missing: {len(missing_paths)}")

    if not args.write:
        print("Dry run. Paths that would be deleted:")
        for p in existing:
            print(p)
        print("Re-run with --write to delete files.")
        return

    deleted = 0
    failed = 0
    pruned = 0

    for p in existing:
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                deleted += 1
                if args.prune_empty_dirs:
                    pruned += prune_empty_dirs(p.parent)
            else:
                print(f"Skipping non-file path: {p}")
        except Exception as exc:
            failed += 1
            print(f"Failed to delete {p}: {exc}")

    print(f"Deleted files: {deleted}")
    print(f"Failed deletions: {failed}")
    if args.prune_empty_dirs:
        print(f"Empty directories removed: {pruned}")


if __name__ == "__main__":
    main()

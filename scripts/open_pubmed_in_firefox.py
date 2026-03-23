#!/usr/bin/env python3
"""Open PubMed pages in Firefox from a file containing PMIDs (one per line)."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path


PMID_RE = re.compile(r"^\d+$")
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open each PMID from a text file in Firefox as a new window."
    )
    parser.add_argument("pmid_file", type=Path, help="Path to a text file with one PMID per line.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs instead of opening Firefox windows.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between launches (default: 0).",
    )
    return parser.parse_args()


def iter_pmids(path: Path) -> list[str]:
    pmids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not PMID_RE.fullmatch(line):
                print(f"Skipping invalid PMID on line {line_number}: {line}")
                continue
            pmids.append(line)
    return pmids


def main() -> None:
    args = parse_args()
    pmid_file = args.pmid_file.expanduser().resolve()

    if not pmid_file.exists():
        raise FileNotFoundError(f"PMID file not found: {pmid_file}")

    pmids = iter_pmids(pmid_file)
    if not pmids:
        print("No valid PMIDs found.")
        return

    print(f"Found {len(pmids)} valid PMIDs in {pmid_file}")

    for pmid in pmids:
        url = PUBMED_URL.format(pmid=pmid)
        if args.dry_run:
            print(url)
        else:
            subprocess.run(["firefox", "--new-window", url], check=False)
        if args.delay > 0:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()

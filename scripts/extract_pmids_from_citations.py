#!/usr/bin/env python3
"""Extract PMIDs from citation-export text files."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PMID_RE = re.compile(r"(?:PubMed\s+)?PMID\s*[:\-]\s*(\d+)", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PMIDs from a citation text file and print one PMID per line."
    )
    parser.add_argument("input_file", type=Path, help="Path to citation text file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path to write PMIDs. Defaults to stdout.",
    )
    return parser.parse_args()


def extract_pmids(text: str) -> list[str]:
    seen: set[str] = set()
    pmids: list[str] = []
    for match in PMID_RE.finditer(text):
        pmid = match.group(1)
        if pmid in seen:
            continue
        seen.add(pmid)
        pmids.append(pmid)
    return pmids


def main() -> int:
    args = parse_args()
    text = args.input_file.read_text(encoding="utf-8-sig")
    pmids = extract_pmids(text)
    if not pmids:
        print(f"No PMIDs found in: {args.input_file}", file=sys.stderr)
        return 1

    output_text = "\n".join(pmids) + "\n"
    if args.output is None:
        sys.stdout.write(output_text)
    else:
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Wrote {len(pmids)} PMIDs to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

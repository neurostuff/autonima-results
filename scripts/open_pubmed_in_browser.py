#!/usr/bin/env python3
"""Open PubMed pages in a browser from a file containing PMIDs (one per line)."""

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
        description="Open each PMID from a text file in Firefox or Chrome as a new window."
    )
    parser.add_argument("pmid_file", type=Path, help="Path to a text file with one PMID per line.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs instead of opening browser windows.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between batch launches (default: 0).",
    )
    parser.add_argument(
        "--max-at-once",
        type=int,
        default=10,
        help=(
            "Maximum number of PMIDs to process per batch before prompting to continue "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--browser",
        choices=["firefox", "chrome"],
        default="firefox",
        help="Browser to use for opening PubMed pages (default: firefox).",
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


def launch_browser(browser: str, urls: list[str]) -> None:
    if not urls:
        return

    if browser == "firefox":
        command = ["firefox", "--new-window", *urls]
    elif browser == "chrome":
        command = ["google-chrome", "--new-window", *urls]
    else:
        raise ValueError(f"Unsupported browser: {browser}")

    try:
        subprocess.run(command, check=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Browser executable not found for '{browser}'. "
            "For chrome, ensure 'google-chrome' is on PATH."
        ) from exc


def main() -> None:
    args = parse_args()
    pmid_file = args.pmid_file.expanduser().resolve()

    if args.max_at_once <= 0:
        raise ValueError("--max-at-once must be a positive integer.")

    if not pmid_file.exists():
        raise FileNotFoundError(f"PMID file not found: {pmid_file}")

    pmids = iter_pmids(pmid_file)
    if not pmids:
        print("No valid PMIDs found.")
        return

    # Open newest/higher PMIDs first by default.
    pmids = sorted(pmids, key=int, reverse=True)
    print(f"Found {len(pmids)} valid PMIDs in {pmid_file}")

    total = len(pmids)
    for start in range(0, total, args.max_at_once):
        batch = pmids[start : start + args.max_at_once]
        batch_end = start + len(batch)
        print(f"Processing PMIDs {start + 1}-{batch_end} of {total}...")
        batch_urls = [PUBMED_URL.format(pmid=pmid) for pmid in batch]

        if args.dry_run:
            for url in batch_urls:
                print(url)
        else:
            launch_browser(args.browser, batch_urls)

        if args.delay > 0:
            time.sleep(args.delay)

        if batch_end >= total:
            break

        next_count = min(args.max_at_once, total - batch_end)
        try:
            response = input(
                f"Press Enter to process the next {next_count} PMID(s), or 'q' to quit: "
            ).strip()
        except EOFError:
            response = "q"
        if response.lower() in {"q", "quit"}:
            print("Stopping before remaining PMIDs.")
            break


if __name__ == "__main__":
    main()

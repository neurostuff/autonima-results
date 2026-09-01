#!/usr/bin/env python3
"""Build a paired PDF/XML benchmark for evaluating PDF table extraction.

Choosing a PDF table extractor (Docling, pdfplumber, Table Transformer, a Rust
pdfium-render pipeline, ...) is currently an argument rather than a measurement. It does not
have to be: for any PMC open-access article we already hold as JATS XML, the XML tables are
ground truth for what a PDF extractor *should* recover from the same paper. Fetch the PDF,
run the extractor, score the coordinates it finds against the coordinates pubget already
parsed. No hand labelling anywhere.

This script does the first two thirds of that. It:

  1. scans pubget article directories for papers that have BOTH a coordinate table in the XML
     AND a redistributable Creative Commons licence (the licence check matters because the
     fetched PDFs get stored, and "in PMC" is not the same as "redistributable" -- roughly a
     tenth of PMC content is publisher-labelled open access with no actual grant);
  2. writes the ground-truth coordinates per article as JSON;
  3. optionally fetches the matching PDF -- and PMC's per-table and per-figure JPEGs -- from the
     PMC Cloud Service on AWS Open Data, which is the only bulk route that still works (see the
     PMC_CLOUD comment below for the four that do not).

Scoring is deliberately left to the caller, because it depends on the extractor under test.
Load ground_truth.json, produce the same shape from your extractor, and compare -- score_stub()
at the bottom shows the intended matching (exact triple match, order-insensitive, per article).

A coordinate row is any table row carrying at least three integer-valued cells in [-120, 120];
an article counts as having a coordinate table if some table has at least three such rows. This
is the same crude rule used elsewhere in this repo for corpus classification. It over-counts
demographic tables full of small integers and under-counts tables that report coordinates as
floats, so treat per-article ground truth as noisy and the aggregate as sound.

Usage:

    # what is available, no network
    python scripts/build_pdf_table_benchmark.py --survey

    # build ground truth for everything eligible
    python scripts/build_pdf_table_benchmark.py --out reports/pdf_table_benchmark

    # ...and fetch the PDFs (polite, sequential, resumable)
    python scripts/build_pdf_table_benchmark.py --out reports/pdf_table_benchmark --fetch-pdfs

    # ...also grabbing PMC's pre-cropped table and figure images
    python scripts/build_pdf_table_benchmark.py --out reports/pdf_table_benchmark \
        --fetch-pdfs --fetch-images --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent.parent

# Article dirs look like <root>/pubget_data/articles/<bucket>/pmcid_<id>/
ARTICLE_GLOBS = (
    "projects/*/*/retrieval/pubget_data/articles/*/pmcid_*",
    "experiments/*/*/retrieval/pubget_data/articles/*/pmcid_*",
)

# Only licences that actually grant redistribution. Deliberately excludes ND (no derivatives,
# which makes a reformatted or excerpted copy awkward), NC (binds if the service ever charges),
# publisher-specific "OpenAccess" labels, and articles with no <license> element at all -- the
# last being where NIH author manuscripts hide, which are free to read but not to redistribute.
REDISTRIBUTABLE = re.compile(
    r"creativecommons\.org/(?:licenses/(?:by|by-sa)/|publicdomain/zero/)", re.I
)

INT_CELL = re.compile(r"^-?\d{1,3}$")
PMCID_FROM_DIR = re.compile(r"pmcid_(\d+)$")

# The PMC Cloud Service on AWS Open Data. This is the only route that currently works for bulk
# PDF retrieval, and it is the sanctioned one:
#   - the OA Web Service (oa.fcgi) 404s on both the old and new PMC domains;
#   - the FTP dataset tree was emptied in August 2026 (see /pub/pmc/readme.txt) in favour of this;
#   - https://pmc.ncbi.nlm.nih.gov/articles/PMC*/pdf/ returns a "Preparing to download ..."
#     bot-mitigation interstitial rather than a PDF;
#   - Europe PMC's fullTextPDF endpoint returned nothing for 30/30 of our candidates.
# No credentials or login are required. Objects are keyed by PMCID and version, and each record
# carries the PDF, the JATS XML, plain text, and -- usefully -- one JPEG per figure AND per table.
PMC_CLOUD = "https://pmc-oa-opendata.s3.amazonaws.com/"
USER_AGENT = "autonima-results pdf-table-benchmark (research use; aid338@eid.utexas.edu)"

S3_KEY = re.compile(r"<Key>([^<]+)</Key>")
# Publishers name table images inconsistently: pone.0042394.t001.jpg, nihms916817t2.jpg, ...
TABLE_IMAGE = re.compile(r"[._]t\d+\.(?:jpg|jpeg|png)$", re.I)
FIGURE_IMAGE = re.compile(r"(?:g\d+|f\d+|Fig\d+)[^/]*\.(?:jpg|jpeg|png)$", re.I)


def is_coordinate_row(row: list[str]) -> bool:
    """True if the row carries at least three plausible stereotactic-coordinate cells."""
    ints = [c for c in row if INT_CELL.match(c.strip())]
    if len(ints) < 3:
        return False
    return sum(1 for c in ints if -120 <= int(c) <= 120) >= 3


def coordinate_rows(table_csv: Path) -> list[list[int]]:
    """Extract (x, y, z) triples from one pubget table CSV."""
    try:
        rows = list(csv.reader(table_csv.open(encoding="utf-8", errors="replace")))
    except (OSError, csv.Error):
        return []
    out: list[list[int]] = []
    for row in rows:
        if not is_coordinate_row(row):
            continue
        ints = [int(c) for c in (c.strip() for c in row) if INT_CELL.match(c)]
        triple = [v for v in ints if -120 <= v <= 120][:3]
        if len(triple) == 3:
            out.append(triple)
    return out


def licence_of(article_xml: Path) -> str | None:
    """Return the CC licence URL if the article carries a redistributable one."""
    try:
        text = article_xml.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = REDISTRIBUTABLE.search(text)
    return match.group(0) if match else None


def scan(roots: Iterable[str]) -> list[dict]:
    """Find every article that has both a coordinate table and a redistributable licence."""
    found: list[dict] = []
    for pattern in roots:
        for article_dir in sorted(REPO.glob(pattern)):
            xml = article_dir / "article.xml"
            if not xml.exists():
                continue

            coords: list[list[int]] = []
            for table in sorted(article_dir.glob("tables/table_*.csv")):
                rows = coordinate_rows(table)
                if len(rows) >= 3:
                    coords.extend(rows)
            if not coords:
                continue

            licence = licence_of(xml)
            pmcid_match = PMCID_FROM_DIR.search(article_dir.name)
            found.append(
                {
                    "pmcid": pmcid_match.group(1) if pmcid_match else None,
                    "article_dir": str(article_dir.relative_to(REPO)),
                    "licence": licence,
                    "redistributable": licence is not None,
                    "n_coordinates": len(coords),
                    "coordinates": coords,
                }
            )
    return found


def _get(url: str, timeout: int = 90) -> bytes | None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError):
        return None


def cloud_assets(pmcid: str) -> dict:
    """List one article's objects in the PMC Cloud bucket.

    Records are versioned (PMC6107443.1/, PMC6107443.2/, ...) and versions are not always
    equivalent -- a .2 is typically the publisher's typeset version replacing an author
    manuscript -- so the highest version wins.
    """
    body = _get(f"{PMC_CLOUD}?list-type=2&prefix=PMC{pmcid}.")
    if body is None:
        return {"pdf": None, "tables": [], "figures": []}
    keys = S3_KEY.findall(body.decode("utf-8", "replace"))
    pdfs = sorted(k for k in keys if k.lower().endswith(".pdf"))
    latest = pdfs[-1] if pdfs else None
    prefix = latest.rsplit("/", 1)[0] + "/" if latest else None
    scoped = [k for k in keys if prefix and k.startswith(prefix)]
    return {
        "pdf": latest,
        "tables": [k for k in scoped if TABLE_IMAGE.search(k)],
        "figures": [k for k in scoped if FIGURE_IMAGE.search(k)],
    }


def fetch_article(pmcid: str, out: Path, images: bool = False, pause: float = 0.34) -> str:
    """Fetch one article's PDF (and optionally its table/figure images). Returns a status."""
    pdf_dest = out / "pdfs" / f"PMC{pmcid}.pdf"
    if pdf_dest.exists() and pdf_dest.stat().st_size > 0 and not images:
        return "cached"

    assets = cloud_assets(pmcid)
    time.sleep(pause)
    if not assets["pdf"]:
        return "absent"

    if not (pdf_dest.exists() and pdf_dest.stat().st_size > 0):
        body = _get(PMC_CLOUD + assets["pdf"])
        time.sleep(pause)
        if body is None:
            return "fetch failed"
        if not body.startswith(b"%PDF"):
            return "not a pdf"
        pdf_dest.write_bytes(body)
        status = f"ok {len(body) // 1024}KB"
    else:
        status = "cached pdf"

    if images:
        got = 0
        for key in assets["tables"] + assets["figures"]:
            dest = out / "images" / f"PMC{pmcid}" / key.rsplit("/", 1)[-1]
            if dest.exists() and dest.stat().st_size > 0:
                continue
            blob = _get(PMC_CLOUD + key)
            time.sleep(pause)
            if blob:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
                got += 1
        status += f" +{got}img ({len(assets['tables'])}t/{len(assets['figures'])}f)"
    return status


def score_stub(ground_truth: list[list[int]], extracted: list[list[int]]) -> dict:
    """Reference scoring: exact triple match, order-insensitive, duplicates preserved.

    Not called by this script. It documents the comparison the benchmark is built for, so an
    extractor evaluation produces numbers comparable across tools.
    """
    from collections import Counter

    want, got = Counter(map(tuple, ground_truth)), Counter(map(tuple, extracted))
    hit = sum((want & got).values())
    return {
        "n_truth": sum(want.values()),
        "n_extracted": sum(got.values()),
        "n_matched": hit,
        "recall": hit / sum(want.values()) if want else None,
        "precision": hit / sum(got.values()) if got else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="directory for ground_truth.json, candidates.csv, pdfs/")
    parser.add_argument("--survey", action="store_true", help="report what is available and exit")
    parser.add_argument("--fetch-pdfs", action="store_true", help="download the matching PMC PDFs")
    parser.add_argument("--fetch-images", action="store_true",
                        help="also download PMC's per-table and per-figure JPEGs")
    parser.add_argument("--limit", type=int, help="cap the number of articles (for a quick pilot)")
    parser.add_argument("--pause", type=float, default=0.34, help="seconds between requests")
    args = parser.parse_args(argv)

    if not args.survey and not args.out:
        parser.error("one of --survey or --out is required")

    articles = scan(ARTICLE_GLOBS)
    eligible = [a for a in articles if a["redistributable"]]

    print(f"articles with a coordinate table       {len(articles)}")
    print(f"  of those, redistributable licence    {len(eligible)}")
    print(f"  total ground-truth coordinates       {sum(a['n_coordinates'] for a in eligible):,}")
    if articles:
        no_pmcid = sum(1 for a in eligible if not a["pmcid"])
        if no_pmcid:
            print(f"  (! {no_pmcid} eligible articles have no parseable PMCID and cannot be fetched)")

    if args.survey:
        return 0

    if args.limit:
        eligible = eligible[: args.limit]

    out = args.out
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    if args.fetch_images:
        (out / "images").mkdir(parents=True, exist_ok=True)

    (out / "ground_truth.json").write_text(json.dumps(eligible, indent=1), encoding="utf-8")
    with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pmcid", "n_coordinates", "licence", "article_dir"])
        for a in eligible:
            writer.writerow([a["pmcid"], a["n_coordinates"], a["licence"], a["article_dir"]])
    print(f"\nwrote {out/'ground_truth.json'} and {out/'candidates.csv'} ({len(eligible)} articles)")

    if not args.fetch_pdfs:
        print("re-run with --fetch-pdfs to download the matching PDFs")
        return 0

    print(f"\nfetching {len(eligible)} articles from the PMC Cloud Service...")
    tally: dict[str, int] = {}
    for i, article in enumerate(eligible, 1):
        pmcid = article["pmcid"]
        if not pmcid:
            tally["no pmcid"] = tally.get("no pmcid", 0) + 1
            continue
        status = fetch_article(pmcid, out, images=args.fetch_images, pause=args.pause)
        key = status.split()[0]
        tally[key] = tally.get(key, 0) + 1
        print(f"  [{i}/{len(eligible)}] PMC{pmcid}: {status}", flush=True)

    print("\nfetch summary:")
    for key, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {key:<12} {count}")
    print(f"\nPDFs in {out/'pdfs'}. Ground truth in {out/'ground_truth.json'}.")
    if args.fetch_images:
        print(f"Per-table and per-figure JPEGs in {out/'images'}.")
    print("Score an extractor with score_stub() in this file for comparable numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

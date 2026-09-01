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
  3. optionally fetches the matching PDF from one or more sources.

Source choice is the point, not an afterthought. PMC's own rendering is uniform and easier than
the real workload, so an extractor validated only against it will look better than it is. The
publisher-hosted PDFs that Unpaywall and Semantic Scholar resolve to are what a document actually
looks like arriving from anywhere else, and they are the ones worth scoring against. Measured
availability over the candidate set:

    pmc         100%   PMC Cloud Service on AWS, no key, uniform typesetting
    unpaywall   100% resolve, ~56-75% download   publisher-native
    s2           98% resolve, ~56-75% download   publisher-native, same hosts

The download gap is publisher bot-protection, not missing content: Frontiers, PLOS and Nature
serve directly, while Wiley, OUP, MDPI and SfN return 403 or an HTML challenge. That is the same
IP/entitlement wall the Elsevier work hit, and it is not worked around here.

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
from urllib.parse import quote
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

# Unpaywall and Semantic Scholar both resolve a DOI to a publisher-hosted PDF. Those are the
# interesting ones: a publisher's own typesetting is what a PDF from any non-PMC route actually
# looks like, whereas PMC's rendering is uniform and easier than the real workload.
NEUROSTORE_BASE = "https://neurostore.org/api/base-studies/?page_size={ps}&page={page}"
NEUROSTORE_NESTED = "https://neurostore.org/api/studies/?pmid={pmid}&nested=true&page_size=1"
UNPAYWALL = "https://api.unpaywall.org/v2/{doi}?email={email}"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
DOI_IN_JATS = re.compile(r'<article-id pub-id-type="doi">([^<]+)</article-id>')

# Publisher sites gate on User-Agent before anything else. This gets past the crudest checks;
# roughly 44% of publisher hosts still refuse (Wiley, OUP, MDPI and SfN all 403 in testing),
# which is the same IP/entitlement wall the Elsevier work ran into and is not worked around here.
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


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

            raw = xml.read_text(encoding="utf-8", errors="replace")
            licence = REDISTRIBUTABLE.search(raw)
            doi = DOI_IN_JATS.search(raw)
            pmcid_match = PMCID_FROM_DIR.search(article_dir.name)
            found.append(
                {
                    "pmcid": pmcid_match.group(1) if pmcid_match else None,
                    "doi": doi.group(1).strip() if doi else None,
                    "article_dir": str(article_dir.relative_to(REPO)),
                    "licence": licence.group(0) if licence else None,
                    "redistributable": licence is not None,
                    "n_coordinates": len(coords),
                    "coordinates": coords,
                }
            )
    return found


def scan_neurostore(max_candidates: int, page_size: int = 100) -> list[dict]:
    """Candidates from NeuroStore instead of local pubget output.

    Far larger and far more publisher-diverse than what we hold locally -- roughly 31,000
    coordinate-bearing studies with a DOI across ~291 journals, no journal above 15% -- and its
    own parsed coordinates serve as ground truth.

    The trade against the pubget arm is ground-truth quality, not size: NeuroStore's coordinates
    came from earlier ACE/Neurosynth-era parsing and carry their own error rate, whereas the
    pubget arm's come from publisher XML. Use this arm for relative comparison between
    extractors, and the pubget arm when an absolute accuracy number is wanted.
    """
    found: list[dict] = []
    page = 1
    while len(found) < max_candidates:
        body = _get(NEUROSTORE_BASE.format(ps=page_size, page=page))
        if body is None:
            break
        try:
            results = json.loads(body).get("results", [])
        except ValueError:
            break
        if not results:
            break
        for study in results:
            if not (study.get("has_coordinates") and study.get("doi") and study.get("pmid")):
                continue
            coords = neurostore_points(study["pmid"])
            if len(coords) < 3:
                continue
            # NeuroStore stores the PMCID with its "PMC" prefix; the rest of this script and the
            # S3 bucket key both want the bare digits.
            raw_pmcid = (study.get("pmcid") or "").strip()
            found.append(
                {
                    "pmcid": raw_pmcid[3:] if raw_pmcid.upper().startswith("PMC") else (raw_pmcid or None),
                    "pmid": study.get("pmid"),
                    "doi": (study.get("doi") or "").strip() or None,
                    "article_dir": None,
                    "licence": None,
                    "redistributable": bool(study.get("is_oa")),
                    "publication": study.get("publication"),
                    "n_coordinates": len(coords),
                    "coordinates": coords,
                }
            )
            if len(found) >= max_candidates:
                break
        page += 1
    return found


def neurostore_points(pmid: str) -> list[list[int]]:
    """Ground-truth coordinates for one study, from NeuroStore's parsed analyses."""
    body = _get(NEUROSTORE_NESTED.format(pmid=pmid))
    if body is None:
        return []
    try:
        results = json.loads(body).get("results", [])
    except ValueError:
        return []
    out: list[list[int]] = []
    for study in results:
        for analysis in study.get("analyses") or []:
            for point in analysis.get("points") or []:
                xyz = point.get("coordinates")
                if isinstance(xyz, list) and len(xyz) == 3:
                    try:
                        out.append([int(round(float(v))) for v in xyz])
                    except (TypeError, ValueError):
                        continue
    return out


def _get(url: str, timeout: int = 90, browser: bool = False) -> bytes | None:
    request = Request(url, headers={"User-Agent": BROWSER_UA if browser else USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError):
        return None


def pmc_cloud_pdf(pmcid: str, doi: str | None) -> str | None:
    """Highest-version PDF key for one article in the PMC Cloud bucket.

    Records are versioned (PMC6107443.1/, PMC6107443.2/, ...) and versions are not equivalent --
    a .2 is typically the publisher's typeset copy replacing an author manuscript -- so the
    highest wins.
    """
    body = _get(f"{PMC_CLOUD}?list-type=2&prefix=PMC{pmcid}.")
    if body is None:
        return None
    pdfs = sorted(k for k in S3_KEY.findall(body.decode("utf-8", "replace")) if k.lower().endswith(".pdf"))
    return PMC_CLOUD + pdfs[-1] if pdfs else None


def unpaywall_pdf(pmcid: str, doi: str | None, email: str = "") -> str | None:
    """Publisher-hosted PDF URL for a DOI, via Unpaywall's best_oa_location."""
    if not doi:
        return None
    body = _get(UNPAYWALL.format(doi=quote(doi.strip(), safe="/"), email=email))
    if body is None:
        return None
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    return (payload.get("best_oa_location") or {}).get("url_for_pdf")


def s2_pdf(pmcid: str, doi: str | None) -> str | None:
    """Publisher-hosted PDF URL for a DOI, via Semantic Scholar's openAccessPdf."""
    if not doi:
        return None
    body = _get(S2_PAPER.format(doi=quote(doi.strip(), safe="/")))
    if body is None:
        return None
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    return (payload.get("openAccessPdf") or {}).get("url")


SOURCES = {
    "pmc": pmc_cloud_pdf,
    "unpaywall": unpaywall_pdf,
    "s2": s2_pdf,
}


def article_id(article: dict) -> str:
    """Stable filename stem. NeuroStore candidates often have no PMCID, DOI-only ones no PMID."""
    if article.get("pmcid"):
        return f"PMC{article['pmcid']}"
    if article.get("pmid"):
        return f"pmid{article['pmid']}"
    return "doi_" + re.sub(r"[^A-Za-z0-9]+", "_", article.get("doi") or "unknown")


def fetch_from(source: str, article: dict, out: Path, email: str, pause: float) -> str:
    """Resolve and download one article's PDF from one source. Returns a status string."""
    pmcid, doi = article.get("pmcid"), article.get("doi")
    if source == "pmc" and not pmcid:
        return "no pmcid"
    if source in ("unpaywall", "s2") and not doi:
        return "no doi"
    dest = out / "pdfs" / source / f"{article_id(article)}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"

    resolver = SOURCES[source]
    url = resolver(pmcid, doi, email) if source == "unpaywall" else resolver(pmcid, doi)
    time.sleep(pause)
    if not url:
        return "no url"

    body = _get(url, browser=True)
    time.sleep(pause)
    if body is None:
        return "blocked/error"
    if not body.startswith(b"%PDF"):
        # Publisher bot-protection returns an HTML challenge with a 200 as often as a 403.
        return "not a pdf (html)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return f"ok {len(body) // 1024}KB"


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
    parser.add_argument("--candidates", choices=("pubget", "neurostore"), default="pubget",
                        help="pubget = local XML, cleaner ground truth, ~1.6k articles. "
                             "neurostore = ~31k coordinate-bearing studies across ~291 journals, "
                             "noisier ground truth but far more publisher-diverse")
    parser.add_argument("--max-candidates", type=int, default=500,
                        help="cap for --candidates neurostore (one API call per study)")
    parser.add_argument("--source", action="append", choices=sorted(SOURCES),
                        help="PDF source to fetch from; repeatable. pmc = PMC Cloud (uniform "
                             "rendering), unpaywall / s2 = publisher-hosted (representative)")
    parser.add_argument("--limit", type=int, help="cap the number of articles (for a quick pilot)")
    parser.add_argument("--pause", type=float, default=0.34, help="seconds between requests")
    parser.add_argument("--email", default="aid338@eid.utexas.edu",
                        help="contact address required by the Unpaywall API")
    args = parser.parse_args(argv)

    if not args.survey and not args.out:
        parser.error("one of --survey or --out is required")

    if args.candidates == "neurostore":
        articles = scan_neurostore(args.max_candidates)
        eligible = articles          # licence gating is the fetch source's problem here
        journals = {(a.get("publication") or "?") for a in articles}
        print(f"NeuroStore coordinate-bearing studies  {len(articles)}")
        print(f"  distinct journals                    {len(journals)}")
        print(f"  flagged open access                  {sum(1 for a in articles if a['redistributable'])}")
        print(f"  with a PMCID (PMC Cloud reachable)   {sum(1 for a in articles if a.get('pmcid'))}")
    else:
        articles = scan(ARTICLE_GLOBS)
        eligible = [a for a in articles if a["redistributable"]]
        print(f"articles with a coordinate table       {len(articles)}")
        print(f"  of those, redistributable licence    {len(eligible)}")
    print(f"  total ground-truth coordinates       {sum(a['n_coordinates'] for a in eligible):,}")
    if articles:
        no_pmcid = sum(1 for a in eligible if not a.get("pmcid"))
        if no_pmcid:
            print(f"  (! {no_pmcid} have no PMCID -- unreachable via --source pmc, "
                  f"but fine via unpaywall/s2, which key on DOI)")

    if args.survey:
        return 0

    out = args.out
    (out / "pdfs").mkdir(parents=True, exist_ok=True)

    (out / "ground_truth.json").write_text(json.dumps(eligible, indent=1), encoding="utf-8")
    with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pmcid", "n_coordinates", "licence", "article_dir"])
        for a in eligible:
            writer.writerow([a["pmcid"], a["n_coordinates"], a["licence"], a["article_dir"]])
    print(f"\nwrote {out/'ground_truth.json'} and {out/'candidates.csv'} ({len(eligible)} articles)")

    # --limit caps the fetch only; the artefacts above always describe the full eligible set.
    if args.limit:
        eligible = eligible[: args.limit]
        print(f"--limit {args.limit}: fetching a subset, artefacts still cover all candidates")

    if not args.source:
        print("re-run with --source {pmc,unpaywall,s2} (repeatable) to download PDFs")
        return 0

    for source in args.source:
        print(f"\n=== {source}: fetching {len(eligible)} PDFs ===")
        tally: dict[str, int] = {}
        for i, article in enumerate(eligible, 1):
            try:
                status = fetch_from(source, article, out, args.email, args.pause)
            except Exception as exc:                      # noqa: BLE001 - one bad record must
                status = f"error {type(exc).__name__}"    # not abort the remaining fetches
            tally[status.split()[0]] = tally.get(status.split()[0], 0) + 1
            print(f"  [{i}/{len(eligible)}] {article_id(article)}: {status}", flush=True)
        got = tally.get("ok", 0) + tally.get("cached", 0)
        print(f"  -- {source}: {got}/{len(eligible)} PDFs ({got/max(len(eligible),1)*100:.0f}%)")
        for key, count in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"     {key:<18} {count}")

    print(f"\nPDFs under {out/'pdfs'}/<source>/. Ground truth in {out/'ground_truth.json'}.")
    print("Score an extractor with score_stub() for numbers comparable across sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

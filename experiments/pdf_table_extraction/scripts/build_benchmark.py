#!/usr/bin/env python3
"""Build a paired PDF/ground-truth benchmark for evaluating PDF table extraction.

Choosing a PDF table extractor (Docling, pdfplumber, Table Transformer, a Rust pdfium-render
pipeline, ...) is currently an argument rather than a measurement. It does not have to be: for any
article whose coordinates we already hold in parsed form, those coordinates are ground truth for
what an extractor *should* recover from the same paper's PDF. No hand labelling anywhere.

Two candidate pools, trading ground-truth quality against size and diversity:

  --candidates pubget      ~1,640 articles from local pubget output. Ground truth is publisher
                           JATS XML, so it is clean -- but the pool is PMC-OA by construction and
                           therefore publisher-skewed before anything is fetched.
  --candidates neurostore  ~31,000 coordinate-bearing studies across ~291 journals, no journal
                           above 15%. Ground truth is NeuroStore's own parsed coordinates, which
                           came from ACE/Neurosynth-era parsing and carry their own error rate.
                           Use this arm for comparing extractors, pubget for absolute accuracy.

Source choice is a validity control, not a convenience. PMC's rendering is uniform and easier than
the real workload, so an extractor validated only against it will look better than it is. Measured
yield with --source auto, which routes each DOI to the API entitled to serve it:

    elsevier    100%   10.1016; needs ELSEVIER_API_KEY and an entitled IP
    pmc          73%   free and keyless, but uniform typesetting
    wiley        50%   10.1002 / 10.1111; needs WILEY_TDM_TOKEN
    s2            7%   free; mostly blocked by publisher bot-protection
    unpaywall     -    same hosts as s2, used as a further fallback

auto tries the entitled publisher API first, then s2, unpaywall and finally PMC, so a paper is
only lost when every route fails. Publisher-native renderings are preferred over PMC's because
they are what a document actually looks like arriving from anywhere else.

Scoring is deliberately left to the caller, since it depends on the extractor under test. Load
ground_truth.json, produce the same shape, and compare -- score_stub() at the bottom shows the
intended matching (exact triple match, order-insensitive, per article).

For the pubget arm, a coordinate row is any table row carrying at least three integer cells in
[-120, 120], and an article qualifies if some table has at least three such rows. This is the same
crude rule used elsewhere in this repo. It over-counts demographic tables full of small integers
and misses coordinates reported as floats, so treat per-article ground truth as noisy and the
aggregate as sound.

Usage:

    # what is available, no network
    python scripts/build_pdf_table_benchmark.py --survey
    python scripts/build_pdf_table_benchmark.py --candidates neurostore --survey

    # build the diverse pool and fetch each paper from whichever API can serve it
    python scripts/build_pdf_table_benchmark.py --candidates neurostore \
        --max-candidates 2000 --out reports/pdf_bench --source auto

    # or pin one source, to compare renderings of the same papers
    python scripts/build_pdf_table_benchmark.py --out reports/pdf_bench \
        --source pmc --source elsevier
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]

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
ELSEVIER_ARTICLE = "https://api.elsevier.com/content/article/doi/{doi}"
WILEY_TDM = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}"

# Which publisher API can serve a DOI, by registrant prefix. Routing on the prefix rather than the
# journal name is exact -- the prefix IS the publisher's Crossref registrant -- and lets --source
# auto send each paper to the one API that can actually serve it.
DOI_ROUTE = {
    "10.1016": "elsevier",
    "10.1002": "wiley",
    "10.1111": "wiley",
}
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
    """Find every article that has both a coordinate table and a redistributable licence.

    Deduplicated by PMCID: the same paper is downloaded independently by every project and run
    that screened it in, so the raw directory count is roughly 3.6x the number of distinct
    papers (1,640 directories -> 451 articles). Counting directories would badly overstate the
    benchmark size.
    """
    found: list[dict] = []
    seen: set[str] = set()
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

            pmcid_match = PMCID_FROM_DIR.search(article_dir.name)
            if pmcid_match and pmcid_match.group(1) in seen:
                continue

            raw = xml.read_text(encoding="utf-8", errors="replace")
            licence = REDISTRIBUTABLE.search(raw)
            doi = DOI_IN_JATS.search(raw)
            if pmcid_match:
                seen.add(pmcid_match.group(1))
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


def scan_neurostore(max_candidates: int, page_size: int = 100, seed: int = 0) -> list[dict]:
    """Candidates from NeuroStore instead of local pubget output.

    Far larger and far more publisher-diverse than what we hold locally -- roughly 31,000
    coordinate-bearing studies with a DOI across ~291 journals, no journal above 15% -- and its
    own parsed coordinates serve as ground truth.

    The trade against the pubget arm is ground-truth quality, not size: NeuroStore's coordinates
    came from earlier ACE/Neurosynth-era parsing and carry their own error rate, whereas the
    pubget arm's come from publisher XML. Use this arm for relative comparison between
    extractors, and the pubget arm when an absolute accuracy number is wanted.
    """
    total = (_json(NEUROSTORE_BASE.format(ps=1, page=1)) or {}).get("metadata", {}).get(
        "total_count", 0
    )
    if not total:
        return []

    # Random pages, not a walk from page 1. The corpus is strongly clustered by ingestion order:
    # coordinate-bearing density measured 14/100 on page 1, 2/100 on page 50 and 98/100 on page
    # 400, and median publication year swings between 2012 and 2024 across those pages. Walking
    # sequentially would both waste most calls on sparse pages and hand back a sample skewed by
    # whenever a study happened to be ingested.
    rng = random.Random(seed)
    pages = list(range(1, max(total // page_size, 1) + 1))
    rng.shuffle(pages)

    # ...and only a handful per page. Studies are clustered *within* a page too -- a page is
    # roughly one ingestion batch, so taking a whole one yields a few journals repeated. Capping
    # per page spreads the sample across many more batches for the same number of studies, and
    # the extra listing calls are cheap next to the one-per-study coordinate lookup.
    per_page = max(1, page_size // 20)

    found: list[dict] = []
    for page in pages:
        if len(found) >= max_candidates:
            break
        payload = _json(NEUROSTORE_BASE.format(ps=page_size, page=page))
        results = (payload or {}).get("results", [])
        rng.shuffle(results)
        taken = 0
        for study in results:
            if taken >= per_page:
                break
            if not (study.get("has_coordinates") and study.get("doi") and study.get("pmid")):
                continue
            coords = neurostore_points(study["pmid"])
            if len(coords) < 3:
                continue
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
                    "year": study.get("year"),
                    "n_coordinates": len(coords),
                    "coordinates": coords,
                }
            )
            taken += 1
            if len(found) >= max_candidates:
                break
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


def merge_pools(pubget: list[dict], neurostore: list[dict]) -> list[dict]:
    """Union the two candidate pools, deduplicated, preferring the cleaner ground truth.

    A paper can appear in both: pubget holds it because we downloaded its PMC XML, NeuroStore
    because someone parsed it years ago. When that happens the pubget record wins, because its
    coordinates come from publisher JATS rather than ACE-era parsing -- but the NeuroStore record
    still contributes its DOI, which pubget records often lack and which every publisher API
    needs.

    Matching is by DOI first (the only identifier all sources agree on), then PMCID, then PMID.
    """
    def keys(rec: dict) -> list[str]:
        out = []
        if rec.get("doi"):
            out.append("doi:" + rec["doi"].strip().lower())
        if rec.get("pmcid"):
            out.append("pmcid:" + str(rec["pmcid"]).lstrip("PMCpmc"))
        if rec.get("pmid"):
            out.append("pmid:" + str(rec["pmid"]))
        return out

    merged: dict[str, dict] = {}
    index: dict[str, str] = {}
    for rec, origin in [(r, "pubget") for r in pubget] + [(r, "neurostore") for r in neurostore]:
        ks = keys(rec)
        hit = next((index[k] for k in ks if k in index), None)
        if hit is None:
            rec = dict(rec, ground_truth_source=origin)
            merged[ks[0]] = rec
            for k in ks:
                index[k] = ks[0]
            continue
        # Already seen. Keep the better ground truth, but backfill missing identifiers.
        existing = merged[hit]
        for field in ("doi", "pmid", "pmcid"):
            if not existing.get(field) and rec.get(field):
                existing[field] = rec[field]
        if existing["ground_truth_source"] == "neurostore" and origin == "pubget":
            existing.update(
                coordinates=rec["coordinates"],
                n_coordinates=rec["n_coordinates"],
                article_dir=rec.get("article_dir"),
                licence=rec.get("licence"),
                ground_truth_source="pubget",
            )
        for k in ks:
            index.setdefault(k, hit)
    return list(merged.values())


def _json(url: str) -> dict | None:
    body = _get(url)
    if body is None:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def _get(url: str, timeout: int = 90, browser: bool = False,
         headers: dict | None = None) -> bytes | None:
    hdrs = {"User-Agent": BROWSER_UA if browser else USER_AGENT}
    hdrs.update(headers or {})
    request = Request(url, headers=hdrs)
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


def unpaywall_pdf(article: dict, cfg: dict) -> bytes | None:
    """Publisher-hosted PDF via Unpaywall's best_oa_location."""
    doi = (article.get("doi") or "").strip()
    if not doi:
        return None
    payload = _json(UNPAYWALL.format(doi=quote(doi, safe="/"), email=cfg.get("email", "")))
    url = (payload or {}).get("best_oa_location", {}) or {}
    return _get(url.get("url_for_pdf"), browser=True) if url.get("url_for_pdf") else None


def s2_pdf(article: dict, cfg: dict) -> bytes | None:
    """Publisher-hosted PDF via Semantic Scholar's openAccessPdf."""
    doi = (article.get("doi") or "").strip()
    if not doi:
        return None
    payload = _json(S2_PAPER.format(doi=quote(doi, safe="/")))
    url = ((payload or {}).get("openAccessPdf") or {}).get("url")
    return _get(url, browser=True) if url else None


def elsevier_pdf(article: dict, cfg: dict) -> bytes | None:
    """Elsevier Article Retrieval API. Entitlement is IP-based, so this only works from a
    network the subscription covers -- the same constraint measured throughout this project."""
    doi, key = (article.get("doi") or "").strip(), cfg.get("elsevier")
    if not (doi and key):
        return None
    headers = {"X-ELS-APIKey": key, "Accept": "application/pdf"}
    if cfg.get("elsevier_insttoken"):
        headers["X-ELS-Insttoken"] = cfg["elsevier_insttoken"]
    return _get(ELSEVIER_ARTICLE.format(doi=quote(doi, safe="/")), headers=headers)


def wiley_pdf(article: dict, cfg: dict) -> bytes | None:
    """Wiley Text and Data Mining API. Needs a client token; also IP-gated."""
    doi, token = (article.get("doi") or "").strip(), cfg.get("wiley")
    if not (doi and token):
        return None
    return _get(WILEY_TDM.format(doi=quote(doi, safe="/")),
                headers={"Wiley-TDM-Client-Token": token})


def pmc_pdf(article: dict, cfg: dict) -> bytes | None:
    """PMC Cloud Service. Free and keyless, but PMC's uniform typesetting is easier than the
    publisher-native renderings above, so results from this arm flatter an extractor."""
    url = pmc_cloud_pdf(article.get("pmcid"), None)
    return _get(url) if url else None


SOURCES = {
    "pmc": pmc_pdf,
    "unpaywall": unpaywall_pdf,
    "s2": s2_pdf,
    "elsevier": elsevier_pdf,
    "wiley": wiley_pdf,
}


def article_id(article: dict) -> str:
    """Stable filename stem. NeuroStore candidates often have no PMCID, DOI-only ones no PMID."""
    if article.get("pmcid"):
        return f"PMC{article['pmcid']}"
    if article.get("pmid"):
        return f"pmid{article['pmid']}"
    return "doi_" + re.sub(r"[^A-Za-z0-9]+", "_", article.get("doi") or "unknown")


def load_credentials(email: str) -> dict:
    """Read publisher credentials from the environment, falling back to ~/.keys/.

    Values are read but never logged. Elsevier's file ships as `export VAR=...` lines; the Wiley
    TDM token is a bare UUID on one line.
    """
    cfg = {
        "email": email,
        "elsevier": os.environ.get("ELSEVIER_API_KEY"),
        "elsevier_insttoken": os.environ.get("ELSEVIER_INSTTOKEN"),
        "wiley": os.environ.get("WILEY_TDM_TOKEN"),
    }
    keys = Path.home() / ".keys"
    if (keys / "elsevier.key").exists():
        # The file holds several publishers' keys, so the variable name has to be matched
        # specifically -- a generic *API_KEY* pattern silently picks up SPRINGER_API_KEY from a
        # later line and hands Elsevier's endpoint the wrong credential.
        text = (keys / "elsevier.key").read_text()
        for field, pattern in (
            ("elsevier", r"ELSEVIER_API_?KEY\s*=\s*[\"']?([^\"'\s]+)"),
            ("elsevier_insttoken", r"ELSEVIER_INSTTOKEN\s*=\s*[\"']?([^\"'\s]+)"),
            ("springer", r"SPRINGER_API_?KEY\s*=\s*[\"']?([^\"'\s]+)"),
        ):
            if not cfg.get(field):
                found = re.search(pattern, text, re.I)
                if found:
                    cfg[field] = found.group(1)
    if not cfg["wiley"] and (keys / "wiley.key").exists():
        raw = (keys / "wiley.key").read_text().strip()
        cfg["wiley"] = raw.split("=", 1)[1].strip().strip("\"'") if "=" in raw else raw
    return cfg


def route_for(article: dict) -> list[str]:
    """Ordered sources to try for one article, best first.

    Routing on the Crossref registrant prefix is exact -- the prefix *is* the publisher -- so the
    entitled API goes first when there is one. The rest is fallback: publisher-native renderings
    are preferred over PMC's uniform one, but a PMC copy beats no copy, and chaining lifts overall
    yield well above any single route.
    """
    prefix = (article.get("doi") or "").strip().split("/")[0]
    chain = []
    if prefix in DOI_ROUTE:
        chain.append(DOI_ROUTE[prefix])
    chain += ["s2", "unpaywall"]
    # PMC is the backstop, deliberately last. It is the most reliable route when a PMCID exists,
    # but its typesetting is uniform and easier than the real workload, so an extractor scored
    # against it looks better than it is. Preferring publisher-native renderings costs about one
    # point of overall yield -- measured 43% with PMC second against 42% with it last -- and buys
    # a test set that resembles the documents this will actually meet.
    if article.get("pmcid"):
        chain.append("pmc")
    return chain


def fetch_auto(article: dict, out: Path, cfg: dict, pause: float) -> str:
    """Try each viable source in turn until one yields a PDF."""
    for source in route_for(article):
        status = fetch_from(source, article, out, cfg, pause)
        if status.startswith(("ok", "cached")):
            return f"{status} [{source}]"
    return "no pdf / blocked"


def fetch_from(source: str, article: dict, out: Path, cfg: dict, pause: float) -> str:
    """Download one article's PDF from one source. Returns a status string."""
    if source == "pmc" and not article.get("pmcid"):
        return "no pmcid"
    if source != "pmc" and not (article.get("doi") or "").strip():
        return "no doi"
    if source in ("elsevier", "wiley") and not cfg.get(source):
        return f"no-{source}-key"
    dest = out / "pdfs" / source / f"{article_id(article)}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"

    body = SOURCES[source](article, cfg)
    time.sleep(pause)
    if body is None:
        return "no pdf / blocked"
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
    parser.add_argument("--out", type=Path, default=EXPERIMENT,
                        help="experiment directory; writes data/ and pdfs/ under it")
    parser.add_argument("--survey", action="store_true", help="report what is available and exit")
    parser.add_argument("--candidates", choices=("pubget", "neurostore", "both"), default="pubget",
                        help="pubget = local XML, cleaner ground truth, ~1.6k articles. "
                             "neurostore = ~31k coordinate-bearing studies across ~291 journals, "
                             "noisier ground truth but far more publisher-diverse. "
                             "both = the deduplicated union, pubget ground truth preferred")
    parser.add_argument("--max-candidates", type=int, default=500,
                        help="cap on the NeuroStore arm only; the local pubget arm is always "
                             "taken in full. One API call per study, so this drives runtime")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for random NeuroStore page sampling (reproducible)")
    parser.add_argument("--source", action="append", choices=sorted(SOURCES) + ["auto"],
                        help="PDF source to fetch from; repeatable. pmc = PMC Cloud (uniform "
                             "rendering), unpaywall / s2 = publisher-hosted (representative)")
    parser.add_argument("--limit", type=int, help="cap the number of articles (for a quick pilot)")
    parser.add_argument("--pause", type=float, default=0.34, help="seconds between requests")
    parser.add_argument("--email", default="aid338@eid.utexas.edu",
                        help="contact address required by the Unpaywall API")
    args = parser.parse_args(argv)

    if args.candidates == "both":
        local = [a for a in scan(ARTICLE_GLOBS) if a["redistributable"]]
        articles = merge_pools(local, scan_neurostore(args.max_candidates, seed=args.seed))
        eligible = articles
        from collections import Counter as _C
        origins = _C(a.get("ground_truth_source") for a in articles)
        print(f"union pool                             {len(articles)}")
        print(f"  ground truth from pubget XML         {origins['pubget']}")
        print(f"  ground truth from NeuroStore         {origins['neurostore']}")
        print(f"  carrying a DOI (publisher APIs)      {sum(1 for a in articles if a.get('doi'))}")
        print(f"  carrying a PMCID (PMC Cloud)         {sum(1 for a in articles if a.get('pmcid'))}")
    elif args.candidates == "neurostore":
        articles = scan_neurostore(args.max_candidates, seed=args.seed)
        eligible = articles          # licence gating is the fetch source's problem here
        journals = {(a.get("publication") or "?") for a in articles}
        print(f"NeuroStore coordinate-bearing studies  {len(articles)}")
        print(f"  distinct journals                    {len(journals)}")
        print(f"  flagged open access                  {sum(1 for a in articles if a['redistributable'])}")
        print(f"  with a PMCID (PMC Cloud reachable)   {sum(1 for a in articles if a.get('pmcid'))}")
    else:
        articles = scan(ARTICLE_GLOBS)
        eligible = [a for a in articles if a["redistributable"]]
        print(f"distinct articles with a coord table   {len(articles)}")
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
    (out / "data").mkdir(parents=True, exist_ok=True)

    (out / "data" / "ground_truth.json").write_text(json.dumps(eligible, indent=1), encoding="utf-8")
    with (out / "data" / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pmcid", "n_coordinates", "licence", "article_dir"])
        for a in eligible:
            writer.writerow([a["pmcid"], a["n_coordinates"], a["licence"], a["article_dir"]])
    print(f"\nwrote {out/'data'}/ground_truth.json + candidates.csv ({len(eligible)} articles)")

    # --limit caps the fetch only; the artefacts above always describe the full eligible set.
    if args.limit:
        eligible = eligible[: args.limit]
        print(f"--limit {args.limit}: fetching a subset, artefacts still cover all candidates")

    if not args.source:
        print("re-run with --source {pmc,unpaywall,s2} (repeatable) to download PDFs")
        return 0

    creds = load_credentials(args.email)
    print(f"\ncredentials: elsevier={'yes' if creds.get('elsevier') else 'NO'} "
          f"wiley={'yes' if creds.get('wiley') else 'NO'}")
    for source in args.source:
        print(f"\n=== {source}: fetching {len(eligible)} PDFs ===")
        tally: dict[str, int] = {}
        for i, article in enumerate(eligible, 1):
            try:
                status = (fetch_auto(article, out, creds, args.pause) if source == "auto"
                          else fetch_from(source, article, out, creds, args.pause))
            except Exception as exc:                      # noqa: BLE001 - one bad record must
                status = f"error {type(exc).__name__}"    # not abort the remaining fetches
            tally[status.split()[0]] = tally.get(status.split()[0], 0) + 1
            print(f"  [{i}/{len(eligible)}] {article_id(article)}: {status}", flush=True)
        got = tally.get("ok", 0) + tally.get("cached", 0)
        print(f"  -- {source}: {got}/{len(eligible)} PDFs ({got/max(len(eligible),1)*100:.0f}%)")
        for key, count in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"     {key:<18} {count}")

    print(f"\nPDFs under {out/'pdfs'}/<source>/. Ground truth in {out/'data'}/ground_truth.json.")
    print("Score an extractor with score_stub() for numbers comparable across sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

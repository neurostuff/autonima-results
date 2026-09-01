# PDF table extraction — feasibility and benchmark

Can we recover coordinate tables from PDFs well enough to use them?

This matters for two things nothing else reaches: **user-uploaded papers** (the Tier 2 delivery
path in [FUTURE_DIRECTIONS.md](../../FUTURE_DIRECTIONS.md)) and **opportunistic PDF sources that
carry no parsed tables** — Semantic Scholar among them, and most supplementary material
(autonima#36, measured at ~25% of otherwise-eligible studies in the schizophrenia pilot).

It is explicitly *not* on the current critical path. The retrieval corpus on disk is **40,534 XML
files and zero PDFs**: pubget returns JATS, Elsevier returns XML. For that path the right tool is a
fast XML parser (`quick-xml`, `lxml`), and the in-text coordinate work (+30–36% coordinates) is
regex over already-structured markup with no PDF parsing and no models at all.

---

## The idea: a benchmark that needs no hand labelling

For any article whose coordinates we **already hold in parsed form**, those coordinates are ground
truth for what an extractor should recover from the same paper's PDF. Fetch the PDF, run the
extractor, score against what we already know. That converts "which PDF tool is best" from an
argument into a measurement, for free.

Two candidate pools, trading ground-truth quality against size and diversity:

| pool | n | ground truth | character |
|---|---|---|---|
| **pubget** (this run) | **451** | publisher JATS XML | clean, but PMC-OA by construction so publisher-skewed |
| **neurostore** | ~31,000 | NeuroStore's parsed coordinates | ACE-era parsing, noisier; ~291 journals, no journal above 15% |

Use pubget when an absolute accuracy number is wanted, NeuroStore for comparing extractors against
each other. A later script will build the large NeuroStore arm; this folder is the small local one.

### Correction worth recording

An earlier survey reported 1,640 articles and 26,486 coordinates. That counted article
*directories*, and the same paper is downloaded independently by every project and run that
screened it in — roughly 3.6× inflation. Deduplicated by PMCID the real figures are **671 distinct
articles with a coordinate table, 451 of them redistributable, carrying 7,256 coordinates**.

---

## The reframe that may remove the need for a model

The downstream consumer is an LLM (`CoordinateParsingClient`), not a schema validator. So what is
needed is not *table structure recognition* but **enough spatial fidelity for an LLM to reconstruct
rows**.

Coordinate tables are unusually forgiving here: the target is `(label, x, y, z)` where x/y/z are
three small integers on a line — numerically distinctive, and robust to imperfect cell segmentation
in a way a financial table would not be. So the cheap path should be tested before the expensive
one is bought.

| tier | approach | tools | cost |
|---|---|---|---|
| **A** | geometry-preserving text, no ML | `pdfium-render` (Rust, permissive), `mutool draw -F stext` (C, AGPL), `pdftotext -bbox-layout` (C++, GPL) | ms/page, no GPU |
| **B** | rule-based table detection | `pdfplumber`, `camelot`, `tabula-java` | slow, CPU |
| **C** | ML table structure | Docling/TableFormer, `marker`/Surya, **Table Transformer** (ONNX-exportable), PaddleOCR PP-Structure | GPU preferred |

Tier B is likely to underperform *here specifically*: neuroimaging coordinate tables are frequently
**unruled** — whitespace-aligned with no borders — which is exactly where lattice methods fail.

**The honest Rust position:** there is no mature Rust equivalent of Docling for table structure.
Rust's real contributions are `pdfium-render` for fast text-plus-geometry and **`ort`** (ONNX
Runtime bindings) for running models without Python overhead, so the pragmatic Rust architecture is
`pdfium-render` + `ort` running Table Transformer or PP-Structure weights. `ferrules` is the closest
packaged attempt and `extractous` the closest general-purpose one; both are working from
recollection here and want verifying against current sources. And for Docling-class tools **the
language is not the bottleneck** — the layout and table models are. A Rust rewrite of orchestration
buys little; a faster ONNX execution provider buys a lot.

### Two source-specific notes

- **Supplementary material is often not a typeset PDF.** It is frequently an author-produced Excel
  or Word export. Try the native format first — `calamine` (Rust) reads xlsx very fast, `docx-rs`
  for Word — since author tables tend to be *cleaner* than publisher typesetting. This may sidestep
  PDF parsing entirely for a share of autonima#36.
- **Pre-2000 papers may be image-only**, needing OCR regardless of tool (Tesseract, Surya). That
  intersects the pre-2004 corpus hole measured elsewhere (34% with coordinates), so it is the same
  shrinking slice twice over.

---

## Getting the PDFs: which routes actually work

Tested 2026-09-01, because most of the documented routes are dead:

| route | result |
|---|---|
| PMC OA Web Service (`oa.fcgi`) | **404** on both `www.ncbi.nlm.nih.gov` and `pmc.ncbi.nlm.nih.gov` |
| PMC FTP dataset tree | **emptied August 2026**; `/pub/pmc/` now holds only `PMC-ids.csv.gz` |
| `pmc.ncbi.nlm.nih.gov/articles/PMC*/pdf/` | HTTP 200, but a "Preparing to download ..." bot-mitigation interstitial |
| Europe PMC `fullTextPDF` | **0 of 30** candidates |
| **PMC Cloud Service** (AWS Open Data) | **25 of 25.** No login, no key |
| **Unpaywall** (`best_oa_location.url_for_pdf`) | **60 of 60 resolve**, all to publisher hosts |
| **Semantic Scholar** (`openAccessPdf`) | 59 of 60 resolve, same hosts |
| OpenAlex (`best_oa_location.pdf_url`) | 19 of 60 — markedly more conservative |
| **Elsevier Article API** | **100%** of `10.1016` — needs `ELSEVIER_API_KEY` + entitled IP |
| **Wiley TDM API** | ~42–50% of `10.1002`/`10.1111` — needs `WILEY_TDM_TOKEN` |

`https://pmc-oa-opendata.s3.amazonaws.com/` is PMC's current sanctioned route and the FTP readme
now points at it. Objects are keyed by PMCID and version (`PMC3434213.1/PMC3434213.1.pdf`); the
highest version wins, since a `.2` is typically the publisher's typeset copy replacing an author
manuscript.

### "Open access" and "obtainable" are different sets

Resolution is near-total; download is not. Of 25 Unpaywall URLs actually fetched:

| outcome | n | hosts |
|---|---|---|
| real PDF | 14 (56%) | Frontiers, PLOS, Nature |
| HTTP 403 + HTML | 6 | SfN, Wiley, MDPI, OUP |
| HTTP 200 + HTML | 5 | BMC redirect pages |

That is publisher bot-protection, not missing content — **the same IP/entitlement wall the Elsevier
work hit, in a different guise**. Closing it means publisher TDM APIs or institutional IP, not
cleverer scraping. And on a *publisher-stratified* sample of 90 studies across 76 journals, Semantic
Scholar alone yields only **19%**, with the successes concentrated in 15 journals. A candidate list
spanning 76 journals collapses to an OA-publisher benchmark the moment you try to fetch it.

---

## Source choice is a validity control

PMC's rendering is uniform and easier than the real workload, so an extractor validated only against
it will look better than it is. **`--source auto` therefore prefers publisher-native renderings and
keeps PMC as the backstop**, in this order:

    entitled publisher API (by DOI prefix) -> s2 -> unpaywall -> pmc

Routing on the Crossref registrant prefix is exact — the prefix *is* the publisher — so
`10.1016` goes to Elsevier and `10.1002`/`10.1111` to Wiley before anything free is tried.

Ordering PMC last costs roughly one point of overall yield (measured 43% with PMC second against
42% with it last) and buys a test set that resembles the documents this will actually meet. The
source is recorded in each PDF's path, so the rendering variable stays controllable at analysis
time, and both renderings can be had for the overlap subset by pinning `--source` explicitly.

---

## Layout

    experiments/pdf_table_extraction/
      README.md                      this file
      scripts/build_benchmark.py     candidate discovery + multi-source fetch + score_stub()
      data/ground_truth.json         451 articles with their XML-derived coordinates
      data/candidates.csv            pmcid, doi, n_coordinates, licence, source dir
      pdfs/<source>/PMC*.pdf         gitignored (~800MB); source is in the path

## Running it

```bash
# what is available, no network
python experiments/pdf_table_extraction/scripts/build_benchmark.py --survey

# fetch the local 451, publisher-native where possible
python experiments/pdf_table_extraction/scripts/build_benchmark.py --candidates pubget --source auto

# pin one source, to compare renderings of the same papers
python experiments/pdf_table_extraction/scripts/build_benchmark.py --source pmc
```

Credentials are read from the environment, falling back to `~/.keys/elsevier.key` and
`~/.keys/wiley.key`. Values are read but never logged. Note that `elsevier.key` also holds
`SPRINGER_API_KEY`, so the Elsevier variable has to be matched by exact name — a generic
`*API_KEY*` pattern silently picks up the wrong credential, which is a bug this script had.

## Scoring

Deliberately left to the caller, since it depends on the extractor under test. Load
`data/ground_truth.json`, produce the same shape from your extractor, and compare. `score_stub()`
in the script documents the intended matching — exact triple match, order-insensitive, per article
— so numbers stay comparable across tools and sources.

Ground truth is noisy per-article and sound in aggregate: a coordinate row is any table row with at
least three integer cells in [-120, 120], which over-counts demographic tables full of small
integers and misses coordinates reported as floats.

## Next

1. Run Tier A (`pdfium-render` or `mutool`) plus the existing LLM parser over `pdfs/`. If it
   recovers ≥90% of coordinates, stop — no model is needed.
2. Score PMC-rendered against publisher-native on the overlap subset, to quantify how much
   PMC-only validation over-estimates.
3. Escalate to Tier C only for the measured residue.
4. Separately, a larger NeuroStore-based arm (~31,000 candidates, ~13,400 obtainable) for
   extractor-vs-extractor comparison at scale.

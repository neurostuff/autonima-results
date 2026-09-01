# Future directions — autonima and automated meta-analysis

Strategy note, started 2026-08-31 as a narrower question about life beyond CBMA and since grown to
cover data sources, deployment, licensing, and tooling. Written after the deactivation and ENIGMA
pilots. Everything quantified here is measured against this project's own runs and corpus rather than
estimated; where a number is a guess or a recollection it says so. Nothing here is committed work.

## Contents

**What to do with the data we have**
- [The premise](#the-premise) — the demonstrated strength, with evidence
- [The limitation is worse than "only coordinates"](#the-limitation-is-worse-than-only-coordinates)
- [The measurement that reframes it](#the-measurement-that-reframes-it) — we discard more than we use
- [Directions, ranked](#directions-ranked) — effect sizes first
- [Recommendation](#recommendation) and [cheapest first step](#cheapest-first-step)

**Where more data lives**
- [The second measurement](#the-second-measurement--where-else-the-results-live) — body text, figures,
  dataset reuse, NeuroVault; and what is ruled out

**How it gets delivered**
- [Delivery](#delivery-what-a-finished-autonima-should-be) — the IP-entitlement constraint, the
  measured corpus ceiling, the three tiers, and the constraint that actually binds
- [Copyright](#copyright-what-actually-matters-and-what-does-not) and
  [provenance display](#provenance-display-what-can-be-shown-by-licence-tier)
- [PDF and table extraction tooling](#pdf-and-table-extraction-tooling)
- [Build order](#build-order)

**Further afield**
- [Appendix: which other domains](#appendix-which-other-domains-and-why) — beyond neuroimaging

---

## The premise

The demonstrated strength is **high-precision selection at both paper and analysis level, at scale,
for almost no money**. Evidence from this project:

| capability | evidence |
|---|---|
| screening matches hand curation | decomposition Q2: ER +0.017, social +0.002 |
| annotation converts a mixed map into a specific one | deactivation: DMN purity 54.7% -> 90.9%, enrichment 2.39x -> 3.98x, from 3x FEWER studies |
| analysis-level selection on crossed labels | ER goal x strategy: +0.244 annotation gain |
| it is cheap | $0.37 for 957 studies / 2,650 analyses (gpt-5.6-luna, reasoning off) |

The stated limitation is that the input modality is **coordinates**.

---

## The limitation is worse than "only coordinates"

Coordinates are lossy, biased, and shrinking.

**Lossy.** A peak says *where*, never *how much*. No effect size means no random-effects model, no
heterogeneity statistic, no moderator analysis, no publication-bias test. CBMA measures convergence
of *reporting*, not magnitude of *effect*.

**Biased.** Only supra-threshold peaks are reported, so the input is pre-filtered by each study's
own threshold and correction choices.

**Shrinking.** The field is migrating to FreeSurfer/parcellation pipelines. The TLE pilot rejected
**60 of 69** full texts as ROI-based, and that share grows every year. Worse, it is exactly the
methodology ENIGMA uses -- which is why ENIGMA can mega-analyse the literature and why CBMA cannot
consume it.

We hit the consequence directly: **Parkinson's has ~183 usable papers and is unanalysable** at
d ~ 0.14, because small effects produce nothing for a convergence method to converge on. MDD
(d ~ 0.06) and antisocial (median d 0.05) are dead for the same reason. Supply was never the
criterion; effect size was.

---

## The measurement that reframes it

Classifying every table pubget extracted across the two experiments (n = 1,347):

| table type | n | share |
|---|---|---|
| descriptive / non-numeric | 644 | 47.8% |
| other numeric (mostly demographics, clinical) | 329 | 24.4% |
| **coordinate tables — all we currently use** | 184 | **13.7%** |
| **ROI + statistics — discarded** | 114 | **8.5%** |
| ROI + numbers, no clear stat header | 76 | 5.6% |

Sample headers from the discarded pile: `Left TLE < healthy controls | ...` with statistics
attached. **We already extract roughly two-thirds as much ROI effect-size data as coordinate data,
and throw all of it away.**

(The classifier is crude -- some coordinate tables are misfiled as demographic -- so treat the exact
percentages as indicative. The direction is not in doubt.)

---

## The second measurement — where else the results live

The section above is a *demand*-side reframing: we already extract more than we use. This one is
*supply*-side: how much of the literature's results are invisible to us entirely.

Classifying 564 pubget articles across both experiments by where their coordinates actually appear
(threshold: 3+ coordinate triples, so incidental numbers do not count):

| where the coordinates are | n | share |
|---|---|---|
| tables — the only place we currently look | 140 | 24.8% |
| **body text only — currently invisible** | 91 | **16.1%** |
| no coordinates anywhere | 333 | 59.0% |

Under the looser "any coordinates" classification, 26 articles report them in *both* places. That
overlap group is the free validation set for anything built on prose extraction.

### 1. In-text coordinates  **[cheapest real gain]**

**+65% more coordinate-bearing studies**, at lower density per study:

| | coords/article, median | mean |
|---|---|---|
| table-derived | 9 | 16 |
| body-text only | 5 | 7 |

So 0.56x the density — sparser, but not by an order of magnitude. Net effect is roughly **+30-36%
more coordinates** corpus-wide. Worth having; not transformative.

Two things cut against the obvious worry that prose results are harder to *select*:

- **Attribution may actually be easier in prose than in tables.** A sentence states its own contrast
  ("relative to controls, patients showed reduced volume in..."), whereas a table header often says
  `Table 2` and nothing more. The sparse-analysis-name problem (autonima#61 — 74% of unnamed
  analyses have no description either) is a *table* problem specifically.
- The real risk is different and worse: **prose coordinates are the authors' highlights**, chosen for
  the narrative, so they are selectively reported *within* a paper. A map built from them could
  inherit that emphasis as spatial bias.

That risk is directly testable on the 26 both-group articles: extract the prose subset, compare it
against the full table from the same paper, and measure whether the prose peaks are spatially biased.
No new corpus needed. Do that before trusting prose-only studies.

### 2. Dataset-reuse detection  **[a validity problem, not a volume one]**

Full-text mentions in Europe PMC, as a scale check:

| corpus mentioned | articles |
|---|---|
| UK Biobank | 13,054 |
| ADNI | 12,489 |
| OSF + fMRI | 8,153 |
| ABIDE | 1,969 |
| "unthresholded" + fMRI | 1,811 |
| NeuroVault + fMRI | 843 |

The interesting consequence is not more data — it is that **CBMA assumes study independence and this
breaks it**. Twelve ADNI papers are not twelve samples. Nobody currently corrects for this because
nobody knows which papers share subjects, and finding out is a full-text screening problem: exactly
the demonstrated strength. This is the one item on the list that fixes a *correctness* flaw rather
than adding volume, and we already have direct evidence the problem is real — the dementia benchmark
aggregated studies that shared a data source, which is part of why its dice coefficients are low.

### 3. Shared unthresholded maps -> IBMA

IBMA on unthresholded maps is far more powerful than CBMA, and we control NeuroVault, so the access
half is solved. The unsolved half is selection, and image selection is *harder* than coordinate
selection.

But there is an asymmetry worth exploiting: a NeuroVault map is usually **linked to a paper**, and
the paper describes the contrast properly even when the map's own label is empty or cryptic. So the
task is not "classify this image" — it is *map -> linked paper -> annotate the contrast from text ->
propagate the label back to the map*. That is paper-side annotation, which is the capability already
demonstrated, and the image never has to be classified directly. Missing map labels are the same
defect as missing analysis names (autonima#61), just in a different container.

### 4. Existing curated corpora — ingestion, not extraction

BrainMap, Neurosynth, NeuroStore. Already-parsed coordinates that need importing rather than
extracting. Cheap, but adds no capability and no novelty; it is a corpus-completeness task. Already
in progress on the NeuroStore side.

### 5. Figures  **[large pool, no measurable yield, research not engineering]**

Of the 333 articles with no coordinates anywhere, **323 (97%) contain at least one figure**, median 3
each. So the pool is the largest of any source here, and figures are the *only* route to those
papers.

Feasibility splits sharply by figure type:

| figure type | tractable? | why |
|---|---|---|
| slice montage with printed slice labels (`z = 24`) | partially | the label fixes one axis exactly; in-plane position must be estimated against the template. Realistically +/-5-10mm |
| surface rendering | no | no coordinate frame at all |
| glass brain / 3D render | barely | projection ambiguity |

The argument in favour: **CBMA already smooths peaks by 10-15mm FWHM**, so +/-10mm localisation error
may be tolerable for a density estimator like MKDA in a way it would never be for anything
effect-size-based.

The argument against, which wins for now: **there is no cheap error signal.** Every other source on
this list has ground truth — the table says what it says. Figure extraction has none unless someone
hand-labels a validation set, and a *systematically* biased localisation would distort maps silently
rather than failing loudly. It is also the only direction here where "LLMs are good at this" is an
assumption rather than something this project has evidence for. Unlike in-text coordinates, the yield
is unmeasured, so any estimate of what it would buy is speculation until a sample is actually
examined.

### Ruled out by measurement

- **Europe PMC as an additional retrieval route.** Tested against 200 actual abstract-included PMIDs:
  27% open access, versus the 28% pubget already achieves. There is no hidden corpus and pubget is
  not underperforming.
- **Preprints.** Negligible volume in these searches.
- **Grey literature** (theses, conference abstracts) — plausible but unmeasured, and reporting quality
  is likely poor.

### Supply-side ordering

**Dataset-reuse detection first** — it is the only one that fixes a correctness flaw, nobody else is
doing it, and it needs no new extraction machinery. **In-text coordinates second**, as the cheap
volume win, with the highlight-bias check on the 26-article overlap group built in from the start.
**NeuroVault labelling third**, as the highest-ceiling item and the one this group is uniquely placed
to do. **Figures last, framed as research** rather than as a pipeline feature.

---

## Directions, ranked

### 1. Extract effect sizes, not just locations  **[highest value]**

Read the ROI tables currently discarded and emit `(region, contrast, d or t/F/p, n1, n2)`.

What it buys:

- the ~59% of structural literature that is currently unusable
- **actual meta-analysis** — random effects, heterogeneity, funnel plots, meta-regression, none of
  which CBMA supports
- **exact ENIGMA validation** instead of qualitative: both sides become per-ROI Cohen's d, so the
  comparison is a correlation across 68 regions rather than "do our clusters overlap their top ROIs"
- disorders CBMA cannot touch — Parkinson's, MDD — become tractable, because pooling magnitudes does
  not care that an effect is small

Hard parts, all normalisation problems and all LLM-tractable: ROI naming -> atlas mapping, effect-size
conversion from t/F/p plus n, and sign/direction conventions. The `analysis-schema` repo already
exists for exactly this class of problem.

### 2. Extract sample and method metadata  **[companion to 1]**

n, age, sex, medication status, illness duration, field strength, smoothing kernel, threshold,
correction method. That is most of the 24.4% "other numeric" bucket, currently invisible.

Enables moderator analysis — what clinicians actually ask for — and lets you audit whether
liberal-threshold studies are driving a cluster. Same tables and same extraction pass as 1, which is
why they should be one project.

### 3. Use screening to find image-based data

IBMA on unthresholded maps is far more powerful than CBMA. The bottleneck is knowing *which* papers
shared maps and where they live — a screening problem, not a statistics problem, and therefore
squarely in the strength. A corpus of "papers with NeuroVault/OSF accessions, linked to their
contrasts" would have standalone value. See supply-side item 3 above for the map-labelling framing,
which is the tractable form of this.

### 4. Stop treating the map as the deliverable

The real capability is **evidence structuring**. The output could be a queryable analysis-level
database — *"every patient>control contrast in a reward task with n>50"* — with meta-analysis as one
query type among many. That is NeuroStore's mission, and autonima is the ingestion engine for it.
See autonima#6 and #27. Already under way on the NeuroStore side, so the autonima-side question is
only what the ingestion contract should be.

### 5. Living meta-analysis

At $0.37 per 1,000 studies, monthly re-runs are free. Living systematic reviews are a recognised
unmet need precisely because nobody can afford them manually. The caching already supports it: a
re-run gap-fills per study rather than redoing the corpus.

### 6. Meta-science

Characterise the literature itself: reporting completeness, threshold practices, coverage bias, what
is systematically never reported. The unreported-space map idea belongs here. Lower prestige,
genuinely novel, and very cheap — the corpus is already on disk.

### 7. Leave neuroimaging

Analysis-level selection is not domain-specific. Any literature whose unit of interest lives inside a
table has the same problem. Expanded below, since picking the right domain matters more than the
idea itself.

---

## Recommendation

There are two independent axes here — *use more of what we already extract* (directions 1-7) and
*extract from places we do not currently look* (supply-side 1-5). They do not compete for the same
work, and the demand side is both cheaper and better evidenced.

**Do 1 + 2 as a single project.** They share one extraction pass, and together they change what
autonima *is*: from a CBMA pipeline constrained by a shrinking input format, into a
structured-evidence extraction engine whose output happens to support CBMA among other things.

Three reasons to prefer it over the alternatives:

1. It directly capitalises on the demonstrated strength — analysis-level precision — rather than
   inventing a new one.
2. Feasibility is already evidenced: those tables are sitting extracted and unused in `pubget_data`
   right now, so a prototype needs no new retrieval, no new screening, and no new corpus.
3. It fixes the validation story. The ENIGMA comparison stops being "do our clusters qualitatively
   overlap their top ROIs" and becomes two independent Cohen's d estimates per ROI, correlated
   across regions.

## Cheapest first step

Take the 114 already-extracted ROI+statistics tables, prompt the model to emit
`(region, contrast, effect, n1, n2)` for each, and check how many normalise cleanly to
Desikan-Killiany. That is a few dollars and an afternoon, needs nothing new, and settles whether the
extraction is reliable enough to build on before anything else is committed.

---

## Related open issues

- autonima#36 — supplementary PDFs as extraction input. Quantified during the schizophrenia pilot at
  **~25% of otherwise-eligible studies**: 8 of 11 studies that passed full-text screening but yielded
  no coordinates had pubget tables containing none, with the coordinate tables most plausibly in
  supplementary material.
- autonima#6, #27 — NeuroStore ingestion, which direction 4 depends on.
- autonima#60 — duplicate coordinate tables; matters more once effect sizes are pooled, since a
  duplicated table would double-weight a study's magnitude rather than just its peaks.

---

## Delivery: what a finished autonima should be

Two measured facts constrain this more than any design preference.

**1. Publisher entitlement is IP-based.** The same Elsevier key returns HTTP 200 / 311KB from the
campus IP and 400s from off-campus. A cloud-hosted autonima has *no* entitlement and is restricted to
the ~28% open-access slice — the exact constraint the retrieval workstream spent weeks fighting.

**2. Runs are hours long and retrieval-bound**, not compute- or LLM-bound. At $0.37 per 1,000
studies, the LLM cost is negligible; the wall-clock is almost entirely fetching.

Both objections evaporate if the hosted tier operates on a **fixed corpus** instead of retrieving:
retrieval is deleted rather than worked around, and a run becomes minutes. So the real question is
what a fixed corpus costs in recall.

### The corpus ceiling, measured

Every gold-standard study from the nine benchmarks, probed against the NeuroStore API
(84,287 base studies; 1,519 unique gold PMIDs, 0 probe failures):

| project | gold | in NeuroStore | % | with parsed coords | % |
|---|---|---|---|---|---|
| social | 238 | 218 | 92% | 209 | 88% |
| cue_reactivity | 191 | 129 | 68% | 105 | 55% |
| executive_function | 171 | 84 | 49% | 72 | 42% |
| decision_making | 153 | 98 | 64% | 94 | 61% |
| problem_solving | 126 | 77 | 61% | 71 | 56% |
| emotion_regulation | 88 | 74 | 84% | 61 | 69% |
| vbm_of_substance_use | 79 | 49 | 62% | 39 | 49% |
| dementia | 74 | 30 | 41% | 29 | 39% |
| vbm_of_ptsd | 21 | 19 | 90% | 8 | 38% |
| **total** | **1,141** | **778** | **68%** | **688** | **60%** |

**A corpus-only hosted tier has a hard recall ceiling of 60%** against hand-curated meta-analyses.

### The comparison that makes it viable

Against what the live pipeline actually delivers end-to-end (gold PMIDs present in each project's
newest NiMADS studyset; emotion_regulation excluded, no studyset at that path):

| | gold recall |
|---|---|
| fixed corpus, with coordinates | **60%** |
| live pipeline, post-screening | **56%** |

These are not like-for-like: 60% is a *supply ceiling* measured before any screening, while 56% is
already post-screening and post-extraction. Applying our screening to the corpus would pull the
hosted number somewhere below 56%. But the gap is single-digit percentage points, and that is the
whole finding — **hosting on a fixed corpus costs a few points of recall and buys away the entire
retrieval problem**: no entitlement to hold, and no hours-long runs.

### Where the corpus is weak: era

Same gold set binned by publication year (years resolved from PubMed for all 1,519):

| era | gold | in NS | % | with coords | % |
|---|---|---|---|---|---|
| <= 2004 | 316 | 144 | 46% | 109 | **34%** |
| 2005-09 | 415 | 263 | 63% | 238 | 57% |
| 2010-14 | 473 | 362 | 77% | 333 | **70%** |
| 2015-19 | 295 | 212 | 72% | 188 | 64% |
| 2020+ | 20 | 13 | 65% | 9 | 45% |

Two different deficits with two different causes. The pre-2004 hole (34%) is digitisation and
parsability — largely permanent. The 2020+ dip is *ingestion lag*, not absence, and it is where users
most want coverage. (n = 20 for that bin, so treat it as a hint, not a measurement.)

This also explains the per-project spread above: social (a 2022 meta-analysis) hits 88%, while
executive_function and dementia — drawing on 1990s-2000s literature — sit at 42% and 39%.

### The architecture that follows

A hosted service on the fixed corpus, plus a **thin local companion** for retrieval. The companion is
not a fallback; it is the ingestion path:

| stage | hosted tier | local companion |
|---|---|---|
| search | filter the corpus snapshot | PubMed Entrez |
| retrieval | *does not exist* | runs where the entitlement is |
| screening / annotation | hosted, minutes, ~$0.37/1k | hosted or local |
| meta-analysis | hosted (NiMARE) | either |
| output | studyset in Compose | pushed up, joins the corpus |

Three properties make this better than either half alone:

1. **The sharing boundary is clean.** What gets *pooled* is extracted content — tables, coordinates,
   analysis records — which is not the publisher's expression and is what NeuroStore already ingests.
   PDFs, if uploaded at all, stay private to the uploader. See the copyright note below: this is a
   permissions boundary, not a restriction on what users may upload.
2. **The fixed corpus stops being fixed.** Every local run raises the ceiling for every hosted user,
   and it raises it preferentially at the recent end — precisely the ingestion-lag deficit above.
3. **The retrieval problem is solved once, by whoever has the entitlement**, rather than by the
   service pretending to have it.

### Copyright: what actually matters (and what does not)

An earlier draft of this note claimed hosting user-uploaded PDFs was a redistribution problem to be
designed around. That was overstated, and the framing was wrong.

**The precedent is directly on point.** Covidence and Rayyan — the systematic-review platforms
Cochrane reviewers actually use — treat full-text PDF upload as a core feature. So do Mendeley,
Paperpile, Elicit, and SciSpace. Allowing users to upload the PDFs they already have access to is
normal, not novel.

**The mechanism that puts the liability on the uploader** is DMCA §512 safe harbour (and the DSA in
the EU). Its conditions are administrative and cheap, not architectural: a designated agent
registered with the Copyright Office, a notice-and-takedown process, a repeat-infringer policy, no
actual knowledge of specific infringement, and no direct financial benefit tied to it.

**The line is not PDF-versus-extracted. It is private-versus-pooled.** ResearchGate is the
cautionary case (Elsevier/ACS litigation, 2017-2023), and it turned on content being *publicly
accessible*, not on it being PDFs. A PDF visible only to its uploader is Covidence; the same PDF
served to other users is a different activity.

That distinction is convenient rather than costly here, because the thing worth pooling is the
extraction, not the file. So: **allow PDF upload, keep uploads private to the uploader, pool only
derived records.** Both halves, separated by a permissions boundary.

**Extraction itself rests on separate and firmer ground** — the EU DSM Directive Art. 3 text-and-data
-mining exception for research organisations, and US fair use for TDM after *HathiTrust* and
*Authors Guild v. Google*.

**The architectural consequence is the useful part.** The IP-entitlement finding is untouched: a user
still needs their own institutional access to obtain a PDF at all. But they can obtain it however
they already do and upload it, which means **PDF upload substitutes for the local companion in v1**.
The companion drops from "required, because the service cannot hold entitlement" to "automation
convenience, because bulk manual download is tedious." That is a real simplification — it moves the
companion later in the build order.

**One caveat that survives.** Publisher licence agreements commonly prohibit systematic downloading
and deposit of licensed content into shared repositories. Those terms bind the *user*, not the
service — which is a second, independent reason to make per-user privacy the default: it keeps users
compliant, rather than protecting us.

None of the above is legal advice, and it is worth a short conversation with UT's IP office before
the hosted tier goes public. But it is a policy-and-permissions question, not an architecture
blocker.

### Provenance display: what can be shown, by licence tier

The extracted layer is error-prone, so users will want to see the sentence a field came from — and
will still read whole papers for PRISMA-grade work. That makes provenance display a product
requirement, not a nicety. It is also the sharpest copyright question in the design, because it is the
first place NeuroStore would reproduce publisher expression rather than facts.

**None of the following is legal advice.** It is the landscape as best I can establish it; the
snippet-length and licence-filter decisions want sign-off from counsel or the library's scholarly
communications office, and the contract question below wants someone to actually read the current
Elsevier agreement.

#### The plan's flaw: "we have a PMC download" is the wrong filter

PMC contains at least three distinct rights situations, and only one is redistributable:

- **OA Subset** — CC-BY / BY-NC / BY-NC-ND / BY-SA / CC0. Redistributable on the licence's terms.
- **Author Manuscript Collection** (NIH public access) — free to read, deposited under a right granted
  to PMC. *Not* a third-party redistribution right.
- **Other free-to-read content** — publisher-labelled "open access" with all rights reserved.

The correct filter is the licence field, and we already capture it. Across the 564 pubget articles on
disk:

| licence as recorded in the JATS XML | share |
|---|---|
| CC-BY (incl. `license-type=creativeCommonsBy`) | ~29% |
| CC-BY-NC-ND | ~8% |
| CC-BY-NC / BY-NC-SA | ~6% |
| CC0 | 0.2% |
| CC reference in prose, machine-unparsed | 44.1% |
| **`license-type=OpenAccess` / `open-access`, no CC reference** | **~10%** |
| **publisher licence-agreement URL (Frontiers, Nature)** | **~1%** |
| **no `<license>` element at all** | **1.6%** |

The bolded rows are the trap: **roughly 10-13% of what a source-based filter would treat as "open
access" carries no identifiable redistribution grant.** A publisher's "OpenAccess" label is a reading
permission, not a licence. So: parse the licence, allow-list the specific CC variants, and default to
the non-OA treatment on anything unrecognised or absent — including the 1.6% with no licence element,
which is where author manuscripts would hide.

Two licence terms carry live obligations even inside the OA set. **ND** (~8%) permits redistributing
the work unmodified — fine for showing full text, worth a thought before reformatting or truncating
it. **NC** (~14% combined) binds if NeuroStore ever charges for access.

#### Non-OA: the escalation ladder

Ordered by exposure, lowest first. The first three are effectively free and should be built
regardless:

1. **The extracted facts themselves.** n = 24, mean age 31.2, task = MID, coordinates, contrast
   labels. Facts are not copyrightable (*Feist v. Rural Telephone*, 1991). This is the entire
   structured layer, and it is solid ground.
2. **Bibliographic metadata and the abstract.** Already settled practice — PubMed redistributes
   abstracts, and NeuroStore already serves them in the `description` field.
3. **Location pointers with no text.** "Methods, paragraph 3, sentence 2", a section label, a table
   caption number, a character offset. Zero reproduction, therefore zero exposure — and genuinely
   useful to any user who has their own access, which is most of the target audience. This option is
   underrated and is the honest default for non-OA.
4. **Short verbatim snippets.** Strong fair-use posture, discussed below.
5. **Full text of non-OA papers.** Not supported by any of the precedent below. Don't.

#### Why snippets are defensible

The controlling US precedent is close to exactly this use case:

- ***Authors Guild v. Google*** (2d Cir. 2015) — scanning entire in-copyright books and displaying
  **snippets** in search results was fair use.
- ***Authors Guild v. HathiTrust*** (2d Cir. 2014) — retaining a full-text corpus for computational
  search was fair use; *displaying* full text to general users was not part of what was upheld.
- ***A.V. v. iParadigms*** (4th Cir. 2009) — Turnitin retaining complete copies for plagiarism
  detection was fair use.

The consistent pattern: **retain full text internally for computation — fine. Display bounded
snippets — fine. Display the whole work — not covered.** That maps directly onto the design.

On the four factors, this case is *stronger* than Google's. Purpose is non-profit research and
transformative in a specific sense: the snippet exists to verify a structured claim, not to convey
the author's expression. Amount is a sentence or two, far below Google's roughly one-eighth of a page.
And market effect — the decisive factor — clearly favours it: a sentence reading "participants were
24 healthy adults (12 female)" cannot substitute for the paper, no licensing market for provenance
snippets exists, and the feature drives traffic to the version of record.

#### Guardrails that make the snippet case hold

Google's snippet view survived because of its *design limits*, and those limits are the checklist:

- **Cap snippet length.** A sentence or two. See the contract note below for a possible hard number.
- **Cap cumulative retrievable text per paper**, per user and in total.
- **Prevent reconstruction.** This is the load-bearing engineering requirement: a determined user must
  not be able to walk a paper by issuing many queries. No adjacent-snippet stitching, no
  offset-walking, rate limits.
- **Keep it purposive, not browsable.** Anchor every snippet to a specific extracted field. Do not
  offer free-text search over non-OA full text returning arbitrary passages — permitted in *Google*,
  but a much harder story than "here is the sentence this number came from".
- **Always link to the DOI / version of record.** Reinforces non-substitution.
- **Honour opt-outs** and have a takedown path, as with the safe-harbour requirements above.

#### The bigger risk is contract, not copyright

This is the part that gets underweighted, and it applies directly because the corpus is built on
institutional licences:

- Publisher and library agreements commonly permit TDM for research while prohibiting redistribution
  and "systematic downloading". **Fair use is a defence to copyright infringement, not to breach of
  contract** — a bulletproof fair-use posture does not cure a licence violation.
- **Elsevier's TDM terms have historically specified a maximum snippet length** (my recollection is on
  the order of 200 characters around a match). If that clause is in the current agreement it is
  effectively the answer for Elsevier-sourced text, and it is far more concrete than a fair-use
  judgement call. **Someone should read the current agreement and get the number.**
- ACE-style scraping of publisher sites likely conflicts with site terms and systematic-download
  clauses regardless of what is later displayed. That is a pre-existing exposure, independent of this
  feature.
- EU/UK: DSM Art. 3 gives research organisations a TDM exception that contracts cannot override
  (Art. 7(1)), and UK CDPA s.29A is similar — but both cover *making copies for mining*, not
  publishing them. Snippet display falls instead to the quotation exception (InfoSoc Art. 5(3)(d)),
  which provenance display plausibly fits.

#### The clean way out for the users who care most

For non-OA papers, **show full text of the user's own upload.** It is already private to the uploader
under the Tier 2 boundary, they demonstrably have access since they supplied the file, and no
redistribution occurs. The verification problem and the upload tier solve each other: the people who
most want to read the source are exactly the people running a PRISMA-grade review, who are already in
Tier 1 or 2.

So the full picture:

| content | OA (allow-listed CC) | non-OA |
|---|---|---|
| extracted facts | yes | yes |
| abstract + metadata | yes | yes |
| location pointer, no text | yes | yes |
| short verbatim snippet | yes | fair use + guardrails; check Elsevier's contractual cap |
| full text | yes, subject to ND/NC | only to the uploader who supplied it |

### The three tiers, and the constraint that actually binds

The natural product is three tiers over one backend, separated by where the full text comes from:

| tier | full text from | PRISMA-complete? | cost |
|---|---|---|---|
| **0 — corpus only** | nowhere; filter already-extracted records | no, 60% ceiling | near zero |
| **1 — local companion** | user's network, extraction runs locally | yes | user's compute |
| **2 — user uploads PDFs** | user's network, extraction runs hosted | yes | ours, small |

Tier 0 is the interesting one because it needs no full text at all: filter on structured metadata and
already-parsed analysis records. It is also already demonstrated — the deactivation experiment did
exactly this, selecting existing parsed analyses by whether they were deactivation contrasts, for
$0.37 across 957 studies / 2,650 analyses.

But its ceiling is not the 60% study-coverage number. Sampling 4,000 NeuroStore analyses at random
(of 208,767 total, 94% coordinate-bearing):

| can you tell what the analysis is, from its record alone? | n | share |
|---|---|---|
| usable description present | 2,138 | 53.4% |
| descriptive name, 3+ words | 201 | 5.0% |
| short / uninformative name only (1-2 words) | 1,043 | 26.1% |
| name is a bare number or ID, no description | 618 | 15.4% |
| **filterable without full text** | **2,339** | **58.5%** |
| **opaque without full text** | **1,661** | **41.5%** |

Restricted to coordinate-bearing analyses — the ones a CBMA would actually use — **56.2% are
filterable**. When labels are present they are excellent, and exactly the right shape for
analysis-level selection: `Neutral cue placebo > cocaine cue placebo`, `Contrast: increase > maintain
emotions`, `Repeated errors > corrected errors`, `Healthy Control Subjects > Borderline Patients-2`.
When they are absent, the record is a bare integer.

**This is the binding constraint, and it is the one neither Tier 1 nor Tier 2 fixes.** Uploading
papers grows *coverage*; it does nothing for analysis labels on the 42,483 base studies already in the
corpus, whose names came from whatever parsed them originally. The two ceilings also compound — a
usable meta-analysis needs the study present *and* its relevant contrast identifiable — so the
realistic Tier 0 recall sits below 60%. (The joint figure is measurable but I have not measured it;
a study with several analyses only needs one labelled, so it is not a simple product.)

### A third job the two options miss: corpus repair

Re-extract analysis labels for the opaque 41.5%. For any paper whose full text is reachable, the
contrast label is recoverable from the table caption and surrounding text — the same task the
coordinate parser already performs, run against records that already have coordinates but no
identity.

It is high leverage in a way corpus *growth* is not: it improves every future query for every user,
on papers already present, with no new retrieval. It is also the direct hosted-tier analogue of
autonima#61.

### Why annotations make hosting compound

A full-corpus annotation pass over all 208,767 analyses costs roughly **$29** at the deactivation
experiment's measured rate ($0.37 / 2,650 analyses). That is cheap enough to run repeatedly.

The more important property: **annotations are reusable across users.** If one user annotates
"is this a deactivation contrast?" corpus-wide, that column can be cached and served to everyone
after. Marginal cost for a popular criterion trends to zero. Local runs never get this — every user
pays the full cost every time. It is the strongest argument for hosting that does not depend on
convenience.

The cost shape does still require a cascade for arbitrary queries: free structured pre-filter on
metadata, then LLM annotation only on survivors. $29 per speculative user query does not scale; $29
once for a cached column does.

### Can a structured evidence layer replace full text for screening?

This is the NeuroStore metadata work's central bet, and it is testable against our own runs: 20,810
full-text screening decisions with written reasons, across every project and version.

**Full text is load-bearing today.** It overturns **33%** of abstract-included studies (6,835
exclusions of 20,810 screened). It is not a rubber stamp, so replacing it is a real substitution, not
a free saving.

**But the grounds for overturning are overwhelmingly field-like.** Mapping each exclusion reason to
the schema field that would have caught it:

| proposed field | exclusions it absorbs | share |
|---|---|---|
| `analysis.contrast_identity` — is this the required contrast? | 3,560 | **52%** |
| `analysis.spatial_scope` — whole-brain vs ROI-only | 2,713 | **40%** |
| `study.imaging_modality` — fMRI / PET / DTI / rs-fMRI / structural | 2,206 | 32% |
| `analysis.coordinate_space` — coordinates reported at all | 650 | 10% |
| `analysis.stimulus_modality` — visual / auditory / taste / imagery | 379 | 6% |
| `study.publication_type` — review, protocol, commentary | 234 | 3% |
| `analysis.level` — group vs single-subject | 146 | 2% |
| `study.sample_size` | 71 | 1% |
| `analysis.survived_correction` | 62 | 1% |
| **absorbed by >= 1 field** | **5,582** | **82%** |
| **residue — needs prose judgment** | **1,253** | **18%** |

Two things stand out. `spatial_scope` is a **single boolean that absorbs 40% of full-text
exclusions** — whole-brain versus ROI-only is the highest-leverage field in the list and the easiest
to extract. And `contrast_identity` at 52% is precisely analysis-level metadata, which is why the
analysis-level half of the NeuroStore work matters more than the paper-level half.

**What the 18% residue actually is:** topical scope, not missing facts. *"Does the Monetary Incentive
Delay task count as cue-reactivity?"* *"Does a slot-machine gambling task count as a drug cue?"* Those
are judgments about whether a study's construct matches the review's construct.

But note the bar. **Fields do not need to decide; they need to carry enough for the LLM to decide.**
Every residue example names its task (`MID`, `slot machine`, `monetary reward expectation`) — so a
`task` or `construct` field would let a screener adjudicate from a 175-token record instead of 13,000
tokens of full text. The residue is *compressible* even where it is not *decidable*.

(Classification is keyword-based against written reasons, with patterns verified by sampling. It both
over- and under-counts — an earlier pass put ROI-only at 31% because it missed phrasings like
"reports only region-of-interest analyses". Treat the percentages as indicative and the ranking as
solid. Many criteria here are cue_reactivity-flavoured, so the specific fields are partly
project-specific; the *classes* generalise.)

### The cost case is strong, and stronger than the cost case

Median pubget article: **13,416 tokens**. A structured record covering the fields above plus contrast
names for a handful of analyses: **~175 tokens**. That is **77x compression**. Across the 20,810
full-text screenings run in this project:

| | input tokens |
|---|---|
| as full text | ~279M |
| as structured records | ~3.6M |

The under-stated benefit is not cost but **capability**: 50 structured records fit in one context
where 50 full texts cannot. That enables *comparative* screening — judging a study against its
neighbours rather than in isolation — which is unavailable at any price with full text.

### The counter-argument to discarding full text

The corpus's **41.5% opaque analyses are themselves the bill for not being able to re-extract.** They
were parsed under a thinner schema, and there is now no cheap way back. `analysis-schema` is at
`0.1.0-alpha.10` and still moving; every alpha changes what the evidence layer should contain.
Discarding source text freezes the layer at whichever schema version happened to be current.

The resolution is narrower than "store everything":

- **Structured layer is the serving substrate.** Screening, filtering, and querying run on it. This
  is right, and the 82% figure supports it.
- **Open-access full text need not be stored** — it is re-fetchable on demand when the schema changes.
- **User-uploaded non-OA full text should be retained** (private to uploader), because it is the only
  copy that will ever exist and the schema *will* change. This is exactly the Tier 2 case, and it is
  the one place the retention question actually bites.

### Consequence for validity, not just cost

The 33% overturn rate cuts both ways. If the structured layer is missing a field that a criterion
depends on, screening does not fail loudly — it **silently includes** studies full text would have
excluded. So field coverage is a validity property, and the audit that matters before Tier 0 goes
public is: for a given review's criteria, is every criterion expressible in available fields? If not,
that review needs Tier 1 or 2, and the service should say so rather than returning a confident map.

### Which of Tier 1 and Tier 2 to build first

**Tier 2, clearly** — and the analysis-label finding sharpens why:

- Extraction runs on one code version, so quality is consistent and **retroactively improvable**.
  Stored PDFs are the substrate for the repair job above, and for adding body-text coordinates
  (+30-36%) later. Tier 1 freezes extraction quality at whatever client version the user happened to
  run, and re-extraction means asking users to re-run.
- No client to distribute, version, or support across platforms.
- Tier 1's only real advantage is automated *fetching*, which the copyright note above establishes is
  a convenience rather than a necessity.

This does imply a retention decision: keep uploaded PDFs (private to uploader) so re-extraction is
possible, rather than discarding them after a single pass. Worth making deliberately.

### Two products, not one service

Tier 0 is not a degraded systematic review; it is a different thing. A 60%-recall map is perfectly
good for hypothesis generation, for deciding whether a question is worth pursuing, for a grant
figure. It is not PRISMA-complete and should never be labelled as though it were. Tiers 1 and 2 are
the publishable-review path. Conflating them is the main product risk here, and the fix is honest
labelling of the recall basis on every output — which the corpus-snapshot provenance requirement
above already demands.

### Where the pooling boundary already sits

Worth noting that NeuroStore is already on the right side of the line and has been: it pools
coordinates, analysis metadata, and **abstracts** (the `description` field on a base study is the
paper's abstract). Abstract redistribution is settled practice — PubMed does it. So the boundary in
operation is: coordinates, analysis records, and abstracts are poolable; full text and verbatim
tables are not. That is a clean, already-established line rather than a new policy to invent.

### PDF and table extraction tooling

Relevant to Tier 2 and to opportunistic PDF sources (Semantic Scholar and similar give PDFs but weak
or absent tables — S2ORC is GROBID-derived, and tables are GROBID's known soft spot, so that gap is
inherited rather than incidental). Supplementary PDFs are the same problem: autonima#36, measured at
**~25% of otherwise-eligible studies** in the schizophrenia pilot.

Note first that this is **not on the current critical path**. The corpus on disk is 40,534 XML files
and **zero PDFs** — pubget returns JATS, Elsevier returns XML. For that path the right tool is a fast
XML parser (`quick-xml` in Rust, `lxml` in Python), and the in-text coordinate work (+30-36%) is regex
over already-structured markup with no PDF parsing and no models at all.

#### The free benchmark we already have

For any PMC OA article held as XML, the PDF can be fetched and extraction scored against the
XML-derived table **exactly, with no hand labelling**. In the two experiments alone:

| | n |
|---|---|
| pubget articles on disk | 564 |
| with a coordinate table in the XML | 132 |
| redistributable CC licence | 545 |
| **both — PDF fetchable, XML ground truth** | **130** |

Thousands corpus-wide. This should be built before any tool is chosen, because it converts the whole
question from argument into measurement.

#### Fetching the PDFs: which routes actually work

Tested 2026-09-01, because most of the documented ones are dead:

| route | result |
|---|---|
| PMC OA Web Service (`oa.fcgi`) | **404** on both `www.ncbi.nlm.nih.gov` and `pmc.ncbi.nlm.nih.gov` |
| PMC FTP dataset tree | **emptied August 2026**; `/pub/pmc/` now holds only `PMC-ids.csv.gz` |
| `pmc.ncbi.nlm.nih.gov/articles/PMC*/pdf/` | HTTP 200 but returns a "Preparing to download ..." bot-mitigation interstitial, not a PDF |
| Europe PMC `fullTextPDF` | **0 of 30** candidates returned a PDF |
| **PMC Cloud Service (AWS Open Data)** | **25 of 25.** No login, no key, HTTPS or S3 |

`https://pmc-oa-opendata.s3.amazonaws.com/` is the current sanctioned route and the FTP readme
now points at it explicitly. Objects are keyed directly by PMCID and version
(`PMC3434213.1/PMC3434213.1.pdf`), so no lookup table is needed — though versions matter, since a
`.2` is typically the publisher's typeset copy replacing an author manuscript, and the highest
version should win.

**The bonus finding is worth more than the PDFs.** Each record also carries **one JPEG per figure
and one per table**, pre-cropped by PMC:

| | share of 25 sampled |
|---|---|
| PDF present | 100% |
| >= 1 table image | 40% (36 images) |
| >= 1 figure image | 88% (89 images) |

That removes a whole pipeline stage from two directions at once. Table-image extraction skips PDF
page segmentation entirely — feed the crop straight to a vision model. And the figure-extraction
direction (supply-side item 5 above), whose main objection was that no cheap validation set exists,
now has one: figures already isolated, already paired with the article's XML coordinates.

For non-PMC sources the ladder is different and untested here: Unpaywall and OpenAlex both expose
`best_oa_location.pdf_url` free and keyless, Crossref carries publisher TDM links
(`intended-application: text-mining`) which is the standard cross-publisher discovery route, and
Wiley and Springer both run TDM APIs that return PDFs to entitled institutions. All of those inherit
the IP-entitlement constraint measured above.

#### The reframe that may remove the need for a model

The downstream consumer is an LLM (`CoordinateParsingClient`), not a schema validator. So what is
needed is not *table structure recognition* but **enough spatial fidelity for an LLM to reconstruct
rows**. Coordinate tables are unusually forgiving here: the target is `(label, x, y, z)` where x/y/z
are three small integers on a line — numerically distinctive, and robust to imperfect cell
segmentation in a way a financial table would not be.

So test the cheap path first.

| tier | approach | tools | cost |
|---|---|---|---|
| **A** | geometry-preserving text, no ML | `pdfium-render` (Rust, permissive), `mutool draw -F stext` (C, AGPL), `pdftotext -bbox-layout` (C++, GPL) | ms/page, no GPU |
| **B** | rule-based table detection | `pdfplumber`, `camelot`, `tabula-java` | slow, CPU |
| **C** | ML table structure | Docling/TableFormer, `marker`/Surya, **Table Transformer** (ONNX-exportable), PaddleOCR PP-Structure | GPU preferred |

Tier B is likely to underperform here specifically: neuroimaging coordinate tables are frequently
**unruled**, whitespace-aligned with no borders, which is exactly where lattice methods fail.

#### The honest Rust position

**There is no mature Rust equivalent of Docling for table structure.** Rust's real contributions are
`pdfium-render` for fast text-plus-geometry and **`ort`** (ONNX Runtime bindings) for running models
without Python overhead. The pragmatic Rust architecture is therefore `pdfium-render` + `ort` running
Table Transformer or PP-Structure weights. `ferrules` is the closest packaged attempt and
`extractous` the closest general-purpose one; both are worth verifying against current sources before
being relied on — this note is working from recollection on those two specifically.

Also worth remembering that for Docling-class tools **the language is not the bottleneck** — layout
and table models are. A Rust rewrite of orchestration buys little; a faster ONNX execution provider
buys a lot.

#### Two source-specific notes

- **Supplementary material is often not a typeset PDF.** It is frequently an author-produced Excel or
  Word export. Try the native format first — `calamine` (Rust) reads xlsx very fast, `docx-rs` for
  Word — since author tables tend to be *cleaner* than publisher typesetting. This may sidestep PDF
  parsing entirely for a share of autonima#36.
- **Pre-2000 papers may be image-only**, needing OCR regardless of tool (Tesseract, Surya). That
  intersects the pre-2004 coverage hole measured above (34% with coordinates), so it is the same
  shrinking slice twice over.

#### Ordering

1. Build the paired benchmark. Free, and every later decision depends on it.
2. Test Tier A plus the existing LLM parser. If it recovers >= 90% of coordinates, stop — no model
   is needed.
3. Escalate to Tier C only for the measured residue.

Licence note for a hosted service: PDFium is permissive; MuPDF and Poppler are AGPL/GPL.

### Verdicts on the interface options

- **Neurosynth Compose** — the *destination and criteria-sharing surface*, and the natural home for
  the hosted tier, since it already has the studyset/annotation data model.
- **MCP** — yes, for **querying** the corpus ("every patient>control reward contrast with n>50").
  That is direction 4 above, and it is a natural fit. Not for launching runs.
- **Chatbot** — yes, for **one step only**: authoring and piloting inclusion criteria. Criteria
  phrasing dominates outcomes (the TLE v1->v2 rewrite; redundant `MRI` and `Humans[MeSH]` clauses
  costing 64% of a search pool), and it is genuinely iterative. It should emit a YAML config, not run
  the pipeline.
- **Local GUI** — keep, scoped to QC and adjudication of borderline calls and parsed tables.
- **CLI + versioned YAML stays ground truth.** PRISMA needs a citable, diffable, re-runnable artifact;
  `execution_manifest.json` plus a git-tracked config is that, and a chat transcript is not.

### Open problem: provenance

"I filtered a corpus snapshot" is a weaker PRISMA identification statement than "I ran this PubMed
query on this date". It is defensible only if the snapshot is **versioned and citable**, and if the
corpus's own inclusion criteria are documented, since they become an inherited and largely invisible
eligibility filter. Worth settling before the hosted tier is public. Two consequences also travel with
the corpus: its parsing quality (including autonima#60, duplicate tables) and its table-only blind
spot — the 16.1% body-text-only share measured above is invisible to a table-derived corpus.

### Build order

1. **Criteria-authoring assistant** — removes the actual adoption barrier, and is cheap.
2. **Compose push as the output contract** — defines the boundary everything else plugs into.
3. **Tier 0 hosted execution**, corpus-only, advertised with its 60% study ceiling and 56%
   analysis-filterability, plus the era profile. Cheapest thing to ship that anyone can use.
4. **Corpus repair / richer evidence layer** — re-extract labels for the opaque 41.5%, and add the
   fields ranked above. Raises the Tier 0 ceiling for everyone with no new retrieval, and is a
   prerequisite for Tier 0 being good rather than merely cheap. Already under way on the NeuroStore
   side. `analysis.spatial_scope` first: one boolean, 40% of full-text exclusions.
5. **Tier 2: user PDF upload**, private to uploader, extraction hosted. Cheap, well-precedented,
   starts the corpus flywheel without shipping a client.
6. **MCP query server** over the resulting corpus, with cached annotation columns.
7. **Tier 1: local companion** — automates the fetching that step 5 makes users do by hand.
   Convenience, not prerequisite.

---

## Appendix: which other domains, and why

Four criteria. A candidate has to clear all of them, and the fourth is the one that eliminates most:

1. **Result units inside tables**, several per paper -- so analysis-level selection is load-bearing
   rather than a nicety.
2. **A common space to pool into** -- whatever plays the role MNI plays for us.
3. **Large, growing literature.**
4. **A synthesis deficit.** Not merely a big literature: a big literature that is *not already being
   synthesised*. This is the criterion that kills most candidates.

### Measuring criterion 4

Meta-analyses per 1,000 papers is a serviceable proxy for whether a field already has a working
synthesis solution. From PubMed, 2000 onward:

| domain | papers | meta-analyses | **MAs per 1,000** |
|---|---|---|---|
| **preclinical animal neuroscience** | 123,453 | 149 | **1.2** |
| **behavioral pharmacology** | 8,900 | 12 | **1.3** |
| EEG / ERP components | 9,717 | 69 | 7 |
| candidate-gene association | 10,336 | 377 | 36 |
| epidemiology risk factors | 177,879 | 8,214 | 46 |

Two eliminations follow immediately. **Epidemiology** has the largest literature of any candidate and
the highest synthesis density -- the field already meta-analyses itself, and Cochrane-adjacent
tooling exists. **Genetics** fails for a different reason: GWAS Catalog already solved the curated
part, and while the candidate-gene literature is not covered by it, the field has moved on.

Also eliminated on prior knowledge rather than these numbers: clinical trials (RobotReviewer,
Trialstreamer, Cochrane infrastructure), and omics (GEO, Expression Atlas). Both are well served.

### Top pick — preclinical animal neuroscience

**123,453 papers against 149 meta-analyses: a 38x lower synthesis density than epidemiology, on a
literature nearly as large.** The deficit is acknowledged inside the field -- CAMARADES and SYRCLE do
this by hand and reviews take years.

It fits the profile:

- outcomes live in tables (group, dose, n, mean +/- SEM), several comparisons per paper
- the pooling space is the standardised mean difference rather than a coordinate system
- analysis-level precision is exactly what is scarce, because a single paper reports many arms
- it is adjacent to this project's domain, so the screening and annotation machinery transfers
- it carries real weight: preclinical-to-clinical translation failure is a named crisis and poor
  synthesis is part of the reason

**The honest difficulty** is outcome heterogeneity -- forced swim time, escape latency, lever presses
-- with no shared ontology. That is a normalisation problem rather than an extraction problem, which
is what LLM extraction is good at, and it is the same shape as the ROI-name-to-atlas mapping in
direction 1 above. If direction 1 works, this works.

### Runner-up — EEG / ERP

The closest structural analogue to the current work: **component x electrode x latency window is a
coordinate space**, and no Neurosynth equivalent exists for ERP. Smaller (9,717 papers) and less
standardised in reporting, but the infrastructure transfers almost directly -- which makes it the
cheapest available test of whether the approach generalises beyond fMRI at all, rather than the most
valuable place to apply it.

### Narrower slice worth noting — behavioral pharmacology

8,900 papers and 12 meta-analyses (1.3 per 1,000). A cleaner subset of the top pick: dose-response
data is unusually tabular and unusually poolable, and the deficit is just as stark. A good pilot
target *within* preclinical work, for the same reason left/right TLE was an attractive pilot within
epilepsy -- tight scope, unambiguous structure.

### Caveat on the metric

Low meta-analysis density can mean "nobody can afford to synthesise this" or "this is not
meta-analysable". For preclinical work the evidence points at the former -- people do it manually and
slowly, and say so -- but that is an inference from the field's own complaints, not something
measured here. Before committing to any of these, the check worth doing is whether a handful of
existing manual reviews in the domain could serve as validation benchmarks, the way the nine
neuroimaging meta-analyses did for this project. A domain with no benchmark is a domain where nothing
can be validated.

# Beyond CBMA — where automated meta-analysis goes next

Strategy note, 2026-08-31. Written after the deactivation and ENIGMA pilots, and grounded in
measurements from those runs rather than speculation. Nothing here is committed work.

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
retrieval problem**: no entitlement, no hours-long runs, no PDF redistribution exposure.

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

1. **It is legally clean.** The companion uploads *extracted* content — tables, coordinates, analysis
   records — not publisher PDFs. Hosting user-uploaded PDFs is redistribution; ingesting derived
   coordinates is what NeuroStore already does.
2. **The fixed corpus stops being fixed.** Every local run raises the ceiling for every hosted user,
   and it raises it preferentially at the recent end — precisely the ingestion-lag deficit above.
3. **The retrieval problem is solved once, by whoever has the entitlement**, rather than by the
   service pretending to have it.

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
3. **Local companion / uploader** — starts the corpus flywheel.
4. **MCP query server** over the resulting corpus.
5. **Hosted execution**, corpus-only, advertised with its 60% ceiling and its era profile.

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

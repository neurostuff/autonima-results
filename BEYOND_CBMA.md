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
contrasts" would have standalone value.

### 4. Stop treating the map as the deliverable

The real capability is **evidence structuring**. The output could be a queryable analysis-level
database — *"every patient>control contrast in a reward task with n>50"* — with meta-analysis as one
query type among many. That is NeuroStore's mission, and autonima is the ingestion engine for it.
See autonima#6 and #27.

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

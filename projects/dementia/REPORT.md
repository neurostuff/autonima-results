# Dementia (bvFTD) case study: screening and meta-analytic performance

Benchmark: Tahmasian et al., bvFTD coordinate-based meta-analysis (PMID 35664889).

---

## Headline

**Given the same input studies, LLM screening follows stated criteria far more accurately
than the standard PubMed-search setup suggests — and that precision carries through
annotation to produce meta-analytic maps closer to the manual result than not using
autonima at all.**

This is the only project where that can be shown, because it is the only one whose
benchmark publishes every screened-and-rejected study together with the rejection reason.
That fixes the candidate pool (removing search recall from the measurement entirely) and
makes it possible to separate genuine eligibility errors from disagreements about data
availability.

**Precision is much higher on a fixed pool.** Holding the criteria constant and changing
only the input from a PubMed query to the fixed 558-PMID list, full-text screening
precision rises **0.395 → 0.508**. And because the rejection reasons let us identify which
apparent false positives were in fact eligibility-correct — studies the raters agreed
qualified but excluded for reporting no usable coordinates — the meaningful figure is higher
still: **0.66–0.76 adjusted precision at screening, and 0.89–0.95 end to end**, versus a
best of 0.593 and a median near 0.36 across the eight PubMed-search projects. Most of what
looks like imprecision in the standard setup is search noise and data availability, not the
model failing to follow criteria.

**That precision cascades into beyond-baseline map performance.** The right counterfactual
is `all_studies` — meta-analysing every study whose coordinates you could extract, with no
targeted annotation. v4 beats it on all four columns in R² (mean **+0.017**; `dementia_all`
R² 0.497 vs baseline 0.468), the only one of five schemas to do so. Screening precision
alone is not sufficient: the two stages have to work together, and the schema that wins is
the one balancing precision against recall rather than maximising either.

**The absolute similarities and the margins are both small, for reasons largely outside the
schemas' control.**

- **Only 49% of gold studies reach the analysis stage.** 93% are obtained as full text and
  91% pass screening, but just 36 of 74 yield an extractable coordinate. That 44-point drop
  is larger than every schema effect in this report combined, and no criteria change touches
  it (§2.3b).
- **The comparison is not strictly fair.** The benchmark merges papers that share a cohort
  into single independent samples — 29 entries covering 70 PMIDs, one bundling 27 papers
  into one n=19 sample — and the pooling happened before coordinates were recorded, so no
  per-paper attribution exists. Our unit is the paper; theirs is the sample. The pools are
  not commensurable and no availability filtering can make them so (§4.1).
- **The margin over doing nothing is thin and fragile.** v4's +0.017 mean R² is real and
  consistent in sign, but four of five schemas are *net negative* against the same baseline,
  and a poor schema is far worse than no annotation (S1, −0.258). Read this as evidence that
  a good schema clears the bar, not that annotation is transformative.

Taken together: high adjusted precision on a controlled pool is the solid result, and it
does propagate to better-than-baseline maps. The ceiling on how much better is set by
coordinate availability and by the benchmark's own structure, not by the criteria.

---

## 1. Why this case is different

Every other project in this evaluation starts from a PubMed query. That makes the
comparison against the published meta-analysis irreducibly confounded: we cannot know
whether a study we missed was missed by our *screening* or was never *retrieved* in the
first place, and PubMed result sets drift over time, so the candidate pool is not even
stable between our own runs.

The dementia case is the one exception. Its supplement tabulates **every study the authors
screened and rejected, together with the rejection reason** — 495 screened records, which
after adjudication against our own list resolves to a fixed pool of **558 PMIDs**. That
single property buys two things:

1. **A pool both processes demonstrably saw.** This is the important one. Because the input
   is a fixed PMID list rather than a query, search recall is removed from the measurement
   entirely. Any disagreement is a screening disagreement, not a retrieval artifact. No
   other project in this set can guarantee that.

2. **Decision-level comparison.** Because each rejection carries a reason (`Not bvFTD`,
   `ROI`, `Data Not Reported`, …), we can ask not just *whether* we disagreed with the
   manual screen but *on what grounds*, and separate genuine eligibility errors from
   disagreements about data availability.

So this project is where we can make the cleanest possible statement about screening
quality: given the same candidate pool, how closely does automated screening reproduce
expert screening?

### Two gold standards

The reasons make a second, looser gold standard available, and both are reported
throughout because they answer different questions.

| | definition | n |
|---|---|---|
| **strict gold** | studies the manual actually included in the meta-analysis | 74 |
| **adjusted gold** | strict gold **+** studies excluded *only* for `Data Not Reported` | 162 |

`Data Not Reported` means the study met every eligibility criterion but did not report
usable coordinates for the required contrast. Selecting such a study is not an eligibility
error, so counting it as a false positive measures data availability rather than screening
judgement. Precision is more interpretable on the adjusted gold; recall is more
interpretable on the strict gold, since only those 74 studies actually contributed
coordinates.

---

## 2. Screening performance

### 2.1 How the schemas were developed, and when Claude entered

There are **three distinct screening schemas** across five run directories. This is easy to
misread from the version numbers, so it is worth stating plainly:

| schema | runs | how it was written |
|---|---|---|
| **S1** | `v1` | Transcribed as close to verbatim from the paper's stated criteria as possible. |
| **S2** | `v2`, `v3` | Revised full-text inclusion using our own reasoning about what the criteria *should* say. `v3` changed **only** the annotation section — its screening criteria are byte-identical to `v2`. |
| **S3** | `v4`, `v5` | Claude-authored, driven by inspection of actual v3 failures. `v5` changed **only** annotation — its screening is byte-identical to `v4`. |

The methodological line worth drawing is **between S2 and S3**, and it is a line about
provenance, not about quality:

- **S1 and S2 were written essentially independently of the outputs.** We looked at how
  each did, but the revisions were reasoned *a priori* — for S1 from the paper's own
  wording, for S2 from our judgement about what those criteria were trying to express.
- **S3 was derived from observed failures.** Claude read the specific studies v3 got wrong
  and wrote rules targeting them. From this point on, the schema is a function of the
  previous run's outputs on this benchmark.

That dependence is a real risk of overfitting to the evaluation set, and it was managed
explicitly rather than assumed away. Rules were required to be stated by *kind* rather than
by instance: when the disease-entity errors were all Parkinson's, MSA, PSP and pathology
subtypes, the rule that went in was "a different clinical diagnosis, or a group defined by
neuropathology or genotype rather than by a clinical bvFTD diagnosis" — never a list of the
specific diseases observed. A first attempt that enumerated them was rejected on exactly
these grounds.

### 2.2 Results

All five runs were executed against the same 558-PMID list (`v1` and `v2` were re-run for
this purpose on 2026-08-20), and all five judge an identical pool: 544 studies screened, 14
skipped for having no retrievable abstract.

Abstract screening is effectively identical across all versions — ~347 included, 73/74 gold
recovered, strict recall 0.99, adjusted recall 0.93. It is permissive by design and is not
where the versions differ. Full-text screening:

`adj.` columns score against the adjusted gold (162). Note that `adj. recall` is recall
over all 162 — a *harder* question than strict recall, not a lenient one (see §2.3b). The
final column applies both corrections in their intended directions: adjusted-gold precision
with an attainable denominator (gold studies actually screened at full text).

| run | schema | included | prec | recall | F1 | | adj. prec | adj. recall | adj. F1 | | **adj+att recall** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v1` | S1 | 182 | 0.37 | 0.91 | 0.52 | | 0.66 | **0.75** | **0.70** | | **0.85** |
| `v2` | S2 | 139 | 0.48 | 0.91 | **0.63** | | **0.76** | 0.65 | **0.70** | | 0.74 |
| `v3` | S2 | 141 | 0.47 | 0.89 | 0.61 | | 0.74 | 0.64 | 0.69 | | 0.73 |
| `v4` | S3 | 157 | 0.43 | 0.91 | 0.58 | | 0.69 | 0.67 | 0.68 | | 0.76 |
| `v5` | S3 | 157 | 0.43 | 0.91 | 0.58 | | 0.69 | 0.67 | 0.68 | | 0.76 |

On strict gold with an attainable denominator, every schema scores **0.96–0.97** — of the
gold studies whose full text we obtained, screening recovers essentially all of them.

**S1 → S2 was a clear improvement.** Precision rose 0.37 → 0.48 at unchanged recall (0.91),
lifting strict F1 from 0.52 to 0.63 — it dropped 43 false positives without losing a single
gold study. Writing the criteria as we understood them beat transcribing the paper's
wording: the paper's prose bundles several requirements into single sentences, which the
model satisfied by finding *any* one of them.

**S2 → S3 slightly reduced screening F1.** Comparing the two same-vintage runs (`v3` and
`v4`): strict F1 0.61 → 0.58, adjusted 0.69 → 0.68. S3 admits 16 more studies (141 → 157)
at lower precision, recovering one more gold study.

Two things are worth reading off the adjusted-gold columns, because strict F1 alone is
misleading here. First, **all three schemas land within 0.02 of each other on adjusted F1**
(0.70 / 0.69 / 0.68) — so on the eligibility standard the manual actually applied, the
schemas are near-equivalent and the strict-F1 ordering is driven mostly by how many
eligible-but-dataless studies each admits. Second, they differ in *where* they sit on the
trade-off:

| schema | adj. precision | adj. recall | character |
|---|---|---|---|
| S1 | 0.66 | 0.75 | recall-leaning |
| S2 | 0.76 | 0.65 | precision-leaning |
| S3 | 0.69 | 0.67 | balanced |

S2 bought precision by giving up recall; S3 moved back toward recall. Under this project's
stated preference — a missed study is unrecoverable downstream, whereas a study admitted
without usable coordinates is filtered later at annotation — S3's balance is the preferable
operating point even at marginally lower F1. S3 was not built to raise screening F1, and
the fact that it did not is not by itself a failure of the schema.

This is also why the screening-stage table should not be read as the verdict on the
schemas. Measured end to end — did the study contribute an eligible *analysis* — the
ordering is monotonic in version and S1 collapses (§3), and on the resulting
meta-analytic maps v4 is the only configuration that beats the no-annotation baselines
(§4). The three metrics disagree, and §3 explains why.

### 2.3 Three caveats on reading any of these numbers

**(a) Screening cannot be evaluated at the screening stage alone.** This is the most
important qualification in the report. The manual screen's decision bundles two judgements
that our pipeline makes at two different stages: *is this study eligible*, and *does it
report an eligible analysis*. Our screening stage only answers the first. Whether an
eligible analysis actually exists is decided later, by annotation, over the parsed
analyses. Consequently a study we "correctly" include at screening can still be lost at
annotation, and a study we include that the manual rejected for `Data Not Reported` is not
a screening error at all. Screening-stage precision against the strict gold is therefore a
lower bound on screening quality, and the adjusted-gold column is the fairer read.

**(b) True recall is capped by coordinate availability.** Some studies will always be
missing coordinates — no full text retrievable, or coordinates present only in a form we
cannot parse. That ceiling is independent of any schema. The attrition funnel across the 74
gold studies (identical for all three re-run versions except at the final step):

| stage | gold retained |
|---|---|
| present in the 558-PMID input | 74 (100%) |
| passing abstract screening | 73 (99%) |
| full text obtained (screened at full text) | 69 (93%) |
| passing full-text screening | 67 (91%) |
| **≥1 coordinate extracted (analysis stage)** | **36 (49%)** |
| entering the meta-analysis (`dementia_all`, v4) | 33 (45%) |

Screening is not the bottleneck. Only 5 gold studies (7%) were never obtained as full text,
and of the 69 we did obtain, full-text screening keeps 67 — an attainable recall of **0.97**.
**The cliff is coordinate extraction: 93% → 49%.** It is identical across all
configurations because it is upstream of anything the schemas control. **This is why recall
should be read as attainable recall** — over the studies for which the data actually exists
— and on that basis the pipeline does well (§3).

Note that the two adjustments used in this report are independent and correct opposite
biases, so they belong on opposite sides of the ledger:

- the **adjusted gold** (§1) fixes *precision* — it stops counting a correct eligibility
  call on a dataless study as an error;
- an **attainable denominator** fixes *recall* — it stops counting data we could never have
  obtained as a miss.

Using the adjusted gold as a *recall* denominator is therefore not a fairness correction at
all but a harder question ("did you find all 162 eligible studies?", 88 of which have no
extractable data). The `adj. recall` column in §2.2 is that harder question, which is why
it is *lower* than strict recall. Applying both corrections in the intended direction —
adjusted-gold precision with an attainable denominator — full-text screening runs
0.66–0.76 precision at 0.73–0.85 recall across the three schemas.

**(c) Run-to-run variance is non-trivial.** `v2` and `v3` have byte-identical screening
criteria and the same underlying model, differing only by a routing prefix on the model
name. Their decisions still diverge on **8.6% of abstracts (47/544)** and **3.7% of
full texts (11/298)**. Differences smaller than a few F1 points should not be attributed to
schema changes — which applies directly to the S2 → S3 comparison above, whose strict-F1
gap of 0.03 is within this range. (`v4` vs `v5` flipped 0/544 and 0/325, as expected — v5
reused v4's screening cache, confirming the variance is resampling noise, not drift.)

### 2.4 Cross-project context

Against the other projects, dementia screening is strong. On the fixed-pool (`*-allstudies`)
runs, which is the like-for-like comparison, only two projects have them:

| project | run | recall | precision | F1 |
|---|---|---|---|---|
| **dementia** | `v3-allstudies` | 0.890 | 0.508 | **0.647** |
| emotional_regulation_2022 | `v3-allstudies` | 0.216 | 0.576 | 0.314 |

On the canonical PubMed-search runs — where search recall is folded in, so numbers are
lower everywhere — dementia ranks third of eight on F1 and second on recall:

| project | run | recall | precision | F1 |
|---|---|---|---|---|
| vbm_of_ptsd | `v1` | 0.762 | 0.593 | 0.667 |
| social | `v3-search-all_pmids-…` | 0.971 | 0.407 | 0.573 |
| **dementia** | `v3` | **0.849** | 0.395 | **0.539** |
| vbm_of_substance_use | `v2` | 0.747 | 0.450 | 0.562 |
| cue_reactivity | `v5` | 0.717 | 0.322 | 0.444 |
| problem_solving | `v1` | 0.397 | 0.182 | 0.249 |
| decision_making | `v2` | 0.542 | 0.192 | 0.283 |
| executive_function | `v1` | 0.327 | 0.095 | 0.147 |

The headline claim the dementia case supports: **given a candidate pool both processes
saw, automated full-text screening recovers 91% of the studies an expert team included,
and 65–75% of everything they judged eligible, at 0.66–0.76 precision against that
eligibility standard — and this holds across three independently authored schemas.**

---

## 3. End-to-end selection: does the study contribute an eligible *analysis*?

This is the metric caveat (a) argues for, and it is the one that corresponds to what the
manual screen actually decided. A study counts as selected only if **at least one of its
parsed analyses was included in a substantive annotation column** (`dementia_all`,
`dementia_decreasing_activity`, `functional`, `structural` — the `all_*` baselines
excluded). Screening alone cannot answer this, because whether an eligible analysis exists
is decided downstream by annotation.

Recall is reported as **attainable recall** — over gold studies for which this run actually
holds coordinates — per caveat (b). True recall is shown alongside to keep the availability
ceiling visible.

**Strict gold (74):**

| run | schema | coord-bearing | selected | TP | FP | precision | true recall | attainable recall | F1 |
|---|---|---|---|---|---|---|---|---|---|
| `v1` | S1 | 119 | 33 | 15 | 18 | 0.45 | 0.20 | 0.42 (15/36) | 0.43 |
| `v2` | S2 | 120 | 37 | 27 | 10 | **0.73** | 0.36 | 0.75 (27/36) | 0.74 |
| `v3` | S2 | 120 | 41 | 30 | 11 | **0.73** | 0.41 | 0.83 (30/36) | **0.78** |
| `v4` | S3 | 121 | 44 | 31 | 13 | 0.70 | 0.42 | 0.86 (31/36) | **0.78** |
| `v5` | S3 | 121 | 50 | 32 | 18 | 0.64 | 0.43 | **0.89** (32/36) | 0.74 |

**Adjusted gold (162):**

| run | schema | selected | TP | FP | precision | true recall | attainable recall | F1 |
|---|---|---|---|---|---|---|---|---|
| `v1` | S1 | 33 | 30 | 3 | 0.91 | 0.19 | 0.41 (30/73) | 0.57 |
| `v2` | S2 | 37 | 35 | 2 | **0.95** | 0.22 | 0.48 (35/73) | 0.64 |
| `v3` | S2 | 41 | 38 | 3 | 0.93 | 0.23 | 0.52 (38/73) | 0.67 |
| `v4` | S3 | 44 | 39 | 5 | 0.89 | 0.24 | 0.53 (39/73) | 0.67 |
| `v5` | S3 | 50 | 45 | 5 | 0.90 | 0.28 | **0.62** (45/73) | **0.73** |

Two results stand out.

**End-to-end, the progression is monotonic — unlike at the screening stage.** Attainable
recall rises at every version: 0.42 → 0.75 → 0.83 → 0.86 → 0.89 on strict gold, and
0.41 → 0.48 → 0.52 → 0.53 → 0.62 on adjusted gold, with precision holding near 0.90
throughout the latter. Adjusted F1 climbs 0.57 → 0.73. **S1's end-to-end collapse
(precision 0.45, attainable recall 0.42 on strict gold) is invisible at the screening
stage**, where it scored a respectable 0.91 recall — it passed studies through and then
failed to identify an eligible analysis in them.

**The gains come from the annotation criteria, not the screening criteria.** Because three
of the four transitions vary exactly one stage, the improvement can be attributed:

| transition | varies | adj. F1 | attainable recall (strict) |
|---|---|---|---|
| v1 → v2 | both stages | 0.57 → 0.64 | 0.42 → 0.75 |
| v2 → v3 | **annotation only** | 0.64 → 0.67 | 0.75 → 0.83 |
| v3 → v4 | **screening only** (S2 → S3) | 0.67 → 0.67 | 0.83 → 0.86 |
| v4 → v5 | **annotation only** | 0.67 → **0.73** | 0.86 → **0.89** |

The one transition that isolates the screening schema (v3 → v4) is flat on F1 and gains
+0.03 attainable recall at a 0.03 precision cost — consistent with §2.2, where S2 → S3 was
also approximately neutral, and consistent with its permissive-by-design intent. Every
substantive end-to-end gain traces to an annotation change.

**A note on denominators.** An earlier draft of this section reported v4/v5 recall over the
61 studies *judged by both runs*, giving 0.76 and 0.90. The tables above use the larger and
more honest denominator — all gold studies for which the run holds coordinates (36 strict,
73 adjusted) — which includes coordinate-bearing studies that received no annotation
decision at all. The precision figures are essentially unchanged; the recall figures are
lower and should be preferred.

Note the disagreement between this section and §4: **v5 is the best configuration for
study-level selection, and v4 is the best for the resulting meta-analytic maps.**

---

## 4. Meta-analytic results

### 4.1 Two limitations specific to this benchmark

**Merged samples.** The published benchmark deduplicates data reuse: where several papers
report the same underlying cohort, the authors merged them into a single independent sample
so the sample is weighted once. This is why the gold NiMADS has comma-joined PMID lists —
29 entries covering 70 PMIDs, one of which bundles **27 papers into a single n=19 sample**.
The pooling happened *before* coordinates were recorded (the raw spreadsheet's `Experiment`
column is already the merged string), so there is no per-paper coordinate attribution
anywhere in the chain. Our unit is the paper; theirs is the independent sample. The pools
are therefore not directly commensurable, and no availability-based filtering of the manual
side can fix it.

**Half the studies.** Per the funnel in §2.3, only 45% of the gold studies reach our
meta-analysis.

### 4.2 Results

All runs scored against the same manual maps, so the comparison between them is valid even
though the absolute values carry the limitations above.

**Dice:**

| column | v1/S1 | v2/S2 | v3/S2 | v4/S3 | v5/S3 |
|---|---|---|---|---|---|
| `dementia_all` | 0.205 | 0.399 | 0.436 | **0.504** | 0.453 |
| `dementia_decreasing_activity` | 0.077 | **0.448** | 0.419 | 0.445 | 0.400 |
| `functional` | 0.031 | **0.370** | 0.289 | 0.188 | 0.137 |
| `structural` | 0.136 | 0.313 | 0.327 | **0.360** | 0.325 |
| **mean** | 0.112 | **0.383** | 0.368 | 0.374 | 0.329 |

**Pearson r:**

| column | v1/S1 | v2/S2 | v3/S2 | v4/S3 | v5/S3 |
|---|---|---|---|---|---|
| `dementia_all` | 0.430 | 0.636 | 0.671 | **0.705** | 0.666 |
| `dementia_decreasing_activity` | 0.322 | **0.664** | 0.652 | 0.658 | 0.645 |
| `functional` | 0.174 | 0.522 | **0.564** | 0.536 | 0.485 |
| `structural` | 0.418 | 0.580 | 0.585 | **0.658** | 0.637 |
| **mean** | 0.336 | 0.601 | 0.618 | **0.639** | 0.608 |

**S1 collapses entirely** — mean dice 0.112, and 0.031 on `functional`. Its maps are barely
related to the manual's, which the screening-stage table (§2.2, F1 0.52) does not begin to
convey.

**There is no single winner among the rest.** v4 leads on `dementia_all` (0.504), on
`structural`, and on mean r (0.639). v2 leads on mean dice (0.383, narrowly over v4's
0.374), on `decrease`, and decisively on `functional` (0.370 vs v4's 0.188). Since
`dementia_all` is the benchmark's headline map, and since v4 is the only configuration that
beats the no-annotation baseline there (§4.3), **v4 remains the defensible choice**.

v2's apparent advantage here is a dice artefact and does not survive on R², where v2 falls
below the no-annotation baseline while v4 stays above it on all four columns (§4.3). Read
the dice table with that in mind: v2 selects the fewest studies of any working run (37),
which flatters a thresholded overlap metric.

**Map quality tracks precision, not recall.** Ordering the runs by end-to-end adjusted
precision (§3) gives v2 0.95 > v3 0.93 > v5 0.90 > v4 0.89 > v1 0.91, and by mean dice
gives v2 0.383 > v4 0.374 > v3 0.368 > v5 0.329 > v1 0.112. v5 has the best attainable
recall of any run (0.89) and the second-worst maps; its extra selections add coordinate mass
that pulls the maps away from the manual's. v1 is the exception that shows precision alone
is insufficient — its precision is respectable (0.91) but its recall is so low (0.41
attainable) that there is almost nothing in the map to be right about. Good maps need
adequate recall *and* high precision; v2–v4 occupy that band.

### 4.3 Against the no-annotation baseline

The relevant counterfactual is **`all_studies`** — every study with parsed coordinates, no
targeted annotation at all. That is what you would get without autonima's annotation stage:
run a PubMed search, extract whatever coordinates you can, meta-analyse the lot. Beating it
is the whole justification for the annotation stage.

**Dice vs the `all_studies` baseline:**

| column | | v1/S1 | v2/S2 | v3/S2 | v4/S3 | v5/S3 |
|---|---|---|---|---|---|---|
| `all` | pipeline | 0.205 | 0.399 | 0.436 | **0.504** | 0.453 |
| | baseline | 0.450 | 0.454 | 0.456 | 0.463 | 0.463 |
| | **delta** | −0.245 | −0.055 | −0.021 | **+0.041** | −0.009 |
| `decrease` | pipeline | 0.077 | 0.448 | 0.419 | 0.445 | 0.400 |
| | baseline | 0.344 | 0.339 | 0.345 | 0.351 | 0.351 |
| | **delta** | −0.267 | **+0.109** | +0.074 | +0.093 | +0.048 |
| `functional` | pipeline | 0.031 | 0.370 | 0.289 | 0.188 | 0.137 |
| | baseline | 0.266 | 0.260 | 0.265 | 0.269 | 0.269 |
| | **delta** | −0.235 | **+0.109** | +0.024 | −0.081 | −0.132 |
| `structural` | pipeline | 0.136 | 0.313 | 0.327 | 0.360 | 0.325 |
| | baseline | 0.265 | 0.266 | 0.269 | 0.277 | 0.277 |
| | **delta** | −0.129 | +0.047 | +0.058 | **+0.083** | +0.048 |
| **MEAN** | **delta** | −0.219 | **+0.053** | +0.034 | +0.034 | −0.011 |

**R² vs the `all_studies` baseline.** Dice depends on a thresholded overlap and so moves
with the number of studies entering each map; R² (squared Pearson correlation of the
unthresholded maps) is far less sensitive to that, which makes it the better arbiter here:

| column | | v1/S1 | v2/S2 | v3/S2 | v4/S3 | v5/S3 |
|---|---|---|---|---|---|---|
| `all` | pipeline | 0.185 | 0.405 | 0.450 | **0.497** | 0.444 |
| | baseline | 0.449 | 0.465 | 0.464 | 0.468 | 0.468 |
| | **delta** | −0.264 | −0.060 | −0.014 | **+0.029** | −0.024 |
| `decrease` | pipeline | 0.104 | 0.441 | 0.425 | 0.433 | 0.415 |
| | baseline | 0.408 | 0.427 | 0.427 | 0.424 | 0.424 |
| | **delta** | −0.304 | +0.015 | −0.002 | **+0.009** | −0.009 |
| `functional` | pipeline | 0.030 | 0.272 | 0.318 | 0.287 | 0.235 |
| | baseline | 0.289 | 0.287 | 0.288 | 0.287 | 0.287 |
| | **delta** | −0.259 | −0.014 | **+0.031** | +0.001 | −0.051 |
| `structural` | pipeline | 0.175 | 0.336 | 0.342 | **0.433** | 0.405 |
| | baseline | 0.378 | 0.398 | 0.395 | 0.404 | 0.403 |
| | **delta** | −0.203 | −0.061 | −0.053 | **+0.029** | +0.002 |
| **MEAN** | **delta** | −0.258 | −0.030 | −0.009 | **+0.017** | −0.020 |

The two metrics tell noticeably different stories, and the R² one is less flattering.

**On R², v4 is the only configuration that beats the baseline at all** — positive on all
four columns (+0.029, +0.009, +0.001, +0.029) and the only positive mean (+0.017). Every
other run is net negative: v2 −0.030, v3 −0.009, v5 −0.020, v1 −0.258.

**v2's dice advantage does not survive the switch.** It led mean dice by +0.053 over
baseline, but on R² it is *below* baseline (−0.030), including on `functional` (−0.014)
where its dice delta was +0.109. That is the signature of a sample-size effect: v2 selects
fewest studies (37), so its thresholded map is sparse and scores well on an overlap metric
while correlating no better with the manual's unthresholded map. **The §4.2 conclusion that
"v2 leads on mean dice" should therefore not be read as v2 producing better maps.**

**The honest summary is that the annotation stage barely beats doing nothing.** v4's margin
is +0.017 mean R² — real, consistent in sign across all four columns, and the only positive
result in the table, but small. The large negative numbers are more informative than the
small positive one: a bad annotation schema (S1, −0.258) is far worse than no annotation,
while a good one is only slightly better. This is an argument for judging annotation
schemas by whether they clear the `all_studies` bar at all, rather than by their margin
over each other.

### 4.4 The headline, stated carefully

**v4 reaches R² = 0.497 (r = 0.705) and dice = 0.504 against the manual `all` map, while
recovering only 45% of the gold studies — and 0.86 of the gold studies for which
coordinates were extractable. It is the only configuration of the five that beats the
`all_studies` no-annotation baseline, doing so on all four columns in R² (mean +0.017).**

That margin is small, and it should be quoted with the margin attached rather than as a
bare R². The stronger version of the claim is about the ceiling, not the margin: a map built
from 45% of the intended studies correlates with the published map at R² ≈ 0.50, and the
binding constraint on getting closer is coordinate extraction (§2.3b), not screening or
annotation.

One number should stop us from over-claiming. Our meta-analysis is *not* built on less
data in coordinate terms:

| | studies | coordinate points | vs manual |
|---|---|---|---|
| manual | 29 samples (70 PMIDs) | 1410 | — |
| v3 | 41 | 1247 | 88% |
| **v4** | 44 | 1520 | **108%** |
| v5 | 49 | 1579 | 112% |

So "half the studies" is true per-study but not per-coordinate — we compensate with more
analyses per paper, plus false positives contributing coordinates the manual never used.
The defensible claim is that v4 matches the manual's map *while recovering under half its
gold studies*, which speaks to the coordinate-availability ceiling. The claim that it does
so on *less evidence* does not hold.

Those two facts together suggest the residual error is about *which* coordinates we
contribute rather than *how many* — consistent with §4.2, where v5's looser annotation
added coordinate mass and made every map worse.

---

## 5. Open items

- **`functional` underperforms its own no-annotation baseline** in v4 and v5, and declines
  monotonically v3 → v4 → v5 (0.289 → 0.188 → 0.137). It is the smallest column and the
  one with the weakest annotation agreement throughout.
- **Three known drafting defects in v5's annotation criteria** account for 2 of its 5
  residual eligibility errors: a contradiction where the sample criterion admits "Pick's
  disease" while the exclusion rejects neuropathology-defined groups; an over-broad
  symptom-subgroup permission; and a pooled-map allowance that admits `All participants`
  maps spanning several patient groups.
- **Coordinate extraction is the dominant constraint**, not full-text availability. Only 5
  of 74 gold studies (7%) were never obtained as full text; 33 more are lost between having
  the text and having a parsed coordinate (93% → 49%). No schema change touches this.
- **`functional` regressed with the annotation changes, not the screening changes.** It
  declines monotonically from v2 onward (0.370 → 0.289 → 0.188 → 0.137) and beats its
  baseline only in v2 and v3. Since v2 → v3 and v4 → v5 both isolate annotation, the cause
  is in the annotation criteria. Recovering v2's `functional` behaviour without giving up
  v4's `all` advantage is the most valuable outstanding change.
- **The annotation stage's margin over doing nothing is thin.** On R² only v4 clears the
  `all_studies` baseline, by +0.017 mean. A bad schema is much worse than no annotation
  (S1, −0.258) while a good one is only slightly better, so schemas should be judged by
  whether they clear that bar at all.
- **A precision-first hybrid was tried and rejected** (not retained in the repo). It paired
  v4's screening with an annotation combining A2's contrast gate, A3's method exclusions,
  a group-identity gate, a connectivity-derived exclusion replacing A2's blanket
  resting-state ban, and a narrow statistical-effect gate. It achieved the best strict
  precision of any run (0.77 vs v4's 0.70) and matched v4 exactly on `decrease`
  (R² 0.433, Δ +0.009) using 44 analyses from 30 studies against v4's 92 from 47 — but
  attainable recall fell to 0.67 and mean R² went **negative** vs the `all_studies`
  baseline (−0.013), with `functional` collapsing to −0.107, the worst of any run. Two
  conclusions worth keeping: pushing precision past v4's operating point does not buy map
  quality (precision *correlates* with map quality across v1–v5 but does not *cause* it
  beyond that point), and the `functional` column is far more sensitive to modality and
  connectivity exclusions than the others because it is small (18 manual analyses, 348
  points). Do not re-attempt a global precision tightening; if anything is worth trying it
  is per-column criteria.

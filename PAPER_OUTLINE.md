# Autonima paper — structure and evidence map

Working thesis: **LLM-based screening and annotation recover meta-analytic maps closer to
expert-curated results than a search-only pipeline can, and the gain comes specifically from
selecting the right *analyses*, not merely the right *papers*.**

Status key: **[have]** analysis exists and is reproducible · **[partial]** exists for some
projects · **[need]** not yet run · **[idea]** aspirational

---

## Framing

The claim to establish is that autonima buys something beyond non-LLM screening. The
instrument is **neurometabench**, the gold-standard benchmark assembled for this work: N
published coordinate-based meta-analyses, their included-study lists, and their per-analysis
annotations.

**Where to put the benchmark caveat.** neurometabench is built from manual meta-analyses,
which have their own limited scope, resolution and internal consistency. That limitation
shapes the interpretation of nearly every result, so it cannot be buried — but it should not
lead, or the paper reads as apologia. Recommended placement: one short paragraph at the end
of the Introduction establishing "best available reference, not ground truth", then invoked
by name at each result where it actually binds. It returns as a first-class topic in the
Discussion, where it motivates the future-use-case argument (§9).

---

## 1. Screening improves precision at almost no cost to recall  **[have]**

**Claim.** Moving search → abstract screening → full-text screening raises precision
monotonically while recall is essentially preserved.

**Evidence.** `reports/cross_project_screening/screening_metrics_top_v_stage_progression.csv`,
8 projects:

| stage | mean precision | mean recall | mean F1 |
|---|---|---|---|
| search | 0.112 | 0.802 | 0.186 |
| abstract | 0.201 | 0.965 | 0.323 |
| full text | 0.332 | 0.934 | 0.472 |

Precision increases at every stage in **8 of 8 projects**, no exceptions — that unanimity is
the strongest single fact in the paper and should be stated as such.

**A metric subtlety to fix before a reviewer finds it.** Recall *rises* from search (0.802)
to abstract (0.965), which is impossible for a cumulative measure — screening cannot recover
a study the search never returned. These are **stage-conditional** retention rates (recall
among gold that reached that stage), not cumulative recall. Two consequences:

- State the denominator explicitly for every recall number in the paper.
- Reframed cumulatively, the finding is *stronger*: screening is close to lossless
  (0.93–0.97 retention), and essentially all recall loss occurs at **search**. That reframing
  also sets up §6, where end-to-end performance is search-limited.

**Adjusted recall.** Recall must be reported over gold studies for which the data actually
exists — full text obtained and coordinates extractable — or it measures corpus availability
rather than screening. Define once, early, and use consistently.

**Figure 1.** Slope plot, precision by stage, one line per project, 8/8 rising. Secondary
panel: stage-conditional recall flat across stages.

---

## 2. The absolute precision objection, and the controlled answer  **[have]**

**Anticipated reaction.** Precision ≈ 0.33–0.40 at full text is unimpressive in absolute
terms. Address this head-on rather than defensively.

**Why it is understated.** For most projects we do not know which pool the original authors
screened. PubMed is a moving target, so our candidate set is not theirs. A "false positive"
may be a study they never saw, or saw and excluded for a reason not recorded. Precision
against an unknown screening pool is a lower bound.

**The controlled case: dementia.** Tahmasian et al. (PMID 35664889) publishes every screened
and rejected study *with its rejection reason* — 495 records, resolving to a fixed 558-PMID
pool. Two things become measurable that are otherwise not:

1. **Same-pool precision.** Holding schema constant and changing only the input from a
   PubMed query to the fixed list: full-text precision **0.401 → 0.468** (v3 vs
   v3-allstudies). The effect replicates across all three schemas that have both runs:

   | schema | search-based | same pool | Δ |
   |---|---|---|---|
   | v1 | 0.254 | 0.368 | +0.114 |
   | v2 | 0.384 | 0.482 | +0.098 |
   | v3 | 0.401 | 0.468 | +0.067 |

   Reporting all three is better than one pair: it shows the gain is a property of controlling
   the pool, not of a particular schema.

   **The cleanest instance of this effect is in social, not dementia**, and it belongs here as
   the lead evidence. `v3-all_pmids` and `v3-search-all_pmids` are byte-identical in
   `screening`, `annotation`, `retrieval` and `parsing` — the only difference is that the
   second adds a PubMed `query` on top of the same `pmids_file`:

   | run | precision | recall | F1 |
   |---|---|---|---|
   | `v3-all_pmids` (fixed pool) | **0.688** | 0.929 | 0.791 |
   | `v3-search-all_pmids` (same list + query) | 0.409 | 0.929 | 0.568 |

   **+0.279 precision at identical recall** (0.9286 to four figures in both). Because recall
   does not move at all, there is no precision/recall trade to argue about, and because the
   two configs differ in exactly one field the comparison isolates pool control better than
   the dementia pairs, which also differ in run vintage and list composition. Social's effect
   is 2–4× the size of dementia's, which is the stronger form of the argument: the amount by
   which an unknown screening pool understates precision is large, and varies by project.

2. **Adjusted precision.** The reasons separate eligibility errors from data-availability
   exclusions. `Data Not Reported` (88 of 484 rejections) means the study *qualified* but
   reported no usable coordinates — selecting it is not an eligibility error. On the same pool
   with schema held at v3, precision rises from **0.47 strict to 0.74 adjusted** at full-text
   screening, and from **0.73 to 0.93** at the end-to-end analysis-selection stage.

**A reporting convention to settle now.** Earlier drafts quoted these as ranges
(0.66–0.76 and 0.89–0.95). Those are **min–max across the five schema versions**, not
confidence intervals, and a reader will misread them as uncertainty:

| version | schema | full-text strict P | full-text adj P | end-to-end strict P | end-to-end adj P |
|---|---|---|---|---|---|
| v1 | S1 | 0.37 | 0.66 | 0.45 | 0.91 |
| v2 | S2 | 0.48 | **0.76** | 0.73 | **0.95** |
| v3 | S2 | 0.47 | 0.74 | 0.73 | 0.93 |
| v4 | S3 | 0.43 | 0.69 | 0.70 | 0.89 |
| v5 | S3 | 0.43 | 0.69 | 0.64 | 0.90 |

**Recommendation: §2 reports a single version throughout (v3), and §3 owns the version
comparison.** Otherwise the two sections double-count the same variance, and §2's headline
becomes a statement about schema choice rather than about pool control — which is the opposite
of its purpose. Ranges elsewhere in the paper should state explicitly what they range over.

Attainable recall on the same pool is **0.96** for v3 (0.96–0.97 across all five versions):
of gold studies whose full text we obtained, screening recovers essentially all of them.

**Emphasis.** The value of the dementia work is the *screening* result, not its brain maps.
Its map comparison belongs in supplement.

**Second controlled case: emotion_regulation_2022 [need].** Worth completing — one
controlled case invites "n=1", and nothing else in the published set records rejection
reasons. ER is a better second case than it first appears, for three reasons:

- **We are in direct contact with the original author.** That is the only route in this
  project to information no paper publishes: the actual screened pool, and potentially the
  per-study rejection reasons that make the adjusted-precision analysis possible. Everything
  in §2 currently rests on dementia having happened to publish its supplement; ER would show
  the analysis is repeatable when the information is obtainable by other means.
- **We already hold at least part of the original search results**, so the same-pool
  comparison does not depend entirely on new correspondence.
- **It is a hard case, deliberately.** ER selection is highly domain-specific cognitive
  judgement — which contrast counts as regulation rather than mere affect, reappraisal versus
  maintenance versus suppression. Dementia is the easy end of the spectrum (a clean
  diagnosis-plus-contrast rule); ER is the difficult end. Showing the method on both bounds
  the claim rather than cherry-picking, and a weaker ER result is still informative because
  it maps where the approach degrades (see §8a).

**Blocker to clear first.** `projects/emotion_regulation_2022/nmb_mappings.json` is still
the unedited template (`MANUAL_NAME1` → `AUTOMATIC_NAME1`), which excludes ER from every
cross-project analysis. The benchmark side exists — 89 gold studies in `included_studies.csv`
and a merged NiMADS — so this is filling in the annotation keys, not new curation.

**Figure 2.** Dementia precision under three definitions (search-based / same-pool /
same-pool adjusted) with the rejection-reason breakdown as a stacked bar.

---

## 3. A cautionary case: v1 was wrong in a way screening metrics could not show  **[have]**

*Scoped deliberately as a case study, not a headline result — and a candidate for supplement
if space is tight. See the note at the end for what would promote it.*

**Claim.** Criteria transcribed verbatim from a published methods section can look perfectly
healthy on screening metrics while being badly wrong, and the error is only visible end to
end. That is an argument about *how to evaluate a schema*, not about how well the method
performs.

**The case.** Dementia S1 was transcribed near-verbatim from Tahmasian et al.'s stated
criteria. On the fixed 558-PMID pool:

| schema | how written | full-text F1 | adjusted F1 | screening recall | **attainable recall, end to end** |
|---|---|---|---|---|---|
| S1 | near-verbatim from the source paper | 0.523 | 0.70 | 0.91 | **0.42** |
| S2 | rewritten from our own reading of intent | **0.629** | 0.70 | 0.91 | 0.75 |
| S3 | revised from inspection of observed failures | 0.580 | 0.68 | 0.91 | 0.86 |

**All three schemas score 0.91 recall at screening — indistinguishable.** End to end, S1
recovers 0.42 of the attainable gold against S3's 0.86. S1 passed studies through and then
failed to find an eligible analysis inside them. Any evaluation stopping at the screening
stage would have judged the three schemas equivalent and shipped the worst one.

**Why verbatim transcription fails.** Published methods sections bundle several requirements
into a single sentence, and a model satisfies the sentence by finding *any* one of them.
S1 → S2 dropped 43 false positives without losing a single gold study.

**The epistemic problem, which is the real point.** We caught this because we had a gold
standard to measure against. A practitioner running a novel meta-analysis has no such
reference, would see 0.91 recall at screening, and would have no signal that anything was
wrong. This is the strongest argument in the paper for human-in-the-loop verification and for
tooling that surfaces *what a schema actually excluded and why* — and it is honest about the
fact that we are not offering a solution, only a diagnosis.

Two supporting observations, both requiring dementia's adjusted metrics:

- **All three schemas land within 0.02 on adjusted F1.** They differ in where they sit on the
  precision/recall trade-off, not in overall quality. Choose an operating point deliberately
  rather than chasing F1.
- **Over-tightening is the characteristic failure of failure-driven revision.** Dementia's
  abandoned v4 collapsed recall 0.96 → 0.50, and emotion_regulation's S3 shows the same
  shape independently (recall 0.477 → 0.216). Iterating against observed errors reliably
  overshoots.

**Scope.** Originally only two projects had undergone genuine schema revision (dementia and
emotion_regulation), both by hand. Four more have since been revised by an LLM agent, which
both widens the base and supports a distinct second claim — see *Agent-authored schema repair*
below.

**The overfitting objection is already answered, and this is where to say so.** Every *later*
schema was iterated against the benchmark, but **v1 was not**: in every project it was written
from the source paper's stated criteria before any results were seen. v1 is therefore a
held-out schema by construction, and §1's headline (precision rising at every stage in 8 of 8
projects) holds at v1 as well as at the tuned versions. No separate frozen-schema experiment is
needed — the corpus already contains one, run nine times. State this explicitly rather than
leaving a reviewer to wonder.

The corollary is that the v1 → later gains in this section are the *only* numbers in the paper
that are tuning-inflated, and they are presented as a case study rather than a headline for
exactly that reason.

### Agent-authored schema repair  **[have]**

**A second, separable claim.** The dementia case above says a human can misread a methods
section. This says something different: **an LLM agent, given the source paper's full text and
its own error reports, can repair a schema** — and that this works reliably for one class of
defect and not another. Four projects were revised this way, with no human expert reviewing the
criteria before the run. Each config carries a provenance header saying so.

| project | what was revised | metric | before | after |
|---|---|---|---|---|
| problem_solving | annotation | mean annotation F1 | 0.770 | **0.948** |
| problem_solving | annotation | fair-comparison dice | 0.514 | **0.633** |
| decision_making | full-text screening | screening F1 | 0.283 | **0.312** |
| social | annotation | fair-comparison dice | 0.514 | 0.550 *(v1 = 0.570)* |
| executive_function | full-text screening | screening F1 | 0.163 | 0.152 *(recall 0.322 → 0.386)* |
| executive_function | annotation | fair-comparison dice | 0.566 | 0.556 |

**Two clear wins, one partial, one negative — and the split is not random.** The revisions
succeeded where the defect was a *factual misreading of the benchmark*, and underperformed where
it was a *precision/recall trade*:

- **problem_solving** — `logical_reasoning` had been defined as formal logic. Reading the
  benchmark's own Sleuth file showed the target it maps to (`Demand_MNI_final`) is
  difficulty manipulations ("Complex > Simple", "Multi-Digit > Single-Digit"). That column went
  0.393 → 0.844, and `verbal` recovered 0.767 → 0.959 because deductive-reasoning contrasts had
  been diverted into it. A fact was wrong; the fix was verifiable.
- **decision_making** — the paper admits only condition-vs-condition contrasts, but the schema
  was accepting contrasts against implicit baseline. Again a checkable fact.
- **social** — the diagnosis was right and quantitatively confirmed (the benchmark assigns 1.34
  construct labels per contrast; the schema assigned 0.94, and the revision moved it to 1.33).
  Recall rose on all four constructs. But map quality still did not beat v1, because the extra
  labels cost precision.
- **executive_function** — the relaxations recovered 11 gold papers at screening, yet F1 and
  map dice both slipped. Its stated criteria contradict its realized inclusions on three axes,
  and relaxation alone cannot repair that without paying in precision.

**The generalisation worth stating:** where a schema encodes a *false belief about the
benchmark*, an agent with the source paper and the error reports can find and fix it, sometimes
dramatically. Where the schema sits at the wrong point on a *precision/recall trade-off*, the
agent moves along the curve rather than off it — and in this corpus map quality tracks
annotation precision, so recall-increasing revisions can look better on annotation metrics while
being no better, or slightly worse, on the maps. **Report both metrics; they disagree, and the
disagreement is the finding.**

**But there is a second, better explanation for the disagreement, and it is measurable.** A
construct may simply not be a well-defined *neural* concept. If the manual meta-analysis's own
maps for two constructs are nearly identical, then no amount of improved analysis selection can
sharpen them — the ceiling is set by the target, not by the annotation. The fair comparison
already measures this: `dice_mean_off_diagonal` is how well a map matches the *other* constructs'
manual targets, so the diagonal-minus-off-diagonal gap is a direct index of how separable the
constructs are.

| project | diag | off-diag | gap | off/diag |
|---|---|---|---|---|
| vbm_of_substance_use | 0.488 | 0.082 | 0.406 | 0.17 |
| vbm_of_ptsd | 0.875 | 0.520 | 0.355 | 0.59 |
| executive_function | 0.556 | 0.355 | 0.200 | 0.64 |
| problem_solving | 0.633 | 0.475 | 0.159 | 0.75 |
| decision_making | 0.390 | 0.233 | 0.157 | 0.60 |
| cue_reactivity | 0.699 | 0.561 | 0.138 | 0.80 |
| dementia | 0.470 | 0.351 | 0.118 | 0.75 |
| **social** | 0.550 | 0.451 | **0.099** | **0.82** |

**`social` is the least separable project in the corpus** — its maps match the wrong construct
82% as well as the right one. And the two projects where annotation gains failed to reach the
maps are exactly the two whose *own source papers* say the constructs share neural substrate:

- **executive_function** — the paper's title is "Meta-analytic evidence for a **superordinate**
  cognitive control network subserving diverse executive functions", and its central finding is
  that one fronto-cingulo-parietal network underlies all six domains. Domain-specific maps are
  not expected to separate; the paper argues they do not.
- **social** — "we allowed for contrasts to be associated with more than one RDoC social domain
  construct", with 1.34 labels per contrast and 29% dual-annotated. The constructs overlap *by
  construction*, not by annotation error.

**Synthesis.** Both mechanisms operate, and which one applies depends on the kind of fix:

- A fix that corrects **misassignment** — analyses going to the wrong construct — improves maps
  even when constructs overlap, because coordinates move to the right target. That is
  problem_solving, whose off/diag ratio is a fairly high 0.75 yet whose dice still rose
  0.514 → 0.633.
- A fix that **adds labels** to already-correct assignments improves annotation recall but
  cannot sharpen maps when the constructs are not separable to begin with. That is social.

**Implication for the paper.** Do not report annotation-metric gains as if they imply better
maps, and do not treat a flat map result as a failed schema. Report the separability index
alongside, and say plainly that for some constructs — RDoC social domains, executive-function
subdomains — the neural target itself is diffuse, which caps what any annotation improvement can
achieve. This is a finding about the constructs, not about the method.

**Caveat, stated up front.** These revisions read the benchmark's error reports, so they are
tuned against it by construction and must be reported as iterated versions, never as held-out
ones. What they demonstrate is repair-given-feedback, not zero-shot schema authoring.

**Still missing [need]:** no config records *what changed* between versions in machine-readable
form; the scope table above had to be reconstructed by diffing configs. A one-line provenance
field would make it auditable.

**Figure 3.** Dementia S1/S2/S3: screening recall (flat at 0.91) beside end-to-end attainable
recall (0.42 → 0.75 → 0.86). The gap between the two panels is the entire point.

---

## 4. Bridge: from screening to annotation

Annotation was originally folded into the all-studies evaluation for a reason worth stating
plainly: when a study was included but contributed nothing usable, we needed to know whether
that was a *screening* failure or a failure to select the right *analysis within* the paper.
Those are different errors with different fixes, and the dementia analysis is what forced the
distinction. Use that as the transition rather than a section break — it motivates why
analysis-level selection deserves separate evaluation.

Also worth a short paragraph here: **where screening fails.** From dementia, residual errors
concentrate in wrong patient group (analyses labelled by another disease or by pathology
subtype) and ROI-only designs. **[have]**

---

## 5. Parsing and analysis-level annotation  **[have — cross-project numbers filled 2026-08-30]**

> **Audit note (2026-08-30).** This section was marked `[have / partial]` because only an EF worked
> example had been written in. Both underlying tables exist and are current (regenerated 2026-08-28,
> after the corpus re-run), so the cross-project version is filled in below and the EF-specific
> figures are corrected. The section's *claims* all survive; three of its *numbers* had drifted.

**Cross-project result — analysis-level annotation.** From
`reports/cross_project_analysis/annotation_aggregates.csv`, with matched fraction from
`parsing_metrics_by_project.csv`.

**Aggregation: `exhausted_manual_assumption`, and dementia excluded.** Two scoping decisions, both
deliberate:

1. *Exhausted-manual is the primary figure*, because it reflects reality. It charges every unmatched
   manual analysis as a miss rather than excusing it. `matched_only` conditions on the matching layer
   having succeeded, which flatters the system by hiding exactly the failures §5 says are an
   independent error source. Where `matched_only` is quoted it must be labelled as such — the two
   differ by **0.32 in pooled precision** (0.863 vs 0.540).
2. *dementia is excluded from analysis-level annotation figures.* Its source meta-analysis **combined
   multiple studies that shared a data source into single analyses**, so there is no clean mapping
   from our per-paper extraction to its gold analyses. Matching failure there is a property of the
   benchmark's construction, not of the pipeline, and scoring it either way is misleading.

| project | matched % | precision | recall | F1 | FN | FP |
|---|---|---|---|---|---|---|
| vbm_of_ptsd | 100% | 1.000 | 0.800 | **0.889** | 2 | 0 |
| vbm_of_substance_use | 98% | 0.584 | 0.890 | 0.705 | 12 | 69 |
| problem_solving | 97% | 0.539 | 0.962 | 0.691 | 13 | 280 |
| social | 94% | 0.618 | 0.708 | 0.660 | 338 | 508 |
| executive_function | 84% | 0.545 | 0.821 | 0.655 | 34 | 130 |
| emotion_regulation_2022 | **63%** | 0.507 | 0.786 | 0.616 | 40 | 143 |
| decision_making | 91% | 0.451 | 0.882 | 0.596 | 11 | 100 |
| cue_reactivity | 95% | 0.427 | 0.950 | 0.589 | 18 | 457 |
| **POOLED (n=8)** | | **0.540** | **0.809** | **0.647** | 468 | 1687 |

Excluding dementia barely moves the pool (0.539 -> 0.540 precision, F1 0.647 either way) — it is
excluded for correctness of the comparison, not to improve the number.

Three things to say:

- **Annotation is recall-heavy under the honest denominator** — pooled recall 0.809 against precision
  0.540. The system finds the manual analyses (0.71–0.96 recall everywhere) and admits a great many
  extra ones. That is the opposite conclusion to `matched_only`, which makes it look
  precision-heavy, and it is why the variant must be stated.
- **The FP count is the story, not the FN count.** 1687 FP against 468 FN. Whether those are true
  errors or analyses the manual meta-analysis simply never considered is the open question, and it
  is the same question §2 addresses for screening precision.
- **Report matched % beside every F1.** ER's rests on **63%** of its manual analyses, the worst in
  the set.

**Corrections to the EF worked example below.** Direction holds, magnitudes drifted: matching now
fails for **19 of 122** manual analyses (was "22 of 108"). Under `matched_only` the EF asymmetry is
3.8:1 false-negative (34 FN vs 9 FP), close to the stated 5:1; under exhausted-manual it inverts to
130 FP vs 34 FN. Both are true of the same run — another reason the variant has to be named.

**5a. Parsing.** Cross-project coordinate extraction performance, from
`reports/cross_project_analysis/parsing_metrics_by_project.csv`. Needed both as a result and
because it defines the matched subset everything downstream is evaluated on.

**5b. Annotation, on analyses where both decisions are known.** Fuzzy-match our parsed
analyses to the benchmark's, keep confident matches, and compare selection decisions.
Worked example — executive_function annotation-only: analysis-level F1 **0.84–0.92** across
four sub-annotations, precision 0.94–1.00, recall 0.75–0.86.

**Report the matching layer honestly.** For EF, 22 of 108 manual analyses fail to match
before annotation is even scored — 14 involving a Talairach→MNI conversion, 13 with a
coordinate-count mismatch. So annotation F1 is conditioned on a matched subset, and the
matching layer is an independent error source. State the matched fraction wherever an
annotation F1 appears.

**An asymmetry worth naming.** EF errors run 5:1 false-negative (30 FN vs 6 FP), and every FN
traced to a criterion in our own schema rather than a model failure. That finding belongs
here and it sets up §3's schema-specification argument with hard numbers.

**Figure 4.** Per-project annotation precision/recall, with matched-fraction annotation.

---

## 6. Does better annotation produce better maps? The fair comparison  **[have]**

**Claim.** Restricted to studies both the manual authors and we had access to, maps built from
annotation-selected analyses are closer to the manual result than maps built from all
extracted analyses.

**Evidence.** `reports/cross_project_manual_vs_auto_meta_fair/`, regenerated 2026-08-25 after
the decision_making and executive_function PMID-keying fixes. Δ is the annotation-selected map
minus that run's own all-analyses baseline, best run per project:

| project | Δdice | Δr | character of the target |
|---|---|---|---|
| vbm_of_ptsd | **+0.355** | **+0.262** | clinical, required population contrast |
| vbm_of_substance_use | **+0.344** | **+0.371** | clinical, required population contrast |
| decision_making | +0.125 | +0.062 | cognitive, moderately specified |
| dementia | +0.116 | +0.087 | *excluded — see below* |
| executive_function | +0.111 | +0.083 | cognitive, domain-specific |
| cue_reactivity | +0.096 | +0.063 | cue/neutral contrast, fairly specified |
| social | +0.045 | +0.008 | cognitive, diffuse |
| problem_solving | +0.008 | **−0.020** | cognitive, most diffuse |

**Annotation helps in 6 of the 7 reportable projects**, and the one failure
(`problem_solving`, Δr −0.020) is the most loosely defined construct in the set.

**The ordering supports the clinical-versus-cognitive hypothesis, which was previously only
an intuition.** The two projects requiring a specific population contrast — PTSD and substance
use — show effects roughly 3× larger than anything else (Δdice ≈ +0.35 against +0.10 or less).
The two weakest are `social` and `problem_solving`, the most diffuse targets. This is now a
tested ordering rather than an assertion, and it is the concrete evidence behind §8a:

> Annotation matters most where the correct analysis is sharply defined and picking the wrong
> one would be an outright error (patients vs controls, and the reverse). It matters least
> where several analyses in a paper are defensibly "the right one".

**What this does not settle.** The ordering is consistent with the squishy-target account but
does not exclude the model-limitation account (§8a): a diffuse construct and a construct the
model handles poorly both produce a small Δ. Distinguishing them still needs the second-rater
work described there.

**Dementia is excluded, and why.** Fair mode requires per-paper attribution on the manual side.
Dementia's benchmark merges 8 of its 29 entries into single independent samples — one bundling
27 papers into one n=19 sample — with pooling applied *before* coordinates were recorded, so no
filtering can recover which coordinates came from which paper. An all-or-nothing group filter
leaks: restricting to studies we could have had still admitted 16 unavailable papers'
coordinates and retained 83% of the manual points. Its row is shown above only to make the
exclusion visible; it should not be reported. Its screening results (§2, §3) are unaffected —
the limitation is specific to per-paper coordinate attribution.

**Soundness audit for the rest [have].** Only dementia has genuine merging:

| project | id style | merged entries | fair mode |
|---|---|---|---|
| cue_reactivity, emotion_regulation_2022, social, vbm_of_ptsd | PMID | 0 | sound |
| executive_function | 171 PMID / 7 author-year | 0 | sound (after PMID-keying fix, `3bea37a`) |
| decision_making | 87 PMID / 3 author-year | 0 | sound (after PMID-keying fix, `b799a39`) |
| problem_solving | 124 / 5 | 0 | sound; 5 non-PMID entries excluded from the join |
| vbm_of_substance_use | 77 / 3 | 0 | sound; 3 non-PMID entries excluded |
| **dementia** | PMID | **8 (max 27)** | **excluded** |

Both keying fixes were needed *because* of this section: decision_making could previously join
only 20 of 90 entries. Regenerating moved it (dice 0.409 → 0.429, r 0.640 → 0.656) while all 14
other runs stayed byte-identical, which is the check that the rebuild was targeted rather than
a global recomputation.

**Caveat to carry.** The original meta-analyses' own resolution bounds how well any selection
can score. Where the benchmark pooled analyses coarsely, a correct fine-grained selection can
look wrong — and that bound is tightest in exactly the diffuse projects at the bottom of the
table, so some of the clinical/cognitive gradient may be benchmark resolution rather than task
difficulty.

**Scope note.** These are `*-annotation-only` runs: the study pool is fixed to the benchmark's
PMIDs, so this isolates annotation from search and screening. It is *not* the end-to-end claim
— that is §7, and the two should not be conflated.

**A secondary result available here [have].** executive_function now has two annotation
schemas in the table. v2 relaxed three criteria that the benchmark demonstrably does not apply
and lifted analysis-level F1 from 0.87 to 0.95 with recall 0.81 → 0.95 — but the maps are a
wash (Δdice +0.111 vs +0.113, Δr +0.083 vs +0.075). Large gains in selection agreement did not
move the resulting maps, which echoes the dementia finding that map quality is insensitive to
selection changes past a point. Worth a sentence: agreement with human selection and map
fidelity are not the same objective.

**Figure 5.** Per-project Δdice and Δr, ordered, with clinical/cognitive marking. Dementia
omitted with the reason in the caption rather than silently dropped.

---

## 7. What it takes in the wild: end-to-end vs a search-only meta-analysis  **[have]**

**Claim.** Running the full pipeline from a plain PubMed search beats what you would get by
searching, extracting every coordinate, and meta-analysing the lot — the Neurosynth-style
approach.

**Two baselines, and why the second is necessary.**

- `all_studies` — every study the project's one broad search returned that yielded
  coordinates, no screening.
- **Per-sub-annotation targeted baselines** — for benchmarks that ran one broad search then
  split the hits into sub-topics, the broad baseline is an unfairly weak opponent. Nobody
  targeting only the alcohol sub-meta-analysis searches the whole substance-use literature.
  Framework: `scripts/run_baseline_searches.py` + `projects/<p>/baselines.yaml`. **[have]**
  for all 9 projects, 35 sub-analysis columns (34 with a targeted arm).

**Full result — 9 projects, 35 sub-analysis columns.** The unit is the COLUMN, not the project,
and each column is scored against the best baseline that could exist for it: the targeted arm
wherever one was defined, the broad arm only where targeting is impossible. Emitted by
`scripts/compile_best_baselines.py` as `reports/cross_project_best_baseline.csv`.

| project | n | autonima | best baseline | Δ |
|---|---|---|---|---|
| emotion_regulation_2022 | 4 | **0.670** | 0.309 | +0.361 |
| vbm_of_ptsd | 1 | **0.456** | 0.247 | +0.209 |
| cue_reactivity | 3 | **0.585** | 0.457 | +0.128 |
| problem_solving | 5 | **0.652** | 0.560 | +0.092 |
| vbm_of_substance_use | 6 | **0.243** | 0.153 | +0.090 |
| social | 5 | **0.584** | 0.527 | +0.057 |
| executive_function | 4 | **0.643** | 0.612 | +0.031 |
| dementia | 4 | **0.280** | 0.275 | +0.005 |
| decision_making | 3 | **0.367** | 0.367 | +0.001 |

*(Regenerated 2026-08-29 against the post-retrieval maps. Previous values, computed before the
corpus re-run: ER 0.663, cue_reactivity 0.568, problem_solving 0.629, social 0.520, dementia 0.330,
vbm_of_su 0.255, EF 0.642, decision_making 0.352.)*

**Pooled over the 35 columns: autonima 0.495 vs 0.396, Δ +0.099, ahead in 30 of 35.** Median Δ is
+0.046 — the mean sits above it because a few large wins skew the distribution, so quote both.
Pooling columns rather than averaging project means is deliberate: it stops a one-column project
(`vbm_of_ptsd`) weighing as much as a six-column one (`vbm_of_substance_use`), and it avoids the
mixed-denominator artefact that arises when a project's targeted margin is averaged over columns
having no targeted arm.

**There are no longer any project-level losses.** Before the retrieval work `social` (−0.007) and
`decision_making` (−0.015) both sat behind their best baseline; after it both are ahead, social
decisively (+0.057) and decision_making by a hair (+0.001, effectively a tie). `social` still has
the corpus's weakest annotation, and `decision_making` remains the closest contest in the set —
worth saying so rather than claiming a clean sweep.
`emotion_regulation_2022` heads the table, but three of its four columns are scored against a
broad arm (see below), so it is the least contested comparison in the set — worth saying so.

### Targeting the search is worth almost nothing, and sometimes less than nothing

Across the 34 columns having both arms, mean targeted − broad is **+0.023 and the median is
exactly 0.000**, despite targeted queries cutting candidate pools several-fold. **In 10 of those
34 columns the broad arm actually scores HIGHER** — narrowing the search made the baseline worse.

`emotion_regulation_2022` supplies the cleanest test of when targeting pays, because all four of
its columns were probed deliberately:

| column | targetable? | targeted pool | recall | sub R² | broad R² |
|---|---|---|---|---|---|
| reappraisal | yes | 135 studies | 0.761 | **0.523** | 0.365 |
| decrease | yes | 195 | 0.709 | 0.307 | **0.316** |
| increase | yes | 167 | 0.667 | 0.173 | **0.198** |
| maintain | **no** | — | — | — | 0.233 |

Only `reappraisal` yields a stronger baseline, and it has the *smallest* pool of the three. The
lesson is that **focus only pays when recall survives it**: the decrease and increase arms shed
more gold studies (recall 0.709, 0.667) than their tighter pools win back.

Two further findings from that probe are worth stating:

- **Direction terms alone are unusable.** Searching `downregulat*` and kin for the decrease column
  returns 9,356 hits for 37 gold studies — the vocabulary belongs to molecular biology. Targeting
  only worked when conjoined with the regulation vocabulary.
- **A control condition cannot be targeted at all.** `maintain` is the Look condition — viewing
  emotional pictures with no regulation instruction. Three plausible researcher queries (emotional
  reactivity, affective-picture viewing, both combined) are ALL worse than the broad arm on both
  recall and precision (0.404 / 0.246 / 0.368 against 0.877). A control condition is not what a
  paper is *about*, so it never reaches the title or abstract. This is the one column in the
  corpus where the broad arm is not a fallback but genuinely the best baseline obtainable.

**This is the paper's sharpest mechanistic point.** What limits a screening-free pool is not
its size but that irrelevant analyses *inside retained papers* still contribute coordinates.
Removing whole irrelevant papers helps far less than removing irrelevant analyses within kept
papers. **It is not a search problem; it is a screening problem** — and it explains why
better queries cannot substitute for the pipeline.

**Figure 6.** Grouped bars per project × arm (autonima / targeted / broad), R².

**Retrieval asymmetry between the arms — narrower than it first appears.** Some projects'
pools were enlarged by **manual downloading** on top of automated PMC/Elsevier retrieval
(`articles/ace_outputs/html/Manual/` and `.../manual/`, 660 PMIDs between them). The concern
was that the per-sub baselines, which are automated-fetch only, would be unfairly starved
relative to broad pools containing hand-fetched papers.

Checked, and mostly not the case: **baseline runs inherit `full_text_sources` from the project
template**, which points at the `ace_outputs` root, and autonima globs that tree by filename.
Manually downloaded papers are therefore visible to a baseline run and a pipeline run alike —
manual downloading enlarged a *shared* local corpus rather than privileging one arm.

The residual asymmetry is real but small: PMIDs that a narrowed search surfaces which are **not
already local** get automated fetch only, and nobody will hand-download for a baseline arm. In
vbm_of_substance_use that residual was 114 newly-surfaced PMIDs yielding **2** coordinate-
bearing studies — negligible against a 190-study control pool. Report the mechanism, not a
correction.

**Manual reliance per project** (coordinate-bearing studies sourced from the manual
directories; `pond` excluded, being a bulk corpus rather than hand-fetching):

| project | coord-bearing | from manual | share |
|---|---|---|---|
| vbm_of_ptsd | 44 | 29 | **66%** |
| emotion_regulation_2022 | 115 | 75 | **65%** |
| decision_making | 357 | 83 | **23%** |
| executive_function | 1128 | 39 | 3% |
| problem_solving | 404 | 12 | 3% |
| dementia | 172 | 5 | 3% |
| cue_reactivity | 561 | 15 | 3% |
| social | 537 | 7 | 1% |
| vbm_of_substance_use | 223 | 0 | **0%** |

Two things follow.

**The project carrying §7 is clean.** vbm_of_substance_use draws 0% from manual sources, so its
+0.105 autonima margin has no retrieval-provenance confound at all. dementia and cue_reactivity
at 3% are effectively clean too — relevant because their baselines are the next to run.

**Where more manual downloading would pay off most.** The bottom of the table is where pools are
thinnest relative to what exists, and any future hand-fetching is best spent there:

- **vbm_of_ptsd** — only 44 coordinate-bearing studies in total, the smallest pool in the
  corpus. Its maps are the most sensitive to a handful of additions, and it is also the
  project with the single largest annotation effect (§6, Δdice +0.355), so its numbers carry
  weight disproportionate to their sample.
- **social (1%) and executive_function (3%)** — the largest pools by raw count but almost
  entirely automated, and EF separately loses 58% of its intended papers to retrieval and
  parsing attrition (§5). Manual fetching there would target a known, quantified gap.
- **vbm_of_substance_use (0%)** — deliberately left alone would preserve it as the clean
  reference case for §7. Worth *not* downloading into, or doing so only after its results are
  locked.
- **cue_reactivity v6** — the corrected search adds ~740 candidates to the `load_excluded`
  arm, so it has a fresh backlog by construction. Its results are provisional until that is
  cleared.

Per-baseline missing-full-text lists are written to
`projects/<p>/reports/baseline_missing_fulltexts.{json,txt}` by the runner, so the targets for
any download session are already enumerated rather than needing reconstruction.

### Recall is capped by retrieval, and correcting for it reorders the projects  **[have]**

The manual-reliance table above says where full texts *came from*. It does not say how many are
missing, and that turns out to matter more. Compiling every project's per-run
`missing_fulltexts.csv` across its registry-designated arms, then subtracting everything actually
on disk, gives the gold-level retrieval gap: **136 gold studies across the corpus have no usable
full text**. "Usable" is stricter than "present" — a file that is a landing or abstract page with
no `<table>` cannot yield coordinates, and 3 of EF's are exactly that.

Because a gold study with no retrievable text can never be recovered by screening, each project
has a hard recall ceiling equal to the fraction of its gold set that is usable. Measuring canonical
recall against that ceiling rather than against the full gold set changes the ordering:

| project | run | recall | gold | unusable | ceiling | **% of ceiling** |
|---|---|---|---|---|---|---|
| social | v3 | 0.929 | 230 | 3 | 0.99 | **94%** |
| dementia | v3 | 0.851 | 73 | 4 | 0.95 | **90%** |
| cue_reactivity | v6 | 0.853 | 191 | 5 | 0.97 | **88%** |
| vbm_of_substance_use | v2 | 0.797 | 79 | 4 | 0.95 | **84%** |
| vbm_of_ptsd | v1 | 0.762 | 22 | 1 | 0.95 | **80%** |
| problem_solving | v2 | 0.540 | 126 | 36 | 0.71 | **76%** |
| emotion_regulation_2022 | v4 | 0.659 | 88 | 9 | 0.90 | **73%** |
| **executive_function** | v3 | **0.386** | 171 | **80** | **0.53** | **73%** |
| **decision_making** | v3 | 0.549 | 87 | 8 | 0.91 | **60%** |

Two conclusions, and they point in opposite directions from the raw numbers.

**executive_function is not the corpus's screening failure.** 80 of its 171 gold studies — 47% —
have no usable full text, so its achievable recall is 0.53, not 1.0. Against that ceiling it
recovers 73%, mid-pack and level with ER. Its headline 0.386 is roughly half retrieval attrition
and half screening. This corroborates, by an independent route, the 58% retrieval-and-parsing
attrition already noted in §5: that figure was computed from pool shrinkage, this one from gold
studies absent from disk, and they agree that EF's binding constraint is upstream of screening.
§3 currently reads EF's false negatives as a specification error; that remains true of the
analysis-level errors examined there, but it should not be extended to the project's headline
recall, which is mostly a retrieval artifact.

**decision_making is the genuine weak point.** Its ceiling is 0.91 — retrieval is nearly fine —
yet it reaches only 60% of it, the worst in the corpus, despite a raw recall (0.549) above EF's.
Any explanation of "which projects screening struggles on" should name decision_making, not EF.

Report the ceiling alongside recall for every project. Reporting raw recall alone rewards projects
whose literature happens to be open-access and penalises ones whose gold sits behind paywalls,
which is a property of the corpus rather than of the method.

**The gap is closable, which is why it is worth reporting rather than just caveating.** Of the 136,
36 resolve to Elsevier and 16 to PMC — both automatable. In this thread ER went from 26 gold
missing to 4, and from 337 to 491 usable full texts of 766, by routing the Elsevier fetcher through
a campus IP (see the retrieval note below). EF holds 24 of the 36 automatable gold papers, so the
single highest-value retrieval action in the corpus is an Elsevier fetch for EF.

**Method note.** The entitlement failures were an IP problem, not a credential one: the same API
key that returns "ScienceDirect rejected FULL view" off-campus returns HTTP 200 with complete
article XML through a SOCKS tunnel to a subscribing IP. No institutional token was required. This
belongs in Methods, because it determines whether a retrieval gap of this kind is a hard limit of
the corpus or a solvable configuration detail — here it was the latter for 213 of 222 attempts.

**A tuned-search outlier to disclose: cue_reactivity v6.** v4/v5's query was written for drug
cues and reached only 24% of the natural-reward gold (drug column: 96%). Two clauses caused it
— the cue/craving clause required "cue"/"craving"/"urge" vocabulary that food and sexual reward
studies do not use, and the reward-type clause carried five natural-reward terms against ~25
drug terms. v6 widens both: natural 24% → 86%, pooled 75% → 93%, drug unchanged at 96%, for
+737 candidates.

The terms were chosen by inspecting which gold studies v4 missed, so **v6 is gold-tuned in a
way no other project's search is**. That is the practice §3 flags as unavailable without a gold
standard, and it makes cue_reactivity an outlier in any cross-project end-to-end comparison.
Disclose it explicitly. Its `load_excluded` arm also grows by ~740 candidates, so its numbers
are provisional until that manual-download backlog is cleared.

**Version labels are not comparable across the two report families, by construction.** The
screening roll-up selects the highest *canonical* run (`v6` for cue_reactivity), while the
analysis and fair-meta roll-ups select the highest *annotation-only* run (`v5-annotation-only`).
The mismatch is not an oversight and must not be "fixed" by producing a `v6-annotation-only`.
v6 differs from its v5 sibling in the `search` block and nothing else — annotation, parsing, retrieval,
screening and output are byte-identical. Annotation-only runs override `search` with a fixed
191-PMID gold list and skip screening, so they are search-independent: a `v6-annotation-only`
config would be identical to `v5-annotation-only`. Running one would re-annotate the same
191 papers and yield a "v6 annotation result" differing from v5 only by LLM nondeterminism —
noise a reader would reasonably misread as an effect of the corrected search. State in Methods
that annotation quality is measured on the fixed gold set and is therefore invariant to the
search version.

**Methodological requirement learned the hard way.** All arms must share a retrieval vintage.
Comparing a pipeline arm from an older corpus against baselines run later silently biases the
result; the effect was large enough here to change the sign of the sub-vs-broad comparison.
Note this in Methods.

---

## 8. Discussion

**8a. The gain varies a lot across projects, and we cannot fully say why.**

The effect of annotation is not uniform: some projects gain substantially, others barely at
all. Two accounts predict that pattern, and they imply opposite things about the method.

- **The target is squishy.** In broad cognitive domains the construct is loosely defined and
  several analyses in a paper are defensibly "the right one". If selection has no single
  correct answer, no selector — human or model — can gain much, and the ceiling belongs to
  the question, not the tool.
- **The model lacks expert judgement.** Choosing the correct contrast is where domain
  expertise lives. A model may handle a clean rule ("bvFTD versus healthy controls") and fail
  a subtler one ("the contrast that isolates the executive demand"), in which case the
  shortfall is a real limitation of LLM annotation.

**§6 now provides one-directional evidence.** The per-project ordering came out as the squishy
account predicts: the two clinical projects requiring a specific population contrast
(vbm_of_ptsd, vbm_of_substance_use) gain ~3× what the cognitive ones do, and the only project
where annotation loses to its baseline (problem_solving, Δr −0.020) is the most diffuse target
in the set. That is a real, tested ordering.

**But it does not discriminate between the two accounts**, because a diffuse construct and a
construct the model handles badly both produce a small Δ. Separating them requires knowing
whether a second qualified rater would have agreed with the published choice. Two gaps block
that:

- **No second rater.** neurometabench carries one decision per analysis — the published
  meta-analysis's. Low LLM–human agreement is therefore ambiguous: the model may be wrong, or
  the target may be loose and the human took one of several defensible options.
- **No human negatives at the analysis level.** The benchmark contains only the analyses the
  authors *included*; within-paper rejections were never recorded. Measured directly, the
  fraction of a paper's analyses that the manual selected is 1.00 in all nine projects,
  because unselected analyses are absent by construction rather than marked as rejected. This
  is also why the existing `assumption` mode has to *reconstruct* negatives, and why its
  precision (0.50–0.64) sits so far below strict precision — that gap is partly an artefact
  of the reconstruction, not evidence of over-selection.

What can be done without new human labour, in rough order of value:

1. **Cross-model agreement** as an ambiguity proxy. Two different models annotating the same
   analyses gives a genuine two-rater measurement. Where they diverge, the construct or its
   specification is ambiguous; where they converge but both differ from the human, that
   points at spec or expertise rather than noise. Some `-gpt` runs already exist. **[partial]**
2. **Test–retest self-consistency.** Re-run annotation with the cache cleared and measure the
   per-project flip rate. Weaker than inter-rater, but it bounds how reliably the spec can be
   applied at all, and we have the screening precedent (8.6% of abstract decisions flip under
   byte-identical criteria). Cheap: one extra run per project. **[need]**
3. **A small hand-rated sample.** ~50 analyses across two contrasting projects (one clinical,
   one cognitive), rated by a second expert, would answer the question properly rather than by
   proxy. This is the only route to real identifiability and is probably worth the cost given
   how load-bearing the claim is. **[need]**
4. **Use the emotion_regulation author contact as the second rater.** This is the strongest
   available option and it is specific to ER: the original author can say whether a contrast
   we selected but they did not was a defensible alternative or a mistake — which is exactly
   the judgement no metric can supply. ER is also the hardest cognitive-selection case in the
   set (§2), so it is where the squishy-target and model-limitation accounts diverge most.
   Even a few dozen adjudicated disagreements would turn §8a from a stated limitation into a
   measurement. **[need]**

**A third possibility the dichotomy misses: the specification was wrong.**
In executive_function all 30 analysis-level false negatives traced to criteria *we* had
written too strictly — an active-control requirement, an overt-response requirement, and a
healthy-adults restriction that the benchmark demonstrably does not apply (it includes
schizophrenia, MDD and adolescent samples). The model applied the specification faithfully
and the specification was wrong.

Two notes on how that was established, because the method matters for what generalises:

- It needed only *our own* rationales checked against the benchmark's inclusions — the model
  states which criterion it applied, so the FNs are attributable without any human negatives.
  That route is available in every project.
- Dementia is separately the only project with recorded human rejection *reasons*, which is
  what let us attribute false *positives* there (wrong patient group, ROI-only) and separate
  eligibility errors from data-availability exclusions. That route is not available elsewhere.

This third mode is indistinguishable from "the model lacks judgement" in any agreement metric
— only reading the rationales separates them, and the fix is entirely different. It is also
the mode §3 is about, which makes §3 and this section mutually reinforcing: before concluding
that a model cannot make a judgement, check whether it was ever asked to.

**A fourth possibility, and it is currently inseparable from the third: the model was never
shown the information.** `pubmed.py:370` reads one `AbstractText` element with `.text`, so a
structured abstract is truncated to its first section. Verified live on PMID 12727696: four
labelled sections, 1,344 characters available, 264 stored. Corpus-wide, stored abstracts under
450 characters run 26.6% (executive_function/v1), 30.6% (dementia/v1) and 28.3%
(cue_reactivity/v6). METHODS and RESULTS — the sections the screening criteria actually ask
about — are frequently gone before the screener reads a word.

This matters because our own decision records cannot tell the two apart. The screening prompt
asks the model to name inclusion IDs "not met **or not demonstrated**"
(`autonima/screening/prompts.py:106`), but only the *met* set is stored structurally, so the
distinction survives nowhere in the data. "The abstract says schizophrenia patients" (criterion
correctly violated) and "the abstract never mentions the sample" (criterion applied to absent
information) both end up as an exclusion with an empty `exclusion_criteria_applied`.

Nor can the not-met set be recovered by subtraction, which is the obvious workaround.
`inclusion_criteria_applied` is *instructed* to be exhaustive (`prompts.py:99`) and inclusion
requires all criteria (`:200`), so in principle not-met = ALL − met. Using included papers as a
ground-truth control — an included paper met every criterion, so it must list every criterion —
that breaks down: of executive_function v1's 1,318 included abstracts only 745 list all four
IDs, so **43.5% under-report**, and the complement invents 716 false "not met" attributions on
records whose true not-met set is empty. Those errors concentrate on I2, the healthy-participants
gate (417 of 716) — the very criterion this section's argument turns on. On excluded papers the
complement matches the prose-named not-met set 90.0% of the time (2,437 vs 270).

So §3's conclusion that executive_function's false negatives are a specification error is
consistent with the evidence but not yet identified against the truncation account. Treat it as
the leading hypothesis, state the confound, and note that recording
`inclusion_criteria_not_met` with an evidence span plus an `abstract_section_count` would settle
it — both are cheap and neither needs a second rater. See DESIGN_DIRECTION.md B1/B1a and D1.

**8b. Reliability.** Run-to-run variance under byte-identical criteria and the same model:
**8.6% of abstract decisions** and 3.7% of full-text decisions flip. Report it — a reviewer
will ask, and it sets a floor below which schema comparisons are noise. Several of our own
apparent schema effects sit inside that band. **[have]**

**8c. The dominant constraint is coordinate extraction, not screening.** Dementia funnel:
99% of gold survives abstract screening, 91% full text, but only **49%** yields an
extractable coordinate. vbm_of_substance_use: 90% search recall × 77% coordinate yield ≈ 69%
ceiling. **That drop is larger than every schema effect in the paper combined.** Presenting
this plainly strengthens rather than weakens the work: it says the LLM components are not the
bottleneck, and it motivates extraction as the next target.

**8d. Where it fails, and what kind of errors.** Wrong patient group; ROI-only designs;
coarse-grained benchmark pooling that makes correct fine selections look wrong; small
sub-annotations where a handful of studies dominate. Also worth stating: **Dice is unusable
at small N** — 8 of 12 sub-annotation comparisons gave exactly 0.000 — so R² on unthresholded
maps should be the reported metric, with Dice at most secondary.

---

## S1. Supplement: cost and scale per project  **[need]**

Not a headline result, but the question every reader with a meta-analysis to run will ask, and
currently unanswered anywhere in the paper.

Per project, report: candidate studies searched, abstracts screened, full texts retrieved,
coordinates extracted, analyses annotated, wall-clock time, and **API cost** — broken out by
stage, since the cost profile is lopsided (abstract screening is high-volume and cheap per
item; full-text screening and annotation are low-volume and expensive per item).

Two things this supports that the main text cannot:

- **The practical case.** If a full project costs on the order of tens of dollars and hours
  rather than months of person-time, that is the argument for adoption, independent of how it
  scores against the benchmark.
- **Where to spend effort.** Pair cost with the funnel from §8c. If coordinate extraction is
  the binding constraint and screening is cheap, the sensible advice is to screen broadly and
  invest in extraction — which is the opposite of where intuition sends people.

Data source: `execution_manifest.json` and `execution_progress.json` per run already record
stage timings and cached-versus-computed counts; cost needs per-stage token accounting, which
may require adding it to the pipeline if it is not already logged. Worth checking before
promising the cost column.

---

## 9. Forward-looking close  **[idea]**

The ceiling on this entire evaluation is the manual meta-analyses themselves — their scale,
resolution and consistency. The natural conclusion is a demonstration that *escapes* the
benchmark:

**A novel use case where manual synthesis was infeasible.** A narrow, specific question where
no one could assemble enough studies by hand, but a broad automated sweep across the
literature surfaces sufficient N to yield a result. Needle-in-a-haystack by construction, and
therefore not validatable against any existing meta-analysis — the validity argument has to
come from §§1–7 instead. **[need]** — no candidate selected yet. Worth choosing the target
question soon, since it likely gates submission.

**Second class of case, and the stronger argument.** Questions requiring precise contrast
selection, where getting the direction wrong is not noise but an inverted result — patients
vs controls being the clearest. This follows directly from §6 and §7: it is exactly where
analysis-level selection is irreplaceable, and it explains *why* the method works where it
does rather than just showing that it does.

---

## Additions worth considering

Beyond the arc above:

1. **An error taxonomy figure** rather than prose. Qualitative FP/FN reports already exist per
   project; a coded taxonomy with frequencies would carry §8d far better than examples.
   **[partial]**
2. **Sensitivity to model, from social.** Social is the natural home for a model comparison,
   because its `v3-*` variants differ along the model axis while holding screening fixed:
   `v3-allstudies` (F1 0.806; renamed from `v3-all_pmids-multi_analysis-ft` on 2026-08-27) vs
   `v3-all_pmids-multi_analysis-ft-gpt52` (0.784) is ready-made. "Does this need a frontier
   model" is the first question a practitioner asks and it is nearly free to answer here.
   Note 0.806 is the *curated-pool* arm, not social's canonical run — canonical `v3` scores
   F1 0.565 on its own search. The model contrast remains valid because both sides screen the
   same 486-PMID pool, but the figure must not be reported as social's headline. **[partial]**

   ~~and the `multi_analysis` variants give a prompt-type one~~ — **STRUCK, 2026-08-26.** There
   is no prompt-type contrast in this project. All 15 social configs resolve to
   `prompt_type: multi_analysis` (9 set it explicitly, 6 omit it and inherit the default, which
   is `multi_analysis` — `annotation/schema.py:30`). No `single_analysis` arm was ever run, in
   social or anywhere else: across all 60 project configs the value is `multi_analysis` or
   unset, and all 57 `config.executed.yaml` files from completed runs record `multi_analysis`. The `-multi_analysis` suffix in the run names is a
   label, not a manipulation: `v3-all_pmids.yaml` and `v3-all_pmids-multi_analysis.yaml` differ
   by exactly one line — the second adds `prompt_type: "multi_analysis"`, i.e. states the
   default the first already inherits. Six of their seven stage hashes are identical; only the
   annotation hash differs, and only because `stage_signature_payloads` hashes the presence of
   the key rather than its effective value, so writing a field's own default busts the cache
   without changing behaviour. Dropped rather than run: a prompt-type arm is not load-bearing
   for any claim in the paper.
3. **Predictors of per-project success.** With 8 projects there is room for a descriptive
   relation between performance and benchmark properties (gold set size, sub-annotation
   granularity, clinical vs cognitive, coordinate availability). Underpowered for inference,
   but it converts §6's hypothesis into something tested rather than asserted.
4. **Reproducibility artifact.** The configs are the method. A released
   `projects/<p>/v*.yaml` set plus the benchmark makes every number here re-derivable and is a
   strong contribution in its own right.

---

## Sequencing note

Three items plausibly gate submission:

1. **emotion_regulation_2022**, which now carries three separate roles — second controlled
   case (§2), hardest cognitive-selection case (§8a), and the only realistic route to
   second-rater adjudication via the author contact (§8a). It also has the smallest immediate
   blocker: a template `nmb_mappings.json` that excludes it from every cross-project analysis.
   Highest leverage per unit of work in the whole plan.
2. **Per-sub-annotation baselines for the other six projects** (§7). The framework exists and
   substance-use is the template, so this is mostly execution and download budget.
3. **A decision on the novel use case** (§9) — likely the longest lead time, so worth choosing
   the target question early even if the analysis comes last.

One cheap item with disproportionate effect: **cost/scale** (§S1) answers the first question a
practitioner asks. Check whether per-stage token accounting is already logged before promising the cost
column.

**Deprioritised for now:** §3 is scoped as a case study rather than a cross-project result,
and is a supplement candidate if space is tight. Promoting it needs two more schema
iterations (n=2 → n=4), which is cheap but not on the critical path.

## Correction: cue_reactivity v6 full-text recall

Commit `ae764e1` records "Full-text recall 0.644 -> 0.970" for cue_reactivity v6. That pair is
wrong — it mixes two different recall definitions and two different comparison runs. From the
per-run `evaluation/performance_metrics.json`:

    run          search recall   fulltext recall_all_meta   fulltext recall_in_search
    v4               0.743              0.644                        0.898
    v5               0.743              0.717                        1.000
    v6               0.932              0.853                        0.970

0.644 is **v4's** `recall_all_meta`; 0.970 is **v6's** `recall_in_search`. Quoting them as a
single before/after overstates the gain twice over.

The correct like-for-like figures against v5 (formerly v5-gpt), whose schema v6 inherits verbatim so that only
the search differs:

  - search recall                0.743 -> 0.932   (+0.189)   [as originally reported, correct]
  - fulltext `recall_all_meta`   0.717 -> 0.853   (+0.136)
  - fulltext `recall_in_search`  1.000 -> 0.970   (-0.030)
  - fulltext precision           0.321 -> 0.310   (-0.011)   [as originally reported, correct]

Two things this changes. First, the end-to-end recall gain is +0.136, not +0.326. Second, and
more useful: `recall_in_search` went slightly DOWN (1.000 -> 0.970, 5 gold papers newly missed at
full text). v6's entire gain comes from the search stage, which is exactly what a search-only
change should do — v5's full-text screening was already perfect on what its search returned.
Report v6 as a search fix, not as a screening improvement.

The cross-project screening roll-up column `recall` is `recall_all_meta`, so the value it lists
for v6 (0.853) is the correct one to quote.

## Note: problem_solving was rebuilt (2026-08-26) — earlier numbers superseded

Two claims made about this project during development were wrong, and the project has since been
rebuilt, so **no problem_solving figure predating 2026-08-26 should be quoted**.

**What was wrong.** (1) I reported that v1's retrieval was broken and that only four papers were
ever screened. That came from a stale `evaluation/` directory dated three months before the
`outputs/` it described; v1 had in fact included 275 studies with 50 gold true positives.
(2) I claimed `max_results: 5000` was a recorded default rather than an authored value — it was
authored, and separately, raising it above 10000 does nothing because NCBI caps esearch at 10,000.

**What actually needed fixing was the query**, which was a transcription error rather than a
design choice: four generic terms (`task`, `number`, `picture`, `verbal`) inflated the pool, and
spatial navigation — one of the source paper's own paradigm groups — was missing entirely.

**Current state**, after replacing the query and collapsing to two versions:

    stage / metric              old query    replacement
    search recall                  0.611          0.881
    screening recall               0.397          0.540
    screening precision            0.182          0.308
    screening F1                   0.249          0.392
    fulltext TP / FP             50 / 225       68 / 153

Better on every axis simultaneously. Old runs are in `projects/problem_solving/archive/` with
`NOTES.md` recording their numbers, including the three-way annotation-only comparison (manual
0.514, GPT-authored 0.485, Claude-authored 0.633 dice) so nothing is lost.

### The generalisable lesson

Per-run `evaluation/` directories can lag their `outputs/` by months, and nothing warns you. A
staleness audit across every run compared in this session found problem_solving/v1 was the only
affected case — decision_making, executive_function and social were all current — but the failure
mode is silent and would have been easy to miss.

The cross-project roll-up regenerates its own evaluations under
`reports/cross_project_screening/evaluations/<project>/<run>/`. **Those are authoritative; the
project-local ones are not.** Any before/after claim should be read from the roll-up, or from a
project-local evaluation whose mtime is confirmed to be newer than its `outputs/`.

## Note: cue_reactivity run naming (v5-gpt -> v5)

`projects/cue_reactivity/v5-gpt` has been renamed to **`v5`** (schema and run directory), and
`v5-annotation-only-gpt` to `v5-annotation-only`. The run itself is unchanged, so every v5 figure
quoted above stands as-is; only the name differs.

The `-gpt` suffix meant that **GPT edited the schema in response to the error reports** — the same
relationship the Claude-generated schemas have to theirs, with GPT as the author. It never meant
"this run called a GPT model": every run in this corpus does, so that would carry no information.
Under the current convention that authorship goes in a header comment rather than a filename, the
suffix is redundant, and each renamed config now states the authorship in its header.

Renamed on the same grounds: `problem_solving/v2-annotation-only-gpt` and
`vbm_of_substance_use/v2-annotation-only-gpt`, both to `v2-annotation-only`.

**Three suffixes are NOT authorship and must stay**, all in `social`:

    v1-annotation-only-gpt5                model variant (gpt-5.2-2025-12-11)
    v1-annotation-only-lc                  config variant (enables local_evidence)
    v3-all_pmids-multi_analysis-ft-gpt52   model variant

Verified: `-gpt5` and `-lc` have annotation criteria *byte-identical* to `v1-annotation-only` and
differ only inside the annotation block. The suffix is the only thing distinguishing them, so
collapsing any would overwrite a run and destroy the comparison. If they should move to the new
convention they need renumbering, not suffix-stripping.

## Naming convention, and an audit against it

**The rule:** the version number tracks the CRITERIA. A suffix tracks anything else — search
window, model, run mode. So multiple `vN-*` variants should share identical criteria and differ
only in those other respects. Corollaries: `-recent` is never canonical (it widens the date window
past the source meta's, so the plain `vN` is the one mirroring the original), and authorship of a
schema edit belongs in a header comment, never in the filename.

Audited every `projects/*/v*.yaml` by fingerprinting screening and annotation criteria separately
(annotation-only runs have no screening criteria, so those are compared on annotation alone).
**Clean:** cue_reactivity, decision_making, dementia, vbm_of_ptsd, vbm_of_substance_use — e.g. `v5`
and `v5-recent` share criteria exactly, as do dementia's `vN` and `vN-allstudies`.

**Violations, all of the same shape** — an `-annotation-only` variant whose annotation criteria
have drifted from the same-numbered full run:

    executive_function v1   v1 / v1-2010 share criteria; v1-annotation-only differs
    executive_function v2   v2 (screening change, v1 annotation) vs v2-annotation-only (annotation change)
    social v2               v2, v2-all_pmids, v2-annotation-only -- three different criteria sets
    social v3               seven variants share criteria; v3-annotation-only differs

`problem_solving v2` was also a violation and is **fixed**: the file I had written as `v2.yaml`
carried annotation criteria byte-identical to `v3-annotation-only`, so it is a v3 and has been
renamed (schema and run directory). `v2-annotation-only` remains the GPT-authored v2.

The `executive_function` and `social` cases are pre-existing and need a decision rather than a
unilateral renumber. For executive_function specifically, v2 currently asserts two different
criteria sets: my `v2.yaml` changed *screening* while keeping v1's annotation, and the
pre-existing `v2-annotation-only` changed *annotation*. Those are two separate criteria versions
sharing one number; one of them should become v3.

## Limitation: the system still depends on a human-written PubMed query

Every stage downstream of search — screening, retrieval, parsing, annotation — is automated, but
the query itself is not. It is written by hand, and problem_solving shows how badly that can go
unnoticed: four generic terms (`task`, `number`, `picture`, `verbal` in Title/Abstract) inflated
the candidate pool to 21,452 while a whole paradigm group from the source paper (spatial
navigation) was missing entirely. Replacing the query raised gold recall from 0.740 to 0.890 while
*cutting* the pool 44%. Nothing in the pipeline flagged this; it surfaced only because we were
scoring against a gold standard.

**The asymmetry worth stating.** Too *narrow* is a correctness failure — unreachable papers cannot
be recovered by any downstream stage, and the ceiling is invisible without a gold standard. Too
*broad* is mostly a cost failure: screening filters the excess, so the price is compute and
credits rather than a wrong answer. The two are not equally bad, and a practitioner without a gold
standard should err broad.

This matters most for OUR evaluation rather than for real use. Because precision here is scored
against a fixed gold set, an over-broad search depresses reported precision even when the extra
hits are legitimately on-topic papers the source meta-analysis simply never screened — the same
floor effect already noted for executive_function (BrainMap-curated pool) and decision_making
(Google Scholar + Web of Science + reference chasing). A real user pays for those extra hits in
compute and sees no correctness penalty.

**Where this leaves the claims.** Search quality is an input to the pipeline, not an output of it,
so it should be reported as a scoping constraint rather than folded into the automation results.
The honest framing: autonima automates everything after the query, and query construction remains
the human bottleneck — one that a capable model can nonetheless improve substantially when given
the source paper and an error report, as the problem_solving replacement demonstrates.

**A hard ceiling worth knowing about: NCBI caps esearch at 10,000 results.** autonima passes
`max_results` straight through as `retmax`, and NCBI returns at most 10,000 ids regardless, with
no warning. Raising `max_results` above 10000 does nothing. problem_solving's replacement query
has a true count of 12,075, so its runs retrieve 9,999 — 83% of the pool.

The damage is limited because autonima searches with `sort="relevance"`, so what is discarded is
the lowest-ranked tail: measured gold recall is 0.890 untruncated versus 0.881 in the slice
actually retrieved, a loss of two papers. Audited the corpus — problem_solving is the only project
anywhere near the cap, the next largest pool being executive_function at 6,447 — so no other
result is affected. But any future query that crosses 10,000 will be truncated silently, and
paging via `retstart` would be the fix if that becomes a constraint.

## Scoping decisions, 2026-08-26

Resolutions to the open items listed above, so they are not re-litigated.

**§S1 cost/scale — descoped from per-project tracking to a single estimate.** There is no token
or cost accounting anywhere in the pipeline outputs; a check across every run found no `usage`,
`prompt_tokens` or `cost` field, and the only timing datum is `started_at`. Rather than
instrument the pipeline for this paper, report a **cost estimate based on expected tokens per
paper**, applied to each project's known volumes (candidates searched, abstracts screened, full
texts screened, analyses annotated — all of which we do have). That answers the practitioner's
question without claiming measurement we did not make. Per-stage tracking is a pipeline feature
request, filed separately, and can inform a later paper.

**§8a inter-rater reliability — not feasible; moves to Discussion as a future direction.** A
second expert rater is not available, so the identifiability limitation stays a stated limitation
rather than becoming a measurement. Do not present cross-model agreement or test-retest as a
substitute for inter-rater reliability; they measure different things. State plainly in Discussion
that whether residual disagreement reflects an ambiguous target or a model limitation is not
identifiable from this data, and name a second-rater study as the way to settle it.

**§3 machine-readable provenance field — deferred.** Version headers now carry authorship and
rationale in prose (and `run_categories.yaml` carries the tier classification), which is enough
for this paper. A structured `provenance:` key in the config schema is a pipeline feature request,
filed separately.

**executive_function verbatim tier = v1.** "Verbatim" means an attempt to stay close to the
paper; a narrowed search window still qualifies, because the CRITERIA are the paper's even though
the 1988-2008 window was inferred from the gold set's range. `v1-2010` is the archived original
window and can be ignored.

### Resolved 2026-08-26 (emotion_regulation_2022)

The ER blockers listed here are cleared. It now has a plain search-driven run (`v2`), so it
appears in the screening roll-up; its annotation criteria were drafted from the paper and
trialled; and its baseline uses the *same query* as the pipeline arm, so the end-to-end
comparison no longer measures a difference between two searches. Its gold was also repaired —
the Sleuth export had been reading the wrong columns.

### The `best` tier, and what peeking bought

`run_categories.yaml` now carries four tiers, ordered by how much gold-standard information
reached the schema:

    verbatim  transcribed from the paper before any results were seen -- the honest attempt
    manual    the author revised by hand, having glanced at a few report examples
    best      THE PREFERRED CONFIG: the run we would put forward, benchmark-informed tuning
              fully allowed. Curated, not derived.
    latest    highest version number. Mechanical, kept as a sanity check and as `best`'s fallback.

`best` was added because version order stopped tracking quality, in two distinct ways. A run can
be renamed *upward* and still be the older, worse set — social's `v5-annotation-only` is the
pre-fix criteria while `v4-annotation-only` carries the multi-label fix. And tuning does not
always help. Naming the preferred run explicitly removes both traps, and `verbatim` → `best` then
measures directly what peeking was worth.

**Selection rule for canonical runs: prefer recall over F1.** Four families had a run that led on
recall and a different run that led on F1. All four were resolved toward recall, on the grounds
that with a mixed pool *it is not knowable why precision fell*. A screened-out "false positive"
may be a perfectly good study that the source meta-analysis never had the chance to reject —
demonstrably so for executive_function, whose benchmark was assembled from BrainMap, and for
decision_making, which drew on Google Scholar and Web of Science as well as PubMed. Precision
against such a benchmark is bounded by the benchmark's own coverage and is therefore a floor
rather than a measurement. Recall carries no equivalent ambiguity: a study the pipeline never
retrieved is unrecoverable by every downstream stage. The rule also matches how the output is
used — a meta-analytic pool tolerates a few extra studies far better than a few missing ones.

| family | recall pick | F1 pick | resolved |
|---|---|---|---|
| emotion_regulation_2022 / canonical | v4 (0.659) | v2 (0.492) | **v4** |
| emotion_regulation_2022 / allstudies | v4 (0.602) | v2 (0.519) | **v4-allstudies** |
| executive_function / canonical | v2, v3 (0.386) | v1 (0.163) | **v3** |
| vbm_of_substance_use / canonical | v2 (0.797) | v1 (0.585) | **v2** |

This rule is stated in `scripts/run_tiers.py` and each entry's `best-reason` records the tradeoff
it accepted, so the cost of the convention stays visible rather than being absorbed into a number.

**Where the honest first attempt still wins.** With the canonical families resolved on recall,
`best` and `latest` now diverge in one place — executive_function / annotation_only, where v1
leads on fair dice (0.566 vs v3's 0.556). Social is the same story one notch weaker: its
v1-annotation-only leads (0.570 vs v4's 0.550), and `best` names v4 there only because v4 carries
the multi-label fix the corpus-wide annotation analysis depends on, which makes v1's edge
non-comparable rather than real.

**This is a result, not bookkeeping.** In two of the corpus's nine projects, benchmark-informed
iteration failed to improve on the honest first attempt at the map level — and both are cognitive
rather than clinical targets (executive function, social processes), the same two that resist
search targeting. Note the asymmetry that the recall rule exposes: in executive_function, peeking
*did* buy better retrieval (recall 0.322 → 0.386) while buying nothing at the map level. Whatever
limits these two projects binds after the studies are in hand. Report `verbatim` → `best` per
project rather than as a corpus mean, because the mean would hide that two of nine are flat.

### Tracked, not yet done

- **Sparse analysis names are a recurring annotation failure, patched per-project.** Analyses
  parsed as `analysis_0` (or with otherwise uninformative names) get rejected for having no
  informative name, and three projects now carry a criterion telling the model that a blank name
  is never itself grounds for exclusion — dementia, executive_function and
  emotion_regulation_2022. Patching it per schema is duplication and will keep being forgotten:
  the fix belongs either in the parser (name analyses from their caption when the label is
  empty) or in autonima's base annotation prompt. Worth doing before the next schema is written.

    **Filed and back-burnered as [autonima#61](https://github.com/neurostuff/autonima/issues/61)
    (2026-08-30), with the first actual measurement.** 1254 of 48992 analyses are sparse-named
    (2.6%) -- executive_function 4.6%, problem_solving 3.2%, ER 2.5%, cue_reactivity 2.1%, social
    1.9%, decision_making 1.6%. Correcting the count above: the patch is in **six configs across
    four projects** (ER v4 and v4-annotation-only, decision_making v3 and v2-annotation-only,
    dementia v5-allstudies, problem_solving v2), each worded differently -- not three projects.
    The workaround is also weaker than it reads: it tells the model to fall back on the
    description, but **74.2% of sparse-named analyses have no description either**, so it prevents
    a spurious exclusion without enabling a correct inclusion. Proposed fix in the issue: parser
    fills `description` from the table caption (closing the 74% gap, and helping every consumer
    rather than only annotation), plus moving the instruction into the base prompt to deduplicate
    the six copies, plus a parse-stage count so this stops being rediscovered. Name synthesis held
    back until the description fix lands. Not on the critical path -- at 2.6% it cannot move the
    headline results.

- **DONE (2026-08-30 audit): ER's deviating screening spec exists, as v4 rather than v3.** The
    entry below was written before v4 and is superseded. v4's header enumerates the exact price of
    each faithful restriction and then drops three of them: film and video now qualify (only
    NON-VISUAL stimuli excluded), any age qualifies, and the strategy definition widened to adjacent
    antecedent-focused regulation and anticipation-period instructions. Result at full text:
    **recall_all_meta 0.500 (v2) -> 0.830 (v4)**, precision 0.484 -> 0.239, F1 0.492 -> 0.372. So
    recall can indeed be bought by departing from the stated criteria, and the answer to "without a
    precision collapse" is **no** -- precision halves. That tradeoff is the finding, and it belongs
    in §3 beside the other cases where a paper does not follow its own criteria. Nothing further to
    run. Original entry retained for the record:

  - ~~**ER needs a v3 screening spec that deviates from the paper to gain recall.**~~ v1 and v2 are
  faithful transcriptions, and faithfulness is costing recall: criterion (4) restricts to static
  pictures, yet 7 of the paper's own 90 included studies use film. A v3 that deliberately departs
  from the stated criteria — admitting film and other non-static visual stimuli, and relaxing
  whatever else the realized inclusions contradict — would test whether recall can be bought
  without a precision collapse. Note this is a *different* v3 from the archived one, which failed
  by being MORE restrictive (its I11 change scored missing coordinates in our retrieved text as
  ineligibility). Numbering will need care.

- **Three naming-convention violations remain**, all pre-existing: executive_function/v1 (its
  `v1-annotation-only` annotation differs from `v1`'s), social/v2, social/v3.
- **cue_reactivity manual-download backlog** for the widened `load_excluded` arm; its numbers stay
  provisional until cleared.

## emotion_regulation_2022: benchmark discrepancies — DIAGNOSED AND FIXED 2026-08-26

**Resolved.** The cause was the Sleuth export, not the source data. `FINAL_DATA_ER.xlsx` was
correct all along; its per-goal sheets each hold MULTIPLE side-by-side Sleuth blocks, and the
export read the wrong ones:

    sheet         blocks (contrasts)        .txt was built from
    Decrease      col A = 128, col H = 152  col H  <-- col H is the reappraisal UNION
    Increase      col A = 23,  col H = 24   col H  (correct)
    Maintain      cols A/E/K = 104 each     truncated to 8
    Reappraisal   col A = 151               (correct, 154)

`Decrease` col H minus col A is exactly 24 labels, 23 of which are the Increase set — so the
"decrease" gold was decrease + increase. File timestamps corroborate it: `Decrease.txt` is dated
May 11 and `Increase.txt` May 13, so Decrease was exported before the increase split existed and
never regenerated.

Rebuilt from `Decrease` col A and `Maintain` col E (cols E/K are cleaned; col A carries two label
typos, `dörfel2014` and `winecoff2011 em42otion>baseline2`). Author originals preserved in
`.original_from_author/`. Post-fix, against Fig. 1:

    construct      before      after     Fig. 1
    reappraisal   154/1590   154/1590   154/1590   exact
    decrease      152/1577   128/1270   130/1284
    increase       24/305     24/305     24/306
    maintain        8/77     104/1408   104/1408   exact

The nimads regenerated to **90 studies — the paper's exact count** — after three
label-convention aliases were added to the fuzzy map (`Albein 2013` -> Albein-Urios 2013,
`MacRae 2012` -> McRae 2012, `vanderVelde 2015a` -> van der Velde 2015a); the Maintain sheet uses
a no-space naming convention the map, built from the other blocks, did not cover. The structural
errors are gone: decrease+increase overlap 23 -> **0**, three-label analyses 23 -> **0**, and
labels-per-analysis is now exactly 1 for maintain (96) and 2 for regulation contrasts (152), which
is what the criteria encode. All four manual reference maps were recomputed.

**Two studies remain unresolved to PMIDs**: `Chen 2017` and `Radke 2017`. Note 90 (paper) = 88
(PMIDs in `included_studies.csv`) + 2, so these are almost certainly the two studies whose PMIDs
were never entered. They carry coordinates and contribute to the ALE maps; only PMID-keyed
comparisons are affected.

### Original diagnosis, retained for the record

Cross-checked the gold Sleuth files against the paper's supplementary Table S1 (266 contrast rows
over 90 studies, carrying per-contrast `Goal` and `Strategy` columns) and against Fig. 1.
**Not corrected** — the gold files are what the original author supplied, so this is a record, not
a patch.

    construct      gold Sleuth   Table S1   Fig. 1     verdict
    studies             --           90        90      S1 agrees with Fig. 1
    reappraisal        154          162       154      gold agrees with Fig. 1
    increase            24           34        24      gold agrees with Fig. 1; S1 counts the
                                                       12 dual-goal contrasts in both buckets
    decrease           152          139       130      was wrong; now 128
    maintain / Look      7          104       104      was wrong; now 104

**Defect 1 — `Decrease.txt` holds the reappraisal union, not the decrease subset.** It is a strict
subset of `Reappraisal.txt` differing by only two labels, and contains 23 of the 24 explicitly
up-regulation contrasts (`Domes 2010 Increase>Maintain`, `Ochsner 2004 Increase>Look Negative`,
`Morawetz 2016b Increase>Look film`, …). Those carry "Increase" in the label itself, so this is not
the legitimate dual-goal set. Expected 130 (Fig. 1) or 139 (S1); the file has 152.

**Defect 2 — `Maintain.txt` is missing ~97 of 104 Look experiments.** Table S1 and Fig. 1 agree
independently on 104; the gold file has 7.

**Consequence for §2's second controlled case.** If ER is scored against the gold as-is, `decrease`
takes ~23 false negatives and `maintain` a large number of false positives, while `reappraisal` and
`increase` should score cleanly. Report only the latter two, or rebuild the two broken files from
Table S1 first. Either way this must not be presented as an ER schema result — the errors are in
the reference data, not the criteria.

### Two things the supplement settled about the criteria

**Decrease and increase are not mutually exclusive.** Table S1 marks 12 contrasts `decr+incr`,
every one a `task > emotion` contrast with `task=reg` — studies that pooled up- and
down-regulation into a single regulate-versus-view contrast. Fig. 1's 130 + 24 = 154 arithmetic
implied disjointness and was misleading; an intermediate version of the ER schema asserted it
before the supplement corrected it.

**The stated static-picture restriction is contradicted by the realized inclusions.** Criterion (4)
says *"Only studies using static visual stimuli (i.e., pictures) were included"*, yet 7 of the 90
included studies are film (Allard 2014, Beauregard 2001, Engen & Singer 2014, Goldin 2008,
Levesque 2003, Levesque 2004, Morawetz 2016a). This is the **third** project where a paper's
stated criteria are contradicted by what it actually included — after executive_function (active
control, age range, overt response) and decision_making (the "healthy adults" clause). That
recurrence is itself a finding worth stating in §3: transcribing a paper's stated criteria
faithfully is not sufficient, because papers do not follow their own stated criteria.

## emotion_regulation_2022 v2: what the screening failures actually are

v2-allstudies loses **12 gold studies to screening** — 4 at abstract, 8 at full text. Every one of
the 12 appears in supplementary Table S1, i.e. the authors did include all of them. Classified by
cause:

### A. Faithful to a stated criterion the meta-analysis itself violates — 9 of 12 (75%)

**A1. The static-picture rule — 4 studies.** Criterion (4) says *"Only studies using static visual
stimuli (i.e., pictures) were included"*. All four excluded studies are among the **7 studies
Table S1 marks `stim = film`**:

    Goldin 2008        17888411   "15-sec film clips"          -> I7
    Allard 2014        24782800   dynamic film/video clips     -> I7 (+ ROI mask)
    Engen & Singer     25698699   film clips (abstract stage)  -> stimulus rule in the objective
    Morawetz 2016a     25631055   extreme-sports film clips    -> I7

**A2. The healthy-adults rule — 5 studies.** Criterion (1) says healthy adults; the gold includes
child and adolescent samples:

    Pitskel 2011       21686071   childhood-to-adolescence      -> no adult group
    Belden 2014        24646887   healthy children              -> no adult group
    Simsek 2017        28372994   girls at risk for depression  -> minors
    Stephanou 2016     26596970   ages 15-25, no adult-only arm -> I8
    Silvers 2015       25439326   ages 10.5-22.9, age analysed continuously -> I8

### B. Other miscategorisations — 3 of 12 (25%)

    Herwig 2007      17588776   I7 read too narrowly. Excluded because regulation targeted the
                                ANTICIPATION of pictures rather than the pictures themselves.
                                S1 lists 3 Herwig contrasts, all task>baseline / goal=decr /
                                stim=picture -- ordinary picture-based down-regulation.
    Kanske 2012      22613776   Judged ROI-only because the paper frames results around AAL
                                amygdala masks. S1 extracts 8 Kanske 2012 contrasts including
                                whole-brain task>emotion (decr), so whole-brain results exist.
    Reinecke 2015    26529426   Judged to lack a separately reported whole-brain contrast for the
                                healthy controls. S1 lists exactly one Reinecke contrast,
                                task>emotion / decr / n=18.

### Why this matters more than the raw recall number

**Three quarters of v2's screening losses are the pipeline being more faithful to the paper than
the paper was to itself.** These are not errors in any ordinary sense — the model applied a stated
criterion correctly and the criterion is one the authors did not follow. Recovering them requires
deliberately deviating from the published methods, which is only knowable *with* a gold standard.
That is the same argument §3 makes from the dementia case, and ER is a second, cleaner instance:
here the deviation is documented in the paper's own supplement.

Effect if each class were fixed (full-text stage, recall within search):

    as-is                       42/50   0.840
    + fix the 5 A-class FNs     47/50   0.940
    + fix the 3 B-class FNs     50/50   1.000

Only the B-class 25% is addressable by better criteria writing. The A-class 75% is addressable
only by choosing to contradict the source paper.

## The pooled best-baseline comparison

Folded into §7 above, which now reports it as the primary result. Retained here as a pointer to
the machinery: `scripts/compile_best_baselines.py` emits
`reports/cross_project_best_baseline.csv`, one row per sub-analysis column carrying both baseline
arms, which arm supplied the best available baseline, whether targeting was possible at all, and
the delta under each definition.

**The rule, for the record:** use the targeted arm wherever one was defined, whether or not it
scores well, because that is what a practitioner aiming at that column would have built. The
alternative — max(targeted, broad) per column — looks more conservative but is the wrong
counterfactual: it lets the baseline switch arms with hindsight, which no practitioner can do.
The two differ by 0.005 (+0.091 vs +0.086), so no conclusion turns on the choice.

## What annotation is worth, given the right studies (§7 addendum)

The cleanest isolation of annotation in the corpus, and it needs no `-allstudies` run — only an
annotation-only run, which every project has. Both sides of the comparison come from the SAME run:
same fixed gold pool, same retrieved texts, same parsed analyses. The only thing that varies is
which analyses enter the map.

    all_analyses   every parsed analysis from those studies, pooled -- perfect study selection,
                   no analysis selection at all
    <column>       only the analyses annotation assigned to that construct

So `annotated − all_analyses` asks: **given that you already found the right papers, how much
closer to the manual meta-analysis does analysis-level selection get you?** Nothing else varies.
Emitted by `scripts/annotation_value.py` as `reports/annotation_value.csv`.

**Result: 35 columns across all 9 projects. Mean dice gain +0.067, median +0.037, positive in
26/35.**

### The gain scales with how selective the target is

    column type              n    mean gain   median   positive
    pooled / global          7      +0.027    +0.009     5/7
    specific sub-analysis   28      +0.077    +0.057    21/28

This is the mechanism, and it is close to arithmetic rather than an empirical surprise. When the
manual column IS essentially every analysis in those papers — `executive_function/all`,
`social/all_merged`, `problem_solving/global_…` — then `all_analyses` is already the right answer
and annotation can only lose by dropping things. Four of the seven pooled columns are negative or
flat. Where the manual column is a genuine subset, annotation is doing real work:
`vbm_of_substance_use/alcohol` +0.446, `stimulants` +0.238, `decision_making/perceptual_dm`
+0.216, `emotion_regulation_2022/maintain` +0.195, `opioids` +0.209.

**This is the same claim §7 already makes mechanistically, now measured directly.** §7 says the
limit on a screening-free pool is that irrelevant analyses *inside retained papers* still
contribute coordinates. Here the papers are held perfect and only the analyses vary, so the effect
is isolated: removing irrelevant analyses within kept papers is worth +0.077 dice on a targeted
column and nothing on a pooled one.

### Annotation F1 does not predict map gain

Pearson r between annotation F1 and dice gain is only **+0.261** across the 35 columns (F1 <0.85:
mean gain +0.052; F1 ≥0.85: +0.075). The two measure different things — F1 scores the labelling
decision on analyses that matched a gold analysis, dice scores the map that results — and a
project can label almost perfectly while gaining nothing, because its target was never selective.
`dementia/decrease` is the clean example: F1 1.000, gain +0.002. Report both; do not treat F1 as a
proxy for map quality.

### Two secondary questions the same family answers

Three projects now have family members sharing criteria byte-for-byte, so they support cross-arm
contrasts (`scripts/decompose_pipeline_stages.py` — note `--version` takes `vN`, not `N`):

    question                                     ER v4    dementia v3    social v3
    1  whole pipeline vs search-only baseline    +0.263      +0.244        +0.172
    2  screening a broad pool vs a curated one   +0.017      -0.054        +0.002
    3  own search vs a hand-assembled pool       +0.008      +0.044        +0.013

Question 3 says running your own search costs almost nothing against a hand-assembled pool,
consistent with §7's finding that search targeting is worth little.

**Question 2 is the one to lead with.** Three independent estimates, spanning +0.017 to -0.054 and
centred on zero, say that automated screening of a broad pool matches hand curation of a targeted
one. It is the most replicated result in the corpus, and unlike question 1 it is not confounded by
the annotation-only arm's gold-restricted pool. It is also the claim most directly relevant to a
reader deciding whether to adopt the method: the labour it replaces is exactly the curation step.

**Not reported: `annotation_only − baseline`.** It is the largest number available (+0.238,
+0.254) and it is tempting, but that arm is restricted to gold studies, so the margin bundles
annotation with being handed a perfect pool. The `all_analyses` comparison above is what that
number was reaching for, done correctly.

**Social became decomposable on 2026-08-27**, and the way it did is worth a Methods sentence.
Its existing arms could not serve: `v2-annotation-only` carried drifted criteria against `v2`, and
the arms that *looked* search-driven (`v3-search-*`) set both a `query` and a `pmids_file`, which
autonima unions rather than choosing between — so each screened its query hits plus the curated
pool containing every gold study, making their search recall 1.000 by construction rather than by
measurement (autonima#57). Two new arms were built to close this: `v3` (the same query, no
injected PMIDs) and `v3-annotation-only` (v3's criteria over the fixed gold list). The decomposition
above uses those. Social's own search reaches 219/230 gold on its own, 0.952, against the 1.000 the
contaminated arms reported.

## What annotation buys once you already have the right studies (§7)

Both sides of this comparison come from the **same annotation-only run**: the same fixed gold
pool, the same retrieved texts, the same parsed analyses. Only the selection differs.

    all_analyses   every parsed analysis from those studies, pooled — perfect study selection,
                   no analysis selection at all
    <column>       only the analyses annotation assigned to that construct

`annotated − all_analyses` therefore answers the practitioner's question directly: **given that
you already found the right papers, how much closer to the manual meta-analysis does
analysis-level selection get you?** Nothing else varies. Emitted by `scripts/annotation_value.py`
as `reports/annotation_value.csv`. Annotation-only runs are used precisely because they have no
screening, so study selection cannot leak into the difference.

**35 columns across all 9 projects: mean dice gain +0.067, median +0.037, positive in 26 of 35.**

    project                   mean gain   positive
    vbm_of_substance_use        +0.156      5/6
    emotion_regulation_2022     +0.144      4/4
    cue_reactivity              +0.081      3/3
    vbm_of_ptsd                 +0.078      1/1
    problem_solving             +0.069      5/5
    decision_making             +0.049      2/3
    dementia                    +0.039      4/4
    social                      -0.007      2/5
    executive_function          -0.028      0/4

The spread is the interesting part. Where a project's sub-analyses are genuinely distinct subsets
of its pool, annotation earns a lot — `vbm_of_substance_use/alcohol` gains **+0.446**, opioids
+0.209, stimulants +0.238, because pooling all analyses from a substance-use corpus buries the
drug-specific signal. Where the sub-analyses are near-synonymous with the whole pool, it earns
nothing: `executive_function` is negative on all four columns, and its `all` column is by
construction almost the same set as `all_analyses`.

### Annotation F1 and map improvement measure different things

Also reported per column is the annotation F1 on matched analyses — of the analyses that matched
a gold analysis, how often annotation put them in the same construct. **Correlation with dice gain
is only r = +0.26**, and the disagreements are systematic rather than noise:

    project              column           annotation F1   dice gain
    executive_function   inhibition           0.957         -0.019
    executive_function   all                  0.955         -0.025
    executive_function   working_memory       0.941         -0.014
    social               all_merged           0.929         -0.025

These are columns where annotation labels almost perfectly and the resulting map is no better than
pooling everything. That is not a failure of annotation — it means **selection had nothing to do**,
because the target subset was already most of the pool. Report both numbers: F1 says whether the
labelling decision is right, dice says whether it changed the answer, and a good schema on an
undifferentiated target will score high on the first and zero on the second.

**Implication for the paper's framing.** Annotation's value is conditional on the sub-analyses
being real subsets. That is a scoping statement about which meta-analyses this tool helps with,
and it is more useful to a reader than an unconditional average would be.

## Correction: social's "search-driven" arms had the gold pool injected (2026-08-27)

Found while checking whether social's existing runs could support a search / screening / annotation
decomposition. They could not, and the reason matters for numbers already in the cross-project
tables.

**The mechanism.** `autonima`'s searcher UNIONS `query` and `pmids_file` when both are set —
`pmids += search_hits`, then `pmids += file_contents` (`autonima/search/pubmed.py`, in `search()`).
The docstring at `pubmed.py:51` claims the query is *ignored* when a PMID source is present, which
is the opposite of what the code does. Filed as
[autonima#57](https://github.com/neurostuff/autonima/issues/57).

**Who was affected.** A corpus-wide audit found exactly four configs setting both keys: social's
three `v3-search-all_pmids-*` arms, plus `projects/template_/v1.yaml` (boilerplate, now commented
out so the pattern stops propagating). No other project was affected.

**Size of the bias.** For `v3-search-all_pmids-multi_analysis-ft` — the arm the cross-project
overrides pointed at:

| | n unique PMIDs |
|---|---|
| query alone | 1024 |
| curated pool (`all_pmids.txt`) | 486 |
| overlap | 445 |
| union actually screened | 1065 |

Gold coverage: **0.952** for the query alone (219/230) → **1.000** for the union. The injected pool
handed the arm 11 gold studies its own search would have missed. So the bias is real but modest —
this is not a case where the pool did the retrieval work. The arm's screening numbers are
optimistic by roughly that margin, and, more importantly, they were never a measurement of search
in isolation.

**What replaces it.** `projects/social/v3.yaml` — query only, no injected PMIDs, criteria and
annotation mode identical to `v3-allstudies` and the new `v3-annotation-only`. The cross-project
overrides now point here. This is social's first genuinely search-driven arm at v3 criteria and the
third leg that makes the project decomposable.

**Two non-differences, corrected.** An earlier note in this outline and in the v3 config headers
listed `prompt_type` and `include_all_analyses` among the reasons social's older runs were not
mode-comparable. Both are wrong: `prompt_type` unset **defaults to** `multi_analysis`
(`config.py:446`), and `include_all_analyses` is the deprecated spelling of
`create_all_included_annotations`, which defaults to `true` — the key is not read anywhere in the
current codebase and is not in the deprecation check's `legacy_keys` list, so it was silently
ignored rather than erroring. Verified by resolving both configs through `ConfigManager` and
comparing `AnnotationConfig.model_dump()`: identical. The two stragglers carrying it
(`social/v2.yaml`, `social/v2-all_pmids.yaml`) were normalised to the current spelling, a
confirmed no-op.

The one real annotation difference stands: **v2 lacks `study_fulltext`** in `metadata_fields`, which
is why it is not a substitute for the new arm.

A caution for anyone counting from these files: `search_results.json` stores the **pre-dedup
concatenation** — 1510 entries for the 1065 unique PMIDs above. An earlier draft of this note
inferred near-zero query/pool overlap from that number before the set arithmetic
(1024 + 486 − 445 = 1065) settled it.

### How much did social's contamination actually buy? (measured 2026-08-27)

`v3` (query only, 1024 studies) vs the archived `v3-search-all_pmids-multi_analysis-ft`
(query ∪ curated pool, 1065 studies), both scored against the same gold set at full-text screening:

| | v3 — clean | v3-search-*-ft — contaminated |
|---|---|---|
| TP / FP | 221 / 323 | 231 / 337 |
| recall (all meta) | 0.93 | 0.97 |
| recall (in search) | 0.98 | 0.97 |
| precision | 0.41 | 0.41 |
| F1 (all-meta recall) | 0.569 | 0.576 |

The injected pool bought **+0.04 all-meta recall at identical precision** — about +0.007 F1. So the
contamination was real but nearly worthless: social's headline screening number is essentially
unchanged, and it is now defensible. This is the reassuring outcome. Had the gap been large, every
social screening number in the cross-project table would have needed a caveat.

Note `recall (in search)` moves the other way, 0.97 → 0.98, because the clean arm's denominator
excludes the 11 gold studies its query never retrieved. That is the metric behaving correctly: the
contaminated arm was penalised for imperfect screening of studies handed to it, while the clean arm
is penalised at the *search* level instead (search recall 0.95, 226/238).

**Not yet measured for v3:** coordinates, annotation, and maps. See the blocker below.

### Blocker: Portkey API key usage limit exhausted (2026-08-27)

The `social/v3` run reached coordinate parsing and then failed every table with

    Error code: 412 - Portkey Error: Portkey API Key Usage Limit Exceeded. Error Code: 04

56 tables attempted, 0 parsed. A direct probe through the gateway reproduces the 412, so the cap is
still in force — this is not transient rate-limiting. The queue was stopped rather than allowed to
continue into `v3-annotation-only`, which would have produced an empty run.

**Nothing was corrupted.** The stages that completed today are valid and cached: search (1024),
abstract screening (recall_all 0.95, precision 0.33), retrieval (835 pubget files, no LLM cost),
full-text screening (the table above). `coordinate_parsing_results.json` still holds the artifact
copied from `v3-allstudies` — the stage was killed before writing — and no per-table or LLM cache
recorded the failures. A re-run resumes at parsing.

**One hazard to watch on that re-run.** `stage_signature_payloads()` in `autonima/execution.py`
builds each stage's signature from **its own config block plus a prompt version only** — signatures
do not chain on upstream artifacts. So the `annotation` artifact copied from `v3-allstudies` is
considered valid for `v3` even though the two runs' study sets differ. Screening demonstrably
gap-fills per item rather than skipping wholesale ("578 to screen, 445 cached"), so annotation
probably does the same, but this must be verified after the re-run: check that
`v3/outputs/annotation_results.json` has a fresh mtime and covers v3's included studies, not
v3-allstudies'.

**Blocked on the cap:** v3 parsing/annotation/output, all of `v3-annotation-only`, social's
decomposition, and the map-level half of the cross-project regeneration. Social's *screening* row
is complete and can be regenerated now.

---

## Retrieval expansion and full-corpus re-run (2026-08-27/28)

The Portkey cap that blocked social is lifted, and everything queued behind it has run. Social's
three decomposition arms (`v3`, `v3-allstudies`, `v3-annotation-only`) are complete, so social joins
dementia and emotion_regulation_2022 as decomposable.

**The cache hazard flagged above is resolved, empirically.** `stage_signature_payloads()` does not
chain on upstream artifacts, so a copied annotation artifact validates against a different study
set. But `_stage_action()` has no wholesale skip: a valid unchanged stage returns `incremental`
("signed entries will be validated against current inputs"), and reuse is gated per study on
`study_input_hash`. On the v3 re-run annotation gap-filled exactly as screening does -- "Processing
330 studies for 'all_studies' annotation (missing or incomplete cache)", then 219, then 161 --
taking decisions from 7,751 to 14,806. No silent skips.

### What the retrieval campaign added

Three routes, in descending yield:

1. **Manual proxied downloads.** ~450 papers via the UT campus SOCKS tunnel.
2. **Elsevier API through the same tunnel.** The 201 "ScienceDirect rejected FULL view" failures were
   an *IP* problem, not a credential problem: the same `ELSEVIER_API_KEY` with no institutional token
   succeeds from a campus IP. Routing the fetcher through the tunnel via `ELSEVIER_HTTPS_PROXY`
   turned 213 of 222 ER attempts into successes, and a later sweep covered every project. The route
   is now exhausted -- zero PMIDs across all nine projects have an Elsevier link and no attempt.
3. **A silent ingest bug worth more than either.** `scripts/ace_ingest_and_export.py` discovers input
   with `glob('*/*')`, which requires exactly one directory level below `html/`. Files written flat
   as `html/<PMID>.html` sit at depth 1 and were **never matched**, and the parent directory doubles
   as the source name. Measured ingestion rate by layout: flat 1/400 = **0.2%**, journal subdirectory
   5343/5373 = 99.4%, `html/Manual/` 546/550 = 99.3%. 17 of 26 ER gold files were flat, so they had
   never reached ACE at all. Moving them and re-ingesting recovered coordinates for 13 of 26 gold
   immediately.

Two data-integrity traps in the ACE database, both of which make it misleading to diagnose from:
every row in `activations` has `article_id` NULL (84k rows), so any `articles`-to-`activations` join
returns nothing; and all rows in `tables` have `n_columns` NULL while `n_activations` is populated.
Coordinates are only attributable through the CSV export, which resolves `pmid` correctly.

### The full-corpus re-run

Affected runs were identified by intersecting each run's own `missing_fulltexts.txt` against what was
newly on disk: **34 of the 38 runs** in `run_categories.yaml` gained text (1,862 studies). All 34 ran,
then 34 metas, then the cross-project reports. Every run and meta exited 0.

Full-text screening, before -> after:

| run | recall | F1 |
|---|---|---|
| emotion_regulation_2022/v4 | 0.659 -> **0.830** (+0.170) | 0.420 -> 0.372 |
| problem_solving/v1, v2 | 0.540 -> 0.579 (+0.040) | 0.392 -> 0.373 |
| vbm_of_substance_use/v1 | 0.759 -> 0.785 (+0.025) | 0.585 -> 0.577 |
| decision_making/v1, v3 | +0.007 | flat |
| executive_function/v1, v3 | +0.006 | flat |

The consistent shape is **recall up, F1 down**: newly retrieved studies bring false positives with
them. `cue_reactivity/v6` is the extreme -- recall unchanged at 0.853 while F1 fell 0.455 -> 0.384,
so its 68 new studies produced only false positives. That is worth stating plainly in §7: retrieval
is not free, and past some point it buys recall by spending precision.

Gold studies still lacking full text fell **179 -> 132**, concentrated in ER (26 -> 5) and
vbm_of_substance_use (13 -> 2). executive_function remains the largest gap at 63.

### PubMed returned a degraded corpus mid-run (RETRACTED as drift, 2026-08-29)

Two runs *lost* recall (`dementia/v1` -0.027, `cue_reactivity/v1` -0.037). Neither loss came from the
retrieval work. `search` is always an incremental stage, so **every re-run silently re-queries
PubMed**, and for these two the corpus came back smaller:

| run | search pool | gold in search |
|---|---|---|
| dementia/v1 | 2027 -> **1282** (-37%) | 71 -> 69 |
| cue_reactivity/v1 | 1170 -> **1057** | 142 -> 134 |

**This was NOT index drift.** Re-probed on 2026-08-29, both queries return their original counts
consistently: cue_reactivity 1170 (stored 1170) and dementia 2028 (stored 2027, +1 new record),
three attempts each. PubMed was in a degraded state during the re-run window and has since
recovered. My original conclusion -- "genuine index drift, not a transient API failure", justified
by the counts reproducing three times -- was wrong. Three identical results seconds apart rule out a
one-off failure; they say nothing about a condition that persists for hours. The classification diff shows
the mechanism -- every bucket lost studies and **gained none** (cue_reactivity TP 125->118, FP
269->260). Decisions did not change; studies left the pool.

Three consequences:

- Those two before/after rows compare different study pools and **must not** be cited as an effect of
  retrieval work.
- `dementia/v1` and `cue_reactivity/v1` must be RE-RUN. Their current outputs were built on a
  corpus PubMed was under-reporting by 37% and 10%, so their metrics and maps are wrong -- including
  the dementia map-level drop (0.330 -> 0.280) reported in the §7 correction, which is an artifact
  of the degraded corpus and not a retrieval effect.
- The reproducibility point survives, but in weaker and more accurate form. `search` is always an
  incremental stage, so every re-run silently re-queries PubMed and inherits whatever state the
  service is in that hour. The failure mode is not slow index drift but transient degradation that
  is invisible unless you diff the corpus size against the stored run. Pinning via `pmids_file`, or
  at minimum asserting the search count against the previous run and refusing to proceed on a large
  drop, would have caught this within seconds. That belongs in the Discussion next to the
  human-written-query limitation at §1060.
- `cue_reactivity/v1` sets no `email` in its search config, which NCBI warns about. Not the cause
  here, but it should be set.

### A measurement trap worth not repeating

`autonima run` does **not** write evaluation metrics. `projects/<p>/<run>/evaluation/` is refreshed
only by `scripts/run_cross_project_screening_reports.py`. Reading those files straight after a run
shows every metric unchanged, which reads as "the re-run did nothing" when it means "evaluation has
not been recomputed". Separately, `compile_missing_fulltexts.py` defaults its gold file to
`<project>/annotation-only-ids.txt`; three projects (cue_reactivity, vbm_of_substance_use, social)
do not have one, and the script silently reports **0 gold missing** rather than failing. Both are
failure modes that produce a plausible wrong number rather than an error.

### Correction: §7 was NOT regenerated with the rest (2026-08-29)

I reported that the map-level headline "did not move, byte-identical to before the re-run", and
explained it as the best-baseline comparison being driven by the gold-restricted `annotation_only`
family. **Both the finding and the explanation were wrong.**

`compile_best_baselines.py` is only an aggregator: it reads
`projects/*/reports/baseline_vs_autonima.csv`. Those files are written by
`scripts/compare_baselines_to_benchmark.py`, which was **not in the report chain**. Every one of
them predated the corpus re-run (newest 2026-08-27 06:43; the re-run started 16:57). The output was
byte-identical because its inputs were never regenerated -- not because the underlying maps are
fixed.

Regenerated for all nine projects at `--tier best`, then re-aggregated. **The map-level results did
move: 30 of 35 columns changed.**

  best AVAILABLE (targeted if any)     0.487 -> 0.495  vs 0.396   +0.091 -> +0.099   ahead 29 -> 30/35
  STRONGEST (max of targeted, broad)   0.487 -> 0.495  vs 0.401   +0.086 -> +0.094   ahead 27 -> 29/35

Largest column moves: social/self_merged +0.176, cue_reactivity/3_natural_neutral +0.109,
social/affiliation_merged +0.076, vbm_of_su/alcohol -0.069, problem_solving/visuospatial +0.066,
dementia/all -0.061.

Two things this changes in the narrative:

- **The retrieval campaign did buy map-level gain**, not just screening recall. §7's central claim
  strengthens rather than staying flat.
- **§7 no longer has two losses to report.** social went -0.007 -> +0.057 and decision_making
  -0.015 -> +0.001. The "report the losses rather than burying them" paragraph has been rewritten;
  decision_making is now a tie rather than a loss, and should be described that way.

`dementia` moved the other way (0.330 -> 0.280) and is now barely ahead of its baseline (+0.005).
Note dementia/v1 is also one of the two runs whose PubMed corpus shrank 37% this cycle, so its
map-level drop is confounded with search drift and should not be read as a retrieval effect.

**Process lesson worth keeping:** an aggregator that runs clean and emits an unchanged file is
indistinguishable from a correctly-unchanged result. When a re-run produces byte-identical
aggregate output, check the mtimes of its inputs before explaining the result.

---

## Dementia is sensitive to the final study list, not to annotation (observation, 2026-08-30)

**The observation, and a correction to my first reading of it.** I described the `allstudies` arm as
the unfiltered pool "before screening or annotation enter the picture". That is wrong: `allstudies`
runs annotation with the *same* criteria as the canonical arm (verified — `annotation.enabled: true`
and identical `n_criteria` in dementia/v5-allstudies, v3-annotation-only and v3, and likewise for ER
and social). What separates the arms is **which studies reach the pool**, not whether annotation
happened. So an allstudies-vs-canonical difference isolates study selection, i.e. screening.

**What prompted it.** dementia/v3's map moved a lot on almost no input change:

| run | studies | analyses | coordinates |
|---|---|---|---|
| dementia/v3 | 62 -> 62 | 332 -> 333 | 1747 -> 1761 (**+14**) |
| cue_reactivity/v6 | 357 -> 446 | 1896 -> 2340 | +2520 |
| emotion_regulation_2022/v4 | 131 -> 207 | 749 -> 1126 | +2394 |

A 0.8% coordinate change moved dementia's r2 by up to 0.06 across all four columns, while the two
projects that gained ~170x more coordinates both improved. Three explanations were tested and two
died:

- *Degraded PubMed corpus* -- **ruled out.** §7 uses dementia/**v3**, whose search pool was 2027
  before and after; only the v1 runs were caught in the degraded window.
- *Meta-analysis nondeterminism* -- **ruled out.** Re-running dementia/v3's meta on byte-identical
  inputs reproduced every column to four decimals (max |delta| 0.0000). There is no seed in
  `autonima/meta.py`, so this was worth checking, but the pipeline is deterministic and the §7
  per-column changes are real signal.
- *The study list itself* -- **the surviving explanation.**

**Supporting numbers.** allstudies -> gold, mean dice: dementia **0.329**, social 0.578, ER 0.598.
Dementia's `functional` column is 0.137, the worst in the set. Against its annotation-only arm at
0.470, dementia gains **+0.141** from a narrower pool where ER gains +0.026 from a pool that already
starts high. Pool sizes differ sharply too: dementia 62 (v3) / 61 (allstudies) / 39
(annotation-only), against ER 207 / 104 / 64.

**Interpretation.** Dementia's map metric is operating in a shallow, unstable region: its pool aligns
weakly with the benchmark, so small membership changes move the thresholded map disproportionately
(dice on thresholded maps is discontinuous -- a few coordinates can tip voxels across a
cluster-forming threshold). The fragility is a symptom of a weakly-aligned pool, not an independent
effect, and it is consistent with dementia's weak §7 margin (+0.005 over baseline) sitting alongside
a strong annotation gain (+0.116): annotation is rescuing a poor pool up to roughly where ER's pool
already begins.

**Follow-up worth doing.** Diff the final studysets directly -- which PMIDs are in
dementia/v3 but not v5-allstudies, and vice versa, and which of those carry the coordinates driving
the moved columns. That converts "sensitive to the study list" from an inference into a named set of
studies, and would say whether a handful of high-leverage papers dominate the metric. The same diff
on ER, where the metric is stable, gives the contrast case.

**Future step: non-search baselines (NeuroQuery, NeuroVLM).** Every baseline in §7 is a *search*
baseline -- a PubMed query, coordinates extracted, pooled. That tests the pipeline against the
Neurosynth-style workflow but not against the current generation of automated meta-analytic map
generators. Adding **NeuroQuery** and **NeuroVLM** as additional arms would ask a different and
harder question: not "does screening beat pooling your own search?" but "does building a curated
studyset beat asking an existing model for a map of the same construct?". Both produce maps directly
from a text query, so they slot into the existing per-column dice/r2 comparison without needing a
studyset at all -- the arm is the map. Worth having for the same reason `baseline_sub` was worth
having: the broad search baseline is a weak opponent, and a reviewer will ask what happens against a
strong one.

**Caveat on the cross-arm table.** Each arm is its own `best` tier, so the rows differ in run *and*
column set (dementia 4 columns for allstudies vs 5 for annotation-only; ER 4 vs 7). The gains above
are indicative, not a controlled contrast.

---

## The dementia diff, and a coordinate-duplication finding (2026-08-30)

Ran the studyset diff flagged above. The answer is sharper than the hypothesis: dementia/v3's map
moved **without any change to its study list at all**.

    dementia/v3 before -> after:  studies 62 -> 62,  added 0,  removed 0
    coordinate counts changed in exactly TWO studies:
      11805245   21 -> 34  (+13)
      27258418   38 -> 39  (+1)

So a 0.06 dice swing across all four columns traces to essentially **one paper**. Contrast ER/v4
over the same window: 76 studies added, +2394 coordinates, and its columns moved +0.007 on average.

### What happened in that paper is a parsing defect, not new data

PMID 11805245's re-parse split two conflated contrasts into four:

    BEFORE                                   AFTER
    FTD (n = 8) vs controls and SemD    6    FTD vs controls                    6
                                             FTD < SemD (direct comparison)     6
    SemD (n = 12) vs controls and FTD   7    SemD vs controls                   7
                                             SemD < FTD (direct comparison)     7

Splitting the conflated label is right. Assigning the *same coordinates to both halves* is not --
the pairs are byte-identical point sets:

    FTD vs controls   vs  FTD < SemD    shared 6/6   (both start 38, 18, -6)
    SemD vs controls  vs  SemD < FTD    shared 7/7   (both start -29, 11, -42)

So the "+13 coordinates" is duplication, and the study is now double-weighted in the pooled
meta-analysis. That is what moved dementia's map, and it moved it the wrong way.

### It is corpus-wide

Scanning every run for studies where two analyses carry *fully identical* coordinate sets:

| run | studies affected | of | duplicated points |
|---|---|---|---|
| executive_function/v3 | 36 | 500 | 1774 |
| executive_function/v1-2010 | 33 | 522 | 1543 |
| executive_function/v2 | 30 | 496 | 1860 |
| cue_reactivity/v6 | 27 | 446 | 678 |
| executive_function/v1 | 27 | 377 | 1389 |
| emotion_regulation_2022/v4 | 14 | 207 | 494 |
| problem_solving/v1 | 14 | 216 | 1242 |
| social/v2 | 13 | 308 | 1549 |

56 runs in total contain at least one such study. In executive_function/v3 that is 7.2% of studies.

**Hedge before this is cited.** Identical coordinate sets across two analyses are *strong* evidence
of one table being assigned to several labels, but not proof: a paper can legitimately report the
same peak table under two headings, and conjunction analyses can share peaks by construction. The
11805245 case was verified by hand against its analysis names and is unambiguous. The corpus-wide
count is a screen, not an audit -- it needs a sample checked against source papers before the
7.2% figure is quoted.

**Why it matters regardless.** Coordinate-based meta-analysis weights studies by their reported
peaks, so a duplicated table silently doubles one study's influence. This is a plausible contributor
to the recurring finding that annotation F1 barely predicts map gain (r = +0.261): if map quality is
partly hostage to parse-level duplication, annotation quality would not track it. Fixing this is a
candidate lever on the map-level results that has nothing to do with screening or annotation.

**Follow-up.** Sample ~20 flagged studies across projects, check against source PDFs, and if the
duplication is confirmed as a defect, dedupe identical point sets within a study at parse time and
re-run the affected metas to measure how much of the map-level signal it was costing.

### Verified, and the fix does NOT improve maps (2026-08-30)

Followed through on the three steps flagged above: sample, verify, dedupe and measure.

**Verification — the duplication is real and its cause is upstream of autonima.** Sampling 20 of the
163 flagged PMIDs, most pairs carry names describing *genuinely different* contrasts while sharing
identical coordinates ("Younger adults - Transient minus Sustained" vs "Older adults - ...",
"Group pattern for words" vs "... for faces", "Main effect of GROUP" vs "Genotype x Alcohol Group
interaction"). Two also appear under byte-identical names, and two pair a real name against `None`
(the known sparse-label problem).

Tracing PMID 17494060 to source: ACE exported **10 table files with only 5 distinct contents** --
each table stored twice under two ids (1368-1372 and 2220-2224). So the same tables were ingested
twice and the LLM, parsing each copy independently, produced *different* analysis names for the same
coordinates. That explains both the identical point sets and the contradictory labels.

Corpus-wide, over `ace_outputs/processed/tables`:

    PMIDs with extracted tables    2551
    PMIDs with duplicate tables     759   (29.8%)
    redundant table files          1412   (22.5% of all tables)

Worst cases are exactly 2x (16 files -> 8 distinct, 12 -> 6, 10 -> 5), consistent with double
ingestion rather than partial re-extraction.

**Not caused by the flat-file move.** The 474 files relocated into `html/Manual/` on 2026-08-28 were
an obvious suspect, since ACE keys work off file paths. They are cleared: duplicate-table rates are
10.5% among Manual PMIDs (which include every moved file) against 12.1% among journal-subdirectory
PMIDs that were never touched. The defect is pre-existing and uniform.

**The measurement, and it refutes my own hypothesis.** I predicted deduping would recover map
quality, on the reasoning that double-weighted studies distort a coordinate-based meta-analysis.
Deduping dementia/v3 by identical point-set within study (6 studies, 10 analyses removed, annotation
notes filtered 513 -> 503 to match) and re-running the meta:

| column | with duplicates | deduped | delta |
|---|---|---|---|
| 0 | 0.4164 | 0.3778 | -0.0386 |
| 1 | 0.4236 | 0.4109 | -0.0127 |
| 2 | 0.3787 | 0.3687 | -0.0100 |
| 3 | 0.2941 | 0.2977 | +0.0036 |
| **mean** | **0.3782** | **0.3638** | **-0.0144** |

Removing the duplicates made agreement with the gold maps slightly **worse**. The double-weighted
studies happened to align well with the benchmark, so their extra weight was accidentally helping.

**What this does and does not license.**

- It remains a genuine data-integrity defect: a study's influence in the meta-analysis should not
  depend on how many times ACE happened to ingest its tables. Two runs of the same pipeline over the
  same corpus can weight a paper differently for reasons that have nothing to do with the paper.
- It does **not** support the claim I made when flagging it -- that fixing this is a lever on the
  map-level results, or that it explains annotation F1 failing to predict map gain (r = +0.261).
  That speculation is withdrawn.
- The dementia test is small (10 analyses over 6 studies). `executive_function/v3` carries 36
  affected studies and 1774 duplicated points and would be a far stronger test; if the effect there
  is also near-zero or negative, the honest conclusion is that CBMA is robust to this level of
  duplication and the fix is worth making for correctness alone.

**Back-burnered.** Filed upstream as
[autonima#60](https://github.com/neurostuff/autonima/issues/60) as a general data-integrity issue to
diagnose and fix, rather than pursued further here. It is not on the critical path for the paper:
the measured map-level effect is slightly negative, so nothing in the results depends on fixing it.
It matters for the reproducibility claim -- a paper's weight should not depend on ingest history --
which is where it belongs in the Discussion, next to the search-corpus point.

Test artifacts: `projects/dementia/v3-dedup` (deduped copy, deliberately not a registry run) and
`projects/dementia/reports/{dedup_test,dedup_baseline}`.

---

## Stage decomposition: all three decomposable projects (2026-08-30)

Social's three matched arms are complete, so the decomposition now covers **three** projects, not
two. Re-ran all three against the current (post-retrieval, post-baseline-rerun) results.

Mean dice per arm, and the three questions the arms can actually answer:

| project | n | baseline | annot-only | allstudies | full | Q1 pipeline vs search-only | Q2 screening vs curation | Q3 own search vs fixed pool |
|---|---|---|---|---|---|---|---|---|
| emotion_regulation_2022 | 4 | 0.328 | 0.585 | 0.601 | 0.609 | **+0.282** | +0.017 | +0.008 |
| dementia | 4 | 0.168 | 0.422 | 0.368 | 0.412 | **+0.244** | **−0.054** | +0.044 |
| social | 5 | 0.408 | 0.570 | 0.572 | 0.585 | **+0.176** | +0.002 | +0.013 |

Social per column:

| column | baseline | annot | allstud | full |
|---|---|---|---|---|
| affiliation_merged | 0.258 | 0.401 | 0.384 | 0.446 |
| all_merged | 0.690 | 0.731 | 0.734 | 0.752 |
| others_merged | 0.468 | 0.655 | 0.665 | 0.644 |
| self_merged | 0.202 | 0.489 | 0.494 | 0.474 |
| soccomm_merged | 0.423 | 0.573 | 0.583 | 0.606 |
| **mean** | **0.408** | **0.570** | **0.572** | **0.585** |

**What holds across all three.**

- **Q1 — the whole pipeline beats a search-only meta-analysis everywhere**, by +0.176 to +0.282.
  This is the §7 claim restated at map level on matched arms, and it is the robust one.
- **Q2 — automated screening matches hand curation in two of three.** ER +0.017 and social +0.002
  are ties; screening a broad pool reaches what a hand-assembled pool reaches. This remains the most
  load-bearing finding in the decomposition, because it is the step a sceptic assumes cannot be
  automated.
- **Q3 — running your own search costs almost nothing**, +0.008 to +0.044, and is *positive* in all
  three. Combined with the earlier result that targeting the search buys ~nothing (+0.023 mean,
  median ~0), the search stage is neither a strength nor a liability: it is close to free.

**Dementia is the exception on Q2 (−0.054), and its cause is now known.** The curated pool still
beats screening there. That sits with everything else established about dementia: the weakest
allstudies→gold alignment in the corpus (mean dice 0.329 against ER 0.598 and social 0.578), the
largest annotation gain (+0.116, rescuing a poor pool), and a map metric so unstable that
13 duplicated coordinates in a single paper moved every column by up to 0.06. Report it as the
project where pool quality dominates, not as evidence that screening generally underperforms
curation — two of three say otherwise.

**Caveat to carry.** None of these is a clean stage effect. Each arm changes the *pool* at the same
time as the stage, and `annotation_only` is restricted to gold studies, so its margin over baseline
bundles annotation with a perfect pool. That is why no annotation share is quoted — the arms cannot
isolate it.

---

## Dementia's benchmark aggregates studies — the likely root cause of its whole profile (2026-08-30)

The dementia source meta-analysis **combined multiple studies sharing a data source into single
analyses**. Our extraction is per paper. So the two sides are not counting the same units, and no
amount of screening or annotation quality closes that gap.

This is a benchmark-comparability property, not a pipeline weakness, and it retro-explains
essentially every dementia anomaly recorded above — all of which I had previously attributed to
"pool quality":

- **Weakest allstudies -> gold alignment in the corpus** (mean dice 0.329 against ER 0.598, social
  0.578). If gold analyses pool several papers, a per-paper map cannot reproduce them.
- **The only negative Q2 in the stage decomposition** (-0.054, curated pool still ahead of
  screening). A hand-curated pool can be assembled to match the benchmark's aggregation; automated
  screening over individual papers cannot.
- **A map metric unstable enough that 13 duplicated coordinates in one paper moved every column by
  up to 0.06.** Fewer, larger gold units means each of our per-paper contributions carries more
  relative weight.
- **The largest annotation gain in the corpus** (+0.116, 0.329 -> 0.470). Annotation is partially
  compensating for a unit mismatch rather than only selecting better analyses.

**Consequences.**

- dementia is excluded from analysis-level annotation figures (§5), for this reason.
- Its map-level numbers should be reported with the caveat attached, not silently pooled. Its §7
  margin (+0.005 over baseline) is a *lower bound* distorted by unit mismatch, not a measurement of
  what the pipeline achieves on a comparably-structured benchmark.
- **Supersedes the earlier interpretation.** The note above concluding dementia is "the project where
  pool quality dominates" is withdrawn: the pool is not obviously bad, it is being scored against
  differently-shaped units.
- Worth checking whether any other benchmark aggregates this way. If dementia is the only one, this
  is a footnote; if two or three do, unit-matching becomes a methods requirement for the whole
  evaluation.

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

## 5. Parsing and analysis-level annotation  **[have / partial]**

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
  for all 8 projects, 31 sub-annotations.

**Full result — all 8 projects, 31 sub-annotations**, mean R² per project:

| project | n | autonima | targeted search | broad search | autonima − targeted |
|---|---|---|---|---|---|
| vbm_of_ptsd | 1 | **0.456** | 0.247 | 0.247 | +0.209 |
| cue_reactivity | 3 | **0.599** | 0.457 | 0.433 | +0.142 |
| vbm_of_substance_use | 6 | **0.255** | 0.153 | 0.123 | +0.102 |
| problem_solving | 5 | **0.629** | 0.560 | 0.549 | +0.069 |
| dementia | 4 | **0.330** | 0.275 | 0.272 | +0.055 |
| executive_function | 4 | **0.633** | 0.612 | 0.598 | +0.021 |
| social | 5 | 0.520 | 0.527 | **0.529** | −0.007 |
| decision_making | 3 | 0.346 | **0.367** | 0.271 | −0.016 |

**autonima wins 6 of 8 projects.** Report the two losses rather than burying them: `social`
is the project with the corpus's weakest annotation, and `decision_making` is the only case
where a narrowed baseline beats the pipeline outright.

And the key secondary finding, which **now holds corpus-wide rather than on one project**:
**targeting the search is worth almost nothing on its own.** Mean targeted − broad across all
eight projects is **+0.022** (per-project range −0.002 to +0.096), despite the targeted queries
cutting candidate pools several-fold while retaining the same gold studies. `decision_making`
is the sole project where targeting buys anything substantial (+0.096) — and notably it is also
the project where the targeted baseline beats autonima, which is consistent rather than
contradictory: where search targeting *does* work, the pipeline's advantage narrows.

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
   `v3-all_pmids-multi_analysis-ft` (F1 0.806) vs `-gpt52` (0.784) is ready-made. "Does this
   need a frontier model" is the first question a practitioner asks and it is nearly free to
   answer here. **[partial]**

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

### Tracked, not yet done

- **emotion_regulation_2022 needs a plain `vN.yaml`** (not `-allstudies`). All three existing ER
  configs are `-allstudies` variants, so the project has no canonical family at all and cannot
  appear in the screening roll-up, which selects plain `vN`. Wanted eventually; no blocker.
- **ER annotation criteria are drafted but untrialled.** They live in `v3-annotation-only.yaml`
  deliberately: trial there first (88 gold PMIDs, screening skipped, annotation is the only stage
  under test), then port to `v3-allstudies.yaml` and re-run that with the stage enabled. Under the
  naming convention those two must then share the criteria byte-for-byte. Numbered v3 because all
  three -allstudies runs already exist and v3 is the latest; its annotation stage was never
  configured rather than deliberately left different.
- **Three naming-convention violations remain**, all pre-existing: executive_function/v1 (its
  `v1-annotation-only` annotation differs from `v1`'s), social/v2, social/v3.
- **cue_reactivity manual-download backlog** for the widened `load_excluded` arm; its numbers stay
  provisional until cleared.

## emotion_regulation_2022: benchmark discrepancies against the supplementary material

Cross-checked the gold Sleuth files against the paper's supplementary Table S1 (266 contrast rows
over 90 studies, carrying per-contrast `Goal` and `Strategy` columns) and against Fig. 1.
**Not corrected** — the gold files are what the original author supplied, so this is a record, not
a patch.

    construct      gold Sleuth   Table S1   Fig. 1     verdict
    studies             --           90        90      S1 agrees with Fig. 1
    reappraisal        154          162       154      gold agrees with Fig. 1
    increase            24           34        24      gold agrees with Fig. 1; S1 counts the
                                                       12 dual-goal contrasts in both buckets
    decrease           152          139       130      GOLD WRONG
    maintain / Look      7          104       104      GOLD WRONG; S1 and Fig. 1 agree at 104

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

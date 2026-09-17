---
title: "Selecting analyses, not papers: a large language model harness for automated neuroimaging meta-analysis"
subtitle: "Nature Methods — Article. Target 3,000 words, 6 display items, ~50 references."
date: "Draft — 2026-09-10"
---

::: {custom-style="Affiliation"}
Alejandro de la Vega^1^, James D. Kent^1^, Nicholas Lee^1^, Thomas E. Nichols^2,3^, Jean-Baptiste Poline^4^, Katherine L. Bottenhorn^5^, Angela R. Laird^6^

^1^ Department of Psychology, University of Texas at Austin, Austin, TX, United States

^2^ Nuffield Department of Population Health, University of Oxford, Oxford, United Kingdom

^3^ Centre for Integrative Neuroimaging, FMRIB, Nuffield Department of Clinical Neurosciences, University of Oxford, Oxford, United Kingdom

^4^ McConnell Brain Imaging Centre, The Neuro (Montreal Neurological Institute-Hospital), McGill University, Montreal, QC, Canada

^5^ Department of Population and Public Health Sciences, Keck School of Medicine of USC, University of Southern California, Los Angeles, CA, United States

^6^ Department of Physics, Florida International University, Miami, FL, United States

Correspondence: Alejandro de la Vega (delavega@utexas.edu)
:::

> **BRIEF — title.** Current: *Selecting analyses, not papers: a large language model harness for
> automated neuroimaging meta-analysis.* The first clause carries the finding, the second is
> harness-first; "large language model" is spelled out rather than LLM for formality. Variants
> if you want the evaluation more explicit:
>
> - …harness for neuroimaging meta-analysis, **evaluated against nine expert syntheses** (longer,
>   but states the evidence in the title)
> - …harness for neuroimaging meta-analysis, **benchmarked against expert syntheses** (shorter
>   version of the same)
>
> Deliberately avoids leading with a gerund ("Evaluating a…"), which reads as a study *of* a
> method rather than a method — the skeleton's framing note is that the evaluation is the proof,
> not the subject (`NATURE_METHODS_SKELETON.md:33`). AutoNIMA is dropped from the title but kept
> as the system name in the abstract and introduction, as with Neuroscout and Neurosynth Compose.

> **BRIEF — author block needs confirming.** Affiliations are carried across from Kent et al.
> (2026, *Imaging Neuroscience*), the most recent paper this group shares, so they are current as
> of that publication but not independently verified here. **Author order is a placeholder**: de
> la Vega is first because the skeleton had it that way, with co-authors in the order you listed
> them. On the Compose paper de la Vega is senior/last author, so if that convention holds here
> the order needs inverting. Kendra Oudyk, Taylor Salo and Julio Peraza are on the Compose paper
> but not on your list — add them if they belong here.

<!--
  HOW TO USE THIS DOCUMENT

  Every section has a heading, a budget/figure line, and one indented BRIEF block. Write over
  the prose; delete the BRIEF block when you no longer need it. Nothing outside a BRIEF block is
  scaffolding, so when the last one is gone the document is the manuscript.

  BRIEF blocks carry three fields:
    CLAIM    the one thing the section must establish
    NUMBERS  the figures to quote, with the CSV they come from
    OPEN     unresolved decisions that affect what you can write here

  NUMBERS are re-derived from reports/, NOT copied from NATURE_METHODS_SKELETON.md or
  PAPER_OUTLINE.md, both of which have drifted. Regenerate or re-check them with:

      pixi run python scripts/manuscript_numbers.py

  Rebuild the Word version (this will OVERWRITE MANUSCRIPT.docx and any prose typed into it —
  write here, in the markdown, or copy the .docx aside first):

      pandoc MANUSCRIPT.md -o MANUSCRIPT.docx --reference-doc=.pandoc/manuscript-reference.docx

  The two planning documents keep their roles and are not superseded:
    NATURE_METHODS_SKELETON.md  section spine, figure numbering, word budget, decision records
    PAPER_OUTLINE.md            evidence map; §§1-9/S1/M1. Lines 1516-2693 are a lab notebook.
-->

# Abstract

_150 words. No figure._

Quantitative synthesis of the neuroimaging literature has been automated at the level of the
publication, but the unit that determines a meta-analytic result is the individual analysis: a
single paper typically reports many statistical contrasts, and only some of them bear on any
given question. Here, we present AutoNIMA, a harness that executes the full
systematic review workflow — search, abstract and full-text screening, coordinate extraction, and
analysis-level annotation — and that selects at both the paper and the analysis level. We
evaluated it against nine published, expert-conducted meta-analyses spanning 32 target contrasts.
Maps produced by the pipeline recovered the published result more closely than the strongest
search-based synthesis available for each contrast (mean *R*² 0.606 vs 0.491; ahead in 26 of 28,
*p* = 3.0 × 10⁻⁶), while pooling fewer analyses. Decomposing this advantage, selecting papers
contributed little (mean Δ*R*² +0.005) and selecting analyses contributed 96% of it (+0.109).

> **CLAIM** — Automated evidence synthesis selects papers, but the analysis is the unit that
> determines the result. We present a pipeline that selects at the analysis level and show it
> recovers published meta-analytic maps better than any search-only synthesis available.
>
> **NUMBERS** — lead with the headline and the unanimity, not the mechanism:
> 9 projects and 32 benchmark contrasts; the map-level comparison uses **28 over 8** (dementia
> excluded). Pipeline r² 0.606 vs 0.491 for the strongest per-column baseline, mean Δ +0.114,
> ahead in 26/28, sign-test p = 3.0 × 10⁻⁶.
> The decomposition is the finding: choosing papers is worth +0.005 (mean, 16/28); choosing
> analyses is worth +0.109 (25/28) — 96% of the total.
>
> **OPEN** — none. §9, the forward-looking case, was **dropped 2026-09-10**, so the abstract no
> longer waits on it and the closing sentence can land on the decomposition.

# Introduction

_~350 words. Cites the neurometabench companion paper._

The functional neuroimaging literature now comprises well over 60,000 published studies, and
synthesizing knowledge across them remains a major bottleneck. Quantitative meta-analysis is the
established remedy, but the traditional workflow — searching databases, screening thousands of
abstracts and full texts against inclusion criteria, and manually extracting coordinates from each
included study — is extraordinarily time consuming, and a single review can demand hundreds of
researcher hours. As a result, published syntheses are expensive to produce and rapidly go stale.

Automated approaches addressed the cost directly. Neurosynth demonstrated that coordinates could
be extracted at scale and meta-analysed automatically, and the resulting maps recover broad
cognitive domains with remarkable similarity to manual efforts. That scale came at a price in
precision, however: because coordinates could not be reliably attributed to the specific contrast
that produced them, the framework aggregated every coordinate reported in a paper into a single
set per study. This is sufficient for mapping broad domains and insufficient for the targeted
questions that motivate most meta-analyses, which pool results only from comparable experimental
conditions or participant groups.

We argue that this is not an incidental limitation of one system but a consequence of operating at
the wrong unit. The unit that determines a meta-analytic result is not the paper but the
*analysis*: a paper contributes k statistical contrasts, and a synthesis of "reappraisal versus
passive viewing" is defined by which of those contrasts enter it. Selecting papers well and then
pooling all of their coordinates answers a different question from the one the reviewer asked.
Coordinate-based meta-analysis is an unusually favourable place to measure the cost of this error,
because the published product is a spatial map that can be compared numerically against a
reproduction.

Here, we present AutoNIMA, a harness that selects at both levels, using large language models
for screening and for analysis-level annotation, and we evaluate it against nine expert-conducted meta-analyses
assembled into a benchmark (companion paper). In the following, we quantify what each stage of the
pipeline costs and buys, compare the end-to-end result against the strongest search-based
synthesis available for each target, and decompose the resulting advantage into the contribution
of selecting papers and the contribution of selecting analyses.

> **CLAIM** — Three moves, in order (`NATURE_METHODS_SKELETON.md:75`):
>
> 1. Evidence synthesis selects *papers*, but the inferential result is the *analysis*. A paper
>    contributes k analyses and typically only some of them address the question.
> 2. Coordinate-based neuroimaging meta-analysis is the one place this error is directly
>    measurable, because the published product is a spatial map that can be compared numerically.
> 3. We built a pipeline that selects at both levels and evaluated it against nine expert
>    meta-analyses.
>
> Frame as a *method* whose evaluation is its proof, not as an evaluation study — Neurosynth
> (2011) is the precedent in this journal (`NATURE_METHODS_SKELETON.md:33`).
>
> The thesis sentence, already drafted (`NATURE_METHODS_SKELETON.md:48`): *Automated evidence
> synthesis has been operating at the wrong unit. Selecting papers is not enough — the analysis
> determines the result, and selecting at that level is both necessary and now feasible.*
>
> **NUMBERS** — none. Resist quoting results here.
>
> **OPEN** — benchmark construction is one sentence plus a citation to the neurometabench
> companion paper, which **needs a Zenodo DOI before submission**.

# Results

## 1. The pipeline, the benchmark, and the unit that matters

_~300 words. **Figure 1** (schematic: a pipeline, b the unit distinction)._

> **CLAIM** — Establish what is being measured before any number appears. Panel b — one paper,
> k analyses, one target contrast — is the argument of the paper in a picture and should be
> drawn first (`NATURE_METHODS_SKELETON.md:96`).
>
> **NUMBERS** (`cross_project_best_baseline.csv`, `analysis_counts.csv`)
>
> - 9 projects, **32** benchmark columns scored
> - pipeline maps pool a median of 227 analyses (range 8–1,699) and 1,870 foci
>
> **FIGURE** — `figures/figure1_pipeline_schematic.svg`, hand-authored pure SVG (added
> 2026-09-10, reproducing the poster schematic). Not generated by
> `make_nature_methods_figures.py`, which is why it lives outside `reports/`. Export with
> `inkscape figures/figure1_pipeline_schematic.svg --export-type=pdf`.
>
> **OPEN** — the schematic covers panel a (the pipeline). Panel b, the unit distinction — one
> paper, k analyses, one target contrast — is **not drawn yet**, and it is the one the skeleton
> calls the argument of the paper in a picture.
> Benchmark construction belongs in the companion paper, which frees this panel.

## 2. Screening is nearly free; annotation is where recall is spent

_~300 words. **Figure 2**; supporting: Supplementary S2, and the raw-retention figure._

Recall reported against the full list of expert-included studies conflates screening judgement
with data availability: a study our query never returned, one whose full text we could not obtain,
and one that yielded no parseable coordinates are all charged to the screener. We therefore report
recall at each stage against an *attainable* denominator, which drops one availability failure at
the stage where it occurs and never drops a study rejected on judgement (Methods).

On this denominator, screening is close to free (Fig. 2). Abstract and full-text screening
together raise precision against the expert inclusion list from 0.103 to 0.323 for 0.074 of
attainable recall, and precision rises at every stage in 9 of 9 projects. Annotation behaves
differently. Requiring that a paper also yield at least one analysis assigned to the target
contrast raises precision by a further 0.147, again in 9 of 9, but costs 0.108 of attainable
recall, also in 9 of 9. Recall falls further than precision rises, and the stage cannot be
described as free.

Whether that trade is worth taking depends on the denominator, which is the clearest argument for
adopting the attainable one. Scored against every expert-included study, annotation appears to lose
0.248 of recall and the gain exceeds the loss in only 1 of 9 projects; scored against the studies
that actually yielded data, the same runs and the same decisions lose 0.108 and the gain exceeds
the loss in 6 of 9. Charging annotation for papers that had nothing to annotate inverts the verdict
on the stage.

The residual precision is a lower bound. Because we do not know which candidate pool the original
authors screened, a false positive may be a study they never considered rather than one they
rejected. Holding the pool fixed raises full-text precision by 0.142 while recall moves far less
(Supplementary S2), so a substantial share of the apparent screening failure is a corpus difference
rather than a screening error.

> **CLAIM** — Screening buys precision at almost no cost to recall. Annotation is where recall is
> actually spent — and the *denominator* decides whether that trade is worth taking, which is the
> argument for reporting recall against what each stage could attainably have kept.
>
> **NUMBERS** (`attainable_recall_by_stage.csv`, `stage_precision_recall.csv`; n = 9 projects)
>
> | stage | precision | attainable recall | raw recall |
> |---|---|---|---|
> | search | 0.103 | 1.000 | 0.858 |
> | abstract screening | 0.196 | 0.970 | 0.833 |
> | full-text screening | 0.323 | 0.926 | 0.728 |
> | annotation | 0.470 | 0.818 | 0.479 |
>
> - screening (abstract + full-text): precision 0.103 → 0.323 for **−0.074** attainable recall
> - annotation: precision **+0.147** (improves 9/9) for **−0.108** attainable recall (falls 9/9)
> - the trade: precision gain exceeds recall loss in **6/9** projects on the attainable
>   denominator and **1/9** on the raw one. Same runs, same decisions — charging annotation for
>   papers that had nothing to annotate inverts the verdict on the stage
> - search is 1.000 by construction (a pure supply stage); the corpus ceiling it carries is a
>   mean of **85.8%** of gold returned, range 60–96%
> - lowest at full text: `executive_function` 0.807, because the 15 gold studies its abstract
>   stage rejects on judgement stay charged against a shrinking denominator
>
> The two sentences that must appear in the body (`NATURE_METHODS_SKELETON.md:212`) point at S2:
> swapping only the study pool raises full-text precision +0.142, so a substantial share of the
> apparent screening failure is a corpus difference.
>
> **OPEN**
>
> - **This heading was changed 2026-09-10 and needs sign-off.** The previous one — "screening
>   raises precision at almost no cost to recall" — is true of screening but contradicted by the
>   annotation stage now on the same axis.
> - **"Adjusted" is used in two contradictory senses** and one must be dropped before submission:
>   here it *narrows* the denominator to attainable studies; in `projects/dementia/REPORT.md:95`
>   "adjusted gold" *widens* it (74 → 162) as a precision correction.
> - S2's recall control reads **−0.044** at full text, not the +0.007 quoted in the skeleton. The
>   control still holds — recall moves far less than precision — but the number needs restating.

## 3. Recovering the analyses, then selecting among them

_~350 words. **Figure 3** (a parsing, b annotation operating points)._

> **CLAIM** — Selection is meaningless until the analyses exist, so this section is in pipeline
> order: recover them, then select among them. Both steps are large effects, and the second only
> looks impressive once you compare it against the right null.
>
> **NUMBERS**
>
> *3a, parsing* (`cross_project_analysis/parsing_metrics_by_project.csv`; n = 9)
>
> - LLM parsing recovers **91.4%** of expert analyses (median 94.7%)
> - tables-only baseline recovers **33.3%** (median 29.9%)
> - margin **+58.1 points**, ahead in **9/9** projects — the strongest single margin in the paper
>
> *3b, annotation in ROC space* (`cross_project_analysis/annotation_aggregates.csv`; **n = 9**,
> all projects)
>
> - **mean TPR − FPR 0.68** against 0 for chance — what the panel reports. The size-matched null
>   is closed-form: a random selector of the same size lands at exactly (k/N, k/N), so one
>   diagonal replaces nine per-project no-skill levels
> - pooled **precision 0.539, recall 0.810**, F1 0.647 over 11,126 candidate analyses and
>   **32** target contrasts
> - lift over prevalence mean **3.1×**, range **1.8–5.6** (`social` to `vbm_of_substance_use`)
> - **dementia is included** (promoted 2026-09-16). It contributes 29 expert assignments against
>   `social`'s 1,159 and sits at a lift of 2.1×, inside the range rather than outside it
>
> **NOTE ON UNITS** — panel a's denominator is *analyses*; panel b's positives are
> *analysis-to-contrast assignments*. `vbm_of_ptsd` has 10 analyses and 1 contrast, so 10
> positives; `social` has 529 analyses across 5 contrasts and 1,159. Say "assignment recall"
> rather than "recall" for panel b.
>
> **OPEN** — the precision-recall rendering is now **unnumbered** (`--only annotationpr`). It is the
> only panel showing the precision-vs-lift reordering ROC space cannot: `social` has the
> third-highest precision (0.618) but the **lowest** lift (1.8×) because its prevalence is
> highest (0.345), while `vbm_of_substance_use` turns a similar 0.584 into **5.6×** off 0.104.
> It still excludes dementia, so its numbers are n = 8 and will not match panel b's.

## 4. The whole pipeline beats a search-only synthesis

_~350 words. **Figure 4**; supporting: Supplementary S5, and the text-to-map figure._

We next tested whether AutoNIMA's end-to-end workflow translated improved evidence selection into
more faithful recovery of published neuroimaging meta-analytic maps. As a baseline comparison akin
to term-based large-scale meta-analysis (e.g. Neurosynth), we generated search-only meta-analyses
that pooled every analysis extracted from articles retrieved through PubMed, without subsequent
article screening or analysis-level selection.

For each target contrast we used the strongest baseline a practitioner could actually have built
in advance: a contrast-targeted search wherever the contrast is a separable search topic (27 of 28
contrasts), and the project's broad search otherwise. The single exception is instructive.
Emotion regulation's *maintain* contrast is a direction within one paradigm rather than a distinct
topic, and no query distinguishes maintaining from up- or down-regulating an emotional response,
so the broad search is not a fallback but genuinely the best baseline anyone could construct.
Because this rule fixes each baseline before seeing the result, we also report a robustness check
in which each contrast takes whichever arm scored higher; narrowing the search in fact made the
baseline *worse* in 8 of 28 contrasts, and the two definitions differ by 0.007 in mean baseline
*R*². All arms shared a retrieval vintage and were meta-analysed identically — MKDA with a 10 mm
spherical kernel in NiMARE, thresholded by Benjamini–Hochberg FDR — so the arms differ only in
which analyses enter them.

Correspondence with the reference maps was quantified as the squared voxelwise Pearson correlation
between unthresholded *z* maps, computed over voxels with finite values in both (*R*²; Methods).
All correlations were positive, so squaring discards no directional information. Dementia is
excluded from this comparison: its reference meta-analysis pools several studies into each gold
analysis, so the number of analyses entering its maps is not comparable with the other projects
(Methods).

> **CLAIM** — Run end to end from a plain PubMed search, the pipeline beats what you would get by
> searching, extracting every coordinate and meta-analysing the lot — the Neurosynth-style
> approach — and it does so while pooling *fewer* analyses. Selection beats volume.
>
> **NUMBERS** (`cross_project_best_baseline_stats.csv`; **28 columns, 8 projects** — dementia
> excluded, see OPEN — metric r²)
>
> | vs | pipeline | baseline | mean Δ | median Δ | 95% CI | ahead |
> |---|---|---|---|---|---|---|
> | best available | 0.606 | 0.491 | +0.114 | **+0.070** | [+0.052, +0.201] | **26/28** |
> | strongest | 0.606 | 0.498 | +0.107 | +0.061 | [+0.044, +0.196] | 25/28 |
>
> - sign test p = **3.0 × 10⁻⁶** (best available), 2.7 × 10⁻⁵ (strongest)
> - cluster bootstrap **over projects, not columns**; 20,000 resamples, seed 0
> - baseline arms: targeted where a contrast is a separable search topic (**27/28**), broad
>   otherwise (only `emotion_regulation_2022/maintain`). Narrowing the search made the baseline
>   *worse* in **8/28** columns, which is why `strongest` is a robustness check and not the
>   primary: it lets the baseline switch arms with hindsight
> - selection beats volume: the pipeline pools fewer analyses than its baseline in **20/28**
>   columns and wins in **19** of those; against the screening-only arm it pools fewer in
>   **28/28** and wins in 26 (`analysis_counts.csv`)
>
> **OPEN**
>
> - The skeleton quotes a second, **stale** headline at L518 (Δ +0.110, p = 3.5 × 10⁻⁶, 31/35)
>   from the 35-column era. Use the table above; the denominator is **32**.
> - The skeleton also wants an **axial-slice panel c** from `make_brain_map_figure.py`. Two
>   generated brain figures are currently unassigned to any slot:
>   `figure_brain_maps_contrast` and `figureS4_brain_maps_all`.
> - `decrease` **cannot be used as the exemplar** — it duplicates `reappraisal` (r = 0.972).
>   Use `reappraisal` instead (`NATURE_METHODS_SKELETON.md:414`).

## 5. The gain comes from analysis selection, not paper selection

_~350 words. **Figure 5**; supporting: Supplementary S3._

> **CLAIM** — Decompose the Result 4 margin into its two selection steps. Choosing the right
> papers is worth almost nothing; choosing the right analyses within them is worth nearly all of
> it. This is the paper's finding, not a supporting detail.
>
> **NUMBERS** (`selection_decomposition.csv`; **28 columns, 8 projects** — same set as Figure 4,
> metric r²)
>
> | step | median | mean | positive in |
> |---|---|---|---|
> | paper selection (screening only) | +0.015 | **+0.005** | 16/28 |
> | analysis selection (annotation) | +0.056 | **+0.109** | **25/28** |
> | total | +0.070 | +0.114 | 26/28 |
>
> - three arms: baseline 0.491 → screening-only 0.496 → pipeline 0.605. The middle arm barely
>   moves, which is the whole point: **96%** of the total is analysis selection
> - the means sum exactly to Figure 4's margin (+0.005 + 0.109 = +0.114), which is the check
>   that the two figures describe the same corpus. Quote means, not medians, when the
>   decomposition needs to add up
> - the sharpest mechanistic sentence in the paper (`PAPER_OUTLINE.md:809`):
>   **it is not a search problem; it is a screening problem** — and more precisely, an
>   analysis-selection problem
>
> *Supplementary S3, the size-matched null* (`annotation_bootstrap_null.csv`)
>
> - median Δr² **+0.179** against each column's own size-matched null; **25/28** clear p < 0.05
> - state carefully: the naive reading overclaims, because 31/32 baselines are targeted searches
>   (`NATURE_METHODS_SKELETON.md:804`)
>
> **OPEN** — the two medians do not sum to the total (+0.015 + 0.056 = +0.071 vs +0.070) because
> medians are not additive; the *means* do exactly (+0.005 + 0.109 = +0.114). Quote the means
> when you need the decomposition to add up, and say which you are quoting.

## 6. Cost and scale

_~200 words (budget table says ~300 — settle this). **Supplementary S5**._

> **CLAIM** — Fully measured, not estimated: a project runs for tens of dollars and hours against
> months of person time. For a methods journal this is not an aside — it is why the method
> matters (`NATURE_METHODS_SKELETON.md:703`).
>
> **NUMBERS** — ⚠ **the only block in this document not derived from a CSV.** Hardcoded as
> `COST_PER_STAGE` in `make_nature_methods_figures.py`, transcribed from `PAPER_OUTLINE.md` S1.
> The module docstring says these should move into a generated CSV before submission; verify
> against `usage_total` in `execution_progress.json` before quoting.
>
> - per call: abstract screening $0.0023 · full-text screening $0.0138 · coordinate parsing
>   $0.0059 · annotation $0.0211
> - **mean $21.55 per project**; all nine from scratch **$194**
> - quote **cost per study that reaches the map** ($0.085 median) rather than cost per hit — it is
>   2.9× tighter and it is the unit a reader can act on
>
> **OPEN** — heading budget says ~200 w, the budget table says ~300.

# Discussion

_~600 words._

> **CLAIM** — Four beats, in order (`NATURE_METHODS_SKELETON.md:706`):
>
> 1. **The benchmark is a reference, not truth.** `vbm_of_substance_use` is the sharpest case:
>    three of its columns have no published result at all, so they are excluded and the
>    denominator is 32 rather than 35. That is a better sentence than "three columns were not
>    significant" (`NATURE_METHODS_SKELETON.md:641`).
> 2. **The gain appears for specific targets and not diffuse ones**, and two accounts remain
>    unresolved — a squishy target versus a model limitation. Say so rather than choosing
>    (`NATURE_METHODS_SKELETON.md:752`). A third account is live: our own specification may have
>    been wrong. A fourth is a confound — `pubmed.py:370` truncates structured abstracts, so
>    26.6–30.6% of stored abstracts are under 450 characters and the model may never have seen the
>    deciding text (`PAPER_OUTLINE.md:1106`).
> 3. **The analysis-unit argument generalises; its measurability does not.** Coordinate-based
>    meta-analysis is unusual in producing a numerically comparable published product.
> 4. **Close on what the method is for.** §9, a forward-looking demonstration on a question no
>    manual synthesis had attempted, was **dropped 2026-09-10** as unnecessary for this paper.
>    Close instead on the practical consequence: the ceiling on this whole evaluation is the
>    manual meta-analyses themselves, so a method that reaches them at this cost changes what is
>    worth attempting rather than only what is worth automating.
>
> Two out-of-scope items to note here rather than answer (`PAPER_OUTLINE.md:1230`, `:1243`):
> MKDA is used throughout even where the source paper used ALE; and there is no human-agreement
> ceiling for screening.
>
> Also worth a clause: the dominant constraint is coordinate extraction, not screening —
> `dementia` yields coordinates for 49% of included studies, "larger than every schema effect in
> the paper combined" (`PAPER_OUTLINE.md:1118`).
>
> **NUMBERS** (Supplementary S1, `tier_progression.csv`) — the mis-specification point, which belongs
> here as a caution about LLM screening generally:
>
> - verbatim → manual criteria: **+0.221** (n = 2 projects)
> - manual → best criteria: **+0.031** (n = 2 projects)
> - roughly **7×**. The drafted line (`NATURE_METHODS_SKELETON.md:1056`): *the live danger in
>   LLM-assisted screening is mis-prompting, and it is roughly seven times larger than the danger
>   of tuning.*
> - **Do not report either without the n.** Both arms are two projects; an indication, not an
>   estimate.
>
> **OPEN** — none. The §9 forward-looking case is dropped; `PAPER_OUTLINE.md:1434` and its
> `[idea]`/`[need]` markers are now stale on that point.

# Methods

_No word limit — Methods sits after references and costs nothing against the 3,000._

> **NOTE** — the five sub-areas below were written against *Imaging Neuroscience's* AI-methods
> guidelines (`PAPER_OUTLINE.md:1313`). Check them against Nature Methods' own requirements
> before submission.

## Pipeline and configuration

> **BRIEF** — stages in order: PubMed search → abstract screening → full-text retrieval →
> coordinate parsing → analysis-level annotation → NiMADS output → meta-analysis. Model and
> version per stage. Retrieval is via Elsevier and pubget, which do their own coordinate
> ingestion; **not ACE**.

## Benchmark and denominator

> **BRIEF** — nine expert meta-analyses, 35 annotation columns, of which **32 are scored**.
> `scripts/benchmark_exclusions.py` is the single source of truth: three `vbm_of_substance_use`
> columns (cannabis, opioids, stimulants) are excluded because the source paper (Hill-Bowen et al.
> 2022, *Drug Alcohol Depend* 240:109625, PMID 36115222) reports no significant result for them
> and the expert map is empty. `nicotine` is deliberately **kept** — that is a reproduction
> failure with identical inputs, not an absent reference.

## The metric/map rule

> **BRIEF** — state once, here; it removes a class of reviewer objection
> (`NATURE_METHODS_SKELETON.md:659`):
> **r² compares unthresholded maps; dice compares FDR-corrected thresholded maps** at z > 1.96.
> `z_corr > 1.96` selects exactly the voxels with FDR-corrected p ≤ 0.05. It uses the two-tailed
> convention; MKDA is one-tailed, so NiMARE's own positive-tail label (≈1.645) flags more voxels
> and 1.96 is the conservative choice.
> r² is the reported metric throughout: dice is degenerate on this corpus (four substance-use
> columns sit at 0.000 for every arm) and reverses sign on `vbm_of_ptsd`.

## Recall against an attainable denominator

> **BRIEF** — `scripts/compute_attainable_recall.py`. Each stage divides by the gold studies it
> could have kept; one availability failure leaves the denominator at the stage where it happens,
> and no judgement ever leaves it.
>
> | stage | denominator | drops |
> |---|---|---|
> | search | gold the search returned | gold the query never returned |
> | abstract | same as search | — nothing becomes unavailable in between |
> | full-text | … minus gold with no *usable* full text | retrieval failures **and `fulltext_incomplete`** |
> | annotation | … minus gold that parsed to zero analyses | nothing to annotate |
>
> Two points a reviewer will probe:
>
> - the annotation denominator is **not** "gold with ≥ 1 parsed analysis" — that set excludes
>   everything full-text screening discarded and would forgive every full-text rejection
> - the retrieval stage counts **usable** text, not the `fulltext_available` flag. The flag is
>   true for 43 gold studies the screener then received as title and abstract only; those are
>   retrieval failures surfacing one stage late, and 43 of the 68 gold studies lost at full-text
>   screening are of that kind against 25 genuine exclusions

## Null models

> **BRIEF** — two spaces, same idea. In map space, 500 size-matched MKDA re-fits per column
> (`scripts/bootstrap_annotation_null.py`). In ROC space it is closed-form: a size-matched random
> selector lands at exactly (k/N, k/N) with hypergeometric spread, so no Monte Carlo is needed for
> the location. Median band width 0.020 in FPR; the two small-N exceptions are `dementia` (0.078)
> and `vbm_of_ptsd` (0.133 off N = 40).

## M1.1 Code and data availability

> **BRIEF** — three repositories: `neurostuff/autonima` (the pipeline, 161 tests), this results
> repository, and neurometabench (the benchmark).
> **OPEN — blocked on Zenodo DOIs.** neurometabench must be at least preprinted with a DOI before
> or alongside this paper; reviewers will ask where the benchmark is.

## M1.2 Generative AI in development

We distinguish two uses of generative AI in this work. The first is the object of study: large
language models perform abstract and full-text screening, coordinate parsing, and analysis-level
annotation within the pipeline, and their behaviour at each of those stages is what the paper
measures. Models, versions, and prompts for every stage are specified above and released with the
code.

The second is the use of generative AI as a development tool, which we declare separately here.
During the preparation of this work, the authors used [MODEL(S)] to assist with writing and
revising code in the analysis repository, and with editing and revising the text of this
manuscript for clarity, readability, and flow. All AI-generated code was reviewed and tested by
the authors, and all AI-assisted text was critically reviewed and revised. No analysis result,
figure, or numerical claim in this paper was produced by an unreviewed AI process; every number
reported here is regenerated from the released artifacts by a script in the analysis repository.
The authors take full responsibility for all content presented in this manuscript, including any
portions assisted by AI.

> **BRIEF** — ⚠ **`[need]` — nothing written yet**, and one of only two outstanding required items
> in the whole plan (`PAPER_OUTLINE.md:1336`). The organising distinction is to keep AI-as-object-
> of-study separate from AI-as-development-tool, and to declare the latter explicitly.

## M1.3 Data leakage mitigation

> **BRIEF** — the four-tier run registry in `run_categories.yaml`: `verbatim` (criteria
> transcribed from the published methods, held out by construction), `manual` (author-revised
> after seeing reports), `best`, `latest`. Report what each tier saw. The `verbatim` tier is a
> held-out schema run nine times, which is the overfitting rebuttal.

## M1.4 Hyperparameter tuning protocol

> **BRIEF** — `scripts/run_tiers.py` and the recorded `best-reason` for each promotion. The
> honest framing is that `verbatim → manual` uncovered major oversights in the criteria rather
> than tuning to the benchmark, which is why it is reported as mis-specification (Discussion)
> rather than as overfitting.

## M1.5 Statistical reporting

> **BRIEF** — `[have]`. Cluster bootstrap **over projects, not columns**, because columns within a
> project are not independent; percentile CIs; sign test over columns. 20,000 resamples, seed 0.
> All arms share a retrieval vintage — a methodological requirement learned the hard way
> (`PAPER_OUTLINE.md:956`), and it needs stating here.

# References

_~50 references. Numbered, Nature style._

1\. [neurometabench companion paper — **needs Zenodo DOI**]

2\. Yarkoni, T. et al. Large-scale automated synthesis of human functional neuroimaging data. *Nat. Methods* **8**, 665–670 (2011).

3\. Hill-Bowen, L. D. et al. *Drug Alcohol Depend.* **240**, 109625 (2022). PMID 36115222.

4\. [NiMARE]

5\. [pubget]

6\. [NeuroQuery]

7\. [NeuroVLM]

> **OPEN** — no bibliography exists. The 603 `.bib` files in the repo are per-run study lists
> (`projects/**/outputs/meta_analysis_results/*/references.bib`), not paper citations.

# Figure legends

## Figure 1

> **BRIEF** — a, pipeline schematic: expert configuration → search → screening (abstract,
> retrieval, full-text) → coordinate parsing → analysis annotation → meta-analysis, with the LLM
> decision step drawn out at each stage that has one. Drawn:
> `figures/figure1_pipeline_schematic.svg`. b, the unit distinction: one paper, k analyses, one
> target contrast — **still to draw**.

## Figure 2

> **BRIEF** — must state: the denominator per stage; that search is 1.000 by construction; and
> that judgement losses stay charged while availability failures leave the denominator. n = 9.

## Figure 3

> **BRIEF** — panel a: dumbbell, tables-only against LLM parsing, all nine projects; must state
> that parsing is scored only on papers from which at least one analysis was extracted. Panel b:
> annotation in ROC space, all nine projects; must state that the diagonal is chance, that each
> open marker is that project's own same-size random selection, and that the unit is
> analysis-to-contrast assignments rather than analyses.

## Figure 4

> **BRIEF** — must say which baseline it is (best available per column, `targeted` for 31 of 32),
> that circle area encodes the number of analyses, and that the bottom rows are small wins rather
> than losses (`NATURE_METHODS_SKELETON.md:470`).

## Figure 5

> **BRIEF** — must state that the three arms are baseline / screening-only / full pipeline, all
> from the canonical run, and that the middle arm isolates paper selection.

## Figure 6

> **BRIEF** — `figure_er_surface_contrasts`: three emotion-regulation maps against a single
> baseline, on inflated surfaces. The honest framing is that this shows what a good case looks
> like while Figures 4 and 5 report the whole distribution including the losses.

# Supplementary Information

> **BRIEF** — renumbered 2026-09-17:
>
> - **S1** criteria tier progression — mis-specification vs overfitting (`--only S1`)
> - **S2** pool mismatch (`--only S2`)
> - **S3** size-matched null (`--only S3`)
> - **S4** brain maps, all columns (`scripts/make_brain_map_figure.py --mode all`)
> - **S5** measured cost per stage (`--only S5`)
>
> Three figures are still built and still correct but no longer cited, so they carry
> descriptive keys rather than S-numbers: `--only text2map` (text-to-map baselines),
> `--only retention` (raw-denominator gold retention) and `--only annotationpr`
> (annotation in precision-recall space). Extended Data: the §3 cautionary case,
> per-project tables behind Figures 2–5, PRISMA funnels.
> Extended Data: the §3 cautionary case, per-project tables behind Figures 2–5, PRISMA funnels.

## Supplementary S2 — how much of the low precision is a pool mismatch?

> **NUMBERS** (`stage_precision_recall.csv` vs `stage_precision_recall_allstudies.csv`; n = 3
> projects: dementia, emotion_regulation_2022, social)
>
> | stage | search pool | fixed pool | Δ precision | Δ recall |
> |---|---|---|---|---|
> | search | 0.106 | 0.276 | +0.169 | −0.057 |
> | abstract | 0.201 | 0.373 | +0.172 | −0.057 |
> | full-text | 0.349 | 0.491 | **+0.142** | −0.044 |
> | annotation | 0.497 | 0.657 | +0.160 | −0.027 |
>
> Recall is the control: it moves far less than precision, so the gain is a corpus difference
> rather than a screening effect. **n = 3 must appear in the caption.**
>
> *Corroboration, no figure of its own:* **59%** of full-text false positives (385 of 649) were
> never in the researchers' pool. **Do not say "most false positives are a pool artefact" without
> the per-project split** (`NATURE_METHODS_SKELETON.md:845`).

## Text-to-map baselines — a term is not an analysis (unnumbered)

> **NUMBERS** (`text_to_map_baselines.csv`; n = 32 columns)
>
> | arm | mean r² | median r² |
> |---|---|---|
> | NeuroQuery | 0.075 | 0.031 |
> | NeuroVLM | 0.238 | 0.231 |
> | best search baseline | 0.476 | 0.424 |
> | pipeline | 0.574 | 0.607 |
>
> Both text-to-map models fall well short of even the search baseline, and they fail on the same
> structure — contrasts and clinical comparisons — which is the analysis-unit argument arriving
> from an independent direction. Panel b reports r² **and** top-k dice; the caption must not quote
> the r² gap alone (`NATURE_METHODS_SKELETON.md:933`).

## Raw denominator, retrieval as its own stage (unnumbered)

> **NUMBERS** (`gold_survival_by_stage.csv`) — marginal gold loss by stage, percentage points:
>
> | stage | median | max | projects > 5pp |
> |---|---|---|---|
> | abstract screening | **1.4** | 8.8 | 2/9 |
> | full-text retrieval | 4.5 | 25.7 | 3/9 |
> | full-text screening | **2.5** | 6.4 | 1/9 |
>
> This is the only figure that still shows the retrieval stage and the raw denominator, so this
> table is not recoverable from Figure 2. Retrieval's median loss is nearly double full-text
> screening's, which is the argument for keeping them as separate stages.

---

# SCAFFOLDING — DELETE BEFORE SUBMISSION

## What actually gates submission

| item | state |
|---|---|
| **neurometabench Zenodo DOI** | blocking. Reviewers will ask where the benchmark is |
| **`projects/emotion_regulation_2022/nmb_mappings.json`** | still the unedited template (`MANUAL_NAME1` → `AUTOMATIC_NAME1`), which excludes ER from every cross-project analysis. Called "highest leverage per unit of work in the whole plan" (`PAPER_OUTLINE.md:1500`) |
| **M1.2 Generative AI in development** | nothing written |
| **"adjusted" used in two senses** | one must be dropped |
| **dementia excluded from Figures 4, 5, S3** | justified but it *helps* the headline (+0.099 → +0.114), so Methods must state the reason and the 32-column result belongs in the text as a robustness check |
| **Figure 1** | panel a drawn (`figures/figure1_pipeline_schematic.svg`); panel b, the unit distinction, still to draw |
| Result 2 heading | changed 2026-09-10, needs sign-off |
| Cost numbers | hardcoded, not CSV-derived |

## Numbers in the planning documents that are stale — do not re-quote

Found while building this skeleton, 2026-09-10. All corrected above.

| document says | actual | source |
|---|---|---|
| parsing 90% vs 31%, +60 pts | **91.4% vs 33.3%, +58.1 pts** | `parsing_metrics_by_project.csv` |
| Fig 4 Δ +0.110, p = 3.5e-06, 31/35 (skeleton L518) | **superseded**, 35-column era | — |
| §7 0.495 vs 0.394, 30/35 (outline L691) | **superseded**, 35-column era | — |
| — | **current: 0.606 vs 0.491, mean Δ +0.114, 26/28, p = 3.0e-06** — the map-level set is 28 columns over 8 projects since dementia was excluded 2026-09-16; the benchmark is still 32 over 9 | `cross_project_best_baseline_stats.csv` |
| annotation lift | **3.1×** for Figure 3b as it now stands (n = 9, dementia included). 3.3× is the n = 8 value and is correct for the unnumbered precision-recall figure only — the two panels have different project sets, which is why this number moved twice | `annotation_aggregates.csv` |
| S2 recall control +0.007 | **−0.044** at full text | `stage_precision_recall*.csv` |
| "35 columns" (skeleton L351/473/518/568/576) | **32** | `benchmark_exclusions.py` |

Regenerate all of the above with `pixi run python scripts/manuscript_numbers.py`.

## Budget arithmetic

Skeleton budget: 150 + 350 + 1,900 + 600 = 3,000. Per-heading Results totals are
300 + 300 + 350 + 350 + 350 + 200 = **1,850**, i.e. 50 short of the stated 1,900. Result 6's
heading (~200 w) also disagrees with the budget table (~300). Settle before writing to length.

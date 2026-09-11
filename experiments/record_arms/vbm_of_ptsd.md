# VBM of PTSD: extraction records as screening input

Benchmark: Pankey et al., *Extended functional connectivity of convergent structural
alterations among individuals with PTSD* (PMID 36100907), an ALE meta-analysis of VBM
studies.

Produced by `scripts/pondie_arm/run_ptsd_record_arms.sh`.

---

## Headline

**Replacing article full text with a machine-generated extraction record costs recall, but
only one of the four lost papers is information loss. The rest are the records applying the
stated criteria correctly to papers the benchmark included in violation of them — and on
one paper the supporting quotations actively cause the miss.**

| arm | precision | recall | F1 |
|---|---|---|---|
| `v1-A1-mini` (article text) | 0.593 | **0.941** | **0.727** |
| `v1-record-with-evidence` | 0.591 | 0.765 | 0.667 |
| `v1-record-no-evidence` | **0.609** | 0.824 | 0.700 |

**One of the sixteen disagreements in this project is an extraction failure.** On 22453299
the `demands` stage enumerated a single analysis — an fMRI localizer contrast — from a
render carrying zero tables, and the VBM between-group comparison the paper states in its
results prose never entered the record. Both record arms lose it (§3.3).

**Evidence quotations make the record behave more like the article, including where the
article is wrong.** The two record arms differ on exactly one paper, 21118656, and it is the
one paper the *text* arm also gets wrong. The with-evidence arm quotes the paper's own
sentence — "statistical VBM analyses were masked for the ROIs under investigation" — and
excludes, reproducing the text arm's only false negative. The no-evidence arm never sees
that sentence, includes, and matches gold. Agreement with the text arm is correspondingly
higher with quotes (κ **0.798** vs **0.757**). Quotes are not neutral: they supply the
disqualifying detail that a summary elides (§3.4).

**Precision is the more misleading number, because the benchmark's criteria are not
encoded.** All three arms are far too inclusive — 55%, 45% and 47% of the cohort against a
gold positive rate of 35%. Eight of eleven false positives map onto criteria Pankey states
plainly and the config never mentions, chiefly that the meta-analysis is **adults only**.
Adding that one exclusion lifts precision to **0.727** for text and **0.722 / 0.737** for
the record arms at no recall cost, because 0 of the 17 gold studies in the cohort have a
paediatric sample and 5 of the 11 false positives do (§4).

Read this as: on this benchmark the record arms' recall deficit is mostly an artifact of a
gold set looser than its own stated criteria, and the one real defect is upstream of the
extraction — a table-less render (§5).

---

## 1. What was compared

Three arms over the same query, criteria and model (`gpt-5-mini-2025-08-07`), differing only
in what the full-text stage reads:

| arm | full-text input |
|---|---|
| `v1-A1-mini` | the article, as retrieved (pubget / Elsevier / ACE HTML) |
| `v1-record-with-evidence` | a rendered pondie extraction record, **with** a supporting quotation under each extracted value |
| `v1-record-no-evidence` | the same records, quotations stripped |

All three configs come from `gen_arm_config.py` against the same baseline
(`projects/vbm_of_ptsd/v1.yaml`), so retrieval sources, model qualification and the
record-format note are generated identically. The two record configs differ from each other
in **three lines**: the mirror-set name in each of the three record sources. The rendered
records behind them differ in one flag, `render_record.py --evidence full|none`.

`build_record_corpus.py` lays the records out as three mirrors that reuse each baseline
source's own `processed_data_path` and `coordinates_path_templates`, so activation tables
and coordinates reach the arms by exactly the route they reached the baseline and the only
thing that changed is the string in the `Full Text Content:` slot. `assert_record_arm.py`
confirms this held: `local_found` is 71 for both arms and 71 for the baseline, and 49 of 50
screened studies read a record.

The record-format note is left in place for the no-evidence arm even though it mentions
quotations. Removing it would change two things at once; `gen_arm_config.py
--no-record-note` is the control for the note itself and is a separate experiment.

Records were extracted by pondie at commit `31e6cea` over the 50 papers in the full-text
screening cohort: nine stages per paper, 8 workers, `service_tier: flex` on `gpt-5.6-luna`.
**50 papers, 0 failures, 8,903,750 input / 1,070,522 output tokens across 409 calls**, ≈$0.03
per paper at flex rates. Each variant was rendered twice and the two passes diffed, because
`study_full_text_content_hash` hashes file content and a non-deterministic renderer would
silently re-screen every paper on resume and read as drift; both were byte-identical over 50
records.

Abstract screening was copied from A1's cache (`--copy-valid-cache-from`), so the arms differ
only downstream of retrieval.

---

## 2. Screening performance

### 2.1 Against gold

50 papers were screened by all three arms; the paired cohort is 49 after dropping 22720021,
which no record arm read a record for — abstract screening is not deterministic, so an arm
can admit a paper the record set never anticipated, retrieval then falls through to real
text, and pairing on it would compare text against text. 17 of the benchmark's 22 included
studies fall inside the cohort.

| arm | included | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| `v1-A1-mini` | 27 | 16 | 11 | 1 | 0.593 | **0.941** | **0.727** |
| `v1-record-with-evidence` | 22 | 13 | 9 | 4 | 0.591 | 0.765 | 0.667 |
| `v1-record-no-evidence` | 23 | 14 | 9 | 3 | **0.609** | 0.824 | 0.700 |

`tables/ptsd_screening.csv`; per-stage metrics from
`compare_screening_to_benchmark.py` in `tables/ptsd_metrics_*.csv`.

Both record arms carry the **same nine** false positives, a strict subset of the text arm's
eleven. The text arm's two extra are papers the records correctly typed out: 19996042
(FreeSurfer parcellation, not VBM) and 28888350 (ROI-only).

The inclusion counts matter as much as the rates. Gold positives are 17 of 49 (35%); the
arms include 55%, 45% and 47%. **All three over-include, and the text arm over-includes
most.** Against a benchmark with this base rate, over-inclusion buys recall cheaply and
costs precision slowly, which is most of why A1's F1 leads.

### 2.2 Arm versus arm

| pair | both include | neither | A1 only | record only | agreement | κ |
|---|---|---|---|---|---|---|
| A1 vs with-evidence | 22 | 22 | 5 | 0 | 0.898 | **0.798** |
| A1 vs no-evidence | 22 | 21 | 5 | 1 | 0.878 | **0.757** |

Exact McNemar over gold positives: p=0.250 (with-evidence), p=0.625 (no-evidence). Over gold
negatives: p=0.500 for both. **Nothing here is significant, and with 5–6 discordant pairs
nothing could be** — see §6(a).

---

## 3. Where the arms disagree, and why

Four papers account for every gold-positive disagreement among the three arms, and they
fail for four different reasons. Per-paper table with attributed causes: `tables/ptsd_errors.csv`.

### 3.1 A lexical collision the schema resolves and prose does not — 19794316

The text arm called this "whole-brain, voxel-wise between-group contrasts" and included it.
The paper reports no whole-brain analysis anywhere. "Whole-brain" occurs four times in the
retrieved HTML and every occurrence is the phrase *whole-brain volume* used as a nuisance
covariate — "After controlling for age, depression and whole-brain volume…", "voxel-based
morphometry analyses with whole-brain volume correction". All results are hippocampus and
rostral ACC inside AAL masks; the paper itself refers to "other regions of interest".

The record cannot make that mistake, because extraction had already filed the same phrase in
its own slot: `source definition: Global grey matter volume` as a model term, and
`spatial scope: roi` for both analyses. The record is right about the paper. It scored wrong
because gold includes the study, which Pankey's own ROI exclusion forbids (§4).

### 3.2 A methods declaration versus a reported result — 21418787

The paper's methods say "We followed up a significant main effect of group with comparisons
between individual groups, inclusively masked with the results of the original group
effect." The text arm credited that sentence and inferred a PTSD-vs-control contrast. But the
results report only the combined clinical-vs-control contrast (Table 1: "the clinical groups
show clusters of decreased brain volume compared with controls") plus four BAI/BDI
regressions in Tables 2–5, and the record enumerated exactly those five analyses.

The record arms had the tables here — this is not a coverage gap. It is that a record states
what a paper reported and prose lets a reader credit what it promised.

### 3.3 Real information loss — 22453299

The only genuine extraction failure in the project. The render carried 0 tables, `demands`
enumerated one analysis — the paper's fMRI localizer ("objects/artifacts than scrambled
images") — and the VBM group comparison stated in the results prose ("at an uncorrected
threshold of P<0.001, control participants had more gray matter volume in the left middle
occipital cortex, as well as the left precentral and the right angular gyri") never entered
the record.

The two arms failed differently, and the difference is instructive. The with-evidence arm
**excluded** at confidence 0.7, treating the absence as a fact about the paper. The
no-evidence arm returned **`fulltext_incomplete`** at confidence 0.35 — which is the better
behaviour, since it flags the record rather than the study, though it still loses the paper.

Note the contradiction that lets a thin parse become a negative decision at all. The render's
own header says: "A field this record does not carry may be one the paper did not report or
one the extraction missed… Either way it is not evidence against the study." The
record-format note that `gen_arm_config.py` injects says: "Otherwise a field that is absent
means the paper did not report it." The note wins.

15 of the 50 records carry exactly one analysis, so the thin-parse state is common; this is
the only case where it flipped a gold positive.

### 3.4 The evidence effect, isolated — 21118656

The one paper on which the two record arms disagree, and the cleanest result in the report
because everything else about the two arms is generated identically.

The paper's methods state "Subsequent statistical VBM analyses were masked for the ROIs under
investigation". The **with-evidence** arm quotes that sentence in its reasoning and excludes
at confidence 0.89 — the correct call under the stated criteria, and the same call the *text*
arm makes; this is the text arm's only false negative. The **no-evidence** arm sees the
record's fields without the underlying sentence, reads a whole-brain VBM with group
contrasts and significant clusters, and includes at 0.88. Gold includes it.

So the quotations did their job — they surfaced the disqualifying detail — and the arm was
penalised for it, because the benchmark includes ROI-masked studies against its own criteria.
This is the whole report in one paper: evidence makes the record behave like the article, the
article-like behaviour is more faithful to the criteria, and the criteria are not what the
gold set actually applied.

---

## 4. The false positives are a criteria gap, not a pipeline gap

Pankey's methods state the inclusion criteria plainly: "peer-reviewed MRI studies, reporting
results among **adult humans**, written in the English language, focused on gray matter
structural differences, and included original data". Exclusions: "trauma or stressful life
event studies not measuring PTSD, other non-voxel-based morphometry methods, **treatment and
longitudinal effects**, papers reporting **regions of interest (ROIs)**, within-group
effects, null effects, **overlapping samples to previous studies**, and studies that did not
report coordinate-based results." Records screened out at title/abstract included
"differences among children or adolescents". The ALE pooled only the GM-*decrease*
direction — 25 contrasts from 22 publications.

Eight of the eleven false positives map onto a stated criterion the config does not encode:

| pmid | why the benchmark excluded it | arms affected |
|---|---|---|
| 19349151, 22948482, 25212487, 30343133 | paediatric/adolescent sample | all three |
| 28888350 | adolescent **and** ROI-only | A1 only |
| 19996042 | FreeSurfer parcellation, not VBM | A1 only |
| 20673548 | results SVC-corrected within ROIs | all three |
| 30127342 | contrast is four diagnoses combined vs controls, direction is a GM *increase* | all three |
| 16371250, 23113800, 32938511 | meet every stated criterion; most plausibly "overlapping samples", which no screener can apply without cross-study comparison | all three |

**The clean fix is one exclusion criterion: adults only.** It is stated in the benchmark's
methods and confirmed independently in the data — 0 of the 17 gold studies in the cohort have
a sub-18 mean age or paediatric wording, against 5 of 11 false positives. Applying it:

| arm | precision now | with adults-only | recall cost |
|---|---|---|---|
| `v1-A1-mini` | 0.593 | **0.727** | none |
| `v1-record-with-evidence` | 0.591 | **0.722** | none |
| `v1-record-no-evidence` | 0.609 | **0.737** | none |

The direction requirement (GM decrease) and a flat ROI exclusion would remove two more, but
tightening ROI would *cost* recall against the benchmark's actual practice, which includes
two ROI-masked papers (§6b).

---

## 5. Input quality: the tables the pipeline never saw

29 of the 50 papers reached extraction with zero tables. Records built from a table-less
render carry **2.4 analyses on average against 4.1** for the rest, which is the material the
screener is failing to find.

| cause | papers |
|---|---|
| HTML contains real article tables, never read | **13** |
| tables cited in prose, absent from the retrieved markup | 7 |
| paper genuinely cites no tables | 6 |
| source file missing | 2 |
| tables delivered as JPEG images (22453299) | 1 |

The 13 are the actionable ones. Their retrieved HTML carries standard scholarly markup —
`<table frame="void" rules="groups">`, `<table class="content" frame="hsides">`, two to six
per paper — and the corpus recorded zero, because `build_ace` in
`scripts/pondie_arm/build_corpus.py` sources tables **exclusively** from ACE's
`processed/tables.csv` export and never inspects the article HTML. Only **11 of the 50**
PMIDs appear in that export at all. The converter needed already exists in the same module
(`html_table_to_markdown`); it is simply fed from ACE's raw files rather than from the
article.

22453299, the one true extraction loss, is the unrepresentative case: it is NeuroReport on
Lippincott, which delivers tables as JPEGs through an image-gallery viewer
(`images["T1-2"]=new image(…,"Table 1",…"Rollover.00001756-201205090-00002.T1-2.jpeg"…)`).
The only `<table>` element in the file is the citation-export widget. For that paper the
table content is pixels on the publisher's server and no parser recovers it.

Note this is orthogonal to the record-vs-text comparison: the text arm reads the same
retrieved HTML, so a missing table is missing from both. It bounds what the records can
contain, not what the comparison measures.

---

## 6. Caveats

**(a) This experiment cannot separate a small effect from noise.** Five to six discordant
pairs, 17 gold positives, one project. Every McNemar test here is non-significant and would
be at any plausible effect size. **The entire with-evidence / no-evidence difference is one
paper** (§3.4) — the κ gap, the F1 gap and the recall gap are all that single flip. Nothing
in §2 should be read as an estimate of how records perform in general; the value of this
project is the per-paper mechanism, which does not depend on the aggregate. The cue-reactivity
cohort is the powered version of the same question.

**(b) The benchmark contradicts its own stated criteria, and the records are penalised for
following them.** Pankey excludes "papers reporting regions of interest (ROIs)" and
"treatment and longitudinal effects", then includes 19794316 (voxel-wise analyses inside AAL
hippocampus/ACC masks), 21118656 ("statistical VBM analyses were masked for the ROIs under
investigation") and 28287194 (a longitudinal EMDR treatment study — which all three arms
include). Two of the four false negatives are that inconsistency. It is not fixable in the
pipeline, and encoding the criteria more faithfully would make the score *worse*.

**(c) Precision is measured against a gold set assembled with a criterion no screener can
apply.** "Overlapping samples to previous studies" is a decision about the corpus, not about
a paper, and the three undetermined false positives are most likely it. A screening stage
reading one paper at a time cannot reproduce it, so an irreducible false-positive floor
exists here independent of criteria quality.

**(d) An earlier version of this comparison was run on a hand-written config with a single
flat record source rather than the generated three-mirror layout.** It reported the two
record arms as identical (0.591 / 0.765 both) and produced one finding — a paper excluded
because the English-language criterion could not be verified from a quote-stripped record —
that **does not reproduce** here. That arm had no `processed_data_path`, so it lost the
baseline's coordinate and table route entirely, which `assert_record_arm.py` would have
caught. Any number in this report supersedes it.

---

## 7. Open items

1. **Align the record-format note with the render.** The render says absence is uncertainty;
   the note `gen_arm_config.py` injects says absence means the paper did not report it. The
   note's wording is what turned a thin parse into an exclusion on 22453299, while the
   no-evidence arm's `fulltext_incomplete` shows the better behaviour is reachable.
2. **Fall back to the article's own HTML tables when ACE's export has no rows for a PMID.**
   13 papers here, 47 in the cue-reactivity corpus (88 tables). Because pondie's cache is
   content-addressed, rebuilding those corpus entries recomputes only the affected papers.
3. **Add the adults-only exclusion** to `v1-*.yaml`, and consider the GM-decrease direction
   requirement. Worth +0.13 precision on every arm at no recall cost.
4. **Flag thin parses.** A render with 0 tables yielding ≤1 analysis is a state the record
   should declare rather than present as a complete account.
5. **Run the same two arms on cue-reactivity**, where 550 papers can distinguish the
   evidence effect from a single coin flip.
6. **The 24 cue-reactivity papers with a missing source file** are a separate retrieval gap.

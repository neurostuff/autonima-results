# Cue reactivity: extraction records as screening input, with power

Benchmark: Hill-Bowen et al., *The cue-reactivity paradigm: An ensemble of networks driving
attention and cognition when viewing drug and natural reward-related stimuli* (PMID
34400176), an ALE meta-analysis over 196 articles — 133 drug, 63 natural reward.

Produced by `scripts/pondie_arm/run_cue_record_arms.sh`.

---

## Headline

**On 542 paired papers the record arms are no longer distinguishable from full text at
finding the right studies, and stripping the supporting quotations improves precision — but
the manual audit says the improvement is right for the wrong reason.**

| arm | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| `v5-gpt-A1-mini` (article text) | 135 | 285 | 5 | 0.321 | **0.964** | 0.482 |
| `v5-gpt-A2F-mini` (previous records) | 126 | 264 | 14 | 0.323 | 0.900 | 0.476 |
| `v5-gpt-record-with-evidence` | 130 | 265 | 10 | 0.329 | 0.929 | 0.486 |
| `v5-gpt-record-no-evidence` | 130 | 251 | 10 | **0.341** | 0.929 | **0.499** |

**The new extraction closes the recall gap the old one had.** Against full text, the previous
vintage was significantly worse on both sides — gold positives p=0.022, gold negatives
p=0.001. The new records take gold positives to **p=0.180**: not distinguishable. What does
not close is the negative side (p=0.003): the records are consistently stricter, rejecting
31 papers the text arm admits against 11 the other way.

**Evidence quotations cost precision, significantly.** The two record arms find *identical*
studies — 130 TP, 10 FN — and differ only in false positives, 265 against 251. Paired: κ
0.833, one gold positive each way (p=1.000), 25 discordant gold negatives to 11, **exact
McNemar p=0.029**. This is the opposite sign from `vbm_of_ptsd`, where quotes helped, and
the mechanism is the same in both: quotes make a record behave more like the article, and
here the article arm is the one that over-includes.

**But reading the discordant papers changes the interpretation (§4).** The criteria admit
"whole-brain **or small-volume-corrected**" results. The no-evidence arm reads `spatial
scope: roi` off the record and excludes anyway — a stricter rule than the config states,
which happens to match what the meta-analysts did, because the benchmark's own annotation
keys are `*_wbonly`. On at least two papers it excludes work that plainly satisfies the
written criteria and gold rewards it. The p=0.029 is a real measurement of a criterion
mismatch, not evidence that quotes harm criterion-following.

---

## 1. What was compared

Four arms, same query, criteria and model (`gpt-5-mini-2025-08-07`), differing only in what
the full-text stage reads. All configs from `gen_arm_config.py` against
`projects/cue_reactivity/v5-gpt.yaml`; the two new ones differ from each other only in the
mirror-set name, and the records behind them only in `render_record.py --evidence full|none`.

Records: pondie at `31e6cea`, 550 papers, **1 failure**, 53,641,022 input / 6,843,647 output
tokens over 2,686 calls. At Luna flex rates (input $0.10/M, cached $0.01/M, output $0.60/M)
that is **≈$8.62, about $0.016 per paper**. 17.6% of input was cached.

Each variant was rendered twice and diffed — byte-identical over 550 — because
`study_full_text_content_hash` hashes content and a non-deterministic renderer would
re-screen everything on resume and read as drift. `assert_record_arm.py` confirms 550 of 561
screened studies read a record, with `local_found` at parity with the baseline (1139/1139).

The extraction ran at 12 workers, then was restarted at 48 after measuring that the account
sits at 0.13% of its 180M TPM ceiling; the content-addressed cache reused 1,706 completed
stage outcomes and throughput went from ~73 to ~380 papers/hour with the 429 rate falling to
zero.

---

## 2. Screening performance

`tables/cue_screening.csv`; per-stage metrics in `tables/cue_metrics_*.csv`.
Paired cohort is 542 of 550 after dropping papers no record arm actually read a record for.
140 of the benchmark's 191 studies fall inside it.

### Arm versus arm

| pair | agreement | κ | gold-positive McNemar | gold-negative McNemar |
|---|---|---|---|---|
| text vs previous records | 0.900 | 0.737 | **p=0.022** | **p=0.001** |
| text vs with-evidence | 0.906 | 0.749 | p=0.180 | **p=0.003** |
| with-evidence vs no-evidence | 0.931 | **0.833** | p=1.000 | **p=0.029** |

Precision sits near 0.33 for every arm. The dominant error in this project is
over-inclusion that none of the four avoids — 251–285 false positives against 126–135 true
ones — and the `vbm_of_ptsd` report argues most of that is criteria the configs never
encode rather than anything the records did.

---

## 3. What the records carry, against the previous vintage

Paired on the same ~163 papers, the new records are richer but cite less:

| per record | new | previous |
|---|---|---|
| extracted values | 181.8 | 119.0 |
| of those, with evidence | 128.8 | 100.2 |
| evidence coverage of extracted | **71.6%** | **84.4%** |

Most of the coverage drop is dilution — the new pipeline extracts 50% more values and the
extra ones are less often quoted. Some is not. At equal extraction volume, `direction` falls
82% → 61%, `level` 89% → 71%, `type` 81% → 51%, and for `direction` the *absolute* count of
evidenced values drops too (6.01 → 4.45 per paper). Repair is not the cause: coverage is
already 74.4% before repair runs. The vintages differ in pipeline version and schema and the
old run logs no model, so this is a measured difference, not a diagnosed one — but
`direction` and `level` are the cells the direction benchmark scores, and they are what the
with-evidence render puts quotes under.

---

## 4. Reading the discordant papers

38 papers separate the two record arms; 26 with-evidence-only, 12 no-evidence-only, and
**only one gold positive in each direction**. The whole effect is on gold negatives, and
almost every case is the same criterion. Ten read by hand:

| pmid | what the article says | include? | arm that was right |
|---|---|---|---|
| 17197102 | "did not survive whole-brain corrections"; drug>neutral from functional ROIs | no | no-evidence |
| 20445032 | photographs of a *rejecting beloved* — not a drug or reward cue at all | no, on stronger grounds than either arm gave | no-evidence |
| 21704307 | "**small volume correction (SVC) tool in SPM2**", plus whole-brain at uncorrected p<.001 | **yes**, SVC is explicitly admitted | with-evidence |
| 24789842 | "p<.005 (required for a **whole brain correction** of p<.05)… study-wide whole-brain mask" | **yes** | with-evidence |
| 25365800 | SVC present, but reports gains-vs-losses, not cue>neutral | no, on I4 | no-evidence |
| 25035299, 26645206, 26096546 | no whole-brain or SVC language anywhere | no | no-evidence |
| 22458676 | 0 tables in the corpus, no coverage language | undecidable from the input | neither |

Six right, two wrong, two undecidable. **The config is looser than the benchmark it is
scored against**: it admits small-volume-corrected results, the gold pooled whole-brain-only
analyses (`1_reward_neutral_2020_wbonly`, `2_drug_neutral_2020_wbonly`,
`3_natural_neutral_2020_wbonly`). The no-evidence arm applies the stricter rule by
over-reading `spatial scope: roi`, and gold agrees with it. Correcting the config to
whole-brain-only would likely reverse the ranking, and that is the experiment to run before
concluding anything about evidence quotes.

Two side findings: 20445032 should be excluded by every arm on topic grounds and all four
argued about ROIs instead; and four of the ten have **0 tables** in the corpus, so their
records could not have shown a coordinate table even where one exists.

---

## 5. Could a deterministic filter do this instead?

Every disagreement in §4 turns on structured fields, not prose, so the question is whether a
rule over the schema decides them. The rule: an analysis with `Effect.kind == contrast`, a
`ModelEstimation` that is `spatial_unit == voxel` at a group stage, and either
`Analysis.spatial_scope == whole_brain` or an `InferenceSettings` with `correction_scope ==
roi` and a named `search_volume`. It deliberately does not judge whether the cue was a drug
or reward cue — the schema stores no construct, so the topic half stays with the model.

| selector | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| deterministic gate alone | 90 | 175 | 50 | 0.340 | 0.643 | 0.444 |
| with-evidence LLM | 130 | 267 | 10 | 0.327 | 0.929 | 0.484 |
| no-evidence LLM | 130 | 253 | 10 | 0.339 | 0.929 | 0.497 |
| with-evidence AND gate | 87 | 139 | 53 | 0.385 | 0.621 | 0.475 |
| no-evidence AND gate | 88 | 135 | 52 | **0.395** | 0.629 | 0.485 |

**The rule is right and the records cannot yet feed it.** As a conjunction it buys real
precision (0.339 → 0.395) and pays more recall than it is worth. The gate misses 50 of 140
gold positives, and the failing clause says why:

| why the gate rejected a gold positive | n |
|---|---|
| no analysis typed `Effect.kind == contrast` | **23** |
| `ModelEstimation.stage` not recognisable as group-level | 11 |
| neither `whole_brain` nor an SVC inference | 9 |
| `spatial_unit` not voxel | 6 |
| no analyses at all | 1 |

Only 9 of the 50 are the substantive question. The rest are extraction typing and one
schema decision: `ModelEstimation.stage` is deliberately an open string whose description
says "nothing derives anything from it" — this filter is the counterexample, and 11 papers
fail on wordings like `second level` and `subject and group`. A second obstacle is type
hygiene: `spatial_scope` comes back as `'whole_brain'` and as `['whole_brain']` in the same
corpus, so any rule needs normalising before it can compare.

---

## 6. Caveats

**(a) The benchmark's criteria and the config's criteria are not the same criteria** (§4).
Until that is reconciled, precision differences between arms measure agreement with an
unstated rule as much as anything else.

**(b) The evidence result is significant and narrow.** p=0.029 on gold negatives, nothing on
positives, one project. It says quotes change which negatives get through, not that records
without quotes are better records.

**(c) Input quality bounds all of it.** 153 of 550 papers reach extraction with zero tables,
and 47 of those have real `<table>` markup in the retrieved HTML that `build_ace` never
reads because it sources tables only from ACE's `tables.csv` export. A further 24 have no
source file at all.

**(d) One paper failed extraction.** 24373127 fails repair with `TypeError: unhashable type:
'dict'` — the model emitted `{"value": "positive"}` with no `extraction_status`, so
`values.read` returns the dict and a set membership test raises. It keeps its unrepaired
build output. 3 of 289 records carry such a wrapper.

---

## 7. Open items

1. **Decide whether the criteria mean whole-brain-only**, and re-run. This is the largest
   single uncertainty in the report and it is cheap to resolve.
2. **Fall back to the article's own HTML tables** when ACE's export has no rows for a PMID —
   47 papers here, 88 tables.
3. **Fix the malformed-wrapper crash** in `values.read`, then recompute repair for the
   affected papers; the cache makes that a few papers, not a re-run.
4. **Make `Effect.kind` and `ModelEstimation.stage` derivable** if the deterministic gate is
   to be used: 34 of the gate's 50 misses are those two fields.
5. **Normalise `spatial_scope`** to a consistent scalar or list.

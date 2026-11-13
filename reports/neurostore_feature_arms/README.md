# NeuroStore feature arms

*Prepared 2026-11-10; run on all nine projects 2026-11-11 to 12 (results at the end). autonima
branch `feat/neurostore-features`, stacked on `feat/neurostore` (neurostuff/autonima#91). Needs
neurostuff/neurostore#1946 (merged and deployed 2026-11-11), which lets the analysis query be
restricted to a list of papers.*

**Question.** Can autonima judge NeuroStore's extracted values instead of each article's full
text, at little loss against NeuroMetaBench and at far fewer tokens? Can the full-text and
analysis-selection judgments be collapsed into one?

**Input.** Since 2026-11-06 NeuroStore serves, for each analysis of a paper's Compose parse,
the features its extraction pipeline (NsPond) resolved:
- the contrast and its two sides;
- the task;
- the spatial scope (whole brain, region of interest or searchlight);
- the effect kind, the groups and n, the medical condition;
- the measure and the design.

Coverage, measured live on 2026-11-09 over each baseline's candidates: 87% of candidates and
81% of the gold studies the search finds have features (lowest: executive function 63%, cue
reactivity 73%).

## Arms

Every arm is the project's NeuroStore search arm, the baseline in
`reports/neurostore_arms/baseline.yaml`, with only the keys below changed.
`scripts/neurostore_feature_arms.py` writes the configs (`projects/<p>/<best>-ns-<arm>.yaml`)
and refuses to run one that differs anywhere else.

| Arm | Change | Calls per abstract-included study | Isolates |
|---|---|---|---|
| `rec` | `retrieval.records.kind: neurostore`: full-text screening and annotation read each study's NeuroStore record instead of the article | 2 (full text, annotation) | record vs full text |
| `rec-comb` | + `screening.fulltext.mode: combined` | 1 | combining the two judgments (vs `rec`) |
| `rec-comb-noabs` | + `screening.abstract.skip_stage: true` | 1, for every candidate | whether the abstract pass pays for itself |
| `rec-global` | `rec` with `screening.fulltext.skip_stage: true`, the full-text criteria added to annotation's global criteria | 1 | annotation alone, deciding per analysis |

**Common to all arms:**
- **No coordinate parsing:** `parsing.parse_coordinates: false`, because the analyses come with the records.
- **Study text reaches annotation:** `study_fulltext` is added to `annotation.metadata_fields` where the baseline lacks it (vbm_of_ptsd, vbm_of_substance_use). Records reach annotation only through it.
- **One `rec-global` adjustment:** in cue_reactivity, a full-text criterion that YAML reads as a one-key mapping is written as `label: text`. In the baseline, screening accepted the mapping and showed the model its dict form.

**Holding the rest fixed:**
- **Same candidates:** the cohort is pinned to the baseline's search results.
- **Shared decisions reused:** each arm's first run seeds its cache from the baseline run. Search and abstract decisions are therefore the baseline's (`rec-comb-noabs` has no abstract decisions).
- **Same judge:** the models are the baseline's. A combined call uses the full-text model.

**Studies without features** are unavailable in every arm, each with its reason in
`documents_unavailable.csv`. They are never screened from their articles.

## Running

```bash
systemd-run --user --unit nsarms-rec-comb --collect --working-directory="$PWD" \
  --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
  bash -c 'scripts/run_neurostore_feature_arms.sh rec-comb vbm_of_ptsd emotion_regulation_2022 > /tmp/arms.log 2>&1'
```

Run it in a systemd unit of its own, not with `setsid nohup` from an app's terminal. On
2026-11-11 systemd-oomd killed the Claude app's cgroup, and two runs started inside it died with
it. annotation saves only when its stage ends, so the whole stage was lost.

1. **Dry run:** without `--live`, `scripts/neurostore_feature_arms.py <project> <arm>` stubs the LLM, writes to `<run>-stub`, and counts the calls a live run would make. It still reads NeuroStore, so it needs #1946 deployed too.
2. **Pilot:** vbm_of_ptsd (50 candidates), then emotion_regulation_2022, then all nine.
3. **After each run:** check the logs for 429s, since autonima exits 0 on quota errors. Then regenerate the reports before reading any metric.

## Scoring

`scripts/compare_neurostore_feature_arms.py` scores every arm present, and the baseline, into
`summary.csv` (one row per run) and `maps.csv` (one row per contrast). Per-run evaluations go
to `<project>/<run>/`. Below is what it was planned to measure; the covered-subset split is
computed by hand in the results.

**Per arm, against gold and against the baseline:**
- **Screening:** precision, recall and F1. Split recall lost to coverage (no features) from recall lost to judging.
- **Covered subset:** the same comparison over the studies that have features, which isolates record vs full text.
- **Selection and coordinates:** study-level selection precision and recall per target, and coordinate recall.
- **Maps:** r and Dice, from the manual-vs-auto reports.
- **Cost:** tokens and cost per stage, from the run's usage.

**`rec-global` has no full-text decisions.** Its study-level inclusion has to be read as "any analysis selected for any target".

## Pilot results (2026-11-11, record format v2)

vbm_of_ptsd and emotion_regulation_2022, all four arms, live (gpt-5-mini via Portkey), no
rate limits and no failed decisions. Scored by `scripts/compare_neurostore_feature_arms.py`
into `summary.csv` and `maps.csv`. "Selected" counts a study as included when any of its
analyses is selected for any target, for every arm alike.

**emotion_regulation_2022** (88 gold studies; 4 contrasts):

| | baseline (full text) | rec | rec-comb | rec-comb-noabs | rec-global |
|---|---|---|---|---|---|
| gold studies selected | 59 | 41 | 45 | 43 | 44 |
| selected precision | 0.34 | 0.33 | 0.34 | 0.33 | 0.27 |
| mean r² of maps | 0.60 | 0.62 | 0.60 | 0.62 | 0.63 |
| mean Dice | 0.58 | 0.61 | 0.59 | 0.61 | 0.61 |
| calls | 683 | 554 | 395 | 705 | 384 |
| cost | not recorded | $3.04 | $2.83 | $4.10 | $3.23 |

**vbm_of_ptsd** (21 gold studies; 1 contrast):

| | baseline (full text) | rec | rec-comb | rec-comb-noabs | rec-global |
|---|---|---|---|---|---|
| gold studies selected | 13 | 10 | 10 | 11 | 10 |
| r² of map | 0.69 | 0.68 | 0.68 | 0.71 | 0.68 |
| cost | not recorded | $0.18 | $0.15 | $0.20 | $0.15 |

**Findings:**
- **Maps match the full-text baseline or beat it** with fewer studies.
- **Study recall is lower.** In emotion regulation, 9 of the combined arm's 17 lost gold studies have no features, or have them only for another version of the study.
- **The PTSD losses were largely a misreading of the record.** Analyses NeuroStore labels whole-brain also listed "corrected within roi" in their inference settings, and the judge applied the protocol's ROI/SVC exclusion to them.
- **Combining the two judgments saves calls, not much money.** Output tokens (the model's reasoning) are about as large as input.
- **Skipping the abstract pass costs about 45% more** with no clear gain, so it is dropped from further runs.
- **The baselines' cost is not recorded:** their progress files hold only the last pass.

**Change after the pilots: record format v3** (autonima e8068f9). Spatial scope decides whole
brain or region of interest. Inference settings are labelled Thresholding, and the record says
a correction listed there does not make a whole-brain analysis an ROI one. The extraction is
unchanged. The results above are v2; the pilots have not been re-run on v3.

## Change after the pilots, corrected: record format v4

v3 was wrong. James's definitions: spatial scope is where the model was fit (an estimate at
every voxel), and the inference settings' correction scope is where significance was tested,
which can be a subset. Whole-brain estimation with inference only within ROIs (small-volume
correction) is a common case, not a contradiction. v4 (autonima 09dc8e0) shows "Model fit over"
and "Inference over" / "Inference restricted to", explains both neutrally, and leaves it to each
review's criteria which one a "whole-brain" criterion means. The PTSD losses traced to this were
therefore the judge applying GLOBAL_E1 as written; the gold includes some SVC studies its own
criteria exclude.

## dementia (2026-11-11, record format v4; rec, rec-comb, rec-global)

74 gold studies; 4 contrasts. No rate limits, no failed decisions.

| | baseline (full text) | rec | rec-comb | rec-global |
|---|---|---|---|---|
| gold studies selected | 32 | 26 | 29 | 30 |
| selected precision | 0.63 | 0.60 | 0.62 | 0.60 |
| mean r² of maps | 0.26 | 0.24 | 0.31 | 0.30 |
| mean Dice | 0.18 | 0.21 | 0.29 | 0.25 |
| calls | 252 | 201 | 154 | 155 |
| cost | not recorded | $1.07 | $1.03 | $1.27 |

Per contrast, r against the expert map: combined beats the baseline on all (0.62 vs 0.55),
decrease (0.59 vs 0.56) and structural (0.59 vs 0.47), and trails it on functional (0.38 vs
0.44, from 18-19 studies). The combined arm's 4 lost gold studies were all excluded at full
text; none is a coverage loss here.

## All nine projects (2026-11-12)

`rec`, `rec-comb` and `rec-global`, live (gpt-5-mini via Portkey), no rate limits. Record format
v4 everywhere except vbm_of_ptsd and emotion_regulation_2022, which are the v2 pilots above and
were not re-run. decision_making is scored against the corrected gold (NeuroMetaBench branch
`fix-decision-making-gold`, ae1ad87: 146 included studies, rebuilt risky coordinates), and so is
its baseline. "Gold selected" counts a gold study when any of its analyses is selected for any
target, for every arm alike.

| Project | Gold | Gold selected: baseline / rec / rec-comb / rec-global | Mean r² of maps: same order | Calls: baseline / rec-comb | rec-comb cost |
|---|---|---|---|---|---|
| cue_reactivity | 191 | 149 / 106 / 117 / 128 | 0.52 / 0.52 / 0.50 / 0.49 | 1317 / 731 | $5.82 |
| decision_making | 146 | 83 / 69 / 70 / 72 | 0.51 / 0.47 / 0.49 / 0.47 | 877 / 677 | $7.23 |
| dementia | 74 | 32 / 26 / 29 / 30 | 0.26 / 0.24 / 0.31 / 0.30 | 252 / 154 | $1.03 |
| emotion_regulation_2022 (v2) | 88 | 59 / 41 / 45 / 44 | 0.60 / 0.62 / 0.60 / 0.63 | 683 / 395 | $2.83 |
| executive_function | 171 | 59 / 49 / 53 / 52 | 0.60 / 0.59 / 0.60 / 0.60 | 1334 / 796 | $7.90 |
| problem_solving | 127 | 65 / 71 / 68 / 71 | 0.60 / 0.59 / 0.59 / 0.61 | 729 / 573 | $5.22 |
| social | 238 | 168 / 150 / 153 / 151 | 0.54 / 0.54 / 0.55 / 0.54 | 1249 / 770 | $6.98 |
| vbm_of_ptsd (v2) | 21 | 13 / 10 / 10 / 10 | 0.69 / 0.68 / 0.68 / 0.68 | 62 / 31 | $0.15 |
| vbm_of_substance_use | 79 | 48 / 45 / 48 / 45 | 0.30 / 0.35 / 0.34 / 0.34 | 231 / 123 | $0.98 |
| **all nine** | 1135 | **676 / 567 / 593 / 603** | **0.51 / 0.51 / 0.52 / 0.52** (mean of projects) | **6734 / 4250** | **$38.12** |

Total cost: `rec` $41.17, `rec-comb` $38.12, `rec-global` $46.24. The baseline's cost is not
recorded, and its calls are counted one per study and stage from its decisions, so they exclude
retries; the arms' calls include them.

**Findings:**
- **Maps are on par with the full-text baseline.** Across the 35 contrasts, the combined arm's r
  is within 0.02 of the baseline's on 19, higher on 9 and lower on 7; the mean difference is
  +0.01. The largest moves: decision making ambiguous 0.50 → 0.42 (lower), dementia structural
  0.47 → 0.59 and substance use cannabis 0.17 → 0.43 (4 studies), both higher.
- **About 12% fewer gold studies are selected** (combined 593 vs 676), at the same precision (2313
  studies selected vs 2316). Of the 125 gold studies the baseline selected and the combined arm
  did not, 51 have no NeuroStore record (no features, or features only for another version of
  the study): a coverage loss. The other 74 were judged out. The combined arm also selected 42
  gold studies the baseline missed. Problem solving and substance use lose nothing overall.
- **The combined arm is the cheapest:** $38 for all nine projects, with 63% of the baseline's
  calls. `rec-global` (annotation alone, the full-text criteria as global ones) selects the
  most gold studies (603) but costs the most, because its one call per study carries every
  criterion and fails validation more often.
- **Validation failures, decision making only.** The model lists an unmet inclusion criterion's
  ID (e.g. `GLOBAL_I5`) under `exclusion_criteria_applied`, which annotation's schema rejects;
  after three attempts the study's annotations are left undecided and retried on the next pass.
  First pass: `rec-global` 82 studies, `rec` 45, combined 22. `rec` and `rec-global` got a
  second retry pass; 6 and 16 studies stay undecided and export as null, which lowers their
  recall. The combined arm decided all of them on its retry. Every other project ended with
  none undecided.

**Caveats:**
- **Executive function's `rec` and `rec-global` costs are undercounts.** Their first annotation
  pass (470 and 616 studies) was killed by systemd-oomd at 20:04 on 2026-11-11 and redone; the
  killed pass's spend was never written to `usage_history`.
- **decision_making's `rec` and `rec-global` include a third pass** in their calls and cost.

## Full-text fallback (2026-11-13)

`rec-comb-ft` is `rec-comb` with `retrieval.records.fallback: fulltext`
(autonima 0ebfc01). A study NeuroStore has coordinates for but no features is reviewed from
its article instead of being left unavailable; the record is NeuroStore's list of its analyses
followed by the article's text. Each run seeded from the project's `rec-comb` run and read its
frozen NeuroStore snapshots, so the two arms differ only in the fallback studies: every
feature-record decision is reused, and only fallback studies were called. Run on the three v4
projects with the largest coverage losses (emotion regulation is still v2 and was left out).
No rate limits, no undecided studies.

| Project | Fallback studies with an article | Gold selected: baseline / rec-comb / rec-comb-ft | Mean r²: same order | Added calls | Added cost |
|---|---|---|---|---|---|
| cue_reactivity | 94 | 149 / 117 / **135** of 191 | 0.52 / 0.50 / 0.50 | 79 | $1.42 |
| social | 84 | 168 / 153 / **163** of 238 | 0.54 / 0.55 / 0.56 | 54 | $1.15 |
| executive_function | 82 | 59 / 53 / **60** of 171 | 0.60 / 0.60 / 0.61 | 51 | $1.24 |
| **three projects** | 260 | **376 / 323 / 358** | | 184 | $3.81 |

**Findings:**
- **The fallback recovers two thirds of the recall the records lose.** Against the baseline,
  the shortfall falls from 53 gold studies (14%) to 18 (5%); executive function now selects
  one more gold study than the baseline. The fallback added 35 gold studies and lost none.
- **Precision holds** (cue 0.23 → 0.24, social 0.33, executive function 0.105 → 0.112): it
  also added 82 non-gold studies.
- **Maps are unchanged or slightly better**, the largest move cue's drug contrast r 0.752 →
  0.763 (159 → 198 studies).
- **The added cost is about 18%** of the combined arm's ($3.81 on $20.70). A fallback call
  reads an article, so it costs about 2.3 times a record call.
- **Still missing:** 35 gold studies the baseline selected, across the three projects. They
  were either judged out or are fallback studies with no retrievable article; not yet split.

**Setup notes.** The first attempt (2026-11-12) failed on a revoked Portkey key, and the venv
lacked ACE's dependencies, so the local HTML article folders failed silently (executive
function got 2 of 164 articles). Both fixed; the runs above are clean re-runs from scratch.
autonima also needed the PMCID lookup fix (74663ee, cherry-picked from
`fix/pmcid-lookup-403`), without which Python 3.14 silently loses PMC articles.

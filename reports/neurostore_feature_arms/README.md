# NeuroStore feature arms

*Prepared 2026-11-10; not run yet. autonima branch `feat/neurostore-features`, stacked on
`feat/neurostore` (neurostuff/autonima#91). Waiting on neurostuff/neurostore#1946, which lets
the analysis query be restricted to a list of papers.*

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
setsid nohup scripts/run_neurostore_feature_arms.sh rec-comb vbm_of_ptsd emotion_regulation_2022 > /tmp/arms.log 2>&1 &
```

1. **Dry run:** without `--live`, `scripts/neurostore_feature_arms.py <project> <arm>` stubs the LLM, writes to `<run>-stub`, and counts the calls a live run would make. It still reads NeuroStore, so it needs #1946 deployed too.
2. **Pilot:** vbm_of_ptsd (50 candidates), then emotion_regulation_2022, then all nine.
3. **After each run:** check the logs for 429s, since autonima exits 0 on quota errors. Then regenerate the reports before reading any metric.

## Scoring (to do once runs exist)

**The existing scripts** (`scripts/compare_neurostore_arms.py`, `scripts/neurostore_same_studies.py`) know only the `coords` and `search` arms. They need the new arm names added.

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

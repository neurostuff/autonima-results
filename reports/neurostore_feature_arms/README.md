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

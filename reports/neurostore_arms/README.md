# NeuroStore as autonima's search index and coordinate source

*2026-11-03. autonima branch `feat/neurostore`; scripts in `scripts/neurostore_*.py`.*

**Question.** Can autonima take its candidate studies and its coordinates from NeuroStore
instead of PubMed and LLM table parsing, at no loss of accuracy against NeuroMetaBench? If
so, it is the token-efficient baseline for the skills implementation (`autonima-skills`),
whose Neurosynth Compose mode works from NeuroStore's index.

## What was built (autonima, `feat/neurostore`)

- **`search.database: neurostore`.** Searches NeuroStore titles and abstracts (or full text)
  with the config's PubMed query, translated, or a native query. Study metadata still comes
  from PubMed, so every later stage is unchanged. What the translation dropped or
  approximated is logged, never silent.
- **`parsing.coordinate_source: neurostore`.** Takes each included study's analyses from one
  NeuroStore version (the first, in a configurable source order, that has coordinates),
  keeping `result`-role analyses only. What each study used is frozen in a per-run snapshot.
- New settings enter the cache signatures only when set: all 55 loadable autonima-results
  configs hash identically on master and the branch. 32 new tests; 321 pass.

## Coverage of NeuroMetaBench

`reports/neurostore_coverage/` (`scripts/neurostore_coverage.py`).

| | Gold studies | In NeuroStore | With result coordinates |
|---|---|---|---|
| All 9 projects | 1,141 | 990 (87%) | 946 (83%) |
| Weakest: dementia | 74 | 50 | 48 (65%) |
| Weakest: executive function | 171 | 114 | 95 (56%) |

Of the 195 gold studies without usable coordinates, 151 are not in NeuroStore at all.

**Coordinate quality.** Gold-peak recall: the fraction of a gold study's expert-curated peaks
with a candidate point within 2 mm (Talairach points also counted at their MNI image, since
NeuroMetaBench stores them converted).

| Gold studies | autonima's LLM parse | NeuroStore |
|---|---|---|
| Covered by both (659) | 0.751 | 0.740 |
| All with gold points (1,017; missing = 0) | 0.493 | 0.584 |

NeuroStore matches autonima's parse where both have the study, and covers more studies,
because it does not depend on autonima retrieving the article's tables. 888 of the 946 studies
use Neurosynth Compose's own parse (`study_schema`); the rest fall back to earlier extractions.

**Search.** best's PubMed queries, translated, against gold:

| Search | Recall | Records to screen |
|---|---|---|
| PubMed (best) | 0.853 | 22.6k |
| PubMed hits that NeuroStore has coordinates for (ceiling for NeuroStore coordinates) | 0.733 | — |
| NeuroStore, titles and abstracts | 0.712 | 8.3k |
| NeuroStore, full text | 0.838 | 73.3k |

Title-and-abstract search comes within 0.02 of that ceiling at 37% of PubMed's records: the
cheap option for abstract screening. Full-text search recovers nearly all of PubMed's recall
but returns 3.2× its records.

## End to end: best's configs on NeuroStore

`scripts/neurostore_arms.py` derives two arms from each project's best config, changing only
NeuroStore keys; `scripts/compare_neurostore_arms.py` scores them with the paper's metrics
(final-include screening against gold; brain-masked r² of unthresholded maps, Dice of FDR maps).

- **coords**: best's search, screening and cohort (reused from its cache, so no screening calls),
  with coordinates from NeuroStore instead of LLM table parsing. Only annotation re-runs.
- **search**: NeuroStore title-and-abstract search as well. Studies best never saw are screened;
  shared studies reuse best's and the coords arm's decisions.

| Project | Contrasts | Recall: best / coords / search | Mean r²: best / coords / search |
|---|---|---|---|
| Cue reactivity | 3 | 0.691 / 0.728 / 0.733 | 0.568 / 0.501 / 0.519 |
| Decision making | 3 | 0.438 / 0.497 / 0.529 | 0.356 / 0.354 / 0.366 |
| Dementia | 4 | 0.446 / 0.554 / 0.459 | 0.292 / 0.330 / 0.257 |
| Emotion regulation | 4 | 0.636 / 0.670 / 0.625 | 0.632 / 0.597 / 0.604 |
| Executive function | 4 | 0.322 / 0.363 / 0.363 | 0.628 / 0.602 / 0.602 |
| Problem solving | 5 | 0.508 / 0.532 / 0.492 | 0.643 / 0.621 / 0.600 |
| Social processing | 5 | 0.618 / 0.714 / 0.634 | 0.550 / 0.571 / 0.538 |
| VBM: PTSD | 1 | 0.381 / 0.476 / 0.571 | 0.578 / 0.655 / 0.686 |
| VBM: substance use | 6 | 0.696 / 0.722 / 0.671 | 0.389 / 0.321 / 0.302 |

Recall is final includes over every gold study, so the arms share a denominator.

| Across projects | best | coords | search |
|---|---|---|---|
| Recall (mean of 9) | 0.526 | 0.584 | 0.564 |
| Precision (mean of 9) | 0.394 | 0.387 | 0.384 |
| Studies in final includes | 2,104 | 2,339 | 2,333 |
| Mean r², 35 contrasts | 0.510 | 0.492 | 0.477 |
| Mean r², 31 contrasts (the paper's 8 projects) | 0.538 | 0.513 | 0.505 |

- **Recall goes up, maps go slightly down.** NeuroStore supplies coordinates for studies whose
  tables best could not parse, so final includes rise 11% and recall by 0.04–0.06, at the same
  precision. Map correspondence falls by 0.02–0.03 r² (coords: worse in 22 of 31 contrasts on
  the paper's eight projects, Wilcoxon P = 0.015; per project, P = 0.31; search: P = 0.004 and
  0.25). The losses concentrate in cue reactivity and VBM substance use; social, dementia and
  PTSD improve.
- **Where the loss comes from.** `scripts/neurostore_same_studies.py` rebuilds each coords map
  from only the studies best put in it (92% of best's study-target pairs survive; the rest have
  no NeuroStore analysis selected). On those studies, NeuroStore coordinates and the selections
  made on them score 0.036 lower than best (worse in 30 of 35 contrasts). Adding the studies
  only NeuroStore supplies recovers half of that (+0.018, better in 26 of 35). So the loss is in
  the coordinates or in how analyses are split and then selected, not in the extra studies.
  Separating those two is the next diagnostic: score the selected analyses' peaks against gold.
- **Cost.** A clean coords arm costs one annotation pass, about $53 for all nine projects plus
  about $1.50 to retry failures; coordinate parsing (one LLM call per table) is gone. The search
  arms added about $5–10. Actual spend was about $150–160, because each early "retry" pass here
  re-ran every study (below); Portkey has the exact figure.
- **Annotation failures remaining** (studies whose LLM output never validated, recorded as
  failures, not exclusions): coords 0–15 per project (decision making 15), search 0–15.

Of the two arms, **coords is the baseline to build on**: same screening as best, more studies,
maps within 0.03 r², and no table parsing. The search arm adds the screening savings of
NeuroStore's smaller result sets (8.3k records against 22.6k) for a further small loss.


## Issues found on the way

- **PMC ID lookups fail silently (autonima, all versions).** NCBI's ID converter answers
  requests 2.34 / urllib3 2.8 on Python 3.14, autonima's default `uv` environment, with 403;
  the standard library and curl succeed. autonima logs it and treats those studies as having
  no full text: a re-run of emotion regulation fell from 305 included to 231. Separate task
  started (branch `fix/pmcid-lookup-403`); the arms here take PMC IDs from best's records.
- **Annotation saves only when the stage ends.** An interrupted run loses every completed
  decision (143 of 285 once here). Separate task started.
- **Annotation validation rejects whole studies.** gpt-5-mini often lists inclusion IDs under
  `exclusion_criteria_applied`; one bad field fails the study after three attempts. In
  decision making, 55 of 388 studies failed in one pass; a retry of just those left 15.
- **Seeding a run's cache overwrites its own (`copy_valid_cache_from`).** autonima copies the
  seed's stage files over the run's, even when the run already holds newer valid results. The
  arm script seeded every pass, so each retry pass replaced the NeuroStore annotations with
  best's and re-annotated everything. Fixed in the script (seed only a fresh run); worth a
  guard in autonima. `usage_total` in `execution_progress.json` is per invocation, which hid it.
- **NeuroStore full-text search applies `NOT` to the text.** A PubMed `NOT review[tiab]`
  removed papers that only cite a review: 19 gold studies across three projects. Fixed in the
  translation: top-level `NOT` clauses stay limited to titles and abstracts.

## For the NeuroStore / Compose API

1. **Field-scoped operators** (`title:`, `abstract:`, `text:`), so a full-text search can keep
   `NOT` and filter clauses to titles and abstracts without a second request.
2. **Prefix matching** (`reactiv*`). Matching is on whole stemmed words, so truncation is
   approximated by appending common endings: long queries, and unreachable forms.
3. **Batch lookup by PMID** (`pmid=1,2,3` returns nothing today; a repeated `pmid` keeps the
   last). Every study is one request now, two with its analyses.
4. **Stable parse versions.** `study_schema` versions are updated in place (several changed the
   day of this test). Expose a parse id or revision and let a client fetch that revision, so a
   review can cite exactly the coordinates it used; autonima snapshots them meanwhile.
5. **Table metadata** for `study_schema` parses (label, caption, footer are null), which
   analysis selection reads.
6. **A version's analyses in the listing call**, e.g. `base-studies/?pmid=…&version_source=study_schema&nested=true`,
   to halve the request count.

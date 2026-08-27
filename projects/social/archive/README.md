# Archived social arms

Archived 2026-08-27. Nothing here is deleted; the run outputs are intact and the configs still
validate. They are out of the main directory so they are not picked up as candidate arms.

## Contaminated: query and pmids_file both set

- `v3-search-all_pmids`
- `v3-search-all_pmids-multi_analysis`
- `v3-search-all_pmids-multi_analysis-ft`

`autonima` UNIONS `query` and `pmids_file` when both are present (`search/pubmed.py`, in
`search()`: `pmids +=` the search hits, then `pmids +=` the file). The docstring at `pubmed.py:51`
claims the query is ignored, which is the opposite of the behaviour — filed upstream as
[autonima#57](https://github.com/neurostuff/autonima/issues/57).

So each of these arms screened its query hits **plus** the 486-PMID curated pool, which contains
every gold study by construction. For `v3-search-all_pmids-multi_analysis-ft`, the arm the
cross-project overrides used to point at:

| | n unique PMIDs |
|---|---|
| query alone | 1024 |
| curated pool | 486 |
| overlap | 445 |
| union actually screened | 1065 |

Gold coverage went from 0.952 (219/230, query alone) to 1.000 — the pool handed the arm 11 gold
studies its own search would have missed. The bias is modest but real, and these arms were never a
measurement of search in isolation. `../v3.yaml` (query only) replaces them.

Note when reading these runs' `search_results.json`: it stores the **pre-dedup concatenation**,
1510 entries for the 1065 unique PMIDs above.

## Exact duplicate

- `v3-all_pmids-multi_analysis`

Resolves identically to `../v3-all_pmids.yaml` in every stage (`search`, `screening`, `retrieval`,
`annotation`) — verified via `ConfigManager` + `model_dump()`. The `-multi_analysis` suffix names
the *default*: `prompt_type` unset already resolves to `multi_analysis` (`config.py:446`), so the
two configs describe the same run.

## Deliberately kept in the main directory

- `../v3-all_pmids.yaml` — differs from `v3-allstudies` **only** by omitting `study_fulltext` from
  annotation `metadata_fields`. That is a clean ablation of whether giving the annotator full text
  helps, so it earns its place.
- `../v3-all_pmids-multi_analysis-ft-gpt52.yaml` — same as `v3-allstudies` but
  `model: gpt-5.2-2025-12-11`. A legitimate model comparison.

Both would read better under the current convention as `v3-allstudies-nofulltext` and
`v3-allstudies-gpt52` (version tracks criteria, suffix tracks everything else). Not renamed yet.

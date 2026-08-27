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

Measured from each arm's own `outputs/abstract_screening_results.json` (all three are identical):

| | n |
|---|---|
| query hits | 1023 |
| curated pool (486 listed, 485 retrievable) | 485 |
| overlap | 445 |
| union, unique PMIDs | 1063 |
| union, screening **records** | 1508 |

Gold coverage went from 0.952 (219/230, query alone) to 1.000 — the pool handed the arm 11 gold
studies its own search would have missed. The bias is modest but real, and these arms were never a
measurement of search in isolation. `../v3.yaml` (query only) replaces them.

There is **no dedup anywhere** — not in `search_results.json` and not before screening. `search()`
builds its `studies` list by iterating the raw concatenated `pmids`, so a PMID in both sources yields
two `Study` objects that flow through independently. The 445 overlapping PMIDs were each screened
**twice, as separate LLM calls** (445/445 with distinct timestamps), which is ~30% redundant
abstract-screening spend on each of these runs. All 445 pairs agreed on the verdict, so the
duplication cost money and inflated the precision denominators but changed no decision.

That padding is why these arms' precision reads low: search precision 0.224 here versus 0.490 for the
same criteria on the curated pool alone (`../v3-allstudies`), and 0.407 versus 0.690 at full text.
Recall is identical across the two, since the extra records are duplicates and non-gold.

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

# End-to-end retest: how much does a run drift, and does the map move?

Re-executes every LLM stage of a project against a **frozen corpus** — identical search results,
identical retrieved full text — so any difference from the original run is model nondeterminism
rather than a change of inputs.

This is the §8a test–retest item from [PAPER_OUTLINE.md](../../PAPER_OUTLINE.md), widened from
annotation-only to end-to-end because the interesting question is not the flip rate itself but
whether it reaches the brain map.

## Projects

Two, chosen as the extremes of the benchmark set, both at their `best` tier per
[run_categories.yaml](../../run_categories.yaml):

| project | run | why | annotation F1 | map dice |
|---|---|---|---|---|
| executive_function | v3 | strong case | 0.879 | 0.643 |
| social | v3 | the outlier — lowest annotation F1 in the set, and alone contributes 338 of 471 pooled false negatives | 0.729 | 0.584 |

## What is frozen, and why

`search` and `retrieval` are reused from the original run; `abstract`, `fulltext`, `parsing`,
`annotation` and `output` are recomputed.

Freezing search is the explicit requirement — PubMed has already burned this project once, when a
transient degradation cost two runs 10% and 37% of their corpora and was initially misread as index
drift. The retest verifies the frozen set matches rather than assuming it: EF re-resolved to the
same 4,881 studies, zero added, zero removed.

Freezing retrieval matters for a subtler reason. Since these runs were built we have expanded
publisher access substantially (Elsevier API, Wiley TDM, manual downloads, the ACE flat-file ingest
fix). Re-running retrieval now would pull in **more** papers than the original had, and that is a
corpus improvement, not drift — it would inflate the apparent instability with our own
infrastructure work. The retrieval directory is therefore copied wholesale into the retest folder.

## Layout

    experiments/e2e_retest/
      README.md
      scripts/compare_retest.py     stage-by-stage diff of a retest against its original
      <project>/<run>/outputs/      the replicate
      <project>/<run>/retrieval/    copied from the original, frozen (gitignored)

## Running it

```bash
autonima run projects/executive_function/v3.yaml experiments/e2e_retest/executive_function/v3 \
  --clear-cache abstract --clear-cache fulltext --clear-cache parsing --clear-cache annotation -j 16
```

`-j` matters: the default is 1, which puts abstract screening for executive_function at roughly
20 hours. At 16 workers it is about 35 minutes.

Then:

```bash
python experiments/e2e_retest/scripts/compare_retest.py \
  --original projects/executive_function/v3 \
  --retest   experiments/e2e_retest/executive_function/v3
```

## What gets measured

The script reports **decision instability** and **consequence** separately, because they answer
different questions and can diverge sharply:

- per-stage flip rates (abstract, full-text, annotation) and coordinate changes per study
- whether any of it reaches the final study set and coordinate pool
- what the retest cost, read from the new per-stage token accounting

A high flip rate that does not reach the studyset means the spec is loose in places that do not
matter — a much better result than the bare rate suggests. The reverse is much worse.

Map-level drift uses the existing `scripts/compare_meta_to_benchmark.py`, which already scores a
run against its manual benchmark; this script produces the studyset and coordinate deltas that
feed it.

## Cost

Estimated before running, from measured per-item token profiles:

| project | abstract | fulltext | parsing | annotation | total |
|---|---|---|---|---|---|
| executive_function | $2.79 | $3.64 | $0.98 | $8.99 | **$16.41** |
| social | $0.59 | $2.64 | $0.63 | $4.80 | **$8.67** |
| | | | | | **$25.07** |

Upper bounds — cached-input savings are not credited. Runs now record their own usage, so the
actual figure lands in `outputs/execution_progress.json` under `usage_total`.

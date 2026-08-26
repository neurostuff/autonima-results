# Archived problem_solving runs (2026-08-26)

These runs are superseded because the project's **PubMed query was replaced**, not because their
criteria were wrong. They are kept for provenance and are not part of any current comparison.

## Why the query was replaced

The original topic clause was:

    "problem solving" OR "calculation" OR "number" OR "picture" OR "task" OR "reasoning"
    OR "math" OR "cognitive process" OR "verbal" OR "visuospatial"     [all Title/Abstract]

Four terms — `task`, `number`, `picture`, `verbal` — match most of the fMRI literature and
contributed pool without contributing gold. The clause also omitted **spatial navigation**
entirely, which is one of the source paper's own paradigm groups (§3.1.3.4); 13 of the 21
reachable gold papers the replacement initially missed were navigation/maze studies.

Measured by exact gold-membership counts against the 127-PMID list, same date window and modality
clause throughout:

    query                                  hits    gold   recall   precision
    original (these archived runs)        21452      94    0.740      0.0044
    replacement (current v1/v2)           12075     113    0.890      0.0094

The replacement is better on both axes — recall +0.150 with 44% fewer hits. The modality clause
was widened too: under the original narrow one only 106 of 127 gold papers were reachable at all;
widening lifts the ceiling to 119.

This was a transcription error rather than a design choice, which is why the old results were
archived rather than retained as a comparison arm. Note also that an over-broad search mostly
costs computation in real use — it is chiefly a scoring problem here, because precision is
measured against a fixed gold standard.

## What is in here

### `v1.yml` + `v1/` — original manual criteria, original query

Freshly re-evaluated before archiving (its own `evaluation/` directory was stale by three months,
dated 2026-05-04 against outputs from 2026-08-14):

    fulltext included 275   TP 50   FP 225   recall 0.397   precision 0.182   F1 0.249

Its retrieval config also pointed at `/data/alejandro/projects/autonima-results/...`, which does
not exist on this machine. A real defect, but a mild one — pubget/PMC supplied enough text that
the run still included 275 studies. The replacement v1 repairs those paths.

### `v3.yaml` + `v3/` — Claude-revised annotation criteria, original query

    fulltext included 264   TP 44   FP 220   recall 0.349   precision 0.167   F1 0.226

Numbered v3 at the time; its criteria now live in the current `v2.yaml` and
`v2-annotation-only.yaml`. Note this run *also* raised `max_results` from 5000 to 20000, so its
pool was wider than archived v1's (9,981 vs 6,927) and the two are not a clean comparison.

### `v2-annotation-only.yaml` + `v2-annotation-only/` — GPT-authored annotation criteria

A third criteria set, distinct from both the manual v1 and the Claude revision. Archived when the
project was collapsed to two versions. Its fair-comparison numbers, so the three-way is
recoverable without re-running:

    run                              dice_diag   pearson_diag
    v1-annotation-only (manual)          0.514          0.695
    v2-annotation-only (GPT)             0.485          0.685
    v3-annotation-only (Claude)          0.633          0.817

The Claude revision is now `v2-annotation-only`; the manual one keeps `v1-annotation-only`.

## Current layout after this change

    v1.yaml / v1                 manual criteria      + replacement query
    v2.yaml / v2                 Claude criteria      + replacement query   (annotation-only diff from v1)
    v1-annotation-only           manual criteria,     fixed PMID list (search-independent, unaffected)
    v2-annotation-only           Claude criteria,     fixed PMID list (search-independent, unaffected)

The two annotation-only runs were **not** re-run: they draw from a fixed 127-PMID list and skip
screening, so a search change cannot affect them.

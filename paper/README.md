# paper/

Everything needed to rebuild the manuscript's display items from the run
artifacts tracked under `projects/`.

```bash
paper/reproduce.sh                  # full chain, ~1 h
paper/reproduce.sh --figures-only   # redraw from existing reports/*.csv, ~3 min
paper/reproduce.sh --skip-slow      # full chain minus the size-matched null
```

The script fails on the first error and ends by checking that all 14 display
items were written this run, so a silent half-rebuild is not possible.

## What it does not do

It does **not** re-run the AutoNIMA pipeline. Stage outputs under
`projects/<project>/<run>/outputs/` are the inputs here, and regenerating them
means re-running the LLM stages against PubMed and the publisher APIs — days of
wall time, paid API calls, and a corpus that has moved on since. The screening
results, coordinate parsing and NiMADS studysets are tracked precisely so the
evaluation is reproducible without them.

**The meta-analytic maps are not yet among them.** `.gitignore` excludes
`meta_analysis_results/`, so a fresh clone has the studysets but none of the maps
that every map-level figure reads, and `reproduce.sh` cannot run there. On this
machine they are present as ignored files left by earlier runs, which is why it
works here. The planned fix is to convert the repository to DataLad and hold the
maps in git-annex against an S3 remote; after that `datalad get` supplies them and
the `meta_analysis_results/` ignore rule comes out. Until then, treat
`reproduce.sh` as reproducible on a machine that has already run the pipeline,
not from a clone.

## Figure names do not match figure numbers

The repository filenames predate the manuscript's final ordering. Two are
inverted; check this table before pasting anything into the document.

| Manuscript | File in `reports/nature_methods_figures/` |
|---|---|
| Figure 1 | *(schematic, drawn by hand — not generated)* |
| Figure 2 | `figure2_precision_and_attainable_recall` |
| Figure 3 | `figure3_recover_and_select_analyses` |
| Figure 4 | `figure4_pipeline_vs_baseline` |
| **Figure 5** | **`figure_er_surface_contrasts`** |
| **Figure 6** | **`figure5_selection_decomposition`** |
| Supplementary S1 | `figureS1_tier_progression` |
| Supplementary S2 | `figureS2_pool_mismatch` |
| Supplementary S3 | `figureS3_size_matched_null` |
| Supplementary S4 | `figureS4_brain_maps_all` |
| Supplementary S5 | `figureS5_measured_cost` |

Three further items are built but not cited: `figure_text_to_map_baselines`,
`figure_annotation_precision_recall`, `figure_brain_maps_contrast`.

## Two interpreters, on purpose

`make_er_surface_figure.py` and `make_brain_map_figure.py` run under the system
`python3`, not the pixi environment. They need nilearn >= 0.13 for surface and
glass-brain rendering; the pixi environment pins nilearn 0.10.1 because NiMARE
0.2.1 requires it. Everything else runs under pixi.

This split is why `figureS4` once sat a week stale at a superseded column count:
it was outside the figure registry and nothing rebuilt it. `reproduce.sh` now
invokes both interpreters explicitly and verifies the outputs.

## Two environments

```bash
pixi run <cmd>          # default: siblings pinned to public commits
pixi run -e dev <cmd>   # development: siblings as editable ../ checkouts
```

`reproduce.sh` uses the default, so it builds against the pinned revisions —
`autonima` at `440de05` and `ace` at `d64291e`. Edits to `../autonima` do **not**
affect that environment; use `-e dev` when working on the siblings.

**One dependency is still unpinnable.** The retrieval used a local `pubget` fix
that exists on no public commit — "Skip an article with no usable PMCID instead of
failing its whole batch" — currently open as
[neuroquery/pubget#61](https://github.com/neuroquery/pubget/pull/61). Both
environments therefore still resolve `pubget` through `../pubget`, and a fresh
clone cannot build either until that merges and the commit is pinned here.

A second caveat belongs with the autonima pin: `440de05` is where master stood
when the evaluation ran, but Supplementary S5's cost figures were measured on
2026-09-04 and come from a later commit (`fef53ac`). No single revision produced
every number in the paper, and `execution_manifest.json` records no version.

## The chain

Each stage writes what the next one reads; the order is not arbitrary.

1. **Per-project baselines** → `projects/*/reports/baseline_vs_autonima.csv`
2. **Per-project map matrices** (annotation-only arm) → `manual_vs_auto_meta_<run>/tables/`
   — `annotation_value.py` reads these rather than recomputing from maps, so they
   must be refreshed first or its correlation columns silently go stale.
3. **Cross-project tables** → `reports/*.csv`. `compile_best_baselines.py` runs
   first; four other scripts read its output.
4. **Size-matched null** → `reports/annotation_bootstrap_null.csv`. The slow step:
   32 contrasts × 500 MKDA refits.
5. **Figures** → `reports/nature_methods_figures/`
6. **Numbers** → `manuscript_numbers.py`, which re-derives every value the
   manuscript quotes so the text can be checked against the tables.

## Metric conventions

Fixed in `scripts/map_mask.py` and applied everywhere: *r*² on unthresholded `z`
maps, Dice on FDR-corrected maps at *z* > 1.96, both computed inside the 228,483-voxel
MNI152 2 mm brain mask. Dice is provably invariant to that mask — no voxel outside
it exceeds zero in any of the 472 corrected maps — which makes it the regression
anchor: if a Dice value moves, something else broke.

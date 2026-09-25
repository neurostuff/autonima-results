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

## What this repository does not contain

**The expert reference maps.** Every map-level comparison is against reference
maps curated in neurometabench, which is a separate repository and a separate
artifact with its own manuscript and DOI. The scripts default to finding it as a
sibling directory (`../neurometabench/analysis`), overridable with
`--manual-analysis-base`. Without it, Figures 4, 5, 6 and S3 cannot be computed.

**The article full texts.** Third-party full texts cannot be redistributed. They
are needed only for the upstream stages -- screening and coordinate parsing --
which therefore cannot be reproduced from a clone. Everything downstream can: the
NiMADS studysets and annotations for all 94 runs are tracked, and
`autonima meta` regenerates the meta-analytic maps from those alone.

**The meta-analytic maps**, for that reason. They are derived data, regenerable
from the tracked studysets, and `.gitignore` excludes them. They are distributed
instead as `maps.tar.gz` on the Zenodo record, because Zenodo's GitHub
integration archives the release zipball and so contains tracked files only:

```bash
tools/bundle_maps.py restore maps.tar.gz   # put them back where the scripts expect
tools/bundle_maps.py verify                # check them against reports/maps_manifest.json
```

`reports/maps_manifest.json` is tracked, so the sha256 of every map the paper was
built from can be checked without downloading the archive.

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

## The manuscript

`latex/` holds the manuscript. Build it with:

```
paper/latex/build.sh
```

`manuscript.tex` is the document -- preamble, front matter, and the order in
which the pieces are assembled. It `\input`s two generated files:

| | |
|---|---|
| `body.tex` | Abstract through the end of the Methods |
| `supplementary.tex` | the supplementary figure captions |

Both are written by `latex/convert.py` from `Autonima - Manuscript.docx`, the
Google Docs export, against `references.bib`, the Paperpile export. **While the
Google Doc is still the source of truth, edit there and re-run `convert.py`;
edits made directly to those two files are overwritten.** Once the Doc is
retired, stop running the script and edit them as ordinary LaTeX.

Four things in the export do not survive a plain `pandoc docx -o tex`, and the
script's module docstring explains each: Paperpile writes hyperlinks rather than
Word field codes, so the only link from a superscript number to a BibTeX entry
is the citation URL; pandoc's markdown writer reads `$` as math and welds the
Fig. S5 cost figures into single words; that caption is separately corrupt in
the Doc itself; and every display equation appears twice, once as the Docs
rendering and once as the LaTeX the author typed beside it.

The script refuses to write anything it cannot verify. It checks that the
Paperpile-id-to-BibTeX-key mapping is a bijection over all 28 references, that
the in-text citations and the reference list name the same set, and -- because
BibTeX renumbers by order of first citation -- that every marker keeps the
number the Doc gave it.

Known defects in the Google Doc, corrected during conversion and worth fixing
upstream if the Doc is ever used again:

- The **Fig. S5 caption**. Docs' equation autoformat consumed the `$...$` spans
  and re-emitted them as Mathematical-Alphanumeric italics with the spaces
  gone, which is visible in the exported PDF too. All seven figures match
  `manuscript_numbers.py`, so the caption is restored from that verified text.
- **No subscripts anywhere** (zero `w:vertAlign` runs), so the similarity and
  count variables set as `scoord`, `slabel`, `nexpert`, `nauto`.
- **`?utm_source=chatgpt.com`** on the GitHub URLs in Data and Code
  availability.

## Figure files and figure numbers

Filenames match the manuscript's numbering. They did not until the LaTeX
conversion: `figure_er_surface_contrasts` was Figure 5 and
`figure5_selection_decomposition` was Figure 6, so the names were one apart
from the numbers for everything after Figure 4. Both were renamed; the
generating scripts and `reproduce.sh` were updated with them.

| Manuscript | File in `reports/nature_methods_figures/` |
|---|---|
| Figure 1 | *(schematic, drawn by hand -- not generated)* |
| Figure 2 | `figure2_precision_and_attainable_recall` |
| Figure 3 | `figure3_recover_and_select_analyses` |
| Figure 4 | `figure4_pipeline_vs_baseline` |
| Figure 5 | `figure5_er_surface_contrasts` |
| Figure 6 | `figure6_selection_decomposition` |
| Supplementary S1 | `figureS1_tier_progression` |
| Supplementary S2 | `figureS2_pool_mismatch` |
| Supplementary S3 | `figureS3_size_matched_null` |
| Supplementary S4 | `figureS4_brain_maps_all` |
| Supplementary S5 | `figureS5_measured_cost` |

Two further items are built but not cited:
`figure_annotation_precision_recall` and `figure_brain_maps_contrast`.

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

All three siblings are now pinned to public commits, so a fresh clone can build
the default environment. `pubget` is pinned to `236b762`, the merge of
[neuroquery/pubget#61](https://github.com/neuroquery/pubget/pull/61), which
carries the fix the retrieval depended on — before it, one unidentifiable record
discarded every sibling article in its batch.

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

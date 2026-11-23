# autonima-paper

Configurations, results, evaluation tables and figure scripts for the manuscript

> **Automated neuroimaging meta-analysis through LLM-guided selection improves recovery of
> published spatial patterns.** de la Vega, Kent, Yang, Nichols, Poline, Bottenhorn, Chen,
> Poldrack and Laird.

Everything here rebuilds the manuscript's figures, Supplementary Table 1 and every number it
quotes. AutoNIMA itself is a separate package:
[neurostuff/autonima](https://github.com/neurostuff/autonima), used here at
[v0.1.1](https://github.com/neurostuff/autonima/releases/tag/v0.1.1).

This repository is a frozen subset of the working repository
[neurostuff/autonima-results](https://github.com/neurostuff/autonima-results); see
[PROVENANCE.md](PROVENANCE.md) for the exact commit and how the subset was chosen.

## Layout

| | |
|---|---|
| `projects/` | one directory per review article: the configuration of every run the paper reads, and its outputs (search results, screening decisions, parsed coordinates, analysis selections, NiMADS studysets) |
| `projects/<project>/baselines/` | the search-only baseline runs |
| `run_categories.yaml` | which run is which configuration tier (initial, manually revised, benchmark-tuned, reported) |
| `scripts/` | evaluation against NeuroMetaBench, stage metrics, the size-matched null, map refitting |
| `reports/` | the evaluation tables the figures are built from |
| `paper/` | figure, table and number scripts, `reproduce.sh`, and the outputs: `paper/figures/`, `paper/tables/` |
| `tools/bundle_maps.py` | restores and verifies the meta-analytic maps |

## Rebuilding the figures and numbers

You need [pixi](https://pixi.sh), the maps bundle from this repository's Zenodo record, and
NeuroMetaBench v1.0 ([10.5281/zenodo.24409323](https://doi.org/10.5281/zenodo.24409323)) unpacked
beside this repository as `../neurometabench` (it supplies the expert-reference maps).

```bash
pixi install --locked
pixi run python tools/bundle_maps.py restore maps.tar.gz   # the maps, into projects/*/*/outputs/
pixi run python tools/bundle_maps.py verify
paper/reproduce.sh                  # everything, ~1.5 h (most of it the size-matched null)
paper/reproduce.sh --figures-only   # redraw from the tables in reports/, ~3 min
```

`paper/manuscript_numbers.py` prints every value the manuscript quotes, each with the table it
comes from. [paper/README.md](paper/README.md) documents the chain step by step.

Maps can also be regenerated from the tracked studysets with AutoNIMA instead of restored:
`pixi run python scripts/refit_maps.py` (AutoNIMA v0.1.1 on NiMARE 0.21.0).

## What is not here

- **Article full texts**, which are third-party. They are needed only to re-run AutoNIMA's
  upstream stages (screening and coordinate parsing); every stage output is tracked, so the
  evaluation does not need them. PubMed Central articles can be re-fetched with
  [pubget](https://github.com/neuroquery/pubget).
- **Re-running the LLM stages.** The configurations here re-run with AutoNIMA v0.1.1, but doing
  so calls the OpenAI API and PubMed, and the literature has moved on since these runs.
- **The skills-based agent runs** behind Fig. S6. Their scores are tracked
  (`projects/*/reports/cbma_skills_*/score.json`); the agent workspaces and transcripts are not.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the manuscript and NeuroMetaBench.

## Licence

MIT. See [LICENSE](LICENSE).

# code/

Everything needed to regenerate the figures and re-derive every number quoted in
`../vbm_of_ptsd.md`, `../cue_reactivity.md` and `../TODO.md`, given the run directories
under `projects/`. No articles or extraction records are kept here — only the evaluation
artefacts in `../data/` and the code that reads them.

## Figures

```
python experiments/record_arms/code/make_record_arm_figures.py
```

Takes no arguments in the normal case: it defaults to `../data/` for the scored CSVs and
`../figures/` for output, and reaches into the repo's `scripts/` for `house_style()`, the
palette, `panel_label()` and `save()` from `make_nature_methods_figures.py`, plus `STAGES`,
`load_gold()` and `survivors()` from `compute_gold_survival.py`. Those are imported rather
than restated so this figure cannot drift from the figure 2 the rest of the repo draws.

Figure 2 additionally reads each run's screening outputs from `projects/<project>/<run>/`,
so it can only be regenerated where those runs exist. Figure 7 needs nothing but
`../data/`.

## Meta-analytic maps

```
python code/run_mkda.py --run-dir projects/vbm_of_ptsd/v1-record-no-evidence --n-cores 6
bash   code/run_all_maps.sh          # the same, over all twelve runs
python code/meta_r2.py               # writes ../data/record_arms_meta_metrics.csv
```

`run_mkda.py` builds a NiMARE dataset from each run's `nimads_studyset.json` restricted to
the analyses one annotation key selects, runs `MKDADensity` with the default kernel and
`FDRCorrector(method="indep")`, and writes `outputs/meta_analysis_results/<key>/z.nii.gz`.
**It skips a key whose `z.nii.gz` already exists**, so delete `meta_analysis_results/` before
re-running after an annotation change or you will score stale maps — that mistake cost a
round trip here.

ALE is not used: it needs per-experiment sample sizes and the studysets carry none.

`meta_r2.py` reuses `compare_meta_to_benchmark.py`'s metric definitions — Dice at z > 1.96,
Pearson r over finite voxels, R² = r² — against each project's manual maps.

## Audits

These are the scripts behind the claims in `../TODO.md`. Each assumes the repository root as
its working directory and the benchmark checked out beside it at `../neurometabench`.

| script | answers |
|---|---|
| `audit/audit_contamination.py` | does any mapped analysis come from a paper its own arm rejected? (the check that must return zero) |
| `audit/audit_annotation_keys.py` | which papers each annotation key selects, per arm |
| `audit/audit_evidence_arm_diff.py` | every decision where `+evidence` and `-evidence` disagree |
| `audit/audit_ptsd_false_negatives.py` | gold papers each arm missed, and at which stage they were lost |
| `audit/audit_ptsd_false_positives.py` | non-gold papers each arm admitted |
| `audit/audit_ptsd_fp_abstracts.py` | the same, with abstracts and age cues, for judging them by hand |
| `audit/audit_ptsd_true_negatives.py` | excluded non-gold papers, split by whether coordinates were extracted |
| `audit/audit_exclusion_methods_abstracts.py` | method sentences in the abstract for papers excluded under E4 |
| `audit/audit_exclusion_methods_fulltext.py` | the same from the full text — **use this one**; abstracts are not sufficient to establish that an analysis was ROI-restricted, and reading region names out of a title certainly is not |

`audit/patch_annotation_metadata_fields.py` swaps `study_abstract` for `study_fulltext` in a
config's `annotation.metadata_fields`, editing only inside the `annotation:` block. This is
the change that made the PTSD arms' annotation see their own document; it also alters
`stage_hash`, which is what invalidates a donated annotation cache. It matches two-space
unquoted list items only — `vbm_of_substance_use/v2.yaml` uses four-space quoted items and
had to be patched separately.

## Provenance

These began as one-off scripts run against the working copy on the analysis host, and are
committed as they were used rather than rewritten, so the numbers in the write-ups can be
reproduced exactly. Paths inside them are therefore literal, not configurable.

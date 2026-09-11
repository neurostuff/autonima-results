# record_arms/

Does screening a **machine-generated extraction record** decide the same way as screening the
article, and do the supporting quotations inside the record matter?

Two benchmarks, three or four arms each, differing in one input:

| arm | full-text stage reads |
|---|---|
| `A1` | the article, as retrieved (pubget / Elsevier / ACE HTML) |
| `record-with-evidence` | a rendered pondie extraction record, quotation under each value |
| `record-no-evidence` | the same records, quotations stripped |
| `A2F` (cue only) | records from the previous extraction vintage |

| | |
|---|---|
| `vbm_of_ptsd.md` | 49 paired papers against Pankey et al. (PMID 36100907). Small enough to read every disagreement by hand; too small to separate a small effect from noise. |
| `cue_reactivity.md` | 542 paired papers against Hill-Bowen et al. (PMID 34400176). The powered version of the same question. |
| `tables/` | the scored CSVs behind both — `*_screening.csv` from `compare_arms.py`, `*_metrics_*.csv` from `compare_screening_to_benchmark.py`, `ptsd_errors.csv` the per-paper adjudication. |
| `code/` | everything needed to regenerate the figures and re-derive the audited numbers; see `code/README.md`. |
| `data/` | evaluation artefacts the figures read: `<project>_record_arms.csv` (screening, from `compare_arms.py`) and `record_arms_meta_metrics.csv` (Dice / r / R² per mapped analysis per arm). |
| `figures/` | `figure2_record_arms` and `figure7_record_arms` as PNG and PDF, plus the earlier confusion-matrix panel. Copies also live in `reports/nature_methods_figures/`. |

## The four benchmarks

All four projects now carry all three arms. `vbm_of_substance_use` (241 papers, Klaming et
al.) and `dementia` (448 papers) joined the two below; their narratives are in `TODO.md`
rather than in separate files.

## Regenerating

```
python experiments/record_arms/code/make_record_arm_figures.py
```

Figure 7 needs only `data/`. Figure 2 also reads each run's screening outputs under
`projects/`, so it can only be drawn where those runs exist.

## What the figures show

Panel c of figure 2 is the one to read for the arm contrast: panels a and b spend three of
their four stages drawing lines that are identical by construction, because the arms inherit
search, abstract screening and retrieval from the same cache. Panel c drops those and plots
full-text screening alone, as a precision–recall trajectory per project.

Two cautions carried from `TODO.md`. Re-running one project's annotation under an unchanged
configuration moved its map R² by 0.034, which is larger than several of the between-arm
gaps figure 7 panel b draws; those differences need error bars before they are read as real.
And VBM PTSD's `+evidence` / `-evidence` screening gap is one paper out of fifty — 21118656,
where the `-evidence` arm is the one that is factually wrong.

## A note on where this sits

This directory is documented as holding work *not* validated against a manual meta-analysis,
kept clear of the benchmark tooling. This experiment is validated against two manual
meta-analyses, so it sits here by the owner's decision rather than by that rule. The
consequence to know: the **run configs stay under `projects/`** —
`projects/vbm_of_ptsd/v1-record-*.yaml` and `projects/cue_reactivity/v5-gpt-record-*.yaml` —
because `run_cross_project_screening_reports.py` and anything globbing `projects/*/v*.yaml`
would otherwise not see the runs at all. Only the narrative and the derived tables live here.

## Reproducing

The runners are in `scripts/pondie_arm/`, not in a local `scripts/`, because every step is
shared with the other record-arm work:

```bash
bash scripts/pondie_arm/run_ptsd_record_arms.sh
bash scripts/pondie_arm/run_cue_record_arms.sh
```

Each chains the repo's own tools end to end — `render_record.py` (twice, with a determinism
diff) → `build_record_corpus.py` → `gen_arm_config.py` → `autonima run` →
`assert_record_arm.py` → `compare_screening_to_benchmark.py` → `compare_arms.py`. Records
come from a pondie extraction run; the cue set is 550 papers at ≈$8.62.

## What the two reports say together

Records hold precision and lose recall against full text, and **most of the recall loss is not
information loss**: on PTSD, three of four misses are the record applying the stated criteria
correctly to papers the benchmark included in violation of them. Only one paper in that
project is a genuine extraction failure.

The evidence quotations change which *negatives* get through, and the sign flips between
benchmarks — quotes helped on PTSD, hurt on cue (McNemar p=0.029). The mechanism is the same
in both: quotes make a record behave more like the article. Whether that helps depends on
whether the article arm is over- or under-including, which is a property of the benchmark and
not of the records.

The largest finding is not about records at all. On cue, the screening config admits
"whole-brain **or** small-volume-corrected" results while the benchmark pooled whole-brain-only
analyses, so an arm that wrongly treats every ROI-scoped record as disqualifying scores
*better*. Reconciling the criteria with the benchmark is worth more than any change measured
here.

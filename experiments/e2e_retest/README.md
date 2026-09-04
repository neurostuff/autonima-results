# End-to-end retest: how much does a run drift, and does the map move?

Re-executes every LLM stage of a project against a **frozen corpus** — identical search results,
identical abstract screening, identical retrieved full text — so any difference from the original
run is model nondeterminism rather than a change of inputs.

This is the §8a test–retest item from [PAPER_OUTLINE.md](../../PAPER_OUTLINE.md), widened from
annotation-only to end-to-end because the interesting question is not the flip rate but whether it
reaches the brain map. It does.

## Headline

**Two runs of the same pipeline, on identical inputs, produce project-level benchmark scores that
differ by ±0.02–0.04 dice and individual annotation columns that differ by up to ±0.12.** The
pipeline's advantage over search-only baselines survives this for the projects with large margins
and does not survive it for the projects with small ones.

---

## Design

| stage | treatment |
|---|---|
| search | **frozen** (verified identical, not assumed) |
| abstract screening | **frozen** |
| retrieval | **frozen** |
| full-text screening | recomputed |
| coordinate parsing | recomputed |
| annotation | recomputed |
| output + meta | recomputed |

Three projects at their `best` tier per [run_categories.yaml](../../run_categories.yaml), chosen to
span the benchmark: **executive_function v3**, **emotion_regulation_2022 v4**, **social v3**.

### Why abstract screening is frozen, and what that cost

An earlier attempt re-ran abstract screening too. That is **incoherent with freezing the corpus**:
when abstract screening flips a study in, retrieval must fetch it, and once retrieval runs it
rebuilds its whole result. The corpus collapsed (social: 1,017 full texts → 145) and the run was
worthless.

So abstract drift was measured separately, in its own arm, before the corpus was frozen:

| project | compared | flips | rate | included before → after |
|---|---|---|---|---|
| executive_function | 4,856 | 109 | **2.24%** | 1,307 → 1,320 |
| social | 1,023 | 29 | **2.83%** | 691 → 690 |

Flips are near-symmetric, so the pool is stable even though individual decisions are not — and both
are well under the 8.6% the outline cites as precedent. Those artifacts are kept in
`<project>/abstract_replicate/`.

### What else had to be fixed to get a valid run

Two real bugs, both filed, both of which produced *plausible but wrong* corpora rather than errors:

- **[autonima#66](https://github.com/neurostuff/autonima/issues/66)** — one failing
  `full_text_source` silently discards all the others. ACE was not importable, and because all
  sources load inside a single `try`, the Elsevier directory (3,785 entries, needs no ACE) was
  discarded too. Full-text availability fell from 3,287 to 76 and the run continued and reported
  success. Fixed here by putting ACE on `PYTHONPATH`; with it, retrieval reproduces the original's
  numbers exactly (3,287 texts, 946 to screen).
- **[autonima#65](https://github.com/neurostuff/autonima/issues/65)** — an unresolvable `table_id`
  aborts the entire run rather than degrading one analysis's metadata.

**Run it with `PYTHONPATH=/home/zorro/repos/ACE`**, or retrieval silently loses most of the corpus.

---

## Results

### Stage-level instability

| stage | executive_function v3 | emotion_regulation_2022 v4 | social v3 |
|---|---|---|---|
| abstract *(frozen)* | 0.00% | 0.00% | 0.00% |
| **full-text screening** | 4.86% | **11.01%** | 7.29% |
| coordinate parsing (studies changed) | 12.2% | 5.8% | 6.5% |
| annotation | 2.42% | 2.87% | 4.50% |

**Only the full-text rate is a clean per-stage measurement.** Its input was genuinely frozen.
Parsing and annotation ran on inputs that had already drifted, so their rates are computed over the
items present in both runs and exclude items that exist in only one — 1,904/1,286 for EF
annotation, 1,704/680 for social. Report those two as end-to-end, not per-stage. Isolating them
needs separate arms (clear only `parsing`, then only `annotation`), not yet run.

### Consequence at the studyset level looks small

| | executive_function | emotion_regulation_2022 | social |
|---|---|---|---|
| studies | 822 → 822 | 580 → 578 | 530 → 529 |
| coordinates | 30,735 → 30,951 | 17,158 → 17,009 | 17,485 → 17,605 |

Under 1% either way. This is the number that misled us: it suggests the drift washes out. It does
not — it redistributes.

### Map drift between the two runs

r² on the uncorrected `z.nii.gz` over voxels finite in both; dice on the FDR-corrected map. Both
match `scripts/compare_meta_to_benchmark.py` so they are comparable to the paper's tables.

| project | manual-matched groups | mean r² | mean dice | worst r² |
|---|---|---|---|---|
| executive_function v3 | 4 | **0.961** | 0.876 | 0.916 (cognitive_flexibility) |
| emotion_regulation_2022 v4 | 4 | **0.908** | 0.805 | 0.800 (increase) |
| social v3 | 5 | **0.893** | 0.856 | 0.797 (affiliation_attachment) |
| *bypass arms* (`all_*`), all projects | 9 | 0.987 | 0.947 | — |

Three things to read off this:

**r² is consistently well above dice.** ER's `increase` is r² 0.800 against dice 0.622. The
continuous pattern is stable; what moves is where the FDR threshold lands. That is a much milder
failure mode than dice alone implies, and it is why r² is the primary metric here.

**The bypass arms barely drift** (r² 0.987). They skip analysis selection and mostly skip full-text
gating, which localises the drift to exactly the stages that were re-run. Averaging them in with
the scored groups would flatter the result, so they are reported separately.

**Instability tracks group size**, r = +0.825 between original voxel count and dice:

| | n | mean dice |
|---|---|---|
| groups < 40k voxels | 4 | 0.692 |
| groups ≥ 40k voxels | 12 | 0.864 |

Small groups swing hard — ER `increase` 19,160 → 36,675 voxels (+91%), EF `planning` 20,556 →
37,983 (+85%). One study entering or leaving moves a small map a great deal. That is the mechanism
by which a sub-1% studyset change becomes a 0.11 dice change.

### Would the paper's numbers change?

Re-scoring each retest against the manual benchmark and comparing to the original:

| project | dice orig → retest | Δ | r² orig → retest | Δ |
|---|---|---|---|---|
| emotion_regulation_2022 | 0.611 → 0.589 | −0.022 | 0.670 → 0.642 | −0.028 |
| social | 0.543 → 0.578 | +0.035 | 0.584 → 0.599 | +0.015 |
| executive_function | 0.575 → 0.617 | +0.042 | 0.643 → 0.660 | +0.018 |

Direction varies, so this is noise rather than bias. But set it against the margin over baseline:

| project | autonima | best baseline | margin | run drift | |
|---|---|---|---|---|---|
| emotion_regulation_2022 | 0.611 | 0.328 | **+0.284** | 0.022 | margin ≫ drift |
| social | 0.543 | 0.428 | **+0.115** | 0.035 | margin ≫ drift |
| executive_function | 0.575 | 0.517 | **+0.058** | 0.042 | **margin ≈ drift** |

And per column the swing is larger still: ER `increase` −0.114 dice, EF `cognitive_flexibility`
+0.117 dice, between identical runs.

**So: a per-column reproducibility floor of roughly ±0.12 dice, and a project-level floor of
±0.04.** Any reported difference smaller than that cannot be distinguished from run-to-run noise on
a single run. In §7's full table that puts decision_making (+0.001), dementia (+0.005) and
executive_function (+0.031) inside the noise band.

---

## Cost

Measured by the token accounting added for this (`usage_total` in `execution_progress.json`), not
estimated:

| project | fulltext | parsing | annotation | total |
|---|---|---|---|---|
| social v3 | $10.48 | $3.84 | $7.65 | **$21.98** |
| emotion_regulation_2022 v4 | $6.27 | $2.53 | $3.96 | **$12.76** |
| executive_function v3 | *(first segment)* | $1.22 | $11.38 | **$12.60** + ~$3.6 |
| | | | | **~$51** |

Worth recording that the pre-hoc estimate was **$28.56 against ~$51 actual** — under by roughly
1.8×, because full texts in these projects are much longer than the 13,416-token corpus median the
estimate extrapolated from. That gap is the argument for the accounting existing at all.

---

## Layout

    experiments/e2e_retest/
      README.md
      scripts/compare_retest.py        stage-by-stage decision diff + studyset consequence
      scripts/map_drift.py             per-group r2 and dice, retest maps vs original maps
      <project>/drift.json             stage diff output
      <project>/map_drift.json         map drift output
      <project>/manual_vs_auto_meta/   retest re-scored against the manual benchmark
      <project>/abstract_replicate/    the separate frozen-search abstract arm
      <project>/<run>/                 the replicate itself (retrieval/ and maps gitignored)

## Reproducing

```bash
PYTHONPATH=/home/zorro/repos/ACE autonima run projects/executive_function/v3.yaml \
  experiments/e2e_retest/executive_function/v3 \
  --clear-cache fulltext --clear-cache parsing --clear-cache annotation -j 12

PYTHONPATH=/home/zorro/repos/ACE autonima meta experiments/e2e_retest/executive_function/v3

python experiments/e2e_retest/scripts/compare_retest.py \
  --original projects/executive_function/v3 --retest experiments/e2e_retest/executive_function/v3
python experiments/e2e_retest/scripts/map_drift.py \
  --original projects/executive_function/v3 --retest experiments/e2e_retest/executive_function/v3

pixi run python scripts/compare_meta_to_benchmark.py --project-dir projects/executive_function \
  --run-dir "$PWD/experiments/e2e_retest/executive_function/v3" \
  --output-dir experiments/e2e_retest/executive_function/manual_vs_auto_meta --no-save-images
```

Stage the replicate by copying `outputs/` and `retrieval/` from the original first, and **delete
any copied `meta_analysis_results/`** — a failed meta will otherwise leave the original's maps in
place and score as perfect agreement. That happened here and was caught only because dice of
exactly 1.0000 was implausible.

`compare_meta_to_benchmark.py` needs seaborn, which is in the pixi env but not the system Python.

## Not done

- **Per-stage isolation.** Separate arms clearing only `parsing`, then only `annotation`, would
  attribute the map drift to a stage. Cheap, since full-text screening stays frozen: roughly $30
  across all three projects.
- **Replicates beyond n=2.** Everything here is a single pair, so the drift figures are a point
  estimate with no interval. A third run per project would give a range.

# Running the record-vs-full-text comparison for one project

Design for a run where the only thing that changes between projects is a single descriptor
file. Every guard below exists because its absence cost a round trip in the first five
projects; the cost is named so none of them gets dropped as ceremony.

## The one file that changes

`experiments/record_arms/arms/<project>.yaml` — nothing else is edited to add a project.

```yaml
project:      emotion_regulation_2022   # directory under projects/, and the key in
                                        # META_TO_PROJECT, which do not always agree
meta_pmid:    "35413444"                # row key in the benchmark's included_studies.csv
baseline_run: v4                        # from run_categories.yaml `best`; the arm-name stem
pondie_run:   emotion_regulation        # name of the extraction run directory
# fulltext_run: v1-A1-mini              # optional; defaults to baseline_run
# only_keys: [increase]                 # optional; defaults to every mapped column
```

Arm run names are derived, never spelled out: `<baseline_run>-record-with-evidence` and
`<baseline_run>-record-no-evidence`. Spelling them by hand is how `v1-Qfull-mini` and
`v5-gpt-record-*` ended up in the same experiment.

`fulltext_run` exists because PTSD and cue reactivity screen full text through a rehomed
`-A1-mini` variant so the text arm and the record arms share a provider. The stem stays `v1`
while the text arm is `v1-A1-mini`, which keeps the arm names derived.

The manual-to-auto column mapping is **not** in the descriptor: it is read from
`projects/<project>/nmb_mappings.json`. Two copies of a mapping is how a figure ends up
pairing the wrong maps. Note which side each name belongs to -- baseline run directories and
manual maps are keyed by the *manual* column, arm maps by the *auto* annotation key.

## Fixed locations

One root, everything under it, no path typed twice.

```
EXP     = /data/james/pondie-vs-fulltext
REPO    = $EXP/repos/autonima-results          # writable; arms run here
BENCH   = $EXP/repos/neurometabench-upstream   # OWN clone, pinned; see step 0
CORPUS  = $EXP/corpus                          # shared, keyed by pmid
MANIFEST= $EXP/pmids/<project>.cohort.csv
IDS     = $EXP/pmids/<project>.arm.tsv
RECORDS = $EXP/records_staged/<project>
RENDER  = $EXP/rendered/<project>-record-{with,no}-evidence
MIRROR  = $EXP/records_text/<project>-record-{with,no}-evidence
```

Mirror and render directories are **project-prefixed**. `--arm` names a directory under a
shared root, so an unprefixed name silently overwrites another project's mirror.

## Step 0 — preflight

Refuses to start rather than producing a result that has to be retracted.

1. **Pin the benchmark.** `git -C $BENCH rev-parse HEAD` recorded into the run log. Do not
   read the benchmark through a symlink into someone else's checkout: that is how emotion
   regulation appeared to have no gold at all (0 rows against 101 upstream) and how dementia
   was scored against a gold set that upstream had since reduced from 74 to 73.
2. **Assert the gold exists.** `meta_pmid` must have rows in
   `$BENCH/data/included_studies.csv`. Do not fall back to
   `projects/<project>/annotation-only-ids.txt` — it is a mirror that drifts (exact for PTSD,
   73 of 74 for dementia, 87 of 101 plus one non-gold paper for emotion regulation).
3. **Assert the manual maps exist**, one per `mapped_keys`, under
   `projects/<project>/reports/manual_vs_auto_meta_fair/fair_manual_meta/manual_analysis/<project>/<key>/z.nii.gz`.
4. **Assert the baseline config is clean**, reading `projects/<project>/<baseline_run>.yaml`:
   - `annotation.metadata_fields` contains `study_fulltext`. Without it the annotation stage
     never sees which document an arm read, and `_study_input_hash` omits the text hash, so a
     donated cache cannot be invalidated — all three PTSD arms produced one identical map.
   - every `model` is gateway-namespaced (`@…/model`). A bare slug in the baseline against
     namespaced arms confounds record-vs-text with provider routing, as in dementia and
     substance use, and donates no valid abstract cache.
   - `retrieval.full_text_sources` roots all exist **on this host**. Substance use's point at
     `/home/zorro/...`, which is why re-running it destroyed a retrieval cache that could not
     be rebuilt.
5. **Assert the toolchain.** `lxml` and `readabilipy` importable in pondie's venv; NiMARE and
   nilearn importable in the scoring venv.

## Step 1 — cohort manifest

```
code/build_manifest.py --project <project>   # writes $MANIFEST and $IDS
```

Emits one row per paper that reached full-text screening in the baseline, with the columns
`build_corpus.py` reads: `pmid, build_source, mirror, source_path, in_cohort` plus the
provenance fields. Route preference is **real ACE HTML → Elsevier → pubget text → ACE
text.csv export**, because the HTML carries headings and tables and the text export carries
neither; dementia's arms were measurably degraded by being built from the inferior route.

Also writes `$IDS` in the `pmid<TAB>study_id<TAB>source` form the extractor requires. A bare
id per line is silently parsed to nothing.

This script must be committed. The original manifest generator is no longer in the tree,
which is why `pmids/cohort_all.csv` could not be regenerated and had to be rebuilt by hand.

**Gate**: every row has a non-empty `build_source`; report the count that do not.

## Step 2 — corpus

```
AUTONIMA_RESULTS=$REPO code/build_corpus.sh --project <project>
```

Wraps `pondie/scripts/build_corpus.py --cohort $MANIFEST --out $CORPUS`. Note that
`--pmids` only *narrows* whatever `--cohort` already contains; passing ids without a manifest
that holds them builds nothing.

`PROJECT_RUNS` in `build_corpus.py` must contain the project, or its stage-1 coordinate
parses are silently skipped and every record reaches the extractor with no analyses
enumerated. `AUTONIMA_RESULTS` must point at a checkout that actually holds the baseline's
`coordinate_parsing_results.json`.

**Gate**: report `built N / M` and the `parsed` / `empty` split. Records built from an
`empty` tier carry no analyses; on PTSD those averaged 2.4 analyses against 4.1.

## Step 3 — extract records

```
code/extract.sh --project <project> --workers 256
```

Settings are fixed across projects so records are comparable: model
`@psyc-aid338-ope-333f18/gpt-5.6-luna`, `--effort low`, `--flavour local`,
`--service-tier flex`. These are read from the descriptor's defaults, not retyped.

**Gate**: record count equals the built count.

Never stop this with `pkill -f "pondie.cli extract"` — the pattern matches the shell holding
the connection. Kill by PID.

## Step 4 — render both variants

```
code/render_arms.sh --project <project>
```

Renders twice per variant and requires the two passes to be byte-identical before continuing.
`study_full_text_content_hash` hashes file content, so a renderer that emitted a timestamp
would re-screen every paper on each resume and read as drift.

The `--evidence full` / `--evidence none` flag is the **only** difference between the two
arms. The record-format note stays in place for both; removing it is a separate experiment.

## Step 5 — mirrors and arm configs

```
code/build_mirrors.sh --project <project>    # build_record_corpus.py, project-prefixed --arm
code/gen_configs.sh   --project <project>    # gen_arm_config.py from the baseline
```

`gen_arm_config.py` copies the baseline and rewrites only `retrieval.full_text_sources`, so
criteria, model and stage settings are inherited rather than re-authored.

**Gate**: diff each generated config against the baseline and assert the only changed keys
are under `retrieval`.

## Step 6 — run the arms

```
code/run_arms.sh --project <project> -j 8
```

Cache policy, which is where most of the damage happened:

- Donate from the baseline **only for stages upstream of full-text screening**. Screening and
  everything after it must be the arm's own.
- **Annotation must re-run per arm.** With `study_fulltext` present (step 0) the key changes
  on its own; do not additionally donate the annotation stage. Donating it gave cue's record
  arms 6–16 papers per key that the arm had rejected, and dementia's no-evidence arm 3–4.
- Arm two donates from **arm one**, not from the baseline, so the two share a cohort and the
  render is the only difference. Donating both from the baseline left substance use's arms
  diverging on 35 papers.

**Gate**: `code/audit/audit_contamination.py` must report zero contaminated keys, and zero
analyses without an annotation note, before anything is scored.

## Step 7 — maps

```
code/make_maps.sh --project <project>        # clears, then regenerates
```

`run_mkda.py` **skips any key whose `z.nii.gz` already exists**, so
`outputs/meta_analysis_results/` is deleted first. Scoring stale maps after an annotation
change cost a full round trip.

Baselines are re-estimated here too, from `projects/<project>/baselines/<key>/`, with the
same estimator. The committed `cross_project_best_baseline.csv` came from another host with an
unrecorded estimator and disagrees with ours by up to 0.66 on shared columns; subtracting it
measures the estimator, not the pipeline.

## Step 8 — score

```
code/score.sh --project <project>
```

Screening, against `$BENCH` gold, reporting **both** denominators:

- *paired*: over papers all arms screened in common — what `compare_arms.py` reports
- *end-to-end*: over the full gold set

They differ a lot. PTSD's paired recall is 0.941; end-to-end it is 16/22 = 0.727. Quote the
paired number only as a paired contrast.

Maps: Dice at z > 1.96 and brain-masked R². Dice is the headline — the repo's whole-volume R²
correlates 88–96% voxels that are zero in both maps. Keep `r2_allfinite` for continuity.

## Step 9 — figures

```
code/make_record_arm_figures.py            # figures 2 and 7
code/make_figure4_record_arms.py           # figure 4, per arm
```

Both read only `data/*.csv`, so they run anywhere the CSVs are.

## The code

`code/recordarms/` — `paths` (every location, env-overridable), `spec` (the descriptor and
derived names), `checks` (preflight and per-step status), `steps` (the eight stages, wrapping
the proven scripts rather than reimplementing them), `scoring`, `cli`.

```
python -m recordarms status                      # every project, every step
python -m recordarms status   --project <p>
python -m recordarms preflight --project <p>
python -m recordarms run-all  --project <p>      # preflight gates it; satisfied steps skip
python -m recordarms <step>   --project <p>      # manifest|corpus|extract|mirrors|
                                                 # configs|arms|maps|score
```

`run-all` skips a step whose artefacts already satisfy its check, so a re-run costs only what
is actually missing. `--force` redoes them; `--from-step` resumes.

## Adding a project: the whole checklist

1. Write `arms/<project>.yaml` (four required fields).
2. Add the project to `PROJECT_RUNS` in `build_corpus.py`.
3. `python -m recordarms run-all --project <project>`.

Everything else is derived. If a step needs a per-project exception, the exception belongs in
the descriptor, not in the script.

## Interpreting the result

Two limits hold for every project measured so far and should be stated with any number:

- **Full-text screening is the smallest loss channel.** It rejects 1–5 gold papers per
  project; search loses 3–49. An arm difference is a difference in the small channel.
- **Re-running annotation unchanged moved mean R² by 0.034.** Differences below that, or of a
  few papers, need repeated runs before they are real. PTSD's entire between-record-arm
  screening difference is one paper out of fifty.

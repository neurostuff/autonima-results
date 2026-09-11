# Record arms: remaining meta-analyses

Status of extending the with-evidence / without-evidence comparison to every benchmark.
No full-text arms — those already exist per project and are not being re-run.

## Done

| project | papers | arms | result |
|---|---|---|---|
| `vbm_of_ptsd` | 50 | full text, +evidence, −evidence | annotation refixed and re-run, see below |
| `cue_reactivity` | 550 | full text, +evidence, −evidence | committed on `record-arms` |
| `vbm_of_substance_use` | 241 | full text, +evidence, −evidence | in the metrics and figures |
| `dementia` | 448 | full text, +evidence, −evidence | in the metrics and figures |

## In flight

Nothing. `vbm_of_substance_use` (241) and `dementia` (448) both finished and are in
`reports/record_arms_meta_metrics.csv` and both figures.

## Queued, corpus buildable

Run one at a time: build corpus → extract at 256 → render both variants → build mirrors →
`gen_arm_config` → two autonima runs → `compare_screening_to_benchmark` → `compare_arms`.

| project | screened | buildable | **with stage-1 parse** | routes |
|---|---|---|---|---|
| `problem_solving` | 779 | 482 | 105 | ace_text 376, elsevier 56, pubget_text 50 |
| `decision_making` | 562 | 556 | 287 | ace_text 409, pubget_text 115, elsevier 32 |
| `social` | 676 | 673 | 316 | ace_text 570, pubget_text 86, elsevier 17 |
| `executive_function` | 993 | 942 | 463 | ace_text 895, elsevier 35, pubget_text 12 |

Estimated at the measured $0.016–0.03/paper: **≈$50–80** for the four, plus eight screening
arms. At the 985 papers/hour measured on substance-use, roughly three hours of extraction.

## Excluded

`emotional_regulation_2022` — 146 papers, text available for all of them, **zero stage-1
coordinate parses** in any of its runs. Every record would reach the extractor with no
analyses enumerated.

## What the "with parse" column means, and why it is the number to watch

`build_corpus.py` transports each paper's stage-1 parse from the baseline run's
`coordinate_parsing_results.json` rather than re-deriving it — deliberately, because
re-parsing costs a model call per table and would let the two arms disagree about how a
table splits. A paper with no parse still builds and still extracts; it just arrives with no
analyses enumerated.

That state is measured, not hypothetical. On `vbm_of_ptsd`, records built from a table-less
render carried **2.4 analyses against 4.1**, and the single genuine extraction loss in that
project — `22453299`, whose VBM contrast never entered the record — was exactly this case.
So of the 2,653 queued papers, expect roughly the 1,171 with a parse to behave like the
cue/ptsd records and the rest to be thin.

## Corpus routes added for this work

Three projects had a frozen cohort and a built corpus; these five had neither, and their
text is not where the baselines said it was.

- **`ace_text`** — builds from ACE's `processed/text.csv` (`title,abstract,body`). The saved
  journal pages cover 97 papers; this export covers 6,632, and that gap is why five projects
  had no corpus. Costs structure: no headings, so `Source:` reports "0 headings". Tables are
  unaffected — both ACE routes take them from the same export.
- **`pubget_text`** — reads `text_by_pmid/<pmid>/text.txt`. Added because **zero of 1,590**
  pubget directories on this machine carry the `article.xml` that `build_pmc` parses, so that
  route fails for every paper. No tables, by construction: pubget's XSL deletes them and
  without the XML there is nothing to rebuild from.
- `PROJECT_RUNS` extended to the five projects so their parses are visible at all. It had
  three entries; the others' parses were being silently skipped.

## Known gaps

1. **360 papers have no text on this machine** and are out of reach without a retrieval job.
2. **`build_pmc` is dead code here** — kept because the route is correct where the XML
   exists, but nothing on this host exercises it.
3. `REPO` in `build_corpus.py` was `Path(__file__).parents[2]`, which resolved to the
   autonima-results root when the script lived there and to `/home` after the move to
   pondie. It now reads `AUTONIMA_RESULTS` or defaults explicitly. The symptom was
   "baseline coordinate parses: 0 pmids" and every paper building with no analyses.

## FIXED: VBM PTSD's record arms never re-ran annotation

`projects/vbm_of_ptsd/*/outputs/annotation_results.json` is **byte-identical** (md5
`1370722a`) across `v1-A1-mini`, `v1-record-with-evidence` and `v1-record-no-evidence`.
`run_ptsd_record_arms.sh` donates both record arms' cache from `v1-A1-mini`, and unlike the
coordinate stage — whose outputs do differ per arm (11 / 10 / 10 studies) — the annotation
stage's signature matched, so it was reused wholesale.

The consequence is that all three PTSD studysets are the same 29 papers and 71 analyses, and
the three meta-analytic maps are the same file. The arms had genuinely disagreed at
screening: they included 28, 23 and 25 papers. None of that reached the map. So VBM PTSD's
`r2 = 0.4912` in `reports/record_arms_meta_metrics.csv`, repeated three times, is a cache
hit rather than a convergence, and `make_record_arm_figures.py` omits the project from
figure 7 panel b for that reason.

The other three projects re-ran annotation per arm — distinct md5s, and 100–225 analyses
differing between each arm pair — so their columns measure the arms.

**Fixed 2026-09-10.** The donation was a symptom; the cause was the config. PTSD's
`annotation.metadata_fields` listed `study_abstract` and not `study_fulltext`, and
`_study_input_hash` only folds `study_full_text_content_hash(study)` into the key when
`study_fulltext` is present. So the arm's text was neither shown to the model nor hashed,
and no donated entry could ever be invalidated by it.

Swapping `study_abstract` for `study_fulltext` in all three PTSD configs — matching
`cue_reactivity` and `dementia` — changes `stage_hash` as well, so every annotation entry
invalidated on its own and the three arms were re-run in place. Screening was untouched
(27 / 22 / 23 included, exactly as before); only annotation and the maps were rebuilt.

| | annotation md5 | decreased_gm analyses | points | contaminated |
|---|---|---|---|---|
| before, all three arms | `1370722a` | 17 | 108 | 0 |
| after, `v1-A1-mini` | `fd6ba061` | 18 | 117 | 0 |
| after, `v1-record-with-evidence` | `322086d8` | 21 | 117 | 0 |
| after, `v1-record-no-evidence` | `3db974c7` | 25 | 124 | 0 |

VBM PTSD is now in figure 7 panel b.

Backups of the pre-edit configs: `/data/james/pondie-vs-fulltext/backup_ptsd_yaml/`.

## Traced: why the studyset is not the arm's screened-in set

Three separate mechanisms, and only the third touches the maps.

**1. `export_excluded_studies: True` in all twelve runs.** `_generate_nimads_output`
([pipeline.py:1446](vendor/autonima/autonima/pipeline.py#L1446)) branches on it and exports
`[s for s in studies if s.analyses]` — screening status ignored — instead of the
`INCLUDED_FULLTEXT and s.analyses` list used when it is false. That is the whole explanation
for a 29-paper studyset over a 27-paper included set, and for 18 of those 29 being papers
the arm excluded. Nothing is inherited here; each run exports its own.

**2. Studyset membership is inherited through the coordinate-parse cache chain.** Because
membership is now "has parsed analyses", and parses are donated by
`--copy-valid-cache-from`, every run in a donation chain exports the same set. Six PTSD runs
(`v1-A1-mini`, `v1-A2-mini`, `v1-A2R-mini`, `v1-A2c-mini`, and both record arms) share the
same 29 papers while their included sets range 21–27.

**3. The maps do not use the studyset — they use the annotation keys, and *those* can be
inherited.** `AnnotationProcessor.process_studies` takes `included_studies` explicitly, so a
freshly-run annotation restricts its criteria keys to that arm's own included set. An
inherited annotation carries the *donor's* included set instead. Measured on every mapped
key in `record_arms_meta_metrics.csv`, counting papers in the key that the arm excluded:

| project | arm | contamination |
|---|---|---|
| cue_reactivity | full text | 0 / 0 / 0 |
| cue_reactivity | +evidence | 6, 7, 14 (of 158, 122, 265) |
| cue_reactivity | −evidence | 7, 9, 16 (of 160, 122, 261) |
| dementia | full text | 0 across 4 keys |
| dementia | +evidence | 0 across 4 keys |
| dementia | −evidence | 4, 3, 0, 4 (of 39, 37, 12, 31) |
| vbm_of_substance_use | all three | 0 across 6 keys |
| vbm_of_ptsd | all three | 0 |

The pattern matches the donation chains exactly. Every full-text arm is clean, because its
annotation ran alongside its own screening. Substance use is clean because its analysis sets
differed enough to change `study_input_hash`, forcing a fresh annotation. Dementia's
`+evidence` arm is clean and its `−evidence` arm is not, because `run_record_arms.sh` makes
`+evidence` the donor for `−evidence` and `−evidence` included 8 fewer papers.

### What this does and does not mean

The PTSD maps are **not** contaminated — its 8 `decreased_gm` papers survive screening in all
three arms. PTSD's identical maps are caused by mechanism (3) in its purest form: annotation
was reused wholesale, so the arms' screening differences never reached it. That is the
text-blind `metadata_fields` problem, not the studyset breadth.

Cue's record arms and dementia's `−evidence` arm **are** contaminated, at 2–6% of papers per
key. Their R² values in figure 7 panel b are computed over maps that include papers those
arms rejected.

### Fixes, applied 2026-09-10

1. **`study_fulltext` in `annotation.metadata_fields`.** Applied to `vbm_of_ptsd`.
   **Not applicable to `vbm_of_substance_use`** — see the failure note below.
2. **Contamination removed.** Deleting each affected arm's donated
   `annotation_results.json` and re-running forces annotation to recompute against that
   arm's own included set. Applied to `cue_reactivity`'s two record arms and `dementia`'s
   no-evidence arm. All twelve runs now report zero contaminated keys and zero analyses
   without a note.
3. **Outstanding, cosmetic.** `export_excluded_studies: False`. Does not change the maps.

## FAILED: the substance-use annotation cannot be made text-aware on beast

`v2.yaml` sets `retrieval.full_text_sources` to `/home/zorro/repos/autonima-results/articles/`
— paths on another host. Beast's `v2` was an imported result directory, not a locally
reproducible run. Re-running it overwrote `fulltext_retrieval_results.json` with zero
studies and it could not be rebuilt: there are no PMCIDs in this project's search results
and PMC's id converter returns 403 from this host.

Recovered by restoring `v2/outputs` from commit `a2701979`, whose vintage matches beast
exactly (234 screening rows, 142 included, 565 retrieved). Configs reverted from
`/data/james/pondie-vs-fulltext/backup_sud_yaml/`. Verified back to 142 / 565 / 2843
decisions.

Because `v2`'s articles are absent, giving only the record arms their own text would leave
the project comparing records-against-record with articles-against-abstract. All three arms
are therefore left text-blind and consistent. To fix properly, run the substance-use arms
where the article corpus lives, or repoint `full_text_sources` at a local copy.

## Measured: annotation is noisy enough to matter

Reverting the substance-use record arms meant re-running annotation under **the same
configuration** as before. Mean map R² moved:

| arm | before | after re-run | delta |
|---|---|---|---|
| `v2-record-with-evidence` | 0.6205 | 0.5867 | −0.034 |
| `v2-record-no-evidence` | 0.6160 | 0.5824 | −0.034 |

Nothing changed but the LLM's sampling. That 0.034 is larger than several of the between-arm
differences figure 7 panel b is drawn to show — cue reactivity separates its full-text and
+evidence arms by 0.002. It also flipped substance use's ordering: the record arms were above
the full-text arm and are now below it.

Panel b's between-arm differences should not be read as real unless they exceed this. The
honest next step is to repeat one project's annotation n times under a fixed config and put
an error bar on each point.

## The record format collides with the annotation stage

With the rendered record in the annotation prompt, the model frequently returns the
**record's own** analysis ids — `a_291_1`, `a_prose_1`, `a_t0010_1`,
`a_pone_0074164_t002_1` — instead of autonima's `<pmid>_analysis_<n>`. Counted per run:
230 errors in cue reactivity, 102 in substance use, 2 in dementia.

Three retries recover them (verified: zero analyses ended without a note in any run), but a
study whose three attempts all fail is dropped from annotation entirely. This is a real
interaction between the pondie record format and `annotation/prompts.py`, not sampling
noise, and it gets worse the more analyses a record enumerates. Worth either renaming the
record's ids at render time or making the prompt state that ids must be copied from the
supplied metadata block.

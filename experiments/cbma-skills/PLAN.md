# Experiment plan: skills-only CBMA versus autonima

## Hypothesis

A general-purpose coding agent with a set of skills can reproduce autonima's
workflow with accuracy that is not practically worse. It needs no specialized LLM
pipeline and no API keys. The workflow is PubMed search, screening, coordinate
extraction, analysis selection and meta-analysis.

"Specialized" means autonima's machinery:
- its prompt construction, structured-output clients and retry logic;
- its cache and signature system;
- its parallel executor.

The skills-only arm may use small generic helpers. Examples are an E-utilities
client, an HTML-to-text normalizer, a JSONL validator, and NiMARE. What it may not
use is a bespoke LLM harness. Every judgment is made by the agent.

**Outcomes that would falsify the hypothesis:**
- final-inclusion recall falls well below autonima's on the same benchmark;
- coordinate recall falls well below autonima's;
- running a benchmark-sized review is infeasible within normal subscription usage
  limits.

## Scope and phases

1. **Phase 1 (this package):** PMC open access as the full-text source; all four
   judged stages done by agents; NiMADS export; NiMARE.
2. **Phase 1b (built; exercised later):** generic local sources. Any folder of
   pre-downloaded articles (publisher HTML from Elsevier or Springer downloads,
   ACE-scraped HTML, JATS XML, plain text) becomes a full-text source.
   - It is declared in `review.yaml` with a path, a glob, and a rule for reading
     the PMID, DOI or PMCID from the filename, folder, a regex or a sidecar JSON.
   - Local files only fill in text for studies the search found, and those studies
     then go through full-text screening like any other.
   - Sources are tried in the configured order.
   - Implemented in `fulltext-sources/scripts/gather_fulltext.py` and tested with a
     local HTML folder.
3. **Phase 2 (decide after the pilot):** whether an MCP server would buy anything.
   See the last section.

## Arms

| Arm | System | Judging model |
|---|---|---|
| A | autonima v0.1.0, results already in autonima-results | as in the manuscript |
| B | cbma-skills in Claude Code | the session's model, recorded per decision |
| C | cbma-skills in Codex | the session's model, recorded per decision |

**Controls:**
- **Same skills:** arms B and C use the same commit of `experiments/cbma-skills`.
  Record the commit SHA in the run notes.
- **Same inputs:** each arm gets the same `review.yaml` criteria, translated
  one-for-one from the autonima config of that project, and the same PubMed query
  and date limits.
- **Blinding:** the agent never sees the gold standard. Keep the gold files outside
  the workspace the agent can read.

## Benchmarks

1. **autonima-results benchmark:** the projects used in the manuscript.
   - For each, translate the autonima config into `review.yaml`: objective,
     abstract and full-text criteria, and annotations as selection targets.
   - Gold: the manuscript's human-labelled includes, plus the analysis-level labels
     where available.
2. **NeuroMetaBench:** the published meta-analyses and their included studies.
   - Gold: each meta-analysis's included-study list, plus coordinates where the
     benchmark provides them.
   - Write a small adapter per benchmark to `benchmark/compare.py`'s CSV format.
     Keep it next to the benchmark data so the mapping can be reviewed.
3. **One difficult case, run first.** Pick the project where autonima did worst
   or where the criteria are hardest. The social-cognition annotation case
   (autonima#39) is the obvious candidate: analysis-level selection with fine
   contrast distinctions. Starting there tells us early whether it is worth
   running the rest.

## Protocol per project

1. **Setup:** write `review.yaml`, run `ledger.py init`, and check the numbered
   criteria against autonima's.
2. **Search:** run the search and check the count against the manuscript's. If it
   differs, record why (the date of the search, and any PubMed indexing changes
   since).
3. **Screening pilot:** screen the first 50 records only (`--limit 50`) and read
   every decision.
   - Fix unclear wording in `review.yaml` now, before the full run.
   - Record every change to `review.yaml` in the run notes. Changes after the pilot
     are protocol deviations and must be reported.
4. **Full run:** run all stages, following the orchestrator's audit checklist
   between stages.
5. **Scoring:** run
   `python benchmark/compare.py REVIEW --gold gold.csv --autonima AUTONIMA_RUN --gold-nimads gold_studyset.json --out report.json`.
6. **Repeatability:** re-run abstract screening and extraction on a fixed random
   subset in a second fresh session. Use about 200 abstracts and about 20 studies.
   - The agreement between the two sessions is the run-to-run variance.
   - Without it, a difference between arms cannot be interpreted.

## Metrics

| Stage | Metric |
|---|---|
| Search | recall of gold includes among retrieved records |
| Abstract | recall (primary: no eligible study lost), precision, workload saved |
| Full-text retrieval | coverage by source; gold includes without text |
| Full-text screening | recall, precision and F1 of final includes, over all gold and over gold with retrievable text |
| Extraction | point recall and precision against gold coordinates (±2 mm), analyses per study, share of points unverified |
| Selection | analysis-level agreement with gold labels where available |
| Meta-analysis | per target: spatial correlation and Dice overlap of thresholded maps with the gold-standard maps |
| Agreement with autonima | Cohen's kappa per stage on studies both systems judged |
| Cost | wall-clock time, subagent count, usage-limit interruptions, human interventions |
| Repeatability | session-to-session agreement on the fixed subset |

Report each with its denominator. The unavailable and incomplete counts are always
reported, never folded into the exclusions.

## Things to record in every run

- The harness and version, and the model ID. The ledger stores `--agent` on every
  decision.
- The cbma-skills commit SHA.
- The date of the PubMed search.
- The batch sizes used per stage, and any batches re-run by hand.
- Each `review.yaml` change, with its reason.

## Known risks to watch

- **Usage limits.** A 5,000-abstract review is about 200 abstract batches at 25
  each. Track where limits hit. This is a real feasibility outcome, not noise.
- **Drift in long sessions.** Mitigated by a fresh subagent per batch. If the
  orchestrator starts processing batches itself after context compaction, treat
  that as a protocol failure.
- **HTML tables.** Local publisher HTML can yield broken grids, and the coordinate
  check then drops good points. Watch `points_unverified` per source.
- **PMC coverage.** PMC open access alone will miss many included studies, so
  report recall given retrievable text separately. Phase 1b, with local sources,
  is how that gap closes.

## Would an MCP server buy anything? (decide after the pilot)

Build one only if the pilot shows a problem it would fix. Concretely:

| Observation in the pilot | What MCP would add |
|---|---|
| Subagents write outside their output files, or edit the ledger | a hard write boundary: tools are the only way to record a decision |
| The agent skips `ingest` or the audit steps, or invents hashes | server-side sequencing and validation that cannot be bypassed |
| Clients without a shell (chat apps) need to run reviews | tools usable without Bash or file access |
| The same review is run from Claude and from Codex, or by several people | one shared ledger service instead of files on one machine |
| Credentials for local publisher sources must stay out of the agent's view | secrets held by the server |

If none of these show up, the skills-plus-scripts design is enough. The ledger
already validates everything that matters; it just relies on the agent calling it.
Also note: moving the ledger into an MCP server is a thin wrapper, because
`ledger.py` already exposes these operations: init, batches, ingest, status,
needs-fulltext and export.

## First run, and the experiments that follow (2026-09-29)

**First run (emotion regulation, full-text mode, Opus 5.5 at medium effort).** Scored by
`scripts/score_cbma_review.py`; report in autonima-results
`projects/emotion_regulation_2022/reports/cbma_skills_v1/`.

**Accuracy.** Against autonima v4:
- **Full-text includes:** precision 0.49 versus 0.24, recall 0.75 versus 0.83, F1 0.59
  versus 0.37.
- **Coordinates:** gold peak recall on the same studies is 0.48 versus 0.42.
- **Maps:** Dice is within 0.03 of autonima on reappraisal, decrease and maintain, and
  better on increase. Pearson r is lower on all four targets.

**Where recall was lost.** Most of it came from the whole-brain criterion (I6/E2), applied
literally: the gold includes studies with small-volume-corrected, ROI-masked and
partial-coverage results.

**Guardrails.** The skills held: no blinding breach, no decision written outside the
ledger, and every failure retried.

**Cost.** 405M input tokens, 63% of them the orchestrator's re-read context. It hit the
five-hour usage limit twice. That led to the stage runners.

**Pre-extracted records.** Pondie records (`articles/pondie/md/<project>/`) summarize each
paper and its analyses, with coordinates. Each is about 3k tokens, against 10–15k for a
normalized full text.
- They exist for 5 projects, built from autonima's abstract passes.
- Coverage is unbiased for emotion_regulation_2022, vbm_of_ptsd and dementia. It is
  slightly biased for vbm_of_substance_use: 99% of gold covered, 87% of other candidates.
  It is strongly biased for cue_reactivity: 79% versus 46%. Score cue_reactivity only on
  candidates with a record.

**Sequence.** Each experiment changes one thing.

| | Experiment | Changes | Needs |
|---|---|---|---|
| E0 | abstract noise floor and low effort | re-screen 200 ER abstracts at low effort in a fresh session | about 2M tokens |
| E1 | records, two passes (ER, v4 criteria) | full text and selection read records; extraction is `ledger.py import-analyses` | record source, import, export of record points |
| E2 | records, one pass (ER) | one judge decides study criteria and every analysis × target; no eligible analysis means excluded | a combined ledger stage |
| E3 | whole-brain criterion (ER) | relax I6/E2 and GE1 together, in the better of E1/E2; reported as gold-tuned | none |
| E4 | the other four projects with records | "best" config tier; vbm_of_ptsd first, then vbm_of_substance_use, dementia, cue_reactivity | a scorer for projects without annotations |
| E5 | the four projects without records | decision_making, executive_function, problem_solving, social: build records first, or run in full-text mode (about 1.2B tokens) | records, or budget |
| E6 | Codex | the winning mode on ER plus one project | Codex install |

**Deferred.**
- **Re-gathering ER with the parser fixes:** only if full text stays in production.
- **A coordinate service or local model:** the records already carry points. A service
  can sit behind the same import step later.

**Before E1.** Make the transcript audit a reusable script, so every run reports tokens
and rule-following the same way.

### E1 result (2026-09-30): records, two passes, emotion regulation

Reports: autonima-results `projects/emotion_regulation_2022/reports/cbma_skills_e1_records/`
(`score.json`, `audit.json`).

**Cost.** 56M input tokens for everything after abstract screening:
- full-text judges 28M, which is 76k per study against 219k in full-text mode;
- selection judges 9M, 99k per study against 237k;
- runners 17M and the orchestrator 2M;
- extraction is free.

The first run spent about 390M on the same stages, part of it one-off debugging. There
was one usage-limit event.

**Guardrails.** The audit is clean: no blinding hit, no stray write, no decision
written outside the ledger. Two PROBLEM verdicts surfaced real protocol ambiguities,
and the user accepted both as judged:
- food-craving regulation under I2;
- pooled up + down contrasts at selection.

**Accuracy is lower than full-text mode.**

| | E1 records | first run, full text | autonima v4 |
|---|---|---|---|
| full-text precision / recall / F1 | 0.51 / 0.59 / 0.55 | 0.49 / 0.75 / 0.59 | 0.24 / 0.83 / 0.37 |
| Dice, reappraisal / decrease / increase / maintain | 0.61 / 0.63 / 0.52 / 0.43 | 0.69 / 0.67 / 0.59 / 0.47 | 0.72 / 0.70 / 0.54 / 0.51 |
| r, reappraisal / decrease / increase / maintain | 0.73 / 0.76 / 0.59 / 0.55 | 0.78 / 0.77 / 0.64 / 0.60 | 0.88 / 0.86 / 0.69 / 0.74 |

**The loss comes from the records, not the judging.**
- **24 gold studies with a record were not included,** against 10 in full-text mode.
  13 of them were included in the first run.
- **Scope labels:** records that label every analysis `roi` and omit the paper's
  supplementary whole-brain analyses. In 17133391 and 23144849 the full text says a
  whole-brain analysis was done, and the record never mentions it. This fails I6/E2.
- **Sparse records:** 5 gold records are stubs or very sparse, and were returned as
  incomplete.
- **Coverage:** 52 abstract passes have no record, 3 of them gold.
- **Studies that added nothing:** 8 of the 13 lost first-run includes had contributed
  no coordinates in the first run. Their loss lowers screening recall, but not the maps.

**Next.**
- **Upstream (Pondie):** fix record completeness (whole-brain analyses reported
  alongside ROI ones, scope labels, stubs, unresolved coordinate links) before
  records mode can match full text.
- **E3 interacts with this:** relaxing the whole-brain criterion would hide the
  scope-label defect rather than fix it.

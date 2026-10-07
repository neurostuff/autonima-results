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
| ~~E3~~ | ~~whole-brain criterion (ER)~~ | dropped (2026-10-01): the scope problem is in the record schema, and is fixed there | |
| E4 | full text, every project | "best" config tier, full-text mode, all eight remaining projects: vbm_of_ptsd, vbm_of_substance_use, dementia, cue_reactivity, decision_making, executive_function, problem_solving, social (about 1.2B tokens for the last four) | full-text sources per project |
| E5 | records, every project that has them | the same review.yaml in records mode, for vbm_of_ptsd, vbm_of_substance_use, dementia and cue_reactivity; starts from a copy of the E4 ledger after abstract screening | the E4 abstract decisions |
| E6 | Codex | full-text mode on all nine projects, with screening on GPT-6-Luna and orchestration on GPT-6-Astra | isolated Codex workspace and Codex transcript audit |

**Revised design (2026-10-01).** Every project runs in full text (E4), and every project
with Pondie records runs again in records mode (E5). The mode is then compared within
each project, and each project has four arms when the traditional harness's
`pondie-text` and `pondie-records` runs are counted. Emotion regulation already has both
skills arms: the first run (full text) and E1 (records).
- **Shared abstract decisions.** E5 copies the E4 ledger after abstract screening. The
  criteria hash is the same, so the ledger keeps those decisions and the two arms
  differ only from full-text screening onward. Record the source ledger and its commit
  in the E5 run notes.
- **Coverage rule.** Records exist only for autonima's abstract passes. Score each E5
  arm two ways: over all its abstract passes, where a study without a record counts as
  a miss, and over the studies that have a record. Report cue_reactivity only the
  second way, because its coverage is biased (79% of gold, 46% of other candidates).
- **After each run:** score with `scripts/score_cbma_review.py`, audit with
  `benchmark/audit_transcripts.py`, and record the NiMARE version and per-role effort.
  The repeatability re-run (protocol step 6) follows per project.

**E6 Codex scope (2026-10-01).** The Codex arm now covers full-text mode for all nine
projects. Full-text is the manuscript comparison; records mode remains an exploratory
deployment experiment and is not part of the planned manuscript results. The arm uses
the same skill files, snapshotted at commit
`f3a68422262350ee3ddcc36fc89f416695f10d10`, across all projects. GPT-6-Luna judges
abstract and full-text screening (low and medium effort respectively); GPT-6-Astra
handles extraction and selection (low and medium effort) and orchestration (high
effort). Each project has its own workspace under
`/home/zorro/repos/cbma-workspace-e6-codex/projects/`, with a local copy of the same
skills snapshot. These separate folders still need an OS/container boundary that hides
the gold before Codex judgments begin. The Codex CLI was present at version
`0.155.0-alpha.16.3` during setup. A separate Codex CLI log reader is now available at
`benchmark/audit_codex_transcripts.py`; it filters session logs by project workspace,
summarizes recorded model usage, and flags tool calls mentioning forbidden paths. Its
direct-write checks are preliminary and should be reviewed manually; the Claude audit
remains the more mature rules audit.

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
- **Not a criteria change (2026-10-01):** relaxing the whole-brain criterion (the old E3)
  is dropped. It would hide the scope-label defect rather than fix it.

## Notes for future skill development (2026-09-30)

### Guardrails against taking the orchestrator off route

The user was strict in the first run and E1: the criteria were never modified.
Nothing mechanical stops a user, or the orchestrator itself, from changing how items
are judged without a new `review.yaml` version. Examples:
- telling the orchestrator "be lenient on ROI studies";
- a runner or judge prompt with added guidance;
- a verbal ruling on a borderline case that shapes later batches.

The ledger hashes `review.yaml` and the skill files, but not the prompt a judge
actually received. Wanted:
- **Iterative refinement as a first-class, stoppable step.** The orchestrator should
  stop and help the user revise criteria, for example after a pilot or when a runner
  reports PROBLEM. Every revision is then a new, versioned protocol, never an
  instruction given in chat.
  - Each version is stored with its date, reason and author.
  - Every decision is tied to the version it was made under.
  - Reports show which decisions each change re-opened.
- **Only the batch file carries judging instructions.** Judges must take criteria and
  guidance from the batch file alone, and the dispatch prompt must be the fixed
  template. The transcript audit should check that every judge's first prompt matches
  the template and carries no extra guidance. It should also flag orchestrator or
  runner messages that restate or reinterpret criteria.
- **Chat rulings go into the protocol or nowhere.** A ruling the user gives in chat
  ("accept as judged") that changes future judging must become a protocol version. If
  it only settles past decisions, it is recorded as such in the run notes.

### Pondie records

E1 lost recall because of how the records present some papers: whole-brain analyses
omitted beside ROI ones, `roi` scope labels, stubs.
- **Check against the traditional harness.** Compare E1's results with the
  in-development Pondie integration in autonima's own pipeline, so that records mode is
  judged against the same inputs in both harnesses.
- **Manual review decides the fix.** The user will review the lost articles by hand
  (PLAN E1 result; `gold_with_text_not_included` in the E1 `score.json`). That decides,
  case by case, whether the fix belongs in the records or in the criteria (relax or
  adjust them). Fixing the records waits until then.

### E2 result (2026-09-30): records, one pass, emotion regulation

Reports: autonima-results `projects/emotion_regulation_2022/reports/cbma_skills_e2_combined/`.

**One pass is worse than two on both counts. Keep two passes.**

| | E2 one pass | E1 two passes | first run, full text |
|---|---|---|---|
| input tokens after abstract screening | 78.7M | 55.9M | about 390M |
| full-text precision / recall / F1 | 0.57 / 0.49 / 0.53 | 0.51 / 0.59 / 0.55 | 0.49 / 0.75 / 0.59 |
| Dice, reappraisal / decrease / increase / maintain | 0.63 / 0.63 / **0.25** / 0.44 | 0.61 / 0.63 / 0.52 / 0.43 | 0.69 / 0.67 / 0.59 / 0.47 |
| r | 0.74 / 0.76 / 0.53 / 0.56 | 0.73 / 0.76 / 0.59 / 0.55 | 0.78 / 0.77 / 0.64 / 0.60 |

- **Cost rose, not fell.** Both judges and runner grew:
  - judges went from 37M to 45M, and output from 85k to 308k, since every study's
    analyses are now judged in the same context as its screening;
  - the one runner went from 17M to 32M, over 122 batches plus retries in one long
    context.

  Selection on records was already cheap in E1, so merging saved little and cost
  context.
- **Recall fell.** 34 studies passed screening but had no eligible analysis, and were
  excluded by rule. 9 of them met every criterion but have empty records (for example
  15488398, whose text reports whole-brain decrease > look). The `increase` target shrank
  to 12 studies, and its map collapsed (Dice 0.25).
- **Guardrail finding.** Under retry pressure, the stage runner added a line to 2 judge
  prompts ("make sure every criterion, including E5, has an entry"). Nothing asked for
  it. The orchestrator caught it and logged it as a deviation, and the new audit check
  (`judge_prompts_with_added_guidance`) finds exactly those 2 prompts. There are none in
  E1 or the first run. The runner's instructions now fix the prompt for every round.
  Judges intermittently dropping E5 is a separate finding, which retries with the same
  prompt should absorb.

### E0 result (2026-09-30): token reductions, and the noise floor

Reports: autonima-results `projects/emotion_regulation_2022/reports/cbma_skills_e0_tokens/`.

**Setup.** A copy of the first run (full-text mode). These were held out and re-judged
with the new setup: text bundles, slim judges, `run_stage.py` dispatch, and per-stage
effort. Other things held equal: the same documents, criteria and default batch sizes.

| judge stage | effort | turns per judge | input per unit | first run | agreement with first run |
|---|---|---|---|---|---|
| abstract (200) | low | 3.0 | 3.0k per record | 8.5k | 189/200 pass or exclude |
| full text (20) | medium | 6.5 | 112k per study | 219k | 19/20 |
| extraction (20) | low | 4.1 | 88k per study | 209k | 755/755 points identical, same 109 analyses |
| selection (20) | medium | 4.0 | 74k per study | 237k | 512/516 analysis × target |

**Coordination.** It nearly vanished: 0.3–0.4M per stage runner, against 5–12M in E1,
and 0.55M for the orchestrator. Judges start at 7k tokens of context, against 28.6k.

**Scaled up.** A full emotion regulation run in full-text mode would come to roughly 70M
input tokens, against 405M for the first run. Records mode would be less.

**Noise floor.** A medium-effort re-run of the same 200 abstracts agreed with the first
run on 191/200 pass or exclude calls, and the low-effort run on 189/200. Neither dropped
a gold study. Low leaned slightly toward excluding: 8 passes lost against 3.

**Found on the way.**
- **Agent frontmatter doesn't reach headless sessions.** A headless `claude -p --agent`
  session ignores the agent's `effort:` and `skills:`. `run_stage.py` now passes
  `--effort` and `--append-system-prompt-file` explicitly.
- **Mislabelled effort.** 200 smoke-test decisions ran at medium while recorded as
  effort-low. They are kept apart, as the medium noise-floor sample.

**Recurring criteria gaps.** Every run's runners raise the same two: food-craving
regulation under I2, and how to label pooled up + down contrasts. Both belong in a
versioned `review.yaml` revision, not in further per-run rulings.

### Draft criteria revision (2026-09-30, not applied)

The user chose four fixes for the gaps the stage runners kept flagging:
1. craving and appetitive regulation is excluded;
2. goal labels require reappraisal as the strategy;
3. pooled up + down contrasts carry reappraisal only;
4. reappraisal induced by experimenter framing counts as instructed.

The same text is drafted in both harnesses:
- `projects/emotion_regulation_2022/drafts/v5.yaml`: autonima, traditional schema, v4
  plus appended lines;
- `experiments/cbma-skills/reviews/emotion_regulation_2022/review.v2.draft.yaml`:
  cbma-skills.

`check_cbma_translation.py` confirms the two match. The draft is parked: improving the
criteria is not the current aim, which is running the other projects against the
original harness. Applying it later needs the recorded-revision tooling
(`ledger.py revise`, see the guardrail notes). It reopens every abstract, full-text and
selection decision of an emotion regulation run.

## Correction, and why the skills maps trail autonima's (2026-09-30)

**Scoring bug (fixed).** `scripts/score_cbma_review.py` derived a cbma map's z from p
as `isf(p)`. NiMARE's MKDA z maps, and the gold and autonima maps, are clipped at 0.
The derivation therefore put negative z where 0.5 < p < 1, and -inf where p = 1. The
-inf also dropped those voxels, about a fifth of the brain, from the shared mask for
every arm.
- **Effect:** Dice was essentially unaffected, but every Pearson r reported for the
  skills runs was too low. First run, corrected: reappraisal 0.847 (was 0.761),
  decrease 0.836, increase 0.674, maintain 0.637, against autonima v4's 0.888, 0.863,
  0.664 and 0.745.
- **Superseded:** the r values in the E1 and E2 tables above. The re-scored reports are
  in `projects/emotion_regulation_2022/reports/`.

**Same records, different judges** (E1 against autonima's `v4-pondie-records`, the
user's evaluation). With inputs held equal, the skills maps still trail: reappraisal
Dice 0.61 against 0.71, maintain r 0.59 against 0.76. Counterfactual maps, which add
or remove one thing at a time (scripts beside that report), locate the gap.

- **Reverse contrasts** ("Look > Reappraise", "Maintain > Decrease") are the largest
  single cause.
  - **Where each arm puts them:** skills judges put them in the regulation targets
    (12% of reappraisal and 11% of decrease selections), following the global guidance
    "either an ACTIVATION or a DEACTIVATION effect qualifies". Autonima's judges put
    them in maintain (14%).
  - **The gold uses none,** under any label.
  - **Effect of removing them** from E1: reappraisal Dice 0.606 to 0.663 and r 0.798
    to 0.854; decrease 0.633 to 0.683 and 0.816 to 0.853. That is most of the way to
    autonima (0.709 and 0.873; 0.700 and 0.859). The same criteria text was read
    differently by the two models.
- **Maintain** is limited by the records.
  - **Missing contrasts:** for 17 of the 19 gold maintain studies that E1 included
    without a maintain label, the record has no unregulated contrast ("Emotion >
    Baseline"), which is what the gold uses.
  - **Why autonima's map scores better:** its labelling of reverse contrasts as maintain
    helps its map (E1 plus its reverse contrasts: r 0.585 to 0.683), although the gold
    never labels them so.
  - **The real fix is upstream:** records should capture unregulated contrasts.
- **Study recall at full text** is a smaller cause. For reappraisal, adding autonima's
  picks from the 10 gold studies E1 excluded at full text (I4, I5, I6) raises r 0.798 to
  0.828.
- **Autonima's extra non-gold studies do not matter.** Removing them leaves its map
  unchanged (r 0.873).

**NiMARE versions differ across arms:** 0.16.0 for the gold maps, 0.2.1 for autonima,
0.21.0 for cbma-skills. Refitting autonima's selection with 0.21 gives r 0.873, against
0.869 from its saved map, so the version effect looks small. Pin one version for future
comparisons.

**Candidate rule for the parked draft revision:** a regulation target's contrast has
the regulation condition as its active condition (regulate > comparison). A reverse
contrast (comparison > regulate) belongs to no regulation target. This matches the
paper's Dec and Inc categories and the gold. It conflicts with the global
"deactivation qualifies" guidance, which the revision would have to reword.

## Potential improvements (backlog, 2026-09-30)

Collected from the emotion regulation runs. None is applied yet. Criteria items belong
in a recorded protocol revision (the parked draft); package items are code changes.

**Criteria (map accuracy)**
1. **Reverse-contrast rule.** A regulation target's contrast has the regulation
   condition as its active condition; a reverse contrast (look > regulate) belongs to no
   regulation target. This means rewording the global "either an ACTIVATION or a
   DEACTIVATION effect qualifies" guidance. It is the largest measured gain: E1
   reappraisal r 0.798 to 0.854, and decrease 0.816 to 0.853.
2. **The four parked choices:** craving excluded, goal labels require reappraisal,
   pooled goals carry reappraisal only, experimenter framing counts.
3. **Full-text strictness** (I4 sample, I5 contrast, I6 whole-brain), decided by the
   user's manual review of the lost gold studies. A smaller gain: +0.03 r on
   reappraisal.
4. **Word guidance so that it cannot be read two ways.** Claude and GPT read the same
   "deactivation qualifies" line in opposite ways.

**Inputs**
5. **Records should capture unregulated contrasts** ("Emotion > Baseline", look >
   baseline). 17 of 19 missed gold maintain studies lack them. This is upstream, in
   Pondie.
6. **Records completeness:** whole-brain analyses reported beside ROI ones, scope labels,
   stubs, unresolved coordinate links. See memory "Pondie record coordinate gaps".
7. **Full-text mode: recover tables that publisher pages link out** (ACE pages), or fall
   back to ACE's or autonima's parsed coordinates.

**Scoring and comparability**
8. **Pin one NiMARE version** across the gold, autonima and cbma maps (now 0.16, 0.2.1
   and 0.21).
9. **Score analysis-level selection** against the gold labels, in `score_cbma_review.py`
   (so far done ad hoc by coordinate matching).

**Tokens**
10. **Bigger batches for records:** full text 10–15, selection 6. Check on the
    noise-floor sample.
11. **Skip studies whose record has no analysis with points,** if maps are the goal.
    About 30% of full-text judging in records mode; PRISMA completeness suffers.
12. **Abstract batches of 50,** shorter reason limits, and smaller models (Haiku or
    Sonnet) for abstract and extraction. Haiku judges for every stage are being tested
    end to end on vbm_of_ptsd (2026-10-01: `~/repos/cbma-workspaces/haiku/`, orchestrator
    and runners on Opus).
16. **Less text per full-text paper.** Implemented 2026-10-01 as
    `screening.fulltext.text: trimmed` (off by default; see review-spec "Trimmed full
    text"): abstract, methods in full, results headings with their first paragraph, all
    tables. A median 49% of the text survives; 4% of documents fall back to full text.
    Next: an arm that tests it against full-text decisions. Original note: Full-text judges read two
    whole papers each; on Opus that made a 1,291-paper queue (executive_function)
    unaffordable. Most criteria an abstract leaves unclear are about the method (task
    fMRI of the right paradigm, whole-brain VBM), which the methods section alone answers.
    Options: send methods plus tables only, or a cheap check of just the criterion the
    abstract left unclear. Changes what judges see, so it needs its own E0-style
    agreement test before use.
17. **Abstract-uncertain studies are half the full-text work for a tenth of the
    includes** (2026-10-01, five Opus runs): 811 of 1,736 full-text judgements, 84 of 567
    includes, 34 of 266 gold kept at full text. Outside dementia only 7 gold came this way;
    in dementia 27 of 58, because its abstracts omit diagnosis details and whether the
    analysis was VBM. "Uncertain" is derived from the criteria (an inclusion criterion
    `unclear`, nothing failed), not from judge confidence, and missing abstracts explain
    almost none of it (18 of 950, none later included). So a project-wide rule to drop
    uncertain studies would cost dementia-like projects real recall; route them through a
    cheaper path instead (Haiku judges, or item 16).

**Guardrails and provenance**
13. **`ledger.py revise`:** versioned protocol snapshots with reason and author, a
    reopen preview, and a refusal to batch when `review.yaml` does not match the last
    recorded version.
14. **Craving and pooled-contrast rulings** keep stopping end-to-end runs. Settle them
    in the revision.
15. **Record the NiMARE version and the per-role effort** in the run notes
    automatically (the audit reports effort already).

## Cross-arm results, full-text mode (2026-10-05)

Workspace folders were renamed on 2026-10-06 to `~/repos/cbma-workspaces/{orchestrator}-{judges}`:
Opus = `opus-opus` (frozen, not being finished for cost; results kept), Haiku = `opus-haiku`,
Astra = `astra-astra`, Sol = `sol-luna5.6`; `sol-luna6` is the Codex Sol / Luna 6 arm, and
Astra / Luna 6 is archived (`_archive/astra-luna6`, not being finished). Old names are symlinks,
so the paths below still resolve.

Every run scored with `scripts/score_cbma_review.py` against the gold standard and the
autonima run its protocol translates (cue_reactivity v6, problem_solving v2, vbm_of_ptsd v1,
vbm_of_substance_use v2, emotion_regulation_2022 v4, the others v3). Reports:
`projects/<p>/reports/cbma_skills_{v1,haiku,delta_astra,delta_sol}/`.

Arms:
- **Opus** (`cbma-workspaces/main`): Claude Code, orchestrator, runners and judges on
  claude-opus-5-5 (abstract and extraction low, full text and selection medium).
- **Haiku** (`cbma-workspaces/haiku`): orchestrator and runners Opus, every judge
  claude-haiku-4-5 (no effort setting).
- **Astra** (Delta run of the `e6-codex-frontier-macos` bundle): Codex, gpt-6-astra max for
  orchestration, screening and selection; gpt-5.6-luna low for extraction.
- **Sol** (same bundle, `luna/`): gpt-6.1-sol medium orchestrates; every judge gpt-5.6-luna
  (low, medium, low, medium). Delta runs are scored with `--decisions-as-recorded` (no `docs/`
  in the package) and `--meta-dir` for their subset-fit maps; see the bundle's `IMPORT_NOTES.md`.

Selection P / R and map r are means over the run's scored targets; the target sets can differ
between arms (Astra substance use has 2 maps, nicotine fell below 10 studies).

| project | arm | screen P | R | F1 | autonima F1 | peak recall (autonima) | sel P / R | maps | map r | autonima r |
|---|---|---|---|---|---|---|---|---|---|---|
| cue_reactivity | Astra | 0.31 | 0.71 | 0.43 | 0.38 | 0.57 (0.56) | 0.34 / 0.61 | 3 | 0.79 | 0.75 |
| decision_making | Astra | 0.37 | 0.38 | 0.37 | 0.31 | 0.77 (0.77) | 0.17 / 0.19 | 3 | 0.50 | 0.57 |
| decision_making | Sol | 0.32 | 0.49 | 0.39 | 0.31 | 0.81 (0.81) | 0.22 / 0.29 | 3 | 0.52 | 0.57 |
| dementia | Astra | 0.47 | 0.76 | 0.58 | 0.55 | 0.57 (0.27) | 0.26 / 0.31 | 4 | 0.56 | 0.54 |
| dementia | Sol | 0.49 | 0.76 | 0.60 | 0.55 | 0.35 (0.25) | 0.22 / 0.25 | 4 | 0.50 | 0.54 |
| emotion_regulation_2022 | Astra | 0.42 | 0.72 | 0.53 | 0.37 | 0.56 (0.49) | 0.56 / 0.49 | 4 | 0.76 | 0.79 |
| emotion_regulation_2022 | Opus | 0.49 | 0.75 | 0.59 | 0.37 | 0.48 (0.42) | 0.58 / 0.46 | 4 | 0.75 | 0.79 |
| executive_function | Astra | 0.10 | 0.42 | 0.16 | 0.15 | 0.30 (0.23) | 0.15 / 0.28 | 4 | 0.73 | 0.78 |
| executive_function | Sol | 0.09 | 0.33 | 0.15 | 0.15 | 0.37 (0.33) | 0.19 / 0.25 | 4 | 0.74 | 0.78 |
| problem_solving | Astra | 0.25 | 0.59 | 0.35 | 0.37 | 0.57 (0.50) | 0.26 / 0.51 | 5 | 0.79 | 0.80 |
| social | Astra | 0.45 | 0.83 | 0.58 | 0.57 | 0.64 (0.61) | 0.43 / 0.43 | 5 | 0.71 | 0.73 |
| social | Sol | 0.42 | 0.82 | 0.55 | 0.57 | 0.76 (0.62) | 0.40 / 0.51 | 5 | 0.73 | 0.73 |
| vbm_of_ptsd | Astra | 0.60 | 0.57 | 0.59 | 0.67 | 0.94 (0.65) | 0.67 / 0.36 | 1 | 0.75 | 0.76 |
| vbm_of_ptsd | Opus | 0.57 | 0.57 | 0.57 | 0.67 | 0.67 (0.67) | 0.70 / 0.32 | 1 | 0.75 | 0.76 |
| vbm_of_ptsd | Haiku | 0.52 | 0.71 | 0.60 | 0.67 | 0.60 (0.60) | 0.70 / 0.32 | 1 | 0.75 | 0.76 |
| vbm_of_substance_use | Astra | 0.48 | 0.75 | 0.59 | 0.56 | 0.75 (0.73) | 0.79 / 0.45 | 2 | 0.75 | 0.69 |
| vbm_of_substance_use | Opus | 0.52 | 0.73 | 0.61 | 0.56 | 0.73 (0.74) | 0.82 / 0.58 | 3 | 0.73 | 0.64 |
| vbm_of_substance_use | Haiku | 0.40 | 0.77 | 0.53 | 0.56 | 0.71 (0.73) | 0.74 / 0.54 | 3 | 0.65 | 0.64 |

Findings:
- **Cheap judges match expensive ones on the Codex side.** Sol (Luna judges) and Astra (Astra
  max) are within 0.03 screening F1 and 0.06 map r on all four shared projects.
- **Haiku judges match Opus on a simple selection (PTSD) but not a crowded one (substance use):**
  Haiku packed overlapping samples and correlation analyses into the targets (alcohol: 29
  analyses from 18 studies against Opus's 15 from 15), so alcohol map r fell 0.80 to 0.66. Judge
  cost was about half of Opus's in both (substance use $53 against $108; PTSD $7 against $14).
  Candidate next arm: Haiku for abstract, full text and extraction, Opus for selection.
- **Against autonima,** screening F1 is as good or better in 6 of 9 projects (emotion regulation
  most: 0.53-0.59 against 0.37), and map r is within about ±0.07: higher in cue reactivity and
  substance use, lower in decision making (0.50 against 0.57) and executive function (0.73
  against 0.78).
- **Codex Astra and Claude Opus agree closely** on the three projects both ran.
- **The hard projects stay hard in every arm:** decision-making selection P/R about 0.2;
  executive-function screening F1 about 0.15 for autonima and every skills arm.
- Social maps are scored here for the first time in any arm (the gold folders are spelled
  `ALL-Merged` and so on; the scorer now matches them).

Caveats: Sol executive_function ran an amended full-text criterion (a protocol deviation,
recorded in the bundle); peak recall is over studies shared with gold, so it moves with the
study set (Astra PTSD's 0.94 is over 12 studies); the Delta runs have no transcripts, so no
blinding or token audit; most Delta reports note no formal stage audit. Not yet scored: Haiku
emotion regulation; the Codex Luna 6 runs (`e6-codex-luna`). The original `e6-codex` runs
(Luna screening, Astra extraction and selection) match none of the compared arms and were
archived on 2026-10-05 (`~/repos/cbma-workspaces/_archive/e6-codex`).

### Update (2026-10-06): seven more runs scored

Added Opus/Haiku cue_reactivity, dementia, emotion_regulation_2022 and problem_solving, and the
archived Codex Astra/Luna 6 runs (emotion_regulation_2022, vbm_of_ptsd, vbm_of_substance_use;
`_archive/astra-luna6`). Arm labels are orchestrator/judges. Reports:
`projects/<p>/reports/cbma_skills_{haiku,astra_luna6}/`. Astra/Luna 6 PTSD fitted no map (decreased GM
had 9 studies) and its substance-use nicotine target had 8, so their map means cover fewer targets.

| project | arm | screen P | R | F1 | autonima F1 | peak recall (aut) | sel P / R (mean over targets) | maps | map r | autonima r |
|---|---|---|---|---|---|---|---|---|---|---|
| cue_reactivity | Opus/Haiku | 0.26 | 0.81 | 0.39 | 0.38 | 0.56 (0.54) | 0.32 / 0.63 | 3 | 0.76 | 0.75 |
| cue_reactivity | Astra/Astra | 0.31 | 0.71 | 0.43 | 0.38 | 0.57 (0.56) | 0.34 / 0.61 | 3 | 0.79 | 0.75 |
| decision_making | Astra/Astra | 0.37 | 0.38 | 0.37 | 0.31 | 0.77 (0.77) | 0.17 / 0.19 | 3 | 0.50 | 0.57 |
| decision_making | Sol/Luna5.6 | 0.32 | 0.49 | 0.39 | 0.31 | 0.81 (0.81) | 0.22 / 0.29 | 3 | 0.52 | 0.57 |
| dementia | Opus/Haiku | 0.35 | 0.72 | 0.47 | 0.55 | 0.28 (0.19) | 0.17 / 0.18 | 4 | 0.44 | 0.54 |
| dementia | Astra/Astra | 0.47 | 0.76 | 0.58 | 0.55 | 0.57 (0.27) | 0.26 / 0.31 | 4 | 0.56 | 0.54 |
| dementia | Sol/Luna5.6 | 0.49 | 0.76 | 0.60 | 0.55 | 0.35 (0.25) | 0.22 / 0.25 | 4 | 0.50 | 0.54 |
| emotion_regulation_2022 | Opus/Opus | 0.49 | 0.75 | 0.59 | 0.37 | 0.48 (0.42) | 0.58 / 0.46 | 4 | 0.75 | 0.79 |
| emotion_regulation_2022 | Opus/Haiku | 0.27 | 0.81 | 0.41 | 0.37 | 0.41 (0.42) | 0.47 / 0.48 | 4 | 0.79 | 0.79 |
| emotion_regulation_2022 | Astra/Astra | 0.42 | 0.72 | 0.53 | 0.37 | 0.56 (0.49) | 0.56 / 0.49 | 4 | 0.76 | 0.79 |
| emotion_regulation_2022 | Astra/Luna6 | 0.33 | 0.68 | 0.44 | 0.37 | 0.56 (0.52) | 0.50 / 0.42 | 4 | 0.73 | 0.79 |
| executive_function | Astra/Astra | 0.10 | 0.42 | 0.16 | 0.15 | 0.30 (0.23) | 0.15 / 0.28 | 4 | 0.73 | 0.78 |
| executive_function | Sol/Luna5.6 | 0.09 | 0.33 | 0.15 | 0.15 | 0.37 (0.33) | 0.19 / 0.25 | 4 | 0.74 | 0.78 |
| problem_solving | Opus/Haiku | 0.28 | 0.52 | 0.36 | 0.37 | 0.53 (0.54) | 0.29 / 0.40 | 5 | 0.75 | 0.80 |
| problem_solving | Astra/Astra | 0.25 | 0.59 | 0.35 | 0.37 | 0.57 (0.50) | 0.26 / 0.51 | 5 | 0.79 | 0.80 |
| social | Astra/Astra | 0.45 | 0.83 | 0.58 | 0.57 | 0.64 (0.61) | 0.43 / 0.43 | 5 | 0.71 | 0.73 |
| social | Sol/Luna5.6 | 0.42 | 0.82 | 0.55 | 0.57 | 0.76 (0.62) | 0.40 / 0.51 | 5 | 0.73 | 0.73 |
| vbm_of_ptsd | Opus/Opus | 0.57 | 0.57 | 0.57 | 0.67 | 0.67 (0.67) | 0.70 / 0.32 | 1 | 0.75 | 0.76 |
| vbm_of_ptsd | Opus/Haiku | 0.52 | 0.71 | 0.60 | 0.67 | 0.60 (0.60) | 0.70 / 0.32 | 1 | 0.75 | 0.76 |
| vbm_of_ptsd | Astra/Astra | 0.60 | 0.57 | 0.59 | 0.67 | 0.94 (0.65) | 0.67 / 0.36 | 1 | 0.75 | 0.76 |
| vbm_of_ptsd | Astra/Luna6 | 0.62 | 0.71 | 0.67 | 0.67 | 0.60 (0.60) | 0.67 / 0.27 | 0 | - | - |
| vbm_of_substance_use | Opus/Opus | 0.52 | 0.73 | 0.61 | 0.56 | 0.73 (0.74) | 0.82 / 0.58 | 3 | 0.73 | 0.64 |
| vbm_of_substance_use | Opus/Haiku | 0.40 | 0.77 | 0.53 | 0.56 | 0.71 (0.73) | 0.74 / 0.54 | 3 | 0.65 | 0.64 |
| vbm_of_substance_use | Astra/Astra | 0.48 | 0.75 | 0.59 | 0.56 | 0.75 (0.73) | 0.79 / 0.45 | 2 | 0.75 | 0.69 |
| vbm_of_substance_use | Astra/Luna6 | 0.50 | 0.70 | 0.59 | 0.56 | 0.70 (0.74) | 0.80 / 0.42 | 2 | 0.70 | 0.69 |

- **Haiku judges keep map quality but lose screening precision where screening is hard.** Map r
  matches or exceeds autonima on cue reactivity (0.76 vs 0.75) and emotion regulation (0.79 vs
  0.79), but screening precision drops (emotion regulation 0.27 against Opus 0.49; cue reactivity
  0.26 against Astra 0.31) as recall rises.
- **Dementia is Haiku's weak project:** screening F1 0.47 against 0.58-0.60 for the Codex arms and
  0.55 for autonima, selection P/R 0.17/0.18, and map r 0.44 against 0.50-0.56.
- **Astra/Luna 6** is close to Astra/Astra on emotion regulation and substance use, and gave the
  best PTSD screening F1 of any arm (0.67, matching autonima).

### Update (2026-10-07): one unknown-space policy for every arm

The earlier tables mixed two treatments of coordinates whose space is unknown. NiMARE 0.21.0
fits null-space points as MNI without saying so (it labels them "UNKNOWN", applies no transform
and relabels them as the dataset's space). Every Claude arm, the archived Astra/Luna 6 runs and
autonima reached NiMARE that way. The Delta Codex runs (Astra/Astra, Sol/Luna 5.6) instead left
unknown-space analyses out of separate fitting copies. `run_meta.py` now takes an explicit
`--unknown-space mni|exclude` (default mni, recorded in each summary.json), and the 13 Delta
reviews were refitted from their full exports with unknown spaces as MNI
(`results/meta_unknown_mni`; reports `cbma_skills_delta_{astra,sol}_mni`). This table uses those
refits, so every row fits unknown spaces as MNI.

| project | arm | screen P | R | F1 | autonima F1 | peak recall (aut) | sel P / R | maps | map r | autonima r |
|---|---|---|---|---|---|---|---|---|---|---|
| cue_reactivity | Opus/Haiku | 0.26 | 0.81 | 0.39 | 0.38 | 0.56 (0.54) | 0.32 / 0.63 | 3 | 0.76 | 0.75 |
| cue_reactivity | Astra/Astra | 0.31 | 0.71 | 0.43 | 0.38 | 0.57 (0.56) | 0.34 / 0.61 | 3 | 0.79 | 0.75 |
| decision_making | Astra/Astra | 0.37 | 0.38 | 0.37 | 0.31 | 0.77 (0.77) | 0.17 / 0.19 | 3 | 0.50 | 0.57 |
| decision_making | Sol/Luna5.6 | 0.32 | 0.49 | 0.39 | 0.31 | 0.81 (0.81) | 0.22 / 0.29 | 3 | 0.53 | 0.57 |
| dementia | Opus/Haiku | 0.35 | 0.72 | 0.47 | 0.55 | 0.28 (0.19) | 0.17 / 0.18 | 4 | 0.44 | 0.54 |
| dementia | Astra/Astra | 0.47 | 0.76 | 0.58 | 0.55 | 0.57 (0.27) | 0.26 / 0.31 | 4 | 0.60 | 0.54 |
| dementia | Sol/Luna5.6 | 0.49 | 0.76 | 0.60 | 0.55 | 0.35 (0.25) | 0.22 / 0.25 | 4 | 0.54 | 0.54 |
| emotion_regulation_2022 | Opus/Opus | 0.49 | 0.75 | 0.59 | 0.37 | 0.48 (0.42) | 0.58 / 0.46 | 4 | 0.75 | 0.79 |
| emotion_regulation_2022 | Opus/Haiku | 0.27 | 0.81 | 0.41 | 0.37 | 0.41 (0.42) | 0.47 / 0.48 | 4 | 0.79 | 0.79 |
| emotion_regulation_2022 | Astra/Astra | 0.42 | 0.72 | 0.53 | 0.37 | 0.56 (0.49) | 0.56 / 0.49 | 4 | 0.76 | 0.79 |
| emotion_regulation_2022 | Astra/Luna6 | 0.33 | 0.68 | 0.44 | 0.37 | 0.56 (0.52) | 0.50 / 0.42 | 4 | 0.73 | 0.79 |
| executive_function | Astra/Astra | 0.10 | 0.42 | 0.16 | 0.15 | 0.30 (0.23) | 0.15 / 0.28 | 4 | 0.74 | 0.78 |
| executive_function | Sol/Luna5.6 | 0.09 | 0.33 | 0.15 | 0.15 | 0.37 (0.33) | 0.19 / 0.25 | 4 | 0.74 | 0.78 |
| problem_solving | Opus/Haiku | 0.28 | 0.52 | 0.36 | 0.37 | 0.53 (0.54) | 0.29 / 0.40 | 5 | 0.75 | 0.80 |
| problem_solving | Astra/Astra | 0.25 | 0.59 | 0.35 | 0.37 | 0.57 (0.50) | 0.26 / 0.51 | 5 | 0.79 | 0.80 |
| social | Astra/Astra | 0.45 | 0.83 | 0.58 | 0.57 | 0.64 (0.61) | 0.43 / 0.43 | 5 | 0.72 | 0.73 |
| social | Sol/Luna5.6 | 0.42 | 0.82 | 0.55 | 0.57 | 0.76 (0.62) | 0.40 / 0.51 | 5 | 0.74 | 0.73 |
| vbm_of_ptsd | Opus/Opus | 0.57 | 0.57 | 0.57 | 0.67 | 0.67 (0.67) | 0.70 / 0.32 | 1 | 0.75 | 0.76 |
| vbm_of_ptsd | Opus/Haiku | 0.52 | 0.71 | 0.60 | 0.67 | 0.60 (0.60) | 0.70 / 0.32 | 1 | 0.75 | 0.76 |
| vbm_of_ptsd | Astra/Astra | 0.60 | 0.57 | 0.59 | 0.67 | 0.94 (0.65) | 0.67 / 0.36 | 1 | 0.74 | 0.76 |
| vbm_of_ptsd | Astra/Luna6 | 0.62 | 0.71 | 0.67 | 0.67 | 0.60 (0.60) | 0.67 / 0.27 | 0 | - | - |
| vbm_of_substance_use | Opus/Opus | 0.52 | 0.73 | 0.61 | 0.56 | 0.73 (0.74) | 0.82 / 0.58 | 3 | 0.73 | 0.64 |
| vbm_of_substance_use | Opus/Haiku | 0.40 | 0.77 | 0.53 | 0.56 | 0.71 (0.73) | 0.74 / 0.54 | 3 | 0.65 | 0.64 |
| vbm_of_substance_use | Astra/Astra | 0.48 | 0.75 | 0.59 | 0.56 | 0.75 (0.73) | 0.79 / 0.45 | 2 | 0.75 | 0.69 |
| vbm_of_substance_use | Astra/Luna6 | 0.50 | 0.70 | 0.59 | 0.56 | 0.70 (0.74) | 0.80 / 0.42 | 2 | 0.70 | 0.69 |

- **The policy matters only for dementia:** mean map r rose by 0.041 in both Delta arms (Astra
  0.556 to 0.597, now above autonima's 0.538; Sol 0.495 to 0.536, level with it). Every other
  project moved by at most 0.012, including executive function, which has the most unknown-space
  points (about 1,000); excluding them was not why its maps trail autonima.
- Screening, peak recall and selection columns are unchanged; only the map columns differ from
  the earlier Delta rows.

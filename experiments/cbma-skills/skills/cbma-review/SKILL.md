---
name: cbma-review
description: Run a coordinate-based neuroimaging meta-analysis as a systematic review, from a PubMed query to NiMARE maps, with the agent itself doing the abstract screening, full-text screening, coordinate-table extraction and analysis selection. Use when the user asks to run, resume, pilot or check the status of a neuroimaging meta-analysis or systematic review (fMRI/PET activation coordinates, PRISMA, NiMADS, NiMARE), or points at a folder containing a review.yaml.
---

# CBMA review (orchestrator)

You are running a systematic review whose output is a coordinate-based
meta-analysis (CBMA). You do the judgment work yourself, through subagents. Small
bundled scripts do the bookkeeping. No LLM API keys are involved: every judgment is
made by you or by a subagent you start.

This skill is the entry point. It tells you the order of stages and the rules. Each
stage has its own skill with the detailed instructions:

| Stage | Skill | Who does it |
|---|---|---|
| 1. Search | `pubmed-search` | a script |
| 2. Abstract screening | `screen-studies` | subagents, in batches |
| 3. Full-text retrieval | `fulltext-sources` | a script (PMC, then any local folders) |
| 4. Full-text screening | `screen-studies` | subagents, in batches |
| 5. Coordinate extraction | `extract-coordinates` | subagents, one study per item |
| 6. Analysis selection | `select-analyses` | subagents, one study per item |
| 7. Meta-analysis | `cbma-nimare` | a script (NiMARE) |

`SKILLS` below means the directory that holds these skill folders (the parent of
this file's directory). `REVIEW` means the review folder, the one with
`review.yaml`.

## Non-negotiable rules

1. **The ledger is the only writer of decisions.** Subagents write their raw output
   to the `output` path named in their batch file, and nowhere else. You then run
   `ledger.py ingest`, which validates it and records it. Never edit anything under
   `REVIEW/decisions/` or `REVIEW/analyses/` by hand, and never write decisions
   yourself outside a batch.
2. **A failure is not a decision.** If a subagent errors, times out, returns
   malformed output, or `ingest` rejects an item, that item stays pending. Re-batch
   and retry it. Never mark it excluded to make progress.
3. **Unavailable is not excluded.** A study with no retrievable full text is reported
   as unavailable. It is never counted as a full-text exclusion.
4. **Criteria come from `review.yaml` only.** If screening shows a criterion is
   ambiguous, stop and propose the edit to the user. Do not reinterpret it silently.
   Once `review.yaml` or a stage skill is edited, the ledger re-opens the affected
   decisions automatically.
5. **Record who judged.** Pass `--agent "<harness>/<model id>"` to every `ingest`,
   for example `--agent "claude-code/claude-opus-5-5"`. Use your real harness and
   model identifiers. A stage runner adds the judges' effort, as
   `claude-code/claude-opus-5-5/effort-low`.
6. **Keep state on disk, not in your context.** After any interruption or context
   compaction, run `ledger.py status REVIEW` and continue from what it reports.

## Setup

```bash
pip install pyyaml            # the only dependency until the meta-analysis stage
python SKILLS/cbma-review/scripts/ledger.py init REVIEW
```

If `REVIEW/review.yaml` does not exist, write it with the user first. The format is
in `references/review-spec.md`, and `references/example-review.yaml` is a starting
point. `init` rejects unknown keys, so a typo fails loudly. Show the user the
numbered criteria that `init` prints before spending any effort on screening.

## Running a judged stage

**Use a stage runner when you can.** Check for `.claude/agents/cbma-stage-runner.md` in
the workspace (`install.sh` puts it there). If it exists, hand each judged stage to one
`cbma-stage-runner` subagent, and do not run the loop below yourself. The runner makes
the batches, dispatches the stage's judge agents, ingests, retries and audits, then
returns a short report. Your context then grows by one report per stage, not by every
batch. Give it this prompt, with the values filled in:

> Run the `<stage>` stage. REVIEW=`<abs path>` SKILLS=`<abs path>` PYTHON=`<python>`
> AGENT=`<harness>/<model id>`. [Batch size `<n>`.] [Pilot: `--limit <N>`.]

The judges' reasoning effort is set per stage in their agent definitions:
- `cbma-abstract-screener`: low;
- `cbma-fulltext-screener`: medium;
- `cbma-extractor`: low;
- `cbma-selector`: medium.

To change one, edit the `effort:` line of that file before the stage, and record the
change in the run notes.

When a runner returns, copy its report into the run notes. Then:
- **verdict OK:** go on to the next stage, if the user asked for an end-to-end run;
  otherwise report and ask.
- **verdict PROBLEM:** stop, show the user the report, and decide with them.

**End-to-end runs.** If the user says to run the review end to end, do every stage in
order without asking between them:
- search;
- abstract runner;
- retrieval;
- full-text runner;
- extraction runner;
- selection runner;
- export and meta-analysis.

Stop only when a runner reports PROBLEM, when a scripted stage fails, or at a pilot the
user asked for.

## The loop for every judged stage

A stage runner follows this loop. Follow it yourself only when there is no stage
runner, as in harnesses without agent definitions. The same five steps run for
abstract screening, full-text screening, extraction and selection:

1. **Make batches.**
   `python SKILLS/cbma-review/scripts/ledger.py batches REVIEW --stage <stage> --size <n>`
   Only pending items are batched, so re-running this after an ingest picks up
   exactly the retries.
   Suggested sizes: abstract 25, fulltext 2, extraction 1, selection 1.
   Add `--limit N` for a pilot.
2. **Dispatch one subagent per batch file.** Run several at once if your harness
   allows it, but keep it to about 8 at a time. Give each subagent this prompt,
   with the paths filled in:

   > Read `SKILLS/<skill>/SKILL.md` and follow it exactly. Your batch file is
   > `<batch path>`. Process every item in it and write your output to the path in
   > the batch's `output` field. Do not read or write any other part of the review
   > folder except the files the batch names. When done, reply with only the number
   > of items you wrote.

   `<skill>` is the batch's `skill` field.

   - **No in-session subagents** (some Codex setups): run each batch in a fresh
     headless session instead, one at a time or a few in parallel. See
     `scripts/run_batches.sh`.
   - **Never process batches in your own context** beyond a pilot of one or two.
     Fresh context per batch is what keeps the thousandth decision as careful as
     the first.
3. **Ingest.**
   `python SKILLS/cbma-review/scripts/ledger.py ingest REVIEW --stage <stage> --agent "<harness>/<model>"`
   It prints accepted and rejected counts and every rejection reason. Exit code 2
   means at least one item was rejected.
4. **Retry the rejected items.** Go back to step 1. If the same item fails twice
   for the same reason, read the rejection and the item yourself. If the problem
   is the input (a broken table, a garbled text), tell the user and leave the item
   pending.
5. **Audit before moving on.** Run the checks in the next section.

## Stage by stage

1. **Search:** follow `pubmed-search`. When it finishes, run `ledger.py status` and
   report the record count. It must equal PubMed's reported count, and the script
   fails if it does not.
2. **Abstract screening:** run the `abstract` stage (a runner, or the loop). Records without an
   abstract are screened on title and metadata; the `screen-studies` skill covers
   this.
3. **Full-text retrieval:** first run
   `python SKILLS/cbma-review/scripts/ledger.py needs-fulltext REVIEW`, then follow
   `fulltext-sources`. Report available, incomplete and unavailable counts by source.
4. **Full-text screening:** run the `fulltext` stage. Items whose text is
   unavailable are never batched.
   **Combined mode** (`screening.fulltext.select_analyses: true`, records mode only): run
   `ledger.py import-analyses` *before* this stage. The full-text runner then decides
   screening and selection together, following the `screen-and-select` skill. An
   included study with no eligible analysis is recorded as excluded. The selection
   stage afterwards has nothing pending.
5. **Extraction:** run the `extraction` stage, one study per batch. In records mode
   (`extraction.records` is set in `review.yaml`) there is nothing to judge: run
   `python SKILLS/cbma-review/scripts/ledger.py import-analyses REVIEW --agent <records source>`
   instead. Report its counts, including studies with no record, which stay pending.
6. **Selection:** run the `selection` stage. Skip it if `review.yaml`
   defines no targets.
7. **Export and meta-analysis:**
   `python SKILLS/cbma-review/scripts/ledger.py export REVIEW`, then follow
   `cbma-nimare`.

## Audit checklist (do this between stages; it is part of the method)

- **`ledger.py status REVIEW`: pending must be 0** before the next stage, unless
  the user agreed to leave items pending.
- **Abstract screening:** read 10 random excludes and 10 random includes from
  `decisions/abstract.jsonl`. If you disagree with more than one, show the user
  the cases before continuing.
- **Full-text screening:** check `decisions_with_ungrounded_evidence`, which counts
  evidence quotes the ledger could not find in the text. A high count means
  subagents are paraphrasing instead of quoting. Re-batch those studies.
- **Extraction:**
  - `points_unverified` should be near 0. Unverified points are dropped at export.
  - Open two extracted studies and compare them with their tables by eye.
- **Selection:** confirm each target has enough studies for a CBMA (usually at least
  17 for ALE and 10 as an absolute floor). Say so if a target falls short.

## Finishing

Report the final `results/prisma.json` counts to the user. Name every stage where
items remain pending or unavailable, and list the files under `results/`. For a
benchmark run, also follow `SKILLS/../benchmark/README.md` if it exists.

## Layout of REVIEW (for orientation; the scripts create it)

```
review.yaml                 the protocol (criteria, sources, targets)
search/records.jsonl        one record per PMID; search/search_log.json
fulltext/index.jsonl        retrieval outcome per PMID; fulltext/raw/ cached downloads
docs/<pmid>/                normalized text.md, tables/<id>.json, meta.json
work/<stage>/batch_*.json   batches waiting for subagents (outputs land beside them)
decisions/<stage>.jsonl     accepted decisions (append-only; the ledger writes these)
analyses/<pmid>.json        accepted, verified coordinate extractions
results/                    prisma.json, criteria.json, nimads/, meta/
```

## Single-response batch transport

Judged stages now use the shared `cbma-review/scripts/single_response.py` transport.
Python loads complete batch inputs; each judge returns one JSON response with tools
disabled, and Python writes the existing raw batch output for ledger validation.
Use the stage runner/optimized launcher rather than dispatching file-reading judges.
Registered eligibility criteria, models and effort settings are unchanged.
Extraction receives every parsed table (including captions, headers and footnotes),
plus source Methods and coordinate-space paragraphs (`tables_space_context`) to
interpret space and analysis labels/boundaries. An explicit extraction
`context_needed` response triggers at most one full-paper expansion per study and
registered input within the existing retry limit;
selection receives the full article and extracted analysis metadata. Legacy full-input
extraction batches retain their original input view. Unreadable inputs, malformed
responses and rejected judgments stay pending for the existing retry mechanism;
scientific incomplete cases follow the original criteria. Inputs are never silently
truncated. Input hashes, raw responses and CLI usage logs are retained under each
stage's `work/STAGE/judge_responses/` directory. New ledger agent strings include
`single-response-v1`; older decisions remain valid. Already-running stage processes
keep their loaded code; this path applies to subsequent runner invocations.

## Runner scheduling, recovery and usage — shared update

New stage-runner invocations use `single-response-v2`. Scientific criteria, inputs,
models and effort/thinking settings are unchanged. Stable criteria and response
schemas now precede varying items and batch identifiers in judge prompts.

Both harnesses launch only available worker slots, cap judge timeouts to the
remaining stage budget, and stop submitting new calls after provider-limit or
authentication failures. `--max-attempts` defaults to 3 and persists for each
unchanged study/criteria/input combination across invocations. Exhausted studies
remain pending and do not block healthy batch mates. `retry_exhausted` is an
execution problem, never a scientific exclusion; repairing scientific inputs gives
a new identity. Raising `--max-attempts` explicitly permits further attempts.
Claude pilots now save their scope so retries never expand into unscreened records.

For syntactically valid JSON responses, complete valid studies are saved while
malformed, duplicate, unresolved or incomplete studies stay pending. Selection
requires all analysis × target pairs; combined-mode scientific completeness remains
validated by the original ledger. Truncated/invalid JSON is not guessed or repaired.

Every new attempt has a unique directory under `work/STAGE/judge_responses/`,
including raw CLI output, input hashes, recovery details and `attempt.json`.
Failures and timeouts are retained. Stage `usage` is cumulative and counts each
attempt once, including rejected judgments; unavailable token usage and cost are
reported explicitly as unknown. Existing legacy logs are retained, but previously
overwritten attempts cannot be recovered. Codex also reports `usage_this_invocation`.
Existing processes are not interrupted; use the usual launch commands on the next
runner invocation. Scientific audits remain required.

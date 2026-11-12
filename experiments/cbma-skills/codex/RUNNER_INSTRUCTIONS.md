## Codex stage execution

Use the dedicated Codex stage runner for every judged stage, from this project workspace:

```bash
.venv/bin/python ../../run_stage.py PROJECT --stage STAGE
```

Replace PROJECT with this project's name and STAGE with abstract, fulltext,
extraction or selection. For an explicitly requested pilot, append `--limit 50`
(or the requested count); repeat with that same limit until the pilot completes.
Do not use a limit for the full stage. Model and effort are fixed by the arm and
stage, matching the project's Models section.

The runner batches pending studies, invokes `run_judge_optimized.sh`, ingests through
the project's own ledger, retries at most three times per unchanged input, and
returns a compact JSON summary. It preserves logs and attempts across calls.
Read its summary instead of whole judge logs or decision files, except for the
scientific audit sample. `usage` includes processed input and cached input where
Codex reports them; it is not a dollar bill. Logs are under
`PROJECT/work/STAGE/codex_runner/`.

- Exit 0 (`JUDGMENTS_DONE`): perform the stage's scientific audit before continuing.
  The runner does not perform or waive scientific audits.
- Exit 3 (`WORK_REMAINS`): when no persistent input/judge failure is reported, rerun
  the same command. A time budget is a checkpoint, not a reason to stop the review.
  `retry_exhausted` means unresolved failures need inspection and reporting; do not
  increase the cap or reset attempts to hide the problem.
- Exit 4: another runner or legacy project worker is active. Wait; do not duplicate
  its work or ingest its output while it is running.
- Exit 1: inspect the reported execution error. Correct auxiliary command mistakes;
  persistent bundled-workflow failures follow this project's stop rules.

Use `../../run_judge_optimized.sh PROJECT STAGE BATCH` only for a deliberate single
batch recovery with no stage runner/other worker active. Judges have their exact
stage skill preloaded and use the batch's `texts_file` first, then the relevant
original tables where needed. They read all content, including chunks after a
truncated tool response. Bundling does not trim article content or change criteria.
Never add scientific guidance to the fixed judge prompt.

## Meta-analysis execution and reporting

Use the bundled `cbma-nimare/scripts/run_meta.py`, the registered `review.yaml`
settings and the installed pinned NiMARE version. Follow the existing skill's
checks for coordinate spaces, repeated samples and implausible maps. Read the
script's summaries, export report and numerical diagnostics for routine reporting.
Do not routinely inspect NiMARE internals, browse documentation, generate extra
diagnostics or open plot images. Investigate those only for a concrete error,
required unresolved scientific check or explicit user request, and record why.
Do not alter thresholds, estimators or correctors as part of that investigation.

## Single-response batch transport

Judged stages now use the shared `cbma-review/scripts/single_response.py` transport.
Python loads complete batch inputs; each judge returns one JSON response with tools
disabled, and Python writes the existing raw batch output for ledger validation.
Use the stage runner/optimized launcher rather than dispatching file-reading judges.
Scientific stage skills, criteria, models and effort settings are unchanged.
Extraction receives every parsed table (including captions, headers and footnotes);
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

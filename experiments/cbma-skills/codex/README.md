# Codex execution layer

This directory contains the Codex transport, judge and stage runner used by the
isolated E6 workspaces. It does not replace scientific skills or ledger validators.

Deploy `portkey_codex.sh` in the workspaces parent and these two files in each arm:

- `run_codex.sh` as `run_codex.sh` (per-project orchestrator defaults);
- `run_stage.py` as `run_stage.py`;
- `run_judge.sh` as `run_judge_optimized.sh`.

Use copies, not symlinks: the runner finds the arm from its own resolved path.
Retain `run_judge.sh` in an arm during migration so existing dispatchers continue
using the legacy launcher. Enable the new execution path per project through the
instructions in `RUNNER_INSTRUCTIONS.md`, replacing PROJECT with its name.

From a project workspace:

```bash
.venv/bin/python ../../run_stage.py dementia --stage fulltext
```

The runner imports the project's ledger snapshot, preserves stage models and
efforts, uses the same batch sizes as Claude, and delegates every scientific
judgment to fresh Codex judges. Only the ledger writes accepted decisions. An
exit-0 summary does not substitute for the stage's scientific audit.

The optimized judge preloads the stage's SKILL.md verbatim in developer instructions.
Combined screening/selection also preloads its two referenced skills. Existing
texts_file bundles supply article text and extraction tables; judges still open
criterion-bearing full-text tables and ambiguous original table grids. No article
trimming is introduced.

The runner uses an OS stage lock and detects active legacy project workers before
adopting their outputs. Optimized judges also take a batch lock. It persists a
three-attempt cap per unchanged criterion/input hash, fixed pilot scopes and JSONL
judge logs in work/STAGE/codex_runner/. The time budget stops new dispatch while
already running judges finish within their individual timeout. It may therefore
return later than the dispatch time budget; invoke it using a terminal or a tool
that supports background process sessions.

Codex turn.completed usage is summed in each invocation's compact report when
available. Missing usage is unknown, not a zero-cost judgment. The logs remain
available for the independent transcript audit and Portkey billing reconciliation.

## Deployment on 2026-10-02

Installed in e6-codex and e6-codex-luna. Enabled through AGENTS.md for eight
projects per arm. vbm_of_substance_use was migrated after the user reported
it finished. emotion_regulation_2022 in both arms remains deferred at the
user's request. Its AGENTS.md, RUN_NOTES.md, skills and legacy judge launcher
remain unchanged. Migrate it explicitly after its current workers finish.

No live review or API calls were launched for this deployment. No implementation
tests were run; the user requested implementation without requesting tests.

## Orchestrator defaults

On 2026-10-02 the user selected gpt-6.1-sol at medium effort for the six
unstarted projects in each arm: cue_reactivity, decision_making, dementia,
executive_function, problem_solving and social. Their `.codex/orchestrator.json`
files are read by run_codex.sh. A missing config preserves Astra/high for started
projects. ORCH_MODEL and ORCH_EFFORT remain explicit overrides. Stage judge
models, scientific protocols and existing results are unchanged. No route/API
smoke test was run for the new orchestrator model.

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

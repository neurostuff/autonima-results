# E6 Codex run: <project>

This workspace runs only `<project>`. The review folder is `./<project>/`.
Use its own ledger and the local skill copy at `.agents/skills/`.

Run the full-text workflow on actual full texts. Do not use Pondie records or copy
decisions from the Claude arm or any other project. Do not inspect or use gold files.
If a criterion is ambiguous, stop and record the issue for a protocol decision.

## Models

- Abstract screening: `gpt-6-luna`, low effort.
- Full-text screening: `gpt-6-luna`, medium effort.
- Coordinate extraction: `gpt-6-astra`, low effort.
- Analysis selection: `gpt-6-astra`, medium effort.
- Orchestration: `gpt-6-astra`, high effort.

Use the exact model and effort in each ledger `--agent` value and the project's
`RUN_NOTES.md`. The existing full-text source paths are preserved in `review.yaml`.

Before judging, check `RUN_NOTES.md`, run ledger init, compare the numbered criteria
with the registered protocol, and complete the 50-record abstract pilot. Stop for the
user to review the pilot before starting the full run.

Correct errors in your own ad hoc diagnostics, preflight commands or JSON access,
then rerun the corrected check and continue. In `results/criteria.json`, target
definitions are under `criteria["selection"]["targets"]`. Stop for persistent bundled
workflow failures, input problems or protocol ambiguities; an auxiliary command you
wrote incorrectly is recoverable without user approval.

Keep decisions in batch output files until the ledger ingests them. Unavailable full
texts remain unavailable; they are not exclusions. Follow the `cbma-review` skill's
audit checklist at each stage.

Gold is outside this workspace. Folder separation is organizational only: a local
Codex process can still read other paths on the host. For strict blinding, run Codex
inside an isolated container or account that omits all gold paths.

## Codex stage execution

Use the dedicated Codex stage runner for every judged stage, from this project workspace:

```bash
.venv/bin/python ../../run_stage.py <project> --stage STAGE
```

Replace <project> with this project's name and STAGE with abstract, fulltext,
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
`<project>/work/STAGE/codex_runner/`.

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

Use `../../run_judge_optimized.sh <project> STAGE BATCH` only for a deliberate single
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

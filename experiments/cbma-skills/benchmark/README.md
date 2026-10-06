# Benchmarking cbma-skills

`compare.py` scores a finished review against:
- a gold standard (`--gold`);
- an autonima run on the same project (`--autonima`);
- gold coordinates (`--gold-nimads`).

```bash
python benchmark/compare.py REVIEW \
  --gold gold.csv \
  --autonima /path/to/autonima-results/<project>/<run> \
  --gold-nimads gold_studyset.json \
  --out REVIEW/results/benchmark.json
```

## Gold file format

```csv
pmid,abstract_included,included
11111111,1,1
22222222,0,0
```

- **`included`** is final inclusion and is the main outcome.
- **`abstract_included`** is optional. When it is absent, the final includes are
  used as the abstract-stage positives. That tests recall only, which is the
  property that matters at that stage.
- **A plain text file** with one PMID per line is read as the final includes.

Each benchmark needs a short adapter from its own files to this shape. Keep the
adapter beside the benchmark data, not in this package, so the mapping is
reviewable.

## Keeping the agent blind

Put gold files outside the agent's workspace, and run `compare.py` yourself after
the review is finished. If you run it through an agent, use a fresh session that
did not run the review.

See [../PLAN.md](../PLAN.md) for the full protocol and metrics.

## Auditing a run from its transcripts

`audit_transcripts.py` reads Claude Code session transcripts. For Codex CLI, use
`audit_codex_transcripts.py`; it reads `~/.codex/sessions` and filters sessions to one
project workspace by the recorded working directory:

```bash
python benchmark/audit_transcripts.py ~/.claude/projects/<workspace path, "/" as "-"> \
  --review REVIEW --forbid /path/to/gold --out audit.json
```

```bash
python benchmark/audit_codex_transcripts.py ~/.codex/sessions \
  --workspace /path/to/cbma-workspace-e6-codex/projects/emotion_regulation_2022 \
  --review /path/to/cbma-workspace-e6-codex/projects/emotion_regulation_2022/emotion_regulation_2022 \
  --forbid /path/to/gold --out /path/to/emotion_regulation_2022/results/codex_audit.json
```

Run the audit after the project's screening and analysis work is complete, or when
stopping a run that will not continue. That captures the sessions accumulated so far;
rerun it after later work to refresh the report. Keep logs until the final audit has
been saved. For a multi-stage or resumed run, use the same workspace path so all its
Codex sessions are included.

- **Tokens:** per role (orchestrator, stage runner, judge) and stage, with each role's
  effort and the usage-limit events.
- **Blinding:** any tool call touching a `--forbid` path.
- **Writes:** judge writes outside their batch output, and decision files written
  anywhere but through the ledger.

The Codex reader summarizes CLI usage records and flags tool calls that mention
forbidden paths. Codex's log schema and tool representations can vary by CLI version;
review `recognized_usage_records` and the session details in the JSON report. Treat
`possible_direct_writes` as candidates for manual review rather than a complete proof
that no direct writes occurred.

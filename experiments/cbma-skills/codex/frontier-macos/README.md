# E6 Codex frontier and Luna bundle — Apple Silicon macOS

A fresh frontier full-text experiment for all nine projects, plus four independent
Luna-arm experiments under luna/. Scientific criteria and stage
skills come from the original E6 arm's clean review templates and skill snapshot
`f3a68422262350ee3ddcc36fc89f416695f10d10`. No earlier search results, judgments,
extracted coordinate records or gold-standard files are included.

| Role | Model | Effort |
|---|---|---|
| Orchestration | gpt-6-astra | max |
| Abstract screening | gpt-6-astra | max |
| Full-text screening | gpt-6-astra | max |
| Coordinate extraction | gpt-6-luna | low |
| Analysis selection | gpt-6-astra | max |

The four Luna projects are decision_making, dementia, executive_function and social.
Their judges use gpt-6-luna at the original efforts (low, medium, low, medium),
and orchestration uses gpt-6.1-sol at medium effort. Their local skills match the
existing Luna snapshot, with separate ledgers/results and shared bundled runtime
and article corpora. No Luna run is included for substance use, PTSD, emotion
regulation, problem solving or cue reactivity.

## Included dependencies and inputs

- Apple Silicon Codex CLI 0.160.0 native binaries and resources; Node/npm is not
  required to launch this native build.
- Relocatable CPython 3.12.15, macOS ARM64 Python dependencies, NiMARE 0.21.0,
  the Elsevier parser, package metadata and offline wheels.
- Actual copies of raw ACE HTML and Elsevier XML article corpora, plus each
  project's cached PMC XML articles. No symlinks or external repository paths
  are required. Shared article files live once under corpora/ to keep size down.
- Thirteen independent project/review folders, local skill copies, fresh initialized
  ledgers, registered protocols, run notes, a scripted stage runner and auditor.

Target: native Apple Silicon macOS, macOS 12 or newer. Intel/Rosetta and Linux
are not supported by these bundled binaries. Bash and standard macOS utilities
come from the operating system. Python dependencies are bundled, so Homebrew,
Conda and pip downloads are not needed. Internet is still required for account
login, Codex model calls, PubMed search and PMC articles not in the copied cache.

This package was assembled on Linux. Its macOS binaries and compiled dependencies
have not been executed on a Mac. Asset checksums, package inventory and fresh
ledger initialization are recorded; no scientific runs or model calls were made.

## Account setup

Unzip the folder anywhere on the Mac, then enter it:

```bash
cd /path/to/e6-codex-frontier-macos
bash setup.sh
bash login.sh
```

Sign in with the free-credit ChatGPT/Codex account. For device-code login, use
`bash login.sh --device-auth`. Account credentials and Codex sessions are stored
outside this package in `~/.codex-e6-frontier`; the ordinary Codex profile is
unchanged. Set FRONTIER_CODEX_HOME to choose another separate profile. Do not
point it to a profile belonging to the existing E6 runs. No API key or Portkey
configuration is needed or bundled. Actual model access depends on this account
and has not been checked by the build.

## Launch a project

```bash
./run_codex.sh dementia
```

Project names: cue_reactivity, decision_making, dementia,
emotion_regulation_2022, executive_function, problem_solving, social,
vbm_of_ptsd, vbm_of_substance_use.

The launcher supplies START_PROMPT.txt automatically when no prompt is given,
so it starts the review and continues through all stages without approval pauses.
You can still append a custom initial prompt:

```bash
./run_codex.sh dementia "Read AGENTS.md and RUN_NOTES.md. Run this fresh review end to end using the configured models and original criteria. Complete and audit the 50-record abstract pilot, then continue without pausing if the audit passes. Continue usable studies when article, table or supplement inputs remain incomplete, recording those cases separately as instructed. Finish selection, export and meta-analysis, and report coverage, deferred studies and skipped targets. Stop for a genuine protocol ambiguity, failed scientific audit or persistent execution/account problem."
```

The user authorized uninterrupted execution: the frontier 50-record pilot is
audited and followed by the full stage when it passes; Luna has no pilot. Each
project AGENTS.md explains the workflow. Do not launch another worker for an
already-running stage.

For a headless frontier run that can be left running:

```bash
./run_review.sh dementia
```

For a Luna project:

```bash
./luna/run_review.sh dementia
```

For all four Luna projects, sequentially, with a log for each:

```bash
./run_all_luna.sh
```

These commands require the separate account login first. They use never-ask
approval settings within the workspace sandbox. Known missing inputs alone do
not pause the review. Genuine protocol ambiguity, failed scientific audits and
persistent account/tool failures remain reportable blockers; no model can finish
valid judgments without its required source data or account access.

## Stage execution

From a project workspace:

```bash
cd projects/dementia
.venv/bin/python ../../run_stage.py dementia --stage fulltext
```

The `.venv/bin/python` files are regular shell wrappers into the bundled runtime,
not virtualenv symlinks. The runner uses the project's unchanged ledger snapshot,
fixed model/effort assignments, fresh headless judges, preloaded skills, full
article bundles, bounded retries and stage/batch locks. It prints a compact report.
Scientific audits remain required. Routine meta reporting reads numerical
summaries; images, implementation details and documentation are inspected only
for a concrete error, scientific check or user request.

## Incomplete-input continuation

Unavailable and genuinely incomplete source material remains separately recorded.
It is not relabeled as scientific exclusion. Complete articles are judged under
the original criteria, including permissions to use figures or supplements.

After bounded retrieval/extraction attempts, record confirmed missing-input cases
by PMID and reason in RUN_NOTES.md. Continue usable studies. For a stage blocked
solely on those documented inputs:

```bash
.venv/bin/python ../../run_stage.py dementia --stage extraction --defer-pmids PMID1 PMID2
```

Deferred studies remain pending in the ledger. This flag does not bypass schema
validation or authorize deferring scientific disagreements, failed audits, invalid
model outputs or account/tool failures. Reapply the list on later checkpoints.
Export with --allow-pending only for these documented input cases, with reduced
coverage explicitly reported. Unknown coordinate spaces must not be guessed;
unusable targets may remain blocked while other valid targets continue. Targets
below the registered minimum are reported as skipped.

## Share the package

```bash
./package.sh
```

This writes a ZIP beside the folder, preserves regular-file permissions and omits
machine state/cache directories. Credentials are outside the folder. The packager
refuses symlinks and known credential files. The supplied pristine ZIP is ready
to share; each colleague logs in separately. arm.json records the asset, corpus
and protocol provenance. MANIFEST.sha256 records the pristine file inventory.
If the experiment produces results after packaging, regenerate the ZIP; the
original pristine manifest still describes the initial inputs.

## Transcript audit

After a run finishes or pauses:

```bash
./bin/python helpers/audit_codex_transcripts.py ~/.codex-e6-frontier/sessions \
  --workspace "$PWD/projects/dementia" \
  --review "$PWD/projects/dementia/dementia" \
  --out "$PWD/projects/dementia/dementia/results/codex_audit.json"
```

The package contains no benchmark gold. Scoring must happen separately in an
evaluator workspace. Account session logs are not automatically included when
sharing the scientific package.

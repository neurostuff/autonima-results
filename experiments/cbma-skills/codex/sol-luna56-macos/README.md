# E6 Codex Sol / Luna 5.6 — five remaining projects

Fresh Apple Silicon macOS runs completing the Sol / Luna 5.6 arm your colleague
ran in the four-project Delta bundle. This package contains only:

- cue_reactivity
- emotion_regulation_2022
- problem_solving
- vbm_of_ptsd
- vbm_of_substance_use

| Role | Model | Effort |
|---|---|---|
| Orchestration | gpt-6.1-sol | medium |
| Abstract screening | gpt-5.6-luna | low |
| Full-text screening | gpt-5.6-luna | medium |
| Coordinate extraction | gpt-5.6-luna | low |
| Analysis selection | gpt-5.6-luna | medium |

The models above match the Delta Sol runs' recorded judge models, rather than the
old bundle's gpt-6-luna label. All five ledgers are fresh. The scientific skills
are snapshot a0ef67e16c3bdcfa31b170b07a58ea7910f80fec, matching those four runs.
Registered protocols are included; only corpus paths change in runnable copies.

## Install and copy the existing articles

Target: native Apple Silicon macOS 12 or newer. The package includes Codex CLI
0.160.0, CPython 3.12.15, NiMARE 0.21.0, the Elsevier parser, dependencies and
offline wheels. No Homebrew, pip downloads or symlinks are needed.

**The ZIP excludes all article corpora and raw PMC caches.** Keep your existing
e6-codex-frontier-macos bundle and unzip this one beside it, into a separate folder.
Do not unzip it over a completed run.

```bash
cd /path/to/e6-codex-sol-luna56-macos
bash setup.sh
./copy_articles.sh ../e6-codex-frontier-macos
bash login.sh
```

Pass the actual old bundle path to copy_articles.sh. It copies only the listed
raw XML/HTML articles and the five projects' original PMC XML caches, validating
their SHA-256 fingerprints. It copies no judgments, coordinates, normalized docs,
exports, settings or credentials. Article files become regular local copies;
the old bundle can remain untouched. Required corpus inputs must be available
before launching. Missing optional PMC cache files are reported and can be
retrieved during the review. ARTICLE_INPUTS.sha256 lists the expected originals.

Login uses the same separate bundle profile as before (~/.codex-e6-frontier), so
if that profile is already signed into the intended account, login can be skipped.
Use bash login.sh --device-auth for device-code login, or set FRONTIER_CODEX_HOME
to select a different profile. Credentials and session logs stay outside the
package. Portkey and API keys are not used. Model access is account-dependent;
no account/model call was made while assembling this package.

Internet is needed for Codex calls, PubMed search and uncached source retrieval.
The macOS binaries were packaged on Linux and have not been executed on a Mac.

## Run all five, unattended

```bash
./run_all_luna.sh
```

This runs the five projects sequentially and records an orchestrator log in each
review's results folder. If one session fails, the script attempts the remaining
projects and returns a nonzero exit status at the end. Rerunning resumes from the
ledgers; completed decisions are not repeated. A successful process exit alone
is not proof of scientific completion: each RUN_NOTES.md reports final coverage,
stage audits, missing inputs and remaining blockers.

For one headless review:

```bash
./luna/run_review.sh vbm_of_ptsd
```

For an interactive terminal session:

```bash
./luna/run_codex.sh vbm_of_ptsd
```

Both launchers inject the end-to-end prompt automatically. Each project's own
AGENTS.md, local skill copy, review folder and ledger remain independent. There
is no pilot or stage-approval pause. The scripted stage runner handles batching,
fixed judge models, bundled full text, retries, locks and validated ingestion.
Keep the Mac awake and connected while running; the command can be prefixed with
the standard macOS utility: caffeinate -i ./run_all_luna.sh.

## Continue with incomplete source inputs

All runs are authorized to proceed through the complete workflow after passing
the required audits. Correct auxiliary command mistakes and retry recoverable
failures within the bounded rules. Missing source articles, tables or supplements
are recorded separately and usable studies continue under the original criteria.
Do not silently exclude missing cases, guess coordinates/spaces, weaken criteria,
or omit studies from fits without reporting the resulting coverage.

For confirmed missing inputs, the runner supports --defer-pmids. Deferred cases
remain pending in the ledger. Export --allow-pending is permitted only for these
documented input cases. Independent valid meta-analysis targets continue; targets
below the registered minimum are reported as skipped. Scientific audit failures,
unresolved protocol ambiguity and persistent execution/account failures remain
blockers. Do not repeat failing calls or reset attempt counters indefinitely.

Routine reporting uses numerical summaries; opening images or browsing package
documentation requires a concrete error, scientific check or explicit request.

## Share again

```bash
./package.sh
```

This regenerates MANIFEST.sha256 and writes an article-free ZIP beside the folder,
preserving executable modes and excluding caches and machine state. Even after
copy_articles.sh has populated this folder, article content remains excluded.
The original bundle and its completed experiments are unchanged.

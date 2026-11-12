# cbma-skills usability review (2026-11-03)

**Purpose.** The skills package was built as a benchmark experiment: can a coding
agent reproduce autonima? That question is answered (yes, at parity — see
Evidence). The new goal is to make the skills package **the primary way users run
autonima**: most users already have a Claude Code or Codex subscription through
their university, need no API key and no new software, and want help *drafting*
criteria, not just executing them. This review assesses the package against that
goal and proposes a design for a uniform interface over two modes:

- **standalone** — everything local, as today;
- **compose** — the review starts on Neurosynth Compose, reads indexed records
  through an MCP server, and pushes decisions and results back.

Two requirements are non-negotiable:
1. **Strict protocol tracking.** A full criteria set is registered before
   screening; every change is versioned; no silent p-hacking.
2. **Token efficiency.** Most users will run on low-tier subscriptions. Whatever
   can be hard-coded must be.

## Evidence base

Everything below is grounded in the nine-project benchmark (three arms, all
scored against NeuroMetaBench with the brain-masked pipeline) and the
failure-mode analysis of the run transcripts and notes.

| Measure | Sol/Luna 5.6 | Astra/Astra | Opus/Haiku | autonima |
|---|---|---|---|---|
| Mean screening F1 | 0.449 | 0.464 | 0.424 | 0.437 |
| Mean map r | 0.687 | 0.711 | 0.676 | 0.706 |
| Cost per project | n/a | n/a | ~US$100 | US$21.55 |

No arm differs significantly from autonima on either accuracy measure (Wilcoxon
across nine projects, P ≥ 0.16). **Accuracy is not the problem. Cost, stops and
governance are:**

- The Opus/Haiku arm cost ~5× autonima. Per stage (Haiku tokens priced at
  gpt-5-mini rates): abstract screening 0.5× autonima, full-text 0.8×,
  **selection 2.3×** (2.25 judge sessions per study from retries; ~4× output;
  ~1.8× input). Orchestration overhead: 13% of tokens, 10% of cost. ~9% of total
  spend was discarded work (a re-extraction forced by a mid-experiment change).
- Every Claude run stopped mid-run despite end-to-end briefs: 1–4 user decisions
  per project, several of which silently changed scientific scope (accepting
  ungrounded quotes, skipping small targets, leaving retry-exhausted studies
  out).
- The ledger's hard rules held (0 decisions written outside the ledger, 0
  blinding breaches across 15 audited runs), but everything *around* the ledger
  was mutable: another session edited live agents, configs and run notes; the
  Codex arms applied unrequested "working defaults" at fitting time, including a
  criterion amendment; run notes proved unusable as provenance because agents
  write them.

## Verdict

**Keep (these are the product):**
- The ledger as sole decision writer, with criteria + input hashes that silently
  invalidate stale decisions and re-batch exactly the re-opened items. This is
  already most of the pre-registration machinery.
- The hard rules table in README (failures are never decisions, unavailable ≠
  excluded, point-vs-table verification, quote grounding, unknown-key rejection).
- The tool-free single-response judge transport (`--json-schema`, no file access)
  — it ended stray writes and is also the token-efficient path.
- Records mode + combined screening/selection — this is, almost exactly, the
  compose-mode consumption shape (see §5).
- The per-stage agent definitions with pinned effort levels.
- The offline test suite (76 tests).

**Blockers for end users, in priority order:**
1. No protocol registration or amendment trail (§3).
2. Cost: selection retries, verbose outputs, full-text inputs, orchestration
   overhead (§4).
3. Mid-run stops with no declared policy; "end to end" is not reliable (§6).
4. No criteria-drafting path; `review.yaml` assumes the user already knows their
   criteria (§2).
5. No human review surface; the audit checklist is agent-executed prose (§7).
6. Experiment residue: SKILL.md has accreted contradictory "update" sections;
   the Codex runner is a diverged fork; benchmark/codex packaging is mixed into
   the user-facing package (§1).

---

## 1. Setup and first contact

**Findings.**
- `install.sh` symlinks skills and copies agents — fine mechanically, but the
  package lives inside `autonima-results` next to benchmark data, gold maps and
  experiment notes. README itself says it is "meant to move to its own
  repository."
- `skills/cbma-review/SKILL.md` is the entry point and currently reads as three
  documents: the original instruction set (dispatch file-reading subagents), plus
  two appended mid-experiment "update" sections (single-response-v1, v2) that
  partially contradict the body. A fresh agent reading top to bottom gets two
  different dispatch protocols. These sections were appended *by agents during
  the experiment* — exactly the accretion the registration design below forbids.
- `codex/run_stage.py` (270 lines) is a fork of
  `skills/cbma-review/scripts/run_stage.py` (354 lines) that has already
  drifted; during the experiment fixes landed in one and not the other.

**Recommendations.**
- **R1.1 (M)** Split into its own repo (working name `autonima-skills`), shipping
  only `skills/`, `agents/`, `install.sh`, tests and the spec reference.
  Benchmark harness, Codex packaging and PLAN stay in autonima-results. Ship as
  a Claude Code plugin (marketplace install = one command) and as a plain folder
  for Codex.
- **R1.2 (S)** Rewrite `cbma-review/SKILL.md` as one coherent current document:
  single-response transport only, stage-runner-first, the loop section demoted to
  a fallback appendix for harnesses without agents. Delete the changelog
  sections; history belongs in git.
- **R1.3 (M)** Merge the two stage runners into one `run_stage.py` with a
  `--harness claude|codex` flag (the judge-invocation line is the only real
  difference). One file, one set of retry/limit semantics, tested once.
- **R1.4 (S)** `install.sh` records the installed skill-set hash into the
  workspace (`.cbma-skills.lock`); `ledger.py status` warns when installed files
  no longer match it (detects the cross-session edits we observed).

## 2. Criteria drafting (goal 3: introspection and seeding)

**Findings.** `init` validates and numbers criteria and `review-spec.md` is a
good reference, but nothing helps a user *arrive at* criteria — the hardest part
of a meta-analysis, as stated. Meanwhile `reviews/` already contains nine real
specs translated from published meta-analyses (the NeuroMetaBench projects):
an unused seed library.

**Recommendations.**
- **R2.1 (M)** New skill: **`draft-review`**. A Q&A planning session that
  interviews the researcher (question, population, modality, contrast directions,
  whole-brain stance, overlap policy…), then drafts `review.yaml` with each
  criterion annotated with its provenance ("adapted from
  emotion_regulation_2022 I3"). It must end by running `ledger.py init` and
  showing the numbered criteria, and must not start screening.
- **R2.2 (S)** Ship `reviews/` (renamed `library/`) with the package as the seed
  corpus, each spec with a one-line README of what the published review was.
  Compose mode later serves this library from the server, where it can grow.
- **R2.3 (M)** Close the loop with the human-review surface (§7): when user
  corrections cluster on one criterion, the orchestrator proposes a concrete
  amendment ("your three overrides all read I3 as allowing SVC-corrected
  results; amend I3 to say so?") — which then goes through `amend` (§3), never
  through silent reinterpretation. The benchmark showed exactly where this
  matters: most lost recall came from one criterion (whole-brain, I6/E2) applied
  literally.
- **R2.4 (S)** A **calibration pilot** as the documented default path: pilot 50
  abstracts (`--limit` exists), human reviews the queue, amend criteria if
  needed, then full run. Cheap, and it front-loads the criterion fixes that
  otherwise surface as mid-run stops.

## 3. Protocol registration and strict version tracking

**Findings.** The hash machinery is strong: every decision stores a criteria
hash (objective + criteria + instructions + stage-skill text) and an input hash,
and a change re-opens exactly the affected decisions. But:
- `review.yaml` can be edited freely; nothing records *that* it changed, *what*
  changed, *when* or *why*. The invalidation is silent.
- Run notes cannot serve as the change record: agents write them, and in the
  benchmark another session wrote false "user authorized" entries into them.
- Skill files are part of the hash but the installed copies were editable live
  (and were edited, by another session, mid-experiment).
- Fitting-time choices (dropping unknown-space, overlapping or duplicate
  analyses) happened entirely outside the ledger in the Codex arms.

**Recommendations — this is the core of the anti-p-hacking requirement.**
- **R3.1 (M)** `ledger.py register REVIEW`: snapshots `review.yaml` + the skill
  hashes + policies (§6) into `protocol/v001.json`, append-only and
  content-addressed. `batches` and `ingest` **refuse to run** when the live
  `review.yaml` hash differs from the registered head: the error says "amend or
  revert". Every decision records `protocol_version`.
- **R3.2 (M)** `ledger.py amend REVIEW --reason "…"`: prints the diff against
  the head, writes `v00N+1`. **Refuses to run without a TTY.** An orchestrator
  runs non-interactively, so this single check mechanically reserves amendments
  for the human — the cheapest strong guardrail available locally. (In compose
  mode the server replaces the TTY: amendments are confirmed on the website.)
- **R3.3 (S)** `export` embeds the full version chain and per-version decision
  counts into `results/prisma.json` and the NiMADS output. The deliverable then
  carries its own audit trail.
- **R3.4 (S)** The meta step is already a fixed script (`run_meta.py` defaults);
  make the remaining fitting knob (`--unknown-space`) part of the registered
  protocol rather than a CLI choice. Any analysis omission at fit time must be a
  ledger decision or it does not happen.
- **R3.5 (S)** `status` verifies installed skills against the lock file (R1.4)
  and against the registered skill hashes, and reports "protocol: v3 (2
  amendments)" — drift from outside the session becomes visible at every stage
  boundary instead of being discovered in a post-hoc audit.

## 4. Token efficiency

**Findings (measured, Opus/Haiku arm).** Abstract screening is already *cheaper*
than autonima (0.5×) and full-text comparable (0.8×). The money goes to:
selection (2.3×: retries 2.25 sessions/study, output ~4×, input ~1.8×),
extraction of full papers, orchestration (13% of tokens), and discarded work.

**Recommendations, by measured lever:**

| # | Lever | Evidence | Effort |
|---|---|---|---|
| R4.1 | **Partial-study selection ingest.** Accept valid (analysis × target) pairs, re-batch only the missing/invalid pairs instead of retrying whole studies | the 2.25×-sessions multiplier is mostly whole-study retries after one malformed pair | M |
| R4.2 | **Compact selection output as the default** (built, currently opt-in via `judge_output_formats.json`) | selection output was ~4× autonima's; per-pair JSON envelopes dominate | S |
| R4.3 | **Trimmed full text as the recommended default** (`text: trimmed`, built) | median 49% of text retained; 4% fallback to full; hashes handle the switch | S |
| R4.4 | **Keep thinking budgets pinned per stage** (built: `judge_thinking_budgets.json`); document them in the registered protocol | thinking was a visible share of selection output | S |
| R4.5 | **Script-first orchestration.** The stage runner relays `run_stage.py`'s JSON report verbatim instead of re-narrating; stage sequencing stays in the orchestrator but each stage is one subagent call, one report | 13% of tokens were orchestration | S |
| R4.6 | **Hard-code overlap triage.** Flag same-group/same-sample candidates by metadata (authors, N, task) in the ledger; judges see the flag instead of being asked open-ended overlap questions | overlap handling caused stops and fitting-time improvisation | M |
| R4.7 | **Compose mode skips extraction entirely** (pre-parsed analyses from the index) and screens from records | extraction + its one forced re-run was the largest single block of discarded + repeated work; records-mode screening measured near-parity with slightly lower recall | — (§5) |

Taken together, R4.1–R4.5 plausibly halve standalone cost; compose mode (R4.7)
removes the extraction stage and shrinks the candidate pool before any judging,
which is where the remaining multiple lives. These are claims to verify with the
existing per-stage usage accounting (`work/STAGE/judge_responses/` +
`usage`), which should become a first-class `ledger.py usage` report so users on
subscriptions can see burn per stage mid-run. **(R4.8, S)**

## 5. The uniform interface: one contract, two providers

**Key structural finding: compose mode already exists in embryo.** Records mode
(`extraction.records` + `import-analyses` + combined screen-and-select) is
precisely the shape of compose consumption — pre-extracted analyses standing in
for extraction, structured records standing in for full text. The work is to
formalize the *provider boundary*, not to build a second pipeline.

**Design.** `review.yaml` gains a `mode` (default `standalone`) and compose
coordinates:

```yaml
mode: compose            # standalone | compose
compose:
  project: <compose project id>     # minted when the user starts on the website
  server: https://compose.neurosynth.org/api
  # token read from $COMPOSE_TOKEN, never stored in the yaml
```

The stage contract — ledger stages, decision schemas, hashes, PRISMA — is
**mode-independent**. Only the providers swap:

| Stage | standalone provider | compose provider |
|---|---|---|
| Search | `pubmed_search.py` (E-utilities) | MCP `search_studies` (text + structured filters) |
| Retrieval | PMC + local folders | not needed for screening (records); optional local folders for the hybrid path |
| Abstract screening | judge on title/abstract | judge on record (same skill, same criteria IDs) |
| Full-text screening | judge on normalized text | judge on record + extracted analyses (`screen-and-select`), hybrid fallback to local text for low-confidence items *(stretch)* |
| Extraction | judged, table-verified | **skipped**: `import-analyses` from MCP `get_analyses` |
| Selection | judged on full text | judged on analyses + context snippets (combined mode folds it into screening) |
| Meta | NiMARE script | NiMARE script, results pushed |
| Tracking | `protocol/` + lock file | same, mirrored to server (registered = pushed) |

Provenance: imported analyses already carry `source` verification status and the
importing agent string; compose imports record the server, index version and
study ids, so a compose review is exactly as auditable as a standalone one.

## 6. End-to-end reliability: declared policies instead of stops

**Findings.** The benchmark's mid-run stops were mostly *policy questions in
disguise*, asked one at a time: accept inexact quotes? raise retries? skip an
11-study target? Each stop risks the "wrong answer" steering the review, and
resumes burn tokens and invite drift.

**Recommendation R6.1 (M).** A `policy` block in `review.yaml`, registered with
the protocol, so these are pre-registered answers rather than mid-run chat:

```yaml
policy:
  retry_exhausted: leave_pending   # | stop        (never an exclusion — rule 2)
  ungrounded_evidence: report      # | stop        (spot-check counts in the stage report)
  small_target: proceed            # | stop        (floor; report either way)
  provider_limit: wait             # | stop        (usage/session limits)
  max_attempts: 3
```

The stage runner consults the policy and stops **only** for: a scripted-stage
failure, a policy set to `stop`, or an amendment-worthy criterion ambiguity
(which routes through §3, not through chat). This converts the benchmark's
1–4 stops per project into, typically, zero.

## 7. Reviewing findings (goal 4: designed UX, not improvised)

**Findings.** Today the audit checklist tells the *orchestrator* to sample
decisions and eyeball extractions. Nothing is designed for the human, and
nothing flags the cases worth their time.

**Recommendations.**
- **R7.1 (M)** `ledger.py review-queue REVIEW --stage <s>`: a deterministic,
  seeded sample written to `results/review_queue.jsonl`, composed of
  (a) flagged items — ungrounded/inexact evidence, pivotal `unclear` states,
  abstract-vs-fulltext disagreements, overlap flags (R4.6), targets near the
  study floor; (b) a few high-confidence includes/excludes as calibration. This
  hard-codes the sampling the orchestrator currently does in prose (and spends
  tokens on).
- **R7.2 (M)** A static HTML review page (no server, written into `results/`):
  each queued decision with its criteria states, reason and evidence quotes
  linked into the text, with accept / flag / note buttons that export a verdicts
  file. Verdicts are ingested as decisions by `agent: human/<name>` — human
  review becomes part of the provenance rather than a side channel.
- **R7.3 (S)** Verdict clusters feed R2.3 (proposed amendments). In compose
  mode the same queue is pushed, and Compose's existing curation UI can render
  it — the queue format, not the page, is the contract.

## 8. Neurosynth Compose / NeuroStore: requested API & MCP surface

What the skills side needs from the server team, shaped so the ledger's
invariants survive the trip. Read side (NeuroStore index):

1. **`search_studies`** — text query + structured filters over extracted
   features (`has_coordinates`, `coordinate_space`, `modality`, `n_analyses`,
   year range, design/task terms…), cursor-paginated, returning a **stable
   total** (the ledger's completeness check — retrieved must equal reported — is
   the rule that caught silent truncation in autonima #53).
2. **`get_study`** — bibliographic record + abstract + feature summary.
3. **`get_analyses`** — parsed analyses with points, space, table
   label/caption and any shareable context snippets: exactly the records-mode
   input. (Full text stays out, per licensing; snippet policy is an open
   question below.)
4. **`get_feature_catalog`** — the available filters **with measured
   precision/recall per feature**, so a protocol can justify using one.

Filter semantics matter for PRISMA integrity: a server-side filter changes the
candidate pool, so the ledger records the filter expression and both counts
(matched / after filters) in `search_log.json`, and filtered-out studies appear
in PRISMA as "excluded by index filter (feature X)". Recommendation to the
team: features advertise themselves as `exclusion-grade` (measured precision
≳99%, e.g. "no coordinates anywhere in the paper") or `triage-grade`
(ranking/batching only); the skills refuse to hard-exclude on triage-grade
features.

Write side (Compose):

5. **`create_protocol` / `amend_protocol`** — the server-side mirror of §3:
   immutable versions, server timestamps, user confirmation in the web UI
   replacing the local TTY gate. This is what makes compose mode a genuine
   pre-registration (a local file cannot prove *when*).
6. **`push_decisions(stage, lines, protocol_version)`** — idempotent (keyed on
   decision hash), batched; returns a receipt hash the ledger stores beside the
   decision (two-way provenance).
7. **`push_studyset` / `push_annotation` / `push_meta_result`** — mapping onto
   Compose's existing studyset/annotation/meta-analysis resources; results
   upload keeps the existing NeuroVault path.
8. **`get_project_state`** — resume a review from another machine.

Auth: the user starts on the website, which mints a project-scoped token; the
agent reads `$COMPOSE_TOKEN`; every tool enforces project scope. The token never
appears in `review.yaml`, prompts or notes.

## 9. Roadmap

**P0 — uniform interface, local testing** *(the package becomes usable)*
1. R1.2 SKILL.md consolidation; R1.4 lock file
2. R3.1–R3.5 registration, amend, stamped export *(the anti-p-hacking core)*
3. R6.1 policy block *(reliable end-to-end)*
4. R4.1–R4.5 token levers + R4.8 usage report; measure before/after on one
   project
5. R2.1–R2.4 draft-review skill + library + calibration pilot
6. R7.1–R7.2 review queue + static review page
7. R1.1/R1.3 repo split, plugin packaging, runner merge

**P1 — compose integration** *(prioritized immediately after local testing)*
8. Provider boundary (§5): compose search + `get_analyses` → records mode;
   hand §8's read-side spec to the server team early, since it gates this
9. Server-side registration (R3 mirror; §8.5)

**P2 — push-back** *(explicitly last)*
10. §8.6–8.8: decisions, studyset, meta results, receipts

**Stretch:** hybrid screening — records first, full text fetched locally for
low-confidence items only, capped fraction, recorded as `text_view: hybrid`.

## 10. Open questions

1. **Name and home** of the user-facing repo/plugin (`autonima-skills`?), and
   whether the autonima webui and the static review page should converge later.
2. **Snippet licensing**: can `get_analyses` ship table captions and short
   Methods context? Combined-mode selection quality depends on it.
3. **Compose resource mapping**: extend the existing `specification` resource
   for protocol versions, or add a new `protocol` resource?
4. **Policy defaults**: ship conservative (`stop`) or autonomous
   (`proceed`/`report`) defaults? Proposal above ships autonomous defaults
   *because* they are registered pre-run, but this is a judgment call.
5. **Subscription budgeting**: after R4, publish a "what fits in a Pro week"
   table (records screened / full texts / selections) from the usage reports, so
   users can size reviews before starting.

# experiments/

Exploratory work that is **not** validated against a manual meta-analysis, kept out of `projects/`
so it cannot be picked up by the benchmark tooling (`run_categories.yaml`, the cross-project report
scripts, and anything globbing `projects/*/v*.yaml`).

| | |
|---|---|
| `README.md` | this file — the §9 candidate set, cascade design and sequencing |
| `deactivation_atlas/` | the run pilot: domain-general deactivation ALE, validated against Yeo-7 |

Each experiment directory holds `scripts/` (numbered, run in order), `maps/`, `tables/`, and a
gitignored `work/` for intermediates.

To reproduce the deactivation atlas from scratch:

```bash
cd experiments/deactivation_atlas/scripts
pixi run python 01_build_candidates.py   # zero-token filter over the ACE db
pixi run python 02_build_dataset.py      # TAL->MNI, drop unusable, NiMARE Dataset
pixi run python 03_run_ale.py            # ALE + FWE montecarlo (~25 min, 6 cores)
pixi run python 04_validate_dmn.py       # cluster table + Yeo-7 enrichment
```

---

# §9 Novel use case — candidates, design, and sequencing

Planning document for the paper's forward-looking section. Status: **no candidate committed.**
Written 2026-08-30. Nothing here has been run.

---

## 1. What §9 has to do

§§1–7 validate the pipeline against manual meta-analyses. That ceilings the evaluation at the
benchmarks' own scale, resolution and consistency. §9 escapes it by demonstrating something the
benchmark set cannot contain: a meta-analysis that **manual synthesis could not have produced**.

Two acceptable shapes, both raised by the author:

- **Externally validatable** — validated against something that is *not* a gold-standard
  meta-analysis (independent modality, independent cohort, molecular atlas, or an internal ordering
  prediction).
- **Unvalidatable by construction** — and that is precisely why it is novel. The resource is the
  contribution.

## 2. Design principles

Stated by the author, in priority order:

1. **Hard and tedious for manual synthesis** — ideally requiring analysis-level selection *and* clean
   screening, the emotion_regulation shape.
2. **Group-level or patient-vs-control selection**, which we know works well.
3. **Start cheap.** High precision preferred over recall; filter clearly-unrelated papers out early
   even at the cost of true positives. Large numbers absorb the loss. Tokens are the budget.
4. **Willing to spend more if the idea is interesting.** Cheapness is a starting constraint, not a
   ceiling.
5. **Annotation is the most interesting and unique capability** and has the most potential —
   screening is worth showing off to the extent it produces a unique novel result, but annotation is
   the crown jewel.

Note that 3 and 5 pull in opposite directions, and the sequencing in §7 below is built around that
tension.

## 3. What the evidence says we can lean on

| capability | evidence from this project |
|---|---|
| analysis-level selection on crossed labels | ER goal × strategy: **+0.244** annotation gain, best in corpus |
| group-contrast selection | schema has `groups` / `group_id` first-class; **22.6%** of parsed analyses carry group/contrast wording |
| direction (activation vs deactivation) | schema normalizes `greater` / `lesser`; **2.7%** of analyses carry deactivation wording |
| direction recoverable *for free* | **62.8%** of coordinate rows carry a signed statistic; **3.1%** of those are negative |
| screening a broad pool ≈ hand curation | decomposition Q2: ER +0.017, social +0.002 |
| own search is near-free | decomposition Q3: +0.008 to +0.044 |
| analysis-level recall | 0.71–0.96 across eight projects (dementia excluded, see §5) |

The pipeline's real edge is **reading tables and making per-analysis distinctions at scale**. The
novel use case should therefore be one where the *unit of interest lives inside the table*, not in
the abstract — that is exactly what makes manual synthesis collapse.

## 4. The NeuroStore reframe

**NeuroStore holds 40k+ papers with parsed analyses and full text.** Retrieval and parsing — which
consumed most of the effort in this project — become free. The only remaining cost is **tokens on
screening and annotation**.

That inverts the optimisation. The goal is no longer "find every paper"; it is **spend nothing on
the 90% that can be excluded structurally, and reserve the model for adjudication.**

### The cascade

**Stage 0 — zero tokens, structured filters.** Field matching and regex over data NeuroStore already
holds:

- has ≥1 parsed analysis with coordinates in a known space
- signal-specific filter (e.g. for deactivation: a **negative signed statistic** on ≥1 coordinate
  row, OR direction wording in the analysis name/description — the union, since the two overlap
  imperfectly)
- not a review or case report

**Stage 1 — cheap model, title + abstract only.** One call per surviving paper, small output, tuned
for precision: when in doubt, drop.

**Stage 2 — annotation on candidate analyses only.** One call per study covering all its candidates
(`multi_analysis`, already the default), and **omit `study_fulltext`** — it dominates prompt size
(see autonima#59) and for most analysis-level judgments the analysis name plus table caption
suffices. This is the single largest token lever available.

Guiding principle: **structured data does the filtering; the model only adjudicates.**

### A calibration step that is also a result

Before touching NeuroStore, run the stage-0 filter against the nine benchmark projects **where gold
labels already exist**, and measure what precision/recall a zero-token filter achieves. This yields:

- a defensible operating point instead of a guessed threshold
- a genuine methods finding — *how much of screening can be done with no model at all?* — which
  strengthens §1's "screening is cheap" argument independently of §9

Costs no LLM calls and de-risks the main run.

## 5. Validation ladder (no gold meta-analysis required)

1. **Independent modality, independent data** — ENIGMA case-control structural effect maps; HCP /
   UK Biobank task contrasts; resting-state networks.
2. **Molecular atlases** — PET receptor/transporter maps via `neuromaps` / JuSpace.
3. **Internal ordering predictions** — dose-response, difficulty gradients, monotonicity.
   Self-validating without any external map.
4. **Unvalidatable by construction** — the resource is the contribution.

## 6. Candidates

All five retained. Ranked within each objective in §7.

### A. Deactivation atlas

**Question.** Where does the brain systematically *deactivate*, by task domain, and does the
domain-general deactivation map recover the default mode network?

**Why manual synthesis cannot do it.** Meta-analyses systematically discard deactivations — documented
inside our own corpus, where problem_solving's benchmark states *"the benchmark includes only BOLD or
rCBF signal INCREASES."* Deactivations sit in the same tables as activations, distinguished only by
sign or a minus in a contrast label, so extracting them requires per-analysis direction assignment
across thousands of tables.

**What autonima uniquely adds.** Direction normalization at analysis level. Stage 0 is nearly free
here because the selection signal is already in the parsed data.

**Validation.** Strongest of any candidate. Predict the domain-general deactivation map ≈ DMN, test
against resting-state networks from HCP/Biobank — independent data, independent modality, zero
literature overlap. Secondary prediction: deactivation extent scales with task difficulty, testable
within the corpus.

**Annotation showcase.** *Moderate.* Much of the work is the free filter. But the hard adjudication —
true deactivation vs a reversed contrast (`B > A` reported as negative) vs a negative correlation —
is a judgment no regex can make, and it is exactly what determines precision.

**Killer risk.** Publication bias: deactivations may be under-reported. Measurable, and itself a
publishable finding about the literature.

### B. Treatment-effect maps validated against PET receptor atlases

**Question.** Where does task-evoked activity change pre- vs post-treatment, per drug class?

**Why manual fails.** Requires selecting *within-group pre/post* contrasts while rejecting
between-group ones, plus clean RCT screening — analysis-level selection layered on strict study
screening. The ER shape, one notch harder.

**Validation.** Compare each drug class's map to its molecular target's density map (SSRI →
5-HT transporter, antipsychotic → D2) using `neuromaps`. Genuinely external, independent molecular
imaging, established method.

**Annotation showcase.** *Strong.* Design classification (pre/post, within-group, drug class) is
pure analysis-level annotation with no free structural proxy.

**Killer risk.** Cell counts — per-drug-class N may be too thin. Needs a feasibility count first.

### C. Transdiagnostic patient > control, validated against ENIGMA

**Question.** Is there a transdiagnostic core of case-control functional difference, and do
per-disorder maps rank-order like ENIGMA's structural ones?

**Why manual is hard.** Screening the whole clinical fMRI literature and selecting between-group
contrasts at analysis level. Plays to the 22.6% supply and to group-level selection.

**Validation.** ENIGMA publishes per-disorder case-control effect maps from mega-analyses of raw
data — different modality, different subjects, no literature overlap.

**Annotation showcase.** *Moderate.* Mostly a single group-contrast label per analysis.

**Killer risk.** **Novelty.** Goodkind et al. 2015 and McTeague et al. did versions of this. The
contribution would be scale and automation, not the question — a weaker §9 than "no one could do
this at all."

### D. Crossed factorial: emotion regulation × clinical status

**Question.** How does regulation-related activity differ by goal (decrease/increase) × strategy ×
clinical status, in cells nobody can assemble by hand?

**Why manual fails.** Cell counts are too small to reach without exhaustive screening of the whole
regulation literature, and each cell needs three crossed analysis-level labels.

**Validation.** *Weak.* No external map; only ordering predictions (e.g. patients deviate from
controls in prefrontal engagement).

**Annotation showcase.** *Strongest of the five.* It is a direct extension of the corpus's
best-performing annotation case (+0.244), with an extra crossed factor.

**Killer risk.** The validation half of the §9 brief is barely satisfied.

### E. The unreported-space map

**Question.** Where are coordinates systematically *absent* — susceptibility dropout, field-of-view
truncation, cerebellar exclusion — written into the literature's coordinate record?

**Why manual fails.** Requires the full coordinate record at scale; no manual synthesis has one.

**Validation.** Unvalidatable against any meta-analysis, since none exists. Partially checkable
against known susceptibility maps.

**Annotation showcase.** *Weak.* Almost entirely structural analysis of the coordinate record.

**Killer risk.** Reads as a methods paper rather than a meta-analysis, and may not carry §9.

## 7. Ranking depends on the objective

**If optimising for cheapest-first (principle 3):** A → C → B → E → D.
A's selection signal is already in the parsed data, so stage 0 does most of the work for free.

**If optimising for annotation showcase (principle 5):** D → B → A → C → E.
D and B require genuine per-analysis judgment with no free structural proxy.

**Recommended sequence, which satisfies both:**

1. **Run the stage-0 calibration** on the nine benchmark projects. No LLM cost. Establishes whether
   the free filter is good enough to build on, and produces a methods result either way.
2. **Pilot A (deactivation atlas)** — cheapest path to a genuine novel result, validates the whole
   cascade end to end, and has the strongest external validation available.
3. **Escalate to B (treatment maps)** once the cascade is proven. B is the best combination of
   annotation showcase and external validation, and shares screening infrastructure with A. If cell
   counts survive, its molecular validation is the more striking result.
4. **Hold D** unless a validation route appears. It is the best annotation demonstration but the
   weakest evidence, and §9 needs both.

## 8. Caveats to write into the paper now

**Generalisability.** Most users will not have a 40k pre-parsed corpus at release. State it plainly,
and frame §9 as *"what becomes possible when retrieval and parsing are solved"* — a demonstration of
the ceiling, with §§1–7 carrying the claim about the normal path. Framed that way the caveat becomes
the point. It also makes autonima#6 (NeuroStore DataPond) and autonima#27 (ingest ACE / ns-pond) the
roadmap items that close the gap.

**Precision-first changes what a null means.** Deliberately trading recall for tokens means a weak or
absent DMN correspondence cannot distinguish "deactivations do not converge" from "our filter dropped
the informative ones." Pre-commit to a sensitivity check: rerun stage 0 at a looser threshold on a
random subsample and confirm the map is stable.

## 9. Blockers and open questions, in order

1. **autonima#6 / #27 are open** — the NeuroStore ingest path does not exist yet. Much smaller than
   retrieval + parsing, but not zero, and it gates everything.
2. **Does NeuroStore preserve signed statistics?** Our ACE export does, at 62.8% coverage. If
   NeuroStore drops the sign, stage 0 for candidate A falls back to wording alone and the free filter
   weakens substantially. **Verify before committing** — it is the difference between a cheap sweep
   and an expensive one.
3. **autonima#60 duplication** matters more here than for maps. The signed-statistic sample above
   showed pmid 10355679 twice with identical values; a doubled deactivation table would double-weight
   in exactly the way the dementia test showed CBMA tolerates but should not.
4. **Feasibility counts for B** — per-drug-class N is unknown and is the single fact that decides
   whether B is viable.

---

# PILOT RESULT — deactivation atlas, local data only, zero LLM tokens (2026-08-30)

Steps 1–2 of the recommended sequence were run. **It worked, and it validated.**

## What was built

Source: `articles/ace_outputs/sqlite.db` only — no NeuroStore, no new retrieval, no new parsing, and
**no LLM calls at any stage**. The whole pipeline below is stage 0.

    activations JOIN tables (84,668 rows, 4,815 tables, 2,534 PMIDs)
      free filter: deactivation wording in caption/notes  OR  >=1 negative signed statistic
        -> 266 candidate tables, 194 studies, 3,053 coordinates
      dedupe identical points within study (autonima#60)      -> 2,681  (-12%)
      drop unknown coordinate space (26) and <3 coords (25)   -> 143 studies
      Talairach -> MNI for 47 studies (nimare.utils.tal2mni)
    ALE, FWE montecarlo, 1,000 iterations, cluster-mass       -> 2,081 significant voxels

## The map recovers the default mode network

Eight significant clusters (z>1.65, k>50 voxels):

| # | x, y, z | peak z | region |
|---|---|---|---|
| 1 | −48, −64, 32 | 3.09 | left angular gyrus / TPJ — **DMN** |
| 2 | −12, 48, 38 | 3.09 | left dorsomedial PFC — **DMN** |
| 3 | 22, −8, −18 | 3.09 | right amygdala / hippocampus |
| 4 | 40, 18, −2 | 3.09 | right insula / IFG |
| 5 | −2, 4, 56 | 2.88 | SMA / dorsal ACC |
| 6 | −6, −56, 36 | 2.75 | precuneus / PCC — **DMN** |
| 7 | −4, 48, −10 | 2.41 | ventromedial PFC — **DMN** |
| 8 | −6, 38, 10 | 1.79 | anterior medial PFC — **DMN** |

Quantified against Yeo-7 (`thick_7`), enrichment = share of significant voxels ÷ share of cortex:

| network | sig voxels | % of sig | enrichment |
|---|---|---|---|
| **Default** | 1062 | **51.0%** | **2.23×** |
| VentAttn / Salience | 384 | 18.5% | 1.78× |
| Frontoparietal | 106 | 5.1% | 0.34× |
| Limbic | 18 | 0.9% | 0.10× |
| Somatomotor | 12 | 0.6% | 0.04× |
| Visual | 0 | 0.0% | **0.00×** |
| DorsalAttn | 0 | 0.0% | **0.00×** |
| (outside cortical atlas) | 499 | 24.0% | — |

**The prediction was DMN, and the prediction held.** Half the significant voxels fall in the DMN at
2.23× enrichment, while the two canonical *task-positive* networks — visual and dorsal attention —
contain **exactly zero** significant voxels. The 24% outside the atlas is subcortical (cluster 3,
amygdala/hippocampus), which Yeo's cortical parcellation does not cover.

## Why this matters for §9

- **It is novel by construction.** Meta-analyses discard deactivations as a matter of routine —
  documented inside our own corpus, where problem_solving's benchmark states it "includes only BOLD
  or rCBF signal INCREASES". There is no gold-standard deactivation meta-analysis to validate
  against, which is exactly why the DMN prediction is the right test.
- **It validated externally.** Yeo-7 comes from resting-state data in an independent cohort — no
  literature overlap, different modality, and the prediction was registered in advance in §6A above.
- **It cost nothing.** Zero tokens. The entire selection was structured filtering over data already
  on disk, which is the strongest possible demonstration of the cascade's stage 0.

## Honest limitations

1. **The corpus is not a neutral sample of the literature.** These 143 studies come from the nine
   benchmark projects (cue_reactivity, social, executive_function, emotion regulation, …), so the
   map is a domain-general deactivation map *of those domains*. Clusters 3–5 (amygdala, insula,
   SMA) plausibly reflect that composition rather than a general task-negative response. A neutral
   sweep needs NeuroStore.
2. **The filter is crude and precision-first by design.** Caption wording OR a negative statistic.
   A negative statistic can mean a reversed contrast (`B > A`) or a negative correlation rather than
   a deactivation. This is the adjudication step an LLM would do, and it was skipped here — so the
   present map is the *zero-token floor*, not the ceiling.
3. **No spin test.** Enrichment is reported against network size, which does not account for spatial
   autocorrelation. A spin/spatial-permutation test is the rigorous version and should be run before
   publication.
4. **Sample-size metadata was stubbed** at n=20 for the ALE kernel, since ACE does not carry it. This
   affects kernel width and therefore cluster extent, though not the network-level conclusion.

## What this licenses

The cheap path works end to end and produces a validated, novel result. Two clear next moves:

- **Add the LLM adjudication layer** to the same 266 candidate tables and measure how much precision
  it buys over the zero-token floor. That is a direct, quantified demonstration of what annotation
  contributes — the capability flagged as most interesting — on a task where the free baseline is
  already known.
- **Rerun on NeuroStore** for a neutral, much larger sample, once autonima#6/#27 land.

Artifacts: `experiments/deactivation_atlas/` — FWE-corrected z map, cluster table, NiMARE dataset,
summary JSON, and the four scripts that produced them.

---

# ANNOTATION EXPERIMENT — what analysis-level selection buys over the free filter (2026-08-30)

The pilot above established a zero-token floor. This is the controlled comparison against it: same
substrate, same downstream pipeline, one difference — regex versus LLM adjudication.

## Design

Both arms are built from **autonima's parsed analyses** (not the raw ACE tables the pilot used), so
the only thing that differs is how analyses are selected:

- **Arm A, floor** — analyses whose own name/description matches the tight direction lexicon.
- **Arm B, LLM** — analyses drawn from a *looser* candidate pool and adjudicated by the model.

Model `gpt-5.6-luna` with `reasoning_effort="none"`. This model **rejects function tools unless that
parameter is passed**, and no autonima call site sends it — `annotation/client.py:305`,
`screening/openai_client.py:92` and `coordinates/openai_client.py:92` all use `functions=` and pass
no model-specific parameters, so the model is unusable across the whole pipeline. Filed as
[autonima#62](https://github.com/neurostuff/autonima/issues/62). The annotation stage here is
therefore a direct API call in `06_annotate.py` rather than an autonima run, which also means it
sidesteps caching, retries and cost accounting — a workaround, not a pattern to copy.

## Cost — measured, not estimated

    957 studies, 2,650 candidate analyses
    473,595 input tokens / 126,960 output tokens
    $0.37, 178 seconds at 10 workers, reasoning tokens = 0

## Selection

**150 of 2,650 analyses included (5.7%).** Precision is the point, and it holds in both directions.
Included: *"Stroop deactivation"*, *"Deactivations (hypoactivations; all words < baseline)"*,
*"Task-Negative Regions (negative activation)"*. Rejected — each a case the regex floor swallows:

- negative *correlations* with craving (a correlation is not a deactivation)
- between-group contrasts (`AUD < CTL`, `Untreated > Treated`)
- active-condition contrasts (`drug < neutral cue`, `Alcohol > No-Alcohol`)
- gPPI connectivity analyses

## Result: fewer studies, stronger and much cleaner map

| | floor (regex) | LLM (annotated) |
|---|---|---|
| studies | 92 | **31** |
| coordinates | 1,326 | 543 |
| significant voxels | 880 | **1,038** |
| clusters | 5 | 4 |
| **% of sig voxels in DMN** | 54.7% | **90.9%** |
| **DMN enrichment** | 2.39× | **3.98×** |

Yeo-7 enrichment, full breakdown:

| network | floor | LLM | Δ |
|---|---|---|---|
| **Default** | 2.39× | **3.98×** | **+1.59** |
| VentAttn / Salience | 2.10× | **0.00×** | −2.10 |
| Frontoparietal | 1.14× | **0.00×** | −1.14 |
| DorsalAttn | 0.00× | 0.12× | +0.12 |
| Visual | 0.00× | 0.03× | +0.03 |
| Somatomotor | 0.00× | 0.00× | 0.00 |

Cluster peaks tell the same story. The floor arm's strongest clusters are anterior cingulate
(−8, 44, 4), pre-SMA (4, 22, 38) and right insula (36, 22, −2) — **task-positive salience regions**,
which is what between-group and active-condition contrasts contribute. The LLM arm's are precuneus
(−6, −52, 28), vmPFC (6, 44, −4), left angular gyrus (−50, −66, 28) and anterior mPFC (4, 56, 10) —
**four canonical DMN nodes and nothing else**.

## What this demonstrates

**Annotation removed two-thirds of the studies and produced a stronger map.** The floor arm's
salience and frontoparietal signal is not weak evidence of deactivation; it is *contamination* — the
regex admits contrasts that are not task-induced deactivations at all, and those contribute
task-positive peaks. Stripping them raises DMN purity from 54.7% to 90.9% and eliminates
salience/frontoparietal entirely.

This is the cleanest available answer to what analysis-level selection is worth: **$0.37, and it
converts a mixed map into a specific one.**

## Limitations

1. **The arms are unequal in power** (92 vs 31 studies) and unequal by construction, since the LLM
   is stricter. The enrichment ratio is partly power-robust and the LLM arm has *more* significant
   voxels despite fewer studies, so the direction is not a power artifact — but a matched-N
   comparison (subsample the floor to 31) would be the rigorous version.
2. **34 of the 72 LLM-selected studies were dropped for unknown coordinate space** — a larger loss
   than the selection itself, and a data-quality limit rather than a method one.
3. **No spin test** on either arm; enrichment is against network size only.
4. **The corpus is still the nine benchmark projects**, so this is not a neutral sample of the
   literature.

Artifacts: `deactivation_atlas/maps/{arm_floor,arm_llm}/`, `tables/clusters_arm_*.csv`,
`tables/arm_comparison.json`. Scripts `05`–`08`.

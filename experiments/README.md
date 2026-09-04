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

---

# §9 REASSESSMENT — B and C, and the topic decision (2026-08-31)

Candidate A (deactivation atlas) is **parked**: the built result stands as proof of concept and can be
revived, but it is not compelling enough to carry §9. B and C are more compelling because they can be
externally validated and represent automation of a much harder outcome — ENIGMA is a
multi-consortium effort.

## Candidate B — DROPPED

Supply-limited even with unlimited fetching. PubMed, structural/functional pre-post treatment fMRI:

| drug class | any fMRI | + pre/post | est. usable at ~12% |
|---|---|---|---|
| SSRI / antidepressant | 600 | 207 | ~25 |
| antipsychotic | 320 | 93 | ~11 |
| stimulant (DAT) | 239 | 33 | ~4 |
| ketamine / NMDA | 190 | 26 | ~3 |
| opioid / MOR | 116 | 28 | ~3 |
| psilocybin / 5HT2A | 141 | 19 | ~2 |
| benzodiazepine / GABA | 60 | 1 | ~0 |

Only SSRI clears ALE's >=20, barely. But B's validation needs SEVERAL receptor-mappable classes —
one class against one PET atlas is an anecdote, not a pattern. The literature does not exist at the
granularity the validation requires, so fetching cannot fix it. `neuromaps` is also not installed.

## Candidate C — PURSUE, structural

Two corrections to the earlier assessment, both from the author:

**1. Structural meta-analysis is already proven here, so the modality-mismatch risk dissolves.**
Three of the nine benchmark projects ARE structural: dementia (104 studies, explicit `structural`
r2=0.249 and `functional` r2=0.217 columns), vbm_of_substance_use (106 studies across six drug-class
strata, alcohol r2=0.476), vbm_of_ptsd (14 studies, r2=0.456). Structural-vs-structural is
modality-matched to ENIGMA — no functional/structural correspondence assumption needed. My earlier
concern was an artifact of assuming C had to be functional.

**2. The validation target is packaged, not a manual extraction.** The ENIGMA Toolbox
(github.com/MICA-MNI/ENIGMA) ships **191 case-control summary-statistic CSVs** in exactly the needed
form — Cohen's d per Desikan-Killiany ROI with CIs and N:

    CorticalThickness,Structure,d_icv,se_icv,low_ci_icv,up_ci_icv,n_controls,n_patients,pobs,fdr_p
    MDDadult_casevsCN,L_bankssts,-0.058,0.031,-0.118,0.002,7571,1781,0.059,0.173

ENIGMA is 50 working groups over 26 diseases. Disorder families with case-control stats: 22q11, ADHD
(4 age strata), epilepsy (5 variants), anorexia, antisocial/conduct, autism (mega AND meta), bipolar
(type I/II, adolescent/adult), MDD (6+ strata), OCD (adult/pediatric x medicated/unmedicated x
anxiety/depression), Parkinson's (by Hoehn-Yahr stage), psychosis/CHR, schizophrenia.

**The stratified contrasts are the opportunity.** ENIGMA publishes not just "OCD vs controls" but
*medicated* OCD, *first-episode* vs *recurrent* MDD, Parkinson's by disease stage, bipolar type I vs
II. Each distinction lives in the methods section and requires analysis-level selection to reproduce
— exactly what annotation is for, and exactly what makes it infeasible by hand.

## THE DECISIVE CRITERION: effect size, not supply

CBMA detects **convergence of reported peaks**, so a disorder with tiny case-control effects yields
few significant peaks to converge on, regardless of how many papers exist. Measured from the ENIGMA
CSVs against PubMed supply:

| target | max abs(d) | median abs(d) | est. usable | verdict |
|---|---|---|---|---|
| **left TLE** (subcortical) | **1.728** | — | ~21 | large effects, marginal N |
| **anorexia** (thickness) | 0.925 | **0.528** | ~20 | strong effects, novel, marginal N |
| schizophrenia (thickness) | 0.536 | 0.317 | **~229** | moderate effects, comfortable N |
| antisocial (thickness) | 0.160 | 0.050 | ~90 | too small |
| Parkinson's (subcortical) | ~0.136 | — | ~183 | **too small — a supply trap** |
| MDD (thickness) | ~0.058 | — | ~90 | too small |

Parkinson's has the best supply of any candidate and is unusable. Supply was never the criterion.

## RECOMMENDED PILOT: left vs right temporal lobe epilepsy

Novel to this project (not one of the nine benchmarks), and the sharpest available qualitative test:

    ENIGMA left hippocampus, Cohen's d
      left TLE  (tlemtsl):  -1.728   massive
      right TLE (tlemtsr):  -0.169   n.s. (p=0.056)

A **10x dissociation in the same structure between two strata.** If our left-TLE map shows left
hippocampal/temporal convergence and our right-TLE map does not, the replication is unambiguous by
eye — which is what the author asked for ("qualitative replication for now").

Four reasons this is the right pilot:

1. **Self-controlling.** Right TLE is an internal negative control for left TLE's positive finding.
   Most validations lack an internal negative; this one has it by construction.
2. **Large effects offset marginal N.** Hippocampal sclerosis is gross pathology, so reported peaks
   are highly consistent — ~20 studies per side is adequate where MDD's d=0.06 would need hundreds.
3. **Genuinely novel.** No published meta-analysis compares left vs right TLE coordinates against
   ENIGMA.
4. **Lateralization is falsifiable and obvious.** A left/right dissociation either appears or it does
   not; there is no ambiguous middle to argue about.

Supply: left TLE 173 hits (~21 usable), right TLE 153 (~18), TLE any 459 (~55), MTS/HS 206 (~25).

**Scale-up options if the pilot works:** schizophrenia (comfortable ~229 studies, moderate effects) or
anorexia (median d=0.53, rarely meta-analyzed, ~20 studies).

## Deferred by decision

- **Quantitative ROI comparison design.** ALE measures spatial convergence of reported findings, not
  effect magnitude, so mapping ALE output onto ENIGMA's per-ROI Cohen's d is non-trivial. Deferred:
  qualitative replication is sufficient for the first pass, and the author will judge the maps
  directly.
- **PTSD and addiction** have ENIGMA working groups but no case-control CSVs in the Toolbox, so our
  two best-populated local projects have no packaged target. This is part of why the pilot goes to a
  novel topic instead of reusing a benchmark corpus.

---

# TLE PILOT — FEASIBILITY RESULT (2026-08-31)

Two runs. Both produced 2 usable studies. **The pilot failed, and the failure is the useful result**:
it establishes the yield rates that determine which ENIGMA replication targets are possible at all.

## v1 — the ROI/coordinate problem

Broad structural search ("gray matter", "atrophy", "hippocampal volume"):

    search 1,114 -> abstract 467 included -> full text 69 -> fulltext 2 included

**60 of 69 full-text rejections were one criterion: no stereotactic coordinates.** The reasons were
consistent -- "ROI-based structural analyses (FreeSurfer cortical thickness and subcortical
volumes)", "atlas-based".

This is structural, not a config bug. The TLE structural literature is ~59% ROI/parcellation-based
(261 ROI vs 182 voxel-wise on PubMed), and ROI studies report per-region values rather than
coordinates. **That is the same methodology ENIGMA uses** -- which is exactly why ENIGMA could
mega-analyse it, and exactly why a coordinate-based meta-analysis cannot consume it. The closer a
literature sits to ENIGMA's method, the less usable it is for CBMA.

## v2 — require a voxel-wise method

Search and screening both now require VBM / voxel-wise / SPM. Pool 1,114 -> 273.

    search 273 -> abstract 105 included -> full text 14 -> fulltext 2 included

Narrowing worked on the intended axis: full-text inclusion given text rose from 2.9% to 14%. But
retrieval became the binding constraint -- only 14 of 105 abstract-included studies had text.

## Measured rates (PMC only)

| stage | rate |
|---|---|
| abstract inclusion | 39.0% |
| PMC full-text coverage | 14% (38 of 269 articles) |
| coordinate table present, given PMC full text | 37% (14 of 38) |
| autonima extracted coordinates, given a table | ~71% (10 of 14) |
| full-text inclusion, given coordinates | 67% |

Net: **~0.7% of searched papers become usable on PMC alone; ~6.9% with full retrieval.**

## Projections, and the two conclusions

| literature | VBM papers | PMC-only | full retrieval | per side |
|---|---|---|---|---|
| **schizophrenia** | 1,042 | 8 | **72** | 36 |
| bipolar | 340 | 3 | 19 | 9 |
| **TLE** | 273 | 2 | **19** | 9 |
| autism | 272 | 2 | 15 | 8 |
| ADHD | 199 | 1 | 11 | 6 |
| OCD | 178 | 1 | 10 | 5 |

**1. PMC alone is not viable for any target.** Every literature caps at 1-8 usable studies. The
Elsevier-via-campus-tunnel top-up is mandatory, not an optimisation -- it is what moves the column
from "PMC-only" to "full retrieval".

**2. TLE is too small.** Even with perfect retrieval it yields ~19 studies, ~9 per side, against the
>=20 per stratum an ALE/MKDA needs. The lateralization design is elegant and self-controlling, and
the literature does not exist to support it. **Only schizophrenia clears the bar** (~72 usable), and
it has no equivalent internal negative control -- so the trade is statistical power against design
elegance.

## Secondary findings

- **A 17% false-exclusion rate at full-text screening.** 2 of 12 exclusions said "no coordinates"
  for studies where pubget *had* extracted a coordinate table (PMIDs 34817680, 26594628). Not
  systemic -- the other 8 genuinely have no coordinate table -- but it means the screener is not
  reading pubget's extracted tables reliably. Related to autonima#44.
- **pubget table extraction works well.** All 46 retrieved articles had non-empty `tables/`
  directories, 246 table files. The bottleneck is PMC *coverage*, not extraction.
- **gpt-5.6-luna ran the annotation stage inside autonima** for the first time, via the autonima#62
  fix and `AUTONIMA_MODEL_PARAMS`.

## What this means for the §9 decision

The ENIGMA replication is not dead, but it is narrower than proposed: **one disorder
(schizophrenia), full retrieval required, and no internal control.** Before committing, the cheap
next step is to re-run this same pilot on schizophrenia with PMC only -- projected 8 usable studies,
which is enough to confirm the rates hold on a larger literature without waiting for the tunnel.

---

# SCHIZOPHRENIA / ENIGMA PILOT — a qualitative replication (2026-08-31)

After the TLE pilot showed that literature size, not design elegance, is the binding constraint, the
target moved to schizophrenia: 5x the VBM literature and higher PMC coverage (46% vs 33%).

## Search widening

The first config scored 916 hits. Measuring each clause's cost showed the disease terms were only
mildly restrictive and two **redundant structural filters** were doing the damage:

    as run                                        916
    wider disease clause (+psychosis/psychotic)  1025
    ...also drop "Humans"[MeSH]                  1092
    ...also drop the MRI clause                  1190
    ...drop BOTH MRI and Humans                  1370
    ...plus a wider method clause                1506

`AND (MRI OR "magnetic resonance imaging"[Mesh])` is redundant -- a VBM paper is MRI by definition --
and silently dropped papers that do not say "MRI" or are not MeSH-indexed. `"Humans"[MeSH]` likewise,
and it penalises recent unindexed work. Both removed; the voxel-wise method requirement was KEPT,
since that is the filter that determines coordinate yield. Pool 916 -> **1,434**. Re-running cost
only 505 new abstract screenings, since the abstract cache is keyed per study.

## Funnel (PMC only, no Elsevier top-up)

    search 1,434 -> abstract 492 included -> full text 136 -> fulltext 32 included
                 -> 21 studies parsed (60 analyses, 257 coordinates)
                 -> scz_atrophy column: 20 analyses across 14 studies

Projected 33 usable, actual 32. The antipsychotic-naive column found 1 study, correctly rejecting
59 analyses -- the selection works, the literature is not there without full retrieval.

## Result: MKDA, 14 studies, and it looks like schizophrenia

MKDA density, Monte-Carlo FWE, 1,000 iterations. Six significant clusters:

| peak (MNI) | z | Harvard-Oxford label |
|---|---|---|
| −40, −12, 16 | 3.09 | Central Opercular (left insula/STG) |
| −56, −6, 4 | 2.65 | Central Opercular (left STG) |
| −36, 14, 8 | 2.65 | Frontal Operculum (left insula) |
| −30, 22, −8 | 2.07 | Frontal Orbital |
| 54, 0, 2 | 2.01 | Central Opercular (right insula/STG) |
| 0, 36, −18 | 1.98 | Frontal Medial |

Against ENIGMA's 12 strongest cortical-thickness effects:

| ENIGMA ROI | d | recovered? |
|---|---|---|
| R/L fusiform | −0.536 / −0.491 | **no** |
| L/R inferiortemporal | −0.449 / −0.439 | no |
| L middletemporal | −0.444 | no |
| L superiortemporal | −0.440 | **yes** (clusters 1, 2) |
| R superiortemporal | −0.438 | **yes** (cluster 5) |
| L superiorfrontal | −0.425 | no |
| R parsopercularis | −0.424 | **yes** (cluster 3 region) |
| L insula | −0.408 | **yes** (clusters 1, 3, 4) |
| R insula | −0.406 | **yes** (cluster 5) |
| L lateralorbitofrontal | −0.398 | **yes** (clusters 4, 6) |

**6 of ENIGMA's top 12 ROIs recovered from 14 studies on PMC alone**, with bilateral insula,
bilateral superior temporal and orbitofrontal all hit. The map is also a textbook schizophrenia VBM
result independent of ENIGMA, which is reassuring about the pipeline rather than about the target.

**The misses are informative.** Fusiform is ENIGMA's single strongest effect and we do not recover it
at all, nor inferior/middle temporal. Two candidate explanations, not yet distinguished: power (14
studies), or a modality difference -- ENIGMA measures cortical *thickness* over the whole ribbon,
while VBM peak coordinates cluster where volume loss is focal. Worth checking whether the fusiform
gap persists at 60+ studies after the tunnel top-up, because if it does it is a substantive finding
about what CBMA can and cannot recover from a thickness-based benchmark.

## Where the studies are lost

Three independent bottlenecks, now quantified:

    492 abstract-included
     -> 136 have full text          PMC OA boundary        -- the Elsevier tunnel fixes this
     ->  32 pass full-text screening
     ->  21 have parseable coordinates   supplementary gap -- nothing currently fixes this
     ->  14 in the scz_atrophy column

The 32 -> 21 loss was diagnosed and is NOT a parser bug: of the 11 lost, 8 have pubget-extracted
tables containing no coordinates at all, 2 are genuine parser misses, 1 had no pubget directory. The
8 are papers whose full text discusses MNI coordinates but whose coordinate tables are not in
pubget's output -- most plausibly in supplementary material. That is autonima#36, now quantified at
**~25% of otherwise-eligible studies**.

Closing both bottlenecks would plausibly take the same 1,434-paper search from 14 studies to 60-80 --
the difference between a pilot and a publishable map.

---

## Update 2026-09-03: bottlenecks closed, and the temporal gap is real

Both bottlenecks were closed and the prediction above was beaten: **115 analyses in the
`scz_atrophy` arm, against 14 in the pilot**.

### How the corpus grew

| route | result |
|---|---|
| Elsevier Article API, through the institutional tunnel | 182 of 183 fetched, 105 with coordinates |
| browser scraping with the new `--proxy` flag | 119 of 174 retrieved |
| manual download | +30 |
| **ACE corpus** | **149 articles, 51 coordinate-bearing** |

| | pilot | now |
|---|---|---|
| studies in studyset | 169 | **318** |
| analyses | 294 | **706** |
| coordinate points | 3,157 | **7,617** |
| passing full-text screening | 32 | **116** |
| `scz_atrophy` analyses selected | 20 | **115** |
| `scz_atrophy_antipsychotic_naive` | 1 | **20** |

The antipsychotic-naive arm produced no map at all before and now yields one (20 experiments,
101 foci, 1 cluster) -- still thin, but no longer empty.

### The new map

MKDA (mkdadensity, Monte Carlo FWE), **1,427 foci from 115 experiments**, five clusters:

| cluster | peak (MNI) | voxels | Harvard-Oxford extent |
|---|---|---|---|
| 1 | -44, 0, 0 | 3,012 | Insular 18%, Central Opercular 16%, Frontal Orbital 9% |
| 5 | 42, 6, 0 | 2,882 | Insular 22%, Frontal Orbital 10%, Central Opercular 9% |
| 4 | 0, 46, -6 | 1,690 | Paracingulate 36%, Frontal Medial 34%, Frontal Pole 18% |
| 3 | 2, -18, 6 | 735 | **bilateral thalamus** (48% R, 44% L) |
| 2 | -2, 10, -8 | 711 | Subcallosal 25% |

Now bilateral where the pilot was left-lateralised, and clusters are roughly ten times larger.

### Validation against ENIGMA

Scored quantitatively this time -- an ENIGMA ROI counts as recovered if significant voxels cover
>= 2% of it -- rather than by reading peak labels against ROI names.

**Cortical thickness (van Erp 2018, top 12):**

| ROI | d | % of ROI significant | recovered |
|---|---|---|---|
| R insula | -0.406 | 52.7% | **yes** |
| L insula | -0.408 | 46.6% | **yes** |
| R parsopercularis | -0.424 | 32.4% | **yes** |
| L lateralorbitofrontal | -0.398 | 17.0% | **yes** |
| L superiortemporal | -0.440 | 0.8% | no |
| R/L fusiform, L/R inferiortemporal, L middletemporal, R superiortemporal, L superiorfrontal | -0.536 to -0.425 | **0.0%** | no |

**4 of 12.** This is *not* comparable to the pilot's "6 of 12": that used a lenient
peak-label reading, this uses a coverage threshold, and the pilot's map cannot be re-scored
because `meta_analysis_results/` is gitignored and has been overwritten. Treat the two numbers as
incommensurable rather than as a regression.

**Subcortical (van Erp 2016)** -- not examined in the pilot, and where the gain actually shows:

| structure | ENIGMA d | % of structure significant |
|---|---|---|
| Accumbens | -0.20 | **63.6%** |
| Thalamus | -0.31 | **25.9%** |
| Amygdala | -0.31 | 11.2% |
| Caudate | +0.06 | 7.4% |
| Putamen | +0.14 | 3.6% |
| **Hippocampus** | **-0.46** | **0.0%** |

### The finding the pilot asked for

The pilot flagged: *"Worth checking whether the fusiform gap persists at 60+ studies after the
tunnel top-up, because if it does it is a substantive finding about what CBMA can and cannot
recover from a thickness-based benchmark."*

**It persists at 115.** Fusiform, inferior temporal and middle temporal are at exactly zero
significant voxels, and superior temporal at 0.8%, with 5.75x the studies. The entire temporal
lobe is absent while insula, operculum, medial frontal, thalamus and accumbens are strongly hit.
So the gap is **not power** -- it is a modality difference. ENIGMA measures cortical thickness
across the whole ribbon, where temporal effects are largest and spatially diffuse; VBM peak
coordinates are reported where volume loss is focal, and those are different places.

Hippocampus is the sharpest case: ENIGMA's single strongest subcortical effect (d = -0.46) and we
recover none of it, while recovering 63.6% of accumbens at less than half the effect size. A
CBMA of reported peaks tracks *where authors localise focal differences*, not where the largest
group effect lies -- which is the coordinate-based limitation this repo argues elsewhere, showing
up as a concrete miss rather than an abstract one.


# Cross-Annotation Precision Failure Report

Generated: 2026-02-20 14:12 local

## Executive Summary

- Precision is lowest for `affiliation_attachment` (0.267 study-level), with high recall (0.842), indicating broad over-inclusion rather than missed signal.
- Precision is also constrained in `social_communication` (0.505) and `perception_others` (0.565), consistent with cross-annotation boundary bleed.
- Across all FP included analyses, the dominant pattern is **construct boundary blur** (co-labeled in sibling sub-annotations), followed by **example overreach** and **context dominance**.
- Prompt/code inspection shows global exclusions are configured, but operational prompt content is driven by per-annotation criteria lists; in this dataset exclusions are effectively not applied in outputs.

## Data Sources

- Evaluation reports: `/home/zorro/repos/autonima-results/projects/social/coordinates/annotation-only/reports/annotation_review_reports`
- Annotation decisions: `/home/zorro/repos/autonima-results/projects/social/coordinates/annotation-only/outputs/annotation_results.json`
- Auto annotation labels: `/home/zorro/repos/autonima-results/projects/social/coordinates/annotation-only/outputs/nimads_annotation.json`
- Manual ground truth (sliced to auto PMID universe): `/home/zorro/repos/neurometabench/data/nimads/social/merged/nimads_annotation.json`
- Pubget evidence: `/home/zorro/repos/autonima-results/projects/social/coordinates/annotation-only/retrieval/pubget_data`
- ACE fallback evidence: `/home/zorro/repos/autonima-results/articles/ace_scrape/processed`
- Prompt/processor code: `/home/zorro/repos/autonima/autonima/annotation/prompts.py`, `/home/zorro/repos/autonima/autonima/annotation/processor.py`, `/home/zorro/repos/autonima/autonima/pipeline.py`

## 1) Annotation-Level Baseline (from `overall_submeta_summary.html`)

| Annotation | Overlap PMIDs | Manual+ | Predicted+ | TP | FP | FN | TN | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `social_processing_all` | 134 | 134 | 132 | 132 | 0 | 2 | 0 | 1.000 | 0.985 | 0.992 | 0.985 |
| `affiliation_attachment` | 134 | 19 | 60 | 16 | 44 | 3 | 71 | 0.267 | 0.842 | 0.405 | 0.649 |
| `social_communication` | 134 | 57 | 101 | 51 | 50 | 6 | 27 | 0.505 | 0.895 | 0.646 | 0.582 |
| `perception_self` | 134 | 44 | 47 | 32 | 15 | 12 | 75 | 0.681 | 0.727 | 0.703 | 0.799 |
| `perception_others` | 134 | 74 | 124 | 70 | 54 | 4 | 6 | 0.565 | 0.946 | 0.707 | 0.567 |

## 2) Cohort Extraction Summary (from per-annotation HTML cards)

| Annotation | Correct docs | FP docs | FN docs | FP included analyses |
|---|---:|---:|---:|---:|
| `social_processing_all` | 120 | 11 | 5 | 33 |
| `affiliation_attachment` | 15 | 45 | 3 | 143 |
| `social_communication` | 45 | 56 | 6 | 205 |
| `perception_self` | 29 | 16 | 14 | 49 |
| `perception_others` | 63 | 61 | 5 | 244 |

## 3) Retrieval Evidence Coverage

- FP PMIDs audited: 105
- Pubget-backed FP PMIDs: 6
- ACE-backed FP PMIDs: 99
- Missing retrieval evidence: 0

## 4) Cross-Annotation Diagnostics

### 4.1 Auto co-assignment density (analysis-level, 4 sub-annotations)

| # positive sub-annotations | Analysis count |
|---:|---:|
| 0 | 34 |
| 1 | 137 |
| 2 | 262 |
| 3 | 139 |
| 4 | 65 |

### 4.2 Pairwise PMID overlap across sub-annotations (auto labels)

| Pair | Overlap PMIDs |
|---|---:|
| `affiliation_attachment` ∩ `perception_others` | 56 |
| `affiliation_attachment` ∩ `perception_self` | 28 |
| `affiliation_attachment` ∩ `social_communication` | 52 |
| `perception_self` ∩ `perception_others` | 43 |
| `social_communication` ∩ `perception_others` | 100 |
| `social_communication` ∩ `perception_self` | 35 |

### 4.3 FP cross-annotation bleed by annotation

| Annotation | FP included analyses | FP with other auto labels | Rate | Top other-label combo (count) |
|---|---:|---:|---:|---|
| `social_processing_all` | 33 | 33 | 1.000 | affiliation_attachment, perception_others, social_communication (14) |
| `affiliation_attachment` | 143 | 143 | 1.000 | perception_others, social_communication, social_processing_all (60) |
| `social_communication` | 205 | 205 | 1.000 | perception_others, social_processing_all (79) |
| `perception_self` | 49 | 49 | 1.000 | affiliation_attachment, perception_others, social_communication, social_processing_all (27) |
| `perception_others` | 244 | 244 | 1.000 | social_communication, social_processing_all (117) |

### 4.4 Criteria-application behavior (`annotation_results.json`)

| Annotation | Includes | Inclusion criteria combo(s) on includes | Exclusion criteria combo(s) (all decisions) | Inconsistent criteria flags |
|---|---:|---|---|---:|
| `social_processing_all` | 610 | `('I1', 'I2'):610` | `():637` | 1 |
| `affiliation_attachment` | 196 | `('I1', 'I2'):196` | `():637` | 2 |
| `social_communication` | 423 | `('I1', 'I2'):423` | `():637` | 0 |
| `perception_self` | 169 | `('I1', 'I2'):169` | `():637` | 0 |
| `perception_others` | 550 | `('I1', 'I2'):550` | `():637` | 1 |

- Observed pattern: all included decisions use `("I1", "I2")`; exclusion criteria are empty for all decisions in this run.

### 4.5 Metadata quality among FP included analyses

| Annotation | FP included analyses | Missing description | Generic analysis name | Zero-point analyses | Global-exclusion-like signals in name/reasoning |
|---|---:|---:|---:|---:|---:|
| `social_processing_all` | 33 | 32 (97.0%) | 13 (39.4%) | 1 (3.0%) | 9 (27.3%) |
| `affiliation_attachment` | 143 | 132 (92.3%) | 21 (14.7%) | 16 (11.2%) | 14 (9.8%) |
| `social_communication` | 205 | 184 (89.8%) | 30 (14.6%) | 12 (5.9%) | 15 (7.3%) |
| `perception_self` | 49 | 47 (95.9%) | 4 (8.2%) | 3 (6.1%) | 3 (6.1%) |
| `perception_others` | 244 | 221 (90.6%) | 35 (14.3%) | 14 (5.7%) | 18 (7.4%) |

### 4.6 Include-bias at FP-study level

| Annotation | FP docs | Mean Pred/Parsed | Median Pred/Parsed | FP docs with ratio >=0.8 | FP docs with ratio =1.0 |
|---|---:|---:|---:|---:|---:|
| `social_processing_all` | 11 | 1.000 | 1.000 | 11 | 11 |
| `affiliation_attachment` | 45 | 0.715 | 0.833 | 24 | 22 |
| `social_communication` | 56 | 0.796 | 1.000 | 37 | 32 |
| `perception_self` | 16 | 0.588 | 0.562 | 7 | 3 |
| `perception_others` | 61 | 0.936 | 1.000 | 52 | 49 |

## 5) Code-Path Diagnosis (Prompting/Execution)

- Study-level prompt can inject very long shared context (`Study Full Text`) before per-analysis decisions (`autonima/annotation/prompts.py:40`, `autonima/annotation/prompts.py:145`).
- Multi-annotation criteria sections are built from `criteria_list` only (`autonima/annotation/prompts.py:79`, `autonima/annotation/prompts.py:89`).
- Annotation processor executes custom annotations using `self.config.annotations` and passes only `annotations_to_process` to the client (`autonima/annotation/processor.py:221`, `autonima/annotation/processor.py:235`, `autonima/annotation/processor.py:307`).
- Pipeline injects global criteria mappings into `annotation_config.inclusion_criteria` / `annotation_config.exclusion_criteria` (`autonima/pipeline.py:649`, `autonima/pipeline.py:652`), but prompt construction for decisions is driven by per-annotation criteria objects and their mappings.
- Practical implication in this run: global exclusions (ROI/connectivity/between-group) are not visibly operationalized in outputs; exclusion criteria fields are empty across all decisions.

## 6) Failure Taxonomy (Evidence-backed)

| Failure mode | FP analyses | FP PMIDs | Share of FP analyses | Example PMIDs |
|---|---:|---:|---:|---|
| `construct_boundary_blur` | 674 | 105 | 1.000 | 14980212, 15006683, 15488424, 15528097, 16035037, 16055351, 16122944, 16171833 |
| `example_overreach` | 189 | 39 | 0.280 | 14980212, 15488424, 17627852, 17964185, 18486491, 18501639, 18537114, 18633788 |
| `global_criteria_dropout` | 59 | 17 | 0.088 | 17627852, 18486491, 18501639, 19439183, 20045478, 23667619, 25534111, 26143208 |
| `context_dominance` | 571 | 94 | 0.847 | 14980212, 15006683, 15488424, 15528097, 16055351, 16122944, 16171833, 16759672 |
| `analysis_granularity_noise` | 103 | 26 | 0.153 | 16759672, 17964185, 18486491, 18501639, 18633846, 18783371, 19048432, 23298748 |

Mode definitions used in this audit:
- `construct_boundary_blur`: FP analysis is simultaneously auto-labeled in sibling sub-annotations.
- `example_overreach`: reasoning/label contains broad example anchors (e.g., cooperation/competition/kinship/social comparison).
- `global_criteria_dropout`: ROI/connectivity/between-group-like terms appear but no exclusion evidence is applied.
- `context_dominance`: long full-text context plus sparse analysis metadata (missing description) for FP decision.
- `analysis_granularity_noise`: generic/underspecified analysis names (e.g., `analysis_0`, `results`, `main effect`).

## 7) Representative Case Studies (all annotations, affiliation-heavy)

Selected cases: 12 (Pubget=2, ACE=10)

### Case 1: PMID 31598216 (`affiliation_attachment`, False Positive, source=pubget)

- Title: Social cognition, behaviour and therapy adherence in frontal lobe epilepsy: a study combining neuroeconomic and neuropsychological methods
- Pred included / parsed: 7 / 7 (ratio=1.000)
- Manual sub-annotation context (PMID-level): perception_others, social_communication
- Key included contrasts: decision-making; results; PATIENTS
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - reasoning repeatedly anchors on broad example terms (e.g., cooperation/competition/kinship)
  - exclusion-like signals (group/ROI/connectivity terms) appear without exclusion enforcement

### Case 2: PMID 29079809 (`affiliation_attachment`, False Positive, source=pubget)

- Title: Differential inter-subject correlation of brain activity when kinship is a variable in moral dilemma
- Pred included / parsed: 3 / 3 (ratio=1.000)
- Manual sub-annotation context (PMID-level): perception_self
- Key included contrasts: clusters genetic > non-genetic; clusters non-genetic > genetic; moral dilemma decision task
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - reasoning repeatedly anchors on broad example terms (e.g., cooperation/competition/kinship)
  - long study context likely dominates sparse analysis-level metadata

### Case 3: PMID 25281889 (`perception_others`, False Positive, source=ace)

- Title: Alexithymic features and the labeling of brief emotional facial expressions - An fMRI study.
- Pred included / parsed: 12 / 12 (ratio=1.000)
- Manual sub-annotation context (PMID-level): social_communication
- Key included contrasts: HA>NE; AN>NE; FE>NE
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - long study context likely dominates sparse analysis-level metadata

### Case 4: PMID 25716010 (`perception_others`, False Positive, source=ace)

- Title: From personal fear to mass panic: The neurological basis of crowd perception.
- Pred included / parsed: 11 / 11 (ratio=1.000)
- Manual sub-annotation context (PMID-level): social_communication
- Key included contrasts: A. Fear>happy+neutral; B. Fear+happy>neutral; C. Fear+neutral>happy
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - analysis labels are generic/underspecified, increasing ambiguity
  - long study context likely dominates sparse analysis-level metadata

### Case 5: PMID 18501639 (`affiliation_attachment`, False Positive, source=ace)

- Title: Face-specific and domain-general characteristics of cortical responses during self-recognition.
- Pred included / parsed: 10 / 15 (ratio=0.667)
- Manual sub-annotation context (PMID-level): perception_others, perception_self, social_communication
- Key included contrasts: Face (simple effect) [Sf–Ff]; Face specific (interaction) [(Sf–Ff)–(Sn–Fn) masked by Sf–Ff]; Name specific (interaction) [(Sn–Fn)–(Sf–Ff) masked by Sn–Fn]
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - reasoning repeatedly anchors on broad example terms (e.g., cooperation/competition/kinship)
  - exclusion-like signals (group/ROI/connectivity terms) appear without exclusion enforcement

### Case 6: PMID 29432769 (`affiliation_attachment`, False Positive, source=ace)

- Title: Self-construals moderate associations between trait creativity and social brain network.
- Pred included / parsed: 8 / 11 (ratio=0.727)
- Manual sub-annotation context (PMID-level): perception_others, perception_self
- Key included contrasts: Friend- vs. Celebrity-judgments; Modulation effect of Interdependence; Functional connectivity
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - reasoning repeatedly anchors on broad example terms (e.g., cooperation/competition/kinship)
  - exclusion-like signals (group/ROI/connectivity terms) appear without exclusion enforcement

### Case 7: PMID 18633788 (`affiliation_attachment`, False Positive, source=ace)

- Title: Theory of mind broad and narrow: reasoning about social exchange engages ToM areas, precautionary reasoning does not.
- Pred included / parsed: 7 / 14 (ratio=0.500)
- Manual sub-annotation context (PMID-level): perception_others
- Key included contrasts: Cards > Rest; Rest > Cards; Social Contracts >Precautions
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - reasoning repeatedly anchors on broad example terms (e.g., cooperation/competition/kinship)
  - long study context likely dominates sparse analysis-level metadata

### Case 8: PMID 24582805 (`affiliation_attachment`, False Positive, source=ace)

- Title: But do you think I'm cool? Developmental differences in striatal recruitment during direct and reflected social self-evaluations.
- Pred included / parsed: 7 / 7 (ratio=1.000)
- Manual sub-annotation context (PMID-level): perception_others, perception_self
- Key included contrasts: Main effect of age group; Main effect of evaluative perspective; Age group × evaluative perspective
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - reasoning repeatedly anchors on broad example terms (e.g., cooperation/competition/kinship)
  - analysis labels are generic/underspecified, increasing ambiguity

### Case 9: PMID 24772075 (`affiliation_attachment`, False Positive, source=ace)

- Title: Cultural influences on social feedback processing of character traits.
- Pred included / parsed: 7 / 7 (ratio=1.000)
- Manual sub-annotation context (PMID-level): perception_others, perception_self
- Key included contrasts: MAIN EFFECT: FEEDBACK ONSET: SELF > OTHER; MAIN EFFECT: FEEDBACK ONSET: OTHER > SELF; INTERACTION: FEEDBACK ONSET: (SELF > OTHER) × (GERMAN > CHINESE)
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - reasoning repeatedly anchors on broad example terms (e.g., cooperation/competition/kinship)
  - analysis labels are generic/underspecified, increasing ambiguity

### Case 10: PMID 27494142 (`affiliation_attachment`, False Positive, source=ace)

- Title: The Neural Responses to Social Cooperation in Gain and Loss Context.
- Pred included / parsed: 6 / 6 (ratio=1.000)
- Manual sub-annotation context (PMID-level): perception_others, perception_self
- Key included contrasts: CC-A; A-CC; Gain-Loss
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - reasoning repeatedly anchors on broad example terms (e.g., cooperation/competition/kinship)

### Case 11: PMID 29582502 (`affiliation_attachment`, False Positive, source=ace)

- Title: The role of the right temporo-parietal junction in social decision-making.
- Pred included / parsed: 6 / 6 (ratio=1.000)
- Manual sub-annotation context (PMID-level): perception_others
- Key included contrasts: analysis_0; Competitive>Cooperative; Cooperative>Competitive
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - reasoning repeatedly anchors on broad example terms (e.g., cooperation/competition/kinship)
  - analysis labels are generic/underspecified, increasing ambiguity

### Case 12: PMID 24726338 (`affiliation_attachment`, False Positive, source=ace)

- Title: Reduced self-referential neural response during intergroup competition predicts competitor harm.
- Pred included / parsed: 5 / 6 (ratio=0.833)
- Manual sub-annotation context (PMID-level): perception_others, perception_self
- Key included contrasts: Self > other; Other > self; Communication > moral (collapsing across self/other)
- Why this likely inflated precision error:
  - included contrasts strongly overlap constructs labeled under sibling annotations
  - reasoning repeatedly anchors on broad example terms (e.g., cooperation/competition/kinship)
  - long study context likely dominates sparse analysis-level metadata

## 8) Prioritized Remediation Roadmap (no threshold changes)

1. **Highest impact: enforce discriminative sibling checks in prompt output contract**
   - Require per included decision: explicit `why_this_annotation_not_siblings` evidence against each sibling sub-annotation.
   - Expected gain: reduces cross-annotation bleed (`construct_boundary_blur`).
2. **Highest impact: operationalize global exclusions in annotation prompt and validation**
   - Inject global exclusions directly into decision criteria for every annotation.
   - Add post-parse guardrail: if exclusion-like signals detected and exclusion criteria not applied, downgrade to exclude/uncertain.
3. **High impact: shorten/segment study full text in annotation context**
   - Prefer analysis-local evidence first (analysis name/description + coordinate-relevant table caption/footer).
   - Include only targeted full-text snippets relevant to that analysis/table.
4. **Medium-high impact: tighten annotation criteria wording**
   - Replace broad overlapping examples with annotation-specific positive and negative counterexamples.
   - Explicitly prohibit generic social paradigms unless the contrast isolates the construct of interest.
5. **Medium impact: improve analysis granularity from parsing stage**
   - Penalize/flag generic analysis names and missing descriptions before annotation.
   - Add table-header/section parsing constraints to avoid weak labels like `results`, `analysis_0`, `main effect` without contrast detail.
6. **Evaluation instrumentation**
   - Add routine confusion diagnostics: per-annotation FP analyses that are positive in sibling annotations, with mode tags and retrieval-backed evidence.

## 9) Validation Checks

- Coverage (all annotations in baseline): PASS
- Bucket consistency (TP/FP/FN header vs parsed doc cards): PASS
- Taxonomy evidence floor (>=3 examples per mode): PASS
- Case source check (exactly one primary source label per case): PASS
- Reproducibility: deterministic sorting by annotation, PMID, and descending included-count used in case selection.

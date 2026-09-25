# Jev vs GPT-5-mini on emotion_regulation_2022

Decision backend swapped, everything else held fixed: same query, same retrieval vintage
(all arms run against a copy of v4's run directory), same coordinate parsing, same MKDA
settings. Only the model making screening and analysis-selection decisions differs.

| run | what it is |
|---|---|
| `v4` | GPT-5-mini, the published arm |
| `v4-jev` | Jev with v4's criteria, verbatim |
| `v5-jev` | Jev with criteria rewritten as propositions |
| `v6-jev` | v5-jev re-run so screening captures `criterion_probabilities` |

## End-to-end, v4 vs v5-jev

| stage | gpt-5-mini | jev | |
|---|---|---|---|
| abstract F1 | 0.247 | **0.352** | jev |
| full-text F1 | 0.383 | **0.481** | jev |
| end-to-end gold recall | **0.830** | 0.784 | gpt |
| annotation F1 (pooled) | 0.659 | **0.675** | tie |
| **map r² (mean, n=4)** | **0.632** | 0.577 | **gpt** |

Jev wins both screening stages on F1, ties on annotation, and loses the maps by 0.055,
ahead in 1 of 4 contrasts. It contributes fewer studies to every map (84v109, 17v28,
52v86, 78v108).

**For coordinate-based meta-analysis, recall buys more than precision.** MKDA aggregates, so
a surplus study mostly adds signal the kernel absorbs while a missing one is unrecoverable.
The higher-precision arm wins every intermediate metric and loses the outcome.

## The threshold was never tuned

Swept offline from `v6-jev`'s stored probabilities, no API calls.

Abstract screening (gold reachable = 79):

| tau | included | gold kept | recall | precision | F1 |
|---|---|---|---|---|---|
| 0.20 | 491 | 78/79 | **0.987** | 0.159 | 0.274 |
| 0.30 | 409 | 76/79 | 0.962 | 0.186 | 0.311 |
| 0.40 | 364 | 75/79 | 0.949 | 0.206 | 0.339 |
| 0.50 | 339 | 74/79 | 0.937 | 0.218 | 0.354 |
| 0.60 | 292 | 74/79 | 0.937 | 0.253 | 0.399 |
| *gpt-5-mini* | *560* | *79/79* | *1.000* | *0.141* | *0.247* |

Full-text screening is flat -- recall 0.944 to 0.958 across the whole range -- so the
abstract stage is the binding constraint.

At tau = 0.20 Jev reaches 0.987 recall on 491 studies, against GPT-5-mini's 1.000 on 560:
near-equal recall from a 12% smaller pool, still at better precision. That is the operating
point the map result argues for, and it is the opposite end of the range from the F1 optimum
(tau = 0.60). **Tuning screening on F1 would have made the maps worse.**

## Tested: tau = 0.20 does NOT close the map gap

`v7-jev` re-runs at abstract tau = 0.20. Screening improved exactly as the sweep predicted
and the maps did not move at all.

| | gpt-5-mini | jev tau=0.50 | jev tau=0.20 |
|---|---|---|---|
| abstract gold recall | 1.000 | 0.937 | **0.987** |
| full-text gold recall | 0.961 | 0.958 | **0.961** |
| full-text F1 | 0.383 | 0.481 | **0.482** |
| **end-to-end gold recall** | **0.830** | 0.784 | **0.830** |
| studies included | 305 | 215 | 227 |
| **map r² (mean)** | **0.632** | 0.577 | **0.573** |

**The hypothesis was wrong.** "Jev's maps lag because it loses gold studies at screening" is
falsified: gold recall is now identical at 73/88 and the maps are unchanged (0.577 -> 0.573),
still behind in 4 of 4 contrasts.

What differs is the SURPLUS. Both arms include the same 73 gold studies; GPT-5-mini includes
**232 non-gold** against Jev's **154**, and contributes more analyses to every map:

| contrast | gpt experiments | jev experiments | r² gap |
|---|---|---|---|
| reappraisal | 279 | 186 | -0.046 |
| decrease | 266 | 236 | **-0.127** |
| increase | 56 | 33 | -0.055 |
| maintain | 204 | 107 | -0.010 |

But volume alone does not explain it. `decrease` has near-equal experiment counts (236 v 266,
11% apart) and the largest r² gap of all four; `maintain` has half the experiments and the
smallest gap. So the remaining difference is about WHICH analyses are selected, not how many
-- which moves the open question from screening to analysis-level selection.

Two readings of the surplus, not yet distinguished:

1. Those "false positives" are largely correct inclusions the expert pool missed. The project's
   own Supplementary S2 supports this -- measured precision rises from 49.7% to 65.7% when
   author-provided candidate lists replace the PubMed search, so much of the apparent FP rate
   is pool mismatch rather than error.
2. MKDA simply benefits from more coordinates. The `decrease` row argues against this being
   the whole story.

Distinguishing them is the next experiment, and it matters beyond Jev: if (1), then screening
precision measured against published inclusion lists is systematically penalising correct
decisions.

## Cost

| | calls | cost |
|---|---|---|
| screening (v4-jev, reused by v5) | 1,580 | $0.2695 |
| annotation v4-jev | 313 | $0.5466 |
| annotation v5-jev | 309 | $0.4850 |
| screening v6-jev (probability capture) | 1,578 | $0.2676 |
| v7-jev at tau=0.20 (abstract fully cached) | 449 | $0.5512 |
| aborted launches while fixing the cache layers | ~470 | ~$0.08 |
| **total** | **~4,700** | **~$2.20** |

The tau change itself cost $0.00 at abstract screening and charged full-text for only the 145
newly included studies, because thresholds are now post-hoc over stored probabilities.

Same work on gpt-5-mini would be several dollars of input alone, before output tokens --
and output is where chat models cost, since they write a rationale for every decision.


## Tested: matching GPT's selection COUNT makes the maps worse

`v8-jev` keeps abstract tau = 0.20 and loosens analysis selection to tau = 0.30, taking
selections from 584 to 626 against GPT-5-mini's 623.

| contrast | gpt-5-mini | jev v5 (.5/.5) | jev v7 (.2/.5) | jev v8 (.2/.3) |
|---|---|---|---|---|
| decrease | 0.745 | 0.620 | 0.619 | 0.569 |
| increase | 0.440 | 0.385 | 0.385 | **0.410** |
| maintain | 0.556 | 0.560 | 0.545 | 0.487 |
| reappraisal | 0.788 | 0.742 | 0.741 | 0.679 |
| **MEAN** | **0.632** | 0.577 | 0.573 | **0.536** |

Worse in three of four. So the deficit is not volume: the 42 analyses added between tau 0.50
and 0.30 are the wrong ones, and adding them costs 0.037 of mean r-squared.

## Study breadth helps; analysis breadth hurts

| arm | studies | analyses | analyses/study | map r² |
|---|---|---|---|---|
| gpt-5-mini | 305 | 623 | **2.04** | **0.632** |
| jev v7 | 227 | 584 | 2.57 | 0.573 |
| jev v8 | 227 | 626 | 2.76 | 0.536 |

Jev already selects MORE analyses per study than GPT-5-mini, and pushing that ratio higher
made correspondence worse monotonically. GPT's advantage is the opposite shape: it draws on
**more studies, more selectively within each**.

Two things follow.

**It supports the paper's central claim rather than undermining it.** Analysis-level
selectivity is what buys correspondence -- loosening it degrades the map even while the raw
count moves toward the better arm. "Pool more analyses" is not the mechanism.

**The remaining gap is screening breadth, and its status is unresolved.** With gold recall
equalised at 73/88, GPT still includes 232 non-gold studies to Jev's 154, and those extra
studies carry the advantage. Whether they are genuinely relevant work the expert pool missed
-- which Supplementary S2's 49.7% to 65.7% precision jump on author-provided lists suggests --
or whether MKDA simply rewards study count, is not settled here and is a question about the
benchmark rather than about Jev.

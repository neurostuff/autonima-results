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

Untested: whether tau = 0.20 actually closes the map gap. It needs screening and annotation
re-run at the new threshold, roughly $0.90.

## Cost

| | calls | cost |
|---|---|---|
| screening (v4-jev, reused by v5) | 1,580 | $0.2695 |
| annotation v4-jev | 313 | $0.5466 |
| annotation v5-jev | 309 | $0.4850 |
| screening v6-jev (probability capture) | 1,578 | $0.2676 |
| **total** | **3,780** | **$1.569** |

Same work on gpt-5-mini would be several dollars of input alone, before output tokens --
and output is where chat models cost, since they write a rationale for every decision.

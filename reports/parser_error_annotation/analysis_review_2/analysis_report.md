# Parser Failure Annotation Analysis

- Reviewed: **185/199** unmatched analyses (93.0%) across **112 papers**.
- Confirmed parser misses: **56** (30.3%).
- Supplemental-only or missing source: **91** (49.2%).
- Reviewed benchmark/matching/expected differences credited as correct: **33**.

## Dispositions

| Disposition | Analyses | Papers | Share |
|---|---:|---:|---:|
| Supplemental-only source | 68 | 29 | 36.8% |
| Parser missed / misparsed analysis | 56 | 35 | 30.3% |
| Source material missing | 23 | 21 | 12.4% |
| Gold standard error | 21 | 21 | 11.4% |
| Expected source/curation difference | 9 | 4 | 4.9% |
| Out of scope | 4 | 3 | 2.2% |
| Matching error; both analyses correct | 3 | 3 | 1.6% |
| Uncertain | 1 | 1 | 0.5% |

## Confirmed Parser Failure Modes

Percentages use confirmed parser misses as denominator; modes are non-exclusive.

| Failure mode | Analyses | Papers | Share |
|---|---:|---:|---:|
| Under-split / merge | 25 | 15 | 44.6% |
| Partial coordinate error | 18 | 11 | 32.1% |
| No failure mode selected | 7 | 4 | 12.5% |
| Over-split | 7 | 7 | 12.5% |

## Accuracy Reconciliation

| Metric | Correct / total | Rate |
|---|---:|---:|
| Current strict baseline | 816/1040 | 78.5% |
| Review-adjusted strict | 849/1040 | 81.6% |
| Current accepted + uncertain | 841/1040 | 80.9% |
| Review-adjusted accepted + uncertain | 874/1040 | 84.0% |
| Parser-evaluable resolved cases | 819/875 | 93.6% |
| All review-resolved factual extraction cases | 849/905 | 93.8% |

Adjusted full-denominator rates reclassify only manually resolved false negatives.

## Unresolved Annotations

- None.

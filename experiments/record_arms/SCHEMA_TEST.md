# Testing the two Group schema changes

`study_schema` 80f7510 added `Group.population_characteristics` and redefined `is_healthy`
as the absence of a medical condition rather than the source's choice of words. This is the
test: ten papers picked because the old records said something the new schema should change,
re-extracted under the new schema with identical settings, then compared.

Run: `pondie extract --run schema_test`, 10 papers, gpt-5.6-luna, effort low, flavour local,
tier flex, 1.67M in / 190k out tokens. Comparison in
`code/audit/schema_test_compare.py`, field quality in `schema_test_field_quality.py`.

## Predictions, made before the run

**A — `is_healthy` should flip True → False**, because each group carries a real diagnosis:

| pmid | group | medical_condition |
|---|---|---|
| 16133128 | healthy male smokers | nicotine dependence |
| 17217932 | tobacco-dependent smokers | Nicotine dependence |
| 18568078 | obese subjects | obesity |
| 17197102 | individuals with cocaine use disorder | cocaine substance dependence, cocaine abuse |
| 18540916 | DRD4.L / OPRM1.G / control | Heavy drinking (all three groups) |

**B — `population_characteristics` should become non-empty**, because a trait defines the
cohort and no diagnosis covers it: 11296095 (social drinkers), 29990584 (long-term
meditators), 28929362 (sedentary/healthy-weight children), 25094019 (undergraduates), and
16565998 (normal-weight women) as a **negative control** — its corpus text is 1,267
characters against a 49,116 median, so I predicted it could *not* fill.

## What actually happened

**B: 10 of 10.** Every paper gained `population_characteristics`, including the five A
papers and including the negative control — **my negative-control prediction was wrong**.
16565998 filled with `normal-weight` because its 1,267 characters are the title and abstract,
and those name the cohort. A short record is not an empty one.

**A: 2 of 5 flipped.**

| pmid | flipped? | what happened |
|---|---|---|
| 17217932 | **yes** | `False`, `pc=[smokers, treatment seeking]` |
| 17197102 | **yes** | `False`, diagnoses kept in `medical_condition` |
| 16133128 | no | still `True`, but `medical_condition` became "nicotine dependence varied from absent to …" — the cohort is no longer described as uniformly dependent, so `True` is now arguably defensible |
| 18568078 | no | still `True` with `medical_condition=[obesity]`. The contradiction the change was meant to remove, unchanged |
| 18540916 | no | still `True`, and `heavy drinking` now appears in **both** `medical_condition` and `population_characteristics` on all three groups |

**Negation cleanup, partial and two-directional.** `29990584` dropped
`Healthy, non-clinical population` from `medical_condition` and `25094019` dropped
`none reported; neurologically and psychologically healthy` — both intended. But `28929362`
*gained* `healthy children without reported metabolic…` and `29990584`'s meditators gained
`medical_condition=[none]`. Negations moved rather than stopped.

## What the new field actually collected

33 entries across the ten papers:

| | entries | |
|---|---|---|
| genuine cohort traits | 22 | 67% |
| restate a demographic that already has its own slot | 8 | 24% |
| duplicate the `medical_condition` | 3 | 9% |

The misplaced ones are specific and fixable:

```
17197102  "All participants were right-handed"     -> handedness_distribution
17197102  "English was their first language"        -> no slot, but not a cohort trait
17197102  "Participants were not color-blind"       -> an exclusion criterion
17197102  "Five participants were female and nine were African-American…"
                                                    -> sex_distribution, race_distribution
28929362  "children aged 8-11 years"                -> age_minimum / age_maximum
28929362  "all Caucasian"                           -> race_distribution
18540916  "heavy drinking" (x3)                     -> already in medical_condition
```

## Conclusions

**The new field works and is over-broad.** Adoption was immediate and complete — 10 of 10,
with no prompt change beyond the slot description — but a third of what it collects belongs
somewhere else. My description says what to include and does not say what to exclude, and the
model filled the gap with any sentence describing participants. The fix is an explicit
exclusion list in the description: not handedness, sex, age, race or ethnicity, which have
their own slots; not exclusion criteria; and not a restatement of the diagnosis. Untested as
yet.

**The `is_healthy` redefinition mostly did not take.** 2 of 5. A description change alone
does not override a source that says "healthy", which is the thing the redefinition was
meant to overrule. `18568078` — `is_healthy=True` with `medical_condition=[obesity]` — is the
clean demonstration: nothing in the prompt makes the model revisit a flag it can justify from
the text. `schema-tutorial.md` already declares this combination a contradiction and points
at a record audit that enforces it; this test says the audit is load-bearing and the
description is not sufficient on its own. Enforce the invariant at validation, and either
reject the pair or derive `is_healthy` from `medical_condition` rather than asking for it.

**One honest caveat on the design.** The description I wrote permits a trait that has become
a diagnosis to appear in both fields, which is what produced 18540916's duplication. That was
my wording and it should be decided rather than permitted: either the diagnosis wins and the
trait is dropped, or the trait is recorded and `medical_condition` left for what a clinician
would code. I would pick the first.

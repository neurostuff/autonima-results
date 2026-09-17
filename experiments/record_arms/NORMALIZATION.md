# Normalising the extraction records

Measured over the 1,817 committed records (`code/audit/survey_record_fields.py`,
`code/audit/normalize_open_fields.py`). Every number below is from those two scripts; rerun
them to reproduce it.

## 1. Two kinds of variance, which need different treatment

**Shape variance** — one field arriving as several python types. This is a defect, not
vocabulary, and it breaks any consumer that indexes the value:

| field | types seen |
|---|---|
| `acquisitions[].acquisition_duration_seconds` | `int`, `float`, `list`, `str` |
| `groups[].age_mean` | `float`, `int`, `list` |
| `groups[].age_standard_deviation` | `float`, `int`, `list` |
| `tasks[].stimuli` | `str`, `list` |
| `model_estimations[].software` | `str`, `list` |
| `devices[].model` | `str`, `list` |

**Vocabulary variance** — one thing written many ways. Distinct values before and after each
normalisation stage:

| field | mentions | raw | case+space | punct+spell | token bag |
|---|---|---|---|---|---|
| `assessments[].name` | 4,359 | 2,607 | 2,423 | 1,993 | **1,830** |
| `tasks[].performance_measures` | 2,541 | 2,011 | 1,935 | 1,923 | **1,875** |
| `groups[].medical_condition` | 4,476 | 1,592 | 1,433 | 1,368 | **1,297** |
| `regions[].name` | 3,238 | 1,317 | 1,249 | 1,178 | **1,156** |
| `model_estimations[].software` | 2,646 | 1,283 | 1,266 | 1,145 | **1,094** |
| `tasks[].name` | 1,449 | 1,075 | 1,029 | 984 | **839** |
| `groups[].medications` | 1,079 | 410 | 350 | 343 | **314** |
| `tasks[].presentation_software` | 495 | 333 | 309 | 244 | **229** |

Rules buy 7–31%. They do not solve any of these fields.

## 2. The rules, in the order they pay off

Each is cheap, reversible, and justified by a case it fixes in this corpus.

1. **Shape first.** Coerce to the slot's declared cardinality before anything else: a scalar
   into a one-item list where the slot is multivalued, a numeric string to a number where
   the slot is numeric, and a list where the slot is scalar is a extraction error to
   surface rather than silently join. `acquisitions[].acquisition_duration_seconds` arriving
   as four different types is the case that matters, because a duration that is sometimes
   `"10 min"` and sometimes `600` cannot be compared at all.
2. **Unicode normalise (NFKD), fold combining marks, unify the seven dash codepoints, fold
   case, squeeze whitespace.** `Alzheimer’s` and `Alzheimer's` differ only by U+2019 and were
   two values; `cue-reactivity` and `cue‑reactivity` differ only by U+2011.
3. **Strip possessives** before tokenising. Without it `Alzheimer's` tokenises to
   `alzheimer` plus a stray `s`, and the possessive and non-possessive forms never meet.
   Worth 171 mentions on one term.
4. **Fold British spellings from a lexicon, not a regex.** `behaviour→behavior`,
   `tumour→tumor`, `oedema→edema`, `paediatric→pediatric`, `-ise→-ize`. This is the single
   highest-value rule here: it merged `behavioral variant frontotemporal dementia` (230
   mentions) with `behavioural variant frontotemporal dementia` (77) into one term of 307.
   A regex general enough to catch the internal `-our-` in "behavioural" also eats *four*,
   *hour*, *tour* and *source*, so the lexicon is the correct shape.
5. **Drop parentheticals and punctuation.** `frontotemporal dementia (FTD)` and
   `frontotemporal dementia` are one thing.
6. **Expand abbreviations from a lexicon** — `PTSD`, `MDD`, `bvFTD`, `MCI`, `MID`, `WM`.
   These are the record's own shorthand and they never cluster with the spelled form.
7. **Drop generic words and sort the remainder to a bag.** `task`, `paradigm`, `fMRI`,
   `protocol`, `session`, `based`, `related`. This is what collapses `emotion regulation
   task`, `Emotion Regulation Task` and `emotion regulation` onto one key, and adding
   `protocol` alone moved `resting state` from 68 mentions to 80.

## 3. What rules cannot do, demonstrated

**The tail is real, not noise.** After every rule above, **81% of task clusters and 70% of
condition clusters are seen exactly once**. Those are not spelling variants of the head
terms; they are different studies describing different paradigms.

**The data cannot supply the hierarchy.** I tried folding each token bag into the most
frequent strictly-larger bag containing it — the case that should merge `cue food` into
`cue food visual`. On tasks it merged 148 clusters, but **86 of those had more than one
candidate host**, and the merges invert the relationship that matters:

```
[emotion regulation]      w=111  ->  [emotion positive regulation]   38 candidate hosts
[frontotemporal dementia] w=143  ->  [behavioral variant FTD]        52 candidate hosts
```

`emotion regulation` is the *general* term and the most frequent in the corpus; subset
merging absorbs it into a rarer, more specific variant. Containment tells you two terms are
related; it cannot tell you which is the parent. **That is the argument for an ontology and
not more rules**: what is missing is an ordering the corpus does not contain.

## 4. Which fields to align, and to what

ONVOC is the **OpenNeuro Vocabulary**. BioPortal returns 403 to automated fetches so I could
not enumerate its classes, and I am not going to assert its contents; what is on the public
record is that OpenNeuro characterises task and cognitive concepts using **Cognitive
Atlas**. So ONVOC is a useful guide to *which fields in a neuroimaging dataset description
deserve controlled terms* — and its task branch inherits exactly the incompleteness noted
below.

| field | align? | target | why |
|---|---|---|---|
| `groups[].medical_condition` | **yes, first** | MONDO, cross-referenced to ICD-10 and the DSM edition already in `groups[].diagnostic_system` | Most concentrated of the open fields: 385 terms cover 80% of mentions. Diagnoses are exactly what disease ontologies are for, and the hierarchy is the thing rules cannot supply — `semantic dementia`, `semantic variant primary progressive aphasia` and `frontotemporal lobar degeneration` are three terms for overlapping entities that only a curated hierarchy can relate. |
| `tasks[].name` | **yes, but hybrid** | Cognitive Atlas where it reaches, extended by the corpus vocabulary in §5 | Cognitive Atlas is incomplete for this literature, as you said and as the tail confirms. Do not discard an unmatched term; route it to review. |
| `assessments[].name` | yes | an instrument ontology — e.g. the [multilayer ontology of instruments for neurological, behavioral and cognitive assessments](https://pmc.ncbi.nlm.nih.gov/articles/PMC4303739/); LOINC for anything with a lab analogue | 2,607 raw values, and the heads are named instruments with official forms (`Structured Clinical Interview…` 85, `Mini-Mental State Examination` 54). Highest raw cardinality of any field and the most mechanically resolvable, because instruments have canonical names and acronyms. |
| `regions[].name` | yes, **after a schema change** | UBERON or NIFSTD for the structure; keep the atlas in `regions[].atlas` | See §6: laterality is currently inside the name, so `amygdala` (145), `left amygdala` (110) and `right amygdala` (98) are three terms for one structure. Fix the schema first or the alignment encodes the defect. |
| `groups[].medications` | yes | RxNorm ingredient, or ATC for class | 314 normalised values, mostly ingredient or class names (`antidepressants`, `benzodiazepines`, `methadone`) which RxNorm/ATC cover directly. |
| `model_estimations[].software`, `preprocessings[].software` | yes, lightly | RRID / SciCrunch | The heads are versioned tool names (`SPM8` 275, `SPM5` 135, `SPM12` 69). Split name from version in a rule, then resolve the name. A full ontology is overkill; an RRID and a version field is the whole requirement. |
| `tasks[].performance_measures` | **no** | small closed enum instead | 2,011 raw values but the heads are `accuracy`, `reaction time`, `d-prime` — a dozen concepts written a thousand ways. Rules barely help (7%) because the variance is phrasing, not vocabulary. This wants an enum with an open-vocabulary escape, like `ResponseModality` already has. |
| `assessments[].assessment_type`, `groups[].diagnostic_system`, `inference_settings[].multiple_comparison_method`, `acquisitions[].modality` | **no** | already enums | Working as intended: `DSM-IV` is 881 of 1,753, `questionnaire`/`clinical scale`/`diagnostic interview` are 2,010 of 4,332. The tails are the open-vocabulary escape doing its job. |
| `analyses[].interpretations`, `*.description`, `groups[].{inclusion,exclusion}_criteria` | **no** | leave free | 8,054 distinct interpretations over 5,887 mentions — these are prose by design. Normalising them would destroy the thing they are for. (Their *criteria* fields do deserve the case rule: `Right-handed` 294 against `right-handed` 208 is pure case.) |

## 5. A data-driven task vocabulary

`data/vocab_tasks_name.csv` and `data/vocab_groups_medical_condition.csv`, generated by
`normalize_open_fields.py`:

| | terms | mentions covered |
|---|---|---|
| tasks | 157 | 767 of 1,449 (53%) |
| conditions | 385 | 3,564 of 4,476 (80%) |

Each row carries the `match_key` (the token bag, for matching), a `canonical_label` (the most
frequent surface form, because `delay incentive monetary` is a key and not a label), the
mention count, and every variant seen. Clusters seen once are excluded — a vocabulary of
hapaxes is a list of strings.

Coverage if you curate only the top N clusters:

| top N | tasks | conditions |
|---|---|---|
| 25 | 31% | 44% |
| 50 | 37% | 54% |
| 100 | 45% | 63% |
| 200 | 56% | 71% |
| 400 | 70% | 80% |

The asymmetry is the finding. **Conditions concentrate and tasks do not**: 100 condition
terms reach 63% of mentions where 100 task terms reach 45%, and tasks need 400 to reach 70%.
Diagnoses come from a finite, already-curated world; task names are minted per paper. That
is why an existing disease ontology will mostly work and an existing task ontology mostly
will not.

Suggested use, in order:
1. Match a new value's token bag against `match_key` — exact, free, and covers the head.
2. Fall back to a classifier over the `canonical_label` set, with a confidence floor.
3. Below the floor, emit the value unmatched with `value_source: unmatched` rather than
   forcing it. The 81% singleton rate means forcing will be wrong most of the time, and an
   unmatched value invites a question where a wrong one does not — the lesson from
   `spatial scope: roi`, which was inferred from one sentence, applied to twelve analyses,
   and cost a paper its inclusion.

## 6. Two defects found on the way, worth fixing before any alignment

**`medical_condition` is used to assert the absence of a condition.** 235 mentions (5%) are
`healthy`, `Healthy controls`, `healthy participants` and the like. "Healthy" is not a
medical condition and will not resolve against any disease ontology. This belongs in a
separate boolean or in the group's role, not in the condition field — and note the two forms
are *still* separate clusters after every rule (`healthy` 109, `control healthy` 101),
because "control" is a meaningful word elsewhere and cannot be added to the generic list.

**`regions[].name` carries laterality.** `amygdala` 145, `left amygdala` 110, `right
amygdala` 98 — one structure, three terms, and the same split runs through the whole field.
A `hemisphere` slot (`left` / `right` / `bilateral` / `midline`) would let the name resolve
to one anatomical term. This is the same shape of problem as `stimulus_modality`: a fact
that criteria and queries need, currently only available inside a prose string.

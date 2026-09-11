# Audit of the record-arm meta-analyses

Everything below is measured, not inferred. Scripts are in `code/audit/`.

## 1. The benchmark checkout on the analysis host was stale

`repos/neurometabench` on beast is a symlink to another user's checkout, sitting at
`a6b59be` while `origin/master` is at `00398b9`. Two gold sets differ:

| meta_pmid | project | stale | upstream |
|---|---|---|---|
| `35413444` | emotion_regulation_2022 | **0** | **101** |
| `35664889` | dementia | 74 | 73 |

The first is why this project looked unbenchmarked: the gold was never missing, the checkout
was behind by the commit that added it. A fresh clone lives at
`repos/neurometabench-upstream`.

`projects/emotion_regulation_2022/annotation-only-ids.txt` (88 pmids) is **not** a substitute:
it holds 87 of the 101 gold papers plus one that is not gold. For `vbm_of_ptsd` that same
file matches the gold exactly (22/22) and for dementia it is 73 of 74, so it is a mirror that
drifts, not the source.

Re-scoring every arm against the upstream gold moves only dementia, and only slightly — but
it moves in the record arms' favour, because the removed paper was a true positive for full
text and a false negative for both record arms:

| arm | stale F1 | upstream F1 |
|---|---|---|
| `v3` | 0.5455 | 0.5391 |
| `v3-record-with-evidence` | 0.5650 | 0.5682 |
| `v3-record-no-evidence` | 0.5325 | 0.5357 |

Still missing everywhere, including upstream: `data/nimads/emotion_regulation_2022/` — the
coordinate gold that this project's own `nmb_mappings.json` `_note` cites as the source of
its four manual keys. It is the only one of nine projects absent from `data/nimads/`. Its
`meta_datasets.csv` row exists upstream but is marked "In Progress" with every criteria
field empty.

## 2. The arms are compared at the smallest of three loss channels

Gold papers, and where each project loses them, against the upstream gold:

| project | gold | reach screening | lost at search | lost search→screening | rejected at screening |
|---|---|---|---|---|---|
| `vbm_of_ptsd` | 22 | 77% | 4 | 1 | **1** |
| `cue_reactivity` | 191 | 73% | **49** | 2 | **5** |
| `dementia` | 73 | 89% | 3 | 5 | **3** |
| `vbm_of_substance_use` | 79 | 82% | 8 | 6 | **2** |
| `emotion_regulation_2022` | 101 | 86% | 11 | 3 | **5** |

Full-text screening is the only stage the record arms vary, and it rejects between one and
five gold papers per project. Search loses an order of magnitude more — 49 of cue
reactivity's 191 never enter the pipeline at all. Any claim that records cost or save recall
is a claim about the smallest channel.

## 3. The reported arm F1 is not end-to-end recall

`compare_arms.py` scores over the papers all arms screened in common (`n_common`), so gold
lost before full-text screening never enters the denominator. On `vbm_of_ptsd` the committed
row reads `tp 16, fp 11, fn 1, recall 0.9412` — but the gold set has 22 papers, so six are
missing and five of them are simply absent from the denominator.

That is a defensible choice for a paired arm contrast; it is not a recall figure. Quoted
without the distinction it reads as 94% recall where the end-to-end number is 16/22 = 73%.

## 4. Configuration inconsistencies across the five projects

| project | run used | registry `best` | annotation reads | model id |
|---|---|---|---|---|
| `vbm_of_ptsd` | `v1-A1-mini` (from `v1`) | `v1` | `study_fulltext` | namespaced |
| `cue_reactivity` | `v5-gpt-A1-mini` | **`v6`** | `study_fulltext` | namespaced |
| `dementia` | `v3` | `v3` | `study_fulltext` | **bare slug** |
| `vbm_of_substance_use` | `v2` | `v2` | **`study_abstract`** | **bare slug** |
| `emotion_regulation_2022` | `v4` | `v4` | `study_fulltext` | namespaced |

Three defects, in descending order of how much they matter:

**`vbm_of_substance_use` annotates from the abstract.** Its annotation stage cannot see
whether it is reading a record or an article, which is the variable this experiment exists to
test. It escaped the failure that hid PTSD's arms only because its analysis sets differed
enough to change `study_input_hash`. Not fixable on this host: `v2.yaml` points
`retrieval.full_text_sources` at `/home/zorro/...`, so beast's `v2` is an imported result.

**`dementia` and `vbm_of_substance_use` run bare model slugs** against namespaced record
arms, so for those two the record-vs-text contrast is confounded with provider routing.
PTSD, cue and emotion regulation are clean.

**`cue_reactivity` is not on its best config.** The registry marks `v6` best ("highest
screening F1 (0.455) and much the best recall (0.853)"); the arms were built on a `v5`
variant.

## 5. What has been fixed

- **Annotation-cache contamination**: zero across all twelve runs, verified by
  `code/audit/audit_contamination.py`. Previously cue's two record arms carried 6–16 papers
  per key that the arm itself had rejected, and dementia's no-evidence arm 3–4.
- **PTSD's text-blind annotation**: `study_abstract` → `study_fulltext`, which also made the
  donated cache self-invalidate.
- **Whole-volume R²**: the repo correlates every finite voxel, which on these sparse maps is
  88–96% voxels that are zero in both. Brain-masked `r2` is now primary; `r2_allfinite`
  keeps the old convention. See `code/meta_r2.py`.

## 6. Known interaction, unfixed

With a rendered record in the annotation prompt the model returns the **record's own**
analysis ids — `a_291_1`, `a_prose_1`, `a_pone_0074164_t002_1` — instead of autonima's
`<pmid>_analysis_<n>`: 230 occurrences in cue, 102 in substance use, 2 in dementia. Three
retries recover them and no analysis was lost, but a study failing all three is dropped from
annotation entirely. Namespacing the record's ids at render time would remove the collision.

## 7. Run-to-run noise bounds what can be claimed

Re-running one project's annotation under an unchanged configuration moved its mean map R² by
**0.034**. Several between-arm gaps in figure 7 panel b are smaller than that, and
`vbm_of_ptsd`'s entire screening difference between the two record arms is **one paper out of
fifty** — 21118656, where the no-evidence arm is the one that is factually wrong. Arm
differences below roughly 0.03 R², or a handful of papers, should not be read as real without
repeated runs.

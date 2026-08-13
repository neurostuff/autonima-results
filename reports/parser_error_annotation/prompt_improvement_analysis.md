# Coordinate Parser Prompt Improvement

## Scope

The coordinate parser prompt was revised from the reviewed parser-failure
annotations. The targeted rerun reparsed all 35 papers marked with at least one
confirmed `parser_missed` disposition:

- Cue reactivity: 15 papers, 29 tables
- Problem solving: 4 papers, 10 tables
- Social processing: 12 papers, 29 tables
- VBM of substance use: 4 papers, 7 tables

All 35 papers produced non-empty replacement analyses. In total, 75 tables
were reparsed with the revised prompt.

## Prompt Changes

The revised prompt now explicitly:

- treats contrast, comparison direction, activation/deactivation, group,
  treatment, condition, session, and time point as analysis-defining axes;
- reads analysis labels from columns and multi-level headers, not only row
  section labels;
- carries labels through blank continuation rows;
- keeps region, lobe, cluster, hemisphere, ROI, local maximum, and other
  anatomical subdivisions within one analysis unless they explicitly define
  a separate statistical map;
- requires every valid coordinate row to be assigned exactly once; and
- distinguishes spatial Z coordinates from Z-statistic columns.

The prompt is centralized in Autonima and versioned in the parsing-stage cache
signature so future prompt changes invalidate stale coordinate caches.

## Gold-Standard Matching Results

The prompt-only comparison uses the original 1,040-analysis denominator and
does not apply human-review corrections.

| Result | Accepted | Uncertain | Unmatched | Strict | Accepted + uncertain |
|---|---:|---:|---:|---:|---:|
| Before prompt revision | 816 | 25 | 199 | 78.5% | 80.9% |
| After prompt revision | 840 | 27 | 173 | 80.8% | 83.4% |
| Change | +24 | +2 | -26 | +2.3 pp | +2.5 pp |

Among the 56 analyses previously confirmed as parser misses:

- 22 now match as accepted;
- 3 now match as uncertain; and
- 31 remain unmatched.

## Human-Review Adjustment

Reviewed dispositions are applied as follows:

- `parser_missed` remains parser-evaluable and can still be penalized.
- `matching_error`, `gold_standard_error`, and `expected_difference` are
  credited as accepted.
- `supplemental_data`, `source_material_missing`, `out_of_scope`, and reviewer
  `uncertain` are excluded from the parser-scoring denominator.

After these corrections, the benchmark contains 944 evaluable analyses:

| Accepted | Uncertain | Unmatched | Strict | Accepted + uncertain |
|---:|---:|---:|---:|---:|
| 872 | 27 | 45 | 92.4% | 95.2% |

The adjusted reports retain each original match status and review disposition
in `match_results_overall.json` for auditability.

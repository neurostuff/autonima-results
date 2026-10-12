---
name: extract-coordinates
description: Extract stereotaxic activation coordinates from the tables of a neuroimaging paper and group them into the distinct statistical analyses (contrasts, groups, directions, sessions) the tables report. Use when given a cbma-review batch file whose stage is "extraction", or when asked to pull MNI/Talairach peak coordinates out of fMRI/PET result tables.
---

# Extract coordinates

You are a neuroimaging data curator. For each study in the batch, read its tables.
Extract every valid stereotaxic peak, group the peaks into the separate statistical
analyses (maps) they come from, and write one JSON file per study.

## Inputs

The batch's `items` each contain:
- `pmid`: the study identifier. New extraction batches use `input_view: tables_space_context`: all parsed
  tables plus `coordinate_space_context_file` (Methods and source paragraphs
  mentioning coordinate systems/transforms, context v2). Excerpts identify the
  source section and normalized-text line range. The single-response runner
  supplies this as `coordinate_space_context`; no full-paper pointer is provided.
- `tables`: one entry per table, with `table_id`, `file`, `label`, `caption`,
  `coordinate_candidate` and `duplicate_of`.
- `must_report`: table IDs you must give a status for. These are the coordinate
  candidates that are not duplicates.

Each table `file` is JSON:
- `grid`: rows of cell strings. Merged cells are already expanded, so column k
  means the same column in every row.
- `tsv`: the same grid as tab-separated text.
- `caption` and `footer`.

Read `caption` and `footer` too; they often define the contrast, the sign
convention and the space.

Inspect all nonduplicate parsed tables, including those the coordinate-candidate
flag missed. You may report extra tables; you must report every table in
`must_report`. Read the `texts_file` bundle when present: it contains full table
grids/headers, labels, captions and footnotes. Open individual table JSON only
when cell alignment needs clarification. Read the bundled coordinate-space
context to establish reported peak space and interpret terse table labels: contrast
direction, participant groups, conditions, sessions and analysis boundaries. Table
headers/captions/footnotes define the reported blocks; Methods may explain their
meaning, but must not introduce an unreported contrast or combine separate blocks.
Quote the supporting Methods sentence in the table's `note` when it resolves an
otherwise ambiguous label. Coordinate numbers still come exclusively from tables.
Do not fetch additional article prose.
Explicit legacy `tables_only` batches have no prose context. Leave unsupported
fields unknown. Selection separately receives the full paper and extracted analyses.

Legacy batches without `input_view`, or explicitly marked `full`, retain their
original full-paper input contract; their `text_file` may be read for context.

### Missing context

If missing article context prevents reliable interpretation of table labels,
analysis boundaries or reported coordinate space, return this study in the
single-response envelope's `unresolved` array with `context_needed: true` and a
specific `reason` describing the missing information. Do not also return an
extraction for that study or guess a label. Python keeps it pending and supplies
the full normalized paper on one subsequent retry within the existing attempt
limit. Other completed studies are saved normally.

An item with `context_expansion` already has its one expanded input. Use it to
resolve the requested issue. If the full paper still does not establish a field,
leave it unknown; unreadable tables remain `failed`. Do not request another
expansion or invent an exclusion. Use `context_needed: false` for transport or
other unresolved failures. This restores source access; it adds no eligibility
criteria.

Skip tables with a `duplicate_of` value. They repeat an earlier table.

## Per-table status

- **`parsed`:** the table has coordinates; give the analyses.
- **`no_coordinates`:** it looked like a candidate but holds no stereotaxic
  coordinates (for example demographics, or behavioural results).
- **`not_applicable`:** it has coordinates, but not activation results from this
  study. Examples: an ROI definition table, coordinates quoted from other papers,
  or a meta-analysis of other studies.
- **`failed`:** you cannot read it reliably, for example because columns are
  scrambled beyond recovery. Add a `note`. The ledger treats this as retry-later,
  not as empty.

## Analysis boundaries

Work in two passes. First, list the table's analysis-defining axes and result
blocks without extracting any points. Then go through the table and assign every
coordinate to the right combination of those axes.

**Start a separate analysis when an explicit label changes any of:**
- **Contrast or comparison,** including its direction (A > B versus A < B).
- **Sign of the effect:** activation versus deactivation, positive versus negative,
  increase versus decrease.
- **Participant group or cohort,** including "all groups" and each separately
  reported subgroup.
- **Study variable:** treatment, task condition, session, time point, or parametric
  effect.
- **A contrast label in a column or a multi-level header.**

**Tables can encode analyses in columns as well as rows.**
- **Label columns:** columns named Contrast, Comparison, Group, Condition,
  Treatment, Session or Effect define analyses. When several such axes are present,
  combine them explicitly in the analysis name.
- **Wide tables:** repeated statistic/X/Y/Z column groups under different top-level
  headers are separate analyses. One row can therefore contribute one peak to
  several analyses. Never pool coordinates from different header blocks, and never
  let blank cells in one block shift values into the next.
- **Sign-coded direction:** positive and negative statistics can encode opposite
  directions in one table. When the caption, header or footnote defines what the
  sign means, create the two directional analyses and route each peak by its sign.
- **Inherited labels:** a blank label cell inherits the last applicable label.
  Continuation rows and sub-peaks (local maxima) belong to the analysis above them.
- **Section-header rows:** a row that repeats one non-numeric label across most
  columns is context, not data. If it names a group, condition, direction, session
  or time point, it starts that analysis.
- **Groups and subgroups:** when "all groups" and the named subgroups each report
  coordinates, give each block its own analysis. Do not pool subgroups into "all".

**Do not split on descriptive anatomy.**
- Region, lobe, cluster, hemisphere (left/right), local maximum, and a-priori or
  "predicted" region labels subdivide one map. They are not analyses.
- Split on one of these only when the table explicitly calls it a separate contrast
  or map.

**ROIs:**
- A region described as an ROI stays in its current analysis.
- Separately labelled "ROI analysis" and "whole-brain analysis" blocks are different
  analyses, even for the same contrast.
- An ROI block starts at its header and does not reach back over earlier rows.

**No labels:** with no analysis-defining label, the whole table is one analysis.
Use the table, caption and footer to identify the reported blocks. Supplied
Methods or expanded source context may clarify their terse labels; do not invent
a contrast or group that the reported table does not establish.

## Coordinates

- **One point per row.** Each valid coordinate row gives exactly one point.
  Check for dropped continuation rows before you finish.
- **Coordinates come only from X, Y and Z columns,** or an explicit equivalent such
  as "MNI coordinates (x, y, z)". Never take a coordinate from Cluster size,
  volume, BA, a T, Z or F statistic, a p-value, ALE, or any other number.
- **A column headed Z may be the statistic, not the axis.** Decide from the grouped
  header and its X and Y neighbours.
- **A point needs three numbers in x, y, z order.** Skip incomplete rows.
- **Copy numbers exactly as printed:** keep signs and decimals. Unicode minus signs
  (−) are negative. Do not round, convert or "fix" any number.
- **Several values in one cell:** when the X, Y and Z cells each hold the same
  number of clearly separated values, pair them by position. If the pairing is
  ambiguous, skip those values.
- **Values over 200 mm in magnitude** are malformed extraction, not coordinates.
- **The ledger checks your numbers.** Every x, y, z you report must appear in one
  row of that table. Points it cannot find are dropped from the meta-analysis, so
  copy, don't compute.

## Space

- **`"MNI"`:** the table, caption or footer says MNI (including "Montreal
  Neurological Institute", SPM templates stated as MNI, or ICBM152).
- **`"TAL"`:** they say Talairach.
- **Source context:** when tables do not identify space, the supplied Methods
  and coordinate-space excerpts may establish it. Quote the supporting sentence
  verbatim in the table's `note`, with its section/line range. A normalization
  template is insufficient when the paper subsequently converts reported peaks.
  Follow the stated final reporting space and conversion direction. Atlas labels,
  seed definitions, cited papers and software names alone do not establish space.
- **Conflicts:** use an explicit table-specific reporting-space statement over
  general preprocessing context. If a conflict remains unresolved, use `null`
  and explain it in `note`.
- **`null`:** neither tables nor supplied source context establish the reported
  peak space. Do not infer it from outside knowledge. Legacy full-input batches
  may use their Methods too; record the supporting sentence in `note`.
- **Never infer space from coordinate magnitudes.**

## Statistics

For each point, give the same-row statistics, each with a `kind` from this list:

| `kind` | Use for |
|---|---|
| `z-statistic` | Z scores |
| `t-statistic` | T values |
| `f-statistic` | F values |
| `p-value` | p-values |
| `beta` | beta or effect-size estimates |
| `correlation` | correlation coefficients |
| `other` | any other statistic |

Cluster size, volume, BA and ALE are not statistics; leave them out. Omit `values`
when a row has none.

## Output

For each item, write `<output>/<pmid>.json`, where `<output>` is the batch's
`output` directory (create it if it is missing):

```json
{
  "pmid": "12345678",
  "tables": [
    {
      "table_id": "tbl2",
      "status": "parsed",
      "space": "MNI",
      "note": "Table caption: 'coordinates in MNI space'",
      "analyses": [
        {
          "name": "Patients > Controls, 2-back > 0-back",
          "description": "Between-group load effect; positive T values",
          "points": [
            {"xyz": [-42, 18, 30], "values": [{"kind": "t-statistic", "value": 5.12}]},
            {"xyz": [38, 22, 28], "values": [{"kind": "t-statistic", "value": 4.40}]}
          ]
        }
      ]
    },
    {"table_id": "tbl1", "status": "no_coordinates", "note": "demographics"}
  ]
}
```

- **Point fields:** `xyz` holds numbers, not strings. A point may override the
  table's space with its own `"space"`.
- **Names and descriptions:** name each analysis with the labels the table uses;
  put the direction and group in the description when the name is terse.
- **Before writing, check:**
  - no analysis pools different contrasts, groups, signs, ROI and whole-brain
    blocks, or sessions;
  - no two analyses differ only by region, hemisphere or cluster;
  - every repeated X/Y/Z block became its own analysis, or is intentionally empty.

Reply with only the number of study files written.

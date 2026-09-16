# records/

The pondie extraction records the record arms screen, one directory per meta-analysis.
These are the input that the whole comparison turns on: the full-text arm reads the article,
the record arms read the rendered form of the file sitting here.

| project | records | size | pondie run |
|---|---|---|---|
| `cue_reactivity` | 550 | 52 MB | `cue` |
| `dementia` | 448 | 36 MB | `dementia` |
| `emotion_regulation_2022` | 525 | 51 MB | `emotion_regulation` |
| `vbm_of_ptsd` | 50 | 4.5 MB | `ptsd` |
| `vbm_of_substance_use` | 244 | 22 MB | `sud` |

Extracted with `@psyc-aid338-ope-333f18/gpt-5.6-luna`, `--effort low`, `--flavour local`,
`--service-tier flex`, through the nine-stage pipeline
(`tables → prose → split → demands → satisfy → fill → evidence → build → repair`).

## MANIFEST.csv

Each directory carries one, with `pmid`, `file`, `bytes`, `sha256`, and
`rendered_by_the_arms`. That last column is not decoration. The arms rendered from a staged
copy of the run rather than from the run directly, and for `vbm_of_substance_use` the two
differ: the run holds 244 records and its arms rendered 241. The complete set is kept here
because that is what the extraction produced, and the three it did not use are marked
`False` rather than dropped, so a rerun can reproduce either set.

## What these are not

Not the articles. A record is a structured extraction of one study against
`study_schema/neuroimaging-study-extraction` -- groups, tasks, acquisitions, analyses,
coordinates -- with an `ExtractedValue` wrapper on every field carrying
`extraction_status`, `value_source` and `evidence`. The rendered text form the arms actually
screen is derived from these by `render_record.py`, with `--evidence full` or `--evidence
none`; the renders are not committed because they are a pure function of these files and the
flag.

Size for that render, measured over 1,814 papers: the article averages 43,295 characters and
10,166 tokens, the no-evidence record 15,709 and 3,702 (64% fewer), the with-evidence record
26,253 and 6,691 (34% fewer).

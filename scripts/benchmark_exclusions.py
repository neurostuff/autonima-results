#!/usr/bin/env python3
"""Benchmark columns excluded from every analysis, and why.

WHY THIS EXISTS AS ONE FILE

Three `vbm_of_substance_use` columns have no published result to recover, so scoring the pipeline
against them measures agreement with nothing. That decision has to reach the denominator of every
pooled statistic and the row set of every plot identically, or the paper will quote 35 columns in
one place and 32 in another. Encoding it once and importing it is the only way that stays true.

THE EXCLUSION

Hill-Bowen et al. 2022, `Drug and Alcohol Dependence` 240:109625 (PMID 36115222) ran ALE
meta-analyses for five drug classes and found significance in two. Of the rest it says, verbatim:
"Drug-specific meta-analyses for cannabis, opioids, and stimulants failed to yield significant
clusters." The paper attributes this to "insufficient power to detect true effects in other
categories of drugs."

The expert maps agree: all three have zero voxels above the corrected q <= 0.05 boundary. So

  * dice is necessarily 0.000 -- not because the pipeline failed but because the reference is
    empty, and for cannabis the expert map peaks at z = 0.496, below any threshold; and
  * r-squared on the unthresholded maps is NOT a rescue. It reads 0.24-0.59 for these columns,
    which is two sub-threshold MKDA density maps agreeing about where the literature happens to
    report coordinates. It measures no recovered finding. An earlier draft of the metric/map note
    took that rise at face value; this is the correction.

So the exclusion is a benchmark SCOPE decision, not a metric choice, and it does not depend on
which metric a figure reports.

NOT EXCLUDED, DELIBERATELY

`nicotine` stays. The paper found significant PCC clusters and the gold inputs match it exactly
(18 experiments, 114 foci), yet neurometabench's MKDA re-analysis finds nothing -- max z 0.935.
That is a reproduction failure in the benchmark rather than an absent reference, and dropping it
would hide the one column where the ALE-to-MKDA substitution can be shown to change whether the
reference result exists at all. It is a finding, so it is reported, not removed.
"""

from __future__ import annotations

# (project, manual annotation) -> short reason, for logs and captions.
EXCLUDED: dict[tuple[str, str], str] = {
    ("vbm_of_substance_use", "cannabis"):
        "no significant result in the source paper (Hill-Bowen 2022); expert map empty",
    ("vbm_of_substance_use", "opioids"):
        "no significant result in the source paper (Hill-Bowen 2022); expert map empty",
    ("vbm_of_substance_use", "stimulants"):
        "no significant result in the source paper (Hill-Bowen 2022); expert map empty",
}

# Files disagree on what the column field is called.
COLUMN_KEYS = ("manual_annotation", "manual_column", "column", "auto_column")


def is_excluded(project: str, column: str) -> bool:
    return (project, column) in EXCLUDED


def reason(project: str, column: str) -> str | None:
    return EXCLUDED.get((project, column))


def _column_of(row: dict) -> str | None:
    for k in COLUMN_KEYS:
        if k in row and row[k]:
            return row[k]
    return None


def split(rows: list[dict], project_key: str = "project") -> tuple[list[dict], list[dict]]:
    """(kept, dropped). Works on any of the report CSVs regardless of its column field name."""
    kept, dropped = [], []
    for r in rows:
        col = _column_of(r)
        (dropped if col and is_excluded(r.get(project_key, ""), col) else kept).append(r)
    return kept, dropped


def filter_rows(rows: list[dict], project_key: str = "project",
                announce: bool = True, label: str = "") -> list[dict]:
    """Drop excluded columns, saying so -- a silent denominator change is how 35 becomes 32
    in one table and stays 35 in another."""
    kept, dropped = split(rows, project_key)
    if announce and dropped:
        where = f" ({label})" if label else ""
        print(f"  excluded {len(dropped)} benchmark column(s){where}:")
        for r in dropped:
            col = _column_of(r)
            print(f"    {r.get(project_key, '?')}/{col} -- {reason(r.get(project_key,''), col)}")
    return kept


if __name__ == "__main__":
    print(__doc__)
    for (p, c), why in EXCLUDED.items():
        print(f"  {p}/{c}\n    {why}")

#!/usr/bin/env python3
"""Shared loaders for benchmark (neurometabench) gold-standard study PMIDs.

The benchmark ships an included_studies CSV relating each published meta-analysis
to the studies it included:

    meta_pmid,study_pmid,doi
    36115222,12345678,...

Callers either pass that CSV plus the meta-analysis PMID to filter on, or a plain
text file with one PMID per line (legacy form).

Extracted because compare_screening_to_benchmark.py and compare_search_to_meta.py
carried near-identical copies of this loader that had drifted apart: only one
stripped the ".0" artifact left when pandas parses integer PMIDs as floats, and
they disagreed on what to do with a CSV containing neither recognised column set.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

__all__ = [
    "normalize_pmid",
    "normalize_pmid_list",
    "load_meta_pmids",
]


def normalize_pmid(value: Any) -> str | None:
    """Normalize a PMID to a plain string, or None if missing/blank."""
    if pd.isna(value):
        return None

    pmid = str(value).strip()
    if not pmid or pmid.lower() == "nan":
        return None

    # Common CSV artifact when integer PMIDs are parsed as floats.
    if pmid.endswith(".0"):
        pmid = pmid[:-2]

    return pmid


def normalize_pmid_list(values: Iterable[Any]) -> list[str]:
    """Normalize an iterable of PMIDs, dropping missing values."""
    return [pmid for value in values if (pmid := normalize_pmid(value)) is not None]


def load_meta_pmids(
    meta_pmids_path: str,
    meta_analysis_pmid: str | None = None,
    *,
    strict_csv: bool = True,
    require_nonempty: bool = True,
) -> list[str]:
    """Load gold-standard included-study PMIDs.

    Accepts either an included_studies CSV (filtered by meta_analysis_pmid when it
    has meta_pmid/study_pmid columns, or read wholesale from a 'pmid' column) or a
    headerless text file with one PMID per line.

    strict_csv: when a .csv has neither recognised column set, raise rather than
        silently re-reading it as a headerless PMID list. compare_search_to_meta
        raised here; compare_screening_to_benchmark fell through to the text path,
        which quietly yields garbage. Default to the strict behaviour.
    """
    path_lower = meta_pmids_path.lower()

    if path_lower.endswith(".csv"):
        df = pd.read_csv(meta_pmids_path)
        columns = set(df.columns)

        if {"meta_pmid", "study_pmid"}.issubset(columns):
            if not meta_analysis_pmid:
                raise ValueError(
                    "CSV input with columns 'meta_pmid' and 'study_pmid' requires "
                    "a meta-analysis PMID (--meta-analysis-pmid, or 'meta_pmid' in "
                    "a project nmb_mappings.json)."
                )
            wanted = str(meta_analysis_pmid).strip()
            filtered = df[df["meta_pmid"].astype(str).str.strip() == wanted]
            pmids = normalize_pmid_list(filtered["study_pmid"].tolist())
            if require_nonempty and not pmids:
                raise ValueError(
                    f"No included study PMIDs found for meta-analysis PMID "
                    f"{meta_analysis_pmid} in {meta_pmids_path}."
                )
            return pmids

        if "pmid" in columns:
            return normalize_pmid_list(df["pmid"].tolist())

        if strict_csv:
            raise ValueError(
                f"CSV file {meta_pmids_path} must contain either "
                f"'meta_pmid' and 'study_pmid' columns, or a 'pmid' column."
            )

    # Text file with one PMID per line.
    df = pd.read_csv(meta_pmids_path, header=None, names=["pmid"])
    return normalize_pmid_list(df["pmid"].tolist())

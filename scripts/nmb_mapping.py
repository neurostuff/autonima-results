#!/usr/bin/env python3
"""Shared loader for project nmb_mappings.json (neurometabench mapping files).

A project mapping file links this repo's automated annotation column names to the
manual annotation names used by the benchmark, and records which published
meta-analysis the project replicates:

    {
      "meta_pmid": "36115222",
      "annotation_mappings": {"<manual_name>": "<auto_annotation_name>", ...}
    }

A legacy flat form is also accepted, where the mapping pairs sit at the top level
alongside "meta_pmid":

    {"meta_pmid": "...", "<manual_name>": "<auto_name>", ...}

This module exists because the same resolve/parse logic was independently
reimplemented in compare_screening_to_benchmark.py, compare_meta_to_benchmark.py,
compare_analyses_to_benchmark.py, run_cross_project_manual_vs_auto_meta_fair.py
and make_pipeline_vs_baselines_plot.py -- with subtly different filename
fallbacks and error messages. Prefer these helpers over adding a sixth copy.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = [
    "MAPPING_FILENAMES",
    "resolve_mapping_path",
    "load_mapping_payload",
    "load_mapping_pairs",
    "load_mappings",
    "load_meta_pmid",
]

# Checked in order. The singular form is legacy but still present in some projects.
MAPPING_FILENAMES = ("nmb_mappings.json", "nmb_mapping.json")


def resolve_mapping_path(
    project_dir: Path | str,
    mapping_path: Path | str | None = None,
    *,
    required: bool = True,
) -> Path | None:
    """Locate a project's mapping file.

    An explicit mapping_path wins and must exist. Otherwise MAPPING_FILENAMES are
    tried inside project_dir. Returns None instead of raising when required=False,
    which callers use to fall back to built-in defaults.
    """
    if mapping_path is not None:
        candidate = Path(mapping_path).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Mapping file not found: {candidate}")
        return candidate

    project_dir = Path(project_dir).expanduser()
    candidates = [project_dir / name for name in MAPPING_FILENAMES]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if not required:
        return None
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate a mapping file for project {project_dir.name}. Searched: {searched}"
    )


def load_mapping_payload(mapping_path: Path | str) -> dict:
    """Read and validate the mapping file as a JSON object."""
    path = Path(mapping_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Mapping file must be a JSON object: {path}")
    return payload


def _raw_mapping_entries(payload: dict, mapping_path: Path) -> dict:
    """Return the manual->auto entries, handling nested and flat layouts."""
    if "annotation_mappings" in payload:
        nested = payload.get("annotation_mappings")
        if not isinstance(nested, dict):
            raise ValueError(
                f"Invalid mapping format at {mapping_path}: "
                "expected 'annotation_mappings' to be a JSON object"
            )
        return nested
    # Legacy flat layout: pairs at top level, minus the meta_pmid marker.
    return {key: value for key, value in payload.items() if str(key).strip() != "meta_pmid"}


def load_mapping_pairs(
    mapping_path: Path | str,
    *,
    require_nonempty: bool = True,
) -> list[tuple[str, str]]:
    """Return [(manual_name, auto_name), ...] in file order.

    Nested container values are skipped (they are metadata, not mapping pairs), as
    are entries where either side is blank after stripping.
    """
    path = Path(mapping_path)
    payload = load_mapping_payload(path)
    raw = _raw_mapping_entries(payload, path)

    pairs: list[tuple[str, str]] = []
    for manual_raw, auto_raw in raw.items():
        if isinstance(auto_raw, (dict, list)):
            continue
        manual = str(manual_raw).strip()
        auto = str(auto_raw).strip()
        if manual and auto:
            pairs.append((manual, auto))

    if require_nonempty and not pairs:
        raise ValueError(f"No valid mapping entries found in {path}")
    return pairs


def load_mappings(mapping_path: Path | str, **kwargs) -> dict[str, str]:
    """Same as load_mapping_pairs but as a manual_name -> auto_name dict."""
    return dict(load_mapping_pairs(mapping_path, **kwargs))


def load_meta_pmid(mapping_path: Path | str) -> str | None:
    """Return the benchmark meta-analysis PMID, or None if not recorded."""
    payload = load_mapping_payload(mapping_path)
    value = str(payload.get("meta_pmid") or "").strip()
    return value or None

"""One reader for the per-project scored CSVs the figures draw from.

Before this, each figure knew its own paths: figure 7 read `<project>_record_arms.csv` from a
directory on the analysis host, figure 4 read a combined `record_arms_meta_metrics.csv`, and
both carried their own hard-coded list of projects and run names. Adding a fifth project
meant editing three lists that could disagree. Now the projects are whatever descriptors
exist, the arms come from the descriptor, and the numbers come from `score`'s output.
"""
from __future__ import annotations

import csv
from pathlib import Path

from . import paths, spec as spec_mod

ARM_ORDER = ["full text", "record + evidence", "record, no evidence"]


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def projects() -> list[str]:
    """Descriptors that have been scored, in the palette's order."""
    from make_nature_methods_figures import PROJECT_ORDER
    have = [p for p in spec_mod.all_projects()
            if (paths.DATA / f"{p}_screening.csv").is_file()]
    return [p for p in PROJECT_ORDER if p in have] + [p for p in have if p not in PROJECT_ORDER]


def screening(metric: str = "endtoend_f1") -> dict[str, dict[str, float]]:
    """project -> arm -> the named screening column.

    `endtoend_*` scores against the whole gold set; `paired_*` scores only over papers every
    arm screened, which is what compare_arms.py reported. They differ by more than the arms
    differ from each other -- PTSD reads 0.653 end-to-end against 0.727 paired -- so which
    one a figure plots belongs in its caption.
    """
    out: dict[str, dict[str, float]] = {}
    for project in projects():
        for row in _read(paths.DATA / f"{project}_screening.csv"):
            v = row.get(metric)
            if v not in (None, ""):
                out.setdefault(project, {})[row["arm"]] = float(v)
    return out


def maps(metric: str = "r2") -> tuple[dict, dict]:
    """(arm values, baseline values), each project -> arm/column -> list of per-column numbers.

    The baseline is one row per manual column with `arm == "baseline"`, re-estimated from its
    own studyset with the arms' settings. It is kept separate rather than treated as a fourth
    arm because it answers a different question.
    """
    arms: dict[str, dict[str, list[float]]] = {}
    base: dict[str, dict[str, float]] = {}
    for project in projects():
        for row in _read(paths.DATA / f"{project}_maps.csv"):
            v = row.get(metric)
            if v in (None, ""):
                continue
            if row["arm"] == "baseline":
                base.setdefault(project, {})[row["manual_analysis"]] = float(v)
            else:
                arms.setdefault(project, {}).setdefault(row["arm"], []).append(float(v))
    return arms, base


def map_columns(metric: str = "r2") -> list[dict]:
    """Tidy per-column rows, for a figure that pairs an arm against its own baseline."""
    rows = []
    for project in projects():
        by_col: dict[str, dict[str, float]] = {}
        for row in _read(paths.DATA / f"{project}_maps.csv"):
            v = row.get(metric)
            if v not in (None, ""):
                by_col.setdefault(row["manual_analysis"], {})[row["arm"]] = float(v)
        for column, per_arm in by_col.items():
            if "baseline" in per_arm and all(a in per_arm for a in ARM_ORDER):
                rows.append({"project": project, "column": column, **per_arm})
    return rows


def arms_for(project: str) -> dict[str, str]:
    return spec_mod.load(project).arms

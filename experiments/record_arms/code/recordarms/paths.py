"""Every location the comparison touches, resolved once.

Overridable by environment so the same code runs on a second host, but with defaults that
match the analysis host rather than silently resolving to something plausible -- `REPO` used
to be `Path(__file__).parents[2]`, which became `/home` when the file moved and produced
"baseline coordinate parses: 0 pmids" with every paper building empty.
"""
from __future__ import annotations

import os
from pathlib import Path

EXP = Path(os.environ.get("RECORD_ARMS_EXP", "/data/james/pondie-vs-fulltext"))
REPO = Path(os.environ.get("AUTONIMA_RESULTS", EXP / "repos/autonima-results"))
#: An own clone, never a symlink into another user's checkout. See WORKFLOW.md step 0.
BENCH = Path(os.environ.get("NEUROMETABENCH", EXP / "repos/neurometabench-upstream"))
PONDIE = Path(os.environ.get("PONDIE_ROOT", "/home/james/pondie"))
AUTONIMA_VENV = Path(os.environ.get("AUTONIMA_VENV", "/home/james/projects/fdcr/.venv"))

CORPUS = EXP / "corpus"
PMIDS = EXP / "pmids"
STAGED = EXP / "records_staged"
RENDERED = EXP / "rendered"
MIRRORS = EXP / "records_text"

HERE = Path(__file__).resolve().parent.parent.parent      # experiments/record_arms
ARMS_DIR = HERE / "arms"
DATA = HERE / "data"
FIGURES = HERE / "figures"
STATE = HERE / "state"

GOLD_CSV = BENCH / "data/included_studies.csv"
RUN_CATEGORIES = REPO / "run_categories.yaml"


def manual_map(project: str, key: str) -> Path:
    """The manual benchmark map for one annotation key."""
    return (REPO / "projects" / project / "reports/manual_vs_auto_meta_fair"
            / "fair_manual_meta/manual_analysis" / project / key / "z.nii.gz")


def auto_map(project: str, run: str, auto_key: str) -> Path:
    return (REPO / "projects" / project / run
            / "outputs/meta_analysis_results" / auto_key / "z.nii.gz")


def baseline_map(project: str, manual_key: str) -> Path:
    """The per-column baseline run's map, re-estimated with the arms' settings.

    Baseline run directories are named for the MANUAL column they target, and each carries a
    single `all_analyses` key.
    """
    return (REPO / "projects" / project / "baselines" / manual_key
            / "outputs/meta_analysis_results/all_analyses/z.nii.gz")


def outputs(project: str, run: str) -> Path:
    return REPO / "projects" / project / run / "outputs"

"""MKDA fits for the paper, through autonima's own meta-analysis code (autonima v0.1.1).

Every map in the paper is produced by `autonima.meta.run_meta_analysis_for_column` -- the
function behind `autonima meta` -- on NiMARE 0.21.0. This module only adapts it to the two
ways the scripts here call it:

  fit_to_dir(studyset_data, ids, output_dir)  one map directory, the file set `autonima meta`
                                              writes (used by refit_maps.py)
  fit(studyset_data, ids)                     raw and FDR-corrected z arrays from one fit,
                                              without writing files (used by the 500-draw
                                              size-matched null in bootstrap_annotation_null.py)

Both select analyses with autonima.meta._select_studyset_data, so the null's draws and the
published maps go through the same selection: selected ids present in the studyset, at least
one finite point, each study/analysis record once, a missing coordinate space fitted as MNI.

WHY autonima v0.1.1. v0.1.0 cannot fit on NiMARE 0.21: it fails to import
(nimare.nimads.Annotation is gone), and its Studyset.slice selection keeps every analysis of a
named study on 0.21.x. v0.1.1 is v0.1.0 plus only those fixes (neurostuff/autonima#88, #93);
refitting with it reproduces the NiMARE 0.2.1 maps to ~1e-6 z except where an analysis mixes
a recorded and a missing space (five social maps, at most 0.0004 r^2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NIMARE_VERSION = "0.21.0"
AUTONIMA_VERSION = "0.1.1"


def selected_ids(annotation: dict, column: str) -> list[str]:
    """Analysis ids whose boolean note for `column` is true (autonima's rule)."""
    from autonima.meta import _analysis_ids_for_column
    if isinstance(annotation, list):
        annotation = annotation[0] if annotation else {}
    return _analysis_ids_for_column(annotation, column)


def subset(studyset_data: dict, ids) -> dict:
    """The studyset dict autonima would fit for these analysis ids."""
    from autonima.meta import _select_studyset_data
    return _select_studyset_data(studyset_data, ids)[0]


def valid_ids(studyset_data: dict, ids) -> list[str]:
    """Ids that would actually be fitted: present in the studyset with a valid point."""
    return [a["id"] for s in subset(studyset_data, ids)["studies"] for a in s["analyses"]]


def _column_annotation(ids) -> dict:
    """A one-column NiMADS annotation selecting exactly `ids`, for run_meta_analysis_for_column."""
    return {"note_keys": {"selected": "boolean"},
            "notes": [{"analysis": i, "note": {"selected": True}} for i in ids]}


def _check_versions() -> None:
    import autonima
    import nimare
    if (autonima.__version__, nimare.__version__) != (AUTONIMA_VERSION, NIMARE_VERSION):
        raise RuntimeError(f"paper maps use autonima {AUTONIMA_VERSION} / NiMARE {NIMARE_VERSION}; "
                           f"found {autonima.__version__} / {nimare.__version__}")


def fit_to_dir(studyset_data: dict, ids, output_dir: Path) -> int:
    """Fit through autonima.meta.run_meta_analysis_for_column; returns analyses fitted."""
    from autonima.meta import run_meta_analysis_for_column
    _check_versions()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_meta_analysis_for_column(studyset_data, None, _column_annotation(list(ids)),
                                          "selected", str(output_dir))
    if result is None:
        raise RuntimeError("autonima selected no analyses with valid coordinates")
    return len(result.estimator.dataset.ids)


def fit(studyset_data: dict, ids):
    """(raw z, FDR-corrected z, n analyses) from one fit, without writing files.

    Same selection, estimator and corrector as fit_to_dir -- autonima's own
    _select_studyset_data, create_estimator("mkdadensity") and create_corrector("fdr") -- but
    skips CBMAWorkflow's FocusCounter diagnostics and file output, which do not touch the maps
    and would multiply the cost of the 13,500-fit null.
    """
    from autonima.meta import create_corrector, create_estimator
    from nimare.nimads import Studyset
    _check_versions()
    sub = subset(studyset_data, ids)
    n = sum(len(s["analyses"]) for s in sub["studies"])
    ds = Studyset(sub).to_dataset()
    if len(ds.ids) != n:
        raise RuntimeError(f"dataset holds {len(ds.ids)} analyses, {n} were selected")
    res = create_estimator("mkdadensity", {}).fit(ds)
    cor = create_corrector("fdr", {}).transform(res)
    return (np.asarray(res.get_map("z").dataobj),
            np.asarray(cor.get_map("z_corr-FDR_method-indep").dataobj),
            len(ds.ids))

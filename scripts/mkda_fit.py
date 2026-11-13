"""MKDA fits for the paper, on NiMARE 0.21.0, with autonima's analysis selection.

WHY THIS EXISTS

The paper's maps were first fitted by `autonima meta` (autonima v0.1.0) on NiMARE 0.2.1. On
NiMARE 0.21 that path is unusable, for two separate reasons:

  1. autonima.meta does not import: it needs nimare.nimads.Annotation, which 0.21 removed.
  2. Its selection relies on Studyset.slice(analyses=...), whose meaning changed. On 0.2.1 it
     keeps exactly the named analyses; on 0.21 it keeps EVERY analysis of the named studies
     (asking for 1 of a study's 5 analyses returns all 5) and raises on an id missing from the
     studyset. Run unchanged it would silently pool whole papers -- the very thing analysis
     selection exists to avoid.

So the selection is done here, on the NiMADS dicts, before NiMARE sees anything. The rules are
autonima v0.1.0's, unchanged:

  - an analysis is selected when the boolean column is true for it in nimads_annotation.json
    (autonima.meta._analysis_ids_for_column);
  - ids absent from the studyset are ignored (0.2.1's slice ignored them silently);
  - an analysis needs at least one finite (x, y, z) point
    (autonima.meta._analysis_has_valid_coordinates);
  - MKDADensity() and FDRCorrector() at NiMARE defaults (method "indep").

Duplicate study or analysis records are fitted once, as 0.2.1 did. One explicit addition: a point with no recorded space is labelled MNI. NiMARE 0.2.1 passed a
null space through untransformed, i.e. fitted it as MNI; 0.21 refuses a dataset without a space,
so the same treatment has to be stated. Refitting the 35 best-run columns this way reproduced the
0.2.1 maps to ~1e-6 z in 32 of them; the three social columns moved by at most 0.0004 r^2,
from two analyses whose points mix a recorded and a missing space.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

NIMARE_VERSION = "0.21.0"


def selected_ids(annotation: dict, column: str) -> list[str]:
    """Analysis ids whose boolean note for `column` is true (autonima's rule)."""
    if isinstance(annotation, list):
        annotation = annotation[0] if annotation else {}
    return [n.get("analysis") for n in annotation.get("notes", [])
            if isinstance(n, dict) and isinstance(n.get("note"), dict)
            and n["note"].get(column) and n.get("analysis")]


def _has_valid_point(analysis: dict) -> bool:
    for p in analysis.get("points") or []:
        c = tuple(p.get("coordinates") or ())[:3]
        if len(c) == 3 and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in c):
            return True
    return False


def subset(studyset_data: dict, ids) -> dict:
    """The studyset holding exactly these analyses (those with a valid point), unknown space
    labelled MNI, study and analysis names set to their ids (as autonima does)."""
    keep = set(ids)
    studies, seen_studies, seen_analyses = [], set(), set()
    for s in studyset_data.get("studies", []):
        # One export (vbm_of_substance_use/v2-annotation-only) repeats three study records
        # verbatim; 0.2.1 fitted each analysis once, 0.21 drops the duplicated study entirely.
        if s.get("id") in seen_studies:
            continue
        seen_studies.add(s.get("id"))
        an = []
        for a in s.get("analyses") or []:
            if a.get("id") in seen_analyses:
                continue
            if a.get("id") in keep and _has_valid_point(a):
                seen_analyses.add(a["id"])
                a = dict(a, name=a["id"],
                         points=[dict(p, space=p.get("space") or "MNI") for p in a.get("points") or []])
                an.append(a)
        if an:
            studies.append(dict(s, name=s["id"], analyses=an))
    out = copy.copy(studyset_data)
    out["studies"] = studies
    return out


def valid_ids(studyset_data: dict, ids) -> list[str]:
    """Ids that would actually be fitted: present in the studyset with a valid point."""
    return [a["id"] for s in subset(studyset_data, ids)["studies"] for a in s["analyses"]]


def dataset(studyset_data: dict, ids):
    from nimare.nimads import Studyset
    sub = subset(studyset_data, ids)
    n = sum(len(s["analyses"]) for s in sub["studies"])
    ds = Studyset(sub).to_dataset()
    if len(ds.ids) != n:
        raise RuntimeError(f"dataset holds {len(ds.ids)} analyses, {n} were selected")
    return ds


def fit(studyset_data: dict, ids):
    """(raw z, FDR-corrected z, n analyses) from one MKDA fit."""
    from nimare.correct import FDRCorrector
    from nimare.meta.cbma import MKDADensity
    ds = dataset(studyset_data, ids)
    res = MKDADensity().fit(ds)
    cor = FDRCorrector().transform(res)
    return (np.asarray(res.get_map("z").dataobj),
            np.asarray(cor.get_map("z_corr-FDR_method-indep").dataobj),
            len(ds.ids))


def fit_to_dir(studyset_data: dict, ids, output_dir: Path) -> int:
    """Fit and write the same file set `autonima meta` wrote (CBMAWorkflow, FocusCounter)."""
    from nimare.correct import FDRCorrector
    from nimare.meta.cbma import MKDADensity
    from nimare.workflows import CBMAWorkflow
    ds = dataset(studyset_data, ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    CBMAWorkflow(estimator=MKDADensity(), corrector=FDRCorrector(),
                 diagnostics="focuscounter", output_dir=str(output_dir)).fit(ds)
    return len(ds.ids)

#!/usr/bin/env python3
"""Refit every paper map with autonima v0.1.1 on NiMARE 0.21.0, in place.

Each map directory listed in reports/maps_manifest.json is <run>/outputs/meta_analysis_results/
<column>; its analyses are the column's selection in that run's nimads_annotation.json, fitted
by autonima.meta.run_meta_analysis_for_column (the code behind `autonima meta`) via mkda_fit. Each new fit is written to a staging directory, compared
with the map it replaces, then swapped in: NiMARE's outputs in the target directory are removed
and the new ones moved over. Old maps are not kept here; maps.tar.gz holds the 0.2.1 set.

    pixi run python scripts/refit_maps.py --jobs 6 --log reports/refit_maps_autonima_v011.csv

Resumable: directories already in the log are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

NIMARE_OUTPUTS = (".nii.gz", ".tsv")
NIMARE_TEXT = ("boilerplate.txt", "references.bib")
FIELDS = ["map_dir", "column", "n_selected", "n_fitted", "r_old_new", "max_abs_dz",
          "supra_old", "supra_new", "seconds", "status"]


def map_dirs() -> list[str]:
    files = json.loads((REPO_ROOT / "reports" / "maps_manifest.json").read_text())["files"]
    return sorted({os.path.dirname(f["path"]) for f in files if f["path"].endswith("/z.nii.gz")})


def refit(rel: str, staging_root: str) -> dict:
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.WARNING)
    import nibabel as nib
    import numpy as np
    from mkda_fit import fit_to_dir, selected_ids
    from map_mask import common_mask

    t = time.time()
    target = REPO_ROOT / rel
    column = target.name
    outputs = target.parent.parent
    row = {"map_dir": rel, "column": column}
    if not (outputs / "nimads_studyset.json").exists() or not (outputs / "nimads_annotation.json").exists():
        row["status"] = "no nimads inputs on disk"
        row["seconds"] = 0
        return row
    ss = json.loads((outputs / "nimads_studyset.json").read_text())
    ann = json.loads((outputs / "nimads_annotation.json").read_text())
    ids = selected_ids(ann, column)
    row["n_selected"] = len(ids)
    stage = Path(tempfile.mkdtemp(dir=staging_root))
    try:
        row["n_fitted"] = fit_to_dir(ss, ids, stage)
        old_z = np.asarray(nib.load(str(target / "z.nii.gz")).dataobj, dtype=float)
        new_z = np.asarray(nib.load(str(stage / "z.nii.gz")).dataobj, dtype=float)
        old_c = np.asarray(nib.load(str(target / "z_corr-FDR_method-indep.nii.gz")).dataobj, dtype=float)
        new_c = np.asarray(nib.load(str(stage / "z_corr-FDR_method-indep.nii.gz")).dataobj, dtype=float)
        m = common_mask(old_z, new_z)
        row["r_old_new"] = round(float(np.corrcoef(old_z[m], new_z[m])[0, 1]), 8)
        row["max_abs_dz"] = float(np.nanmax(np.abs(old_z - new_z)))
        row["supra_old"], row["supra_new"] = int((old_c > 1.96).sum()), int((new_c > 1.96).sum())
        for f in target.iterdir():
            if f.is_file() and (f.name.endswith(NIMARE_OUTPUTS) or f.name in NIMARE_TEXT):
                f.unlink()
        for f in stage.iterdir():
            shutil.move(str(f), str(target / f.name))
        row["status"] = "ok"
    except Exception as e:  # recorded, not raised: one bad directory must not stop the rest
        row["status"] = f"error: {type(e).__name__}: {e}"[:300]
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    row["seconds"] = round(time.time() - t, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--log", type=Path, default=REPO_ROOT / "reports" / "refit_maps_autonima_v011.csv")
    ap.add_argument("--staging", type=Path, default=None, help="scratch dir for fits (default: system temp)")
    ap.add_argument("--only", nargs="*", default=None, help="substring filters on map_dir")
    args = ap.parse_args()

    from mkda_fit import _check_versions
    _check_versions()

    done = set()
    if args.log.exists():
        done = {r["map_dir"] for r in csv.DictReader(open(args.log)) if r["status"] == "ok"}
    todo = [d for d in map_dirs() if d not in done and (not args.only or any(o in d for o in args.only))]
    print(f"{len(todo)} map directories to refit ({len(done)} already done)", flush=True)
    staging = str(args.staging or tempfile.gettempdir())
    new_log = not args.log.exists()
    with open(args.log, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_log:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(refit, d, staging): d for d in todo}
            for i, fu in enumerate(as_completed(futs), 1):
                row = fu.result()
                w.writerow(row)
                fh.flush()
                print(f"[{i}/{len(todo)}] {row['status'][:60]:<8} r={row.get('r_old_new', '')} "
                      f"n={row.get('n_fitted', '')} {row['map_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Re-run each mapped annotation's ALE from a run's own nimads files.

The published `dice_matrix_*` / `pearson_matrix_*` tables were computed from
`<run>/outputs/meta_analysis_results/<name>/z.nii.gz`, and those maps never left the host
that made them. The selection behind them is recoverable though -- recomputing analyses and
unique pmids per annotation from nimads_studyset.json + nimads_annotation.json reproduces
`annotation_counts_by_run.csv` exactly, 7/7 on dementia v3-allstudies -- so re-running the
estimator is a settings question, not a data question.

MKDA density with NiMARE's default kernel, chosen by the owner. ALE was not an option: its
kernel derives its width from each experiment's sample size, and the studyset carries none --
`sample_size` appears zero times -- so whatever produced the original maps cannot have used
it either. Correction is FDR independent, which the comparison script pins by expecting
`z_corr-FDR_method-indep.nii.gz`.

Every run is regenerated the same way, baselines included, so the arms are comparable to
each other and to their own baseline. They are NOT comparable to the committed
`dice_matrix_*` tables, which were made on another host with an unrecorded estimator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def selected_ids(run_out: Path, key: str) -> set[str]:
    notes = json.loads((run_out / "nimads_annotation.json").read_text()).get("notes") or []
    return {n["analysis"] for n in notes if (n.get("note") or {}).get(key)}


def note_keys(run_out: Path) -> list[str]:
    ann = json.loads((run_out / "nimads_annotation.json").read_text())
    keys = ann.get("note_keys")
    if isinstance(keys, dict):
        return list(keys)
    out: set[str] = set()
    for n in ann.get("notes") or []:
        out |= set((n.get("note") or {}).keys())
    return sorted(out)


def build_dataset(run_out: Path, keep: set[str]):
    """A NiMARE Dataset holding only the selected analyses."""
    from nimare.io import convert_nimads_to_dataset
    from nimare.nimads import Studyset

    raw = json.loads((run_out / "nimads_studyset.json").read_text())
    studies = []
    for s in raw.get("studies", []):
        kept = [a for a in (s.get("analyses") or []) if a.get("id") in keep]
        if kept:
            studies.append({**s, "analyses": kept})
    if not studies:
        return None, 0, 0
    trimmed = {**raw, "studies": studies}
    n_an = sum(len(s["analyses"]) for s in studies)
    n_pt = sum(len(a.get("points") or []) for s in studies for a in s["analyses"])
    return convert_nimads_to_dataset(Studyset(trimmed)), n_an, n_pt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--keys", nargs="*", help="annotation keys; default every mapped key")
    ap.add_argument("--n-iters", type=int, default=5000)
    ap.add_argument("--n-cores", type=int, default=4)
    ap.add_argument("--min-analyses", type=int, default=2)
    args = ap.parse_args()

    out = args.run_dir / "outputs"
    dest_root = out / "meta_analysis_results"
    keys = args.keys or note_keys(out)
    print(f"{args.run_dir.name}: {len(keys)} annotation key(s)")

    from nimare.correct import FDRCorrector
    from nimare.meta.cbma import MKDADensity

    for key in keys:
        dest = dest_root / key
        if (dest / "z.nii.gz").exists():
            print(f"  {key}: already done")
            continue
        keep = selected_ids(out, key)
        dset, n_an, n_pt = build_dataset(out, keep)
        if dset is None or n_an < args.min_analyses:
            print(f"  {key}: {n_an} analyses -- skipped")
            continue
        print(f"  {key}: {n_an} analyses, {n_pt} points ... ", end="", flush=True)
        res = MKDADensity(null_method="approximate", n_cores=args.n_cores).fit(dset)
        corr = FDRCorrector(method="indep", alpha=0.05).transform(res)
        dest.mkdir(parents=True, exist_ok=True)
        corr.save_maps(output_dir=str(dest), prefix="")
        print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

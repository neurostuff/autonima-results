#!/usr/bin/env python3
"""Axial-slice panels comparing baseline, pipeline and expert maps for selected columns.

WHY THIS EXISTS

The Nature Methods figures are all scatters and distributions. For a neuroimaging paper that is a
conspicuous gap: every number in Result 4 is a similarity between two brain maps, and the reader
never sees one. This renders the three arms side by side so the similarity score has something to
be a summary *of*.

WHICH COLUMNS

Selection is by margin over the best available baseline, which is cherry-picking and is treated as
such: `--mode showcase` takes the largest margins for the main figure, `--mode contrast` pairs the
best with the worst so a failure is shown beside a success, and `--mode all` renders every column
for the supplement. The honest framing is that the main figure shows what a good case looks like
while Figures 4 and 5 report the whole distribution including the losses -- so run `--mode all`
for the supplement rather than only shipping the flattering three.

CORRECTNESS

Maps are resolved the way compare_baselines_to_benchmark.py resolves them: the canonical run from
run_categories.yaml at the requested tier, the manual annotation key translated through the
project's nmb_mappings.json, and the same FDR-corrected z map. Because a silently wrong path would
produce a plausible-looking figure that does not match the text, every triplet is verified by
recomputing the pipeline-vs-expert similarity and comparing against
cross_project_best_baseline.csv; a mismatch is reported and the column is skipped.

The metric is read from that table rather than assumed, because compile_best_baselines.py is
parameterised (--metric dice|r2|pearson_r) and now defaults to dice. Checking a dice column against
an r-squared computation rejects 29 of 35 columns as path errors when the paths are fine.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tiers import resolve_tier  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MANUAL_BASE = Path("/home/zorro/repos/neurometabench/analysis")
MAP_NAME = "z_corr-FDR_method-indep.nii.gz"
POOLED = REPO_ROOT / "reports" / "cross_project_best_baseline.csv"
DEFAULT_OUT = REPO_ROOT / "reports" / "nature_methods_figures"

SINGLE_COL, DOUBLE_COL = 89 / 25.4, 183 / 25.4
INK, MUTED = "#1a1a1a", "#5a5a5a"

DISPLAY = {
    "cue_reactivity": "Cue reactivity", "decision_making": "Decision making",
    "dementia": "Dementia", "emotion_regulation_2022": "Emotion regulation",
    "executive_function": "Executive function", "problem_solving": "Problem solving",
    "social": "Social", "vbm_of_ptsd": "VBM PTSD", "vbm_of_substance_use": "VBM substance use",
}


def auto_column(project: str, key: str) -> str:
    """Translate a manual annotation key into the pipeline's column name."""
    path = REPO_ROOT / "projects" / project / "nmb_mappings.json"
    if path.exists():
        try:
            mapping = json.loads(path.read_text()).get("annotation_mappings") or {}
        except ValueError:
            mapping = {}
        return mapping.get(key, key)
    return key


def resolve_maps(project: str, key: str, source: str, tier: str) -> dict[str, Path] | None:
    """Paths for the expert, pipeline and best-available-baseline maps, or None if incomplete."""
    run = resolve_tier(project, "canonical", tier)
    if not run:
        return None

    gold_dir = MANUAL_BASE / project / key
    if not (gold_dir / MAP_NAME).exists():
        # neurometabench directory names vary in case and separators.
        matches = [d for d in (MANUAL_BASE / project).glob("*")
                   if d.is_dir() and d.name.lower().replace("-", "_") == key.lower().replace("-", "_")]
        gold_dir = matches[0] if matches else gold_dir

    auto = (REPO_ROOT / "projects" / project / run / "outputs" / "meta_analysis_results"
            / auto_column(project, key) / MAP_NAME)

    # "targeted" means the per-column baseline search; "broad" means the project's all_studies arm.
    if source == "targeted":
        base = (REPO_ROOT / "projects" / project / "baselines" / key / "outputs"
                / "meta_analysis_results" / "all_analyses" / MAP_NAME)
    else:
        base = (REPO_ROOT / "projects" / project / run / "outputs" / "meta_analysis_results"
                / "all_studies" / MAP_NAME)

    paths = {"gold": gold_dir / MAP_NAME, "autonima": auto, "baseline": base}
    return paths if all(p.exists() for p in paths.values()) else None


def verify(paths: dict[str, Path], expected: float, metric: str = "r2",
           tol: float = 0.02) -> float | None:
    """Recompute the pipeline-vs-expert similarity; None if it contradicts the table.

    compile_best_baselines.py is metric-parameterised, so the check has to match whichever metric
    the table was built with or it rejects every column as a path error.
    """
    from nilearn.image import resample_to_img
    import nibabel as nib

    gold = nib.load(paths["gold"])
    auto = nib.load(paths["autonima"])
    if auto.shape != gold.shape:
        auto = resample_to_img(auto, gold, interpolation="continuous", force_resample=True,
                               copy_header=True)
    # Match compare_baselines_to_benchmark.py exactly: correlate over ALL finite voxels, not
    # just voxels non-zero in either map. Restricting to non-zero drops the shared background and
    # produces a completely different r -- which is what an earlier version of this check did, and
    # why it rejected all 35 columns as path errors when the paths were fine.
    gd, ad = gold.get_fdata(), auto.get_fdata()
    if gd.shape != ad.shape:
        return None
    mask = np.isfinite(gd) & np.isfinite(ad)
    if mask.sum() < 100:
        return None
    g, a = gd[mask].ravel(), ad[mask].ravel()
    if np.all(g == g[0]) or np.all(a == a[0]):
        return None
    if metric == "dice":
        # Same rule as compute_dice in compare_baselines_to_benchmark.py: z > 1.96, one-sided.
        ba, bb = g > 1.96, a > 1.96
        denom = int(ba.sum()) + int(bb.sum())
        got = float(2.0 * int((ba & bb).sum()) / denom) if denom else 0.0
    else:
        got = float(np.corrcoef(g, a)[0, 1] ** 2)
        if metric == "pearson_r":
            got = got ** 0.5
    return got if abs(got - expected) <= tol else None


def suprathreshold(path: Path, threshold: float) -> int:
    """Voxels surviving the display threshold. A row where nothing survives shows an empty brain."""
    import nibabel as nib
    d = np.nan_to_num(nib.load(str(path)).get_fdata())
    return int((np.abs(d) > threshold).sum())


def one_per_project(rows: list[dict]) -> list[dict]:
    """Best-margin column from each project.

    Without this the top of the list is four emotion_regulation columns, which shows the same
    contrast four times and tells the reader nothing about breadth. One row per project makes each
    row carry new information.
    """
    best: dict[str, dict] = {}
    for r in rows:
        cur = best.get(r["project"])
        if cur is None or r["delta"] > cur["delta"]:
            best[r["project"]] = r
    return sorted(best.values(), key=lambda x: -x["delta"])


def candidates(tier: str) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(POOLED)):
        paths = resolve_maps(r["project"], r["manual_annotation"],
                             r["best_available_source"], tier)
        if paths:
            rows.append({**r, "paths": paths, "delta": float(r["delta_vs_available"])})
    rows.sort(key=lambda x: -x["delta"])
    return rows


def render(rows: list[dict], out_dir: Path, name: str, cut_coords, threshold: float) -> None:
    from nilearn import plotting

    arms = [("baseline", "Search-only baseline"), ("autonima", "Full pipeline"),
            ("gold", "Expert meta-analysis")]
    n = len(rows)
    # nilearn leaves generous margins inside each axes, so rows need to be pulled together
    # explicitly or the figure is mostly whitespace.
    fig, axes = plt.subplots(n, 3, figsize=(DOUBLE_COL, 0.62 * n),
                             squeeze=False, facecolor="white")
    fig.subplots_adjust(hspace=0.02, wspace=0.02, left=0.19, right=0.93, top=0.94, bottom=0.06)
    for i, row in enumerate(rows):
        for j, (arm, arm_label) in enumerate(arms):
            ax = axes[i][j]
            plotting.plot_stat_map(
                str(row["paths"][arm]), display_mode="z", cut_coords=cut_coords,
                threshold=threshold, colorbar=False, axes=ax, annotate=False,
                black_bg=False, draw_cross=False, cmap="cold_hot",
            )
            if i == 0:
                ax.set_title(arm_label, fontsize=6.5, color=INK, pad=2)
        lbl = f"{DISPLAY.get(row['project'], row['project'])}\n{row['manual_annotation']}"
        axes[i][0].text(-0.04, 0.5, lbl, transform=axes[i][0].transAxes,
                        ha="right", va="center", fontsize=5.6, color=INK, linespacing=1.5)
        axes[i][2].text(1.01, 0.5, f"$\\Delta$ {row.get('metric', 'dice')}\n{row['delta']:+.3f}",
                        transform=axes[i][2].transAxes, ha="left", va="center",
                        fontsize=5.6, color=MUTED, linespacing=1.4)
    fig.text(0.5, 0.005, f"axial slices at z = {list(cut_coords)}, "
             f"FDR-corrected z thresholded at |z| > {threshold}",
             ha="center", fontsize=5.2, color=MUTED)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 400})):
        fig.savefig(out_dir / f"{name}{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  {name}.pdf / .png   ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("showcase", "contrast", "all"), default="contrast")
    ap.add_argument("--n", type=int, default=3, help="rows per group")
    ap.add_argument("--tier", default="best")
    ap.add_argument("--threshold", type=float, default=2.3)
    ap.add_argument("--cut-coords", type=int, nargs="*", default=[-12, 0, 12, 24, 36])
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    # 500 rather than a token 50. Measured counts in the legible rows run from ~5,000 to ~48,000
    # suprathreshold voxels; VBM PTSD has 401 in the expert map, 32 in the pipeline and 101 in the
    # baseline, which passes any small floor and still renders as three empty brains at this size.
    ap.add_argument("--min-voxels", type=int, default=500,
                    help="skip columns with fewer suprathreshold voxels than this (see comment)")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    rows = candidates(args.tier)
    print(f"columns with all three maps present: {len(rows)}/35")
    if not rows:
        print("  nothing to render"); return 1

    if not args.skip_verify:
        keep = []
        for r in rows:
            # The metric must come from the table. compile_best_baselines.py is parameterised
            # (--metric dice|r2|pearson_r) and now defaults to dice, so checking a dice column
            # against an r-squared computation rejects almost everything as a path error.
            expected = float(r["autonima"]) if "autonima" in r else float(r["autonima_r2"])
            got = verify(r["paths"], expected, metric=(r.get("metric") or "r2").lower())
            if got is None:
                print(f"  SKIP {r['project']}/{r['manual_annotation']}: "
                      f"recomputed metric disagrees with the table (path likely wrong)")
                continue
            keep.append(r)
        print(f"  verified against cross_project_best_baseline.csv: {len(keep)}/{len(rows)}")
        rows = keep

    if args.mode == "all":
        render(rows, args.output_dir, "figureS_brain_maps_all", args.cut_coords, args.threshold)
        return 0

    # Drop columns where the expert map has nothing above the display threshold. VBM PTSD is the
    # case in point: it has a healthy +0.209 margin but only 21 gold studies, so almost nothing
    # survives FDR correction and all three panels render as empty brains. That is a true fact
    # about a small meta-analysis, but it occupies a row without showing the reader anything, and
    # the margin is already reported in Figure 4.
    # The expert map must have signal (it is the reference) AND at least one of the two compared
    # arms must too, or the row is a three-way comparison of empty brains. Requiring all three
    # would be wrong in the other direction: a row where the baseline finds nothing and the
    # pipeline finds the expert pattern is among the most informative rows there is.
    def showable_row(r: dict) -> bool:
        gold_ok = suprathreshold(r["paths"]["gold"], args.threshold) >= args.min_voxels
        arms_ok = any(suprathreshold(r["paths"][a], args.threshold) >= args.min_voxels
                      for a in ("baseline", "autonima"))
        return gold_ok and arms_ok

    showable = [r for r in rows if showable_row(r)]
    dropped = len(rows) - len(showable)
    if dropped:
        print(f"  {dropped} columns would render as empty brains at |z| > {args.threshold} "
              f"(<{args.min_voxels} voxels); excluded from selection")

    # One row per project, so each row shows a different target rather than repeating one.
    diverse = one_per_project(showable)
    if args.mode == "showcase":
        render(diverse[:args.n], args.output_dir, "figure_brain_maps_showcase",
               args.cut_coords, args.threshold)
    else:
        # Best and worst together: a figure of only wins invites the reader to assume they are
        # typical, and Figures 4 and 5 already report that they are not.
        picked = diverse[:args.n] + diverse[-max(args.n - 1, 1):]
        render(picked, args.output_dir, "figure_brain_maps_contrast",
               args.cut_coords, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

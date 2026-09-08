#!/usr/bin/env python3
"""Emotion regulation on the cortical surface: three contrasts against one search-only baseline.

WHY THIS FIGURE

Figure 4 shows emotion regulation holding the three largest advantages in the whole benchmark, and
Figure 5 shows the gain surviving with the study pool held fixed. Both are scatters. The claim
underneath them is spatial and belongs on a brain: the pipeline produces a DIFFERENT map for each
regulation contrast, matching a different expert map each time, while a search-only synthesis of
the same literature produces ONE map that cannot tell the contrasts apart. That is the paper's
analysis-unit argument in a single display item.

So the layout is the argument. Three columns, one per contrast, expert above and pipeline below.
One brain to the right: the fixed-pool arm, which is a single map shared by all three columns
because nothing in it is contrast-specific.

WHICH THREE CONTRASTS, AND WHY NOT `decrease`

The project has four annotated contrasts and `decrease` is deliberately excluded. Its gold map
correlates r = 0.972 with `reappraisal`'s -- they are effectively the same map, because
neurometabench's Decrease.txt holds the reappraisal union rather than the decrease contrast (a
defect recorded in the project's nmb_mappings.json and confirmed here by direct comparison).
Showing both would present one expert map twice as if it were two independent successes, and
`decrease` happens to carry the single largest advantage in Figure 4, so the temptation is real.

`maintain` is included but is the weakest of the three: the same note reports its gold file is
missing most of the Look experiments. Its expert map correlates 0.03-0.09 with every other ER
contrast, which is consistent with a genuinely distinct contrast and equally consistent with a
broken one, so this cannot be settled from the maps alone. Use --columns to drop it.

Surfaces are fsaverage5, projected with vol_to_surf. Every panel shares one colour scale, computed
across all displayed maps, so panel-to-panel brightness is comparable rather than per-panel
normalised.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tiers import resolve_tier  # noqa: E402

MANUAL_BASE = Path("/home/zorro/repos/neurometabench/analysis")
MAP_NAME = "z_corr-FDR_method-indep.nii.gz"
PROJECT = "emotion_regulation_2022"
DEFAULT_OUT = REPO_ROOT / "reports" / "nature_methods_figures"

SINGLE_COL, DOUBLE_COL = 89 / 25.4, 183 / 25.4
INK, MUTED, RULE = "#1a1a1a", "#5a5a5a", "#c8c8c8"

# Excludes `decrease`; see the module docstring.
DEFAULT_COLUMNS = ["increase", "maintain", "reappraisal"]
PRETTY = {
    "increase": "Increase\nemotion",
    "maintain": "Maintain\n(look)",
    "reappraisal": "Reappraisal",
    "decrease": "Decrease\nemotion",
}


def house_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6, "axes.labelsize": 6.5,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def auto_column(key: str) -> str:
    path = REPO_ROOT / "projects" / PROJECT / "nmb_mappings.json"
    if path.exists():
        try:
            mapping = json.loads(path.read_text()).get("annotation_mappings") or {}
        except ValueError:
            mapping = {}
        return mapping.get(key, key)
    return key


def gold_map(key: str) -> Path | None:
    d = MANUAL_BASE / PROJECT / key
    if (d / MAP_NAME).exists():
        return d / MAP_NAME
    for c in (MANUAL_BASE / PROJECT).glob("*"):
        if c.is_dir() and c.name.lower().replace("-", "_") == key.lower().replace("-", "_"):
            return c / MAP_NAME
    return None


def resolve(columns: list[str], tier: str) -> tuple[str, dict[str, dict[str, Path]], Path]:
    run = resolve_tier(PROJECT, "canonical", tier)
    if not run:
        raise SystemExit(f"no canonical run for {PROJECT} at tier {tier}")
    mdir = REPO_ROOT / "projects" / PROJECT / run / "outputs" / "meta_analysis_results"

    maps: dict[str, dict[str, Path]] = {}
    for key in columns:
        g, a = gold_map(key), mdir / auto_column(key) / MAP_NAME
        if not (g and g.exists()):
            raise SystemExit(f"missing expert map for {key}")
        if not a.exists():
            raise SystemExit(f"missing pipeline map for {key} at {a}")
        maps[key] = {"expert": g, "pipeline": a}

    baseline = mdir / "all_studies" / MAP_NAME
    if not baseline.exists():
        raise SystemExit(f"missing fixed-pool baseline at {baseline}")
    return run, maps, baseline


def r2_against(expert: Path, other: Path) -> float:
    """R-squared between two whole-brain maps, over voxels finite in both.

    Computed from the maps this figure actually draws rather than read from
    cross_project_best_baseline.csv. That table's baseline is the *best available* per column,
    which for `increase` and `reappraisal` is a per-column targeted search rather than the
    fixed-pool arm shown here -- so quoting it beside these brains would caption one map with
    another map's number.
    """
    import nibabel as nib

    a, b = nib.load(str(expert)).get_fdata(), nib.load(str(other)).get_fdata()
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {expert} {a.shape} vs {other} {b.shape}")
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m].ravel(), b[m].ravel())[0, 1] ** 2)


def project_to_surface(path: Path, fsavg, mesh_kind: str):
    """Volume -> per-hemisphere vertex arrays on the fsaverage mesh."""
    from nilearn.surface import vol_to_surf
    import nibabel as nib

    img = nib.load(str(path))
    out = {}
    for hemi in ("left", "right"):
        # Sample along the cortical ribbon rather than at a single depth: an MKDA density map is
        # smooth but sparse, and one-depth sampling drops blobs that sit off the pial surface.
        # Linear, not nearest: these are continuous z maps. nilearn's nearest option is
        # "nearest_most_frequent" and is for label images.
        out[hemi] = vol_to_surf(
            img, fsavg[mesh_kind].parts[hemi], interpolation="linear",
            radius=3.0, n_samples=10,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    ap.add_argument("--tier", default="best")
    ap.add_argument("--threshold", type=float, default=2.3,
                    help="z threshold; 2.3 matches make_brain_map_figure.py")
    ap.add_argument("--mesh", default="inflated", choices=("inflated", "pial"))
    ap.add_argument("--panels", nargs="*", default=["left:lateral", "left:medial"],
                    help='"hemi:view" pairs drawn for every map, e.g. left:lateral right:lateral')
    ap.add_argument("--vmax", type=float, default=5.0,
                    help="shared colour ceiling. The maps differ enormously in z scale -- the "
                         "fixed-pool baseline peaks at 23 while `increase` peaks at 6.4 -- so a "
                         "ceiling set from the maximum renders the weaker maps invisible. 5 keeps "
                         "all of them legible and saturates the strongest, which is the "
                         "conservative direction: the claim is that the baseline is "
                         "undifferentiated, not that it is weak.")
    ap.add_argument("--cmap", default="YlOrRd")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--name", default="figure_er_surface_contrasts")
    args = ap.parse_args(argv)

    from nilearn import datasets
    from nilearn.plotting import plot_surf_stat_map

    house_style()
    run, maps, baseline = resolve(args.columns, args.tier)
    print(f"run {run}; columns {', '.join(args.columns)}")

    fsavg = datasets.load_fsaverage("fsaverage5")
    surf_mesh = fsavg[args.mesh]
    bg = datasets.load_fsaverage_data(mesh="fsaverage5", data_type="sulcal", mesh_type=args.mesh)

    # Project every map first, so one colour scale can be computed across all of them.
    surfaces: dict[tuple[str, str], dict] = {}
    for key, arms in maps.items():
        for arm, path in arms.items():
            surfaces[(arm, key)] = project_to_surface(path, fsavg, args.mesh)
    surfaces[("baseline", "all_studies")] = project_to_surface(baseline, fsavg, args.mesh)

    panels = [tuple(p.split(":", 1)) for p in args.panels]
    vmax = args.vmax
    print(f"shared colour scale: threshold {args.threshold}, vmax {vmax}")
    for (arm, key), surf in surfaces.items():
        peak = max(float(np.nanmax(surf[h])) for h, _ in panels)
        print(f"    {arm:9} {key:12} peak on shown surfaces {peak:6.2f}"
              f"{'  (saturated)' if peak > vmax else ''}")

    ncol, nview = len(args.columns), len(panels)
    # Contrast columns, a spacer, then the single baseline column.
    width_ratios = [1.0] * (ncol * nview) + [0.30] + [1.0] * nview
    fig = plt.figure(figsize=(DOUBLE_COL, 2.35))
    gs = fig.add_gridspec(2, len(width_ratios), width_ratios=width_ratios,
                          wspace=0.0, hspace=0.0, left=0.055, right=0.995,
                          top=0.87, bottom=0.155)

    def draw(ax, data, hemi, view):
        plot_surf_stat_map(
            surf_mesh.parts[hemi], data[hemi], hemi=hemi, view=view,
            bg_map=bg.data.parts[hemi], bg_on_data=True,
            threshold=args.threshold, vmax=vmax, cmap=args.cmap,
            colorbar=False, axes=ax, figure=fig, engine="matplotlib",
        )
        ax.set_axis_off()
        # Zoom the 3D content inside its own axes rather than enlarging the axes box. Expanding
        # the box overlaps neighbours, and a 3D axes clips against its own bounds -- which is what
        # sliced a vertical edge off the baseline brain in the first draft.
        ax.set_box_aspect(None, zoom=1.45)

    for r, arm in enumerate(("expert", "pipeline")):
        for c, key in enumerate(args.columns):
            for v, (hemi, view) in enumerate(panels):
                ax = fig.add_subplot(gs[r, c * nview + v], projection="3d")
                draw(ax, surfaces[(arm, key)], hemi, view)
        # One row label per arm, on the left edge.
        fig.text(0.012, 0.68 - 0.40 * r, {"expert": "Expert", "pipeline": "Pipeline"}[arm],
                 rotation=90, va="center", ha="center", fontsize=6.5, color=INK)

    # The single fixed-pool baseline, drawn once and spanning both rows.
    for v, (hemi, view) in enumerate(panels):
        ax = fig.add_subplot(gs[:, ncol * nview + 1 + v], projection="3d")
        draw(ax, surfaces[("baseline", "all_studies")], hemi, view)

    # Column headers, centred over each contrast's panel group.
    span = (0.995 - 0.055)
    unit = span / sum(width_ratios)
    x = 0.055
    for c, key in enumerate(args.columns):
        w = unit * nview
        fig.text(x + w / 2, 0.90, PRETTY.get(key, key), ha="center", va="bottom",
                 fontsize=6.5, color=INK, linespacing=1.25)
        x += w
    x += unit * width_ratios[ncol * nview]
    fig.text(x + unit * nview / 2, 0.90, "Search-only baseline\n(one map, all contrasts)",
             ha="center", va="bottom", fontsize=6.5, color=INK, linespacing=1.25)

    # Similarity to the expert map, for the maps drawn here.
    x = 0.055
    print("  R^2 vs expert (maps as drawn):")
    for c, key in enumerate(args.columns):
        w = unit * nview
        rp = r2_against(maps[key]["expert"], maps[key]["pipeline"])
        rb = r2_against(maps[key]["expert"], baseline)
        print(f"    {key:12} pipeline {rp:.3f}   baseline {rb:.3f}")
        fig.text(x + w / 2, 0.085,
                 f"$R^2$  pipeline {rp:.2f}  \u00b7  baseline {rb:.2f}",
                 ha="center", va="bottom", fontsize=5.6, color=MUTED)
        x += w

    # Shared colour bar, small and out of the way.
    import matplotlib as mpl
    cax = fig.add_axes([0.845, 0.075, 0.105, 0.024])
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=plt.get_cmap(args.cmap), orientation="horizontal",
        norm=mpl.colors.Normalize(vmin=args.threshold, vmax=vmax))
    cb.set_ticks([args.threshold, vmax])
    cb.set_ticklabels([f"{args.threshold:g}", f"\u2265{vmax:g}"])
    cb.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=5.2, length=1.6, width=0.4, colors=MUTED, pad=1.5)
    cax.set_title("z (FDR)", fontsize=5.2, color=MUTED, pad=2)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 400})):
        fig.savefig(out / f"{args.name}{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  {args.name}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

WHICH THREE CONTRASTS, AND WHY NOT `reappraisal`

The project has four annotated contrasts and `reappraisal` is deliberately excluded, leaving the
three regulation GOALS: increase, decrease, maintain. Reappraisal is a STRATEGY -- a different
axis -- and it is not independent of decrease: in neurometabench's committed annotation
`decrease` holds 128 experiments and `reappraisal` 154, with ALL 128 shared. decrease is a
perfect subset, Jaccard 0.831, and the two gold maps correlate r = 0.970. Showing both would
present those 128 experiments twice as if they were two independent successes.

THIS REVERSES AN EARLIER CHOICE (2026-11-06). The figure previously dropped `decrease` and kept
`reappraisal`, on the stated grounds that neurometabench's Decrease.txt was a corrupted file
"holding the reappraisal union". That was wrong -- see projects/emotion_regulation_2022/
nmb_mappings.json, corrected the same day. The overlap is nested labelling, not a bad file, so
the question was never which file to trust but which axis to report. Keeping the three goals is
the coherent set.

The exclusion now lives in scripts/benchmark_exclusions.py so it reaches the pooled statistics
too, not just this figure. Before that it did not: all FOUR ER columns were entering
cross_project_best_baseline.csv and therefore the 28-column denominator, double-counting those
128 experiments in every aggregate.

`maintain` is the weakest of the three. The committed annotation holds 94 maintain experiments
against an expected 104, so ~10 are unaccounted for -- not the ~97 an earlier note claimed. Its
expert map correlates 0.03-0.09 with every other ER contrast, which is consistent with a
genuinely distinct contrast and equally consistent with a lossy one, so that cannot be settled
from the maps alone. Use --columns to drop it.

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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_tiers import resolve_tier  # noqa: E402
from map_mask import common_mask  # noqa: E402

MANUAL_BASE = REPO_ROOT.parent / "neurometabench" / "analysis"
CORRECTED_MAP = "z_corr-FDR_method-indep.nii.gz"
UNCORRECTED_MAP = "z.nii.gz"
MAP_NAME = CORRECTED_MAP  # what gets rendered
PROJECT = "emotion_regulation_2022"
DEFAULT_OUT = REPO_ROOT / "reports" / "nature_methods_figures"

SINGLE_COL, DOUBLE_COL = 89 / 25.4, 183 / 25.4
INK, MUTED, RULE = "#1a1a1a", "#5a5a5a", "#c8c8c8"

# Excludes `decrease`; see the module docstring.
# Deck preset, matching the other figure scripts: --font-scale raises label sizes relative to the
# panels, which is what projected legibility needs. Default 1.0 leaves publication output alone.
FONT_SCALE = 1.0


def fs(size: float) -> float:
    return size * FONT_SCALE


DEFAULT_COLUMNS = ["increase", "decrease", "maintain"]
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
        "font.size": fs(6), "axes.labelsize": fs(6.5),
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


# Matches compare_baselines_to_benchmark.DICE_THRESHOLD, and it is not an arbitrary display
# choice: on these maps `z_corr > 1.96` selects EXACTLY the voxels with FDR-corrected p <= 0.05.
# Verified voxel-identical on five columns across three projects -- z_corr is a strictly monotone
# transform of the corrected p map (Spearman -1.0000 over tested voxels), so thresholding the z is
# thresholding the q.
#
# One caveat worth knowing before defending the number. That equivalence uses the TWO-tailed z
# convention, while MKDA is one-tailed (positive only), for which q <= 0.05 is z ~ 1.645. NiMARE's
# own `label_corr-FDR_method-indep_tail-positive` mask uses the one-tailed boundary and so flags
# more voxels than 1.96 does -- 4,865 vs 2,539 for emotion regulation `increase`, 22,096 vs 18,427
# for `reappraisal`. So 1.96 is CONSERVATIVE: it discards voxels NiMARE calls significant. That is
# the safe direction for a figure making a positive claim, and it keeps this figure's dice
# comparable with every other dice number in the paper, which is why it is kept.
DICE_THRESHOLD = 1.96


def similarity(expert: Path, other: Path, metric: str, threshold: float) -> float:
    """Dice of the thresholded maps, or unthresholded R-squared.

    Dice is the right metric for this figure and r-squared is not, for the same reason the
    axial-slice figure gives: these panels are rendered *thresholded*, and dice measures overlap
    of thresholded maps, so it describes what the reader can actually see. R-squared measures
    unthresholded correlation over the whole volume, most of which is not on the page.

    That only holds if the dice threshold is the threshold being displayed, which is why the two
    are the same number here rather than separately configurable.

    Computed from the maps this figure draws rather than read from cross_project_best_baseline.csv,
    whose baseline for two of these columns is a targeted search and not the map shown.
    """
    import nibabel as nib

    # The metric decides the map, not the caller: r-squared compares unthresholded maps and dice
    # compares thresholded ones, so asking for r2 swaps both paths to the raw z.
    if metric == "r2":
        expert, other = (expert.parent / UNCORRECTED_MAP, other.parent / UNCORRECTED_MAP)
    a, b = nib.load(str(expert)).get_fdata(), nib.load(str(other)).get_fdata()
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {expert} {a.shape} vs {other} {b.shape}")
    if metric == "r2":
        m = common_mask(a, b)
        return float(np.corrcoef(a[m].ravel(), b[m].ravel())[0, 1] ** 2)
    ba, bb = a > threshold, b > threshold
    total = ba.sum() + bb.sum()
    return 0.0 if total == 0 else float(2.0 * (ba & bb).sum() / total)


def verify_pipeline(columns: list[str], values: dict[str, float], metric: str,
                    threshold: float) -> None:
    """Check the pipeline-vs-expert numbers against the committed per-project table.

    A silently wrong map path or a mismatched threshold would produce a plausible figure whose
    numbers do not match the text, which is the failure worth engineering against. The table's
    dice is computed at 1.96, so a figure rendered at any other threshold cannot agree with it --
    this is what catches that.
    """
    table = REPO_ROOT / "projects" / PROJECT / "reports" / "baseline_vs_autonima.csv"
    if not table.exists():
        print("  (no per-project table to verify against)")
        return
    want = {r["manual_annotation"]: r for r in csv.DictReader(open(table))
            if r["arm"] == "autonima"}
    key = "dice" if metric == "dice" else "r2"
    for c in columns:
        if c not in want or not want[c].get(key):
            continue
        expected, got = float(want[c][key]), values[c]
        flag = "ok" if abs(expected - got) < 0.005 else "MISMATCH"
        note = "" if flag == "ok" else (
            f"  <- table {key} is computed at z > {DICE_THRESHOLD}; this figure renders at "
            f"{threshold}")
        print(f"    verify {c:12} figure {got:.4f}  table {expected:.4f}  {flag}{note}")


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



def draw_rows_layout(fig, args, surfaces, maps, baseline, panels, draw, fs_, INK_, MUTED_):
    """Reference / pipeline / baseline as three stacked rows, one column per contrast.

    WHY THIS EXISTS alongside the columns layout. In the columns layout the search-only baseline
    sits as a single brain off to the right, spanning both rows. That placement is logically
    right -- it is one map shared by every contrast -- but it reads as a fourth contrast, which
    is the opposite of the point. Stacking it as its own ROW makes the repetition visible: the
    reference and pipeline rows change across columns, the baseline row does not.

    The arcs carry the comparison the figure is actually about. Each contrast gets a gutter
    holding two curved connectors from the reference row: one to the pipeline row, one to the
    baseline row, each labelled with that pair's similarity. Both start at the reference because
    that is the only comparison that matters here -- pipeline-to-baseline is not a claim the
    figure makes.
    """
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch

    ncol, nview = len(args.columns), len(panels)
    # The gutter has to hold the deeper arc AND its label, so it is sized from those two
    # numbers rather than guessed. BULGE/LABEL_PAD are in FIGURE coordinates.
    # Both arcs now span ONE row (reference sits between the two things compared to it), so a
    # single bulge serves both and they mirror rather than nest.
    BULGE, LABEL_PAD = 0.020, 0.005
    GUT = 0.95
    width_ratios = []
    for _ in range(ncol):
        width_ratios += [1.0] * nview + [GUT]
    # 2.95in, not 3.45: three rows of 3D axes leave a lot of internal padding, so the
    # rows drifted apart and the brains read small. Shortening the canvas closes the
    # gaps without touching the per-axes zoom, which clips if pushed.
    fig.set_size_inches(DOUBLE_COL, 2.95)
    gs = fig.add_gridspec(3, len(width_ratios), width_ratios=width_ratios,
                          wspace=0.0, hspace=0.0, left=0.075, right=0.995,
                          top=0.885, bottom=0.075)

    # AutoNIMA / reference / baseline. The reference goes in the MIDDLE so that each comparison
    # it anchors is one row away: the two arcs are then the same length and mirror each other,
    # instead of one being twice the span of the other and having to nest around it.
    ARMS = ("pipeline", "expert", "baseline")
    ROW_LABEL = {"expert": "Reference", "pipeline": "AutoNIMA",
                 "baseline": "Search-only\nbaseline"}
    for r, arm in enumerate(ARMS):
        for c, key in enumerate(args.columns):
            # The baseline is ONE map; drawing it in every column is the point, not an error.
            surf = surfaces[("baseline", "all_studies")] if arm == "baseline" \
                else surfaces[(arm, key)]
            for v, (hemi, view) in enumerate(panels):
                ax = fig.add_subplot(gs[r, c * (nview + 1) + v], projection="3d")
                draw(ax, surf, hemi, view)
        fig.text(0.018, 0.80 - 0.272 * r, ROW_LABEL[arm], rotation=90, va="center",
                 ha="center", fontsize=fs_(6.2), color=INK_, linespacing=1.15)

    span = 0.995 - 0.075
    unit = span / sum(width_ratios)
    label = "Dice" if args.metric == "dice" else "$R^2$"

    top, bottom = 0.885, 0.075
    rowh = (top - bottom) / 3
    y = [top - rowh * (i + 0.5) for i in range(3)]

    over = fig.add_axes([0, 0, 1, 1], zorder=5)
    over.set_axis_off()
    over.set_xlim(0, 1)
    over.set_ylim(0, 1)
    over.patch.set_alpha(0)

    def arc(x0, y0, y1, bulge, color):
        """A quadratic Bezier bulging right by `bulge` figure-units at its apex.

        arc3's `rad` is a fraction of the CHORD, so the reference->baseline connector (twice the
        span of reference->pipeline) bulged twice as far and ran clean into the next column. An
        explicit control point fixes the apex where the gutter can hold it, whatever the span.
        """
        ctrl = x0 + 2 * bulge          # apex of a quadratic Bezier sits at half the control offset
        pth = MPath([(x0, y0), (ctrl, (y0 + y1) / 2), (x0, y1)],
                    [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
        over.add_patch(PathPatch(pth, fill=False, lw=0.9, edgecolor=color, zorder=6))
        return x0 + bulge              # where the apex actually is, for the label

    x = 0.075
    print(f"  {args.metric} vs reference (maps as drawn, z > {args.threshold}):")
    pipeline_vals = {}
    for c, key in enumerate(args.columns):
        w = unit * nview
        fig.text(x + w / 2, 0.905, PRETTY.get(key, key), ha="center", va="bottom",
                 fontsize=fs_(6.5), color=INK_, linespacing=1.25)
        rp = similarity(maps[key]["expert"], maps[key]["pipeline"], args.metric, args.threshold)
        rb = similarity(maps[key]["expert"], baseline, args.metric, args.threshold)
        pipeline_vals[key] = rp
        print(f"    {key:12} pipeline {rp:.3f}   baseline {rb:.3f}")

        gx = x + w + unit * GUT * 0.10
        # Both arcs start at the reference row (y[1]): up to AutoNIMA, down to the baseline.
        for (y0, y1), val, col, weight in (
                ((y[1], y[0]), rp, INK_, "bold"),
                ((y[1], y[2]), rb, MUTED_, "normal")):
            apex = arc(gx, y0, y1, BULGE, col)
            over.text(apex + LABEL_PAD, (y0 + y1) / 2, f"{val:.2f}", ha="left", va="center",
                      fontsize=fs_(5.6), color=col, fontweight=weight, zorder=7)
        x += w + unit * GUT

    fig.text(0.075 + (span) / 2, 0.012,
             f"arcs: {label} against the reference map "
             f"(upper = AutoNIMA, lower = search-only baseline)",
             ha="center", va="bottom", fontsize=fs_(5.4), color=MUTED_)
    return pipeline_vals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    ap.add_argument("--tier", default="best")
    ap.add_argument("--metric", choices=("dice", "r2"), default="dice",
                    help="dice of the thresholded maps (default -- describes the rendered "
                         "panels) or unthresholded R-squared")
    ap.add_argument("--threshold", type=float, default=DICE_THRESHOLD,
                    help="z threshold for BOTH the rendering and the dice, deliberately one "
                         "number. Defaults to 1.96, which is the threshold the project's dice "
                         "tables use, so the reported dice matches both the picture and the "
                         "rest of the paper.")
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
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help="scale all label sizes; use ~1.5 for slides")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--name", default="figure5_er_surface_contrasts")
    ap.add_argument("--layout", choices=("columns", "rows"), default="columns",
                    help="columns (default): expert over pipeline, baseline as a separate brain "
                         "to the right. rows: reference / pipeline / baseline stacked as three "
                         "rows, with arcs in a gutter beside each contrast giving that "
                         "contrast's similarity to the reference.")
    args = ap.parse_args(argv)

    from nilearn import datasets
    from nilearn.plotting import plot_surf_stat_map

    global FONT_SCALE
    FONT_SCALE = args.font_scale
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

    def draw(ax, data, hemi, view):
        plot_surf_stat_map(
            surf_mesh.parts[hemi], data[hemi], hemi=hemi, view=view,
            bg_map=bg.data.parts[hemi], bg_on_data=True,
            threshold=args.threshold, vmax=vmax, cmap=args.cmap,
            colorbar=False, axes=ax, figure=fig, engine="matplotlib",
        )
        ax.set_axis_off()
        ax.set_box_aspect(None, zoom=1.45)

    if args.layout == "rows":
        fig = plt.figure(figsize=(DOUBLE_COL, 2.95))
        pipeline_vals = draw_rows_layout(fig, args, surfaces, maps, baseline, panels,
                                         draw, fs, INK, MUTED)
        verify_pipeline(args.columns, pipeline_vals, args.metric, args.threshold)
        add_colorbar(fig, args, vmax, rect=[0.845, 0.028, 0.105, 0.020])
        return save(fig, args)

    # Contrast columns, a spacer, then the single baseline column.
    width_ratios = [1.0] * (ncol * nview) + [0.30] + [1.0] * nview
    fig = plt.figure(figsize=(DOUBLE_COL, 2.35))
    gs = fig.add_gridspec(2, len(width_ratios), width_ratios=width_ratios,
                          wspace=0.0, hspace=0.0, left=0.055, right=0.995,
                          top=0.87, bottom=0.155)

    for r, arm in enumerate(("expert", "pipeline")):
        for c, key in enumerate(args.columns):
            for v, (hemi, view) in enumerate(panels):
                ax = fig.add_subplot(gs[r, c * nview + v], projection="3d")
                draw(ax, surfaces[(arm, key)], hemi, view)
        # One row label per arm, on the left edge.
        fig.text(0.012, 0.68 - 0.40 * r, {"expert": "Expert", "pipeline": "Pipeline"}[arm],
                 rotation=90, va="center", ha="center", fontsize=fs(6.5), color=INK)

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
                 fontsize=fs(6.5), color=INK, linespacing=1.25)
        x += w
    x += unit * width_ratios[ncol * nview]
    fig.text(x + unit * nview / 2, 0.90, "Search-only baseline\n(one map, all contrasts)",
             ha="center", va="bottom", fontsize=fs(6.5), color=INK, linespacing=1.25)

    # Similarity to the expert map, for the maps drawn here.
    label = "Dice" if args.metric == "dice" else "$R^2$"
    x = 0.055
    print(f"  {args.metric} vs expert (maps as drawn, z > {args.threshold}):")
    pipeline_vals = {}
    for c, key in enumerate(args.columns):
        w = unit * nview
        rp = similarity(maps[key]["expert"], maps[key]["pipeline"], args.metric, args.threshold)
        rb = similarity(maps[key]["expert"], baseline, args.metric, args.threshold)
        pipeline_vals[key] = rp
        print(f"    {key:12} pipeline {rp:.3f}   baseline {rb:.3f}")
        # At deck font scale the long form runs into its neighbours -- three of these across
        # three columns is already tight at 1.0. Compact it rather than let them overlap.
        text = (f"{label}  pipeline {rp:.2f}  \u00b7  baseline {rb:.2f}" if FONT_SCALE <= 1.2
                else f"{label} {rp:.2f} vs {rb:.2f}")
        fig.text(x + w / 2, 0.085, text,
                 ha="center", va="bottom", fontsize=fs(5.6), color=MUTED)
        x += w
    verify_pipeline(args.columns, pipeline_vals, args.metric, args.threshold)

    add_colorbar(fig, args, vmax, rect=[0.845, 0.075, 0.105, 0.024])
    return save(fig, args)


def add_colorbar(fig, args, vmax, rect) -> None:
    """The shared colour bar. Both layouts use it; only its rectangle differs."""
    import matplotlib as mpl
    cax = fig.add_axes(rect)
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=plt.get_cmap(args.cmap), orientation="horizontal",
        norm=mpl.colors.Normalize(vmin=args.threshold, vmax=vmax))
    cb.set_ticks([args.threshold, vmax])
    cb.set_ticklabels([f"{args.threshold:g}", f"\u2265{vmax:g}"])
    cb.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=5.2, length=1.6, width=0.4, colors=MUTED, pad=1.5)
    cax.set_title("z (FDR)", fontsize=fs(5.2), color=MUTED, pad=2)


def save(fig, args) -> int:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 400})):
        fig.savefig(out / f"{args.name}{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  {args.name}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

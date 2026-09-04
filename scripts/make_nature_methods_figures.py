#!/usr/bin/env python3
"""Publication figures for the Nature Methods submission, at Nature's format specs.

Distinct from make_cross_project_publication_plots.py, which targets a poster: large bold type,
tinted backgrounds, one idea per slide read from three metres away. Nature figures are the
opposite -- 89mm or 183mm wide, 5-7pt sans-serif, white, dense, read at arm's length with a
caption doing the explaining. Rather than parameterise the poster module into unreadability, this
is a separate module with its own house style.

Figures map onto NATURE_METHODS_SKELETON.md:

    Figure 2  gold recovery + precision gain     Result 2  (§1)
    Figure 3  analysis-level annotation          Result 3  (§5)
    Figure 4  pipeline vs best baseline          Result 4  (§7)  <- headline
    Figure 5  where the gain comes from          Result 5  (§6)  <- the thesis
    Figure 6  measured cost per stage            Result 6  (S1)

Figure 1 is a schematic (pipeline, benchmark, and the paper-vs-analysis unit) and is not
generated here -- it wants a vector editor, and panel c is an argument rather than a plot.

Every panel is generated from a committed report CSV, so figures cannot drift from the numbers
in the text. The exception is Figure 6, whose per-stage costs currently live only in the S1 table
in PAPER_OUTLINE.md; those are transcribed below as COST_PER_STAGE and should move into a
generated CSV before submission.

Usage:
    python scripts/make_nature_methods_figures.py                  # all figures
    python scripts/make_nature_methods_figures.py --only 4 5       # just those
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "reports" / "nature_methods_figures"

# Nature: single column 89mm, double column 183mm, max height 247mm.
SINGLE_COL = 89 / 25.4
DOUBLE_COL = 183 / 25.4

# Ordered so a project keeps its colour across every figure -- a reader tracking dementia through
# Figures 2-5 should not have to re-learn the legend each time.
PROJECT_ORDER = [
    "cue_reactivity", "decision_making", "dementia", "emotion_regulation_2022",
    "executive_function", "problem_solving", "social", "vbm_of_ptsd", "vbm_of_substance_use",
]
DISPLAY = {
    "cue_reactivity": "Cue reactivity", "decision_making": "Decision making",
    "dementia": "Dementia", "emotion_regulation_2022": "Emotion regulation",
    "executive_function": "Executive function", "problem_solving": "Problem solving",
    "social": "Social", "vbm_of_ptsd": "VBM PTSD", "vbm_of_substance_use": "VBM substance use",
}
# Colour-blind safe qualitative set, which matters at 6pt where shape cues are weak. Okabe-Ito
# with its yellow (#F0E442) replaced by a violet: yellow-on-white is too low-contrast to read as a
# 3pt marker, which is the size these actually print at.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#785EF0", "#000000", "#7F7F7F"]
COLORS = dict(zip(PROJECT_ORDER, PALETTE))

INK, MUTED, RULE = "#1a1a1a", "#5a5a5a", "#c8c8c8"

# From S1 in PAPER_OUTLINE.md, measured from per-stage token accounting (usage_total in
# execution_progress.json). gpt-5-mini at $0.25 / $0.03 / $2.00 per 1M in / cached / out.
COST_PER_STAGE = [
    # stage, calls, input/call, cached fraction, output/call, $/call
    ("Abstract\nscreening", 708, 1007, 0.109, 1034, 0.0023),
    ("Full-text\nscreening", 1211, 46310, 0.017, 1211, 0.0138),
    ("Coordinate\nparsing", 1299, 4962, 0.102, 2359, 0.0059),
    ("Annotation", 1091, 52117, 0.070, 4424, 0.0211),
]
PRICE_IN, PRICE_CACHED, PRICE_OUT = 0.25e-6, 0.03e-6, 2.00e-6


def house_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "axes.labelsize": 6.5, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "grid.color": RULE, "grid.linewidth": 0.4,
        "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "lines.linewidth": 0.9, "lines.markersize": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text, not outlines -- Nature requires it
    })


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.06) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="top", ha="left", color=INK)


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 400})):
        fig.savefig(out_dir / f"{name}{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- Figure 2

def figure2(out_dir: Path) -> None:
    """Where the gold standard is lost, and what screening does to precision.

    An earlier version plotted the stage-progression file's recall column directly and it rose
    across stages, which is impossible for cumulative recall -- screening can only discard. The
    cause is that the file holds *conditional retention* (the share of studies entering a stage
    that survive it), not cumulative recall, so its denominator changes at every stage. Plotted as
    "recall against gold standard" it invited exactly the wrong reading.

    Panel a instead reports the quantity a reader actually wants: what fraction of the gold
    standard the pipeline ends up with, and where the rest went. It uses two numbers that are
    unambiguous -- search recall, and end-to-end cumulative recall -- so the three shares sum to
    exactly 100% by construction and no stage-conditional arithmetic is involved.

    The decomposition is also the more useful result. Loss is dominated by the search query in
    several projects (decision_making 40%, executive_function 26%), which is the human-written
    query limitation, not a screening failure.
    """
    prog: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in read(REPO_ROOT / "reports" / "cross_project_screening"
                  / "screening_metrics_top_v_stage_progression.csv"):
        if r["metric"] == "recall":
            try:
                prog[r["project_name"]][r["stage"]] = float(r["value"])
            except ValueError:
                continue
    cum: dict[str, float] = {}
    prec: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in read(REPO_ROOT / "reports" / "cross_project_screening"
                  / "screening_metrics_top_v.csv"):
        if r["stage"] == "fulltext":
            try:
                cum[r["project_name"]] = float(r["recall"])
            except ValueError:
                continue
    for r in read(REPO_ROOT / "reports" / "cross_project_screening"
                  / "screening_metrics_top_v_stage_progression.csv"):
        if r["metric"] == "precision":
            try:
                prec[r["project_name"]][r["stage"]] = float(r["value"])
            except ValueError:
                continue

    projects = [p for p in PROJECT_ORDER if p in cum and "search" in prog.get(p, {})]
    projects.sort(key=lambda p: cum[p])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.3),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # a: what happened to the gold standard, as shares that sum to 1
    ax = axes[0]
    y = range(len(projects))
    never = [(1 - prog[p]["search"]) * 100 for p in projects]
    after = [(prog[p]["search"] - cum[p]) * 100 for p in projects]
    kept = [cum[p] * 100 for p in projects]
    left = [0.0] * len(projects)
    for vals, colr, lab in ((kept, "#009E73", "Recovered"),
                            (after, "#E69F00", "Lost after search"),
                            (never, "#D55E00", "Never found by search")):
        ax.barh(list(y), vals, 0.68, left=left, color=colr, lw=0, label=lab)
        left = [l + v for l, v in zip(left, vals)]
    for i, p in enumerate(projects):
        ax.text(kept[i] - 1.5, i, f"{kept[i]:.0f}%", va="center", ha="right",
                fontsize=5.5, color="white")
    ax.set_yticks(list(y))
    ax.set_yticklabels([DISPLAY[p] for p in projects])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of gold-standard studies (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3,
              handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    panel_label(ax, "a", dx=-0.34)

    # b: precision, which does climb monotonically and is the claim being made
    ax = axes[1]
    stages = ["search", "abstract", "fulltext"]
    for p in projects:
        ys = [prec.get(p, {}).get(s) for s in stages]
        if any(v is None for v in ys):
            continue
        ax.plot(range(len(stages)), ys, "-o", color=COLORS[p], markeredgewidth=0, alpha=0.85)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(["Search", "Abstract", "Full text"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Precision vs gold standard")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    panel_label(ax, "b", dx=-0.24)

    fig.subplots_adjust(wspace=0.42)
    save(fig, out_dir, "figure2_gold_recovery_and_precision")


# --------------------------------------------------------------------------- Figure 3

def figure3(out_dir: Path) -> None:
    """Analysis-level annotation, exhausted-manual basis, dementia excluded.

    dementia is dropped because its source meta-analysis pools several studies into one gold
    analysis, so per-paper extraction has no clean mapping to its units -- an artefact of the
    benchmark, not a result about the method.
    """
    rows = read(REPO_ROOT / "reports" / "cross_project_analysis" / "annotation_aggregates.csv")
    keep = [r for r in rows
            if r["level"] == "analysis"
            and r["variant"] == "exhausted_manual_assumption"
            and r["scope"] == "project"
            and r["project_name"] != "dementia"]
    keep = [r for r in keep if r["project_name"] in DISPLAY]
    keep.sort(key=lambda r: float(r["f1"]))
    if not keep:
        print("  figure3: no rows matched; skipped")
        return

    labels = [DISPLAY[r["project_name"]] for r in keep]
    y = range(len(keep))
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 0.26 * len(keep) + 0.9))
    for i, r in enumerate(keep):
        p, rec = float(r["precision"]), float(r["recall"])
        ax.plot([min(p, rec), max(p, rec)], [i, i], color=RULE, lw=1.4, zorder=1)
        ax.scatter([p], [i], s=14, color="#0072B2", zorder=3, edgecolors="white", linewidths=0.4)
        ax.scatter([rec], [i], s=14, color="#D55E00", zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Precision / recall vs expert annotation")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#0072B2", markersize=3.4, label="Precision"),
                       Line2D([], [], marker="o", ls="", color="#D55E00", markersize=3.4, label="Recall")],
              loc="lower right", handletextpad=0.3)
    save(fig, out_dir, "figure3_analysis_annotation")


# --------------------------------------------------------------------------- Figure 4

def figure4(out_dir: Path) -> None:
    """Headline: the pipeline against the strongest baseline available per column."""
    rows = read(REPO_ROOT / "reports" / "cross_project_best_baseline.csv")
    stats = {r["comparison"]: r for r in
             read(REPO_ROOT / "reports" / "cross_project_best_baseline_stats.csv")}
    s = stats.get("best_available_r2")

    pts = [(r["project"], float(r["autonima_r2"]), float(r["best_available_r2"]),
            float(r["delta_vs_available"])) for r in rows]
    pts.sort(key=lambda t: t[3])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # a: every column, autonima against its own best baseline
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=RULE, lw=0.6, zorder=1)
    for proj, auto, base, _ in pts:
        ax.scatter([base], [auto], s=13, color=COLORS.get(proj, "#7F7F7F"),
                   edgecolors="white", linewidths=0.35, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Best baseline $R^2$"); ax.set_ylabel("Pipeline $R^2$")
    ax.set_aspect("equal")
    ax.text(0.04, 0.93, "above the line =\npipeline better", fontsize=5.5, color=MUTED,
            transform=ax.transAxes, va="top")
    ax.grid(alpha=0.6); ax.set_axisbelow(True)
    panel_label(ax, "a", dx=-0.20)

    # b: per-column delta, sorted, with the pooled cluster-bootstrap CI behind it
    ax = axes[1]
    if s:
        lo, hi, mean = float(s["ci_low"]), float(s["ci_high"]), float(s["mean_delta"])
        ax.axhspan(lo, hi, color="#0072B2", alpha=0.10, lw=0, zorder=0)
        ax.axhline(mean, color="#0072B2", lw=0.8, zorder=2)
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    for i, (proj, _, _, dl) in enumerate(pts):
        ax.bar(i, dl, width=0.78, color=COLORS.get(proj, "#7F7F7F"), lw=0, zorder=3)
    ax.set_xlim(-1, len(pts))
    ax.set_xticks([])
    ax.set_xlabel(f"{len(pts)} benchmark columns, ordered by advantage")
    ax.set_ylabel("$\\Delta R^2$ vs best baseline")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    if s:
        ax.text(0.03, 0.96,
                f"$\\Delta$ = {mean:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]\n"
                f"sign test $P$ = {float(s['sign_test_p']):.1e}",
                transform=ax.transAxes, va="top", fontsize=5.5, color=INK, linespacing=1.5)
    panel_label(ax, "b", dx=-0.18)

    handles = [Line2D([], [], marker="o", ls="", color=COLORS[p], markersize=3.2,
                      label=DISPLAY[p]) for p in PROJECT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.16),
               handletextpad=0.3, columnspacing=1.1)
    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "figure4_pipeline_vs_baseline")


# --------------------------------------------------------------------------- Figure 5

def figure5(out_dir: Path) -> None:
    """The thesis: with the study pool held fixed, analysis selection still improves the map."""
    rows = read(REPO_ROOT / "reports" / "annotation_value.csv")
    pts = []
    for r in rows:
        try:
            ann, allan = float(r["dice_annotated"]), float(r["dice_all_analyses"])
        except (ValueError, KeyError):
            continue
        pts.append((r["project"], ann, allan, ann - allan))
    if not pts:
        print("  figure5: no rows; skipped")
        return
    pts.sort(key=lambda t: t[3])
    gains = [p[3] for p in pts]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # a: same studies, annotation on vs off
    ax = axes[0]
    for proj, ann, allan, _ in pts:
        ax.plot([0, 1], [allan, ann], color=COLORS.get(proj, "#7F7F7F"), lw=0.7, alpha=0.75)
        ax.scatter([0, 1], [allan, ann], s=9, color=COLORS.get(proj, "#7F7F7F"),
                   edgecolors="white", linewidths=0.3, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["All analyses\nfrom same studies", "Annotation-\nselected"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Dice vs expert map")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    panel_label(ax, "a", dx=-0.22)

    # b: the gain, by project, with the pooled median
    ax = axes[1]
    by_proj: dict[str, list[float]] = collections.defaultdict(list)
    for proj, _, _, g in pts:
        by_proj[proj].append(g)
    order = sorted(by_proj, key=lambda p: st.median(by_proj[p]))
    ax.axvline(0, color=INK, lw=0.6, zorder=2)
    for i, proj in enumerate(order):
        vals = by_proj[proj]
        ax.scatter(vals, [i] * len(vals), s=11, color=COLORS.get(proj, "#7F7F7F"),
                   edgecolors="white", linewidths=0.3, zorder=3, alpha=0.9)
        ax.plot([st.median(vals)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.0, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY.get(p, p) for p in order])
    ax.set_xlabel("Dice gain from analysis selection")
    ax.grid(axis="x", alpha=0.6); ax.set_axisbelow(True)
    ax.text(0.97, 0.04, f"median {st.median(gains):+.3f}\n{sum(1 for g in gains if g > 0)}"
            f"/{len(gains)} columns improve",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=MUTED,
            linespacing=1.5)
    panel_label(ax, "b", dx=-0.30)
    fig.subplots_adjust(wspace=0.55)
    save(fig, out_dir, "figure5_where_the_gain_comes_from")


# --------------------------------------------------------------------------- Figure 6

def figure6(out_dir: Path) -> None:
    """Measured cost per call, split by what is actually being paid for."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.0),
                            gridspec_kw={"width_ratios": [1, 1]})

    names = [c[0] for c in COST_PER_STAGE]
    x = range(len(names))

    # a: what a call costs, decomposed into uncached input / cached input / output
    ax = axes[0]
    fresh, cached, out = [], [], []
    for _, _, inp, cfrac, outp, _ in COST_PER_STAGE:
        fresh.append(inp * (1 - cfrac) * PRICE_IN)
        cached.append(inp * cfrac * PRICE_CACHED)
        out.append(outp * PRICE_OUT)
    bottom = [0.0] * len(names)
    for vals, colr, lab in ((fresh, "#0072B2", "Input"),
                            (cached, "#56B4E9", "Cached input"),
                            (out, "#D55E00", "Output")):
        ax.bar(x, vals, 0.6, bottom=bottom, color=colr, lw=0, label=lab)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("USD per call")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper left", handlelength=1.0, handletextpad=0.4)
    ax.text(0.02, 0.60, "output dominates\nthe cheap stage", transform=ax.transAxes,
            fontsize=5.5, color=MUTED, va="top", linespacing=1.5)
    panel_label(ax, "a", dx=-0.20)

    # b: output share -- the counter-intuitive part, and the one actionable lever
    ax = axes[1]
    share = [o / (f + c + o) * 100 for f, c, o in zip(fresh, cached, out)]
    ax.bar(x, share, 0.6, color="#D55E00", lw=0)
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylabel("Output share of cost (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    for i, v in enumerate(share):
        ax.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=5.5, color=INK)
    panel_label(ax, "b", dx=-0.20)

    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "figure6_measured_cost")


FIGURES = {2: figure2, 3: figure3, 4: figure4, 5: figure5, 6: figure6}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", type=int, nargs="*", choices=sorted(FIGURES),
                    help="figure numbers to build (default: all)")
    args = ap.parse_args()

    house_style()
    wanted = args.only or sorted(FIGURES)
    print(f"writing to {args.output_dir.relative_to(REPO_ROOT)}")
    for n in wanted:
        try:
            FIGURES[n](args.output_dir)
        except FileNotFoundError as exc:
            print(f"  figure{n}: missing input ({exc.filename}); skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

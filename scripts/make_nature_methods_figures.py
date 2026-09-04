#!/usr/bin/env python3
"""Publication figures for the Nature Methods submission, at Nature's format specs.

Distinct from make_cross_project_publication_plots.py, which targets a poster: large bold type,
tinted backgrounds, one idea per slide read from three metres away. Nature figures are the
opposite -- 89mm or 183mm wide, 5-7pt sans-serif, white, dense, read at arm's length with a
caption doing the explaining. Rather than parameterise the poster module into unreadability, this
is a separate module with its own house style.

Figures map onto NATURE_METHODS_SKELETON.md:

    Figure 2  gold retention + precision gain    Result 2  (§1)
    Figure 3  parsing + annotation               Result 3  (§5)
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
    """Cumulative gold recovery barely falls through screening; precision climbs.

    This is §1's claim, and getting it right required not using
    screening_metrics_top_v_stage_progression.csv. That file holds *conditional retention* -- the
    share of studies entering a stage that survive it -- so its denominator moves at every stage
    and its recall column rises across the funnel, which cumulative recall cannot do. Chaining the
    retentions does not fix it either, because retrieval loss is absent from that file entirely
    (executive_function's chained product is 0.645 against an actual 0.392).

    Panel a instead uses scripts/compute_gold_survival.py, which counts gold PMIDs against a fixed
    denominator at each stage, so the curve is monotone by construction and every drop belongs to
    a named stage. Retrieval is shown separately from screening because they fail for different
    reasons -- no obtainable full text is a supply problem, a rejection is a judgement -- and in
    several projects retrieval is the larger loss.
    """
    rows = read(REPO_ROOT / "reports" / "gold_survival_by_stage.csv")
    stages = ["search", "abstract", "retrieval", "fulltext"]
    labels = ["Search", "Abstract\nscreening", "Full-text\nretrieval", "Full-text\nscreening"]
    surv: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in rows:
        if r["cumulative_recall"]:
            surv[r["project"]][r["stage"]] = float(r["cumulative_recall"])

    prec: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in read(REPO_ROOT / "reports" / "cross_project_screening"
                  / "screening_metrics_top_v_stage_progression.csv"):
        if r["metric"] == "precision":
            try:
                prec[r["project_name"]][r["stage"]] = float(r["value"])
            except ValueError:
                continue

    projects = [p for p in PROJECT_ORDER if len(surv.get(p, {})) == len(stages)]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))

    # a: cumulative share of the gold standard still in play
    ax = axes[0]
    for p in projects:
        ys = [surv[p][s] * 100 for s in stages]
        ax.plot(range(len(stages)), ys, "-o", color=COLORS[p], markeredgewidth=0,
                alpha=0.9, label=DISPLAY[p])
        ax.annotate(f"{ys[-1]:.0f}", (len(stages) - 1, ys[-1]), textcoords="offset points",
                    xytext=(4, -1.5), fontsize=5.2, color=COLORS[p])
    losses = [(surv[p][stages[i - 1]] - surv[p][stages[i]]) * 100
              for p in projects for i in range(1, len(stages))]
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.25, len(stages) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Gold-standard studies retained (%)")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    med_abs = st.median([(surv[p]["search"] - surv[p]["abstract"]) * 100 for p in projects])
    ax.text(0.03, 0.06, f"abstract screening costs a\nmedian {med_abs:.1f} points",
            transform=ax.transAxes, fontsize=5.5, color=MUTED, linespacing=1.5)
    panel_label(ax, "a", dx=-0.20)

    # b: precision over the stages that change it
    ax = axes[1]
    pstages = ["search", "abstract", "fulltext"]
    for p in projects:
        ys = [prec.get(p, {}).get(s) for s in pstages]
        if any(v is None for v in ys):
            continue
        ax.plot(range(len(pstages)), ys, "-o", color=COLORS[p], markeredgewidth=0, alpha=0.9)
    ax.set_xticks(range(len(pstages)))
    ax.set_xticklabels(["Search", "Abstract\nscreening", "Full-text\nscreening"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Precision vs gold standard")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    panel_label(ax, "b", dx=-0.20)

    handles = [Line2D([], [], marker="o", ls="-", color=COLORS[p], markersize=2.8,
                      label=DISPLAY[p]) for p in projects]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.20),
               handletextpad=0.3, columnspacing=1.1, handlelength=1.2)
    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "figure2_gold_retention_and_precision")


# --------------------------------------------------------------------------- Figure 3

def figure3(out_dir: Path) -> None:
    """Analysis-level work, in pipeline order: recover the analyses, then select among them.

    Two operations that both live below the paper, combined into one display item to stay inside
    Nature's six. Panel a is recovery -- did the pipeline reconstruct the analyses the experts
    worked from at all -- against a table-only baseline that takes coordinate tables as parsed
    without an LLM reading them. Panel b is selection among what was recovered.

    dementia appears in a but not b. Its parsing is fine (100%); it is excluded from
    analysis-level annotation because its source meta-analysis pools several studies into one gold
    analysis, so per-paper extraction has no clean mapping to its units. That is an artefact of
    that benchmark rather than a result about the method, and the asymmetry is marked rather than
    hidden.
    """
    parse = read(REPO_ROOT / "reports" / "cross_project_analysis" / "parsing_metrics_by_project.csv")
    # mode_id "combined" is what §5 reports (pooled 0.539/0.810 across all nine, 0.540 with
    # dementia dropped); "accepted" is the stricter variant and differs by about a point. Both
    # rows are present for every project, so filtering on it is not optional -- without it each
    # project is plotted twice.
    ann = [r for r in read(REPO_ROOT / "reports" / "cross_project_analysis"
                           / "annotation_aggregates.csv")
           if r["level"] == "analysis" and r["variant"] == "exhausted_manual_assumption"
           and r["scope"] == "project" and r["mode_id"] == "combined"
           and r["project_name"] in DISPLAY and r["project_name"] != "dementia"]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.6))

    # a: LLM parsing against a table-only baseline
    rows = []
    for r in parse:
        if r["project_name"] not in DISPLAY:
            continue
        try:
            llm = float(r["manual_matched_pct"])
            tab = float(r["table_only_baseline_matched_pct"])
        except (ValueError, KeyError):
            continue
        rows.append((r["project_name"], tab * 100, llm * 100, int(r["manual_analyses_total"])))
    rows.sort(key=lambda x: x[2])

    ax = axes[0]
    for i, (proj, tab, llm, _) in enumerate(rows):
        ax.plot([tab, llm], [i, i], color=RULE, lw=1.5, zorder=1)
        ax.scatter([tab], [i], s=15, color="#7F7F7F", zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.scatter([llm], [i], s=15, color=COLORS[proj], zorder=3,
                   edgecolors="white", linewidths=0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DISPLAY[p] + (" *" if p == "dementia" else "") for p, _, _, _ in rows])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Expert analyses recovered (%)")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    pooled_llm = sum(r[2] * r[3] for r in rows) / sum(r[3] for r in rows)
    pooled_tab = sum(r[1] * r[3] for r in rows) / sum(r[3] for r in rows)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#7F7F7F", markersize=3.4,
                              label=f"Tables only ({pooled_tab:.0f}%)"),
                       Line2D([], [], marker="o", ls="", color=INK, markersize=3.4,
                              label=f"LLM parsing ({pooled_llm:.0f}%)")],
              loc="lower right", handletextpad=0.3)
    panel_label(ax, "a", dx=-0.40)

    # b: annotation as an operating point in precision-recall space, against the no-skill line
    #
    # An earlier version drew precision and recall as a dumbbell, which was wrong: a connector
    # implies a before/after, and these are two coordinates of one operating point, not a
    # progression. Worse, it made panel b look like panel a, where the connector genuinely does
    # run baseline -> outcome.
    #
    # The connector here runs prevalence -> achieved precision, which IS that relationship. A
    # random selector achieves precision equal to the prevalence of true positives at every
    # recall, so prevalence is the no-skill line in precision-recall space. It also makes the
    # absolute numbers interpretable, and it reorders them: social's 0.618 precision is the third
    # highest but the *lowest* lift at 1.8x, because its prevalence is the highest in the set,
    # while vbm_of_substance_use turns a similar 0.584 into 5.6x off a prevalence of 0.104.
    ax = axes[1]
    pts = []
    for r in ann:
        tp, fp, fn, tn = (int(float(r[k])) for k in ("tp", "fp", "fn", "tn"))
        total = tp + fp + fn + tn
        if not total:
            continue
        pts.append((r["project_name"], float(r["recall"]), float(r["precision"]),
                    (tp + fn) / total))
    pooled_prev = st.mean([p[3] for p in pts])

    ax.axhline(pooled_prev, color=MUTED, lw=0.7, ls=(0, (4, 2)), zorder=1)
    # Parked at the far left: every project sits at recall > 0.7, so the left half of the axis is
    # empty and the label cannot collide with data.
    ax.text(0.02, pooled_prev + 0.025, f"random selector (mean prevalence {pooled_prev:.2f})",
            fontsize=5.2, color=MUTED, ha="left")
    for proj, rec, prec, prev in pts:
        ax.plot([rec, rec], [prev, prec], color=COLORS[proj], lw=0.8, alpha=0.55, zorder=2)
        ax.scatter([rec], [prev], s=9, facecolors="white", edgecolors=COLORS[proj],
                   linewidths=0.7, zorder=3)
        ax.scatter([rec], [prec], s=17, color=COLORS[proj], zorder=4,
                   edgecolors="white", linewidths=0.4)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, markersize=3.4,
                              label="Achieved"),
                       Line2D([], [], marker="o", ls="", markerfacecolor="white",
                              markeredgecolor=INK, color="none", markersize=3.0,
                              label="Random (prevalence)")],
              loc="upper left", handletextpad=0.3, borderaxespad=0.4)
    ax.text(0.02, 0.055, f"vertical span = lift over random\nmean {st.mean([p[2] / p[3] for p in pts]):.1f}x",
            transform=ax.transAxes, ha="left", fontsize=5.2, color=MUTED, linespacing=1.5)
    panel_label(ax, "b", dx=-0.24)

    fig.text(0.5, -0.06, "* dementia excluded from b: its gold analyses pool several studies each",
             ha="center", fontsize=5.2, color=MUTED)
    fig.subplots_adjust(wspace=0.42)
    save(fig, out_dir, "figure3_recover_and_select_analyses")


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

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
    Figure 5  gain from analysis selection       Result 5  (§6)  <- the thesis
    Figure S1 measured cost per stage            Supplementary  (S1)  <- was Figure 6

Moved to the supplement 2026-09-09: Nature allows six display items, and the emotion-regulation
surface figure (scripts/make_er_surface_figure.py) is a stronger use of the slot than a cost
bar chart. Cost is a paragraph in the text plus S1.

Figure 1 is a schematic (pipeline, benchmark, and the paper-vs-analysis unit) and is not
generated here -- it wants a vector editor, and panel c is an argument rather than a plot.

Every panel is generated from a committed report CSV, so figures cannot drift from the numbers
in the text. The exception is Figure S1, whose per-stage costs currently live only in the S1 table
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_brain_map_figure import pretty_column  # noqa: E402
from benchmark_exclusions import filter_rows  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "reports" / "nature_methods_figures"
DECK_OUT = REPO_ROOT / "reports" / "deck_figures"

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
# The two VBM projects previously took #000000 and #7F7F7F. Black and grey are now reserved for
# the cross-project mean line, which has to be unmistakably not-a-project, so they moved to wine
# and brown -- both dark and warm, which also groups the two VBM projects visually, and both
# distinguishable from the vermillion and pink already in use at 3pt.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#785EF0", "#882255", "#8C564B"]
COLORS = dict(zip(PROJECT_ORDER, PALETTE))

# Reserved: no project may use these.
MEAN_COLOR = "#000000"
MEAN_KW = dict(color=MEAN_COLOR, lw=1.9, zorder=6, solid_capstyle="round")

# Direct labels for dense panels, where a legend would cost more space than it saves and colour
# alone leaves a reader unable to name a point.
SHORT = {
    "cue_reactivity": "Cue", "decision_making": "Decision", "dementia": "Dementia",
    "emotion_regulation_2022": "Emo. reg.", "executive_function": "Exec. fn.",
    "problem_solving": "Problem", "social": "Social", "vbm_of_ptsd": "PTSD",
    "vbm_of_substance_use": "Subst. use",
}

# Deck preset. Projected legibility needs type that is LARGER RELATIVE TO THE AXES -- scaling the
# figure and its fonts together just yields the same picture at more pixels. So the deck preset
# raises every font size and gives the figure extra HEIGHT to absorb it, keeping the column width
# (and so the slide fit) unchanged. Publication output is untouched: preset "print" is the default
# and both scales are 1.0.
FONT_SCALE = 1.0
HEIGHT_SCALE = 1.0


def fs(size: float) -> float:
    """A font size, scaled for the active preset."""
    return size * FONT_SCALE


def fh(inches: float) -> float:
    """A figure height, scaled for the active preset."""
    return inches * HEIGHT_SCALE


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
        "font.size": fs(6),
        "axes.labelsize": fs(6.5), "axes.titlesize": fs(7),
        "xtick.labelsize": fs(6), "ytick.labelsize": fs(6), "legend.fontsize": fs(6),
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
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fs(8), fontweight="bold",
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

    The fixed-pool arm used to appear here as a dotted overlay on both panels. It was removed
    2026-09-09: only three of nine projects have one, so it drew six extra part-width lines that
    invited the reader to compare curves covering different project sets, and the comparison it
    supports now has a figure of its own (Supplementary S3) where it gets the recall control that
    makes it interpretable. This figure is about what screening costs and buys on the pool we
    actually search.
    """
    rows = read(REPO_ROOT / "reports" / "gold_survival_by_stage.csv")
    stages = ["search", "abstract", "retrieval", "fulltext"]
    labels = ["Search", "Abstract\nscreening", "Full-text\nretrieval", "Full-text\nscreening"]
    surv: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in rows:
        if not r["cumulative_recall"]:
            continue
        # The fixed-pool family is no longer drawn here -- that comparison is Supplementary S3.
        if r.get("family") == "allstudies":
            continue
        surv[r["project"]][r["stage"]] = float(r["cumulative_recall"])

    prec: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for src, dest in (("screening_metrics_top_v_stage_progression.csv", prec),):
        try:
            src_rows = read(REPO_ROOT / "reports" / "cross_project_screening" / src)
        except FileNotFoundError:
            continue
        for r in src_rows:
            if r["metric"] == "precision":
                try:
                    dest[r["project_name"]][r["stage"]] = float(r["value"])
                except ValueError:
                    continue

    projects = [p for p in PROJECT_ORDER if len(surv.get(p, {})) == len(stages)]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fh(2.4)))

    # a: cumulative share of the gold standard still in play
    ax = axes[0]
    ends_a = []
    for p in projects:
        ys = [surv[p][s] * 100 for s in stages]
        ax.plot(range(len(stages)), ys, "-o", color=COLORS[p], markeredgewidth=0,
                alpha=0.9, label=DISPLAY[p])
        ends_a.append([ys[-1], f"{ys[-1]:.0f}", COLORS[p], False])
    # Cross-project mean, over the same projects the panel draws.
    means_a = [st.mean([surv[p][s] * 100 for p in projects]) for s in stages]
    ax.plot(range(len(stages)), means_a, marker="o", ms=3.6, mec="white", mew=0.5, **MEAN_KW)
    ends_a.append([means_a[-1], f"{means_a[-1]:.0f}", MEAN_COLOR, True])

    # End labels collide where projects finish close together -- 85/83/80/76 were printed on top
    # of one another, leaving one of them unreadable. Push them apart on the y axis, keeping
    # order, so each label still sits beside its own line.
    ends_a.sort(key=lambda e: e[0])
    gap_a = 3.1 * FONT_SCALE
    for i in range(1, len(ends_a)):
        if ends_a[i][0] - ends_a[i - 1][0] < gap_a:
            ends_a[i][0] = ends_a[i - 1][0] + gap_a
    for yy, txt, col, is_mean in ends_a:
        ax.annotate(txt, (len(stages) - 1, yy), textcoords="offset points",
                    xytext=(4, -1.5), fontsize=fs(5.6) if is_mean else 5.2,
                    fontweight="bold" if is_mean else "normal", color=col,
                    annotation_clip=False)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.25, len(stages) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Gold-standard studies retained (%)")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    med_abs = st.median([(surv[p]["search"] - surv[p]["abstract"]) * 100 for p in projects])
    ax.text(0.03, 0.06, f"abstract screening costs a\nmedian {med_abs:.1f} points",
            transform=ax.transAxes, fontsize=fs(5.5), color=MUTED, linespacing=1.5)
    panel_label(ax, "a", dx=-0.20 - 0.06 * (FONT_SCALE - 1.0))

    # b: precision over the stages that change it
    ax = axes[1]
    pstages = ["search", "abstract", "fulltext"]
    for p in projects:
        ys = [prec.get(p, {}).get(s) for s in pstages]
        if not any(v is None for v in ys):
            ax.plot(range(len(pstages)), ys, "-o", color=COLORS[p], markeredgewidth=0, alpha=0.9)
    have_prec = [p for p in projects
                 if all(prec.get(p, {}).get(s) is not None for s in pstages)]
    if have_prec:
        means_b = [st.mean([prec[p][s] for p in have_prec]) for s in pstages]
        ax.plot(range(len(pstages)), means_b, marker="o", ms=3.6, mec="white", mew=0.5,
                **MEAN_KW)
    ax.set_xticks(range(len(pstages)))
    ax.set_xticklabels(["Search", "Abstract\nscreening", "Full-text\nscreening"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Precision vs gold standard")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    panel_label(ax, "b", dx=-0.20 - 0.06 * (FONT_SCALE - 1.0))

    handles = [Line2D([], [], marker="o", ls="-", color=COLORS[p], markersize=2.8,
                      label=DISPLAY[p]) for p in projects]
    handles.append(Line2D([], [], ls="-", color=MEAN_COLOR, lw=1.9, marker="o", markersize=3.2,
                          label="mean across projects"))
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

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fh(2.6)))

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
    # Each project's baseline is drawn as a short horizontal RULE, not a dot. A random selector
    # picking each analysis with probability p gets recall = p and precision = prevalence,
    # independent of p -- the selected set has the pool's composition whatever its size. So the
    # no-skill baseline is a horizontal line spanning all recall, not a point, and a dot invites
    # the reasonable question of why "random" sits at recall 0.8. The rule says "this is a level";
    # the connector samples it at our recall, which is the like-for-like comparison.
    #
    # No pooled baseline line is drawn. Prevalence ranges 0.104 to 0.345 across these projects, a
    # 3.3x spread, so a single mean line would read as THE baseline and misplace most projects.
    # Labels sit BESIDE each point, not above it. Projects cluster tightly in recall (three pairs
    # within 0.02 of each other) but spread in precision, so horizontal placement separates them
    # where vertical placement overprints. The two rightmost flip to the left of their marker to
    # stay inside the axis, and PTSD at precision 1.0 would clip the top edge if labelled above.
    for proj, rec, prec, prev in sorted(pts, key=lambda z: z[1]):
        right = rec < 0.92
        ax.plot([rec, rec], [prev, prec], color=COLORS[proj], lw=0.8, alpha=0.55, zorder=2)
        ax.plot([rec - 0.012, rec + 0.012], [prev, prev], color=COLORS[proj], lw=1.1,
                alpha=0.85, zorder=3)
        ax.scatter([rec], [prec], s=17, color=COLORS[proj], zorder=4,
                   edgecolors="white", linewidths=0.4)
        ax.annotate(SHORT[proj], (rec, prec), textcoords="offset points",
                    xytext=(5 if right else -5, 0), va="center",
                    ha="left" if right else "right",
                    fontsize=fs(5.2), color=COLORS[proj])
    ax.set_xlim(0.62, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, markersize=3.4,
                              label="Achieved"),
                       Line2D([], [], marker="_", ls="", color=INK, markersize=5,
                              markeredgewidth=1.2, label="Random, at this recall")],
              loc="upper left", handletextpad=0.4, borderaxespad=0.4)
    ax.text(0.03, 0.055,
            "a random selector scores its own prevalence\n"
            f"at every recall; span = lift, mean {st.mean([p[2] / p[3] for p in pts]):.1f}x",
            transform=ax.transAxes, ha="left", fontsize=fs(5.2), color=MUTED, linespacing=1.5)
    panel_label(ax, "b", dx=-0.24)

    fig.text(0.5, -0.06, "* dementia excluded from b: its gold analyses pool several studies each",
             ha="center", fontsize=fs(5.2), color=MUTED)
    fig.subplots_adjust(wspace=0.42)
    save(fig, out_dir, "figure3_recover_and_select_analyses")


# --------------------------------------------------------------------------- Figure 4

# Marker areas for the Figure 4 size encoding. Analyses per column span 8 to 1699, a 200-fold
# range, so area proportional to N would leave the smallest columns invisible next to the largest.
# Log scaling keeps all 35 legible while preserving the ordering.
SIZE_MIN, SIZE_MAX = 5.0, 62.0
SIZE_TICKS = (10, 100, 1000)


def size_scale(counts: list[int]):
    """Map an analysis count to a marker area, and back for the legend."""
    import math
    lo, hi = math.log10(max(1, min(counts))), math.log10(max(counts))
    span = hi - lo or 1.0

    def area(n: int) -> float:
        f = (math.log10(max(1, n)) - lo) / span
        return SIZE_MIN + (SIZE_MAX - SIZE_MIN) * min(1.0, max(0.0, f))

    return area


def figure4(out_dir: Path) -> None:
    """Headline: the pipeline against the strongest baseline available per column.

    Circle area encodes how many analyses the pipeline pooled for that column. Without it a reader
    cannot tell whether the advantage rides on well-populated columns, and the answer is the
    opposite of the obvious guess: the pipeline beats its baseline in 30 of 35 columns, and in 23
    of those it does so while pooling FEWER analyses than the baseline (median baseline 2.5x
    larger, up to 11x for emotion regulation's `increase`). Selection beating volume is the
    paper's thesis, so the N belongs in the headline figure rather than in the text alone.
    """
    rows = filter_rows(read(REPO_ROOT / "reports" / "cross_project_best_baseline.csv"),
                       label="figure 4")
    stats = {r["comparison"]: r for r in
             read(REPO_ROOT / "reports" / "cross_project_best_baseline_stats.csv")}
    s = stats.get("best_available")
    metric = (rows[0].get("metric") if rows else None) or "dice"
    axis = {"dice": "Dice", "r2": "$R^2$", "pearson_r": "Pearson $r$"}.get(metric, metric)

    # Analyses per column, keyed the same way as the baseline table so the join is exact.
    counts_path = REPO_ROOT / "reports" / "analysis_counts.csv"
    counts: dict[tuple[str, str], int] = {}
    if counts_path.exists():
        for r in read(counts_path):
            if r["n_analyses_autonima"]:
                counts[(r["project"], r["manual_annotation"])] = int(r["n_analyses_autonima"])

    pts = [(r["project"], float(r["autonima"]), float(r["best_available"]),
            float(r["delta_vs_available"]),
            counts.get((r["project"], r["manual_annotation"]))) for r in rows]
    pts.sort(key=lambda t: t[3])
    known = [n for *_, n in pts if n]
    area = size_scale(known) if known else (lambda n: 13.0)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fh(2.5)),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # a: every column, autonima against its own best baseline
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=RULE, lw=0.6, zorder=1)
    # Large circles last, so a big column cannot hide a small one behind it.
    for proj, auto, base, _, n in sorted(pts, key=lambda t: -(t[4] or 0)):
        ax.scatter([base], [auto], s=area(n) if n else 13.0,
                   color=COLORS.get(proj, "#7F7F7F"),
                   edgecolors="white", linewidths=0.35, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"Best baseline {axis}"); ax.set_ylabel(f"Pipeline {axis}")
    ax.set_aspect("equal")
    ax.text(0.04, 0.93, "above the line =\npipeline better", fontsize=fs(5.5), color=MUTED,
            transform=ax.transAxes, va="top")
    ax.grid(alpha=0.6); ax.set_axisbelow(True)
    if known:
        # Size key in the lower right: that corner is below the diagonal and far from it, where
        # only a heavy loss would land, and none of the five losses is that severe.
        keys = [n for n in SIZE_TICKS if min(known) <= n <= max(known)] or [min(known), max(known)]
        handles = [Line2D([], [], marker="o", ls="", color=MUTED, alpha=0.55,
                          markeredgecolor="white", markeredgewidth=0.35,
                          markersize=area(n) ** 0.5, label=f"{n:,}") for n in keys]
        key = ax.legend(handles=handles, loc="lower right", title="Analyses pooled",
                        labelspacing=0.85, borderpad=0.5, handletextpad=0.7,
                        fontsize=fs(5.2), title_fontsize=fs(5.2), borderaxespad=0.4)
        key.get_title().set_color(MUTED)
        for t in key.get_texts():
            t.set_color(MUTED)
    panel_label(ax, "a", dx=-0.20)

    # b: per-column delta, sorted, with the pooled cluster-bootstrap CI behind it
    ax = axes[1]
    if s:
        lo, hi, mean = float(s["ci_low"]), float(s["ci_high"]), float(s["mean_delta"])
        ax.axhspan(lo, hi, color="#0072B2", alpha=0.10, lw=0, zorder=0)
        ax.axhline(mean, color="#0072B2", lw=0.8, zorder=2)
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    for i, (proj, _, _, dl, _n) in enumerate(pts):
        ax.bar(i, dl, width=0.78, color=COLORS.get(proj, "#7F7F7F"), lw=0, zorder=3)
    ax.set_xlim(-1, len(pts))
    ax.set_xticks([])
    ax.set_xlabel(f"{len(pts)} benchmark columns, ordered by advantage")
    ax.set_ylabel(f"$\\Delta$ {axis} vs best baseline")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    if s:
        ax.text(0.03, 0.96,
                f"$\\Delta$ = {mean:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]\n"
                f"sign test $P$ = {float(s['sign_test_p']):.1e}",
                transform=ax.transAxes, va="top", fontsize=fs(5.5), color=INK, linespacing=1.5)
    panel_label(ax, "b", dx=-0.18)

    handles = [Line2D([], [], marker="o", ls="", color=COLORS[p], markersize=3.2,
                      label=DISPLAY[p]) for p in PROJECT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.16),
               handletextpad=0.3, columnspacing=1.1)
    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "figure4_pipeline_vs_baseline")


# --------------------------------------------------------------------------- Figure 5

def figure5(out_dir: Path) -> None:
    """Which selection step earns the advantage: choosing papers, or choosing analyses?

    Promoted to the main text 2026-09-09, replacing the size-matched-null forest that was Figure 5
    (now Supplementary S5). It makes the same claim more directly -- two scatters that look
    different, no null model needed to read them -- and it decomposes the Figure 4 margin rather
    than opening a separate comparison.

    A third map is inserted between the baseline and the full pipeline: the canonical run's
    `all_analyses` column, which is every parsed analysis from the studies that survived
    screening, with no annotation. Papers chosen, analyses not.

    Both panels share axes, so the shape carries the result: panel a's points sit ON the diagonal
    and panel b's sit ABOVE it.

    A caveat the caption must carry, because the naive reading overclaims. `all_analyses` is one
    map per project scored against each of its contrasts -- the honest representation of having no
    analysis selection. But 31 of 32 baselines are TARGETED searches built per contrast, so the
    baseline is not an unselected corpus either: it selects papers by query where the pipeline
    selects them by LLM. So panel a does not show that paper selection is worthless. It shows that
    **two different ways of selecting papers come out even**, and that everything the pipeline
    gains comes from the step a search cannot perform at all.
    """
    path = REPO_ROOT / "reports" / "selection_decomposition.csv"
    if not path.exists():
        print("  figureS5: run scripts/decompose_selection_gain.py first; skipped")
        return
    rows = read(path)
    for r in rows:
        for k in ("r2_baseline", "r2_screening_only", "r2_pipeline",
                  "gain_paper_selection", "gain_analysis_selection"):
            r[k] = float(r[k])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fh(3.05)))
    panels = (
        ("a", "r2_baseline", "r2_screening_only", "Search baseline $R^2$",
         "Screening only $R^2$", "gain_paper_selection",
         "choosing papers", "on the line = no gain"),
        ("b", "r2_screening_only", "r2_pipeline", "Screening only $R^2$",
         "Full pipeline $R^2$", "gain_analysis_selection",
         "choosing analyses", "above the line = gain"),
    )
    for ax, (letter, xk, yk, xl, yl, gk, what, hint) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], color=RULE, lw=0.7, zorder=1)
        for r in rows:
            ax.scatter([r[xk]], [r[yk]], s=15, color=COLORS.get(r["project"], "#7F7F7F"),
                       edgecolors="white", linewidths=0.35, zorder=3)
        g = [r[gk] for r in rows]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.grid(alpha=0.6); ax.set_axisbelow(True)
        ax.set_title(f"gain from {what}", fontsize=fs(7.5), color=INK, pad=4)
        ax.text(0.035, 0.965, hint, transform=ax.transAxes, va="top", ha="left",
                fontsize=fs(5.4), color=MUTED)
        ax.text(0.97, 0.05,
                f"mean $\\Delta$ {st.mean(g):+.3f}\nmedian {st.median(g):+.3f}\n"
                f"{sum(1 for x in g if x > 0)}/{len(g)} improve",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=fs(5.6),
                color=INK, linespacing=1.5)
        panel_label(ax, letter, dx=-0.24)

    handles = [Line2D([], [], marker="o", ls="", color=COLORS[p], markersize=3.2,
                      label=DISPLAY[p]) for p in PROJECT_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.19),
               handletextpad=0.3, columnspacing=1.1)
    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "figure5_selection_decomposition")


def figureS5(out_dir: Path) -> None:
    """Every column against its own size-matched null.

    Promoted out of the main text 2026-09-09. This was Figure 5a; the decomposition that is now
    Figure 5 makes the same claim more directly and without a null model, so this becomes the
    supporting evidence rather than the headline. Its old panel b -- the same deltas rolled up per
    project -- is dropped outright: it duplicated Figure 4's cross-project view and carried
    nothing this panel does not already show column by column.

    What it still uniquely rules out: that any smaller subset of the same analyses would have done
    as well. Holding the study pool AND the subset size fixed, 500 random draws per column give the
    distribution a chance selection would produce, and the annotated map's position in it is the
    effect of selecting well.
    """
    path = REPO_ROOT / "reports" / "annotation_bootstrap_null.csv"
    if not path.exists():
        print("  figure5: run scripts/bootstrap_annotation_null.py first; skipped")
        return
    rows = filter_rows([r for r in read(path) if r.get("status") == "ok"],
                       label="figure 5")
    if not rows:
        print("  figure5: no usable rows; skipped")
        return

    for r in rows:
        for k in ("observed_r2", "null_mean", "null_p05", "null_p50", "null_p95", "p_value"):
            r[k] = float(r[k])
        r["delta"] = r["observed_r2"] - r["null_mean"]

    rank = {p: i for i, p in enumerate(PROJECT_ORDER)}
    rows.sort(key=lambda r: (rank.get(r["project"], 99), -r["delta"]))

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.55 * FONT_SCALE,
                                    fh(0.115 * len(rows) + 1.0)))

    # a: every column against its own size-matched null
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        c = COLORS.get(r["project"], "#7F7F7F")
        ax.plot([r["null_p05"], r["null_p95"]], [y, y], color=RULE, lw=1.6,
                solid_capstyle="butt", zorder=2)
        ax.plot([r["null_p50"]], [y], marker="|", color=MUTED, ms=3.2, mew=0.7, zorder=3)
        ax.scatter([r["observed_r2"]], [y], s=11, color=c, edgecolors="white",
                   linewidths=0.35, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{SHORT.get(r['project'], r['project'])} \u00b7 "
         f"{pretty_column(r['project'], r['manual_column'])}" for r in reversed(rows)],
        fontsize=fs(4.6))
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlim(0, 1)
    n_boot = min(int(r["n_boot"]) for r in rows)
    # In the axis label rather than floated inside the panel: at 35 rows the bottom rows reach the
    # lower right, which is the only corner an in-panel note fits, and it collided with them.
    ax.set_xlabel("$R^2$ against the expert map\n"
                  f"grey bar = 5-95% of {n_boot} size-matched draws; tick = median")
    ax.grid(axis="x", alpha=0.6); ax.set_axisbelow(True)

    gains = [r["delta"] for r in rows]
    beat = sum(1 for r in rows if r["p_value"] < 0.05)
    floor = sum(1 for r in rows if int(r.get("n_null_ge_observed", 1) or 1) == 0)
    # Above the axes rather than inside: the bottom-right corner is where the strongest columns
    # put their dots, and the summary was grazing them.
    ax.text(0.995, 1.012,
            f"median $\\Delta R^2$ {st.median(gains):+.3f}   \u00b7   {beat}/{len(rows)} "
            f"columns $P$ < 0.05   \u00b7   {floor} outside all {n_boot} draws",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=fs(5.6), color=INK,
            linespacing=1.5)
    save(fig, out_dir, "figureS5_size_matched_null")


# -------------------------------------------------------------------- Supplementary S2

def figureS2(out_dir: Path) -> None:
    """Two different effects live in the tier progression, and only the second one is overfitting.

    run_categories.yaml orders runs by how much benchmark information shaped their criteria. It is
    tempting to read the whole `verbatim -> best` rise as leakage, and an earlier version of this
    figure did. That is wrong, and the split says why:

      verbatim -> manual   the project author, having seen reports, fixed MAJOR OVERSIGHTS in the
                           criteria -- things that were simply mis-specified. n = 2, mean +0.221.
      manual   -> best     criteria rewritten against the error reports in full. This is the
                           overfitting estimate. n = 2, mean +0.031.

    So almost the entire rise is the cost of getting the criteria wrong, not the benefit of having
    seen the answers. That reframes the result from a caveat into a finding: the live risk in
    LLM screening is mis-prompting, and it is far larger than the risk of tuning. It is also a
    risk the authors ran into themselves, which is worth saying plainly.

    Both n = 2, so neither number is more than an indication. Only four projects were ever
    hand-revised, and two of those register the same run at `manual` and `best`, contributing no
    second-segment measurement.
    """
    path = REPO_ROOT / "reports" / "tier_progression.csv"
    if not path.exists():
        print("  figureS2: run scripts/compile_tier_progression.py first; skipped")
        return
    rows = read(path)
    TIER_ORDER = ["verbatim", "manual", "best"]
    LABEL = {"verbatim": "verbatim\n(from the paper,\nheld out)",
             "manual": "manual\n(oversights\nfixed by hand)",
             "best": "best\n(tuned on\nerror reports)"}

    by: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    runs: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for r in rows:
        by[r["project"]][r["tier"]].append(float(r["r2"]))
        runs[r["project"]][r["tier"]] = r["run"]

    means = {p: {t: st.mean(v) for t, v in tiers.items()} for p, tiers in by.items()}

    def seg(a: str, b: str) -> list[float]:
        """Paired deltas, excluding projects where both tiers resolve to the same run."""
        out = []
        for proj, m in means.items():
            if a in m and b in m and runs[proj].get(a) != runs[proj].get(b):
                out.append(m[b] - m[a])
        return out

    vm, mb = seg("verbatim", "manual"), seg("manual", "best")

    fig, ax = plt.subplots(figsize=(SINGLE_COL, fh(2.9)))
    ends, all_y = [], []
    for proj in PROJECT_ORDER:
        m = means.get(proj)
        if not m:
            continue
        xs = [i for i, t in enumerate(TIER_ORDER) if t in m]
        ys = [m[TIER_ORDER[i]] for i in xs]
        all_y.extend(ys)
        c = COLORS.get(proj, "#7F7F7F")
        n_runs = len({runs[proj].get(t) for t in TIER_ORDER if t in m})
        if len(xs) == 1:
            # Only one tier has maps at all. Missing data, not a measured flat progression, so it
            # must not carry the open-circle marker that means "same run at several tiers".
            ax.scatter(xs, ys, s=13, color=c, edgecolors="white", linewidths=0.35, zorder=4)
        elif n_runs == 1:
            # Registered at several tiers but resolving to one run: a line would imply a
            # progression was measured when none was.
            ax.scatter(xs[-1:], ys[-1:], s=15, facecolors="none", edgecolors=c, linewidths=0.9,
                       zorder=4)
        else:
            ax.plot(xs, ys, marker="o", ms=3.2, mec="white", mew=0.35, color=c, lw=0.9, zorder=3)
        ends.append([xs[-1], ys[-1], proj, c])

    # Mean over the projects present at ALL THREE tiers, so the line is one consistent subset
    # rather than three different ones.
    complete = [p for p in PROJECT_ORDER if p in means
                and all(t in means[p] for t in TIER_ORDER)]
    if complete:
        mline = [st.mean([means[p][t] for p in complete]) for t in TIER_ORDER]
        ax.plot(range(len(TIER_ORDER)), mline, marker="o", ms=4.0, mec="white", mew=0.5,
                **MEAN_KW)
        all_y.extend(mline)
        ends.append([len(TIER_ORDER) - 1, mline[-1], "_mean", MEAN_COLOR])

    ends.sort(key=lambda e: e[1])
    span = (max(all_y) - min(all_y)) or 1.0
    gap = span * 0.055
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < gap:
            ends[i][1] = ends[i - 1][1] + gap
    for x, y, proj, c in ends:
        txt = f"mean ({len(complete)})" if proj == "_mean" else SHORT.get(proj, proj)
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=(5, 0),
                    fontsize=fs(4.8), color=c, va="center", annotation_clip=False,
                    fontweight="bold" if proj == "_mean" else "normal")

    ax.set_xticks(range(len(TIER_ORDER)))
    ax.set_xticklabels([LABEL[t] for t in TIER_ORDER], linespacing=1.25, fontsize=fs(5.4))
    ax.set_xlim(-0.35, len(TIER_ORDER) - 0.22)
    lo, hi = min(all_y), max(all_y)
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.30 * (hi - lo))
    ax.set_ylabel("Mean $R^2$ against the expert map")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)

    # The two segments carry different meanings, so label them separately rather than quoting one
    # end-to-end number.
    y0, y1 = ax.get_ylim()
    band = y1 - 0.055 * (y1 - y0)
    for (x0, x1), lab, deltas in ((( -0.02, 0.98), "fixing\nmis-specification", vm),
                                  ((1.02, 1.98), "overfitting", mb)):
        ax.annotate("", xy=(x1, band), xytext=(x0, band),
                    arrowprops=dict(arrowstyle="<->", lw=0.6, color=MUTED, shrinkA=0, shrinkB=0))
        txt = f"{lab}\nmean {st.mean(deltas):+.3f} (n={len(deltas)})" if deltas else lab
        ax.text((x0 + x1) / 2, band - 0.012 * (y1 - y0), txt, ha="center", va="top",
                fontsize=fs(4.9), color=MUTED, linespacing=1.35)

    ax.text(0.98, 0.03, "open circle = one run registered at several tiers",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=fs(4.6), color=MUTED)
    save(fig, out_dir, "figureS2_tier_progression")


# -------------------------------------------------------------------- Supplementary S4

def figureS4(out_dir: Path) -> None:
    """A term is not an analysis: text-to-map prediction against curated synthesis.

    Every baseline in Result 4 is a *search* baseline, which tests the pipeline against the
    Neurosynth-style workflow but not against the current generation of automated map generators.
    NeuroQuery predicts a map from free text with no studyset, no screening and no coordinate
    extraction, so it is the strongest available "do nothing" arm and the one a reviewer will name.

    The overall gap is large -- mean r-squared 0.075 against 0.574, behind on 32 of 32 columns --
    but reporting only that would miss the finding, and would invite the fair suspicion that the
    comparison is rigged. The structure of *where* it fails is the result:

      canonical cognitive terms   executive function 0.308, working memory 0.278,
                                  problem solving 0.256, mental arithmetic 0.240
      condition contrasts         all three emotion-regulation contrasts 0.002
      clinical group comparisons  alcohol 0.002, dementia functional 0.002, PTSD 0.022

    NeuroQuery encodes term-level association. Where a benchmark column essentially *is* a term it
    does respectably; where the column is a contrast between conditions or a between-group clinical
    comparison it has no representation for the thing being asked and scores near zero. That is
    this paper's analysis-unit argument arriving from an independent direction.

    Two caveats belong in the caption. This is not like-for-like: NeuroQuery answers a different
    and much cheaper question, so the comparison shows that term-level prediction cannot substitute
    for contrast-level synthesis, not that NeuroQuery is poor at its own task. And r-squared is the
    only applicable metric -- NeuroQuery produces no FDR-corrected map, so dice would require a
    threshold with no error control, which the metric/map rule forbids.
    """
    path = REPO_ROOT / "reports" / "text_to_map_baselines.csv"
    if not path.exists():
        print("  figureS4: run scripts/text_to_map_baselines.py first; skipped")
        return
    rows = read(path)
    for r in rows:
        for k in ("neuroquery_r2", "autonima_r2", "best_baseline_r2"):
            r[k] = float(r[k])
        r["neurovlm_r2"] = float(r["neurovlm_r2"]) if r.get("neurovlm_r2") else None
    has_nvlm = all(r["neurovlm_r2"] is not None for r in rows)

    # Prompt-sensitivity ranges from query_sensitivity.py. The interval is the min-to-max over
    # five UNIFORM query strategies, i.e. what one global prompt decision is worth -- not the
    # per-column oracle, which is an unreachable upper bound and would overstate the range.
    sens_path = REPO_ROOT / "reports" / "query_sensitivity.csv"
    prompt_range: dict[tuple[str, str], tuple[float, float]] = {}
    overall_range: dict[str, tuple[float, float]] = {}
    if sens_path.exists():
        sens = read(sens_path)
        for key, arm in (("neuroquery_r2", "neuroquery_r2"), ("neurovlm_r2", "neurovlm_r2")):
            per_strategy_project: dict[str, dict[str, list[float]]] = collections.defaultdict(
                lambda: collections.defaultdict(list))
            per_strategy_all: dict[str, list[float]] = collections.defaultdict(list)
            for x in sens:
                if not x.get(key):
                    continue
                v = float(x[key])
                per_strategy_project[x["project"]][x["strategy"]].append(v)
                per_strategy_all[x["strategy"]].append(v)
            for proj, by_strat in per_strategy_project.items():
                means = [st.mean(v) for v in by_strat.values() if v]
                if len(means) > 1:
                    prompt_range[(proj, arm)] = (min(means), max(means))
            means = [st.mean(v) for v in per_strategy_all.values() if v]
            if len(means) > 1:
                overall_range[arm] = (min(means), max(means))

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fh(2.7)),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # a: per-project means for the three arms, ordered by how well NeuroQuery does
    ax = axes[0]
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[r["project"]].append(r)
    rank_key = "neurovlm_r2" if has_nvlm else "neuroquery_r2"
    order = sorted(by, key=lambda p: st.mean([r[rank_key] for r in by[p]]))
    # Colour, not shades of grey: four grey bars at this size were not separable. Hue encodes the
    # KIND of arm and lightness the ordering within a kind -- the two text->map models share a
    # blue family (light = weaker), the search baseline is orange, the pipeline black. Okabe-Ito,
    # so it survives colour-blind viewing; S4 draws no per-project colour, so there is no clash
    # with the project palette used in the other figures.
    arms = [("neuroquery_r2", "NeuroQuery (text \u2192 map)", "#56B4E9")]
    if has_nvlm:
        arms.append(("neurovlm_r2", "NeuroVLM (text \u2192 map)", "#0072B2"))
    arms += [("best_baseline_r2", "best search baseline", "#E69F00"),
             ("autonima_r2", "full pipeline", MEAN_COLOR)]
    h = 0.86 / len(arms)
    for j, (key, label, colour) in enumerate(arms):
        off = (j - (len(arms) - 1) / 2) * h
        ys = [i + off for i in range(len(order))]
        vals = [st.mean([r[key] for r in by[p]]) for p in order]
        ax.barh(ys, vals, height=h * 0.9, color=colour, lw=0, label=label, zorder=3)
        # Prompt-sensitivity whisker, text->map arms only: the search baseline and the pipeline
        # take no query, so they have no such range.
        if key in ("neuroquery_r2", "neurovlm_r2"):
            lo = [prompt_range.get((p, key), (v, v))[0] for p, v in zip(order, vals)]
            hi = [prompt_range.get((p, key), (v, v))[1] for p, v in zip(order, vals)]
            ax.hlines(ys, lo, hi, color=INK, lw=0.7, zorder=5)
            ax.vlines(lo, [y - h * 0.28 for y in ys], [y + h * 0.28 for y in ys],
                      color=INK, lw=0.7, zorder=5)
            ax.vlines(hi, [y - h * 0.28 for y in ys], [y + h * 0.28 for y in ys],
                      color=INK, lw=0.7, zorder=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY.get(p, p) for p in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Mean $R^2$ against the expert map")
    ax.set_xlim(0, 1.12)
    ax.grid(axis="x", alpha=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=fs(5.0), handlelength=1.1, handletextpad=0.4,
              borderaxespad=0.5)
    # The two projects where NeuroQuery is not near zero are the two whose columns are closest to
    # being plain terms, which is the point of ordering the panel this way.
    # Top right: the two highest-NeuroQuery projects top out around 0.67, so this corner is free.
    # Mid-height on the right: no bar in the middle rows passes 0.63 and the x limit is 1.12.
    panel_label(ax, "a", dx=-0.42)

    # b: the same three arms under two measures, because r-squared is not a fair one here
    ax = axes[1]
    # All three arms must have a top-k value, or the three bars would be means over different
    # column sets.
    have = [r for r in rows
            if all(r.get(f"topk_dice_{n}") not in ("", None)
                   for n in (["neuroquery"] + (["neurovlm"] if has_nvlm else [])
                             + ["best_baseline", "pipeline"]))]
    r2_keys = [a[0] for a in arms]
    tk_names = {"neuroquery_r2": "neuroquery", "neurovlm_r2": "neurovlm",
                "best_baseline_r2": "best_baseline", "autonima_r2": "pipeline"}
    groups = [("$R^2$\n(all voxels)", [st.mean([r[k] for r in rows]) for k in r2_keys])]
    if have:
        groups.append(("top-$k$ dice\n(ranked, form-free)",
                       [st.mean([float(r[f"topk_dice_{tk_names[k]}"]) for r in have])
                        for k in r2_keys]))
    w = 0.82 / len(arms)
    for j, (key, label, colour) in enumerate(arms):
        off = (j - (len(arms) - 1) / 2) * w
        xs2 = [i + off for i in range(len(groups))]
        ax.bar(xs2, [g[1][j] for g in groups], width=w * 0.88, color=colour, lw=0,
               zorder=3)
        # Only the r-squared group (index 0) has a prompt range: per-strategy top-k dice was not
        # computed for the variants, so a whisker there would be fabricated.
        if key in overall_range:
            lo, hi = overall_range[key]
            ax.vlines(xs2[0], lo, hi, color=INK, lw=0.7, zorder=5)
            ax.hlines([lo, hi], xs2[0] - w * 0.22, xs2[0] + w * 0.22, color=INK, lw=0.7,
                      zorder=5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], linespacing=1.3)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("Mean agreement with the expert map")
    ax.set_ylim(0, max(max(g[1]) for g in groups) * 1.75)
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    if len(groups) == 2:
        # Kept deliberately short: the form argument and the baseline shares are caption material
        # and live in the docstring and NATURE_METHODS_SKELETON.md. Only what cannot be read off
        # the bars goes in the panel -- what the whisker means, and the resulting range.
        i_base = len(arms) - 2
        lines = []
        if has_nvlm and "neurovlm_r2" in overall_range:
            nv_lo, nv_hi = overall_range["neurovlm_r2"]
            nq_lo, nq_hi = overall_range.get("neuroquery_r2", (0.0, 0.0))
            lines += [
                "whiskers = min\u2013max over five uniform query strategies",
                f"NeuroVLM {nv_lo:.2f}\u2013{nv_hi:.2f} (bar = {groups[0][1][1]:.2f} "
                f"pre-registered)",
                f"NeuroQuery {nq_lo:.2f}\u2013{nq_hi:.2f}; even NeuroVLM's best prompt",
                "stays below the search baseline",
            ]
        lines.append(f"NeuroVLM leads NeuroQuery "
                     f"{groups[0][1][1] / groups[0][1][0]:.1f}$\\times$ on $R^2$, "
                     f"{groups[1][1][1] / groups[1][1][0]:.1f}$\\times$ ranked")
        ax.text(0.5, 0.985, "\n".join(lines), transform=ax.transAxes, ha="center", va="top",
                fontsize=fs(4.6), color=INK, linespacing=1.5)
    panel_label(ax, "b", dx=-0.26)

    fig.subplots_adjust(wspace=0.42)
    save(fig, out_dir, "figureS4_text_to_map_baselines")


# -------------------------------------------------------------------- Supplementary S3

def _screening_metric(filename: str, metric: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    path = REPO_ROOT / "reports" / "cross_project_screening" / filename
    if not path.exists():
        return out
    for r in read(path):
        if r.get("metric") != metric:
            continue
        try:
            out[r["project_name"]][r["stage"]] = float(r["value"])
        except (ValueError, KeyError):
            continue
    return out


def figureS3(out_dir: Path) -> None:
    """How much of the low screening precision is a pool mismatch rather than a screening failure?

    Precision against the expert inclusion list is the weakest headline number in the paper, and
    there are two candidate explanations. Either the screener admits studies the experts rejected,
    or the *pool* it screens is not the pool the experts drew from -- our PubMed query returns a
    different corpus than the one behind the published review, so studies that could never have
    been in the expert list count as false positives no matter how well screening works.

    Three projects have a fixed-pool arm: the same criteria, the same screening, over a pool
    assembled without search-driven narrowing. Holding screening constant and swapping only the
    pool separates the two explanations, and the answer is that a large share is the pool.

    Panel b carries the control that makes the claim, and without it the panel a gap would be
    uninterpretable: the fixed pool buys precision at essentially no cost to RECALL from abstract
    screening onward (mean +0.000 and +0.007). A change that raised precision by discarding
    borderline true positives would show up there as a recall loss, and it does not.

    The search-stage recall delta is shown but should not be read as a screening result: at that
    stage the two arms are different corpora by construction, which is why emotion regulation
    reads -0.261 there and ~0 at every later stage.
    """
    STAGES = ["search", "abstract", "fulltext"]
    LABELS = ["Search", "Abstract\nscreening", "Full-text\nscreening"]
    SEARCH_FILE = "screening_metrics_top_v_stage_progression.csv"
    FIXED_FILE = "screening_metrics_top_v_allstudies_stage_progression.csv"

    prec_s, prec_f = (_screening_metric(SEARCH_FILE, "precision"),
                      _screening_metric(FIXED_FILE, "precision"))
    rec_s, rec_f = (_screening_metric(SEARCH_FILE, "recall"),
                    _screening_metric(FIXED_FILE, "recall"))
    projects = [p for p in PROJECT_ORDER
                if all(p in d for d in (prec_s, prec_f, rec_s, rec_f))
                and all(st_ in prec_s[p] and st_ in prec_f[p] for st_ in STAGES)]
    if not projects:
        print("  figureS3: no project has both pools; skipped")
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fh(2.5)),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    xs = range(len(STAGES))

    # a: precision under each pool, same screening
    ax = axes[0]
    for p in projects:
        c = COLORS.get(p, "#7F7F7F")
        ax.plot(xs, [prec_s[p][s] for s in STAGES], "-o", color=c, ms=3.0,
                markeredgewidth=0, alpha=0.9)
        ax.plot(xs, [prec_f[p][s] for s in STAGES], ":o", color=c, ms=3.0,
                markeredgewidth=0, alpha=0.9, lw=1.1)
    m_s = [st.mean([prec_s[p][s] for p in projects]) for s in STAGES]
    m_f = [st.mean([prec_f[p][s] for p in projects]) for s in STAGES]
    ax.fill_between(list(xs), m_s, m_f, color=MEAN_COLOR, alpha=0.10, lw=0, zorder=1)
    ax.plot(xs, m_s, marker="o", ms=4.0, mec="white", mew=0.5, **MEAN_KW)
    ax.plot(xs, m_f, marker="o", ms=4.0, mec="white", mew=0.5, ls=":",
            **{k: v for k, v in MEAN_KW.items() if k != "solid_capstyle"})
    ax.text(0.02, 0.98,
            f"shaded = pool contribution\nat full-text screening: {m_f[-1] - m_s[-1]:+.3f} "
            f"precision\n({m_s[-1]:.3f} search pool \u2192 {m_f[-1]:.3f} fixed)",
            transform=ax.transAxes, ha="left", va="top", fontsize=fs(5.2), color=INK,
            linespacing=1.5)
    ax.set_xticks(list(xs)); ax.set_xticklabels(LABELS)
    ax.set_xlim(-0.2, len(STAGES) - 0.35)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Precision vs expert inclusion list")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    panel_label(ax, "a", dx=-0.19)

    # b: the paired deltas, precision against recall
    ax = axes[1]
    width = 0.17
    for j, (label, ds, df, colour) in enumerate((
            ("precision", prec_s, prec_f, "#0072B2"),
            ("recall", rec_s, rec_f, "#D55E00"))):
        for i, s in enumerate(STAGES):
            vals = [df[p][s] - ds[p][s] for p in projects
                    if s in ds[p] and s in df[p]]
            if not vals:
                continue
            base = i + (j - 0.5) * 2 * width
            ax.bar(base, st.mean(vals), width=width * 1.7, color=colour, alpha=0.30, lw=0,
                   zorder=2, label=f"mean \u0394 {label}" if i == 0 else None)
            ax.scatter([base] * len(vals), vals, s=9, color=colour, edgecolors="white",
                       linewidths=0.3, zorder=4)
    ax.axhline(0, color=INK, lw=0.6, zorder=3)
    ax.set_xticks(list(xs)); ax.set_xticklabels(LABELS)
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    # Headroom above the tallest precision point so the note clears it.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.30 * (hi - lo))
    ax.set_ylabel("Fixed pool \u2212 search pool")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=fs(5.2), handlelength=1.0, handletextpad=0.4,
              borderaxespad=0.4)
    ax.text(0.02, 0.98,
            "precision rises at every stage;\nrecall unchanged from abstract on.\n"
            "The search-stage recall dip is one\nproject and is a corpus difference,\n"
            "not a screening result.",
            transform=ax.transAxes, ha="left", va="top", fontsize=fs(4.9), color=MUTED,
            linespacing=1.45)
    panel_label(ax, "b", dx=-0.19)

    handles = [Line2D([], [], marker="o", ls="-", color=COLORS[p], markersize=2.8,
                      label=DISPLAY[p]) for p in projects]
    handles += [Line2D([], [], ls="-", color=MEAN_COLOR, lw=1.9, label="mean, search pool"),
                Line2D([], [], ls=":", color=MEAN_COLOR, lw=1.9, label="mean, fixed pool")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.17),
               handletextpad=0.3, columnspacing=1.1, handlelength=1.3)
    fig.subplots_adjust(wspace=0.30)
    save(fig, out_dir, "figureS3_pool_mismatch")


# -------------------------------------------------------------------- Supplementary S1

def figureS1(out_dir: Path) -> None:
    """Measured cost per call, split by what is actually being paid for."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fh(2.0)),
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
            fontsize=fs(5.5), color=MUTED, va="top", linespacing=1.5)
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
        ax.text(i, v + 2.5, f"{v:.0f}%", ha="center", fontsize=fs(5.5), color=INK)
    panel_label(ax, "b", dx=-0.20)

    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "figureS1_measured_cost")


# Keys are strings because the cost figure moved to the supplement: it is "S1", not 6. Nature
# allows six display items and the brain-surface figure is a stronger use of the slot.
FIGURES = {"2": figure2, "3": figure3, "4": figure4, "5": figure5,
           "S1": figureS1, "S2": figureS2, "S3": figureS3, "S4": figureS4, "S5": figureS5}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=("print", "deck"), default="print",
                    help="print = Nature specs (default). deck = larger type for projection, "
                         "written to reports/deck_figures so publication output is never "
                         "overwritten.")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), metavar="FIG",
                    help=f"figures to build, from {' '.join(sorted(FIGURES))} (default: all)")
    args = ap.parse_args()

    global FONT_SCALE, HEIGHT_SCALE
    if args.preset == "deck":
        FONT_SCALE, HEIGHT_SCALE = 1.55, 1.28
    out_dir = args.output_dir or (DEFAULT_OUT if args.preset == "print" else DECK_OUT)
    args.output_dir = out_dir

    house_style()
    wanted = args.only or sorted(FIGURES)
    print(f"writing to {args.output_dir.relative_to(REPO_ROOT)}")
    for n in wanted:
        try:
            FIGURES[n](args.output_dir)
        except FileNotFoundError as exc:
            print(f"  figure {n}: missing input ({exc.filename}); skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

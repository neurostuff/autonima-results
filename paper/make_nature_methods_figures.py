#!/usr/bin/env python3
"""Publication figures for the Nature Methods submission, at Nature's format specs.

Distinct from make_cross_project_publication_plots.py, which targets a poster: large bold type,
tinted backgrounds, one idea per slide read from three metres away. Nature figures are the
opposite -- 89mm or 183mm wide, 5-7pt sans-serif, white, dense, read at arm's length with a
caption doing the explaining. Rather than parameterise the poster module into unreadability, this
is a separate module with its own house style.

Figures map onto NATURE_METHODS_SKELETON.md:

    Figure 2   precision gain + attainable recall   Result 2  (§1)
    Figure 3   parsing + annotation in ROC space      Result 3  (§5)
    Figure 4   pipeline vs best baseline              Result 4  (§7)  <- headline
    Figure 5   gain from analysis selection           Result 5  (§6)  <- the thesis
    Figure S1  criteria tier progression              --only S1
    Figure S2  pool mismatch                          --only S2
    Figure S3  size-matched null                      --only S3
    Figure S4  brain maps, all columns                --only S4  (~2 min, shells out)
    Figure S5  measured cost per stage                --only S5

Supplementary figures were renumbered 2026-09-16. Two that previously carried S-numbers are
still built and still correct but are no longer cited, so they keep descriptive keys instead:
`--only retention` (raw-denominator gold retention, was S6) and `--only annotationpr`
(annotation in precision-recall space, was S7).

Moved to the supplement 2026-09-09: Nature allows six display items, and the emotion-regulation
surface figure (paper/make_er_surface_figure.py) is a stronger use of the slot than a cost
bar chart. Cost is a paragraph in the text plus S5.

Figure 1 is a schematic (pipeline, benchmark, and the paper-vs-analysis unit) and is not
generated here -- it wants a vector editor, and panel c is an argument rather than a plot.

Every panel is generated from a committed report CSV, so figures cannot drift from the numbers
in the text. The exception is Figure S5, whose per-stage costs currently live only in the S1 table
in PAPER_OUTLINE.md; those are transcribed below as COST_PER_STAGE and should move into a
generated CSV before submission.

Usage:
    python paper/make_nature_methods_figures.py                  # all figures
    python paper/make_nature_methods_figures.py --only 4 5       # just those
"""

from __future__ import annotations

import argparse
import collections
import subprocess
import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

import sys
# Both explicitly: scripts/ for the shared libraries, and this directory for the sibling
# figure module. Relying on Python adding the script's own directory is what breaks the
# moment anything imports this file rather than running it -- see scripts/map_mask.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
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
# Dementia is excluded from the MAP-LEVEL figures (4, 5, S5). Its source meta-analysis pools
# several studies into one gold analysis, so the number of analyses entering its maps is not
# comparable with the other projects and a per-column margin against them is not interpretable.
# This is a project-level call about map-level comparability, distinct from benchmark_exclusions,
# which drops individual columns with no published result. Dementia is deliberately KEPT in the
# analysis-level figures (2, 3), where its matched analyses score normally -- see figure3.
# compile_best_baselines.py --exclude-project dementia applies the same rule upstream, so the
# bootstrap CI and sign test are computed on the same 28 columns rather than filtered after.
MAP_LEVEL_EXCLUDED_PROJECTS = ("dementia",)

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
    ("Analysis\nselection", 1091, 52117, 0.070, 4424, 0.0211),
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


def deoverlap_labels(fig, ax, texts, blockers=(), markers=(), marker_r_pt: float = 3.2,
                     pad_pt: float = 0.9, max_shift_pt: float = 40.0,
                     headroom_pt: float = 16.0) -> None:
    """Place point labels so they clear each other, the plotted markers, and the axes edge.

    Two passes, deliberately separated:

    1. *Side choice.* Each label is tried on both sides of its marker and keeps whichever
       overlaps fewer of the ``markers`` (data-space (x, y) points) and stays inside the axes.
       Ties keep the caller's alternation. Markers only ever influence this choice -- letting
       them push labels vertically cascades, since the interesting region of an ROC panel is
       exactly where the markers are, and everything ends up jammed against the top.
    2. *Vertical de-overlap.* Labels are then nudged up (or under, when there is no headroom)
       until no two rendered boxes touch, considering only other labels and ``blockers``.

    Both passes measure real text extents from the renderer rather than assuming a gap in data
    units. A data-unit gap has to be retuned every time the font scale changes, and when it is
    not, labels silently overprint at one preset while looking clean at the other.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    frame, pad = ax.get_window_extent(rend), pad_pt * px
    # Labels may sit a little ABOVE the axes: annotation_clip is off and save() writes with
    # bbox_inches="tight", so anything just outside the box is still in the image. Treating the
    # axes top as a hard ceiling is what jammed the deck preset -- points at TPR 0.96 against a
    # 1.01 limit had nowhere to go, so crowded labels ducked back down into each other.
    ceiling = frame.y1 + headroom_pt * px
    r = marker_r_pt * px
    dots = []
    for mx, my in markers:
        cx, cy = ax.transData.transform((mx, my))
        dots.append(Bbox.from_extents(cx - r, cy - r, cx + r, cy + r))

    def _hits(box):
        return sum(box.x0 < q.x1 + pad and q.x0 < box.x1 + pad
                   and box.y0 < q.y1 + pad and q.y0 < box.y1 + pad for q in dots)

    for t in texts:
        box = t.get_window_extent(rend)
        dx, dy = t.xyann
        ax_x = ax.transData.transform(t.xy)[0]
        w, off = box.x1 - box.x0, abs(dx) * px
        # Own marker sits under the anchor, so it is discounted on both sides and cannot
        # bias the choice.
        here = Bbox.from_extents(box.x0, box.y0, box.x1, box.y1)
        x0 = ax_x + off if dx < 0 else ax_x - off - w
        there = Bbox.from_extents(x0, box.y0, x0 + w, box.y1)
        score_h = _hits(here) + 99 * (here.x0 < frame.x0 or here.x1 > frame.x1)
        score_t = _hits(there) + 99 * (there.x0 < frame.x0 or there.x1 > frame.x1)
        if score_t < score_h:
            t.set_ha("left" if dx < 0 else "right")
            t.xyann = (-dx, dy)

    placed = [b.get_window_extent(rend) for b in blockers]
    for t in texts:
        dx, dy = t.xyann
        for _ in range(60):
            box = t.get_window_extent(rend)
            hit = next((q for q in placed
                        if box.x0 < q.x1 + pad and q.x0 < box.x1 + pad
                        and box.y0 < q.y1 + pad and q.y0 < box.y1 + pad), None)
            if hit is None:
                break
            step = (hit.y1 + pad) - box.y0                  # clear the blocker upward
            if box.y1 + step > ceiling:
                step = (hit.y0 - pad) - box.y1              # no headroom: duck under it
            if abs(dy + step / px) > max_shift_pt:
                break                                       # keep the label on its marker
            dy += step / px
            t.xyann = (dx, dy)
        placed.append(t.get_window_extent(rend))


# Deck type is 1.55x larger against the same axes width, so the two-line stage ticks collide
# ("screeningscreening"). The stage is named in the slide title anyway, so the qualifier is
# dropped rather than shrinking type that has to survive projection. Mapped explicitly, not by
# truncating at the newline: figure 2 carries both a retrieval and a screening full-text stage,
# and truncating would give it two ticks reading "Full-text".
# The fourth stage is labelled "Analysis selection" for readers, not "Annotation". The paper
# never uses "annotation" as a stage name -- it says analysis-level selection, and Fig. 3b reports
# assignment precision and recall. The INTERNAL key stays `annotation`: it is the pipeline's own
# stage name and it keys annotation_aggregates.csv, nimads_annotation.json and every run
# artifact, so renaming it would be a large refactor with no reader-facing benefit. Display name
# and data key differ on purpose.
DECK_TICKS = {
    "Abstract\nscreening": "Abstract",
    "Analysis\nselection": "Analysis",
    "Full-text\nscreening": "Full-text",
    "Full-text\nretrieval": "Retrieval",
}


def stage_ticks(labels):
    """Stage tick labels, shortened when the active preset needs the room."""
    if FONT_SCALE <= 1.2:
        return list(labels)
    return [DECK_TICKS.get(x, x) for x in labels]


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.06) -> None:
    t = ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fs(8), fontweight="bold",
                va="top", ha="left", color=INK)
    t.set_gid("panel-label")           # save() shifts it clear of the y-label if it lands on it


def _clear_panel_labels(fig) -> None:
    """Shift any panel label that has landed on its own y-label further left.

    dx is an axes-fraction constant, so a longer y-label or a larger font moves the y-label
    under the letter without moving the letter. That is how "a" ended up printed across the
    "(%)" of figure 2's y-label at deck scale, at 1.55x type against the same axes width.
    Measuring the two and pushing the letter out fixes every figure at once, instead of each
    dx being retuned per preset.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax in fig.axes:
        letter = next((t for t in ax.texts if t.get_gid() == "panel-label"), None)
        if letter is None or not ax.get_ylabel():
            continue
        span = ax.get_window_extent(rend).width or 1.0
        for _ in range(40):
            a, b = letter.get_window_extent(rend), ax.yaxis.label.get_window_extent(rend)
            if not (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1):
                break
            dx, dy = letter.get_position()
            letter.set_position((dx - (a.x1 - b.x0 + 2.0) / span, dy))


def save(fig, out_dir: Path, name: str) -> None:
    _clear_panel_labels(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 400})):
        fig.savefig(out_dir / f"{name}{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------- Raw-denominator gold retention (unnumbered)

def figure_gold_retention_raw_fig(out_dir: Path) -> None:
    """Cumulative gold recovery on the RAW denominator, with retrieval as its own stage.

    Was Figure 2 until 2026-09-10, when the attainable-denominator version took that slot. It is
    kept because it is the only figure that still shows the retrieval stage and the raw
    denominator, which is what the loss-by-stage table in NATURE_METHODS_SKELETON.md §1 is
    computed from: abstract screening costs a median 1.4 points, retrieval 4.5, full-text
    screening 2.5. Main-text Figure 2 folds retrieval into the denominator instead, so that
    breakdown is not recoverable from it.

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
    supports now has a figure of its own (Supplementary S2) where it gets the recall control that
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
        # The fixed-pool family is no longer drawn here -- that comparison is Supplementary S2.
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
    ax.set_xticklabels(stage_ticks(labels))
    ax.set_xlim(-0.25, len(stages) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Gold-standard studies retained (%)")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    # in-panel result text removed 2026-09-18; the caption carries it.
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
    ax.set_xticklabels(stage_ticks(["Search", "Abstract\nscreening",
                                    "Full-text\nscreening"]))
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
    save(fig, out_dir, "figure_gold_retention_raw")


# --------------------------------------------------------------------------- Figure 2

def figure2(out_dir: Path) -> None:
    """The precision climb beside recall on an attainable denominator.

    Panels swapped 2026-09-18 so precision leads: it is the claim Result 2 makes, and the recall
    panel is the control that makes it interpretable rather than a finding in its own right.

    Promoted from an alternate to Figure 2 on 2026-09-10. The raw-denominator version it
    replaced is the raw-retention figure (--only retention), which keeps the retrieval stage
    and the loss-by-stage
    breakdown. figure2alt, the same four stages on the raw denominator, was dropped as redundant
    on 2026-09-10; its findings are recorded below.

    The raw-retention figure divides every stage by all gold studies, which bills four
    availability failures to screening judgement: a study the query never returned, one whose
    full text we could not obtain, one whose full text arrived too thin to screen
    (`fulltext_incomplete`), and one that parsed to zero analyses. This panel removes each at the
    stage where it happens and removes nothing else, so each point answers "of the gold studies
    this stage could have kept, how many did it keep?".

    Judgement losses stay charged -- a study rejected at abstract or full-text screening, or
    parsed and then assigned to no construct, stays in the denominator. That is why the
    annotation denominator is not simply "gold with >= 1 parsed analysis": that set excludes
    everything full-text screening discarded and would forgive every full-text rejection.
    scripts/compute_attainable_recall.py builds it and documents the arithmetic.

    What it changes, raw -> attainable: abstract 0.833 -> 0.970, full-text 0.728 -> 0.926,
    annotation 0.479 -> 0.818. Annotation is the headline -- most of the apparent annotation loss
    is papers with nothing to annotate rather than annotation failing.

    The full-text stage moved most when `fulltext_incomplete` was reclassified as supply, which
    it plainly is: the retriever reported the text as available, then handed the screener title
    and abstract only. 43 of the 68 gold studies lost at full-text screening corpus-wide are of
    that kind against 25 genuine exclusions, and executive_function is the extreme -- 24 of its
    25, taking its full-text recall from 0.626 to 0.807. Note that this makes
    gold_survival_by_stage.csv's `retrieval` stage, which figure2 still plots, optimistic by
    those 43 studies; it counts `fulltext_available` rather than usable text.

    This does not rescue a weak project. executive_function is still lowest at full text (0.807)
    because its abstract stage loses 15 gold studies to judgement, and those stay charged
    against a denominator that has shrunk, which is why its recall keeps falling.

    THE ANNOTATION STAGE, AND WHY THE DENOMINATOR DECIDES THE VERDICT

    Carried here from figure2alt, dropped 2026-09-10, which was the only record of it. A paper
    survives annotation if it passed full-text screening AND had at least one analysis assigned
    to a construct column -- the pipeline's own answer to "did this paper actually yield data for
    a target contrast?". The stage exists because the screener is asked whether a paper *meets
    the criteria* and often correctly says yes about a paper that then yields no usable
    coordinates; the benchmark counts that as not included, so a decision right on the merits
    scores as a false positive.

    Across 9 projects the stage moves precision +0.147 (0.323 -> 0.470), improving in 9 of 9,
    and costs recall -0.108 attainable (0.926 -> 0.818), falling in 9 of 9. Both unanimous.

    The denominator decides whether that trade is worth taking, and this is the sharpest
    illustration of why the figure was rebuilt: the precision gain exceeds the recall loss in
    **6 of 9 projects on the attainable denominator and only 1 of 9 on the raw one** (where the
    loss reads -0.248). Same runs, same decisions -- the raw denominator charges annotation for
    papers that had nothing to annotate, and that alone inverts the conclusion.

    Search sits at 1.000 by construction -- it is a pure supply stage -- and is kept on the axis
    so the renormalisation is visible. The corpus ceiling it used to carry is stated in the
    panel note instead of being dropped silently.

    This is NOT the "adjusted gold" of projects/dementia/REPORT.md, which ADDS the studies
    excluded only for `Data Not Reported` (74 -> 162) as a precision correction; that report is
    explicit that using it as a recall denominator asks a harder question, not a fairer one, and
    it is defined for one project only.
    """
    path = REPO_ROOT / "reports" / "attainable_recall_by_stage.csv"
    if not path.exists():
        print("  figure2recall: run scripts/compute_attainable_recall.py first; skipped")
        return
    stages = ["search", "abstract", "fulltext", "annotation"]
    labels = ["Search", "Abstract\nscreening", "Full-text\nscreening", "Analysis\nselection"]

    adj: dict[str, dict[str, float]] = collections.defaultdict(dict)
    raw: dict[str, dict[str, float]] = collections.defaultdict(dict)
    attainable: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for r in read(path):
        if r["recall_attainable"] != "":
            adj[r["project"]][r["stage"]] = float(r["recall_attainable"])
            raw[r["project"]][r["stage"]] = float(r["recall_raw"])
            attainable[r["project"]][r["stage"]] = int(r["n_attainable"])
    projects = [p for p in PROJECT_ORDER if len(adj.get(p, {})) == len(stages)]
    if not projects:
        print("  figure2recall: no project has every stage; skipped")
        return

    prec: dict[str, dict[str, float]] = collections.defaultdict(dict)
    pr_path = REPO_ROOT / "reports" / "stage_precision_recall.csv"
    if pr_path.exists():
        for r in read(pr_path):
            if r["precision"] != "":
                prec[r["project"]][r["stage"]] = float(r["precision"])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fh(2.4)))

    # a: precision over the four stages
    ax = axes[0]
    have = [p for p in projects if all(prec.get(p, {}).get(s) is not None for s in stages)]
    for p in have:
        ax.plot(range(len(stages)), [prec[p][s] for s in stages], "-o", color=COLORS[p],
                markeredgewidth=0, alpha=0.9)
    if have:
        means_b = [st.mean([prec[p][s] for p in have]) for s in stages]
        ax.plot(range(len(stages)), means_b, marker="o", ms=3.6, mec="white", mew=0.5,
                **MEAN_KW)
        ax.annotate(f"{means_b[-1]:.2f}", (len(stages) - 1, means_b[-1]),
                    textcoords="offset points", xytext=(5, -1.5), fontsize=fs(5.8),
                    fontweight="bold", color=MEAN_COLOR, annotation_clip=False)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stage_ticks(labels))
    ax.set_xlim(-0.25, len(stages) - 0.42)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Precision vs gold standard")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    panel_label(ax, "a", dx=-0.20 - 0.06 * (FONT_SCALE - 1.0))

    # b: recall on the attainable denominator
    ax = axes[1]
    ends = []
    for p in projects:
        ys = [adj[p][s] for s in stages]
        ax.plot(range(len(stages)), ys, "-o", color=COLORS[p], markeredgewidth=0,
                alpha=0.9, label=DISPLAY[p])
        ends.append([ys[-1], f"{ys[-1]:.2f}", COLORS[p], False])
    means = [st.mean([adj[p][s] for p in projects]) for s in stages]
    ax.plot(range(len(stages)), means, marker="o", ms=3.6, mec="white", mew=0.5, **MEAN_KW)
    ends.append([means[-1], f"{means[-1]:.2f}", MEAN_COLOR, True])

    # End labels: same de-overlap as figure2, on a 0-1 axis, clamped so a crowded stack stays
    # on the plot instead of floating above it.
    ends.sort(key=lambda e: e[0])
    gap, ceiling = 0.036 * FONT_SCALE, 1.02
    for n in range(1, len(ends)):
        if ends[n][0] - ends[n - 1][0] < gap:
            ends[n][0] = ends[n - 1][0] + gap
    if ends and ends[-1][0] > ceiling:
        ends[-1][0] = ceiling
        for n in range(len(ends) - 2, -1, -1):
            if ends[n + 1][0] - ends[n][0] < gap:
                ends[n][0] = ends[n + 1][0] - gap
    for yy, txt, col, is_mean in ends:
        ax.annotate(txt, (len(stages) - 1, yy), textcoords="offset points",
                    xytext=(4, -1.5), fontsize=fs(5.6) if is_mean else fs(5.2),
                    fontweight="bold" if is_mean else "normal", color=col,
                    annotation_clip=False)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stage_ticks(labels))
    ax.set_xlim(-0.25, len(stages) - 0.42)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Recall vs attainable gold")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    # The grey in-panel note was removed 2026-09-18. It carried three things: the denominator
    # rule, the corpus ceiling (search returned a mean 86% of gold, 60-96%) and the raw-
    # denominator comparison (0.48 at analysis selection). All three are now in the caption --
    # printed here so they are not lost if anyone wonders what the panel used to say.
    ceil_raw = [raw[p]["search"] for p in projects]
    d_ann = st.mean([raw[p]["annotation"] for p in projects])
    print(f"  figure2 caption facts: search returned {st.mean(ceil_raw):.0%} of gold "
          f"({min(ceil_raw):.0%}-{max(ceil_raw):.0%}); "
          f"raw-denominator value at analysis selection {d_ann:.2f}")
    panel_label(ax, "b", dx=-0.20 - 0.06 * (FONT_SCALE - 1.0))

    handles = [Line2D([], [], marker="o", ls="-", color=COLORS[p], markersize=2.8,
                      label=DISPLAY[p]) for p in projects]
    handles.append(Line2D([], [], ls="-", color=MEAN_COLOR, lw=1.9, marker="o", markersize=3.2,
                          label="mean across projects"))
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.20),
               handletextpad=0.3, columnspacing=1.1, handlelength=1.2)
    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "figure2_precision_and_attainable_recall")


# --------------------------------------------------------------------------- Figure 3

def figure3(out_dir: Path) -> None:
    """Recover the analyses, then select among them, with selection scored in ROC space.

    Promoted from an alternate to Figure 3 on 2026-09-16. The precision-recall version it
    replaced is the precision-recall figure (--only annotationpr), which keeps the
    precision-vs-lift reordering that ROC space
    cannot show.

    All nine projects appear in both panels. The PR version dropped dementia from panel b
    because its source meta-analysis pools several studies into one gold analysis; that is a
    real caveat about how many of its analyses can be matched at all, but the ones that do match
    are scored no differently from any other project's, so excluding the project overstated the
    problem. Dementia simply contributes fewer matched analyses (29 expert assignments against
    social's 1,159) and sits at a lift of 2.1x, inside the 1.8-5.6 range rather than outside it.

    WHY ROC RATHER THAN PRECISION-RECALL

    The PR version needs a SEPARATE no-skill level per project, because a random selector's
    precision equals that project's prevalence and prevalence ranges 0.104 to 0.345 here. Nine
    baselines on one panel is the readability problem: the eye has to find the right rule for
    each point before the lift means anything.

    In ROC space there is one chance locus for every project -- the diagonal -- because a random
    selector has TPR = FPR whatever the prevalence. The lift becomes vertical distance from a
    single line, which is what makes it readable at a glance.

    THE NULL IS ANALYTIC, NOT SIMULATED

    A size-matched random selector -- one that picks the same number of analyses k out of N that
    annotation picked -- lands at exactly (k/N, k/N). Both coordinates reduce to the selection
    fraction: E[TPR] = k/N, and E[FPR] = k(1 - P/N)/(N - P) = k/N. So each project's own null
    point is a specific spot ON the diagonal, and no Monte Carlo is needed for the location.

    Only the spread needs a distribution, and that is hypergeometric in closed form, so it is
    exact rather than sampled. It is NOT drawn: the bands are narrow enough (median width 0.020
    in FPR) that nine of them added clutter without changing any reading, and every project's
    observed point sits far outside its own band regardless.

    Two projects are the exception and the caption has to say so, because the figure no longer
    shows it: dementia's band is 0.078 wide and vbm_of_ptsd's is 0.133 off N = 40. Both points
    still clear their band comfortably -- dementia observed 0.291 against a null of 0.388-0.466,
    PTSD 0.000 against 0.133-0.267 -- but the small-N softness is real and belongs in words now
    that it is not visible.

    This is the same size-matched-null logic as Figure 5, in a different space, which is worth a
    clause: the paper then uses one idea for "beat an arbitrary selection of the same size" in
    both places.
    """
    parse = read(REPO_ROOT / "reports" / "cross_project_analysis" / "parsing_metrics_by_project.csv")
    ann = [r for r in read(REPO_ROOT / "reports" / "cross_project_analysis"
                           / "annotation_aggregates.csv")
           if r["level"] == "analysis" and r["variant"] == "exhausted_manual_assumption"
           and r["scope"] == "project" and r["mode_id"] == "combined"
           and r["project_name"] in DISPLAY]
    if not ann:
        print("  figure3: no annotation rows; skipped")
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fh(2.75)),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # a: unchanged from Figure 3 -- parsing recovery against a table-only baseline.
    ax = axes[0]
    rows = []
    for r in parse:
        try:
            rows.append((r["project_name"], float(r["manual_matched_pct"]) * 100,
                         float(r["table_only_baseline_matched_pct"]) * 100))
        except (ValueError, KeyError):
            continue
    rows.sort(key=lambda z: z[1])
    # Shape encodes the ARM, colour only identifies the project -- the same division of labour
    # as every other panel. The two arms used to be grey against the project colour, which made
    # colour carry the comparison: the one job colour cannot do safely, since grey against a
    # mid-lightness hue like #8C564B separates poorly under deuteranopia and not at all in
    # greyscale. Open vs filled survives both.
    for i, (proj, llm, tab) in enumerate(rows):
        c = COLORS.get(proj, "#7F7F7F")
        ax.plot([tab, llm], [i, i], color=c, lw=1.6, alpha=0.45, zorder=1,
                solid_capstyle="round")
        ax.scatter([tab], [i], s=16, facecolors="white", edgecolors=c, linewidths=1.1, zorder=3)
        ax.scatter([llm], [i], s=16, color=c, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DISPLAY.get(p, p) for p, _, _ in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Expert analyses recovered (%)")
    ax.grid(axis="x", alpha=0.6); ax.set_axisbelow(True)
    pooled_llm = st.mean([llm for _, llm, _ in rows])
    pooled_tab = st.mean([tab for _, _, tab in rows])
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor="white",
                              markeredgecolor=INK, markeredgewidth=1.1,
                              color=INK, markersize=3.4,
                              label=f"Tables only ({pooled_tab:.0f}%)"),
                       Line2D([], [], marker="o", ls="", color=INK, markersize=3.4,
                              label=f"LLM parsing ({pooled_llm:.0f}%)")],
              # Above the axes, not inside them: every row's dumbbell spans most of the x
              # range, so there is no in-panel corner that does not sit on data. At lower right
              # it covered the Executive function and Emotion regulation rows.
              # Two columns at publication scale; stacked at deck scale, where 1.55x type makes
              # the pair wider than panel a and it ran into panel b's label.
              loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=2 if FONT_SCALE <= 1.2 else 1, frameon=False,
              handletextpad=0.3, columnspacing=1.4, borderaxespad=0.25, fontsize=fs(5.6))
    panel_label(ax, "a", dx=-0.40)

    # b: annotation in ROC space, each project against its own size-matched null
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=RULE, lw=0.8, zorder=1)
    js, labels, dots = [], [], []
    for r in sorted(ann, key=lambda r: float(r["recall"])):
        proj = r["project_name"]
        tp, fp, fn, tn = (int(float(r[k])) for k in ("tp", "fp", "fn", "tn"))
        N, P, k = tp + fp + fn + tn, tp + fn, tp + fp
        if not (N and P and (N - P)):
            continue
        tpr, fpr = tp / P, fp / (N - P)
        js.append(tpr - fpr)
        c = COLORS.get(proj, "#7F7F7F")
        null = k / N
        # Dotted and faint: the connector only has to say which null belongs to which
        # point. Drawn solid, nine of them read as data and crowded the panel.
        ax.plot([null, fpr], [null, tpr], color=c, lw=0.6, alpha=0.35, ls=":", zorder=2)
        ax.scatter([null], [null], s=13, facecolors="white", edgecolors=c, linewidths=0.9,
                   zorder=3)
        ax.scatter([fpr], [tpr], s=18, color=c, edgecolors="white", linewidths=0.4, zorder=4)
        labels.append([fpr, tpr, SHORT.get(proj, proj), c])
        dots += [(fpr, tpr), (null, null)]
    # Points cluster in TPR (six of nine between 0.78 and 0.96), so labels placed at their own
    # y overprint each other. Alternating sides halves the crowding; whatever still collides is
    # resolved in display space by deoverlap_labels() once the axes are final. Sizing the gap in
    # data units instead looked fine at publication scale and broke at deck scale, twice.
    labels.sort(key=lambda e: e[1])
    texts = []
    for n, (lx, ly, txt, c) in enumerate(labels):
        left = (n % 2 == 1) and lx > 0.05   # only PTSD, at x = 0, cannot take a
        texts.append(ax.annotate(           # left label without running off the axes
            txt, (lx, ly), textcoords="offset points", xytext=(-6 if left else 6, 0),
            va="center", ha="right" if left else "left",
            fontsize=fs(5.2), color=c, annotation_clip=False))

    # Full 0-1 on both axes so the whole ROC square is shown and the diagonal reaches both
    # corners: the chance line is the reference the panel is built on, and cropping it at the
    # data hid that. Every point sits below FPR 0.5, which is the result rather than a reason to
    # zoom. Not set_aspect("equal"): with adjustable="box" matplotlib resizes the axes at draw
    # time, after deoverlap_labels() has measured it, and the labels collide.
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate (recall)")
    ax.grid(alpha=0.6); ax.set_axisbelow(True)
    summary = ax.text(0.975, 0.215, f"mean TPR \u2212 FPR  {st.mean(js):.2f}\n(chance = 0)",
                      transform=ax.transAxes, va="top", ha="right", fontsize=fs(5.6),
                      color=INK, linespacing=1.5)
    # "open marker = same-size random draw" removed 2026-09-18; the caption defines the glyph.
    panel_label(ax, "b", dx=-0.24)

    fig.subplots_adjust(wspace=0.40)
    deoverlap_labels(fig, ax, texts, blockers=[summary], markers=dots)
    save(fig, out_dir, "figure3_recover_and_select_analyses")


# ------------------------------------ Annotation precision-recall space (unnumbered)

def figure_annotation_pr(out_dir: Path) -> None:
    """Annotation as an operating point in precision-recall space.

    Was Figure 3 until 2026-09-16, when the ROC version took that slot. It is kept because it
    is the only panel that shows the precision-vs-lift reordering: social has the third-highest
    precision (0.618) but the LOWEST lift (1.8x) because its prevalence is the highest in the
    set (0.345), while vbm_of_substance_use turns a similar 0.584 into 5.6x off a prevalence of
    0.104. ROC space replaces the nine per-project no-skill levels with one diagonal, which is
    the readability win, and loses exactly that comparison.

    Panel b here excludes dementia; main-text Figure 3 does not. See that docstring for why the
    exclusion was dropped.

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
    # in-panel explainer removed 2026-09-18; the caption carries it.
    panel_label(ax, "b", dx=-0.24)

    fig.text(0.5, -0.06, "* dementia excluded from b: its gold analyses pool several studies each",
             ha="center", fontsize=fs(5.2), color=MUTED)
    fig.subplots_adjust(wspace=0.42)
    save(fig, out_dir, "figure_annotation_precision_recall")


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
    # in-panel explainer removed 2026-09-18; the caption defines it.
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

    # Legend from the projects actually drawn, not the fixed order: with dementia excluded
    # from the map-level figures a fixed list advertises a colour absent from the panel.
    drawn = {r["project"] for r in rows}
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[p], markersize=3.2,
                      label=DISPLAY[p]) for p in PROJECT_ORDER if p in drawn]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.16),
               handletextpad=0.3, columnspacing=1.1)
    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "figure4_pipeline_vs_baseline")


# --------------------------------------------------------------------------- Figure 5

def figure5(out_dir: Path) -> None:
    """Which selection step earns the advantage: choosing papers, or choosing analyses?

    Promoted to the main text 2026-09-09, replacing the size-matched-null forest that was Figure 5
    (now Supplementary S3). It makes the same claim more directly -- two scatters that look
    different, no null model needed to read them -- and it decomposes the Figure 4 margin rather
    than opening a separate comparison.

    THE TWO PANELS ARE TWO DIFFERENT DESIGNS, ON PURPOSE

    Panel a is a step in the end-to-end chain: the canonical run's `all_analyses` column -- every
    parsed analysis from the studies that survived screening, no annotation -- against the search
    baseline. Articles chosen, analyses not.

    Panel b changed 2026-09-18 and is NOT the next step in that chain. It is the annotation-only
    arm, which starts from the expert inclusion list and never screens, so the study pool is held
    fixed and the only thing that varies is whether analyses were selected. Its baseline is the
    `all_analyses` map from that same pool. This is the design the Results text describes, and it
    is the cleaner test of analysis selection because no article-level difference can leak into
    it.

    WHAT THIS COSTS: the panels no longer sum. The old panel b was the second half of an additive
    decomposition of Figure 4's margin (+0.005 and +0.109 summing to +0.114). Panel b now lives on
    a different arm and a different pool, so nothing adds up and the caption must not claim it
    does. What is gained is a second, independent estimate of the same quantity: +0.109 from the
    chain and +0.100 with the pool held fixed. The additive decomposition still exists in
    reports/selection_decomposition.csv and can be quoted from there.

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
        print("  figure5: run scripts/decompose_selection_gain.py first; skipped")
        return
    rows = filter_rows(read(path), announce=False)
    rows = [r for r in rows if r["project"] not in MAP_LEVEL_EXCLUDED_PROJECTS]
    for r in rows:
        for k in ("r2_baseline", "r2_screening_only", "gain_paper_selection"):
            r[k] = float(r[k])

    # Panel b: the annotation-only arm, pool held fixed. r-squared from the stored Pearson r,
    # which compare_meta_to_benchmark computes on the UNTHRESHOLDED maps, per the metric/map
    # rule. All correlations in this corpus are positive, so squaring loses no sign.
    apath = REPO_ROOT / "reports" / "annotation_value.csv"
    if not apath.exists():
        print("  figure5: run scripts/annotation_value.py first; skipped")
        return
    arows = filter_rows(read(apath), project_key="project", announce=False)
    arows = [r for r in arows if r["project"] not in MAP_LEVEL_EXCLUDED_PROJECTS
             and r["pearson_annotated"] and r["pearson_all_analyses"]]
    for r in arows:
        r["r2_all_analyses"] = float(r["pearson_all_analyses"]) ** 2
        r["r2_annotated"] = float(r["pearson_annotated"]) ** 2
        r["gain_analysis_only"] = r["r2_annotated"] - r["r2_all_analyses"]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, fh(3.05)))
    panels = (
        ("a", "r2_baseline", "r2_screening_only", "Search-only baseline $R^2$",
         "Article screening only $R^2$", "gain_paper_selection",
         "Article selection only"),
        ("b", "r2_all_analyses", "r2_annotated", "All analyses, fixed pool $R^2$",
         "Analysis selection only $R^2$", "gain_analysis_only",
         "Analysis selection only"),
    )
    for ax, (letter, xk, yk, xl, yl, gk, what) in zip(axes, panels):
        src = rows if letter == "a" else arows
        ax.plot([0, 1], [0, 1], color=RULE, lw=0.7, zorder=1)
        for r in src:
            ax.scatter([r[xk]], [r[yk]], s=26, color=COLORS.get(r["project"], "#7F7F7F"),
                       edgecolors="white", linewidths=0.45, zorder=3)
        g = [r[gk] for r in src]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.grid(alpha=0.6); ax.set_axisbelow(True)
        ax.set_title(what, fontsize=fs(7.0), color=INK, pad=4)
        # in-panel explainer removed 2026-09-18; the caption defines the geometry.
        ax.text(0.97, 0.05,
                f"mean $\\Delta$ {st.mean(g):+.3f}\nmedian {st.median(g):+.3f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=fs(5.6),
                color=INK, linespacing=1.5)
        panel_label(ax, letter, dx=-0.24)

    # Legend from the projects actually drawn, not the fixed order: with dementia excluded
    # from the map-level figures a fixed list advertises a colour absent from the panel.
    drawn = {r["project"] for r in rows} | {r["project"] for r in arows}
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[p], markersize=3.8,
                      label=DISPLAY[p]) for p in PROJECT_ORDER if p in drawn]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.19),
               handletextpad=0.3, columnspacing=1.1)
    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "figure5_selection_decomposition")


# -------------------------------------------------------------------- Supplementary S3

def figureS3(out_dir: Path) -> None:
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
    rows = filter_rows([r for r in read(path) if r.get("status") == "ok"
                        and r["project"] not in MAP_LEVEL_EXCLUDED_PROJECTS],
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
    save(fig, out_dir, "figureS3_size_matched_null")


# -------------------------------------------------------------------- Supplementary S1

def figureS1(out_dir: Path) -> None:
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
        print("  figureS1: run scripts/compile_tier_progression.py first; skipped")
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

    def seg(a: str, b: str) -> dict[str, float]:
        """Paired deltas by project, excluding projects where both tiers are the same run.

        Keyed by project rather than a bare list so segments can be compared by membership --
        the verbatim -> best segment below is defined as the projects in neither paired segment.
        """
        return {proj: m[b] - m[a] for proj, m in means.items()
                if a in m and b in m and runs[proj].get(a) != runs[proj].get(b)}

    vm, mb = seg("verbatim", "manual"), seg("manual", "best")
    # Four projects never had a distinct manual stage, so they contribute to neither paired
    # segment above and their lines run straight from verbatim to best. Quoting only the paired
    # segments left them invisible in the summary while being visible in the panel, which
    # overstated how general the large mis-specification gain is: end to end they move +0.018
    # against +0.228 for the two that were reworked by hand.
    vb = seg("verbatim", "best")
    vb_nomanual = {k: v for k, v in vb.items() if k not in (set(vm) | set(mb))}

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
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.46 * (hi - lo))   # headroom for three arrows
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
        txt = (f"{lab}\nmean {st.mean(deltas.values()):+.3f} (n={len(deltas)})"
               if deltas else lab)
        ax.text((x0 + x1) / 2, band - 0.012 * (y1 - y0), txt, ha="center", va="top",
                fontsize=fs(4.9), color=MUTED, linespacing=1.35)

    # The four projects with no distinct manual stage, end to end, drawn under the paired pair
    # so the contrast in magnitude is the thing the eye picks up.
    if vb_nomanual:
        band2 = y1 - 0.205 * (y1 - y0)
        ax.annotate("", xy=(1.98, band2), xytext=(-0.02, band2),
                    arrowprops=dict(arrowstyle="<->", lw=0.6, color=MUTED, shrinkA=0, shrinkB=0))
        ax.text(0.98, band2 - 0.012 * (y1 - y0),
                f"no manual stage\nmean {st.mean(vb_nomanual.values()):+.3f} "
                f"(n={len(vb_nomanual)})",
                ha="center", va="top", fontsize=fs(4.9), color=MUTED, linespacing=1.35)
    save(fig, out_dir, "figureS1_tier_progression")


# -------------------------------------------------------------------- Supplementary S2

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


def figureS2(out_dir: Path) -> None:
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
    STAGES = ["search", "abstract", "fulltext", "annotation"]
    LABELS = ["Search", "Abstract\nscreening", "Full-text\nscreening", "Analysis\nselection"]
    # Both arms now come from compute_stage_precision_recall.py rather than the older
    # cross_project_screening tables, so the annotation stage is available and both metrics come
    # from one source. Verified identical to the committed tables on all 36 shared stage-values.
    SEARCH_FILE = REPO_ROOT / "reports" / "stage_precision_recall.csv"
    FIXED_FILE = REPO_ROOT / "reports" / "stage_precision_recall_allstudies.csv"

    def series(path: Path, metric: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = collections.defaultdict(dict)
        if not path.exists():
            return out
        for r in read(path):
            if r.get(metric) not in ("", None):
                out[r["project"]][r["stage"]] = float(r[metric])
        return out

    prec_s, prec_f = series(SEARCH_FILE, "precision"), series(FIXED_FILE, "precision")
    rec_s, rec_f = series(SEARCH_FILE, "recall"), series(FIXED_FILE, "recall")
    projects = [p for p in PROJECT_ORDER
                if all(p in d for d in (prec_s, prec_f, rec_s, rec_f))
                and all(st_ in prec_s[p] and st_ in prec_f[p] for st_ in STAGES)]
    if not projects:
        print("  figureS2: no project has both pools; skipped")
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
            f"shaded = pool contribution\nat {LABELS[-1].replace(chr(10), ' ').lower()}: "
            f"{m_f[-1] - m_s[-1]:+.3f} precision\n"
            f"({m_s[-1]:.3f} search pool \u2192 {m_f[-1]:.3f} fixed)",
            transform=ax.transAxes, ha="left", va="top", fontsize=fs(5.2), color=INK,
            linespacing=1.5)
    ax.set_xticks(list(xs)); ax.set_xticklabels(stage_ticks(LABELS))
    ax.set_xlim(-0.2, len(STAGES) - 0.35)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("Precision vs expert inclusion list")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    panel_label(ax, "a", dx=-0.19)

    # b: the paired deltas, precision against recall
    ax = axes[1]
    width = 0.15
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
    ax.set_xticks(list(xs)); ax.set_xticklabels(stage_ticks(LABELS))
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    # Headroom above the tallest precision point so the six-line note clears it.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.72 * (hi - lo))
    ax.set_ylabel("Fixed pool \u2212 search pool")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=fs(5.2), handlelength=1.0, handletextpad=0.4,
              borderaxespad=0.4)
    # in-panel commentary removed 2026-09-18; the caption carries it.
    panel_label(ax, "b", dx=-0.19)

    handles = [Line2D([], [], marker="o", ls="-", color=COLORS[p], markersize=2.8,
                      label=DISPLAY[p]) for p in projects]
    handles += [Line2D([], [], ls="-", color=MEAN_COLOR, lw=1.9, label="mean, search pool"),
                Line2D([], [], ls=":", color=MEAN_COLOR, lw=1.9, label="mean, fixed pool")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.17),
               handletextpad=0.3, columnspacing=1.1, handlelength=1.3)
    fig.subplots_adjust(wspace=0.30)
    save(fig, out_dir, "figureS2_pool_mismatch")


# -------------------------------------------------------------------- Supplementary S4

BRAIN_MAP_SCRIPT = Path(__file__).resolve().parent / "make_brain_map_figure.py"


def figureS4(out_dir: Path) -> None:
    """Axial slices for every scored column, all three arms side by side.

    Shelled out rather than drawn here, for two reasons worth stating so nobody "tidies" it
    into this module. It needs nilearn 0.13 for the current plot_stat_map API and the pixi
    environment pins 0.10.1, so it runs under the system interpreter. And it takes about two
    minutes against a couple of seconds for every other figure, so main() reports and skips it
    on a full run; `--only S4` builds it.

    Its column set is read from cross_project_best_baseline.csv, so it inherits the dementia
    exclusion automatically and stays at 28 rows, consistent with Figures 4, 5 and S3. Before
    this was wired up the figure was outside the registry entirely, which meant a full rebuild
    silently left it stale -- it sat at the pre-exclusion 32-column version for a week.
    """
    cmd = ["python3", str(BRAIN_MAP_SCRIPT), "--mode", "all",
           "--output-dir", str(out_dir), "--name", "figureS4_brain_maps_all"]
    if FONT_SCALE != 1.0:
        cmd += ["--font-scale", f"{FONT_SCALE}"]
    print(f"  figureS4: rendering via {BRAIN_MAP_SCRIPT.name} under system python3 "
          f"(~2 min)...")
    try:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  figureS4: FAILED ({exc}). Build it directly with:")
        print("    python3 paper/make_brain_map_figure.py --mode all")


# -------------------------------------------------------------------- Supplementary S5

def figureS5(out_dir: Path) -> None:
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
    # in-panel explainer removed 2026-09-18; the caption defines it.
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
    save(fig, out_dir, "figureS5_measured_cost")


# Keys are strings because the cost figure moved to the supplement: it is "S1", not 6. Nature
# allows six display items and the brain-surface figure is a stronger use of the slot.
# S4 shells out to make_brain_map_figure.py and is SKIPPED on a full run -- see SLOW_FIGURES.
# The three figures with descriptive keys lost their S-numbers on 2026-09-16; they are
# still built and still correct, just no longer cited.
FIGURES = {"2": figure2, "3": figure3, "4": figure4, "5": figure5,
           "S1": figureS1, "S2": figureS2, "S3": figureS3, "S4": figureS4,
           "S5": figureS5,
           "retention": figure_gold_retention_raw_fig,
           "annotationpr": figure_annotation_pr}


SLOW_FIGURES = {"S4": "~2 min, shells out to make_brain_map_figure.py "
                      "under system python3 for nilearn 0.13"}


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
        # Reported rather than silently omitted: the whole reason S4 is in the registry is that
        # being outside it let a full rebuild leave the figure stale without saying so.
        if n in SLOW_FIGURES and not args.only:
            print(f"  figure {n}: {SLOW_FIGURES[n]} -- skipped on a full run, "
                  f"build with --only {n}")
            continue
        try:
            FIGURES[n](args.output_dir)
        except FileNotFoundError as exc:
            print(f"  figure {n}: missing input ({exc.filename}); skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

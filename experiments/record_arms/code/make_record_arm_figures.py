#!/usr/bin/env python3
"""Nature-format figures for the record arms.

    figure2_record_arms   gold retention and precision through the funnel, by arm
    figure7_record_arms   screening F1 against map R^2, by arm

Companions to make_nature_methods_figures.py, whose house style, palette, panel labels and
save() this imports rather than restates. Figure 2's stage definitions come from
compute_gold_survival.py the same way -- `survivors()` and `load_gold()` are imported, not
re-implemented, so a change to what "retrieval" counts cannot drift between the two figures.

Only the four projects with record arms appear. A legend entry for a series absent from every
panel costs a reader more than it tells them.

Figure 2 is drawn as arms rather than projects because the arms share their first three
stages: search, abstract screening and retrieval are copied from the baseline's cache, so
the three lines are identical until full-text screening and every visible difference belongs
to that stage. The exception is dementia, whose baseline ran a bare model slug and donated no
valid abstract cache, so its arms re-screened abstracts; that is annotated on the figure.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

#: house_style/palette and the stage definitions live with the repo's other figure code;
#: import them rather than restate them, so this figure cannot drift from figure 2 proper.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compute_gold_survival import STAGES, load_gold, survivors  # noqa: E402
from make_nature_methods_figures import (  # noqa: E402
    COLORS, DISPLAY, DOUBLE_COL, INK, MUTED, RULE, house_style, panel_label, read, save,
)

HERE = Path(__file__).resolve().parent.parent          # experiments/record_arms
DEFAULT_OUT = HERE / "figures"
DEFAULT_DATA = HERE / "data"

PROJECTS = ["cue_reactivity", "dementia", "vbm_of_substance_use", "vbm_of_ptsd"]
ARMS = ["full text", "record + evidence", "record, no evidence"]
RUNS = {
    "cue_reactivity": ("v5-gpt-A1-mini", "v5-gpt-record-with-evidence",
                       "v5-gpt-record-no-evidence"),
    "dementia": ("v3", "v3-record-with-evidence", "v3-record-no-evidence"),
    "vbm_of_substance_use": ("v2", "v2-record-with-evidence", "v2-record-no-evidence"),
    "vbm_of_ptsd": ("v1-A1-mini", "v1-record-with-evidence", "v1-record-no-evidence"),
}
#: Shape as well as colour, so the arms stay separable in greyscale at 3pt.
STYLE = {"full text": ("o", "-", INK),
         "record + evidence": ("s", "-", "#0072B2"),
         "record, no evidence": ("^", "--", "#D55E00")}
SCREENING = {"vbm_of_ptsd": "ptsd_record_arms.csv", "cue_reactivity": "cue_record_arms.csv",
             "dementia": "dementia_record_arms.csv",
             "vbm_of_substance_use": "sud_record_arms.csv"}
#: Single words: at three panels across, the two-line labels collided with their neighbours.
#: The caption carries the full stage names.
STAGE_LABELS = ["Search", "Abstract", "Retrieval", "Full text"]


def arm_of(run: str) -> str | None:
    if run.endswith("record-with-evidence"):
        return "record + evidence"
    if run.endswith("record-no-evidence"):
        return "record, no evidence"
    if run.endswith("A1-mini") or run in ("v2", "v3"):
        return "full text"
    return None


# ------------------------------------------------------------------ figure 2, by arm

def gather_survival(projects_root: Path) -> tuple[dict, dict]:
    """cumulative gold recall, and precision, per project x arm x stage."""
    gold = load_gold()
    recall: dict = defaultdict(lambda: defaultdict(dict))
    precision: dict = defaultdict(lambda: defaultdict(dict))
    for project in PROJECTS:
        g = gold.get(project) or set()
        if not g:
            print(f"  {project}: no gold set", file=sys.stderr)
            continue
        for run, arm in zip(RUNS[project], ARMS):
            outputs = projects_root / project / run / "outputs"
            if not outputs.is_dir():
                print(f"  {project}/{run}: no outputs", file=sys.stderr)
                continue
            got = survivors(outputs)
            for stage in STAGES:
                s = got.get(stage)
                if s is None:
                    continue
                hit = s & g
                recall[project][arm][stage] = len(hit) / len(g)
                if s:
                    precision[project][arm][stage] = len(hit) / len(s)
    return recall, precision


def figure2_arms(out_dir: Path, projects_root: Path) -> None:
    recall, precision = gather_survival(projects_root)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.4))

    for ax, data, ylabel, pct in (
        (axes[0], recall, "Gold-standard studies retained (%)", True),
        (axes[1], precision, "Precision vs gold standard", False),
    ):
        for project in PROJECTS:
            for arm in ARMS:
                ys = [data.get(project, {}).get(arm, {}).get(s) for s in STAGES]
                if any(v is None for v in ys):
                    continue
                marker, ls, colour = STYLE[arm]
                ax.plot(range(len(STAGES)), [v * 100 if pct else v for v in ys],
                        marker=marker, linestyle=ls, color=COLORS[project],
                        markeredgewidth=0, markersize=2.6, lw=0.8, alpha=0.85)
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGE_LABELS)
        ax.set_xlim(-0.25, len(STAGES) - 0.55)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 1.02)

    # Panel c. Panels a and b spend three of their four stages showing lines that are
    # identical by construction -- the arms inherit search, abstract screening and retrieval
    # from the same cache -- so the contrast the figure exists to show is compressed into the
    # last tick. This panel drops the shared stages and plots only full-text screening, as a
    # precision-recall trajectory per project: one point per arm, joined in the order
    # full text -> +evidence -> -evidence. Up-and-left means an arm kept more gold and
    # admitted more non-gold; down-and-right means it rejected more of both.
    ax = axes[2]
    for project in PROJECTS:
        pts = []
        for arm in ARMS:
            r = recall.get(project, {}).get(arm, {}).get("fulltext")
            pr = precision.get(project, {}).get(arm, {}).get("fulltext")
            if r is None or pr is None:
                pts = []
                break
            pts.append((pr, r * 100, arm))
        if not pts:
            continue
        ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts],
                color=COLORS[project], lw=0.7, alpha=0.55, zorder=1)
        for x, y, arm in pts:
            marker, _, _ = STYLE[arm]
            ax.plot([x], [y], marker=marker, color=COLORS[project], markersize=3.4,
                    markeredgewidth=0.6, zorder=2,
                    markerfacecolor=COLORS[project] if arm != "record, no evidence" else "white")
    ax.set_xlabel("Precision vs gold standard")
    ax.set_ylabel("Gold-standard studies retained (%)")
    ax.set_title("Full-text screening only", pad=4, fontsize=6)
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.10, y=0.10)   # cue reactivity's trajectory sat on the left spine

    panel_label(axes[0], "a", dx=-0.24)
    panel_label(axes[1], "b", dx=-0.24)
    panel_label(axes[2], "c", dx=-0.24)

    drops = [(recall[p]["full text"]["fulltext"] - recall[p]["record + evidence"]["fulltext"]) * 100
             for p in PROJECTS if recall.get(p, {}).get("full text", {}).get("fulltext")
             and recall[p].get("record + evidence", {}).get("fulltext")]
    if drops:
        axes[0].text(0.03, 0.06,
                     f"records cost a median\n{st.median(drops):.1f} points at full-text screening",
                     transform=axes[0].transAxes, fontsize=5.5, color=MUTED, linespacing=1.5)

    handles = [Line2D([], [], marker="o", ls="-", color=COLORS[p], markersize=2.8,
                      label=DISPLAY[p]) for p in PROJECTS]
    handles += [Line2D([], [], marker=STYLE[a][0], ls=STYLE[a][1], color=INK,
                       markersize=2.8, label=a) for a in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.28),
               handletextpad=0.3, columnspacing=1.1, handlelength=1.4)
    fig.text(0.5, -0.36,
             "Stages in a and b are search, abstract screening, full-text retrieval and "
             "full-text screening. Arms share the first three through the baseline's cache, so "
             "they\nseparate only at the last; dementia's arms re-screened abstracts. Panel c "
             "drops the shared stages and shows that last step alone: each line is one project, "
             "each\nmarker one arm, joined full text to +evidence to -evidence. Down-and-right "
             "means an arm rejected more gold and more non-gold than the arm before it.",
             ha="center", fontsize=5, color=MUTED, linespacing=1.6)
    fig.subplots_adjust(wspace=0.42)
    save(fig, out_dir, "figure2_record_arms")


# ------------------------------------------------------------------ figure 7, by arm

def figure7_arms(out_dir: Path, reports: Path, r2_path: Path) -> None:
    f1: dict = defaultdict(dict)
    for project, name in SCREENING.items():
        path = reports / name
        if not path.is_file():
            print(f"  missing {path}", file=sys.stderr)
            continue
        for row in read(path):
            arm = arm_of(row["arm"])
            if arm:
                f1[project][arm] = float(row["f1"])

    raw: dict = defaultdict(lambda: defaultdict(list))
    for row in read(r2_path):
        if row.get("r2"):
            raw[row["project"]][row["arm"]].append(float(row["r2"]))
    r2 = {p: {a: st.mean(v) for a, v in arms.items()} for p, arms in raw.items()}
    n_pairs = {p: len(next(iter(arms.values()), [])) for p, arms in raw.items()}
    # All four projects appear. VBM PTSD was excluded until its annotation stage was fixed:
    # its `metadata_fields` omitted `study_fulltext`, which both hid the arm's own text from
    # the annotation model and kept it out of `study_input_hash`, so a donated annotation
    # cache could not tell the arms apart and all three maps were one file. Swapping
    # `study_abstract` for `study_fulltext` -- matching cue_reactivity and dementia -- made
    # the stage arm-sensitive and forced it to re-run. The arms now select 18, 21 and 25
    # analyses out of their own 27-, 22- and 23-paper included sets.
    shown = [p for p in PROJECTS if p in r2]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.34))
    #: Coincident markers hide each other -- the white-filled arm would erase the two beneath
    #: it. A fixed offset per arm, the same in both panels, keeps every arm visible.
    offset = {"full text": -0.13, "record + evidence": 0.0, "record, no evidence": 0.13}
    for ax, data, ylabel, title, projects, ylim in (
        (axes[0], f1, "Screening F1", "Paper-level screening", PROJECTS, (0.4, 0.8)),
        (axes[1], r2, "Map $R^2$ vs manual", "Meta-analytic map", shown, (0.2, 0.7)),
    ):
        xs = range(len(projects))
        for arm in ARMS:
            marker, _, colour = STYLE[arm]
            ax.plot([x + offset[arm] for x in xs],
                    [data.get(p, {}).get(arm) for p in projects], marker=marker,
                    color=colour, linestyle="none", markersize=4, markeredgewidth=0.7,
                    markerfacecolor=colour if arm != "record, no evidence" else "white")
        for x in xs:
            ax.axvline(x, color=RULE, linewidth=0.4, zorder=0)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(
            [DISPLAY[p] + (f"\n({n_pairs[p]} analysis)" if n_pairs.get(p) == 1
                           else f"\n({n_pairs[p]} analyses)") if ax is axes[1] else DISPLAY[p]
             for p in projects], fontsize=5.6)
        ax.set_xlim(-0.5, len(projects) - 0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=4)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", linewidth=0.4)
        ax.set_axisbelow(True)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")

    handles = [Line2D([], [], marker=STYLE[a][0], color=STYLE[a][2], linestyle="none",
                      markersize=4, markeredgewidth=0.7,
                      markerfacecolor=STYLE[a][2] if a != "record, no evidence" else "white",
                      label=a) for a in ARMS]
    axes[1].legend(handles=handles, loc="lower right", handletextpad=0.4, borderpad=0.2)
    # Hard-wrapped rather than left to the renderer: fig.text does not wrap, and at this
    # width the unwrapped caption ran off both edges of the canvas.
    fig.text(0.5, -0.34,
             "MKDA density, NiMARE default kernel, FDR-independent correction; maps regenerated "
             "identically for every arm.\nPanel b averages each project's mapped analyses, "
             "counted under its label, and offsets the arms horizontally so coincident points "
             "stay visible.\nNo mapped analysis comes from a paper its own arm rejected. "
             "Cue reactivity, dementia and VBM PTSD annotate from each arm's own\ndocument; "
             "VBM substance use annotates from title and abstract only, because its full-text "
             "arm's articles are not on this host.\nRe-running one project's annotation "
             "unchanged moved its arms by 0.034 $R^2$, which exceeds several of the "
             "between-arm gaps shown here.",
             ha="center", va="top", fontsize=5, color=MUTED, linespacing=1.6)
    save(fig, out_dir, "figure7_record_arms")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects",
                    help="run directories; only needed for figure 2, which reads each run's "
                         "screening outputs")
    ap.add_argument("--reports", type=Path, default=DEFAULT_DATA,
                    help="directory holding <project>_record_arms.csv")
    ap.add_argument("--r2", type=Path, default=DEFAULT_DATA / "record_arms_meta_metrics.csv")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    house_style()
    figure2_arms(args.out_dir, args.projects_root)
    figure7_arms(args.out_dir, args.reports, args.r2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Composition of the criteria changes on the path to each `best` configuration.

WHY TWO PANELS AND NOT ONE

The same question has two defensible units and they disagree, so showing one alone would be a
choice disguised as a fact:

  by transition      each tier step counts once, regardless of how much it changed. This is the
                     unit of DECISION -- how many times did someone revise the criteria, and what
                     kind of revision was it.
  by criteria change each added / removed / reworded criterion counts once. This is the unit of
                     EDIT -- how much of the final criteria set came from each kind of change.

`no-source-basis` is 1 of 10 transitions (10%) but 28 of 130 individual changes (22%), because
executive_function's single tuned step rewrote 28 criteria at once -- the largest of any step in
the corpus. Quoting 10% alone understates how much of the reported criteria text has no source
argument; quoting 22% alone implies more separate acts of tuning than actually happened.

SCOPE. Only transitions on the path to the run registered as `best` for each project's canonical
family. vbm_of_ptsd contributes none: its `best` IS its first transcription, never revised.

PALETTE. Okabe-Ito members, validated with the dataviz skill's checker: all six pass the
lightness band, chroma floor, CVD separation (worst adjacent pair ΔE 15.8 deutan) and
normal-vision floor. Two slots carry a contrast WARN against white, which obligates visible
labels -- every wedge is directly labelled, so identity is never colour-alone.

    scripts/plot_best_path_composition.py [--output reports/best_path_composition]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_tiers import load_registry, resolve_tier  # noqa: E402

# Ordered so the four source-tracking categories read as a block before the two that do not.
ORDER = ["source-uncovered", "source-misencoded", "source-underspecified",
         "source-unenforced", "no-source-basis", "non-criteria"]
COLOR = {
    "source-uncovered":      "#0072B2",
    "source-misencoded":     "#009E73",
    "source-underspecified": "#56B4E9",
    "source-unenforced":     "#785EF0",
    "no-source-basis":       "#D55E00",
    "non-criteria":          "#CC79A7",
}
TRACKS_SOURCE = ORDER[:4]

INK, MUTED = "#1a1a1a", "#5a5a5a"


def best_path_rows() -> list[dict]:
    """Every tier transition on the path to each project's registered `best` run."""
    rows = {(r["project"], r["from"], r["to"]): r
            for r in csv.DictReader((REPO_ROOT / "reports" / "criteria_changes.csv").open())}
    out = []
    for project in sorted(load_registry()):
        best = resolve_tier(project, "canonical", "best")
        rungs: list[str] = []
        for tier in ("initial", "untuned", "benchmark_tuned"):
            run = resolve_tier(project, "canonical", tier)
            if run and (not rungs or rungs[-1] != run):
                rungs.append(run)
        stop = rungs.index(best) + 1 if best in rungs else len(rungs)
        for k in range(stop - 1):
            r = rows.get((project, rungs[k], rungs[k + 1]))
            if r:
                out.append(r)
    return out


def pie(ax, counts: Counter, title: str, unit: str) -> None:
    labels = [k for k in ORDER if counts.get(k)]
    vals = [counts[k] for k in labels]
    total = sum(vals)
    wedges, _ = ax.pie(
        vals, startangle=90, counterclock=False,
        colors=[COLOR[k] for k in labels],
        # A 2px surface gap between fills: white edges, not a stroke in the series colour.
        wedgeprops=dict(linewidth=1.6, edgecolor="white"),
    )
    for w, k, v in zip(wedges, labels, vals):
        ang = (w.theta1 + w.theta2) / 2
        import math
        x, y = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        ha = "left" if x >= 0 else "right"
        ax.annotate(
            f"{k}\n{v} {unit if v != 1 else unit.rstrip('s')} · {100 * v / total:.0f}%",
            xy=(0.82 * x, 0.82 * y), xytext=(1.26 * x, 1.26 * y),
            ha=ha, va="center", fontsize=6.4, color=INK, linespacing=1.35,
            arrowprops=dict(arrowstyle="-", color="#c8c8c8", linewidth=0.6,
                            shrinkA=0, shrinkB=2),
        )
    ax.set_title(title, fontsize=7.4, color=INK, pad=14)
    tracked = sum(v for k, v in counts.items() if k in TRACKS_SOURCE)
    ax.text(0, -1.62, f"{tracked} of {total} track the source paper",
            ha="center", va="center", fontsize=6.6, color=MUTED, transform=ax.transData)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=str(REPO_ROOT / "reports" / "best_path_composition"))
    args = ap.parse_args()

    rows = best_path_rows()
    by_transition = Counter(r["change_type"] for r in rows)
    by_change: Counter = Counter()
    for r in rows:
        by_change[r["change_type"]] += int(r["n_changes"])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3))
    pie(axes[0], by_transition, "By revision\n(each tier step counts once)", "steps")
    pie(axes[1], by_change, "By individual criterion\n(each added/removed/reworded line)",
        "changes")

    fig.suptitle("What kind of criteria change produced each reported configuration",
                 fontsize=8.4, color=INK, y=0.975)
    fig.text(0.5, 0.035,
             "Canonical family, nine projects. Only transitions on the path to the run "
             "registered as `best`. vbm_of_ptsd contributes none — its best IS its first "
             "transcription, never revised.",
             ha="center", fontsize=6.2, color=MUTED, wrap=True)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.84, bottom=0.14, wspace=0.55)

    out = Path(args.output)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.with_suffix('.png').relative_to(REPO_ROOT)} and .pdf")
    print(f"  by transition: {dict(by_transition)}  (total {sum(by_transition.values())})")
    print(f"  by change:     {dict(by_change)}  (total {sum(by_change.values())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

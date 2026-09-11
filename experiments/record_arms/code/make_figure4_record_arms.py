#!/usr/bin/env python3
"""Figure 4, redrawn for the three record arms.

`figure4_pipeline_vs_baseline` asks whether the pipeline beats the strongest baseline
available for each manual column, over 35 columns in 9 projects: mean delta +0.101, 95% CI
[+0.045, +0.185], sign test P = 2.2e-05. This asks the same question separately for each
arm, over the 14 columns in the 4 projects that have record arms.

METRIC. `reports/cross_project_best_baseline.csv` cannot be used here. Its numbers were
produced on another host with an unrecorded estimator, and on shared columns the committed
pipeline values differ from ours by up to 0.66 -- cannabis is 0.021 there against 0.686 here.
Subtracting those baselines from our maps would measure the estimator change, not the
pipeline. A first version of this figure did exactly that and reported a spurious +0.24.

So `score_baselines.py` re-estimates all 14 baselines from their own studysets with the
settings the arms used, and both sides are scored by `meta_r2.py`'s masks. The default metric
is the brain-masked r^2 that figure 7 uses; `--metric r2_allfinite` reproduces the repo's
whole-volume convention as a sensitivity check.

The statistics are imported from `compile_best_baselines.py`, not restated: a percentile
bootstrap resampling whole projects rather than columns, because columns within a project
share a corpus, a search and a screening run.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from compile_best_baselines import cluster_bootstrap, sign_test  # noqa: E402
from recordarms import results  # noqa: E402
from make_nature_methods_figures import (  # noqa: E402
    COLORS, DISPLAY, DOUBLE_COL, INK, MUTED, RULE, house_style, panel_label, read, save,
)

ARMS = results.ARM_ORDER
STYLE = {"full text": ("o", INK),
         "record + evidence": ("s", "#0072B2"),
         "record, no evidence": ("^", "#D55E00")}
RESAMPLES, SEED = 20000, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="r2", choices=["r2", "r2_allfinite"],
                    help="r2 = brain-masked (default); r2_allfinite = whole-volume")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--out-csv", type=Path, default=HERE / "data/record_arms_vs_baseline.csv")
    args = ap.parse_args()
    house_style()

    # Both sides come from `score`: each project's maps CSV carries its three arms and a
    # `baseline` row per manual column, all estimated with the same settings. Only columns
    # with all four are drawn -- emotion regulation's `maintain` has no baseline run.
    rows_in = results.map_columns(args.metric)
    base = {(r["project"], r["column"]): r["baseline"] for r in rows_in}
    matched = [((r["project"], r["column"]), {a: r[a] for a in ARMS}) for r in rows_in]
    if not matched:
        sys.exit("no column has all three arms and a baseline; run `score` first")
    print(f"  {len(matched)} columns with all three arms and a baseline")

    rows, deltas, by_project = [], defaultdict(list), defaultdict(lambda: defaultdict(list))
    for (project, column), per_arm in matched:
        for arm in ARMS:
            d = per_arm[arm] - base[(project, column)]
            deltas[arm].append(d)
            by_project[arm][project].append(d)
            rows.append({"project": project, "manual_annotation": column, "arm": arm,
                         "metric": args.metric,
                         "arm_r2": round(per_arm[arm], 4),
                         "baseline_r2": round(base[(project, column)], 4),
                         "delta_vs_available": round(d, 4)})
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out_csv}  ({len(rows)} rows, {len(matched)} columns)")

    stats = {}
    for arm in ARMS:
        d = deltas[arm]
        boot = cluster_bootstrap(by_project[arm], RESAMPLES, SEED)
        stats[arm] = {"mean": st.mean(d), "median": st.median(d),
                      "lo": boot["ci_low"], "hi": boot["ci_high"],
                      "p": sign_test(d), "ahead": sum(1 for x in d if x > 0), "n": len(d)}
        s = stats[arm]
        print(f"  {arm:20} delta {s['mean']:+.3f}  95% CI [{s['lo']:+.3f}, {s['hi']:+.3f}]  "
              f"ahead {s['ahead']}/{s['n']}  sign P {s['p']:.2e}")

    order = sorted(range(len(matched)),
                   key=lambda i: matched[i][1]["full text"] - base[matched[i][0]])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=RULE, lw=0.6, zorder=1)
    for key, per_arm in matched:
        project, _column = key
        for arm in ARMS:
            marker, _ = STYLE[arm]
            ax.scatter([base[key]], [per_arm[arm]], s=12, marker=marker,
                       color=COLORS.get(project, "#7F7F7F"), edgecolors="white",
                       linewidths=0.3, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Best baseline $R^2$"); ax.set_ylabel("Arm $R^2$")
    ax.text(0.04, 0.93, "above the line =\narm better", fontsize=5.5, color=MUTED,
            transform=ax.transAxes, va="top")
    ax.grid(alpha=0.6); ax.set_axisbelow(True)
    panel_label(ax, "a", dx=-0.20)

    ax = axes[1]
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    # Colour means project and shape means arm in BOTH panels; a second colour encoding for
    # the arms here would contradict panel a. The three arm means are within 0.014 of each
    # other, so they are drawn as one grey band rather than three lines that would overplot.
    lo_m, hi_m = min(stats[a]["mean"] for a in ARMS), max(stats[a]["mean"] for a in ARMS)
    ax.axhspan(lo_m, hi_m, color=MUTED, alpha=0.18, lw=0, zorder=1)
    for x, i in enumerate(order):
        key, per_arm = matched[i]
        project = key[0]
        ys = [per_arm[a] - base[key] for a in ARMS]
        ax.plot([x, x], [min(ys), max(ys)], color=RULE, lw=0.5, zorder=2)
        for arm, y in zip(ARMS, ys):
            marker, _ = STYLE[arm]
            colour = COLORS.get(project, "#7F7F7F")
            ax.plot([x], [y], marker=marker, color=colour, markersize=3,
                    markeredgewidth=0.5, zorder=3,
                    markerfacecolor=colour if arm != "record, no evidence" else "white")
    ax.set_xlim(-1, len(matched))
    ax.set_xticks([])
    ax.set_xlabel(f"{len(matched)} benchmark columns, ordered by full-text advantage")
    ax.set_ylabel("$\\Delta$ $R^2$ vs best baseline")
    ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
    txt = "\n".join(
        f"{arm}: {stats[arm]['mean']:+.3f}  [{stats[arm]['lo']:+.3f}, {stats[arm]['hi']:+.3f}]"
        for arm in ARMS)
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=5,
            color=INK, linespacing=1.5)
    panel_label(ax, "b", dx=-0.18)

    handles = [Line2D([], [], marker="o", ls="", color=COLORS[p], markersize=3.2,
                      label=DISPLAY[p]) for p in dict.fromkeys(k[0] for k, _ in matched)]
    handles += [Line2D([], [], marker=STYLE[a][0], ls="", color=INK, markersize=3.2,
                       markerfacecolor=INK if a != "record, no evidence" else "white",
                       label=a) for a in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.20),
               handletextpad=0.3, columnspacing=1.1)
    fig.text(0.5, -0.30,
             f"Figure 4's comparison, per arm, over the {len(matched)} columns whose projects "
             "have record arms. Baselines are re-estimated from their own studysets with the "
             "settings the arms used,\nbecause the committed baseline table came from a "
             "different estimator. Both sides use the brain-masked $R^2$. CI is a percentile "
             "bootstrap resampling projects, not columns.\nColour is the project and shape is "
             "the arm in both panels; the grey band in b spans the three arm means, which differ by "
             f"{hi_m - lo_m:.3f}.",
             ha="center", fontsize=5, color=MUTED, linespacing=1.6)
    fig.subplots_adjust(wspace=0.34)
    save(fig, args.out_dir, "figure4_record_arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

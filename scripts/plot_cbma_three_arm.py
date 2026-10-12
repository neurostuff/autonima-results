#!/usr/bin/env python3
"""Dot plot of the three cbma-skills arms against autonima, per project.

    python scripts/plot_cbma_three_arm.py [--out reports/cbma_skills_three_arm]

Reads projects/<p>/reports/cbma_skills_{delta_sol_mni,delta_astra_mni,haiku}/score.json (every map
fits unknown coordinate spaces as MNI) and writes a two-panel figure: screening F1 and mean map
correlation with the expert map (Pearson r of unthresholded z), one row per project plus the mean,
with autonima as a grey reference tick. The same numbers go to a CSV beside the figure.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ["cue_reactivity", "decision_making", "dementia", "emotion_regulation_2022", "executive_function",
            "problem_solving", "social", "vbm_of_ptsd", "vbm_of_substance_use"]
LABELS = {"cue_reactivity": "Cue reactivity", "decision_making": "Decision making", "dementia": "Dementia",
          "emotion_regulation_2022": "Emotion regulation", "executive_function": "Executive function",
          "problem_solving": "Problem solving", "social": "Social processing", "vbm_of_ptsd": "VBM: PTSD",
          "vbm_of_substance_use": "VBM: substance use"}
# Categorical slots 1-3 of the reference palette, in fixed order; validated for CVD separation on the
# light surface. Shapes are a second channel, so identity never rests on colour alone.
ARMS = [("Sol / Luna 5.6 (Codex)", "delta_sol_mni", "#2a78d6", "o"),
        ("Astra / Astra (Codex)", "delta_astra_mni", "#eb6834", "s"),
        ("Opus / Haiku (Claude)", "haiku", "#1baf7a", "D")]
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd", "#52514e"


def scores(project: str, key: str):
    s = json.loads((ROOT / "projects" / project / "reports" / f"cbma_skills_{key}" / "score.json").read_text())
    maps = [v for v in (s.get("maps") or {}).values() if "pearson_r" in v]
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {"f1": (s["screening"]["cbma"].get("final") or {}).get("f1"),
            "r": mean([v["pearson_r"]["cbma"] for v in maps]),
            "maps": len(maps),
            "autonima_f1": (s["screening"]["autonima"].get("final") or {}).get("f1"),
            "autonima_r": mean([v["pearson_r"]["autonima"] for v in maps if v["pearson_r"].get("autonima") is not None])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "cbma_skills_three_arm")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    data = {p: {key: scores(p, key) for _, key, _, _ in ARMS} for p in PROJECTS}
    # The autonima reference is the same run for every arm; take it from the first arm's report.
    ref = {p: data[p][ARMS[0][1]] for p in PROJECTS}

    rows = PROJECTS + ["mean"]
    def value(p, key, metric):
        if p == "mean":
            xs = [data[q][key][metric] for q in PROJECTS if data[q][key][metric] is not None]
            return sum(xs) / len(xs)
        return data[p][key][metric]
    def ref_value(p, metric):
        if p == "mean":
            xs = [ref[q][f"autonima_{metric}"] for q in PROJECTS if ref[q][f"autonima_{metric}"] is not None]
            return sum(xs) / len(xs)
        return ref[p][f"autonima_{metric}"]

    with (args.out / "three_arm_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "metric", "autonima"] + [a for a, *_ in ARMS])
        for p in rows:
            for metric in ("f1", "r"):
                w.writerow([p, "screening_f1" if metric == "f1" else "mean_map_r",
                            f"{ref_value(p, metric):.3f}"] + [f"{value(p, k, metric):.3f}" for _, k, _, _ in ARMS])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.4), sharey=True, facecolor=SURFACE)
    offsets = [-0.22, 0.0, 0.22]
    ypos = {p: i for i, p in enumerate(reversed(rows))}
    panels = [("f1", "Screening F1 (final includes vs gold)", (0.0, 0.8)),
              ("r", "Mean map r with the expert map", (0.3, 0.9))]
    for ax, (metric, title, lim) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for p in rows:
            y = ypos[p]
            if p == "mean":
                ax.axhspan(y - 0.45, y + 0.45, color="#f0efea", zorder=0, lw=0)
            rv = ref_value(p, metric)
            ax.plot([rv, rv], [y - 0.36, y + 0.36], color=REF, lw=2, solid_capstyle="round", zorder=2)
            for (name, key, color, marker), dy in zip(ARMS, offsets):
                v = value(p, key, metric)
                ax.scatter(v, y + dy, s=46, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.set_xlim(*lim)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=0, labelsize=8)
    axes[0].set_yticks([ypos[p] for p in rows])
    axes[0].set_yticklabels([LABELS.get(p, "Mean of 9 projects") for p in rows])
    for lab in axes[0].get_yticklabels():
        if lab.get_text().startswith("Mean"):
            lab.set_fontweight("bold")
    handles = [plt.Line2D([], [], ls="", marker=m, ms=7, color=c, markeredgecolor=SURFACE, label=n) for n, _, c, m in ARMS]
    handles.append(plt.Line2D([], [], color=REF, lw=2, label="autonima (reference)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.55, 0.995), labelcolor=INK)
    fig.text(0.01, 0.01, "Arms are orchestrator / judges. Maps fit unknown coordinate spaces as MNI (NiMARE 0.21.0, "
             "MKDA density, FDR). Map r is averaged over each project's scorable targets.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    for ext in ("png", "svg"):
        fig.savefig(args.out / f"three_arm_comparison.{ext}", dpi=200, facecolor=SURFACE)
    print(args.out / "three_arm_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

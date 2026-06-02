#!/usr/bin/env python3
"""Create poster-sized validation plots with one shared legend."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from make_cross_project_publication_plots import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_MANUAL_META_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROJECTS_ROOT,
    DEFAULT_SCREENING_DIR,
    display_project,
    filter_primary_manual_rows,
    float_or_none,
    int_or_none,
    load_manual_diagonal_rows,
    load_manual_run_rows,
    read_csv_rows,
    select_highest_version_rows,
)


POSTER_BG = "#F4FAFB"
POSTER_PANEL_BG = "#FBFEFE"
POSTER_HEADER = "#DFE9EB"
POSTER_BORDER = "#587A85"
POSTER_GRID = "#C8D8DD"
POSTER_TEXT = "#111111"
POSTER_MUTED = "#3F4A56"
TABLE_ONLY_COLOR = "#B85B16"
MEAN_COLOR = "#111827"
MAP_BASELINE_COLOR = "#6B7280"
PAIR_LINE_COLOR = "#A7B1BA"

PROJECT_COLORS = {
    "cue_reactivity": "#2F83B7",
    "decision_making": "#9DB8D9",
    "dementia": "#F28E2B",
    "executive_function": "#FFC48A",
    "problem_solving": "#45A84A",
    "social": "#9CDB93",
    "vbm_of_ptsd": "#D83C3E",
    "vbm_of_substance_use": "#F79599",
}

PROJECT_ORDER = [
    "cue_reactivity",
    "decision_making",
    "dementia",
    "executive_function",
    "problem_solving",
    "social",
    "vbm_of_ptsd",
    "vbm_of_substance_use",
]

LEGEND_PROJECT_LABELS = {
    "cue_reactivity": "Cue",
    "decision_making": "Decision-Making",
    "dementia": "Dementia",
    "executive_function": "Executive",
    "problem_solving": "Problem Solving",
    "social": "Social",
    "vbm_of_ptsd": "VBM PTSD",
    "vbm_of_substance_use": "VBM SUD",
}

AXIS_PROJECT_LABELS = {
    **LEGEND_PROJECT_LABELS,
    "decision_making": "DM",
    "problem_solving": "PS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-dir", type=Path, default=DEFAULT_SCREENING_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--manual-meta-dir", type=Path, default=DEFAULT_MANUAL_META_DIR)
    parser.add_argument("--projects-root", type=Path, default=DEFAULT_PROJECTS_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "poster_validation_plots",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--layout-dpi", type=int, default=200)
    parser.add_argument("--panel-width-px", type=int, default=580)
    parser.add_argument("--screening-width-px", type=int, default=640)
    parser.add_argument("--parsing-width-px", type=int, default=460)
    parser.add_argument("--annotation-width-px", type=int, default=500)
    parser.add_argument("--map-width-px", type=int, default=720)
    parser.add_argument("--panel-height-px", type=int, default=440)
    parser.add_argument("--legend-width-px", type=int, default=2400)
    parser.add_argument("--legend-height-px", type=int, default=105)
    return parser.parse_args()


def configure_poster_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": POSTER_BG,
            "axes.facecolor": POSTER_PANEL_BG,
            "axes.edgecolor": POSTER_BORDER,
            "axes.labelcolor": POSTER_TEXT,
            "axes.linewidth": 1.0,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "grid.color": POSTER_GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": POSTER_BG,
            "savefig.transparent": False,
            "xtick.color": POSTER_TEXT,
            "ytick.color": POSTER_TEXT,
        }
    )


def project_color(project: str) -> str:
    return PROJECT_COLORS.get(project, "#6B7280")


def axis_project_label(project: str) -> str:
    return AXIS_PROJECT_LABELS.get(project, display_project(project))


def legend_project_label(project: str) -> str:
    return LEGEND_PROJECT_LABELS.get(project, display_project(project))


def save_exact(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{fmt}"
        kwargs: dict[str, Any] = {"facecolor": POSTER_BG}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def new_panel(args: argparse.Namespace, width_px: int | None = None) -> tuple[plt.Figure, Any]:
    width_px = width_px or args.panel_width_px
    fig = plt.figure(
        figsize=(width_px / args.layout_dpi, args.panel_height_px / args.layout_dpi),
        dpi=args.layout_dpi,
        facecolor=POSTER_BG,
    )
    return fig, None


def style_axes(ax: Any, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.grid(axis="x" if xgrid else "y", alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(POSTER_BORDER)
    ax.spines["bottom"].set_color(POSTER_BORDER)
    ax.tick_params(length=3, width=0.8)


def plot_screening(stage_rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> list[Path]:
    rows = [
        row
        for row in stage_rows
        if str(row.get("metric", "")) in {"recall", "precision"}
        and str(row.get("stage", "")) in {"search", "abstract", "fulltext"}
        and float_or_none(row.get("value")) is not None
    ]
    fig, _ = new_panel(args, args.screening_width_px)
    axes = [fig.add_axes([0.12, 0.22, 0.36, 0.57]), fig.add_axes([0.58, 0.22, 0.36, 0.57])]
    stage_order = ["search", "abstract", "fulltext"]
    stage_labels = ["Search", "Abstract", "Full"]
    stage_x = {stage: idx for idx, stage in enumerate(stage_order)}

    projects = sorted({str(row["project_name"]) for row in rows})
    for ax, metric, label in zip(axes, ["recall", "precision"], ["Recall", "Precision"]):
        stage_to_values: dict[str, list[float]] = {stage: [] for stage in stage_order}
        for project in projects:
            vals = []
            xs = []
            for row in sorted(
                [r for r in rows if r["project_name"] == project and r["metric"] == metric],
                key=lambda r: stage_x[str(r["stage"])],
            ):
                xs.append(stage_x[str(row["stage"])])
                value = float(row["value"])
                vals.append(value)
                stage_to_values[str(row["stage"])].append(value)
            if vals:
                ax.plot(xs, vals, marker="o", markersize=3.7, linewidth=1.35, color=project_color(project), alpha=0.9)
        mean_vals = []
        mean_xs = []
        for stage in stage_order:
            vals = stage_to_values[stage]
            if vals:
                mean_xs.append(stage_x[stage])
                mean_vals.append(sum(vals) / len(vals))
        if mean_vals:
            ax.plot(mean_xs, mean_vals, color=MEAN_COLOR, marker="D", markersize=3.8, linewidth=2.0, zorder=5)
        ax.set_title(label, pad=3, fontsize=9.5, fontweight="bold")
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(range(3))
        ax.set_xticklabels(stage_labels, fontsize=6.9)
        ax.tick_params(axis="y", labelsize=7.0)
        style_axes(ax)
    axes[0].set_ylabel("Score", fontsize=8.0, labelpad=1)
    axes[1].set_yticklabels([])
    fig.text(0.5, 0.91, "Screening", ha="center", va="center", fontsize=13, fontweight="bold", color=POSTER_TEXT)
    return save_exact(fig, output_dir, "01_screening", args.dpi)


def plot_parsing(parsing_rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> list[Path]:
    rows = []
    for row in parsing_rows:
        project = str(row.get("project_name", "")).strip()
        pct = float_or_none(row.get("manual_matched_pct"))
        if project and pct is not None:
            rows.append({"project": project, "pct": pct})
    values = [row["pct"] for row in rows]
    mean_value = sum(values) / len(values) if values else 0.0
    table_values = [
        float(row["table_only_baseline_matched_pct"])
        for row in parsing_rows
        if float_or_none(row.get("table_only_baseline_matched_pct")) is not None
    ]
    table_mean = sum(table_values) / len(table_values) if table_values else None

    fig, _ = new_panel(args, args.parsing_width_px)
    ax = fig.add_axes([0.22, 0.19, 0.58, 0.66])
    x = 1.0
    offsets = [0.0] if len(rows) <= 1 else [-0.19 + 0.38 * (i / (len(rows) - 1)) for i in range(len(rows))]
    for row, offset in zip(sorted(rows, key=lambda r: r["project"]), offsets):
        ax.scatter(x + offset, row["pct"], s=56, color=project_color(row["project"]), edgecolor=POSTER_TEXT, linewidth=0.9, zorder=3)
    ax.hlines(mean_value, x - 0.32, x + 0.32, color=MEAN_COLOR, linewidth=3.0, zorder=4)
    label_box = {"boxstyle": "round,pad=0.12", "facecolor": POSTER_PANEL_BG, "edgecolor": "none", "alpha": 0.92}
    ax.text(x + 0.30, mean_value, f"Mean {mean_value:.0%}", va="center", ha="left", fontsize=7.0, color=POSTER_TEXT, bbox=label_box)
    if table_mean is not None:
        ax.axhline(table_mean, color=TABLE_ONLY_COLOR, linestyle="-.", linewidth=2.4, zorder=2)
        ax.text(x + 0.30, table_mean, f"Table-only {table_mean:.0%}", va="center", ha="left", fontsize=7.0, color=TABLE_ONLY_COLOR, bbox=label_box)
    ax.set_xlim(0.55, 1.62)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([x])
    ax.set_xticklabels(["Projects"], fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Matched analyses", fontsize=8.0, labelpad=1)
    ax.tick_params(axis="y", labelsize=7.0)
    style_axes(ax)
    fig.text(0.5, 0.91, "Parsing", ha="center", va="center", fontsize=13, fontweight="bold", color=POSTER_TEXT)
    return save_exact(fig, output_dir, "02_parsing", args.dpi)


def plot_analysis_f1(version_rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> list[Path]:
    rows = select_highest_version_rows(version_rows)
    rows = [row for row in rows if float_or_none(row.get("f1")) is not None]
    rows.sort(key=lambda row: float(row["f1"]))
    fig, _ = new_panel(args, args.annotation_width_px)
    ax = fig.add_axes([0.24, 0.20, 0.68, 0.67])
    ys = list(range(len(rows)))
    vals = [float(row["f1"]) for row in rows]
    ax.barh(ys, vals, color=[project_color(str(row["project_name"])) for row in rows], edgecolor=POSTER_TEXT, linewidth=0.55, alpha=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([axis_project_label(str(row["project_name"])) for row in rows], fontsize=6.9)
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("F1", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.0)
    for y, val in zip(ys, vals):
        ax.text(min(val + 0.025, 0.97), y, f"{val:.2f}", va="center", ha="left", fontsize=6.8, color=POSTER_MUTED)
    style_axes(ax, xgrid=True, ygrid=False)
    fig.text(0.5, 0.91, "Analysis Annotation", ha="center", va="center", fontsize=13, fontweight="bold", color=POSTER_TEXT)
    return save_exact(fig, output_dir, "03_analysis_f1", args.dpi)


def plot_meta_pearson(
    run_rows: list[dict[str, Any]],
    diagonal_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[Path]:
    chart_run_rows, chart_diagonal_rows = filter_primary_manual_rows(run_rows, diagonal_rows)
    projects = sorted({str(row["project_name"]) for row in chart_run_rows})
    project_to_pairs: dict[str, list[dict[str, float]]] = {project: [] for project in projects}
    for row in chart_diagonal_rows:
        project = str(row.get("project_name", ""))
        value = float_or_none(row.get("pearson_r"))
        baseline = float_or_none(row.get("all_analyses_pearson_for_manual"))
        if project in project_to_pairs and value is not None and baseline is not None:
            project_to_pairs[project].append({"value": value, "baseline": baseline})
    projects = [project for project in projects if project_to_pairs.get(project)]
    projects.sort(
        key=lambda project: sum(pair["value"] for pair in project_to_pairs[project])
        / len(project_to_pairs[project]),
        reverse=True,
    )

    fig, _ = new_panel(args, args.map_width_px)
    ax = fig.add_axes([0.10, 0.28, 0.88, 0.58])
    xs = list(range(1, len(projects) + 1))
    for x, project in zip(xs, projects):
        pairs = project_to_pairs[project]
        pairs.sort(key=lambda pair: (pair["value"], pair["baseline"]))
        offsets = [0.0] if len(pairs) <= 1 else [-0.28 + 0.56 * (i / (len(pairs) - 1)) for i in range(len(pairs))]
        for offset, pair in zip(offsets, pairs):
            point_x = x + offset
            ax.plot(
                [point_x, point_x],
                [pair["baseline"], pair["value"]],
                color=PAIR_LINE_COLOR,
                linewidth=1.7,
                zorder=1,
            )
            ax.scatter(
                [point_x],
                [pair["baseline"]],
                s=38,
                facecolor=POSTER_PANEL_BG,
                edgecolor=POSTER_TEXT,
                linewidth=1.15,
                zorder=3,
            )
            ax.scatter(
                [point_x],
                [pair["value"]],
                s=28,
                color=project_color(project),
                edgecolor=POSTER_TEXT,
                linewidth=0.65,
                zorder=4,
            )
    ax.set_xlim(0.35, len(projects) + 0.65)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(xs)
    ax.set_xticklabels([axis_project_label(project) for project in projects], rotation=25, ha="right", fontsize=6.4)
    ax.set_ylabel("Pearson r", fontsize=8.0, labelpad=1)
    ax.tick_params(axis="y", labelsize=7.0)
    style_axes(ax)
    fig.text(0.5, 0.91, "Meta-Analytic Reproducibility", ha="center", va="center", fontsize=13, fontweight="bold", color=POSTER_TEXT)
    return save_exact(fig, output_dir, "04_map_similarity", args.dpi)


def make_project_legend(projects: list[str], args: argparse.Namespace, output_dir: Path) -> list[Path]:
    ordered = [project for project in PROJECT_ORDER if project in projects]
    ordered.extend(sorted(project for project in projects if project not in ordered))
    fig = plt.figure(
        figsize=(args.legend_width_px / args.layout_dpi, args.legend_height_px / args.layout_dpi),
        dpi=args.layout_dpi,
        facecolor=POSTER_BG,
    )
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.84])
    ax.axis("off")
    handles: list[Any] = [
            Line2D([0], [0], marker="o", linestyle="", markersize=6.0, markerfacecolor=project_color(project), markeredgecolor=POSTER_TEXT, label=legend_project_label(project))
        for project in ordered
    ]
    handles.extend(
        [
            Line2D([0], [0], color=MEAN_COLOR, linewidth=2.7, label="Mean"),
            Line2D([0], [0], color=TABLE_ONLY_COLOR, linestyle="-.", linewidth=2.4, label="Table-only"),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6.0,
                markerfacecolor=POSTER_PANEL_BG,
                markeredgecolor=POSTER_TEXT,
                label="Baseline (All Studies)",
            ),
        ]
    )
    ax.legend(
        handles=handles,
        loc="center",
        ncol=min(len(handles), 11),
        columnspacing=0.55,
        handletextpad=0.25,
        fontsize=8.1,
        frameon=False,
    )
    return save_exact(fig, output_dir, "00_shared_legend", args.dpi)


def make_preview(output_dir: Path) -> Path | None:
    try:
        from PIL import Image
    except Exception:
        return None
    panel_names = ["01_screening.png", "02_parsing.png", "03_analysis_f1.png", "04_map_similarity.png"]
    panels = [Image.open(output_dir / name).convert("RGB") for name in panel_names]
    legend = Image.open(output_dir / "00_shared_legend.png").convert("RGB")
    gutter = 28
    margin = 18
    width = sum(panel.width for panel in panels) + gutter * 3 + margin * 2
    height = max(panel.height for panel in panels) + legend.height + margin * 3
    canvas = Image.new("RGB", (width, height), POSTER_BG)
    x = margin
    y = margin
    for panel in panels:
        canvas.paste(panel, (x, y))
        x += panel.width + gutter
    legend_x = (width - legend.width) // 2
    canvas.paste(legend, (legend_x, margin * 2 + max(panel.height for panel in panels)))
    path = output_dir / "poster_bottom_row_preview.png"
    canvas.save(path)
    return path


def main() -> int:
    args = parse_args()
    configure_poster_matplotlib()

    screening_dir = args.screening_dir.expanduser().resolve()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    manual_meta_dir = args.manual_meta_dir.expanduser().resolve()
    projects_root = args.projects_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    screening_rows = read_csv_rows(screening_dir / "screening_metrics_top_v_stage_progression.csv")
    parsing_rows = read_csv_rows(analysis_dir / "parsing_metrics_by_project.csv")
    analysis_rows = read_csv_rows(analysis_dir / "analysis_assumption_strict_by_version.csv")
    manual_run_rows = load_manual_run_rows(manual_meta_dir)
    manual_diagonal_rows = load_manual_diagonal_rows(manual_meta_dir, projects_root, manual_run_rows)

    project_set: set[str] = set()
    for group in (screening_rows, parsing_rows, analysis_rows, manual_run_rows, manual_diagonal_rows):
        for row in group:
            project = str(row.get("project_name", "")).strip()
            if project:
                project_set.add(project)

    outputs: list[Path] = []
    outputs.extend(make_project_legend(sorted(project_set), args, output_dir))
    outputs.extend(plot_screening(screening_rows, args, output_dir))
    outputs.extend(plot_parsing(parsing_rows, args, output_dir))
    outputs.extend(plot_analysis_f1(analysis_rows, args, output_dir))
    outputs.extend(plot_meta_pearson(manual_run_rows, manual_diagonal_rows, args, output_dir))
    preview = make_preview(output_dir)
    if preview is not None:
        outputs.append(preview)

    for path in outputs:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

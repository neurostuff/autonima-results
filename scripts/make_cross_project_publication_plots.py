#!/usr/bin/env python3
"""Create publication-ready cross-project plots from existing report outputs.

This script is intentionally plot-only: it reads CSV/table artifacts produced by
the cross-project report scripts and does not rerun screening, parsing,
annotation, or meta-analysis comparisons.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_REPORTS_ROOT = REPO_ROOT / "reports"
DEFAULT_SCREENING_DIR = DEFAULT_REPORTS_ROOT / "cross_project_screening"
DEFAULT_ANALYSIS_DIR = DEFAULT_REPORTS_ROOT / "cross_project_analysis"
DEFAULT_MANUAL_META_DIR = DEFAULT_REPORTS_ROOT / "cross_project_manual_vs_auto_meta_fair"
DEFAULT_PROJECTS_ROOT = REPO_ROOT / "projects"
DEFAULT_OUTPUT_DIR = DEFAULT_REPORTS_ROOT / "cross_project_publication_plots"

ANNOTATION_ONLY_RUN_RE = re.compile(r"^v(?P<version>\d+)-annotation-only(?:-.+)?$")

PROJECT_LABELS = {
    "cue_reactivity": "Cue reactivity",
    "decision_making": "Decision making",
    "dementia": "Dementia",
    "emotional_regulation_2022": "Emotional regulation",
    "executive_function": "Executive function",
    "problem_solving": "Problem solving",
    "social": "Social",
    "vbm_of_ptsd": "VBM PTSD",
    "vbm_of_substance_use": "VBM substance use",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screening-dir",
        type=Path,
        default=DEFAULT_SCREENING_DIR,
        help="cross_project_screening report directory.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="cross_project_analysis report directory.",
    )
    parser.add_argument(
        "--manual-meta-dir",
        type=Path,
        default=DEFAULT_MANUAL_META_DIR,
        help="cross_project_manual_vs_auto_meta_fair report directory.",
    )
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=DEFAULT_PROJECTS_ROOT,
        help="Projects root, used as a fallback for project-level manual-vs-auto tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where standalone plots should be written.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        help="Output formats. Accepts space-separated or comma-separated values.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=450,
        help="DPI for raster outputs.",
    )
    parser.add_argument(
        "--raw-project-labels",
        action="store_true",
        help="Use raw project directory names instead of publication display labels.",
    )
    return parser.parse_args()


def normalize_formats(values: list[str]) -> list[str]:
    formats: list[str] = []
    allowed = {"png", "pdf", "svg", "eps", "tiff"}
    for value in values:
        for token in str(value).split(","):
            fmt = token.strip().lower().lstrip(".")
            if not fmt:
                continue
            if fmt not in allowed:
                raise ValueError(f"Unsupported output format: {fmt}")
            if fmt not in formats:
                formats.append(fmt)
    return formats or ["png"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except Exception:
        return None
    if math.isnan(number):
        return None
    return number


def int_or_none(value: Any) -> int | None:
    number = float_or_none(value)
    if number is None:
        return None
    return int(number)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def display_project(project: str, raw_labels: bool = False) -> str:
    if raw_labels:
        return project
    return PROJECT_LABELS.get(project, project.replace("_", " ").title())


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#111827",
            "axes.labelcolor": "#111827",
            "axes.linewidth": 0.8,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "grid.color": "#94a3b8",
            "grid.linewidth": 0.7,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "xtick.color": "#111827",
            "xtick.labelsize": 8,
            "ytick.color": "#111827",
            "ytick.labelsize": 8,
        }
    )


def make_project_colors(projects: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {project: cmap(idx % cmap.N) for idx, project in enumerate(sorted(set(projects)))}


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if fmt in {"png", "tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def collect_projects(*row_groups: list[dict[str, Any]]) -> list[str]:
    projects: set[str] = set()
    for rows in row_groups:
        for row in rows:
            project = str(row.get("project_name", "")).strip()
            if project:
                projects.add(project)
    return sorted(projects)


def plot_screening_stage_progression(
    stage_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    project_colors: dict[str, Any],
    raw_project_labels: bool,
) -> list[Path]:
    rows = [
        row
        for row in stage_rows
        if str(row.get("metric", "")).strip() in {"recall", "precision"}
        and str(row.get("stage", "")).strip() in {"search", "abstract", "fulltext"}
        and float_or_none(row.get("value")) is not None
    ]
    if not rows:
        return []

    stage_order = ["search", "abstract", "fulltext"]
    stage_labels = {"search": "Search", "abstract": "Abstract", "fulltext": "Full text"}
    stage_x = {stage: idx for idx, stage in enumerate(stage_order)}
    metric_specs = [("recall", "Recall"), ("precision", "Precision")]
    projects = sorted({str(row["project_name"]) for row in rows})

    fig_w = max(9.2, 0.55 * len(projects) + 8.2)
    fig, axes_arr = plt.subplots(1, 2, figsize=(fig_w, 4.4), sharey=True)
    axes = list(axes_arr)

    for ax, (metric_key, metric_label) in zip(axes, metric_specs):
        stage_to_values: dict[str, list[float]] = {stage: [] for stage in stage_order}
        for project in projects:
            project_rows = [
                row
                for row in rows
                if str(row.get("project_name")) == project and str(row.get("metric")) == metric_key
            ]
            project_rows.sort(key=lambda row: stage_x[str(row["stage"])])
            xs: list[int] = []
            ys: list[float] = []
            for row in project_rows:
                stage = str(row["stage"])
                value = float(row["value"])
                xs.append(stage_x[stage])
                ys.append(value)
                stage_to_values[stage].append(value)
            if not ys:
                continue
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=4.8,
                linewidth=1.7,
                color=project_colors.get(project),
                alpha=0.88,
                label=display_project(project, raw_project_labels),
            )

        mean_xs: list[int] = []
        mean_ys: list[float] = []
        for stage in stage_order:
            values = stage_to_values[stage]
            if values:
                mean_xs.append(stage_x[stage])
                mean_ys.append(sum(values) / len(values))
        if mean_ys:
            ax.plot(
                mean_xs,
                mean_ys,
                color="#111827",
                marker="D",
                markersize=4.6,
                linewidth=2.2,
                label="Mean",
                zorder=5,
            )

        ax.set_title(metric_label)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks([stage_x[stage] for stage in stage_order])
        ax.set_xticklabels([stage_labels[stage] for stage in stage_order])
        ax.set_xlabel("Screening stage")
        ax.grid(axis="y", alpha=0.32)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(5, len(labels)),
            columnspacing=1.2,
            handlelength=2.4,
        )
    fig.suptitle("Top Canonical V Run: PRF Stage Progression", y=1.02, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    return save_figure(
        fig,
        output_dir,
        "screening_top_v_stage_progression_recall_precision",
        formats,
        dpi,
    )


def plot_parsing_matched_pct(
    parsing_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    project_colors: dict[str, Any],
    raw_project_labels: bool,
) -> list[Path]:
    rows = []
    for row in parsing_rows:
        pct = float_or_none(row.get("manual_matched_pct"))
        project = str(row.get("project_name", "")).strip()
        if pct is None or not project:
            continue
        rows.append(
            {
                "project_name": project,
                "pct": pct,
                "matched_count": int_or_none(row.get("matched_count")),
                "manual_total": int_or_none(row.get("manual_analyses_total")),
                "table_only_pct": float_or_none(row.get("table_only_baseline_matched_pct")),
                "table_only_matched_count": int_or_none(row.get("table_only_baseline_matched_count")),
                "table_only_manual_total": int_or_none(
                    row.get("table_only_baseline_manual_analyses_total")
                ),
            }
        )
    if not rows:
        return []

    rows.sort(key=lambda row: row["pct"], reverse=True)
    ys = list(range(len(rows)))
    fig_h = max(3.8, 0.42 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    colors = [project_colors.get(row["project_name"]) for row in rows]
    has_table_baseline = any(row.get("table_only_pct") is not None for row in rows)
    if has_table_baseline:
        h = 0.34
        parsed_ys = [y - h / 2 for y in ys]
        baseline_ys = [y + h / 2 for y in ys]
        ax.barh(
            parsed_ys,
            [row["pct"] for row in rows],
            height=h,
            color=colors,
            alpha=0.86,
            edgecolor="#111827",
            linewidth=0.5,
            label="Parsed analyses",
        )
        ax.barh(
            baseline_ys,
            [row["table_only_pct"] or 0.0 for row in rows],
            height=h,
            color="#b45309",
            alpha=0.68,
            edgecolor="#111827",
            linewidth=0.5,
            label="Table-only baseline",
        )
    else:
        ax.barh(
            ys,
            [row["pct"] for row in rows],
            color=colors,
            alpha=0.86,
            edgecolor="#111827",
            linewidth=0.5,
        )
    ax.invert_yaxis()
    ax.set_yticks(ys)
    ax.set_yticklabels([display_project(row["project_name"], raw_project_labels) for row in rows])
    ax.set_xlim(0.0, 1.08)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Manual analyses matched")
    ax.set_title("Parsing Performance Across Projects")
    ax.grid(axis="x", alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for y, row in zip(ys, rows):
        matched = row["matched_count"]
        total = row["manual_total"]
        count_text = f" ({matched}/{total})" if matched is not None and total is not None else ""
        ax.text(
            row["pct"] + 0.015,
            y - 0.17 if has_table_baseline else y,
            f"{row['pct']:.0%}{count_text}",
            va="center",
            ha="left",
            fontsize=8,
            color="#374151",
        )
        if has_table_baseline and row.get("table_only_pct") is not None:
            tb_matched = row.get("table_only_matched_count")
            tb_total = row.get("table_only_manual_total")
            tb_count_text = (
                f" ({tb_matched}/{tb_total})"
                if tb_matched is not None and tb_total is not None
                else ""
            )
            ax.text(
                float(row["table_only_pct"]) + 0.015,
                y + 0.17,
                f"{float(row['table_only_pct']):.0%}{tb_count_text}",
                va="center",
                ha="left",
                fontsize=8,
                color="#374151",
            )

    if has_table_baseline:
        ax.legend(loc="lower right")

    fig.tight_layout()
    return save_figure(fig, output_dir, "parsing_matched_percentage_by_project", formats, dpi)


def plot_parsing_overall_matched_pct(
    parsing_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    project_colors: dict[str, Any],
    raw_project_labels: bool,
) -> list[Path]:
    rows = []
    for row in parsing_rows:
        pct = float_or_none(row.get("manual_matched_pct"))
        project = str(row.get("project_name", "")).strip()
        if pct is None or not project:
            continue
        rows.append({"project_name": project, "pct": pct})
    if not rows:
        return []

    rows.sort(key=lambda row: row["project_name"])
    values = [float(row["pct"]) for row in rows]
    mean_value = sum(values) / len(values)
    table_only_values = [
        float(row["table_only_baseline_matched_pct"])
        for row in parsing_rows
        if float_or_none(row.get("table_only_baseline_matched_pct")) is not None
    ]
    table_only_mean = sum(table_only_values) / len(table_only_values) if table_only_values else None

    fig, ax = plt.subplots(figsize=(4.6, 5.0))
    center_x = 1.0
    offsets = deterministic_offsets(len(rows), span=0.18)
    for row, offset in zip(rows, offsets):
        project = str(row["project_name"])
        ax.scatter(
            center_x + offset,
            float(row["pct"]),
            s=58,
            color=project_colors.get(project),
            edgecolor="#111827",
            linewidth=0.65,
            alpha=0.95,
            zorder=3,
            label=display_project(project, raw_project_labels),
        )

    ax.hlines(
        mean_value,
        center_x - 0.28,
        center_x + 0.28,
        color="#111827",
        linewidth=2.4,
        zorder=4,
        label=f"Mean = {mean_value:.0%}",
    )
    if table_only_mean is not None:
        ax.axhline(
            table_only_mean,
            color="#b45309",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.95,
            zorder=2,
            label=f"Table-only baseline = {table_only_mean:.0%}",
        )

    ax.set_xlim(0.55, 1.45)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([center_x])
    ax.set_xticklabels(["Projects"])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Manual analyses matched")
    ax.set_title("Overall Parsing Performance")
    ax.grid(axis="y", alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "parsing_overall_matched_percentage", formats, dpi)


def select_highest_version_rows(version_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_project: dict[str, dict[str, Any]] = {}
    for row in version_rows:
        project = str(row.get("project_name", "")).strip()
        if not project or float_or_none(row.get("f1")) is None:
            continue
        current = best_by_project.get(project)
        row_key = (int_or_none(row.get("version")) or -1, str(row.get("run", "")))
        if current is None:
            best_by_project[project] = row
            continue
        current_key = (int_or_none(current.get("version")) or -1, str(current.get("run", "")))
        if row_key > current_key:
            best_by_project[project] = row
    return sorted(best_by_project.values(), key=lambda row: str(row["project_name"]))


def plot_analysis_best_v_f1(
    version_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    project_colors: dict[str, Any],
    raw_project_labels: bool,
) -> list[Path]:
    rows = select_highest_version_rows(version_rows)
    if not rows:
        return []
    rows.sort(key=lambda row: float(row["f1"]), reverse=True)

    ys = list(range(len(rows)))
    fig_h = max(3.8, 0.42 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    colors = [project_colors.get(str(row["project_name"])) for row in rows]
    vals = [float(row["f1"]) for row in rows]
    ax.barh(ys, vals, color=colors, alpha=0.86, edgecolor="#111827", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_yticks(ys)
    ax.set_yticklabels([display_project(str(row["project_name"]), raw_project_labels) for row in rows])
    ax.set_xlim(0.0, 1.08)
    ax.set_xlabel("F1")
    ax.set_title("Analysis F1 Exhausted-Manual (Strict): Best V-Version")
    ax.grid(axis="x", alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for y, row, value in zip(ys, rows, vals):
        ax.text(
            value + 0.015,
            y,
            f"{value:.2f}  {row.get('run', '')}",
            va="center",
            ha="left",
            fontsize=8,
            color="#374151",
        )

    fig.tight_layout()
    return save_figure(fig, output_dir, "analysis_f1_exhausted_manual_strict_best_v", formats, dpi)


def parse_annotation_only_version(run_name: str) -> int | None:
    match = ANNOTATION_ONLY_RUN_RE.fullmatch(run_name)
    if not match:
        return None
    return int(match.group("version"))


def select_chart_runs_by_project(run_rows: list[dict[str, Any]]) -> dict[str, str]:
    selected: dict[str, tuple[int, int, str]] = {}
    for row in run_rows:
        project = str(row.get("project_name", "")).strip()
        run = str(row.get("run", "")).strip()
        if not project or not run:
            continue
        version = parse_annotation_only_version(run)
        if version is None:
            version = -1
        is_plain_annotation_only = int(bool(re.fullmatch(r"^v\d+-annotation-only$", run)))
        key = (version, is_plain_annotation_only, run)
        current = selected.get(project)
        if current is None or key > current:
            selected[project] = key
    return {project: key[2] for project, key in selected.items()}


def read_matrix_row_values(matrix_path: Path, row_name: str) -> dict[str, float]:
    if not matrix_path.exists():
        return {}
    with matrix_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        index_key = reader.fieldnames[0]
        for row in reader:
            if str(row.get(index_key, "")).strip() != row_name:
                continue
            values: dict[str, float] = {}
            for key, value in row.items():
                if key == index_key:
                    continue
                parsed = float_or_none(value)
                if parsed is not None:
                    values[str(key)] = parsed
            return values
    return {}


def load_manual_run_rows(manual_meta_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv_rows(manual_meta_dir / "run_metrics.csv"):
        project = str(row.get("project_name", "")).strip()
        run = str(row.get("run", "")).strip()
        if not project or not run:
            continue
        rows.append(
            {
                "project_name": project,
                "run": run,
                "dice_mean_diagonal": float_or_none(row.get("dice_mean_diagonal")),
                "pearson_mean_diagonal": float_or_none(row.get("pearson_mean_diagonal")),
                "all_analyses_dice": float_or_none(row.get("all_analyses_dice")),
                "all_analyses_pearson": float_or_none(row.get("all_analyses_pearson")),
            }
        )
    return rows


def project_report_dirs_from_status(
    manual_meta_dir: Path,
    projects_root: Path,
    run_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    report_dirs: dict[str, Path] = {}
    for row in read_csv_rows(manual_meta_dir / "project_status.csv"):
        project = str(row.get("project_name", "")).strip()
        path_text = str(row.get("project_report_dir", "")).strip()
        if project and path_text:
            report_dirs[project] = Path(path_text).expanduser()

    for row in run_rows:
        project = str(row.get("project_name", "")).strip()
        if project and project not in report_dirs:
            report_dirs[project] = projects_root / project / "reports" / "manual_vs_auto_meta_fair"
    return report_dirs


def load_manual_diagonal_rows(
    manual_meta_dir: Path,
    projects_root: Path,
    run_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    report_dirs = project_report_dirs_from_status(manual_meta_dir, projects_root, run_rows)
    diagonal_rows: list[dict[str, Any]] = []
    for project, report_dir in sorted(report_dirs.items()):
        diagonal_path = report_dir / "tables" / "diagonal_metrics.csv"
        if not diagonal_path.exists():
            continue
        dice_baseline_by_run: dict[str, dict[str, float]] = {}
        pearson_baseline_by_run: dict[str, dict[str, float]] = {}
        for row in read_csv_rows(diagonal_path):
            run_name = str(row.get("run", "")).strip()
            manual_name = str(row.get("manual_name", "")).strip()
            dice = float_or_none(row.get("dice"))
            pearson = float_or_none(row.get("pearson_r"))
            if not run_name or not manual_name or dice is None or pearson is None:
                continue

            if run_name not in dice_baseline_by_run:
                safe_run_name = sanitize_name(run_name)
                tables_dir = report_dir / "tables"
                dice_baseline_by_run[run_name] = read_matrix_row_values(
                    tables_dir / f"dice_matrix_{safe_run_name}.csv",
                    "all_analyses",
                )
                pearson_baseline_by_run[run_name] = read_matrix_row_values(
                    tables_dir / f"pearson_matrix_{safe_run_name}.csv",
                    "all_analyses",
                )

            dice_baseline = dice_baseline_by_run.get(run_name, {}).get(manual_name)
            pearson_baseline = pearson_baseline_by_run.get(run_name, {}).get(manual_name)
            diagonal_rows.append(
                {
                    "project_name": project,
                    "run": run_name,
                    "manual_name": manual_name,
                    "auto_name": str(row.get("auto_name", "")).strip(),
                    "dice": dice,
                    "pearson_r": pearson,
                    "all_analyses_dice_for_manual": dice_baseline,
                    "all_analyses_pearson_for_manual": pearson_baseline,
                    "dice_minus_all_analyses": (
                        dice - dice_baseline if dice_baseline is not None else None
                    ),
                    "pearson_minus_all_analyses": (
                        pearson - pearson_baseline if pearson_baseline is not None else None
                    ),
                }
            )
    return diagonal_rows


def filter_primary_manual_rows(
    run_rows: list[dict[str, Any]],
    diagonal_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_runs = select_chart_runs_by_project(run_rows)
    if not selected_runs:
        return run_rows, diagonal_rows
    filtered_run_rows = [
        row
        for row in run_rows
        if selected_runs.get(str(row.get("project_name", ""))) == str(row.get("run", ""))
    ]
    filtered_diagonal_rows = [
        row
        for row in diagonal_rows
        if selected_runs.get(str(row.get("project_name", ""))) == str(row.get("run", ""))
    ]
    return filtered_run_rows, filtered_diagonal_rows


def deterministic_offsets(n_values: int, span: float) -> list[float]:
    if n_values <= 1:
        return [0.0]
    return [-span + (2 * span) * (idx / (n_values - 1)) for idx in range(n_values)]


def plot_manual_pearson_per_run(
    run_rows: list[dict[str, Any]],
    diagonal_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    project_colors: dict[str, Any],
    raw_project_labels: bool,
) -> list[Path]:
    chart_run_rows, chart_diagonal_rows = filter_primary_manual_rows(run_rows, diagonal_rows)
    if not chart_run_rows:
        return []

    projects = sorted({str(row["project_name"]) for row in chart_run_rows})
    project_to_vals: dict[str, list[float]] = {project: [] for project in projects}
    if chart_diagonal_rows:
        for row in chart_diagonal_rows:
            project = str(row.get("project_name", ""))
            value = float_or_none(row.get("pearson_r"))
            if project in project_to_vals and value is not None:
                project_to_vals[project].append(value)
    else:
        for row in chart_run_rows:
            project = str(row.get("project_name", ""))
            value = float_or_none(row.get("pearson_mean_diagonal"))
            if project in project_to_vals and value is not None:
                project_to_vals[project].append(value)

    projects = [project for project in projects if project_to_vals.get(project)]
    if not projects:
        return []

    baseline_by_project: dict[str, float] = {}
    for project in projects:
        baselines = [
            float(row["all_analyses_pearson"])
            for row in chart_run_rows
            if str(row.get("project_name", "")) == project and row.get("all_analyses_pearson") is not None
        ]
        if baselines:
            baseline_by_project[project] = sum(baselines) / len(baselines)

    positions = list(range(1, len(projects) + 1))
    fig_w = max(7.8, 1.05 * len(projects) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, 4.9))
    data = [project_to_vals[project] for project in projects]
    bp = ax.boxplot(data, positions=positions, widths=0.48, patch_artist=True, showfliers=False)
    for patch, project in zip(bp["boxes"], projects):
        patch.set_facecolor(project_colors.get(project))
        patch.set_alpha(0.22)
        patch.set_edgecolor(project_colors.get(project))
        patch.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.4)
    for whisker in bp["whiskers"]:
        whisker.set_color("#475569")
        whisker.set_linewidth(1.0)
    for cap in bp["caps"]:
        cap.set_color("#475569")
        cap.set_linewidth(1.0)

    for pos, project in zip(positions, projects):
        vals = project_to_vals[project]
        xs = [pos + offset for offset in deterministic_offsets(len(vals), span=0.24)]
        ax.scatter(
            xs,
            vals,
            s=32,
            color=project_colors.get(project),
            edgecolor="#111827",
            linewidth=0.55,
            alpha=0.95,
            zorder=3,
        )
        project_mean = sum(vals) / len(vals)
        ax.hlines(project_mean, pos - 0.24, pos + 0.24, color="#111827", linewidth=2.1, zorder=4)
        if project in baseline_by_project:
            ax.hlines(
                baseline_by_project[project],
                pos - 0.28,
                pos + 0.28,
                color=project_colors.get(project),
                linestyle="--",
                linewidth=2.0,
                zorder=4,
            )

    ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.55)
    ax.set_title("Per-Run Pearson Chart")
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("Project")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [display_project(project, raw_project_labels) for project in projects],
        rotation=25,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#111827", linewidth=2.1, label="Project mean"),
            Line2D([0], [0], color="#475569", linestyle="--", linewidth=2.0, label="all_analyses baseline"),
        ],
        loc="lower left",
    )
    fig.tight_layout()
    return save_figure(fig, output_dir, "manual_vs_auto_pearson_per_run", formats, dpi)


def plot_manual_dice_delta(
    run_rows: list[dict[str, Any]],
    diagonal_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    project_colors: dict[str, Any],
    raw_project_labels: bool,
) -> list[Path]:
    _chart_run_rows, chart_diagonal_rows = filter_primary_manual_rows(run_rows, diagonal_rows)
    rows = [
        row
        for row in chart_diagonal_rows
        if str(row.get("project_name", "")).strip()
        and float_or_none(row.get("dice_minus_all_analyses")) is not None
    ]
    if not rows:
        return []

    projects = sorted({str(row["project_name"]) for row in rows})
    project_to_vals: dict[str, list[float]] = {project: [] for project in projects}
    for row in rows:
        project_to_vals[str(row["project_name"])].append(float(row["dice_minus_all_analyses"]))
    all_vals = [value for values in project_to_vals.values() for value in values]
    if not all_vals:
        return []

    mean_delta = sum(all_vals) / len(all_vals)
    fig_h = max(4.0, 0.55 * len(projects) + 1.8)
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    positions = list(range(len(projects)))
    for pos, project in zip(positions, projects):
        vals = project_to_vals[project]
        ys = [pos + offset for offset in deterministic_offsets(len(vals), span=0.30)]
        ax.scatter(
            vals,
            ys,
            s=36,
            color=project_colors.get(project),
            edgecolor="#111827",
            linewidth=0.55,
            alpha=0.95,
            zorder=3,
        )
        project_mean = sum(vals) / len(vals)
        ax.scatter(
            [project_mean],
            [pos],
            marker="o",
            s=84,
            facecolors="none",
            edgecolors="#111827",
            linewidth=1.8,
            zorder=4,
        )

    ax.axvline(0.0, color="#374151", linewidth=1.4, alpha=0.9, zorder=1)
    ax.axvline(
        mean_delta,
        color="#b91c1c",
        linestyle=":",
        linewidth=2.1,
        alpha=0.95,
        zorder=2,
        label=f"Mean difference = {mean_delta:.3f}",
    )
    max_abs = max(abs(min(all_vals)), abs(max(all_vals)), abs(mean_delta), 0.02)
    xpad = max(0.012, 0.12 * max_abs)
    ax.set_xlim(-max_abs - xpad, max_abs + xpad)
    ax.set_yticks(positions)
    ax.set_yticklabels([display_project(project, raw_project_labels) for project in projects])
    ax.set_xlabel("Diagonal Dice - all_analyses Dice")
    ax.set_title("Per-Diagonal Dice Difference vs Baseline")
    ax.grid(axis="x", alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_figure(fig, output_dir, "manual_vs_auto_dice_delta_vs_baseline", formats, dpi)


def main() -> int:
    args = parse_args()
    configure_matplotlib()

    screening_dir = args.screening_dir.expanduser().resolve()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    manual_meta_dir = args.manual_meta_dir.expanduser().resolve()
    projects_root = args.projects_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    formats = normalize_formats(args.formats)

    screening_rows = read_csv_rows(screening_dir / "screening_metrics_top_v_stage_progression.csv")
    parsing_rows = read_csv_rows(analysis_dir / "parsing_metrics_by_project.csv")
    analysis_rows = read_csv_rows(analysis_dir / "analysis_assumption_strict_by_version.csv")
    manual_run_rows = load_manual_run_rows(manual_meta_dir)
    manual_diagonal_rows = load_manual_diagonal_rows(manual_meta_dir, projects_root, manual_run_rows)

    all_projects = collect_projects(
        screening_rows,
        parsing_rows,
        analysis_rows,
        manual_run_rows,
        manual_diagonal_rows,
    )
    project_colors = make_project_colors(all_projects)

    manifest: list[dict[str, Any]] = []

    plot_jobs = [
        (
            "screening_top_v_stage_progression_recall_precision",
            plot_screening_stage_progression(
                screening_rows,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                project_colors=project_colors,
                raw_project_labels=args.raw_project_labels,
            ),
            screening_dir / "screening_metrics_top_v_stage_progression.csv",
        ),
        (
            "parsing_matched_percentage_by_project",
            plot_parsing_matched_pct(
                parsing_rows,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                project_colors=project_colors,
                raw_project_labels=args.raw_project_labels,
            ),
            analysis_dir / "parsing_metrics_by_project.csv",
        ),
        (
            "parsing_overall_matched_percentage",
            plot_parsing_overall_matched_pct(
                parsing_rows,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                project_colors=project_colors,
                raw_project_labels=args.raw_project_labels,
            ),
            analysis_dir / "parsing_metrics_by_project.csv",
        ),
        (
            "analysis_f1_exhausted_manual_strict_best_v",
            plot_analysis_best_v_f1(
                analysis_rows,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                project_colors=project_colors,
                raw_project_labels=args.raw_project_labels,
            ),
            analysis_dir / "analysis_assumption_strict_by_version.csv",
        ),
        (
            "manual_vs_auto_pearson_per_run",
            plot_manual_pearson_per_run(
                manual_run_rows,
                manual_diagonal_rows,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                project_colors=project_colors,
                raw_project_labels=args.raw_project_labels,
            ),
            manual_meta_dir / "run_metrics.csv",
        ),
        (
            "manual_vs_auto_dice_delta_vs_baseline",
            plot_manual_dice_delta(
                manual_run_rows,
                manual_diagonal_rows,
                output_dir=output_dir,
                formats=formats,
                dpi=args.dpi,
                project_colors=project_colors,
                raw_project_labels=args.raw_project_labels,
            ),
            manual_meta_dir / "project_status.csv",
        ),
    ]

    for figure_id, paths, source_path in plot_jobs:
        if not paths:
            print(f"[SKIP] {figure_id}: no usable source rows")
            continue
        for path in paths:
            manifest.append(
                {
                    "figure_id": figure_id,
                    "format": path.suffix.lstrip("."),
                    "path": str(path),
                    "source": str(source_path),
                }
            )
            print(f"[OK] {path}")

    if manifest:
        write_csv_rows(
            output_dir / "manifest.csv",
            manifest,
            ["figure_id", "format", "path", "source"],
        )
        return 0

    print("[ERROR] No plots were created. Check that the cross-project report outputs exist.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

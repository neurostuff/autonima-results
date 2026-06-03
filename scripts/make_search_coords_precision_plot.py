#!/usr/bin/env python3
"""Compare Search-Coords precision with FT-Coords precision by project."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from compare_screening_to_benchmark import (
    load_meta_pmids,
    normalize_pmid,
    normalize_pmid_list,
    resolve_meta_analysis_pmid,
)
from make_cross_project_publication_plots import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROJECTS_ROOT,
    DEFAULT_SCREENING_DIR,
    float_or_none,
    read_csv_rows,
)
from make_poster_validation_plots import (
    AXIS_PROJECT_LABELS,
    MEAN_COLOR,
    POSTER_BG,
    POSTER_BORDER,
    POSTER_GRID,
    POSTER_MUTED,
    POSTER_PANEL_BG,
    POSTER_TEXT,
    PROJECT_COLORS,
    configure_poster_matplotlib,
    project_color,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_META_PMIDS = REPO_ROOT.parent / "neurometabench" / "data" / "included_studies.csv"
DEFAULT_OUT_DIR = DEFAULT_OUTPUT_DIR / "search_coords_precision"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-root", type=Path, default=DEFAULT_PROJECTS_ROOT)
    parser.add_argument("--screening-dir", type=Path, default=DEFAULT_SCREENING_DIR)
    parser.add_argument("--meta-pmids", type=Path, default=DEFAULT_META_PMIDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--layout-dpi", type=int, default=200)
    parser.add_argument("--width-px", type=int, default=1700)
    parser.add_argument("--height-px", type=int, default=920)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
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


def axis_project_label(project: str) -> str:
    if project == "dementia_allstudies":
        return "Dementia All"
    return AXIS_PROJECT_LABELS.get(project, project.replace("_", " ").title())


def display_project_key(row: dict[str, Any]) -> str:
    condition = str(row.get("condition", "")).strip()
    if condition:
        return condition
    return str(row.get("project_name", "")).strip()


def is_dementia_allstudies(row: dict[str, Any]) -> bool:
    return display_project_key(row) == "dementia_allstudies"


def resolve_repo_path(path_value: Any) -> Path | None:
    if path_value is None:
        return None
    raw = str(path_value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()
    marker = "autonima-results/"
    if marker in raw:
        remapped = REPO_ROOT / raw.split(marker, 1)[1]
        if remapped.exists():
            return remapped.resolve()
    return path.resolve()


def load_search_pmids(run_dir: Path) -> set[str]:
    path = run_dir / "outputs" / "search_results.json"
    if not path.exists():
        return set()
    payload = read_json(path)
    return set(normalize_pmid_list([study.get("pmid") for study in payload.get("studies", [])]))


def load_run_config(run_dir: Path) -> dict[str, Any]:
    final_results_path = run_dir / "outputs" / "final_results.json"
    if final_results_path.exists():
        payload = read_json(final_results_path)
        config = payload.get("config")
        if isinstance(config, dict):
            return config
    return {}


def has_coordinate_content(path: Path) -> bool:
    try:
        payload = read_json(path)
    except Exception:
        return path.stat().st_size > 0

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            if "points" in value and isinstance(value["points"], list) and value["points"]:
                return True
            if "coordinates" in value and isinstance(value["coordinates"], list) and value["coordinates"]:
                return True
            return any(walk(child) for child in value.values())
        if isinstance(value, list):
            return any(walk(child) for child in value)
        return False

    return walk(payload)


def collect_pubget_coordinate_pmids(run_dir: Path) -> set[str]:
    pubget_dir = run_dir / "retrieval" / "pubget_data"
    metadata_path = pubget_dir / "metadata.csv"
    coordinates_path = pubget_dir / "coordinates.csv"
    if not metadata_path.exists() or not coordinates_path.exists():
        return set()

    pmid_by_pmcid: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pmcid = normalize_pmid(row.get("pmcid"))
            pmid = normalize_pmid(row.get("pmid"))
            if pmcid and pmid:
                pmid_by_pmcid[pmcid] = pmid

    pmids: set[str] = set()
    with coordinates_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pmcid = normalize_pmid(row.get("pmcid"))
            if pmcid and pmcid in pmid_by_pmcid:
                pmids.add(pmid_by_pmcid[pmcid])
    return pmids


def collect_processed_coordinate_pmids(processed_path: Path | None) -> set[str]:
    if processed_path is None:
        return set()
    coordinates_path = processed_path / "coordinates.csv" if processed_path.is_dir() else processed_path
    if not coordinates_path.exists():
        return set()
    pmids: set[str] = set()
    with coordinates_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pmid = normalize_pmid(row.get("pmid"))
            if pmid:
                pmids.add(pmid)
    return pmids


def collect_folder_coordinate_pmids(root_path: Path | None, templates: list[Any]) -> set[str]:
    if root_path is None or not root_path.exists() or not root_path.is_dir():
        return set()
    template_strings = [str(template) for template in templates if str(template).strip()]
    if not template_strings:
        template_strings = ["coordinates.json"]

    pmids: set[str] = set()
    for child in root_path.iterdir():
        if not child.is_dir():
            continue
        pmid = normalize_pmid(child.name)
        if not pmid:
            continue
        for template in template_strings:
            candidate = child / template
            if candidate.exists() and has_coordinate_content(candidate):
                pmids.add(pmid)
                break
    return pmids


def collect_source_coordinate_pmids(run_dir: Path) -> tuple[set[str], dict[str, int]]:
    config = load_run_config(run_dir)
    retrieval = config.get("retrieval", {}) if isinstance(config, dict) else {}
    full_text_sources = retrieval.get("full_text_sources", []) if isinstance(retrieval, dict) else []

    pubget_pmids = collect_pubget_coordinate_pmids(run_dir)
    elsevier_pmids: set[str] = set()
    ace_pmids: set[str] = set()
    other_pmids: set[str] = set()

    for source in full_text_sources:
        if not isinstance(source, dict):
            continue
        root_path = resolve_repo_path(source.get("root_path"))
        processed_path = resolve_repo_path(source.get("processed_data_path"))
        templates = source.get("coordinates_path_templates", ["coordinates.json"])
        if not isinstance(templates, list):
            templates = [templates]

        processed_pmids = collect_processed_coordinate_pmids(processed_path)
        folder_pmids = collect_folder_coordinate_pmids(root_path, templates)

        source_text = " ".join(str(source.get(key, "")) for key in ("root_path", "processed_data_path")).lower()
        if "ace" in source_text:
            ace_pmids.update(processed_pmids or folder_pmids)
        elif "elsevier" in source_text:
            elsevier_pmids.update(folder_pmids or processed_pmids)
        else:
            other_pmids.update(processed_pmids | folder_pmids)

    all_pmids = pubget_pmids | elsevier_pmids | ace_pmids | other_pmids
    counts = {
        "pubget_coord_pmids": len(pubget_pmids),
        "elsevier_coord_pmids": len(elsevier_pmids),
        "ace_coord_pmids": len(ace_pmids),
        "other_coord_pmids": len(other_pmids),
        "source_coord_pmids": len(all_pmids),
    }
    return all_pmids, counts


def load_meta_set(project_dir: Path, meta_pmids_path: Path) -> set[str]:
    meta_pmid = resolve_meta_analysis_pmid(str(project_dir), explicit_meta_analysis_pmid=None)
    return set(load_meta_pmids(str(meta_pmids_path), meta_analysis_pmid=meta_pmid))


def load_ft_coords_counts(performance_metrics_path: Path) -> tuple[int | None, int | None]:
    if not performance_metrics_path.exists():
        return None, None
    true_positives = None
    false_positives = None
    for row in read_csv_rows(performance_metrics_path):
        if str(row.get("stage", "")) != "fulltext_with_coords":
            continue
        if str(row.get("metric", "")) == "true_positives":
            true_positives = int(float(row["value"]))
        elif str(row.get("metric", "")) == "false_positives":
            false_positives = int(float(row["value"]))
    if true_positives is None or false_positives is None:
        return true_positives, None
    return true_positives, true_positives + false_positives


def load_stage_precision_counts(performance_metrics_path: Path, stage: str) -> tuple[float | None, int | None, int | None]:
    if not performance_metrics_path.exists():
        return None, None, None
    precision = None
    true_positives = None
    false_positives = None
    for row in read_csv_rows(performance_metrics_path):
        if str(row.get("stage", "")) != stage:
            continue
        metric = str(row.get("metric", ""))
        if metric == "precision":
            precision = float_or_none(row.get("value"))
        elif metric == "true_positives":
            true_positives = int(float(row["value"]))
        elif metric == "false_positives":
            false_positives = int(float(row["value"]))
    denominator = None
    if true_positives is not None and false_positives is not None:
        denominator = true_positives + false_positives
    return precision, true_positives, denominator


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    top_v_rows = read_csv_rows(args.screening_dir / "screening_metrics_top_v.csv")
    top_v_allstudies_rows = [
        {
            **row,
            "condition": "dementia_allstudies",
        }
        for row in read_csv_rows(args.screening_dir / "screening_metrics_top_v_allstudies.csv")
        if str(row.get("project_name", "")).strip() == "dementia"
    ]
    rows: list[dict[str, Any]] = []
    for selected in top_v_rows + top_v_allstudies_rows:
        project = str(selected.get("project_name", "")).strip()
        run_name = str(selected.get("run_name", "")).strip()
        if not project or not run_name:
            continue
        project_dir = args.projects_root / project
        run_dir = project_dir / run_name
        search_pmids = load_search_pmids(run_dir)
        source_coord_pmids, source_counts = collect_source_coordinate_pmids(run_dir)
        search_coord_pmids = search_pmids & source_coord_pmids
        meta_pmids = load_meta_set(project_dir, args.meta_pmids)
        search_coords_tp = search_coord_pmids & meta_pmids
        search_coords_precision = len(search_coords_tp) / len(search_coord_pmids) if search_coord_pmids else math.nan

        performance_metrics_path = args.screening_dir / "evaluations" / project / run_name / "performance_metrics.csv"
        search_precision, search_tp, search_total = load_stage_precision_counts(performance_metrics_path, "search")
        ft_precision, ft_tp, ft_total = load_stage_precision_counts(performance_metrics_path, "fulltext")
        ft_coords_precision = None
        for row in read_csv_rows(performance_metrics_path):
            if str(row.get("stage", "")) == "fulltext_with_coords" and str(row.get("metric", "")) == "precision":
                ft_coords_precision = float_or_none(row.get("value"))
                break
        ft_coords_tp, ft_coords_total = load_ft_coords_counts(performance_metrics_path)

        rows.append(
            {
                "project_name": project,
                "run_name": run_name,
                "condition": str(selected.get("condition", project)).strip() or project,
                "n_search": len(search_pmids),
                "n_manual": len(meta_pmids),
                "search_precision": search_precision if search_precision is not None else math.nan,
                "n_search_true_positive": "" if search_tp is None else search_tp,
                "n_search_precision_denominator": "" if search_total is None else search_total,
                "ft_precision": ft_precision if ft_precision is not None else math.nan,
                "n_ft": "" if ft_total is None else ft_total,
                "n_ft_true_positive": "" if ft_tp is None else ft_tp,
                "n_source_coords": len(source_coord_pmids),
                "n_search_coords": len(search_coord_pmids),
                "n_search_coords_true_positive": len(search_coords_tp),
                "search_coords_precision": search_coords_precision,
                "ft_coords_precision": ft_coords_precision if ft_coords_precision is not None else math.nan,
                "n_ft_coords": "" if ft_coords_total is None else ft_coords_total,
                "n_ft_coords_true_positive": "" if ft_coords_tp is None else ft_coords_tp,
                **source_counts,
            }
        )
    return rows


def plot_rows(rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> list[Path]:
    plot_rows = sorted(
        [
            row
            for row in rows
            if not math.isnan(float(row["search_coords_precision"]))
            and not math.isnan(float(row["ft_coords_precision"]))
        ],
        key=lambda row: (float(row["ft_coords_precision"]), float(row["search_coords_precision"])),
        reverse=True,
    )

    fig = plt.figure(
        figsize=(args.width_px / args.layout_dpi, args.height_px / args.layout_dpi),
        dpi=args.layout_dpi,
        facecolor=POSTER_BG,
    )
    ax = fig.add_axes([0.08, 0.25, 0.88, 0.59])
    xs = list(range(len(plot_rows)))
    search_vals = [float(row["search_coords_precision"]) for row in plot_rows]
    ft_vals = [float(row["ft_coords_precision"]) for row in plot_rows]

    for x, row, search_val, ft_val in zip(xs, plot_rows, search_vals, ft_vals):
        project = str(row["project_name"])
        color = PROJECT_COLORS.get(project, project_color(project))
        ax.plot(
            [x, x],
            [search_val, ft_val],
            color="#A7B1BA",
            linewidth=2.0,
            linestyle="--" if is_dementia_allstudies(row) else "-",
            zorder=1,
        )
        ax.scatter(
            [x],
            [search_val],
            s=78,
            marker="o",
            facecolor=POSTER_PANEL_BG,
            edgecolor=color,
            linewidth=2.0,
            zorder=3,
        )
        ax.scatter(
            [x],
            [ft_val],
            s=78,
            marker="o",
            color=color,
            edgecolor=POSTER_TEXT,
            linewidth=0.7,
            zorder=4,
        )
        ax.text(
            x,
            min(max(search_val, ft_val) + 0.045, 0.98),
            f"n={int(row['n_search_coords'])}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=POSTER_MUTED,
        )

    ax.set_title("Search-Coords vs FT-Coords Precision", fontsize=16, fontweight="bold", pad=10)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.55, len(plot_rows) - 0.45)
    ax.set_xticks(xs)
    ax.set_xticklabels([axis_project_label(display_project_key(row)) for row in plot_rows], fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", color=POSTER_GRID, linewidth=0.9, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(POSTER_BORDER)
    ax.spines["bottom"].set_color(POSTER_BORDER)
    ax.tick_params(axis="y", labelsize=9)

    stage_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=POSTER_PANEL_BG,
            markeredgecolor=MEAN_COLOR,
            markeredgewidth=1.8,
            label="Search-Coords",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=MEAN_COLOR,
            markeredgecolor=POSTER_TEXT,
            label="FT-Coords",
        ),
    ]
    ax.legend(
        handles=stage_handles,
        loc="upper left",
        frameon=False,
        fontsize=9.0,
        labelspacing=0.55,
        handletextpad=0.55,
    )
    fig.text(
        0.08,
        0.11,
        "Search-Coords restricts the Search-stage pool to articles with coordinates in PubGet, Elsevier, or ACE source data.\n"
        "n labels show Search-Coords denominators.",
        ha="left",
        va="center",
        fontsize=8.0,
        color=POSTER_MUTED,
    )
    return save_figure(fig, output_dir, "search_coords_vs_ft_coords_precision", args.dpi)


def plot_search_ft_rows(rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> list[Path]:
    plot_rows = sorted(
        [
            row
            for row in rows
            if not math.isnan(float(row["search_precision"]))
            and not math.isnan(float(row["ft_precision"]))
        ],
        key=lambda row: (float(row["ft_precision"]), float(row["search_precision"])),
        reverse=True,
    )

    fig = plt.figure(
        figsize=(args.width_px / args.layout_dpi, args.height_px / args.layout_dpi),
        dpi=args.layout_dpi,
        facecolor=POSTER_BG,
    )
    ax = fig.add_axes([0.08, 0.25, 0.88, 0.59])
    xs = list(range(len(plot_rows)))
    search_vals = [float(row["search_precision"]) for row in plot_rows]
    ft_vals = [float(row["ft_precision"]) for row in plot_rows]

    for x, row, search_val, ft_val in zip(xs, plot_rows, search_vals, ft_vals):
        project = str(row["project_name"])
        color = PROJECT_COLORS.get(project, project_color(project))
        ax.plot(
            [x, x],
            [search_val, ft_val],
            color="#A7B1BA",
            linewidth=2.0,
            linestyle="--" if is_dementia_allstudies(row) else "-",
            zorder=1,
        )
        ax.scatter(
            [x],
            [search_val],
            s=78,
            marker="o",
            facecolor=POSTER_PANEL_BG,
            edgecolor=color,
            linewidth=2.0,
            zorder=3,
        )
        ax.scatter(
            [x],
            [ft_val],
            s=78,
            marker="o",
            color=color,
            edgecolor=POSTER_TEXT,
            linewidth=0.7,
            zorder=4,
        )
        ax.text(
            x,
            min(max(search_val, ft_val) + 0.045, 0.98),
            f"n={int(row['n_search_precision_denominator'])}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=POSTER_MUTED,
        )

    ax.set_title("Search vs FT Precision", fontsize=16, fontweight="bold", pad=10)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.55, len(plot_rows) - 0.45)
    ax.set_xticks(xs)
    ax.set_xticklabels([axis_project_label(display_project_key(row)) for row in plot_rows], fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", color=POSTER_GRID, linewidth=0.9, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(POSTER_BORDER)
    ax.spines["bottom"].set_color(POSTER_BORDER)
    ax.tick_params(axis="y", labelsize=9)

    stage_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=POSTER_PANEL_BG,
            markeredgecolor=MEAN_COLOR,
            markeredgewidth=1.8,
            label="Search",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=MEAN_COLOR,
            markeredgecolor=POSTER_TEXT,
            label="FT",
        ),
    ]
    ax.legend(
        handles=stage_handles,
        loc="upper left",
        frameon=False,
        fontsize=9.0,
        labelspacing=0.55,
        handletextpad=0.55,
    )
    fig.text(
        0.08,
        0.11,
        "Search is precision across all Search-stage hits; FT is precision after full-text screening.\n"
        "n labels show Search precision denominators.",
        ha="left",
        va="center",
        fontsize=8.0,
        color=POSTER_MUTED,
    )
    return save_figure(fig, output_dir, "search_vs_ft_precision", args.dpi)


def plot_poster_style_rows(rows: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> list[Path]:
    plot_rows = [
        row
        for row in rows
        if not math.isnan(float(row["search_coords_precision"]))
        and not math.isnan(float(row["ft_coords_precision"]))
    ]
    plot_rows.sort(
        key=lambda row: (float(row["ft_coords_precision"]), float(row["search_coords_precision"])),
        reverse=True,
    )
    fig = plt.figure(
        figsize=(1280 / args.layout_dpi, 880 / args.layout_dpi),
        dpi=args.layout_dpi,
        facecolor=POSTER_BG,
    )
    ax = fig.add_axes([0.12, 0.22, 0.76, 0.57])
    stage_x = [0, 1]
    stage_to_values = {"search_coords": [], "ft_coords": []}
    for row in plot_rows:
        project = str(row["project_name"])
        search_val = float(row["search_coords_precision"])
        ft_val = float(row["ft_coords_precision"])
        if not is_dementia_allstudies(row):
            stage_to_values["search_coords"].append(search_val)
            stage_to_values["ft_coords"].append(ft_val)
        ax.plot(
            stage_x,
            [search_val, ft_val],
            marker="o",
            markersize=5.2,
            linewidth=1.7,
            linestyle="--" if is_dementia_allstudies(row) else "-",
            color=PROJECT_COLORS.get(project, project_color(project)),
            alpha=0.92,
            zorder=3,
        )

    mean_vals = [
        sum(stage_to_values["search_coords"]) / len(stage_to_values["search_coords"]),
        sum(stage_to_values["ft_coords"]) / len(stage_to_values["ft_coords"]),
    ]
    ax.plot(stage_x, mean_vals, color=MEAN_COLOR, marker="D", markersize=5.4, linewidth=2.5, zorder=5)

    ax.set_title("Coordinate Precision", pad=10, fontsize=19, fontweight="bold")
    ax.set_ylabel("Score", fontsize=14, labelpad=4)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.12, 1.12)
    ax.set_xticks(stage_x)
    ax.set_xticklabels(["Search-Coords", "FT-Coords"], fontsize=12)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", color=POSTER_GRID, linewidth=1.0, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(POSTER_BORDER)
    ax.spines["bottom"].set_color(POSTER_BORDER)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    return save_figure(fig, output_dir, "search_coords_vs_ft_coords_precision_poster_style", args.dpi)


def main() -> int:
    args = parse_args()
    args.projects_root = args.projects_root.expanduser().resolve()
    args.screening_dir = args.screening_dir.expanduser().resolve()
    args.meta_pmids = args.meta_pmids.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    configure_poster_matplotlib()

    rows = build_rows(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "search_coords_vs_ft_coords_precision.csv"
    search_ft_table_path = output_dir / "search_vs_ft_precision.csv"
    write_csv(
        table_path,
        rows,
        [
            "project_name",
            "run_name",
            "condition",
            "n_search",
            "n_manual",
            "n_source_coords",
            "n_search_coords",
            "n_search_coords_true_positive",
            "search_coords_precision",
            "n_ft_coords",
            "n_ft_coords_true_positive",
            "ft_coords_precision",
            "pubget_coord_pmids",
            "elsevier_coord_pmids",
            "ace_coord_pmids",
            "other_coord_pmids",
        ],
    )
    write_csv(
        search_ft_table_path,
        rows,
        [
            "project_name",
            "run_name",
            "condition",
            "n_search_precision_denominator",
            "n_search_true_positive",
            "search_precision",
            "n_ft",
            "n_ft_true_positive",
            "ft_precision",
        ],
    )

    outputs = [table_path, search_ft_table_path]
    outputs.extend(plot_rows(rows, args, output_dir))
    outputs.extend(plot_search_ft_rows(rows, args, output_dir))
    outputs.extend(plot_poster_style_rows(rows, args, output_dir))
    for path in outputs:
        print(f"[OK] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run screening benchmark comparisons across projects and build a cross-project report."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PROJECTS_ROOT = REPO_ROOT / "projects"
DEFAULT_COMPARE_SCRIPT = SCRIPT_DIR / "compare_screening_to_benchmark.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "cross_project_screening"
DEFAULT_META_PMIDS = REPO_ROOT.parent / "neurometabench" / "data" / "included_studies.csv"

VERSION_RUN_RE = re.compile(r"^v(?P<version>\d+)(?:-.+)?$")
CANONICAL_VERSION_RUN_RE = re.compile(r"^v(?P<version>\d+)$")
ALLSTUDIES_VERSION_RUN_RE = re.compile(r"^v(?P<version>\d+)-allstudies$")
LATEST_VERSION_RUN_RE = re.compile(r"^v(?P<version>\d+)-latest$")
SKIP_SUBSTRING = "annotation-only"
SCREENING_ARTIFACTS = (
    "outputs/search_results.json",
    "outputs/abstract_screening_results.json",
    "outputs/fulltext_screening_results.json",
    "outputs/final_results.json",
)


@dataclass
class RunSelection:
    project_name: str
    run_name: str
    run_dir: Path
    version: int
    status: str
    reason: str


@dataclass
class RunExecutionResult:
    selection: RunSelection
    rerun_status: str
    return_code: int | None
    log_path: Path | None
    evaluation_output_dir: Path
    metrics_stage: str | None
    recall: float | None
    precision: float | None
    f1: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=DEFAULT_PROJECTS_ROOT,
        help="Projects root directory (default: repo/projects).",
    )
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="Project name to include (repeatable). If omitted, include all projects.",
    )
    parser.add_argument(
        "--compare-script",
        type=Path,
        default=DEFAULT_COMPARE_SCRIPT,
        help="Path to compare_screening_to_benchmark.py.",
    )
    parser.add_argument(
        "--meta-pmids",
        type=Path,
        default=DEFAULT_META_PMIDS,
        help=(
            "Path to benchmark PMIDs input (default: ../neurometabench/data/included_studies.csv). "
            "Supports compare_screening_to_benchmark.py formats."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for logs, CSVs, plots, and HTML dashboard.",
    )
    return parser.parse_args()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def project_dirs_from_filters(projects_root: Path, filters: list[str]) -> list[Path]:
    all_dirs = sorted([p for p in projects_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not filters:
        return all_dirs

    requested = set(filters)
    selected = [p for p in all_dirs if p.name in requested]
    missing = sorted(requested - {p.name for p in selected})
    if missing:
        raise ValueError(
            "Unknown --project values: "
            + ", ".join(missing)
            + ". Available: "
            + ", ".join(p.name for p in all_dirs)
        )
    return selected


def has_screening_artifacts(run_dir: Path) -> bool:
    return any((run_dir / rel).exists() for rel in SCREENING_ARTIFACTS)


def discover_run_selections(projects_root: Path, project_filters: list[str]) -> list[RunSelection]:
    selections: list[RunSelection] = []
    for project_dir in project_dirs_from_filters(projects_root, project_filters):
        for child in sorted(project_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue

            match = VERSION_RUN_RE.fullmatch(child.name)
            if not match:
                continue

            version = safe_int(match.group("version"), default=-1)
            lowered = child.name.lower()
            if SKIP_SUBSTRING in lowered:
                selections.append(
                    RunSelection(
                        project_name=project_dir.name,
                        run_name=child.name,
                        run_dir=child,
                        version=version,
                        status="skipped",
                        reason="Excluded annotation-only run.",
                    )
                )
                continue

            if not has_screening_artifacts(child):
                selections.append(
                    RunSelection(
                        project_name=project_dir.name,
                        run_name=child.name,
                        run_dir=child,
                        version=version,
                        status="skipped",
                        reason=(
                            "No screening artifacts found under outputs/ "
                            "(search/abstract/fulltext/final)."
                        ),
                    )
                )
                continue

            selections.append(
                RunSelection(
                    project_name=project_dir.name,
                    run_name=child.name,
                    run_dir=child,
                    version=version,
                    status="selected",
                    reason="Eligible versioned run with screening artifacts.",
                )
            )

    selections.sort(key=lambda s: (s.project_name, s.version, s.run_name))
    return selections


def run_compare_for_selection(
    *,
    selection: RunSelection,
    compare_script: Path,
    meta_pmids: Path,
    output_dir: Path,
) -> tuple[str, int | None, Path, Path]:
    logs_dir = output_dir / "logs"
    eval_dir = output_dir / "evaluations" / selection.project_name / selection.run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"{selection.project_name}__{selection.run_name}.log"

    cmd = [
        sys.executable,
        str(compare_script),
        str(meta_pmids),
        str(selection.run_dir),
        "--output_dir",
        str(eval_dir),
        "--skip-qualitative-report",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    combined_log = (
        f"$ {' '.join(cmd)}\n\n"
        f"exit_code={proc.returncode}\n\n"
        "STDOUT:\n"
        f"{proc.stdout}\n\n"
        "STDERR:\n"
        f"{proc.stderr}\n"
    )
    log_path.write_text(combined_log, encoding="utf-8")

    return ("success" if proc.returncode == 0 else "failed"), proc.returncode, log_path, eval_dir


def _metric_or_none(block: dict[str, Any], key: str) -> float | None:
    value = block.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _compute_f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def is_canonical_version_run(run_name: str) -> bool:
    return CANONICAL_VERSION_RUN_RE.fullmatch(str(run_name)) is not None


def is_allstudies_version_run(run_name: str) -> bool:
    return ALLSTUDIES_VERSION_RUN_RE.fullmatch(str(run_name)) is not None


def is_latest_version_run(run_name: str) -> bool:
    return LATEST_VERSION_RUN_RE.fullmatch(str(run_name).lower()) is not None


def is_recent_run(run_name: str) -> bool:
    return "recent" in str(run_name).lower()


def extract_run_metrics(performance_metrics_path: Path) -> tuple[str | None, float | None, float | None, float | None]:
    if not performance_metrics_path.exists():
        return None, None, None, None

    payload = load_json(performance_metrics_path)
    if not isinstance(payload, dict):
        return None, None, None, None

    # Prefer fulltext screening metrics when present, then abstract, then search.
    if "fulltext" in payload and isinstance(payload["fulltext"], dict):
        metrics = payload["fulltext"].get("metrics", {})
        if isinstance(metrics, dict):
            recall = _metric_or_none(metrics, "recall_all_meta")
            if recall is None:
                recall = _metric_or_none(metrics, "recall_in_search")
            precision = _metric_or_none(metrics, "precision_fulltext_only")
            if precision is None:
                precision = _metric_or_none(metrics, "precision")
            return "fulltext", recall, precision, _compute_f1(precision, recall)

    if "abstract" in payload and isinstance(payload["abstract"], dict):
        metrics = payload["abstract"].get("metrics", {})
        if isinstance(metrics, dict):
            recall = _metric_or_none(metrics, "recall_all_meta")
            if recall is None:
                recall = _metric_or_none(metrics, "recall_in_search")
            precision = _metric_or_none(metrics, "precision")
            return "abstract", recall, precision, _compute_f1(precision, recall)

    if "search" in payload and isinstance(payload["search"], dict):
        metrics = payload["search"].get("metrics", {})
        if isinstance(metrics, dict):
            recall = _metric_or_none(metrics, "recall")
            precision = _metric_or_none(metrics, "precision")
            return "search", recall, precision, _compute_f1(precision, recall)

    return None, None, None, None


def select_top_v_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_project: dict[str, dict[str, Any]] = {}
    for row in rows:
        project = str(row.get("project_name", ""))
        if not project:
            continue
        run_name = str(row.get("run_name", ""))
        if not is_canonical_version_run(run_name):
            continue
        current = best_by_project.get(project)
        if current is None:
            best_by_project[project] = row
            continue
        row_version = safe_int(row.get("version", -1))
        current_version = safe_int(current.get("version", -1))
        row_name = run_name
        current_name = str(current.get("run_name", ""))
        row_f1 = float(row.get("f1", 0.0))
        current_f1 = float(current.get("f1", 0.0))
        row_key = (row_version, row_f1, row_name)
        current_key = (current_version, current_f1, current_name)
        if row_key > current_key:
            best_by_project[project] = row

    return sorted(best_by_project.values(), key=lambda r: str(r.get("project_name", "")))


def select_top_v_allstudies_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_project: dict[str, dict[str, Any]] = {}
    for row in rows:
        project = str(row.get("project_name", ""))
        if not project:
            continue
        run_name = str(row.get("run_name", ""))
        if not is_allstudies_version_run(run_name):
            continue
        current = best_by_project.get(project)
        if current is None:
            best_by_project[project] = row
            continue
        row_version = safe_int(row.get("version", -1))
        current_version = safe_int(current.get("version", -1))
        row_name = run_name
        current_name = str(current.get("run_name", ""))
        row_f1 = float(row.get("f1", 0.0))
        current_f1 = float(current.get("f1", 0.0))
        row_key = (row_version, row_f1, row_name)
        current_key = (current_version, current_f1, current_name)
        if row_key > current_key:
            best_by_project[project] = row

    return sorted(best_by_project.values(), key=lambda r: str(r.get("project_name", "")))


def write_across_runs_plot(output_dir: Path, metric_rows: list[dict[str, Any]]) -> Path | None:
    if not metric_rows:
        return None

    projects = sorted({str(row["project_name"]) for row in metric_rows})
    if not projects:
        return None

    fig_w = max(11.0, 1.3 * len(projects))
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, 13.0), sharex=True)
    plot_specs = [
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("f1", "F1"),
    ]
    cmap = plt.get_cmap("tab20")
    color_for_project = {project: cmap(idx % cmap.N) for idx, project in enumerate(projects)}

    centers = {project: idx + 1 for idx, project in enumerate(projects)}

    for ax, (metric_key, label) in zip(axes, plot_specs):
        for project in projects:
            rows = [row for row in metric_rows if row["project_name"] == project]
            rows.sort(key=lambda r: (safe_int(r.get("version", -1)), str(r.get("run_name", ""))))
            vals = [row.get(metric_key) for row in rows]
            vals = [float(v) for v in vals if v is not None]
            if not vals:
                continue

            n_vals = len(vals)
            center = centers[project]
            if n_vals == 1:
                xs = [center]
            else:
                span = 0.34
                xs = [center - span + (2 * span) * (i / (n_vals - 1)) for i in range(n_vals)]

            project_color = color_for_project[project]
            ax.plot(xs, vals, color=project_color, linewidth=1.5, alpha=0.72, zorder=2)
            ax.scatter(xs, vals, s=42, color=project_color, edgecolor="#111827", linewidth=0.5, zorder=3)

            kept_rows = [row for row in rows if row.get(metric_key) is not None]
            for x, row, y in zip(xs, kept_rows, vals):
                ax.text(
                    x,
                    y,
                    str(row["run_name"]),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    rotation=32,
                    color="#334155",
                )

        metric_values = [float(row[metric_key]) for row in metric_rows if row.get(metric_key) is not None]
        if metric_values:
            mean_val = sum(metric_values) / len(metric_values)
            ax.axhline(
                mean_val,
                color="#9f1239",
                linestyle=":",
                linewidth=2.0,
                alpha=0.9,
                label=f"Mean {label} = {mean_val:.4f}",
            )
            ax.legend(loc="lower right")

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"{label} Across Runs by Project", fontweight="bold")
    handles = [
        plt.Line2D([0], [0], marker="o", color=color_for_project[p], label=p, linewidth=1.5, markersize=5)
        for p in projects
    ]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)

    axes[-1].set_xlim(0.5, len(projects) + 0.5)
    axes[-1].set_xticks([centers[p] for p in projects])
    axes[-1].set_xticklabels(projects)
    axes[-1].set_xlabel("Project")

    fig.tight_layout()
    plot_path = output_dir / "screening_prf_across_runs_by_project.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def write_top_v_plot(
    output_dir: Path,
    top_rows: list[dict[str, Any]],
    *,
    filename: str = "screening_prf_top_v_by_project.png",
    title: str = "Top-V Screening Metrics by Project",
) -> Path | None:
    if not top_rows:
        return None

    ordered = sorted(top_rows, key=lambda r: (float(r.get("f1", 0.0)), str(r.get("project_name", ""))), reverse=True)
    projects = [str(row["project_name"]) for row in ordered]
    recalls = [float(row.get("recall", 0.0)) for row in ordered]
    precisions = [float(row.get("precision", 0.0)) for row in ordered]
    f1s = [float(row.get("f1", 0.0)) for row in ordered]

    x = list(range(len(projects)))
    w = 0.24
    fig_h = max(5.5, 0.38 * len(projects) + 3.0)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))

    ax.bar([i - w for i in x], recalls, width=w, label="Recall", color="#0b7285", alpha=0.9)
    ax.bar(x, precisions, width=w, label="Precision", color="#2b8a3e", alpha=0.9)
    ax.bar([i + w for i in x], f1s, width=w, label="F1", color="#d9480f", alpha=0.9)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Metric")
    ax.set_xlabel("Project")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(projects, rotation=22, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    for i, row in enumerate(ordered):
        ax.text(i, min(float(row.get("f1", 0.0)) + 0.03, 0.985), str(row["run_name"]), ha="center", va="bottom", fontsize=8, color="#334155")

    fig.tight_layout()
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def build_top_v_vs_allstudies_pairs(
    top_v_rows: list[dict[str, Any]],
    top_v_allstudies_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical_by_project = {str(row["project_name"]): row for row in top_v_rows}
    allstudies_by_project = {str(row["project_name"]): row for row in top_v_allstudies_rows}

    pairs: list[dict[str, Any]] = []
    for project in sorted(set(canonical_by_project) & set(allstudies_by_project)):
        canonical = canonical_by_project[project]
        allstudies = allstudies_by_project[project]
        canonical_v = safe_int(canonical.get("version", -1))
        allstudies_v = safe_int(allstudies.get("version", -1))
        if canonical_v != allstudies_v:
            continue

        pairs.append(
            {
                "project_name": project,
                "version": canonical_v,
                "canonical_run_name": str(canonical.get("run_name", "")),
                "allstudies_run_name": str(allstudies.get("run_name", "")),
                "canonical_recall": float(canonical.get("recall", 0.0)),
                "canonical_precision": float(canonical.get("precision", 0.0)),
                "canonical_f1": float(canonical.get("f1", 0.0)),
                "allstudies_recall": float(allstudies.get("recall", 0.0)),
                "allstudies_precision": float(allstudies.get("precision", 0.0)),
                "allstudies_f1": float(allstudies.get("f1", 0.0)),
            }
        )

    return pairs


def write_top_v_vs_allstudies_comparison_plot(
    output_dir: Path,
    pairs: list[dict[str, Any]],
) -> Path | None:
    if not pairs:
        return None

    ordered = sorted(
        pairs,
        key=lambda row: (
            float(row.get("allstudies_f1", 0.0) - row.get("canonical_f1", 0.0)),
            str(row.get("project_name", "")),
        ),
        reverse=True,
    )
    metric_keys = ["recall", "precision", "f1"]
    metric_labels = ["Recall", "Precision", "F1"]
    metric_colors = ["#0b7285", "#2b8a3e", "#d9480f"]

    # Compact small-N layout: one panel per project with grouped metric bars.
    if len(ordered) <= 2:
        n = len(ordered)
        fig, axes_arr = plt.subplots(1, n, figsize=(6.6 * n, 4.8), sharey=True)
        axes = [axes_arr] if n == 1 else list(axes_arr)
        x = list(range(len(metric_keys)))
        w = 0.34

        for ax, row in zip(axes, ordered):
            canonical_vals = [float(row[f"canonical_{m}"]) for m in metric_keys]
            allstudies_vals = [float(row[f"allstudies_{m}"]) for m in metric_keys]
            bars_c = ax.bar(
                [i - (w / 2) for i in x],
                canonical_vals,
                width=w,
                color=metric_colors,
                alpha=0.55,
                label="vN",
            )
            bars_a = ax.bar(
                [i + (w / 2) for i in x],
                allstudies_vals,
                width=w,
                color=metric_colors,
                alpha=0.92,
                label="vN-allstudies",
            )
            for bc, ba in zip(bars_c, bars_a):
                x1 = bc.get_x() + bc.get_width() / 2
                x2 = ba.get_x() + ba.get_width() / 2
                y1 = bc.get_height()
                y2 = ba.get_height()
                ax.plot([x1, x2], [y1, y2], color="#334155", linewidth=1.0, alpha=0.65)

            ax.set_xticks(x)
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(f"{row['project_name']} (v{int(row['version'])})", fontweight="bold")
        axes[0].set_ylabel("Metric")
        axes[0].legend(loc="upper left")
        fig.suptitle("Comparable Runs: Top vN vs Top vN-Allstudies", fontweight="bold")
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    else:
        projects = [str(row["project_name"]) for row in ordered]
        x_labels = [f"{row['project_name']} (v{int(row['version'])})" for row in ordered]
        x = list(range(len(projects)))
        w = 0.32

        fig_h = max(6.0, 0.42 * len(projects) + 3.0)
        fig, axes = plt.subplots(3, 1, figsize=(13.5, fig_h), sharex=True)
        metric_specs = [
            ("recall", "Recall", "#0b7285"),
            ("precision", "Precision", "#2b8a3e"),
            ("f1", "F1", "#d9480f"),
        ]

        for ax, (metric, label, color) in zip(axes, metric_specs):
            canonical_vals = [float(row[f"canonical_{metric}"]) for row in ordered]
            allstudies_vals = [float(row[f"allstudies_{metric}"]) for row in ordered]

            ax.bar([i - (w / 2) for i in x], canonical_vals, width=w, label="vN", color=color, alpha=0.55)
            ax.bar([i + (w / 2) for i in x], allstudies_vals, width=w, label="vN-allstudies", color=color, alpha=0.9)

            for i, (c_val, a_val) in enumerate(zip(canonical_vals, allstudies_vals)):
                ax.plot([i - (w / 2), i + (w / 2)], [c_val, a_val], color="#334155", linewidth=1.0, alpha=0.65)

            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.25)
            ax.set_title(f"Top vN vs Top vN-allstudies: {label}", fontweight="bold")

        axes[0].legend(loc="upper right")
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(x_labels, rotation=22, ha="right")
        axes[-1].set_xlabel("Comparable Runs by Project")

    fig.tight_layout()
    plot_path = output_dir / "screening_prf_top_v_vs_top_v_allstudies_comparison.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def extract_stage_metrics(performance_metrics_path: Path) -> dict[str, dict[str, float] | None]:
    stage_rows: dict[str, dict[str, float] | None] = {
        "search": None,
        "abstract": None,
        "fulltext": None,
    }
    if not performance_metrics_path.exists():
        return stage_rows

    payload = load_json(performance_metrics_path)
    if not isinstance(payload, dict):
        return stage_rows

    search_block = payload.get("search")
    if isinstance(search_block, dict):
        metrics = search_block.get("metrics", {})
        if isinstance(metrics, dict):
            recall = _metric_or_none(metrics, "recall")
            precision = _metric_or_none(metrics, "precision")
            f1 = _compute_f1(precision, recall)
            if recall is not None and precision is not None and f1 is not None:
                stage_rows["search"] = {"recall": recall, "precision": precision, "f1": f1}

    abstract_block = payload.get("abstract")
    if isinstance(abstract_block, dict):
        metrics = abstract_block.get("metrics", {})
        if isinstance(metrics, dict):
            # For stage-progression plots, use abstract recall in-search.
            recall = _metric_or_none(metrics, "recall_in_search")
            if recall is None:
                recall = _metric_or_none(metrics, "recall_all_meta")
            precision = _metric_or_none(metrics, "precision")
            f1 = _compute_f1(precision, recall)
            if recall is not None and precision is not None and f1 is not None:
                stage_rows["abstract"] = {"recall": recall, "precision": precision, "f1": f1}

    fulltext_block = payload.get("fulltext")
    if isinstance(fulltext_block, dict):
        metrics = fulltext_block.get("metrics", {})
        if isinstance(metrics, dict):
            # For stage-progression plots, use "Recall (full-text)" specifically.
            recall = _metric_or_none(metrics, "recall_fulltext_only")
            if recall is None:
                recall = _metric_or_none(metrics, "recall_in_search")
            if recall is None:
                recall = _metric_or_none(metrics, "recall_all_meta")
            precision = _metric_or_none(metrics, "precision_fulltext_only")
            if precision is None:
                precision = _metric_or_none(metrics, "precision")
            f1 = _compute_f1(precision, recall)
            if recall is not None and precision is not None and f1 is not None:
                stage_rows["fulltext"] = {"recall": recall, "precision": precision, "f1": f1}

    return stage_rows


def write_top_v_stage_progression_plot(
    output_dir: Path,
    stage_rows: list[dict[str, Any]],
    *,
    filename: str = "screening_prf_top_v_stage_progression.png",
    title_prefix: str = "Top Canonical V Run",
    layout: str = "vertical",
    add_mean_across_projects: bool = False,
    annotate_run_names: bool = True,
    legend_position: str = "right",
) -> Path | None:
    if not stage_rows:
        return None

    stage_order = ["search", "abstract", "fulltext"]
    stage_label = {"search": "Search", "abstract": "Abstract", "fulltext": "Fulltext"}
    x_lookup = {stage: idx for idx, stage in enumerate(stage_order)}

    projects = sorted({str(row["project_name"]) for row in stage_rows})
    if not projects:
        return None

    if layout == "horizontal":
        fig_w = max(13.5, 1.1 * len(projects) + 8.0)
        fig, axes_arr = plt.subplots(1, 3, figsize=(fig_w, 4.6), sharey=True)
        axes = list(axes_arr)
    else:
        fig_w = max(12.0, 1.2 * len(projects))
        fig, axes_arr = plt.subplots(3, 1, figsize=(fig_w, 12.5), sharex=True)
        axes = list(axes_arr)
    metric_specs = [
        ("recall", "Recall", "#0b7285"),
        ("precision", "Precision", "#2b8a3e"),
        ("f1", "F1", "#d9480f"),
    ]
    cmap = plt.get_cmap("tab20")
    color_for_project = {project: cmap(idx % cmap.N) for idx, project in enumerate(projects)}

    for ax, (metric_key, metric_label, metric_color) in zip(axes, metric_specs):
        del metric_color
        stage_to_values: dict[str, list[float]] = {stage: [] for stage in stage_order}
        for project in projects:
            rows = [row for row in stage_rows if str(row.get("project_name")) == project]
            rows.sort(key=lambda row: x_lookup.get(str(row.get("stage")), -1))
            xs = []
            ys = []
            for row in rows:
                if row.get("metric") != metric_key:
                    continue
                if row.get("value") is None:
                    continue
                stage_name = str(row["stage"])
                value = float(row["value"])
                xs.append(x_lookup[stage_name])
                ys.append(value)
                if stage_name in stage_to_values:
                    stage_to_values[stage_name].append(value)

            if not ys:
                continue

            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                markersize=5.5,
                color=color_for_project[project],
                alpha=0.85,
                label=project,
            )

            top_row = next((row for row in rows if row.get("metric") == metric_key and row.get("stage") == "fulltext"), None)
            if annotate_run_names and top_row is not None and top_row.get("value") is not None:
                run_name = str(top_row.get("run_name", ""))
                ax.text(
                    x_lookup["fulltext"] + 0.03,
                    float(top_row["value"]),
                    run_name,
                    fontsize=7,
                    va="center",
                    ha="left",
                    color="#334155",
                )

        if add_mean_across_projects:
            mean_xs: list[int] = []
            mean_ys: list[float] = []
            for stage in stage_order:
                vals = stage_to_values.get(stage, [])
                if not vals:
                    continue
                mean_xs.append(x_lookup[stage])
                mean_ys.append(float(sum(vals) / len(vals)))
            if mean_ys:
                ax.plot(
                    mean_xs,
                    mean_ys,
                    color="#000000",
                    linewidth=2.2,
                    linestyle="-",
                    marker="D",
                    markersize=4.8,
                    label="Mean (all projects)",
                    zorder=5,
                )

        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric_label)
        ax.set_title(f"{title_prefix}: {metric_label} Across Stages", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    if layout == "horizontal":
        for ax in axes:
            ax.set_xticks([x_lookup[s] for s in stage_order])
            ax.set_xticklabels([stage_label[s] for s in stage_order])
            ax.set_xlabel("Screening Stage")
        if legend_position == "bottom":
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.02),
                    ncol=min(len(labels), 6),
                    fontsize=8,
                    frameon=False,
                )
        else:
            axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)
    else:
        axes[-1].set_xticks([x_lookup[s] for s in stage_order])
        axes[-1].set_xticklabels([stage_label[s] for s in stage_order])
        axes[-1].set_xlabel("Screening Stage")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)

    fig.tight_layout()
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def order_results_top_first(
    results: list[RunExecutionResult],
    top_run_keys: set[tuple[str, str]],
    top_allstudies_run_keys: set[tuple[str, str]],
) -> list[RunExecutionResult]:
    return sorted(
        results,
        key=lambda r: (
            0 if (r.selection.project_name, r.selection.run_name) in top_run_keys else 1,
            0 if (r.selection.project_name, r.selection.run_name) in top_allstudies_run_keys else 1,
            r.selection.project_name,
            r.selection.version,
            r.selection.run_name,
        ),
    )


def build_html_report(
    *,
    generated_at_utc: str,
    selections: list[RunSelection],
    results: list[RunExecutionResult],
    output_dir: Path,
    across_runs_plot_path: Path | None,
    top_v_plot_path: Path | None,
    top_v_stage_progression_plot_path: Path | None,
    top_v_allstudies_plot_path: Path | None,
    top_v_allstudies_stage_progression_plot_path: Path | None,
    top_v_vs_allstudies_comparison_plot_path: Path | None,
    top_run_keys: set[tuple[str, str]],
    top_allstudies_run_keys: set[tuple[str, str]],
) -> str:
    selected_count = len([s for s in selections if s.status == "selected"])
    skipped_count = len([s for s in selections if s.status != "selected"])
    success_count = len([r for r in results if r.rerun_status == "success"])
    failed_count = len([r for r in results if r.rerun_status == "failed"])

    row_html: list[str] = []
    for result in order_results_top_first(results, top_run_keys, top_allstudies_run_keys):
        sel = result.selection
        log_rel = ""
        if result.log_path is not None and result.log_path.exists():
            log_rel = os.path.relpath(result.log_path.resolve(), output_dir.resolve())
        log_cell = f'<a href="{escape(log_rel)}">log</a>' if log_rel else ""
        is_top = (sel.project_name, sel.run_name) in top_run_keys
        is_top_allstudies = (sel.project_name, sel.run_name) in top_allstudies_run_keys

        row_html.append(
            "<tr>"
            f"<td>{escape(sel.project_name)}</td>"
            f"<td>{escape(sel.run_name)}</td>"
            f"<td>{sel.version}</td>"
            f"<td>{escape(sel.status)}</td>"
            f"<td>{escape(sel.reason)}</td>"
            f"<td>{escape(result.rerun_status)}</td>"
            f"<td>{'' if result.return_code is None else result.return_code}</td>"
            f"<td>{escape(result.metrics_stage or '')}</td>"
            f"<td>{'' if result.recall is None else f'{result.recall:.4f}'}</td>"
            f"<td>{'' if result.precision is None else f'{result.precision:.4f}'}</td>"
            f"<td>{'' if result.f1 is None else f'{result.f1:.4f}'}</td>"
            f"<td>{'yes' if is_top else ''}</td>"
            f"<td>{'yes' if is_top_allstudies else ''}</td>"
            f"<td>{log_cell}</td>"
            "</tr>"
        )

    skipped_items = [
        f"<li><code>{escape(s.project_name)}/{escape(s.run_name)}</code>: {escape(s.reason)}</li>"
        for s in sorted(selections, key=lambda x: (x.project_name, x.version, x.run_name))
        if s.status != "selected"
    ]

    failed_items = [
        f"<li><code>{escape(r.selection.project_name)}/{escape(r.selection.run_name)}</code>: return_code={r.return_code}</li>"
        for r in sorted(results, key=lambda x: (x.selection.project_name, x.selection.version, x.selection.run_name))
        if r.rerun_status == "failed"
    ]

    sections: list[str] = []
    if top_v_plot_path is not None and top_v_plot_path.exists():
        rel = os.path.relpath(top_v_plot_path.resolve(), output_dir.resolve())
        sections.append(
            "<section>"
            "<h2>Top-V PRF by Project</h2>"
            f"<img src=\"{escape(rel)}\" alt=\"Top-V PRF plot\" style=\"max-width:100%; height:auto; border:1px solid #d7dee6; border-radius:8px;\">"
            "</section>"
        )
    if top_v_stage_progression_plot_path is not None and top_v_stage_progression_plot_path.exists():
        rel = os.path.relpath(top_v_stage_progression_plot_path.resolve(), output_dir.resolve())
        sections.append(
            "<section>"
            "<h2>Top Canonical V Run: PRF Stage Progression</h2>"
            f"<img src=\"{escape(rel)}\" alt=\"Top canonical V run stage progression PRF plot\" style=\"max-width:100%; height:auto; border:1px solid #d7dee6; border-radius:8px;\">"
            "</section>"
        )
    if across_runs_plot_path is not None and across_runs_plot_path.exists():
        rel = os.path.relpath(across_runs_plot_path.resolve(), output_dir.resolve())
        sections.append(
            "<section>"
            "<h2>Across-Runs PRF by Project</h2>"
            f"<img src=\"{escape(rel)}\" alt=\"Across-runs PRF plot\" style=\"max-width:100%; height:auto; border:1px solid #d7dee6; border-radius:8px;\">"
            "</section>"
        )
    if top_v_allstudies_plot_path is not None and top_v_allstudies_plot_path.exists():
        rel = os.path.relpath(top_v_allstudies_plot_path.resolve(), output_dir.resolve())
        sections.append(
            "<section>"
            "<h2>Top V-Allstudies PRF by Project</h2>"
            f"<img src=\"{escape(rel)}\" alt=\"Top V-allstudies PRF plot\" style=\"max-width:100%; height:auto; border:1px solid #d7dee6; border-radius:8px;\">"
            "</section>"
        )
    if (
        top_v_allstudies_stage_progression_plot_path is not None
        and top_v_allstudies_stage_progression_plot_path.exists()
    ):
        rel = os.path.relpath(top_v_allstudies_stage_progression_plot_path.resolve(), output_dir.resolve())
        sections.append(
            "<section>"
            "<h2>Top V-Allstudies Run: PRF Stage Progression</h2>"
            f"<img src=\"{escape(rel)}\" alt=\"Top V-allstudies run stage progression PRF plot\" style=\"max-width:100%; height:auto; border:1px solid #d7dee6; border-radius:8px;\">"
            "</section>"
        )
    if (
        top_v_vs_allstudies_comparison_plot_path is not None
        and top_v_vs_allstudies_comparison_plot_path.exists()
    ):
        rel = os.path.relpath(top_v_vs_allstudies_comparison_plot_path.resolve(), output_dir.resolve())
        sections.append(
            "<section>"
            "<h2>Comparable Runs: Top vN vs Top vN-Allstudies</h2>"
            f"<img src=\"{escape(rel)}\" alt=\"Comparable runs top vN vs top vN-allstudies PRF plot\" style=\"max-width:100%; height:auto; border:1px solid #d7dee6; border-radius:8px;\">"
            "</section>"
        )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Cross-Project Screening Report</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #1f2933;
      --line: #d7dee6;
      --accent: #0f766e;
    }}
    body {{ margin: 0; padding: 1.1rem; background: var(--bg); color: var(--ink); font-family: \"IBM Plex Sans\", \"Segoe UI\", sans-serif; }}
    header, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }}
    .kpis {{ display: grid; gap: 0.8rem; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); margin-top: 0.6rem; }}
    .kpi {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfdff; padding: 0.65rem; }}
    .kpi .label {{ color: #4a5b70; font-size: 0.83rem; }}
    .kpi .value {{ color: var(--accent); font-size: 1.3rem; font-weight: 700; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border: 1px solid var(--line); padding: 0.4rem; text-align: left; vertical-align: top; }}
    th {{ background: #e9eff6; }}
    code {{ background: #eef2f7; border: 1px solid #d7dee7; border-radius: 4px; padding: 0.05rem 0.25rem; }}
    ul {{ margin: 0.4rem 0 0 1.1rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Cross-Project Screening Report</h1>
    <p>Generated at {escape(generated_at_utc)} by re-running <code>compare_screening_to_benchmark.py</code> on all version runs except annotation-only runs. Includes top-per-project canonical <code>vN</code> runs and top-per-project <code>vN-allstudies</code> runs.</p>
    <div class=\"kpis\">
      <div class=\"kpi\"><div class=\"label\">Runs Selected</div><div class=\"value\">{selected_count}</div></div>
      <div class=\"kpi\"><div class=\"label\">Runs Skipped</div><div class=\"value\">{skipped_count}</div></div>
      <div class=\"kpi\"><div class=\"label\">Compare Success</div><div class=\"value\">{success_count}</div></div>
      <div class=\"kpi\"><div class=\"label\">Compare Failed</div><div class=\"value\">{failed_count}</div></div>
    </div>
  </header>

  <section>
    <h2>Run-Level Results</h2>
    <div class=\"table-wrap\">
      <table>
        <thead>
          <tr>
            <th>Project</th><th>Run</th><th>V</th><th>Selection</th><th>Reason</th>
            <th>Re-run</th><th>Code</th><th>Metric Stage</th><th>Recall</th><th>Precision</th><th>F1</th><th>Top Canonical V</th><th>Top V-Allstudies</th><th>Log</th>
          </tr>
        </thead>
        <tbody>
          {''.join(row_html)}
        </tbody>
      </table>
    </div>
    <p>CSVs: <code>run_selection.csv</code>, <code>screening_metrics_by_run.csv</code>, <code>screening_metrics_top_v.csv</code>, <code>screening_metrics_top_v_stage_progression.csv</code>, <code>screening_metrics_top_v_allstudies.csv</code>, <code>screening_metrics_top_v_allstudies_stage_progression.csv</code>, <code>screening_metrics_top_v_vs_top_v_allstudies_pairs.csv</code></p>
  </section>

  {''.join(sections)}

  <section>
    <h2>Skipped Runs</h2>
    {('<ul>' + ''.join(skipped_items) + '</ul>') if skipped_items else '<p>None.</p>'}
  </section>

  <section>
    <h2>Failed Re-runs</h2>
    {('<ul>' + ''.join(failed_items) + '</ul>') if failed_items else '<p>None.</p>'}
  </section>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    projects_root = args.projects_root.expanduser().resolve()
    compare_script = args.compare_script.expanduser().resolve()
    meta_pmids = args.meta_pmids.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not projects_root.exists() or not projects_root.is_dir():
        raise FileNotFoundError(f"Projects root not found: {projects_root}")
    if not compare_script.exists():
        raise FileNotFoundError(f"Compare script not found: {compare_script}")
    if not meta_pmids.exists():
        raise FileNotFoundError(f"Meta PMIDs input not found: {meta_pmids}")

    selections = discover_run_selections(projects_root, args.project)
    print(f"Discovered {len(selections)} versioned runs under {projects_root}")

    results: list[RunExecutionResult] = []
    for selection in selections:
        if selection.status != "selected":
            results.append(
                RunExecutionResult(
                    selection=selection,
                    rerun_status="skipped",
                    return_code=None,
                    log_path=None,
                    evaluation_output_dir=output_dir / "evaluations" / selection.project_name / selection.run_name,
                    metrics_stage=None,
                    recall=None,
                    precision=None,
                    f1=None,
                )
            )
            continue

        print(f"[RUN] {selection.project_name}/{selection.run_name}")
        rerun_status, return_code, log_path, eval_dir = run_compare_for_selection(
            selection=selection,
            compare_script=compare_script,
            meta_pmids=meta_pmids,
            output_dir=output_dir,
        )
        perf_path = eval_dir / "performance_metrics.json"
        stage, recall, precision, f1 = extract_run_metrics(perf_path)

        print(
            f"[{rerun_status.upper()}] {selection.project_name}/{selection.run_name} "
            f"code={return_code} stage={stage or '-'} "
            f"R={'' if recall is None else f'{recall:.3f}'} "
            f"P={'' if precision is None else f'{precision:.3f}'} "
            f"F1={'' if f1 is None else f'{f1:.3f}'}"
        )

        results.append(
            RunExecutionResult(
                selection=selection,
                rerun_status=rerun_status,
                return_code=return_code,
                log_path=log_path,
                evaluation_output_dir=eval_dir,
                metrics_stage=stage,
                recall=recall,
                precision=precision,
                f1=f1,
            )
        )

    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for result in sorted(results, key=lambda r: (r.selection.project_name, r.selection.version, r.selection.run_name)):
        sel = result.selection
        selection_rows.append(
            {
                "project_name": sel.project_name,
                "run_name": sel.run_name,
                "version": sel.version,
                "selection_status": sel.status,
                "selection_reason": sel.reason,
                "rerun_status": result.rerun_status,
                "return_code": "" if result.return_code is None else result.return_code,
                "metrics_stage": result.metrics_stage or "",
                "recall": "" if result.recall is None else f"{result.recall:.6f}",
                "precision": "" if result.precision is None else f"{result.precision:.6f}",
                "f1": "" if result.f1 is None else f"{result.f1:.6f}",
                "run_dir": str(sel.run_dir),
                "evaluation_output_dir": str(result.evaluation_output_dir),
                "log_path": str(result.log_path) if result.log_path is not None else "",
            }
        )

        if result.recall is None or result.precision is None or result.f1 is None:
            continue

        metric_rows.append(
            {
                "project_name": sel.project_name,
                "run_name": sel.run_name,
                "version": sel.version,
                "stage": result.metrics_stage or "",
                "recall": float(result.recall),
                "precision": float(result.precision),
                "f1": float(result.f1),
            }
        )

    top_v_rows = select_top_v_rows(metric_rows)
    top_v_allstudies_rows = select_top_v_allstudies_rows(metric_rows)
    top_run_keys = {(str(row["project_name"]), str(row["run_name"])) for row in top_v_rows}
    top_allstudies_run_keys = {
        (str(row["project_name"]), str(row["run_name"]))
        for row in top_v_allstudies_rows
    }
    ordered_results = order_results_top_first(results, top_run_keys, top_allstudies_run_keys)

    across_runs_rows = [
        row
        for row in metric_rows
        if not is_latest_version_run(str(row.get("run_name", "")))
        and not is_recent_run(str(row.get("run_name", "")))
        and not is_allstudies_version_run(str(row.get("run_name", "")))
    ]
    across_runs_plot_path = write_across_runs_plot(output_dir, across_runs_rows)
    top_v_plot_path = write_top_v_plot(output_dir, top_v_rows)
    top_v_allstudies_plot_path = write_top_v_plot(
        output_dir,
        top_v_allstudies_rows,
        filename="screening_prf_top_v_allstudies_by_project.png",
        title="Top V-Allstudies Screening Metrics by Project",
    )
    top_v_vs_allstudies_pairs = build_top_v_vs_allstudies_pairs(
        top_v_rows=top_v_rows,
        top_v_allstudies_rows=top_v_allstudies_rows,
    )
    top_v_vs_allstudies_comparison_plot_path = write_top_v_vs_allstudies_comparison_plot(
        output_dir,
        top_v_vs_allstudies_pairs,
    )

    top_stage_rows: list[dict[str, Any]] = []
    result_by_key = {
        (r.selection.project_name, r.selection.run_name): r
        for r in results
    }
    for row in top_v_rows:
        key = (str(row["project_name"]), str(row["run_name"]))
        result = result_by_key.get(key)
        if result is None:
            continue
        perf_path = result.evaluation_output_dir / "performance_metrics.json"
        stage_metrics = extract_stage_metrics(perf_path)
        for stage in ("search", "abstract", "fulltext"):
            metrics = stage_metrics.get(stage)
            if not metrics:
                continue
            for metric in ("recall", "precision", "f1"):
                top_stage_rows.append(
                    {
                        "project_name": row["project_name"],
                        "run_name": row["run_name"],
                        "version": row["version"],
                        "stage": stage,
                        "metric": metric,
                        "value": float(metrics[metric]),
                    }
                )

    top_v_stage_progression_plot_path = write_top_v_stage_progression_plot(
        output_dir,
        top_stage_rows,
        filename="screening_prf_top_v_stage_progression.png",
        title_prefix="Top Canonical V Run",
        layout="horizontal",
        add_mean_across_projects=True,
        annotate_run_names=False,
        legend_position="bottom",
    )

    top_allstudies_stage_rows: list[dict[str, Any]] = []
    for row in top_v_allstudies_rows:
        key = (str(row["project_name"]), str(row["run_name"]))
        result = result_by_key.get(key)
        if result is None:
            continue
        perf_path = result.evaluation_output_dir / "performance_metrics.json"
        stage_metrics = extract_stage_metrics(perf_path)
        for stage in ("search", "abstract", "fulltext"):
            metrics = stage_metrics.get(stage)
            if not metrics:
                continue
            for metric in ("recall", "precision", "f1"):
                top_allstudies_stage_rows.append(
                    {
                        "project_name": row["project_name"],
                        "run_name": row["run_name"],
                        "version": row["version"],
                        "stage": stage,
                        "metric": metric,
                        "value": float(metrics[metric]),
                    }
                )

    top_v_allstudies_stage_progression_plot_path = write_top_v_stage_progression_plot(
        output_dir,
        top_allstudies_stage_rows,
        filename="screening_prf_top_v_allstudies_stage_progression.png",
        title_prefix="Top V-Allstudies Run",
    )

    write_csv(
        output_dir / "run_selection.csv",
        [
            {
                **row,
                "is_top_canonical_v_run": "yes"
                if (str(row["project_name"]), str(row["run_name"])) in top_run_keys
                else "",
                "is_top_v_allstudies_run": "yes"
                if (str(row["project_name"]), str(row["run_name"])) in top_allstudies_run_keys
                else "",
            }
            for row in sorted(
                selection_rows,
                key=lambda row: (
                    0 if (str(row["project_name"]), str(row["run_name"])) in top_run_keys else 1,
                    0
                    if (str(row["project_name"]), str(row["run_name"])) in top_allstudies_run_keys
                    else 1,
                    str(row["project_name"]),
                    safe_int(row["version"]),
                    str(row["run_name"]),
                ),
            )
        ],
        [
            "project_name",
            "run_name",
            "version",
            "is_top_canonical_v_run",
            "is_top_v_allstudies_run",
            "selection_status",
            "selection_reason",
            "rerun_status",
            "return_code",
            "metrics_stage",
            "recall",
            "precision",
            "f1",
            "run_dir",
            "evaluation_output_dir",
            "log_path",
        ],
    )

    write_csv(
        output_dir / "screening_metrics_by_run.csv",
        [
            {
                "project_name": row["project_name"],
                "run_name": row["run_name"],
                "version": row["version"],
                "stage": row["stage"],
                "recall": f"{float(row['recall']):.6f}",
                "precision": f"{float(row['precision']):.6f}",
                "f1": f"{float(row['f1']):.6f}",
            }
            for row in sorted(metric_rows, key=lambda r: (r["project_name"], safe_int(r["version"]), r["run_name"]))
        ],
        ["project_name", "run_name", "version", "stage", "recall", "precision", "f1"],
    )

    write_csv(
        output_dir / "screening_metrics_top_v.csv",
        [
            {
                "project_name": row["project_name"],
                "run_name": row["run_name"],
                "version": row["version"],
                "stage": row["stage"],
                "recall": f"{float(row['recall']):.6f}",
                "precision": f"{float(row['precision']):.6f}",
                "f1": f"{float(row['f1']):.6f}",
            }
            for row in top_v_rows
        ],
        ["project_name", "run_name", "version", "stage", "recall", "precision", "f1"],
    )
    write_csv(
        output_dir / "screening_metrics_top_v_stage_progression.csv",
        [
            {
                "project_name": row["project_name"],
                "run_name": row["run_name"],
                "version": row["version"],
                "stage": row["stage"],
                "metric": row["metric"],
                "value": f"{float(row['value']):.6f}",
            }
            for row in sorted(
                top_stage_rows,
                key=lambda r: (
                    str(r["project_name"]),
                    safe_int(r["version"]),
                    str(r["run_name"]),
                    {"search": 0, "abstract": 1, "fulltext": 2}.get(str(r["stage"]), 99),
                    {"recall": 0, "precision": 1, "f1": 2}.get(str(r["metric"]), 99),
                ),
            )
        ],
        ["project_name", "run_name", "version", "stage", "metric", "value"],
    )
    write_csv(
        output_dir / "screening_metrics_top_v_allstudies.csv",
        [
            {
                "project_name": row["project_name"],
                "run_name": row["run_name"],
                "version": row["version"],
                "stage": row["stage"],
                "recall": f"{float(row['recall']):.6f}",
                "precision": f"{float(row['precision']):.6f}",
                "f1": f"{float(row['f1']):.6f}",
            }
            for row in top_v_allstudies_rows
        ],
        ["project_name", "run_name", "version", "stage", "recall", "precision", "f1"],
    )
    write_csv(
        output_dir / "screening_metrics_top_v_allstudies_stage_progression.csv",
        [
            {
                "project_name": row["project_name"],
                "run_name": row["run_name"],
                "version": row["version"],
                "stage": row["stage"],
                "metric": row["metric"],
                "value": f"{float(row['value']):.6f}",
            }
            for row in sorted(
                top_allstudies_stage_rows,
                key=lambda r: (
                    str(r["project_name"]),
                    safe_int(r["version"]),
                    str(r["run_name"]),
                    {"search": 0, "abstract": 1, "fulltext": 2}.get(str(r["stage"]), 99),
                    {"recall": 0, "precision": 1, "f1": 2}.get(str(r["metric"]), 99),
                ),
            )
        ],
        ["project_name", "run_name", "version", "stage", "metric", "value"],
    )
    write_csv(
        output_dir / "screening_metrics_top_v_vs_top_v_allstudies_pairs.csv",
        [
            {
                "project_name": row["project_name"],
                "version": row["version"],
                "canonical_run_name": row["canonical_run_name"],
                "allstudies_run_name": row["allstudies_run_name"],
                "canonical_recall": f"{float(row['canonical_recall']):.6f}",
                "canonical_precision": f"{float(row['canonical_precision']):.6f}",
                "canonical_f1": f"{float(row['canonical_f1']):.6f}",
                "allstudies_recall": f"{float(row['allstudies_recall']):.6f}",
                "allstudies_precision": f"{float(row['allstudies_precision']):.6f}",
                "allstudies_f1": f"{float(row['allstudies_f1']):.6f}",
                "delta_recall_allstudies_minus_canonical": f"{float(row['allstudies_recall'] - row['canonical_recall']):.6f}",
                "delta_precision_allstudies_minus_canonical": f"{float(row['allstudies_precision'] - row['canonical_precision']):.6f}",
                "delta_f1_allstudies_minus_canonical": f"{float(row['allstudies_f1'] - row['canonical_f1']):.6f}",
            }
            for row in top_v_vs_allstudies_pairs
        ],
        [
            "project_name",
            "version",
            "canonical_run_name",
            "allstudies_run_name",
            "canonical_recall",
            "canonical_precision",
            "canonical_f1",
            "allstudies_recall",
            "allstudies_precision",
            "allstudies_f1",
            "delta_recall_allstudies_minus_canonical",
            "delta_precision_allstudies_minus_canonical",
            "delta_f1_allstudies_minus_canonical",
        ],
    )

    html = build_html_report(
        generated_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        selections=selections,
        results=ordered_results,
        output_dir=output_dir,
        across_runs_plot_path=across_runs_plot_path,
        top_v_plot_path=top_v_plot_path,
        top_v_stage_progression_plot_path=top_v_stage_progression_plot_path,
        top_v_allstudies_plot_path=top_v_allstudies_plot_path,
        top_v_allstudies_stage_progression_plot_path=top_v_allstudies_stage_progression_plot_path,
        top_v_vs_allstudies_comparison_plot_path=top_v_vs_allstudies_comparison_plot_path,
        top_run_keys=top_run_keys,
        top_allstudies_run_keys=top_allstudies_run_keys,
    )
    html_path = output_dir / "cross_project_screening_report.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"Wrote {output_dir / 'run_selection.csv'}")
    print(f"Wrote {output_dir / 'screening_metrics_by_run.csv'}")
    print(f"Wrote {output_dir / 'screening_metrics_top_v.csv'}")
    print(f"Wrote {output_dir / 'screening_metrics_top_v_stage_progression.csv'}")
    print(f"Wrote {output_dir / 'screening_metrics_top_v_allstudies.csv'}")
    print(f"Wrote {output_dir / 'screening_metrics_top_v_allstudies_stage_progression.csv'}")
    print(f"Wrote {output_dir / 'screening_metrics_top_v_vs_top_v_allstudies_pairs.csv'}")
    if across_runs_plot_path is not None:
        print(f"Wrote {across_runs_plot_path}")
    if top_v_plot_path is not None:
        print(f"Wrote {top_v_plot_path}")
    if top_v_stage_progression_plot_path is not None:
        print(f"Wrote {top_v_stage_progression_plot_path}")
    if top_v_allstudies_plot_path is not None:
        print(f"Wrote {top_v_allstudies_plot_path}")
    if top_v_allstudies_stage_progression_plot_path is not None:
        print(f"Wrote {top_v_allstudies_stage_progression_plot_path}")
    if top_v_vs_allstudies_comparison_plot_path is not None:
        print(f"Wrote {top_v_vs_allstudies_comparison_plot_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()

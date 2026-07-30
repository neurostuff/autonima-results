#!/usr/bin/env python3
"""Re-run analysis/annotation reports for selected projects and build a cross-project dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROJECTS_ROOT = REPO_ROOT / "projects"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "cross_project_analysis"
COMPARE_SCRIPT = SCRIPT_DIR / "compare_analyses_to_benchmark.py"

ANNOTATION_ONLY_RUN_RE = re.compile(r"^v(?P<version>\d+)-annotation-only(?P<suffix>.*)$")
REQUIRED_RUN_FILES = (
    "outputs/annotation_results.json",
    "outputs/coordinate_parsing_results.json",
    "outputs/nimads_annotation.json",
)
MATCH_RESULTS_FILE = "reports/match_results_overall.json"
ANNOTATION_METRICS_FILE = "reports/annotation_review_reports/annotation_metrics_by_mode.json"
ANALYSIS_REPORT_FILE = "reports/analysis_fuzzy_matching_report.html"
ANNOTATION_SUMMARY_FILE = "reports/annotation_review_reports/overall_submeta_summary.html"
ANALYSIS_ASSUMPTION_STRICT_ID = "analysis_assumption_strict"

ANNOTATION_AGG_SPECS = [
    {
        "id": "study_strict",
        "mode_id": "accepted",
        "mode_label": "strict",
        "level": "study",
        "variant": "study",
        "block_key": "study_metrics",
        "display_label": "Study (Strict)",
    },
    {
        "id": "study_combined",
        "mode_id": "combined",
        "mode_label": "combined",
        "level": "study",
        "variant": "study",
        "block_key": "study_metrics",
        "display_label": "Study (Combined)",
    },
    {
        "id": "analysis_baseline_strict",
        "mode_id": "accepted",
        "mode_label": "strict",
        "level": "analysis",
        "variant": "matched_only",
        "block_key": "analysis_metrics",
        "display_label": "Analysis Matched-Only (Strict)",
    },
    {
        "id": "analysis_baseline_combined",
        "mode_id": "combined",
        "mode_label": "combined",
        "level": "analysis",
        "variant": "matched_only",
        "block_key": "analysis_metrics",
        "display_label": "Analysis Matched-Only (Combined)",
    },
    {
        "id": "analysis_assumption_strict",
        "mode_id": "accepted",
        "mode_label": "strict",
        "level": "analysis",
        "variant": "exhausted_manual_assumption",
        "block_key": "analysis_metrics_exhausted_manual_assumption",
        "display_label": "Analysis Exhausted-Manual (Strict)",
    },
    {
        "id": "analysis_assumption_combined",
        "mode_id": "combined",
        "mode_label": "combined",
        "level": "analysis",
        "variant": "exhausted_manual_assumption",
        "block_key": "analysis_metrics_exhausted_manual_assumption",
        "display_label": "Analysis Exhausted-Manual (Combined)",
    },
]


@dataclass
class CandidateRun:
    run_dir: Path
    version: int
    is_strict: bool
    missing_required_files: list[str]


@dataclass
class ProjectSelection:
    project_name: str
    status: str
    reason: str
    selected_run_dir: Path | None = None
    selected_version: int | None = None
    matched_candidate_names: list[str] | None = None


@dataclass
class ProjectExecutionResult:
    selection: ProjectSelection
    rerun_status: str
    return_code: int | None
    log_path: Path | None
    parsing_metrics: dict[str, Any] | None
    annotation_metrics: dict[str, dict[str, Any]] | None
    analysis_report_path: Path | None
    annotation_summary_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=PROJECTS_ROOT,
        help="Projects root directory (default: repo/projects).",
    )
    parser.add_argument(
        "--compare-script",
        type=Path,
        default=COMPARE_SCRIPT,
        help="Path to compare_analyses_to_benchmark.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for cross-project dashboard and CSVs.",
    )
    parser.add_argument(
        "--decimal-manual-coordinate-handling",
        choices=("exclude", "convert_to_talairach", "keep", "match_best_space"),
        default="match_best_space",
        help=(
            "Forwarded to compare_analyses_to_benchmark.py. "
            "Default is match_best_space for cross-project reruns."
        ),
    )
    return parser.parse_args()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def compute_prf_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    total = tp + fp + fn + tn
    accuracy = float((tp + tn) / total) if total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def discover_project_selections(projects_root: Path) -> list[ProjectSelection]:
    selections: list[ProjectSelection] = []
    if not projects_root.exists() or not projects_root.is_dir():
        return selections

    for project_dir in sorted(projects_root.iterdir(), key=lambda p: p.name):
        if not project_dir.is_dir():
            continue
        matched_candidates: list[CandidateRun] = []
        matched_names: list[str] = []

        for child in sorted(project_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            match = ANNOTATION_ONLY_RUN_RE.fullmatch(child.name)
            if not match:
                continue
            version = safe_int(match.group("version"), default=-1)
            is_strict = str(match.group("suffix") or "") == ""
            matched_names.append(child.name)
            missing_files = [name for name in REQUIRED_RUN_FILES if not (child / name).exists()]
            matched_candidates.append(
                CandidateRun(
                    run_dir=child,
                    version=version,
                    is_strict=is_strict,
                    missing_required_files=missing_files,
                )
            )

        if not matched_candidates:
            selections.append(
                ProjectSelection(
                    project_name=project_dir.name,
                    status="skipped",
                    reason="No directories matched annotation-only pattern ^vN-annotation-only.*$",
                    matched_candidate_names=[],
                )
            )
            continue

        valid_candidates = [c for c in matched_candidates if not c.missing_required_files]
        if not valid_candidates:
            missing_text = "; ".join(
                f"{c.run_dir.name}: missing {', '.join(c.missing_required_files)}"
                for c in sorted(matched_candidates, key=lambda x: x.version)
            )
            selections.append(
                ProjectSelection(
                    project_name=project_dir.name,
                    status="skipped",
                    reason=f"Matched pattern but required files missing ({missing_text})",
                    matched_candidate_names=matched_names,
                )
            )
            continue

        strict_candidates = [c for c in valid_candidates if c.is_strict]
        candidate_pool = strict_candidates or valid_candidates
        selected = max(candidate_pool, key=lambda c: (c.version, int(c.is_strict)))
        reason = (
            "Selected highest strict vN-annotation-only candidate"
            if selected.is_strict
            else "Selected highest annotation-only fallback candidate"
        )
        selections.append(
            ProjectSelection(
                project_name=project_dir.name,
                status="selected",
                reason=reason,
                selected_run_dir=selected.run_dir,
                selected_version=selected.version,
                matched_candidate_names=matched_names,
            )
        )

    return selections


def run_compare_for_project(
    *,
    selection: ProjectSelection,
    compare_script: Path,
    output_dir: Path,
    decimal_manual_coordinate_handling: str,
) -> tuple[str, int | None, Path]:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{selection.project_name}.log"

    if selection.selected_run_dir is None:
        log_path.write_text("Skipped: no selected run directory.\n", encoding="utf-8")
        return "skipped", None, log_path

    python_executable = REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "python"
    if not python_executable.exists():
        python_executable = Path(sys.executable)
    cmd = [
        str(python_executable),
        str(compare_script),
        "--project-output-dir",
        str(selection.selected_run_dir),
        "--decimal-manual-coordinate-handling",
        str(decimal_manual_coordinate_handling),
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
    return ("success" if proc.returncode == 0 else "failed"), proc.returncode, log_path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_parsing_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / MATCH_RESULTS_FILE
    if not path.exists():
        return None

    payload = load_json(path)
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    accepted = safe_int(summary.get("accepted", 0))
    uncertain = safe_int(summary.get("uncertain", 0))
    unmatched = safe_int(summary.get("unmatched", 0))
    manual_total = safe_int(summary.get("manual_analyses_total", 0))
    matched_count = accepted + uncertain
    matched_pct = float(matched_count / manual_total) if manual_total else 0.0
    table_baseline = payload.get("table_only_baseline", {}) if isinstance(payload, dict) else {}
    if not isinstance(table_baseline, dict):
        table_baseline = {}
    table_baseline_available = bool(table_baseline.get("available"))
    table_baseline_manual_total = safe_int(table_baseline.get("manual_analyses_total", 0))
    table_baseline_matched_count = safe_int(table_baseline.get("matched_count", 0))
    table_baseline_matched_pct = (
        float(table_baseline.get("matched_pct"))
        if table_baseline_available and table_baseline.get("matched_pct") is not None
        else None
    )
    return {
        "manual_analyses_total": manual_total,
        "accepted": accepted,
        "uncertain": uncertain,
        "unmatched": unmatched,
        "matched_count": matched_count,
        "matched_pct": matched_pct,
        "table_only_baseline_available": table_baseline_available,
        "table_only_baseline_manual_analyses_total": table_baseline_manual_total,
        "table_only_baseline_accepted": safe_int(table_baseline.get("accepted", 0)),
        "table_only_baseline_uncertain": safe_int(table_baseline.get("uncertain", 0)),
        "table_only_baseline_unmatched": safe_int(table_baseline.get("unmatched", 0)),
        "table_only_baseline_matched_count": table_baseline_matched_count,
        "table_only_baseline_matched_pct": table_baseline_matched_pct,
        "table_only_baseline_table_units": safe_int(table_baseline.get("table_units", 0)),
        "table_only_baseline_coordinate_rows": safe_int(table_baseline.get("coordinate_rows", 0)),
        "table_only_baseline_source": str(table_baseline.get("source", "")),
        "match_results_path": str(path),
    }


def aggregate_annotation_for_project(run_dir: Path) -> dict[str, dict[str, Any]] | None:
    path = run_dir / ANNOTATION_METRICS_FILE
    if not path.exists():
        return None

    payload = load_json(path)
    metrics_by_mode = payload.get("metrics_by_mode", {}) if isinstance(payload, dict) else {}
    if not isinstance(metrics_by_mode, dict):
        return None

    result: dict[str, dict[str, Any]] = {}
    for spec in ANNOTATION_AGG_SPECS:
        mode_metrics = metrics_by_mode.get(spec["mode_id"], {})
        if not isinstance(mode_metrics, dict):
            mode_metrics = {}

        tp = fp = fn = tn = 0
        annotation_count = 0
        activated_pmids = 0
        added_assumed_negative = 0
        for annotation_name, annotation_payload in mode_metrics.items():
            if not isinstance(annotation_payload, dict):
                continue
            block = annotation_payload.get(spec["block_key"], {})
            if not isinstance(block, dict):
                continue
            annotation_count += 1
            tp += safe_int(block.get("tp", 0))
            fp += safe_int(block.get("fp", 0))
            fn += safe_int(block.get("fn", 0))
            tn += safe_int(block.get("tn", 0))
            if spec["variant"] == "exhausted_manual_assumption":
                assumption_meta = annotation_payload.get("assumed_negative_expansion", {})
                if isinstance(assumption_meta, dict):
                    activated_pmids += safe_int(assumption_meta.get("activated_pmids", 0))
                    added_assumed_negative += safe_int(
                        assumption_meta.get("added_assumed_negative_analyses", 0)
                    )

        metrics = compute_prf_from_confusion(tp=tp, fp=fp, fn=fn, tn=tn)
        result[spec["id"]] = {
            "mode_id": spec["mode_id"],
            "mode_label": spec["mode_label"],
            "level": spec["level"],
            "variant": spec["variant"],
            "display_label": spec["display_label"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "annotation_count": annotation_count,
            "activated_pmids": activated_pmids,
            "added_assumed_negative_analyses": added_assumed_negative,
            "metrics_path": str(path),
        }
    return result


def compute_cross_project_annotation_aggregates(
    project_annotation_metrics: list[dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for spec in ANNOTATION_AGG_SPECS:
        totals[spec["id"]] = {
            "mode_id": spec["mode_id"],
            "mode_label": spec["mode_label"],
            "level": spec["level"],
            "variant": spec["variant"],
            "display_label": spec["display_label"],
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "annotation_count": 0,
            "project_count": 0,
            "activated_pmids": 0,
            "added_assumed_negative_analyses": 0,
        }

    for project_metrics in project_annotation_metrics:
        for spec in ANNOTATION_AGG_SPECS:
            row = project_metrics.get(spec["id"])
            if not row:
                continue
            total_row = totals[spec["id"]]
            total_row["project_count"] += 1
            total_row["annotation_count"] += safe_int(row.get("annotation_count", 0))
            total_row["tp"] += safe_int(row.get("tp", 0))
            total_row["fp"] += safe_int(row.get("fp", 0))
            total_row["fn"] += safe_int(row.get("fn", 0))
            total_row["tn"] += safe_int(row.get("tn", 0))
            total_row["activated_pmids"] += safe_int(row.get("activated_pmids", 0))
            total_row["added_assumed_negative_analyses"] += safe_int(
                row.get("added_assumed_negative_analyses", 0)
            )

    for total_row in totals.values():
        metrics = compute_prf_from_confusion(
            tp=safe_int(total_row["tp"]),
            fp=safe_int(total_row["fp"]),
            fn=safe_int(total_row["fn"]),
            tn=safe_int(total_row["tn"]),
        )
        total_row["precision"] = metrics["precision"]
        total_row["recall"] = metrics["recall"]
        total_row["f1"] = metrics["f1"]
        total_row["accuracy"] = metrics["accuracy"]

    return totals


def html_link(base_dir: Path, target_path: Path | None, label: str) -> str:
    if target_path is None or not target_path.exists():
        return "<span class=\"muted\">missing</span>"
    href = Path(os.path.relpath(target_path.resolve(), base_dir.resolve()))
    return f"<a href=\"{escape(str(href))}\">{escape(label)}</a>"


def bar_cell(value: float) -> str:
    pct = max(0.0, min(1.0, float(value)))
    width = pct * 100.0
    return (
        "<div class=\"bar-wrap\">"
        f"<div class=\"bar-fill\" style=\"width:{width:.2f}%\"></div>"
        f"<span class=\"bar-label\">{pct:.3f}</span>"
        "</div>"
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def collect_analysis_assumption_strict_rows(projects_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not projects_root.exists() or not projects_root.is_dir():
        return rows

    for selection in discover_project_selections(projects_root):
        if selection.selected_run_dir is None:
            continue
        metrics = aggregate_annotation_for_project(selection.selected_run_dir)
        if not metrics:
            continue
        strict_row = metrics.get(ANALYSIS_ASSUMPTION_STRICT_ID, {})
        if not strict_row:
            continue
        rows.append(
            {
                "project_name": selection.project_name,
                "run": selection.selected_run_dir.name,
                "version": selection.selected_version or -1,
                "f1": float(strict_row.get("f1", 0.0)),
            }
        )

    rows.sort(key=lambda item: (str(item["project_name"]), int(item["version"]), str(item["run"])))
    return rows


def select_best_version_rows(version_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_project: dict[str, dict[str, Any]] = {}
    for row in version_rows:
        project = str(row.get("project_name", ""))
        if not project:
            continue
        current = best_by_project.get(project)
        if current is None:
            best_by_project[project] = row
            continue
        row_key = (int(row.get("version", -1)), str(row.get("run", "")))
        cur_key = (int(current.get("version", -1)), str(current.get("run", "")))
        if row_key > cur_key:
            best_by_project[project] = row
    return sorted(best_by_project.values(), key=lambda item: str(item["project_name"]))


def write_analysis_assumption_strict_plots(
    output_dir: Path,
    version_rows: list[dict[str, Any]],
) -> tuple[Path | None, Path | None]:
    if not version_rows:
        return None, None

    best_rows = select_best_version_rows(version_rows)
    best_path: Path | None = None
    trend_path: Path | None = None

    if best_rows:
        best_sorted = sorted(best_rows, key=lambda item: float(item.get("f1", 0.0)), reverse=True)
        projects = [str(row["project_name"]) for row in best_sorted]
        vals = [float(row["f1"]) for row in best_sorted]
        colors = plt.get_cmap("tab20").colors
        fig_h = max(4.5, 0.42 * len(projects) + 1.8)
        fig, ax = plt.subplots(figsize=(10.5, fig_h))
        ys = list(range(len(projects)))
        bar_colors = [colors[idx % len(colors)] for idx in range(len(projects))]
        ax.barh(ys, vals, color=bar_colors, alpha=0.85, edgecolor="#111827", linewidth=0.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(projects)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("F1")
        ax.set_title("Analysis F1 Exhausted-Manual (Strict): Best V per Project", fontweight="bold")
        ax.grid(axis="x", alpha=0.28)
        for y, row in zip(ys, best_sorted):
            ax.text(
                min(float(row["f1"]) + 0.01, 0.985),
                y,
                str(row["run"]),
                va="center",
                ha="left",
                fontsize=8,
                color="#374151",
            )
        fig.tight_layout()
        best_path = output_dir / "analysis_assumption_strict_best_v_plot.png"
        fig.savefig(best_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    projects = sorted({str(row["project_name"]) for row in version_rows})
    if projects:
        positions = {project: idx + 1 for idx, project in enumerate(projects)}
        fig_w = max(10, 1.45 * len(projects))
        fig, ax = plt.subplots(figsize=(fig_w, 6))
        cmap = plt.get_cmap("tab20")
        project_to_color = {project: cmap(idx % cmap.N) for idx, project in enumerate(projects)}

        for project in projects:
            rows = [row for row in version_rows if str(row["project_name"]) == project]
            rows.sort(key=lambda item: (int(item.get("version", -1)), str(item.get("run", ""))))
            vals = [float(row["f1"]) for row in rows]
            n_vals = len(vals)
            center = positions[project]
            if n_vals == 1:
                xs = [center]
            else:
                span = 0.30
                xs = [center - span + (2 * span) * (idx / (n_vals - 1)) for idx in range(n_vals)]

            ax.plot(xs, vals, color=project_to_color[project], linewidth=1.5, alpha=0.85, zorder=2)
            ax.scatter(
                xs,
                vals,
                s=42,
                color=project_to_color[project],
                edgecolor="#111827",
                linewidth=0.55,
                zorder=3,
            )
            for x, row, y in zip(xs, rows, vals):
                ax.text(
                    x,
                    y,
                    str(row["run"]),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    color="#374151",
                    rotation=35,
                )

        overall_mean = float(sum(float(row["f1"]) for row in version_rows) / len(version_rows))
        ax.axhline(
            overall_mean,
            color="#b91c1c",
            linestyle=":",
            linewidth=2.1,
            alpha=0.95,
            label=f"Mean F1 = {overall_mean:.4f}",
            zorder=1,
        )
        ax.set_xlim(0.5, len(projects) + 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([positions[project] for project in projects])
        ax.set_xticklabels(projects)
        ax.set_ylabel("F1")
        ax.set_xlabel("Project")
        ax.set_title("Analysis F1 Exhausted-Manual (Strict): Across Versions by Project", fontweight="bold")
        ax.grid(axis="y", alpha=0.28)
        ax.legend(loc="upper right")
        fig.tight_layout()
        trend_path = output_dir / "analysis_assumption_strict_across_versions_plot.png"
        fig.savefig(trend_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return best_path, trend_path


def build_report_html(
    *,
    generated_at_utc: str,
    selections: list[ProjectSelection],
    results: list[ProjectExecutionResult],
    parsing_rows: list[dict[str, Any]],
    parsing_weighted_pct: float,
    parsing_unweighted_pct: float,
    annotation_overall: dict[str, dict[str, Any]],
    annotation_project_rows: list[dict[str, Any]],
    output_dir: Path,
    analysis_assumption_strict_best_plot_path: Path | None,
    analysis_assumption_strict_trend_plot_path: Path | None,
) -> str:
    missing_html = "<span class='muted'>missing</span>"
    na_html = "<span class='muted'>n/a</span>"

    selected_count = len([s for s in selections if s.status == "selected"])
    skipped_count = len([s for s in selections if s.status != "selected"])
    rerun_success_count = len([r for r in results if r.rerun_status == "success"])
    rerun_failed_count = len([r for r in results if r.rerun_status == "failed"])

    selection_table_rows: list[str] = []
    for result in sorted(results, key=lambda r: r.selection.project_name):
        sel = result.selection
        run_text = str(sel.selected_run_dir) if sel.selected_run_dir else ""
        log_href = (
            str(result.log_path.relative_to(output_dir))
            if result.log_path is not None and result.log_path.exists()
            else ""
        )
        log_cell = f"<a href=\"{escape(log_href)}\">log</a>" if log_href else na_html
        selection_table_rows.append(
            "<tr>"
            f"<td>{escape(sel.project_name)}</td>"
            f"<td>{escape(sel.status)}</td>"
            f"<td>{escape(str(sel.selected_version or ''))}</td>"
            f"<td><code>{escape(run_text)}</code></td>"
            f"<td>{escape(result.rerun_status)}</td>"
            f"<td>{escape(str(result.return_code) if result.return_code is not None else '')}</td>"
            f"<td>{escape(sel.reason)}</td>"
            f"<td>{log_cell}</td>"
            "</tr>"
        )

    parsing_plot_rows: list[str] = []
    for row in sorted(parsing_rows, key=lambda r: float(r.get("matched_pct", 0.0)), reverse=True):
        table_baseline_pct = row.get("table_only_baseline_matched_pct")
        if table_baseline_pct is None:
            table_baseline_bar = na_html
            table_baseline_pct_text = na_html
            table_baseline_count_text = na_html
        else:
            table_baseline_pct_float = float(table_baseline_pct)
            table_baseline_bar = bar_cell(table_baseline_pct_float)
            table_baseline_pct_text = f"{table_baseline_pct_float:.3f}"
            table_baseline_count_text = (
                f"{int(row.get('table_only_baseline_matched_count', 0))}/"
                f"{int(row.get('table_only_baseline_manual_analyses_total', 0))}"
            )
        parsing_plot_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('project_name', '')))}</td>"
            f"<td>{bar_cell(float(row.get('matched_pct', 0.0)))}</td>"
            f"<td>{float(row.get('matched_pct', 0.0)):.3f}</td>"
            f"<td>{int(row.get('matched_count', 0))}/{int(row.get('manual_analyses_total', 0))}</td>"
            f"<td>{table_baseline_bar}</td>"
            f"<td>{table_baseline_pct_text}</td>"
            f"<td>{table_baseline_count_text}</td>"
            f"<td>{row.get('analysis_report_link', missing_html)}</td>"
            f"<td>{row.get('annotation_summary_link', missing_html)}</td>"
            "</tr>"
        )

    annotation_overall_rows: list[str] = []
    for spec in ANNOTATION_AGG_SPECS:
        row = annotation_overall.get(spec["id"], {})
        annotation_overall_rows.append(
            "<tr>"
            f"<td>{escape(spec['display_label'])}</td>"
            f"<td>{int(row.get('project_count', 0))}</td>"
            f"<td>{int(row.get('annotation_count', 0))}</td>"
            f"<td>{int(row.get('tp', 0))}</td>"
            f"<td>{int(row.get('fp', 0))}</td>"
            f"<td>{int(row.get('fn', 0))}</td>"
            f"<td>{int(row.get('tn', 0))}</td>"
            f"<td>{float(row.get('precision', 0.0)):.3f}</td>"
            f"<td>{float(row.get('recall', 0.0)):.3f}</td>"
            f"<td>{float(row.get('f1', 0.0)):.3f}</td>"
            f"<td>{float(row.get('accuracy', 0.0)):.3f}</td>"
            f"<td>{int(row.get('activated_pmids', 0))}</td>"
            f"<td>{int(row.get('added_assumed_negative_analyses', 0))}</td>"
            "</tr>"
        )

    annotation_project_table_rows: list[str] = []
    for row in sorted(annotation_project_rows, key=lambda r: r["project_name"]):
        annotation_project_table_rows.append(
            "<tr>"
            f"<td>{escape(row['project_name'])}</td>"
            f"<td>{bar_cell(float(row.get('study_strict_f1', 0.0)))}</td>"
            f"<td>{bar_cell(float(row.get('study_combined_f1', 0.0)))}</td>"
            f"<td>{bar_cell(float(row.get('analysis_baseline_strict_f1', 0.0)))}</td>"
            f"<td>{bar_cell(float(row.get('analysis_assumption_strict_f1', 0.0)))}</td>"
            f"<td>{bar_cell(float(row.get('analysis_baseline_combined_f1', 0.0)))}</td>"
            f"<td>{bar_cell(float(row.get('analysis_assumption_combined_f1', 0.0)))}</td>"
            f"<td>{row.get('analysis_report_link', missing_html)}</td>"
            f"<td>{row.get('annotation_summary_link', missing_html)}</td>"
            "</tr>"
        )

    skipped_rows = [
        "<li><code>{}</code>: {}</li>".format(escape(sel.project_name), escape(sel.reason))
        for sel in sorted(selections, key=lambda s: s.project_name)
        if sel.status != "selected"
    ]
    failed_rows = [
        "<li><code>{}</code>: return_code={} (see log)</li>".format(
            escape(result.selection.project_name),
            escape(str(result.return_code)),
        )
        for result in sorted(results, key=lambda r: r.selection.project_name)
        if result.rerun_status == "failed"
    ]

    analysis_plot_sections = ""
    if (
        analysis_assumption_strict_best_plot_path is not None
        and analysis_assumption_strict_best_plot_path.exists()
    ):
        rel = os.path.relpath(
            analysis_assumption_strict_best_plot_path.resolve(),
            output_dir.resolve(),
        )
        analysis_plot_sections += (
            '<section>'
            '<h2>Analysis F1 Exhausted-Manual (Strict): Best V-Version</h2>'
            f'<img src="{escape(rel)}" alt="Best V-version Analysis F1 Exhausted-Manual Strict plot" '
            'style="max-width: 100%; height: auto; border: 1px solid #d5dee8; border-radius: 8px;">'
            '</section>'
        )
    if (
        analysis_assumption_strict_trend_plot_path is not None
        and analysis_assumption_strict_trend_plot_path.exists()
    ):
        rel = os.path.relpath(
            analysis_assumption_strict_trend_plot_path.resolve(),
            output_dir.resolve(),
        )
        analysis_plot_sections += (
            '<section>'
            '<h2>Analysis F1 Exhausted-Manual (Strict): Across Versions by Project</h2>'
            f'<img src="{escape(rel)}" alt="Across-version Analysis F1 Exhausted-Manual Strict plot" '
            'style="max-width: 100%; height: auto; border: 1px solid #d5dee8; border-radius: 8px;">'
            '</section>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cross-Project Analysis Dashboard</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #1f2933;
      --line: #d5dee8;
      --accent: #0f766e;
      --bar-bg: #edf2f7;
      --bar-fill: linear-gradient(90deg, #0f766e 0%, #0284c7 100%);
    }}
    body {{ margin: 0; padding: 1.1rem; background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }}
    header, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }}
    h1, h2 {{ margin-top: 0; }}
    .kpis {{ display: grid; gap: 0.8rem; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); margin-top: 0.75rem; }}
    .kpi {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfdff; padding: 0.65rem; }}
    .kpi .label {{ color: #4a5b70; font-size: 0.84rem; }}
    .kpi .value {{ color: var(--accent); font-size: 1.35rem; font-weight: 700; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border: 1px solid var(--line); padding: 0.45rem; text-align: left; vertical-align: top; }}
    th {{ background: #e9eff6; }}
    .muted {{ color: #5b6b80; font-size: 0.84rem; }}
    .bar-wrap {{ position: relative; min-width: 170px; height: 1.05rem; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; background: var(--bar-bg); }}
    .bar-fill {{ height: 100%; background: var(--bar-fill); }}
    .bar-label {{ position: absolute; right: 0.35rem; top: 50%; transform: translateY(-50%); font-size: 0.75rem; font-weight: 700; color: #0f172a; }}
    ul {{ margin: 0.5rem 0 0 1.15rem; }}
    li {{ margin: 0.25rem 0; }}
    code {{ background: #eef2f7; border: 1px solid #d7dee7; border-radius: 4px; padding: 0.05rem 0.25rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Cross-Project Analysis Dashboard</h1>
    <p>Generated at {escape(generated_at_utc)}. This dashboard re-runs <code>compare_analyses_to_benchmark.py</code> for each selected strict annotation-only run and aggregates parsing + annotation metrics.</p>
    <div class="kpis">
      <div class="kpi"><div class="label">Projects Selected</div><div class="value">{selected_count}</div></div>
      <div class="kpi"><div class="label">Projects Skipped</div><div class="value">{skipped_count}</div></div>
      <div class="kpi"><div class="label">Re-run Successes</div><div class="value">{rerun_success_count}</div></div>
      <div class="kpi"><div class="label">Re-run Failures</div><div class="value">{rerun_failed_count}</div></div>
      <div class="kpi"><div class="label">Parsing Matched % (Weighted)</div><div class="value">{parsing_weighted_pct:.3f}</div></div>
      <div class="kpi"><div class="label">Parsing Matched % (Unweighted)</div><div class="value">{parsing_unweighted_pct:.3f}</div></div>
    </div>
  </header>

  <section>
    <h2>Selection and Re-run Status</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Project</th>
            <th>Selection</th>
            <th>Version</th>
            <th>Selected Run</th>
            <th>Re-run</th>
            <th>Return Code</th>
            <th>Reason</th>
            <th>Log</th>
          </tr>
        </thead>
        <tbody>
          {''.join(selection_table_rows)}
        </tbody>
      </table>
    </div>
    <p class="muted">CSV: <code>project_selection.csv</code></p>
  </section>

  <section>
    <h2>Parsing Performance Across Projects</h2>
    <p>Manual matched % is computed as <code>(accepted + uncertain) / manual_analyses_total</code> from <code>match_results_overall.json</code>.</p>
    <p>Table-only baseline groups raw extracted coordinates by source table before fuzzy matching, representing performance if table/contrast parsing were skipped.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Project</th>
            <th>Matched % Bar</th>
            <th>Matched %</th>
            <th>Matched / Manual</th>
            <th>Table-Only % Bar</th>
            <th>Table-Only %</th>
            <th>Table-Only / Manual</th>
            <th>Fuzzy Report</th>
            <th>Annotation Summary</th>
          </tr>
        </thead>
        <tbody>
          {''.join(parsing_plot_rows)}
        </tbody>
      </table>
    </div>
    <p class="muted">CSV: <code>parsing_metrics_by_project.csv</code></p>
  </section>

  <section>
    <h2>Annotation Aggregates (Micro-Pooled Across Projects and Annotations)</h2>
    <p>Modes: strict (<code>accepted</code>) and combined. Levels: study and analysis. Analysis is shown for both matched-only baseline and exhausted-manual-assumption variants.</p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Metric Slice</th>
            <th>Projects</th>
            <th>Annotations</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
            <th>TN</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Accuracy</th>
            <th>Activated PMIDs</th>
            <th>Added Assumed Negatives</th>
          </tr>
        </thead>
        <tbody>
          {''.join(annotation_overall_rows)}
        </tbody>
      </table>
    </div>
    <p class="muted">CSV: <code>annotation_aggregates.csv</code></p>
  </section>

  <section>
    <h2>Per-Project Annotation F1 Summary</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Project</th>
            <th>Study F1 (Strict)</th>
            <th>Study F1 (Combined)</th>
            <th>Analysis F1 Matched-Only (Strict)</th>
            <th>Analysis F1 Exhausted-Manual (Strict)</th>
            <th>Analysis F1 Matched-Only (Combined)</th>
            <th>Analysis F1 Exhausted-Manual (Combined)</th>
            <th>Fuzzy Report</th>
            <th>Annotation Summary</th>
          </tr>
        </thead>
        <tbody>
          {''.join(annotation_project_table_rows)}
        </tbody>
      </table>
    </div>
  </section>
  {analysis_plot_sections}

  <section>
    <h2>Skipped Projects</h2>
    {('<ul>' + ''.join(skipped_rows) + '</ul>') if skipped_rows else '<p class="muted">No skipped projects.</p>'}
  </section>

  <section>
    <h2>Failed Re-runs</h2>
    {('<ul>' + ''.join(failed_rows) + '</ul>') if failed_rows else '<p class="muted">No re-run failures.</p>'}
  </section>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    projects_root = args.projects_root.expanduser().resolve()
    compare_script = args.compare_script.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    decimal_manual_coordinate_handling = str(args.decimal_manual_coordinate_handling)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not compare_script.exists():
        raise FileNotFoundError(f"Compare script not found: {compare_script}")
    if not projects_root.exists():
        raise FileNotFoundError(f"Projects root not found: {projects_root}")

    selections = discover_project_selections(projects_root)
    print(f"Discovered {len(selections)} project directories under {projects_root}")

    for selection in selections:
        if selection.status == "selected":
            print(
                f"[SELECT] {selection.project_name}: "
                f"{selection.selected_run_dir} (v{selection.selected_version})"
            )
        else:
            print(f"[SKIP]   {selection.project_name}: {selection.reason}")

    results: list[ProjectExecutionResult] = []
    for selection in selections:
        if selection.status != "selected":
            results.append(
                ProjectExecutionResult(
                    selection=selection,
                    rerun_status="skipped",
                    return_code=None,
                    log_path=None,
                    parsing_metrics=None,
                    annotation_metrics=None,
                    analysis_report_path=None,
                    annotation_summary_path=None,
                )
            )
            continue

        print(
            f"[RUN]    {selection.project_name}: rerunning compare script "
            f"(decimal_handling={decimal_manual_coordinate_handling})"
        )
        rerun_status, return_code, log_path = run_compare_for_project(
            selection=selection,
            compare_script=compare_script,
            output_dir=output_dir,
            decimal_manual_coordinate_handling=decimal_manual_coordinate_handling,
        )
        print(f"[{rerun_status.upper():7}] {selection.project_name}: return_code={return_code}")

        run_dir = selection.selected_run_dir
        assert run_dir is not None
        parsing_metrics = load_parsing_metrics(run_dir)
        annotation_metrics = aggregate_annotation_for_project(run_dir)
        analysis_report_path = run_dir / ANALYSIS_REPORT_FILE
        annotation_summary_path = run_dir / ANNOTATION_SUMMARY_FILE

        results.append(
            ProjectExecutionResult(
                selection=selection,
                rerun_status=rerun_status,
                return_code=return_code,
                log_path=log_path,
                parsing_metrics=parsing_metrics,
                annotation_metrics=annotation_metrics,
                analysis_report_path=analysis_report_path if analysis_report_path.exists() else None,
                annotation_summary_path=annotation_summary_path if annotation_summary_path.exists() else None,
            )
        )

    selection_csv_rows: list[dict[str, Any]] = []
    parsing_csv_rows: list[dict[str, Any]] = []
    annotation_csv_rows: list[dict[str, Any]] = []
    annotation_project_rows: list[dict[str, Any]] = []
    parsing_display_rows: list[dict[str, Any]] = []

    total_manual = 0
    total_matched = 0
    parsing_pcts: list[float] = []

    annotation_for_overall: list[dict[str, dict[str, Any]]] = []

    for result in sorted(results, key=lambda r: r.selection.project_name):
        selection = result.selection
        log_rel = (
            str(result.log_path.relative_to(output_dir))
            if result.log_path is not None and result.log_path.exists()
            else ""
        )
        selection_csv_rows.append(
            {
                "project_name": selection.project_name,
                "selection_status": selection.status,
                "selection_reason": selection.reason,
                "selected_version": selection.selected_version or "",
                "selected_run_dir": str(selection.selected_run_dir) if selection.selected_run_dir else "",
                "rerun_status": result.rerun_status,
                "return_code": result.return_code if result.return_code is not None else "",
                "log_path": log_rel,
            }
        )

        if result.parsing_metrics is not None:
            parsing = result.parsing_metrics
            matched_pct = float(parsing.get("matched_pct", 0.0))
            manual_total = safe_int(parsing.get("manual_analyses_total", 0))
            matched_count = safe_int(parsing.get("matched_count", 0))
            total_manual += manual_total
            total_matched += matched_count
            parsing_pcts.append(matched_pct)

            analysis_link = html_link(output_dir, result.analysis_report_path, "analysis_fuzzy_matching_report.html")
            annotation_link = html_link(output_dir, result.annotation_summary_path, "overall_submeta_summary.html")

            parsing_row = {
                "project_name": selection.project_name,
                "selected_run_dir": str(selection.selected_run_dir) if selection.selected_run_dir else "",
                "manual_analyses_total": manual_total,
                "accepted": safe_int(parsing.get("accepted", 0)),
                "uncertain": safe_int(parsing.get("uncertain", 0)),
                "unmatched": safe_int(parsing.get("unmatched", 0)),
                "matched_count": matched_count,
                "matched_pct": matched_pct,
                "table_only_baseline_matched_pct": parsing.get("table_only_baseline_matched_pct"),
                "table_only_baseline_matched_count": safe_int(
                    parsing.get("table_only_baseline_matched_count", 0)
                ),
                "table_only_baseline_manual_analyses_total": safe_int(
                    parsing.get("table_only_baseline_manual_analyses_total", 0)
                ),
                "analysis_report_link": analysis_link,
                "annotation_summary_link": annotation_link,
            }
            parsing_csv_rows.append(
                {
                    "project_name": parsing_row["project_name"],
                    "selected_run_dir": parsing_row["selected_run_dir"],
                    "manual_analyses_total": parsing_row["manual_analyses_total"],
                    "accepted": parsing_row["accepted"],
                    "uncertain": parsing_row["uncertain"],
                    "unmatched": parsing_row["unmatched"],
                    "matched_count": parsing_row["matched_count"],
                    "manual_matched_pct": f"{matched_pct:.6f}",
                    "table_only_baseline_available": int(
                        bool(parsing.get("table_only_baseline_available", False))
                    ),
                    "table_only_baseline_accepted": safe_int(
                        parsing.get("table_only_baseline_accepted", 0)
                    ),
                    "table_only_baseline_uncertain": safe_int(
                        parsing.get("table_only_baseline_uncertain", 0)
                    ),
                    "table_only_baseline_unmatched": safe_int(
                        parsing.get("table_only_baseline_unmatched", 0)
                    ),
                    "table_only_baseline_matched_count": safe_int(
                        parsing.get("table_only_baseline_matched_count", 0)
                    ),
                    "table_only_baseline_manual_analyses_total": safe_int(
                        parsing.get("table_only_baseline_manual_analyses_total", 0)
                    ),
                    "table_only_baseline_matched_pct": (
                        f"{float(parsing['table_only_baseline_matched_pct']):.6f}"
                        if parsing.get("table_only_baseline_matched_pct") is not None
                        else ""
                    ),
                    "table_only_baseline_table_units": safe_int(
                        parsing.get("table_only_baseline_table_units", 0)
                    ),
                    "table_only_baseline_coordinate_rows": safe_int(
                        parsing.get("table_only_baseline_coordinate_rows", 0)
                    ),
                    "table_only_baseline_source": str(
                        parsing.get("table_only_baseline_source", "")
                    ),
                }
            )
            parsing_display_rows.append(parsing_row)
        else:
            parsing_row = None

        if result.annotation_metrics is not None:
            annotation_for_overall.append(result.annotation_metrics)
            project_summary = {
                "project_name": selection.project_name,
                "analysis_report_link": html_link(output_dir, result.analysis_report_path, "analysis_fuzzy_matching_report.html"),
                "annotation_summary_link": html_link(output_dir, result.annotation_summary_path, "overall_submeta_summary.html"),
            }
            for spec in ANNOTATION_AGG_SPECS:
                row = result.annotation_metrics.get(spec["id"], {})
                project_summary[f"{spec['id']}_f1"] = float(row.get("f1", 0.0))
                annotation_csv_rows.append(
                    {
                        "scope": "project",
                        "project_name": selection.project_name,
                        "mode_id": spec["mode_id"],
                        "level": spec["level"],
                        "variant": spec["variant"],
                        "display_label": spec["display_label"],
                        "tp": safe_int(row.get("tp", 0)),
                        "fp": safe_int(row.get("fp", 0)),
                        "fn": safe_int(row.get("fn", 0)),
                        "tn": safe_int(row.get("tn", 0)),
                        "precision": f"{float(row.get('precision', 0.0)):.6f}",
                        "recall": f"{float(row.get('recall', 0.0)):.6f}",
                        "f1": f"{float(row.get('f1', 0.0)):.6f}",
                        "accuracy": f"{float(row.get('accuracy', 0.0)):.6f}",
                        "annotation_count": safe_int(row.get("annotation_count", 0)),
                        "project_count": 1,
                        "activated_pmids": safe_int(row.get("activated_pmids", 0)),
                        "added_assumed_negative_analyses": safe_int(
                            row.get("added_assumed_negative_analyses", 0)
                        ),
                    }
                )
            annotation_project_rows.append(project_summary)

    annotation_overall = compute_cross_project_annotation_aggregates(annotation_for_overall)
    for spec in ANNOTATION_AGG_SPECS:
        row = annotation_overall.get(spec["id"], {})
        annotation_csv_rows.append(
            {
                "scope": "overall",
                "project_name": "__all__",
                "mode_id": spec["mode_id"],
                "level": spec["level"],
                "variant": spec["variant"],
                "display_label": spec["display_label"],
                "tp": safe_int(row.get("tp", 0)),
                "fp": safe_int(row.get("fp", 0)),
                "fn": safe_int(row.get("fn", 0)),
                "tn": safe_int(row.get("tn", 0)),
                "precision": f"{float(row.get('precision', 0.0)):.6f}",
                "recall": f"{float(row.get('recall', 0.0)):.6f}",
                "f1": f"{float(row.get('f1', 0.0)):.6f}",
                "accuracy": f"{float(row.get('accuracy', 0.0)):.6f}",
                "annotation_count": safe_int(row.get("annotation_count", 0)),
                "project_count": safe_int(row.get("project_count", 0)),
                "activated_pmids": safe_int(row.get("activated_pmids", 0)),
                "added_assumed_negative_analyses": safe_int(row.get("added_assumed_negative_analyses", 0)),
            }
        )

    parsing_weighted_pct = float(total_matched / total_manual) if total_manual else 0.0
    parsing_unweighted_pct = float(sum(parsing_pcts) / len(parsing_pcts)) if parsing_pcts else 0.0

    write_csv(
        output_dir / "project_selection.csv",
        selection_csv_rows,
        [
            "project_name",
            "selection_status",
            "selection_reason",
            "selected_version",
            "selected_run_dir",
            "rerun_status",
            "return_code",
            "log_path",
        ],
    )
    write_csv(
        output_dir / "parsing_metrics_by_project.csv",
        parsing_csv_rows,
        [
            "project_name",
            "selected_run_dir",
            "manual_analyses_total",
            "accepted",
            "uncertain",
            "unmatched",
            "matched_count",
            "manual_matched_pct",
            "table_only_baseline_available",
            "table_only_baseline_accepted",
            "table_only_baseline_uncertain",
            "table_only_baseline_unmatched",
            "table_only_baseline_matched_count",
            "table_only_baseline_manual_analyses_total",
            "table_only_baseline_matched_pct",
            "table_only_baseline_table_units",
            "table_only_baseline_coordinate_rows",
            "table_only_baseline_source",
        ],
    )
    write_csv(
        output_dir / "annotation_aggregates.csv",
        annotation_csv_rows,
        [
            "scope",
            "project_name",
            "mode_id",
            "level",
            "variant",
            "display_label",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "annotation_count",
            "project_count",
            "activated_pmids",
            "added_assumed_negative_analyses",
        ],
    )

    analysis_assumption_strict_version_rows = collect_analysis_assumption_strict_rows(projects_root)
    write_csv(
        output_dir / "analysis_assumption_strict_by_version.csv",
        [
            {
                "project_name": row["project_name"],
                "run": row["run"],
                "version": int(row["version"]),
                "f1": f"{float(row['f1']):.6f}",
            }
            for row in analysis_assumption_strict_version_rows
        ],
        ["project_name", "run", "version", "f1"],
    )
    (
        analysis_assumption_strict_best_plot_path,
        analysis_assumption_strict_trend_plot_path,
    ) = write_analysis_assumption_strict_plots(
        output_dir=output_dir,
        version_rows=analysis_assumption_strict_version_rows,
    )

    html = build_report_html(
        generated_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        selections=selections,
        results=results,
        parsing_rows=parsing_display_rows,
        parsing_weighted_pct=parsing_weighted_pct,
        parsing_unweighted_pct=parsing_unweighted_pct,
        annotation_overall=annotation_overall,
        annotation_project_rows=annotation_project_rows,
        output_dir=output_dir,
        analysis_assumption_strict_best_plot_path=analysis_assumption_strict_best_plot_path,
        analysis_assumption_strict_trend_plot_path=analysis_assumption_strict_trend_plot_path,
    )
    html_path = output_dir / "cross_project_analysis_report.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"Wrote {output_dir / 'project_selection.csv'}")
    print(f"Wrote {output_dir / 'parsing_metrics_by_project.csv'}")
    print(f"Wrote {output_dir / 'annotation_aggregates.csv'}")
    print(f"Wrote {output_dir / 'analysis_assumption_strict_by_version.csv'}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()

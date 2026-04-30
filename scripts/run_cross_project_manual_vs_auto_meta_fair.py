#!/usr/bin/env python3
"""Run fair manual-vs-auto meta comparisons across annotation-only runs.

Fair IDs are constrained by BOTH:
1) PMIDs listed in each run's YAML search.pmids_file
2) PMIDs with >=1 parsed coordinate point in outputs/coordinate_parsing_results.json

For each project, the script computes the intersection across eligible runs'
effective PMID sets, builds a filtered manual NiMADS subset once, runs manual
meta-analyses once, and compares against all eligible annotation-only runs.
It also writes a cross-project summary report.
"""

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
from matplotlib.lines import Line2D
import yaml
from autonima import meta as autonima_meta


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PROJECTS_ROOT = REPO_ROOT / "projects"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports" / "cross_project_manual_vs_auto_meta_fair"
DEFAULT_MANUAL_NIMADS_BASE = Path("/home/zorro/repos/neurometabench/data/nimads")
DEFAULT_MANUAL_ANALYSIS_MAP_FILENAME = "z.nii.gz"
COMPARE_SCRIPT = SCRIPT_DIR / "compare_meta_to_benchmark.py"

ANNOTATION_ONLY_RUN_RE = re.compile(r"^v(?P<version>\d+)-annotation-only(?:-.+)?$")
REQUIRED_RUN_OUTPUTS = (
    "outputs/meta_analysis_results",
    "outputs/coordinate_parsing_results.json",
)
MISSING_HTML = '<span class="muted">missing</span>'


@dataclass
class RunRecord:
    run_name: str
    run_dir: Path
    yaml_path: Path
    pmids_file: Path
    input_ids: set[str]
    output_ids_with_points: set[str]
    effective_ids: set[str]


@dataclass
class ProjectResult:
    project_name: str
    status: str
    reason: str
    report_dir: Path | None = None
    report_html_path: Path | None = None
    run_count: int = 0
    fair_id_count: int = 0
    manual_meta_status: str = "not_run"
    compare_status: str = "not_run"
    compare_return_code: int | None = None
    compare_log_path: Path | None = None
    project_run_summary_path: Path | None = None
    project_diag_summary_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="Project name to process (repeatable). If omitted, process all projects.",
    )
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=DEFAULT_PROJECTS_ROOT,
        help="Projects root directory (default: repo/projects).",
    )
    parser.add_argument(
        "--manual-nimads-base",
        type=Path,
        default=DEFAULT_MANUAL_NIMADS_BASE,
        help="Base directory for manual merged NiMADS data by project.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Cross-project output directory for dashboard and CSVs.",
    )
    parser.add_argument(
        "--compare-script",
        type=Path,
        default=COMPARE_SCRIPT,
        help="Path to compare_meta_to_benchmark.py.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute fair manual subset/meta outputs even if cached artifacts exist.",
    )
    parser.add_argument(
        "--estimator",
        choices=["ale", "mkdadensity", "kda"],
        default="mkdadensity",
        help="Manual fair meta estimator (default: mkdadensity).",
    )
    parser.add_argument(
        "--corrector",
        choices=["fdr", "montecarlo", "bonferroni"],
        default="fdr",
        help="Manual fair meta corrector (default: fdr).",
    )
    parser.add_argument(
        "--estimator-args",
        type=str,
        default="{}",
        help="JSON estimator args forwarded to autonima.meta.",
    )
    parser.add_argument(
        "--corrector-args",
        type=str,
        default="{}",
        help="JSON corrector args forwarded to autonima.meta.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def read_pmid_file(path: Path) -> set[str]:
    pmids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = str(line).strip()
            if not value or value.startswith("#"):
                continue
            pmids.add(value)
    return pmids


def extract_output_ids_with_points(coordinate_parsing_results_path: Path) -> set[str]:
    payload = read_json(coordinate_parsing_results_path)
    studies = payload.get("studies", []) if isinstance(payload, dict) else []
    if not isinstance(studies, list):
        raise ValueError(
            "coordinate_parsing_results.json must contain a list at key 'studies': "
            f"{coordinate_parsing_results_path}"
        )

    pmids_with_points: set[str] = set()
    for study in studies:
        if not isinstance(study, dict):
            continue
        pmid = str(study.get("pmid") or "").strip()
        if not pmid:
            continue

        analyses = study.get("analyses", [])
        if not isinstance(analyses, list):
            continue

        has_any_parsed_point = False
        for analysis in analyses:
            if not isinstance(analysis, dict):
                continue
            points = analysis.get("points", [])
            if not isinstance(points, list):
                continue
            for point in points:
                if not isinstance(point, dict):
                    continue
                coords = point.get("coordinates")
                if not isinstance(coords, (list, tuple)) or len(coords) != 3:
                    continue
                try:
                    float(coords[0])
                    float(coords[1])
                    float(coords[2])
                except Exception:
                    continue
                has_any_parsed_point = True
                break
            if has_any_parsed_point:
                break

        if has_any_parsed_point:
            pmids_with_points.add(pmid)

    return pmids_with_points


def project_dirs_from_args(projects_root: Path, project_filters: list[str]) -> list[Path]:
    if not projects_root.exists() or not projects_root.is_dir():
        raise FileNotFoundError(f"Projects root does not exist: {projects_root}")

    all_dirs = sorted([path for path in projects_root.iterdir() if path.is_dir()], key=lambda p: p.name)
    if not project_filters:
        return all_dirs

    requested = set(project_filters)
    selected = [path for path in all_dirs if path.name in requested]
    missing = sorted(requested - {path.name for path in selected})
    if missing:
        raise ValueError(
            "Unknown --project values: "
            + ", ".join(missing)
            + ". Available projects: "
            + ", ".join(path.name for path in all_dirs)
        )
    return selected


def discover_annotation_only_runs(project_dir: Path) -> list[Path]:
    matches: list[tuple[int, str, Path]] = []
    for child in project_dir.iterdir():
        if not child.is_dir():
            continue
        match = ANNOTATION_ONLY_RUN_RE.fullmatch(child.name)
        if not match:
            continue
        version = int(match.group("version"))
        matches.append((version, child.name, child))
    matches.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in matches]


def resolve_mapping_path(project_dir: Path) -> Path:
    candidates = [project_dir / "nmb_mappings.json", project_dir / "nmb_mapping.json"]
    existing = next((path for path in candidates if path.exists()), None)
    if existing is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Missing mapping file for project {project_dir.name}. Searched: {searched}")
    return existing


def load_mapping_manual_names(mapping_path: Path) -> list[str]:
    return [manual_name for manual_name, _auto_name in load_mapping_pairs(mapping_path)]


def load_mapping_pairs(mapping_path: Path) -> list[tuple[str, str]]:
    payload = read_json(mapping_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Mapping file must be a JSON object: {mapping_path}")
    raw = payload.get("annotation_mappings", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid mapping payload in {mapping_path}: expected object mappings")

    mapping_pairs: list[tuple[str, str]] = []
    for manual_name, auto_name in raw.items():
        if str(manual_name).strip() == "meta_pmid":
            continue
        if isinstance(auto_name, (dict, list)):
            continue
        manual_name_clean = str(manual_name).strip()
        auto_name_clean = str(auto_name).strip()
        if manual_name_clean and auto_name_clean:
            mapping_pairs.append((manual_name_clean, auto_name_clean))
    if not mapping_pairs:
        raise ValueError(f"No valid mapping entries found in {mapping_path}")
    return mapping_pairs


def collect_run_records(project_dir: Path) -> tuple[list[RunRecord], list[dict[str, str]]]:
    run_dirs = discover_annotation_only_runs(project_dir)
    run_records: list[RunRecord] = []
    skipped: list[dict[str, str]] = []

    for run_dir in run_dirs:
        run_name = run_dir.name
        missing_required = [rel for rel in REQUIRED_RUN_OUTPUTS if not (run_dir / rel).exists()]
        if missing_required:
            skipped.append(
                {
                    "run_name": run_name,
                    "reason": "missing required outputs: " + ", ".join(missing_required),
                }
            )
            continue

        yaml_path = project_dir / f"{run_name}.yaml"
        if not yaml_path.exists():
            skipped.append({"run_name": run_name, "reason": f"missing run YAML: {yaml_path}"})
            continue

        try:
            config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            skipped.append({"run_name": run_name, "reason": f"failed to parse YAML: {exc}"})
            continue

        if not isinstance(config, dict):
            skipped.append({"run_name": run_name, "reason": "run YAML root is not a mapping object"})
            continue

        search_cfg = config.get("search", {})
        if not isinstance(search_cfg, dict):
            skipped.append({"run_name": run_name, "reason": "search config missing or invalid"})
            continue

        pmids_file_raw = search_cfg.get("pmids_file")
        if not pmids_file_raw:
            skipped.append({"run_name": run_name, "reason": "search.pmids_file missing in run YAML"})
            continue

        pmids_file = Path(str(pmids_file_raw)).expanduser()
        if not pmids_file.is_absolute():
            pmids_file = (yaml_path.parent / pmids_file).resolve()

        if not pmids_file.exists():
            skipped.append(
                {
                    "run_name": run_name,
                    "reason": f"search.pmids_file does not exist: {pmids_file}",
                }
            )
            continue

        try:
            input_ids = read_pmid_file(pmids_file)
        except Exception as exc:
            skipped.append({"run_name": run_name, "reason": f"failed reading pmids_file: {exc}"})
            continue

        if not input_ids:
            skipped.append({"run_name": run_name, "reason": "pmids_file resolved but had zero IDs"})
            continue

        coord_path = run_dir / "outputs" / "coordinate_parsing_results.json"
        try:
            output_ids_with_points = extract_output_ids_with_points(coord_path)
        except Exception as exc:
            skipped.append(
                {
                    "run_name": run_name,
                    "reason": f"failed parsing coordinate_parsing_results.json: {exc}",
                }
            )
            continue

        effective_ids = input_ids & output_ids_with_points
        if not effective_ids:
            skipped.append(
                {
                    "run_name": run_name,
                    "reason": (
                        "run-effective ID set is empty after "
                        "input_ids ∩ output_ids_with_points"
                    ),
                }
            )
            continue

        run_records.append(
            RunRecord(
                run_name=run_name,
                run_dir=run_dir,
                yaml_path=yaml_path,
                pmids_file=pmids_file,
                input_ids=input_ids,
                output_ids_with_points=output_ids_with_points,
                effective_ids=effective_ids,
            )
        )

    return run_records, skipped


def parse_maybe_list_payload(payload: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(payload, dict):
        return payload, False
    if isinstance(payload, list):
        if not payload:
            return {}, True
        if isinstance(payload[0], dict):
            return payload[0], True
    raise ValueError("Expected JSON payload to be an object or list with object first entry.")


def rebuild_payload(base_obj: dict[str, Any], wrapped_as_list: bool) -> Any:
    if wrapped_as_list:
        return [base_obj]
    return base_obj


def build_filtered_manual_nimads_subset(
    *,
    manual_studyset_path: Path,
    manual_annotation_path: Path,
    include_pmids: set[str],
    output_studyset_path: Path,
    output_annotation_path: Path,
) -> dict[str, int]:
    studyset_payload_raw = read_json(manual_studyset_path)
    annotation_payload_raw = read_json(manual_annotation_path)

    studyset_payload, studyset_wrapped = parse_maybe_list_payload(studyset_payload_raw)
    annotation_payload, annotation_wrapped = parse_maybe_list_payload(annotation_payload_raw)

    studies = studyset_payload.get("studies", [])
    if not isinstance(studies, list):
        raise ValueError(f"Invalid studyset payload at {manual_studyset_path}: missing list studies")

    filtered_studies: list[dict[str, Any]] = []
    kept_analysis_ids: set[str] = set()
    for study in studies:
        if not isinstance(study, dict):
            continue
        study_id = str(study.get("id") or study.get("pmid") or "").strip()
        if not study_id or study_id not in include_pmids:
            continue
        analyses = study.get("analyses", [])
        if not isinstance(analyses, list):
            analyses = []
        new_analyses: list[dict[str, Any]] = []
        for analysis in analyses:
            if not isinstance(analysis, dict):
                continue
            analysis_id = str(analysis.get("id") or "").strip()
            if not analysis_id:
                continue
            kept_analysis_ids.add(analysis_id)
            new_analyses.append(analysis)
        new_study = dict(study)
        new_study["analyses"] = new_analyses
        filtered_studies.append(new_study)

    filtered_studyset_obj = dict(studyset_payload)
    filtered_studyset_obj["studies"] = filtered_studies

    notes = annotation_payload.get("notes", [])
    if not isinstance(notes, list):
        raise ValueError(f"Invalid annotation payload at {manual_annotation_path}: missing list notes")

    filtered_notes = []
    for note_row in notes:
        if not isinstance(note_row, dict):
            continue
        analysis_id = str(note_row.get("analysis") or "").strip()
        if analysis_id and analysis_id in kept_analysis_ids:
            filtered_notes.append(note_row)

    filtered_annotation_obj = dict(annotation_payload)
    filtered_annotation_obj["notes"] = filtered_notes

    write_json(output_studyset_path, rebuild_payload(filtered_studyset_obj, studyset_wrapped))
    write_json(output_annotation_path, rebuild_payload(filtered_annotation_obj, annotation_wrapped))

    return {
        "n_input_pmids": len(include_pmids),
        "n_filtered_studies": len(filtered_studies),
        "n_kept_analysis_ids": len(kept_analysis_ids),
        "n_filtered_notes": len(filtered_notes),
    }


def manual_meta_outputs_complete(
    manual_analysis_project_dir: Path,
    manual_names: list[str],
    map_filename: str,
) -> bool:
    return all((manual_analysis_project_dir / manual_name / map_filename).exists() for manual_name in manual_names)


def run_manual_meta_for_project(
    *,
    project_name: str,
    manual_names: list[str],
    filtered_studyset_path: Path,
    filtered_annotation_path: Path,
    manual_analysis_base: Path,
    map_filename: str,
    estimator: str,
    estimator_args: dict[str, Any],
    corrector: str,
    corrector_args: dict[str, Any],
    force: bool,
) -> str:
    manual_analysis_project_dir = manual_analysis_base / project_name
    manual_analysis_project_dir.mkdir(parents=True, exist_ok=True)

    if not force and manual_meta_outputs_complete(
        manual_analysis_project_dir=manual_analysis_project_dir,
        manual_names=manual_names,
        map_filename=map_filename,
    ):
        return "cached"

    results = autonima_meta.run_meta_analyses_from_files(
        studyset_file=filtered_studyset_path,
        annotation_file=filtered_annotation_path,
        output_dir=manual_analysis_project_dir,
        estimator_name=estimator,
        estimator_args=estimator_args,
        corrector_name=corrector,
        corrector_args=corrector_args,
        include_ids=None,
        skip_existing=False,
        columns=manual_names,
        fail_fast=True,
        debug=False,
        generate_reports=False,
    )

    _ = results
    if not manual_meta_outputs_complete(
        manual_analysis_project_dir=manual_analysis_project_dir,
        manual_names=manual_names,
        map_filename=map_filename,
    ):
        raise RuntimeError(
            "Manual fair meta run finished but required map outputs are missing for one or more mapped manual annotations."
        )
    return "recomputed"


def run_compare_meta(
    *,
    compare_script: Path,
    project_dir: Path,
    run_records: list[RunRecord],
    manual_analysis_base: Path,
    output_dir: Path,
    log_path: Path,
) -> tuple[str, int]:
    cmd = [
        sys.executable,
        str(compare_script),
        "--project-dir",
        str(project_dir),
        "--manual-analysis-base",
        str(manual_analysis_base),
        "--output-dir",
        str(output_dir),
    ]
    for run_record in run_records:
        cmd.extend(["--run-dir", str(run_record.run_dir)])

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    combined = (
        f"$ {' '.join(cmd)}\n\n"
        f"exit_code={proc.returncode}\n\n"
        "STDOUT:\n"
        f"{proc.stdout}\n\n"
        "STDERR:\n"
        f"{proc.stderr}\n"
    )
    log_path.write_text(combined, encoding="utf-8")
    return ("success" if proc.returncode == 0 else "failed"), proc.returncode


def relative_link(from_dir: Path, target: Path | None, label: str) -> str:
    if target is None or not target.exists():
        return "<span class='muted'>missing</span>"
    href = os.path.relpath(target.resolve(), from_dir.resolve())
    return f"<a href=\"{escape(href)}\">{escape(label)}</a>"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_matrix_row_mean(matrix_path: Path, row_name: str) -> float | None:
    if not matrix_path.exists():
        return None
    with matrix_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        index_key = reader.fieldnames[0]
        for row in reader:
            if str(row.get(index_key, "")).strip() != row_name:
                continue
            values: list[float] = []
            for key, value in row.items():
                if key == index_key:
                    continue
                try:
                    values.append(float(value))
                except Exception:
                    continue
            if not values:
                return None
            return float(sum(values) / len(values))
    return None


def read_all_analyses_baseline(project_report_dir: Path, run_name: str) -> tuple[float | None, float | None]:
    safe_run_name = sanitize_name(run_name)
    tables_dir = project_report_dir / "tables"
    dice_matrix_path = tables_dir / f"dice_matrix_{safe_run_name}.csv"
    pearson_matrix_path = tables_dir / f"pearson_matrix_{safe_run_name}.csv"
    return (
        read_matrix_row_mean(dice_matrix_path, "all_analyses"),
        read_matrix_row_mean(pearson_matrix_path, "all_analyses"),
    )


def read_off_diagonal_mean(
    matrix_path: Path,
    mapping_pairs: list[tuple[str, str]],
    baseline_row_name: str = "all_analyses",
) -> float | None:
    if not matrix_path.exists():
        return None

    diagonal_cells = {(auto_name, manual_name) for manual_name, auto_name in mapping_pairs}
    values: list[float] = []

    with matrix_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        index_key = reader.fieldnames[0]
        for row in reader:
            row_name = str(row.get(index_key, "")).strip()
            if not row_name or row_name == baseline_row_name:
                continue
            for col_name, raw_value in row.items():
                if col_name == index_key:
                    continue
                cell_key = (row_name, str(col_name))
                if cell_key in diagonal_cells:
                    continue
                try:
                    values.append(float(raw_value))
                except Exception:
                    continue

    if not values:
        return None
    return float(sum(values) / len(values))


def read_run_summary_rows(
    run_summary_path: Path,
    project_name: str,
    project_report_dir: Path | None = None,
    mapping_pairs: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not run_summary_path.exists():
        return rows
    with run_summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_name = str(row.get("run", ""))
            all_analyses_dice: float | None = None
            all_analyses_pearson: float | None = None
            off_diagonal_dice: float | None = None
            off_diagonal_pearson: float | None = None
            if project_report_dir is not None and run_name:
                all_analyses_dice, all_analyses_pearson = read_all_analyses_baseline(
                    project_report_dir=project_report_dir,
                    run_name=run_name,
                )
                if mapping_pairs:
                    safe_run_name = sanitize_name(run_name)
                    tables_dir = project_report_dir / "tables"
                    off_diagonal_dice = read_off_diagonal_mean(
                        tables_dir / f"dice_matrix_{safe_run_name}.csv",
                        mapping_pairs=mapping_pairs,
                    )
                    off_diagonal_pearson = read_off_diagonal_mean(
                        tables_dir / f"pearson_matrix_{safe_run_name}.csv",
                        mapping_pairs=mapping_pairs,
                    )
            rows.append(
                {
                    "project_name": project_name,
                    "run": run_name,
                    "dice_mean_diagonal": float(row.get("dice_mean_diagonal", 0.0)),
                    "pearson_mean_diagonal": float(row.get("pearson_mean_diagonal", 0.0)),
                    "dice_mean_off_diagonal": off_diagonal_dice,
                    "pearson_mean_off_diagonal": off_diagonal_pearson,
                    "n_rows": int(float(row.get("n_rows", 0) or 0)),
                    "n_cols": int(float(row.get("n_cols", 0) or 0)),
                    "all_analyses_dice": all_analyses_dice,
                    "all_analyses_pearson": all_analyses_pearson,
                }
            )
    return rows


def read_diagonal_metric_rows(diagonal_metrics_path: Path, project_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not diagonal_metrics_path.exists():
        return rows
    with diagonal_metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dice = float(row.get("dice", "nan"))
            except Exception:
                continue
            try:
                pearson = float(row.get("pearson_r", "nan"))
            except Exception:
                continue
            rows.append(
                {
                    "project_name": project_name,
                    "run": str(row.get("run", "")),
                    "manual_name": str(row.get("manual_name", "")),
                    "auto_name": str(row.get("auto_name", "")),
                    "dice": dice,
                    "pearson_r": pearson,
                }
            )
    return rows


def build_cross_project_html(
    *,
    output_root: Path,
    generated_at_utc: str,
    project_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    global_summary: dict[str, float],
    run_metrics_dice_plot_path: Path | None,
    run_metrics_pearson_plot_path: Path | None,
) -> str:
    project_rows_html: list[str] = []
    for row in sorted(project_rows, key=lambda item: str(item["project_name"])):
        project_rows_html.append(
            "<tr>"
            f"<td>{escape(str(row['project_name']))}</td>"
            f"<td>{escape(str(row['status']))}</td>"
            f"<td>{escape(str(row['reason']))}</td>"
            f"<td>{int(row.get('run_count', 0))}</td>"
            f"<td>{int(row.get('fair_id_count', 0))}</td>"
            f"<td>{escape(str(row.get('manual_meta_status', '')))}</td>"
            f"<td>{escape(str(row.get('compare_status', '')))}</td>"
            f"<td>{row.get('project_report_link', MISSING_HTML)}</td>"
            f"<td>{row.get('compare_log_link', MISSING_HTML)}</td>"
            "</tr>"
        )

    run_rows_html: list[str] = []
    for row in sorted(run_rows, key=lambda item: (str(item["project_name"]), str(item["run"]))):
        dice_baseline = row.get("all_analyses_dice")
        pearson_baseline = row.get("all_analyses_pearson")
        dice_offdiag = row.get("dice_mean_off_diagonal")
        pearson_offdiag = row.get("pearson_mean_off_diagonal")
        run_rows_html.append(
            "<tr>"
            f"<td>{escape(str(row['project_name']))}</td>"
            f"<td>{escape(str(row['run']))}</td>"
            f"<td>{float(row.get('dice_mean_diagonal', 0.0)):.4f}</td>"
            f"<td>{float(row.get('pearson_mean_diagonal', 0.0)):.4f}</td>"
            f"<td>{f'{float(dice_baseline):.4f}' if dice_baseline is not None else 'n/a'}</td>"
            f"<td>{f'{float(pearson_baseline):.4f}' if pearson_baseline is not None else 'n/a'}</td>"
            f"<td>{f'{float(dice_offdiag):.4f}' if dice_offdiag is not None else 'n/a'}</td>"
            f"<td>{f'{float(pearson_offdiag):.4f}' if pearson_offdiag is not None else 'n/a'}</td>"
            f"<td>{int(row.get('n_rows', 0))}</td>"
            f"<td>{int(row.get('n_cols', 0))}</td>"
            "</tr>"
        )

    run_metrics_plot_html = ""
    if run_metrics_dice_plot_path is not None and run_metrics_dice_plot_path.exists():
        rel = os.path.relpath(run_metrics_dice_plot_path.resolve(), output_root.resolve())
        run_metrics_plot_html += (
            '<section>'
            '<h2>Per-Run Dice Chart</h2>'
            f'<img src="{escape(rel)}" alt="Per-run Dice chart" '
            'style="max-width: 100%; height: auto; border: 1px solid #d5dee8; border-radius: 8px;">'
            '</section>'
        )
    if run_metrics_pearson_plot_path is not None and run_metrics_pearson_plot_path.exists():
        rel = os.path.relpath(run_metrics_pearson_plot_path.resolve(), output_root.resolve())
        run_metrics_plot_html += (
            '<section>'
            '<h2>Per-Run Pearson Chart</h2>'
            f'<img src="{escape(rel)}" alt="Per-run Pearson chart" '
            'style="max-width: 100%; height: auto; border: 1px solid #d5dee8; border-radius: 8px;">'
            '</section>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cross-Project Fair Manual-vs-Auto Meta Summary</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #1f2933;
      --line: #d5dee8;
      --accent: #0f766e;
    }}
    body {{ margin: 0; padding: 1rem; background: var(--bg); color: var(--ink); font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }}
    header, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }}
    h1, h2 {{ margin-top: 0; }}
    .kpis {{ display: grid; gap: 0.7rem; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
    .kpi {{ border: 1px solid var(--line); border-radius: 8px; background: #fbfdff; padding: 0.65rem; }}
    .kpi .label {{ color: #4a5b70; font-size: 0.85rem; }}
    .kpi .value {{ color: var(--accent); font-size: 1.3rem; font-weight: 700; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border: 1px solid var(--line); padding: 0.45rem; text-align: left; vertical-align: top; }}
    th {{ background: #e9eff6; }}
    .muted {{ color: #5b6b80; }}
  </style>
</head>
<body>
  <header>
    <h1>Cross-Project Fair Manual-vs-Auto Meta Summary</h1>
    <p>Generated at {escape(generated_at_utc)}.</p>
    <div class="kpis">
      <div class="kpi"><div class="label">Projects Processed</div><div class="value">{int(global_summary.get("n_projects", 0))}</div></div>
      <div class="kpi"><div class="label">Projects Succeeded</div><div class="value">{int(global_summary.get("n_projects_success", 0))}</div></div>
      <div class="kpi"><div class="label">Runs Summarized</div><div class="value">{int(global_summary.get("n_runs", 0))}</div></div>
      <div class="kpi"><div class="label">Mean Dice Diagonal</div><div class="value">{float(global_summary.get("dice_mean_diagonal", 0.0)):.4f}</div></div>
      <div class="kpi"><div class="label">Mean Pearson Diagonal</div><div class="value">{float(global_summary.get("pearson_mean_diagonal", 0.0)):.4f}</div></div>
    </div>
  </header>

  <section>
    <h2>Project Status</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Project</th>
            <th>Status</th>
            <th>Reason</th>
            <th>Eligible Runs</th>
            <th>Fair IDs</th>
            <th>Manual Meta</th>
            <th>Compare</th>
            <th>Project Report</th>
            <th>Compare Log</th>
          </tr>
        </thead>
        <tbody>
          {''.join(project_rows_html)}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Per-Run Metrics</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Project</th>
            <th>Run</th>
            <th>Dice Mean Diagonal</th>
            <th>Pearson Mean Diagonal</th>
            <th>Dice Baseline (all_analyses)</th>
            <th>Pearson Baseline (all_analyses)</th>
            <th>Dice Mean Off-Diagonal</th>
            <th>Pearson Mean Off-Diagonal</th>
            <th>N Rows</th>
            <th>N Cols</th>
          </tr>
        </thead>
        <tbody>
          {''.join(run_rows_html)}
        </tbody>
      </table>
    </div>
    <p class="muted">Primary summary metrics are mapped diagonal means. Off-diagonal excludes mapped diagonals and the all_analyses baseline row.</p>
  </section>
  {run_metrics_plot_html}
</body>
</html>
"""


def write_run_metrics_plots(
    output_root: Path,
    run_rows: list[dict[str, Any]],
    diagonal_rows: list[dict[str, Any]],
) -> tuple[Path | None, Path | None]:
    if not run_rows and not diagonal_rows:
        return None, None

    rows = sorted(run_rows, key=lambda item: (str(item["project_name"]), str(item["run"])))
    project_names_from_rows = {str(row.get("project_name", "")) for row in rows}
    project_names_from_diags = {str(row.get("project_name", "")) for row in diagonal_rows}
    unique_project_names = sorted(project_names_from_rows | project_names_from_diags)
    projects = unique_project_names
    fig_w = max(10, 1.6 * len(unique_project_names))
    unique_projects = sorted(set(projects))
    cmap = plt.get_cmap("tab20")
    project_to_color = {
        project: cmap(idx % cmap.N)
        for idx, project in enumerate(unique_projects)
    }
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=project_to_color[project], label=project)
        for project in unique_projects
    ]

    project_to_dice_vals: dict[str, list[float]] = {project: [] for project in unique_projects}
    project_to_pearson_vals: dict[str, list[float]] = {project: [] for project in unique_projects}
    if diagonal_rows:
        for row in diagonal_rows:
            project = str(row.get("project_name", ""))
            if project not in project_to_dice_vals:
                continue
            project_to_dice_vals[project].append(float(row.get("dice", 0.0)))
            project_to_pearson_vals[project].append(float(row.get("pearson_r", 0.0)))
    else:
        for row in rows:
            project = str(row.get("project_name", ""))
            project_to_dice_vals[project].append(float(row.get("dice_mean_diagonal", 0.0)))
            project_to_pearson_vals[project].append(float(row.get("pearson_mean_diagonal", 0.0)))

    project_baseline_dice: dict[str, float] = {}
    project_baseline_pearson: dict[str, float] = {}
    for project in unique_projects:
        dice_baselines = [
            float(row["all_analyses_dice"])
            for row in rows
            if str(row.get("project_name", "")) == project
            and row.get("all_analyses_dice") is not None
        ]
        pearson_baselines = [
            float(row["all_analyses_pearson"])
            for row in rows
            if str(row.get("project_name", "")) == project
            and row.get("all_analyses_pearson") is not None
        ]
        if dice_baselines:
            project_baseline_dice[project] = float(sum(dice_baselines) / len(dice_baselines))
        if pearson_baselines:
            project_baseline_pearson[project] = float(sum(pearson_baselines) / len(pearson_baselines))

    baseline_legend_handle = Line2D(
        [0], [0], color="#374151", linestyle="--", linewidth=1.5, label="Project baseline (all_analyses)"
    )
    mean_legend_handle = Line2D(
        [0], [0], color="#111827", linestyle="-", linewidth=2.2, label="Project mean (diagonal dots)"
    )

    positions = list(range(1, len(unique_projects) + 1))

    fig, ax = plt.subplots(figsize=(fig_w, 6))
    dice_data = [project_to_dice_vals[project] for project in unique_projects]
    bp = ax.boxplot(
        dice_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    for patch, project in zip(bp["boxes"], unique_projects):
        patch.set_facecolor(project_to_color[project])
        patch.set_alpha(0.22)
        patch.set_edgecolor(project_to_color[project])
        patch.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.5)

    for pos, project in zip(positions, unique_projects):
        vals = project_to_dice_vals[project]
        n = len(vals)
        if n == 1:
            xs = [pos]
        else:
            span = 0.26
            xs = [pos - span + (2 * span) * (idx / (n - 1)) for idx in range(n)]
        ax.scatter(
            xs,
            vals,
            s=30,
            color=project_to_color[project],
            edgecolor="#111827",
            linewidth=0.6,
            alpha=0.95,
            zorder=3,
        )

    ax.set_title("Dice Diagonal Values by Project", fontweight="bold")
    ax.set_ylabel("Dice")
    ax.set_xlabel("Project")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(unique_projects, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.3)
    for project in unique_projects:
        vals = project_to_dice_vals.get(project, [])
        if not vals:
            continue
        pos = positions[unique_projects.index(project)]
        project_mean = float(sum(vals) / len(vals))
        ax.hlines(
            project_mean,
            pos - 0.24,
            pos + 0.24,
            colors=["#111827"],
            linestyles="-",
            linewidth=2.2,
            alpha=1.0,
            zorder=4,
        )
    for project, baseline in project_baseline_dice.items():
        if project not in unique_projects:
            continue
        pos = positions[unique_projects.index(project)]
        ax.hlines(
            baseline,
            pos - 0.28,
            pos + 0.28,
            colors=[project_to_color[project]],
            linestyles="--",
            linewidth=2.2,
            alpha=1.0,
            zorder=4,
        )
    ax.legend(
        handles=legend_handles + [mean_legend_handle, baseline_legend_handle],
        title="Project",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    fig.tight_layout()
    dice_plot_path = output_root / "run_metrics_dice_plot.png"
    fig.savefig(dice_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(fig_w, 6))
    pearson_data = [project_to_pearson_vals[project] for project in unique_projects]
    bp = ax.boxplot(
        pearson_data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    for patch, project in zip(bp["boxes"], unique_projects):
        patch.set_facecolor(project_to_color[project])
        patch.set_alpha(0.22)
        patch.set_edgecolor(project_to_color[project])
        patch.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.5)

    for pos, project in zip(positions, unique_projects):
        vals = project_to_pearson_vals[project]
        n = len(vals)
        if n == 1:
            xs = [pos]
        else:
            span = 0.26
            xs = [pos - span + (2 * span) * (idx / (n - 1)) for idx in range(n)]
        ax.scatter(
            xs,
            vals,
            s=30,
            color=project_to_color[project],
            edgecolor="#111827",
            linewidth=0.6,
            alpha=0.95,
            zorder=3,
        )

    ax.set_title("Pearson Diagonal Values by Project", fontweight="bold")
    ax.set_ylabel("Pearson")
    ax.set_xlabel("Project")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(unique_projects, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.3)
    for project in unique_projects:
        vals = project_to_pearson_vals.get(project, [])
        if not vals:
            continue
        pos = positions[unique_projects.index(project)]
        project_mean = float(sum(vals) / len(vals))
        ax.hlines(
            project_mean,
            pos - 0.24,
            pos + 0.24,
            colors=["#111827"],
            linestyles="-",
            linewidth=2.2,
            alpha=1.0,
            zorder=4,
        )
    for project, baseline in project_baseline_pearson.items():
        if project not in unique_projects:
            continue
        pos = positions[unique_projects.index(project)]
        ax.hlines(
            baseline,
            pos - 0.28,
            pos + 0.28,
            colors=[project_to_color[project]],
            linestyles="--",
            linewidth=2.2,
            alpha=1.0,
            zorder=4,
        )
    ax.legend(
        handles=legend_handles + [mean_legend_handle, baseline_legend_handle],
        title="Project",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    fig.tight_layout()
    pearson_plot_path = output_root / "run_metrics_pearson_plot.png"
    fig.savefig(pearson_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return dice_plot_path, pearson_plot_path


def main() -> int:
    args = parse_args()
    projects_root = args.projects_root.expanduser().resolve()
    manual_nimads_base = args.manual_nimads_base.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    compare_script = args.compare_script.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not compare_script.exists():
        raise FileNotFoundError(f"Compare script not found: {compare_script}")
    if not manual_nimads_base.exists():
        raise FileNotFoundError(f"Manual NiMADS base not found: {manual_nimads_base}")

    try:
        estimator_args = json.loads(args.estimator_args)
        corrector_args = json.loads(args.corrector_args)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse estimator/corrector args JSON: {exc}") from exc

    project_dirs = project_dirs_from_args(projects_root, [str(value) for value in args.project])
    print(f"Discovered {len(project_dirs)} project(s) to evaluate.")

    project_results: list[ProjectResult] = []
    all_run_summary_rows: list[dict[str, Any]] = []
    all_diagonal_rows: list[dict[str, Any]] = []
    project_summary_rows: list[dict[str, Any]] = []

    for project_dir in project_dirs:
        project_name = project_dir.name
        print(f"\n{'=' * 90}")
        print(f"Project: {project_name}")
        print(f"{'=' * 90}")
        try:
            mapping_path = resolve_mapping_path(project_dir)
            mapping_pairs = load_mapping_pairs(mapping_path)
            manual_names = [manual_name for manual_name, _auto_name in mapping_pairs]
        except Exception as exc:
            reason = f"mapping setup failed: {exc}"
            print(f"[SKIP] {project_name}: {reason}")
            project_results.append(ProjectResult(project_name=project_name, status="skipped", reason=reason))
            continue

        run_records, skipped_runs = collect_run_records(project_dir)
        if skipped_runs:
            for skipped in skipped_runs:
                print(f"[SKIP RUN] {project_name}/{skipped['run_name']}: {skipped['reason']}")

        if not run_records:
            reason = "no eligible annotation-only runs after validation"
            print(f"[SKIP] {project_name}: {reason}")
            project_results.append(ProjectResult(project_name=project_name, status="skipped", reason=reason))
            continue

        fair_ids = set.intersection(*(record.effective_ids for record in run_records))
        if not fair_ids:
            reason = "project fair ID intersection is empty"
            print(f"[SKIP] {project_name}: {reason}")
            project_results.append(
                ProjectResult(
                    project_name=project_name,
                    status="skipped",
                    reason=reason,
                    run_count=len(run_records),
                )
            )
            continue

        project_report_dir = (project_dir / "reports" / "manual_vs_auto_meta_fair").resolve()
        fair_root = project_report_dir / "fair_manual_meta"
        fair_subset_dir = fair_root / "nimads_subset"
        fair_manual_analysis_base = fair_root / "manual_analysis"
        fair_logs_dir = fair_root / "logs"
        project_report_dir.mkdir(parents=True, exist_ok=True)
        fair_logs_dir.mkdir(parents=True, exist_ok=True)

        provenance_path = fair_root / "fair_id_provenance.json"
        provenance_payload = {
            "project": project_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_regex": ANNOTATION_ONLY_RUN_RE.pattern,
            "effective_id_rule": (
                "effective_ids = input_ids(search.pmids_file) "
                "intersect output_ids_with_points(coordinate_parsing_results.json)"
            ),
            "project_fair_rule": "fair_ids = intersection of run-effective ID sets",
            "run_records": [
                {
                    "run_name": record.run_name,
                    "run_dir": str(record.run_dir),
                    "yaml_path": str(record.yaml_path),
                    "pmids_file": str(record.pmids_file),
                    "n_input_ids": len(record.input_ids),
                    "n_output_ids_with_points": len(record.output_ids_with_points),
                    "n_effective_ids": len(record.effective_ids),
                }
                for record in run_records
            ],
            "skipped_runs": skipped_runs,
            "n_fair_ids": len(fair_ids),
            "fair_ids": sorted(fair_ids),
        }
        write_json(provenance_path, provenance_payload)

        manual_project_merged_dir = manual_nimads_base / project_name / "merged"
        manual_studyset_path = manual_project_merged_dir / "nimads_studyset.json"
        manual_annotation_path = manual_project_merged_dir / "nimads_annotation.json"
        if not manual_studyset_path.exists() or not manual_annotation_path.exists():
            reason = (
                "manual merged NiMADS missing: "
                f"{manual_studyset_path} and/or {manual_annotation_path}"
            )
            print(f"[SKIP] {project_name}: {reason}")
            project_results.append(
                ProjectResult(
                    project_name=project_name,
                    status="skipped",
                    reason=reason,
                    report_dir=project_report_dir,
                    run_count=len(run_records),
                    fair_id_count=len(fair_ids),
                )
            )
            continue

        filtered_studyset_path = fair_subset_dir / "nimads_studyset.json"
        filtered_annotation_path = fair_subset_dir / "nimads_annotation.json"
        subset_stats_path = fair_root / "subset_stats.json"

        try:
            if args.force or not (filtered_studyset_path.exists() and filtered_annotation_path.exists()):
                subset_stats = build_filtered_manual_nimads_subset(
                    manual_studyset_path=manual_studyset_path,
                    manual_annotation_path=manual_annotation_path,
                    include_pmids=fair_ids,
                    output_studyset_path=filtered_studyset_path,
                    output_annotation_path=filtered_annotation_path,
                )
                write_json(subset_stats_path, subset_stats)
                print(
                    f"[OK] Built filtered manual subset for {project_name}: "
                    f"studies={subset_stats['n_filtered_studies']} "
                    f"analyses={subset_stats['n_kept_analysis_ids']}"
                )
            else:
                print(f"[OK] Reusing cached filtered manual subset for {project_name}")
        except Exception as exc:
            reason = f"failed building filtered manual subset: {exc}"
            print(f"[FAIL] {project_name}: {reason}")
            project_results.append(
                ProjectResult(
                    project_name=project_name,
                    status="failed",
                    reason=reason,
                    report_dir=project_report_dir,
                    run_count=len(run_records),
                    fair_id_count=len(fair_ids),
                )
            )
            continue

        manual_meta_status = "not_run"
        try:
            manual_meta_status = run_manual_meta_for_project(
                project_name=project_name,
                manual_names=manual_names,
                filtered_studyset_path=filtered_studyset_path,
                filtered_annotation_path=filtered_annotation_path,
                manual_analysis_base=fair_manual_analysis_base,
                map_filename=DEFAULT_MANUAL_ANALYSIS_MAP_FILENAME,
                estimator=str(args.estimator),
                estimator_args=estimator_args,
                corrector=str(args.corrector),
                corrector_args=corrector_args,
                force=bool(args.force),
            )
            print(f"[OK] Manual fair meta for {project_name}: {manual_meta_status}")
        except Exception as exc:
            reason = f"manual fair meta failed: {exc}"
            print(f"[FAIL] {project_name}: {reason}")
            project_results.append(
                ProjectResult(
                    project_name=project_name,
                    status="failed",
                    reason=reason,
                    report_dir=project_report_dir,
                    run_count=len(run_records),
                    fair_id_count=len(fair_ids),
                    manual_meta_status="failed",
                )
            )
            continue

        compare_log_path = fair_logs_dir / "compare_meta_to_benchmark.log"
        compare_status, compare_return_code = run_compare_meta(
            compare_script=compare_script,
            project_dir=project_dir,
            run_records=run_records,
            manual_analysis_base=fair_manual_analysis_base,
            output_dir=project_report_dir,
            log_path=compare_log_path,
        )
        print(
            f"[{compare_status.upper()}] compare_meta_to_benchmark for {project_name} "
            f"(return_code={compare_return_code})"
        )

        run_summary_path = project_report_dir / "tables" / "run_summary.csv"
        diag_summary_path = project_report_dir / "tables" / "diagonal_metrics.csv"
        report_html_path = project_report_dir / "manual_vs_auto_meta_report.html"

        if compare_status == "success" and run_summary_path.exists():
            run_summary_rows = read_run_summary_rows(
                run_summary_path,
                project_name=project_name,
                project_report_dir=project_report_dir,
                mapping_pairs=mapping_pairs,
            )
            all_run_summary_rows.extend(run_summary_rows)
            all_diagonal_rows.extend(
                read_diagonal_metric_rows(
                    diagonal_metrics_path=diag_summary_path,
                    project_name=project_name,
                )
            )

        status = "success" if compare_status == "success" else "failed"
        reason = "completed" if status == "success" else "compare_meta_to_benchmark failed"

        project_results.append(
            ProjectResult(
                project_name=project_name,
                status=status,
                reason=reason,
                report_dir=project_report_dir,
                report_html_path=report_html_path if report_html_path.exists() else None,
                run_count=len(run_records),
                fair_id_count=len(fair_ids),
                manual_meta_status=manual_meta_status,
                compare_status=compare_status,
                compare_return_code=compare_return_code,
                compare_log_path=compare_log_path,
                project_run_summary_path=run_summary_path if run_summary_path.exists() else None,
                project_diag_summary_path=diag_summary_path if diag_summary_path.exists() else None,
            )
        )

    for result in project_results:
        run_subset = [row for row in all_run_summary_rows if row["project_name"] == result.project_name]
        if run_subset:
            dice_mean = sum(float(row["dice_mean_diagonal"]) for row in run_subset) / len(run_subset)
            pearson_mean = sum(float(row["pearson_mean_diagonal"]) for row in run_subset) / len(run_subset)
        else:
            dice_mean = 0.0
            pearson_mean = 0.0
        project_summary_rows.append(
            {
                "project_name": result.project_name,
                "status": result.status,
                "reason": result.reason,
                "run_count": result.run_count,
                "fair_id_count": result.fair_id_count,
                "manual_meta_status": result.manual_meta_status,
                "compare_status": result.compare_status,
                "compare_return_code": result.compare_return_code if result.compare_return_code is not None else "",
                "project_report_dir": str(result.report_dir) if result.report_dir else "",
                "project_report_html": str(result.report_html_path) if result.report_html_path else "",
                "compare_log_path": str(result.compare_log_path) if result.compare_log_path else "",
                "dice_mean_diagonal_avg": f"{dice_mean:.6f}",
                "pearson_mean_diagonal_avg": f"{pearson_mean:.6f}",
            }
        )

    run_rows_out: list[dict[str, Any]] = []
    for row in all_run_summary_rows:
        run_rows_out.append(
            {
                "project_name": row["project_name"],
                "run": row["run"],
                "dice_mean_diagonal": f"{float(row['dice_mean_diagonal']):.6f}",
                "pearson_mean_diagonal": f"{float(row['pearson_mean_diagonal']):.6f}",
                "dice_mean_off_diagonal": (
                    f"{float(row['dice_mean_off_diagonal']):.6f}"
                    if row.get("dice_mean_off_diagonal") is not None
                    else ""
                ),
                "pearson_mean_off_diagonal": (
                    f"{float(row['pearson_mean_off_diagonal']):.6f}"
                    if row.get("pearson_mean_off_diagonal") is not None
                    else ""
                ),
                "all_analyses_dice": (
                    f"{float(row['all_analyses_dice']):.6f}"
                    if row.get("all_analyses_dice") is not None
                    else ""
                ),
                "all_analyses_pearson": (
                    f"{float(row['all_analyses_pearson']):.6f}"
                    if row.get("all_analyses_pearson") is not None
                    else ""
                ),
                "n_rows": int(row["n_rows"]),
                "n_cols": int(row["n_cols"]),
            }
        )

    write_csv(
        output_root / "project_status.csv",
        project_summary_rows,
        [
            "project_name",
            "status",
            "reason",
            "run_count",
            "fair_id_count",
            "manual_meta_status",
            "compare_status",
            "compare_return_code",
            "project_report_dir",
            "project_report_html",
            "compare_log_path",
            "dice_mean_diagonal_avg",
            "pearson_mean_diagonal_avg",
        ],
    )
    write_csv(
        output_root / "run_metrics.csv",
        run_rows_out,
        [
            "project_name",
            "run",
            "dice_mean_diagonal",
            "pearson_mean_diagonal",
            "dice_mean_off_diagonal",
            "pearson_mean_off_diagonal",
            "all_analyses_dice",
            "all_analyses_pearson",
            "n_rows",
            "n_cols",
        ],
    )

    run_metrics_dice_plot_path, run_metrics_pearson_plot_path = write_run_metrics_plots(
        output_root=output_root,
        run_rows=all_run_summary_rows,
        diagonal_rows=all_diagonal_rows,
    )

    n_runs = len(all_run_summary_rows)
    dice_global = (
        sum(float(row["dice_mean_diagonal"]) for row in all_run_summary_rows) / n_runs if n_runs else 0.0
    )
    pearson_global = (
        sum(float(row["pearson_mean_diagonal"]) for row in all_run_summary_rows) / n_runs
        if n_runs
        else 0.0
    )

    project_rows_for_html: list[dict[str, Any]] = []
    for row in project_summary_rows:
        project_name = str(row["project_name"])
        result = next(item for item in project_results if item.project_name == project_name)
        row_copy = dict(row)
        row_copy["project_report_link"] = relative_link(
            output_root,
            result.report_html_path,
            "manual_vs_auto_meta_report.html",
        )
        row_copy["compare_log_link"] = relative_link(output_root, result.compare_log_path, "compare log")
        project_rows_for_html.append(row_copy)

    html = build_cross_project_html(
        output_root=output_root,
        generated_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        project_rows=project_rows_for_html,
        run_rows=all_run_summary_rows,
        global_summary={
            "n_projects": len(project_results),
            "n_projects_success": len([result for result in project_results if result.status == "success"]),
            "n_runs": n_runs,
            "dice_mean_diagonal": dice_global,
            "pearson_mean_diagonal": pearson_global,
        },
        run_metrics_dice_plot_path=run_metrics_dice_plot_path,
        run_metrics_pearson_plot_path=run_metrics_pearson_plot_path,
    )
    html_path = output_root / "cross_project_manual_vs_auto_meta_fair_report.html"
    html_path.write_text(html, encoding="utf-8")

    print("\nWrote outputs:")
    print(f"- {output_root / 'project_status.csv'}")
    print(f"- {output_root / 'run_metrics.csv'}")
    print(f"- {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

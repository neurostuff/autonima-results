#!/usr/bin/env python3
"""Compare automated meta-analysis maps against manual benchmark maps.

This script ports the generic notebook workflow into a reusable CLI and produces
CSV tables, PNG figures, and an HTML report.

Examples:
    python scripts/compare_meta_to_benchmark.py \
        --project-dir projects/cue_reactivity

    python scripts/compare_meta_to_benchmark.py \
        --project-dir projects/cue_reactivity \
        --run-dir v1 \
        --manual-meta-run manual_meta_v1

    python scripts/compare_meta_to_benchmark.py \
        --project-dir projects/cue_reactivity \
        --output-dir projects/cue_reactivity/reports/manual_vs_auto_meta_custom \
        --no-save-images
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


AGGREGATE_ANALYSIS_NAME_VARIANTS = (
    ("all_analyses", "all_studies", "all_abstract"),
    ("all_analyses", "all_search", "all_abstract_screened"),
    ("all_analyses", "all_studies"),
    # Single-name fallback. For an annotation-only run there is no search or abstract
    # screening stage, so all_analyses / all_studies / all_abstract would be the SAME
    # map; such runs therefore emit only all_analyses by design. Every variant above
    # requires >=2 maps, so those runs matched none, were skipped wholesale, and lost
    # all_analyses despite it existing -- which silently blanked the baseline in the
    # fair cross-project report (dementia and executive_function both showed no
    # baseline and were dropped from the dice-delta-vs-baseline plot). Emitting one
    # aggregate is correct, not a misconfiguration, so a 1-name set must be honoured.
    ("all_analyses",),
)
MANUAL_META_MARKERS = ("manual_meta", "manual-meta", "manual metas", "manual_metas")
ANALYSIS_ID_RE = re.compile(r"^(?P<pmid>.+?)_analysis_(?P<index>\d+)$")


@dataclass(frozen=True)
class RunInfo:
    name: str
    run_dir: Path
    meta_results_dir: Path


@dataclass(frozen=True)
class MappingPair:
    manual_name: str
    auto_name: str
    manual_path: Path
    auto_paths: dict[str, Path]


@dataclass
class ComparisonResults:
    run_names: list[str]
    manual_order: list[str]
    auto_order: list[str]
    run_aggregate_names_by_run: dict[str, list[str]]
    dice_matrices: dict[str, pd.DataFrame]
    pearson_matrices: dict[str, pd.DataFrame]
    diag_df: pd.DataFrame
    summary_df: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-dir",
        required=True,
        type=Path,
        help="Path to project directory (contains runs and mapping file).",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=None,
        help=(
            "Path to mapping JSON. Defaults to {project_dir}/nmb_mappings.json, "
            "falling back to {project_dir}/nmb_mapping.json."
        ),
    )
    parser.add_argument(
        "--manual-analysis-base",
        type=Path,
        default=Path("/home/zorro/repos/neurometabench/analysis"),
        help="Root containing manual benchmark maps by project.",
    )
    parser.add_argument(
        "--manual-nimads-base",
        type=Path,
        default=Path("/home/zorro/repos/neurometabench/data/nimads"),
        help="Root containing merged manual benchmark nimads datasets by project.",
    )
    parser.add_argument(
        "--map-filename",
        type=str,
        default="z.nii.gz",
        help="Map filename expected in each analysis directory.",
    )
    parser.add_argument(
        "--corrected-map-filename",
        type=str,
        default="z_corr-FDR_method-indep.nii.gz",
        help="FDR-corrected map filename used for orthogonal stat-map snapshots.",
    )
    parser.add_argument(
        "--stat-map-display-mode",
        type=str,
        default="ortho",
        help="Display mode for corrected stat-map snapshots (e.g., ortho, x, y, z).",
    )
    parser.add_argument(
        "--stat-map-cut-coords",
        nargs="+",
        type=float,
        default=[0.0, 0.0, 0.0],
        help=(
            "Cut coordinates for corrected stat-map snapshots. Provide one or more values. "
            "For display_mode=ortho, exactly three values are required."
        ),
    )
    parser.add_argument(
        "--dice-threshold",
        type=float,
        default=1.96,
        help="Threshold applied before Dice computation.",
    )
    parser.add_argument(
        "--show-figures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display figures interactively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report output dir. Defaults to {project_dir}/reports/manual_vs_auto_meta.",
    )
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write PNG plots to output images dir.",
    )
    parser.add_argument(
        "--save-tables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write CSV tables to output tables dir.",
    )
    parser.add_argument(
        "--save-html",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write HTML report to output dir.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help=(
            "Run directory to include (repeatable). If relative, resolved under --project-dir. "
            "If omitted, runs are auto-discovered under project dir."
        ),
    )
    parser.add_argument(
        "--meta-results-subpath",
        type=Path,
        default=Path("outputs/meta_analysis_results"),
        help="Subpath under each run directory containing analysis maps.",
    )
    parser.add_argument(
        "--manual-meta-run",
        action="append",
        default=[],
        help=(
            "Run name to treat as manual-meta (repeatable). If omitted, heuristic run-name "
            "matching is used."
        ),
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def manual_name_candidates(manual_name: str) -> list[str]:
    candidates = [str(manual_name)]
    if str(manual_name).endswith(".txt"):
        candidates.append(str(manual_name)[:-4])

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def heuristic_is_manual_meta_run(run_name: str) -> bool:
    run_name_lc = str(run_name).lower()
    if run_name_lc.startswith("manual"):
        return True
    return any(marker in run_name_lc for marker in MANUAL_META_MARKERS)


def normalize_note_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def parse_pmid_from_analysis_id(analysis_id: str) -> str | None:
    analysis_text = str(analysis_id).strip()
    if not analysis_text:
        return None

    match = ANALYSIS_ID_RE.match(analysis_text)
    if match:
        return match.group("pmid")

    if "_" in analysis_text:
        candidate = analysis_text.split("_", 1)[0].strip()
        return candidate or None

    return None


def resolve_project_dir(project_dir: Path) -> Path:
    resolved = project_dir.expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {resolved}")
    return resolved


def resolve_mapping_path(project_dir: Path, mapping_path: Path | None) -> Path:
    if mapping_path is not None:
        candidate = mapping_path.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Mapping file not found: {candidate}")
        return candidate

    candidates = [
        project_dir / "nmb_mappings.json",
        project_dir / "nmb_mapping.json",
    ]
    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is None:
        searched = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Could not locate mapping file. Searched:\n{searched}")
    return resolved


def resolve_output_dir(project_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return (project_dir / "reports" / "manual_vs_auto_meta").resolve()


def resolve_run_infos(
    project_dir: Path,
    meta_results_subpath: Path,
    run_dir_args: list[str],
) -> list[RunInfo]:
    run_dirs: list[Path] = []
    if run_dir_args:
        seen: set[Path] = set()
        for value in run_dir_args:
            run_dir = Path(value).expanduser()
            if not run_dir.is_absolute():
                run_dir = project_dir / run_dir
            run_dir = run_dir.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            run_dirs.append(run_dir)
    else:
        run_dirs = sorted(
            [
                entry
                for entry in project_dir.iterdir()
                if entry.is_dir() and (entry / meta_results_subpath).is_dir()
            ],
            key=lambda path: path.name,
        )

    if not run_dirs:
        raise RuntimeError(
            f"No run directories found under {project_dir} with subpath {meta_results_subpath}."
        )

    run_infos: list[RunInfo] = []
    seen_names: set[str] = set()
    for run_dir in run_dirs:
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        meta_results_dir = run_dir / meta_results_subpath
        if not meta_results_dir.is_dir():
            raise FileNotFoundError(
                f"Run directory {run_dir} missing expected meta results dir: {meta_results_dir}"
            )
        run_name = run_dir.name
        if run_name in seen_names:
            raise ValueError(
                f"Duplicate run name detected ({run_name}). Provide distinct run dirs."
            )
        seen_names.add(run_name)
        run_infos.append(
            RunInfo(name=run_name, run_dir=run_dir, meta_results_dir=meta_results_dir)
        )

    return run_infos


def load_mappings(mapping_path: Path) -> dict[str, str]:
    with mapping_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Mapping file must be a non-empty JSON object: {mapping_path}")

    raw_mappings: dict[str, object]
    if "annotation_mappings" in payload:
        nested = payload.get("annotation_mappings")
        if not isinstance(nested, dict):
            raise ValueError(
                f"Invalid mapping format at {mapping_path}: "
                "expected 'annotation_mappings' to be a JSON object"
            )
        raw_mappings = nested
    else:
        raw_mappings = {
            key: value
            for key, value in payload.items()
            if str(key).strip() != "meta_pmid"
        }

    mappings: dict[str, str] = {}
    for manual_name_raw, auto_name_raw in raw_mappings.items():
        if isinstance(auto_name_raw, (dict, list)):
            continue
        manual_name = str(manual_name_raw).strip()
        auto_name = str(auto_name_raw).strip()
        if not manual_name or not auto_name:
            continue
        mappings[manual_name] = auto_name
    if not mappings:
        raise ValueError(f"Mapping file did not contain any usable mappings: {mapping_path}")
    return mappings


def load_analysis_to_pmid_map_from_studyset(studyset_path: Path) -> dict[str, str]:
    with studyset_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    studies = payload.get("studies", [])
    if not isinstance(studies, list):
        raise ValueError(f"Invalid nimads_studyset format (missing list 'studies'): {studyset_path}")

    analysis_to_pmid: dict[str, str] = {}
    for study in studies:
        if not isinstance(study, dict):
            continue
        pmid = str(study.get("pmid") or study.get("id") or "").strip()
        analyses = study.get("analyses", [])
        if not isinstance(analyses, list):
            continue
        for analysis in analyses:
            if not isinstance(analysis, dict):
                continue
            analysis_id = str(analysis.get("id") or "").strip()
            if analysis_id and pmid:
                analysis_to_pmid[analysis_id] = pmid

    return analysis_to_pmid


def count_annotation_membership(
    annotation_path: Path,
    studyset_path: Path,
    target_annotation_names: list[str],
) -> dict[str, tuple[int, int]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    notes = payload.get("notes", [])
    if not isinstance(notes, list):
        raise ValueError(f"Invalid nimads_annotation format (missing list 'notes'): {annotation_path}")

    analysis_to_pmid = load_analysis_to_pmid_map_from_studyset(studyset_path)
    target_by_normalized_key: dict[str, list[str]] = defaultdict(list)
    for target_name in target_annotation_names:
        target_by_normalized_key[normalize_note_key(target_name)].append(target_name)

    analysis_ids_by_annotation: dict[str, set[str]] = {
        target_name: set() for target_name in target_annotation_names
    }
    pmids_by_annotation: dict[str, set[str]] = {
        target_name: set() for target_name in target_annotation_names
    }
    fallback_pmids_used: set[str] = set()

    for note_row in notes:
        if not isinstance(note_row, dict):
            continue

        analysis_id = str(note_row.get("analysis", "")).strip()
        if not analysis_id:
            continue

        note = note_row.get("note", {})
        if not isinstance(note, dict):
            continue

        true_keys = {
            normalize_note_key(key)
            for key, value in note.items()
            if bool(value)
        }
        if not true_keys:
            continue

        pmid = analysis_to_pmid.get(analysis_id, "")
        if not pmid:
            parsed = parse_pmid_from_analysis_id(analysis_id)
            pmid = parsed or ""
            if pmid:
                fallback_pmids_used.add(analysis_id)

        for normalized_key in true_keys:
            for target_name in target_by_normalized_key.get(normalized_key, []):
                analysis_ids_by_annotation[target_name].add(analysis_id)
                if pmid:
                    pmids_by_annotation[target_name].add(pmid)

    if fallback_pmids_used:
        preview = ", ".join(sorted(fallback_pmids_used)[:5])
        suffix = "" if len(fallback_pmids_used) <= 5 else f" ... (+{len(fallback_pmids_used)-5} more)"
        print(
            f"[WARN] {annotation_path}: {len(fallback_pmids_used)} analyses missing studyset PMID "
            f"mapping; used analysis-id parsing fallback. Examples: {preview}{suffix}"
        )

    return {
        target_name: (
            len(analysis_ids_by_annotation[target_name]),
            len(pmids_by_annotation[target_name]),
        )
        for target_name in target_annotation_names
    }


def load_available_annotation_names(annotation_path: Path) -> list[str]:
    with annotation_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    names: set[str] = set()
    note_keys = payload.get("note_keys", {})
    if isinstance(note_keys, dict):
        names.update(str(key) for key in note_keys.keys())

    for note_row in payload.get("notes", []):
        if not isinstance(note_row, dict):
            continue
        note = note_row.get("note", {})
        if isinstance(note, dict):
            names.update(str(key) for key in note.keys())

    return sorted(name for name in names if str(name).strip())


def compute_automated_annotation_counts(
    mapping_pairs: list[MappingPair],
    included_run_infos: list[RunInfo],
    manual_meta_by_run: dict[str, bool],
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    mapped_auto_names = [pair.auto_name for pair in mapping_pairs]
    mapped_auto_name_set = set(mapped_auto_names)
    manual_name_by_auto_name = {
        pair.auto_name: pair.manual_name for pair in mapping_pairs
    }
    annotation_order: list[str] = []

    automated_run_names = [
        run_info.name for run_info in included_run_infos if not manual_meta_by_run[run_info.name]
    ]
    rows: list[dict[str, object]] = []
    manual_column_labels_by_run: dict[str, dict[str, str]] = {}

    for run_info in included_run_infos:
        run_name = run_info.name
        if manual_meta_by_run[run_name]:
            continue

        outputs_dir = run_info.meta_results_dir.parent
        annotation_path = outputs_dir / "nimads_annotation.json"
        studyset_path = outputs_dir / "nimads_studyset.json"

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Run {run_name} missing annotation file for count extraction: {annotation_path}"
            )
        if not studyset_path.exists():
            raise FileNotFoundError(
                f"Run {run_name} missing studyset file for PMID extraction: {studyset_path}"
            )

        available_annotations = load_available_annotation_names(annotation_path)
        all_annotation_names = [
            name
            for name in available_annotations
            if normalize_note_key(name).startswith("all_") or normalize_note_key(name) == "all"
        ]
        target_annotation_names = mapped_auto_names + [
            name for name in all_annotation_names if name not in mapped_auto_name_set
        ]
        for annotation_name in target_annotation_names:
            if annotation_name not in annotation_order:
                annotation_order.append(annotation_name)

        counts = count_annotation_membership(
            annotation_path=annotation_path,
            studyset_path=studyset_path,
            target_annotation_names=target_annotation_names,
        )

        run_labels: dict[str, str] = {}
        for annotation_name in target_annotation_names:
            n_analyses, n_unique_pmids = counts.get(annotation_name, (0, 0))
            manual_name = manual_name_by_auto_name.get(annotation_name, "")
            rows.append(
                {
                    "run": run_name,
                    "manual_name": manual_name,
                    "auto_name": annotation_name,
                    "n_analyses": int(n_analyses),
                    "n_unique_pmids": int(n_unique_pmids),
                    "is_mapped": bool(manual_name),
                }
            )
            if manual_name:
                run_labels[manual_name] = manual_name

        manual_column_labels_by_run[run_name] = run_labels

    counts_df = pd.DataFrame(rows)
    if not counts_df.empty:
        counts_df["run"] = pd.Categorical(
            counts_df["run"], categories=automated_run_names, ordered=True
        )
        counts_df["auto_name"] = pd.Categorical(
            counts_df["auto_name"], categories=annotation_order, ordered=True
        )
        counts_df = counts_df.sort_values(["auto_name", "run"]).reset_index(drop=True)

    return counts_df, manual_column_labels_by_run


def compute_manual_benchmark_totals(project_name: str, manual_nimads_base: Path) -> pd.DataFrame:
    merged_dir = manual_nimads_base / project_name / "merged"
    annotation_path = merged_dir / "nimads_annotation.json"
    studyset_path = merged_dir / "nimads_studyset.json"

    missing_files = [path for path in (annotation_path, studyset_path) if not path.exists()]
    if missing_files:
        missing_str = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Manual benchmark counts requested, but missing NimADS files: " + missing_str
        )

    annotation_names = load_available_annotation_names(annotation_path)
    counts_by_annotation = count_annotation_membership(
        annotation_path=annotation_path,
        studyset_path=studyset_path,
        target_annotation_names=annotation_names,
    )

    analysis_to_pmid = load_analysis_to_pmid_map_from_studyset(studyset_path)
    total_n_analyses = len(analysis_to_pmid)
    total_n_unique_pmids = len({pmid for pmid in analysis_to_pmid.values() if pmid})

    rows: list[dict[str, object]] = []
    for annotation_name in annotation_names:
        n_analyses, n_unique_pmids = counts_by_annotation.get(annotation_name, (0, 0))
        rows.append(
            {
                "project": project_name,
                "annotation": annotation_name,
                "n_analyses": int(n_analyses),
                "n_unique_pmids": int(n_unique_pmids),
                "is_dataset_total": False,
                "annotation_path": str(annotation_path),
                "studyset_path": str(studyset_path),
            }
        )

    rows.append(
        {
            "project": project_name,
            "annotation": "__dataset_total__",
            "n_analyses": int(total_n_analyses),
            "n_unique_pmids": int(total_n_unique_pmids),
            "is_dataset_total": True,
            "annotation_path": str(annotation_path),
            "studyset_path": str(studyset_path),
        }
    )

    return pd.DataFrame(rows)


def build_mapping_pairs(
    mappings: dict[str, str],
    run_infos: list[RunInfo],
    manual_analysis_base: Path,
    project_name: str,
    map_filename: str,
) -> tuple[list[MappingPair], dict[str, list[str]], pd.DataFrame]:
    mapping_pairs: list[MappingPair] = []
    manual_missing_errors: list[str] = []
    availability_rows: list[dict[str, object]] = []

    run_missing_pairs = {run_info.name: [] for run_info in run_infos}

    for manual_name, auto_name in mappings.items():
        candidates = manual_name_candidates(manual_name)

        manual_path: Path | None = None
        manual_paths_checked: list[Path] = []
        for candidate in candidates:
            candidate_path = manual_analysis_base / project_name / candidate / map_filename
            manual_paths_checked.append(candidate_path)
            if candidate_path.exists():
                manual_path = candidate_path
                break

        if manual_path is None:
            checked_str = ", ".join(str(path) for path in manual_paths_checked)
            manual_missing_errors.append(
                f"Missing manual map for mapping {manual_name} -> {auto_name}. Checked: {checked_str}"
            )

        auto_paths: dict[str, Path] = {}
        for run_info in run_infos:
            auto_file = run_info.meta_results_dir / auto_name / map_filename
            auto_paths[run_info.name] = auto_file
            if not auto_file.exists():
                run_missing_pairs[run_info.name].append(f"{manual_name} -> {auto_name}")

        if manual_path is None:
            # This row will never be used because we fail below, but keeps the type stable.
            manual_path = manual_paths_checked[0]

        mapping_pairs.append(
            MappingPair(
                manual_name=manual_name,
                auto_name=auto_name,
                manual_path=manual_path,
                auto_paths=auto_paths,
            )
        )

        row: dict[str, object] = {
            "manual_name": manual_name,
            "auto_name": auto_name,
            "manual_exists": manual_path.exists(),
        }
        for run_info in run_infos:
            row[f"run::{run_info.name}"] = auto_paths[run_info.name].exists()
        availability_rows.append(row)

    if manual_missing_errors:
        error_lines = "\n".join(f"- {error}" for error in manual_missing_errors)
        raise FileNotFoundError(
            "Strict manual validation failed. Every mapped manual output must exist.\n"
            f"{error_lines}"
        )

    availability_df = pd.DataFrame(availability_rows)
    return mapping_pairs, run_missing_pairs, availability_df


def filter_complete_runs(
    run_infos: list[RunInfo],
    run_missing_pairs: dict[str, list[str]],
) -> tuple[list[RunInfo], dict[str, list[str]]]:
    included: list[RunInfo] = []
    skipped: dict[str, list[str]] = {}

    for run_info in run_infos:
        missing_pairs = run_missing_pairs[run_info.name]
        if missing_pairs:
            skipped[run_info.name] = missing_pairs
            continue
        included.append(run_info)

    if not included:
        raise RuntimeError(
            "No runs have a complete mapped set of z maps. At least one run must contain all mapped outputs."
        )

    return included, skipped


def classify_manual_meta_runs(
    included_run_infos: list[RunInfo],
    explicit_manual_meta_runs: list[str],
) -> dict[str, bool]:
    included_names = [run_info.name for run_info in included_run_infos]
    included_set = set(included_names)

    if explicit_manual_meta_runs:
        explicit_set = set(explicit_manual_meta_runs)
        unknown = sorted(explicit_set - included_set)
        if unknown:
            raise ValueError(
                "Unknown --manual-meta-run values (not in included runs): "
                + ", ".join(unknown)
            )
        return {run_name: run_name in explicit_set for run_name in included_names}

    return {run_name: heuristic_is_manual_meta_run(run_name) for run_name in included_names}


def collect_aggregate_paths(
    included_run_infos: list[RunInfo],
    manual_meta_by_run: dict[str, bool],
    map_filename: str,
) -> tuple[dict[str, dict[str, Path]], dict[str, str]]:
    run_aggregate_paths: dict[str, dict[str, Path]] = {
        run_info.name: {} for run_info in included_run_infos
    }
    skipped_aggregate_runs: dict[str, str] = {}

    for run_info in included_run_infos:
        run_name = run_info.name
        if manual_meta_by_run[run_name]:
            continue

        matched_variant: tuple[str, ...] | None = None
        for variant in sorted(AGGREGATE_ANALYSIS_NAME_VARIANTS, key=len, reverse=True):
            if all(
                (run_info.meta_results_dir / aggregate_name / map_filename).exists()
                for aggregate_name in variant
            ):
                matched_variant = variant
                break

        if matched_variant is None:
            expected_variants = [
                " + ".join(variant) for variant in AGGREGATE_ANALYSIS_NAME_VARIANTS
            ]
            skipped_aggregate_runs[run_name] = (
                "missing aggregate analysis maps. Expected one of: "
                f"{'; '.join(expected_variants)}"
            )
            continue

        for aggregate_name in matched_variant:
            run_aggregate_paths[run_name][aggregate_name] = (
                run_info.meta_results_dir / aggregate_name / map_filename
            )

    return run_aggregate_paths, skipped_aggregate_runs


def load_maps_and_vectors(
    mapping_pairs: list[MappingPair],
    run_names: list[str],
    run_aggregate_paths: dict[str, dict[str, Path]],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], tuple[int, ...], int]:
    manual_data: dict[str, np.ndarray] = {}
    auto_data_by_run: dict[str, dict[str, np.ndarray]] = {run_name: {} for run_name in run_names}
    shape_records: list[tuple[str, tuple[int, ...]]] = []

    for pair in mapping_pairs:
        manual_name = pair.manual_name
        auto_name = pair.auto_name

        manual_arr = nib.load(str(pair.manual_path)).get_fdata()
        manual_data[manual_name] = manual_arr
        shape_records.append((f"manual::{manual_name}", manual_arr.shape))

        for run_name in run_names:
            auto_path = pair.auto_paths[run_name]
            auto_arr = nib.load(str(auto_path)).get_fdata()
            auto_data_by_run[run_name][auto_name] = auto_arr
            shape_records.append((f"{run_name}::{auto_name}", auto_arr.shape))

    for run_name in run_names:
        for aggregate_name, aggregate_path in run_aggregate_paths[run_name].items():
            aggregate_arr = nib.load(str(aggregate_path)).get_fdata()
            auto_data_by_run[run_name][aggregate_name] = aggregate_arr
            shape_records.append((f"{run_name}::{aggregate_name}", aggregate_arr.shape))

    unique_shapes = sorted({shape for _, shape in shape_records})
    if len(unique_shapes) != 1:
        shape_lines = "\n".join(f"- {name}: {shape}" for name, shape in shape_records)
        raise ValueError(
            "All maps must have identical shapes before comparison. Found mismatched shapes:\n"
            f"{shape_lines}"
        )

    common_shape = unique_shapes[0]
    mask = np.ones(common_shape, dtype=bool)

    for arr in manual_data.values():
        mask &= np.isfinite(arr)
    for run_data in auto_data_by_run.values():
        for arr in run_data.values():
            mask &= np.isfinite(arr)

    n_valid_voxels = int(mask.sum())
    if n_valid_voxels == 0:
        raise ValueError("No common finite voxels remained after masking.")

    manual_vectors = {name: arr[mask].ravel() for name, arr in manual_data.items()}
    auto_vectors_by_run = {
        run_name: {name: arr[mask].ravel() for name, arr in run_data.items()}
        for run_name, run_data in auto_data_by_run.items()
    }

    return manual_vectors, auto_vectors_by_run, common_shape, n_valid_voxels


def compute_dice(vec_a: np.ndarray, vec_b: np.ndarray, threshold: float) -> float:
    binary_a = vec_a > threshold
    binary_b = vec_b > threshold
    intersection = np.sum(binary_a & binary_b)
    volume_sum = np.sum(binary_a) + np.sum(binary_b)
    if volume_sum == 0:
        return 0.0
    return float((2.0 * intersection) / volume_sum)


def compute_pearson(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size < 2 or vec_b.size < 2:
        return float("nan")
    if np.all(vec_a == vec_a[0]) or np.all(vec_b == vec_b[0]):
        return float("nan")
    return float(pearsonr(vec_a, vec_b)[0])


def compute_comparison_results(
    mapping_pairs: list[MappingPair],
    run_names: list[str],
    manual_vectors: dict[str, np.ndarray],
    auto_vectors_by_run: dict[str, dict[str, np.ndarray]],
    run_aggregate_paths: dict[str, dict[str, Path]],
    dice_threshold: float,
) -> ComparisonResults:
    manual_order = [pair.manual_name for pair in mapping_pairs]
    auto_order = [pair.auto_name for pair in mapping_pairs]

    run_aggregate_names_by_run = {
        run_name: list(run_aggregate_paths[run_name].keys())
        for run_name in run_names
    }

    dice_matrices: dict[str, pd.DataFrame] = {}
    pearson_matrices: dict[str, pd.DataFrame] = {}

    for run_name in run_names:
        run_auto_vectors = auto_vectors_by_run[run_name]
        matrix_rows = auto_order + run_aggregate_names_by_run[run_name]
        matrix_columns = manual_order

        row_vectors = {name: run_auto_vectors[name] for name in matrix_rows}
        column_vectors = {name: manual_vectors[name] for name in manual_order}

        dice_df = pd.DataFrame(index=matrix_rows, columns=matrix_columns, dtype=float)
        pearson_df = pd.DataFrame(index=matrix_rows, columns=matrix_columns, dtype=float)

        for row_name in matrix_rows:
            row_vec = row_vectors[row_name]
            for column_name in matrix_columns:
                compare_vec = column_vectors[column_name]
                dice_df.loc[row_name, column_name] = compute_dice(
                    row_vec, compare_vec, dice_threshold
                )
                pearson_df.loc[row_name, column_name] = compute_pearson(row_vec, compare_vec)

        dice_matrices[run_name] = dice_df
        pearson_matrices[run_name] = pearson_df

    # Diagonal metrics stay restricted to mapped (auto_name, manual_name) pairs only.
    diag_rows: list[dict[str, object]] = []
    for run_name in run_names:
        for pair in mapping_pairs:
            diag_rows.append(
                {
                    "run": run_name,
                    "manual_name": pair.manual_name,
                    "auto_name": pair.auto_name,
                    "dice": dice_matrices[run_name].loc[pair.auto_name, pair.manual_name],
                    "pearson_r": pearson_matrices[run_name].loc[pair.auto_name, pair.manual_name],
                }
            )

    diag_df = pd.DataFrame(diag_rows)
    diag_df["run"] = pd.Categorical(diag_df["run"], categories=run_names, ordered=True)
    diag_df["auto_name"] = pd.Categorical(
        diag_df["auto_name"], categories=auto_order, ordered=True
    )
    diag_df = diag_df.sort_values(["auto_name", "run"]).reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for run_name in run_names:
        dice_values = dice_matrices[run_name].to_numpy().ravel()
        pearson_values = pearson_matrices[run_name].to_numpy().ravel()

        diag_dice = [
            dice_matrices[run_name].loc[pair.auto_name, pair.manual_name]
            for pair in mapping_pairs
        ]
        diag_pearson = [
            pearson_matrices[run_name].loc[pair.auto_name, pair.manual_name]
            for pair in mapping_pairs
        ]

        summary_rows.append(
            {
                "run": run_name,
                "n_rows": int(dice_matrices[run_name].shape[0]),
                "n_cols": int(dice_matrices[run_name].shape[1]),
                "dice_mean_full": float(np.nanmean(dice_values)),
                "dice_mean_diagonal": float(np.nanmean(diag_dice)),
                "pearson_mean_full": float(np.nanmean(pearson_values)),
                "pearson_mean_diagonal": float(np.nanmean(diag_pearson)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).set_index("run")

    return ComparisonResults(
        run_names=run_names,
        manual_order=manual_order,
        auto_order=auto_order,
        run_aggregate_names_by_run=run_aggregate_names_by_run,
        dice_matrices=dice_matrices,
        pearson_matrices=pearson_matrices,
        diag_df=diag_df,
        summary_df=summary_df,
    )


def relabel_manual_columns(
    df: pd.DataFrame,
    run_name: str,
    manual_column_labels_by_run: dict[str, dict[str, str]],
) -> pd.DataFrame:
    labels = manual_column_labels_by_run.get(run_name, {})
    if not labels:
        return df
    rename_map = {column_name: labels.get(column_name, column_name) for column_name in df.columns}
    return df.rename(columns=rename_map)


def set_heatmap_xtick_alignment(ax: plt.Axes, labels: list[str]) -> None:
    # Keep xticks centered on each heatmap column and anchor rotated labels to the tick.
    tick_positions = np.arange(len(labels), dtype=float) + 0.5
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=2)


def write_tables(
    output_tables_dir: Path,
    save_tables: bool,
    results: ComparisonResults,
    availability_df: pd.DataFrame,
    annotation_counts_df: pd.DataFrame,
    manual_benchmark_totals_df: pd.DataFrame,
    manual_column_labels_by_run: dict[str, dict[str, str]],
) -> dict[str, Path]:
    table_paths: dict[str, Path] = {}
    if not save_tables:
        return table_paths

    output_tables_dir.mkdir(parents=True, exist_ok=True)

    availability_path = output_tables_dir / "availability_summary.csv"
    availability_df.to_csv(availability_path, index=False)
    table_paths["availability_summary"] = availability_path

    if not annotation_counts_df.empty:
        annotation_counts_path = output_tables_dir / "annotation_counts_by_run.csv"
        annotation_counts_df.to_csv(annotation_counts_path, index=False)
        table_paths["annotation_counts_by_run"] = annotation_counts_path

        n_analyses_matrix_path = output_tables_dir / "n_analyses_matrix.csv"
        (
            annotation_counts_df.pivot(index="auto_name", columns="run", values="n_analyses")
            .sort_index()
            .to_csv(n_analyses_matrix_path)
        )
        table_paths["n_analyses_matrix"] = n_analyses_matrix_path

        n_unique_pmids_matrix_path = output_tables_dir / "n_unique_pmids_matrix.csv"
        (
            annotation_counts_df.pivot(index="auto_name", columns="run", values="n_unique_pmids")
            .sort_index()
            .to_csv(n_unique_pmids_matrix_path)
        )
        table_paths["n_unique_pmids_matrix"] = n_unique_pmids_matrix_path

    if not manual_benchmark_totals_df.empty:
        manual_totals_path = output_tables_dir / "manual_benchmark_totals.csv"
        manual_benchmark_totals_df.to_csv(manual_totals_path, index=False)
        table_paths["manual_benchmark_totals"] = manual_totals_path

    for run_name in results.run_names:
        safe_run_name = sanitize_name(run_name)
        dice_path = output_tables_dir / f"dice_matrix_{safe_run_name}.csv"
        pearson_path = output_tables_dir / f"pearson_matrix_{safe_run_name}.csv"
        relabel_manual_columns(
            results.dice_matrices[run_name],
            run_name=run_name,
            manual_column_labels_by_run=manual_column_labels_by_run,
        ).to_csv(dice_path)
        relabel_manual_columns(
            results.pearson_matrices[run_name],
            run_name=run_name,
            manual_column_labels_by_run=manual_column_labels_by_run,
        ).to_csv(pearson_path)
        table_paths[f"dice_matrix::{run_name}"] = dice_path
        table_paths[f"pearson_matrix::{run_name}"] = pearson_path

    diag_path = output_tables_dir / "diagonal_metrics.csv"
    summary_path = output_tables_dir / "run_summary.csv"
    results.diag_df.to_csv(diag_path, index=False)
    results.summary_df.to_csv(summary_path)
    table_paths["diagonal_metrics"] = diag_path
    table_paths["run_summary"] = summary_path

    return table_paths


def maybe_show_figure(show_figures: bool) -> None:
    if show_figures:
        plt.show()


def write_images(
    output_images_dir: Path,
    save_images: bool,
    show_figures: bool,
    results: ComparisonResults,
    dice_threshold: float,
    manual_column_labels_by_run: dict[str, dict[str, str]],
) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    if not save_images and not show_figures:
        return image_paths

    if save_images:
        output_images_dir.mkdir(parents=True, exist_ok=True)

    for run_name in results.run_names:
        safe_run_name = sanitize_name(run_name)
        run_dice_df = relabel_manual_columns(
            results.dice_matrices[run_name],
            run_name=run_name,
            manual_column_labels_by_run=manual_column_labels_by_run,
        )
        run_pearson_df = relabel_manual_columns(
            results.pearson_matrices[run_name],
            run_name=run_name,
            manual_column_labels_by_run=manual_column_labels_by_run,
        )

        fig, ax = plt.subplots(
            figsize=(
                max(8, 1.4 * len(run_dice_df.columns)),
                max(6, 1.1 * len(run_dice_df.index)),
            )
        )
        sns.heatmap(
            run_dice_df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Dice coefficient"},
        )
        ax.set_title(f"Dice Matrix: {run_name} (Automated vs Manual)", fontweight="bold")
        ax.set_xlabel("Manual benchmark annotation")
        ax.set_ylabel("Automated annotation + aggregate automated analyses")
        set_heatmap_xtick_alignment(ax, [str(label) for label in run_dice_df.columns])
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        if save_images:
            image_path = output_images_dir / f"dice_heatmap_{safe_run_name}.png"
            plt.savefig(image_path, dpi=300, bbox_inches="tight")
            image_paths[f"dice_heatmap::{run_name}"] = image_path
        maybe_show_figure(show_figures)
        plt.close(fig)

        fig, ax = plt.subplots(
            figsize=(
                max(8, 1.4 * len(run_pearson_df.columns)),
                max(6, 1.1 * len(run_pearson_df.index)),
            )
        )
        sns.heatmap(
            run_pearson_df,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            ax=ax,
            cbar_kws={"label": "Pearson r"},
        )
        ax.set_title(f"Pearson Matrix: {run_name} (Automated vs Manual)", fontweight="bold")
        ax.set_xlabel("Manual benchmark annotation")
        ax.set_ylabel("Automated annotation + aggregate automated analyses")
        set_heatmap_xtick_alignment(ax, [str(label) for label in run_pearson_df.columns])
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        if save_images:
            image_path = output_images_dir / f"pearson_heatmap_{safe_run_name}.png"
            plt.savefig(image_path, dpi=300, bbox_inches="tight")
            image_paths[f"pearson_heatmap::{run_name}"] = image_path
        maybe_show_figure(show_figures)
        plt.close(fig)

    plot_df = results.diag_df.copy()

    fig, ax = plt.subplots(figsize=(max(12, 1.6 * len(results.auto_order)), 6))
    sns.barplot(
        data=plot_df,
        x="auto_name",
        y="dice",
        hue="run",
        order=results.auto_order,
        hue_order=results.run_names,
        errorbar=None,
        ax=ax,
    )
    ax.set_title(f"Diagonal Dice (z > {dice_threshold}) by Run", fontweight="bold")
    ax.set_xlabel("Automated annotation (mapped to manual)")
    ax.set_ylabel("Dice coefficient")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Run", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_images:
        image_path = output_images_dir / "dice_diagonal_grouped.png"
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        image_paths["dice_diagonal_grouped"] = image_path
    maybe_show_figure(show_figures)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(12, 1.6 * len(results.auto_order)), 6))
    sns.barplot(
        data=plot_df,
        x="auto_name",
        y="pearson_r",
        hue="run",
        order=results.auto_order,
        hue_order=results.run_names,
        errorbar=None,
        ax=ax,
    )
    ax.set_title("Diagonal Pearson r by Run", fontweight="bold")
    ax.set_xlabel("Automated annotation (mapped to manual)")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Run", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_images:
        image_path = output_images_dir / "pearson_diagonal_grouped.png"
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        image_paths["pearson_diagonal_grouped"] = image_path
    maybe_show_figure(show_figures)
    plt.close(fig)

    return image_paths


def aggregate_name_candidates() -> list[str]:
    candidates: list[str] = ["all_analyses"]
    for variant in AGGREGATE_ANALYSIS_NAME_VARIANTS:
        for name in variant:
            if name not in candidates:
                candidates.append(name)
    return candidates


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_stat_map_cut_coords(
    display_mode: str,
    raw_cut_coords: list[float] | tuple[float, ...],
) -> float | tuple[float, ...]:
    cut_coords = tuple(float(value) for value in raw_cut_coords)
    if not cut_coords:
        raise ValueError("--stat-map-cut-coords requires at least one numeric value.")

    if str(display_mode).lower() == "ortho" and len(cut_coords) != 3:
        raise ValueError(
            "--stat-map-display-mode=ortho requires exactly 3 values for --stat-map-cut-coords "
            f"(got {len(cut_coords)}: {cut_coords})."
        )

    if len(cut_coords) == 1:
        return cut_coords[0]

    return cut_coords


def write_corrected_stat_map_images(
    output_images_dir: Path,
    save_images: bool,
    show_figures: bool,
    project_name: str,
    manual_analysis_base: Path,
    corrected_map_filename: str,
    stat_map_display_mode: str,
    stat_map_cut_coords: float | tuple[float, ...],
    mapping_pairs: list[MappingPair],
    included_run_infos: list[RunInfo],
    manual_meta_by_run: dict[str, bool],
    run_aggregate_paths: dict[str, dict[str, Path]],
) -> dict[str, list[tuple[str, Path]]]:
    if not save_images and not show_figures:
        return {}

    try:
        from nilearn import datasets, plotting
    except ImportError as exc:
        print(
            "Skipping corrected stat-map plotting: nilearn is not installed "
            f"({exc})."
        )
        return {}

    stat_maps_root = output_images_dir / "stat_maps"
    if save_images:
        stat_maps_root.mkdir(parents=True, exist_ok=True)

    aggregate_candidates = aggregate_name_candidates()
    plot_paths_by_version: dict[str, list[tuple[str, Path]]] = {}

    unique_pairs: list[MappingPair] = []
    seen_auto_names: set[str] = set()
    for pair in mapping_pairs:
        if pair.auto_name in seen_auto_names:
            continue
        seen_auto_names.add(pair.auto_name)
        unique_pairs.append(pair)

    def plot_one(
        *,
        version_label: str,
        annotation_label: str,
        stat_map_path: Path,
        out_path: Path | None,
    ) -> None:
        display = plotting.plot_stat_map(
            stat_map_img=str(stat_map_path),
            cut_coords=stat_map_cut_coords,
            display_mode=stat_map_display_mode,
            title=f"{version_label}: {annotation_label}",
            annotate=True,
            draw_cross=False,
        )
        if out_path is not None:
            display.savefig(str(out_path))
        if show_figures:
            maybe_show_figure(show_figures)
        display.close()

    # Manual benchmark first.
    manual_version_label = "manual_benchmark"
    manual_version_dir = stat_maps_root / sanitize_name(manual_version_label)
    if save_images:
        manual_version_dir.mkdir(parents=True, exist_ok=True)
    manual_entries: list[tuple[str, Path]] = []

    manual_aggregate_paths = [
        manual_analysis_base / project_name / candidate / corrected_map_filename
        for candidate in aggregate_candidates
    ]
    manual_aggregate_path = first_existing_path(manual_aggregate_paths)
    if manual_aggregate_path is not None:
        out_path = manual_version_dir / "all_analyses.png" if save_images else None
        plot_one(
            version_label=manual_version_label,
            annotation_label="all_analyses",
            stat_map_path=manual_aggregate_path,
            out_path=out_path,
        )
        if out_path is not None:
            manual_entries.append(("all_analyses", out_path))
    else:
        checked = ", ".join(str(path) for path in manual_aggregate_paths)
        print(
            "Manual benchmark aggregate map missing for corrected stat-map plot. "
            f"Checked: {checked}"
        )

    for pair in unique_pairs:
        candidate_paths = [
            manual_analysis_base / project_name / candidate / corrected_map_filename
            for candidate in manual_name_candidates(pair.manual_name)
        ]
        manual_map_path = first_existing_path(candidate_paths)
        if manual_map_path is None:
            checked = ", ".join(str(path) for path in candidate_paths)
            print(
                f"Missing manual corrected map for {pair.manual_name} "
                f"(label={pair.auto_name}). Checked: {checked}"
            )
            continue
        out_path = manual_version_dir / f"{sanitize_name(pair.auto_name)}.png" if save_images else None
        plot_one(
            version_label=manual_version_label,
            annotation_label=pair.auto_name,
            stat_map_path=manual_map_path,
            out_path=out_path,
        )
        if out_path is not None:
            manual_entries.append((pair.auto_name, out_path))

    if manual_entries:
        plot_paths_by_version[manual_version_label] = manual_entries

    # Then each automated meta-analysis run.
    for run_info in included_run_infos:
        run_name = run_info.name
        if manual_meta_by_run.get(run_name, False):
            continue

        run_version_dir = stat_maps_root / sanitize_name(run_name)
        if save_images:
            run_version_dir.mkdir(parents=True, exist_ok=True)
        run_entries: list[tuple[str, Path]] = []

        preferred_aggregate_names = list(run_aggregate_paths.get(run_name, {}).keys())
        aggregate_name_order = ["all_analyses"] + [
            name for name in preferred_aggregate_names if name != "all_analyses"
        ]
        aggregate_name_order.extend(
            name for name in aggregate_candidates if name not in aggregate_name_order
        )
        run_aggregate_map_path = first_existing_path(
            [
                run_info.meta_results_dir / aggregate_name / corrected_map_filename
                for aggregate_name in aggregate_name_order
            ]
        )

        if run_aggregate_map_path is not None:
            out_path = run_version_dir / "all_analyses.png" if save_images else None
            plot_one(
                version_label=run_name,
                annotation_label="all_analyses",
                stat_map_path=run_aggregate_map_path,
                out_path=out_path,
            )
            if out_path is not None:
                run_entries.append(("all_analyses", out_path))
        else:
            print(
                f"Missing corrected aggregate map for run {run_name}; skipped all_analyses plot."
            )

        for pair in unique_pairs:
            auto_map_path = run_info.meta_results_dir / pair.auto_name / corrected_map_filename
            if not auto_map_path.exists():
                print(
                    f"Missing corrected map for run {run_name}, annotation {pair.auto_name}: "
                    f"{auto_map_path}"
                )
                continue
            out_path = run_version_dir / f"{sanitize_name(pair.auto_name)}.png" if save_images else None
            plot_one(
                version_label=run_name,
                annotation_label=pair.auto_name,
                stat_map_path=auto_map_path,
                out_path=out_path,
            )
            if out_path is not None:
                run_entries.append((pair.auto_name, out_path))

        if run_entries:
            plot_paths_by_version[run_name] = run_entries

    return plot_paths_by_version


def to_html_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>No rows.</em></p>"
    return df.to_html(classes="report-table", border=0, escape=False)


def format_missing_pairs(missing_pairs: list[str], max_items: int = 6) -> str:
    preview = ", ".join(missing_pairs[:max_items])
    if len(missing_pairs) > max_items:
        preview += f" ... (+{len(missing_pairs) - max_items} more)"
    return preview


def build_html_report(
    output_dir: Path,
    project_dir: Path,
    mapping_path: Path,
    manual_analysis_base: Path,
    map_filename: str,
    corrected_map_filename: str,
    stat_map_display_mode: str,
    stat_map_cut_coords: float | tuple[float, ...],
    dice_threshold: float,
    meta_results_subpath: Path,
    run_infos: list[RunInfo],
    included_run_infos: list[RunInfo],
    skipped_run_missing_pairs: dict[str, list[str]],
    manual_meta_by_run: dict[str, bool],
    availability_df: pd.DataFrame,
    annotation_counts_df: pd.DataFrame,
    manual_benchmark_totals_df: pd.DataFrame,
    results: ComparisonResults,
    table_paths: dict[str, Path],
    image_paths: dict[str, Path],
    stat_map_paths_by_version: dict[str, list[tuple[str, Path]]],
) -> str:
    run_overview_rows = []
    run_info_by_name = {run_info.name: run_info for run_info in run_infos}
    for run_name in results.run_names:
        run_info = run_info_by_name[run_name]
        run_overview_rows.append(
            {
                "run": run_name,
                "run_dir": str(run_info.run_dir),
                "manual_meta": manual_meta_by_run[run_name],
                "aggregate_rows": ", ".join(results.run_aggregate_names_by_run[run_name])
                if results.run_aggregate_names_by_run[run_name]
                else "",
            }
        )
    run_overview_df = pd.DataFrame(run_overview_rows)

    skipped_rows = []
    for run_name, missing_pairs in skipped_run_missing_pairs.items():
        skipped_rows.append(
            {
                "run": run_name,
                "missing_count": len(missing_pairs),
                "missing_preview": format_missing_pairs(missing_pairs),
            }
        )
    skipped_df = pd.DataFrame(skipped_rows)

    diag_dice_pivot = results.diag_df.pivot(index="auto_name", columns="run", values="dice").round(3)
    diag_pearson_pivot = (
        results.diag_df.pivot(index="auto_name", columns="run", values="pearson_r").round(3)
    )
    annotation_n_analyses_matrix = pd.DataFrame()
    annotation_n_unique_pmids_matrix = pd.DataFrame()
    if not annotation_counts_df.empty:
        annotation_n_analyses_matrix = (
            annotation_counts_df.pivot(index="auto_name", columns="run", values="n_analyses")
            .sort_index()
        )
        annotation_n_unique_pmids_matrix = (
            annotation_counts_df.pivot(index="auto_name", columns="run", values="n_unique_pmids")
            .sort_index()
        )

    links_html = []
    for label, path in sorted(table_paths.items()):
        rel = path.relative_to(output_dir).as_posix()
        links_html.append(f'<li><a href="{escape(rel)}">{escape(path.name)}</a> ({escape(label)})</li>')

    image_sections: list[str] = []
    for run_name in results.run_names:
        dice_key = f"dice_heatmap::{run_name}"
        pearson_key = f"pearson_heatmap::{run_name}"
        if dice_key in image_paths:
            rel = image_paths[dice_key].relative_to(output_dir).as_posix()
            image_sections.append(
                f"<h4>Dice Heatmap: {escape(run_name)}</h4>"
                f'<img src="{escape(rel)}" alt="Dice heatmap {escape(run_name)}" class="plot-img" />'
            )
        if pearson_key in image_paths:
            rel = image_paths[pearson_key].relative_to(output_dir).as_posix()
            image_sections.append(
                f"<h4>Pearson Heatmap: {escape(run_name)}</h4>"
                f'<img src="{escape(rel)}" alt="Pearson heatmap {escape(run_name)}" class="plot-img" />'
            )

    for aggregate_plot_key, title in (
        ("dice_diagonal_grouped", "Diagonal Dice Grouped"),
        ("pearson_diagonal_grouped", "Diagonal Pearson Grouped"),
    ):
        if aggregate_plot_key in image_paths:
            rel = image_paths[aggregate_plot_key].relative_to(output_dir).as_posix()
            image_sections.append(
                f"<h4>{escape(title)}</h4>"
                f'<img src="{escape(rel)}" alt="{escape(title)}" class="plot-img" />'
            )

    stat_map_sections: list[str] = []
    for version_label, entries in stat_map_paths_by_version.items():
        stat_map_sections.append(f"<h3>{escape(version_label)}</h3>")
        for annotation_label, path in entries:
            rel = path.relative_to(output_dir).as_posix()
            stat_map_sections.append(
                f"<h4>{escape(annotation_label)}</h4>"
                f'<img src="{escape(rel)}" alt="Stat map {escape(version_label)} {escape(annotation_label)}" class="plot-img" />'
            )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Manual vs Auto Meta Comparison Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; line-height: 1.45; }}
    h1, h2, h3, h4 {{ margin-top: 1.2em; margin-bottom: 0.4em; }}
    .muted {{ color: #444; }}
    .report-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; }}
    .report-table th, .report-table td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
    .report-table th {{ background: #f7f7f7; text-align: left; }}
    .section {{ margin-top: 24px; }}
    .plot-img {{ display: block; max-width: 100%; height: auto; border: 1px solid #ccc; margin: 8px 0 20px; }}
    ul {{ margin-top: 6px; }}
    code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Manual vs Automated Meta-Analysis Comparison</h1>
  <p class="muted">Generated by <code>scripts/compare_meta_to_benchmark.py</code>.</p>

  <div class="section">
    <h2>Configuration</h2>
    <ul>
      <li><strong>project_dir:</strong> {escape(str(project_dir))}</li>
      <li><strong>output_dir:</strong> {escape(str(output_dir))}</li>
      <li><strong>mapping_path:</strong> {escape(str(mapping_path))}</li>
      <li><strong>manual_analysis_base:</strong> {escape(str(manual_analysis_base))}</li>
      <li><strong>map_filename:</strong> {escape(map_filename)}</li>
      <li><strong>corrected_map_filename:</strong> {escape(corrected_map_filename)}</li>
      <li><strong>stat_map_display_mode:</strong> {escape(stat_map_display_mode)}</li>
      <li><strong>stat_map_cut_coords:</strong> {escape(str(stat_map_cut_coords))}</li>
      <li><strong>dice_threshold:</strong> {dice_threshold:.4g}</li>
      <li><strong>meta_results_subpath:</strong> {escape(str(meta_results_subpath))}</li>
      <li><strong>discovered_runs:</strong> {len(run_infos)}</li>
      <li><strong>included_runs:</strong> {len(included_run_infos)}</li>
      <li><strong>n_mappings:</strong> {len(results.manual_order)}</li>
    </ul>
  </div>

  <div class="section">
    <h2>Run Inclusion Summary</h2>
    {to_html_table(run_overview_df)}
    <h3>Skipped Incomplete Runs</h3>
    {to_html_table(skipped_df)}
  </div>

  <div class="section">
    <h2>Availability Matrix</h2>
    {to_html_table(availability_df)}
  </div>

  <div class="section">
    <h2>Automated Annotation Counts</h2>
    <p class="muted">Counts are computed from each run's <code>nimads_annotation.json</code> and PMID-linked through <code>nimads_studyset.json</code>. Includes mapped annotations and <code>all*</code> annotations.</p>
    {to_html_table(annotation_counts_df)}
    <h3>n_analyses Matrix (annotation x run)</h3>
    {to_html_table(annotation_n_analyses_matrix)}
    <h3>n_unique_pmids Matrix (annotation x run)</h3>
    {to_html_table(annotation_n_unique_pmids_matrix)}
  </div>

  <div class="section">
    <h2>Manual Benchmark Totals By Annotation</h2>
    {to_html_table(manual_benchmark_totals_df)}
  </div>

  <div class="section">
    <h2>Per-Run Summary</h2>
    {to_html_table(results.summary_df.round(3))}
  </div>

  <div class="section">
    <h2>Diagonal Dice (auto_name x run)</h2>
    {to_html_table(diag_dice_pivot)}

    <h2>Diagonal Pearson r (auto_name x run)</h2>
    {to_html_table(diag_pearson_pivot)}
  </div>

  <div class="section">
    <h2>Table Artifacts</h2>
    <ul>
      {''.join(links_html) if links_html else '<li>No table files were written.</li>'}
    </ul>
  </div>

  <div class="section">
    <h2>Visualizations</h2>
    {''.join(image_sections) if image_sections else '<p><em>No images available.</em></p>'}
  </div>

  <div class="section">
    <h2>Orthogonal Stat Maps (FDR Corrected)</h2>
    <p class="muted">Ordered as manual benchmark first, then each automated run. Includes only <code>all_analyses</code> and mapped custom annotations.</p>
    {''.join(stat_map_sections) if stat_map_sections else '<p><em>No corrected stat-map images available.</em></p>'}
  </div>
</body>
</html>
"""

    return html


def write_html_report(output_dir: Path, save_html: bool, html_content: str) -> Path | None:
    if not save_html:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "manual_vs_auto_meta_report.html"
    html_path.write_text(html_content, encoding="utf-8")
    return html_path


def print_configuration_summary(
    project_dir: Path,
    mapping_path: Path,
    manual_analysis_base: Path,
    manual_nimads_base: Path,
    map_filename: str,
    corrected_map_filename: str,
    stat_map_display_mode: str,
    stat_map_cut_coords: float | tuple[float, ...],
    dice_threshold: float,
    output_dir: Path,
    meta_results_subpath: Path,
    run_infos: list[RunInfo],
) -> None:
    print("Configuration Summary")
    print("=" * 80)
    print(f"project_dir:         {project_dir}")
    print(f"mapping_path:        {mapping_path}")
    print(f"manual_analysis_base:{manual_analysis_base}")
    print(f"manual_nimads_base:  {manual_nimads_base}")
    print(f"map_filename:        {map_filename}")
    print(f"corrected_map_file:  {corrected_map_filename}")
    print(f"stat_map_mode:       {stat_map_display_mode}")
    print(f"stat_map_cut_coords: {stat_map_cut_coords}")
    print(f"dice_threshold:      {dice_threshold}")
    print(f"output_dir:          {output_dir}")
    print(f"meta_results_subpath:{meta_results_subpath}")
    print(f"discovered_runs:     {[run_info.name for run_info in run_infos]}")


def print_run_selection_summary(
    included_run_infos: list[RunInfo],
    skipped_run_missing_pairs: dict[str, list[str]],
    skipped_aggregate_runs: dict[str, str],
    manual_meta_by_run: dict[str, bool],
    run_aggregate_paths: dict[str, dict[str, Path]],
) -> None:
    print("\nIncluded complete runs")
    print("=" * 80)
    print([run_info.name for run_info in included_run_infos])

    if skipped_run_missing_pairs:
        print("\nSkipped incomplete runs (missing mapped outputs):")
        for run_name, missing_pairs in skipped_run_missing_pairs.items():
            print(
                f"  - {run_name}: missing {len(missing_pairs)} mapped outputs: "
                f"{format_missing_pairs(missing_pairs, max_items=4)}"
            )
    if skipped_aggregate_runs:
        print("\nSkipped aggregate analysis rows (missing supported aggregate set):")
        for run_name, reason in skipped_aggregate_runs.items():
            print(f"  - {run_name}: {reason}")

    print("\nAggregate analyses added as matrix rows")
    for run_info in included_run_infos:
        run_name = run_info.name
        aggregate_names = list(run_aggregate_paths[run_name].keys())
        print(
            f"  - {run_name}: manual_meta={manual_meta_by_run[run_name]} "
            f"aggregates={aggregate_names}"
        )


def main() -> None:
    args = parse_args()

    project_dir = resolve_project_dir(args.project_dir)
    output_dir = resolve_output_dir(project_dir, args.output_dir)
    mapping_path = resolve_mapping_path(project_dir, args.mapping_path)
    run_infos = resolve_run_infos(project_dir, args.meta_results_subpath, args.run_dir)

    if not args.show_figures:
        plt.ioff()

    mappings = load_mappings(mapping_path)
    stat_map_cut_coords = resolve_stat_map_cut_coords(
        display_mode=args.stat_map_display_mode,
        raw_cut_coords=args.stat_map_cut_coords,
    )

    print_configuration_summary(
        project_dir=project_dir,
        mapping_path=mapping_path,
        manual_analysis_base=args.manual_analysis_base,
        manual_nimads_base=args.manual_nimads_base,
        map_filename=args.map_filename,
        corrected_map_filename=args.corrected_map_filename,
        stat_map_display_mode=args.stat_map_display_mode,
        stat_map_cut_coords=stat_map_cut_coords,
        dice_threshold=args.dice_threshold,
        output_dir=output_dir,
        meta_results_subpath=args.meta_results_subpath,
        run_infos=run_infos,
    )

    mapping_pairs, run_missing_pairs, availability_df = build_mapping_pairs(
        mappings=mappings,
        run_infos=run_infos,
        manual_analysis_base=args.manual_analysis_base,
        project_name=project_dir.name,
        map_filename=args.map_filename,
    )

    included_run_infos, skipped_run_missing_pairs = filter_complete_runs(
        run_infos=run_infos,
        run_missing_pairs=run_missing_pairs,
    )

    run_names = [run_info.name for run_info in included_run_infos]

    mapping_pairs = [
        MappingPair(
            manual_name=pair.manual_name,
            auto_name=pair.auto_name,
            manual_path=pair.manual_path,
            auto_paths={run_name: pair.auto_paths[run_name] for run_name in run_names},
        )
        for pair in mapping_pairs
    ]

    manual_meta_by_run = classify_manual_meta_runs(
        included_run_infos=included_run_infos,
        explicit_manual_meta_runs=args.manual_meta_run,
    )

    run_aggregate_paths, skipped_aggregate_runs = collect_aggregate_paths(
        included_run_infos=included_run_infos,
        manual_meta_by_run=manual_meta_by_run,
        map_filename=args.map_filename,
    )

    annotation_counts_df, manual_column_labels_by_run = compute_automated_annotation_counts(
        mapping_pairs=mapping_pairs,
        included_run_infos=included_run_infos,
        manual_meta_by_run=manual_meta_by_run,
    )
    manual_benchmark_totals_df = compute_manual_benchmark_totals(
        project_name=project_dir.name,
        manual_nimads_base=args.manual_nimads_base.expanduser().resolve(),
    )

    print_run_selection_summary(
        included_run_infos=included_run_infos,
        skipped_run_missing_pairs=skipped_run_missing_pairs,
        skipped_aggregate_runs=skipped_aggregate_runs,
        manual_meta_by_run=manual_meta_by_run,
        run_aggregate_paths=run_aggregate_paths,
    )

    print("\nAnnotation Count Summary")
    print("=" * 80)
    if annotation_counts_df.empty:
        print("No automated runs available for annotation count summary.")
    else:
        print(annotation_counts_df.to_string(index=False))
    print("\nManual benchmark totals by annotation")
    print(manual_benchmark_totals_df.to_string(index=False))

    manual_vectors, auto_vectors_by_run, common_shape, n_valid_voxels = load_maps_and_vectors(
        mapping_pairs=mapping_pairs,
        run_names=run_names,
        run_aggregate_paths=run_aggregate_paths,
    )

    print("\nLoaded Data Summary")
    print("=" * 80)
    print(f"common_shape:      {common_shape}")
    print(f"valid_voxels:      {n_valid_voxels}")
    print(f"n_manual_maps:     {len(manual_vectors)}")
    print(f"n_automated_runs:  {len(auto_vectors_by_run)}")

    results = compute_comparison_results(
        mapping_pairs=mapping_pairs,
        run_names=run_names,
        manual_vectors=manual_vectors,
        auto_vectors_by_run=auto_vectors_by_run,
        run_aggregate_paths=run_aggregate_paths,
        dice_threshold=args.dice_threshold,
    )

    output_tables_dir = output_dir / "tables"
    output_images_dir = output_dir / "images"

    table_paths = write_tables(
        output_tables_dir=output_tables_dir,
        save_tables=args.save_tables,
        results=results,
        availability_df=availability_df,
        annotation_counts_df=annotation_counts_df,
        manual_benchmark_totals_df=manual_benchmark_totals_df,
        manual_column_labels_by_run=manual_column_labels_by_run,
    )

    image_paths = write_images(
        output_images_dir=output_images_dir,
        save_images=args.save_images,
        show_figures=args.show_figures,
        results=results,
        dice_threshold=args.dice_threshold,
        manual_column_labels_by_run=manual_column_labels_by_run,
    )
    stat_map_paths_by_version = write_corrected_stat_map_images(
        output_images_dir=output_images_dir,
        save_images=args.save_images,
        show_figures=args.show_figures,
        project_name=project_dir.name,
        manual_analysis_base=args.manual_analysis_base.expanduser().resolve(),
        corrected_map_filename=args.corrected_map_filename,
        stat_map_display_mode=args.stat_map_display_mode,
        stat_map_cut_coords=stat_map_cut_coords,
        mapping_pairs=mapping_pairs,
        included_run_infos=included_run_infos,
        manual_meta_by_run=manual_meta_by_run,
        run_aggregate_paths=run_aggregate_paths,
    )

    html_content = build_html_report(
        output_dir=output_dir,
        project_dir=project_dir,
        mapping_path=mapping_path,
        manual_analysis_base=args.manual_analysis_base,
        map_filename=args.map_filename,
        corrected_map_filename=args.corrected_map_filename,
        stat_map_display_mode=args.stat_map_display_mode,
        stat_map_cut_coords=stat_map_cut_coords,
        dice_threshold=args.dice_threshold,
        meta_results_subpath=args.meta_results_subpath,
        run_infos=run_infos,
        included_run_infos=included_run_infos,
        skipped_run_missing_pairs=skipped_run_missing_pairs,
        manual_meta_by_run=manual_meta_by_run,
        availability_df=availability_df,
        annotation_counts_df=annotation_counts_df,
        manual_benchmark_totals_df=manual_benchmark_totals_df,
        results=results,
        table_paths=table_paths,
        image_paths=image_paths,
        stat_map_paths_by_version=stat_map_paths_by_version,
    )
    html_path = write_html_report(
        output_dir=output_dir,
        save_html=args.save_html,
        html_content=html_content,
    )

    print("\nOutputs")
    print("=" * 80)
    print(f"save_tables: {args.save_tables}")
    if args.save_tables:
        print(f"tables_dir:  {output_tables_dir}")
    print(f"save_images: {args.save_images}")
    if args.save_images:
        print(f"images_dir:  {output_images_dir}")
        print(f"stat_maps:   {output_images_dir / 'stat_maps'}")
    print(f"save_html:   {args.save_html}")
    if html_path is not None:
        print(f"html_report: {html_path}")


def _main_with_exit() -> None:
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main_with_exit()

#!/usr/bin/env python3
"""Compare automated meta-analysis maps against manual benchmark maps.

This script ports the generic notebook workflow into a reusable CLI and produces
CSV tables, PNG figures, and an HTML report.

Examples:
    python scripts/compare_auto_vs_manual_generic.py \
        --project-dir projects/cue_reactivity

    python scripts/compare_auto_vs_manual_generic.py \
        --project-dir projects/cue_reactivity \
        --run-dir v1 \
        --manual-meta-run manual_meta_v1

    python scripts/compare_auto_vs_manual_generic.py \
        --project-dir projects/cue_reactivity \
        --output-dir projects/cue_reactivity/reports/manual_vs_auto_meta_custom \
        --no-save-images
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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
)
MANUAL_META_MARKERS = ("manual_meta", "manual-meta", "manual metas", "manual_metas")


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
        "--map-filename",
        type=str,
        default="z.nii.gz",
        help="Map filename expected in each analysis directory.",
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
        mappings = json.load(f)
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError(f"Mapping file must be a non-empty JSON object: {mapping_path}")
    return {str(k): str(v) for k, v in mappings.items()}


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
) -> dict[str, dict[str, Path]]:
    run_aggregate_paths: dict[str, dict[str, Path]] = {
        run_info.name: {} for run_info in included_run_infos
    }
    aggregate_missing_errors: list[str] = []

    for run_info in included_run_infos:
        run_name = run_info.name
        if manual_meta_by_run[run_name]:
            continue

        matched_variant: tuple[str, ...] | None = None
        for variant in AGGREGATE_ANALYSIS_NAME_VARIANTS:
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
            aggregate_missing_errors.append(
                f"{run_name}: missing aggregate analysis maps. Expected one of: "
                f"{'; '.join(expected_variants)}"
            )
            continue

        for aggregate_name in matched_variant:
            run_aggregate_paths[run_name][aggregate_name] = (
                run_info.meta_results_dir / aggregate_name / map_filename
            )

    if aggregate_missing_errors:
        error_lines = "\n".join(f"- {error}" for error in aggregate_missing_errors)
        raise FileNotFoundError(
            "Automated meta-analysis runs must include a supported aggregate set.\n"
            f"{error_lines}"
        )

    return run_aggregate_paths


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


def write_tables(
    output_tables_dir: Path,
    save_tables: bool,
    results: ComparisonResults,
    availability_df: pd.DataFrame,
) -> dict[str, Path]:
    table_paths: dict[str, Path] = {}
    if not save_tables:
        return table_paths

    output_tables_dir.mkdir(parents=True, exist_ok=True)

    availability_path = output_tables_dir / "availability_summary.csv"
    availability_df.to_csv(availability_path, index=False)
    table_paths["availability_summary"] = availability_path

    for run_name in results.run_names:
        safe_run_name = sanitize_name(run_name)
        dice_path = output_tables_dir / f"dice_matrix_{safe_run_name}.csv"
        pearson_path = output_tables_dir / f"pearson_matrix_{safe_run_name}.csv"
        results.dice_matrices[run_name].to_csv(dice_path)
        results.pearson_matrices[run_name].to_csv(pearson_path)
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
) -> dict[str, Path]:
    image_paths: dict[str, Path] = {}
    if not save_images and not show_figures:
        return image_paths

    if save_images:
        output_images_dir.mkdir(parents=True, exist_ok=True)

    for run_name in results.run_names:
        safe_run_name = sanitize_name(run_name)

        fig, ax = plt.subplots(
            figsize=(
                max(8, 1.4 * len(results.dice_matrices[run_name].columns)),
                max(6, 1.1 * len(results.dice_matrices[run_name].index)),
            )
        )
        sns.heatmap(
            results.dice_matrices[run_name],
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
        ax.tick_params(axis="x", rotation=45)
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
                max(8, 1.4 * len(results.pearson_matrices[run_name].columns)),
                max(6, 1.1 * len(results.pearson_matrices[run_name].index)),
            )
        )
        sns.heatmap(
            results.pearson_matrices[run_name],
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
        ax.tick_params(axis="x", rotation=45)
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
    dice_threshold: float,
    meta_results_subpath: Path,
    run_infos: list[RunInfo],
    included_run_infos: list[RunInfo],
    skipped_run_missing_pairs: dict[str, list[str]],
    manual_meta_by_run: dict[str, bool],
    availability_df: pd.DataFrame,
    results: ComparisonResults,
    table_paths: dict[str, Path],
    image_paths: dict[str, Path],
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
  <p class="muted">Generated by <code>scripts/compare_auto_vs_manual_generic.py</code>.</p>

  <div class="section">
    <h2>Configuration</h2>
    <ul>
      <li><strong>project_dir:</strong> {escape(str(project_dir))}</li>
      <li><strong>output_dir:</strong> {escape(str(output_dir))}</li>
      <li><strong>mapping_path:</strong> {escape(str(mapping_path))}</li>
      <li><strong>manual_analysis_base:</strong> {escape(str(manual_analysis_base))}</li>
      <li><strong>map_filename:</strong> {escape(map_filename)}</li>
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
    map_filename: str,
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
    print(f"map_filename:        {map_filename}")
    print(f"dice_threshold:      {dice_threshold}")
    print(f"output_dir:          {output_dir}")
    print(f"meta_results_subpath:{meta_results_subpath}")
    print(f"discovered_runs:     {[run_info.name for run_info in run_infos]}")


def print_run_selection_summary(
    included_run_infos: list[RunInfo],
    skipped_run_missing_pairs: dict[str, list[str]],
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

    print_configuration_summary(
        project_dir=project_dir,
        mapping_path=mapping_path,
        manual_analysis_base=args.manual_analysis_base,
        map_filename=args.map_filename,
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

    run_aggregate_paths = collect_aggregate_paths(
        included_run_infos=included_run_infos,
        manual_meta_by_run=manual_meta_by_run,
        map_filename=args.map_filename,
    )

    print_run_selection_summary(
        included_run_infos=included_run_infos,
        skipped_run_missing_pairs=skipped_run_missing_pairs,
        manual_meta_by_run=manual_meta_by_run,
        run_aggregate_paths=run_aggregate_paths,
    )

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
    )

    image_paths = write_images(
        output_images_dir=output_images_dir,
        save_images=args.save_images,
        show_figures=args.show_figures,
        results=results,
        dice_threshold=args.dice_threshold,
    )

    html_content = build_html_report(
        output_dir=output_dir,
        project_dir=project_dir,
        mapping_path=mapping_path,
        manual_analysis_base=args.manual_analysis_base,
        map_filename=args.map_filename,
        dice_threshold=args.dice_threshold,
        meta_results_subpath=args.meta_results_subpath,
        run_infos=run_infos,
        included_run_infos=included_run_infos,
        skipped_run_missing_pairs=skipped_run_missing_pairs,
        manual_meta_by_run=manual_meta_by_run,
        availability_df=availability_df,
        results=results,
        table_paths=table_paths,
        image_paths=image_paths,
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

#!/usr/bin/env python3
"""Plot top-V manual/auto/all-studies map triptychs for poster inspection."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np

from make_cross_project_publication_plots import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROJECTS_ROOT,
    DEFAULT_SCREENING_DIR,
    float_or_none,
    read_csv_rows,
    read_matrix_row_values,
    sanitize_name,
)
from make_poster_validation_plots import (
    POSTER_BG,
    POSTER_MUTED,
    POSTER_PANEL_BG,
    POSTER_TEXT,
    axis_project_label,
)


DEFAULT_MANUAL_ANALYSIS_BASE = Path("/home/zorro/repos/neurometabench/analysis")
DEFAULT_MAP_FILENAME = "z.nii.gz"
DEFAULT_CUT_COORDS = [-18, 0, 18, 36]
AUTO_DIFF_CMAP = ListedColormap(["#B2182B"])
BASELINE_DIFF_CMAP = ListedColormap(["#2166AC"])
BOTH_DIFF_CMAP = ListedColormap(["#7B3294"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-dir", type=Path, default=DEFAULT_SCREENING_DIR)
    parser.add_argument("--projects-root", type=Path, default=DEFAULT_PROJECTS_ROOT)
    parser.add_argument("--manual-analysis-base", type=Path, default=DEFAULT_MANUAL_ANALYSIS_BASE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "poster_validation_plots" / "brain_map_comparisons",
    )
    parser.add_argument("--map-filename", type=str, default=DEFAULT_MAP_FILENAME)
    parser.add_argument("--threshold", type=float, default=1.96)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--cut-coords", nargs="+", type=float, default=DEFAULT_CUT_COORDS)
    parser.add_argument("--display-mode", type=str, default="z")
    parser.add_argument(
        "--top-n-per-project",
        type=int,
        default=2,
        help="Number of top Dice-delta pairings to show per project.",
    )
    return parser.parse_args()


def selected_top_v_runs(screening_rows: list[dict[str, Any]]) -> dict[str, str]:
    selected_runs: dict[str, str] = {}
    for row in screening_rows:
        project = str(row.get("project_name", "")).strip()
        run = str(row.get("run_name", "")).strip()
        if project and run:
            selected_runs[project] = run
    return selected_runs


def load_top_v_allstudies_dice_rows(
    screening_rows: list[dict[str, Any]],
    projects_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for project, run_name in sorted(selected_top_v_runs(screening_rows).items()):
        tables_dir = projects_root / project / "reports" / "manual_vs_auto_meta" / "tables"
        diagonal_path = tables_dir / "diagonal_metrics.csv"
        matrix_path = tables_dir / f"dice_matrix_{sanitize_name(run_name)}.csv"
        if not diagonal_path.exists() or not matrix_path.exists():
            continue
        all_studies_by_manual = read_matrix_row_values(matrix_path, "all_studies")
        if not all_studies_by_manual:
            continue
        for row in read_csv_rows(diagonal_path):
            if str(row.get("run", "")).strip() != run_name:
                continue
            manual_name = str(row.get("manual_name", "")).strip()
            dice = float_or_none(row.get("dice"))
            baseline = float_or_none(all_studies_by_manual.get(manual_name))
            if manual_name and dice is not None and baseline is not None:
                rows.append(
                    {
                        "project_name": project,
                        "run": run_name,
                        "manual_name": manual_name,
                        "auto_name": str(row.get("auto_name", "")).strip(),
                        "score": dice,
                        "baseline_score": baseline,
                        "delta": dice - baseline,
                    }
                )
    return rows


def top_delta_rows(rows: list[dict[str, Any]], top_n_per_project: int) -> list[dict[str, Any]]:
    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = float_or_none(row.get("score"))
        baseline = float_or_none(row.get("baseline_score"))
        if value is None or baseline is None:
            continue
        item = dict(row)
        item["delta"] = value - baseline
        by_project[str(item["project_name"])].append(item)

    winners: list[dict[str, Any]] = []
    projects = sorted(
        by_project,
        key=lambda project: max(float(row["delta"]) for row in by_project[project]),
        reverse=True,
    )
    for project in projects:
        project_rows = sorted(by_project[project], key=lambda row: float(row["delta"]), reverse=True)
        winners.extend(project_rows[: max(1, top_n_per_project)])
    return winners


def map_paths_for_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Path]:
    project = str(row["project_name"])
    run = str(row["run"])
    manual = str(row["manual_name"])
    auto = str(row["auto_name"])
    run_maps_root = args.projects_root / project / run / "outputs" / "meta_analysis_results"
    return {
        "Manual": resolve_manual_map_path(args.manual_analysis_base, project, manual, args.map_filename),
        "Auto": run_maps_root / auto / args.map_filename,
        "Baseline": run_maps_root / "all_studies" / args.map_filename,
    }


def loose_name_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_manual_map_path(
    manual_analysis_base: Path,
    project: str,
    manual_name: str,
    map_filename: str,
) -> Path:
    exact_path = manual_analysis_base / project / manual_name / map_filename
    if exact_path.exists():
        return exact_path

    project_dir = manual_analysis_base / project
    target_key = loose_name_key(manual_name)
    if project_dir.exists():
        for candidate_dir in sorted(path for path in project_dir.iterdir() if path.is_dir()):
            if loose_name_key(candidate_dir.name) == target_key:
                candidate_path = candidate_dir / map_filename
                if candidate_path.exists():
                    return candidate_path

    return exact_path


def validate_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        paths = map_paths_for_row(row, args)
        missing = [f"{label}: {path}" for label, path in paths.items() if not path.exists()]
        if missing:
            print(f"[SKIP] {row['project_name']} {row['manual_name']} -> {row['auto_name']}")
            for item in missing:
                print(f"       missing {item}")
            continue
        valid_rows.append(row)
    return valid_rows


def add_stat_map(
    *,
    plotting: Any,
    stat_map_path: Path,
    ax: plt.Axes,
    title: str,
    args: argparse.Namespace,
    colorbar: bool = False,
) -> None:
    plotting.plot_stat_map(
        str(stat_map_path),
        axes=ax,
        display_mode=args.display_mode,
        cut_coords=args.cut_coords,
        threshold=args.threshold,
        cmap="cold_hot",
        colorbar=colorbar,
        annotate=False,
        draw_cross=False,
        black_bg=False,
        title=None,
    )
    ax.set_title(title, fontsize=8.8, color=POSTER_TEXT, pad=3)


def add_combined_difference_map(
    *,
    plotting: Any,
    diff_imgs: dict[str, nib.Nifti1Image],
    ax: plt.Axes,
    title: str,
    args: argparse.Namespace,
) -> None:
    display = plotting.plot_anat(
        axes=ax,
        display_mode=args.display_mode,
        cut_coords=args.cut_coords,
        annotate=False,
        draw_cross=False,
        black_bg=False,
        title=None,
    )
    for image_key, cmap in [
        ("Auto", AUTO_DIFF_CMAP),
        ("Baseline", BASELINE_DIFF_CMAP),
        ("Both", BOTH_DIFF_CMAP),
    ]:
        data = np.asanyarray(diff_imgs[image_key].dataobj)
        if np.any(data > 0):
            display.add_overlay(
                diff_imgs[image_key],
                threshold=0.5,
                cmap=cmap,
                colorbar=False,
                alpha=0.88,
            )
    if title:
        ax.set_title(title, fontsize=8.5, color=POSTER_TEXT, pad=3)


def load_binary_mask(path: Path, threshold: float) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    mask = np.nan_to_num(data, nan=0.0) > threshold
    return mask, img


def binary_difference_img(
    include_mask: np.ndarray,
    exclude_mask: np.ndarray,
    reference_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    diff = np.logical_and(include_mask, np.logical_not(exclude_mask)).astype(np.float32)
    return nib.Nifti1Image(diff, reference_img.affine, reference_img.header)


def mask_img(
    mask: np.ndarray,
    reference_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    return nib.Nifti1Image(mask.astype(np.float32), reference_img.affine, reference_img.header)


def combined_difference_group(
    auto_related_mask: np.ndarray,
    baseline_related_mask: np.ndarray,
    reference_img: nib.Nifti1Image,
) -> dict[str, nib.Nifti1Image]:
    both = np.logical_and(auto_related_mask, baseline_related_mask)
    auto_only = np.logical_and(auto_related_mask, np.logical_not(baseline_related_mask))
    baseline_only = np.logical_and(baseline_related_mask, np.logical_not(auto_related_mask))
    return {
        "Auto": mask_img(auto_only, reference_img),
        "Baseline": mask_img(baseline_only, reference_img),
        "Both": mask_img(both, reference_img),
    }


def combined_difference_images_for_row(
    row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, dict[str, nib.Nifti1Image]]:
    paths = map_paths_for_row(row, args)
    manual_mask, manual_img = load_binary_mask(paths["Manual"], args.threshold)
    auto_mask, auto_img = load_binary_mask(paths["Auto"], args.threshold)
    baseline_mask, _baseline_img = load_binary_mask(paths["Baseline"], args.threshold)
    extra_auto = np.logical_and(auto_mask, np.logical_not(manual_mask))
    extra_baseline = np.logical_and(baseline_mask, np.logical_not(manual_mask))
    missing_from_auto = np.logical_and(manual_mask, np.logical_not(auto_mask))
    missing_from_baseline = np.logical_and(manual_mask, np.logical_not(baseline_mask))
    return {
        "Extra vs Manual": combined_difference_group(extra_auto, extra_baseline, auto_img),
        "Manual Missing": combined_difference_group(missing_from_auto, missing_from_baseline, manual_img),
    }


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=POSTER_BG)
        print(f"[OK] {path}")
    plt.close(fig)
    return paths


def rows_by_project(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["project_name"])].append(row)
    for project_rows in grouped.values():
        project_rows.sort(key=lambda row: float(row["delta"]), reverse=True)
    return dict(
        sorted(
            grouped.items(),
            key=lambda item: max(float(row["delta"]) for row in item[1]),
            reverse=True,
        )
    )


def score_label(row: dict[str, Any]) -> str:
    return f"Dice={float(row['score']):.2f}"


def baseline_score_label(row: dict[str, Any]) -> str:
    return f"Dice={float(row['baseline_score']):.2f}"


def row_annotation_label(row: dict[str, Any]) -> str:
    return f"{axis_project_label(str(row['project_name']))}\n{row['manual_name']}\nΔDice={float(row['delta']):+.2f}"


def plot_full_panel(rows: list[dict[str, Any]], args: argparse.Namespace, plotting: Any) -> list[Path]:
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(15.5, max(2.3 * n_rows, 4.5)),
        facecolor=POSTER_BG,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.92, bottom=0.04, hspace=0.25, wspace=0.02)
    fig.suptitle("Top-V Maps vs All-Studies Baseline", fontsize=18, fontweight="bold", color=POSTER_TEXT)

    for col, label in enumerate(["Manual", "Auto", "Baseline"]):
        axes[0, col].text(
            0.5,
            1.18,
            label,
            transform=axes[0, col].transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=POSTER_TEXT,
        )

    for row_idx, row in enumerate(rows):
        paths = map_paths_for_row(row, args)
        axes[row_idx, 0].text(
            -0.04,
            0.5,
            row_annotation_label(row),
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            color=POSTER_MUTED,
        )
        titles = {
            "Manual": str(row["manual_name"]),
            "Auto": f"{row['auto_name']} ({score_label(row)})",
            "Baseline": f"All Studies ({baseline_score_label(row)})",
        }
        for col_idx, label in enumerate(["Manual", "Auto", "Baseline"]):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(POSTER_PANEL_BG)
            add_stat_map(
                plotting=plotting,
                stat_map_path=paths[label],
                ax=ax,
                title=titles[label],
                args=args,
            )

    return save_figure(fig, args.output_dir, "top_v_manual_auto_baseline_z_maps_panel", args.dpi)


def plot_individual_triptychs(rows: list[dict[str, Any]], args: argparse.Namespace, plotting: Any) -> list[Path]:
    outputs: list[Path] = []
    for project, project_rows in rows_by_project(rows).items():
        fig, axes = plt.subplots(
            len(project_rows),
            3,
            figsize=(11.5, max(2.55 * len(project_rows), 3.2)),
            facecolor=POSTER_BG,
            squeeze=False,
        )
        fig.subplots_adjust(left=0.12, right=0.99, top=0.82, bottom=0.06, hspace=0.34, wspace=0.02)
        fig.suptitle(
            f"{axis_project_label(project)}: Top {len(project_rows)} Dice Improvements",
            fontsize=12,
            fontweight="bold",
            color=POSTER_TEXT,
        )
        for col, label in enumerate(["Manual", "Auto", "Baseline"]):
            axes[0, col].text(
                0.5,
                1.20,
                label,
                transform=axes[0, col].transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=POSTER_TEXT,
            )
        for row_idx, row in enumerate(project_rows):
            paths = map_paths_for_row(row, args)
            axes[row_idx, 0].text(
                -0.04,
                0.5,
                f"{row['manual_name']}\nΔDice={float(row['delta']):+.2f}",
                transform=axes[row_idx, 0].transAxes,
                ha="right",
                va="center",
                fontsize=8.2,
                color=POSTER_MUTED,
            )
            titles = [
                ("Manual", str(row["manual_name"])),
                ("Auto", f"{row['auto_name']} ({score_label(row)})"),
                ("Baseline", f"All Studies ({baseline_score_label(row)})"),
            ]
            for ax, (label, title) in zip(axes[row_idx], titles):
                add_stat_map(
                    plotting=plotting,
                    stat_map_path=paths[label],
                    ax=ax,
                    title=title,
                    args=args,
                )
        stem = f"top_v_z_maps_{project}"
        outputs.extend(save_figure(fig, args.output_dir, stem, args.dpi))
    return outputs


def plot_difference_panel(rows: list[dict[str, Any]], args: argparse.Namespace, plotting: Any) -> list[Path]:
    columns = ["Extra Regions vs Manual", "Manual Regions Missed"]
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows,
        len(columns),
        figsize=(11.0, max(2.1 * n_rows, 4.5)),
        facecolor=POSTER_BG,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.16, right=0.99, top=0.89, bottom=0.04, hspace=0.28, wspace=0.05)
    fig.suptitle(
        f"Binarized Difference Maps (z > {args.threshold:g})",
        fontsize=18,
        fontweight="bold",
        color=POSTER_TEXT,
    )
    fig.text(
        0.5,
        0.925,
        "Red = Top-V Auto    Blue = All-Studies Baseline    Purple = Both",
        ha="center",
        va="center",
        fontsize=10,
        color=POSTER_MUTED,
    )
    for col, label in enumerate(columns):
        axes[0, col].text(
            0.5,
            1.18,
            label,
            transform=axes[0, col].transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=POSTER_TEXT,
        )

    for row_idx, row in enumerate(rows):
        axes[row_idx, 0].text(
            -0.04,
            0.5,
            row_annotation_label(row),
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            color=POSTER_MUTED,
        )
        diff_imgs = combined_difference_images_for_row(row, args)
        plot_items = [
            ("Extra vs Manual", ""),
            ("Manual Missing", ""),
        ]
        for col_idx, (image_key, title) in enumerate(plot_items):
            add_combined_difference_map(
                plotting=plotting,
                diff_imgs=diff_imgs[image_key],
                ax=axes[row_idx, col_idx],
                title=title,
                args=args,
            )

    return save_figure(fig, args.output_dir, "top_v_binarized_difference_maps_panel", args.dpi)


def plot_individual_difference_panels(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    plotting: Any,
) -> list[Path]:
    outputs: list[Path] = []
    columns = [
        ("Extra vs Manual", "Extra Regions vs Manual"),
        ("Manual Missing", "Manual Regions Missed"),
    ]
    for project, project_rows in rows_by_project(rows).items():
        fig, axes = plt.subplots(
            len(project_rows),
            len(columns),
            figsize=(8.6, max(2.55 * len(project_rows), 3.2)),
            facecolor=POSTER_BG,
            squeeze=False,
        )
        fig.subplots_adjust(left=0.13, right=0.99, top=0.76, bottom=0.06, hspace=0.34, wspace=0.05)
        fig.suptitle(
            f"{axis_project_label(project)}: Top {len(project_rows)} Dice Difference Maps",
            fontsize=12,
            fontweight="bold",
            color=POSTER_TEXT,
        )
        fig.text(
            0.5,
            0.82,
            "Red = Top-V Auto    Blue = All-Studies Baseline    Purple = Both",
            ha="center",
            va="center",
            fontsize=8,
            color=POSTER_MUTED,
        )
        for row_idx, row in enumerate(project_rows):
            axes[row_idx, 0].text(
                -0.04,
                0.5,
                f"{row['manual_name']}\nΔDice={float(row['delta']):+.2f}",
                transform=axes[row_idx, 0].transAxes,
                ha="right",
                va="center",
                fontsize=8.2,
                color=POSTER_MUTED,
            )
            diff_imgs = combined_difference_images_for_row(row, args)
            for ax, (image_key, title) in zip(axes[row_idx], columns):
                add_combined_difference_map(
                    plotting=plotting,
                    diff_imgs=diff_imgs[image_key],
                    ax=ax,
                    title=title,
                    args=args,
                )
        outputs.extend(save_figure(fig, args.output_dir, f"top_v_binarized_differences_{project}", args.dpi))
    return outputs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from nilearn import plotting

    screening_rows = read_csv_rows(args.screening_dir / "screening_metrics_top_v_stage_progression.csv")
    all_rows = load_top_v_allstudies_dice_rows(screening_rows, args.projects_root)
    rows = validate_rows(top_delta_rows(all_rows, args.top_n_per_project), args)
    if not rows:
        raise RuntimeError("No complete manual/auto/baseline map triplets found.")

    outputs: list[Path] = []
    outputs.extend(plot_full_panel(rows, args, plotting))
    outputs.extend(plot_individual_triptychs(rows, args, plotting))
    outputs.extend(plot_difference_panel(rows, args, plotting))
    outputs.extend(plot_individual_difference_panels(rows, args, plotting))
    print(f"[DONE] wrote {len(outputs)} files to {args.output_dir}")


if __name__ == "__main__":
    main()

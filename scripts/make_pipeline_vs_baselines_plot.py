#!/usr/bin/env python3
"""Whole-pipeline vs baselines: similarity to the manual meta-analysis.

Separate from the annotation-only isolation plot. This one is computed on the
top canonical-V run that takes a PubMed search as input, so all three conditions
come from the same run and the same corpus:

  annotated     each annotation column vs its mapped manual map (nmb_mappings.json)
  all_analyses  screened studies, every analysis, no annotation filtering
  all_studies   everything the PubMed search returned (quasi-Neurosynth)

Reports R^2 (variance of the manual map explained). "-recent" runs are excluded:
their search is not date-limited.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECTS_ROOT = REPO_ROOT / "projects"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "cross_project_publication_plots"

# Same selection rules as run_cross_project_screening_reports.py
CANONICAL_VERSION_RUN_RE = re.compile(r"^v(?P<version>\d+)$")
CANONICAL_RUN_OVERRIDES = {"social": "v3-search-all_pmids-multi_analysis-ft"}

CONDITIONS = ("annotated", "all_analyses", "all_studies")

# Poster styling, mirrored from make_poster_validation_plots.py
POSTER_BG = "#F4FAFB"
POSTER_PANEL_BG = "#FBFEFE"
POSTER_BORDER = "#587A85"
POSTER_TEXT = "#111111"
PAIR_LINE_COLOR = "#A7B1BA"
ALL_STUDIES_FILL = "#C3CDD4"
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
AXIS_PROJECT_LABELS = {
    "cue_reactivity": "Cue",
    "decision_making": "DM",
    "dementia": "Dementia",
    "executive_function": "Executive",
    "problem_solving": "PS",
    "social": "Social",
    "vbm_of_ptsd": "VBM PTSD",
    "vbm_of_substance_use": "VBM SUD",
}


def is_recent_run(name: str) -> bool:
    return "recent" in str(name).lower()


def read_matrix(path: Path) -> tuple[list[str], dict[str, list[float | None]]]:
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = header[1:]
        rows: dict[str, list[float | None]] = {}
        for row in reader:
            if not row or not row[0]:
                continue
            vals: list[float | None] = []
            for cell in row[1:]:
                try:
                    vals.append(float(cell))
                except (TypeError, ValueError):
                    vals.append(None)
            rows[row[0]] = vals
    return cols, rows


def select_run(project_dir: Path) -> str | None:
    """Top canonical-V run taking a PubMed search as input."""
    project = project_dir.name
    if project in CANONICAL_RUN_OVERRIDES:
        return CANONICAL_RUN_OVERRIDES[project]
    # Select from run directories, not configs: some run dirs (e.g.
    # cue_reactivity/v5) have outputs but no matching yaml. This matches
    # select_top_v_rows() in run_cross_project_screening_reports.py.
    best: tuple[int, str] | None = None
    for run_dir in sorted(d for d in project_dir.iterdir() if d.is_dir()):
        name = run_dir.name
        if is_recent_run(name):
            continue
        m = CANONICAL_VERSION_RUN_RE.fullmatch(name)
        if not m:
            continue
        if not (run_dir / "outputs").is_dir():
            continue
        key = (int(m.group("version")), name)
        if best is None or key > best:
            best = key
    return best[1] if best else None


def collect(projects_root: Path) -> list[dict]:
    out: list[dict] = []
    for project_dir in sorted(p for p in projects_root.iterdir() if p.is_dir()):
        mapping_path = project_dir / "nmb_mappings.json"
        if not mapping_path.exists():
            continue
        mapping = (json.loads(mapping_path.read_text()) or {}).get("annotation_mappings") or {}
        if not mapping:
            continue
        run = select_run(project_dir)
        if not run:
            continue
        matrix_path = project_dir / "reports" / "manual_vs_auto_meta" / "tables" / f"pearson_matrix_{run}.csv"
        if not matrix_path.exists():
            continue
        cols, rows = read_matrix(matrix_path)
        for idx, manual_col in enumerate(cols):
            annotation_row = mapping.get(manual_col)
            if not annotation_row or annotation_row not in rows:
                continue
            rec = {
                "project_name": project_dir.name,
                "run": run,
                "manual_column": manual_col,
                "annotation_column": annotation_row,
            }
            ok = True
            for cond, row_name in (("annotated", annotation_row),
                                   ("all_analyses", "all_analyses"),
                                   ("all_studies", "all_studies")):
                vals = rows.get(row_name)
                r = vals[idx] if vals and idx < len(vals) else None
                if r is None:
                    ok = False
                    break
                rec[f"{cond}_r"] = r
                rec[f"{cond}_r2"] = r * r
            if ok:
                out.append(rec)
    return out


def project_means(records: list[dict]) -> list[dict]:
    by_project: dict[str, list[dict]] = {}
    for rec in records:
        by_project.setdefault(rec["project_name"], []).append(rec)
    summary = []
    for project, recs in sorted(by_project.items()):
        row = {"project_name": project, "run": recs[0]["run"], "n_columns": len(recs)}
        for cond in CONDITIONS:
            row[f"{cond}_r2"] = statistics.fmean(r[f"{cond}_r2"] for r in recs)
            row[f"{cond}_r"] = statistics.fmean(r[f"{cond}_r"] for r in recs)
        row["gain_vs_all_analyses"] = row["annotated_r2"] - row["all_analyses_r2"]
        row["gain_vs_all_studies"] = row["annotated_r2"] - row["all_studies_r2"]
        summary.append(row)
    return summary


def write_plot(records: list[dict], output_dir: Path, formats: list[str], dpi: int,
               metric: str = "r2") -> list[Path]:
    """Dumbbell in the poster style: one filled project-coloured dot per annotation
    column, plus two hollow baseline dots (all_analyses, all_studies)."""
    key = (lambda cond: f"{cond}_{metric}")
    by_project: dict[str, list[dict]] = {}
    for rec in records:
        by_project.setdefault(rec["project_name"], []).append(rec)
    projects = sorted(
        by_project,
        key=lambda p: statistics.fmean(r[key("annotated")] for r in by_project[p]),
        reverse=True,
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": POSTER_PANEL_BG})
    fig = plt.figure(figsize=(4.6, 3.1), dpi=200, facecolor=POSTER_BG)
    ax = fig.add_axes([0.14, 0.33, 0.83, 0.53])

    xs = list(range(1, len(projects) + 1))
    for x, project in zip(xs, projects):
        recs = sorted(by_project[project], key=lambda r: r[key("annotated")])
        n = len(recs)
        offsets = [0.0] if n <= 1 else [-0.28 + 0.56 * (i / (n - 1)) for i in range(n)]
        for offset, rec in zip(offsets, recs):
            px = x + offset
            vals = [rec[key(c)] for c in CONDITIONS]
            ax.plot([px, px], [min(vals), max(vals)],
                    color=PAIR_LINE_COLOR, linewidth=1.7, zorder=1)
            ax.scatter([px], [rec[key("all_studies")]], s=40, facecolor=ALL_STUDIES_FILL,
                       edgecolor=POSTER_TEXT, linewidth=1.0, zorder=2)
            ax.scatter([px], [rec[key("all_analyses")]], s=38, facecolor=POSTER_PANEL_BG,
                       edgecolor=POSTER_TEXT, linewidth=1.15, zorder=3)
            ax.scatter([px], [rec[key("annotated")]], s=28,
                       color=PROJECT_COLORS.get(project, "#6B7280"),
                       edgecolor=POSTER_TEXT, linewidth=0.65, zorder=4)

    ax.set_xlim(0.35, len(projects) + 0.65)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(xs)
    ax.set_xticklabels([AXIS_PROJECT_LABELS.get(p, p.replace("_", " ")) for p in projects],
                       rotation=25, ha="right", fontsize=6.4)
    ax.set_ylabel("$R^2$ vs manual map" if metric == "r2" else "Pearson r",
                  fontsize=8.0, labelpad=1)
    ax.tick_params(axis="y", labelsize=7.0)
    ax.grid(axis="y", alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(POSTER_BORDER)
    ax.spines["bottom"].set_color(POSTER_BORDER)
    ax.tick_params(length=3, width=0.8)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", markersize=5.4,
                   markerfacecolor="#8FA0AC", markeredgecolor=POSTER_TEXT,
                   markeredgewidth=0.65, label="Annotation"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markersize=6.2,
                   markerfacecolor=POSTER_PANEL_BG, markeredgecolor=POSTER_TEXT,
                   markeredgewidth=1.15, label="All analyses"),
        plt.Line2D([0], [0], marker="o", linestyle="none", markersize=6.4,
                   markerfacecolor=ALL_STUDIES_FILL, markeredgecolor=POSTER_TEXT,
                   markeredgewidth=1.0, label="All studies"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.6, loc="upper center",
              ncol=3, handletextpad=0.35, columnspacing=1.4,
              bbox_to_anchor=(0.5, -0.34))
    fig.text(0.5, 0.945, "Top-V Map Similarity vs Baselines", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=POSTER_TEXT)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    stem = f"pipeline_vs_baselines_{metric}"
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi, facecolor=POSTER_BG, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--projects-root", type=Path, default=DEFAULT_PROJECTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=450)
    args = parser.parse_args()

    records = collect(args.projects_root)
    if not records:
        print("No records found.")
        return 1
    summary = project_means(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_col = args.output_dir / "pipeline_vs_baselines_by_column.csv"
    with per_col.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    per_proj = args.output_dir / "pipeline_vs_baselines_by_project.csv"
    with per_proj.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"[OK] {per_col}")
    print(f"[OK] {per_proj}")
    for path in write_plot(records, args.output_dir, args.formats, args.dpi, "r2"):
        print(f"[OK] {path}")

    print(f"\n{'project':<24}{'run':<40}{'n':>3}{'annot':>9}{'all_anal':>10}{'all_stud':>10}{'gain':>8}")
    for row in summary:
        print(f"{row['project_name']:<24}{row['run']:<40}{row['n_columns']:>3}"
              f"{row['annotated_r2']:>9.3f}{row['all_analyses_r2']:>10.3f}"
              f"{row['all_studies_r2']:>10.3f}{row['gain_vs_all_studies']:>+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

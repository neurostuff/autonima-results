#!/usr/bin/env python3
"""Summarize parser-failure annotations and reconcile them with matching metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LEGACY_DISPOSITIONS = {
    "missed_unit_confirmed": "parser_missed",
    "missed_unit_supplemental_data": "supplemental_data",
    "missed_unit_out_of_scope": "out_of_scope",
    "gold_standard_wrong": "gold_standard_error",
}

DISPOSITION_LABELS = {
    "(blank)": "No disposition selected",
    "expected_difference": "Expected source/curation difference",
    "gold_standard_error": "Gold standard error",
    "matching_error": "Matching error; both analyses correct",
    "out_of_scope": "Out of scope",
    "parser_missed": "Parser missed / misparsed analysis",
    "source_material_missing": "Source material missing",
    "supplemental_data": "Supplemental-only source",
    "uncertain": "Uncertain",
}

MODE_LABELS = {
    "under_split_merge": "Under-split / merge",
    "partial_coord_error": "Partial coordinate error",
    "over_split": "Over-split",
    "no_failure_mode_selected": "No failure mode selected",
}

CORRECTED_DISPOSITIONS = {
    "matching_error",
    "gold_standard_error",
    "expected_difference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--prior-review", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def disposition(entry: dict[str, Any]) -> str:
    current = str(entry.get("unmatched_gold_disposition") or "")
    if current:
        return current
    legacy = str(entry.get("missed_unit_disposition") or "")
    return LEGACY_DISPOSITIONS.get(legacy, legacy or "(blank)")


def percentage(numerator: int, denominator: int, digits: int = 1) -> str:
    return f"{100 * numerator / denominator:.{digits}f}%" if denominator else "n/a"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    review_path = args.review.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    baseline_path = args.baseline_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    review = load_json(review_path)
    manifest = load_json(manifest_path)
    baseline_rows = load_csv(baseline_path)
    entries = review.get("entries", []) or []
    manifest_ids = {str(unit["unit_id"]) for unit in manifest.get("units", [])}
    entry_ids = [str(entry.get("unit_id") or "") for entry in entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("Review contains duplicate unit IDs")
    missing_ids = sorted(set(entry_ids) - manifest_ids)
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} reviewed units are absent from the manifest")

    disposition_counts = Counter(disposition(entry) for entry in entries)
    disposition_rows: list[dict[str, Any]] = []
    for value, count in sorted(disposition_counts.items(), key=lambda item: (-item[1], item[0])):
        subset = [entry for entry in entries if disposition(entry) == value]
        disposition_rows.append(
            {
                "disposition": value,
                "label": DISPOSITION_LABELS.get(value, value),
                "analysis_count": count,
                "paper_count": len({(entry["project"], entry["pmid"]) for entry in subset}),
                "percent_of_reviewed": f"{100 * count / len(entries):.3f}",
            }
        )
    write_csv(
        output_dir / "disposition_summary.csv",
        disposition_rows,
        ["disposition", "label", "analysis_count", "paper_count", "percent_of_reviewed"],
    )

    dispositions_by_project: list[dict[str, Any]] = []
    annotation_by_project: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        annotation_by_project[str(entry["project"])][disposition(entry)] += 1
    for project, counts in sorted(annotation_by_project.items()):
        project_entries = [entry for entry in entries if entry["project"] == project]
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            dispositions_by_project.append(
                {
                    "project": project,
                    "disposition": value,
                    "label": DISPOSITION_LABELS.get(value, value),
                    "analysis_count": count,
                    "paper_count": len(
                        {
                            entry["pmid"]
                            for entry in project_entries
                            if disposition(entry) == value
                        }
                    ),
                    "percent_of_project_reviewed": f"{100 * count / len(project_entries):.3f}",
                }
            )
    write_csv(
        output_dir / "disposition_by_project.csv",
        dispositions_by_project,
        [
            "project",
            "disposition",
            "label",
            "analysis_count",
            "paper_count",
            "percent_of_project_reviewed",
        ],
    )

    parser_misses = [entry for entry in entries if disposition(entry) == "parser_missed"]
    mode_counts: Counter[str] = Counter()
    mode_papers: dict[str, set[tuple[str, str]]] = defaultdict(set)
    mode_projects: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in parser_misses:
        modes = entry.get("failure_modes") or ["no_failure_mode_selected"]
        for mode in set(modes):
            mode_counts[mode] += 1
            mode_papers[mode].add((str(entry["project"]), str(entry["pmid"])))
            mode_projects[mode][str(entry["project"])] += 1
    mode_rows = [
        {
            "failure_mode": mode,
            "label": MODE_LABELS.get(mode, mode),
            "analysis_count": count,
            "paper_count": len(mode_papers[mode]),
            "percent_of_confirmed_parser_misses": f"{100 * count / len(parser_misses):.3f}",
            "project_counts": "; ".join(
                f"{project}={n}" for project, n in sorted(mode_projects[mode].items())
            ),
        }
        for mode, count in sorted(mode_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    write_csv(
        output_dir / "failure_mode_summary.csv",
        mode_rows,
        [
            "failure_mode",
            "label",
            "analysis_count",
            "paper_count",
            "percent_of_confirmed_parser_misses",
            "project_counts",
        ],
    )

    parser_miss_rows: list[dict[str, Any]] = []
    reason_combinations: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
    for entry in parser_misses:
        modes = tuple(sorted(entry.get("failure_modes") or [])) or (
            "no_failure_mode_selected",
        )
        auxiliary = tuple(
            sorted(entry.get("parsing_reasons") or entry.get("legacy_reasons") or [])
        ) or ("none",)
        reason_combinations[(modes, auxiliary)] += 1
        parser_miss_rows.append(
            {
                "unit_id": entry["unit_id"],
                "project": entry["project"],
                "pmid": entry["pmid"],
                "failure_modes": "|".join(modes),
                "auxiliary_reasons": "|".join(auxiliary) if auxiliary != ("none",) else "",
                "note": entry.get("note", ""),
            }
        )
    write_csv(
        output_dir / "confirmed_parser_miss_reasons.csv",
        parser_miss_rows,
        [
            "unit_id",
            "project",
            "pmid",
            "failure_modes",
            "auxiliary_reasons",
            "note",
        ],
    )
    reason_combination_rows = [
        {
            "failure_modes": " + ".join(modes),
            "auxiliary_reasons": " + ".join(auxiliary),
            "analysis_count": count,
            "percent_of_parser_misses": f"{100 * count / len(parser_misses):.3f}",
        }
        for (modes, auxiliary), count in sorted(
            reason_combinations.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    write_csv(
        output_dir / "failure_reason_combinations.csv",
        reason_combination_rows,
        [
            "failure_modes",
            "auxiliary_reasons",
            "analysis_count",
            "percent_of_parser_misses",
        ],
    )

    accuracy_rows: list[dict[str, Any]] = []
    for baseline in baseline_rows:
        project = baseline["project_name"]
        total = int(baseline["manual_analyses_total"])
        accepted = int(baseline["accepted"])
        uncertain = int(baseline["uncertain"])
        unmatched = int(baseline["unmatched"])
        counts = annotation_by_project[project]
        reviewed = sum(counts.values())
        corrections = sum(counts[value] for value in CORRECTED_DISPOSITIONS)
        parser_correct = accepted + counts["matching_error"]
        parser_total = parser_correct + counts["parser_missed"]
        resolved_correct = accepted + corrections
        resolved_total = resolved_correct + counts["parser_missed"]
        accuracy_rows.append(
            {
                "project": project,
                "manual_analyses_total": total,
                "baseline_accepted": accepted,
                "baseline_uncertain": uncertain,
                "baseline_unmatched": unmatched,
                "reviewed_unmatched": reviewed,
                "review_coverage_of_unmatched": f"{reviewed / unmatched:.6f}" if unmatched else "1.000000",
                "confirmed_parser_missed": counts["parser_missed"],
                "matching_error_correct": counts["matching_error"],
                "gold_standard_error_correct": counts["gold_standard_error"],
                "expected_difference_correct": counts["expected_difference"],
                "corrections_added": corrections,
                "baseline_strict_rate": f"{accepted / total:.6f}",
                "review_adjusted_strict_correct": accepted + corrections,
                "review_adjusted_strict_rate": f"{(accepted + corrections) / total:.6f}",
                "baseline_combined_rate": f"{(accepted + uncertain) / total:.6f}",
                "review_adjusted_combined_correct": accepted + uncertain + corrections,
                "review_adjusted_combined_rate": f"{(accepted + uncertain + corrections) / total:.6f}",
                "parser_evaluable_correct": parser_correct,
                "parser_evaluable_total": parser_total,
                "parser_evaluable_rate": f"{parser_correct / parser_total:.6f}" if parser_total else "",
                "review_resolved_correct": resolved_correct,
                "review_resolved_total": resolved_total,
                "review_resolved_rate": f"{resolved_correct / resolved_total:.6f}" if resolved_total else "",
            }
        )

    integer_fields = [
        "manual_analyses_total",
        "baseline_accepted",
        "baseline_uncertain",
        "baseline_unmatched",
        "reviewed_unmatched",
        "confirmed_parser_missed",
        "matching_error_correct",
        "gold_standard_error_correct",
        "expected_difference_correct",
        "corrections_added",
        "review_adjusted_strict_correct",
        "review_adjusted_combined_correct",
        "parser_evaluable_correct",
        "parser_evaluable_total",
        "review_resolved_correct",
        "review_resolved_total",
    ]
    overall: dict[str, Any] = {"project": "__all__"}
    for field in integer_fields:
        overall[field] = sum(int(row[field]) for row in accuracy_rows)
    total = overall["manual_analyses_total"]
    accepted = overall["baseline_accepted"]
    uncertain = overall["baseline_uncertain"]
    unmatched = overall["baseline_unmatched"]
    reviewed = overall["reviewed_unmatched"]
    overall.update(
        {
            "review_coverage_of_unmatched": f"{reviewed / unmatched:.6f}",
            "baseline_strict_rate": f"{accepted / total:.6f}",
            "review_adjusted_strict_rate": f"{overall['review_adjusted_strict_correct'] / total:.6f}",
            "baseline_combined_rate": f"{(accepted + uncertain) / total:.6f}",
            "review_adjusted_combined_rate": f"{overall['review_adjusted_combined_correct'] / total:.6f}",
            "parser_evaluable_rate": f"{overall['parser_evaluable_correct'] / overall['parser_evaluable_total']:.6f}",
            "review_resolved_rate": f"{overall['review_resolved_correct'] / overall['review_resolved_total']:.6f}",
        }
    )
    accuracy_rows.append(overall)
    write_csv(
        output_dir / "accuracy_adjustment_by_project.csv",
        accuracy_rows,
        list(accuracy_rows[0].keys()),
    )

    prior_summary: dict[str, Any] = {}
    if args.prior_review is not None:
        prior = load_json(args.prior_review.expanduser().resolve())
        prior_by_id = {str(entry["unit_id"]): entry for entry in prior.get("entries", [])}
        current_by_id = {str(entry["unit_id"]): entry for entry in entries}
        shared = set(prior_by_id) & set(current_by_id)
        new_ids = set(current_by_id) - set(prior_by_id)
        changed = [
            unit_id
            for unit_id in shared
            if disposition(prior_by_id[unit_id]) != disposition(current_by_id[unit_id])
        ]
        prior_summary = {
            "new_unit_annotations": len(new_ids),
            "carried_forward_units": len(shared),
            "prior_units_not_current": len(set(prior_by_id) - set(current_by_id)),
            "changed_carried_forward_dispositions": len(changed),
            "new_annotation_dispositions": dict(
                Counter(disposition(current_by_id[unit_id]) for unit_id in new_ids)
            ),
        }

    unresolved = [
        {
            "unit_id": entry["unit_id"],
            "project": entry["project"],
            "pmid": entry["pmid"],
            "note": entry.get("note", ""),
        }
        for entry in entries
        if disposition(entry) == "(blank)"
    ]
    summary = {
        "inputs": {
            "review": str(review_path),
            "manifest": str(manifest_path),
            "baseline": str(baseline_path),
        },
        "annotation_coverage": {
            "reviewed_units": len(entries),
            "manifest_units": len(manifest.get("units", [])),
            "coverage": len(entries) / len(manifest.get("units", [])),
            "reviewed_papers": len({(entry["project"], entry["pmid"]) for entry in entries}),
            **prior_summary,
        },
        "dispositions": dict(disposition_counts),
        "failure_modes_among_parser_missed": dict(mode_counts),
        "parsing_reasons_among_parser_missed": dict(
            Counter(
                reason
                for entry in parser_misses
                for reason in (entry.get("parsing_reasons") or entry.get("legacy_reasons") or [])
            )
        ),
        "accuracy_methodology": {
            "review_adjusted": (
                "Adds reviewed unmatched matching_error, gold_standard_error, and "
                "expected_difference units to the baseline correct count. It does not "
                "impute unreviewed or non-evaluable units."
            ),
            "parser_evaluable": (
                "Assumes baseline accepted units are correct and includes only reviewed "
                "matching_error (correct) and parser_missed (incorrect) disagreements."
            ),
            "review_resolved": (
                "Also treats reviewed gold_standard_error and expected_difference units "
                "as factually correct extraction outcomes."
            ),
        },
        "overall_accuracy": overall,
        "unresolved_annotations": unresolved,
    }
    (output_dir / "annotation_analysis.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Parser Failure Annotation Analysis",
        "",
        f"- Reviewed: **{len(entries)}/{len(manifest['units'])}** unmatched analyses "
        f"({percentage(len(entries), len(manifest['units']))}) across "
        f"**{summary['annotation_coverage']['reviewed_papers']} papers**.",
        f"- Confirmed parser misses: **{disposition_counts['parser_missed']}** "
        f"({percentage(disposition_counts['parser_missed'], len(entries))}).",
        f"- Supplemental-only or missing source: "
        f"**{disposition_counts['supplemental_data'] + disposition_counts['source_material_missing']}** "
        f"({percentage(disposition_counts['supplemental_data'] + disposition_counts['source_material_missing'], len(entries))}).",
        f"- Reviewed benchmark/matching/expected differences credited as correct: "
        f"**{sum(disposition_counts[value] for value in CORRECTED_DISPOSITIONS)}**.",
        "",
        "## Dispositions",
        "",
        "| Disposition | Analyses | Papers | Share |",
        "|---|---:|---:|---:|",
    ]
    for row in disposition_rows:
        lines.append(
            f"| {row['label']} | {row['analysis_count']} | {row['paper_count']} | "
            f"{float(row['percent_of_reviewed']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Confirmed Parser Failure Modes",
            "",
            "Percentages use confirmed parser misses as denominator; modes are non-exclusive.",
            "",
            "| Failure mode | Analyses | Papers | Share |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in mode_rows:
        lines.append(
            f"| {row['label']} | {row['analysis_count']} | {row['paper_count']} | "
            f"{float(row['percent_of_confirmed_parser_misses']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Accuracy Reconciliation",
            "",
            "| Metric | Correct / total | Rate |",
            "|---|---:|---:|",
            f"| Current strict baseline | {accepted}/{total} | {percentage(accepted, total)} |",
            f"| Review-adjusted strict | {overall['review_adjusted_strict_correct']}/{total} | "
            f"{percentage(overall['review_adjusted_strict_correct'], total)} |",
            f"| Current accepted + uncertain | {accepted + uncertain}/{total} | "
            f"{percentage(accepted + uncertain, total)} |",
            f"| Review-adjusted accepted + uncertain | "
            f"{overall['review_adjusted_combined_correct']}/{total} | "
            f"{percentage(overall['review_adjusted_combined_correct'], total)} |",
            f"| Parser-evaluable resolved cases | {overall['parser_evaluable_correct']}/"
            f"{overall['parser_evaluable_total']} | "
            f"{percentage(overall['parser_evaluable_correct'], overall['parser_evaluable_total'])} |",
            f"| All review-resolved factual extraction cases | {overall['review_resolved_correct']}/"
            f"{overall['review_resolved_total']} | "
            f"{percentage(overall['review_resolved_correct'], overall['review_resolved_total'])} |",
            "",
            "Adjusted full-denominator rates reclassify only manually resolved false negatives.",
            "",
            "## Unresolved Annotations",
            "",
        ]
    )
    if unresolved:
        for item in unresolved:
            lines.append(f"- `{item['unit_id']}`: {item['note'] or '(no note)'}")
    else:
        lines.append("- None.")
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = analyze(args)
    overall = summary["overall_accuracy"]
    print(
        f"Reviewed {summary['annotation_coverage']['reviewed_units']}/"
        f"{summary['annotation_coverage']['manifest_units']} units; "
        f"strict baseline={float(overall['baseline_strict_rate']):.3%}, "
        f"adjusted={float(overall['review_adjusted_strict_rate']):.3%}"
    )
    print(f"Wrote analysis artifacts to {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()

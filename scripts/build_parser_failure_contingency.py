#!/usr/bin/env python3
"""Turn a manifest.json (from generate_parser_failure_annotation_report.py) plus one or
more exported human-review JSON files into:

- a failure-mode x trigger-variable contingency table (rates computed over ALL
  reviewed units at each trigger level, not failures-only)
- a random-vs-systematic verdict per (failure_mode, trigger_variable, trigger_level) cell
  (a simple rate-ratio heuristic, NOT a statistical significance test)
- a corrupted contrast<->map training-pair rate (overall, by project, by trigger level)
- a ranked fix list (systematic cells only, ranked by frequency x severity weight)

Scope: coordinate-separation correctness only. See parser_failure_taxonomy.py for the
failure-mode / trigger-variable definitions this script cross-tabulates.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import parser_failure_taxonomy as taxonomy  # noqa: E402

CAUTION_TEXT = (
    "This is a simple rate-ratio heuristic (>= {multiple:.1f}x baseline rate with "
    ">= {min_count} occurrences), not a statistical significance test. With small "
    "per-paper N, apparent concentration can be noise -- treat 'systematic' flags as a "
    "prioritization signal for manual follow-up, not a proven causal claim."
)
CORRUPTED_PAIR_NOTE = (
    "A unit counts as producing a corrupted contrast<->map training pair if it carries "
    "any of: {modes}. Partial coordinate error and missed unit are excluded by design -- "
    "a partially wrong peak list or an absent analysis doesn't corrupt an *existing* pair "
    "the way a wrong unit boundary or wrong coordinate attachment does."
)
RANKED_FIX_LIST_OMISSION_NOTE = (
    "Only 'concentrated/systematic' cells are listed here. Cells classified as "
    "'roughly uniform/random-ish' are intentionally omitted -- see contingency.csv for "
    "the full table, and the plan's guidance to fix only systematic modes."
)

CROWDING_BINS = [(2, "1-2"), (5, "3-5"), (10, "6-10")]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", nargs="+", required=True, help="Review export JSON file(s), glob-friendly.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--corrupted-pair-modes",
        nargs="+",
        default=list(taxonomy.DEFAULT_CORRUPTED_PAIR_MODES),
    )
    parser.add_argument("--systematic-rate-multiple", type=float, default=taxonomy.DEFAULT_SYSTEMATIC_RATE_MULTIPLE)
    parser.add_argument("--systematic-min-count", type=int, default=taxonomy.DEFAULT_SYSTEMATIC_MIN_COUNT)
    parser.add_argument("--severity-weight-overrides", type=Path, default=None)
    return parser.parse_args()


def expand_review_paths(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.update(Path(m).resolve() for m in matches)
        else:
            direct = Path(pattern)
            if direct.exists():
                paths.add(direct.resolve())
    return sorted(paths)


def merge_review_exports(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    candidates_by_unit: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for path in paths:
        payload = load_json(path)
        for entry in payload.get("entries", []) or []:
            unit_id = str(entry.get("unit_id") or "")
            if not unit_id:
                continue
            candidates_by_unit.setdefault(unit_id, []).append((entry, path))

    winners: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    for unit_id, candidates in candidates_by_unit.items():
        candidates_sorted = sorted(
            candidates, key=lambda item: str(item[0].get("updated_at") or ""), reverse=True
        )
        winner_entry, winner_path = candidates_sorted[0]
        winners[unit_id] = winner_entry
        if len(candidates_sorted) > 1:
            runner_entry, runner_path = candidates_sorted[1]
            try:
                delta_seconds = abs(
                    (
                        datetime.fromisoformat(str(winner_entry.get("updated_at")).replace("Z", "+00:00"))
                        - datetime.fromisoformat(str(runner_entry.get("updated_at")).replace("Z", "+00:00"))
                    ).total_seconds()
                )
            except Exception:
                delta_seconds = None
            if delta_seconds is not None and delta_seconds < 60 and winner_entry != runner_entry:
                warnings.append(
                    {
                        "unit_id": unit_id,
                        "winner_path": str(winner_path),
                        "runner_up_path": str(runner_path),
                        "delta_seconds": delta_seconds,
                        "note": "possible-concurrent-edit, verify manually",
                    }
                )
    return winners, warnings


def bin_crowding(crowding: int | None) -> str:
    if crowding is None:
        return "(unknown)"
    for threshold, label in CROWDING_BINS:
        if crowding <= threshold:
            return label
    return "11+"


def effective_failure_modes(entry: dict[str, Any]) -> set[str]:
    modes = set(entry.get("failure_modes") or [])
    if (
        entry.get("unmatched_gold_disposition") == "parser_missed"
        or entry.get("missed_unit_disposition") == "missed_unit_confirmed"
    ):
        modes.add("missed_unit")
    if entry.get("spurious_disposition") == "spurious_fabricated":
        modes.add("spurious_unit")
    return modes


def is_parser_evaluable(entry: dict[str, Any]) -> bool:
    unit_kind = str(entry.get("unit_kind") or "")
    if unit_kind == "auto_only_unit":
        return entry.get("spurious_disposition") == "spurious_fabricated"

    if unit_kind != "gold_unit":
        return False
    if entry.get("match_status") == "unmatched":
        disposition = entry.get("unmatched_gold_disposition")
        if not disposition:
            legacy = entry.get("missed_unit_disposition")
            disposition = {
                "missed_unit_confirmed": "parser_missed",
                "missed_unit_supplemental_data": "supplemental_data",
                "missed_unit_out_of_scope": "out_of_scope",
                "gold_standard_wrong": "gold_standard_error",
            }.get(legacy, legacy)
        return disposition in taxonomy.PARSER_EVALUABLE_UNMATCHED_DISPOSITIONS

    parsing_disposition = entry.get("parsing_disposition")
    if parsing_disposition in {"correct", "error"}:
        return True
    return bool(entry.get("failure_modes"))


def compute_trigger_levels(entry: dict[str, Any], crowding: int | None) -> dict[str, str | None]:
    levels: dict[str, str | None] = {}
    for tv in taxonomy.TRIGGER_VARIABLES:
        if tv.id == "crowding":
            levels[tv.id] = bin_crowding(crowding)
        elif tv.kind == "dropdown":
            value = str(entry.get(tv.id) or "").strip()
            levels[tv.id] = value or None
        elif tv.kind == "checkbox":
            levels[tv.id] = "true" if entry.get(tv.id) else "false"
    return levels


def join_manifest_and_reviews(
    manifest: dict[str, Any], entries: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_units_by_id = {u["unit_id"]: u for u in manifest.get("units", [])}
    reviewed_units: list[dict[str, Any]] = []
    for unit_id, entry in entries.items():
        manifest_unit = manifest_units_by_id.get(unit_id)
        crowding = manifest_unit.get("crowding") if manifest_unit else None
        reviewed_units.append(
            {
                "unit_id": unit_id,
                "unit_kind": str(entry.get("unit_kind") or ""),
                "project": str(entry.get("project") or ""),
                "pmid": str(entry.get("pmid") or ""),
                "match_status": entry.get("match_status") or "",
                "parser_evaluable": is_parser_evaluable(entry),
                "effective_failure_modes": effective_failure_modes(entry),
                "trigger_levels": compute_trigger_levels(entry, crowding),
            }
        )
    not_yet_reviewed = [u for u in manifest.get("units", []) if u["unit_id"] not in entries]
    return reviewed_units, not_yet_reviewed


def classify_cell(rate: float, overall_rate: float, count: int, multiple: float, min_count: int) -> str:
    if count >= min_count and overall_rate > 0 and rate >= multiple * overall_rate:
        return "concentrated/systematic"
    return "roughly uniform/random-ish"


def build_contingency(
    reviewed_units: list[dict[str, Any]], multiple: float, min_count: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in taxonomy.FAILURE_MODES:
        applicable = [
            u
            for u in reviewed_units
            if u["parser_evaluable"]
            and (mode.applies_to == "both" or u["unit_kind"] == mode.applies_to)
        ]
        overall_den = len(applicable)
        overall_num = sum(1 for u in applicable if mode.id in u["effective_failure_modes"])
        overall_rate = overall_num / overall_den if overall_den else 0.0

        for tv in taxonomy.TRIGGER_VARIABLES:
            level_counts: dict[str, dict[str, int]] = {}
            for u in applicable:
                level = u["trigger_levels"].get(tv.id)
                if level is None:
                    continue
                counts = level_counts.setdefault(level, {"num": 0, "den": 0})
                counts["den"] += 1
                if mode.id in u["effective_failure_modes"]:
                    counts["num"] += 1
            for level in sorted(level_counts.keys()):
                counts = level_counts[level]
                rate = counts["num"] / counts["den"] if counts["den"] else 0.0
                rows.append(
                    {
                        "failure_mode": mode.id,
                        "trigger_variable": tv.id,
                        "trigger_level": level,
                        "numerator": counts["num"],
                        "denominator": counts["den"],
                        "rate": rate,
                        "overall_failure_mode_rate": overall_rate,
                        "systematic_flag": classify_cell(rate, overall_rate, counts["num"], multiple, min_count),
                    }
                )
    return rows


def compute_corrupted_pair_rate(
    reviewed_units: list[dict[str, Any]], corrupted_pair_modes: list[str]
) -> dict[str, Any]:
    reviewed_units = [u for u in reviewed_units if u["parser_evaluable"]]
    modes_set = set(corrupted_pair_modes)

    def is_corrupted(unit: dict[str, Any]) -> bool:
        return bool(unit["effective_failure_modes"] & modes_set)

    total = len(reviewed_units)
    corrupted_count = sum(1 for u in reviewed_units if is_corrupted(u))
    overall = {
        "corrupted": corrupted_count,
        "total": total,
        "rate": corrupted_count / total if total else 0.0,
    }

    by_project: dict[str, dict[str, int]] = {}
    for u in reviewed_units:
        row = by_project.setdefault(u["project"], {"corrupted": 0, "total": 0})
        row["total"] += 1
        if is_corrupted(u):
            row["corrupted"] += 1
    project_rows = [
        {
            "project": project,
            "corrupted": row["corrupted"],
            "total": row["total"],
            "rate": row["corrupted"] / row["total"] if row["total"] else 0.0,
        }
        for project, row in sorted(by_project.items())
    ]

    by_trigger: dict[tuple[str, str], dict[str, int]] = {}
    for tv in taxonomy.TRIGGER_VARIABLES:
        for u in reviewed_units:
            level = u["trigger_levels"].get(tv.id)
            if level is None:
                continue
            key = (tv.id, level)
            row = by_trigger.setdefault(key, {"corrupted": 0, "total": 0})
            row["total"] += 1
            if is_corrupted(u):
                row["corrupted"] += 1
    trigger_rows = [
        {
            "trigger_variable": tv_id,
            "trigger_level": level,
            "corrupted": row["corrupted"],
            "total": row["total"],
            "rate": row["corrupted"] / row["total"] if row["total"] else 0.0,
        }
        for (tv_id, level), row in sorted(by_trigger.items())
    ]

    return {"overall": overall, "by_project": project_rows, "by_trigger": trigger_rows}


def load_severity_weights(path: Path | None) -> dict[str, float]:
    weights = {mode.id: mode.severity_weight for mode in taxonomy.FAILURE_MODES}
    if path is not None:
        overrides = load_json(path)
        for key, value in overrides.items():
            weights[key] = float(value)
    return weights


def rank_fix_candidates(
    contingency_rows: list[dict[str, Any]], severity_weights: dict[str, float]
) -> list[dict[str, Any]]:
    systematic = [dict(row) for row in contingency_rows if row["systematic_flag"] == "concentrated/systematic"]
    for row in systematic:
        weight = severity_weights.get(row["failure_mode"], 1.0)
        row["severity_weight"] = weight
        row["score"] = row["numerator"] * weight
    systematic.sort(key=lambda row: row["score"], reverse=True)
    for idx, row in enumerate(systematic, start=1):
        row["rank"] = idx
    return systematic


def write_csv_with_preface(path: Path, preface_lines: list[str], rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        for line in preface_lines:
            f.write(f"# {line}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def render_contingency_html(
    manifest: dict[str, Any],
    reviewed_units: list[dict[str, Any]],
    not_yet_reviewed: list[dict[str, Any]],
    merge_warnings: list[dict[str, Any]],
    contingency_rows: list[dict[str, Any]],
    corrupted_pair: dict[str, Any],
    fix_list: list[dict[str, Any]],
    caution_text: str,
    corrupted_pair_note: str,
    args: argparse.Namespace,
) -> str:
    def contingency_table_rows() -> str:
        out = []
        for row in contingency_rows:
            css_class = "systematic" if row["systematic_flag"] == "concentrated/systematic" else "random-ish"
            out.append(
                f"<tr class=\"{css_class}\">"
                f"<td>{escape(row['failure_mode'])}</td>"
                f"<td>{escape(row['trigger_variable'])}</td>"
                f"<td>{escape(row['trigger_level'])}</td>"
                f"<td>{row['numerator']}/{row['denominator']}</td>"
                f"<td>{row['rate']:.3f}</td>"
                f"<td>{row['overall_failure_mode_rate']:.3f}</td>"
                f"<td>{escape(row['systematic_flag'])}</td>"
                "</tr>"
            )
        return "".join(out)

    def fix_list_rows() -> str:
        out = []
        for row in fix_list:
            out.append(
                "<tr>"
                f"<td>{row['rank']}</td>"
                f"<td>{escape(row['failure_mode'])}</td>"
                f"<td>{escape(row['trigger_variable'])}</td>"
                f"<td>{escape(row['trigger_level'])}</td>"
                f"<td>{row['numerator']}</td>"
                f"<td>{row['rate']:.3f}</td>"
                f"<td>{row['severity_weight']:.2f}</td>"
                f"<td>{row['score']:.2f}</td>"
                "</tr>"
            )
        return "".join(out)

    def project_rows() -> str:
        return "".join(
            f"<tr><td>{escape(r['project'])}</td><td>{r['corrupted']}/{r['total']}</td>"
            f"<td>{r['rate']:.3f}</td></tr>"
            for r in corrupted_pair["by_project"]
        )

    def trigger_rows() -> str:
        return "".join(
            f"<tr><td>{escape(r['trigger_variable'])}</td><td>{escape(r['trigger_level'])}</td>"
            f"<td>{r['corrupted']}/{r['total']}</td><td>{r['rate']:.3f}</td></tr>"
            for r in corrupted_pair["by_trigger"]
        )

    warnings_html = ""
    if merge_warnings:
        items = "".join(
            f"<li><code>{escape(w['unit_id'])}</code>: {escape(w['winner_path'])} vs "
            f"{escape(w['runner_up_path'])} (Δ{w['delta_seconds']:.0f}s) -- {escape(w['note'])}</li>"
            for w in merge_warnings
        )
        warnings_html = f"<section><h2>Merge Warnings</h2><ul>{items}</ul></section>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Parser Failure Contingency Report</title>
  <style>
    :root {{ --bg: #f7f6f2; --panel: #ffffff; --ink: #1d2730; --line: #d8dde3; }}
    body {{ margin: 0; padding: 1.25rem; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    header, section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }}
    .caution {{ background: #fff7e6; border: 1px solid #e8d8ad; border-radius: 8px; padding: 0.6rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border: 1px solid var(--line); padding: 0.4rem; text-align: left; }}
    th {{ background: #edf2f5; }}
    tr.random-ish {{ color: #8a97a3; }}
    tr.systematic {{ font-weight: 600; background: #fff0f0; }}
  </style>
</head>
<body>
  <header>
    <h1>Parser Failure Contingency Report</h1>
    <p class="caution">{escape(caution_text)}</p>
    <p><strong>Reviewed units:</strong> {len(reviewed_units)} |
    <strong>Parser-evaluable:</strong> {sum(1 for unit in reviewed_units if unit["parser_evaluable"])} |
    <strong>Not yet reviewed:</strong> {len(not_yet_reviewed)} |
    <strong>Systematic-rate multiple:</strong> {args.systematic_rate_multiple:.1f} |
    <strong>Systematic min count:</strong> {args.systematic_min_count}</p>
  </header>

  <section>
    <h2>Corrupted Pair Rate</h2>
    <p class="resource-note">{escape(corrupted_pair_note)}</p>
    <p><strong>Overall:</strong> {corrupted_pair['overall']['corrupted']}/{corrupted_pair['overall']['total']}
    = {corrupted_pair['overall']['rate']:.3f}</p>
    <h3>By project</h3>
    <table><thead><tr><th>Project</th><th>Corrupted/Total</th><th>Rate</th></tr></thead>
    <tbody>{project_rows()}</tbody></table>
    <h3>By trigger level</h3>
    <table><thead><tr><th>Trigger variable</th><th>Level</th><th>Corrupted/Total</th><th>Rate</th></tr></thead>
    <tbody>{trigger_rows()}</tbody></table>
  </section>

  <section>
    <h2>Contingency Table</h2>
    <p class="resource-note">Rows classified "roughly uniform/random-ish" are shown grayed out.</p>
    <table>
      <thead><tr><th>Failure mode</th><th>Trigger variable</th><th>Trigger level</th>
      <th>Num/Den</th><th>Rate</th><th>Overall mode rate</th><th>Verdict</th></tr></thead>
      <tbody>{contingency_table_rows()}</tbody>
    </table>
  </section>

  <section>
    <h2>Ranked Fix List (systematic only)</h2>
    <p class="resource-note">{escape(RANKED_FIX_LIST_OMISSION_NOTE)}</p>
    <table>
      <thead><tr><th>Rank</th><th>Failure mode</th><th>Trigger variable</th><th>Trigger level</th>
      <th>Count</th><th>Rate</th><th>Severity weight</th><th>Score</th></tr></thead>
      <tbody>{fix_list_rows()}</tbody>
    </table>
  </section>
  {warnings_html}
</body>
</html>
"""


def write_outputs(
    output_dir: Path,
    contingency_rows: list[dict[str, Any]],
    corrupted_pair: dict[str, Any],
    fix_list: list[dict[str, Any]],
    merge_warnings: list[dict[str, Any]],
    html: str,
    caution_text: str,
    corrupted_pair_note: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv_with_preface(
        output_dir / "contingency.csv",
        [caution_text],
        contingency_rows,
        [
            "failure_mode",
            "trigger_variable",
            "trigger_level",
            "numerator",
            "denominator",
            "rate",
            "overall_failure_mode_rate",
            "systematic_flag",
        ],
    )

    corrupted_rows = [{"scope": "overall", "key": "", **corrupted_pair["overall"]}]
    corrupted_rows += [{"scope": "project", "key": r["project"], **r} for r in corrupted_pair["by_project"]]
    corrupted_rows += [
        {"scope": "trigger", "key": f"{r['trigger_variable']}={r['trigger_level']}", **r}
        for r in corrupted_pair["by_trigger"]
    ]
    write_csv_with_preface(
        output_dir / "corrupted_pair_rate.csv",
        [corrupted_pair_note],
        corrupted_rows,
        ["scope", "key", "corrupted", "total", "rate"],
    )

    write_csv_with_preface(
        output_dir / "ranked_fix_list.csv",
        [RANKED_FIX_LIST_OMISSION_NOTE],
        fix_list,
        [
            "rank",
            "failure_mode",
            "trigger_variable",
            "trigger_level",
            "numerator",
            "rate",
            "severity_weight",
            "score",
        ],
    )

    (output_dir / "contingency_report.html").write_text(html, encoding="utf-8")

    if merge_warnings:
        (output_dir / "merge_warnings.json").write_text(json.dumps(merge_warnings, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    manifest = load_json(args.manifest.expanduser().resolve())
    review_paths = expand_review_paths(args.reviews)
    if not review_paths:
        raise FileNotFoundError(f"No review export files matched: {args.reviews}")
    print(f"Loading {len(review_paths)} review export file(s):")
    for path in review_paths:
        print(f"  {path}")

    entries, merge_warnings = merge_review_exports(review_paths)
    reviewed_units, not_yet_reviewed = join_manifest_and_reviews(manifest, entries)
    print(f"Reviewed units: {len(reviewed_units)} | Not yet reviewed: {len(not_yet_reviewed)}")
    if merge_warnings:
        print(f"WARNING: {len(merge_warnings)} possible concurrent-edit collisions -- see merge_warnings.json")

    contingency_rows = build_contingency(reviewed_units, args.systematic_rate_multiple, args.systematic_min_count)
    corrupted_pair = compute_corrupted_pair_rate(reviewed_units, args.corrupted_pair_modes)
    severity_weights = load_severity_weights(args.severity_weight_overrides)
    fix_list = rank_fix_candidates(contingency_rows, severity_weights)

    caution_text = CAUTION_TEXT.format(
        multiple=args.systematic_rate_multiple, min_count=args.systematic_min_count
    )
    corrupted_pair_note = CORRUPTED_PAIR_NOTE.format(modes=", ".join(args.corrupted_pair_modes))

    html = render_contingency_html(
        manifest,
        reviewed_units,
        not_yet_reviewed,
        merge_warnings,
        contingency_rows,
        corrupted_pair,
        fix_list,
        caution_text,
        corrupted_pair_note,
        args,
    )
    write_outputs(
        output_dir,
        contingency_rows,
        corrupted_pair,
        fix_list,
        merge_warnings,
        html,
        caution_text,
        corrupted_pair_note,
    )

    print(f"Wrote {output_dir / 'contingency.csv'}")
    print(f"Wrote {output_dir / 'corrupted_pair_rate.csv'}")
    print(f"Wrote {output_dir / 'ranked_fix_list.csv'}")
    print(f"Wrote {output_dir / 'contingency_report.html'}")
    if merge_warnings:
        print(f"Wrote {output_dir / 'merge_warnings.json'}")


if __name__ == "__main__":
    main()

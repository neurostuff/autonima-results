#!/usr/bin/env python3
"""Diff a retest run against the run it replicates, stage by stage.

The retest re-executes every LLM stage against a frozen corpus -- same search results, same
retrieved full text -- so any difference between the two runs is model nondeterminism rather than
a change of inputs. This reports where that lands.

Two things it deliberately reports separately:

- **Decision instability** -- what fraction of individual calls came out differently. This is the
  number people expect, and on its own it is not very informative: a screener can flip 5% of
  borderline abstracts and change nothing downstream.
- **Consequence** -- whether the final study set and coordinate pool actually moved. A large flip
  rate that does not reach the studyset means the spec is loose in places that do not matter,
  which is a much better result than the bare rate suggests, and the reverse is much worse.

Map-level drift is left to `scripts/compare_meta_to_benchmark.py`, which already knows how to
score a run against its manual benchmark; this script reports the studyset and coordinate deltas
that feed it.

Usage:
    python experiments/e2e_retest/scripts/compare_retest.py \\
        --original projects/executive_function/v3 \\
        --retest   experiments/e2e_retest/executive_function/v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _screening_decisions(run: Path, artifact: str) -> Dict[str, str]:
    data = _load(run / "outputs" / artifact) or {}
    rows = data.get("screening_results") or []
    out: Dict[str, str] = {}
    for row in rows:
        pmid = str(row.get("study_id") or "").strip()
        if pmid:
            out[pmid] = "include" if "includ" in str(row.get("decision", "")).lower() else "exclude"
    return out


def _annotation_decisions(run: Path) -> Dict[Tuple[str, str], bool]:
    data = _load(run / "outputs" / "annotation_results.json")
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    out: Dict[Tuple[str, str], bool] = {}
    for row in data or []:
        key = (str(row.get("analysis_id")), str(row.get("annotation_name")))
        out[key] = bool(row.get("include"))
    return out


def _coordinates(run: Path) -> Dict[str, list]:
    """Per-study coordinate triples, order-insensitive at comparison time."""
    data = _load(run / "outputs" / "coordinate_parsing_results.json") or {}
    out: Dict[str, list] = {}
    for study in data.get("studies") or []:
        pmid = str(study.get("pmid") or "").strip()
        if not pmid:
            continue
        points = []
        for analysis in study.get("analyses") or []:
            for point in analysis.get("points") or analysis.get("coordinates") or []:
                if isinstance(point, dict):
                    xyz = (point.get("x"), point.get("y"), point.get("z"))
                elif isinstance(point, (list, tuple)) and len(point) >= 3:
                    xyz = tuple(point[:3])
                else:
                    continue
                if all(isinstance(v, (int, float)) for v in xyz):
                    points.append(tuple(round(float(v)) for v in xyz))
        out[pmid] = points
    return out


def _studyset(run: Path) -> Dict[str, int]:
    """PMIDs in the final NiMADS studyset, with their coordinate counts."""
    data = _load(run / "outputs" / "nimads_studyset.json") or {}
    out: Dict[str, int] = {}
    for study in data.get("studies") or []:
        pmid = str(study.get("pmid") or (study.get("metadata") or {}).get("pmid") or "").strip()
        if not pmid:
            continue
        out[pmid] = sum(
            len(a.get("points") or []) for a in (study.get("analyses") or [])
        )
    return out


def _flip_report(label: str, a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[str, Any]:
    shared = set(a) & set(b)
    flips = [k for k in shared if a[k] != b[k]]
    return {
        "stage": label,
        "compared": len(shared),
        "only_original": len(set(a) - set(b)),
        "only_retest": len(set(b) - set(a)),
        "flips": len(flips),
        "flip_rate": (len(flips) / len(shared)) if shared else None,
        "examples": flips[:5],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--original", type=Path, required=True, help="the run being replicated")
    parser.add_argument("--retest", type=Path, required=True, help="the replicate")
    parser.add_argument("--json-out", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    orig, retest = args.original, args.retest
    report: Dict[str, Any] = {"original": str(orig), "retest": str(retest), "stages": []}

    # --- guard: the whole comparison is meaningless if the inputs were not actually frozen.
    so = {s.get("pmid") for s in (_load(orig / "outputs" / "search_results.json") or {}).get("studies", [])}
    sr = {s.get("pmid") for s in (_load(retest / "outputs" / "search_results.json") or {}).get("studies", [])}
    if so != sr:
        print(f"!! SEARCH DRIFTED: {len(so)} -> {len(sr)} studies, "
              f"+{len(sr - so)} / -{len(so - sr)}. Downstream deltas are confounded.")
    else:
        print(f"search frozen: {len(so)} studies, identical\n")
    report["search_frozen"] = so == sr

    print(f"{'stage':<14}{'compared':>9}{'flips':>7}{'rate':>8}{'only-A':>8}{'only-B':>8}")
    print("-" * 54)
    for label, artifact in (("abstract", "abstract_screening_results.json"),
                            ("fulltext", "fulltext_screening_results.json")):
        row = _flip_report(label, _screening_decisions(orig, artifact),
                           _screening_decisions(retest, artifact))
        report["stages"].append(row)
        rate = f"{row['flip_rate']*100:.2f}%" if row["flip_rate"] is not None else "-"
        print(f"{label:<14}{row['compared']:>9,}{row['flips']:>7,}{rate:>8}"
              f"{row['only_original']:>8}{row['only_retest']:>8}")

    ann = _flip_report("annotation", _annotation_decisions(orig), _annotation_decisions(retest))
    report["stages"].append(ann)
    rate = f"{ann['flip_rate']*100:.2f}%" if ann["flip_rate"] is not None else "-"
    print(f"{'annotation':<14}{ann['compared']:>9,}{ann['flips']:>7,}{rate:>8}"
          f"{ann['only_original']:>8}{ann['only_retest']:>8}")

    # --- parsing: compare coordinate multisets per study rather than decisions
    from collections import Counter
    co, cr = _coordinates(orig), _coordinates(retest)
    shared = set(co) & set(cr)
    changed = [p for p in shared if Counter(co[p]) != Counter(cr[p])]
    tot_o = sum(len(v) for v in co.values())
    tot_r = sum(len(v) for v in cr.values())
    print(f"\nparsing       studies compared {len(shared):,}   "
          f"with different coordinates {len(changed):,} "
          f"({len(changed)/len(shared)*100:.1f}%)" if shared else "\nparsing       no overlap")
    print(f"              total coordinates {tot_o:,} -> {tot_r:,}  ({tot_r - tot_o:+,})")
    report["parsing"] = {"studies_compared": len(shared), "studies_changed": len(changed),
                         "coordinates_original": tot_o, "coordinates_retest": tot_r}

    # --- consequence: does any of it reach the final studyset?
    so_, sr_ = _studyset(orig), _studyset(retest)
    if so_ or sr_:
        added, removed = set(sr_) - set(so_), set(so_) - set(sr_)
        moved = [p for p in set(so_) & set(sr_) if so_[p] != sr_[p]]
        print(f"\nstudyset      {len(so_):,} -> {len(sr_):,} studies   "
              f"+{len(added)} / -{len(removed)}   {len(moved)} with changed coordinate counts")
        print(f"              coordinates {sum(so_.values()):,} -> {sum(sr_.values()):,}")
        report["studyset"] = {"original": len(so_), "retest": len(sr_),
                              "added": sorted(added)[:20], "removed": sorted(removed)[:20],
                              "n_added": len(added), "n_removed": len(removed),
                              "coord_changed_studies": len(moved)}
    else:
        print("\nstudyset      not built yet in one or both runs")

    # --- what this execution cost, now that runs record it
    usage = (_load(retest / "outputs" / "execution_progress.json") or {}).get("usage_total")
    if usage:
        cost = usage.get("cost_usd")
        print(f"\nretest cost   {usage['calls']:,} calls, "
              f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out, "
              + (f"${cost:,.2f}" if cost is not None else "cost unknown"))
        report["retest_usage"] = usage

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

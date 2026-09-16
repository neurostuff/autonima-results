#!/usr/bin/env python3
"""Every number the manuscript briefs quote, re-derived from the report CSVs.

WHY THIS EXISTS

MANUSCRIPT.md carries a writing brief per section listing the numbers to quote. Those numbers
were NOT transcribed from NATURE_METHODS_SKELETON.md or PAPER_OUTLINE.md, because both have
drifted from the artifacts. Four errors found on 2026-09-10 while building the skeleton:

    parsing "90% vs 31%, +60 pts"        actual 91.4% vs 33.3%, +58.1 pts
    Figure 4 "+0.110, p=3.5e-06, 31/35"  superseded; 35-column era
    outline S7 "0.495 vs 0.394, 30/35"   superseded; 35-column era
    "35 columns" in five places          32; three substance-use columns are excluded

Run this to regenerate the brief numbers, or to check the ones in MANUSCRIPT.md after any re-run.
Each block prints the CSV it came from so a reader can go straight to the source.

    pixi run python scripts/manuscript_numbers.py

Computations mirror scripts/make_nature_methods_figures.py so the text cannot disagree with the
figure it sits beside. Where a figure applies benchmark_exclusions, so does this.
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_exclusions import filter_rows  # noqa: E402

REPORTS = REPO_ROOT / "reports"


def read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def head(title: str, source: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n  source: {source}\n{'-' * 78}")


def result1() -> None:
    head("RESULT 1 / Figure 1 — the pipeline, the benchmark, and the unit",
         "reports/analysis_counts.csv, reports/cross_project_best_baseline.csv")
    rows = filter_rows(read(REPORTS / "cross_project_best_baseline.csv"), announce=False)
    projects = sorted({r["project"] for r in rows})
    print(f"benchmark columns scored      {len(rows)}")
    print(f"projects                      {len(projects)}")
    counts = filter_rows(read(REPORTS / "analysis_counts.csv"), announce=False)
    a_pipe = [int(r["n_analyses_autonima"]) for r in counts if r.get("n_analyses_autonima")]
    f_pipe = [int(r["n_foci_autonima"]) for r in counts if r.get("n_foci_autonima")]
    if a_pipe:
        print(f"analyses per map, pipeline    median {int(st.median(a_pipe))} "
              f"(range {min(a_pipe)}-{max(a_pipe)})")
        print(f"foci per map, pipeline        median {int(st.median(f_pipe))}")

    # Selection beats volume: the pipeline wins while pooling FEWER analyses, so the advantage
    # cannot be a sample-size effect. Compared against the arm Figure 4 actually scores it
    # against -- best_available_source is `targeted` for 31 of 32 columns and `broad` for one --
    # and separately against the screening-only arm, which is the Figure 5 contrast.
    key = {"targeted": "n_analyses_targeted", "broad": "n_analyses_all_abstract"}
    by_col = {(r["project"], r["manual_annotation"]): r for r in counts}
    base = filter_rows(read(REPORTS / "cross_project_best_baseline.csv"), announce=False)
    for label, getter in (
        ("its Figure 4 baseline", lambda r, c: c.get(key.get(r["best_available_source"], ""))),
        ("the screening-only arm", lambda r, c: c.get("n_analyses_all_analyses")),
    ):
        pairs, won_fewer = [], 0
        for r in base:
            c = by_col.get((r["project"], r["manual_annotation"]))
            other = getter(r, c) if c else None
            if not (c and other and c.get("n_analyses_autonima")):
                continue
            mine, theirs = int(c["n_analyses_autonima"]), int(other)
            pairs.append((mine, theirs))
            if mine < theirs and float(r["delta_vs_available"]) > 0:
                won_fewer += 1
        fewer = sum(1 for a, b in pairs if a < b)
        print(f"pools fewer than {label:<24} {fewer}/{len(pairs)} columns"
              f"   (and wins there: {won_fewer})")


def result2() -> None:
    head("RESULT 2 / Figure 2 — screening is nearly free; annotation spends recall",
         "reports/attainable_recall_by_stage.csv, reports/stage_precision_recall.csv")
    at, rw = {}, {}
    for r in read(REPORTS / "attainable_recall_by_stage.csv"):
        at.setdefault(r["project"], {})[r["stage"]] = float(r["recall_attainable"])
        rw.setdefault(r["project"], {})[r["stage"]] = float(r["recall_raw"])
    pr = {}
    for r in read(REPORTS / "stage_precision_recall.csv"):
        if r["precision"] != "":
            pr.setdefault(r["project"], {})[r["stage"]] = float(r["precision"])
    stages = ["search", "abstract", "fulltext", "annotation"]
    ma = {s: st.mean(at[p][s] for p in at) for s in stages}
    mr = {s: st.mean(rw[p][s] for p in rw) for s in stages}
    mp = {s: st.mean(pr[p][s] for p in pr) for s in stages}
    print(f"n projects                    {len(at)}")
    print("stage        precision   attainable recall   raw recall")
    for s in stages:
        print(f"  {s:<10} {mp[s]:>9.3f} {ma[s]:>18.3f} {mr[s]:>12.3f}")
    print(f"\nscreening (abstract+full-text): precision {mp['search']:.3f} -> {mp['fulltext']:.3f}"
          f", attainable recall {ma['search']:.3f} -> {ma['fulltext']:.3f} "
          f"({ma['fulltext']-ma['search']:+.3f})")
    print(f"annotation:                     precision {mp['fulltext']:.3f} -> "
          f"{mp['annotation']:.3f}, attainable recall {ma['fulltext']:.3f} -> "
          f"{ma['annotation']:.3f} ({ma['annotation']-ma['fulltext']:+.3f})")
    dp = [pr[p]["annotation"] - pr[p]["fulltext"] for p in pr]
    da = [at[p]["annotation"] - at[p]["fulltext"] for p in at]
    dr = [rw[p]["annotation"] - rw[p]["fulltext"] for p in rw]
    n = len(dp)
    print(f"\nannotation stage, per project (n={n}):")
    print(f"  precision improves in       {sum(x > 0 for x in dp)}/{n}   mean {st.mean(dp):+.3f}")
    print(f"  attainable recall falls in  {sum(x < 0 for x in da)}/{n}   mean {st.mean(da):+.3f}")
    print(f"  raw recall falls in         {sum(x < 0 for x in dr)}/{n}   mean {st.mean(dr):+.3f}")
    print(f"  precision gain > recall loss: {sum(1 for a, b in zip(dp, da) if a > -b)}/{n} "
          f"attainable   vs   {sum(1 for a, b in zip(dp, dr) if a > -b)}/{n} raw")
    lo = min(at, key=lambda p: at[p]["fulltext"])
    print(f"\nlowest at full text:          {lo} {at[lo]['fulltext']:.3f}")
    ceil = [rw[p]["search"] for p in rw]
    print(f"corpus ceiling (search, raw): mean {st.mean(ceil):.1%} "
          f"({min(ceil):.0%}-{max(ceil):.0%})")


def result3() -> None:
    head("RESULT 3 / Figure 3 — recover the analyses, then select among them",
         "reports/cross_project_analysis/{parsing_metrics_by_project,annotation_aggregates}.csv")
    rows = read(REPORTS / "cross_project_analysis" / "parsing_metrics_by_project.csv")
    llm = [float(r["manual_matched_pct"]) * 100 for r in rows if r["manual_matched_pct"]]
    tab = [float(r["table_only_baseline_matched_pct"]) * 100 for r in rows
           if r["table_only_baseline_matched_pct"]]
    print(f"3a parsing, n={len(llm)} projects")
    print(f"  LLM parsing recovers        mean {st.mean(llm):.1f}%  median {st.median(llm):.1f}%")
    print(f"  tables-only baseline        mean {st.mean(tab):.1f}%  median {st.median(tab):.1f}%")
    print(f"  margin                      {st.mean(llm)-st.mean(tab):+.1f} pts, "
          f"LLM ahead in {sum(1 for a, b in zip(llm, tab) if a > b)}/{len(llm)}")

    # figure3 panel b EXCLUDES dementia -- its source meta-analysis pools several studies into
    # one gold analysis, so per-analysis annotation cannot be scored against it. Omitting this
    # filter here computed the lift over nine projects and reported 3.1x against the figure's
    # 3.3x, which is exactly the text-disagrees-with-its-own-figure failure this script exists
    # to prevent. Panel a keeps all nine; only panel b drops it.
    ann = [r for r in read(REPORTS / "cross_project_analysis" / "annotation_aggregates.csv")
           if r["level"] == "analysis" and r["variant"] == "exhausted_manual_assumption"
           and r["scope"] == "project" and r["mode_id"] == "combined"
           and r["project_name"] != "dementia"]
    pts = []
    for r in ann:
        tp, fp, fn, tn = (int(float(r[k])) for k in ("tp", "fp", "fn", "tn"))
        N, P = tp + fp + fn + tn, tp + fn
        if not (N and P):
            continue
        pts.append((r["project_name"], float(r["recall"]), float(r["precision"]), P / N,
                    tp - fp, N, P, tp + fp))
    lifts = [p[2] / p[3] for p in pts]
    print(f"\n3b annotation, n={len(pts)} projects")
    print(f"  precision                   mean {st.mean(p[2] for p in pts):.3f}")
    print(f"  prevalence (no-skill)       mean {st.mean(p[3] for p in pts):.3f}")
    print(f"  lift over prevalence        mean {st.mean(lifts):.1f}x  "
          f"(range {min(lifts):.1f}-{max(lifts):.1f})")
    js = [(p[1] - (p[7] - (p[1] * p[6])) / (p[5] - p[6])) for p in pts]
    print("  (ROC alternate, figure3alt)")
    tprfpr = []
    for name, recall, prec, prev, _, N, P, k in pts:
        tp = recall * P
        fpr = (k - tp) / (N - P)
        tprfpr.append(recall - fpr)
    print(f"  mean TPR - FPR              {st.mean(tprfpr):.2f}  (chance = 0)")
    worst = min(pts, key=lambda p: p[2] / p[3])
    best = max(pts, key=lambda p: p[2] / p[3])
    print(f"  lowest lift                 {worst[0]} {worst[2]/worst[3]:.1f}x "
          f"(precision {worst[2]:.3f}, prevalence {worst[3]:.3f})")
    print(f"  highest lift                {best[0]} {best[2]/best[3]:.1f}x "
          f"(precision {best[2]:.3f}, prevalence {best[3]:.3f})")


def result4() -> None:
    head("RESULT 4 / Figure 4 — the whole pipeline beats a search-only synthesis",
         "reports/cross_project_best_baseline_stats.csv, reports/cross_project_best_baseline.csv")
    for r in read(REPORTS / "cross_project_best_baseline_stats.csv"):
        print(f"vs {r['comparison']:<16} n={r['n_columns']} columns, {r['n_projects']} projects, "
              f"metric {r['metric']}")
        print(f"  pipeline {float(r['autonima_mean']):.3f}  baseline "
              f"{float(r['baseline_mean']):.3f}   mean D {float(r['mean_delta']):+.4f}  "
              f"median D {float(r['median_delta']):+.4f}")
        print(f"  95% CI [{float(r['ci_low']):+.3f}, {float(r['ci_high']):+.3f}]   "
              f"ahead {r['columns_ahead']}  tied {r['columns_tied']}  behind {r['columns_behind']}"
              f"   sign test p={r['sign_test_p']}")
        print(f"  {r['bootstrap']}, {r['resamples']} resamples, seed {r['seed']}\n")


def result5() -> None:
    head("RESULT 5 / Figure 5 — the gain is analysis selection, not paper selection",
         "reports/selection_decomposition.csv")
    rows = filter_rows(read(REPORTS / "selection_decomposition.csv"), announce=False)
    gp = [float(r["gain_paper_selection"]) for r in rows]
    ga = [float(r["gain_analysis_selection"]) for r in rows]
    gt = [float(r["gain_total"]) for r in rows]
    n = len(rows)
    print(f"n={n} columns, {len({r['project'] for r in rows})} projects, metric r^2")
    for name, v in (("paper selection", gp), ("analysis selection", ga), ("total", gt)):
        print(f"  {name:<20} median {st.median(v):+.3f}  mean {st.mean(v):+.3f}  "
              f"positive in {sum(x > 0 for x in v)}/{n}")
    print(f"  arms: baseline {st.mean(float(r['r2_baseline']) for r in rows):.3f} -> "
          f"screening-only {st.mean(float(r['r2_screening_only']) for r in rows):.3f} -> "
          f"pipeline {st.mean(float(r['r2_pipeline']) for r in rows):.3f}")

    head("SUPPLEMENTARY S5 — every column against its own size-matched null",
         "reports/annotation_bootstrap_null.csv")
    nr = [r for r in read(REPORTS / "annotation_bootstrap_null.csv") if r.get("status") == "ok"]
    nr = filter_rows(nr, project_key="project", announce=False)
    d = [float(r["observed_r2"]) - float(r["null_mean"]) for r in nr]
    beat = sum(1 for r in nr if float(r["p_value"]) < 0.05)
    print(f"n={len(nr)} columns   median D R^2 {st.median(d):+.3f}   "
          f"{beat}/{len(nr)} clear p<0.05")


def result6() -> None:
    head("RESULT 6 / Supplementary S1 — cost and scale",
         "!! NOT CSV-DERIVED — hardcoded as COST_PER_STAGE in make_nature_methods_figures.py, "
         "transcribed from PAPER_OUTLINE.md S1")
    print("  abstract screening   $0.0023/call      full-text screening  $0.0138/call")
    print("  coordinate parsing   $0.0059/call      annotation           $0.0211/call")
    print("  mean per project     $21.55            all nine from scratch  $194")
    print("  per study reaching the map  $0.085 median")
    print("\n  The module docstring says these 'should move into a generated CSV before")
    print("  submission'. Until they do, this is the one block in the manuscript that is not")
    print("  machine-checked. Verify against usage_total in execution_progress.json.")


def supplementary() -> None:
    head("SUPPLEMENTARY S2 — mis-specification, not overfitting",
         "reports/tier_progression.csv")
    # Mirrors figureS2 exactly: average a project's columns first, then take paired PROJECT
    # deltas, skipping projects where the two tiers resolve to the same run (no contrast there).
    rows = filter_rows(read(REPORTS / "tier_progression.csv"), announce=False)
    by, runs = {}, {}
    for r in rows:
        if not r["r2"]:
            continue
        by.setdefault(r["project"], {}).setdefault(r["tier"], []).append(float(r["r2"]))
        runs.setdefault(r["project"], {})[r["tier"]] = r["run"]
    means = {p: {t: st.mean(v) for t, v in tiers.items()} for p, tiers in by.items()}

    def seg(a, b):
        return {p: m[b] - m[a] for p, m in means.items()
                if a in m and b in m and runs[p].get(a) != runs[p].get(b)}

    vm, mb = seg("verbatim", "manual"), seg("manual", "best")
    print(f"  verbatim -> manual   mean {st.mean(vm.values()):+.3f}  "
          f"(n={len(vm)} projects: {', '.join(sorted(vm))})")
    print(f"  manual   -> best     mean {st.mean(mb.values()):+.3f}  "
          f"(n={len(mb)} projects: {', '.join(sorted(mb))})")
    if st.mean(mb.values()):
        print(f"  ratio                {st.mean(vm.values())/st.mean(mb.values()):.1f}x")
    print(f"  NOTE n={len(vm)} and n={len(mb)} projects; an indication, not an estimate. "
          "Do not report without the n.")

    head("SUPPLEMENTARY S3 — how much of the low precision is a pool mismatch?",
         "reports/stage_precision_recall.csv, reports/stage_precision_recall_allstudies.csv")
    def load(name):
        out = {}
        for r in read(REPORTS / name):
            if r["precision"] != "":
                out.setdefault(r["project"], {})[r["stage"]] = (float(r["precision"]),
                                                                float(r["recall"]))
        return out
    search, fixed = load("stage_precision_recall.csv"), load("stage_precision_recall_allstudies.csv")
    shared = sorted(set(search) & set(fixed))
    print(f"  n={len(shared)} projects with both arms: {', '.join(shared)}")
    for stage in ("search", "abstract", "fulltext", "annotation"):
        ok = [p for p in shared if stage in search[p] and stage in fixed[p]]
        if not ok:
            continue
        dp = st.mean(fixed[p][stage][0] - search[p][stage][0] for p in ok)
        dr = st.mean(fixed[p][stage][1] - search[p][stage][1] for p in ok)
        sp = st.mean(search[p][stage][0] for p in ok)
        print(f"  {stage:<11} precision {sp:.3f} -> {sp+dp:.3f} ({dp:+.3f})   "
              f"recall {dr:+.3f}  (n={len(ok)})")
    print("  recall is the control: it barely moves, so the precision gain is a corpus "
          "difference, not a screening effect.")

    head("SUPPLEMENTARY S4 — a term is not an analysis",
         "reports/text_to_map_baselines.csv")
    rows = filter_rows(read(REPORTS / "text_to_map_baselines.csv"), announce=False)
    for label, key in (("NeuroQuery", "neuroquery_r2"), ("NeuroVLM", "neurovlm_r2"),
                       ("best search baseline", "best_baseline_r2"), ("pipeline", "autonima_r2")):
        v = [float(r[key]) for r in rows if r.get(key)]
        if v:
            print(f"  {label:<22} mean r^2 {st.mean(v):.3f}  median {st.median(v):.3f}  "
                  f"(n={len(v)})")

    head("SUPPLEMENTARY S6 — raw denominator, retrieval as its own stage",
         "reports/gold_survival_by_stage.csv")
    rows = [r for r in read(REPORTS / "gold_survival_by_stage.csv")
            if r["family"] == "canonical" and r["cumulative_recall"]]
    by = {}
    for r in rows:
        by.setdefault(r["project"], {})[r["stage"]] = float(r["cumulative_recall"]) * 100
    for name, a, b in (("abstract screening", "search", "abstract"),
                       ("full-text retrieval", "abstract", "retrieval"),
                       ("full-text screening", "retrieval", "fulltext")):
        v = [by[p][a] - by[p][b] for p in by]
        print(f"  {name:<20} median {st.median(v):.1f}pp  max {max(v):.1f}pp  "
              f"projects >5pp {sum(1 for x in v if x > 5)}/{len(v)}")
    print("  retrieval counts USABLE text: fulltext_available AND not fulltext_incomplete.")

    head("FALSE-POSITIVE COMPOSITION — did the researchers ever see them?",
         "reports/false_positive_composition.csv")
    rows = read(REPORTS / "false_positive_composition.csv")
    ft = [r for r in rows if r["stage"] == "fulltext"]
    if ft:
        tot = sum(int(r["n_false_positives"]) for r in ft)
        seen = sum(int(r["fp_considered_and_rejected"]) for r in ft)
        print(f"  n={len(ft)} projects with a fixed-pool arm")
        print(f"  full-text false positives   {tot}, of which {tot-seen} "
              f"({(tot-seen)/tot:.0%}) were never in the researchers' pool")
        print("  Do not say 'most false positives are a pool artefact' without the per-project "
              "split.")


def main() -> int:
    print("Manuscript brief numbers, re-derived from reports/. "
          "Every figure here is what MANUSCRIPT.md should quote.")
    for fn in (result1, result2, result3, result4, result5, result6, supplementary):
        try:
            fn()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"\n!! {fn.__name__} failed: {type(exc).__name__}: {exc}")
    print(f"\n{'=' * 78}\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

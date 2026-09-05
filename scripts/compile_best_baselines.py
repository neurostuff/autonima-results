#!/usr/bin/env python3
"""Pool every sub-analysis into one autonima-vs-best-baseline table.

WHY THIS EXISTS

Per-project baseline tables are not directly comparable, because the baselines are not all the
same kind of thing. Most sub-analyses have a targeted arm -- a narrowed search a real competitor
would plausibly run -- but some have none, because the sub-analysis is not a separable search
topic. emotion_regulation_2022's decrease/increase/maintain are contrast DIRECTIONS within one
paradigm: there is no query that targets "down-regulation" as opposed to "up-regulation", so the
broad arm is not a fallback there, it is genuinely the best baseline anyone could build.

Averaging a project's targeted margin over columns that have no targeted arm silently mixes the
two. This script instead resolves the best baseline PER COLUMN and pools the columns, so every
row compares autonima against the strongest competitor that could exist for that particular
sub-analysis.

THE RULE: use the targeted arm wherever one was defined, and the broad arm only where targeting
is impossible.

  available  <- PRIMARY. The targeted arm when one exists, else the broad arm. This is the
             honest counterfactual: what would a competent practitioner aiming at this column
             actually have built? If they would have narrowed the search, that narrowed search is
             their baseline, whether or not it happens to score well.
  strongest  max(targeted, broad). Reported as a robustness check only. It is tempting because it
             looks conservative, but it is the wrong counterfactual -- it lets the baseline switch
             arms per column with hindsight, picking whichever scored better after the fact. No
             practitioner gets to do that.

In 8 of 35 columns the broad arm outscores the targeted one, i.e. narrowing the search made the
baseline WORSE. That is a result about when sub-targeting helps, not a reason to substitute the
broad arm. The two definitions differ by 0.004 anyway, so the conclusion does not turn on it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import math
import random
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


METRICS = ("dice", "r2", "pearson_r")


def load_columns(projects_root: Path, metric: str) -> dict[tuple[str, str], dict[str, float]]:
    by: dict[tuple[str, str], dict[str, float]] = collections.defaultdict(dict)
    for f in sorted(glob.glob(str(projects_root / "*" / "reports" / "baseline_vs_autonima.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                by[(r["project"], r["manual_annotation"])][r["arm"]] = float(r[metric])
            except (ValueError, TypeError, KeyError):
                continue
    return by


def cluster_bootstrap(
    deltas_by_project: dict[str, list[float]],
    resamples: int,
    seed: int,
) -> dict[str, float]:
    """Percentile CI for the pooled mean delta, resampling PROJECTS rather than columns.

    The 35 columns are not 35 independent observations: cue_reactivity's three columns come from
    one corpus, one search and one screening run, so resampling columns treats correlated
    measurements as independent and reports a narrower interval than the evidence supports. This
    resamples whole projects, carrying each one's columns along with it.

    The correction is not cosmetic. emotion_regulation_2022 alone carries a mean delta of +0.369
    against a pooled +0.101, so whether it lands in a given resample moves the mean a long way --
    which is precisely the uncertainty a column bootstrap hides. Measured at 20,000 resamples the
    cluster interval came out 1.8x wider than the naive one.

    A percentile bootstrap is used rather than a t-based interval because the deltas are visibly
    right-skewed (mean +0.101 against median +0.059), so a symmetric interval would be misplaced.
    """
    rng = random.Random(seed)
    projects = sorted(deltas_by_project)
    means = []
    for _ in range(resamples):
        drawn = rng.choices(projects, k=len(projects))
        values = [v for p in drawn for v in deltas_by_project[p]]
        if values:
            means.append(st.mean(values))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    return {"ci_low": lo, "ci_high": hi, "n_projects": len(projects), "resamples": len(means)}


def sign_test(deltas: list[float]) -> float:
    """Exact two-sided binomial test that autonima beats the baseline no more often than chance.

    Ties count as non-wins rather than being dropped, and that choice matters here.

    The usual convention discards ties as evidence for neither side. Under dice that is actively
    misleading, because dice is a thresholded overlap measure and produces genuine exact ties where
    r2 resolves a difference: the same 35 columns give 30 wins / 5 losses / 0 ties under r2, but
    30 / 1 / 4 under dice. Dropping those four shrinks n from 35 to 31 and drops p from 2.2e-05 to
    3.0e-08 -- three orders of magnitude of apparent significance bought entirely by the metric
    being coarser.

    Counting a tie as a non-win keeps the test measuring the directional claim ("autonima beats the
    baseline") on the full column set, and makes the result metric-stable: p = 2.2e-05 under both.
    """
    wins = sum(1 for d in deltas if d > 0)
    trials = len(deltas)
    if not trials:
        return 1.0
    tail = sum(math.comb(trials, i) for i in range(wins, trials + 1)) / 2 ** trials
    return min(2 * tail, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects-root", type=Path, default=REPO_ROOT / "projects")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "cross_project_best_baseline.csv")
    ap.add_argument("--resamples", type=int, default=20000,
                    help="bootstrap resamples for the pooled CI (default 20000)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed, so the CI is reproducible")
    ap.add_argument("--metric", choices=METRICS, default="dice",
                    help="map-similarity metric. dice is the default because it is the only one "
                         "annotation_value.csv also carries, so Results 4 and 5 can report the "
                         "same units. The headline is unchanged under all three -- the same 30 of "
                         "35 columns win -- and re-running with --metric r2 reproduces that check")
    args = ap.parse_args()

    by = load_columns(args.projects_root, args.metric)
    rows = []
    for (proj, col), arms in sorted(by.items()):
        auto = arms.get("autonima")
        sub, broad = arms.get("baseline_sub"), arms.get("baseline_broad")
        if auto is None:
            continue
        candidates = {k: v for k, v in (("targeted", sub), ("broad", broad)) if v is not None}
        if not candidates:
            continue
        avail_src = "targeted" if sub is not None else "broad"
        avail = candidates[avail_src]
        strong_src = max(candidates, key=lambda k: candidates[k])
        strong = candidates[strong_src]
        rows.append({
            "project": proj, "manual_annotation": col, "metric": args.metric,
            "autonima": round(auto, 4),
            "baseline_sub": round(sub, 4) if sub is not None else "",
            "baseline_broad": round(broad, 4) if broad is not None else "",
            "targeting_possible": "yes" if sub is not None else "no",
            "best_available": round(avail, 4), "best_available_source": avail_src,
            "strongest": round(strong, 4), "strongest_source": strong_src,
            "delta_vs_available": round(auto - avail, 4),
            "delta_vs_strongest": round(auto - strong, 4),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    a = [r["autonima"] for r in rows]
    stats_rows = []
    for lbl, key in (("best AVAILABLE (targeted if any)", "best_available"),
                     ("STRONGEST (max of targeted, broad)", "strongest")):
        b = [r[key] for r in rows]
        d = [x - y for x, y in zip(a, b)]
        by_project: dict[str, list[float]] = collections.defaultdict(list)
        for row, delta in zip(rows, d):
            by_project[row["project"]].append(delta)
        boot = cluster_bootstrap(by_project, args.resamples, args.seed)
        p = sign_test(d)
        print(f"  {lbl:<36} autonima {st.mean(a):.3f}  baseline {st.mean(b):.3f}  "
              f"delta {st.mean(d):+.3f}  ahead {sum(1 for x in d if x > 0)}/{len(d)}")
        ties = sum(1 for x in d if x == 0)
        print(f"  {'':<36} 95% CI [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]  "
              f"sign test p = {p:.2e}  (median {st.median(d):+.3f}"
              f"{f', {ties} tied' if ties else ''})")
        stats_rows.append({
            "comparison": key, "metric": args.metric,
            "n_columns": len(d), "n_projects": boot["n_projects"],
            "autonima_mean": round(st.mean(a), 4), "baseline_mean": round(st.mean(b), 4),
            "mean_delta": round(st.mean(d), 4), "median_delta": round(st.median(d), 4),
            "ci_low": round(boot["ci_low"], 4), "ci_high": round(boot["ci_high"], 4),
            "columns_ahead": sum(1 for x in d if x > 0),
            "columns_tied": sum(1 for x in d if x == 0),
            "columns_behind": sum(1 for x in d if x < 0),
            "sign_test_p": f"{p:.3e}",
            "bootstrap": "cluster over projects", "resamples": boot["resamples"], "seed": args.seed,
        })

    # Written beside the per-column table so the paper quotes a generated number rather than one
    # transcribed from a console run that nobody can re-derive later.
    stats_path = args.output.with_name(args.output.stem + "_stats.csv")
    with open(stats_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0]))
        w.writeheader()
        w.writerows(stats_rows)
    no_target = [r for r in rows if r["targeting_possible"] == "no"]
    print(f"\n  columns: {len(rows)}   targetable: {len(rows) - len(no_target)}   "
          f"no targeting possible: {len(no_target)}")
    for r in no_target:
        print(f"    {r['project']}/{r['manual_annotation']}")
    def _rel(path: Path) -> str:
        return str(path.relative_to(REPO_ROOT)) if str(path).startswith(str(REPO_ROOT)) else str(path)

    print(f"\n  wrote {_rel(args.output)}")
    print(f"  wrote {_rel(stats_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

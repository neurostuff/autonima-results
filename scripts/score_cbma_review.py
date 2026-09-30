#!/usr/bin/env python3
"""Score a finished cbma-skills review against the NeuroMetaBench gold and an autonima run.

    python scripts/score_cbma_review.py --project emotion_regulation_2022 \\
        --review ~/repos/cbma-workspace/emotion_regulation_2022 --autonima-run v4 \\
        --out projects/emotion_regulation_2022/reports/cbma_skills_v1

Needs numpy, scipy and nibabel (the cbma workspace venv, which has NiMARE, is enough).

Run this from a session that did not run the review, and never write its output into the
review's workspace: it is gold-derived.

WHAT IT MEASURES, for both arms on the same footing

  screening    benchmark/compare.py's scores (search, abstract, final, final given text, kappa),
               plus every gold study the review had text for but did not include, with the
               criteria that failed. Those are the recall losses a protocol change could buy back.
  coordinates  recall of gold peaks within 2 mm, one-to-one, over the gold studies each arm
               included, and over the studies both included. Coordinates are compared as
               reported: the gold stores the papers' own coordinates labelled MNI (converting
               the arms' Talairach peaks lowers the match rate), so no transform is applied.
               Precision is not reported, because the arms export every analysis and the gold
               only the analyses the meta-analysis used.
  selection    per target, study-level precision and recall: a study counts for a target if
               any of its analyses is selected for it.
  maps         the manuscript's metric: Dice of FDR-corrected z maps above 1.96 and Pearson r of
               unthresholded z maps, inside the NiMARE brain mask (scripts/map_mask.py). Both
               arms are scored on one common mask. A cbma run saves p but not z, so z is
               z_from_p(p): isf(p) clipped at 0, which reproduces NiMARE's z maps exactly.
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import norm, pearsonr

REPO_ROOT = Path(__file__).resolve().parents[1]
NMB = REPO_ROOT.parent / "neurometabench"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from map_mask import common_mask  # noqa: E402

spec = importlib.util.spec_from_file_location("compare", REPO_ROOT / "experiments/cbma-skills/benchmark/compare.py")
compare = importlib.util.module_from_spec(spec)
sys.modules["compare"] = compare
spec.loader.exec_module(compare)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def study_of(studyset: dict) -> dict[str, str]:
    return {a["id"]: str(s.get("pmid") or s["id"]) for s in studyset["studies"] for a in s["analyses"]}


def points_by_study(studyset: dict) -> dict[str, np.ndarray]:
    out: dict[str, list] = collections.defaultdict(list)
    for s in studyset["studies"]:
        for a in s["analyses"]:
            out[str(s.get("pmid") or s["id"])].extend(p["coordinates"] for p in a["points"])
    return {k: np.array(v, float) for k, v in out.items() if v}


def peak_recall(ours: dict, gold: dict, ids, tol: float = 2.0) -> dict:
    hit = total = 0
    for pid in ids:
        g, o = gold[pid], ours.get(pid)
        total += len(g)
        if o is None:
            continue
        used: set = set()
        for q in g:
            d = np.linalg.norm(o - q, axis=1)
            for j in np.argsort(d):
                if d[j] > tol:
                    break
                if j not in used:
                    used.add(j)
                    hit += 1
                    break
    return {"studies": len(ids), "gold_peaks": total, "recovered": hit,
            "recall": round(hit / total, 4) if total else None}


def z_from_p(p: np.ndarray) -> np.ndarray:
    """One-tailed z from p, as NiMARE's MKDA z maps hold it: z = isf(p), clipped at 0.

    NiMARE's z maps (and the gold and autonima maps) are 0 wherever p >= 0.5; this
    reproduces them to within 3e-7. The earlier plain isf(p) got two things wrong on a
    cbma map:
    - negative z where 0.5 < p < 1;
    - -inf where p = 1, for voxels no kernel reaches. That dropped those voxels from the
      common mask, for every arm.

    Emotion regulation E1, reappraisal: r 0.726 with plain isf, against 0.798 now.
    """
    z = norm.isf(np.clip(p, 1e-300, 1))
    z[~np.isfinite(z)] = 0.0
    return np.maximum(z, 0.0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--autonima-run", required=True, help="run folder name under projects/<project>/")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    review = args.review.expanduser().resolve()
    project = REPO_ROOT / "projects" / args.project
    run = project / args.autonima_run
    gold_csv = project / "cbma_skills_gold" / "gold.csv"
    merged = NMB / "data" / "nimads" / args.project / "merged"
    manual_maps = NMB / "analysis" / args.project
    targets = [t["name"] for t in __import__("yaml").safe_load((review / "review.yaml").read_text())["selection"]["targets"]]
    report: dict = {"review": str(review), "autonima_run": str(run), "targets": targets}

    # ---- screening
    gold = compare.load_gold(gold_csv)
    ours, theirs = compare.skills_decisions(review), compare.autonima_decisions(run)
    report["screening"] = {"gold_final_includes": sum(1 for g in gold.values() if g.get("included")),
                           "cbma": compare.score(ours, gold), "autonima": compare.score(theirs, gold),
                           "agreement": {
                               "abstract": compare.kappa(ours["abstract_pass"], theirs["abstract_pass"],
                                                         ours["abstract_judged"] & theirs["abstract_judged"]),
                               "final": compare.kappa(ours["fulltext_include"], theirs["fulltext_include"],
                                                      ours["fulltext_judged"] & theirs["fulltext_judged"])}}
    ft = {r["pmid"]: r for r in read_jsonl(review / "decisions" / "fulltext.jsonl")}
    aut_ft = {r["study_id"]: r["decision"] for r in
              json.loads((run / "outputs" / "fulltext_screening_results.json").read_text())["screening_results"]}
    gold_pos = {p for p, g in gold.items() if g.get("included")}
    losses = []
    for p in sorted(gold_pos & set(ft), key=int):
        d = ft[p]
        if d["decision"] == "include":
            continue
        failed = [k for k, v in d["criteria"].items()
                  if (k.startswith("I") and v == "not_met") or (k.startswith("E") and v == "met")]
        losses.append({"pmid": p, "decision": d["decision"], "failed": failed, "autonima": aut_ft.get(p),
                       "reason": d["reason"]})
    report["screening"]["gold_with_text_not_included"] = losses

    # ---- coordinates
    gold_ss = json.loads((merged / "nimads_studyset.json").read_text())
    cbma_ss = json.loads((review / "results" / "nimads" / "studyset.json").read_text())
    aut_ss = json.loads((run / "outputs" / "nimads_studyset.json").read_text())
    gp, cp, ap_ = points_by_study(gold_ss), points_by_study(cbma_ss), points_by_study(aut_ss)
    inc_c, inc_a = ours["fulltext_include"], theirs["fulltext_include"]
    report["coordinates"] = {
        "cbma": peak_recall(cp, gp, sorted(set(gp) & inc_c)),
        "autonima": peak_recall(ap_, gp, sorted(set(gp) & inc_a)),
        "same_studies": {"cbma": peak_recall(cp, gp, sorted(set(gp) & inc_c & inc_a)),
                         "autonima": peak_recall(ap_, gp, sorted(set(gp) & inc_c & inc_a))},
    }

    # ---- selection, study level
    def by_target(notes, analysis_to_study):
        out = collections.defaultdict(set)
        n = collections.Counter()
        for note in notes:
            for t in targets:
                if note["note"].get(t) is True:
                    out[t].add(analysis_to_study[note["analysis"]])
                    n[t] += 1
        return out, n
    gold_notes = json.loads((merged / "nimads_annotation.json").read_text())["notes"]
    aut_notes = json.loads((run / "outputs" / "nimads_annotation.json").read_text())["notes"]
    gs, gn = by_target(gold_notes, study_of(gold_ss))
    as_, an = by_target(aut_notes, study_of(aut_ss))
    cs, cn = collections.defaultdict(set), collections.Counter()
    for r in read_jsonl(review / "decisions" / "selection.jsonl"):
        if r["include"]:
            cs[r["target"]].add(r["pmid"])
            cn[r["target"]] += 1
    sel = {}
    for t in targets:
        row = {"gold_studies": len(gs[t]), "gold_analyses": gn[t]}
        for arm, s, n in (("cbma", cs[t], cn[t]), ("autonima", as_[t], an[t])):
            tp = len(s & gs[t])
            row[arm] = {"studies": len(s), "analyses": n, "tp": tp, "fp": len(s - gs[t]), "fn": len(gs[t] - s),
                        "precision": round(tp / len(s), 4) if s else None,
                        "recall": round(tp / len(gs[t]), 4) if gs[t] else None}
        sel[t] = row
    report["selection_study_level"] = sel

    # ---- maps
    def load(path: Path) -> np.ndarray:
        return nib.load(str(path)).get_fdata()

    def dice(a, b, thr=1.96):
        a, b = a > thr, b > thr
        s = a.sum() + b.sum()
        return float(2 * (a & b).sum() / s) if s else 0.0
    maps = {}
    for t in targets:
        m_dir, a_dir, c_dir = manual_maps / t, run / "outputs" / "meta_analysis_results" / t, review / "results" / "meta" / t
        mz, az = load(m_dir / "z.nii.gz"), load(a_dir / "z.nii.gz")
        cz = z_from_p(load(c_dir / "p.nii.gz"))
        fdr = "z_corr-FDR_method-indep.nii.gz"
        mf, af, cf = load(m_dir / fdr), load(a_dir / fdr), load(c_dir / fdr)
        mask = common_mask(mz, az, cz, mf, af, cf)
        maps[t] = {"voxels_in_mask": int(mask.sum()),
                   "fdr_voxels": {"gold": int((mf[mask] > 1.96).sum()), "cbma": int((cf[mask] > 1.96).sum()),
                                  "autonima": int((af[mask] > 1.96).sum())},
                   "dice": {"cbma": round(dice(mf[mask], cf[mask]), 4), "autonima": round(dice(mf[mask], af[mask]), 4)},
                   "pearson_r": {"cbma": round(float(pearsonr(mz[mask], cz[mask])[0]), 4),
                                 "autonima": round(float(pearsonr(mz[mask], az[mask])[0]), 4)}}
    report["maps"] = maps

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "score.json").write_text(json.dumps(report, indent=1) + "\n")

    s = report["screening"]
    print(f"gold includes {s['gold_final_includes']}")
    for arm in ("cbma", "autonima"):
        f = s[arm]["final"]
        print(f"  {arm:9s} final tp {f['tp']} fp {f['fp']} fn {f['fn']}  P {f['precision']} R {f['recall']} F1 {f['f1']}")
    print(f"  gold with text not included by cbma: {len(losses)}")
    c = report["coordinates"]
    print(f"coordinates (gold peak recall, same studies): cbma {c['same_studies']['cbma']['recall']}  "
          f"autonima {c['same_studies']['autonima']['recall']}")
    for t in targets:
        r, m = sel[t], maps[t]
        print(f"{t:12s} studies P/R cbma {r['cbma']['precision']}/{r['cbma']['recall']}  autonima "
              f"{r['autonima']['precision']}/{r['autonima']['recall']} | dice {m['dice']['cbma']} vs "
              f"{m['dice']['autonima']} | r {m['pearson_r']['cbma']} vs {m['pearson_r']['autonima']}")
    print(f"wrote {args.out / 'score.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

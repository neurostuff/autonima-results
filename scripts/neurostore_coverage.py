"""How much of NeuroMetaBench can NeuroStore supply: studies, coordinates and search hits.

  python scripts/neurostore_coverage.py probe     # gold PMIDs -> NeuroStore versions (network)
  python scripts/neurostore_coverage.py search    # best's queries, translated, in NeuroStore (network)
  python scripts/neurostore_coverage.py report    # tables from the two above (offline)

Needs autonima with NeuroStore support (feat/neurostore) and NiMARE (TAL -> MNI).

probe   For every gold study (neurometabench included_studies.csv, per project's meta PMID),
        every NeuroStore version and its points. The raw snapshot (several MB, and NeuroStore
        changes as it re-ingests) goes to reports/neurostore_coverage/raw/, which is not
        committed; the report keeps what it measured.
search  Each project's best PubMed query (run_categories.yaml), translated by autonima, run in
        NeuroStore over titles and abstracts and over full text, with best's date limits.
report  coverage.csv (per project: gold, in NeuroStore, with coordinates), coordinates.csv
        (gold-peak recall of NeuroStore's coordinates against autonima's own LLM parse from the
        annotation-only run, which parsed every gold study it could retrieve), search.csv
        (recall and result counts against best's PubMed search) and gold_studies.csv (per PMID).

Gold-peak recall: the fraction of a gold study's curated peaks with a candidate point within
2 mm on every axis. A TAL point counts at its raw position or at its MNI image (NiMARE's
tal2mni), because NeuroMetaBench stores Talairach studies converted to MNI. A study with no
candidate points scores 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
NMB = REPO.parent / "neurometabench"
OUT = REPO / "reports" / "neurostore_coverage"
RAW = OUT / "raw"
PROJECTS = ["cue_reactivity", "decision_making", "dementia", "emotion_regulation_2022", "executive_function",
            "problem_solving", "social", "vbm_of_ptsd", "vbm_of_substance_use"]
TOL_MM = 2.0
POLICY = ["study_schema", "nuextract-v21", "llm", "neurostore", "pubget", "nuextract-v19", "neuroquery", "neurosynth"]


def categories():
    return yaml.safe_load((REPO / "run_categories.yaml").read_text())["projects"]


def gold_pmids() -> dict:
    rows = list(csv.DictReader(open(NMB / "data" / "included_studies.csv")))
    out = {}
    for project in PROJECTS:
        meta = str(json.loads((REPO / "projects" / project / "nmb_mappings.json").read_text())["meta_pmid"])
        out[project] = sorted({r["study_pmid"].split(".")[0] for r in rows
                               if r["meta_pmid"].split(".")[0] == meta and r["study_pmid"]})
    return out


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def probe(args) -> None:
    from autonima.neurostore.client import NeuroStoreClient

    client = NeuroStoreClient()

    def one(pmid):
        listing = client.get("base-studies/", {"pmid": pmid, "info": "true"}) or {}
        matches = [r for r in listing.get("results") or [] if str(r.get("pmid")) == pmid]
        if not matches:
            return pmid, {"in_store": False}
        base = matches[0]
        versions = []
        for listed in base.get("versions") or []:
            study = client.study(listed["id"]) or {}
            versions.append({
                "id": listed["id"], "source": listed.get("source"), "updated_at": listed.get("updated_at"),
                "analyses": [{"role": a.get("role"),
                              "points": [[p.get("coordinates"), p.get("space")] for p in a.get("points") or []]}
                             for a in study.get("analyses") or []],
            })
        return pmid, {"in_store": True, "base_id": base["id"], "has_coordinates": base.get("has_coordinates"),
                      "versions": versions}

    pmids = sorted({p for ps in gold_pmids().values() for p in ps})
    with ThreadPoolExecutor(args.workers) as pool:
        results = dict(pool.map(one, pmids))
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "gold_probe.json").write_text(json.dumps(results))
    print(f"probed {len(results)} gold PMIDs -> {RAW / 'gold_probe.json'}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search(args) -> None:
    from autonima.neurostore.client import NeuroStoreClient
    from autonima.neurostore.query import translate_pubmed_query

    client = NeuroStoreClient()
    year = lambda s: int(str(s)[:4]) if s else None  # noqa: E731
    out = {}
    for project in PROJECTS:
        best = categories()[project]["canonical"]["best"]
        config = yaml.safe_load((REPO / "projects" / project / f"{best}.yaml").read_text())["search"]
        translation = translate_pubmed_query(config["query"])
        row = {"best": best, "neurostore_query": translation.query, "dropped": translation.dropped,
               "approximated": translation.approximated}
        limits = dict(year_min=year(config.get("date_from")), year_max=year(config.get("date_to")),
                      data_type="coordinate")
        for full_text in (False, True):
            if full_text and translation.exclude:
                # As autonima does: NOT clauses stay limited to titles and abstracts.
                results = list(client.search_base_studies(translation.include, search_full_text=True, **limits))
                excluded = {r["id"] for clause in translation.exclude
                            for r in client.search_base_studies(clause, search_full_text=False, **limits)}
                results = [r for r in results if r["id"] not in excluded]
            else:
                results = list(client.search_base_studies(translation.query, search_full_text=full_text, **limits))
            key = "fulltext" if full_text else "tiab"
            row[f"{key}_base_studies"] = len(results)
            row[f"{key}_pmids"] = sorted({str(r["pmid"]) for r in results if r.get("pmid")})
        out[project] = row
        print(project, row["tiab_base_studies"], row["fulltext_base_studies"], flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "search.json").write_text(json.dumps(out))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _gold_points(project):
    studyset = json.loads((NMB / "data" / "nimads" / project / "merged" / "nimads_studyset.json").read_text())
    out = defaultdict(list)
    for study in studyset["studies"]:
        for analysis in study.get("analyses", []):
            for point in analysis.get("points", []):
                xyz = point.get("coordinates")
                if xyz and None not in xyz:
                    out[str(study["id"])].append(xyz)
    return out


def _candidates(points):
    import numpy as np
    from nimare.utils import tal2mni

    raw = [c for c, _ in points if c and None not in c]
    tal = [c for c, s in points if c and None not in c and str(s).upper().startswith("TAL")]
    return raw + (tal2mni(np.array(tal, float)).tolist() if tal else [])


def _recall(gold, candidates):
    import numpy as np

    if not candidates:
        return 0.0
    g, c = np.array(gold, float), np.array(candidates, float)
    return float((np.abs(g[:, None, :] - c[None, :, :]).max(axis=2).min(axis=1) <= TOL_MM).mean())


def _policy_points(record):
    by_source = {}
    for version in record.get("versions", []):
        points = [(p[0], p[1]) for a in version["analyses"] if (a["role"] or "result") == "result" for p in a["points"]]
        if points and version["source"] not in by_source:
            by_source[version["source"]] = points
    source = next((s for s in POLICY if s in by_source), None)
    return source, (by_source[source] if source else [])


def _autonima_points(project, run):
    studyset = json.loads((REPO / "projects" / project / run / "outputs" / "nimads_studyset.json").read_text())
    return {str(s["id"]): [(p["coordinates"], p.get("space")) for a in s.get("analyses", []) for p in a.get("points", [])]
            for s in studyset["studies"]}


def report(args) -> None:
    import logging

    logging.getLogger("nimare.utils").setLevel(logging.WARNING)
    probe_data = json.loads((RAW / "gold_probe.json").read_text())
    gold = gold_pmids()
    cats = categories()
    OUT.mkdir(parents=True, exist_ok=True)

    per_study, coverage, coords = [], [], []
    for project in PROJECTS:
        gold_xyz = _gold_points(project)
        annotation_run = cats[project]["annotation_only"]["best"]
        autonima = _autonima_points(project, annotation_run)
        c = Counter()
        rec = defaultdict(list)
        for pmid in gold[project]:
            record = probe_data.get(pmid, {})
            source, points = _policy_points(record)
            c["gold"] += 1
            c["in_neurostore"] += bool(record.get("in_store"))
            c["with_coordinates"] += source is not None
            row = {"project": project, "pmid": pmid, "in_neurostore": bool(record.get("in_store")),
                   "neurostore_source": source or "", "neurostore_points": len(points),
                   "gold_points": len(gold_xyz.get(pmid, [])), "in_autonima_parse": pmid in autonima,
                   "autonima_points": len(autonima.get(pmid, []))}
            if pmid in gold_xyz:
                row["recall_neurostore"] = round(_recall(gold_xyz[pmid], _candidates(points)), 4)
                row["recall_autonima"] = round(_recall(gold_xyz[pmid], _candidates(autonima.get(pmid, []))), 4)
                rec["all"].append((row["recall_autonima"], row["recall_neurostore"]))
                if source and pmid in autonima:
                    rec["both"].append((row["recall_autonima"], row["recall_neurostore"]))
            per_study.append(row)
        coverage.append({"project": project, **c})
        mean = lambda xs, i: round(sum(x[i] for x in xs) / len(xs), 3) if xs else ""  # noqa: E731
        coords.append({"project": project, "annotation_only_run": annotation_run,
                       "gold_studies_with_points": len(rec["all"]),
                       "recall_autonima_all": mean(rec["all"], 0), "recall_neurostore_all": mean(rec["all"], 1),
                       "studies_in_both": len(rec["both"]),
                       "recall_autonima_both": mean(rec["both"], 0), "recall_neurostore_both": mean(rec["both"], 1)})

    def total(rows, keys):
        return {k: sum(r[k] for r in rows) for k in keys}

    coverage.append({"project": "TOTAL", **total(coverage, ["gold", "in_neurostore", "with_coordinates"])})
    by_pmid_all = [r for r in per_study if "recall_neurostore" in r]
    by_pmid_both = [r for r in by_pmid_all if r["neurostore_source"] and r["in_autonima_parse"]]
    coords.append({"project": "TOTAL", "annotation_only_run": "",
                   "gold_studies_with_points": len(by_pmid_all),
                   "recall_autonima_all": round(sum(r["recall_autonima"] for r in by_pmid_all) / len(by_pmid_all), 3),
                   "recall_neurostore_all": round(sum(r["recall_neurostore"] for r in by_pmid_all) / len(by_pmid_all), 3),
                   "studies_in_both": len(by_pmid_both),
                   "recall_autonima_both": round(sum(r["recall_autonima"] for r in by_pmid_both) / len(by_pmid_both), 3),
                   "recall_neurostore_both": round(sum(r["recall_neurostore"] for r in by_pmid_both) / len(by_pmid_both), 3)})

    searches = []
    if (RAW / "search.json").exists():
        search_data = json.loads((RAW / "search.json").read_text())
        for project in PROJECTS:
            s = search_data[project]
            best = s["best"]
            pubmed = {str(x["pmid"]) for x in json.loads(
                (REPO / "projects" / project / best / "outputs" / "search_results.json").read_text())["studies"]}
            g = set(gold[project])
            with_xyz = {r["pmid"] for r in per_study if r["project"] == project and r["neurostore_source"]}
            tiab, ft = set(s["tiab_pmids"]), set(s["fulltext_pmids"])
            searches.append({
                "project": project, "gold": len(g),
                "pubmed_records": len(pubmed), "pubmed_recall": round(len(g & pubmed) / len(g), 3),
                "pubmed_and_neurostore_coords_recall": round(len(g & pubmed & with_xyz) / len(g), 3),
                "ns_tiab_records": len(tiab), "ns_tiab_recall": round(len(g & tiab) / len(g), 3),
                "ns_fulltext_records": len(ft), "ns_fulltext_recall": round(len(g & ft) / len(g), 3),
                "dropped_clauses": len(s["dropped"]), "approximated_clauses": len(s["approximated"]),
            })

    def write(name, rows):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(OUT / name)

    write("coverage.csv", coverage)
    write("coordinates.csv", coords)
    write("gold_studies.csv", per_study)
    if searches:
        write("search.csv", searches)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["probe", "search", "report"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    {"probe": probe, "search": search, "report": report}[args.step](args)


if __name__ == "__main__":
    main()

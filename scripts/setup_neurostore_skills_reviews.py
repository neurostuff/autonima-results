"""Set up autonima-skills reviews with the NeuroStore baseline's inputs, for a like-for-like comparison.

    python scripts/setup_neurostore_skills_reviews.py WORKSPACE [--projects P ...] [--skills ~/repos/autonima-skills]

The baseline (reports/neurostore_arms/baseline.yaml) is each project's best config run end to end
on NeuroStore: title-and-abstract search, NeuroStore coordinates. Each review here gets:
- the criteria: the registered cbma-skills protocol, experiments/cbma-skills/reviews/<p>/review.yaml,
  which carries best's criteria verbatim;
- the search: the baseline's NeuroStore search, frozen. search.neurostore.query is the query the
  baseline sent, and frozen/neurostore_search_log.json holds its PMIDs (the arm's search records);
- the full text: what the baseline read, from disk. The arm's pubget cache (PMC JATS), the Elsevier
  XML corpus and the ACE HTML corpus;
- the coordinates: the baseline's frozen NeuroStore snapshot, frozen/neurostore_analyses.json.

WORKSPACE must be outside this repository, so judge sessions cannot reach the gold standard; nothing
from projects/*/cbma_skills_gold or neurometabench is copied. The script then installs the skills,
registers each protocol and runs the search, which needs network access to NCBI only. It judges
nothing.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ARTICLES = REPO / "articles"


def local_sources(project: str, run: str) -> list:
    """The baseline's full texts, in the order it preferred them."""
    return [
        {"type": "local", "name": "pmc_pubget", "format": "jats", "id_kind": "pmcid", "id_from": "regex",
         "regex": r"pmcid_(?P<id>\d+)", "pattern": "*/pmcid_*/article.xml",
         "path": str(REPO / "projects" / project / run / "retrieval" / "pubget_data" / "articles")},
        {"type": "local", "name": "elsevier_xml", "format": "elsevier", "id_kind": "pmid", "id_from": "parent_dir",
         "pattern": "*/article.xml", "path": str(ARTICLES / "elsevier_output")},
        {"type": "local", "name": "ace_html", "format": "html", "id_kind": "pmid", "id_from": "filename",
         "pattern": "**/*.html", "path": str(ARTICLES / "ace_outputs" / "html")},
    ]


def year(date):
    return int(str(date)[:4]) if date else None


def setup(project: str, run: str, best: str, workspace: Path) -> Path:
    arm = REPO / "projects" / project / run / "outputs"
    log = json.loads((arm / "neurostore_search_log.json").read_text())
    records = json.loads((arm / "search_results.json").read_text())
    records = records["studies"] if isinstance(records, dict) else records
    spec = yaml.safe_load((REPO / "experiments" / "cbma-skills" / "reviews" / project / "review.yaml").read_text())

    search = {k: v for k, v in spec["search"].items() if k != "query"}
    search["neurostore"] = {"query": log["neurostore_query"], "snapshot": "frozen/neurostore_search_log.json"}
    params = {"query": log["neurostore_query"], "search_full_text": bool(log["search_full_text"]),
              "data_type": log["data_type"], "year_min": year(search.get("date_from")),
              "year_max": year(search.get("date_to"))}
    if (params["year_min"], params["year_max"]) != (log["year_min"], log["year_max"]):
        sys.exit(f"{project}: review dates give {params['year_min']}-{params['year_max']}, the baseline "
                 f"searched {log['year_min']}-{log['year_max']}")
    spec["search"] = search
    spec["fulltext"] = {"sources": local_sources(project, run)}
    extraction = {k: v for k, v in (spec.get("extraction") or {}).items() if k != "records"}
    extraction["neurostore"] = {"snapshot": "frozen/neurostore_analyses.json"}
    spec["extraction"] = extraction

    review = workspace / project
    (review / "frozen").mkdir(parents=True, exist_ok=False)
    pmids = sorted({str(r["pmid"]) for r in records}, key=int)
    (review / "frozen" / "neurostore_search_log.json").write_text(json.dumps({
        "params": params, "base_url": log["base_url"], "searched_at": log["timestamp"],
        "reported_total": log["base_studies"], "base_studies": log["base_studies"], "pmids": pmids,
        "without_pmid": log["without_pmid"],
        "derived_from": {"run": f"{project}/{run}", "pubmed_query": log["pubmed_query"],
                         "note": f"NeuroStore returned {log['pmids']} PMIDs; these are the {len(pmids)} the arm "
                                 "had PubMed records for"}}, indent=1))
    shutil.copy2(arm / "neurostore_analyses.json", review / "frozen" / "neurostore_analyses.json")
    header = (f"# {project}: the registered cbma-skills protocol (experiments/cbma-skills/reviews/{project}/\n"
              f"# review.yaml, autonima {best}'s criteria verbatim) with the NeuroStore baseline's inputs, frozen\n"
              f"# from autonima-results projects/{project}/{run}: its NeuroStore search, its full texts from\n"
              f"# disk, and its NeuroStore coordinates. Written by scripts/setup_neurostore_skills_reviews.py.\n\n")
    (review / "review.yaml").write_text(header + yaml.safe_dump(spec, sort_keys=False, width=100, allow_unicode=True))
    return review


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workspace", type=Path)
    ap.add_argument("--projects", nargs="*")
    ap.add_argument("--skills", type=Path, default=Path.home() / "repos" / "autonima-skills")
    args = ap.parse_args()
    workspace = args.workspace.expanduser().resolve()
    if REPO in workspace.parents or workspace == REPO:
        sys.exit("the workspace must be outside autonima-results")
    baseline = yaml.safe_load((REPO / "reports" / "neurostore_arms" / "baseline.yaml").read_text())["projects"]
    projects = args.projects or list(baseline)
    workspace.mkdir(parents=True, exist_ok=True)
    subprocess.run(["bash", str(args.skills / "install.sh"), str(workspace)], check=True, stdout=subprocess.DEVNULL)
    scripts = args.skills / "skills"
    for project in projects:
        review = setup(project, baseline[project]["run"], baseline[project]["best"], workspace)
        reason = (f"Benchmark protocol (signed off 2026-09-28) with the NeuroStore baseline's inputs, frozen from "
                  f"autonima-results {project}/{baseline[project]['run']}")
        for cmd in (["cbma-review/scripts/ledger.py", "init"], ["cbma-review/scripts/ledger.py", "register"],
                    ["neurostore/scripts/neurostore.py", "search"], ["pubmed-search/scripts/pubmed_search.py"]):
            extra = ["--reason", reason] if cmd[-1] == "register" else []
            out = subprocess.run([sys.executable, str(scripts / cmd[0]), *cmd[1:], str(review), *extra],
                                 capture_output=True, text=True)
            if out.returncode:
                sys.exit(f"{project}: {' '.join(cmd)} failed:\n{out.stderr[-2000:]}")
        log = json.loads((review / "search" / "search_log.json").read_text())
        print(f"{project}: {log['records_written']} records ({len(log['missing_from_efetch'])} without a PubMed "
              f"record) from {log.get('neurostore_pmids')} NeuroStore candidates", flush=True)


if __name__ == "__main__":
    main()

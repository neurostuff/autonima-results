#!/usr/bin/env python3
"""Write a NeuroMetaBench gold standard in the shape cbma-skills' benchmark/compare.py reads.

    # a project of this repo (meta-analysis PMID from projects/<project>/nmb_mappings.json)
    python scripts/nmb_to_cbma_gold.py --project emotion_regulation_2022

    # any NeuroMetaBench meta-analysis, by its PMID
    python scripts/nmb_to_cbma_gold.py --meta-pmid 31872334 --out <dir>

WHY ONE ADAPTER FOR BOTH BENCHMARKS

Every project in this repo takes its gold from NeuroMetaBench: the included-study list
is neurometabench/data/included_studies.csv filtered on the meta-analysis PMID, and the
gold coordinates and analysis labels are neurometabench/data/nimads/<project>/merged/.
The manuscript's screening scores read exactly that (scripts/benchmark_pmids.py), so the
"autonima-results benchmark" and "NeuroMetaBench" are one source at two scopes: the nine
manuscript projects, which have merged coordinates, and the other NeuroMetaBench
meta-analyses, which have only an included-study list.

WHAT IT WRITES (to --out, default projects/<project>/cbma_skills_gold/)

    gold.csv            pmid,included -- one row per included study, included=1.
                        No abstract_included column: NeuroMetaBench has no abstract-stage
                        labels for these, so compare.py scores the abstract stage on
                        recall of the final includes, which is what matters there.
    gold_manifest.json  source paths with sha256, the neurometabench commit, counts, and
                        the gold studyset to pass as compare.py --gold-nimads.

The gold studyset is referenced in place rather than copied; the manifest's sha256 shows
whether it changed since the gold was written.

PMIDs are normalised as scripts/benchmark_pmids.py does (strip, drop a trailing ".0") and
de-duplicated, so the counts match the manuscript's meta_total.

NOT USED: neurometabench/data/all_studies.csv has per-candidate status columns for two
meta-analyses (dementia, social). Their meaning at the abstract stage is not documented,
so they are left out rather than guessed at.

Keep the output outside any cbma-skills review folder: the agent running a review must
not be able to read it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NMB = REPO_ROOT.parent / "neurometabench"


def normalize_pmid(value: str) -> str | None:
    pmid = (value or "").strip()
    if not pmid or pmid.lower() == "nan":
        return None
    if pmid.endswith(".0"):
        pmid = pmid[:-2]
    return pmid


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(repo: Path, files: list[Path]) -> str | None:
    """HEAD of the neurometabench clone, marked +modified if any file read here differs from it."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", *map(str, files)],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out.stdout.strip() + ("+modified" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def included_pmids(included_csv: Path, meta_pmid: str) -> list[str]:
    seen: dict[str, None] = {}
    with included_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("meta_pmid") or "").strip() != meta_pmid:
                continue
            pmid = normalize_pmid(row.get("study_pmid"))
            if pmid is None:
                continue
            if not pmid.isdigit():
                raise ValueError(f"non-numeric study_pmid {pmid!r} for meta-analysis {meta_pmid}")
            seen.setdefault(pmid)
    if not seen:
        raise ValueError(f"no included studies for meta-analysis PMID {meta_pmid} in {included_csv}")
    return list(seen)


def describe_studyset(studyset: Path, annotation: Path | None) -> dict:
    ss = json.loads(studyset.read_text())
    analyses = [a for s in ss["studies"] for a in s.get("analyses", [])]
    out = {"path": str(studyset), "sha256": sha256(studyset), "studies": len(ss["studies"]),
           "analyses": len(analyses), "points": sum(len(a.get("points", [])) for a in analyses)}
    if annotation and annotation.exists():
        ann = json.loads(annotation.read_text())
        labels = Counter(k for n in ann.get("notes", []) for k, v in n.get("note", {}).items() if v is True)
        out["annotation"] = {"path": str(annotation), "sha256": sha256(annotation),
                             "notes": len(ann.get("notes", [])), "true_labels": dict(sorted(labels.items()))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--project", help="a project under projects/ with an nmb_mappings.json")
    which.add_argument("--meta-pmid", help="a NeuroMetaBench meta-analysis PMID")
    ap.add_argument("--nmb-root", type=Path, default=DEFAULT_NMB, help=f"neurometabench clone (default {DEFAULT_NMB})")
    ap.add_argument("--out", type=Path, help="output folder (required with --meta-pmid)")
    args = ap.parse_args(argv)

    nmb = args.nmb_root.expanduser().resolve()
    included_csv = nmb / "data" / "included_studies.csv"
    if args.project:
        mapping = REPO_ROOT / "projects" / args.project / "nmb_mappings.json"
        meta_pmid = str(json.loads(mapping.read_text())["meta_pmid"]).strip()
        out = args.out or REPO_ROOT / "projects" / args.project / "cbma_skills_gold"
        merged = nmb / "data" / "nimads" / args.project / "merged"
    else:
        if not args.out:
            ap.error("--out is required with --meta-pmid")
        meta_pmid, out, merged = args.meta_pmid.strip(), args.out, None

    pmids = included_pmids(included_csv, meta_pmid)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "gold.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pmid", "included"])
        w.writerows([p, 1] for p in sorted(pmids, key=int))

    manifest = {
        "meta_pmid": meta_pmid,
        "project": args.project,
        "included_studies": len(pmids),
        "sources": {"included_studies_csv": {"path": str(included_csv), "sha256": sha256(included_csv)}},
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/nmb_to_cbma_gold.py",
    }
    studyset = merged / "nimads_studyset.json" if merged else None
    if studyset and studyset.exists():
        manifest["gold_nimads"] = describe_studyset(studyset, merged / "nimads_annotation.json")
    else:
        manifest["gold_nimads"] = None
    read = [included_csv] + ([studyset, merged / "nimads_annotation.json"] if manifest["gold_nimads"] else [])
    manifest["neurometabench_commit"] = git_head(nmb, read)
    (out / "gold_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"meta-analysis {meta_pmid}: {len(pmids)} included studies -> {out / 'gold.csv'}")
    if manifest["gold_nimads"]:
        g = manifest["gold_nimads"]
        print(f"gold coordinates: {g['studies']} studies, {g['analyses']} analyses, {g['points']} points "
              f"-> pass --gold-nimads {g['path']}")
    else:
        print("no merged gold studyset for this meta-analysis; coordinates cannot be scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())

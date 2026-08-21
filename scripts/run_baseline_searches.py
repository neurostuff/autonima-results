#!/usr/bin/env python3
"""Build and run per-sub-annotation search baselines for a project.

WHY
---
The project-level `all_studies` / `all_analyses` baseline stands in for a coarse,
Neurosynth-style meta-analysis: take a broad PubMed search, extract every coordinate you
can, pool them. It is the thing LLM screening has to beat.

That baseline is unfair to itself when a benchmark ran one broad search and then split the
hits into sub-topics. Nobody targeting only the alcohol sub-meta-analysis would search the
whole substance-use literature -- they would put "alcohol" in the query. Comparing our
alcohol map against a baseline drawn from the broad search hands the baseline a worse search
than a real competitor would use, and so overstates what our screening bought.

This script builds a fairer baseline per sub-annotation: same modality and publication-type
constraints, topic clause narrowed to that sub-topic. Each baseline is a normal autonima run
with screening skipped, so "included" == "everything the search returned that we could parse
coordinates from" -- a Neurosynth-style pool over a better-targeted search.

The study pools across arms are NOT equalised. Each arm gets whatever its own search plus
our retrieval could actually obtain, which is the real-world question: given the same effort,
whose map lands closer to the manual result. Pool sizes are reported alongside the metrics so
the asymmetry stays visible rather than being silently corrected away.

SPEC
----
projects/<project>/baselines.yaml

    meta_pmid: "36115222"
    shared:
      modality: '...'          # clause shared by every baseline
      exclude:  '...'          # applied as NOT (...)
      date_from: null
      date_to: "2019/7/31"
    baselines:
      - manual_annotation: alcohol      # must be a key in nmb_mappings.json
        topic: '...'                    # narrowed topic clause
        # query: '...'                  # optional: overrides the assembly entirely

Query assembly:  (modality) AND (topic) AND <dates>  NOT (exclude)

LAYOUT
------
Generated configs and their run outputs are kept together under a single folder so they
do not clutter the project root alongside the hand-written v*.yaml runs:

    projects/<project>/baselines.yaml            the spec (hand-written)
    projects/<project>/baselines/<key>.yaml      generated config, do not hand-edit
    projects/<project>/baselines/<key>/          that baseline's run outputs
    projects/<project>/reports/baseline_*        cross-baseline reports

autonima derives a run's output dir from the config path (config_path.with_suffix("")),
so nesting the config under baselines/ nests its outputs there automatically. The older
flat layout (<project>/baseline-<key>{,.yaml}) is still read by the evaluator.

USAGE
-----
    # inspect the assembled queries and PubMed hit counts, write nothing
    python scripts/run_baseline_searches.py --project vbm_of_substance_use --dry-run

    # generate run configs only
    python scripts/run_baseline_searches.py --project vbm_of_substance_use --generate-only

    # generate, run the pipeline, and run meta for each baseline
    python scripts/run_baseline_searches.py --project vbm_of_substance_use --run --meta
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PIXI_PYTHON = REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "python"
AUTONIMA = REPO_ROOT / ".pixi" / "envs" / "default" / "bin" / "autonima"
BASELINE_PREFIX = "baseline-"   # legacy flat layout: <project>/baseline-<key>{,.yaml}
BASELINES_DIR = "baselines"     # current layout: <project>/baselines/<key>{,.yaml}
# With screening skipped nothing is excluded, so this column is every searched study we
# could parse coordinates from. `all_studies`/`all_abstract` are not emitted in that case.
BASELINE_COLUMN = "all_analyses"


def collapse(text: str) -> str:
    """Fold a YAML block scalar into one line.

    YAML folded scalars join lines with a single space, which is what PubMed wants, but
    stray double spaces make queries harder to diff against a hand-written original.
    """
    return re.sub(r"\s+", " ", str(text or "")).strip()


def assemble_query(shared: dict[str, Any], entry: dict[str, Any]) -> str:
    """Build the PubMed query for one baseline entry."""
    override = collapse(entry.get("query") or "")
    if override:
        return override

    modality = collapse(shared.get("modality") or "")
    topic = collapse(entry.get("topic") or "")
    if not modality or not topic:
        raise SystemExit(
            f"baseline {entry.get('manual_annotation')!r}: needs either `query` or both "
            "`shared.modality` and `topic`"
        )
    query = f"(({modality}) AND ({topic}))"

    date_from = shared.get("date_from")
    date_to = shared.get("date_to")
    if date_from or date_to:
        lo = date_from or "1800"
        hi = date_to or "3000"
        query = f'{query} AND ("{lo}"[Date - Publication] : "{hi}"[Date - Publication])'

    exclude = collapse(shared.get("exclude") or "")
    if exclude:
        query = f"({query}) NOT ({exclude})"
    return query


def load_spec(project_dir: Path) -> dict[str, Any]:
    spec_path = project_dir / "baselines.yaml"
    if not spec_path.exists():
        raise SystemExit(f"no baseline spec at {spec_path}")
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    if not spec.get("baselines"):
        raise SystemExit(f"{spec_path}: no `baselines` entries")
    return spec


def validate_against_mappings(project_dir: Path, spec: dict[str, Any]) -> list[str]:
    """Check every baseline names a real manual annotation. Returns warnings."""
    warnings: list[str] = []
    mp = project_dir / "nmb_mappings.json"
    if not mp.exists():
        return [f"no nmb_mappings.json in {project_dir}; cannot validate annotation keys"]
    mapping = json.loads(mp.read_text(encoding="utf-8"))
    known = set((mapping.get("annotation_mappings") or {}).keys())
    spec_meta = str(spec.get("meta_pmid") or "").strip()
    map_meta = str(mapping.get("meta_pmid") or "").strip()
    if spec_meta and map_meta and spec_meta != map_meta:
        warnings.append(f"meta_pmid mismatch: baselines.yaml={spec_meta} nmb_mappings={map_meta}")
    for entry in spec["baselines"]:
        key = entry.get("manual_annotation")
        if not key:
            raise SystemExit("every baseline entry needs `manual_annotation`")
        if key not in known:
            warnings.append(
                f"baseline {key!r} is not a key in nmb_mappings.json "
                f"(known: {sorted(known)}) -- it will have no manual map to compare against"
            )
    covered = {e["manual_annotation"] for e in spec["baselines"]}
    for missing in sorted(known - covered):
        warnings.append(f"manual annotation {missing!r} has no baseline entry")
    return warnings


def gold_by_annotation(project: str, manual_nimads_base: Path) -> dict[str, set[str]]:
    """Gold PMIDs per manual annotation column, from the benchmark's NiMADS.

    Study ids may be comma-joined lists of PMIDs (merged independent samples) and may also
    contain author-year fragments, so keep only well-formed PMIDs. Passing a malformed token
    into an `[uid]` clause silently breaks the whole query rather than erroring.
    """
    merged = manual_nimads_base / project / "merged"
    ss_path, ann_path = merged / "nimads_studyset.json", merged / "nimads_annotation.json"
    if not ss_path.exists() or not ann_path.exists():
        return {}
    studyset = json.loads(ss_path.read_text(encoding="utf-8"))
    studies = studyset.get("studies", studyset)
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    notes = ann.get("notes", ann) if isinstance(ann, dict) else ann

    analysis_to_study: dict[str, str] = {}
    for study in studies:
        for analysis in study.get("analyses") or []:
            analysis_to_study[str(analysis.get("id"))] = str(study.get("id"))

    pmid_re = re.compile(r"^\d{7,8}$")
    out: dict[str, set[str]] = {}
    for note in notes:
        payload = note.get("note") or {}
        study_id = analysis_to_study.get(str(note.get("analysis")))
        if not study_id:
            continue
        pmids = {t.strip() for t in study_id.split(",") if pmid_re.match(t.strip())}
        for column, flag in payload.items():
            if flag:
                out.setdefault(column, set()).update(pmids)
    return out


def gold_found(query: str, pmids: set[str], email: str | None) -> set[str] | None:
    """Which of `pmids` a query returns."""
    if not pmids:
        return set()
    try:
        from Bio import Entrez
    except Exception:
        return None
    if email:
        Entrez.email = email
    term = "(" + " OR ".join(f"{p}[uid]" for p in sorted(pmids)) + f") AND ({query})"
    for attempt in range(3):
        try:
            handle = Entrez.esearch(db="pubmed", term=term, retmax=5000)
            result = Entrez.read(handle)
            handle.close()
            time.sleep(0.4)
            return set(result["IdList"])
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def pubmed_count(query: str, email: str | None) -> int | None:
    """Total PubMed hits for a query, or None if the lookup fails."""
    try:
        from Bio import Entrez
    except Exception:
        return None
    if email:
        Entrez.email = email
    for attempt in range(3):
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
            result = Entrez.read(handle)
            handle.close()
            time.sleep(0.4)
            return int(result["Count"])
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def build_run_config(
    *, project_dir: Path, template_run: dict[str, Any], query: str, spec: dict[str, Any]
) -> dict[str, Any]:
    """An autonima config for one baseline: search + retrieve + parse, no screening.

    Retrieval and parsing settings are inherited from a template run in the same project so
    the baseline reads the same local full-text sources as the pipeline it is compared with.
    Screening is skipped, which is the whole point: the baseline applies no eligibility
    judgement at all.
    """
    cfg: dict[str, Any] = {}
    search = dict(template_run.get("search") or {})
    search.pop("pmids_file", None)
    search["query"] = query
    for key in ("date_from", "date_to"):
        if spec.get("shared", {}).get(key):
            search[key] = spec["shared"][key]
    cfg["search"] = search

    retrieval = dict(template_run.get("retrieval") or {})
    # Nothing is excluded when screening is skipped, so this flag is inert; pin it off so a
    # reader does not assume the baseline is pulling an excluded arm.
    retrieval["load_excluded"] = False
    cfg["retrieval"] = retrieval

    cfg["screening"] = {"abstract": {"skip_stage": True}, "fulltext": {"skip_stage": True}}
    cfg["parsing"] = dict(template_run.get("parsing") or {"parse_coordinates": True})
    # Only the all_* columns; no criteria, because a baseline makes no judgement.
    cfg["annotation"] = {
        "enabled": True,
        "model": (template_run.get("annotation") or {}).get("model", "gpt-5-mini-2025-08-07"),
        "create_all_included_annotations": True,
        "annotations": [],
    }
    cfg["output"] = {"formats": ["csv"], "nimads": True}
    return cfg


def pick_template_run(project_dir: Path, explicit: str | None) -> tuple[str, dict[str, Any]]:
    """Choose the config whose retrieval/parsing settings the baselines inherit."""
    if explicit:
        path = project_dir / explicit
        if not path.exists():
            raise SystemExit(f"template config not found: {path}")
        return path.name, yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    candidates = sorted(
        p for p in project_dir.glob("v*.yaml") if not p.name.startswith(BASELINE_PREFIX)
    )
    if not candidates:
        raise SystemExit(f"no v*.yaml in {project_dir} to inherit retrieval settings from")
    # Prefer the highest plain vN.yaml, which is the project's main search-based run.
    plain = [p for p in candidates if re.fullmatch(r"v\d+\.yaml", p.name)]
    chosen = (plain or candidates)[-1]
    return chosen.name, yaml.safe_load(chosen.read_text(encoding="utf-8")) or {}


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\nexit={proc.returncode}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n",
        encoding="utf-8",
    )
    return proc.returncode


def summarise_run(run_dir: Path) -> dict[str, Any]:
    """Pool size and coordinate mass for a finished baseline run."""
    out: dict[str, Any] = {"studies": None, "with_coords": None, "points": None, "missing_fulltexts": None}
    ss = run_dir / "outputs" / "nimads_studyset.json"
    if ss.exists():
        payload = json.loads(ss.read_text(encoding="utf-8"))
        studies = payload.get("studies", payload)
        out["studies"] = len(studies)
        n_with = 0
        n_pts = 0
        for study in studies:
            pts = sum(len(a.get("points") or []) for a in (study.get("analyses") or []))
            n_pts += pts
            if pts:
                n_with += 1
        out["with_coords"] = n_with
        out["points"] = n_pts
    missing = run_dir / "outputs" / "missing_fulltexts.txt"
    if missing.exists():
        out["missing_fulltexts"] = len([x for x in missing.read_text().split() if x.strip()])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--project", required=True, help="project name under projects/")
    ap.add_argument("--only", action="append", default=[], help="limit to these manual_annotation keys (repeatable)")
    ap.add_argument("--template-config", default=None, help="config to inherit retrieval/parsing from (default: highest vN.yaml)")
    ap.add_argument("--dry-run", action="store_true", help="print assembled queries and PubMed counts; write nothing")
    ap.add_argument("--generate-only", action="store_true", help="write run configs, do not execute")
    ap.add_argument("--run", action="store_true", help="execute the pipeline for each baseline")
    ap.add_argument("--meta", action="store_true", help="run autonima meta after each pipeline run")
    ap.add_argument("--workers", type=int, default=25, help="-j for autonima run")
    ap.add_argument("--no-counts", action="store_true", help="skip PubMed hit-count lookups")
    ap.add_argument("--check-recall", action="store_true",
                    help="report each baseline's gold recall, and the ceiling imposed by the "
                         "shared modality clause, against the benchmark's annotation columns")
    ap.add_argument("--manual-nimads-base", default="/home/zorro/repos/neurometabench/data/nimads",
                    help="benchmark NiMADS base dir (for --check-recall)")
    args = ap.parse_args()

    project_dir = REPO_ROOT / "projects" / args.project
    if not project_dir.is_dir():
        raise SystemExit(f"no such project: {project_dir}")

    spec = load_spec(project_dir)
    for warning in validate_against_mappings(project_dir, spec):
        print(f"WARNING: {warning}", file=sys.stderr)

    template_name, template_run = pick_template_run(project_dir, args.template_config)
    print(f"project        : {args.project}")
    print(f"template config: {template_name} (retrieval/parsing inherited)")
    print(f"baseline column: {BASELINE_COLUMN}\n")

    entries = spec["baselines"]
    if args.only:
        entries = [e for e in entries if e["manual_annotation"] in set(args.only)]
        if not entries:
            raise SystemExit(f"--only matched nothing; available: {[e['manual_annotation'] for e in spec['baselines']]}")

    email = (template_run.get("search") or {}).get("email")
    gold: dict[str, set[str]] = {}
    if args.check_recall:
        gold = gold_by_annotation(args.project, Path(args.manual_nimads_base))
        if not gold:
            print("WARNING: --check-recall found no benchmark NiMADS for this project", file=sys.stderr)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        key = entry["manual_annotation"]
        row_recall: dict[str, Any] = {}
        query = assemble_query(spec.get("shared") or {}, entry)
        hits = None if (args.no_counts or args.dry_run is False and not args.dry_run) else None
        if not args.no_counts:
            hits = pubmed_count(query, email)
        print(f"[{key}]")
        print(f"  query chars : {len(query)}")
        if hits is not None:
            print(f"  pubmed hits : {hits}")
        if entry.get("query"):
            print("  NOTE        : hand-written `query` override in use")
        if args.check_recall:
            g = gold.get(key) or set()
            if not g:
                print(f"  gold recall : no gold column {key!r} in the benchmark")
            else:
                narrow = gold_found(query, g, email)
                ceiling = gold_found(f"({collapse((spec.get('shared') or {}).get('modality') or '')})", g, email)
                if narrow is None or ceiling is None:
                    print("  gold recall : lookup failed")
                else:
                    # Gold the topic clause cannot reach even though the modality clause can:
                    # typically multi-substance papers indexed under a different sub-topic.
                    unreachable = sorted(ceiling - narrow)
                    print(f"  gold recall : {len(narrow)}/{len(g)} ({len(narrow)/len(g):.0%})"
                          f"   modality ceiling {len(ceiling)}/{len(g)} ({len(ceiling)/len(g):.0%})")
                    if unreachable:
                        print(f"                {len(unreachable)} pass modality but not this topic: {unreachable[:6]}")
                    row_recall.update({"gold_total": len(g), "gold_found": len(narrow),
                                       "gold_recall": round(len(narrow)/len(g), 4),
                                       "modality_ceiling": len(ceiling),
                                       "topic_unreachable": unreachable})
        if args.dry_run:
            print(f"  query       : {query}\n")
            rows.append({"manual_annotation": key, "query": query, "pubmed_hits": hits, **row_recall})
            continue

        baselines_dir = project_dir / BASELINES_DIR
        baselines_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = baselines_dir / f"{key}.yaml"
        cfg = build_run_config(
            project_dir=project_dir, template_run=template_run, query=query, spec=spec
        )
        header = (
            f"# GENERATED by scripts/run_baseline_searches.py from baselines.yaml -- do not\n"
            f"# hand-edit; edit the spec instead. Search baseline for manual annotation\n"
            f"# {key!r}. Screening is skipped, so this is a Neurosynth-style pool over a\n"
            f"# topic-targeted search: every hit whose coordinates we could parse. The\n"
            f"# comparison column is '{BASELINE_COLUMN}'.\n"
        )
        if entry.get("note"):
            header += "#\n# " + collapse(entry["note"]) + "\n"
        cfg_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, width=100), encoding="utf-8")
        print(f"  wrote config: {cfg_path.relative_to(REPO_ROOT)}")

        # autonima derives the run dir from the config path (config_path.with_suffix("")),
        # so nesting the config under baselines/ nests the outputs there too.
        run_dir = baselines_dir / key
        row: dict[str, Any] = {"manual_annotation": key, "query": query, "pubmed_hits": hits,
                               "config": cfg_path.relative_to(REPO_ROOT).as_posix(), **row_recall}
        if args.run:
            rc = run_cmd(
                [str(AUTONIMA), "run", str(cfg_path.relative_to(REPO_ROOT)), "-j", str(args.workers)],
                run_dir / "logs" / "run.log",
            )
            print(f"  run rc      : {rc}")
            row["run_rc"] = rc
            if rc == 0 and args.meta:
                rc_meta = run_cmd(
                    [str(AUTONIMA), "meta", str(run_dir.relative_to(REPO_ROOT))],
                    run_dir / "logs" / "meta.log",
                )
                print(f"  meta rc     : {rc_meta}")
                row["meta_rc"] = rc_meta
            row.update(summarise_run(run_dir))
            print(f"  pool        : studies={row.get('studies')} with_coords={row.get('with_coords')} points={row.get('points')}")
        rows.append(row)
        print()

    if not args.dry_run:
        report_dir = project_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / "baseline_searches.json"
        out.write_text(json.dumps({"project": args.project, "template_config": template_name,
                                   "baseline_column": BASELINE_COLUMN, "baselines": rows},
                                  indent=2), encoding="utf-8")
        print(f"wrote {out.relative_to(REPO_ROOT)}")
        # Roll up per-baseline missing full texts so manual downloading can be targeted.
        missing: dict[str, list[str]] = {}
        for entry in entries:
            key = entry["manual_annotation"]
            path = project_dir / BASELINES_DIR / key / "outputs" / "missing_fulltexts.txt"
            if not path.exists():   # legacy flat layout
                path = project_dir / f"{BASELINE_PREFIX}{key}" / "outputs" / "missing_fulltexts.txt"
            if path.exists():
                missing[key] = sorted({x.strip() for x in path.read_text().split() if x.strip()})
        if missing:
            union = sorted(set().union(*missing.values()))
            roll = report_dir / "baseline_missing_fulltexts.json"
            roll.write_text(json.dumps({"by_baseline": {k: len(v) for k, v in missing.items()},
                                        "union_count": len(union), "union": union}, indent=2),
                            encoding="utf-8")
            (report_dir / "baseline_missing_fulltexts.txt").write_text("\n".join(union) + "\n", encoding="utf-8")
            print(f"wrote {roll.relative_to(REPO_ROOT)} (union of {len(union)} PMIDs to chase)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

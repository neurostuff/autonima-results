"""Run a pondie-record document arm against a project's best autonima run.

  python scripts/pondie_arms.py <project> <text|records> [--live] [--workers N]
  python scripts/pondie_arms.py <project> <text|records> --write-config

The arm's config is the project's `best` config (run_categories.yaml) with only what documents
require changed, and the script refuses to run if anything else differs:

  documents:                      added, pointing at articles/pondie/md/<project>/<kind>
  parsing.parse_coordinates       false for `records` (the records carry the analyses)
  annotation.metadata_fields      + study_fulltext when best lacks it (autonima refuses a
                                  document run whose annotation never reads the document --
                                  this makes annotation differ from best, and is reported)
                                  + analysis_document for `records`
  output.directory                projects/<project>/<best>-pondie-<kind>, or
                                  <best>-ft-pondie-<kind> when study_fulltext was added

The arm reuses best's cache through --copy-valid-cache-from, and PubMed esearch returns best's
own PMIDs, so the cohort, search metadata and abstract decisions are best's. Without --live
the LLM is stubbed (full-text decisions replay best's, annotation returns placeholders, and any
abstract or coordinate-parsing call raises), which checks the whole run for free. --live
needs OPENAI_API_KEY and OPENAI_API_GATEWAY in the environment.

Documents come from scripts/compile_pondie_records.py. The `text` kind re-runs article
retrieval, which needs a full ACE import (ACE, xmltodict, seleniumbase) on PYTHONPATH.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

REPO = Path(__file__).resolve().parents[1]
DESCRIPTION = "a structured extraction record of the article produced by pondie"


def best_run(project: str) -> str:
    categories = yaml.safe_load((REPO / "run_categories.yaml").read_text())
    return categories["projects"][project]["canonical"]["best"]


def arm_config(project: str, kind: str):
    best = best_run(project)
    base = yaml.safe_load((REPO / "projects" / project / f"{best}.yaml").read_text())
    arm = deepcopy(base)
    notes = []
    for source in arm.get("retrieval", {}).get("full_text_sources", []) or []:
        for key in ("root_path", "processed_data_path"):
            if source.get(key) and not Path(source[key]).is_absolute():
                source[key] = str(REPO / source[key])
    arm["documents"] = {
        "enabled": True,
        "kind": kind,
        "root": str(REPO / "articles" / "pondie" / "md" / project / kind),
        "description": DESCRIPTION,
    }
    if kind == "records":
        arm.setdefault("parsing", {})["parse_coordinates"] = False
    fields = list(arm.setdefault("annotation", {}).get("metadata_fields") or [])
    name = f"{best}-pondie-{kind}"
    if "study_fulltext" not in fields:
        fields.append("study_fulltext")
        name = f"{best}-ft-pondie-{kind}"  # "-ft": annotation reads the document, best did not
        notes.append("annotation.metadata_fields gains study_fulltext: best annotates without "
                     "the article, so annotation is not like-for-like with best")
    if kind == "records":
        fields.append("analysis_document")
    arm["annotation"]["metadata_fields"] = fields
    arm.setdefault("output", {})["directory"] = str(REPO / "projects" / project / name)
    return best, base, arm, notes


def assert_equivalent(base: dict, arm: dict) -> None:
    """Fail unless the arm differs from best only in the keys documents require."""
    allowed = {
        ("documents",),
        ("parsing", "parse_coordinates"),
        ("annotation", "metadata_fields"),
        ("output", "directory"),
    }

    def flatten(d, prefix=()):
        for key, value in d.items():
            path = prefix + (key,)
            if isinstance(value, dict) and path != ("documents",):
                yield from flatten(value, path)
            else:
                yield path, value

    b, a = dict(flatten(base)), dict(flatten(arm))
    changed = {p for p in set(a) | set(b) if a.get(p) != b.get(p)}
    # Repo-relative paths made absolute are the same path.
    changed = {
        p for p in changed
        if not (p[:2] == ("retrieval", "full_text_sources")
                and _same_sources(b.get(p), a.get(p)))
    }
    unexpected = sorted(changed - allowed)
    if unexpected:
        sys.exit(f"arm config differs from best outside the allowed keys: {unexpected}")


def _same_sources(base_sources, arm_sources):
    if not isinstance(base_sources, list) or not isinstance(arm_sources, list):
        return False
    resolved = []
    for source in deepcopy(base_sources):
        for key in ("root_path", "processed_data_path"):
            if source.get(key) and not Path(source[key]).is_absolute():
                source[key] = str(REPO / source[key])
        resolved.append(source)
    return resolved == arm_sources


def write_run_config(project: str, kind: str) -> Path:
    """Write the arm's config beside the project's others, as projects/<project>/<run>.yaml.

    Paths stay repo-relative and output.directory is left out, as in every other run config:
    the CLI names the output folder after the file.
    """
    best, base, config, notes = arm_config(project, kind)
    assert_equivalent(base, config)
    run = Path(config["output"].pop("directory")).name
    if not config["output"]:
        del config["output"]
    config["retrieval"]["full_text_sources"] = deepcopy(
        base.get("retrieval", {}).get("full_text_sources")
    )
    if config["retrieval"]["full_text_sources"] is None:
        del config["retrieval"]["full_text_sources"]
    config["documents"]["root"] = str(Path(config["documents"]["root"]).relative_to(REPO))
    changes = [
        "documents: added (kind %s, root %s)" % (kind, config["documents"]["root"]),
    ]
    if kind == "records":
        changes.append("parsing.parse_coordinates: false (the records carry the analyses)")
    if notes:
        changes.append("annotation.metadata_fields: + study_fulltext (best annotates without the "
                       "article, so annotation is not like-for-like with best; hence -ft)")
    if kind == "records":
        changes.append("annotation.metadata_fields: + analysis_document")
    header = "\n".join([
        f"# {run} -- pondie-record document arm of {best}, the best {project} run.",
        "#",
        f"# Generated by scripts/pondie_arms.py from {best}.yaml; do not edit by hand. The only",
        "# differences from it:",
        *[f"#   - {c}" for c in changes],
        "#",
        "# Reproduce (reuses best's cache and pins the cohort to best's PMIDs; running this file",
        "# with `autonima run` alone would re-query PubMed and reuse nothing):",
        f"#   python scripts/pondie_arms.py {project} {kind} --live",
        "",
    ])
    path = REPO / "projects" / project / f"{run}.yaml"
    path.write_text(header + yaml.safe_dump(config, sort_keys=False, width=100))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project")
    ap.add_argument("kind", choices=["text", "records"])
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--write-config", action="store_true",
                    help="write projects/<project>/<run>.yaml and exit")
    args = ap.parse_args()
    if args.write_config:
        print(write_run_config(args.project, args.kind))
        return

    best, base, config, notes = arm_config(args.project, args.kind)
    assert_equivalent(base, config)
    run_dir = Path(config["output"]["directory"])
    if not args.live:
        run_dir = run_dir.with_name(run_dir.name + "-stub")
        config["output"]["directory"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "arm_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for note in notes:
        print(f"NOTE: {note}")
    if args.live and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("--live needs OPENAI_API_KEY and OPENAI_API_GATEWAY in the environment")

    from autonima.annotation.prompts import create_study_multi_annotation_prompt
    from autonima.annotation.schema import AnnotationDecision
    from autonima.config import ConfigManager
    from autonima.pipeline import AutonimaPipeline
    from autonima.screening.schema import FullTextScreeningOutput

    best_out = REPO / "projects" / args.project / best / "outputs"
    search = json.loads((best_out / "search_results.json").read_text())["studies"]
    title_to_pmid = {s["title"]: s["pmid"] for s in search}
    replay = {
        r["study_id"]: r["decision"]
        for r in json.loads((best_out / "fulltext_screening_results.json").read_text())["screening_results"]
    }
    log = {"fulltext": 0, "annotation": 0, "abstract": 0, "parsing": 0}

    async def best_pmids(self, query):
        return [s["pmid"] for s in search]

    class StubScreening:
        def screen_abstract(self, prompt, model, **kwargs):
            log["abstract"] += 1
            raise AssertionError("abstract screening should come from best's cache")

        def screen_fulltext(self, prompt, model, **kwargs):
            log["fulltext"] += 1
            pmid = title_to_pmid.get(re.search(r"^Title: (.*)$", prompt, re.M).group(1))
            decision = replay.get(pmid, "included_fulltext")
            return FullTextScreeningOutput(
                decision="INCLUDED" if decision == "included_fulltext" else "EXCLUDED",
                confidence=0.9, reason=f"stub replay of {decision}",
                fulltext_incomplete=decision == "fulltext_incomplete",
            )

    class StubAnnotation:
        def __init__(self, *a, **k):
            pass

        def make_decision(self, group, criteria, fields, model="", model_params=None, prompt_type=""):
            log["annotation"] += 1
            create_study_multi_annotation_prompt(group, criteria, fields)
            return [AnnotationDecision(annotation_name=c.name, analysis_id=a.analysis_id,
                                       study_id=group.study_id, include=False,
                                       reasoning="stub", model_used=model)
                    for a in group.analyses for c in criteria]

    class StubParsing:
        def __init__(self, *a, **k):
            pass

        def parse_analyses(self, *a, **k):
            log["parsing"] += 1
            raise AssertionError("coordinate parsing should come from best's cache")

    with ExitStack() as stack:
        stack.enter_context(patch("autonima.search.pubmed.PubMedSearch._execute_search", best_pmids))
        if not args.live:
            stack.enter_context(patch("autonima.screening.screener.GenericLLMClient", StubScreening))
            stack.enter_context(patch("autonima.annotation.processor.AnnotationClient", StubAnnotation))
            stack.enter_context(patch("autonima.coordinates.processor.CoordinateParsingClient", StubParsing))
        pipeline = AutonimaPipeline(
            ConfigManager().load_from_dict(config),
            num_workers=args.workers,
            copy_valid_cache_from=str(REPO / "projects" / args.project / best),
        )
        result = asyncio.run(pipeline.run())

    print(json.dumps({
        "project": args.project, "best": best, "kind": args.kind, "run": str(run_dir),
        "live": args.live, "stub_calls": None if args.live else log,
        "retrieval": result.execution_stats.get("retrieval"),
        "prisma": result.execution_stats.get("prisma_stats"),
        "notes": notes,
    }, indent=1))


if __name__ == "__main__":
    main()

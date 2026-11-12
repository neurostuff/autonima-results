"""NeuroStore feature arms: judge NeuroStore's extracted values instead of the full text.

  python scripts/neurostore_feature_arms.py <project> <arm> --write-config
  python scripts/neurostore_feature_arms.py <project> <arm> [--live] [--workers N]
  python scripts/neurostore_feature_arms.py --check            # load every arm config

Needs autonima with NeuroStore records (branch feat/neurostore-features, venv in
~/repos/autonima-nsfeat). Every arm is the project's NeuroStore search arm, <best>-ns-search
(reports/neurostore_arms/baseline.yaml), with only these keys changed, and the script refuses
to run if anything else differs:

  rec             retrieval.records.kind: neurostore. Full-text screening and annotation read
                  each study's NeuroStore record (each analysis with its extracted features)
                  instead of the article; the analyses come with the records, so
                  parsing.parse_coordinates is false. Separate full-text and annotation calls.
  rec-comb        rec, with screening.fulltext.mode: combined. One call per study screens it
                  and selects its analyses.
  rec-comb-noabs  rec-comb with abstract screening skipped: every candidate gets the combined
                  call.
  rec-global      rec with full-text screening skipped and its criteria added to the annotation
                  stage's global criteria: annotation alone decides, per analysis.

All arms also add study_fulltext to annotation.metadata_fields where the baseline lacks it
(the records reach annotation only through it).

The cohort is pinned to the baseline arm's search results, and the first run of an arm seeds
its cache from the baseline arm, so search and abstract decisions are the baseline's (the
noabs arm has none). Without --live the LLM is stubbed: full-text decisions replay the
baseline's, analysis decisions are placeholders, so a run checks the whole pipeline for free and
counts the calls a live run would make. --live needs OPENAI_API_KEY and OPENAI_API_GATEWAY.
Both modes read NeuroStore's analysis query, which must be deployed.
"""

import argparse
import asyncio
import json
import os
import sys
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

REPO = Path(__file__).resolve().parents[1]

ARMS = {
    "rec": "retrieval.records.kind: neurostore (NeuroStore's extracted values instead of the full text)",
    "rec-comb": "screening.fulltext.mode: combined (one call screens each study and selects its analyses)",
    "rec-comb-noabs": "screening.abstract.skip_stage: true (every candidate gets the combined call)",
    "rec-global": "screening.fulltext.skip_stage: true, its criteria added to annotation's global criteria",
}


def baseline_run(project: str) -> str:
    return yaml.safe_load((REPO / "reports" / "neurostore_arms" / "baseline.yaml").read_text())["projects"][project]["run"]


def criterion_text(criterion) -> str:
    """A criterion as text. An unquoted "Label: text" item in YAML reads as a one-key mapping
    (cue_reactivity v6's third full-text criterion): screening accepts it and shows the model
    its dict form, while annotation requires strings, so it becomes "Label: text"."""
    if isinstance(criterion, dict):
        return "; ".join(f"{k}: {v}" for k, v in criterion.items())
    return str(criterion)


def arm_config(project: str, arm: str):
    """The baseline's config as it ran (its frozen arm_config.yaml), and the arm's.

    Read from the baseline run, not regenerated from best: the baseline does not follow later
    changes to best in run_categories.yaml.
    """
    entry = yaml.safe_load((REPO / "reports" / "neurostore_arms" / "baseline.yaml").read_text())["projects"][project]
    best = entry["best"]
    base = yaml.safe_load((REPO / "projects" / project / entry["run"] / "arm_config.yaml").read_text())
    if Path(base["output"]["directory"]).name != entry["run"]:
        sys.exit(f"{project}: {entry['run']}/arm_config.yaml writes to {base['output']['directory']}")
    config = deepcopy(base)
    config["parsing"] = {"parse_coordinates": False}
    config["retrieval"]["records"] = {"enabled": True, "kind": "neurostore"}
    annotation = config["annotation"]
    fields = annotation.setdefault("metadata_fields", [])
    if "study_fulltext" not in fields:
        fields.append("study_fulltext")
    if arm in ("rec-comb", "rec-comb-noabs"):
        config["screening"]["fulltext"]["mode"] = "combined"
    if arm == "rec-comb-noabs":
        config["screening"]["abstract"] = {"skip_stage": True}
    if arm == "rec-global":
        fulltext = config["screening"]["fulltext"]
        annotation["inclusion_criteria"] = list(annotation.get("inclusion_criteria") or []) + [
            criterion_text(c) for c in fulltext.get("inclusion_criteria") or []]
        annotation["exclusion_criteria"] = list(annotation.get("exclusion_criteria") or []) + [
            criterion_text(c) for c in fulltext.get("exclusion_criteria") or []]
        config["screening"]["fulltext"] = {"skip_stage": True}
    config["output"]["directory"] = str(REPO / "projects" / project / f"{best}-ns-{arm}")
    return best, base, config


def _flatten(d, prefix=()):
    for key, value in d.items():
        path = prefix + (key,)
        if isinstance(value, dict) and value:
            yield from _flatten(value, path)
        else:
            yield path, value


def assert_equivalent(base: dict, arm_cfg: dict, arm: str) -> None:
    """Fail unless the arm differs from the baseline only where the arm says it does."""
    allowed_prefixes = [("parsing",), ("retrieval", "records"), ("annotation", "metadata_fields"),
                        ("output", "directory")]
    if arm in ("rec-comb", "rec-comb-noabs"):
        allowed_prefixes.append(("screening", "fulltext", "mode"))
    if arm == "rec-comb-noabs":
        allowed_prefixes.append(("screening", "abstract"))
    if arm == "rec-global":
        allowed_prefixes += [("screening", "fulltext"), ("annotation", "inclusion_criteria"),
                             ("annotation", "exclusion_criteria")]
    b, a = dict(_flatten(base)), dict(_flatten(arm_cfg))
    changed = {p for p in set(a) | set(b) if a.get(p) != b.get(p)}
    unexpected = sorted(p for p in changed if not any(p[:len(q)] == q for q in allowed_prefixes))
    if unexpected:
        sys.exit(f"{arm} config differs from the baseline outside its keys: {unexpected}")


def write_run_config(project: str, arm: str) -> Path:
    best, base, config = arm_config(project, arm)
    assert_equivalent(base, config, arm)
    run = Path(config["output"].pop("directory")).name
    if not config["output"]:
        del config["output"]
    # The baseline's article sources, relative to the repository as configs keep them; a
    # records run never reads them.
    for source in config["retrieval"].get("full_text_sources") or []:
        for key in ("root_path", "processed_data_path"):
            if source.get(key) and Path(source[key]).is_relative_to(REPO):
                source[key] = str(Path(source[key]).relative_to(REPO))
    changes = [ARMS["rec"], "parsing.parse_coordinates: false (the analyses come with the records)"]
    if arm != "rec":
        changes.append(ARMS[arm] if arm != "rec-comb-noabs" else ARMS["rec-comb"])
    if arm == "rec-comb-noabs":
        changes.append(ARMS["rec-comb-noabs"])
    if arm == "rec-global" and any(
        not isinstance(c, str)
        for key in ("inclusion_criteria", "exclusion_criteria")
        for c in base["screening"]["fulltext"].get(key) or []
    ):
        changes.append("a full-text criterion YAML read as a mapping is written as 'label: text'")
    if "study_fulltext" not in (base.get("annotation", {}).get("metadata_fields") or []):
        changes.append("annotation.metadata_fields: + study_fulltext (records reach annotation through it)")
    baseline = baseline_run(project)
    header = "\n".join([
        f"# {run} -- NeuroStore feature arm '{arm}' of {baseline}, the {project} NeuroStore search arm.",
        "#",
        f"# Generated by scripts/neurostore_feature_arms.py from {baseline}; do not edit by hand.",
        "# The only differences from it:",
        *[f"#   - {c}" for c in changes],
        "#",
        f"# Reproduce (pins the cohort to {baseline}'s search and seeds from its cache):",
        f"#   python scripts/neurostore_feature_arms.py {project} {arm} --live",
        "",
    ])
    path = REPO / "projects" / project / f"{run}.yaml"
    path.write_text(header + yaml.safe_dump(config, sort_keys=False, width=100))
    return path


def check_all() -> int:
    """Load every arm config with autonima's validation; report the first problem per config."""
    from autonima.config import ConfigManager

    failures = 0
    for project in yaml.safe_load((REPO / "reports" / "neurostore_arms" / "baseline.yaml").read_text())["projects"]:
        for arm in ARMS:
            _, base, config = arm_config(project, arm)
            assert_equivalent(base, config, arm)
            try:
                ConfigManager().load_from_dict(deepcopy(config))
                print(f"ok   {project} {arm}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"FAIL {project} {arm}: {exc}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", nargs="?")
    ap.add_argument("arm", nargs="?", choices=ARMS)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--write-config", action="store_true", help="write projects/<project>/<run>.yaml and exit")
    ap.add_argument("--check", action="store_true", help="load every arm config and exit")
    args = ap.parse_args()
    if args.check:
        sys.exit(1 if check_all() else 0)
    if not (args.project and args.arm):
        ap.error("project and arm are required")
    if args.write_config:
        print(write_run_config(args.project, args.arm))
        return

    best, base, config = arm_config(args.project, args.arm)
    assert_equivalent(base, config, args.arm)
    run_dir = Path(config["output"]["directory"])
    if not args.live:
        run_dir = run_dir.with_name(run_dir.name + "-stub")
        config["output"]["directory"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "arm_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if args.live and not (os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API_GATEWAY")):
        sys.exit("--live needs OPENAI_API_KEY and OPENAI_API_GATEWAY in the environment")

    from autonima.annotation.schema import AnnotationDecision
    from autonima.config import ConfigManager
    from autonima.pipeline import AutonimaPipeline
    from autonima.screening.schema import AbstractScreeningOutput, FullTextScreeningOutput

    baseline_dir = REPO / "projects" / args.project / baseline_run(args.project)
    baseline_out = baseline_dir / "outputs"
    cohort = [s["pmid"] for s in json.loads((baseline_out / "search_results.json").read_text())["studies"]]
    replay = {
        r["study_id"]: r["decision"]
        for r in json.loads((baseline_out / "fulltext_screening_results.json").read_text())["screening_results"]
    }
    log = {"abstract": 0, "fulltext": 0, "combined": 0, "annotation": 0}

    async def pinned_search(self, query):
        return list(cohort)

    def replayed(pmid):
        decision = replay.get(str(pmid), "excluded_fulltext")
        return ("INCLUDED" if decision == "included_fulltext" else "EXCLUDED"), decision

    class StubScreening:
        def screen_abstract(self, prompt, model, **kwargs):
            log["abstract"] += 1
            return AbstractScreeningOutput(decision="INCLUDED", confidence=0.9, reason="stub")

        def screen_fulltext(self, prompt, model, **kwargs):
            log["fulltext"] += 1
            return FullTextScreeningOutput(decision="EXCLUDED", confidence=0.9, reason="stub")

    class StubAnnotation:
        def __init__(self, *a, **k):
            pass

        def make_decision(self, group, criteria, fields, model="", model_params=None, prompt_type=""):
            log["annotation"] += 1
            return [AnnotationDecision(annotation_name=c.name, analysis_id=a.analysis_id, study_id=group.study_id,
                                       include=False, reasoning="stub", model_used=model)
                    for a in group.analyses for c in criteria]

        def make_combined_decision(self, group, criteria, fields, block, model, model_params=None, **kwargs):
            log["combined"] += 1
            decision, replayed_as = replayed(group.study_id)
            screening = {"decision": decision, "confidence": 0.9, "reason": f"stub replay of {replayed_as}",
                         "fulltext_incomplete": False, "inclusion_criteria_applied": [],
                         "exclusion_criteria_applied": []}
            decisions = [
                AnnotationDecision(annotation_name=c.name, analysis_id=a.analysis_id, study_id=group.study_id,
                                   include=False, reasoning="stub", model_used=model)
                for a in group.analyses for c in criteria
            ] if decision == "INCLUDED" else []
            return screening, decisions

    with ExitStack() as stack:
        stack.enter_context(patch("autonima.search.neurostore.NeuroStoreSearch._execute_search", pinned_search))
        if not args.live:
            stack.enter_context(patch("autonima.screening.screener.GenericLLMClient", StubScreening))
            stack.enter_context(patch("autonima.annotation.processor.AnnotationClient", StubAnnotation))
            stack.enter_context(patch("autonima.screening.combined.AnnotationClient", StubAnnotation))
        # Seed only an arm's first run: a later pass would have its own stage files replaced
        # by the baseline's, and re-decide every study (found in neurostore_arms.py, 2026-11-03).
        seed = None if (run_dir / "outputs" / "execution_manifest.json").exists() else baseline_dir
        print(f"cache seed: {seed.name if seed else 'none (resuming this arm)'}", flush=True)
        pipeline = AutonimaPipeline(
            ConfigManager().load_from_dict(config),
            num_workers=args.workers,
            copy_valid_cache_from=str(seed) if seed else None,
        )
        result = asyncio.run(pipeline.run())

    print(json.dumps({
        "project": args.project, "arm": args.arm, "baseline": baseline_dir.name, "run": str(run_dir),
        "live": args.live, "calls": log,
        "retrieval": result.execution_stats.get("retrieval"),
        "prisma": result.execution_stats.get("prisma_stats"),
    }, indent=1, default=str))


if __name__ == "__main__":
    main()

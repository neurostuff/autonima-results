"""Retry annotation for one finished run, holding every other stage fixed.

`autonima run` would re-query PubMed and try to fetch missing full texts, either of which can
change the corpus. This rebuilds the included studies from final_results.json, runs only the
AnnotationProcessor (which re-annotates only studies whose cached decisions are missing or
failed), and rewrites nimads_annotation.json. Nothing else in the run directory is touched.

Default is a dry run: no API calls, just the list of studies that would be re-annotated. The
cache signatures double as a fidelity check -- if the rebuilt studies or config differed from
the originals, every study would show as pending, and the script refuses to go on.

    python3 tools/retry_annotation.py projects/decision_making/v3 --expect 22
    python3 tools/retry_annotation.py projects/decision_making/v3 --expect 22 --execute
"""

import argparse
import dataclasses
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from autonima.annotation.processor import AnnotationProcessor
from autonima.config import ConfigManager
from autonima.coordinates.nimads_models import (
    create_annotations_from_results,
    sanitize_annotation_dict,
)
from autonima.coordinates.schema import Analysis
from autonima.models.types import ActivationTable, Study, StudyStatus
from autonima.utils.criteria import load_criteria_mapping


def load_config(run_dir: Path):
    manager = ConfigManager()
    load = getattr(manager, "load_from_file", None) or manager.load_config
    return load(str(run_dir / "outputs" / "config.executed.yaml"))


def annotation_config_with_mapping(config, run_dir: Path):
    """Mirror of AutonimaPipeline._execute_annotation_phase's criteria-mapping injection."""
    annotation_config = deepcopy(config.annotation)
    criteria_mapping = load_criteria_mapping(str(run_dir))
    if not (criteria_mapping and "annotation" in criteria_mapping):
        return annotation_config
    data = criteria_mapping["annotation"]
    global_mapping = data.get("global", {})
    global_inclusion = global_mapping.get("inclusion", {})
    global_exclusion = global_mapping.get("exclusion", {})
    if global_mapping and annotation_config.inclusion_criteria:
        annotation_config.inclusion_criteria = list(global_inclusion.values())
        annotation_config.exclusion_criteria = list(global_exclusion.values())
    for annotation in annotation_config.annotations:
        local = data.get("annotations", {}).get(annotation.name)
        if local is None:
            continue
        local_inclusion = local.get("inclusion", {})
        local_exclusion = local.get("exclusion", {})
        annotation.criteria_mapping = {
            "inclusion": {**global_inclusion, **local_inclusion},
            "exclusion": {**global_exclusion, **local_exclusion},
            "global_inclusion": global_inclusion,
            "global_exclusion": global_exclusion,
            "local_inclusion": local_inclusion,
            "local_exclusion": local_exclusion,
        }
    return annotation_config


def rebuild_study(row: dict, run_dir: Path) -> Study:
    names = {f.name for f in dataclasses.fields(Study)}
    kwargs = {k: v for k, v in row.items() if k in names}
    kwargs["activation_tables"] = [ActivationTable(**t) for t in row.get("activation_tables") or []]
    kwargs["analyses"] = [Analysis(**a) for a in row.get("analyses") or []]
    study = Study(**kwargs)
    if not study.full_text_output_dir:
        study.full_text_output_dir = str(run_dir)
    return study


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--expect", type=int, required=True,
                    help="number of studies the audit says are missing decisions")
    ap.add_argument("--execute", action="store_true", help="make the API calls")
    ap.add_argument("--workers", type=int, default=1, help="studies annotated in parallel")
    args = ap.parse_args()
    run_dir = args.run_dir

    config = load_config(run_dir)
    annotation_config = annotation_config_with_mapping(config, run_dir)
    final = json.loads((run_dir / "outputs" / "final_results.json").read_text())
    included = [
        rebuild_study(row, run_dir) for row in final["studies"]
        if row.get("status") == StudyStatus.INCLUDED_FULLTEXT.value and row.get("analyses")
    ]

    processor = AnnotationProcessor(annotation_config, num_workers=args.workers)
    cached = processor._load_cached_results(str(run_dir))
    decided = {
        (d.analysis_id, d.annotation_name) for d in cached if not d.failed
    }
    names = [a.name for a in annotation_config.annotations]

    def undecided(study):
        return any(
            (f"{study.pmid}_analysis_{i}", name) not in decided
            for i in range(len(study.analyses)) for name in names
        )

    pending = [
        study for study in included
        if len(processor._get_annotations_with_complete_results_for_study(
            cached, study.pmid, annotation_config.annotations, study))
        < len(names)
    ]
    # Only fill gaps. A study with every pair decided but a stale signature would also be
    # redone by a normal run; here it is left exactly as the published outputs used it.
    retry = [study for study in pending if undecided(study)]
    stale = [study for study in pending if not undecided(study)]
    system_ok = processor._get_studies_with_complete_results(cached, "all_analyses", included)

    print(f"{run_dir}: {len(included)} included studies, {len(retry)} with missing or "
          f"failed decisions")
    print("  retry:", " ".join(sorted(s.pmid for s in retry)))
    if stale:
        print("  left as is (decided, but inputs no longer hash the same):",
              " ".join(sorted(s.pmid for s in stale)))
    if len(system_ok) != len(included):
        print(f"  REFUSING: all_analyses signatures match for only {len(system_ok)} of "
              f"{len(included)} studies, so the rebuilt inputs differ from the originals.")
        return 2
    if len(retry) != args.expect:
        print(f"  REFUSING: expected {args.expect} studies to retry, found {len(retry)}.")
        return 2
    if not args.execute:
        print("  dry run: no API calls made. Re-run with --execute.")
        return 0
    # Without a key every call fails; it is recorded as a failure, but costs a pass for nothing.
    if (annotation_config.backend or "openai").lower() == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("  REFUSING: OPENAI_API_KEY is not set in this shell.")
        return 2

    # Passing only the retried studies leaves every other row in annotation_results.json as
    # it is: the save merges per study.
    results = processor.process_studies(included_studies=retry, output_dir=str(run_dir))
    stats = processor.cache_stats
    print(f"  processed={stats['processed']} reused={stats['reused']} "
          f"failed={stats['failed']} failed_studies={stats['failed_studies']}")

    # Rewrite nimads_annotation.json against the unchanged studyset, keeping its IDs.
    studyset_path = run_dir / "outputs" / "nimads_studyset.json"
    studyset_json = json.loads(studyset_path.read_text())
    studyset = SimpleNamespace(studies=[
        SimpleNamespace(id=s["id"], analyses=[SimpleNamespace(id=a["id"]) for a in s["analyses"]])
        for s in studyset_json["studies"]
    ])
    included_ids = {s.pmid for s in included}
    annotation = create_annotations_from_results(
        studyset_json["id"], studyset, processor._load_cached_results(str(run_dir)),
        unknown_when_missing={a.name: included_ids for a in annotation_config.annotations},
    )
    out = run_dir / "outputs" / "nimads_annotation.json"
    out.write_text(json.dumps(sanitize_annotation_dict(annotation.to_dict()), indent=2))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

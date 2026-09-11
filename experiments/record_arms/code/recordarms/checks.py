"""Preflight assertions and per-step status.

`status()` is also the migration report: for a project run before this package existed it
says which steps its artefacts already satisfy and which have to be redone.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import paths
from .spec import Spec


@dataclass
class Result:
    ok: bool
    detail: str

    def __str__(self) -> str:
        return ("PASS " if self.ok else "FAIL ") + self.detail


def gold(meta_pmid: str) -> set[str]:
    if not paths.GOLD_CSV.is_file():
        return set()
    with paths.GOLD_CSV.open() as fh:
        return {r["study_pmid"] for r in csv.DictReader(fh) if r["meta_pmid"] == meta_pmid}


def bench_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(paths.BENCH), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------- preflight

def check_benchmark(spec: Spec) -> Result:
    if paths.BENCH.is_symlink():
        return Result(False, f"BENCH is a symlink ({paths.BENCH}); clone it instead -- a "
                             "stale checkout is why one project read as having no gold")
    g = gold(spec.meta_pmid)
    if not g:
        return Result(False, f"no gold rows for meta_pmid {spec.meta_pmid} in {paths.GOLD_CSV}")
    return Result(True, f"gold {len(g)} papers, benchmark at {bench_commit()}")


def check_manual_maps(spec: Spec) -> Result:
    m = spec.mapping
    if not m:
        return Result(False, "nmb_mappings.json has no usable annotation_mappings")
    missing = [k for k in m if not paths.manual_map(spec.project, k).is_file()]
    if missing:
        return Result(False, f"no manual map for {', '.join(missing)}")
    return Result(True, f"{len(m)} manual maps present ({', '.join(sorted(m))})")


def check_baseline_config(spec: Spec) -> Result:
    path = spec.baseline_config()
    if not path.is_file():
        return Result(False, f"no baseline config at {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    problems = []

    fields = ((cfg.get("annotation") or {}).get("metadata_fields")) or []
    if "study_fulltext" not in fields:
        problems.append("annotation.metadata_fields lacks study_fulltext, so the annotation "
                        "stage cannot tell the arms apart")

    bare = sorted({m for m in _models(cfg) if not m.startswith("@")})
    if bare:
        problems.append(f"bare model slug(s) {bare}: record-vs-text would be confounded with "
                        "provider routing")

    # The baseline's retrieval roots only have to exist here if the baseline itself would
    # have to run. It never does: the arms read their own record mirrors and consume the
    # baseline's cached outputs. Re-running a baseline whose sources live on another host is
    # what destroyed substance use's retrieval cache, so the condition is "outputs absent",
    # not "roots absent" -- an absent root with outputs present is a note, not a failure.
    absent = [src.get("root_path") for src in
              ((cfg.get("retrieval") or {}).get("full_text_sources") or [])
              if src.get("root_path") and not Path(src["root_path"]).exists()]
    consumable = (paths.outputs(spec.project, spec.baseline_run)
                  / "final_results.json").is_file()
    notes = []
    if absent:
        if consumable:
            notes.append(f"{len(absent)} retrieval root(s) live on another host; fine while "
                         "the baseline is only a cache donor, but it cannot be re-run here")
        else:
            problems.append(f"retrieval roots absent and no cached outputs to consume: "
                            f"{absent[:2]}")

    detail = "; ".join(problems + notes) or ("study_fulltext set, models namespaced, "
                                             "retrieval roots present")
    return Result(not problems, detail)


def _normalise(cfg):
    """Config with the two differences `gen_arm_config` introduces by design removed.

    Those are every model id gaining the gateway namespace, and the record-format note
    landing in `screening.fulltext.additional_instructions`. The note is the arm's whole
    point -- it tells the screener the input is an extraction record, not prose -- so its
    presence must not read as config drift. Anything else differing is drift.
    """
    out = _strip_models(cfg)
    ft = ((out.get("screening") or {}).get("fulltext"))
    if isinstance(ft, dict):
        ft.pop("additional_instructions", None)
    return out


def _strip_models(node):
    """The config with every `model` value reduced to its bare slug."""
    if isinstance(node, dict):
        return {k: (v.split("/")[-1] if k.endswith("model") and isinstance(v, str)
                    else _strip_models(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [_strip_models(v) for v in node]
    return node


def _models(cfg) -> list[str]:
    found = []
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "model" and isinstance(v, str):
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(cfg)
    return found


def check_toolchain(_spec: Spec) -> Result:
    missing = []
    for venv, mods in ((paths.PONDIE / ".venv/bin/python", ["lxml", "readabilipy"]),
                       (paths.AUTONIMA_VENV / "bin/python", ["nimare", "nilearn"])):
        if not venv.is_file():
            missing.append(f"{venv} absent")
            continue
        code = "import " + ", ".join(mods)
        r = subprocess.run([str(venv), "-c", code], capture_output=True, text=True, timeout=180)
        if r.returncode:
            missing.append(f"{venv.parent.parent.name}: {r.stderr.strip().splitlines()[-1]}")
    return Result(not missing, "; ".join(missing) or "pondie and scoring venvs complete")


PREFLIGHT = [("benchmark", check_benchmark), ("manual maps", check_manual_maps),
             ("baseline config", check_baseline_config), ("toolchain", check_toolchain)]


def preflight(spec: Spec) -> list[tuple[str, Result]]:
    return [(name, fn(spec)) for name, fn in PREFLIGHT]


# --------------------------------------------------------------------- step status

def _screened(spec: Spec, run: str) -> set[str]:
    f = paths.outputs(spec.project, run) / "fulltext_screening_results.json"
    if not f.is_file():
        return set()
    return {str(r["study_id"])
            for r in json.loads(f.read_text()).get("screening_results", [])}


def step_manifest(spec: Spec) -> Result:
    if not spec.manifest.is_file():
        return Result(False, "no cohort manifest")
    with spec.manifest.open() as fh:
        rows = list(csv.DictReader(fh))
    routed = [r for r in rows if r.get("build_source")]
    base = _screened(spec, spec.baseline_run)
    covers = len({r["pmid"] for r in rows} & base)
    ok = bool(rows) and covers == len(base) and spec.ids_tsv.is_file()
    return Result(ok, f"{len(rows)} rows, {len(routed)} routed, covers {covers}/{len(base)} "
                      f"screened, ids_tsv={'yes' if spec.ids_tsv.is_file() else 'NO'}")


def step_corpus(spec: Spec) -> Result:
    if not spec.manifest.is_file():
        return Result(False, "manifest first")
    with spec.manifest.open() as fh:
        want = {r["pmid"] for r in csv.DictReader(fh) if r.get("build_source")}
    built = {p for p in want if (paths.CORPUS / p).is_dir()}
    return Result(bool(want) and len(built) >= len(want) - 1,
                  f"{len(built)}/{len(want)} papers in the corpus")


def step_extract(spec: Spec) -> Result:
    rec = paths.PONDIE / "data/runs" / spec.pondie_run / "records"
    if not rec.is_dir():
        return Result(False, f"no extraction run at {rec}")
    n = len(list(rec.glob("*.extraction.json")))
    with spec.manifest.open() as fh:
        want = sum(1 for r in csv.DictReader(fh)
                   if r.get("build_source") and (paths.CORPUS / r["pmid"]).is_dir())
    return Result(n >= want - 1 and n > 0, f"{n} records against {want} built")


def arm_mirror(spec: Spec, arm_label: str) -> tuple[str, bool]:
    """The mirror an arm actually reads, and whether it is the canonical name.

    Runs made before this package existed used inconsistent mirror names -- bare
    `record-with-evidence` for PTSD, `sud-record-*`, `cue-record-*` -- so the check reads the
    generated config rather than asserting the name it would choose today. Renaming them
    would break the configs that point at them for no gain; new runs get the canonical name.
    """
    canonical = spec.mirror_name(arm_label)
    cfg_path = spec.arm_config(arm_label)
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        for src in ((cfg.get("retrieval") or {}).get("full_text_sources") or []):
            root = str(src.get("root_path") or "")
            if str(paths.MIRRORS) in root:
                name = Path(root.rstrip("/")).relative_to(paths.MIRRORS).parts[0]
                return name, name == canonical
    return canonical, True


def step_mirrors(spec: Spec) -> Result:
    notes, bad = [], []
    for label in spec.record_arms:
        name, canonical = arm_mirror(spec, label)
        if not (paths.MIRRORS / name).is_dir():
            bad.append(f"no mirror {name}")
        elif not canonical:
            notes.append(name)
    if bad:
        return Result(False, "; ".join(bad))
    if notes:
        return Result(True, f"present under legacy names {sorted(set(notes))}")
    return Result(True, "both record mirrors present, canonical names")


def step_configs(spec: Spec) -> Result:
    """The arms must differ from the FULL-TEXT ARM only in retrieval and the record note.

    Compared against the text arm rather than the generation stem, because the text arm is
    the control the record arms are contrasted with. For PTSD they diverge: `v1.yaml` is the
    stem and still annotates from `study_abstract`, while the text arm `v1-A1-mini.yaml` and
    both record arms were corrected to `study_fulltext`. Comparing to the stem would report
    that correction as drift; comparing to the control shows the three arms agree, which is
    what has to be true. Regenerating from the stem would reintroduce the defect -- that is
    recorded in the descriptor, not papered over here.
    """
    problems = []
    control = spec.control_config()
    base = yaml.safe_load(control.read_text()) if control.is_file() else {}
    for label in spec.record_arms:
        p = spec.arm_config(label)
        if not p.is_file():
            problems.append(f"no {p.name}")
            continue
        cfg = yaml.safe_load(p.read_text()) or {}
        # gen_arm_config namespaces every model id by design, which shows up under
        # annotation/parsing/screening. Normalise that away before comparing, so the gate
        # catches a real criteria or stage change rather than the intended rewrite.
        b, c = _normalise(base), _normalise(cfg)
        changed = sorted(k for k in set(b) | set(c) if b.get(k) != c.get(k))
        extra = [k for k in changed if k != "retrieval"]
        if extra:
            problems.append(f"{p.name} differs from the baseline outside retrieval, "
                            f"model naming and the record note: {extra}")
    return Result(not problems, "; ".join(problems) or "arm configs differ only under retrieval")


def step_arms(spec: Spec) -> Result:
    missing = [r for r in spec.record_arms.values()
               if not (paths.outputs(spec.project, r) / "final_results.json").is_file()]
    if missing:
        return Result(False, "; ".join(f"{r} not run" for r in missing))
    counts = {label: len(_screened(spec, run)) for label, run in spec.arms.items()}
    return Result(True, "screened " + ", ".join(f"{k.split()[0]}={v}" for k, v in counts.items()))


def step_maps(spec: Spec) -> Result:
    gaps = []
    for manual_key, auto_key in spec.mapping.items():
        for run in spec.arms.values():
            if not paths.auto_map(spec.project, run, auto_key).is_file():
                gaps.append(f"{run}/{auto_key}")
        if not paths.baseline_map(spec.project, manual_key).is_file():
            gaps.append(f"baselines/{manual_key}")
    return Result(not gaps, f"{len(gaps)} missing: {gaps[:4]}" if gaps
                  else f"{len(spec.mapping)} columns x 3 arms + baselines")


def step_score(spec: Spec) -> Result:
    f = paths.DATA / f"{spec.project}_metrics.csv"
    return Result(f.is_file(), f"{'present' if f.is_file() else 'absent'}: {f.name}")


STEPS = [("1 manifest", step_manifest), ("2 corpus", step_corpus), ("3 extract", step_extract),
         ("4 mirrors", step_mirrors), ("5 configs", step_configs), ("6 arms", step_arms),
         ("7 maps", step_maps), ("8 score", step_score)]


def status(spec: Spec) -> list[tuple[str, Result]]:
    out = []
    for name, fn in STEPS:
        try:
            out.append((name, fn(spec)))
        except Exception as exc:                      # a missing input is a status, not a crash
            out.append((name, Result(False, f"{type(exc).__name__}: {exc}")))
    return out

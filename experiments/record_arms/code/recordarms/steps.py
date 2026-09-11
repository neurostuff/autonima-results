"""The eight steps. Heavy stages wrap the proven scripts rather than reimplementing them.

Rewriting the renderer or the extractor to live here would put a second implementation of
each behind the same name, which is how the two copies of build_corpus.py came to disagree.
What this module owns is the interface, the ordering and the gates.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import paths
from .spec import Spec

PONDIE_PY = paths.PONDIE / ".venv/bin/python"
AUTO_PY = paths.AUTONIMA_VENV / "bin/python"
AUTONIMA = paths.AUTONIMA_VENV / "bin/autonima"
ARM_SCRIPTS = paths.REPO / "scripts/pondie_arm"
CODE = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    import os
    e = {**os.environ, "AUTONIMA_RESULTS": str(paths.REPO)}
    if env:
        e.update(env)
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd or paths.REPO), env=e)
    if r.returncode:
        sys.exit(f"failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")


# ------------------------------------------------------------------ 1 manifest

#: Real ACE HTML first: it carries headings and tables, the text.csv export carries neither,
#: and dementia's arms were measurably degraded by being built from the inferior route.
ROUTES = [
    ("ace", "ace", lambda exp: {p.stem: str(p.relative_to(exp))
                                for p in (exp / "articles/ace_outputs/html").rglob("*.html")}),
    ("elsevier", "elsevier", lambda exp: {
        d.name: f"articles/elsevier_output/{d.name}"
        for d in (exp / "articles/elsevier_output").iterdir() if (d / "text.txt").is_file()}),
    ("pubget_text", "pubget", lambda exp: {
        d.name: f"pubget_pmc/text_by_pmid/{d.name}"
        for d in (exp / "pubget_pmc/text_by_pmid").iterdir() if (d / "text.txt").is_file()}),
]


def manifest(spec: Spec) -> None:
    o = paths.outputs(spec.project, spec.baseline_run)
    rows_scr = json.loads((o / "fulltext_screening_results.json").read_text())["screening_results"]
    decision = {str(r["study_id"]): r.get("decision", "") for r in rows_scr}
    ret = json.loads((o / "fulltext_retrieval_results.json").read_text())
    meta = {str(s["pmid"]): s for s in ret.get("studies_with_fulltext", []) if s.get("pmid")}

    tables = []
    for name, mirror, build in ROUTES:
        try:
            tables.append((name, mirror, build(paths.EXP)))
        except FileNotFoundError:
            tables.append((name, mirror, {}))

    out, counts = [], {}
    for pmid in sorted(decision):
        route = mirror = src = ""
        for name, mir, table in tables:
            if pmid in table:
                route, mirror, src = name, mir, table[pmid]
                break
        counts[route or "NONE"] = counts.get(route or "NONE", 0) + 1
        e = meta.get(pmid, {})
        out.append({"project": spec.project, "pmid": pmid, "baseline_source": route,
                    "decision": decision[pmid],
                    "coordinates_found": bool(e.get("coordinates_found")),
                    "screened": True, "in_annotation_only": False,
                    "has_baseline_fulltext": pmid in meta,
                    "pmc_recoverable": pmid in tables[2][2],
                    "build_source": route, "mirror": mirror, "source_path": src,
                    "in_cohort": bool(route)})

    spec.manifest.parent.mkdir(parents=True, exist_ok=True)
    with spec.manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    #: `pmid<TAB>study_id<TAB>source`; a bare id per line is silently parsed to nothing.
    with spec.ids_tsv.open("w") as fh:
        for r in out:
            if r["in_cohort"]:
                fh.write(f"{r['pmid']}\t{r['pmid']}\t{r['build_source']}\n")
    print(f"  {len(out)} rows -> {spec.manifest}")
    print(f"  routes: {counts}")


# ------------------------------------------------------------------ 2 corpus

def corpus(spec: Spec) -> None:
    run([PONDIE_PY, paths.PONDIE / "scripts/build_corpus.py",
         "--cohort", spec.manifest, "--out", paths.CORPUS], cwd=paths.PONDIE)


# ------------------------------------------------------------------ 3 extract

def extract(spec: Spec, workers: int = 256) -> None:
    e = spec.extract
    run([PONDIE_PY, "-m", "pondie.cli", "extract",
         "--pmids", spec.ids_tsv, "--run", spec.pondie_run, "--corpus", paths.CORPUS,
         "--model", e["model"], "--flavour", e["flavour"], "--effort", e["effort"],
         "--service-tier", e["service_tier"], "--workers", str(workers),
         "--no-progress", "--env", paths.PONDIE / ".env"], cwd=paths.PONDIE)


# ------------------------------------------------------------------ 4 render + mirrors

def mirrors(spec: Spec) -> None:
    spec.staged.parent.mkdir(parents=True, exist_ok=True)
    if spec.staged.exists():
        shutil.rmtree(spec.staged)
    spec.staged.mkdir(parents=True)
    src = paths.PONDIE / "data/runs" / spec.pondie_run / "records"
    n = 0
    for f in src.glob("*.extraction.json"):
        shutil.copy2(f, spec.staged / f.name); n += 1
    print(f"  staged {n} records")

    for label in spec.record_arms:
        arm = spec.mirror_name(label)
        evidence = "full" if "+" in label else "none"
        # Render twice and require byte-identical output: study_full_text_content_hash hashes
        # file content, so a renderer emitting a timestamp would re-screen every paper on
        # resume and read as drift.
        outs = []
        for p in (1, 2):
            d = paths.RENDERED / f"{arm}.{p}"
            if d.exists():
                shutil.rmtree(d)
            run([AUTO_PY, paths.PONDIE / "scripts/render_record.py",
                 "--records", spec.staged, "--out", d, "--evidence", evidence,
                 "--corpus", paths.CORPUS])
            outs.append(d)
        if subprocess.run(["diff", "-r", str(outs[0]), str(outs[1])],
                          capture_output=True).returncode:
            sys.exit(f"renderer is not deterministic for {arm}")
        final = paths.RENDERED / arm
        if final.exists():
            shutil.rmtree(final)
        outs[0].rename(final)
        shutil.rmtree(outs[1])
        n_rendered = len(list(final.iterdir()))
        print(f"  {arm}: deterministic over {n_rendered} records")
        # --cohort is required, not optional: build_record_corpus defaults to
        # pmids/cohort.csv and silently drops every record absent from it as
        # `not_in_cohort`. Omitting it here put 11 of 525 papers in the mirror and the arms
        # would have screened that as the whole corpus.
        run([AUTO_PY, ARM_SCRIPTS / "build_record_corpus.py",
             "--rendered", final, "--arm", arm, "--cohort", spec.manifest])
        # Count from the builder's own MANIFEST.csv, not the filesystem: the routes do not
        # share a layout -- ace writes a flat <pmid>.txt while elsevier and pubget write
        # <pmid>/text.txt -- so globbing text.txt undercounts by every ace paper.
        mf = paths.MIRRORS / arm / "MANIFEST.csv"
        built = sum(1 for _ in mf.open()) - 1 if mf.is_file() else 0
        if built < n_rendered * 0.9:
            sys.exit(f"mirror {arm} holds {built} papers against {n_rendered} rendered; "
                     "check that the manifest covers the cohort")
        print(f"  {arm}: mirror holds {built} papers")


# ------------------------------------------------------------------ 5 configs

def configs(spec: Spec) -> None:
    for label in spec.record_arms:
        run([AUTO_PY, ARM_SCRIPTS / "gen_arm_config.py",
             "--baseline", spec.baseline_config(),
             "--out", spec.arm_config(label), "--arm", spec.mirror_name(label)])


# ------------------------------------------------------------------ 6 arms

def arms(spec: Spec, jobs: int = 8) -> None:
    """Donate upstream of screening only, and arm two from arm one.

    Donating the annotation stage gave cue's record arms 6-16 papers per key that the arm
    itself had rejected. Donating both arms from the baseline left substance use's arms
    diverging on 35 papers, which makes the evidence contrast a two-variable comparison.
    """
    donor = paths.REPO / "projects" / spec.project / spec.baseline_run
    for label, run_name in spec.record_arms.items():
        _seed_manifest(donor, paths.REPO / "projects" / spec.project / run_name)
        run([AUTONIMA, "run", spec.arm_config(label), "-j", str(jobs),
             "--copy-valid-cache-from", donor])
        # annotation is never donated; drop any entry the copy brought across
        stale = paths.outputs(spec.project, run_name) / "annotation_results.json"
        donor = paths.REPO / "projects" / spec.project / run_name
        print(f"  {run_name} done (next donor: {donor.name})")
        if stale.is_file():
            print(f"  note: {stale.name} present; contamination gate will verify it")


def _seed_manifest(donor: Path, dest: Path) -> None:
    """Put the donor's execution_manifest.json beside the artefacts it will donate.

    `--copy-valid-cache-from` copies result files but not the manifest, and autonima then
    refuses: `_has_cache_artifacts` is true while `manifest_is_modern` is false, so
    `unsupported_cache` trips and the run aborts with "cannot be verified by this version".
    Donation into a virgin run directory therefore always fails -- the four earlier projects
    only worked because their arm directories already held a manifest from a previous run.

    The donor's manifest is the right one to seed: it describes the stage hashes the copied
    artefacts were produced under, so stages whose hashes differ under the arm's config --
    retrieval and everything downstream of it -- are recomputed, which is the intent.
    """
    src = donor / "outputs/execution_manifest.json"
    out = dest / "outputs"
    if not src.is_file():
        return
    out.mkdir(parents=True, exist_ok=True)
    target = out / "execution_manifest.json"
    if not target.is_file():
        shutil.copy2(src, target)
        print(f"  seeded {target.relative_to(paths.REPO)} from {donor.name}")


# ------------------------------------------------------------------ 7 maps

def maps(spec: Spec, cores: int = 6) -> None:
    """Clear before regenerating: run_mkda skips any key whose z.nii.gz already exists."""
    targets = [paths.REPO / "projects" / spec.project / r for r in spec.arms.values()]
    targets += [paths.REPO / "projects" / spec.project / "baselines" / k
                for k in spec.mapping]
    for t in targets:
        if not (t / "outputs/nimads_studyset.json").is_file():
            print(f"  skip {t.name}: no studyset")
            continue
        mar = t / "outputs/meta_analysis_results"
        if mar.exists():
            shutil.rmtree(mar)
        run([AUTO_PY, CODE / "run_mkda.py", "--run-dir", t, "--n-cores", str(cores)])

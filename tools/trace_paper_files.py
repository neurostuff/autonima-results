#!/usr/bin/env python3
"""Turn an strace of paper/reproduce.sh into the export file lists for autonima-paper.

HOW THE TRACE IS MADE (in a clean checkout, maps restored, neurometabench beside it):

    strace -f -qq -e trace=openat,execve -o trace_raw.log  bash run_trace.sh

where run_trace.sh runs paper/reproduce.sh (null cut to a few draws: it opens the same inputs),
then scripts/plot_cbma_three_arm.py and scripts/compile_table_s1.py.

WHAT IS KEPT. Every file opened successfully, for reading or writing, that is tracked in git:
inputs the chain reads and the tracked outputs it regenerates (reports, figures, tables).
Opens under meta_analysis_results/ are collected separately as map directories -- the maps are
deposited as maps.tar.gz, not tracked. Untracked opens (article full texts, caches) are reported
and dropped.

    tools/trace_paper_files.py TRACE_RAW --repo CHECKOUT [--by-script]

Writes paper/export/files.txt and paper/export/map_dirs.txt in this repository.

RUN SCOPE. Two dashboard steps (the cross-project screening report and gold survival) scan EVERY
run folder of a project, so the raw trace touches runs the paper never reports -- intermediate
versions, NeuroStore arms. Runs are kept only if run_categories.yaml registers them (any tier,
any family) or they are search-only baselines; files of other runs, and their per-run screening
evaluations under reports/cross_project_screening/evaluations/, are dropped. Re-running
reproduce.sh in the exported tree then regenerates those dashboards over the paper's runs only.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPEN = re.compile(r'^(\d+)\s+openat\(AT_FDCWD, "([^"]+)", ([A-Z_|]+)[^)]*\)\s+=\s+(-?\d+)')
EXEC = re.compile(r'^(\d+)\s+execve\("([^"]+)", \[(.*?)\]')
MAPS = "/meta_analysis_results/"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--repo", type=Path, required=True, help="the traced checkout")
    ap.add_argument("--by-script", action="store_true", help="print which script opened what")
    args = ap.parse_args()
    root = str(args.repo.resolve()) + "/"
    tracked = set(subprocess.run(["git", "-C", str(args.repo), "ls-files"], check=True,
                                 capture_output=True, text=True).stdout.splitlines())

    cmd: dict[str, str] = {}
    files, map_dirs, untracked = set(), set(), collections.Counter()
    by_script: dict[str, set] = collections.defaultdict(set)
    for line in args.trace.open(errors="replace"):
        m = EXEC.match(line)
        if m and " = 0" in line:
            argv = re.findall(r'"([^"]*)"', m.group(3))
            script = next((a for a in argv if a.endswith((".py", ".sh")) or "/scripts/" in a
                           or "/paper/" in a), m.group(2))
            cmd[m.group(1)] = script.replace(root, "")
            continue
        m = OPEN.match(line)
        if not m or int(m.group(4)) < 0:
            continue
        pid, path = m.group(1), m.group(2)
        if path.startswith(root):
            rel = path[len(root):]
        elif path.startswith("/"):
            continue
        else:
            rel = path
        rel = os.path.normpath(rel)
        if rel.startswith(".."):
            continue
        # A module imported from cached bytecode is never opened as source: Python stats the
        # .py and opens only __pycache__/<name>.cpython-XY.pyc. Map those back to the source.
        pyc = re.match(r"^(.*?)/?__pycache__/([^/.]+)\.[^/]*\.pyc$", rel)
        if pyc:
            rel = f"{pyc.group(1)}/{pyc.group(2)}.py" if pyc.group(1) else f"{pyc.group(2)}.py"
        if MAPS in "/" + rel:
            head, tail = rel.split("/meta_analysis_results/", 1)
            if "/" in tail:
                map_dirs.add(f"{head}/meta_analysis_results/{tail.split('/')[0]}")
            continue
        if rel in tracked:
            files.add(rel)
            by_script[cmd.get(pid, "?")].add(rel)
        elif not rel.startswith((".git", ".pixi")) and "__pycache__" not in rel:
            untracked[rel.split("/")[0]] += 1

    # Keep registered runs and baselines; drop runs only the scanning dashboards touched.
    import yaml
    reg = yaml.safe_load((args.repo / "run_categories.yaml").read_text())
    registered = set()
    for project, families in (reg.get("projects") or {}).items():
        for tiers in (families or {}).values():
            if isinstance(tiers, dict):
                for v in tiers.values():
                    if isinstance(v, str) and re.fullmatch(r"v\d[\w.-]*", v):
                        registered.add((project, v))

    def keep(f: str) -> bool:
        parts = f.split("/")
        if parts[0] == "projects" and len(parts) >= 4 and parts[2] not in ("baselines", "reports") \
                and not parts[2].endswith((".yaml", ".json", ".md", ".csv")):
            return (parts[1], parts[2]) in registered
        if f.startswith("reports/cross_project_screening/evaluations/") and len(parts) >= 5:
            return (parts[3], parts[4]) in registered
        return True

    dropped = {f for f in files if not keep(f)}
    files -= dropped
    map_dirs = {d for d in map_dirs if keep(d)}
    print(f"dropped {len(dropped)} files of unregistered runs")

    # The whitelist: what the Data availability statement promises for every kept run, whether or
    # not a script happens to open it, plus the environment, licence and map tools.
    kept_runs = {"/".join(f.split("/")[:4]) if "/baselines/" in f else "/".join(f.split("/")[:3])
                 for f in files if f.startswith("projects/") and f.count("/") >= 3
                 and f.split("/")[2] != "reports"}
    kept_runs |= {f"projects/{p}/{v}" for p, v in registered if (args.repo / "projects" / p / v).is_dir()}
    stage_outputs = ("config.executed.yaml", "search_results.json", "abstract_screening_results.json",
                     "fulltext_retrieval_results.json", "fulltext_screening_results.json",
                     "missing_fulltexts.csv", "coordinate_parsing_results.json",
                     "annotation_results.json", "nimads_studyset.json", "nimads_annotation.json",
                     "criteria_mapping.json", "execution_progress.json",
                     # text-free pubget table coordinates for the table-only baseline, where a
                     # run has them (tools/snapshot_table_baseline.py)
                     "table_only_baseline/metadata.csv", "table_only_baseline/tables.csv",
                     "table_only_baseline/coordinates.csv")
    extra = set()
    for run in kept_runs:
        extra |= {f"{run}/outputs/{name}" for name in stage_outputs}
        if "/baselines/" in run:
            extra.add(f"{run}.yaml")                             # projects/<p>/baselines/<key>.yaml
        else:
            project, version = run.split("/")[1], run.split("/")[2]
            extra.add(f"projects/{project}/{version}.yaml")      # the run's configuration
    # paper/reproduce.sh is listed explicitly: the trace runs a copy of it (pointed at the
    # traced checkout), so the original is never opened.
    extra |= {"LICENSE", "pyproject.toml", "pixi.lock", "run_categories.yaml", ".gitattributes",
              "paper/README.md", "paper/reproduce.sh", "tools/bundle_maps.py", "scripts/refit_maps.py",
              "scripts/mkda_fit.py", "scripts/compile_table_s1.py", "scripts/plot_cbma_three_arm.py",
              "reports/criteria_changes.csv", "reports/refit_maps_autonima_v011.csv"}
    extra |= {f for f in tracked if f.startswith("projects/") and f.count("/") == 2
              and f.endswith(("nmb_mappings.json", "baselines.yaml"))}
    extra = sorted(f for f in extra if f in tracked and f not in files)

    out = REPO_ROOT / "paper" / "export"
    out.mkdir(parents=True, exist_ok=True)
    (out / "extra.txt").write_text(
        "# Generated by tools/trace_paper_files.py: per-run configurations and stage outputs\n"
        "# promised by the Data availability statement, plus environment and licence files.\n"
        + "\n".join(extra) + "\n")
    print(f"{len(extra)} whitelisted files added (extra.txt)")
    header = ("# Generated by tools/trace_paper_files.py from an strace of paper/reproduce.sh.\n"
              "# Do not edit by hand; extra.txt holds the deliberate additions.\n")
    (out / "files.txt").write_text(header + "\n".join(sorted(files)) + "\n")
    (out / "map_dirs.txt").write_text(header + "\n".join(sorted(map_dirs)) + "\n")
    runs = {"/".join(f.split("/")[:4]) if "/baselines/" in f else "/".join(f.split("/")[:3])
            for f in files if f.startswith("projects/") and f.count("/") >= 3}
    print(f"{len(files)} tracked files ({len(runs)} run folders), {len(map_dirs)} map dirs")
    if untracked:
        print("untracked opens dropped (top-level):", dict(untracked.most_common(8)))
    if args.by_script:
        for script, fs in sorted(by_script.items(), key=lambda kv: -len(kv[1])):
            rr = {"/".join(f.split("/")[:4]) if "/baselines/" in f else "/".join(f.split("/")[:3])
                  for f in fs if f.startswith("projects/") and f.count("/") >= 3}
            print(f"  {len(fs):5d} files  {len(rr):4d} runs  {script}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

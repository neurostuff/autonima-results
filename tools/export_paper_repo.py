#!/usr/bin/env python3
"""Export the paper's subset of this repository as a self-contained `autonima-paper` tree.

WHY THIS EXISTS

autonima-results is a working repository: alongside the runs behind the AutoNIMA manuscript it
holds archived configurations, exploratory arms (NeuroStore, document arms, pilots) and the
scaffolding of later studies -- 339 run folders, of which the paper reads about a third. The
paper's archive should contain exactly what rebuilds its figures, tables and numbers, and
nothing that belongs to other work. This script produces that tree from a committed file list,
so the export can be regenerated (and re-verified) rather than assembled by hand.

WHERE THE FILE LIST COMES FROM

paper/export/files.txt was derived empirically, not by guessing: paper/reproduce.sh (plus the
Fig. S6 and Table S1 scripts) was run under `strace -f -e trace=openat` in a clean checkout,
and every tracked file it opened was kept. A short whitelist (paper/export/extra.txt) adds what
the Data availability statement promises but the scripts do not happen to open -- each copied
run's configuration and screening outputs, the licence and environment files.
paper/export/map_dirs.txt lists the meta_analysis_results/<column> directories the same trace
read; those maps go into the deposit's maps.tar.gz, not into git.

    tools/export_paper_repo.py --target ../autonima-paper                 # files + overlay
    tools/export_paper_repo.py --target ../autonima-paper --maps-out ../maps.tar.gz

The target is created if missing. An existing target must be a git repository (its .git is
kept and everything else is replaced), so re-exporting into the published repo shows the change
as an ordinary diff.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORT = REPO_ROOT / "paper" / "export"
sys.path.insert(0, str(REPO_ROOT / "tools"))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def clear_target(target: Path) -> None:
    if not target.exists():
        target.mkdir(parents=True)
        return
    if any(target.iterdir()) and not (target / ".git").exists():
        raise SystemExit(f"{target} exists, is not empty and is not a git repository; refusing")
    for child in target.iterdir():
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--maps-out", type=Path, default=None,
                    help="also write the deposit's maps.tar.gz (needs the maps present in this tree)")
    args = ap.parse_args()
    target = args.target.resolve()
    if target == REPO_ROOT or REPO_ROOT in target.parents:
        raise SystemExit("target must be outside this repository")

    tracked = set(git("ls-files").splitlines())
    files = sorted(set(read_list(EXPORT / "files.txt")) | set(read_list(EXPORT / "extra.txt")))
    missing = [f for f in files if f not in tracked]
    if missing:
        raise SystemExit(f"{len(missing)} listed files are not tracked here, e.g. {missing[:3]}")
    map_dirs = read_list(EXPORT / "map_dirs.txt")

    clear_target(target)
    for rel in files:
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)

    # The paper repository's own README, citation and ignore rules replace this repo's.
    overlay = EXPORT / "overlay"
    for src in overlay.rglob("*"):
        if src.is_file():
            dst = target / src.relative_to(overlay)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Same environment, the paper repository's own name.
    py = target / "pyproject.toml"
    py.write_text(py.read_text().replace('name = "autonima-results"', 'name = "autonima-paper"', 1))

    # The refit log (bundle_maps.py's list of paper maps) restricted to the exported map dirs.
    log_src = REPO_ROOT / "reports" / "refit_maps_autonima_v011.csv"
    keep = set(map_dirs)
    with open(log_src, newline="") as fh:
        rows = list(csv.DictReader(fh))
    log_dst = target / "reports" / "refit_maps_autonima_v011.csv"
    log_dst.parent.mkdir(parents=True, exist_ok=True)
    with open(log_dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(r for r in rows if r["map_dir"] in keep)
    not_ok = keep - {r["map_dir"] for r in rows if r["status"] == "ok"}
    if not_ok:
        raise SystemExit(f"{len(not_ok)} map dirs lack an ok refit row, e.g. {sorted(not_ok)[:2]}")

    # Maps: manifest into the tree (tracked), archive to --maps-out (deposited, not tracked).
    import bundle_maps as bm
    mfiles = bm.map_files(keep)
    present = {str(f.parent) for f in mfiles}
    if args.maps_out is not None:
        if present != keep:
            raise SystemExit(f"{len(keep - present)} map dirs absent from this tree")
        manifest = bm.build_manifest(mfiles)
        (target / "reports" / "maps_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        out = args.maps_out.resolve()
        with tarfile.open(out, "w:gz") as tar:
            tar.add(target / "reports" / "maps_manifest.json", arcname="maps_manifest.json")
            for rel in mfiles:
                tar.add(REPO_ROOT / rel, arcname=str(rel))
        print(f"wrote {out} ({out.stat().st_size / 2**20:.0f} MB, {len(mfiles)} files, {len(keep)} dirs)")

    commit = git("rev-parse", "HEAD")
    prov = (EXPORT / "PROVENANCE.template.md").read_text().format(
        commit=commit, short=commit[:8], date=dt.date.today().isoformat(),
        n_files=len(files), n_runs=len({"/".join(f.split("/")[:4]) if "/baselines/" in f
                                        else "/".join(f.split("/")[:3])
                                        for f in files if f.startswith("projects/")
                                        and f.count("/") >= 3 and f.split("/")[2] != "reports"}),
        n_maps=len(keep))
    (target / "PROVENANCE.md").write_text(prov)
    print(f"exported {len(files)} files from autonima-results {commit[:8]} into {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

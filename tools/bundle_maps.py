#!/usr/bin/env python3
"""Bundle the meta-analytic maps for deposit, and put them back afterwards.

WHY THIS EXISTS

The maps under projects/*/*/outputs/meta_analysis_results/ are derived data --
regenerable from the tracked NiMADS studysets with `autonima meta` -- so they are
gitignored and absent from any clone. Every map-level figure reads them, which
makes paper/reproduce.sh runnable only on a machine that has already run the
pipeline.

Zenodo's GitHub integration archives the release zipball, which contains tracked
files only, so the maps do not reach the deposit by themselves either. This
bundles them into one archive to upload alongside the record, and restores them
into the right places on the other side.

    tools/bundle_maps.py bundle            # -> maps.tar.gz + reports/maps_manifest.json
    tools/bundle_maps.py restore maps.tar.gz
    tools/bundle_maps.py verify            # check what is on disk against the manifest

The manifest is committed. That is deliberate: it is small, and it lets anyone
check whether the maps they have are the ones the paper was built from, without
downloading 1.1 GB to find out.

SAFETY

Restore refuses any archive member with an absolute path or a `..` component, and
refuses to write outside the repository root. It verifies sha256 after extracting
and reports mismatches rather than trusting the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "reports" / "maps_manifest.json"
DIR_NAME = "meta_analysis_results"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def map_files() -> list[Path]:
    """Every file under a meta_analysis_results directory, repo-relative, sorted."""
    out: list[Path] = []
    for d in (REPO_ROOT / "projects").rglob(DIR_NAME):
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.is_file():
                out.append(f.relative_to(REPO_ROOT))
    return sorted(out)


def build_manifest(files: list[Path]) -> dict:
    entries = []
    total = 0
    for i, rel in enumerate(files, 1):
        p = REPO_ROOT / rel
        size = p.stat().st_size
        total += size
        entries.append({"path": str(rel), "size": size, "sha256": sha256(p)})
        if i % 200 == 0:
            print(f"    hashed {i}/{len(files)}", file=sys.stderr)
    return {
        "description": "Meta-analytic maps for the AutoNIMA evaluation. Derived data, "
                       "regenerable from the tracked NiMADS studysets with `autonima meta`.",
        "n_files": len(entries),
        "total_bytes": total,
        "files": entries,
    }


def cmd_bundle(args: argparse.Namespace) -> int:
    files = map_files()
    if not files:
        print(f"no {DIR_NAME}/ files found -- nothing to bundle", file=sys.stderr)
        return 1
    print(f"  {len(files)} files under {DIR_NAME}/")
    manifest = build_manifest(files)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"  wrote {MANIFEST.relative_to(REPO_ROOT)} "
          f"({MANIFEST.stat().st_size / 1024:.0f} KB)")

    out = Path(args.output).resolve()
    with tarfile.open(out, "w:gz") as tar:
        # the manifest travels inside the archive too, so it is self-describing
        tar.add(MANIFEST, arcname="maps_manifest.json")
        for i, rel in enumerate(files, 1):
            tar.add(REPO_ROOT / rel, arcname=str(rel))
            if i % 200 == 0:
                print(f"    added {i}/{len(files)}", file=sys.stderr)
    print(f"  wrote {out}  ({out.stat().st_size / 1048576:.0f} MB "
          f"from {manifest['total_bytes'] / 1048576:.0f} MB)")
    return 0


def _safe_members(tar: tarfile.TarFile):
    """Yield members that stay inside the repo. Anything else is a hard error."""
    for m in tar.getmembers():
        name = m.name
        if name == "maps_manifest.json":
            continue
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise SystemExit(f"refusing archive: unsafe member path {name!r}")
        dest = (REPO_ROOT / p).resolve()
        if not str(dest).startswith(str(REPO_ROOT) + os.sep):
            raise SystemExit(f"refusing archive: {name!r} escapes the repository")
        if not m.isfile() and not m.isdir():
            raise SystemExit(f"refusing archive: {name!r} is not a regular file")
        yield m


def cmd_restore(args: argparse.Namespace) -> int:
    archive = Path(args.archive).resolve()
    if not archive.exists():
        print(f"no such archive: {archive}", file=sys.stderr)
        return 1
    with tarfile.open(archive, "r:gz") as tar:
        members = list(_safe_members(tar))
        existing = [m for m in members
                    if m.isfile() and (REPO_ROOT / m.name).exists()]
        if existing and not args.force:
            print(f"  {len(existing)} of {len(members)} files already exist; "
                  f"pass --force to overwrite")
            return 1
        print(f"  extracting {len(members)} entries into {REPO_ROOT}")
        for m in members:
            tar.extract(m, REPO_ROOT)
        try:
            mf = tar.extractfile("maps_manifest.json")
            manifest = json.load(mf) if mf else None
        except KeyError:
            manifest = None

    if manifest is None:
        print("  archive carries no manifest; extracted without verification")
        return 0
    return _verify(manifest, label="restored")


def _verify(manifest: dict, label: str) -> int:
    missing, bad = [], []
    for e in manifest["files"]:
        p = REPO_ROOT / e["path"]
        if not p.exists():
            missing.append(e["path"])
        elif p.stat().st_size != e["size"] or sha256(p) != e["sha256"]:
            bad.append(e["path"])
    ok = manifest["n_files"] - len(missing) - len(bad)
    print(f"  {ok}/{manifest['n_files']} {label} and verified")
    for name, items in (("missing", missing), ("checksum mismatch", bad)):
        if items:
            print(f"  {len(items)} {name}:")
            for x in items[:5]:
                print(f"    {x}")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")
    return 1 if (missing or bad) else 0


def cmd_verify(args: argparse.Namespace) -> int:
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST}; run `bundle` first", file=sys.stderr)
        return 1
    return _verify(json.loads(MANIFEST.read_text()), label="present")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bundle", help="write maps.tar.gz and the manifest")
    b.add_argument("--output", default=str(REPO_ROOT / "maps.tar.gz"))
    b.set_defaults(func=cmd_bundle)

    r = sub.add_parser("restore", help="put a bundle back into the tree")
    r.add_argument("archive")
    r.add_argument("--force", action="store_true", help="overwrite existing files")
    r.set_defaults(func=cmd_restore)

    v = sub.add_parser("verify", help="check the tree against the committed manifest")
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

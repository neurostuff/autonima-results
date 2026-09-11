"""`python -m recordarms <command> --project <name>`"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

from . import checks, paths, scoring, spec as spec_mod, steps


@contextlib.contextmanager
def lock(project: str):
    """One run per project at a time.

    Killing a `run-all` wrapper does not kill the `autonima` it has already spawned, so a
    relaunch can leave two -- and on one occasion three -- processes writing the same run
    directory concurrently. Nothing downstream detects that; the outputs simply become a mix.
    An exclusive lock file makes the second launch refuse instead.
    """
    paths.STATE.mkdir(parents=True, exist_ok=True)
    path = paths.STATE / f"{project}.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = path.read_text().strip() or "unknown"
        sys.exit(f"{project} is already being run by pid {holder} ({path}).\n"
                 f"If that process is gone: check `ps` for a stray `autonima run` on this "
                 f"project first, then remove the lock.")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)

COMMANDS = {
    "status": None, "preflight": None,
    "manifest": steps.manifest, "corpus": steps.corpus, "extract": steps.extract,
    "mirrors": steps.mirrors, "configs": steps.configs, "arms": steps.arms,
    "maps": steps.maps, "score": scoring.write,
}
#: order for `run-all`; preflight gates it
PIPELINE = ["manifest", "corpus", "extract", "mirrors", "configs", "arms", "maps", "score"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="recordarms", description=__doc__)
    ap.add_argument("command", choices=list(COMMANDS) + ["run-all"])
    ap.add_argument("--project", help="descriptor name; omit for status over all projects")
    ap.add_argument("--workers", type=int, default=256, help="extraction workers")
    ap.add_argument("--jobs", type=int, default=8, help="autonima -j")
    ap.add_argument("--from-step", help="run-all: resume at this step")
    ap.add_argument("--force", action="store_true", help="run-all: redo satisfied steps")
    args = ap.parse_args(argv)

    if args.command == "status" and not args.project:
        for project in spec_mod.all_projects():
            report(spec_mod.load(project))
        return 0
    if not args.project:
        ap.error("--project is required for " + args.command)
    s = spec_mod.load(args.project)

    if args.command == "status":
        report(s); return 0
    if args.command == "preflight":
        bad = [r for _, r in checks.preflight(s) if not r.ok]
        for name, r in checks.preflight(s):
            print(f"  {name:18} {r}")
        return 1 if bad else 0

    if args.command == "run-all":
        with lock(s.project):
            return _run_all(s, args)

    with lock(s.project):
        _dispatch(args.command, s, args)
    return 0


def _run_all(s, args) -> int:
    if True:
        for name, r in checks.preflight(s):
            print(f"  {name:18} {r}")
            if not r.ok:
                return 1
        done = {n.split()[1]: r.ok for n, r in checks.status(s)}
        started = args.from_step is None
        for name in PIPELINE:
            if name == args.from_step:
                started = True
            if not started:
                continue
            if done.get(name) and not args.force:
                print(f"== {name}: already satisfied, skipping")
                continue
            print(f"== {name}")
            _dispatch(name, s, args)
        report(s)
        return 0


def _dispatch(name, s, args) -> None:
    fn = COMMANDS[name]
    if name == "extract":
        fn(s, workers=args.workers)
    elif name == "arms":
        fn(s, jobs=args.jobs)
    else:
        fn(s)


def report(s) -> None:
    print(f"\n=== {s.project}  (baseline {s.baseline_run}, meta {s.meta_pmid})")
    for name, r in checks.preflight(s):
        print(f"   preflight {name:16} {r}")
    for name, r in checks.status(s):
        print(f"   {name:14} {r}")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build a shadow projects/ tree exposing only one tier's runs, for recomputing at that tier.

WHY A TREE RATHER THAN A FLAG

Half the evaluation chain takes --tier; the other half does not. compute_gold_survival's
`canonical_run` (shared by compute_stage_precision_recall, compute_attainable_recall and
false_positive_composition) picks the HIGHEST-NUMBERED vN directory it can see, and
annotation_value hardcodes the `best` tier. Pointed at the real projects/ they would silently
compute the tuned result while everything else computed the untuned one -- the exact failure the
`untuned` tier exists to prevent, and one that would leave no trace in the output.

So the tier is enforced by what exists on disk. Each project directory here contains symlinks to
the registered runs for ONE tier and to nothing else version-shaped, which makes
`max(versions)` correct by construction. Everything that is not a run directory -- configs,
baselines/, PMID lists, gold files -- is linked through unchanged, and reports/ is a real
directory so writes land here instead of over the tuned artifacts.

    scripts/build_tier_tree.py --tier untuned --root manual-tier
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tiers import TIERS, load_registry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECTS = REPO_ROOT / "projects"
FAMILIES = ("canonical", "annotation_only", "allstudies")
VERSION_SHAPED = re.compile(r"^v\d+")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=TIERS, default="untuned")
    ap.add_argument("--root", default=str(REPO_ROOT / "manual-tier"))
    ap.add_argument("--keep-reports", action="store_true",
                    help="do not wipe existing reports/ in the tree")
    args = ap.parse_args(argv)

    reg = load_registry()
    root = Path(args.root)
    (root / "reports").mkdir(parents=True, exist_ok=True)

    print(f"tier={args.tier}  root={root.relative_to(REPO_ROOT) if root.is_relative_to(REPO_ROOT) else root}")
    missing: list[str] = []
    for project in sorted(reg):
        pdir = PROJECTS / project
        if not pdir.is_dir():
            continue
        dest = root / "projects" / project
        if dest.exists() and not args.keep_reports:
            shutil.rmtree(dest)
        (dest / "reports").mkdir(parents=True, exist_ok=True)

        wanted: set[str] = set()
        for family in FAMILIES:
            fam = (reg[project] or {}).get(family)
            if not isinstance(fam, dict):
                continue
            run = fam.get(args.tier)
            if run:
                wanted.add(str(run))
            elif fam.get("best"):
                # an unregistered tier for a family that HAS a tuned run is a real gap:
                # say so rather than letting the tree quietly omit the family
                missing.append(f"{project}/{family}")

        linked = 0
        for item in sorted(pdir.iterdir()):
            if item.name == "reports":
                continue
            if item.is_dir() and VERSION_SHAPED.match(item.name) and item.name not in wanted:
                continue  # a run of another tier: hide it, so max(versions) cannot reach it
            (dest / item.name).symlink_to(item.resolve())
            linked += 1
        runs = sorted(w for w in wanted if (dest / w).exists())
        print(f"  {project:<26} runs={','.join(runs) or '(none)':<48} links={linked}")

    if missing:
        print(f"\n  families with no `{args.tier}` run registered: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

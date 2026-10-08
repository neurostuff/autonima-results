#!/usr/bin/env python3
"""Check that a cbma-skills review.yaml carries an autonima config's criteria verbatim.

    python scripts/check_cbma_translation.py projects/<project>/<run>.yaml REVIEW/review.yaml

The cbma-skills arm is only comparable with autonima if both judge against the same
text. This compares every criterion-bearing field after YAML parsing, so a folded
line break or a dropped trailing space counts as a difference.

Scopes, autonima -> review.yaml:

    screening.<stage>.{inclusion,exclusion}_criteria, additional_instructions
        -> screening.<stage>.{inclusion,exclusion}, instructions
    annotation.{inclusion,exclusion}_criteria, additional_instructions
        -> selection.global.{inclusion,exclusion}, selection.instructions
    annotation.annotations[i].{inclusion,exclusion}_criteria, additional_instructions
        -> selection.targets[i].{inclusion,exclusion}, instructions (same names, same order)
    screening.<stage>.objective -> screening.<stage>.objective, or objective
    search.query / date_from / date_to / email -> the same keys under search

A line autonima lists as a criterion may instead sit in the same scope's
instructions, as one paragraph of it (paragraphs are separated by a newline, which a
blank line gives inside a folded YAML scalar). That is how guidance that is not a
property of a study or analysis leaves the criteria list without being reworded. The
rule enforced per scope is:

    * every autonima line appears exactly once, as a criterion or as a paragraph;
    * the criteria that remain keep autonima's order;
    * the instructions hold nothing but moved lines and autonima's own
      additional_instructions.

Moved lines are reported with their autonima index. The script then lists the
autonima settings that have no cbma-skills counterpart, so the run notes can say what
was left behind. Exit code 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Settings that change how autonima runs but carry no criteria text.
NOT_MAPPED = {
    "search": {"database", "max_results"},
    "screening.stage": {"model", "confidence_reporting", "skip_stage", "backend", "model_params",
                        "inclusion_threshold", "exclusion_threshold"},
    "annotation": {"enabled", "model", "create_all_included_annotations", "prompt_type", "metadata_fields",
                   "model_params", "backend", "inclusion_threshold", "exclusion_threshold"},
}


RENDERED: list = []


def _as_text(items):
    """autonima renders each criterion as f"- {criterion}", so a criterion YAML parsed as a mapping
    (a "key: value" line) reached its model as the mapping's str(). Compare that."""
    if not isinstance(items, list):
        return items
    for x in items:
        if not isinstance(x, str):
            RENDERED.append(str(x)[:90])
    return [x if isinstance(x, str) else str(x) for x in items]


def paragraphs(text) -> list[str]:
    return [p for p in (text or "").split("\n") if p.strip()]


class Check:
    def __init__(self):
        self.problems: list[str] = []
        self.moved: list[str] = []
        self.fields = 0

    def same(self, label, a, b):
        self.fields += 1
        if a in (None, "", []) and b in (None, "", []):
            return
        if a != b:
            self.problems.append(f"{label}:\n    autonima: {a!r}\n    review:   {b!r}")

    def scope(self, label, src: dict, inc: list, exc: list, instructions):
        """One criteria scope: autonima's two lists and instructions against the review's."""
        self.fields += 1
        src = {k: _as_text(v) if k.endswith("_criteria") else v for k, v in src.items()}
        pars = paragraphs(instructions)
        own = paragraphs(src.get("additional_instructions"))
        for kind, a_list, r_list in (("inclusion", src.get("inclusion_criteria") or [], inc or []),
                                     ("exclusion", src.get("exclusion_criteria") or [], exc or [])):
            kept = [c for c in a_list if c not in pars]
            if r_list != kept:
                self.problems.append(f"{label} {kind}: the criteria are not autonima's, minus the moved lines, "
                                     f"in autonima's order:\n    expected: {kept!r}\n    review:   {r_list!r}")
            for i, c in enumerate(a_list):
                if c in pars:
                    self.moved.append(f"{label} {kind}_criteria[{i}] -> instructions: {c[:70]}...")
        known = (src.get("inclusion_criteria") or []) + (src.get("exclusion_criteria") or []) + own
        for p in pars:
            if p not in known:
                self.problems.append(f"{label} instructions carry text autonima does not have: {p!r}")
            elif pars.count(p) > 1:
                self.problems.append(f"{label} instructions repeat a paragraph: {p!r}")
        for p in own:
            if p not in pars:
                self.problems.append(f"{label}: autonima's additional_instructions paragraph is missing: {p!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("autonima_config", type=Path)
    ap.add_argument("review_yaml", type=Path)
    args = ap.parse_args(argv)

    src = yaml.safe_load(args.autonima_config.read_text())
    dst = yaml.safe_load(args.review_yaml.read_text())
    ck, unmapped = Check(), []

    # Each screening stage's objective: review.yaml's screening.<stage>.objective if set, else its
    # top-level objective.
    for stage in ("abstract", "fulltext"):
        dst_obj = ((dst.get("screening") or {}).get(stage) or {}).get("objective") or dst.get("objective")
        ck.same(f"objective ({stage})", (src["screening"].get(stage) or {}).get("objective"), dst_obj)

    for key in ("query", "date_from", "date_to", "email"):
        ck.same(f"search.{key}", (src.get("search") or {}).get(key), (dst.get("search") or {}).get(key))
    unmapped += [f"search.{k}" for k in (src.get("search") or {}) if k in NOT_MAPPED["search"]]

    for stage in ("abstract", "fulltext"):
        a = src["screening"].get(stage) or {}
        b = (dst.get("screening") or {}).get(stage) or {}
        ck.scope(f"screening.{stage}", a, b.get("inclusion"), b.get("exclusion"), b.get("instructions"))
        unmapped += [f"screening.{stage}.{k}" for k in a if k in NOT_MAPPED["screening.stage"]]

    ann = src.get("annotation") or {}
    sel = dst.get("selection") or {}
    if ann.get("enabled", True) and ann.get("annotations"):
        glob = sel.get("global") or {}
        ck.scope("annotation (global)", ann, glob.get("inclusion"), glob.get("exclusion"), sel.get("instructions"))
        src_t, dst_t = ann["annotations"], sel.get("targets") or []
        ck.same("selection.targets (names, in order)", [t["name"] for t in src_t], [t.get("name") for t in dst_t])
        for a, b in zip(src_t, dst_t):
            ck.same(f"target {a['name']}.description", a.get("description"), b.get("description"))
            ck.scope(f"target {a['name']}", a, b.get("inclusion"), b.get("exclusion"), b.get("instructions"))
        unmapped += [f"annotation.{k}" for k in ann if k in NOT_MAPPED["annotation"]]
    elif sel.get("targets"):
        ck.problems.append("review.yaml defines selection targets but the autonima config has no annotations")

    for line in ck.moved:
        print(f"moved: {line}")
    for r in RENDERED:
        print(f"note: a criterion autonima parsed as a mapping is compared as its str(): {r}...")
    for label in unmapped:
        print(f"not mapped (no cbma-skills counterpart): {label}")
    if ck.problems:
        print(f"\n{len(ck.problems)} mismatch(es) in {ck.fields} fields and scopes:", file=sys.stderr)
        for p in ck.problems:
            print("  " + p, file=sys.stderr)
        return 1
    print(f"\nOK: {ck.fields} fields and scopes carry autonima's text, {len(ck.moved)} line(s) moved to instructions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Resolve a project's run for a given comparison tier, from run_categories.yaml.

The comparison scripts historically auto-picked the highest version number per project. That
conflates three different things, which matters for the overfitting argument:

    verbatim  criteria transcribed from the source paper before any results were seen. The
              honest attempt: no peeking at the benchmark.
    manual    the author revised BY HAND, having glanced at a few report examples. Light peeking.
    best      THE PREFERRED CONFIG. Curated, not derived: the run we would put forward as the
              system's best showing, with benchmark-informed tuning fully allowed. Usually the
              highest performer; not necessarily the highest version number.

              Selection rule for CANONICAL (search-driven) runs: prefer RECALL over F1. With a
              mixed pool it is not knowable why precision fell -- a "false positive" may be a
              perfectly good paper the source meta-analysis never screened, which is demonstrably
              the case for executive_function (its benchmark searched BrainMap) and
              decision_making (Google Scholar and Web of Science too). Recall has no such
              ambiguity: a paper the pipeline never retrieved is unrecoverable downstream.
    latest    absolute highest version number. Mechanical, kept as a sanity check and as the
              fallback when `best` is unset.

`best` exists because version order stopped tracking quality. A run renamed upward can be older
and worse than a lower-numbered one (social's v5-annotation-only is the pre-fix set, v4 is the
fixed one), and tuning does not always help -- in executive_function and social the honest v1
still beats every tuned successor at map level. Naming the preferred run explicitly removes both
traps, and the gap between `verbatim` and `best` is then a direct measure of what peeking bought.

Runs are registered per project AND per family, because the roll-ups select from different
families: the screening roll-up takes plain `vN` (family "canonical"), while the analysis and
fair roll-ups take `vN-annotation-only` (family "annotation_only").

`resolve_tier` returns None when a tier genuinely does not exist for a project/family. Callers
should fall back to their own auto-pick in that case rather than skipping the project -- the
gaps are real (only three families support a full three-way contrast) and a hard failure would
silently shrink every comparison.
"""

from __future__ import annotations

from pathlib import Path

import yaml

__all__ = [
    "TIERS",
    "NON_TIER_FAMILIES",
    "DEFAULT_REGISTRY",
    "load_registry",
    "resolve_tier",
    "families_for",
    "resolve_decomposition",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "run_categories.yaml"
TIERS = ("verbatim", "manual", "best", "latest")

# Families keyed by role rather than by tier. They live in the same registry because they name
# runs of the same project, but asking them for a tier is a caller error, not a missing entry.
NON_TIER_FAMILIES = frozenset({"decomposition"})

_CACHE: dict[str, dict] = {}


def load_registry(path: Path | str | None = None) -> dict:
    """Load and cache the registry. Returns {} if the file is absent, so callers degrade to
    their own auto-pick rather than failing."""
    p = Path(path) if path else DEFAULT_REGISTRY
    key = str(p)
    if key not in _CACHE:
        if not p.exists():
            _CACHE[key] = {}
        else:
            payload = yaml.safe_load(p.read_text()) or {}
            _CACHE[key] = payload.get("projects") or {}
    return _CACHE[key]


def families_for(project: str, registry: dict | None = None) -> list[str]:
    reg = registry if registry is not None else load_registry()
    entry = reg.get(project) or {}
    return [
        k
        for k, v in entry.items()
        if isinstance(v, dict) and k not in NON_TIER_FAMILIES
    ]


def resolve_tier(
    project: str,
    family: str = "canonical",
    tier: str = "latest",
    registry: dict | None = None,
) -> str | None:
    """Run name for (project, family, tier), or None if unregistered/absent.

    If the requested family is missing but exactly one family is registered, that one is used --
    projects like emotion_regulation_2022 only have `allstudies`, and a caller asking for
    `canonical` should still get the sensible answer rather than nothing.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
    if family in NON_TIER_FAMILIES:
        raise ValueError(
            f"family {family!r} is keyed by role, not by tier; "
            f"use resolve_decomposition({project!r}) instead"
        )
    reg = registry if registry is not None else load_registry()
    entry = reg.get(project) or {}
    fam = entry.get(family)
    if not isinstance(fam, dict):
        candidates = [
            v
            for k, v in entry.items()
            if isinstance(v, dict) and k not in NON_TIER_FAMILIES
        ]
        if len(candidates) != 1:
            return None
        fam = candidates[0]
    value = fam.get(tier)
    if not value and tier == "best":
        # `best` is curated and may be unset; fall back to the mechanical highest version so a
        # caller asking for the preferred run always gets something rather than nothing.
        value = fam.get("latest")
    return str(value) if value else None


def add_tier_argument(parser, default: str = "best") -> None:
    """Attach a uniform --tier flag. Kept here so every script spells it the same way."""
    parser.add_argument(
        "--tier",
        choices=TIERS,
        default=default,
        help=(
            "Which registered run to compare, from run_categories.yaml: "
            "'verbatim' (paper-faithful, no peeking), 'manual' (hand-iterated), "
            "'best' (curated preferred run, tuning allowed), 'latest' (highest version number). "
            f"Default {default}. Falls back to auto-pick when the tier is unregistered."
        ),
    )


def resolve_decomposition(project: str, registry: dict | None = None) -> dict[str, str]:
    """The matched search / pool / annotation arms for `project`, or {} if not registered.

    These three arms share criteria and annotation mode and differ only in how studies enter
    the pipeline, which is what makes a search / screening / annotation split interpretable.
    """
    reg = registry if registry is not None else load_registry()
    fam = (reg.get(project) or {}).get("decomposition")
    if not isinstance(fam, dict):
        return {}
    return {
        role: str(name)
        for role, name in fam.items()
        if role != "note" and name
    }

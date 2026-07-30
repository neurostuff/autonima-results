#!/usr/bin/env python3
"""Shared taxonomy + trigger-variable constants for parser coordinate-separation
error annotation.

Single source of truth for:
- scripts/generate_parser_failure_annotation_report.py (sampling + fillable review UI)
- scripts/build_parser_failure_contingency.py (contingency table / corrupted-pair rate / fix ranking)

Scope: coordinate-separation correctness only -- did the parser carve tables into the
right analysis units with the right coordinates attached. Not expressivity, not facet
extraction.

Deliberately has no numpy/scipy/nimare/matplotlib imports so it stays cheap to import
from either script (or, in principle, from compare_analyses_to_benchmark.py in a future
refactor without adding to that script's import cost).
"""

from __future__ import annotations

from dataclasses import dataclass, field


EXPECTED_DIFFERENCE_DISPOSITION = "expected_difference"

PARSER_EVALUABLE_UNMATCHED_DISPOSITIONS: frozenset[str] = frozenset(
    {"parser_missed", "matching_error"}
)


@dataclass(frozen=True)
class FailureMode:
    id: str
    label: str
    description: str
    severity_weight: float
    is_corrupted_pair: bool
    applies_to: str  # "gold_unit" | "auto_only_unit" | "both"


FAILURE_MODES: list[FailureMode] = [
    FailureMode(
        id="over_split",
        label="Over-split",
        description="A single true (gold) analysis was shattered into multiple predicted units.",
        severity_weight=3.0,
        is_corrupted_pair=True,
        applies_to="gold_unit",
    ),
    FailureMode(
        id="under_split_merge",
        label="Under-split / merge",
        description="Multiple true (gold) analyses were collapsed into one predicted unit.",
        severity_weight=4.0,
        is_corrupted_pair=True,
        applies_to="gold_unit",
    ),
    FailureMode(
        id="misattribution",
        label="Misattribution",
        description=(
            "Right unit count, but the wrong coordinates are attached to this unit "
            "(peaks belonging to analysis A placed under B)."
        ),
        severity_weight=5.0,
        is_corrupted_pair=True,
        applies_to="both",
    ),
    FailureMode(
        id="partial_coord_error",
        label="Partial coordinate error",
        description=(
            "Right unit, right attribution, but individual peaks were dropped, added, "
            "or garbled."
        ),
        severity_weight=1.0,
        is_corrupted_pair=False,
        applies_to="gold_unit",
    ),
    FailureMode(
        id="missed_unit",
        label="Missed unit",
        description="A true (gold) analysis was not recovered at all.",
        severity_weight=0.5,
        is_corrupted_pair=False,
        applies_to="gold_unit",
    ),
    FailureMode(
        id="spurious_unit",
        label="Spurious / fabricated unit",
        description=(
            "A predicted analysis with no true counterpart that is fabricated -- distinct "
            "from a correct analysis the gold set simply didn't include (out of scope)."
        ),
        severity_weight=3.0,
        is_corrupted_pair=True,
        applies_to="auto_only_unit",
    ),
]

FAILURE_MODE_BY_ID: dict[str, FailureMode] = {mode.id: mode for mode in FAILURE_MODES}

# Crosswalk from the existing HUMAN_REVIEW_EXTRACTION_REASONS ids (defined in
# compare_analyses_to_benchmark.py, ~line 49) to zero or more of the FAILURE_MODES ids
# above. Tags are NOT mutually exclusive -- a unit can carry more than one failure mode.
#
# "contrast_label_missed_or_truncated" and "other_extraction_issue" map to no failure
# mode: they are naming-only issues (out of this plan's coordinate-separation scope) or
# an unmapped catch-all, and are surfaced as auxiliary free-standing "legacy reason"
# checkboxes rather than folded into the 6-mode taxonomy.
LEGACY_REASON_TO_FAILURE_MODES: dict[str, list[str]] = {
    "multiple_analyses_merged_into_one": ["under_split_merge"],
    "single_analysis_split_into_multiple": ["over_split"],
    "section_header_parsed_as_analysis": ["spurious_unit"],
    "coordinate_rows_assigned_wrong_analysis": ["misattribution"],
    "contrast_label_missed_or_truncated": [],
    "table_structure_misparsed": ["over_split", "under_split_merge", "misattribution"],
    "coordinates_missed_or_incomplete": ["partial_coord_error"],
    "other_extraction_issue": [],
}


@dataclass(frozen=True)
class TriggerVariable:
    id: str
    label: str
    kind: str  # "dropdown" | "checkbox" | "readonly"
    options: tuple[str, ...] = field(default_factory=tuple)
    human_judgment: bool = True  # False => precomputed/read-only, not asked of the reviewer


TRIGGER_VARIABLES: list[TriggerVariable] = [
    TriggerVariable(
        id="table_layout",
        label="Table layout",
        kind="dropdown",
        options=(
            "single_row_per_region",
            "stacked_contrasts",
            "multi_analysis_per_table",
            "one_analysis_one_table",
            "other",
        ),
    ),
    TriggerVariable(
        id="design_type",
        label="Design type",
        kind="dropdown",
        options=(
            "subtractive",
            "parametric",
            "multivariate",
            "conjunction",
            "omnibus",
            "other",
        ),
    ),
    TriggerVariable(
        id="multi_header",
        label="Multi-column-header / merged-cell structure",
        kind="checkbox",
    ),
    TriggerVariable(
        id="spans_multiple_tables",
        label="Analysis coordinates span multiple tables",
        kind="checkbox",
    ),
    TriggerVariable(
        id="footnote_carried_context",
        label="Contrast identity/threshold stated only in caption/footnote",
        kind="checkbox",
    ),
    TriggerVariable(
        id="crowding",
        label="Number of analyses in paper",
        kind="readonly",
        human_judgment=False,
    ),
]

TRIGGER_VARIABLE_BY_ID: dict[str, TriggerVariable] = {tv.id: tv for tv in TRIGGER_VARIABLES}

# A unit "produces a corrupted contrast<->map training pair" if it carries any tag in
# this set. Over-split/merge/misattribution/spurious all corrupt an existing or implied
# pair; partial-coordinate-error and missed-unit are excluded by design (a partially
# wrong peak list or an absent analysis doesn't corrupt an *existing* pair the way a
# wrong unit boundary or wrong coordinate attachment does). Overridable via CLI in
# build_parser_failure_contingency.py.
DEFAULT_CORRUPTED_PAIR_MODES: tuple[str, ...] = tuple(
    mode.id for mode in FAILURE_MODES if mode.is_corrupted_pair
)

# Random-vs-systematic heuristic defaults (overridable via CLI in
# build_parser_failure_contingency.py). This is a simple rate-ratio heuristic, not a
# statistical significance test -- see build_parser_failure_contingency.py's caution text.
DEFAULT_SYSTEMATIC_RATE_MULTIPLE: float = 3.0
DEFAULT_SYSTEMATIC_MIN_COUNT: int = 3

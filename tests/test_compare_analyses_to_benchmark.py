from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_analyses_to_benchmark.py"
)


@pytest.fixture(scope="module")
def matcher():
    spec = importlib.util.spec_from_file_location("compare_analyses_to_benchmark", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_coordinate_score_is_order_independent(matcher):
    manual = [(1.0, 2.0, 3.0), (10.0, 20.0, 30.0), (-4.0, 5.0, 6.0)]
    automatic = list(reversed(manual))

    score, metadata, _reasons = matcher.compute_coord_score(manual, automatic)

    assert score == pytest.approx(1.0)
    assert metadata["strict_exact_coord_set"] is True
    assert metadata["mean_paired_distance_mm"] == pytest.approx(0.0)


def test_coordinate_assignment_minimizes_distance_beyond_score_cutoff(matcher):
    manual = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
    automatic = [(115.0, 0.0, 0.0), (15.0, 0.0, 0.0)]

    _score, metadata, _reasons = matcher.compute_coord_score(manual, automatic)

    assert metadata["mean_paired_distance_mm"] == pytest.approx(15.0)
    assert metadata["max_paired_distance_mm"] == pytest.approx(15.0)


def test_best_space_recovers_preconverted_manual_coordinates(matcher):
    automatic = [
        (-1.0, -2.0, 55.0),
        (7.0, 18.0, 43.0),
        (33.0, -8.0, 51.0),
        (-22.0, -19.0, 51.0),
    ]
    preconverted = [
        tuple(row)
        for row in matcher.mni2tal(np.asarray(automatic, dtype=float)).tolist()
    ]
    manual = {
        "name": "Planning",
        "points": preconverted,
        "coordinate_variants": [
            {
                "name": "manual_original",
                "points": preconverted,
                "use_exact_axis_tolerance": False,
            },
            {
                "name": "manual_talairach_to_mni",
                "points": matcher.convert_coords_talairach_to_mni(preconverted),
                "use_exact_axis_tolerance": True,
            },
            {
                "name": "manual_mni_to_talairach",
                "points": matcher.convert_coords_mni_to_talairach(preconverted),
                "use_exact_axis_tolerance": True,
            },
        ],
    }
    auto = {"name": "Planning", "points": automatic}

    result = matcher.score_pair(
        manual,
        auto,
        coord_accept_override_threshold=0.80,
        converted_talairach_exact_axis_tolerance=1.0,
    )

    assert result["coordinate_variant"] == "manual_talairach_to_mni"
    assert result["coord_score"] == pytest.approx(1.0)
    assert result["coord_override_accepted"] is True


def test_high_coverage_override_requires_spatial_agreement(matcher):
    manual = [(0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (40.0, 0.0, 0.0)]
    nearby = [(5.0, 0.0, 0.0), (25.0, 0.0, 0.0), (45.0, 0.0, 0.0)]
    unrelated = [(100.0, 100.0, 100.0), (120.0, 100.0, 100.0), (140.0, 100.0, 100.0)]

    nearby_result = matcher.score_pair(
        {"name": "manual", "points": manual},
        {"name": "automatic", "points": nearby},
        coord_accept_override_threshold=0.80,
        converted_talairach_exact_axis_tolerance=1.0,
    )
    unrelated_result = matcher.score_pair(
        {"name": "manual", "points": manual},
        {"name": "automatic", "points": unrelated},
        coord_accept_override_threshold=0.80,
        converted_talairach_exact_axis_tolerance=1.0,
    )

    assert nearby_result["high_coverage_coord_set"] is True
    assert nearby_result["coord_override_accepted"] is True
    assert unrelated_result["high_coverage_coord_set"] is False
    assert unrelated_result["coord_override_accepted"] is False


def test_lower_threshold_does_not_accept_incomplete_coordinate_set(matcher):
    manual = [
        (0.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (40.0, 0.0, 0.0),
        (60.0, 0.0, 0.0),
    ]
    automatic = [*manual, (80.0, 0.0, 0.0)]

    result = matcher.score_pair(
        {"name": "manual", "points": manual},
        {"name": "automatic", "points": automatic},
        coord_accept_override_threshold=0.80,
        converted_talairach_exact_axis_tolerance=1.0,
    )

    assert result["coord_score"] == pytest.approx(0.8)
    assert result["equal_coord_count"] is False
    assert result["coord_override_accepted"] is False


def test_parser_review_credits_and_excludes_non_parser_failures(
    matcher,
    tmp_path,
):
    match_result = {
        "matching_policy": {},
        "summary": {},
        "pmids": {
            "123": {
                "manual_analyses": [
                    {
                        "manual_analysis_id": "analysis_1",
                        "match_status": "unmatched",
                        "combined_score": 0.1,
                    },
                    {
                        "manual_analysis_id": "analysis_2",
                        "match_status": "unmatched",
                        "combined_score": 0.2,
                    },
                    {
                        "manual_analysis_id": "analysis_3",
                        "match_status": "unmatched",
                        "combined_score": 0.3,
                    },
                ],
                "pmid_summary": {},
            }
        },
    }
    review_entries = {
        "project:123:analysis_1": {
            "unit_id": "project:123:analysis_1",
            "unmatched_gold_disposition": "gold_standard_error",
        },
        "project:123:analysis_2": {
            "unit_id": "project:123:analysis_2",
            "unmatched_gold_disposition": "supplemental_data",
        },
        "project:123:analysis_3": {
            "unit_id": "project:123:analysis_3",
            "unmatched_gold_disposition": "parser_missed",
        },
    }

    adjustment = matcher.apply_parser_review_adjustments(
        match_result,
        review_entries=review_entries,
        project_name="project",
        review_path=tmp_path / "review.json",
    )

    retained = match_result["pmids"]["123"]["manual_analyses"]
    excluded = match_result["pmids"]["123"][
        "review_excluded_manual_analyses"
    ]
    assert [entry["match_status"] for entry in retained] == [
        "accepted",
        "unmatched",
    ]
    assert retained[0]["raw_match_status"] == "unmatched"
    assert excluded[0]["manual_analysis_id"] == "analysis_2"
    assert match_result["summary"]["manual_analyses_total"] == 2
    assert match_result["summary"]["accepted"] == 1
    assert match_result["summary"]["unmatched"] == 1
    assert adjustment["credited_as_accepted"] == 1
    assert adjustment["excluded_non_parser_evaluable"] == 1
    assert adjustment["confirmed_parser_misses"] == 1

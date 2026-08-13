from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_parser_failure_annotation_report.py"
)


@pytest.fixture(scope="module")
def report_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_parser_failure_annotation_report",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_unit(module, unit_id: str, pmid: str = "123"):
    return module.Unit(
        unit_id=unit_id,
        unit_kind="gold_unit",
        sample_bucket="failure",
        project="project",
        run_dir="/new/run",
        pmid=pmid,
        study_name="Study",
        match_status="unmatched",
        manual_analysis_id=unit_id.rsplit(":", 1)[-1],
        auto_index=0,
        combined_score=0.2,
        table_id="1",
        crowding=2,
        spans_multiple_tables_hint=False,
        pubget_available=False,
    )


def test_existing_review_is_filtered_to_current_sample(report_generator):
    surviving_id = "project:123:analysis_1"
    payload = {
        "reviewer": "AD",
        "entries": [
            {
                "unit_id": surviving_id,
                "match_status": "accepted",
                "unmatched_gold_disposition": "parser_missed",
                "note": "Keep this decision",
                "updated_at": "2026-07-29T12:00:00Z",
            },
            {
                "unit_id": "project:456:analysis_2",
                "unmatched_gold_disposition": "matching_error",
            },
        ],
        "paper_notes": [
            {
                "paper_key": "project:123",
                "missed_table": True,
                "updated_at": "2026-07-29T12:00:00Z",
            },
            {"paper_key": "project:456", "missed_table": True},
        ],
    }

    store, stats = report_generator.build_seeded_browser_store(
        payload,
        [make_unit(report_generator, surviving_id)],
    )

    assert set(store["entries"]) == {surviving_id}
    assert store["entries"][surviving_id]["match_status"] == "unmatched"
    assert store["entries"][surviving_id]["run_dir"] == "/new/run"
    assert store["entries"][surviving_id]["note"] == "Keep this decision"
    assert set(store["paper_notes"]) == {"project:123"}
    assert store["reviewer"] == "AD"
    assert stats == {
        "source_entries": 2,
        "seeded_entries": 1,
        "dropped_entries": 1,
        "source_paper_notes": 2,
        "seeded_paper_notes": 1,
    }


def test_default_sample_focuses_on_unmatched_gold(report_generator, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH)])

    args = report_generator.parse_args()

    assert args.accepted_sample_rate == 0.0
    assert args.accepted_sample_min_per_project == 0
    assert args.spurious_candidate_sample_rate == 0.0
    assert args.include_uncertain_gold is False
    assert args.load_existing_review is True


def test_expected_difference_is_available_for_unmatched_gold(report_generator):
    controls = report_generator.render_unit_review_controls(
        make_unit(report_generator, "project:123:analysis_1"),
        in_sample=True,
        predicted_choices=[],
    )

    assert 'value="expected_difference"' in controls
    assert "Expected source/curation difference" in controls
    assert "[d]" in controls


def test_expected_difference_is_not_counted_as_parser_evaluable():
    script_path = SCRIPT_PATH.with_name("build_parser_failure_contingency.py")
    spec = importlib.util.spec_from_file_location(
        "build_parser_failure_contingency",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.is_parser_evaluable(
        {
            "unit_kind": "gold_unit",
            "match_status": "unmatched",
            "unmatched_gold_disposition": "expected_difference",
        }
    ) is False

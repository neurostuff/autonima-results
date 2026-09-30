"""Records mode: pre-extracted records stand in for full text and for extraction."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[1] / "skills"
FIX = Path(__file__).parent / "fixtures"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pubmed = load("pubmed_search", SKILLS / "pubmed-search/scripts/pubmed_search.py")
gather = load("gather_fulltext", SKILLS / "fulltext-sources/scripts/gather_fulltext.py")
ledger = load("ledger", SKILLS / "cbma-review/scripts/ledger.py")

REVIEW_YAML = """
name: records_test
objective: fMRI studies of working memory in schizophrenia with coordinates
search:
  query: schizophrenia working memory fmri
screening:
  abstract:
    inclusion: [Human participants with schizophrenia]
  fulltext:
    inclusion: [Whole-brain coordinates reported]
fulltext:
  sources:
    - type: local
      name: records
      path: records/text
      pattern: "*.md"
      id_from: filename
      format: text
extraction:
  records: records/analyses
selection:
  targets:
    - name: patients_gt_controls
      inclusion: [Patients greater than controls]
"""

RECORD_TEXT = "# Extraction record\n\n## Analyses\n\n" + "A whole-brain analysis of patients versus controls. " * 80


def record(analyses):
    return json.dumps({"document_sha256": "x", "analyses": analyses})


@pytest.fixture
def review(tmp_path):
    rv = tmp_path / "review"
    (rv / "search").mkdir(parents=True)
    (rv / "review.yaml").write_text(REVIEW_YAML)
    recs = pubmed.parse_pubmed_xml((FIX / "pubmed_efetch.xml").read_bytes())
    (rv / "search" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    text, an = rv / "records" / "text", rv / "records" / "analyses"
    text.mkdir(parents=True)
    an.mkdir()
    for pmid in ("11111111", "33333333"):
        (text / f"{pmid}.md").write_text(RECORD_TEXT)
    # 33333333 has text but no analyses record: it must stay pending, never be emptied.
    (an / "11111111.analyses.json").write_text(record([
        {"key": "ana_pt_gt_hc", "name": "Patients > Controls", "description": "load effect", "table_id": "tbl2",
         "points": [{"coordinates": [-42.0, 18.0, 30.0], "space": None, "values": []},
                    {"coordinates": [4.0, -62.0, 44.0], "space": "TALAIRACH", "values": []}],
         "document": "- name: Patients > Controls\n- spatial scope: whole_brain\n- coordinate space: MNI"},
        {"key": "ana_roi", "name": "ROI only", "description": "", "table_id": None, "points": [],
         "document": "- spatial scope: roi"},
    ]))
    return rv


def run_ledger(*args):
    return ledger.main([str(a) for a in args])


def fake_output(batch, lines):
    Path(batch["output"]).write_text("".join(json.dumps(x) + "\n" for x in lines))


def test_records_stand_in_for_full_text_and_extraction(review, capsys):
    assert run_ledger("init", review) == 0
    assert run_ledger("batches", review, "--stage", "abstract", "--size", "10") == 0
    batch = json.loads((review / "work/abstract/batch_0001.json").read_text())
    fake_output(batch, [{"pmid": p, "decision": d, "criteria": {"I1": s}, "reason": "r"} for p, d, s in
                        (("11111111", "include", "met"), ("22222222", "exclude", "not_met"),
                         ("33333333", "include", "met"))])
    assert run_ledger("ingest", review, "--stage", "abstract", "--agent", "test") == 0

    # the record's text is the study's full text
    spec, records = gather.load_spec(review), gather.load_records(review)
    entries = gather.gather(review, ["11111111", "33333333"], gather.build_sources(spec, review, records))
    gather.write_index(review, entries)
    assert all(e["status"] == "available" and e["source"] == "records" for e in entries)

    assert run_ledger("batches", review, "--stage", "fulltext", "--size", "5") == 0
    batch = json.loads((review / "work/fulltext/batch_0001.json").read_text())
    fake_output(batch, [{"pmid": p, "decision": "include", "criteria": {"I1": "met"}, "reason": "r",
                         "evidence": [{"criterion": "I1", "quote": "A whole-brain analysis of patients versus controls."}]}
                        for p in ("11111111", "33333333")])
    assert run_ledger("ingest", review, "--stage", "fulltext", "--agent", "test") == 0
    capsys.readouterr()

    # extraction is an import, not a judged stage
    assert run_ledger("import-analyses", review, "--agent", "pondie") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["imported"] == 1 and report["analyses"] == 1 and report["points"] == 2
    assert report["no_record_pmids"] == ["33333333"]
    assert report["space_from_analysis"] == 1
    a = json.loads((review / "analyses/11111111.json").read_text())
    (x,) = a["analyses"]                                              # the pointless ROI analysis is left out
    assert x["analysis_id"] == "11111111-ana_pt_gt_hc" and "spatial scope: whole_brain" in x["description"]
    assert [p["space"] for p in x["points"]] == ["MNI", "TAL"]        # stated space, and TALAIRACH normalised
    assert {p["verification"] for p in x["points"]} == {"source"}
    rv = ledger.Review(review)
    assert rv.pending("extraction") == ["33333333"]                   # no record: pending, not empty
    assert {p["verification"] for p in rv.analyses("11111111")["analyses"][0]["points"]} == {"source"}

    assert run_ledger("batches", review, "--stage", "selection") == 0
    batch = json.loads((review / "work/selection/batch_0001.json").read_text())
    assert "coordinate space: MNI" in batch["items"][0]["analyses"][0]["description"]
    fake_output(batch, [{"pmid": "11111111", "analysis_id": "11111111-ana_pt_gt_hc", "target": "patients_gt_controls",
                         "include": True, "criteria": {"I1": "met"}, "reason": "r"}])
    assert run_ledger("ingest", review, "--stage", "selection", "--agent", "test") == 0
    capsys.readouterr()
    assert run_ledger("export", review, "--allow-pending") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["studies"] == 1 and out["dropped"].get("points_unverified", 0) == 0
    studyset = json.loads((review / "results/nimads/studyset.json").read_text())
    assert len(studyset["studies"][0]["analyses"][0]["points"]) == 2
    assert run_ledger("status", review) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["extraction"]["points_from_records"] == 2 and status["extraction"]["pending"] == 1


def test_a_records_folder_changes_the_extraction_hash_only(review):
    before = ledger.Review(review).criteria
    (review / "review.yaml").write_text(REVIEW_YAML.replace("  records: records/analyses\n", ""))
    after = ledger.Review(review).criteria
    assert before["extraction"]["hash"] != after["extraction"]["hash"]
    assert all(before[s]["hash"] == after[s]["hash"] for s in ("abstract", "fulltext", "selection"))


def test_combined_mode_screens_and_selects_in_one_pass(review, capsys):
    text = (review / "review.yaml").read_text().replace(
        "    inclusion: [Whole-brain coordinates reported]\n",
        "    inclusion: [Whole-brain coordinates reported]\n    select_analyses: true\n")
    (review / "review.yaml").write_text(text)
    an = review / "records" / "analyses"
    (an / "33333333.analyses.json").write_text(record([
        {"key": "a1", "name": "Controls > Patients", "description": "", "table_id": "t1",
         "points": [{"coordinates": [1.0, 2.0, 3.0], "space": "MNI", "values": []}], "document": "- x"}]))
    assert run_ledger("init", review) == 0
    rv = ledger.Review(review)
    assert rv.combined and rv.criteria["fulltext"]["hash"] == rv.criteria["selection"]["hash"]
    assert run_ledger("batches", review, "--stage", "abstract", "--size", "10") == 0
    batch = json.loads((review / "work/abstract/batch_0001.json").read_text())
    fake_output(batch, [{"pmid": p, "decision": d, "criteria": {"I1": s}, "reason": "r"} for p, d, s in
                        (("11111111", "include", "met"), ("22222222", "exclude", "not_met"),
                         ("33333333", "include", "met"))])
    assert run_ledger("ingest", review, "--stage", "abstract", "--agent", "test") == 0
    spec, records = gather.load_spec(review), gather.load_records(review)
    gather.write_index(review, gather.gather(review, ["11111111", "33333333"],
                                             gather.build_sources(spec, review, records)))
    assert ledger.Review(review).pending("fulltext") == []          # waits for the analyses
    assert run_ledger("import-analyses", review) == 0
    capsys.readouterr()
    assert ledger.Review(review).pending("fulltext") == ["11111111", "33333333"]

    assert run_ledger("batches", review, "--stage", "fulltext", "--size", "5") == 0
    batch = json.loads((review / "work/fulltext/batch_0001.json").read_text())
    assert batch["skill"] == "screen-and-select" and "patients_gt_controls" in batch["targets"]
    assert [a["analysis_id"] for a in batch["items"][0]["analyses"]] == ["11111111-ana_pt_gt_hc"]
    quote = [{"criterion": "I1", "quote": "A whole-brain analysis of patients versus controls."}]
    fake_output(batch, [
        {"pmid": "11111111", "decision": "include", "criteria": {"I1": "met"}, "reason": "r", "evidence": quote,
         "analyses": [{"analysis_id": "11111111-ana_pt_gt_hc", "target": "patients_gt_controls", "include": True,
                       "criteria": {"I1": "met"}, "reason": "patients > controls"}]},
        # screened in, but its only analysis fits no target: recorded as excluded
        {"pmid": "33333333", "decision": "include", "criteria": {"I1": "met"}, "reason": "r", "evidence": quote,
         "analyses": [{"analysis_id": "33333333-a1", "target": "patients_gt_controls", "include": False,
                       "criteria": {"I1": "not_met"}, "reason": "wrong direction"}]}])
    assert run_ledger("ingest", review, "--stage", "fulltext", "--agent", "test") == 0
    capsys.readouterr()
    rv = ledger.Review(review)
    ft = rv.valid_decisions("fulltext")
    assert ft[("11111111",)]["decision"] == "include"
    assert ft[("33333333",)]["decision"] == "exclude" and ft[("33333333",)]["no_eligible_analysis"]
    assert ft[("33333333",)]["judged_decision"] == "include"
    assert rv.pending("selection") == [] and len(rv.valid_decisions("selection")) == 2
    assert run_ledger("export", review) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["studies"] == 1
    assert run_ledger("status", review) == 0
    assert json.loads(capsys.readouterr().out)["fulltext_screening"]["excluded_no_eligible_analysis"] == 1


def test_combined_mode_rejects_an_included_study_missing_pairs(review, capsys):
    (review / "review.yaml").write_text((review / "review.yaml").read_text().replace(
        "    inclusion: [Whole-brain coordinates reported]\n",
        "    inclusion: [Whole-brain coordinates reported]\n    select_analyses: true\n"))
    assert run_ledger("init", review) == 0
    assert run_ledger("batches", review, "--stage", "abstract", "--size", "10") == 0
    batch = json.loads((review / "work/abstract/batch_0001.json").read_text())
    fake_output(batch, [{"pmid": p, "decision": "include" if p == "11111111" else "exclude",
                         "criteria": {"I1": "met" if p == "11111111" else "not_met"}, "reason": "r"}
                        for p in ("11111111", "22222222", "33333333")])
    assert run_ledger("ingest", review, "--stage", "abstract", "--agent", "test") == 0
    spec, records = gather.load_spec(review), gather.load_records(review)
    gather.write_index(review, gather.gather(review, ["11111111"], gather.build_sources(spec, review, records)))
    assert run_ledger("import-analyses", review) == 0
    assert run_ledger("batches", review, "--stage", "fulltext") == 0
    batch = json.loads((review / "work/fulltext/batch_0001.json").read_text())
    fake_output(batch, [{"pmid": "11111111", "decision": "include", "criteria": {"I1": "met"}, "reason": "r",
                         "evidence": [], "analyses": []}])
    assert run_ledger("ingest", review, "--stage", "fulltext", "--agent", "test") == 2   # pairs missing
    assert ledger.Review(review).pending("fulltext") == ["11111111"]

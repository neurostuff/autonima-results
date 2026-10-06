"""Extraction context access and one-expansion retry, without model calls."""
import importlib.util
import json
from pathlib import Path

SKILLS = Path(__file__).resolve().parents[1] / "skills"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SKILLS / "cbma-review/scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transport = load("single_response")
trim = load("trim")


def batch(tmp_path):
    root = tmp_path / "review"
    doc = root / "docs/123"
    doc.mkdir(parents=True)
    full = "# Study\n\n## Methods\n\nThe separate patient and control cohorts performed task A. Coordinates are MNI.\n\n## Results\n\nAdditional contrast details.\n\n## Discussion\n\n" + "Unneeded background. " * 100
    (doc / "text.md").write_text(full)
    context, _ = trim.write_coordinate_space_context(doc)
    table = doc / "T1.json"
    table.write_text(json.dumps({"table_id": "T1", "grid": [["x", "y", "z"]], "caption": "Peaks", "footer": ""}))
    work = root / "work/extraction"
    work.mkdir(parents=True)
    path = work / "batch_0001.json"
    data = {"stage": "extraction", "skill": "extract-coordinates", "criteria": {},
            "objective": "Synthetic transport fixture", "batch_id": "extraction-1",
            "input_view": "tables_space_context", "output": str(work / "output"),
            "_criteria_hash": "criteria1", "_input_hashes": {"123": "source1"},
            "items": [{"pmid": "123", "title": "Study", "must_report": ["T1"],
                       "coordinate_space_context_file": str(context),
                       "tables": [{"table_id": "T1", "file": str(table)}]}]}
    path.write_text(json.dumps(data))
    return path, data, full


def request(prepared, needed=True, pmid="123"):
    return transport.save(prepared, {"items": [], "unresolved": [
        {"pmid": pmid, "reason": "Need Results to distinguish contrast blocks", "context_needed": needed}]})


def test_methods_and_space_evidence_keep_source_text():
    text = "# Study\n\n## Methods\n\nTwo independent cohorts.\n\n## Results\n\nReported peaks were converted from MNI to Talairach.\n\n## References\n\nTalairach citation only.\n\n" + "Other material. " * 100
    context, info = trim.coordinate_space_context(text)
    assert "Two independent cohorts." in context
    assert "Reported peaks were converted from MNI to Talairach." in context
    assert "Talairach citation only." not in context
    assert all(e["line_start"] <= e["line_end"] for e in info["excerpts"])


def test_request_expands_once_across_prepare_restarts(tmp_path):
    path, data, full = batch(tmp_path)
    prepared = transport.prepare(path, SKILLS)
    assert full not in prepared["prompt"]
    assert "separate patient and control cohorts" in prepared["prompt"]
    recovered = request(prepared)
    assert recovered["pending_pmids"] == ["123"]
    expanded = transport.prepare(path, SKILLS)
    assert "Additional contrast details." in expanded["prompt"]
    assert len(expanded["expanded_context"]) == 1
    state = json.loads(transport.context_request_path(path, data, "123").read_text())
    assert state["expansion_used"] is True
    again = transport.prepare(path, SKILLS)
    assert "Additional contrast details." not in again["prompt"]
    assert again["expanded_context"] == []
    assert not prepared["output"].exists()


def test_changed_source_does_not_reuse_request(tmp_path):
    path, data, _ = batch(tmp_path)
    request(transport.prepare(path, SKILLS))
    data["_input_hashes"]["123"] = "source2"
    path.write_text(json.dumps(data))
    prepared = transport.prepare(path, SKILLS)
    assert prepared["expanded_context"] == []


def test_other_failures_and_invalid_ids_do_not_request_context(tmp_path):
    path, _, _ = batch(tmp_path)
    prepared = transport.prepare(path, SKILLS)
    request(prepared, needed=False)
    request(prepared, pmid="unknown")
    assert not (path.parent / "context_requests").exists()

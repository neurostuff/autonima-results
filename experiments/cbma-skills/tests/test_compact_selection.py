"""Compact selection replies: one record per analysis, global criteria once, expanded to pairs."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SKILLS = Path(__file__).resolve().parents[1] / "skills"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SKILLS / "cbma-review/scripts" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ledger = load("ledger")
transport = load("single_response")

TARGETS = {"pos": {"description": None, "criteria": {"I1": "Patients > controls"}},
           "neg": {"description": None, "criteria": {"I1": "Controls > patients", "E1": "Medicated"}}}
GLOBAL = {"GI1": "Whole-brain voxelwise", "GE1": "ROI only"}


def prepared(tmp_path, analyses=("a1", "a2")):
    batch = {"stage": "selection", "skill": "select-analyses", "criteria": GLOBAL, "selection_criteria": GLOBAL,
             "targets": TARGETS, "selection_format": "compact",
             "items": [{"pmid": "111", "analyses": [{"analysis_id": a} for a in analyses]}]}
    out = tmp_path / "batch_0001.out.jsonl"
    return {"batch": batch, "schema": transport.response_schema(batch), "ready": {"111"},
            "unreadable": [], "output": out, "path": tmp_path / "batch_0001.json"}


def failing(aid):
    return {"pmid": "111", "analysis_id": aid, "global": {"GI1": "not_met", "GE1": "not_met"},
            "global_reason": "ROI-restricted (GI1 not met).", "targets": []}


def passing(aid, pos=True):
    return {"pmid": "111", "analysis_id": aid, "global": {"GI1": "met", "GE1": "not_met"}, "global_reason": "",
            "targets": [{"target": "pos", "include": pos, "criteria": {"I1": "met" if pos else "not_met"},
                         "reason": "Patients > controls." if pos else "Wrong direction."},
                        {"target": "neg", "include": False, "criteria": {"I1": "not_met", "E1": "not_met"},
                         "reason": "Wrong direction."}]}


def test_schema_states_global_criteria_once_per_analysis():
    batch = prepared(Path("/tmp"))["batch"]
    record = transport.response_schema(batch)["properties"]["items"]["items"]
    assert set(record["required"]) == {"pmid", "analysis_id", "global", "global_reason", "targets"}
    assert set(record["properties"]["global"]["required"]) == set(GLOBAL)
    entry = record["properties"]["targets"]["items"]["anyOf"]
    # a target entry carries only that target's own criteria, never the globals
    assert {tuple(e["properties"]["criteria"]["required"]) for e in entry} == {("I1",), ("I1", "E1")}


def test_compact_reply_expands_to_pairs_the_ledger_accepts(tmp_path):
    prep = prepared(tmp_path)
    recovery = transport.save(prep, {"items": [failing("a1"), passing("a2")], "unresolved": []})
    assert recovery["studies_saved"] == ["111"] and not recovery["pending_pmids"]
    rows = [json.loads(line) for line in prep["output"].read_text().splitlines()]
    assert {(r["analysis_id"], r["target"]) for r in rows} == {("a1", "pos"), ("a1", "neg"), ("a2", "pos"), ("a2", "neg")}
    a1 = [r for r in rows if r["analysis_id"] == "a1"]
    assert all(r["include"] is False and r["criteria"] == {"GI1": "not_met", "GE1": "not_met"} for r in a1)
    assert all(r["reason"] == "ROI-restricted (GI1 not met)." for r in a1)
    a2pos = next(r for r in rows if r["analysis_id"] == "a2" and r["target"] == "pos")
    assert a2pos["include"] is True and a2pos["criteria"] == {"GI1": "met", "GE1": "not_met", "I1": "met"}
    # The ledger's own validator takes the expanded pairs unchanged.
    errs, recs = ledger.validate_selection([{k: v for k, v in r.items() if k != "pmid"} for r in rows],
                                           prep["batch"]["items"][0], TARGETS, list(GLOBAL))
    assert errs == [], errs
    assert sum(1 for r in recs if r.get("target_criteria_omitted")) == 2       # the two a1 rows


@pytest.mark.parametrize("bad", [
    lambda: dict(passing("a2"), targets=passing("a2")["targets"][:1]),                     # a target missing
    lambda: dict(passing("a2"), targets=passing("a2")["targets"] + passing("a2")["targets"][:1]),  # repeated
    lambda: dict(failing("a1"), global_reason=" "),                                        # failure, no reason
    lambda: {**passing("a2"), "global": {"GI1": "unclear", "GE1": "not_met"}, "targets": []},  # unclear is not a failure
])
def test_an_incomplete_compact_record_leaves_the_study_pending(tmp_path, bad):
    prep = prepared(tmp_path, analyses=("a1", "a2"))
    record = bad()
    others = [failing("a1")] if record["analysis_id"] == "a2" else [passing("a2")]
    recovery = transport.save(prep, {"items": others + [record], "unresolved": []})
    assert recovery["pending_pmids"] == ["111"] and not recovery["studies_saved"]
    assert not prep["output"].exists()


def test_a_missing_analysis_leaves_the_study_pending(tmp_path):
    prep = prepared(tmp_path)
    recovery = transport.save(prep, {"items": [failing("a1")], "unresolved": []})
    assert recovery["pending_pmids"] == ["111"]


def test_the_format_is_opt_in_per_workspace(tmp_path):
    review = tmp_path / "ws" / "review"
    review.mkdir(parents=True)
    assert ledger.selection_output_format(review) == "pairs"
    agents = tmp_path / "ws" / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "judge_output_formats.json").write_text('{"selection": "compact"}')
    assert ledger.selection_output_format(review) == "compact"
    (agents / "judge_output_formats.json").write_text('{"selection": "terse"}')
    with pytest.raises(ledger.LedgerError):
        ledger.selection_output_format(review)


def test_the_compact_instruction_only_reaches_compact_batches():
    assert "Compact selection format" in transport.COMPACT_SELECTION
    assert "Compact selection format" not in transport.ROLE


@pytest.mark.parametrize("fmt", ["pairs", "compact"])
def test_the_cli_schema_is_lenient_and_the_ledger_checks_criterion_ids(tmp_path, fmt):
    """The CLI schema has no per-target anyOf and no required "unresolved" (both cost Haiku
    correction turns). The transport checks only the shape; which criterion IDs each target
    needs is the ledger's check. So a clear global failure that leaves the target criteria out
    is kept (the strict schema used to reject it), while a passing analysis that leaves out one
    of its target's own criteria is rejected by the ledger."""
    prep = prepared(tmp_path)
    if fmt == "pairs":
        prep["batch"].pop("selection_format")
        prep["schema"] = transport.response_schema(prep["batch"])
    cli = transport.cli_schema(prep["batch"])
    assert cli["required"] == ["items"] and "anyOf" not in json.dumps(cli)
    rec = cli["properties"]["items"]["items"]
    entry = rec["properties"]["targets"]["items"] if fmt == "compact" else rec
    assert entry["properties"]["target"]["enum"] == ["pos", "neg"]

    def judged(neg_criteria):
        if fmt == "compact":
            bad = passing("a2")
            bad["targets"] = [bad["targets"][0], dict(bad["targets"][1], criteria=neg_criteria)]
            return [failing("a1"), bad]
        gfail = {"GI1": "not_met", "GE1": "not_met"}
        rows = [{"pmid": "111", "analysis_id": "a1", "target": t, "include": False, "criteria": dict(gfail),
                 "reason": "ROI only."} for t in ("pos", "neg")]                  # target criteria left out
        base = {"GI1": "met", "GE1": "not_met"}
        rows += [{"pmid": "111", "analysis_id": "a2", "target": "pos", "include": True,
                  "criteria": {**base, "I1": "met"}, "reason": "Patients > controls."},
                 {"pmid": "111", "analysis_id": "a2", "target": "neg", "include": False,
                  "criteria": {**base, **neg_criteria}, "reason": "Wrong direction."}]
        return rows

    def ledger_errors(rows):
        recs = [{k: v for k, v in r.items() if k != "pmid"} for r in rows]
        return ledger.validate_selection(recs, prep["batch"]["items"][0], TARGETS, list(GLOBAL))[0]

    good = transport.save(prep, {"items": judged({"I1": "not_met", "E1": "not_met"}), "unresolved": []})
    assert good["studies_saved"] == ["111"]
    rows = [json.loads(line) for line in prep["output"].read_text().splitlines()]
    assert ledger_errors(rows) == []                                   # the omitted a1 criteria are fine

    prep["output"].unlink()
    # neg's E1 left out with nothing clearly failed: not a permitted omission
    transport.save(prep, {"items": judged({"I1": "met"}), "unresolved": []})
    rows = [json.loads(line) for line in prep["output"].read_text().splitlines()]
    assert any("criteria missing ['E1']" in e for e in ledger_errors(rows))   # rejected, so it stays pending

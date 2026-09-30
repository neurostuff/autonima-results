"""benchmark/compare.py on the shapes real benchmark files take."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare", ROOT / "benchmark/compare.py")
compare = importlib.util.module_from_spec(spec)
sys.modules["compare"] = compare
spec.loader.exec_module(compare)


def studyset(studies):
    return {"studies": [{"id": sid, "pmid": pmid, "analyses": [{"points": [{"coordinates": xyz} for xyz in pts]}]}
                        for sid, pmid, pts in studies]}


def test_coordinates_tolerate_gold_studies_without_a_pmid(tmp_path):
    # NeuroMetaBench's merged studysets name a few studies instead of giving a PMID.
    review = tmp_path / "review"
    (review / "results" / "nimads").mkdir(parents=True)
    (review / "results" / "nimads" / "studyset.json").write_text(json.dumps(studyset([
        ("123", "123", [[-42, 18, 30], [10, 10, 10]]), ("456", "456", [[0, 0, 0]])])))
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(studyset([
        ("123", None, [[-43, 18, 30]]), ("Chen 2017", None, [[1, 2, 3]]), ("99", None, [[5, 5, 5]])])))
    out = compare.compare_coordinates(review, gold, 2.0)
    assert out["studies_compared"] == 1 and out["point_recall"] == 1.0 and out["point_precision"] == 0.5
    assert out["only_gold"] == ["99", "Chen 2017"] and out["only_ours"] == ["456"]


audit_spec = importlib.util.spec_from_file_location("audit_transcripts", ROOT / "benchmark/audit_transcripts.py")
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)


def _line(kind, content, **extra):
    return json.dumps({"type": kind, "message": {"content": content, **extra.pop("msg", {})}, **extra}) + "\n"


def _assistant(mid, tools, usage, effort="low"):
    content = [{"type": "tool_use", "name": n, "input": i} for n, i in tools]
    return _line("assistant", content, perTurnEffort=effort,
                 msg={"id": mid, "model": "m", "usage": usage})


def test_audit_roles_tokens_and_writes(tmp_path, capsys):
    review = tmp_path / "review"
    (review / "work" / "abstract").mkdir(parents=True)
    out = review / "work" / "abstract" / "batch_0001.out.jsonl"
    (review / "work" / "abstract" / "batch_0001.json").write_text(json.dumps({"output": str(out)}))
    tr = tmp_path / "transcripts"
    (tr / "s1" / "subagents").mkdir(parents=True)
    (tr / "s1.jsonl").write_text(_line("user", "go") + _assistant("o1", [("Bash", {"command": "python ledger.py status R"})],
                                                                  {"input_tokens": 1, "cache_read_input_tokens": 100}, "medium"))
    (tr / "s1" / "subagents" / "agent-r.jsonl").write_text(
        _line("user", "Run the `abstract` stage. REVIEW=x") +
        _assistant("r1", [("Bash", {"command": "python ledger.py batches R --stage abstract"})], {"input_tokens": 5}, "medium"))
    judge = _line("user", f"Your batch file is `{review}/work/abstract/batch_0001.json`. Follow the skill")
    # the same message logged twice (streaming): counted once, at its largest usage
    judge += _assistant("j1", [], {"input_tokens": 2, "output_tokens": 1})
    judge += _assistant("j1", [("Bash", {"command": f"cat > {out} <<'EOF'\n{{\"reason\": \"reappraise > look\"}}\nEOF"}),
                               ("Write", {"file_path": "/tmp/scratch/w.py", "content": "x"}),
                               ("Read", {"file_path": "/data/gold/gold.csv"})],
                         {"input_tokens": 2, "output_tokens": 40})
    (tr / "s1" / "subagents" / "agent-j.jsonl").write_text(judge)
    assert audit.main([str(tr), "--review", str(review), "--forbid", "/data/gold", "--out", str(tmp_path / "a.json")]) == 0
    r = json.loads((tmp_path / "a.json").read_text())
    roles = r["tokens"]["by_role"]
    assert set(roles) == {"orchestrator", "runner abstract", "judge abstract"}
    assert roles["judge abstract"]["output_tokens"] == 40 and roles["judge abstract"]["effort"] == {"low": 2}
    assert [w["target"] for w in r["judge_writes_outside_output"]] == ["/tmp/scratch/w.py"]   # not "look"
    assert [b["hit"] for b in r["blinding_hits"]] == ["/data/gold"]
    # a forbidden path matches at a path boundary only
    assert audit.main([str(tr), "--review", str(review), "--forbid", "/data/go", "--out", str(tmp_path / "b.json")]) == 0
    assert json.loads((tmp_path / "b.json").read_text())["blinding_hits"] == []
    assert r["ledger_calls"] == {"orchestrator:status": 1, "runner:batches": 1}

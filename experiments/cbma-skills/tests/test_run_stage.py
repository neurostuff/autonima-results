"""run_stage.py: the scripted judge loop, with a fake claude CLI."""

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
FIX = Path(__file__).parent / "fixtures"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pubmed = load("pubmed_search", SKILLS / "pubmed-search/scripts/pubmed_search.py")
ledger = load("ledger", SKILLS / "cbma-review/scripts/ledger.py")
run_stage = load("run_stage", SKILLS / "cbma-review/scripts/run_stage.py")

FAKE_CLAUDE = r'''
import json, re, sys
args = sys.argv[1:]
prompt = args[args.index("--") + 1]
agent = args[args.index("--agent") + 1]
open(sys.argv[0] + ".calls", "a").write(json.dumps({"agent": agent, "prompt": prompt, "argv": args[:args.index("--")]}) + "\n")
batch = json.load(open(re.search(r"Your batch file is `([^`]+)`", prompt).group(1)))
open(batch["output"], "w").write("".join(json.dumps({
    "pmid": it["pmid"], "decision": "uncertain", "reason": "fake",
    "criteria": {k: "unclear" for k in batch["criteria"]}}) + "\n" for it in batch["items"]))
print(json.dumps({"result": str(len(batch["items"])), "total_cost_usd": 0.01,
                  "usage": {"input_tokens": 3, "cache_read_input_tokens": 1000, "output_tokens": 50}}))
'''


def test_run_stage_dispatches_fixed_prompts_to_the_judge_agent(tmp_path, capsys, monkeypatch):
    ws = tmp_path / "ws"
    rv = ws / "review"
    (rv / "search").mkdir(parents=True)
    (rv / "review.yaml").write_text("""
objective: test
search: {query: q}
screening:
  abstract: {inclusion: [Human participants]}
  fulltext: {inclusion: [Whole brain]}
""")
    recs = pubmed.parse_pubmed_xml((FIX / "pubmed_efetch.xml").read_bytes())
    (rv / "search" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    shutil.copytree(ROOT / "agents", ws / ".claude" / "agents")
    fake = tmp_path / "fake_claude.py"
    fake.write_text(FAKE_CLAUDE)
    monkeypatch.chdir(ws)
    code = run_stage.main([str(rv), "--stage", "abstract", "--model", "m1", "--size", "2",
                           "--claude", f"{sys.executable} {fake}"])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["done"] and summary["pending"] == 0
    assert summary["accepted"] == 3 and summary["agent"] == "claude-code/m1/effort-low"
    assert summary["usage"]["judges"] == 2 and summary["usage"]["cache_read_input_tokens"] == 2000
    calls = [json.loads(line) for line in open(f"{fake}.calls")]
    assert {c["agent"] for c in calls} == {"cbma-abstract-screener"}
    # frontmatter does not reach a headless --agent session: effort and skill go as flags
    for c in calls:
        argv = c["argv"]
        assert argv[argv.index("--effort") + 1] == "low"
        assert argv[argv.index("--append-system-prompt-file") + 1].endswith("screen-studies/SKILL.md")
    assert all(run_stage.PROMPT.split("`{batch}`")[0] in c["prompt"] and "lenient" not in c["prompt"] for c in calls)
    decisions = [json.loads(line) for line in open(rv / "decisions" / "abstract.jsonl")]
    assert {d["agent"] for d in decisions} == {"claude-code/m1/effort-low"}
    assert not list((rv / "work" / "abstract").glob("batch_*.judge.json"))      # logs archived with batches
    assert len(list((rv / "work" / "abstract" / "done").glob("*.judge.json"))) == 2


def test_a_pilot_limit_holds_across_rounds(tmp_path, capsys, monkeypatch):
    ws = tmp_path / "ws"
    rv = ws / "review"
    (rv / "search").mkdir(parents=True)
    (rv / "review.yaml").write_text("objective: t\nsearch: {query: q}\nscreening:\n  abstract: {inclusion: [H]}\n"
                                    "  fulltext: {inclusion: [W]}\n")
    recs = pubmed.parse_pubmed_xml((FIX / "pubmed_efetch.xml").read_bytes())
    (rv / "search" / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    shutil.copytree(ROOT / "agents", ws / ".claude" / "agents")
    fake = tmp_path / "fake_claude.py"
    fake.write_text(FAKE_CLAUDE)
    monkeypatch.chdir(ws)
    run_stage.main([str(rv), "--stage", "abstract", "--model", "m1", "--size", "1", "--limit", "1",
                    "--claude", f"{sys.executable} {fake}"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["accepted"] == 1 and summary["pending"] == 2 and summary["done"]

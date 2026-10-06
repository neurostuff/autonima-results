"""run_stage.py: the scripted judge loop, with a fake claude CLI."""

import importlib.util
import json
import os
import shutil
import socket
import subprocess
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
import json, os, re, sys
# The single-response transport: the prompt arrives on stdin, tools are disabled, and the
# judge's whole answer is one JSON object in the CLI result's "result" field.
args = sys.argv[1:]
prompt = sys.stdin.read()
open(sys.argv[0] + ".calls", "a").write(json.dumps({"prompt": prompt, "argv": args}) + "\n")
batch = json.load(open(re.search(r"^CBMA_SINGLE_RESPONSE_V\d batch=(.+)$", prompt, re.M).group(1)))
items = json.loads(prompt.split("\n")[1])["items"]
reply = json.dumps({"items": [{"pmid": it["pmid"], "decision": "uncertain", "reason": "fake",
                               "criteria": {k: "unclear" for k in batch["criteria"]}} for it in items],
                    "unresolved": []})
result = {"result": reply, "num_turns": 1, "is_error": False, "total_cost_usd": 0.01,
          "usage": {"input_tokens": 3, "cache_read_input_tokens": 1000, "output_tokens": 50}}
if os.environ.get("FAKE_FENCE") == "1":
    result["result"] = "```json\n" + reply + "\n```"
if os.environ.get("FAKE_STRUCTURED") == "1":
    # As the CLI does with --json-schema: the schema-valid answer in structured_output, two
    # turns, and free text that would not parse on its own.
    schema = json.loads(args[args.index("--json-schema") + 1])
    assert schema["required"] == ["items"]           # the lenient CLI schema: "unresolved" optional
    result.update(structured_output=json.loads(reply), num_turns=3,
                  result="```json\n" + reply + "\n```\n\n**3 lines written.**")
print(json.dumps(result))
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
    assert summary["accepted"] == 3 and summary["agent"] == "claude-code/m1/effort-low/single-response-v2"
    assert summary["usage"]["judges"] == 2 and summary["usage"]["cache_read_input_tokens"] == 2000
    assert summary["usage"]["attempts_with_unknown_usage"] == 0
    calls = [json.loads(line) for line in open(f"{fake}.calls")]
    for c in calls:
        argv = c["argv"]
        # One tool-free turn with the judge's effort as a flag (agent frontmatter does not reach -p).
        assert argv[argv.index("--effort") + 1] == "low"
        assert argv[argv.index("--tools") + 1] == "" and argv[argv.index("--max-turns") + 1] == "4"
        assert "--json-schema" in argv                              # the CLI enforces the reply shape
        system = Path(argv[argv.index("--system-prompt-file") + 1]).read_text()
        assert "## Scientific skill: screen-studies" in system      # the stage skill is in the system prompt
        # The prompt is the fixed envelope plus inline items: no added guidance.
        assert c["prompt"].startswith("Judge these supplied items using the registered configuration")
        assert "CBMA_SINGLE_RESPONSE_V2 batch=" in c["prompt"] and "lenient" not in c["prompt"]
    decisions = [json.loads(line) for line in open(rv / "decisions" / "abstract.jsonl")]
    assert {d["agent"] for d in decisions} == {"claude-code/m1/effort-low/single-response-v2"}
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
    # pending counts the pilot's own scope; the rest of the stage is reported separately and
    # is never batched by a pilot's retries.
    assert summary["accepted"] == 1 and summary["pending"] == 0 and summary["done"]
    assert summary["total_stage_pending"] == 2


def workspace(tmp_path):
    """A workspace with three abstract records pending and a fake claude CLI."""
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
    return ws, rv, fake


def test_a_leftover_judge_log_is_not_mistaken_for_a_batch(tmp_path, capsys, monkeypatch):
    """A judge writes its log before its output. A process that dies in between leaves
    batch_NNNN.judge.json beside a batch_NNNN.json whose output is still missing, so the log
    survives ingest's cleanup. A bare batch_*.json glob matched the log, and the next call tried
    to judge it: KeyError: 'skill' (executive_function)."""
    ws, rv, fake = workspace(tmp_path)
    ledger.cmd_batches(ledger.Review(rv), "abstract", 2, None, discard=False)
    work = rv / "work" / "abstract"
    batch = sorted(work.glob("batch_*.json"))[0]                     # no log beside it yet
    log = batch.with_suffix(".judge.json")                           # its own log, output not yet written
    log.write_text(json.dumps({"usage": {"input_tokens": 7}, "total_cost_usd": 0.5}))
    monkeypatch.chdir(ws)
    code = run_stage.main([str(rv), "--stage", "abstract", "--model", "m1", "--size", "2",
                           "--claude", f"{sys.executable} {fake}"])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["done"] and summary["accepted"] == 3
    judged = [json.loads(line)["prompt"] for line in open(f"{fake}.calls")]
    assert not any(".judge.json" in prompt for prompt in judged)     # no judge was sent a log to judge
    assert len(list((work / "done").glob("*.judge.json"))) >= 1      # the log is kept, with its usage


def test_rebatching_archives_a_leftover_judge_log_rather_than_deleting_it(tmp_path):
    """`batches` wipes the work folder with a batch_* glob, which deleted the judges' logs and
    the token usage they record. They belong in done/ with the rest of the run's accounting."""
    ws, rv, _ = workspace(tmp_path)
    ledger.cmd_batches(ledger.Review(rv), "abstract", 2, None, discard=False)
    work = rv / "work" / "abstract"
    log = work / "batch_0000.judge.json"
    log.write_text(json.dumps({"usage": {"input_tokens": 7}, "total_cost_usd": 0.5}))
    ledger.cmd_batches(ledger.Review(rv), "abstract", 2, None, discard=True)
    assert not log.exists()
    archived = list((work / "done").glob("*batch_0000.judge.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["total_cost_usd"] == 0.5


def test_a_second_process_will_not_work_a_locked_stage(tmp_path, capsys, monkeypatch):
    """A runner stopped by a usage limit leaves its run_stage.py running. A second call on the
    same stage must refuse rather than judge the same records alongside it."""
    ws, rv, fake = workspace(tmp_path)
    work = rv / "work" / "abstract"
    work.mkdir(parents=True)
    (work / "run_stage.lock").write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                                                     "stage": "abstract", "started": "now"}))
    monkeypatch.chdir(ws)
    code = run_stage.main([str(rv), "--stage", "abstract", "--model", "m1", "--size", "2",
                           "--claude", f"{sys.executable} {fake}"])
    out = json.loads(capsys.readouterr().out)
    assert code == 4 and out["stage_busy"]["pid"] == os.getpid()
    assert not (rv / "decisions" / "abstract.jsonl").exists()        # nothing was judged
    assert not os.path.exists(f"{fake}.calls")
    assert (work / "run_stage.lock").exists()                        # the holder's lock is untouched


def test_a_lock_left_by_a_dead_process_is_taken_over(tmp_path, capsys, monkeypatch):
    ws, rv, fake = workspace(tmp_path)
    work = rv / "work" / "abstract"
    work.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()                                                      # a pid that is gone and reaped
    (work / "run_stage.lock").write_text(json.dumps({"host": socket.gethostname(), "pid": dead.pid,
                                                     "stage": "abstract", "started": "earlier"}))
    monkeypatch.chdir(ws)
    code = run_stage.main([str(rv), "--stage", "abstract", "--model", "m1", "--size", "2",
                           "--claude", f"{sys.executable} {fake}"])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["accepted"] == 3
    assert summary["took_over_lock_from"]["pid"] == dead.pid         # and says so, for the run notes
    assert not (work / "run_stage.lock").exists()                    # released on the way out


def test_the_lock_is_claimed_atomically(tmp_path):
    """Two processes starting at the same moment must not both claim the stage, so the claim is
    an exclusive create rather than a check followed by a write."""
    work = tmp_path / "work" / "abstract"
    path, stale = run_stage.take_lock(work, "abstract")
    assert stale is None and json.loads(path.read_text())["pid"] == os.getpid()
    try:
        run_stage.take_lock(work, "abstract")                        # a second claim, lock still held
    except run_stage.StageBusy as exc:
        assert exc.args[0]["pid"] == os.getpid()
    else:
        raise AssertionError("the second claim should have been refused")


def test_a_reply_wrapped_in_a_json_fence_is_accepted(tmp_path, capsys, monkeypatch):
    """Haiku wraps its single-response JSON in a ```json fence despite the instruction not to.
    Every one of the 156 failed attempts in the Haiku cue_reactivity and problem_solving
    selection runs was such a reply, complete but rejected at character 0 and judged again."""
    ws, rv, fake = workspace(tmp_path)
    monkeypatch.chdir(ws)
    monkeypatch.setenv("FAKE_FENCE", "1")
    code = run_stage.main([str(rv), "--stage", "abstract", "--model", "m1", "--size", "2",
                           "--claude", f"{sys.executable} {fake}"])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["done"] and summary["accepted"] == 3
    assert summary["usage"]["attempts"] == 2 and not summary["judge_failures"]    # no re-judging


def test_the_cli_structured_output_is_used_and_the_schema_is_passed(tmp_path, capsys, monkeypatch):
    """With --json-schema the CLI returns the schema-valid answer in structured_output, after
    one internal tool call (two turns). Haiku's free text around it (a fence, then "3 lines
    written") must not matter."""
    ws, rv, fake = workspace(tmp_path)
    monkeypatch.chdir(ws)
    monkeypatch.setenv("FAKE_STRUCTURED", "1")
    code = run_stage.main([str(rv), "--stage", "abstract", "--model", "m1", "--size", "2",
                           "--claude", f"{sys.executable} {fake}"])
    summary = json.loads(capsys.readouterr().out)
    assert code == 0 and summary["done"] and summary["accepted"] == 3 and not summary["judge_failures"]
    argv = json.loads(open(f"{fake}.calls").readline())["argv"]
    assert argv[argv.index("--max-turns") + 1] == "4" and argv[argv.index("--tools") + 1] == ""

"""Run one judged stage mechanically: batches, one headless judge per batch, ingest, retries.

    python run_stage.py REVIEW --stage fulltext --model claude-opus-5-5 \\
        [--size 2] [--limit N] [--parallel 8] [--time-budget 540] [--add-dir PACKAGE]

Run it from the workspace: the folder whose `.claude/agents/` holds the judge
definitions. Each batch is judged by a fresh `claude -p --agent <judge>` session, with
the one dispatch prompt below and nothing else. The judge's effort, tools and preloaded
skill come from its agent definition, so no coordinating model sits between the ledger
and the judges, and no one can add guidance to a prompt.

The script is resumable, and it stops at --time-budget seconds. It stops launching
judges well before then and waits for those running, so a caller whose commands time
out (an agent's Bash tool) runs it again until it reports done. Each call:
  1. ingests outputs already written;
  2. if no batch is waiting, batches whatever is pending (at most --limit in the first
     round);
  3. dispatches judges for batches without output, up to --parallel at once;
  4. ingests;
  5. repeats for retries, up to --rounds rounds per call, while time remains.

It prints one JSON summary: accepted, rejected, pending, judge failures, and the
judges' token usage from their `--output-format json` logs (saved beside each batch as
`.judge.json`).

Exit codes: 0 when nothing the stage can batch is pending; 3 when work remains; 1 on a
ledger error.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402

JUDGE = {"abstract": "cbma-abstract-screener", "fulltext": "cbma-fulltext-screener",
         "extraction": "cbma-extractor", "selection": "cbma-selector"}
SIZE = {"abstract": 25, "fulltext": 2, "extraction": 1, "selection": 1}
# The skill goes into the judge's system prompt (--append-system-prompt-file), where it is
# cached and costs no turn. Agent frontmatter does not reach a headless --agent session:
# its effort and preloaded skills are ignored there, so both are passed as flags.
PROMPT = ("Your batch file is `{batch}`. Follow the `{skill}` skill, which is already in your "
          "system prompt (source: `{skills}/{skill}/SKILL.md`). Write your output to the path in "
          "the batch's `output` field. Reply with only the number of items you wrote.")
USAGE = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def agent_effort(agents_dir: Path, judge: str) -> str:
    m = re.search(r"^effort:\s*(\S+)", (agents_dir / f"{judge}.md").read_text(), re.M)
    return m.group(1) if m else "default"


def judge_one(batch: Path, stage: str, args, skills: Path) -> dict:
    data = json.loads(batch.read_text())
    prompt = PROMPT.format(batch=batch, skills=skills, skill=data["skill"])
    effort = agent_effort(args.agents_dir, JUDGE[stage])
    cmd = shlex.split(args.claude) + ["-p", "--agent", JUDGE[stage], "--model", args.model,
                                      "--output-format", "json", "--allowedTools", "Read,Write",
                                      "--append-system-prompt-file", str(skills / data["skill"] / "SKILL.md")]
    if effort != "default":
        cmd += ["--effort", effort]
    for d in args.add_dir:
        cmd += ["--add-dir", d]
    cmd += ["--", prompt]
    log = batch.with_suffix(".judge.json")
    started = time.time()
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=args.judge_timeout, stdin=subprocess.DEVNULL)
        log.write_text(run.stdout or json.dumps({"stderr": run.stderr[-4000:]}))
        ok = run.returncode == 0
        err = None if ok else (run.stderr or run.stdout)[-500:]
    except subprocess.TimeoutExpired:
        ok, err = False, f"timed out after {args.judge_timeout}s"
    return {"batch": batch.name, "ok": ok, "error": err, "seconds": round(time.time() - started)}


def usage_of(work: Path) -> dict:
    total = {k: 0 for k in USAGE}
    total["judges"] = total["cost_usd"] = 0
    for log in list(work.glob("batch_*.judge.json")) + list((work / "done").glob("*.judge.json")):
        try:
            r = json.loads(log.read_text())
        except json.JSONDecodeError:
            continue
        u = r.get("usage") or {}
        if not u:
            continue
        total["judges"] += 1
        total["cost_usd"] += r.get("total_cost_usd") or 0
        for k in USAGE:
            total[k] += int(u.get(k) or 0)
    total["cost_usd"] = round(total["cost_usd"], 2)
    return total


def ingest(review: Path, stage: str, agent: str, summary: dict) -> None:
    rv = ledger.Review(review)
    work = rv.work_dir(stage)
    # Keep each judge's log with its batch when the ledger archives the batch.
    logs = {p.name.replace(".judge.json", ""): p for p in work.glob("batch_*.judge.json")}
    report = ledger.cmd_ingest(rv, stage, agent)
    done = work / "done"
    for stem, log in logs.items():
        if not (work / f"{stem}.json").exists() and log.exists():
            done.mkdir(parents=True, exist_ok=True)
            log.rename(done / f"{time.strftime('%Y%m%dT%H%M%S')}_{log.name}")
    summary["accepted"] += report["accepted"]
    summary["rejected"] += report["rejected"]
    summary["errors"] += report["errors"][:20]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review_dir", type=Path)
    ap.add_argument("--stage", choices=list(JUDGE), required=True)
    ap.add_argument("--model", required=True, help="the judges' model id, recorded with every decision")
    ap.add_argument("--size", type=int)
    ap.add_argument("--limit", type=int, help="batch at most this many items in the first round (pilots)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3, help="dispatch rounds per call: the first, plus retries")
    ap.add_argument("--time-budget", type=int, default=540, help="seconds before this call returns")
    ap.add_argument("--judge-timeout", type=int, default=900)
    ap.add_argument("--agents-dir", type=Path, default=Path(".claude/agents"))
    ap.add_argument("--add-dir", action="append", default=[], help="passed to each judge (e.g. the package)")
    ap.add_argument("--claude", default="claude", help="the claude CLI command (tests pass a fake)")
    ap.add_argument("--harness", default="claude-code")
    args = ap.parse_args(argv)

    review = args.review_dir.resolve()
    skills = Path(ledger.SKILLS_ROOT)
    judge = JUDGE[args.stage]
    effort = agent_effort(args.agents_dir, judge)
    agent = f"{args.harness}/{args.model}/effort-{effort}"
    deadline = time.time() + args.time_budget
    summary = {"stage": args.stage, "judge": judge, "effort": effort, "agent": agent, "rounds": 0,
               "batches_dispatched": 0, "accepted": 0, "rejected": 0, "judge_failures": [], "errors": []}
    try:
        rv = ledger.Review(review)
        work = rv.work_dir(args.stage)
        ingest(review, args.stage, agent, summary)          # outputs left by an earlier call
        first = not any((work / "done").glob("*_batch_*.json")) if (work / "done").exists() else True
        while summary["rounds"] < args.rounds and time.time() < deadline - 60:
            waiting = sorted(p for p in work.glob("batch_*.json")) if work.exists() else []
            if not waiting:
                if not ledger.Review(review).pending(args.stage):
                    break
                if args.limit and not first:
                    break                                   # a pilot: retry its items, batch no new ones
                ledger.cmd_batches(ledger.Review(review), args.stage, args.size or SIZE[args.stage],
                                   args.limit if first else None, discard=False)
                waiting = sorted(work.glob("batch_*.json"))
            first = False
            todo = [b for b in waiting if not ledger._output_path(b, args.stage).exists()]
            summary["rounds"] += 1
            with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
                futures = {}
                for b in todo:
                    # Leave room for a judge to finish: never launch within the last few minutes.
                    if time.time() > deadline - min(args.judge_timeout, 300):
                        break
                    futures[pool.submit(judge_one, b, args.stage, args, skills)] = b
                for fut in cf.as_completed(futures):
                    res = fut.result()
                    summary["batches_dispatched"] += 1
                    if not res["ok"]:
                        summary["judge_failures"].append(res)
            ingest(review, args.stage, agent, summary)
            if len(futures) < len(todo):
                break                                       # out of time; the next call resumes
        rv = ledger.Review(review)
        pending = rv.pending(args.stage)
        waiting = sorted(work.glob("batch_*.json")) if work.exists() else []
        batched_pending = sorted({it["pmid"] for b in waiting for it in json.loads(b.read_text())["items"]})
        summary.update(pending=len(pending), waiting_batches=len(waiting), batched_but_pending=batched_pending[:50],
                       usage=usage_of(work))
    except ledger.LedgerError as exc:
        print(json.dumps({"ledger_error": str(exc)}))
        return 1
    limited = any(re.search(r"limit|429", f.get("error") or "", re.I) for f in summary["judge_failures"])
    summary["usage_limited"] = limited
    summary["done"] = not waiting and (not pending or bool(args.limit))
    print(json.dumps(summary, indent=1))
    return 0 if summary["done"] else 3


if __name__ == "__main__":
    sys.exit(main())

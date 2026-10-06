"""Run one judged stage mechanically: batches, one headless judge per batch, ingest, retries.

    python run_stage.py REVIEW --stage fulltext --model claude-opus-5-5 \\
        [--size 2] [--limit N] [--parallel 8] [--time-budget 540] [--add-dir PACKAGE]

Run it from the workspace: the folder whose `.claude/agents/` holds the judge
definitions. Each batch is judged by a fresh tool-free `claude -p` session. Python supplies the
scientific skills and complete stage inputs; the model returns one JSON response.
Python saves existing output formats for unchanged ledger validation. Effort and
thinking budgets retain their workspace configuration.

The script is resumable, and it stops at --time-budget seconds. It stops launching
judges well before then and waits for those running, so a caller whose commands time
out (an agent's Bash tool) runs it again until it reports done. Each call:
  1. ingests outputs already written;
  2. if no batch is waiting, batches whatever is pending (at most --limit in the first
     round);
  3. dispatches only available slots, with timeouts capped to the remaining budget;
  4. ingests;
  5. repeats while time remains; --max-attempts persists across calls for unchanged inputs.

It prints one JSON summary: accepted, rejected, pending, judge failures, and the
judges' cumulative reported usage from immutable per-attempt records. Missing CLI
usage is explicitly counted as unknown; native `.judge.json` logs remain compatible.

Workspaces can opt into model-specific fixed thinking budgets with
`.claude/agents/judge_thinking_budgets.json` (a `model` ID and `stages` mapping).
The configured budget sets MAX_THINKING_TOKENS only for judge subprocesses and is
recorded in their result logs, the stage summary, and the ledger agent string.

Only one live process may work a stage: the call takes a lock in the stage's work folder
and exits 4 if another process holds it, so a runner stopped by a usage limit (whose script
keeps running) cannot be joined by a second runner on the same stage.

Exit codes: 0 when nothing the stage can batch is pending; 3 when work remains; 1 on a
ledger error; 4 when another live process holds the stage.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402
import single_response  # noqa: E402
import stage_bookkeeping as bookkeeping  # noqa: E402

JUDGE = {"abstract": "cbma-abstract-screener", "fulltext": "cbma-fulltext-screener",
         "extraction": "cbma-extractor", "selection": "cbma-selector"}
SIZE = {"abstract": 25, "fulltext": 2, "extraction": 1, "selection": 1}
USAGE = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def judge_thinking_budget(agents_dir: Path, model: str, stage: str) -> Optional[int]:
    """Read an opt-in, workspace-local fixed thinking budget for this judge."""
    path = agents_dir / "judge_thinking_budgets.json"
    if not path.exists():
        return None
    config = json.loads(path.read_text())
    configured_model = config["model"]
    if model != configured_model and not model.startswith(configured_model + "-"):
        return None
    budget = config["stages"][stage]
    if type(budget) is not int or budget < 1024:
        raise ValueError(f"{path}: {stage} thinking budget must be an integer >= 1024")
    return budget


def agent_effort(agents_dir: Path, judge: str) -> str:
    m = re.search(r"^effort:\s*(\S+)", (agents_dir / f"{judge}.md").read_text(), re.M)
    return m.group(1) if m else "default"


def judge_one(batch: Path, stage: str, args, skills: Path, timeout=None) -> dict:
    started = time.monotonic()
    effort = agent_effort(args.agents_dir, JUDGE[stage])
    directory = single_response.new_attempt(batch, 'claude', args.model, effort)
    recovery = None
    try:
        prepared = single_response.prepare(batch, skills)
        recovery = single_response.claude(prepared, shlex.split(args.claude), args.model,
            effort, timeout if timeout is not None else args.judge_timeout,
            getattr(args, "max_thinking_tokens", None), directory=directory)
        ok, err = True, None
    except subprocess.TimeoutExpired:
        ok, err = False, f"timed out after {timeout or args.judge_timeout}s"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ok, err = False, str(exc)[-500:]
    # Includes preparation failures, which occur before the CLI adapter starts.
    record = json.loads((directory / 'attempt.json').read_text())
    if record.get('status') == 'started':
        single_response.finish_attempt(directory, status='failed', error=err)
    return {"batch": batch.name, "ok": ok, "error": err, "attempt_id": directory.name,
            "audit": str(directory), "recovery": recovery, "provider_blocked": bookkeeping.provider_blocked(err),
            "judge_path": single_response.VERSION, "seconds": round(time.monotonic() - started)}


def usage_of(work: Path) -> dict:
    return single_response.usage_of(work, 'claude')


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


class StageBusy(Exception):
    """Another live process holds this stage's lock."""


def alive(held: dict) -> bool:
    """Is the process in a lock record still running? A lock from another host cannot be
    checked, so it counts as live: refusing a call is cheap, judging twice is not."""
    if held.get("host") and held["host"] != socket.gethostname():
        return True
    pid = held.get("pid")
    if not isinstance(pid, int):
        return False                                    # an unreadable lock holds nothing
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                                     # alive, owned by someone else
    return True


def take_lock(work: Path, stage: str) -> Tuple[Path, Optional[dict]]:
    """Claim the stage for this process. Returns the lock path and, when a dead process left
    a lock behind, the record it left, so the caller can report the interrupted run.

    A runner stopped by a usage limit leaves its run_stage.py running in the background. A
    second runner then judged the same stage at the same time (executive_function). The lock
    names the host and pid, so a stale lock from a crashed run is taken over, while a live
    one refuses the call.
    """
    work.mkdir(parents=True, exist_ok=True)
    path = work / "run_stage.lock"
    mine = json.dumps({"host": socket.gethostname(), "pid": os.getpid(), "stage": stage,
                       "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1)
    stale = None
    for attempt in range(2):
        try:
            # O_EXCL, so two processes starting at once cannot both claim the stage.
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                held = {}
            if alive(held) or attempt:
                raise StageBusy(held or {"unreadable": True})
            stale = held or {"unreadable": True}
            path.unlink(missing_ok=True)                    # its process is gone; take it over
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(mine)
        return path, stale
    raise StageBusy(stale)                                  # unreachable: the loop returns or raises


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review_dir", type=Path)
    ap.add_argument("--stage", choices=list(JUDGE), required=True)
    ap.add_argument("--model", required=True, help="the judges' model id, recorded with every decision")
    ap.add_argument("--size", type=int)
    ap.add_argument("--limit", type=int, help="batch at most this many items in the first round (pilots)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3, help="dispatch rounds per call: the first, plus retries")
    ap.add_argument("--max-attempts", type=int, default=3, help="persistent attempts per unchanged item")
    ap.add_argument("--time-budget", type=int, default=540, help="seconds before this call returns")
    ap.add_argument("--judge-timeout", type=int, default=900)
    ap.add_argument("--agents-dir", type=Path, default=Path(".claude/agents"))
    ap.add_argument("--add-dir", action="append", default=[], help="passed to each judge (e.g. the package)")
    ap.add_argument("--claude", default="claude", help="the claude CLI command (tests pass a fake)")
    ap.add_argument("--harness", default="claude-code")
    args = ap.parse_args(argv)
    if any(n <= 0 for n in (args.size or SIZE[args.stage], args.parallel, args.rounds,
                            args.max_attempts, args.time_budget, args.judge_timeout)) or (
                            args.limit is not None and args.limit <= 0):
        ap.error("numeric controls must be positive")
    try:
        args.max_thinking_tokens = judge_thinking_budget(args.agents_dir, args.model, args.stage)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ap.error(f"invalid judge thinking-budget configuration: {exc}")

    review = args.review_dir.resolve()
    skills = Path(ledger.SKILLS_ROOT)
    judge = JUDGE[args.stage]
    effort = agent_effort(args.agents_dir, judge)
    agent = f"{args.harness}/{args.model}/effort-{effort}"
    if args.max_thinking_tokens is not None:
        agent += f"/thinking-budget-{args.max_thinking_tokens}"
    agent += "/" + single_response.VERSION
    deadline = time.monotonic() + args.time_budget
    summary = {"stage": args.stage, "judge": judge, "effort": effort, "agent": agent, "rounds": 0,
               "batches_dispatched": 0, "accepted": 0, "rejected": 0, "judge_failures": [], "errors": []}
    if args.max_thinking_tokens is not None:
        summary["configured_max_thinking_tokens"] = args.max_thinking_tokens
    lock = None
    try:
        rv = ledger.Review(review)
        work = rv.work_dir(args.stage)
        lock, stale = take_lock(work, args.stage)
        if stale:
            # An earlier call died holding the stage. Its batches, if any, are picked up below.
            summary["took_over_lock_from"] = stale
        state_dir = work / 'claude_runner'
        state_dir.mkdir(exist_ok=True)
        state_path = state_dir / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'attempts': {}, 'pilots': {}}
        attempts = state['attempts']
        scope = None
        if args.limit:
            key = rv.criteria[args.stage]['hash'] + ':' + str(args.limit)
            existing = {i['pmid'] for b in ledger.batch_files(work) for i in json.loads(b.read_text())['items']}
            # Adopt legacy pilot history before ingestion shrinks its batches.
            # A completed legacy pilot must never silently start the next N records.
            historical = []
            for archived in sorted((work / 'done').glob('*_batch_*.json')):
                if not re.fullmatch(r'.*_batch_\d+\.json', archived.name):
                    continue
                old = json.loads(archived.read_text())
                if old.get('_criteria_hash') == rv.criteria[args.stage]['hash']:
                    for item in old.get('items', []):
                        if item['pmid'] not in historical:
                            historical.append(item['pmid'])
                        if len(historical) >= args.limit:
                            break
                if len(historical) >= args.limit:
                    break
            initial = historical or (sorted(existing) if existing else rv.pending(args.stage)[:args.limit])
            if historical and len(initial) < args.limit:
                initial += [p for p in sorted(existing) if p not in initial]
            if len(initial) > args.limit:
                raise ledger.LedgerError('existing batches exceed pilot scope')
            scope = state['pilots'].setdefault(key, initial)
            if existing - set(scope):
                raise ledger.LedgerError('existing batches exceed saved pilot scope')
            bookkeeping.write_state(state_path, state)

        def pending():
            values = ledger.Review(review).pending(args.stage)
            return [p for p in values if scope is None or p in scope]

        ingest(review, args.stage, agent, summary)
        blocked = False
        exhausted = set()
        while pending() and summary['rounds'] < args.rounds and time.monotonic() < deadline - 5 and not blocked:
            waiting = ledger.batch_files(work)
            if not waiting:
                scoped = ledger.Review(review)
                if scope is not None:
                    original_pending = scoped.pending
                    scoped.pending = lambda stage: [p for p in original_pending(stage)
                                                    if stage != args.stage or p in scope]
                ledger.cmd_batches(scoped, args.stage, args.size or SIZE[args.stage], None, discard=False)
                waiting = ledger.batch_files(work)
            todo, current_exhausted = bookkeeping.eligible_batches(
                waiting, pending(), attempts, args.max_attempts, state_dir)
            exhausted.update(current_exhausted)
            if not todo:
                break
            summary['rounds'] += 1
            with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
                running = {}
                queue = iter(todo)
                while True:
                    while len(running) < args.parallel and time.monotonic() < deadline - 5 and not blocked:
                        job = next(queue, None)
                        if job is None:
                            break
                        batch, keys = job
                        timeout = min(args.judge_timeout, max(0.1, deadline - time.monotonic() - 5))
                        for key in keys:
                            attempts[key] = attempts.get(key, 0) + 1
                        bookkeeping.write_state(state_path, state)
                        running[pool.submit(judge_one, batch, args.stage, args, skills, timeout)] = batch
                    if not running:
                        break
                    done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                    for future in done:
                        running.pop(future)
                        row = future.result()
                        summary['batches_dispatched'] += 1
                        if not row['ok']:
                            summary['judge_failures'].append(row)
                        blocked = blocked or row['provider_blocked']
            ingest(review, args.stage, agent, summary)
        scoped_pending = pending()
        waiting = ledger.batch_files(work)
        summary.update(pending=len(scoped_pending), total_stage_pending=len(ledger.Review(review).pending(args.stage)),
                       waiting_batches=len(waiting), retry_exhausted=sorted(exhausted),
                       max_attempts=args.max_attempts, provider_blocked=blocked, usage_limited=blocked,
                       batched_but_pending=sorted({it['pmid'] for b in waiting
                           for it in json.loads(b.read_text())['items']})[:50], usage=usage_of(work),
                       done=not scoped_pending, logs=str(state_dir))
        bookkeeping.write_state(state_dir / 'last_summary.json', summary)

    except StageBusy as exc:
        held = exc.args[0]
        print(json.dumps({"stage_busy": held, "stage": args.stage,
                          "hint": "another run_stage.py process holds this stage; wait for it or "
                                  "stop that process before running the stage again"}, indent=1))
        return 4
    except ledger.LedgerError as exc:
        print(json.dumps({"ledger_error": str(exc)}))
        return 1
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)

    print(json.dumps(summary, indent=1))
    return 0 if summary["done"] else 3


if __name__ == "__main__":
    sys.exit(main())

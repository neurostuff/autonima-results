#!/usr/bin/env python3
"""Codex stage bookkeeping: fixed judges, ledger ingestion, bounded retries, summaries.

Install beside run_judge_optimized.sh in an arm root. Run with the project's venv:
  .venv/bin/python ../../run_stage.py PROJECT --stage fulltext
Exit 0: judgments finished (scientific audit still required); 3: work remains;
4: another worker owns the stage/project; 1: execution/ledger error.
No scientific criteria or model settings are changed by this runner.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

SIZES = {'abstract': 25, 'fulltext': 2, 'extraction': 1, 'selection': 1}


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=1) + '\n')
    temp.replace(path)


def identity(batch, item):
    # Retries remain bounded across invocations and re-numbered batches. Changed
    # scientific inputs get a distinct identity; old attempts remain auditable.
    payload = [batch['_criteria_hash'], item['pmid'],
               batch['_input_hashes'][item['pmid']],
               batch.get('_analyses_hashes', {}).get(item['pmid'])]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def other_workers(ws):
    """Refuse adoption while legacy judges or project Python dispatchers are live.

    A root interactive Codex orchestrator is expected and is not a judge. Only
    PIDs, never full command lines (which may contain secrets), are reported.
    """
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            cwd = (p / 'cwd').resolve()
        except (OSError, RuntimeError):
            continue
        if not args:
            continue
        name = Path(args[0]).name
        judge = ('codex' in name or name == 'node') and 'exec' in args and cwd == ws
        dispatcher = name.startswith('python') and cwd == ws
        launcher = any(Path(a).name in {'run_judge.sh', 'run_judge_optimized.sh'} for a in args)
        if judge or dispatcher or (launcher and str(ws.name) in args):
            found.append(int(p.name))
    return found


def judge_one(launcher, project, stage, batch, timeout, logs):
    log = logs / f'{batch.stem}.{time.time_ns()}.{uuid.uuid4().hex[:6]}.jsonl'
    directory = single_response.new_attempt(batch, 'codex')
    record = json.loads((directory / 'attempt.json').read_text())
    record['launcher_log'] = str(log)
    single_response.atomic_json(directory / 'attempt.json', json.dumps(record))
    started = time.monotonic()
    code = 1
    with log.open('w') as output:
        try:
            proc = subprocess.Popen([str(launcher), project, stage, str(batch)],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, CBMA_JUDGE_ATTEMPT_DIR=str(directory)))
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                code = 124
        except OSError as exc:
            output.write(f'{type(exc).__name__}: {exc}\n')
    record = json.loads((directory / 'attempt.json').read_text())
    usage, reported = single_response.event_usage(directory / 'events.jsonl')
    # The outer runner can finalize an attempt even if its CLI/adapter was killed.
    status = record.get('status')
    if status == 'started':
        status = 'timeout' if code == 124 else 'failed'
    single_response.finish_attempt(directory, status=status, launcher_exit_code=code,
        launcher_log=str(log), usage=usage if reported else record.get('usage'),
        usage_reported=reported or record.get('usage_reported', False))
    error_text = log.read_text(errors='replace')
    stderr = directory / 'stderr.txt'
    if stderr.exists():
        error_text += stderr.read_text(errors='replace')
    return {'batch': batch.name, 'exit_code': code, 'attempt_id': directory.name,
            'audit': str(directory), 'provider_blocked': bool(code) and bookkeeping.provider_blocked(error_text),
            'seconds': round(time.monotonic() - started), 'log': str(log), 'usage': usage}


def main():
    global single_response, bookkeeping
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('project')
    ap.add_argument('--stage', choices=list(SIZES), required=True)
    ap.add_argument('--size', type=int)
    ap.add_argument('--limit', type=int, help='persistent pilot scope, never expands on retry')
    ap.add_argument('--parallel', type=int, default=8)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--time-budget', type=int, default=540)
    ap.add_argument('--judge-timeout', type=int, default=900)
    args = ap.parse_args()
    if os.environ.get('CBMA_CODEX_BATCH_JUDGE') == '1':
        ap.error('batch judges cannot launch stage runners')
    if not args.project or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in args.project):
        ap.error('invalid project name')
    if any(n <= 0 for n in [args.size or SIZES[args.stage], args.parallel, args.rounds,
                            args.max_attempts, args.time_budget, args.judge_timeout]) or (args.limit is not None and args.limit <= 0):
        ap.error('numeric controls must be positive')
    arm = Path(__file__).resolve().parent
    if arm.name not in {'e6-codex', 'e6-codex-luna'}:
        ap.error('install runner in an E6 Codex arm')
    ws = (arm / 'projects' / args.project).resolve()
    review = ws / args.project
    ledger_path = ws / '.agents/skills/cbma-review/scripts/ledger.py'
    launcher = arm / 'run_judge_optimized.sh'
    if not ledger_path.is_file() or not launcher.is_file():
        ap.error('workspace ledger or optimized launcher is missing')
    # Import the project's own snapshot: its hashes and validators stay unchanged.
    sys.path.insert(0, str(ledger_path.parent))
    import single_response
    import stage_bookkeeping as bookkeeping
    spec = importlib.util.spec_from_file_location('ledger', ledger_path)
    ledger = importlib.util.module_from_spec(spec)
    sys.modules['ledger'] = ledger
    spec.loader.exec_module(ledger)
    model = 'gpt-6-luna' if arm.name == 'e6-codex-luna' or args.stage in {'abstract', 'fulltext'} else 'gpt-6-astra'
    effort = 'medium' if args.stage in {'fulltext', 'selection'} else 'low'
    agent = f'codex-cli/{model}/effort-{effort}/{single_response.VERSION}'
    work = review / 'work' / args.stage
    work.mkdir(parents=True, exist_ok=True)
    log_dir = work / 'codex_runner'
    log_dir.mkdir(exist_ok=True)
    result = {'stage': args.stage, 'agent': agent, 'accepted': 0, 'rejected': 0,
              'batches_dispatched': 0, 'rounds': 0, 'errors': [], 'judge_failures': [],
              'usage': {}, 'audit_required': True}
    state_path = log_dir / 'state.json'
    deadline = time.monotonic() + args.time_budget
    # OS lock is released even on a crash; no PID guessing or stale-lock deletion.
    with (work / 'codex_stage.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'stage_busy': args.stage, 'hint': 'another Codex stage runner holds the lock'}))
            return 4
        blockers = other_workers(ws)
        if blockers:
            print(json.dumps({'workers_busy': blockers, 'hint': 'wait for existing judges/dispatchers before adopting this runner'}))
            return 4
        state = json.loads(state_path.read_text()) if state_path.exists() else {'attempts': {}, 'pilots': {}}
        attempts = state['attempts']
        rv = ledger.Review(review)
        scope = None
        if args.limit:
            pilot_key = rv.criteria[args.stage]['hash'] + ':' + str(args.limit)
            scope = state['pilots'].setdefault(pilot_key, rv.pending(args.stage)[:args.limit])
            write_json(state_path, state)
            existing = {i['pmid'] for b in ledger.batch_files(work) for i in json.loads(b.read_text())['items']}
            if existing - set(scope):
                raise ledger.LedgerError('existing batches exceed pilot scope; resolve them before starting this pilot')

        def ingest():
            report = ledger.cmd_ingest(ledger.Review(review), args.stage, agent)
            result['accepted'] += report['accepted']
            result['rejected'] += report['rejected']
            result['errors'] = (result['errors'] + report['errors'])[-20:]

        def pending():
            values = ledger.Review(review).pending(args.stage)
            return [p for p in values if scope is None or p in scope]

        ingest()  # recover finished, un-ingested batches from an interrupted run
        blocked = False
        while pending() and result['rounds'] < args.rounds and time.monotonic() < deadline - 5 and not blocked:
            batches = ledger.batch_files(work)
            if not batches:
                # Keep attempts and logs in codex_runner/, outside batch_* cleanup.
                scoped_rv = ledger.Review(review)
                if scope is not None:
                    original_pending = scoped_rv.pending
                    scoped_rv.pending = lambda stage: [p for p in original_pending(stage)
                                                      if stage != args.stage or p in scope]
                batches = ledger.cmd_batches(scoped_rv, args.stage,
                                             args.size or SIZES[args.stage],
                                             len(pending()) if scope is not None else None, discard=False)
                if scope is not None and any(i['pmid'] not in scope for b in batches for i in json.loads(b.read_text())['items']):
                    raise ledger.LedgerError('pilot inputs changed; stop rather than judge outside the original pilot')
            todo, exhausted = bookkeeping.eligible_batches(
                batches, pending(), attempts, args.max_attempts, log_dir)
            if exhausted:
                result['retry_exhausted'] = sorted(set(result.get('retry_exhausted', [])) | exhausted)
            if not todo:
                break
            result['rounds'] += 1
            # Submit only available slots, not hundreds of queued futures. A later
            # invocation resumes batches not launched within this time budget.
            with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
                running = {}
                queue = iter(todo)
                while True:
                    while len(running) < args.parallel and time.monotonic() < deadline - 5 and not blocked:
                        job = next(queue, None)
                        if job is None:
                            break
                        batch, keys = job
                        for key in keys:
                            attempts[key] = attempts.get(key, 0) + 1
                        write_json(state_path, state)
                        fut = pool.submit(judge_one, launcher, args.project, args.stage,
                                          batch, min(args.judge_timeout, max(0.1, deadline - time.monotonic() - 5)), log_dir)
                        running[fut] = batch
                    if not running:
                        break
                    done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                    for fut in done:
                        running.pop(fut)
                        row = fut.result()
                        result['batches_dispatched'] += 1
                        blocked = blocked or row['provider_blocked']
                        if row['exit_code']:
                            result['judge_failures'].append(row)
                        for k, v in row['usage'].items():
                            result['usage'][k] = result['usage'].get(k, 0) + v
            ingest()
        result['provider_blocked'] = blocked
        result['usage_limited'] = blocked
        result['usage_this_invocation'] = result['usage']
        result['usage'] = single_response.usage_of(work, 'codex')
        result['max_attempts'] = args.max_attempts
        result['judge_failure_count'] = len(result['judge_failures'])
        result['judge_failures'] = result['judge_failures'][-10:]
        result.update(pending=len(pending()), total_stage_pending=len(ledger.Review(review).pending(args.stage)),
                      pending_pmids=pending()[:50], waiting_batches=len(ledger.batch_files(work)),
                      verdict='JUDGMENTS_DONE' if not pending() else 'WORK_REMAINS',
                      logs=str(log_dir))
        write_json(log_dir / 'last_summary.json', result)
        print(json.dumps(result, indent=1))
        return 0 if not pending() else 3


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({'runner_error': f'{type(exc).__name__}: {exc}',
                          'hint': 'leave scientific decisions unchanged; resolve the execution error'}))
        sys.exit(1)

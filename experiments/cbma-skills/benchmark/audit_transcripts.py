"""Audit a cbma-skills run from its Claude Code session transcripts: tokens, and rules kept.

    python audit_transcripts.py ~/.claude/projects/<workspace-dir> --review REVIEW \\
        [--forbid PATH ...] [--allow PATH ...] [--out audit.json]

The transcript folder is the one Claude Code keeps for the workspace the review ran in
(the workspace path with "/" replaced by "-"). Every session file in it is read, and so
is every subagent file under <session>/subagents/.

ROLES. Each transcript is classified by its first prompt:
    judge <stage>    names a batch file (work/<stage>/batch_...)
    runner <stage>   asks for a stage ("Run the `<stage>` stage")
    orchestrator     any top-level session
    other            anything else (a subagent that is neither)

TOKENS. Assistant messages are de-duplicated by message id, taking the largest value of
each usage field, because a streamed message can be logged on several lines. Recorded
output tokens for subagents can undercount: some lines keep the usage from the start of
the stream.

RULES CHECKED
    prompts     a judge's first prompt holding anything beyond the dispatch template:
                guidance that reached a judge outside the versioned protocol
    blinding    any tool call whose path or command touches a --forbid prefix
    writes      a judge writing anywhere but its batch output (Write/Edit targets, and
                Bash redirections or file writes); anyone else writing to decisions/,
                analyses/ or a batch output directly
    failures    tool results flagged as errors, and usage-limit (429) messages
    effort      the perTurnEffort each role ran at, which should match its definition
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

USAGE = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
LIMIT = re.compile(r"hit your (session|usage|weekly) limit|rate_limit|\b429\b", re.I)


def classify(first_prompt: str, top_level: bool) -> tuple:
    m = re.search(r"work/(abstract|fulltext|extraction|selection)/batch_\d+\.json", first_prompt)
    if m and not top_level:
        return "judge", m.group(1), m.group(0)
    m = re.search(r"Run the `?(abstract|fulltext|extraction|selection)`? stage", first_prompt)
    if m and not top_level:
        return "runner", m.group(1), None
    return ("orchestrator" if top_level else "other"), None, None


def tool_paths(name: str, inp: dict) -> List[str]:
    out = [str(inp[k]) for k in ("file_path", "path", "notebook_path") if inp.get(k)]
    if name == "Bash" and inp.get("command"):
        out.append(inp["command"])
    if name in ("Grep", "Glob") and inp.get("pattern"):
        out.append(str(inp.get("path") or ""))
    return out


HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n.*?\n\s*\1\b", re.S)
QUOTED = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"")
REDIRECT = re.compile(r"(?<![0-9&<>=-])>>?\s*(?!&|=)([^\s;|&<>()]+)|\btee\s+(?:-a\s+)?([^\s;|&]+)")
PY_OPEN = re.compile(r"open\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wa]")
# The sentences of the stage runner's judge prompt (agents/cbma-stage-runner.md). Anything a
# judge's first prompt holds beyond them is guidance the skills did not give.
JUDGE_TEMPLATE = [
    r"Your batch file is `[^`]+`\.",
    r"Follow the skill at\s+`[^`]+`,?(\s*where `[^`]+` is the batch's `skill` field\.)?",
    r"Write your output to the path in the batch's `output` field\.",
    r"Reply with only the number of items you wrote\.",
    # the orchestrator skill's older loop template, for runs without stage runners
    r"Read `[^`]+SKILL\.md` and follow it exactly\.",
    r"Process every item in it and write your output to the path in the batch's `output` field\.",
    r"Do not read or write any other part of the review folder except the files the batch names\.",
    r"When done, reply with only the number of items you wrote\.",
]


def prompt_additions(prompt: str) -> str:
    rest = prompt
    for sentence in JUDGE_TEMPLATE:
        rest = re.sub(sentence, " ", rest)
    return re.sub(r"\s+", " ", rest).strip(" .")


LEDGER_SUBCOMMANDS = {"init", "batches", "ingest", "status", "export", "needs-fulltext", "import-analyses"}


def bash_write_targets(command: str) -> List[str]:
    """Files a shell command writes: redirections and tee at the shell level (after
    removing heredoc bodies and quoted strings, where "look > view" is text), plus
    python open(path, "w"/"a") anywhere."""
    shell = QUOTED.sub("''", HEREDOC.sub("", command))
    out = [m.group(1) or m.group(2) for m in REDIRECT.finditer(shell) if m.group(1) or m.group(2)]
    return out + PY_OPEN.findall(command)


def same_file(target: str, output: str, review: Path) -> bool:
    """A write lands on the judge's own output (the .out.jsonl file, or inside the .out/ dir)."""
    target_path = Path(target) if Path(target).is_absolute() else review / target
    out = Path(output)
    try:
        target_path = target_path.resolve()
    except OSError:
        pass
    return target_path == out or out in target_path.parents


def read_transcript(path: Path) -> dict:
    messages: Dict[str, dict] = {}
    effort: collections.Counter = collections.Counter()
    models: collections.Counter = collections.Counter()
    tools: List[tuple] = []
    errors, limits = 0, []
    first_prompt = None
    for line in path.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = d.get("message") or {}
        if d.get("type") == "user":
            content = msg.get("content")
            if first_prompt is None:
                first_prompt = content if isinstance(content, str) else " ".join(
                    c.get("text", "") for c in content or [] if isinstance(c, dict))
            for c in content if isinstance(content, list) else []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    text = json.dumps(c.get("content"))[:2000]
                    if c.get("is_error"):
                        errors += 1
                    if LIMIT.search(text) and "limit" in text.lower():
                        limits.append(d.get("timestamp"))
        elif d.get("type") == "assistant":
            mid = msg.get("id") or d.get("uuid")
            u = msg.get("usage") or {}
            prev = messages.setdefault(mid, {k: 0 for k in USAGE})
            for k in USAGE:
                prev[k] = max(prev[k], int(u.get(k) or 0))
            if d.get("perTurnEffort"):
                effort[d["perTurnEffort"]] += 1
            if msg.get("model"):
                models[msg["model"]] += 1
            for c in msg.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    tools.append((c.get("name"), c.get("input") or {}, d.get("timestamp")))
            text = json.dumps(msg.get("content"))[:2000]
            if LIMIT.search(text) and "limit" in text.lower() and not msg.get("usage", {}).get("output_tokens"):
                limits.append(d.get("timestamp"))
    totals = {k: sum(m[k] for m in messages.values()) for k in USAGE}
    return {"messages": len(messages), "tokens": totals, "effort": dict(effort), "models": dict(models),
            "tools": tools, "tool_errors": errors, "limit_events": sorted(set(t for t in limits if t)),
            "first_prompt": first_prompt or ""}


def limit_events(stamps: List[str], gap_minutes: int = 30) -> List[dict]:
    """Group usage-limit messages into events: one per burst, split by quiet gaps."""
    import datetime as dt
    events: List[dict] = []
    for s in stamps:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if events and (t - events[-1]["_last"]).total_seconds() < gap_minutes * 60:
            events[-1]["_last"] = t
            events[-1]["messages"] += 1
        else:
            events.append({"first": s, "_last": t, "messages": 1})
    return [{"first": e["first"], "last": e["_last"].isoformat(), "messages": e["messages"]} for e in events]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcripts", type=Path)
    ap.add_argument("--review", type=Path, required=True, help="the review folder, for batch outputs")
    ap.add_argument("--forbid", action="append", default=[], help="path prefix no tool call may touch")
    ap.add_argument("--allow", action="append", default=[], help="prefix exempt from --forbid")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    review = args.review.expanduser().resolve()
    top = sorted(args.transcripts.glob("*.jsonl"))
    subs = sorted(args.transcripts.glob("*/subagents/*.jsonl"))
    rows = []
    for path in top + subs:
        t = read_transcript(path)
        role, stage, batch = classify(t["first_prompt"], path in top)
        if role == "orchestrator" and t["messages"] == 0:
            continue
        t.update(file=str(path.relative_to(args.transcripts)), role=role, stage=stage, batch=batch)
        rows.append(t)

    # ---- tokens by role and stage
    by: Dict[str, dict] = {}
    for t in rows:
        key = t["role"] if t["role"] in ("orchestrator", "other") else f"{t['role']} {t['stage']}"
        agg = by.setdefault(key, {"transcripts": 0, **{k: 0 for k in USAGE}, "effort": collections.Counter()})
        agg["transcripts"] += 1
        for k in USAGE:
            agg[k] += t["tokens"][k]
        agg["effort"].update(t["effort"])
    for agg in by.values():
        agg["all_input"] = agg["input_tokens"] + agg["cache_creation_input_tokens"] + agg["cache_read_input_tokens"]
        agg["effort"] = dict(agg["effort"])
    total_input = sum(a["all_input"] for a in by.values())

    # ---- rules
    # A forbidden path matches only at a path boundary: /x/workspace must not match
    # /x/workspace-records.
    forbid = {str(Path(p).expanduser()): re.compile(re.escape(str(Path(p).expanduser())) + r"(?=$|[/\s'\"`;)|&])")
              for p in args.forbid}
    allow = [str(Path(p).expanduser()) for p in args.allow]
    blinding, stray_writes, direct_decision_writes, unresolved_writes = [], [], [], []
    for t in rows:
        output = None
        if t["role"] == "judge":
            try:
                output = json.loads((review / t["batch"]).read_text())["output"]
            except (OSError, KeyError, json.JSONDecodeError):
                done = sorted((review / Path(t["batch"]).parent / "done").glob(f"*_{Path(t['batch']).name}"))
                output = json.loads(done[-1].read_text())["output"] if done else None
        for name, inp, ts in t["tools"]:
            for s in tool_paths(name, inp):
                hit = next((f for f, rx in forbid.items() if rx.search(s) and not any(a in s for a in allow)), None)
                if hit:
                    blinding.append({"file": t["file"], "role": t["role"], "tool": name, "hit": hit, "at": ts,
                                     "what": s[:200]})
            targets = [str(inp.get("file_path") or inp.get("notebook_path"))] if name in WRITE_TOOLS else []
            if name == "Bash":
                command = inp.get("command") or ""
                # A relative target is relative to the directory the command cd's into.
                cd = re.search(r"\bcd\s+['\"]?([^\s;&|'\"]+)", command)
                base = Path(cd.group(1)) if cd and Path(cd.group(1)).is_absolute() else None
                targets += [str(base / w) if base and not Path(w).is_absolute() else w
                            for w in bash_write_targets(command)]
            for target in targets:
                if target in ("/dev/null", "None", "", "''") or target.startswith("&"):
                    continue
                if t["role"] == "judge":
                    if output and same_file(target, output, review):
                        continue
                    # Written by name after a cd, or through a shell variable: its own output
                    # when the name is the output's, or a <pmid>.json inside an extraction .out/.
                    name_only = re.sub(r"^\$\{?\w+\}?/", "", target)
                    if output and (name_only == Path(output).name or
                                   (output.endswith(".out") and re.fullmatch(r"\d+\.json", name_only))):
                        continue
                    if "$" in target:
                        unresolved_writes.append({"file": t["file"], "stage": t["stage"], "target": target[:200]})
                        continue
                    stray_writes.append({"file": t["file"], "stage": t["stage"], "tool": name, "target": target[:200]})
                elif re.search(r"/(decisions|analyses)/|/work/\w+/batch_\d+\.out", target):
                    direct_decision_writes.append({"file": t["file"], "role": t["role"], "tool": name,
                                                   "target": target[:200]})

    prompt_extra = [{"file": t["file"], "stage": t["stage"], "batch": t["batch"],
                     "added": prompt_additions(t["first_prompt"])[:300]}
                    for t in rows if t["role"] == "judge" and prompt_additions(t["first_prompt"])]

    ledger_calls = collections.Counter()
    for t in rows:
        if t["role"] in ("orchestrator", "runner"):
            for name, inp, _ in t["tools"]:
                for sub in re.findall(r"ledger\.py\s+([a-z-]+)", inp.get("command") or "") if name == "Bash" else []:
                    if sub in LEDGER_SUBCOMMANDS:
                        ledger_calls[f"{t['role']}:{sub}"] += 1

    report = {
        "transcripts": len(rows),
        "models": dict(sum((collections.Counter(t["models"]) for t in rows), collections.Counter())),
        "tokens": {"total_input": total_input, "by_role": by},
        "limit_events": limit_events(sorted({e for t in rows for e in t["limit_events"]})),
        "tool_errors": sum(t["tool_errors"] for t in rows),
        "blinding_hits": blinding,
        "judge_writes_outside_output": stray_writes,
        "judge_writes_unresolved": unresolved_writes,
        "judge_prompts_with_added_guidance": prompt_extra,
        "decision_files_written_directly": direct_decision_writes,
        "ledger_calls": dict(ledger_calls),
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")

    print(f"{len(rows)} transcripts; models {report['models']}")
    print(f"{'role':22s} {'n':>4s} {'all input':>12s} {'cache read':>11s} {'output':>9s}  effort")
    for key in sorted(by, key=lambda k: -by[k]["all_input"]):
        a = by[key]
        print(f"{key:22s} {a['transcripts']:4d} {a['all_input']/1e6:10.1f}M {a['cache_read_input_tokens']/max(a['all_input'],1):10.0%} "
              f"{a['output_tokens']/1e3:8.1f}k  {a['effort']}")
    print(f"total input {total_input/1e6:.1f}M | usage-limit events {len(report['limit_events'])} | tool errors {report['tool_errors']}")
    print(f"blinding hits {len(blinding)} | judge writes outside output {len(stray_writes)} "
          f"({len({w['file'] for w in stray_writes})} judges; {len(unresolved_writes)} unresolved) | "
          f"decision files written directly {len(direct_decision_writes)}")
    print(f"judge prompts with guidance beyond the template: {len(prompt_extra)}")
    for x in prompt_extra[:5]:
        print(f"   {x['batch']}: {x['added'][:140]}")
    print(f"ledger calls {report['ledger_calls']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

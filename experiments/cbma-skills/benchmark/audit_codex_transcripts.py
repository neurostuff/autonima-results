"""Audit Codex CLI rollout JSONL logs for one cbma-skills project.

    python benchmark/audit_codex_transcripts.py ~/.codex/sessions \\
        --workspace /path/to/workspace --review /path/to/workspace/PROJECT \\
        [--forbid /path/to/gold ...] [--out /path/to/audit.json]

Codex stores sessions below ~/.codex/sessions. This script scans JSONL files
recursively and includes only sessions whose recorded cwd matches --workspace.
It reads token_usage_record entries and function/custom tool calls from the
Codex rollout format. Codex log schemas can change; unrecognized records are
ignored and the report records the number of recognized usage records.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Optional

LIMIT = re.compile(r"rate[_ -]?limit|usage limit|weekly limit|\b429\b", re.I)
STAGE = re.compile(r"work/(abstract|fulltext|extraction|selection)/batch_\d+\.json")


def text_content(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(x.get("text", "")) for x in value if isinstance(x, dict))
    return ""


def tool_records(payload: dict) -> list[tuple[str, str, str]]:
    """Return (name, serialized input, output) for common Codex tool records."""
    kind = payload.get("type")
    if kind in ("function_call", "custom_tool_call"):
        name = str(payload.get("name") or "")
        args = payload.get("arguments", payload.get("input", {}))
        if isinstance(args, str):
            args_text = args
        else:
            args_text = json.dumps(args, ensure_ascii=False)
        return [(name, args_text, "")]
    if kind in ("function_call_output", "custom_tool_call_output"):
        return [("<output>", "", str(payload.get("output", "")))]
    return []


def read_session(path: Path) -> dict:
    meta = {}
    first_prompt = ""
    tools = []
    usages = {}
    models = collections.Counter()
    efforts = collections.Counter()
    errors = 0
    limit_texts = []
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = row.get("type")
        payload = row.get("payload") or {}
        if kind == "session_meta":
            meta.update(payload)
        elif kind == "turn_context":
            if payload.get("model"):
                models[payload["model"]] += 1
            if payload.get("effort"):
                efforts[payload["effort"]] += 1
        elif kind == "response_item":
            if payload.get("type") == "message" and payload.get("role") == "user" and not first_prompt:
                first_prompt = text_content(payload.get("content"))
            tools.extend((name, args, out, row.get("timestamp")) for name, args, out in tool_records(payload))
            body = json.dumps(payload, ensure_ascii=False)
            if LIMIT.search(body):
                limit_texts.append(row.get("timestamp"))
            if payload.get("type") == "function_call_output" and re.search(r"\b(error|failed)\b", str(payload.get("output", "")), re.I):
                errors += 1
        elif kind == "token_usage_record":
            usage = payload.get("usage") or {}
            key = payload.get("response_id") or f"{payload.get('turn_id')}:{row.get('ordinal')}"
            # One record per response. Keep the latest if a CLI version streams updates.
            usages[key] = {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
                "cache_write_input_tokens": int(usage.get("cache_write_input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "reasoning_output_tokens": int(usage.get("reasoning_output_tokens") or 0),
            }
    usage_total = {k: sum(u[k] for u in usages.values()) for k in
                   ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens", "reasoning_output_tokens")}
    marker = re.search(r"(?m)^CBMA_SINGLE_RESPONSE_V[12] batch=([^\n]+)", first_prompt)
    stage_match = STAGE.search(marker.group(1) if marker else first_prompt)
    role = "judge" if stage_match else "orchestrator"
    return {"file": str(path), "cwd": meta.get("cwd"), "session_id": meta.get("session_id"),
            "parent_thread_id": meta.get("parent_thread_id"), "cli_version": meta.get("cli_version"),
            "models": dict(models), "effort": dict(efforts), "role": role,
            "stage": stage_match.group(1) if stage_match else None, "first_prompt": first_prompt,
            "usage_records": len(usages), "tokens": usage_total, "tools": tools,
            "tool_errors": errors, "limit_events": sorted(set(x for x in limit_texts if x))}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", type=Path, help="Codex sessions directory, usually ~/.codex/sessions")
    ap.add_argument("--workspace", type=Path, required=True, help="project workspace root recorded as session cwd")
    ap.add_argument("--review", type=Path, required=True, help="review folder containing work/ and ledger files")
    ap.add_argument("--forbid", action="append", default=[], help="path that tool inputs must not reference; repeatable")
    ap.add_argument("--out", type=Path, help="write full JSON report here")
    args = ap.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    review = args.review.expanduser().resolve()
    rows = []
    for path in sorted(args.sessions.expanduser().rglob("*.jsonl")):
        row = read_session(path)
        if row["cwd"] and Path(row["cwd"]).expanduser().resolve() == workspace:
            rows.append(row)

    forbidden = [str(Path(p).expanduser().resolve()) for p in args.forbid]
    blinding = []
    writes = []
    ledger_calls = collections.Counter()
    by = collections.defaultdict(lambda: {"sessions": 0, "usage_records": 0, "tokens": collections.Counter(), "models": collections.Counter(), "effort": collections.Counter()})
    for row in rows:
        key = row["role"] + (f" {row['stage']}" if row["stage"] else "")
        agg = by[key]
        agg["sessions"] += 1
        agg["usage_records"] += row["usage_records"]
        agg["tokens"].update(row["tokens"])
        agg["models"].update(row["models"])
        agg["effort"].update(row["effort"])
        for name, raw, output, stamp in row["tools"]:
            haystack = raw + " " + output
            for prefix in forbidden:
                if re.search(re.escape(prefix) + r"(?=$|[/\s'\"`;)|&])", haystack):
                    blinding.append({"file": row["file"], "tool": name, "path": prefix, "at": stamp})
            if name in ("exec_command", "Bash"):
                for target in re.findall(r"(?:^|\s)(?:>>?|tee\s+)([^\s;|&]+)", raw):
                    writes.append({"file": row["file"], "role": row["role"], "target": target})
                ledger_calls.update(f"{row['role']}:{x}" for x in re.findall(r"ledger\.py\s+([a-z-]+)", raw))
            if name in ("apply_patch", "write_file"):
                if re.search(r"/(?:decisions|analyses)/|/work/\w+/batch_\d+\.out", raw):
                    writes.append({"file": row["file"], "role": row["role"], "target": raw[:300]})

    details = [{"file": x["file"], "session_id": x["session_id"], "cli_version": x["cli_version"],
                "role": x["role"], "stage": x["stage"], "models": x["models"], "effort": x["effort"],
                "usage_records": x["usage_records"], "tokens": x["tokens"], "tool_errors": x["tool_errors"]}
               for x in rows]
    report = {"workspace": str(workspace), "review": str(review), "sessions": len(rows),
              "recognized_usage_records": sum(x["usage_records"] for x in rows),
              "by_role": {k: {"sessions": v["sessions"], "usage_records": v["usage_records"],
                              "tokens": dict(v["tokens"]), "models": dict(v["models"]), "effort": dict(v["effort"])}
                          for k, v in by.items()},
              "tool_errors": sum(x["tool_errors"] for x in rows),
              "limit_events": sorted({e for x in rows for e in x["limit_events"]}),
              "blinding_hits": blinding, "possible_direct_writes": writes,
              "ledger_calls": dict(ledger_calls), "session_details": details}
    if args.out:
        args.out.expanduser().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} Codex sessions in {workspace}; recognized usage records {report['recognized_usage_records']}")
    for role, data in sorted(report["by_role"].items()):
        tokens = data["tokens"]
        print(f"{role:22s} sessions={data['sessions']} input={tokens.get('input_tokens', 0):,} "
              f"cached={tokens.get('cached_input_tokens', 0):,} output={tokens.get('output_tokens', 0):,} "
              f"models={data['models']} effort={data['effort']}")
    print(f"tool errors {report['tool_errors']} | limit events {len(report['limit_events'])} | "
          f"blinding hits {len(blinding)} | possible direct writes {len(writes)}")
    if not rows:
        print("No sessions matched. Codex logs must exist, and their recorded cwd must equal --workspace.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

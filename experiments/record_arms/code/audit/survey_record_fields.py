#!/usr/bin/env python3
"""Field-level variance across every committed extraction record.

Walks `records/<project>/*.extraction.json`, flattens each to (path, ExtractedValue) pairs
with list indices collapsed, and reports per path: how often it is present and filled, the
python types its `value` takes, how many distinct values, and the top values. The point is
to separate three kinds of field:

  controlled    few distinct values, already an enum -- nothing to normalise
  normalisable  many distinct values that are variants of a few things (case, plurals,
                punctuation, units) -- rules can collapse them
  open          many distinct values naming real-world entities (tasks, conditions,
                instruments) -- rules cannot; these need an ontology
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

RECORDS = Path(__file__).resolve().parents[2] / "records"
EV_KEYS = {"value", "extraction_status", "value_source", "evidence"}


def walk(node, path="", out=None):
    """Yield (path, extracted_value_dict) for every ExtractedValue in the tree."""
    if out is None:
        out = []
    if isinstance(node, dict):
        if EV_KEYS & set(node) and "extraction_status" in node:
            out.append((path, node))
            return out
        for k, v in node.items():
            walk(v, f"{path}.{k}" if path else k, out)
    elif isinstance(node, list):
        for v in node:
            walk(v, f"{path}[]", out)
    return out


def main() -> int:
    present = Counter()
    filled = Counter()
    types = defaultdict(Counter)
    values = defaultdict(Counter)
    status = defaultdict(Counter)
    source = defaultdict(Counter)
    n_records = 0

    for project_dir in sorted(p for p in RECORDS.iterdir() if p.is_dir()):
        for f in sorted(project_dir.glob("*.extraction.json")):
            n_records += 1
            try:
                doc = json.loads(f.read_text())
            except Exception as exc:
                print(f"  unreadable {f}: {exc}", file=sys.stderr)
                continue
            for path, ev in walk(doc):
                present[path] += 1
                status[path][str(ev.get("extraction_status"))] += 1
                source[path][str(ev.get("value_source"))] += 1
                v = ev.get("value")
                if v is None or v == [] or v == "":
                    continue
                filled[path] += 1
                types[path][type(v).__name__] += 1
                for item in (v if isinstance(v, list) else [v]):
                    if isinstance(item, (str, int, float, bool)):
                        values[path][str(item)] += 1

    print(f"records read: {n_records}\n")
    rows = []
    for path, n in present.items():
        vals = values[path]
        rows.append((path, n, filled[path], len(vals),
                     dict(types[path]), vals.most_common(3)))
    # the interesting fields are the filled ones with many distinct values
    rows.sort(key=lambda r: (-r[3], -r[2]))
    print(f"{'path':58} {'filled':>7} {'distinct':>8}  types / top values")
    for path, n, fl, nd, ty, top in rows:
        if fl == 0:
            continue
        t = ",".join(f"{k}" for k in ty)
        sample = "; ".join(f"{v[:28]}({c})" for v, c in top)
        print(f"{path[:58]:58} {fl:>7} {nd:>8}  {t:<12} {sample[:70]}")
    print(f"\nfields never filled: "
          f"{sorted(p for p in present if filled[p] == 0)[:12]} "
          f"({sum(1 for p in present if filled[p] == 0)} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

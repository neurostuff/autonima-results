"""Compile pondie extraction records into compact autonima document sources.

One-time conversion for testing autonima's `documents:` support against pondie records. For
each project under --records it writes, under --out/<project>/:

  text/<pmid>.md               the record as compact markdown, pondie's ids left as written
                               (documents kind `text`: coordinates still come from the article)
  records/<pmid>.md            the same render with every analysis id written as
                               {{analysis:<local_id>}}, for autonima to rewrite to its own ids
  records/<pmid>.analyses.json one entry per record analysis, in record order, with
                               coordinates joined from pondie's final stage-1 parse
  MANIFEST.csv                 per paper: characters, analyses, analyses with coordinates

The render keeps extracted values only: evidence quotations, fields that were not reported
or not applicable, and extraction bookkeeping are dropped.

A record analysis links to its coordinates through source_table_analysis, a key of the form
"<table_id>#<n>": entry n (1-based) among that table's entries in
<corpus>/<pmid>/stage1/analyses.json, the parse as it stands after pondie's prose-foci and
sign-split stages. Links that do not resolve travel with no points and are counted.

  python scripts/compile_pondie_records.py \
      --records articles/pondie/records --corpus articles/pondie/corpus \
      --out articles/pondie/md
"""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

DROP_KEYS = {"evidence", "extraction_metadata", "source_table_analysis", "local_id"}
DROP_VALUES = {"not_applicable"}


def _extracted(field):
    """The value of an ExtractedValue wrapper, or None when there is nothing to show."""
    if field.get("extraction_status") != "extracted":
        return None
    value = field.get("value")
    if isinstance(value, list):
        value = [v for v in value if str(v) not in DROP_VALUES]
        return "; ".join(map(str, value)) if value else None
    if value in (None, "") or str(value) in DROP_VALUES:
        return None
    return str(value)


def render_fields(obj, ref, depth=0):
    """One line per extracted value, nesting shown by indentation."""
    pad, lines = "  " * depth, []
    for key, value in obj.items():
        if key in DROP_KEYS:
            continue
        label = key.replace("_", " ")
        if isinstance(value, dict) and "extraction_status" in value:
            text = _extracted(value)
            if text is not None:
                lines.append(f"{pad}- {label}: {text}")
        elif isinstance(value, dict):
            sub = render_fields(value, ref, depth + 1)
            if sub:
                lines += [f"{pad}- {label}:"] + sub
        elif isinstance(value, list):
            scalars = [ref(str(v)) for v in value if not isinstance(v, dict)]
            if scalars:
                lines.append(f"{pad}- {label}: {', '.join(scalars)}")
            for item in value:
                if isinstance(item, dict):
                    sub = render_fields(item, ref, depth + 1)
                    if sub:
                        lines += [f"{pad}- {label}:"] + sub
        elif value not in (None, "") and str(value) not in DROP_VALUES:
            lines.append(f"{pad}- {label}: {ref(str(value))}")
    return lines


def render_record(record, ref):
    lines = ["# Extraction record"]
    for key, value in record.items():
        if key in DROP_KEYS:
            continue
        title = key.replace("_", " ").capitalize()
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            lines.append(f"\n## {title}")
            for item in value:
                local_id = item.get("local_id", "")
                heading = f"Analysis {ref(local_id)}" if key == "analyses" else local_id
                lines.append(f"\n### {heading}")
                lines += render_fields(item, ref)
        else:
            body = render_fields({key: value}, ref)
            if body:
                lines += [f"\n## {title}"] + body
    return "\n".join(lines) + "\n"


def stage1_index(corpus, pmid):
    """'<table_id>#<n>' -> parse entry, or None when the paper has no stage-1 parse."""
    path = corpus / pmid / "stage1" / "analyses.json"
    if not path.is_file():
        return None
    counts, index = collections.Counter(), {}
    for entry in json.loads(path.read_text())["analyses"]:
        table_id = str(entry.get("table_id"))
        counts[table_id] += 1
        index[f"{table_id}#{counts[table_id]}"] = entry
    return index


def points_of(entry):
    return [
        {k: point[k] for k in ("coordinates", "space", "values") if point.get(k) is not None}
        for point in entry.get("points") or []
    ]


def compile_project(records_dir, corpus, out):
    (out / "text").mkdir(parents=True, exist_ok=True)
    (out / "records").mkdir(parents=True, exist_ok=True)
    stats, manifest = collections.Counter(), []

    for path in sorted(records_dir.glob("*.extraction.json")):
        pmid = path.name.split(".")[0]
        record = json.loads(path.read_text())
        analyses = record.get("analyses", [])
        local_ids = {a["local_id"] for a in analyses}
        wrap = lambda v: f"{{{{analysis:{v}}}}}" if v in local_ids else v

        text_doc = render_record(record, lambda v: v)
        (out / "text" / f"{pmid}.md").write_text(text_doc, encoding="utf-8")
        records_doc = render_record(record, wrap).encode("utf-8")
        (out / "records" / f"{pmid}.md").write_bytes(records_doc)

        index = stage1_index(corpus, pmid)
        entries, with_points = [], 0
        for analysis in analyses:
            link = (analysis.get("source_table_analysis") or {}).get("value")
            parsed = index.get(link) if (index and link) else None
            if not link:
                stats["no link in record"] += 1
            elif index is None:
                stats["no stage-1 parse for paper"] += 1
            elif parsed is None:
                stats["link does not resolve"] += 1
            elif not parsed.get("points"):
                stats["resolves, no points"] += 1
            else:
                stats["resolves with points"] += 1
                with_points += 1
            name = analysis.get("name") or {}
            definition = analysis.get("definition") or {}
            entries.append({
                "key": analysis["local_id"],
                "name": name.get("value") if isinstance(name, dict) else None,
                "description": definition.get("value") if isinstance(definition, dict) else None,
                "table_id": str(parsed["table_id"]) if parsed else None,
                "points": points_of(parsed) if parsed else [],
                "document": "\n".join(render_fields(analysis, wrap)) or None,
            })
        (out / "records" / f"{pmid}.analyses.json").write_text(
            json.dumps({
                "document_sha256": hashlib.sha256(records_doc).hexdigest(),
                "analyses": entries,
            }, indent=1),
            encoding="utf-8",
        )
        stats["records"] += 1
        stats["analyses"] += len(analyses)
        manifest.append({
            "pmid": pmid,
            "source_bytes": path.stat().st_size,
            "text_chars": len(text_doc),
            "analyses": len(analyses),
            "analyses_with_points": with_points,
        })

    with open(out / "MANIFEST.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, required=True, help="directory of <project>/<pmid>.extraction.json")
    ap.add_argument("--corpus", type=Path, required=True, help="pondie corpus with <pmid>/stage1/analyses.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    summary = {}
    for project_dir in sorted(p for p in args.records.iterdir() if p.is_dir()):
        summary[project_dir.name] = dict(
            compile_project(project_dir, args.corpus, args.out / project_dir.name)
        )
        print(project_dir.name, json.dumps(summary[project_dir.name]))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

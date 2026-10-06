"""Trimmed full text for full-text screening: what the criteria need, without the rest.

    python trim.py REVIEW [--pmid PMID] [--show]   # how much trimming would cut, per source

Full-text judges read two whole papers each, and most of a paper is introduction,
discussion, references and publisher furniture. Screening criteria are mostly about the
sample, the task and the analysis, with a few about what the results report. So the
trimmed view keeps, in this order:

  * the title;
  * the abstract;
  * every methods section, whole;
  * each results heading with its first paragraph;
  * every table (label, caption, footer and cells), duplicates excepted.

It is deterministic: sections are found by their headings in the normalized markdown
that `docnorm.py` writes for every source (PMC, Elsevier XML and publisher HTML alike).
When no methods heading is found, or the methods are an implausible share of the text,
the full text is used instead and the item says why. A miss costs tokens, never the
evidence a judge would otherwise have seen.

Bump TRIM_VERSION whenever the output for the same document can change: it is part of
the full-text input hash, so a new version reopens trimmed decisions.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

TRIM_VERSION = "1"
SPACE_CONTEXT_VERSION = "2"

# Match coordinate-system evidence, not generic anatomical region names. Keep
# whole source paragraphs so atlas mentions and conversion direction have context.
_SPACE_EVIDENCE = re.compile(
    r"\b(?:MNI|Talairach|Montreal\s+Neurological\s+Institute|ICBM(?:[- ]?\d+)?"
    r"|stereotaxic|stereotactic|coordinate\s+(?:space|system)|Brett|Lancaster"
    r"|tal2mni|mni2tal|icbm2tal)\b", re.I)

_NUMBERING = r"(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?"
METHODS = re.compile(
    r"^" + _NUMBERING + r"((?:[\w-]+,?\s+){1,3}and\s+methods?|methods?\s+and\s+materials?|methods?|methodology"
    r"|experimental\s+procedures?"
    r"|study\s+design|participants?|subjects?|patients?)\b", re.I)
RESULTS = re.compile(r"^" + _NUMBERING + r"results?\b", re.I)
ABSTRACT = re.compile(r"^(abstract|summary)\b", re.I)
# Headings that end a methods or results span.
STOP = re.compile(r"^" + _NUMBERING + r"(results?|discussion|conclusions?|general\s+discussion|references"
                  r"|acknowledg|funding|supplementary|appendix|declaration|conflicts?\s+of\s+interest"
                  r"|author\s+contributions?)\b", re.I)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_PLACEHOLDER = re.compile(r"^\[TABLE [^\]]*\]$")

MIN_METHODS_SHARE = 0.03
MAX_METHODS_SHARE = 0.70


class Section:
    def __init__(self, level: int, title: str, start: int):
        self.level, self.title, self.start = level, title, start
        self.lines: List[str] = []

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


# A standalone line that is exactly a section name, as some publisher pages print them
# ("METHOD", "Subjects and methods") without heading markup.
_BARE_SECTION = re.compile(
    r"^" + _NUMBERING + r"(introduction|background|materials?\s+and\s+methods?|methods?\s+and\s+materials?|methods?"
    r"|methodology|experimental\s+procedures?|(?:subjects?|participants?|patients?)\s+and\s+methods?"
    r"|results?|discussion|comment|conclusions?|general\s+discussion|references|acknowledge?ments?)\s*:?$", re.I)


def promote_bare_sections(text: str) -> str:
    """Give heading markup to standalone section-name lines, only in a document that has no
    methods heading at all: a line alone in its paragraph, matching a section name exactly."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        alone = (i == 0 or not lines[i - 1].strip()) and (i + 1 == len(lines) or not lines[i + 1].strip())
        out.append(f"## {line.strip().rstrip(':')}" if alone and _BARE_SECTION.match(line.strip()) else line)
    return "\n".join(out)


def sections(text: str) -> List[Section]:
    """The document as a flat list of sections, one per heading. Text before the first
    heading is a level-0 section with an empty title."""
    out = [Section(0, "", 0)]
    for i, line in enumerate(text.splitlines()):
        m = _HEADING.match(line)
        if m:
            out.append(Section(len(m.group(1)), m.group(2).strip(), i))
        else:
            out[-1].lines.append(line)
    return out


def _span(secs: List[Section], i: int) -> int:
    """End (exclusive) of the span opened by section i: the next heading at or above its
    level that starts results, discussion or back matter, or failing that the next
    heading above its level. Subsections and sibling method sections ("Participants",
    "fMRI acquisition") stay inside."""
    level = secs[i].level
    for j in range(i + 1, len(secs)):
        if secs[j].level <= level and STOP.match(secs[j].title) and not METHODS.match(secs[j].title):
            return j
        if secs[j].level < level:
            return j
    return len(secs)


def _first_paragraph(body: str) -> str:
    for para in re.split(r"\n\s*\n", body):
        lines = [ln for ln in para.strip().splitlines() if not _TABLE_PLACEHOLDER.match(ln.strip())]
        if " ".join(lines).strip():
            return "\n".join(lines).strip()
    return ""


def _tables(tables_dir: Optional[Path]) -> List[str]:
    out = []
    if not tables_dir or not tables_dir.exists():
        return out
    for p in sorted(tables_dir.glob("*.json")):
        t = json.loads(p.read_text())
        if t.get("duplicate_of"):
            continue
        out.append(f"#### TABLE {t['table_id']}: {t.get('label') or ''}\n"
                   f"caption: {t.get('caption') or ''}\nfooter: {t.get('footer') or ''}\n\n{t.get('tsv') or ''}")
    return out


def trim(text: str, tables_dir: Optional[Path] = None) -> Tuple[str, dict]:
    """Return (text the judge reads, info). info["view"] is "trimmed", or "full" with a
    "fallback" reason when the trimmed view could not be trusted."""
    secs = sections(text)
    promoted = False
    if not any(s.level and METHODS.match(s.title) for s in secs):
        promoted_text = promote_bare_sections(text)
        if promoted_text != text:
            secs, promoted = sections(promoted_text), True
    total = len(text) or 1
    used = set()
    # A "Patients" or "Participants" subheading inside Results is not a methods section.
    in_results = set()
    for k, s in enumerate(secs):
        if s.level and RESULTS.match(s.title):
            in_results.update(range(k + 1, _span(secs, k)))
    methods_spans = []
    i = 0
    while i < len(secs):
        if secs[i].level and METHODS.match(secs[i].title) and i not in used and i not in in_results:
            j = _span(secs, i)
            methods_spans.append((i, j))
            used.update(range(i, j))
            i = j
        else:
            i += 1
    methods_chars = sum(len(s.body) + len(s.title) for a, b in methods_spans for s in secs[a:b])
    info = {"trim_version": TRIM_VERSION, "full_chars": len(text), "methods_share": round(methods_chars / total, 3),
            "bare_section_names": promoted}
    if not methods_spans:
        return text, dict(info, view="full", fallback="no methods heading found")
    if not MIN_METHODS_SHARE <= methods_chars / total <= MAX_METHODS_SHARE:
        return text, dict(info, view="full", fallback=f"methods are {methods_chars / total:.0%} of the text")

    parts, omitted = [], []
    title = next((s.title for s in secs if s.level == 1), "")
    if title:
        parts.append(f"# {title}")
    for k, s in enumerate(secs):
        if s.level and ABSTRACT.match(s.title) and k not in used:
            parts.append(f"## {s.title}\n\n{s.body}")
            used.add(k)
    for a, b in methods_spans:
        for s in secs[a:b]:
            parts.append(f"{'#' * s.level} {s.title}\n\n{s.body}".rstrip())
    k = 0
    while k < len(secs):
        if secs[k].level and RESULTS.match(secs[k].title) and k not in used:
            j = _span(secs, k)
            for s in secs[k:j]:
                first = _first_paragraph(s.body)
                parts.append(f"{'#' * s.level} {s.title}" + (f"\n\n{first}" if first else ""))
                if len(first) < len(s.body):
                    parts.append("[rest of this results section omitted]")
            used.update(range(k, j))
            k = j
        else:
            k += 1
    for k, s in enumerate(secs):
        if k not in used and s.level and s.level <= 2 and s.title and s.title != title:
            omitted.append(s.title)
    tables = _tables(tables_dir)
    if tables:
        parts.append("## Tables (all tables of the paper)\n\n" + "\n\n".join(tables))
    note = (f"[Trimmed text, trim v{TRIM_VERSION}: title, abstract, methods in full, the first paragraph "
            f"under each results heading, and every table. Other sections are omitted"
            + (f" ({'; '.join(omitted[:12])}{'; ...' if len(omitted) > 12 else ''})" if omitted else "")
            + ". The full text is the item's full_text_file.]")
    out = note + "\n\n" + "\n\n".join(parts) + "\n"
    return out, dict(info, view="trimmed", trimmed_chars=len(out), omitted_sections=len(omitted))


def write_trimmed(doc_dir: Path) -> Tuple[Path, dict]:
    """Write doc_dir/text.trimmed.md (or reuse the full text on a fallback) and return the
    path the judge should read, with the trim info."""
    text = (doc_dir / "text.md").read_text(encoding="utf-8")
    out, info = trim(text, doc_dir / "tables")
    if info["view"] == "full":
        return doc_dir / "text.md", info
    path = doc_dir / "text.trimmed.md"
    path.write_text(out, encoding="utf-8")
    return path, info


def coordinate_space_context(text: str) -> Tuple[str, dict]:
    """Verbatim Methods plus space-evidence paragraphs outside Methods.

    Reuse screening's heading/span detection, but never use its full-paper
    fallback. If Methods cannot be identified reliably, preserve matching source
    paragraphs only. Do not infer a coordinate space or silently truncate spans.
    Line numbers always refer to the original normalized text, including when
    bare section names were promoted for detection.
    """
    secs = sections(text)
    if not any(s.level and METHODS.match(s.title) for s in secs):
        secs = sections(promote_bare_sections(text))
    lines = text.splitlines()
    in_results = set()
    for i, sec in enumerate(secs):
        if sec.level and RESULTS.match(sec.title):
            in_results.update(range(i + 1, _span(secs, i)))
    spans = []
    used = set()
    for i, sec in enumerate(secs):
        if sec.level and METHODS.match(sec.title) and i not in used and i not in in_results:
            j = _span(secs, i)
            end = secs[j].start if j < len(secs) else len(lines)
            spans.append((sec.start, end, sec.title, "methods"))
            used.update(range(i, j))
    methods_chars = sum(len("\n".join(lines[a:b])) for a, b, _, _ in spans)
    fallback = None
    if not spans:
        fallback = "no methods heading found; coordinate-space paragraphs only"
    elif methods_chars / max(len(text), 1) > MAX_METHODS_SHARE:
        spans = []
        fallback = "implausible methods span; coordinate-space paragraphs only"
    for i, sec in enumerate(secs):
        # References cannot establish this study's output space.
        if re.match(r"^(?:references|bibliography|literature cited)\b", sec.title, re.I):
            continue
        end = secs[i + 1].start if i + 1 < len(secs) else len(lines)
        start = sec.start + (1 if sec.level else 0)
        cursor = start
        while cursor < end:
            while cursor < end and not lines[cursor].strip():
                cursor += 1
            stop = cursor
            while stop < end and lines[stop].strip():
                stop += 1
            body = "\n".join(lines[cursor:stop])
            if body and _SPACE_EVIDENCE.search(sec.title + "\n" + body) and not any(a <= cursor and stop <= b for a, b, _, _ in spans):
                spans.append((cursor, stop, sec.title or "preamble", "space_paragraph"))
            cursor = stop + 1
    spans.sort()
    parts = [f"[Source coordinate-space context v{SPACE_CONTEXT_VERSION}; excerpts from normalized text.md. "
             "Methods plus coordinate-system paragraphs; no space inferred.]"]
    excerpts = []
    for a, b, title, kind in spans:
        parts.append(f"\n[Source section: {title}; lines {a + 1}-{b}; {kind}]\n" + "\n".join(lines[a:b]))
        excerpts.append({"section": title, "line_start": a + 1, "line_end": b, "kind": kind})
    if not spans:
        parts.append("[No Methods or coordinate-space evidence found; leave unsupported space null.]")
    out = "\n".join(parts) + "\n"
    return out, {"version": SPACE_CONTEXT_VERSION, "source_chars": len(text),
                 "context_chars": len(out), "fallback": fallback, "excerpts": excerpts}


def write_coordinate_space_context(doc_dir: Path) -> Tuple[Path, dict]:
    text = (doc_dir / "text.md").read_text(encoding="utf-8")
    out, info = coordinate_space_context(text)
    path = doc_dir / "text.coordinate-space.md"
    path.write_text(out, encoding="utf-8")
    return path, info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("review_dir", type=Path)
    ap.add_argument("--pmid", help="print one document's trimmed text")
    args = ap.parse_args(argv)
    docs = args.review_dir / "docs"
    if args.pmid:
        out, info = trim((docs / args.pmid / "text.md").read_text(), docs / args.pmid / "tables")
        print(out)
        print(json.dumps(info), file=sys.stderr)
        return 0
    by: dict = {}
    for d in sorted(p for p in docs.iterdir() if (p / "text.md").exists()):
        source = json.loads((d / "meta.json").read_text()).get("source") if (d / "meta.json").exists() else "?"
        text = (d / "text.md").read_text()
        out, info = trim(text, d / "tables")
        tables = sum(len(t) for t in _tables(d / "tables"))
        s = by.setdefault(source, {"docs": 0, "fallback": 0, "full_chars": 0, "judge_chars": 0})
        s["docs"] += 1
        s["fallback"] += info["view"] == "full"
        # What a judge reads today: the text, then the tables it opens (counted in full).
        s["full_chars"] += len(text) + tables
        s["judge_chars"] += len(out) + (tables if info["view"] == "full" else 0)
    for s in by.values():
        s["kept"] = round(s["judge_chars"] / max(s["full_chars"], 1), 3)
    print(json.dumps(by, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

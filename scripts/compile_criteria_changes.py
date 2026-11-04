#!/usr/bin/env python3
"""Every criteria change on every project's version ladder, as a spreadsheet to annotate.

WHY

The tier ladder (`initial` -> `manual` -> `untuned` -> `best`) says WHICH run sits at each level
of benchmark exposure. It does not say WHAT CHANGED between them. That distinction carries the
overfitting argument: "the criteria were revised" means something very different depending on
whether the revision restored something the source paper states and the transcriber missed,
corrected a misreading, or invented a rule to catch papers the benchmark marked wrong.

This pulls every change out mechanically and leaves two columns for the author to fill in.

    change_type   what the change IS, measured against the source paper:
                    source-uncovered      paper states it; the config never encoded it
                    source-misencoded     config encoded it wrongly; this corrects it
                    source-underspecified paper is vague; this operationalises it, same scope
                    no-source-basis       no counterpart in the paper -- tuning
                    non-criteria          wording/plumbing only; asks nothing new of a study

    provenance    who made it and what they had seen:
                    human-paper     author, source paper only
                    human-reports   author, having looked at error reports (usually <5 papers)
                    llm-paper       LLM-drafted from the paper, no benchmark output in context
                    llm-reports     LLM-drafted with error reports in context -- CONFOUNDED:
                                    you cannot tell from the artifact whether the paper or the
                                    error list surfaced a given criterion

Keeping the axes separate is the point. A change can be llm-reports in provenance and still be
source-uncovered in content -- which is what vbm_of_ptsd v1 -> v2 turned out to be.

`source_overlap` is a crude content-word overlap against the registry's one-line summary of the
paper's criteria. A HINT, not a verdict: a real restoration scores low whenever the paper used
different words. The full source text is on the `source_criteria` sheet for judging by eye.

USAGE

    scripts/compile_criteria_changes.py                    # -> reports/criteria_changes.{csv,xlsx}
    scripts/compile_criteria_changes.py --project dementia
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "neurometabench" / "data" / "meta_datasets.csv"

# The six lists that carry criteria. `annotation.metadata_fields` and `retrieval.sources` are also
# lists of strings but ask nothing of a study, so they are not criteria.
CRITERIA_PATHS = [
    ("screening", "abstract", "inclusion_criteria"),
    ("screening", "abstract", "exclusion_criteria"),
    ("screening", "fulltext", "inclusion_criteria"),
    ("screening", "fulltext", "exclusion_criteria"),
    ("annotation", "inclusion_criteria"),
    ("annotation", "exclusion_criteria"),
]

# Above this, a removed/added pair reads as one criterion REWORDED rather than an independent
# deletion plus addition. 0.60 picked against the real diffs: reworded pairs sit at 0.7-0.9,
# unrelated criteria in the same block rarely clear 0.5.
REWORD_RATIO = 0.60

STOPWORDS = frozenset("""
a an and are as at be been but by for from had has have if in into is it its of on or that the
their them then there these this to was were which while with without not no only also any all
must should may can when where who whom whose than other others same such each both more most
study studies paper papers report reports reported reporting include included including exclude
excluded excluding criterion criteria analysis analyses results result group groups subject
subjects participant participants
""".split())

# ---------------------------------------------------------------- my best guess per transition
#
# ONE ROW PER SCHEMA, not per criterion: the author annotates a whole v(N) -> v(N+1) step.
#
# `confidence` is mine and it is not decoration. high = the target config's own header states
# the answer, or I verified it against outcomes. medium = provenance is stated outright but the
# CONTENT label is inferred from the diff. low = the header is one terse line and the label is
# mostly inference; treat these as prompts for the author, not as findings.
GUESS: dict[tuple[str, str, str], dict[str, str]] = {
    ("cue_reactivity", "v1", "v2"): dict(
        change_type="source-misencoded", provenance="human-paper", confidence="high",
        reasoning="The header ('correct error that only healthy adult participants should be "
                  "included') reads as if a restriction were being ADDED. The diff goes the "
                  "other way: v2 REMOVES it. 'from the healthy/control group and is "
                  "generalizable to the healthy adult population' is deleted from annotation "
                  "inclusion, and 'without original fMRI data in healthy adults' becomes "
                  "'without original human fMRI data'. Checked against the source paper (pmid "
                  "34400176): its stated criteria impose NO population restriction at all -- "
                  "they are about visual cue-reactivity, reported coordinates, and whole-brain "
                  "or SVC within-participant contrasts. So v1 invented a healthy-adult limit "
                  "the paper does not have, and v2 repairs that. Note a second, looser edit in "
                  "the same step: 'drug or natural reward-related cue-reactivity' is shortened "
                  "to 'reward-related cue-reactivity', which drops a distinction the paper does "
                  "make -- worth a second look."),
    ("cue_reactivity", "v2", "v3"): dict(
        change_type="source-underspecified", provenance="human-paper", confidence="low",
        reasoning="Header: 'explicitly revoke correlational analyses and include between group "
                  "contrasts.' Looks like operationalising which contrast types count, a "
                  "distinction the source paper likely assumes rather than spells out. No "
                  "mention of reports."),
    ("cue_reactivity", "v3", "v4"): dict(
        change_type="source-underspecified", provenance="human-reports", confidence="low",
        reasoning="Header: 'revert correlational analysis exclusion criteria' -- it undoes v3. A "
                  "revert usually means the restriction was seen to cost something, which "
                  "implies report inspection, but the header does not say so. Author input "
                  "needed on whether anything was looked at."),
    ("cue_reactivity", "v4", "v5"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="Header is explicit: 'GPT edited this schema. Modified from v4 using feedback "
                  "from reports (annotated inspection of false negatives primarily) ... it is "
                  "benchmark-tuned.' Provenance certain. Content guessed: criteria written to "
                  "recover specific false negatives usually have no counterpart in the paper, "
                  "but I have not checked each one against the source text the way I did for "
                  "vbm_of_ptsd."),
    ("decision_making", "v1", "v2"): dict(
        change_type="source-underspecified", provenance="human-paper", confidence="high",
        reasoning="Header states it outright: author-edited with ChatGPT as a phrasing tool, and "
                  "'NOT BENCHMARK-TUNED' -- an earlier header claimed otherwise and the claim "
                  "was withdrawn because the diff does not support it. ChatGPT as an editing "
                  "tool keeps this human-paper, not llm-paper."),
    ("decision_making", "v2", "v3"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="Header: Claude, inputs were the source paper AND v2's error reports, 'Tuned "
                  "against the benchmark'. Provenance certain; content not individually checked."),
    ("dementia", "v1", "v2"): dict(
        change_type="source-misencoded", provenance="human-paper", confidence="low",
        reasoning="Header is one line ('changes in inclusion'). Labelled from the author's own "
                  "account that dementia is the project where the first transcription was wrong "
                  "on the face of the source text. Needs confirming against the diff."),
    ("dementia", "v2", "v3"): dict(
        change_type="source-underspecified", provenance="human-reports", confidence="low",
        reasoning="Header: 'only changes to annotation section'. Nothing else to go on."),
    ("dementia", "v3", "v4"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="Header: Claude, 'tuned against the benchmark', inputs included v3's 94-paper "
                  "false-positive report. Hurt recall and was not promoted to best."),
    ("emotion_regulation_2022", "v2", "v4"): dict(
        change_type="no-source-basis", provenance="human-reports", confidence="high",
        reasoning="The clearest tuning case in the corpus, and the header says so without "
                  "flinching: 'DELIBERATELY LESS FAITHFUL THAN THE PAPER, to buy recall.' It "
                  "then enumerates the exact gold PMIDs each faithful restriction was costing "
                  "(static-visual-stimuli -5, healthy-adults -5) and removes those restrictions. "
                  "Fitted to the realised gold list by construction -- and note it moves AWAY "
                  "from the paper's stated criteria, which is the opposite of vbm_of_ptsd."),
    ("executive_function", "v1", "v2"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="Header: Claude, inputs the source paper plus v1's false-negative report, "
                  "'tuned against the benchmark'. Relaxed full-text screening."),
    ("executive_function", "v2", "v3"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="Header: Claude, 'Benchmark-tuned'. Combines v2's screening relaxation with "
                  "the annotation relaxation from v3-annotation-only."),
    ("problem_solving", "v1", "v2"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="high",
        reasoning="Header names the headline fix: 'logical_reasoning maps to the benchmark's "
                  "Demand_MNI_final set, which is difficulty manipulations (Complex > Simple), "
                  "not formal logic.' That is a criterion rewritten to match the BENCHMARK'S "
                  "label semantics, not the paper's -- no-source-basis almost by definition, "
                  "whatever its merits as a correction."),
    ("social", "v3", "v4"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="Header: Claude, 'tuned against the benchmark'. A null result -- full-text "
                  "0.406/0.995 -> 0.402/0.981, maps within +/-0.02."),
    ("vbm_of_ptsd", "v1", "v2"): dict(
        change_type="source-uncovered", provenance="llm-reports", confidence="high",
        reasoning="VERIFIED BY OUTCOME, not just by reading the header. Both added criteria are "
                  "in the source paper's own stated criteria (pmid 36100907): inclusion "
                  "'reporting results among adult humans', exclusion 'other non-voxel-based "
                  "morphometry methods'. v1 encodes neither. Exactly 6 papers flip "
                  "include->exclude and every one is caught by a plain reading of those two "
                  "criteria -- 5 paediatric (ages 8-18, 12-21, 8-18, 7-14, 8-19) and 1 "
                  "FreeSurfer/surface-based with no VBM contrast. Zero gold lost, zero reverse "
                  "flips, all 4 maps byte-identical. CONFOUND: the agent had the 11-FP report in "
                  "context, so the error list may be how it NOTICED the omission -- that is "
                  "provenance, not content. Not promoted (author's decision 2026-11-04) because "
                  "it is LLM-authored."),
    ("vbm_of_substance_use", "v1", "v2"): dict(
        change_type="source-misencoded", provenance="human-paper", confidence="high",
        reasoning="Header calls it 'wording only', but the two edits it quotes are substantive "
                  "mis-encodings being repaired: 'fMRI data' -> 'structural MRI data' (a plain "
                  "bug in a VBM project) and 'cue type' -> 'substance use characteristics' (a "
                  "leftover from the cue-reactivity config this one was templated from). No "
                  "criterion added or removed. Also note the header explicitly WITHDRAWS an "
                  "earlier 'Benchmark-tuned' claim as wrong."),
    ("vbm_of_substance_use", "v2", "v3"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="Header: Claude, 'tuned against the benchmark', v2's reports in context. Hurt "
                  "recall and was not promoted to best."),
}


# (legacy per-change map, retained for the CSV)
# Pre-filled annotations, keyed (project, from, to). Everything else ships blank for the author.
# A blank row is NOT a claim that the change was ordinary -- it means nobody has read that diff
# against the source paper yet.
ANNOTATIONS: dict[tuple[str, str, str], dict[str, str]] = {
    ("vbm_of_ptsd", "v1", "v2"): dict(
        change_type="source-uncovered",
        provenance="llm-reports",
        note=(
            "Benchmark tuning plus an LLM read of the paper surfaced two criteria the human "
            "annotator never encoded. Both are in the source paper's own stated criteria "
            '(pmid 36100907): inclusion "reporting results among adult humans"; exclusion '
            '"other non-voxel-based morphometry methods". v1 encodes neither. Verified by '
            "outcome: exactly 6 papers flip include->exclude, all caught by the plain reading of "
            "those two criteria (5 paediatric: ages 8-18, 12-21, 8-18, 7-14, 8-19; 1 "
            "FreeSurfer/surface-based with no VBM contrast). Zero gold lost, zero reverse flips, "
            "all 4 maps byte-identical. Confound: the agent had the 11-FP report in context, so "
            "the error list may be how it noticed the omission -- that is provenance, not "
            "content. Consequence: v1 is defective as this project's initial/untuned/best run. "
            "Not promoted (author's decision 2026-11-04) because it is LLM-authored."
        ),
    ),
}


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in STOPWORDS}


def meta_pmid(project: str) -> str | None:
    m = REPO_ROOT / "projects" / project / "nmb_mappings.json"
    if m.exists():
        try:
            payload = json.loads(m.read_text())
        except json.JSONDecodeError:
            payload = {}
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key in ("meta_pmid", "pmid", "meta_analysis_pmid"):
                    if isinstance(node.get(key), (str, int)) and re.fullmatch(r"\d{7,8}", str(node[key])):
                        return str(node[key])
                stack += list(node.values())
            elif isinstance(node, list):
                stack += node
    for cfg in sorted((REPO_ROOT / "projects" / project).glob("v*.yaml")):
        hit = re.search(r'meta_pmid:\s*["\']?(\d{7,8})', cfg.read_text())
        if hit:
            return hit.group(1)
    return None


def source_criteria(project: str) -> dict:
    pmid = meta_pmid(project)
    if not pmid or not BENCH.exists():
        return {}
    for row in csv.DictReader(BENCH.open()):
        if str(row.get("pmid")) == str(pmid):
            return {"project": project, "pmid": pmid,
                    "inclusion": (row.get("inclusion") or "").strip(),
                    "exclusion": (row.get("exclusion") or "").strip()}
    return {"project": project, "pmid": pmid, "inclusion": "", "exclusion": ""}


def ladder(project: str) -> list[tuple[str, Path]]:
    """Plain `vN.yaml` in version order. Variant configs (-annotation-only, -allstudies, -pondie-*,
    -jev, -portkey-test, -2010) vary the POOL or the PLUMBING at a fixed criteria set, so putting
    them on the criteria ladder would report phantom changes at every branch point."""
    out = []
    for p in (REPO_ROOT / "projects" / project).glob("v*.yaml"):
        m = re.fullmatch(r"v(\d+)\.yaml", p.name)
        if m:
            out.append((int(m.group(1)), f"v{m.group(1)}", p))
    return [(name, path) for _, name, path in sorted(out)]


def criteria_of(path: Path) -> dict[str, list[str]]:
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        print(f"  !! {path.name}: {exc}", file=sys.stderr)
        return {}
    out = {}
    for keys in CRITERIA_PATHS:
        node = cfg
        for k in keys:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, list):
            out[".".join(keys)] = [str(x) for x in node]
    return out


def diff_block(old: list[str], new: list[str]) -> list[dict]:
    removed = [x for x in old if x not in set(new)]
    added = [x for x in new if x not in set(old)]
    changes, used = [], set()
    for r in removed:
        best, best_ratio, best_i = None, 0.0, -1
        for i, a in enumerate(added):
            if i in used:
                continue
            ratio = difflib.SequenceMatcher(None, r, a).ratio()
            if ratio > best_ratio:
                best, best_ratio, best_i = a, ratio, i
        if best is not None and best_ratio >= REWORD_RATIO:
            used.add(best_i)
            changes.append({"kind": "reworded", "before": r, "after": best,
                            "similarity": round(best_ratio, 3)})
        else:
            changes.append({"kind": "removed", "before": r, "after": "", "similarity": ""})
    for i, a in enumerate(added):
        if i not in used:
            changes.append({"kind": "added", "before": "", "after": a, "similarity": ""})
    return changes


def overlap(text: str, src: dict) -> tuple[float, str]:
    words = content_words(text)
    if not src or not words:
        return 0.0, ""
    best_field, best = "", 0.0
    for field in ("inclusion", "exclusion"):
        sw = content_words(src.get(field, ""))
        if sw:
            score = len(words & sw) / len(words)
            if score > best:
                best, best_field = score, field
    return round(best, 3), best_field


def collect(project: str) -> list[dict]:
    src = source_criteria(project)
    rungs = ladder(project)
    rows = []
    for (a_name, a_path), (b_name, b_path) in zip(rungs, rungs[1:]):
        a, b = criteria_of(a_path), criteria_of(b_path)
        for block in sorted(set(a) | set(b)):
            for ch in diff_block(a.get(block, []), b.get(block, [])):
                text = ch["after"] or ch["before"] or ""
                score, field = overlap(text, src)
                ann = ANNOTATIONS.get((project, a_name, b_name), {})
                rows.append({
                    "project": project, "from": a_name, "to": b_name,
                    "stage": block.replace("_criteria", "").replace("screening.", ""),
                    "kind": ch["kind"],
                    "change_type": ann.get("change_type", ""),
                    "provenance": ann.get("provenance", ""),
                    "note": ann.get("note", ""),
                    "source_overlap": score, "source_field": field,
                    "similarity": ch.get("similarity", ""),
                    "before": " ".join(str(ch["before"]).split()),
                    "after": " ".join(str(ch["after"]).split()),
                })
    return rows


HEADER_LINES = 14   # how much of a config's banner to quote as evidence


def header_quote(project: str, version: str) -> str:
    """The target config's own provenance banner. This is the single most useful column for
    annotating: in most projects the author or the agent already stated what the change was."""
    f = REPO_ROOT / "projects" / project / f"{version}.yaml"
    if not f.exists():
        return ""
    out = []
    for ln in f.read_text().split("\n")[:40]:
        if ln.startswith("#"):
            out.append(ln.lstrip("#").strip())
        elif out:
            break
    return " ".join(x for x in out[:HEADER_LINES] if x)


COLUMNS = ["project", "from", "to", "n_changes", "blocks", "added", "removed", "reworded",
           "my_change_type", "my_provenance", "my_confidence", "my_reasoning",
           "your_change_type", "your_provenance", "your_note",
           "target_header", "changes"]

MINE = ("my_change_type", "my_provenance", "my_confidence", "my_reasoning")
YOURS = ("your_change_type", "your_provenance", "your_note")


def transition_rows(project: str) -> list[dict]:
    """One row per v(N) -> v(N+1) step, which is the unit the author annotates."""
    per_change = collect(project)
    rows = []
    for key in dict.fromkeys((r["from"], r["to"]) for r in per_change):
        ch = [r for r in per_change if (r["from"], r["to"]) == key]
        g = GUESS.get((project, *key), {})
        bullets = []
        for r in ch:
            sign = {"added": "+", "removed": "-", "reworded": "~"}[r["kind"]]
            if r["kind"] == "reworded":
                bullets.append(f"{sign} [{r['stage']}] {r['before']}\n    -> {r['after']}")
            else:
                bullets.append(f"{sign} [{r['stage']}] {r['after'] or r['before']}")
        rows.append({
            "project": project, "from": key[0], "to": key[1],
            "n_changes": len(ch),
            "blocks": ", ".join(sorted({r["stage"] for r in ch})),
            "added": sum(r["kind"] == "added" for r in ch),
            "removed": sum(r["kind"] == "removed" for r in ch),
            "reworded": sum(r["kind"] == "reworded" for r in ch),
            "my_change_type": g.get("change_type", ""),
            "my_provenance": g.get("provenance", ""),
            "my_confidence": g.get("confidence", ""),
            "my_reasoning": g.get("reasoning", ""),
            "your_change_type": "", "your_provenance": "", "your_note": "",
            "target_header": header_quote(project, key[1]),
            "changes": "\n\n".join(bullets),
        })
    return rows


def write_xlsx(path: Path, rows: list[dict], sources: list[dict]) -> bool:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        return False

    wb = Workbook()
    ws = wb.active
    ws.title = "schemas"
    head_font = Font(bold=True)
    yours = PatternFill("solid", fgColor="FFF2CC")   # for the author to fill
    mine = PatternFill("solid", fgColor="E8F1FB")    # my guess, overwrite freely
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = head_font
    for r in rows:
        ws.append([r.get(c, "") for c in COLUMNS])

    widths = {"project": 22, "from": 7, "to": 7, "n_changes": 11, "blocks": 26, "added": 8,
              "removed": 9, "reworded": 10, "my_change_type": 20, "my_provenance": 15,
              "my_confidence": 12, "my_reasoning": 85, "your_change_type": 20,
              "your_provenance": 15, "your_note": 45, "target_header": 70, "changes": 100}
    for i, c in enumerate(COLUMNS, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(c, 18)
    for col, fill in [(c, mine) for c in MINE] + [(c, yours) for c in YOURS]:
        letter = ws.cell(1, COLUMNS.index(col) + 1).column_letter
        for cell in ws[letter]:
            cell.fill = fill
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = ws.dimensions

    n = len(rows) + 1
    for col, opts in [
        ("your_change_type", "source-uncovered,source-misencoded,source-underspecified,"
                             "no-source-basis,non-criteria"),
        ("your_provenance", "human-paper,human-reports,llm-paper,llm-reports"),
    ]:
        dv = DataValidation(type="list", allow_blank=True, formula1=f'"{opts}"')
        ws.add_data_validation(dv)
        letter = ws.cell(1, COLUMNS.index(col) + 1).column_letter
        dv.add(f"{letter}2:{letter}{n}")

    sh = wb.create_sheet("source_criteria")
    sh.append(["project", "pmid", "inclusion", "exclusion"])
    for c in sh[1]:
        c.font = head_font
    for s_ in sources:
        sh.append([s_.get("project", ""), s_.get("pmid", ""), s_.get("inclusion", ""),
                   s_.get("exclusion", "")])
    for letter, width in (("A", 24), ("B", 12), ("C", 85), ("D", 85)):
        sh.column_dimensions[letter].width = width
    for row in sh.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    sh.freeze_panes = "A2"

    key = wb.create_sheet("key")
    key.append(["column", "value", "meaning"])
    for c in key[1]:
        c.font = head_font
    for v, m in [
        ("source-uncovered", "paper states it; the config never encoded it. A DEFECT IN THE "
                             "EARLIER CONFIG being repaired, not a refinement past the paper."),
        ("source-misencoded", "config encoded it wrongly -- contradicts or distorts the paper. "
                              "This corrects it."),
        ("source-underspecified", "paper is vague; this operationalises it at the SAME scope. "
                                  "Watch this one: an 'operationalisation' that quietly narrows "
                                  "or widens scope is really no-source-basis."),
        ("no-source-basis", "no counterpart in the paper. This is the category that makes a "
                            "result circular, and it is tuning whatever its provenance."),
        ("non-criteria", "wording/plumbing only; asks nothing new of a study."),
    ]:
        key.append(["change_type", v, m])
    for v, m in [
        ("human-paper", "author, source paper only. An LLM used as a PHRASING tool still counts "
                        "as human-paper."),
        ("human-reports", "author, having looked at error reports -- usually under five papers."),
        ("llm-paper", "LLM-drafted from the paper, no benchmark output in context."),
        ("llm-reports", "LLM-drafted with error reports in context. CONFOUNDED by construction: "
                        "you cannot tell from the artifact whether the paper or the error list "
                        "surfaced a given criterion. Axis A still decides whether the CONTENT is "
                        "defensible."),
    ]:
        key.append(["provenance", v, m])
    for v, m in [
        ("high", "the target config's header states the answer, or I verified it against outcomes"),
        ("medium", "provenance stated outright; the CONTENT label inferred from the diff"),
        ("low", "one terse header line; mostly inference. Prompts, not findings."),
    ]:
        key.append(["my_confidence", v, m])
    key.append(["target_header", "", "the target config's own provenance banner -- usually the "
                                     "fastest way to annotate a row"])
    key.append(["changes", "+ / - / ~", "added / removed / reworded, prefixed by the block"])
    for letter, width in (("A", 20), ("B", 24), ("C", 96)):
        key.column_dimensions[letter].width = width
    for row in key.iter_rows(min_row=2):
        row[2].alignment = Alignment(vertical="top", wrap_text=True)
    key.freeze_panes = "A2"

    wb.save(path)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(REPO_ROOT / "reports" / "criteria_changes.csv"))
    ap.add_argument("--xlsx", default=str(REPO_ROOT / "reports" / "criteria_changes.xlsx"))
    ap.add_argument("--project", nargs="*", default=None)
    args = ap.parse_args()

    projects = args.project or sorted(
        p.name for p in (REPO_ROOT / "projects").iterdir()
        if p.is_dir() and not p.name.startswith(("template", "_")) and list(p.glob("v*.yaml")))

    rows, sources = [], []
    for project in projects:
        rows += transition_rows(project)
        sources.append(source_criteria(project))

    with Path(args.csv).open("w", newline="") as fh:
        ww = csv.DictWriter(fh, fieldnames=COLUMNS)
        ww.writeheader()
        ww.writerows({c: r.get(c, "") for c in COLUMNS} for r in rows)

    ok = write_xlsx(Path(args.xlsx), rows, sources)
    guessed = sum(1 for r in rows if r["my_change_type"])
    print(f"wrote {Path(args.csv).relative_to(REPO_ROOT)}"
          + (f" and {Path(args.xlsx).relative_to(REPO_ROOT)}" if ok
             else "  (no openpyxl -- xlsx skipped)"))
    print(f"  {len(rows)} schema transitions, {sum(r['n_changes'] for r in rows)} criteria "
          f"changes, {guessed} with my guess filled in")
    from collections import Counter
    for label, c in (("change_type", Counter(r["my_change_type"] for r in rows if r["my_change_type"])),
                     ("confidence", Counter(r["my_confidence"] for r in rows if r["my_confidence"]))):
        print(f"  {label}: " + "  ".join(f"{k}={v}" for k, v in c.most_common()))
    missing = [(r["project"], r["from"], r["to"]) for r in rows if not r["my_change_type"]]
    for m in missing:
        print(f"  NO GUESS: {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

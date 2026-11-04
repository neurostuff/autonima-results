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

# --------------------------------------------------------------- benchmark registry defects
#
# Two rows of neurometabench/data/meta_datasets.csv cannot be used as reference text, and both
# were found by reading them rather than by any check. Passing them through silently would be
# worse than having no reference at all, because a wrong source text reads as authoritative:
# I came within one step of citing executive_function's contaminated exclusion field to justify
# that project's relaxation as source-grounded.
SOURCE_DEFECTS = {
    "executive_function": (
        "CONTAMINATED -- do not use. Row 22282036 is labelled correctly (topic 'Executive "
        "Function', first_author Niendam, 2012) but its search, inclusion and exclusion fields "
        "belong to a NICOTINE meta-analysis: search is ('fMRI' OR 'PET' OR 'neuroimaging') AND "
        "('nicotine' OR 'cigarette' OR 'smok*'), and inclusion describes nAChR agonist "
        "administration. Niendam 2012 is a cognitive-control meta-analysis and has nothing to do "
        "with nicotine. Any content judgement for this project needs the real paper (PMC3660731); "
        "it CANNOT be made from the registry."
    ),
    "emotion_regulation_2022": (
        "MISSING -- row 35413444 (Morawetz) has empty search and inclusion fields, so there is no "
        "reference text at all. Judge this project from the config headers only."
    ),
}


# ---------------------------------------------------------------- my best guess per transition
#
# ONE ROW PER TIER TRANSITION. Only steps that CHANGE CATEGORY are annotated: initial -> untuned
# and untuned -> benchmark_tuned, resolved from run_categories.yaml. Intermediate versions are
# not rows of their own -- their changes are folded into the cumulative diff and their headers
# are quoted as evidence, because an intermediate's banner often explains the provenance of the
# version that follows it.
#
# `confidence` is mine and it is not decoration. high = verified against the source paper or the
# config states it outright. medium = provenance is stated but the CONTENT label is inferred.
# low = terse headers, missing reference text, or mostly inference -- prompts, not findings.
GUESS: dict[tuple[str, str, str], dict[str, str]] = {
    ("cue_reactivity", "v1", "v4"): dict(
        change_type="source-misencoded", provenance="human-paper", confidence="high",
        reasoning="Net of three hand edits. The substantive one is v2 REMOVING a healthy-adult "
                  "restriction: 'from the healthy/control group and is generalizable to the "
                  "healthy adult population' is deleted, and 'without original fMRI data in "
                  "healthy adults' becomes '...original human fMRI data'. The source paper (pmid "
                  "34400176) imposes NO population restriction -- its criteria are about visual "
                  "cue-reactivity, reported coordinates, and whole-brain or SVC within-"
                  "participant contrasts. So v1 invented the limit and v2 repaired it. NOTE the "
                  "v2 header reads 'correct error that only healthy adult participants should be "
                  "included', which sounds like the opposite; the diff settles it. v3 added a "
                  "correlational-analysis exclusion and v4 reverted it, so they CANCEL and leave "
                  "no trace in the cumulative diff -- the churn is invisible here by design. One "
                  "loosening to check: 'drug or natural reward-related cue-reactivity' is "
                  "shortened to 'reward-related cue-reactivity', dropping a distinction the "
                  "paper does make."),
    ("cue_reactivity", "v4", "v6"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="v5's header is explicit: 'GPT edited this schema. Modified from v4 using "
                  "feedback from reports (annotated inspection of false negatives primarily) ... "
                  "it is benchmark-tuned.' v6 inherits v5's criteria unchanged and changes only "
                  "the search, itself chosen by reading which gold studies v4 missed. Provenance "
                  "certain; content inferred -- criteria written to recover named false "
                  "negatives usually have no counterpart in the paper, but I did not check each "
                  "against the source text."),
    ("decision_making", "v1", "v2"): dict(
        change_type="source-underspecified", provenance="human-paper", confidence="high",
        reasoning="The header states it: author-edited with ChatGPT as a phrasing tool, and "
                  "'NOT BENCHMARK-TUNED'. An earlier header claimed otherwise and the claim was "
                  "withdrawn because the diff does not support it. An LLM used as an editing "
                  "tool keeps this human-paper, not llm-paper."),
    ("decision_making", "v2", "v3"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="Header: Claude, inputs the source paper AND v2's error reports, 'Tuned "
                  "against the benchmark'. Provenance certain; content not individually checked."),
    ("dementia", "v1", "v3"): dict(
        change_type="source-uncovered", provenance="human-reports", confidence="medium",
        reasoning="REVISED after reading the cumulative diff against the source (pmid 35664889), "
                  "which I had not done when I first guessed source-misencoded. Most of the 18 "
                  "changes RESTORE criteria the paper states and v1 omitted: 'Analysis must "
                  "compare the target bvFTD group to healthy controls' answers the paper's "
                  "exclusion 'No Control'; the new exclusion for pooling bvFTD with other "
                  "patient groups answers its 'no other forms of dementia'; the >=6-participant "
                  "rule and the VBM/fMRI/FDG-PET modality restriction are both stated inclusion "
                  "criteria. TWO EXCEPTIONS worth your eye. (1) 'Study should not focus on "
                  "Parkinson's disease' has no counterpart in the paper -- likely report-driven. "
                  "(2) The participant rule is made STRICTER than the source: the paper says "
                  "'at least six participants in EITHER the patient or healthy group', v3 "
                  "requires >=6 in the patient group AND >=6 in the control group if present. "
                  "A mixed row; the dominant pattern is restoration."),
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
                  "Fitted to the realised gold list by construction, and note it moves AWAY from "
                  "the paper's stated criteria -- the opposite of vbm_of_ptsd. Judged from the "
                  "header alone: this project's registry row has no criteria text at all."),
    ("executive_function", "v1", "v3"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="low",
        reasoning="Provenance is certain -- v2 and v3 both carry 'AUTOMATICALLY GENERATED by "
                  "Claude', 'tuned against the benchmark', inputs including v1's false-negative "
                  "report. CONTENT CANNOT BE JUDGED FROM THE REGISTRY: row 22282036's criteria "
                  "fields are contaminated with a nicotine meta-analysis (see SOURCE_DEFECTS), "
                  "so the reference text is unusable and my label is inference from the shape of "
                  "the diff, not a reading against the paper. The diff relaxes hard: the 18-60 "
                  "healthy-only restriction goes, 'any participant group qualifies, including "
                  "clinical and developmental samples'; the active-control requirement goes, "
                  "rest or fixation now qualifies. Several additions are annotator INSTRUCTIONS "
                  "rather than study criteria ('You are judging ONE analysis, not the paper', "
                  "'How to read the analysis'), which may deserve non-criteria instead. Settle "
                  "this against the real Niendam 2012 (PMC3660731), not the registry."),
    ("problem_solving", "v1", "v2"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="high",
        reasoning="The header names the headline fix: 'logical_reasoning maps to the benchmark's "
                  "Demand_MNI_final set, which is difficulty manipulations (Complex > Simple), "
                  "not formal logic.' That is a criterion rewritten to match the BENCHMARK'S "
                  "label semantics rather than the paper's -- no-source-basis almost by "
                  "definition, whatever its merits as a correction."),
    ("social", "v3", "v4"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="Header: Claude, inputs the source paper plus v3's criterion-level attribution "
                  "tables, 'tuned against the benchmark'. A null result -- full-text 0.406/0.995 "
                  "-> 0.402/0.981, maps within +/-0.02."),
    ("vbm_of_ptsd", "v1", "v2"): dict(
        change_type="source-uncovered", provenance="llm-reports", confidence="high",
        reasoning="VERIFIED BY OUTCOME, not just by reading the header. Both added criteria are "
                  "in the source paper's stated criteria (pmid 36100907): inclusion 'reporting "
                  "results among adult humans', exclusion 'other non-voxel-based morphometry "
                  "methods'. v1 encodes neither. Exactly 6 papers flip include->exclude and "
                  "every one is caught by a plain reading of those two criteria -- 5 paediatric "
                  "(ages 8-18, 12-21, 8-18, 7-14, 8-19) and 1 FreeSurfer/surface-based with no "
                  "VBM contrast. Zero gold lost, zero reverse flips, all 4 maps byte-identical. "
                  "CONFOUND: the agent had the 11-FP report in context, so the error list may be "
                  "how it NOTICED the omission -- that is provenance, not content. Not promoted "
                  "(author's decision 2026-11-04) because it is LLM-authored."),
    ("vbm_of_substance_use", "v1", "v2"): dict(
        change_type="source-misencoded", provenance="human-paper", confidence="high",
        reasoning="The header calls it 'wording only', but the two edits it quotes are "
                  "substantive mis-encodings being repaired: 'fMRI data' -> 'structural MRI "
                  "data' (a plain bug in a VBM project) and 'cue type' -> 'substance use "
                  "characteristics' (a leftover from the cue-reactivity config this one was "
                  "templated from). No criterion added or removed. The header also explicitly "
                  "WITHDRAWS an earlier 'Benchmark-tuned' claim as wrong."),
    ("vbm_of_substance_use", "v2", "v3"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        reasoning="Header: Claude, 'tuned against the benchmark', v2's reports in context. Hurt "
                  "recall and was not promoted to best."),
}


# ----------------------------------------------------------------------------------- loading

def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in STOPWORDS}


def meta_pmid(project: str) -> str | None:
    """Project -> source meta-analysis PMID, from nmb_mappings.json or any config's meta_pmid."""
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
                    v = node.get(key)
                    if isinstance(v, (str, int)) and re.fullmatch(r"\d{7,8}", str(v)):
                        return str(v)
                stack += list(node.values())
            elif isinstance(node, list):
                stack += node
    for cfg in sorted((REPO_ROOT / "projects" / project).glob("v*.yaml")):
        hit = re.search(r'meta_pmid:\s*["\']?(\d{7,8})', cfg.read_text())
        if hit:
            return hit.group(1)
    return None


def source_criteria(project: str) -> dict:
    """The source meta-analysis's own stated criteria, plus whether they can be trusted.

    `usable` is false for a project in SOURCE_DEFECTS and for any row whose criteria fields are
    empty. Callers must NOT fall back to the raw text when it is false: a wrong reference text
    reads as authoritative, which is strictly worse than having none.
    """
    pmid = meta_pmid(project)
    defect = SOURCE_DEFECTS.get(project, "")
    base = {"project": project, "pmid": pmid or "", "inclusion": "", "exclusion": "",
            "defect": defect, "usable": not defect}
    if not pmid or not BENCH.exists():
        return {**base, "usable": False,
                "defect": defect or "no registry row found for this project"}
    for row in csv.DictReader(BENCH.open()):
        if str(row.get("pmid")) == str(pmid):
            inc = (row.get("inclusion") or "").strip()
            exc = (row.get("exclusion") or "").strip()
            if not inc and not exc and not defect:
                defect = "registry row has empty inclusion and exclusion fields"
            return {**base, "inclusion": inc, "exclusion": exc,
                    "defect": defect, "usable": not defect}
    return {**base, "usable": False, "defect": defect or "pmid not present in the registry"}


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
    """Added / removed / reworded criteria in one block. Reworded pairs are matched greedily by
    best similarity so a lightly edited criterion is not reported as a delete plus an unrelated
    add."""
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


TIER_LADDER = ("initial", "untuned", "benchmark_tuned")


def header_quote(project: str, version: str, lines: int = 14) -> str:
    """A config's own provenance banner -- usually the fastest way to annotate a row."""
    f = REPO_ROOT / "projects" / project / f"{version}.yaml"
    if not f.exists():
        return ""
    out = []
    for ln in f.read_text().split("\n")[:40]:
        if ln.startswith("#"):
            out.append(ln.lstrip("#").strip())
        elif out:
            break
    return " ".join(x for x in out[:lines] if x)


def vnum(name: str) -> int:
    m = re.match(r"v(\d+)", str(name))
    return int(m.group(1)) if m else -1


def tier_transitions(project: str) -> list[dict]:
    """One row per CATEGORY CHANGE on the canonical ladder.

    Only initial -> untuned and untuned -> benchmark_tuned are rows. A tier that resolves to the
    same run as the previous one contributes no row (executive_function's initial and untuned are
    both v1, so it has a single row). The diff is CUMULATIVE -- start config against end config
    directly -- and every version strictly between them has its header quoted, because an
    intermediate's banner routinely explains the provenance of the version after it.
    """
    from run_tiers import resolve_tier
    src = source_criteria(project)
    rungs = []
    for tier in TIER_LADDER:
        run = resolve_tier(project, "canonical", tier)
        if run and (not rungs or rungs[-1][1] != run):
            rungs.append((tier, run))

    rows = []
    for (ta, a), (tb, b) in zip(rungs, rungs[1:]):
        pa = REPO_ROOT / "projects" / project / f"{a}.yaml"
        pb = REPO_ROOT / "projects" / project / f"{b}.yaml"
        if not (pa.exists() and pb.exists()):
            continue
        A, B = criteria_of(pa), criteria_of(pb)
        changes, bullets = [], []
        for block in sorted(set(A) | set(B)):
            for ch in diff_block(A.get(block, []), B.get(block, [])):
                stage = block.replace("_criteria", "").replace("screening.", "")
                changes.append({**ch, "stage": stage})
                sign = {"added": "+", "removed": "-", "reworded": "~"}[ch["kind"]]
                if ch["kind"] == "reworded":
                    bullets.append(f"{sign} [{stage}] {' '.join(str(ch['before']).split())}"
                                   f"\n    -> {' '.join(str(ch['after']).split())}")
                else:
                    bullets.append(f"{sign} [{stage}] "
                                   f"{' '.join(str(ch['after'] or ch['before']).split())}")

        between = [p.stem for p in sorted((REPO_ROOT / "projects" / project).glob("v*.yaml"))
                   if re.fullmatch(r"v\d+", p.stem) and vnum(a) < vnum(p.stem) < vnum(b)]
        inter = "\n\n".join(f"[{v}] {header_quote(project, v, 8)}" for v in between)

        g = GUESS.get((project, a, b), {})
        rows.append({
            "project": project, "from_tier": ta, "to_tier": tb, "from": a, "to": b,
            "intermediate": ", ".join(between) or "-",
            "n_changes": len(changes),
            "blocks": ", ".join(sorted({c["stage"] for c in changes})),
            "added": sum(c["kind"] == "added" for c in changes),
            "removed": sum(c["kind"] == "removed" for c in changes),
            "reworded": sum(c["kind"] == "reworded" for c in changes),
            "my_change_type": g.get("change_type", ""),
            "my_provenance": g.get("provenance", ""),
            "my_confidence": g.get("confidence", ""),
            "my_reasoning": g.get("reasoning", ""),
            "your_change_type": "", "your_provenance": "", "your_note": "",
            "source_usable": "NO -- see source_criteria sheet" if not src.get("usable") else "yes",
            "target_header": header_quote(project, b),
            "intermediate_headers": inter or "-",
            "changes": "\n\n".join(bullets),
        })
    return rows


COLUMNS = ["project", "from_tier", "to_tier", "from", "to", "intermediate", "n_changes",
           "blocks", "added", "removed", "reworded",
           "my_change_type", "my_provenance", "my_confidence", "my_reasoning",
           "your_change_type", "your_provenance", "your_note",
           "source_usable", "target_header", "intermediate_headers", "changes"]

MINE = ("my_change_type", "my_provenance", "my_confidence", "my_reasoning")
YOURS = ("your_change_type", "your_provenance", "your_note")


def write_xlsx(path: Path, rows: list[dict], sources: list[dict]) -> bool:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        return False

    wb = Workbook()
    ws = wb.active
    ws.title = "tier_transitions"
    head_font = Font(bold=True)
    yours = PatternFill("solid", fgColor="FFF2CC")   # for the author to fill
    mine = PatternFill("solid", fgColor="E8F1FB")    # my guess, overwrite freely
    ws.append(COLUMNS)
    for c in ws[1]:
        c.font = head_font
    for r in rows:
        ws.append([r.get(c, "") for c in COLUMNS])

    widths = {"project": 22, "from_tier": 14, "to_tier": 16, "from": 7, "to": 7,
              "intermediate": 14, "n_changes": 11, "blocks": 26, "added": 8, "removed": 9,
              "reworded": 10, "my_change_type": 20, "my_provenance": 15, "my_confidence": 12,
              "my_reasoning": 90, "your_change_type": 20, "your_provenance": 15,
              "your_note": 45, "source_usable": 16, "target_header": 70,
              "intermediate_headers": 70, "changes": 100}
    for i, c in enumerate(COLUMNS, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(c, 18)
    for col, fill in [(c, mine) for c in MINE] + [(c, yours) for c in YOURS]:
        letter = ws.cell(1, COLUMNS.index(col) + 1).column_letter
        for cell in ws[letter]:
            cell.fill = fill
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "F2"
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
    sh.append(["project", "pmid", "status", "defect", "inclusion", "exclusion"])
    for c in sh[1]:
        c.font = head_font
    for s_ in sources:
        sh.append([s_.get("project", ""), s_.get("pmid", ""),
                   "OK" if s_.get("usable") else "DEFECT",
                   s_.get("defect", ""), s_.get("inclusion", ""), s_.get("exclusion", "")])
    for letter, width in (("A", 24), ("B", 12), ("C", 10), ("D", 80), ("E", 80), ("F", 80)):
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
        rows += tier_transitions(project)
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
    print(f"  {len(rows)} tier transitions, {sum(r['n_changes'] for r in rows)} cumulative "
          f"criteria changes, {guessed} with my guess filled in")
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

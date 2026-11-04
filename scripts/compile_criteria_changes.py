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
                    source-unenforced     paper STATES the rule but its own included studies
                                          violate it; the change matches the paper's PRACTICE
                                          over its TEXT. Checkable against the gold list, and
                                          that check is what separates it from no-source-basis
                    no-source-basis       no counterpart in the paper's criteria OR its realised
                                          inclusions -- tuning
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
        summary="REMOVES an invented healthy-adult restriction (annotation inclusion + fulltext "
                "exclusion). Loosens 'drug or natural reward-related' to 'reward-related'. Folds "
                "the between-group exclusion into the whole-brain inclusion criterion.",
        reasoning="Net of three hand edits. The substantive one is v2 removing the healthy-adult "
                  "restriction; the source (pmid 34400176) imposes NO population restriction, so "
                  "v1 invented it. The v2 header reads 'correct error that only healthy adult "
                  "participants should be included', which sounds like the opposite -- the diff "
                  "settles it. v3 added a correlational-analysis exclusion and v4 reverted it, so "
                  "they CANCEL and leave no trace in the cumulative diff. One loosening to check: "
                  "dropping 'drug or natural' loses a distinction the paper does make."),
    ("cue_reactivity", "v4", "v6"): dict(
        change_type="source-underspecified", provenance="llm-reports", confidence="medium",
        summary="Admits ROI analyses THAT REPORT COORDINATES (new explicit exception). Admits "
                "small-volume-corrected maps alongside whole-brain. Narrows the cue exclusion to "
                "paradigms with no visual component. Demotes reporting-completeness from a "
                "requirement to a preference.",
        reasoning="RELABELLED from no-source-basis. The source excludes ROI analyses expressly "
                  "'given the absence of coordinates' -- that is its stated RATIONALE, so an ROI "
                  "analysis that does report coordinates satisfies the source's intent even "
                  "though it breaches the literal rule. Same for SVC: the source's third "
                  "inclusion criterion already admits 'whole-brain OR small-volume corrected' "
                  "analyses, so v6 is restoring something v4 dropped. Against that, the criterion "
                  "names itself 'ROI exception for BENCHMARK COMPATIBILITY', and provenance is "
                  "llm-reports off a false-negative report -- so read the label as the content "
                  "being defensible, not as the derivation being clean."),
    ("decision_making", "v1", "v2"): dict(
        change_type="source-underspecified", provenance="human-paper", confidence="high",
        summary="Adds four annotation-side disambiguation rules: how to read a generic analysis "
                "label, how to resolve subtype overlap by primary uncertainty source, and that "
                "subtype assignment is per-study and mutually exclusive.",
        reasoning="The header states it: author-edited with ChatGPT as a phrasing tool, and 'NOT "
                  "BENCHMARK-TUNED'. An earlier header claimed otherwise and was withdrawn "
                  "because the diff does not support it. Every addition is about HOW TO APPLY the "
                  "existing criteria consistently, not about what qualifies -- scope is "
                  "unchanged. An LLM used as an editing tool keeps this human-paper."),
    ("decision_making", "v2", "v3"): dict(
        change_type="source-uncovered", provenance="llm-reports", confidence="high",
        summary="ADDS a PET exclusion, an exclusion for non-risky/ambiguous/perceptual tasks "
                "(delay discounting, moral, social), and an explicit 'any participant group "
                "qualifies' inclusion naming substance users and neuropsychiatric conditions. "
                "Excludes implicit-baseline-only contrasts.",
        reasoning="CORRECTED from no-source-basis, and the correction is not close. The source "
                  "inclusion says 'We excluded PET studies to remove heterogeneity' -- v3's PET "
                  "exclusion quotes that reasoning back. It also says coordinates from studies "
                  "'including healthy controls, substance users, as well as participants with "
                  "other neuropsychiatric conditions (internet addiction, gambling, internet "
                  "gaming disorder, obesity, depressive disorders...)' were used, and that "
                  "'participants from all age groups (range 10-69) were also included'. v3's new "
                  "participant criterion reproduces that list almost verbatim. These are the "
                  "paper's own criteria that v2 never encoded."),
    ("dementia", "v1", "v3"): dict(
        change_type="source-uncovered", provenance="human-reports", confidence="medium",
        summary="ADDS the control-group requirement, an exclusion for pooling bvFTD with other "
                "patient groups, a >=6-participant rule, and a modality restriction (excludes "
                "seed-based FC, DTI, cortical thickness). Extends bvFTD to picks disease. Adds a "
                "Parkinson's exclusion and a no-neuroimaging exclusion at abstract.",
        reasoning="Most of the 18 changes restore stated criteria v1 omitted: the control-group "
                  "requirement answers the paper's 'No Control' exclusion; the no-pooling rule "
                  "answers 'no other forms of dementia'; the >=6 rule and the VBM/fMRI/FDG-PET "
                  "modality list are both stated inclusions. TWO EXCEPTIONS: the Parkinson's "
                  "exclusion has no counterpart in the stated criteria (though the source SEARCH "
                  "string scopes to bvFTD/picks), and the participant rule is made STRICTER than "
                  "the source -- the paper says '>=6 in EITHER the patient or healthy group', v3 "
                  "requires >=6 in the patient group AND >=6 in the control group. That second "
                  "one is a candidate source-unenforced in reverse and is worth your call."),
    ("dementia", "v3", "v4"): dict(
        change_type="source-unenforced", provenance="llm-reports", confidence="low",
        summary="RELAXES the psychiatric/comorbidity exclusion from 'no concurrent diagnosis' to "
                "'the group is PREDOMINANTLY some other diagnosis'. Tightens ROI to cover "
                "mask-confined and SVC analyses. ADDS an explicit requirement that peak "
                "coordinates exist.",
        reasoning="MIXED, and relabelled from no-source-basis. The coordinates requirement is "
                  "straight source-uncovered -- the paper's fourth inclusion criterion is "
                  "'reported the coordinates of between-group contrasts'. The comorbidity "
                  "relaxation is the interesting one: the paper STATES 'no concurrent psychiatric "
                  "diagnosis... other forms of dementia', and v4 loosens it to a predominance "
                  "test, which is the ER pattern. UNVERIFIED: I have not checked whether the "
                  "dementia gold list actually contains studies with comorbid or mixed samples. "
                  "Confidence is low until someone does. This run hurt recall and was not "
                  "promoted."),
    ("emotion_regulation_2022", "v2", "v4"): dict(
        change_type="source-unenforced", provenance="human-reports", confidence="high",
        summary="REMOVES three stated restrictions: healthy-adults-only (now any group at any "
                "age), static-images-only (now film and video qualify), and reappraisal-narrowly-"
                "read (now any instructed regulation strategy). Softens the ROI exclusion to "
                "explicit ROI only.",
        reasoning="VERIFIED, and relabelled from no-source-basis -- this is the case that forced "
                  "the new category. The v4 header names 10 gold PMIDs the paper included in "
                  "breach of its own stated criteria, and ALL TEN are in the 88-study gold list: "
                  "25631055, 17888411, 24782800, 17588776, 25698699 (film, against 'static visual "
                  "images'); 21686071, 24646887, 28372994, 25439326, 26596970 (child, adolescent "
                  "or mixed 10-25 samples, against 'healthy adults'). So the paper did not apply "
                  "its own text, and v4 follows its practice instead. The header's framing -- "
                  "'DELIBERATELY LESS FAITHFUL THAN THE PAPER, to buy recall' -- is what made me "
                  "call this no-source-basis; it is faithful to what the paper DID. Note the "
                  "honest residue: the change was found by counting lost gold, so provenance "
                  "stays human-reports."),
    ("executive_function", "v1", "v3"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="high",
        summary="RELAXES hard: the 18-60 healthy-only restriction goes (any group, clinical and "
                "developmental included); the active-control requirement goes (rest or fixation "
                "qualifies); partial-brain coverage now qualifies. ADDS SVC to the ROI exclusion. "
                "ADDS several annotator INSTRUCTIONS rather than study criteria ('You are judging "
                "ONE analysis, not the paper').",
        reasoning="SET BY THE AUTHOR, 2026-11-04, from knowledge of the project. I had left this "
                  "unlabelled because registry row 22282036's criteria fields are contaminated "
                  "with a nicotine meta-analysis (see SOURCE_DEFECTS), so there is no usable "
                  "reference text to judge against -- and because the relaxations superficially "
                  "resemble the source-unenforced pattern. The author's reading is that they are "
                  "not: these go beyond anything Niendam 2012 supports. Provenance was never in "
                  "doubt -- v2 and v3 both carry 'AUTOMATICALLY GENERATED by Claude' and 'tuned "
                  "against the benchmark'. The annotator-instruction additions ('You are judging "
                  "ONE analysis, not the paper') are arguably non-criteria within this row."),
    ("problem_solving", "v1", "v2"): dict(
        change_type="source-uncovered", provenance="llm-reports", confidence="high",
        summary="ADDS an exclusion for deactivations and reverse contrasts (increases only), an "
                "exclusion for between-group and group x condition contrasts, and a 'group-level "
                "effects in healthy participants' inclusion. ADDS a closed list of five permitted "
                "contrast types and a source-based definition of problem solving.",
        reasoning="CORRECTED from no-source-basis, which I gave you at HIGH confidence last turn. "
                  "I was misled by the config's own wording -- it says 'the BENCHMARK includes "
                  "only BOLD or rCBF signal INCREASES' -- but that is the source paper's "
                  "inclusion criterion 2, verbatim. Likewise 'group-level effects in healthy "
                  "participants' is its criterion 3, and the between-group/age/gender exclusion "
                  "is its stated exclusion 1. Three of the largest additions are the paper's own "
                  "criteria that v1 never encoded. The one genuinely benchmark-derived item is "
                  "the logical_reasoning -> Demand_MNI_final mapping and the closed list of five "
                  "admitted contrast types; those alone would be no-source-basis."),
    ("social", "v3", "v4"): dict(
        change_type="source-unenforced", provenance="llm-reports", confidence="low",
        summary="RELAXES the 18-65 age restriction to exclude only wholly-child/adolescent or "
                "wholly-geriatric samples. ADDS an exclusion for multivariate-only whole-brain "
                "results (searchlight MVPA, classification accuracy, RSA).",
        reasoning="Relabelled from no-source-basis, but UNVERIFIED. The source states 'healthy "
                  "participants, age 18-60', and v4 loosens it -- the ER pattern. Whether the "
                  "social gold list actually contains out-of-range samples is the check that "
                  "would settle it, and I have not run it. The MVPA exclusion is separate and "
                  "looks source-underspecified (the paper asks for whole-brain coordinates; a "
                  "classification-accuracy map is not that). A null result either way."),
    ("vbm_of_ptsd", "v1", "v2"): dict(
        change_type="source-uncovered", provenance="llm-reports", confidence="high",
        summary="ADDS an adult-sample inclusion plus a paediatric exclusion, and an exclusion for "
                "non-voxel-based morphometry (surface/vertex thickness, FreeSurfer parcellation, "
                "shape analyses, TBM) where no VBM contrast is reported.",
        reasoning="VERIFIED BY OUTCOME. Both added criteria are in the source's stated criteria "
                  "(pmid 36100907): 'reporting results among adult humans' and 'other non-voxel-"
                  "based morphometry methods'. v1 encodes neither. Exactly 6 papers flip "
                  "include->exclude and every one is caught by a plain reading of those two -- 5 "
                  "paediatric (ages 8-18, 12-21, 8-18, 7-14, 8-19) and 1 FreeSurfer/surface-based "
                  "with no VBM contrast. Zero gold lost, zero reverse flips, all 4 maps "
                  "byte-identical. CONFOUND: the agent had the 11-FP report in context, so the "
                  "error list may be how it NOTICED the omission -- provenance, not content. Not "
                  "promoted (author's decision 2026-11-04) because it is LLM-authored."),
    ("vbm_of_substance_use", "v1", "v2"): dict(
        change_type="source-misencoded", provenance="human-paper", confidence="high",
        summary="Repairs two template bugs: 'fMRI data' -> 'structural MRI data' in a VBM "
                "project, and 'cue type' -> 'substance use characteristics' (left over from the "
                "cue-reactivity config). Tightens 'whole brain analysis' to 'of grey matter "
                "volume'. Rewrites the multiple-experiments rule for clarity.",
        reasoning="The header calls it 'wording only', but 'fMRI' in a VBM project is a plain "
                  "mis-encoding, not phrasing. No criterion added or removed. The header also "
                  "explicitly WITHDRAWS an earlier 'Benchmark-tuned' claim as wrong."),
    ("vbm_of_substance_use", "v2", "v3"): dict(
        change_type="no-source-basis", provenance="llm-reports", confidence="medium",
        summary="EXPANDS the measure beyond GM volume to density, concentration and surface-based "
                "measures including cortical thickness. EXPANDS to any substance of abuse. "
                "RELAXES the psychiatric exclusion to 'patient group is the grouping variable'. "
                "ADDS an atlas-parcellation exclusion and a significant-foci requirement.",
        reasoning="The one tuned row I still read as no-source-basis. The source inclusion is "
                  "specific -- 'GM VOLUME differences' -- and v3 admits cortical thickness and "
                  "other surface measures, which is not a vague rule being operationalised but a "
                  "different measure being allowed in. Note vbm_of_ptsd moved the OPPOSITE way on "
                  "the same question, adding an exclusion for exactly those methods because its "
                  "source names them. The psychiatric relaxation is a source-unenforced candidate "
                  "(the paper states that exclusion) and would need a gold check. Hurt recall, "
                  "not promoted."),
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

    NOTE what this does and does not give you. These are the paper's STATED criteria. They are
    not what the paper DID -- see the source-unenforced category, where the two disagree and the
    gold list is the only way to tell.
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
            "criteria_summary": g.get("summary", ""),
            "change_type": g.get("change_type", "") or "UNLABELLED - see reasoning",
            "provenance": g.get("provenance", ""),
            "confidence": g.get("confidence", ""),
            "reasoning": g.get("reasoning", ""),
            "source_usable": "NO -- see source_criteria sheet" if not src.get("usable") else "yes",
            "target_header": header_quote(project, b),
            "intermediate_headers": inter or "-",
            "changes": "\n\n".join(bullets),
        })
    return rows


COLUMNS = ["project", "from_tier", "to_tier", "from", "to", "intermediate", "n_changes",
           "blocks", "added", "removed", "reworded", "criteria_summary",
           "change_type", "provenance", "confidence", "reasoning",
           "source_usable", "target_header", "intermediate_headers", "changes"]

# One set of annotation columns, not two. The earlier my_*/your_* split implied my guesses were
# a separate artifact to be adjudicated; they are a first pass to be overwritten in place, and
# the author has already overwritten one of them (executive_function).
EDITABLE = ("change_type", "provenance", "confidence", "reasoning")


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
              "reworded": 10, "criteria_summary": 85, "my_change_type": 22, "my_provenance": 15, "my_confidence": 12,
              "my_reasoning": 90, "your_change_type": 20, "your_provenance": 15,
              "your_note": 45, "source_usable": 16, "target_header": 70,
              "intermediate_headers": 70, "changes": 100}
    for i, c in enumerate(COLUMNS, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(c, 18)
    for col, fill in [(c, yours) for c in EDITABLE]:
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
        ("change_type", "source-uncovered,source-misencoded,source-underspecified,"
                        "source-unenforced,no-source-basis,non-criteria"),
        ("provenance", "human-paper,human-reports,llm-paper,llm-reports"),
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
        ("source-unenforced", "the paper STATES the rule, but its own included studies violate "
                              "it. The change follows the paper's PRACTICE rather than its TEXT. "
                              "CHECKABLE: look for gold studies that breach the stated rule. "
                              "Verified for emotion_regulation_2022 -- all 10 PMIDs its v4 header "
                              "names as breaching 'static images only' and 'healthy adults only' "
                              "are in the 88-study gold list."),
        ("no-source-basis", "no counterpart in the paper's criteria OR in its realised "
                            "inclusions. This is the category that makes a result circular, and "
                            "it is tuning whatever its provenance."),
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
        ("high", "verified against the source criteria or against outcomes (gold retention)"),
        ("medium", "provenance stated outright; the CONTENT label inferred from the diff"),
        ("low", "one terse header line; mostly inference. Prompts, not findings."),
    ]:
        key.append(["confidence", v, m])
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
    guessed = sum(1 for r in rows if r["change_type"])
    print(f"wrote {Path(args.csv).relative_to(REPO_ROOT)}"
          + (f" and {Path(args.xlsx).relative_to(REPO_ROOT)}" if ok
             else "  (no openpyxl -- xlsx skipped)"))
    print(f"  {len(rows)} tier transitions, {sum(r['n_changes'] for r in rows)} cumulative "
          f"criteria changes, {guessed} with my guess filled in")
    from collections import Counter
    for label, c in (("change_type", Counter(r["change_type"] for r in rows if r["change_type"])),
                     ("confidence", Counter(r["confidence"] for r in rows if r["confidence"]))):
        print(f"  {label}: " + "  ".join(f"{k}={v}" for k, v in c.most_common()))
    missing = [(r["project"], r["from"], r["to"]) for r in rows if not r["change_type"]]
    for m in missing:
        print(f"  NO GUESS: {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

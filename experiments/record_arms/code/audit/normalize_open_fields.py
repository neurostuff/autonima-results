#!/usr/bin/env python3
"""Measure how far rule-based normalisation gets on the open-vocabulary fields.

The point is to separate what rules can fix from what needs an ontology. Each stage is
applied in order and the distinct-value count after it is reported, so the value of every
rule is visible rather than asserted.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from survey_record_fields import RECORDS, walk  # noqa: E402

#: Words that carry no discriminating content in a task or condition name. Dropping them is
#: what collapses "emotion regulation task", "emotion regulation paradigm" and "emotion
#: regulation fMRI task" onto one key.
GENERIC = {
    "task", "tasks", "paradigm", "paradigms", "fmri", "mri", "pet", "scan", "scanning",
    "functional", "imaging", "study", "experiment", "version", "modified", "adapted",
    "the", "a", "an", "of", "and", "with", "during", "in", "for", "test", "battery",
    # added after reading the first pass: these split clusters that are one thing.
    # "resting state" and "resting-state protocol" were separate keys worth 68 and 11.
    "protocol", "session", "run", "block", "design", "based", "related", "condition",
    "conditions", "procedure", "measure", "assessment", "diagnosis", "group", "patients",
    "participants", "subjects", "type", "using", "via", "new", "standard", "classic",
}
#: Abbreviations seen in the records, expanded so they cluster with their spelled forms.
EXPAND = {
    "wm": "working memory", "rt": "reaction time", "rs": "resting state",
    "er": "emotion regulation", "mid": "monetary incentive delay",
    "ptsd": "posttraumatic stress disorder", "mdd": "major depressive disorder",
    "ocd": "obsessive compulsive disorder", "scz": "schizophrenia",
    "bvftd": "behavioral variant frontotemporal dementia",
    "ftd": "frontotemporal dementia", "ad": "alzheimers disease",
    "mci": "mild cognitive impairment", "aud": "alcohol use disorder",
    "sud": "substance use disorder", "tbi": "traumatic brain injury",
    "hc": "healthy control", "n-back": "nback",
}
DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def stage_case(s: str) -> str:
    """Unicode-normalise, fold case, unify dashes, squeeze whitespace."""
    s = unicodedata.normalize("NFKD", s).translate(DASHES)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


#: British spellings and possessives, folded before tokenising. Without the first,
#: "behavioral variant frontotemporal dementia" and "behavioural ..." are two clusters
#: worth 230 and 77 mentions of the same diagnosis. Without the second, "Alzheimer's"
#: tokenises to "alzheimer" plus a stray "s".
#: A lexicon, not a regex. `\w+?our\b` misses "behavioural" (the -our- is internal, so
#: "behavioral variant frontotemporal dementia" stayed split from "behavioural ..." at 230
#: and 77 mentions) and a regex loose enough to catch it also eats four, hour, tour, source.
BRITISH = {"behaviour": "behavior", "colour": "color", "favour": "favor",
           "humour": "humor", "tumour": "tumor", "vapour": "vapor", "odour": "odor",
           "honour": "honor", "labour": "labor", "neighbour": "neighbor",
           "haemo": "hemo", "oedema": "edema", "paediatric": "pediatric",
           "foetal": "fetal", "anaesthe": "anesthe", "orthopaedic": "orthopedic"}

SPELLING = [
            (re.compile(r"\b(\w+?)ise(d|s|r)?\b"), r"\1ize\2"),
            (re.compile(r"\b(\w+?)isation\b"), r"\1ization"),
            (re.compile(r"\b(\w+?)aemia\b"), r"\1emia"),
            (re.compile(r"\b(\w+?)oedema\b"), r"\1edema")]


def stage_punct(s: str) -> str:
    """Drop possessives, fold spelling, drop punctuation and parenthetical asides."""
    s = re.sub(r"\u2019", "'", s)
    s = re.sub(r"'s\b|\u2019s\b|s'\b", "", s)       # Alzheimer's -> Alzheimer
    for brit, us in BRITISH.items():
        s = s.replace(brit, us)
    for pat, rep in SPELLING:
        s = pat.sub(rep, s)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def stage_tokens(s: str) -> str:
    """Expand abbreviations, drop generic words, singularise, sort to a bag."""
    out = []
    for tok in s.split():
        tok = EXPAND.get(tok, tok)
        for part in tok.split():
            if part in GENERIC:
                continue
            if len(part) > 3 and part.endswith("s") and not part.endswith("ss"):
                part = part[:-1]
            out.append(part)
    return " ".join(sorted(set(out)))


STAGES = [("raw", lambda s: s), ("case+space", stage_case),
          ("punct+spell", lambda s: stage_punct(stage_case(s))),
          ("token bag", lambda s: stage_tokens(stage_punct(stage_case(s))))]


def subset_merge(keys: dict[str, object]) -> tuple[dict, list]:
    """Fold a token bag into the most frequent strictly-larger bag that contains it.

    "cue food" into "cue food visual" is the intended case. It is reported separately
    rather than folded into the stage list because it is lossy in a way the earlier stages
    are not: a shorter bag can be contained by two unrelated longer ones, and "cue food"
    is contained by both "cue food visual" and "cue food reactivity". Every merge it makes
    is listed so the over-merges can be seen.
    """
    weight = {k: sum(v.values()) for k, v in keys.items()}
    sets = {k: frozenset(k.split()) for k in keys}
    merged, moves = dict(keys), []
    for k in sorted(keys, key=lambda x: weight[x]):
        if len(sets[k]) < 2:
            continue
        hosts = [o for o in keys if o != k and sets[k] < sets[o]]
        if not hosts:
            continue
        host = max(hosts, key=lambda o: weight[o])
        if k in merged and host in merged:
            for v, n in merged.pop(k).items():
                merged[host][v] += n
            moves.append((k, host, weight[k], len(hosts)))
    return merged, moves


def collect(paths: set[str]) -> dict[str, Counter]:
    out = defaultdict(Counter)
    for project_dir in sorted(p for p in RECORDS.iterdir() if p.is_dir()):
        for f in project_dir.glob("*.extraction.json"):
            try:
                doc = json.loads(f.read_text())
            except Exception:
                continue
            for path, ev in walk(doc):
                if path not in paths:
                    continue
                v = ev.get("value")
                if not v:
                    continue
                for item in (v if isinstance(v, list) else [v]):
                    if isinstance(item, str) and item.strip():
                        out[path][item.strip()] += 1
    return out


FIELDS = {"tasks[].name", "groups[].medical_condition", "assessments[].name",
          "regions[].name", "model_estimations[].software", "groups[].medications",
          "tasks[].performance_measures", "tasks[].presentation_software"}


def emit_vocabulary(field: str, keys: dict, dest: Path, min_mentions: int = 2) -> None:
    """Write the clusters as a candidate controlled vocabulary.

    The canonical label is the most frequent surface form in the cluster, not the token bag:
    the bag is a matching key, and "delay incentive monetary" is not what anyone would put in
    a vocabulary. Clusters seen once are excluded -- 81% of task clusters are singletons and a
    vocabulary of hapaxes is a list of strings, not a vocabulary.
    """
    import csv
    rows = []
    for key, variants in keys.items():
        total = sum(variants.values())
        if total < min_mentions or not key:
            continue
        canonical = variants.most_common(1)[0][0]
        rows.append({"match_key": key, "canonical_label": canonical, "mentions": total,
                     "n_variants": len(variants),
                     "variants": " | ".join(v for v, _ in variants.most_common())})
    rows.sort(key=lambda r: -r["mentions"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    covered = sum(r["mentions"] for r in rows)
    print(f"  wrote {dest.name}: {len(rows)} terms covering {covered} mentions")


def main() -> int:
    data = collect(FIELDS)
    print(f"{'field':38} {'mentions':>9} " + " ".join(f"{n:>12}" for n, _ in STAGES))
    reduced = {}
    for field in sorted(FIELDS):
        c = data.get(field)
        if not c:
            continue
        counts, keys = [], None
        for _, fn in STAGES:
            keys = defaultdict(Counter)
            for v, n in c.items():
                keys[fn(v)][v] += n
            counts.append(len(keys))
        reduced[field] = keys
        print(f"{field:38} {sum(c.values()):>9} " + " ".join(f"{n:>12}" for n in counts))

    for field in ("tasks[].name", "groups[].medical_condition"):
        keys = reduced.get(field)
        if not keys:
            continue
        print(f"\n{'='*94}\n{field}: largest clusters after the token bag\n")
        ranked = sorted(keys.items(), key=lambda kv: -sum(kv[1].values()))
        for key, variants in ranked[:14]:
            total = sum(variants.values())
            forms = ", ".join(f"{v}({n})" for v, n in variants.most_common(4))
            print(f"  {total:>4}  [{key[:40]:40}] {forms[:100]}")
        singles = sum(1 for v in keys.values() if sum(v.values()) == 1)
        print(f"\n  clusters: {len(keys)};  seen once only: {singles} "
              f"({100*singles/len(keys):.0f}%)")
        total_mentions = sum(sum(v.values()) for v in keys.values())
        ranked_w = sorted((sum(v.values()) for v in keys.values()), reverse=True)
        print("  coverage by a curated vocabulary of the top N clusters:")
        run = 0
        for n in (25, 50, 100, 200, 400):
            run = sum(ranked_w[:n])
            print(f"     top {n:>3}: {100*run/total_mentions:5.1f}% of mentions")
        if field == "groups[].medical_condition":
            hc = sum(sum(v.values()) for k, v in keys.items()
                     if k and set(k.split()) <= {"healthy", "control", "normal", "no",
                                                 "non", "comparison", "volunteer"})
            print(f"  values that assert the ABSENCE of a condition: {hc} mentions "
                  f"({100*hc/total_mentions:.0f}%) -- these are not conditions")
        emit_vocabulary(field, keys,
                        RECORDS.parent / "data" /
                        f"vocab_{field.replace('[].', '_').replace('.', '_')}.csv")
        merged, moves = subset_merge(keys)
        ambiguous = [m for m in moves if m[3] > 1]
        print(f"  subset merge: {len(keys)} -> {len(merged)} clusters via {len(moves)} "
              f"merges, of which {len(ambiguous)} had more than one candidate host")
        for k, host, w, n in sorted(ambiguous, key=lambda m: -m[2])[:4]:
            print(f"     ambiguous: [{k[:34]}] (w={w}) -> [{host[:34]}], {n} hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

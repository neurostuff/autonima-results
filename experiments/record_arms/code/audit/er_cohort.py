"""Cohort manifest for emotion_regulation_2022, in the format build_corpus.py reads.

The script that originally produced pmids/cohort_all.csv is no longer in the tree, so the
rows are rebuilt from the same evidence it used: v4's screening and retrieval outputs for the
per-paper fields, and a filesystem probe for the build route.

Route preference is real ACE HTML first, then Elsevier, then pubget text, then ACE's text.csv
export. That order is deliberate: the HTML carries headings and tables, ace_text carries
neither, and dementia's arms were measurably degraded by being built from the inferior route.
"""
import csv, json, sys
from pathlib import Path

# ACE's text.csv holds whole article bodies in one field, well past the default
# 128 KiB field limit.
csv.field_size_limit(sys.maxsize)

EXP = Path("/data/james/pondie-vs-fulltext")
O = EXP / "repos/autonima-results/projects/emotion_regulation_2022/v4/outputs"
rows_scr = json.loads((O / "fulltext_screening_results.json").read_text())["screening_results"]
decision = {str(r["study_id"]): r.get("decision", "") for r in rows_scr}
screened = set(decision)
ret = json.loads((O / "fulltext_retrieval_results.json").read_text())
meta = {str(s["pmid"]): s for s in ret.get("studies_with_fulltext", []) if s.get("pmid")}

ace_html = {p.stem: p for p in (EXP / "articles/ace_outputs/html").rglob("*.html")}
els = {d.name for d in (EXP / "articles/elsevier_output").iterdir()
       if (d / "text.txt").is_file()}
pub = {d.name for d in (EXP / "pubget_pmc/text_by_pmid").iterdir()
       if (d / "text.txt").is_file()}
ace_text = set()
tc = EXP / "articles/ace_outputs/processed/text.csv"
if tc.is_file():
    with tc.open() as fh:
        for r in csv.DictReader(fh):
            ace_text.add(str(r.get("pmid")))

def route(pmid):
    if pmid in ace_html:
        return "ace", "ace", str(ace_html[pmid].relative_to(EXP))
    if pmid in els:
        return "elsevier", "elsevier", f"articles/elsevier_output/{pmid}"
    if pmid in pub:
        return "pubget_text", "pubget", f"pubget_pmc/text_by_pmid/{pmid}"
    if pmid in ace_text:
        return "ace_text", "ace", "articles/ace_outputs/processed/text.csv"
    return "", "", ""

out, counts = [], {}
for pmid in sorted(screened):
    b, m, sp = route(pmid)
    counts[b or "NONE"] = counts.get(b or "NONE", 0) + 1
    e = meta.get(pmid, {})
    out.append({
        "project": "emotion_regulation_2022", "pmid": pmid,
        "baseline_source": b, "decision": decision.get(pmid, ""),
        "coordinates_found": bool(e.get("coordinates_found")),
        "screened": True, "in_annotation_only": False,
        "has_baseline_fulltext": pmid in meta, "pmc_recoverable": pmid in pub,
        "build_source": b, "mirror": m, "source_path": sp,
        "in_cohort": bool(b),
    })
dest = EXP / "pmids/emotion_regulation_2022.cohort.csv"
with dest.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)
print("  wrote", dest, len(out), "rows")
print("  routes:", counts)
print("  in_cohort True:", sum(1 for r in out if r["in_cohort"]))

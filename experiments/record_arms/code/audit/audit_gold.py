"""Re-score every arm against the upstream gold, beside the stale checkout's gold."""
import csv, json
from pathlib import Path

R = Path("/data/james/pondie-vs-fulltext/repos/autonima-results")
STALE = Path("/data/alejandro/projects/neurometabench/data/included_studies.csv")
UP = Path("/data/james/pondie-vs-fulltext/repos/neurometabench-upstream/data/included_studies.csv")
META = {"cue_reactivity": "34400176", "dementia": "35664889",
        "vbm_of_ptsd": "36100907", "vbm_of_substance_use": "36115222",
        "emotion_regulation_2022": "35413444"}
ARMS = {
    "vbm_of_ptsd": ("v1-A1-mini", "v1-record-with-evidence", "v1-record-no-evidence"),
    "cue_reactivity": ("v5-gpt-A1-mini", "v5-gpt-record-with-evidence", "v5-gpt-record-no-evidence"),
    "dementia": ("v3", "v3-record-with-evidence", "v3-record-no-evidence"),
    "vbm_of_substance_use": ("v2", "v2-record-with-evidence", "v2-record-no-evidence"),
}
def load(p):
    g = {}
    for r in csv.DictReader(p.open()):
        g.setdefault(r["meta_pmid"], set()).add(r["study_pmid"])
    return g
stale, up = load(STALE), load(UP)

def f1(inc, gold):
    tp = len(inc & gold); fp = len(inc - gold); fn = len(gold - inc)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

print("%-22s %-22s %-28s %-28s" % ("project", "arm", "stale gold", "upstream gold"))
for proj, runs in ARMS.items():
    gs, gu = stale.get(META[proj], set()), up.get(META[proj], set())
    tag = "" if gs == gu else "   <-- GOLD CHANGED %d -> %d" % (len(gs), len(gu))
    print("== %s%s" % (proj, tag))
    for run in runs:
        d = R / "projects" / proj / run / "outputs" / "fulltext_screening_results.json"
        if not d.is_file():
            continue
        rows = json.loads(d.read_text())["screening_results"]
        inc = {str(r["study_id"]) for r in rows if r["decision"] == "included_fulltext"}
        a = f1(inc, gs); b = f1(inc, gu)
        mark = "" if abs(a[5] - b[5]) < 1e-9 else "  <-- F1 moves"
        print("   %-28s F1 %.4f (tp%3d fp%3d fn%3d)   F1 %.4f (tp%3d fp%3d fn%3d)%s" % (
            run, a[5], a[0], a[1], a[2], b[5], b[0], b[1], b[2], mark))

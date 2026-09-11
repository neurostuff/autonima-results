"""Where each project's gold is lost, and how much the reported F1 hides."""
import csv, json
from pathlib import Path
R = Path("/data/james/pondie-vs-fulltext/repos/autonima-results")
UP = Path("/data/james/pondie-vs-fulltext/repos/neurometabench-upstream/data/included_studies.csv")
META = {"cue_reactivity": "34400176", "dementia": "35664889", "vbm_of_ptsd": "36100907",
        "vbm_of_substance_use": "36115222", "emotion_regulation_2022": "35413444"}
FULLTEXT = {"vbm_of_ptsd": "v1-A1-mini", "cue_reactivity": "v5-gpt-A1-mini",
            "dementia": "v3", "vbm_of_substance_use": "v2", "emotion_regulation_2022": "v4"}
gold = {}
for r in csv.DictReader(UP.open()):
    gold.setdefault(r["meta_pmid"], set()).add(r["study_pmid"])

print("%-24s %5s %8s %9s %9s %9s" % ("project", "gold", "in search", "screened", "included", "ceiling"))
for proj, run in FULLTEXT.items():
    g = gold.get(META[proj], set())
    o = R / "projects" / proj / run / "outputs"
    if not o.is_dir():
        print("  %-22s (no run)" % proj); continue
    sr = json.loads((o / "search_results.json").read_text())
    rows = sr if isinstance(sr, list) else (sr.get("studies") or sr.get("results") or [])
    found = {str(x.get("pmid")) for x in rows}
    fs = json.loads((o / "fulltext_screening_results.json").read_text())["screening_results"]
    scr = {str(r["study_id"]) for r in fs}
    inc = {str(r["study_id"]) for r in fs if r["decision"] == "included_fulltext"}
    print("  %-22s %5d %8d %9d %9d %8.0f%%" % (
        proj, len(g), len(g & found), len(g & scr), len(g & inc),
        100 * len(g & scr) / len(g) if g else 0))
    print("     lost at search %3d | lost between search and screening %3d | rejected %3d" % (
        len(g - found), len((g & found) - scr), len((g & scr) - inc)))

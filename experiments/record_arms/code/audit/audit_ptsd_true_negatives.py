import csv, json
from pathlib import Path
GOLD = {r["study_pmid"] for r in csv.DictReader(
    open("../neurometabench/data/included_studies.csv")) if r["meta_pmid"] == "36100907"}
arms = {"full text":"v1-A1-mini", "+evidence":"v1-record-with-evidence",
        "-evidence":"v1-record-no-evidence"}
dec, tn = {}, {}
for name, run in arms.items():
    d = Path("projects/vbm_of_ptsd")/run/"outputs"
    rows = json.loads((d/"fulltext_screening_results.json").read_text())["screening_results"]
    dec[name] = {str(r["study_id"]): r for r in rows}
    excluded = {p for p, r in dec[name].items() if r["decision"] != "included_fulltext"}
    tn[name] = excluded - GOLD
    print("%-11s screened=%d  excluded=%d  TN(excluded & not gold)=%d" % (
        name, len(dec[name]), len(excluded), len(tn[name])))
# analyses extracted per study (from the full-text arm studyset, which exports excluded too)
d = Path("projects/vbm_of_ptsd/v1-A1-mini/outputs")
ss = json.loads((d/"nimads_studyset.json").read_text())
an = {str(s.get("pmid")): [(a.get("name"), len(a.get("points") or []))
                           for a in (s.get("analyses") or [])] for s in ss["studies"]}
meta = {str(s["pmid"]): s.get("title") for s in
        json.loads((d/"final_results.json").read_text())["studies"]}
print("\n=== TN papers (-evidence arm) that nonetheless have extracted analyses:")
for p in sorted(tn["-evidence"]):
    if an.get(p):
        print("  %-10s %s" % (p, (meta.get(p) or "?")[:74]))
        for nm, k in an[p]:
            print("      %-58s points=%d" % (str(nm)[:58], k))
print("\n=== TN papers with NO extracted analyses: %d" % sum(1 for p in tn["-evidence"] if not an.get(p)))

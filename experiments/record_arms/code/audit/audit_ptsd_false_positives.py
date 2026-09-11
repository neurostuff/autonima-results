import csv, json
from pathlib import Path
GOLD = {r["study_pmid"] for r in csv.DictReader(
    open("../neurometabench/data/included_studies.csv")) if r["meta_pmid"] == "36100907"}
arms = {"full text":"v1-A1-mini", "+evidence":"v1-record-with-evidence",
        "-evidence":"v1-record-no-evidence"}
incl, meta, dec = {}, {}, {}
for name, run in arms.items():
    d = Path("projects/vbm_of_ptsd")/run/"outputs"
    rows = json.loads((d/"fulltext_screening_results.json").read_text())["screening_results"]
    dec[name] = {str(r["study_id"]): r for r in rows}
    incl[name] = {p for p, r in dec[name].items() if r["decision"] == "included_fulltext"}
    for s in json.loads((d/"final_results.json").read_text())["studies"]:
        meta[str(s["pmid"])] = (s.get("title"), s.get("publication_date"), s.get("journal"))
allfp = sorted(set().union(*[incl[n] - GOLD for n in arms]))
print("union of false positives across arms:", len(allfp))
print("%-10s %-4s %-4s %-4s  %-11s %s" % ("pmid", "ft", "+ev", "-ev", "date", "title"))
for p in allfp:
    flags = ["Y" if p in incl[n] else "." for n in arms]
    t, dt, j = meta.get(p, (None, None, None))
    print("%-10s %-4s %-4s %-4s  %-11s %s" % (p, flags[0], flags[1], flags[2],
          (dt or "?")[:11], (t or "?")[:78]))

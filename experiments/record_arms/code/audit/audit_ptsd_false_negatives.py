import csv, json
from pathlib import Path
GOLD = {r["study_pmid"] for r in csv.DictReader(open("../neurometabench/data/included_studies.csv"))
        if r["meta_pmid"] == "36100907"}
print("gold papers:", len(GOLD))
arms = {"full text":"v1-A1-mini", "record + evidence":"v1-record-with-evidence",
        "record, no evidence":"v1-record-no-evidence"}
dec, incl = {}, {}
for name, run in arms.items():
    d = Path("projects/vbm_of_ptsd")/run/"outputs"
    rows = json.loads((d/"fulltext_screening_results.json").read_text())["screening_results"]
    dec[name] = {str(r["study_id"]): r for r in rows}
    incl[name] = {p for p, r in dec[name].items() if r["decision"] == "included_fulltext"}
    tp = GOLD & incl[name]; fn = GOLD - incl[name]; fp = incl[name] - GOLD
    print("\n%-20s included=%2d  TP=%2d FP=%2d FN=%2d" % (name, len(incl[name]), len(tp), len(fp), len(fn)))
    print("   FN pmids:", sorted(fn))
# which gold papers were reached at all (i.e. present in the screening file)
seen = set(dec["full text"])
print("\ngold papers never reaching full-text screening:", sorted(GOLD - seen))
print("\n=== per-gold-paper decision across arms:")
for p in sorted(GOLD):
    row = []
    for name in arms:
        r = dec[name].get(p)
        row.append("%-22s" % (r["decision"] if r else "NOT SCREENED"))
    print("  %-10s %s" % (p, " ".join(row)))

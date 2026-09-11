import csv, json
from pathlib import Path
GOLD = {r["study_pmid"] for r in csv.DictReader(
    open("../neurometabench/data/included_studies.csv")) if r["meta_pmid"] == "36100907"}
arms = {"+evidence":"v1-record-with-evidence", "-evidence":"v1-record-no-evidence",
        "full text":"v1-A1-mini"}
dec = {}
for n, run in arms.items():
    d = Path("projects/vbm_of_ptsd")/run/"outputs"
    dec[n] = {str(r["study_id"]): r for r in
              json.loads((d/"fulltext_screening_results.json").read_text())["screening_results"]}
inc = {n: {p for p, r in v.items() if r["decision"] == "included_fulltext"} for n, v in dec.items()}
print("included:", {n: len(v) for n, v in inc.items()})
diff = sorted((inc["+evidence"] | inc["-evidence"]) - (inc["+evidence"] & inc["-evidence"]))
print("\npapers where the two record arms disagree on INCLUSION: %d" % len(diff))
for p in diff:
    print("  %-10s +ev=%-20s -ev=%-20s full=%-20s gold=%s" % (
        p, dec["+evidence"][p]["decision"], dec["-evidence"][p]["decision"],
        dec["full text"][p]["decision"], p in GOLD))
alldiff = sorted({p for p in dec["+evidence"]
                  if dec["+evidence"][p]["decision"] != dec["-evidence"][p]["decision"]})
print("\nany decision-label difference (incl. incomplete): %d" % len(alldiff))
for p in alldiff:
    print("  %-10s +ev=%-20s -ev=%-20s gold=%s" % (
        p, dec["+evidence"][p]["decision"], dec["-evidence"][p]["decision"], p in GOLD))

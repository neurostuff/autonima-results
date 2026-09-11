import csv, json
from pathlib import Path
RUNMAP = {}
for r in csv.DictReader(open("reports/record_arms_meta_metrics.csv")):
    RUNMAP.setdefault((r["project"], r["run"]), set()).add(r["auto_analysis"])
for (proj, run), keys in sorted(RUNMAP.items()):
    d = Path("projects")/proj/run/"outputs"
    incl = {str(s["pmid"]) for s in json.loads((d/"final_results.json").read_text())["studies"]}
    ss = json.loads((d/"nimads_studyset.json").read_text())
    by_id = {a["id"]: str(s.get("pmid")) for s in ss["studies"] for a in (s.get("analyses") or [])}
    ann = json.loads((d/"nimads_annotation.json").read_text())
    per = {}
    for n in ann["notes"]:
        for k, v in (n.get("note") or {}).items():
            if v and k in keys:
                per.setdefault(k, set()).add(by_id.get(n["analysis"]))
    print("%-22s %-28s included=%4d" % (proj, run, len(incl)))
    for k in sorted(keys):
        pm = (per.get(k) or set()) - {None}
        bad = pm - incl
        flag = "  <-- CONTAMINATED" if bad else ""
        print("      %-32s pmids=%4d  excluded_by_this_arm=%3d%s" % (k, len(pm), len(bad), flag))

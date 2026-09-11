import json
from pathlib import Path
for proj, runs in {
  "vbm_of_ptsd": ("v1-A1-mini","v1-record-with-evidence","v1-record-no-evidence"),
  "dementia": ("v3","v3-record-with-evidence","v3-record-no-evidence"),
}.items():
    print("==", proj)
    for run in runs:
        d = Path("projects")/proj/run/"outputs"
        fr = json.loads((d/"final_results.json").read_text())
        incl = {str(s["pmid"]) for s in fr["studies"]}
        ss = json.loads((d/"nimads_studyset.json").read_text())
        by_id = {}
        allpm = set()
        for s in ss["studies"]:
            allpm.add(str(s.get("pmid")))
            for a in s.get("analyses") or []:
                by_id[a["id"]] = str(s.get("pmid"))
        ann = json.loads((d/"nimads_annotation.json").read_text())
        keys = {}
        for n in ann["notes"]:
            for k, v in (n.get("note") or {}).items():
                if v:
                    keys.setdefault(k, set()).add(by_id.get(n["analysis"]))
        print("   %-28s studyset=%3d  final_included=%3d" % (run, len(allpm), len(incl)))
        for k in sorted(keys):
            pm = keys[k] - {None}
            print("        %-16s pmids=%3d   == included? %-5s  extra_vs_included=%3d" % (
                k, len(pm), pm == incl, len(pm - incl)))

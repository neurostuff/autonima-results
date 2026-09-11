import csv, json, re, textwrap
from pathlib import Path
GOLD = {r["study_pmid"] for r in csv.DictReader(
    open("../neurometabench/data/included_studies.csv")) if r["meta_pmid"] == "36100907"}
d = Path("projects/vbm_of_ptsd/v1-A1-mini/outputs")
studies = {str(s["pmid"]): s for s in json.loads((d/"final_results.json").read_text())["studies"]}
FP = ["16371250","19349151","19996042","20673548","22948482","23113800",
      "25212487","28888350","30127342","30343133","32938511"]
AGE = re.compile(r"\b(child|children|adolescen\w*|pediatric|paediatric|youth|boys|girls|"
                 r"aged?\s+\d+[–\-]\d+|mean age[^.;]{0,40})", re.I)
for p in FP:
    s = studies.get(p)
    if not s: print(p, "-- no metadata in v1-A1-mini"); continue
    ab = (s.get("abstract") or "")
    hits = sorted({m.group(0).lower() for m in AGE.finditer(ab)})
    print("=" * 98)
    print("%s  %s" % (p, (s.get("title") or "")[:86]))
    print("   date: %s" % s.get("publication_date"))
    print("   age cues: %s" % (", ".join(hits[:6]) or "none found"))
    print(textwrap.fill(ab[:620], 96, initial_indent="   ", subsequent_indent="   "))

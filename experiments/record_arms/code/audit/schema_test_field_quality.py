import json, re
from pathlib import Path
NEW = Path("/home/james/pondie/data/runs/schema_test/records")
DEMO = re.compile(r"\b(right-?handed|left-?handed|handedness|female|male|caucasian|"
                  r"african|hispanic|asian|age[ds]?\b|years old|color-?blind|"
                  r"english was|native speaker|sex|gender|IQ)\b", re.I)
tot = dup = demo = trait = 0
rows = []
for f in sorted(NEW.glob("*.extraction.json")):
    doc = json.loads(f.read_text())
    for g in doc.get("groups") or []:
        mc = (g.get("medical_condition") or {}).get("value") or []
        if isinstance(mc, str): mc = [mc]
        mcl = {str(c).lower().strip() for c in mc if c}
        pc = (g.get("population_characteristics") or {}).get("value") or []
        if isinstance(pc, str): pc = [pc]
        for v in pc:
            v = str(v); tot += 1
            if v.lower().strip() in mcl:
                dup += 1; rows.append(("duplicates medical_condition", f.stem[:9], v[:52]))
            elif DEMO.search(v):
                demo += 1; rows.append(("belongs to a demographic slot", f.stem[:9], v[:52]))
            else:
                trait += 1
print(f"population_characteristics entries across the 10 papers: {tot}")
print(f"   genuine cohort traits              {trait:>3}  ({100*trait/tot:.0f}%)")
print(f"   duplicate the medical_condition    {dup:>3}  ({100*dup/tot:.0f}%)")
print(f"   restate a demographic already slotted {demo:>3}  ({100*demo/tot:.0f}%)")
print("\n   the misplaced ones:")
for kind, pmid, v in rows[:12]:
    print(f"     [{kind[:30]:30}] {pmid:9} {v}")

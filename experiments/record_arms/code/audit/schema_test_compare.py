import json
from pathlib import Path
OLD = Path("/data/james/pondie-vs-fulltext/repos/autonima-results/experiments/record_arms/records")
NEW = Path("/home/james/pondie/data/runs/schema_test/records")
PRED = {"16133128":"A","17217932":"A","18568078":"A","17197102":"A","18540916":"A",
        "11296095":"B","29990584":"B","28929362":"B","16565998":"B*","25094019":"B"}

def load(pmid, root):
    for f in root.rglob(f"{pmid}.extraction.json"):
        return json.loads(f.read_text())
    return None

def groups(doc):
    out = []
    for g in doc.get("groups") or []:
        mc = (g.get("medical_condition") or {}).get("value") or []
        if isinstance(mc, str): mc = [mc]
        pc = (g.get("population_characteristics") or {}).get("value") or []
        if isinstance(pc, str): pc = [pc]
        out.append({"name": ((g.get("name") or {}).get("value")) or "",
                    "healthy": (g.get("is_healthy") or {}).get("value"),
                    "mc": [str(c) for c in mc if c],
                    "pc": [str(c) for c in pc if c]})
    return out

for pmid, pred in PRED.items():
    o, n = load(pmid, OLD), load(pmid, NEW)
    print("=" * 96)
    print(f"{pmid}  prediction {pred}")
    if not (o and n):
        print("   missing a side"); continue
    og, ng = groups(o), groups(n)
    for i in range(max(len(og), len(ng))):
        a = og[i] if i < len(og) else None
        b = ng[i] if i < len(ng) else None
        if a:
            print(f"   OLD {a['name'][:34]:34} healthy={str(a['healthy']):5} mc={str(a['mc'])[:44]}")
        if b:
            flag = ""
            if a and a["healthy"] is True and b["healthy"] is False: flag = "  <- FLIPPED"
            if b["pc"]: flag += "  <- pc FILLED"
            print(f"   NEW {b['name'][:34]:34} healthy={str(b['healthy']):5} mc={str(b['mc'])[:44]}{flag}")
            if b["pc"]:
                print(f"       population_characteristics: {b['pc']}")

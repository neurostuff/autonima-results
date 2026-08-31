"""Build NiMARE datasets for both arms: regex floor vs LLM-selected."""
import json, os, sqlite3, collections, numpy as np, nimare
from nimare.utils import tal2mni
from _paths import WORK, ACE_DB

space = {str(r[0]): (r[1] or '').upper() for r in sqlite3.connect(ACE_DB).execute("SELECT id, space FROM articles")}

def build(coords_by_pmid, name):
    ds, kept = {}, collections.Counter()
    for pm, cs in coords_by_pmid.items():
        sp = space.get(pm, '')
        cs = sorted({tuple(round(float(v),1) for v in c) for c in cs if len(c)==3})
        cs = [c for c in cs if all(abs(v)<=100 for v in c)]
        if sp not in ('MNI','TAL'): kept['drop_space']+=1; continue
        if len(cs) < 3: kept['drop_lt3']+=1; continue
        arr = np.array(cs, dtype=float)
        if sp=='TAL': arr = tal2mni(arr); kept['TAL']+=1
        else: kept['MNI']+=1
        ds[pm]={"contrasts":{"1":{"coords":{"space":"MNI","x":arr[:,0].tolist(),
                "y":arr[:,1].tolist(),"z":arr[:,2].tolist()},"metadata":{"sample_sizes":[20]}}}}
    p=os.path.join(WORK,f"{name}.json"); json.dump(ds,open(p,"w"))
    nimare.dataset.Dataset(p).save(os.path.join(WORK,f"{name}.pkl"))
    print(f"  {name:<12} studies {len(ds):>4}  coords {sum(len(v['contrasts']['1']['coords']['x']) for v in ds.values()):>5}  {dict(kept)}")
    return ds

floor = json.load(open(os.path.join(WORK,"arm_floor.json")))
build(floor, "arm_floor")

ann = json.load(open(os.path.join(WORK,"annotations.json")))
crd = json.load(open(os.path.join(WORK,"arm_llm_coords.json")))
inc_ids = {d['analysis_id'] for v in ann.values() for d in v if d.get('include')}
llm = collections.defaultdict(list)
for pm, entries in crd.items():
    for e in entries:
        if e['id'] in inc_ids: llm[pm].extend(e['coords'])
print(f"  LLM included analyses: {len(inc_ids)}  spanning {len(llm)} studies")
build(dict(llm), "arm_llm")

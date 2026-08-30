import sqlite3, re, collections, json, numpy as np
import nimare
from nimare.utils import tal2mni
c=sqlite3.connect('/home/zorro/repos/autonima-results/articles/ace_outputs/sqlite.db')
space={r[0]: (r[1] or '').upper() for r in c.execute("SELECT id, space FROM articles")}
S='/tmp/claude-1000/-home-zorro-repos-autonima-results/1fa96b56-7ac4-4233-80c7-5f42c02ec2da/scratchpad'
cand=json.load(open(f'{S}/deact_candidates.json'))
ds={}
kept=collections.Counter()
for pmid, coords in cand.items():
    sp=space.get(int(pmid),'')
    if sp not in ('MNI','TAL'): kept['dropped_unknown_space']+=1; continue
    if len(coords)<3: kept['dropped_lt3_coords']+=1; continue
    arr=np.array(coords, dtype=float)
    if sp=='TAL':
        arr=tal2mni(arr); kept['converted_TAL']+=1
    else: kept['native_MNI']+=1
    ds[pmid]={"contrasts":{"1":{"coords":{"space":"MNI",
        "x":arr[:,0].tolist(),"y":arr[:,1].tolist(),"z":arr[:,2].tolist()},
        "metadata":{"sample_sizes":[20]}}}}
print("  ", dict(kept))
print(f"  studies in dataset: {len(ds)}   coords: {sum(len(v['contrasts']['1']['coords']['x']) for v in ds.values())}")
json.dump(ds, open(f'{S}/deact_nimare.json','w'))
d=nimare.dataset.Dataset(f'{S}/deact_nimare.json')
d.save(f'{S}/deact_dataset.pkl')
print(f"  NiMARE Dataset built: {len(d.ids)} ids, {d.coordinates.shape[0]} coordinate rows")

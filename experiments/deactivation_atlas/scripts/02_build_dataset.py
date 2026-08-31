"""Convert candidates to MNI, drop unusable studies, emit a NiMARE Dataset."""
import sqlite3, json, collections, os, numpy as np, nimare
from nimare.utils import tal2mni
from _paths import ACE_DB, WORK

space = {r[0]: (r[1] or '').upper() for r in sqlite3.connect(ACE_DB).execute("SELECT id, space FROM articles")}
cand = json.load(open(os.path.join(WORK, "deact_candidates.json")))
ds, kept = {}, collections.Counter()
for pmid, coords in cand.items():
    sp = space.get(int(pmid), '')
    if sp not in ('MNI', 'TAL'): kept['dropped_unknown_space'] += 1; continue
    if len(coords) < 3:          kept['dropped_lt3_coords']   += 1; continue
    arr = np.array(coords, dtype=float)
    if sp == 'TAL': arr = tal2mni(arr); kept['converted_TAL'] += 1
    else:                                kept['native_MNI']    += 1
    ds[pmid] = {"contrasts": {"1": {
        "coords": {"space": "MNI", "x": arr[:,0].tolist(), "y": arr[:,1].tolist(), "z": arr[:,2].tolist()},
        "metadata": {"sample_sizes": [20]}}}}      # ACE carries no n; stub for the ALE kernel
print(dict(kept))
print(f"studies {len(ds)}  coords {sum(len(v['contrasts']['1']['coords']['x']) for v in ds.values())}")
json.dump(ds, open(os.path.join(WORK, "deact_nimare.json"), "w"))
nimare.dataset.Dataset(os.path.join(WORK, "deact_nimare.json")).save(os.path.join(WORK, "deact_dataset.pkl"))
print("wrote", os.path.join(WORK, "deact_dataset.pkl"))

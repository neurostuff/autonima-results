"""ALE with Monte-Carlo FWE correction. ~25 min at 1000 iterations on 6 cores."""
import os, nimare
from nimare.meta.cbma.ale import ALE
from nimare.correct import FWECorrector
from _paths import WORK, MAPS

d = nimare.dataset.Dataset.load(os.path.join(WORK, "deact_dataset.pkl"))
print(f"studies={len(d.ids)} coords={d.coordinates.shape[0]}", flush=True)
res = ALE().fit(d)
cres = FWECorrector(method="montecarlo", n_iters=1000, n_cores=6).transform(res)
cres.save_maps(output_dir=MAPS, prefix="deact")
print("maps:", sorted(os.listdir(MAPS)))

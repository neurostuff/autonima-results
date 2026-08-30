import nimare, os
from nimare.meta.cbma.ale import ALE
from nimare.correct import FWECorrector
S='/tmp/claude-1000/-home-zorro-repos-autonima-results/1fa96b56-7ac4-4233-80c7-5f42c02ec2da/scratchpad'
d=nimare.dataset.Dataset.load(f'{S}/deact_dataset.pkl')
print(f"studies={len(d.ids)} coords={d.coordinates.shape[0]}", flush=True)
res=ALE().fit(d)
print("ALE fit done", flush=True)
corr=FWECorrector(method="montecarlo", n_iters=1000, n_cores=6)
cres=corr.transform(res)
os.makedirs(f'{S}/deact_ale', exist_ok=True)
cres.save_maps(output_dir=f'{S}/deact_ale', prefix='deact')
print("maps:", sorted(os.listdir(f'{S}/deact_ale')), flush=True)

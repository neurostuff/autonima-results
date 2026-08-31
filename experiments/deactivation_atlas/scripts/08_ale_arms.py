"""ALE + FWE for both arms, then Yeo-7 enrichment for each."""
import os, sys, json, numpy as np, nibabel as nib, nimare
from nimare.meta.cbma.ale import ALE
from nimare.correct import FWECorrector
from nilearn import datasets, image
from nilearn.reporting import get_clusters_table
from _paths import WORK, MAPS, TABLES

yeo = None
def enrich(img):
    global yeo
    if yeo is None:
        yeo = nib.load(datasets.fetch_atlas_yeo_2011()['thick_7'])
    Y = np.asarray(image.resample_to_img(yeo, img, interpolation='nearest').dataobj).squeeze().astype(int)
    sig = img.get_fdata() > 1.65
    names = {1:'Visual',2:'Somatomotor',3:'DorsalAttn',4:'VentAttn',5:'Limbic',6:'Frontoparietal',7:'Default'}
    tot, brain = int(sig.sum()), (Y>0)
    if tot == 0: return tot, {}
    out={}
    for k,n in names.items():
        net=(Y==k); i=int((sig&net).sum()); f=net.sum()/brain.sum()
        out[n]=(i, i/tot*100, (i/tot)/f if f else 0)
    return tot, out

summary={}
for arm in ("arm_floor","arm_llm"):
    d = nimare.dataset.Dataset.load(os.path.join(WORK, f"{arm}.pkl"))
    print(f"\n=== {arm}: {len(d.ids)} studies, {d.coordinates.shape[0]} coords ===", flush=True)
    res = ALE().fit(d)
    cres = FWECorrector(method="montecarlo", n_iters=1000, n_cores=6).transform(res)
    od = os.path.join(MAPS, arm); os.makedirs(od, exist_ok=True)
    cres.save_maps(output_dir=od, prefix=arm)
    z = nib.load(os.path.join(od, f"{arm}_z_desc-mass_level-cluster_corr-FWE_method-montecarlo.nii.gz"))
    tot, e = enrich(z)
    tbl = get_clusters_table(z, stat_threshold=1.65, cluster_threshold=50)
    tbl.to_csv(os.path.join(TABLES, f"clusters_{arm}.csv"), index=False)
    summary[arm]={"studies":len(d.ids),"coords":int(d.coordinates.shape[0]),
                  "sig_voxels":tot,"clusters":len(tbl),
                  "enrichment":{k:round(v[2],2) for k,v in e.items()},
                  "pct":{k:round(v[1],1) for k,v in e.items()}}
    print(f"  sig voxels {tot}  clusters {len(tbl)}")
    for k,v in sorted(e.items(), key=lambda x:-x[1][2]): print(f"    {k:<16}{v[1]:>6.1f}%  {v[2]:>6.2f}x")
json.dump(summary, open(os.path.join(TABLES,"arm_comparison.json"),"w"), indent=1)
print("\nwrote arm_comparison.json")

"""Validate: cluster table + enrichment of significant voxels across Yeo-7 networks."""
import os, numpy as np, nibabel as nib
from nilearn import datasets, image
from nilearn.reporting import get_clusters_table
from _paths import MAPS, TABLES

Z = os.path.join(MAPS, "deact_z_desc-mass_level-cluster_corr-FWE_method-montecarlo.nii.gz")
img = nib.load(Z)
tbl = get_clusters_table(img, stat_threshold=1.65, cluster_threshold=50)
tbl.to_csv(os.path.join(TABLES, "clusters.csv"), index=False)
print(f"clusters (z>1.65, k>50): {len(tbl)}")
print(tbl.to_string(index=False))

yeo = image.resample_to_img(nib.load(datasets.fetch_atlas_yeo_2011()['thick_7']), img, interpolation='nearest')
Y = np.asarray(yeo.dataobj).squeeze().astype(int)
sig = img.get_fdata() > 1.65
names = {1:'Visual', 2:'Somatomotor', 3:'DorsalAttn', 4:'VentAttn/Salience',
         5:'Limbic', 6:'Frontoparietal', 7:'Default'}
tot, brain = int(sig.sum()), (Y > 0)
print(f"\nsignificant voxels: {tot}")
print(f"{'network':<20}{'sig':>7}{'% sig':>9}{'enrichment':>12}")
rows = []
for k, n in names.items():
    net = (Y == k); inter = int((sig & net).sum())
    frac = net.sum() / brain.sum()
    rows.append((n, inter, inter/tot*100, (inter/tot)/frac if frac else 0))
for n, i, p, e in sorted(rows, key=lambda r: -r[3]):
    print(f"{n:<20}{i:>7}{p:>8.1f}%{e:>11.2f}x")
out = int((sig & (Y == 0)).sum())
print(f"{'(outside atlas)':<20}{out:>7}{out/tot*100:>8.1f}%")

import nibabel as nib, numpy as np
from nilearn.reporting import get_clusters_table
from nilearn.image import math_img
S='/home/zorro/repos/autonima-results'
D='/tmp/claude-1000/-home-zorro-repos-autonima-results/1fa96b56-7ac4-4233-80c7-5f42c02ec2da/scratchpad/deact_ale'
img=nib.load(f'{D}/deact_z_desc-mass_level-cluster_corr-FWE_method-montecarlo.nii.gz')
d=img.get_fdata()
print(f"  z map: nonzero voxels {int((d!=0).sum())}, max z {d.max():.2f}")
# cluster-FWE thresholded at z corresponding to p<.05 -> the map is already corrected; threshold z>1.65
tbl=get_clusters_table(img, stat_threshold=1.65, cluster_threshold=50)
print(f"  clusters (z>1.65, k>50): {len(tbl)}")
print(tbl.to_string(index=False, max_colwidth=18))
tbl.to_csv(f'{D}/clusters.csv', index=False)

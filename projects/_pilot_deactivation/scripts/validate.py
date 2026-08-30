import numpy as np, nibabel as nib, pandas as pd
from nilearn import datasets, image
D='/tmp/claude-1000/-home-zorro-repos-autonima-results/1fa96b56-7ac4-4233-80c7-5f42c02ec2da/scratchpad/deact_ale'
y=datasets.fetch_atlas_yeo_2011()
yeo=nib.load(y['thick_7'])
z=nib.load(f'{D}/deact_z_desc-mass_level-cluster_corr-FWE_method-montecarlo.nii.gz')
yeo_r=image.resample_to_img(yeo, z, interpolation='nearest')
Y=np.asarray(yeo_r.dataobj).squeeze().astype(int)
Z=z.get_fdata()
sig = Z>1.65
names={1:'Visual',2:'Somatomotor',3:'DorsalAttn',4:'VentAttn/Salience',5:'Limbic',6:'Frontoparietal',7:'Default'}
tot_sig=int(sig.sum())
print(f"  significant voxels: {tot_sig}")
print(f"\n  {'network':<20}{'sig voxels':>11}{'% of sig':>10}{'net size':>10}{'enrichment':>12}")
brain=(Y>0)
rows=[]
for k,n in names.items():
    net=(Y==k)
    inter=int((sig&net).sum())
    pct=inter/tot_sig*100
    netfrac=net.sum()/brain.sum()
    enr=(pct/100)/netfrac if netfrac>0 else 0
    rows.append((n,inter,pct,int(net.sum()),enr))
for n,i,p,s,e in sorted(rows,key=lambda r:-r[4]):
    print(f"  {n:<20}{i:>11}{p:>9.1f}%{s:>10}{e:>11.2f}x")
unl=int((sig&(Y==0)).sum())
print(f"  {'(outside atlas)':<20}{unl:>11}{unl/tot_sig*100:>9.1f}%")

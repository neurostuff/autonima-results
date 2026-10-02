"""Counterfactual MKDA maps: E1's selection plus pieces of autonima-pondie's, scored against gold."""
import json, csv, collections, sys, glob, warnings, logging
import numpy as np, nibabel as nib
from scipy.stats import norm, pearsonr
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
sys.path.insert(0, '/home/zorro/repos/autonima-results/scripts')
from map_mask import common_mask
from nimare.nimads import Studyset
from nimare.meta.cbma import MKDADensity
from nimare.correct import FDRCorrector
P = '/home/zorro/repos/autonima-results/projects/emotion_regulation_2022/'
R = '/home/zorro/repos/cbma-workspace-records/emotion_regulation_2022/'
M = '/home/zorro/repos/neurometabench/analysis/emotion_regulation_2022/'
G = json.load(open('/home/zorro/repos/neurometabench/data/nimads/emotion_regulation_2022/merged/nimads_studyset.json'))
GN = json.load(open('/home/zorro/repos/neurometabench/data/nimads/emotion_regulation_2022/merged/nimads_annotation.json'))
g2s = {an['id']: str(s.get('pmid') or s['id']) for s in G['studies'] for an in s['analyses']}
gold_t = collections.defaultdict(set)
for n in GN['notes']:
    for t, v in n['note'].items():
        if v is True: gold_t[t].add(g2s[n['analysis']])
E = json.load(open(R + 'results/nimads/studyset.json')); EN = json.load(open(R + 'results/nimads/annotation.json'))
A = json.load(open(P + 'v4-pondie-records/outputs/nimads_studyset.json')); AN = json.load(open(P + 'v4-pondie-records/outputs/nimads_annotation.json'))
studies = {}
def add(ss, prefix):
    for s in ss['studies']:
        pm = str(s.get('pmid') or s['id'])
        st = studies.setdefault(pm, {'id': pm, 'name': pm, 'analyses': []})
        for an in s['analyses']:
            st['analyses'].append({'id': prefix + an['id'], 'name': an.get('name') or an['id'],
                                   'points': [{'coordinates': p['coordinates'], 'space': p.get('space') or 'MNI'} for p in an['points']]})
add(E, 'E:'); add(A, 'A:')
merged = Studyset({'id': 'cf', 'name': 'cf', 'studies': list(studies.values())})
e_sel = collections.defaultdict(set); a_sel = collections.defaultdict(set)
for n in EN['notes']:
    for t, v in n['note'].items():
        if v is True: e_sel[t].add('E:' + n['analysis_id'])
a_study = {an['id']: str(s.get('pmid') or s['id']) for s in A['studies'] for an in s['analyses']}
for n in AN['notes']:
    for t in ('reappraisal', 'decrease', 'increase', 'maintain'):
        if n['note'].get(t) is True and n['analysis'] in a_study: a_sel[t].add('A:' + n['analysis'])
st_of = lambda i: i.split(':', 1)[1].split('-')[0].split('_')[0]
L = lambda p: nib.load(p).get_fdata()
def score(ids, t):
    ds = merged.slice(analyses=sorted(ids)).to_dataset()
    res = MKDADensity().fit(ds)
    cres = FDRCorrector(method='indep', alpha=0.05).transform(res)
    z = cres.get_map('z', return_type='image').get_fdata()
    zf = cres.get_map('z_corr-FDR_method-indep', return_type='image').get_fdata()
    mz, mf = L(M + t + '/z.nii.gz'), L(M + t + '/z_corr-FDR_method-indep.nii.gz')
    m = common_mask(mz, mf, z, zf)
    a, b = mf[m] > 1.96, zf[m] > 1.96
    return round(2 * (a & b).sum() / (a.sum() + b.sum()), 3), round(pearsonr(mz[m], z[m])[0], 3), len({st_of(i) for i in ids}), len(ids)
for t in sys.argv[1:]:
    e_st = {st_of(i) for i in e_sel[t]}
    extra = {i for i in a_sel[t] if st_of(i) not in e_st}
    shared = {i for i in a_sel[t] if st_of(i) in e_st}
    conds = {
        'E1 (baseline)': e_sel[t],
        'autonima-pondie (baseline)': a_sel[t],
        'E1 + autonima picks in its extra GOLD studies': e_sel[t] | {i for i in extra if st_of(i) in gold_t[t]},
        'E1 + autonima picks in ALL its extra studies': e_sel[t] | extra,
        'E1 + autonima picks within E1 studies': e_sel[t] | shared,
        'autonima minus its extra NON-gold studies': {i for i in a_sel[t] if st_of(i) in e_st or st_of(i) in gold_t[t]},
    }
    print(f'== {t}')
    for name, ids in conds.items():
        d, r, ns, na = score(ids, t)
        print(f'  {name:48s} studies {ns:3d} analyses {na:3d} | dice {d:.3f} r {r:.3f}', flush=True)

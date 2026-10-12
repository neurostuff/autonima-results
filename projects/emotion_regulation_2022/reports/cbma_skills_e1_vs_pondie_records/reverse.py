import json, re, sys, glob, warnings, logging, runpy
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING)
sys.argv = ['x']                      # load the counterfactual helpers without running targets
cf = runpy.run_path('/tmp/claude-1000/-home-zorro-repos-autonima-results/057837f3-33ce-4a6e-8ae0-9673202c4e0b/scratchpad/cf/counterfactual.py')
R = '/home/zorro/repos/cbma-workspace-records/emotion_regulation_2022/'
rev = re.compile(r'^\W*(look|view|watch|maintain|attend|permit|no[ -]?reg|react|experience|observe|passive)[^>]*>\s*[^>]*(decreas|increas|reapprais|regulat|reduc|distanc|suppress|enhanc|reinterpret|down|up)', re.I)
name = {}
for f in glob.glob(R + 'analyses/*.json'):
    for x in json.load(open(f))['analyses']: name['E:' + x['analysis_id']] = x['name'] or ''
e_sel, score = cf['e_sel'], cf['score']
reverse = {i for i in set().union(*e_sel.values()) if rev.search(name.get(i, ''))}
for t in ('reappraisal', 'decrease'):
    for label, ids in ((f'E1', e_sel[t]), (f'E1 without reverse contrasts', e_sel[t] - reverse)):
        d, r, ns, na = score(ids, t); print(f'{t:12s} {label:34s} studies {ns:3d} analyses {na:3d} | dice {d:.3f} r {r:.3f}', flush=True)
rev_reg = {i for i in reverse if any(i in e_sel[t] for t in ('reappraisal', 'decrease', 'increase'))}
for label, ids in (('E1', e_sel['maintain']), ('E1 + its reverse contrasts', e_sel['maintain'] | rev_reg)):
    d, r, ns, na = score(ids, 'maintain'); print(f"maintain     {label:34s} studies {ns:3d} analyses {na:3d} | dice {d:.3f} r {r:.3f}", flush=True)

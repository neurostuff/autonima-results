"""Candidate set for the annotation experiment, from autonima's PARSED ANALYSES.

Both arms of the comparison are built from this same substrate so the only
difference is regex-floor vs LLM adjudication:
  arm A (floor) : analyses whose own text matches the tight direction lexicon
  arm B (llm)   : analyses the model marks as task-induced deactivations,
                  drawn from the looser candidate pool
"""
import json, glob, re, collections, os, sqlite3
from _paths import WORK, ACE_DB

TIGHT = re.compile(r'(deactivat|de-activat|decreas\w+ (?:activation|activity|signal|response|BOLD)|'
                   r'negative (?:BOLD|activation|activity|signal|effect|correlation)|task[- ]negative|'
                   r'reduced (?:activation|activity|BOLD)|less activation|signal decrease)', re.I)
LOOSE = re.compile(r'(deactivat|decreas|negative|reduc|less activation|below baseline|<)', re.I)

# best parse per PMID (most analyses)
best = {}
for f in glob.glob(os.path.join(os.path.dirname(ACE_DB), '..', '..', 'projects', '*', '*', 'outputs',
                                'coordinate_parsing_results.json')):
    try: sts = json.load(open(f))['studies']
    except Exception: continue
    for x in sts:
        pm = str(x['pmid']); an = x.get('analyses') or []
        if pm not in best or len(an) > len(best[pm]): best[pm] = an

# table captions per pmid, for prompt context
caps = collections.defaultdict(list)
for aid, cap in sqlite3.connect(ACE_DB).execute(
        "SELECT article_id, caption FROM tables WHERE caption IS NOT NULL AND LENGTH(TRIM(caption))>10"):
    caps[str(aid)].append(cap.strip()[:200])

def txt(a): return f"{a.get('name') or ''} {a.get('description') or ''}".strip()
def npts(a): return len(a.get('points') or a.get('coordinates') or [])

floor, cand = {}, {}
for pm, ans in best.items():
    keep_f = [a for a in ans if npts(a) and TIGHT.search(txt(a))]
    keep_c = [a for a in ans if npts(a) and LOOSE.search(txt(a))]
    if keep_f: floor[pm] = keep_f
    if keep_c: cand[pm] = keep_c

def coords(ans):
    out = []
    for a in ans:
        for p in (a.get('points') or a.get('coordinates') or []):
            c = p.get('coordinates') if isinstance(p, dict) else p
            if c and len(c) == 3: out.append([float(v) for v in c])
    return out

json.dump({p: coords(v) for p, v in floor.items()}, open(os.path.join(WORK, "arm_floor.json"), "w"))
payload = {p: {"analyses": [{"id": f"{p}_{i}", "name": a.get('name'), "desc": a.get('description'),
                             "n_points": npts(a)} for i, a in enumerate(v)],
               "captions": caps.get(p, [])[:6]} for p, v in cand.items()}
json.dump(payload, open(os.path.join(WORK, "arm_llm_candidates.json"), "w"))
json.dump({p: [ {"id": f"{p}_{i}", "coords": coords([a])} for i,a in enumerate(v)] for p,v in cand.items()},
          open(os.path.join(WORK, "arm_llm_coords.json"), "w"))
print(f"parsed PMIDs {len(best)}")
print(f"ARM A floor : {len(floor)} studies, {sum(len(v) for v in floor.values())} analyses, "
      f"{sum(len(coords(v)) for v in floor.values())} coords")
print(f"ARM B pool  : {len(cand)} studies, {sum(len(v) for v in cand.values())} analyses")

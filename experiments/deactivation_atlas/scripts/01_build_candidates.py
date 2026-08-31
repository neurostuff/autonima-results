"""Stage 0: zero-token filter over the ACE database -> deactivation candidate coordinates."""
import sqlite3, re, collections, json, os
from _paths import ACE_DB, WORK

DEACT = re.compile(
    r'(deactivat|de-activat|decreas\w+ (?:activation|activity|signal|response|BOLD)|'
    r'negative (?:BOLD|activation|activity|signal|effect|correlation)|task[- ]negative|'
    r'reduced (?:activation|activity|BOLD)|less activation|signal decrease)', re.I)

c = sqlite3.connect(ACE_DB)
q = """SELECT t.article_id, t.id, t.caption, t.notes, a.x, a.y, a.z, a.statistic
       FROM activations a JOIN tables t ON a.table_id = t.id"""
by = collections.defaultdict(list)
for aid, tid, cap, notes, x, y, z, st in c.execute(q):
    is_cap = bool(DEACT.search(f"{cap or ''} {notes or ''}"))
    try:    v = float(st) if st not in (None, '') else None
    except Exception: v = None
    if not (is_cap or (v is not None and v < 0)):
        continue
    try:    X, Y, Z = float(x), float(y), float(z)
    except Exception: continue
    if not all(abs(t) <= 100 for t in (X, Y, Z)):
        continue
    by[str(aid)].append((round(X,1), round(Y,1), round(Z,1)))

# dedupe identical points within a study (see autonima#60)
ded = {k: sorted(set(v)) for k, v in by.items()}
out = os.path.join(WORK, "deact_candidates.json")
json.dump({k: [list(p) for p in v] for k, v in ded.items()}, open(out, "w"))
print(f"studies {len(ded)}  coords pre-dedup {sum(map(len, by.values()))} "
      f"-> deduped {sum(map(len, ded.values()))}")
print("wrote", out)

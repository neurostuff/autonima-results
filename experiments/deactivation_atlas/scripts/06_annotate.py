"""LLM adjudication of the candidate pool. gpt-5.6-luna, reasoning disabled.

Usage: 06_annotate.py [--limit N] [--out FILE]
gpt-5.6-luna rejects function tools unless reasoning_effort='none' is passed,
which autonima's client does not send -- hence the direct call here.
"""
import os, sys, json, time, argparse, concurrent.futures as cf
from openai import OpenAI
sys.path.insert(0, '/home/zorro/repos/autonima')
from autonima.llm.client import resolve_model_name
from _paths import WORK

ap = argparse.ArgumentParser()
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--out', default='annotations.json')
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--sample', type=int, default=0, help='random sample of N studies')
a = ap.parse_args()

MODEL = resolve_model_name("gpt-5.6-luna")
cli = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_API_GATEWAY'])

CRITERIA = """Decide, for each analysis, whether it reports TASK-INDUCED DEACTIVATION:
a decrease in BOLD/rCBF signal during a task condition relative to rest, baseline,
or a lower-demand control condition, in a single group of participants.

INCLUDE  - contrasts explicitly described as deactivation, decreased activation,
           negative BOLD, task-negative, or signal decrease relative to rest/baseline.
         - "rest > task" or "baseline > task" contrasts.
EXCLUDE  - between-group contrasts (patients vs controls, young vs old): a group
           difference is not a task-induced deactivation.
         - correlations with a behavioural or clinical variable, including negative ones.
         - ordinary activations, and reversed contrasts between two active conditions
           where neither is rest/baseline.
         - anything where direction cannot be determined from the text given."""

SCHEMA = {"name": "decide", "description": "Deactivation decision per analysis",
  "parameters": {"type": "object", "properties": {"decisions": {"type": "array", "items": {
      "type": "object", "properties": {
          "analysis_id": {"type": "string"},
          "include": {"type": "boolean"},
          "reasoning": {"type": "string", "description": "one short sentence"}},
      "required": ["analysis_id", "include", "reasoning"]}}}, "required": ["decisions"]}}

pool = json.load(open(os.path.join(WORK, "arm_llm_candidates.json")))
items = list(pool.items())
if a.sample:
    import random; random.seed(7); items = random.sample(items, min(a.sample, len(items)))
elif a.limit: items = items[:a.limit]

def one(kv):
    pm, d = kv
    lines = [f"{x['id']}: {x['name'] or '(no name)'} | {(x['desc'] or '')[:180]} | {x['n_points']} peaks"
             for x in d['analyses']]
    caps = "\n".join(f"- {c}" for c in d['captions'])
    prompt = f"{CRITERIA}\n\nTABLE CAPTIONS FROM THIS PAPER:\n{caps or '(none)'}\n\nANALYSES:\n" + "\n".join(lines)
    for attempt in range(3):
        try:
            r = cli.chat.completions.create(model=MODEL,
                messages=[{"role": "system", "content": "You are a neuroimaging meta-analysis expert. Respond using the decide function."},
                          {"role": "user", "content": prompt}],
                functions=[SCHEMA], function_call={"name": "decide"}, reasoning_effort="none")
            u = r.usage
            return pm, json.loads(r.choices[0].message.function_call.arguments)['decisions'], u.prompt_tokens, u.completion_tokens
        except Exception as e:
            if attempt == 2: return pm, [], 0, 0
            time.sleep(2 * (attempt + 1))

res, tin, tout, t0 = {}, 0, 0, time.time()
with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
    for pm, dec, i, o in ex.map(one, items):
        res[pm] = dec; tin += i; tout += o
json.dump(res, open(os.path.join(WORK, a.out), "w"))
inc = sum(1 for v in res.values() for d in v if d.get('include'))
tot = sum(len(v) for v in res.values())
el = time.time() - t0
cost = tin/1e6*0.25 + tout/1e6*2.00
print(f"studies {len(res)}  analyses {tot}  included {inc} ({inc/max(tot,1)*100:.1f}%)")
print(f"tokens in {tin:,} out {tout:,}   {el:.0f}s   est USD ${cost:.4f}")
print(f"per-study: {tin/max(len(res),1):.0f} in, {tout/max(len(res),1):.0f} out")

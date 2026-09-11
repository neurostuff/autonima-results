import json, re, textwrap
from pathlib import Path
E4 = ["11950456","16199014","17892884","19914045","19996042","25000505",
      "26424424","26535944","27082610","28888350","29761009","31662209"]
d = Path("projects/vbm_of_ptsd/v1-record-no-evidence/outputs")
dec = {str(r["study_id"]): r for r in
       json.loads((d/"fulltext_screening_results.json").read_text())["screening_results"]}
sr = json.loads((d/"search_results.json").read_text())
rows = sr if isinstance(sr, list) else (sr.get("studies") or sr.get("results") or [])
meta = {str(r.get("pmid")): r for r in rows}
CUE = re.compile(r"[^.]*?\b(voxel-based morphometry|VBM|region[s]? of interest|ROI|"
                 r"manual(ly)? (trac|segment)\w*|volumetr\w+|whole[- ]brain|FreeSurfer|"
                 r"segmentation|tracing)\b[^.]*\.", re.I)
for p in E4:
    m = meta.get(p, {})
    ab = m.get("abstract") or ""
    sents = [s.strip() for s in CUE.findall(ab)] if False else [
        s.group(0).strip() for s in CUE.finditer(ab)]
    print("=" * 98)
    print("%s  %s" % (p, (m.get("title") or "?")[:82]))
    if sents:
        for s in sents[:3]:
            print(textwrap.fill(s, 94, initial_indent="   METHOD> ", subsequent_indent="            "))
    else:
        print("   METHOD> (no method cue in abstract)")

import re, html
from pathlib import Path
PATHS = {
 "11950456":"/data/james/pondie-vs-fulltext/articles/ace_outputs/html/manual/11950456.html",
 "16199014":"/data/james/pondie-vs-fulltext/articles/ace_outputs/html/manual/16199014.html",
 "17892884":"/data/james/pondie-vs-fulltext/articles/elsevier_output/17892884/text.txt",
 "19996042":"/data/james/pondie-vs-fulltext/articles/ace_outputs/html/manual/19996042.html",
 "25000505":"/data/james/pondie-vs-fulltext/pubget_pmc/text_by_pmid/25000505/text.txt",
 "27082610":"/data/james/pondie-vs-fulltext/pubget_pmc/text_by_pmid/27082610/text.txt",
 "31662209":"/data/james/pondie-vs-fulltext/articles/ace_outputs/html/manual/31662209.html",
}
CUE = re.compile(r"[^.]{0,200}\b(whole[- ]brain|region[s]? of interest|ROI|small volume correct\w*|"
                 r"SVC|manual(ly)? (trac|segment)\w*|voxel-based morphometry|VBM|"
                 r"FreeSurfer|mask\w*)\b[^.]{0,200}\.", re.I)
for p, f in PATHS.items():
    t = Path(f).read_text(errors="ignore")
    if f.endswith(".html"):
        t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S|re.I)
        t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"\s+", " ", t)
    hits, seen = [], set()
    for m in CUE.finditer(t):
        s = m.group(0).strip()
        k = s[:60].lower()
        if k not in seen and len(s) > 40:
            seen.add(k); hits.append(s)
    print("=" * 98)
    print("%s   (%s)" % (p, Path(f).name))
    for s in hits[:4]:
        print("   -", s[:300])
    if not hits: print("   (no method cue found)")

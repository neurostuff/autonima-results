import re, sys
from pathlib import Path

def patch(path):
    lines = Path(path).read_text().splitlines(keepends=True)
    out, in_ann, in_mf, changed = [], False, False, False
    for ln in lines:
        if re.match(r"^annotation:\s*$", ln):
            in_ann, in_mf = True, False
        elif in_ann and re.match(r"^[A-Za-z_][A-Za-z0-9_]*:", ln):
            in_ann, in_mf = False, False
        if in_ann and re.match(r"^\s{2}metadata_fields:\s*$", ln):
            in_mf = True
            out.append(ln); continue
        if in_mf:
            if re.match(r"^\s{2}-\s", ln):
                if ln.strip() == "- study_abstract":
                    out.append(ln.replace("study_abstract", "study_fulltext"))
                    changed = True
                    continue
            else:
                in_mf = False
        out.append(ln)
    if not changed:
        print("  %s: NO CHANGE (no study_abstract in annotation.metadata_fields)" % path)
        return False
    Path(path).write_text("".join(out))
    print("  %s: patched" % path)
    return True

for p in sys.argv[1:]:
    patch(p)

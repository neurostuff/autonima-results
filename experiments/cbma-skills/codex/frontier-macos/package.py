"""Write a shareable ZIP with no symlinks, credentials, caches or machine state."""
from pathlib import Path
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root.with_suffix('.zip')
if out.is_relative_to(root):
    sys.exit('ZIP must be outside the package folder')
skip = {'.run-state', '__pycache__', '.git', '.DS_Store'}
files = sorted(p for p in root.rglob('*') if not any(part in skip for part in p.relative_to(root).parts))
if any(p.is_symlink() for p in files):
    sys.exit('refusing to package symlinks')
if any(p.name in {'auth.json', 'portkey.key', '.env'} for p in files):
    sys.exit('refusing to package credential files')
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as z:
    for p in files:
        if p.is_file():
            z.write(p, root.name + '/' + str(p.relative_to(root)))
print(out)

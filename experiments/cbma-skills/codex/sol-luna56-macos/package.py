"""Build an article-free ZIP with matching fingerprints and regular-file modes."""
from pathlib import Path
import hashlib
import os
import sys
import zipfile


root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else root.with_suffix('.zip')
if out.is_relative_to(root):
    sys.exit('ZIP must be outside the package folder')


def excluded(rel):
    parts = rel.parts
    if any(p in {'.run-state', '__pycache__', '.git', '.DS_Store'} for p in parts):
        return True
    if parts[0] == 'corpora' or parts[:2] == ('runtime', 'data'):
        return True
    return any(parts[i:i + 2] == ('fulltext', 'raw') for i in range(len(parts) - 1))


files = []
for parent, directories, filenames in os.walk(root, followlinks=False):
    directory = Path(parent)
    for name in list(directories):
        item = directory / name
        if excluded(item.relative_to(root)):
            directories.remove(name)
        elif item.is_symlink():
            sys.exit(f'refusing to package symlink: {item.relative_to(root)}')
    for name in filenames:
        item = directory / name
        relative = item.relative_to(root)
        if excluded(relative) or name == 'MANIFEST.sha256':
            continue
        if item.is_symlink():
            sys.exit(f'refusing to package symlink: {relative}')
        if name in {'auth.json', 'portkey.key', '.env', '.env.local'}:
            sys.exit(f'refusing to package credential file: {relative}')
        files.append(item)
files.sort()
manifest = root / 'MANIFEST.sha256'
with manifest.open('w') as output:
    for item in files:
        with item.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        output.write(f'{digest}  {item.relative_to(root)}\n')
files.append(manifest)
out.parent.mkdir(parents=True, exist_ok=True)
temporary = out.with_name(out.name + '.partial')
with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                     compresslevel=4, allowZip64=True) as archive:
    for item in files:
        archive.write(item, root.name + '/' + str(item.relative_to(root)))
temporary.replace(out)
print(out)

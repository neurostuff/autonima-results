"""Copy only fingerprinted raw article inputs from a prior bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


def entries():
    for line in (ROOT / 'ARTICLE_INPUTS.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts or len(digest) != 64:
            raise ValueError('invalid article manifest entry')
        corpus = (len(rel.parts) >= 3 and rel.parts[0] == 'corpora'
                  and rel.parts[1] in {'elsevier_output', 'elsevier_problem_solving', 'ace_html'}
                  and rel.suffix in {'.xml', '.html'})
        pmc = (len(rel.parts) == 8 and rel.parts[:2] == ('luna', 'projects')
               and rel.parts[2] == rel.parts[3]
               and rel.parts[4:7] == ('fulltext', 'raw', 'pmc') and rel.suffix == '.xml')
        if not (corpus or pmc):
            raise ValueError(f'non-article manifest path: {rel}')
        yield digest, rel


def fingerprint(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', nargs='?', type=Path,
                    help='existing e6-codex-frontier-macos folder')
    ap.add_argument('--check', action='store_true', help='require listed local corpus files before model calls')
    args = ap.parse_args()
    records = list(entries())
    if args.check:
        missing = [str(rel) for _, rel in records
                   if rel.parts[0] == 'corpora' and not (ROOT / rel).is_file()]
        if missing:
            print(f'{len(missing)} required corpus files missing; first: {missing[0]}', file=sys.stderr)
            print('Run ./copy_articles.sh /path/to/e6-codex-frontier-macos before launching.', file=sys.stderr)
            return 2
        return 0
    if args.source is None:
        ap.error('source folder is required unless --check is used')
    source = args.source.expanduser().resolve()
    if source == ROOT:
        ap.error('source must be the existing bundle, not this new folder')
    if not source.is_dir():
        ap.error('source folder does not exist')
    summary = {'copied': 0, 'already_present': 0, 'missing_optional_pmc': [], 'errors': []}
    for digest, rel in records:
        destination = ROOT / rel
        if destination.is_symlink():
            summary['errors'].append(f'destination is a symlink: {rel}')
            continue
        if any(parent.is_symlink() for parent in destination.parents if parent != ROOT and ROOT in parent.parents):
            summary['errors'].append(f'destination parent is a symlink: {rel}')
            continue
        if destination.is_file():
            if fingerprint(destination) == digest:
                summary['already_present'] += 1
            else:
                summary['errors'].append(f'existing input differs; left unchanged: {rel}')
            continue
        candidates = [source / rel]
        if rel.parts[0] == 'luna':
            candidates.append(source / Path(*rel.parts[1:]))
        matching = None
        for candidate in candidates:
            if candidate.is_file() and fingerprint(candidate) == digest:
                matching = candidate
                break
        if matching is None:
            if any(p.is_file() for p in candidates):
                summary['errors'].append(f'input fingerprint differs: {rel}')
            elif rel.parts[0] == 'corpora':
                summary['errors'].append(f'required source input missing: {rel}')
            else:
                summary['missing_optional_pmc'].append(str(rel))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.copying')
        shutil.copyfile(matching, temporary, follow_symlinks=True)
        if fingerprint(temporary) != digest:
            temporary.unlink()
            summary['errors'].append(f'source changed while copying: {rel}')
            continue
        temporary.replace(destination)
        summary['copied'] += 1
    print(json.dumps(summary, indent=2))
    return 2 if summary['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())

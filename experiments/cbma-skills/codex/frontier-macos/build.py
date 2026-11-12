"""Build the Apple Silicon frontier arm from clean E6 inputs and pinned assets.

Run with the existing Linux CBMA Python. Mac binaries are packaged, not executed.
Requires a populated dependencies/wheels-macos-arm64 and requirements.lock.txt.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

SOURCE = Path(__file__).resolve().parent
ARM = Path(sys.argv[1]).resolve()
OLD = Path('/home/zorro/repos/cbma-workspaces/e6-codex')
SKILLS = OLD / 'projects/dementia/.agents/skills'
PROJECTS = sorted(p.name for p in (OLD / 'reviews').iterdir() if (p / 'review.yaml').exists())
MODELS = {stage: {'model': 'gpt-6-luna' if stage == 'extraction' else 'gpt-6-astra',
                  'effort': 'low' if stage == 'extraction' else 'max'}
          for stage in ('abstract', 'fulltext', 'extraction', 'selection')}
ASSETS = []


def digest(path, algo='sha256'):
    h = hashlib.new(algo)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, expected, algo='sha256'):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as f:
            shutil.copyfileobj(response, f)
    if digest(path, algo) != expected:
        raise RuntimeError('asset checksum mismatch: ' + path.name)
    ASSETS.append({'file': str(path.relative_to(ARM)), 'url': url, algo: expected})
    return path


def unpack_copy(archive, component, destination):
    # Extract only into staging; follow links into regular copies in the package.
    # tarfile's data filter rejects escaping paths and links.
    with tempfile.TemporaryDirectory(prefix='cbma-macos-') as tmp:
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter='data')
        shutil.copytree(Path(tmp) / component, destination, symlinks=False,
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))


def write(path, content, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(0o755)


def runtime():
    release_url = 'https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/20261001'
    with urllib.request.urlopen(release_url, timeout=60) as response:
        release = json.load(response)
    asset = next(a for a in release['assets'] if a['name'] ==
                 'cpython-3.12.15+20261001-aarch64-apple-darwin-install_only_stripped.tar.gz')
    archive = download(asset['browser_download_url'], ARM / 'dependencies' / asset['name'],
                       asset['digest'].split(':', 1)[1])
    unpack_copy(archive, 'python', ARM / 'runtime/python')
    url = 'https://registry.npmjs.org/@openai/codex/0.160.0-darwin-arm64'
    with urllib.request.urlopen(url, timeout=60) as response:
        package = json.load(response)
    expected = base64.b64decode(package['dist']['integrity'].split('-', 1)[1]).hex()
    archive = download(package['dist']['tarball'], ARM / 'dependencies/codex-0.160.0-darwin-arm64.tgz',
                       expected, 'sha512')
    unpack_copy(archive, 'package', ARM / 'runtime/codex')
    site = ARM / 'runtime/site-packages'
    site.mkdir(parents=True, exist_ok=True)
    # Materialize wheels, including their bundled native libraries and metadata.
    for wheel in sorted((ARM / 'dependencies/wheels-macos-arm64').glob('*.whl')):
        with zipfile.ZipFile(wheel) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                parts = Path(member.filename).parts
                if '..' in parts or Path(member.filename).is_absolute():
                    raise RuntimeError('unsafe wheel path')
                if parts[0].endswith('.data'):
                    if len(parts) < 3 or parts[1] not in {'purelib', 'platlib'}:
                        continue
                    parts = parts[2:]
                target = site.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(member))
                mode = (member.external_attr >> 16) & 0o777
                if mode:
                    target.chmod(mode)
        ASSETS.append({'file': str(wheel.relative_to(ARM)), 'sha256': digest(wheel)})
    parser = Path('/home/zorro/repos/elsevier_coordinate_extractor/elsevier_coordinate_extraction')
    shutil.copytree(parser, site / parser.name, symlinks=False, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(Path('/home/zorro/repos/elsevier_coordinate_extractor/elsevier_coordinate_extraction.egg-info'),
                    site / 'elsevier_coordinate_extraction.egg-info', symlinks=False, dirs_exist_ok=True)
    write(ARM / 'bin/python', '''#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "This runtime requires Apple Silicon macOS (run Terminal natively, not under Rosetta)." >&2; exit 2
fi
export PYTHONHOME="$root/runtime/python"
export PYTHONPATH="$root/runtime/site-packages"
export PYTHONNOUSERSITE=1
export SSL_CERT_FILE="$root/runtime/site-packages/certifi/cacert.pem"
export NILEARN_DATA="$root/runtime/data/nilearn"
exec "$root/runtime/python/bin/python3.12" "$@"
''', True)
    shutil.copy2(ARM / 'bin/python', ARM / 'bin/python3')
    write(ARM / 'bin/codex', '''#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "This Codex binary requires Apple Silicon macOS." >&2; exit 2
fi
vendor="$root/runtime/codex/vendor/aarch64-apple-darwin"
export PATH="$root/bin:$vendor/codex-path:/usr/bin:/bin:/usr/sbin:/sbin"
profile="${FRONTIER_CODEX_HOME:-$HOME/.codex-e6-frontier}"
mkdir -p "$profile"
exec env CODEX_HOME="$profile" "$vendor/bin/codex" "$@"
''', True)
    print('Mac Python, Codex and Python dependencies packaged', flush=True)


def corpora():
    stats = {}
    for name in ['elsevier_output', 'elsevier_problem_solving', 'ace_html']:
        src = (OLD / 'corpora' / name).resolve()
        count = size = 0
        for p in sorted(src.rglob('*')):
            if not p.is_file() or any(part.startswith('.') for part in p.relative_to(src).parts):
                continue
            # Only original article inputs, not saved extracted coordinates/tables.
            if (name == 'ace_html' and p.suffix.lower() != '.html') or \
               (name != 'ace_html' and p.suffix.lower() != '.xml'):
                continue
            dst = ARM / 'corpora' / name / p.relative_to(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p.resolve(), dst)
            count += 1
            size += p.stat().st_size
        stats[name] = {'article_files': count, 'bytes': size}
        print('Copied', name, count, 'articles', flush=True)
    return stats


def runner():
    source = (SOURCE.parent / 'run_stage.py').read_text()
    source = source.replace("if arm.name not in {'e6-codex', 'e6-codex-luna'}:",
                            "if not (arm / 'arm.json').is_file():")
    source = source.replace("model = 'gpt-6-luna' if arm.name == 'e6-codex-luna' or args.stage in {'abstract', 'fulltext'} else 'gpt-6-astra'\n    effort = 'medium' if args.stage in {'fulltext', 'selection'} else 'low'",
                            "cfg = json.loads((arm / 'arm.json').read_text())['stage_models'][args.stage]\n    model, effort = cfg['model'], cfg['effort']")
    start = source.index('def other_workers(ws):')
    end = source.index('\ndef judge_one(', start)
    source = source[:start] + '''def other_workers(ws):
    result = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True, check=True)
    found = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit() or int(fields[0]) == os.getpid():
            continue
        command = fields[1]
        if str(ws / ws.name / 'work') in command and (' exec ' in command or 'judge.py' in command):
            found.append(int(fields[0]))
    return found

''' + source[end:]
    source = source.replace("ap.add_argument('--limit',", "ap.add_argument('--defer-pmids', nargs='*', default=[], help='documented missing-input cases only; do not use for scientific exclusions or tool failures')\n    ap.add_argument('--limit',")
    source = source.replace("return [p for p in values if scope is None or p in scope]",
                            "return [p for p in values if (scope is None or p in scope) and p not in args.defer_pmids]")
    source = source.replace("ingest()  # recover finished, un-ingested batches from an interrupted run",
                            """ingest()  # recover finished, un-ingested batches from an interrupted run
        for batch in ledger.batch_files(work):
            data = json.loads(batch.read_text())
            if any(i['pmid'] in args.defer_pmids for i in data['items']):
                ledger._archive(batch, ledger._output_path(batch, args.stage), log_dir / 'deferred')
        deferred = sorted(set(args.defer_pmids) & set(ledger.Review(review).pending(args.stage)))
        result['deferred_input_pmids'] = deferred""")
    # _archive requires the destination folder to exist, including the first deferral.
    source = source.replace("ledger._archive(batch, ledger._output_path(batch, args.stage), log_dir / 'deferred')",
                            "(log_dir / 'deferred').mkdir(exist_ok=True)\n                ledger._archive(batch, ledger._output_path(batch, args.stage), log_dir / 'deferred')")
    source = source.replace("if scope is not None:\n                    original_pending", "if scope is not None or args.defer_pmids:\n                    original_pending")
    source = source.replace("if stage != args.stage or p in scope]", "if stage != args.stage or ((scope is None or p in scope) and p not in args.defer_pmids)]")
    source = source.replace("verdict='JUDGMENTS_DONE' if not pending() else 'WORK_REMAINS'",
                            "verdict=('DONE_WITH_INCOMPLETE_INPUTS' if deferred else 'JUDGMENTS_DONE') if not pending() else 'WORK_REMAINS'")
    write(ARM / 'run_stage.py', source, True)


def projects():
    records = {}
    for project in PROJECTS:
        ws = ARM / 'projects' / project
        review = ws / project
        review.mkdir(parents=True, exist_ok=True)
        protocol = OLD / 'reviews' / project / 'review.yaml'
        text = protocol.read_text().replace('path: ../corpora/', 'path: ../../../corpora/')
        write(review / 'review.yaml', text)
        shutil.copy2(protocol, review / 'registered-review.yaml')
        shutil.copytree(SKILLS, ws / '.agents/skills', symlinks=False, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        write(ws / '.venv/bin/python', '''#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec "$root/bin/python" "$@"
''', True)
        shutil.copy2(ws / '.venv/bin/python', ws / '.venv/bin/python3')
        write(ws / 'AGENTS.md', instructions(project))
        # Warm only original PMC article inputs. No search results or prior judgments.
        pmc = review / 'fulltext/raw/pmc'
        pmc.mkdir(parents=True, exist_ok=True)
        n = 0
        for arm in ['e6-codex', 'e6-codex-luna', 'main', 'haiku']:
            root = OLD.parent / arm
            for old_review in [root / 'projects' / project / project, root / project / project]:
                for raw in (old_review / 'fulltext/raw/pmc').glob('*.xml'):
                    if not (pmc / raw.name).exists():
                        shutil.copy2(raw.resolve(), pmc / raw.name)
                        n += 1
        for helper in ('single_response.py', 'stage_bookkeeping.py'):
            shutil.copy2(SOURCE.parent.parent / 'skills/cbma-review/scripts' / helper,
                         ws / '.agents/skills/cbma-review/scripts' / helper)
        subprocess.run([sys.executable, str(ws / '.agents/skills/cbma-review/scripts/ledger.py'),
                        'init', str(review)], check=True, stdout=subprocess.DEVNULL)
        source_sha = digest(protocol)
        packaged_sha = digest(review / 'review.yaml')
        notes = f'''# Frontier E6 run notes: {project}

- New arm: e6-codex-frontier-macos; no previous searches or judgments copied.
- Original E6 scientific skill snapshot: f3a68422262350ee3ddcc36fc89f416695f10d10.
- Source protocol SHA-256: {source_sha} (registered-review.yaml).
- Packaged protocol SHA-256: {packaged_sha} (review.yaml).
- Packaging deviation: local corpus paths changed to ../../../corpora; scientific fields unchanged.
- Models: Astra/max orchestration, abstract screening, full-text screening and selection;
  Luna/low coordinate extraction. Ledger attribution records actual model and effort.
- Account: separate ChatGPT/Codex login; no Portkey or API key copied.
- Target: Apple Silicon macOS. Bundled CPython 3.12.15, Codex 0.160.0,
  NiMARE 0.21.0 and pinned macOS dependencies. Original Python environment was 3.12.3.
- Cached PMC articles copied: {n}; these are source material, not decisions.
- Incomplete-input continuation: missing source cases remain unavailable/incomplete/pending;
  usable studies proceed under the original criteria. Explicit missing-input deferrals
  must be listed here; final export may use --allow-pending with those cases disclosed.
- Search date/count:
- Stage counts and scientific audits:
- Missing-input PMIDs, reasons and retry attempts:
- Export/meta-analysis outputs and skipped targets:
- Actual account/model availability and Codex version:
- Run-time deviations:
'''
        write(review / 'RUN_NOTES.md', notes)
        records[project] = {'source_protocol_sha256': source_sha, 'protocol_sha256': packaged_sha,
                            'cached_pmc_articles': n}
        print('Initialized fresh review', project, flush=True)
    return records


def instructions(project):
    return f'''# Frontier E6 Codex: {project}

Run only `{project}` from this workspace. REVIEW=`./{project}`. Use the local
`.agents/skills/` snapshot and `.venv/bin/python` (a regular wrapper file, not a
symlink). All dependencies and corpora are inside the arm's enclosing folder.

## Models and account

- Orchestration: gpt-6-astra, max effort.
- Abstract screening: gpt-6-astra, max effort.
- Full-text screening: gpt-6-astra, max effort.
- Coordinate extraction: gpt-6-luna, low effort.
- Analysis selection: gpt-6-astra, max effort.

Use the bundled launcher and separate ChatGPT/Codex login. Do not use Portkey,
native spawn_agent or a different account/provider. The runner selects stage models
and records codex-cli/<model>/effort-<effort> through ledger ingestion.

## Scientific rules

Follow cbma-review and each scientific stage skill exactly. Use full article text;
no trimmed or records mode. Criteria and review-specific permissions come from
review.yaml, including permissions to use figures or supplements. Do not add a
table-only eligibility requirement when the registered criteria permit other evidence.
registered-review.yaml preserves the source protocol for criteria comparison;
only local corpus paths changed in the runnable copy. Gold, other arms' judgments
and extracted coordinate records are not inputs and are not bundled.

## Execution

Before judgment, run ledger init, compare numbered criteria with the registered
protocol, and record hashes/model settings in RUN_NOTES.md. Preserve the original
50-record abstract pilot; audit it and continue automatically if it passes. The user
authorized uninterrupted end-to-end execution. Never rerun completed ledger decisions.

For each judged stage use:

```bash
.venv/bin/python ../../run_stage.py {project} --stage STAGE
```

Use --limit 50 for the pilot and repeat with the same limit until it completes.
For full stages omit --limit. The runner batches, dispatches fresh headless judges,
ingests through this project's ledger, retries boundedly and returns a compact
summary. Judges have exact skills preloaded and read texts_file bundles in full.
Read original tables where required by criteria or table alignment checks.

- Exit 0: perform the scientific stage audit, then continue.
- Exit 3: work remains; repeat after a time-budget checkpoint. Inspect persistent
  failures after bounded retries rather than resetting attempt counters.
- Exit 4: another worker is active; wait, do not duplicate work.
- Exit 1: inspect an execution failure. Fix auxiliary command errors and continue;
  report persistent bundled-tool/model/account failures.

## Incomplete inputs: continue the usable cases

An unavailable or incomplete article is not a scientific exclusion. Apply the
original screening skill: genuinely incomplete source files receive incomplete,
return to retrieval and remain separately reported. A complete article whose
eligibility evidence is not demonstrated follows the original full-text rules.
Do not invent evidence, coordinates, space labels or criterion revisions.

Retry missing-source retrieval within the workflow's bounded recovery rules. If
the source/table/supplement remains missing, record PMID, reason and attempts in
RUN_NOTES.md and continue the other studies. For extraction/selection pending
solely because of those documented inputs, rerun the stage with:

```bash
.venv/bin/python ../../run_stage.py {project} --stage STAGE --defer-pmids PMID1 PMID2
```

The runner leaves deferred items pending in the ledger, preserves their batches
and processes usable studies. Deferral is only for confirmed missing source input,
not contradictory judgments, invalid model output, scientific audit failures or
account/tool errors. Reapply the same deferral list on later checkpoints.

Missing-input cases alone do not stop downstream stages. Stage audits must pass
for completed decisions. Export with --allow-pending when documented missing-input
cases remain; name them and the resulting reduced study coverage in the report.
Targets below the registered minimum are skipped by the meta script; report and
continue the remaining targets. Genuine criterion ambiguity still requires a
recorded user protocol decision. Failures are never converted to exclusions.

## Meta-analysis and reporting

Use bundled cbma-nimare/scripts/run_meta.py and review.yaml defaults, with pinned
NiMARE 0.21.0. Check coordinate spaces, repeated samples and numerical diagnostics.
Do not change thresholds or guess unknown spaces. Routine reporting uses summaries
and the export report; do not routinely open plots, inspect NiMARE internals or
browse documentation. Investigate only a concrete error, required scientific check
or explicit user request, and record why. Report stage counts, missing inputs,
deferred studies, dropped points, skipped targets and output files.
'''


def main():
    runtime()
    article_stats = corpora()
    for filename in ['codex_session.sh', 'run_codex.sh', 'run_review.sh', 'login.sh']:
        shutil.copy2(SOURCE / filename, ARM / filename)
        (ARM / filename).chmod(0o755)
    (ARM / 'helpers').mkdir(exist_ok=True)
    shutil.copy2(SOURCE / 'judge.py', ARM / 'helpers/judge.py')
    shutil.copy2(SOURCE / 'package.py', ARM / 'helpers/package.py')
    shutil.copy2(SOURCE.parent.parent / 'benchmark/audit_codex_transcripts.py', ARM / 'helpers/audit_codex_transcripts.py')
    write(ARM / 'run_judge_optimized.sh', '''#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$root/bin/python" "$root/helpers/judge.py" "$@"
''', True)
    write(ARM / 'package.sh', '''#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$root/bin/python" "$root/helpers/package.py" "$@"
''', True)
    runner()
    project_info = projects()
    manifest = {'arm': ARM.name, 'platform': 'macOS Apple Silicon', 'stage_models': MODELS,
                'orchestrator': {'model': 'gpt-6-astra', 'effort': 'max'},
                'account': 'separate ChatGPT/Codex login', 'source_arm': 'e6-codex clean review templates',
                'python': '3.12.15', 'nimare': '0.21.0', 'codex': '0.160.0',
                'corpora': article_stats, 'projects': project_info, 'assets': ASSETS,
                'mac_runtime_executed': False}
    write(ARM / 'arm.json', json.dumps(manifest, indent=2) + '\n')
    symlinks = [str(p.relative_to(ARM)) for p in ARM.rglob('*') if p.is_symlink()]
    if symlinks:
        raise RuntimeError('unexpected symlinks: ' + str(symlinks[:10]))
    print('Package assembled; no symlinks. Mac binaries were not executed.', flush=True)


if __name__ == '__main__':
    main()

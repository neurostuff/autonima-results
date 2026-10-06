"""Add four fresh Luna reviews to the Apple Silicon bundle, without duplicating runtime."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(sys.argv[1]).resolve()
SOURCE = Path(__file__).resolve().parent
OLD = Path('/home/zorro/repos/cbma-workspaces/e6-codex-luna')
PROJECTS = ['decision_making', 'dementia', 'executive_function', 'social']
spec = importlib.util.spec_from_file_location('frontier_build', SOURCE / 'build.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
LUNA = ROOT / 'luna'
LUNA.mkdir(exist_ok=True)

# All bundled experiments use the authorized uninterrupted-run instructions.
for project in build.PROJECTS:
    build.write(ROOT / 'projects' / project / 'AGENTS.md', build.instructions(project))
for name in ['run_codex.sh', 'run_review.sh', 'START_PROMPT.txt']:
    shutil.copy2(SOURCE / name, ROOT / name)
    if name.endswith('.sh'):
        (ROOT / name).chmod(0o755)

models = {s: {'model': 'gpt-6-luna', 'effort': 'medium' if s in {'fulltext', 'selection'} else 'low'}
          for s in ('abstract', 'fulltext', 'extraction', 'selection')}
manifest = {'arm': 'e6-codex-luna-macos', 'stage_models': models,
            'orchestrator': {'model': 'gpt-6.1-sol', 'effort': 'medium'},
            'shared_runtime': '../runtime', 'shared_corpora': '../corpora',
            'account': 'separate bundle ChatGPT/Codex login', 'projects': {}}
for name in ['run_stage.py', 'run_judge_optimized.sh']:
    shutil.copy2(ROOT / name, LUNA / name)
(LUNA / 'helpers').mkdir(exist_ok=True)
shutil.copy2(ROOT / 'helpers/judge.py', LUNA / 'helpers/judge.py')
for name in ['run_codex.sh', 'run_review.sh']:
    text = (ROOT / name).read_text().replace('gpt-6-astra max', 'gpt-6.1-sol medium')
    build.write(LUNA / name, text, True)
transport = (ROOT / 'codex_session.sh').read_text().replace(
    '$(dirname "${BASH_SOURCE[0]}")" && pwd)', '$(dirname "${BASH_SOURCE[0]}")/.." && pwd)')
build.write(LUNA / 'codex_session.sh', transport, True)
shutil.copy2(SOURCE / 'START_PROMPT.txt', LUNA / 'START_PROMPT.txt')
for name in ['python', 'python3', 'codex']:
    build.write(LUNA / 'bin' / name, f'''#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
exec "$root/bin/{name}" "$@"
''', True)

for project in PROJECTS:
    original_ws = OLD / 'projects' / project
    ws = LUNA / 'projects' / project
    review = ws / project
    review.mkdir(parents=True, exist_ok=True)
    protocol = original_ws / project / 'review.yaml'
    build.write(review / 'review.yaml', protocol.read_text().replace('path: ../corpora/', 'path: ../../../../corpora/'))
    shutil.copy2(protocol, review / 'registered-review.yaml')
    shutil.copytree(original_ws / '.agents/skills', ws / '.agents/skills', symlinks=False,
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    text = build.instructions(project).replace('# Frontier E6 Codex:', '# Luna E6 Codex:')
    text = text.replace('- Orchestration: gpt-6-astra, max effort.', '- Orchestration: gpt-6.1-sol, medium effort.')
    text = text.replace('- Abstract screening: gpt-6-astra, max effort.', '- Abstract screening: gpt-6-luna, low effort.')
    text = text.replace('- Full-text screening: gpt-6-astra, max effort.', '- Full-text screening: gpt-6-luna, medium effort.')
    text = text.replace('- Analysis selection: gpt-6-astra, max effort.', '- Analysis selection: gpt-6-luna, medium effort.')
    text = text.replace('only local corpus paths changed in the runnable copy.', 'only local corpus paths changed in the runnable copy.')
    text = text.replace('Preserve the original\n50-record abstract pilot; audit it and continue automatically if it passes. The user\nauthorized uninterrupted end-to-end execution.',
                        'This Luna arm has no pilot. The user authorized uninterrupted end-to-end execution.')
    text = text.replace('Use --limit 50 for the pilot and repeat with the same limit until it completes.\nFor full stages omit --limit.',
                        'For full stages omit --limit; a pilot is only run on an explicit user request.')
    build.write(ws / 'AGENTS.md', text)
    build.write(ws / '.venv/bin/python', '''#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
exec "$root/bin/python" "$@"
''', True)
    shutil.copy2(ws / '.venv/bin/python', ws / '.venv/bin/python3')
    # Actual raw article copies, not ledger state from another experiment.
    pmc_source = ROOT / 'projects' / project / project / 'fulltext/raw/pmc'
    shutil.copytree(pmc_source, review / 'fulltext/raw/pmc', symlinks=False, dirs_exist_ok=True)
    subprocess.run([sys.executable, str(ws / '.agents/skills/cbma-review/scripts/ledger.py'),
                    'init', str(review)], check=True, stdout=subprocess.DEVNULL)
    build.write(review / 'RUN_NOTES.md', f'''# Luna E6 macOS run notes: {project}

- Fresh experiment copied from the unstarted e6-codex-luna project setup.
- No searches, decisions, extracted coordinates or gold files copied.
- Skills snapshot: a0ef67e16c3bdcfa31b170b07a58ea7910f80fec.
- Source protocol SHA-256: {build.digest(protocol)}.
- Packaged protocol SHA-256: {build.digest(review / 'review.yaml')}.
- Corpus path-only packaging deviation: ../../../../corpora; scientific fields unchanged.
- Models: gpt-6-luna judges (abstract low, full-text medium, extraction low,
  selection medium); gpt-6.1-sol orchestration at medium effort.
- Transport deviation: separate ChatGPT/Codex login instead of the original Portkey route.
- Runtime: shared bundled Mac CPython 3.12.15, Codex 0.160.0, NiMARE 0.21.0.
- Uninterrupted end-to-end run authorized; no pilot or stage approval pauses.
- Missing-input continuation authorized: record bounded attempts and missing PMIDs,
  leave them unavailable/incomplete/pending, complete usable cases under original criteria.
- Search date and counts:
- Stage counts/audits:
- Missing inputs, reasons and deferred PMIDs:
- Export coverage, dropped points, skipped/blocked targets and result paths:
- Actual account model access and run-time deviations:
''')
    manifest['projects'][project] = {'source_protocol_sha256': build.digest(protocol),
                                     'protocol_sha256': build.digest(review / 'review.yaml'),
                                     'cached_pmc_articles': len(list((review / 'fulltext/raw/pmc').glob('*.xml')))}
    print('Added fresh Luna review', project, flush=True)
build.write(LUNA / 'arm.json', json.dumps(manifest, indent=2) + '\n')
root_manifest = json.loads((ROOT / 'arm.json').read_text())
root_manifest['additional_arms'] = {'luna': {'manifest': 'luna/arm.json', 'projects': PROJECTS}}
root_manifest['uninterrupted_execution'] = True
build.write(ROOT / 'arm.json', json.dumps(root_manifest, indent=2) + '\n')
build.write(LUNA / 'README.md', '''# Luna arm inside the Apple Silicon bundle

Projects: decision_making, dementia, executive_function and social.
All judges use gpt-6-luna at the original stage efforts; the orchestrator uses
gpt-6.1-sol at medium effort. Scientific skills come from the existing Luna arm.
Ledgers are fresh. Runtime and copied corpora are shared by relative paths inside
the enclosing bundle; every project has its own skill copy and output folder.

From the enclosing bundle, sign in once with `bash login.sh`, then run:

```bash
./luna/run_codex.sh dementia
```

For a headless end-to-end run:

```bash
./luna/run_review.sh dementia
```

Both commands automatically supply the uninterrupted-run prompt. There is no pilot
or stage-approval pause. Missing source cases are reported and usable cases continue;
scientific criteria, audits and ledger validation are preserved. Run each project
in its own terminal/process. Use the enclosing package.sh to share the whole bundle;
this subfolder requires the enclosing runtime and corpora.
''')
print('Luna arm added; no runs or model calls launched.', flush=True)

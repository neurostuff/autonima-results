"""Mechanical retry identities and pending-batch filtering, shared by both harnesses."""
import hashlib
import json
from pathlib import Path
import re
import uuid

import single_response


def identity(batch, item):
    payload = [batch['_criteria_hash'], item['pmid'], batch['_input_hashes'][item['pmid']],
               batch.get('_analyses_hashes', {}).get(item['pmid'])]
    if batch.get('stage') == 'extraction':
        # Attempts from another extraction input (tables_only, the full paper, ...) are a different
        # job: re-extracting from the full paper must not inherit a tables-only attempt count.
        payload.append(batch.get('input_view', 'full'))
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def write_state(path, value):
    single_response.atomic_json(Path(path), json.dumps(value, indent=1) + '\n')


def eligible_batches(batches, pending, attempts, maximum, state_dir):
    """Exhausted items stay scientifically pending without blocking healthy batch mates."""
    pending = set(pending)
    jobs, exhausted = [], set()
    for path in batches:
        batch = json.loads(path.read_text())
        eligible = []
        for item in batch['items']:
            if item['pmid'] not in pending:
                continue
            if attempts.get(identity(batch, item), 0) >= maximum:
                exhausted.add(item['pmid'])
            else:
                eligible.append(item)
        if len(eligible) != len(batch['items']):
            archive = Path(state_dir) / 'filtered_batches'
            archive.mkdir(parents=True, exist_ok=True)
            write_state(archive / (path.stem + '.' + uuid.uuid4().hex + '.json'), batch)
            if not eligible:
                path.unlink()
                continue
            batch['items'] = eligible
            pmids = {i['pmid'] for i in eligible}
            for field in ('_input_hashes', '_analyses_hashes'):
                if field in batch:
                    batch[field] = {p: v for p, v in batch[field].items() if p in pmids}
            write_state(path, batch)
        jobs.append((path, [identity(batch, i) for i in eligible]))
    return jobs, exhausted


def provider_blocked(error):
    # Claude Code says "You've hit your session limit · resets ..."; it was not matched, so the
    # runner kept dispatching through two session limits (the Haiku social and decision_making
    # runs: 216 and 728 refused calls), each counted against the study's attempt cap.
    return bool(re.search(r'\b429\b|rate[_ -]?limit|usage limit|session limit|weekly limit|'
                          r'hit your [a-z ]*limit|quota|'
                          r'authentication|unauthorized|HTTP (?:401|403)|insufficient credits',
                          error or '', re.I))

"""Portable Codex adapter for the shared single-response judge path."""
from pathlib import Path
import re
import sys


def main():
    if len(sys.argv) != 4 or not re.fullmatch(r'[a-z0-9_]+', sys.argv[1]):
        sys.exit('usage: run_judge_optimized.sh PROJECT STAGE BATCH')
    arm = Path(__file__).resolve().parents[1]
    scripts = arm / 'projects' / sys.argv[1] / '.agents/skills/cbma-review/scripts'
    sys.path.insert(0, str(scripts))
    import single_response
    try:
        return single_response.codex_entry(arm, sys.argv[1:], portable=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'{single_response.VERSION}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

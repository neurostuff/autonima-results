#!/usr/bin/env python3
"""Re-sign cached results written before thresholds became post-hoc.

Thresholds used to be part of the screening and annotation stage hashes. They no longer are,
because a calibrated backend stores probabilities and the threshold is applied to them on
read. Caches written under the old scheme therefore carry a hash the new code cannot match,
and would be recomputed once for nothing.

Nothing about the stored ANSWERS changes -- probabilities are threshold-independent, and the
loader re-gates them at whatever threshold is configured. Only the signature is rewritten, and
only when the existing one is verified to be the old scheme for this exact config. Anything
else is left alone.

    python scripts/migrate_posthoc_cache_signatures.py <run-dir> --config <yaml>
    python scripts/migrate_posthoc_cache_signatures.py <run-dir> --config <yaml> --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/zorro/repos/autonima")
from autonima.backends.jev import POST_HOC_CONFIG_KEYS  # noqa: E402
from autonima.config import ConfigManager  # noqa: E402
from autonima.execution import stable_hash  # noqa: E402
from autonima.screening.screener import (  # noqa: E402
    ABSTRACT_SCREENING_PROMPT_VERSION,
    FULLTEXT_SCREENING_PROMPT_VERSION,
)


def screening_hashes(stage_cfg: dict, prompt_version: str) -> tuple[str, str]:
    """(old scheme, new scheme) stage hashes for one screening stage."""
    old = stable_hash({**stage_cfg, "prompt_version": prompt_version})
    new = stable_hash({**{k: v for k, v in stage_cfg.items()
                          if k not in POST_HOC_CONFIG_KEYS},
                       "prompt_version": prompt_version})
    return old, new


def migrate_file(path: Path, old: str, new: str, apply: bool) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    blob = json.load(open(path))
    rows = blob.get("screening_results", blob if isinstance(blob, list) else [])
    matched = skipped = 0
    for r in rows:
        sig = r.get("cache_signature") or {}
        if sig.get("stage_hash") == old:
            sig["stage_hash"] = new
            matched += 1
        elif sig.get("stage_hash") != new:
            skipped += 1
    if apply and matched:
        json.dump(blob, open(path, "w"), indent=2, default=str)
    return matched, skipped


def migrate_manifest(run_dir: Path, config_path: Path, apply: bool) -> tuple[int, int]:
    """Re-sign execution_manifest.json's per-stage hashes.

    The execution layer compares freshly computed stage hashes against the ones the previous
    run stored here, and invalidates the artifact FILE when they differ -- before the
    per-study cache is consulted. A manifest written under the old scheme therefore forces a
    full recompute no matter how well the per-study signatures are migrated.
    """
    from autonima.execution import (DEPLOYMENT_KEYS, POST_HOC_KEYS,
                                    pipeline_config_to_dict, stage_hashes)

    path = run_dir / "outputs" / "execution_manifest.json"
    if not path.exists():
        return 0, 0
    manifest = json.load(open(path))
    stored = manifest.get("stage_hashes") or {}

    cfg = pipeline_config_to_dict(ConfigManager().load_from_file(str(config_path)))
    new = stage_hashes(cfg)

    # recompute the OLD scheme: thresholds still in the screening payloads
    import autonima.execution as ex
    old_payloads = ex.stage_signature_payloads(cfg)
    screening = cfg.get("screening") or {}
    for stage, pv in (("abstract", ex.ABSTRACT_SCREENING_PROMPT_VERSION),
                      ("fulltext", ex.FULLTEXT_SCREENING_PROMPT_VERSION)):
        block = {k: v for k, v in (screening.get(stage) or {}).items()
                 if k not in DEPLOYMENT_KEYS}
        old_payloads[stage] = {**block, "prompt_version": pv}
    old = {k: stable_hash(v) for k, v in old_payloads.items()}

    matched = skipped = 0
    for stage, h in list(stored.items()):
        if stage in old and h == old[stage] and h != new[stage]:
            stored[stage] = new[stage]
            matched += 1
        elif stage in new and h != new[stage]:
            skipped += 1
    if apply and matched:
        manifest["stage_hashes"] = stored
        json.dump(manifest, open(path, "w"), indent=2, default=str)
    return matched, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--apply", action="store_true", help="write; otherwise report only")
    args = ap.parse_args(argv)

    cfg = ConfigManager().load_from_file(str(args.config))
    total_m = total_s = 0
    for stage, fname, pv in (
        ("abstract", "abstract_screening_results.json", ABSTRACT_SCREENING_PROMPT_VERSION),
        ("fulltext", "fulltext_screening_results.json", FULLTEXT_SCREENING_PROMPT_VERSION),
    ):
        old, new = screening_hashes(getattr(cfg.screening, stage), pv)
        m, s = migrate_file(args.run_dir / "outputs" / fname, old, new, args.apply)
        total_m += m; total_s += s
        print(f"  {stage:<10} re-signed {m:>5}   left alone {s:>5}")
    m, s = migrate_manifest(args.run_dir, args.config, args.apply)
    print(f"  {'manifest':<10} re-signed {m:>5}   left alone {s:>5}")
    total_m += m; total_s += s
    print(f"  {'TOTAL':<10} re-signed {total_m:>5}   left alone {total_s:>5}"
          + ("" if args.apply else "   (dry run -- pass --apply)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

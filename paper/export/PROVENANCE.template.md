# Provenance

This repository is a frozen subset of
[neurostuff/autonima-results](https://github.com/neurostuff/autonima-results) at commit
[`{short}`](https://github.com/neurostuff/autonima-results/commit/{commit}), exported on {date}
by `tools/export_paper_repo.py` in that repository.

- **{n_files} files**, covering {n_runs} run folders under `projects/`, the scripts that rebuild the
  manuscript's figures, tables and numbers, and the evaluation tables they produce.
- **{n_maps} meta-analytic map directories**, distributed as `maps.tar.gz` on the Zenodo record
  (checksums in `reports/maps_manifest.json`).

**How the subset was chosen.** `paper/reproduce.sh`, plus the Fig. S6 and Table S1 scripts, was run
under `strace` in a clean checkout of autonima-results, and every tracked file it opened was kept.
A short whitelist adds each run's configuration and screening outputs (promised in the
manuscript's Data availability statement) and the licence and environment files.

**What stays in autonima-results.** Its full git history; archived and superseded configurations;
exploratory arms that are not part of this paper (NeuroStore retrieval, document arms, pilot
studies); per-run HTML review pages; and corpus-construction tools. None of it is read by
anything here.

**Not included anywhere.** Article full texts (third-party, copyrighted; PubMed Central articles
can be re-fetched with pubget), AutoNIMA itself (pinned in `pyproject.toml` as
[v0.1.1](https://github.com/neurostuff/autonima/releases/tag/v0.1.1)) and the expert-reference
maps, which belong to NeuroMetaBench v1.0
([10.5281/zenodo.24409323](https://doi.org/10.5281/zenodo.24409323)).

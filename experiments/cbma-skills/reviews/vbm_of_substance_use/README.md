# vbm_of_substance_use (cbma-skills arm): registered protocol

`review.yaml` is the protocol as set up on 2026-09-30, before any screening. It
translates autonima's v2 (the registry's "best"). `scripts/check_cbma_translation.py`
shows it carries v2's criteria verbatim, with three guidance lines moved into
`instructions`. The review runs outside this repository, in
`~/repos/cbma-workspace-vbm_of_substance_use/vbm_of_substance_use/`. The gold is in
`projects/vbm_of_substance_use/cbma_skills_gold/`. Score it with the benchmark's column
exclusions (`scripts/benchmark_exclusions.py`: cannabis, opioids and stimulants have no
published result).

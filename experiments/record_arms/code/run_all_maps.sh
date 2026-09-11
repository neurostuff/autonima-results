set -u
PY=/home/james/projects/fdcr/.venv/bin/python
cd /data/james/pondie-vs-fulltext/repos/autonima-results
run () { echo "### $1"; $PY /tmp/ale/run_ale.py --run-dir "projects/$1" --n-cores 6 2>&1 | grep -vE "FutureWarning|res = MKDA"; }
for r in vbm_of_ptsd/v1-A1-mini vbm_of_ptsd/v1-record-with-evidence vbm_of_ptsd/v1-record-no-evidence \
         cue_reactivity/v5-gpt-A1-mini cue_reactivity/v5-gpt-record-with-evidence cue_reactivity/v5-gpt-record-no-evidence \
         dementia/v3 dementia/v3-record-with-evidence dementia/v3-record-no-evidence \
         vbm_of_substance_use/v2 vbm_of_substance_use/v2-record-with-evidence vbm_of_substance_use/v2-record-no-evidence; do
  run "$r"
done

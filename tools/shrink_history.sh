#!/usr/bin/env bash
#
# One-time history rewrite, to be run ONCE before the DataLad conversion.
#
# WHY THIS EXISTS AS A SCRIPT
#
# Several directories were removed from tracking during the pre-submission cleanup
# and copied to attic/. Untracking stops history growing; it does not shrink what
# is already there. This removes those paths from every commit.
#
# It must run BEFORE `datalad create` / git-annex. Rewriting after annexing would
# rewrite the annex pointer commits and can orphan content in the S3 remote.
#
# WHAT IT REMOVES, and what that costs in history blob bytes:
#
#   projects/*/archive/               408 MB   superseded runs
#   8 exploratory run dirs            389 MB   *-recent and the social variants
#   reports/{neuroquery,neurovlm}*    229 MB   text-to-map maps, arm withdrawn
#   5 retired report trees            120 MB   poster plots, deck figures, etc.
#   downloaded_files/                  <1 MB   pyautogui/selenium locks, redirect stubs
#                                   -------
#                                    1144 MB   13% of 8837 MB of blob bytes
#
# WHAT IT DELIBERATELY KEEPS
#
#   search_results.json  (1139 MB, 12.9% -- the single largest category)
#
# These are not a cache. Each records what PubMed returned on the date the search
# ran, and re-running the query later returns a different set as the literature
# grows. They are provenance for the corpus and must survive.
#
# CONSEQUENCES -- read before running
#
#   * Every commit SHA changes. Existing clones and forks are invalidated; anyone
#     working from one must re-clone, not pull.
#   * filter-repo strips the remote. It is re-added at the end, but the next push
#     must be --force.
#   * The .git reduction will be smaller than 1144 MB: that figure is uncompressed
#     blob bytes, and the pack is delta-compressed. Measured before/after below.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BACKUP="${BACKUP:-$REPO/../autonima-results-prerewrite.git}"
REMOTE_NAME="anthonyangg"
REMOTE_URL="$(git remote get-url "$REMOTE_NAME" 2>/dev/null || true)"

die() { echo "ERROR: $*" >&2; exit 1; }

# ---- preconditions -------------------------------------------------------
[ -z "$(git status --porcelain --untracked-files=no)" ] \
  || die "working tree has uncommitted tracked changes; commit or discard first"

if [ -n "$(git worktree list | tail -n +2)" ]; then
  echo "Worktrees present. filter-repo cannot rewrite with these attached:"
  git worktree list | tail -n +2
  die "remove them first: git worktree remove <path>"
fi

command -v git-filter-repo >/dev/null || die "git-filter-repo not installed"

# ---- backup --------------------------------------------------------------
# A mirror clone is the only real undo. filter-repo's own .git/filter-repo/
# backup is easy to lose to a later gc.
[ -e "$BACKUP" ] && die "backup already exists at $BACKUP -- move it aside"
echo "==> mirroring to $BACKUP"
git clone --mirror "$REPO" "$BACKUP" >/dev/null
echo "    restore with: rm -rf .git && git clone $BACKUP --no-local ."

before=$(du -sb .git | cut -f1)

# ---- rewrite -------------------------------------------------------------
echo "==> rewriting history"
git filter-repo --force --invert-paths \
  --path downloaded_files/ \
  --path-glob 'projects/*/archive/*' \
  --path projects/cue_reactivity/v5-recent \
  --path projects/cue_reactivity/v5-recent.yaml \
  --path projects/vbm_of_ptsd/v1-recent \
  --path projects/vbm_of_ptsd/v1-recent.yaml \
  --path projects/vbm_of_substance_use/v2-recent \
  --path projects/vbm_of_substance_use/v2-recent.yaml \
  --path projects/social/v1-annotation-only-gpt5 \
  --path projects/social/v1-annotation-only-gpt5.yaml \
  --path projects/social/v1-annotation-only-lc \
  --path projects/social/v1-annotation-only-lc.yaml \
  --path projects/social/v2-all_pmids \
  --path projects/social/v2-all_pmids.yaml \
  --path projects/social/v3-all_pmids \
  --path projects/social/v3-all_pmids.yaml \
  --path projects/social/v3-all_pmids-multi_analysis-ft-gpt52 \
  --path projects/social/v3-all_pmids-multi_analysis-ft-gpt52.yaml \
  --path reports/neuroquery_maps \
  --path reports/neurovlm_maps \
  --path reports/neurovlm_variant_maps \
  --path reports/cross_project_publication_plots \
  --path reports/deck_figures \
  --path reports/cross_project_manual_vs_auto_meta_fair \
  --path reports/coordinate_miss_triage \
  --path reports/gold_gap

# ---- repack and report ---------------------------------------------------
echo "==> repacking"
git reflog expire --expire=now --all
git gc --prune=now --aggressive --quiet
after=$(du -sb .git | cut -f1)
printf '    .git %.0f MB -> %.0f MB  (%.0f MB reclaimed)\n' \
  "$((before/1048576))" "$((after/1048576))" "$(((before-after)/1048576))"

# ---- restore the remote --------------------------------------------------
if [ -n "$REMOTE_URL" ]; then
  git remote add "$REMOTE_NAME" "$REMOTE_URL" 2>/dev/null || true
  echo "==> remote '$REMOTE_NAME' restored. The next push MUST be:"
  echo "    git push --force --all $REMOTE_NAME && git push --force --tags $REMOTE_NAME"
fi

# ---- verify --------------------------------------------------------------
echo "==> verifying the kept paths survived"
n=$(git log --all --pretty=format: --name-only | grep -c 'search_results\.json$' || true)
[ "$n" -gt 0 ] || die "search_results.json vanished -- restore from $BACKUP"
echo "    search_results.json still present in history ($n path references)"
echo "==> done. Check the working tree, then remove $BACKUP once satisfied."

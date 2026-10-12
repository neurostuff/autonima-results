#!/usr/bin/env bash
# Restore executable permissions after a ZIP extraction; dependencies are prebuilt.
set -euo pipefail
shopt -s nullglob
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "This package targets native Apple Silicon macOS." >&2; exit 2
fi
chmod +x "$root"/*.sh "$root/bin/python" "$root/bin/python3" "$root/bin/codex"
chmod +x "$root/runtime/python/bin/"* "$root/runtime/codex/vendor/aarch64-apple-darwin/bin/"*
if [[ -d "$root/luna" ]]; then
  chmod +x "$root/luna/"*.sh "$root/luna/bin/"*
  for ws in "$root/luna/projects/"*; do
    chmod +x "$ws/.venv/bin/python" "$ws/.venv/bin/python3"
  done
fi
mkdir -p "$root/.run-state/matplotlib" "$root/runtime/data/nilearn"
echo "Ready. Run ./copy_articles.sh /path/to/e6-codex-frontier-macos, then bash login.sh if needed."
echo "Launch all five reviews with ./run_all_luna.sh, or one with ./luna/run_review.sh PROJECT."

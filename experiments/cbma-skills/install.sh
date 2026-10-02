#!/usr/bin/env bash
# Link the cbma-skills skill folders into a workspace for Claude Code and Codex.
#   ./install.sh [WORKSPACE]      (default: current directory)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ws="$(cd "${1:-.}" && pwd)"
for target in "$ws/.claude/skills" "$ws/.agents/skills"; do
  mkdir -p "$target"
  for skill in "$here"/skills/*/; do
    name="$(basename "$skill")"
    ln -sfn "${skill%/}" "$target/$name"
  done
  echo "linked $(ls "$here/skills" | wc -l) skills into $target"
done
# Claude Code agent definitions: the stage runner and the per-stage judges, with their
# reasoning effort. Copied, not linked: a session reads them at start, and a copy keeps a
# running review on the definitions it started with. Re-run install.sh to update them.
mkdir -p "$ws/.claude/agents"
cp "$here"/agents/*.md "$ws/.claude/agents/"
echo "copied $(ls "$here"/agents/*.md | wc -l) agent definitions into $ws/.claude/agents"

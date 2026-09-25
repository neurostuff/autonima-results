#!/usr/bin/env bash
# Build the manuscript PDF. Runs pdflatex/BibTeX to a fixed point.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
echo
echo "pages: $(pdfinfo manuscript.pdf 2>/dev/null | awk '/^Pages/{print $2}')"

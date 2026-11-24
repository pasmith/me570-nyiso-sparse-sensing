#!/usr/bin/env bash
# Build the final report: Markdown + pandoc-header.tex -> PDF (xelatex), IEEE numeric citations.
set -euo pipefail
cd "$(dirname "$0")"
CSL=/usr/local/texlive/2026/texmf-dist/tex/latex/citation-style-language/styles/ieee.csl
pandoc final-report.md \
  -H pandoc-header.tex \
  --lua-filter=chit.lua \
  --citeproc --csl "$CSL" --bibliography refs.bib \
  --pdf-engine=xelatex -V geometry:margin=1in \
  -o final-report.pdf
echo "wrote final-report.pdf"

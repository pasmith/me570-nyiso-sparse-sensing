#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CSL=/usr/local/texlive/2026/texmf-dist/tex/latex/citation-style-language/styles/ieee.csl
pandoc final-report.md \
  -H pandoc-header.tex \
  --lua-filter=chit.lua \
  --citeproc --csl "$CSL" --bibliography refs.bib \
  --pdf-engine=xelatex \
  -V geometry:margin=0.7in \
  -V classoption=twocolumn \
  -V fontsize=10pt \
  -o final-report.pdf
echo "wrote final-report.pdf"

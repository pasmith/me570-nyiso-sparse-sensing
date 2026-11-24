# Report & Literature Review Directory (`report/`)

**Project:** Sparse Sensor Placement for Classification and Regression in NYISO DART Spread Virtual Bidding  
**Authors:** Patrick Smith (University of Washington) & Tyler Darwin (University of Washington)

---

## Directory Contents

| File | Description |
| :--- | :--- |
| **`project-proposal.md` / `project-proposal.qmd`** | Initial project scoping proposals and conceptual sensor placement diagram (`images/selected-sensors.png`). |
| **`literature_review_outline.md`** | Structured 6-paragraph literature review outline, 34-paper annotated reference matrix (`[1]–[34]`), and key methodological distinctions across Notebooks `01`–`04`. |
| **`literature_review_prose.md`** | Full academic prose literature review contrasting classification-driven sparse sensing (`SSPOC`) against reconstruction/regression sparse sensing (`QR-Pivoting`) and forced dynamical control (`DMDc`). |
| **`progress-update.md`** | Joint project progress report outlining Patrick's 3-class regime classification branch (`SSPOC`), Tyler's continuous sparse sensor regression branch (`QR-Pivoting`), dataset scope, and baseline reproductions. |
| **`final-report.md`** | Main IEEE-style 2-column research paper source in Pandoc Markdown. Contains the YAML metadata/abstract, full-width `TikZ` causal filtration & cross-paradigm architecture diagram (`Figure 1`), mathematical formulations (`DMDc`, `SSPOC`, `QR-Pivoting`), out-of-sample results (`2025–2026`), full-width comparison table (`Table 1`), `chit.lua` callout placeholders for figures and Tyler's continuous regression results, and `[@citekey]` bibliography tags. |
| **`build.sh`** | Executable build script that compiles `final-report.md` into `final-report.pdf` via `pandoc`, `xelatex`, `--citeproc`, `ieee.csl`, `chit.lua`, and `refs.bib`. |
| **`pandoc-header.tex`** | LaTeX preamble configuring 2-column spacing, `booktabs` tables, `tcolorbox` placeholder environments (`placeholderfig`, `placeholdertable`), `TikZ` diagram libraries, and Unicode math character mappings. |
| **`chit.lua`** | Pandoc Lua filter that converts fenced Markdown divs (`::: {.figbox}` and `::: {.tablebox}`) into styled LaTeX `tcolorbox` callout boxes. |
| **`refs.bib`** | BibTeX bibliography database (`30` references) covering wholesale electricity markets, continuous price forecasting, modal decomposition (`DMD`, `BOP-DMD`, `DMDc`), sparse sensor placement (`SSPOC`, `QR-Pivoting`, `Klishin et al., 2026`), and backtest hygiene. |

---

## Building the PDF (`build.sh`)

### Prerequisites
* **Pandoc** (with Lua filter and `--citeproc` support)
* **TeX Live / XeLaTeX** (`2026` distribution with `ieee.csl`, `booktabs`, `tcolorbox`, `tikz`, and `newunicodechar`)

### Compile Command
From inside the `report/` directory, run:

```bash
chmod +x build.sh
./build.sh
```

This runs:
```bash
pandoc final-report.md \
  -H pandoc-header.tex \
  --lua-filter=chit.lua \
  --citeproc --csl /usr/local/texlive/2026/texmf-dist/tex/latex/citation-style-language/styles/ieee.csl \
  --bibliography refs.bib \
  --pdf-engine=xelatex \
  -V geometry:margin=0.7in \
  -V classoption=twocolumn \
  -V fontsize=10pt \
  -o final-report.pdf
```


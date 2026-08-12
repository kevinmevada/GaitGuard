# GaitGuard — Overleaf / MDPI *Sensors* manuscript

## Quick start (compiles today)

1. Go to [overleaf.com](https://www.overleaf.com) → **New Project** → **Upload Project**.
2. Zip this folder locally:
   ```bash
   cd docs/paper/overleaf
   # Windows PowerShell:
   Compress-Archive -Path * -DestinationPath gaitguard-overleaf.zip
   ```
3. Upload `gaitguard-overleaf.zip`. Set the main document to **`main.tex`**.
4. Compile with **pdfLaTeX** (default). You should get a complete draft PDF immediately.

`main.tex` is a **self-contained** article (standard `article` class). It does **not** use the proprietary MDPI `Definitions/mdpi` class, so it compiles without downloading the MDPI ZIP.

## Submit to *Sensors* (required template)

MDPI requires their official LaTeX template for submission ([Sensors instructions](https://www.mdpi.com/journal/Sensors/instructions)).

1. Download the latest **MDPI LaTeX template** from [https://www.mdpi.com/authors/latex](https://www.mdpi.com/authors/latex) (prefer the MDPI site over the Overleaf gallery; gallery copies can lag).
2. New Overleaf project from that ZIP (keep the `Definitions/` folder).
3. In the template `template.tex` (or equivalent), set:
   ```latex
   \documentclass[sensors,article,submit,pdftex,moreauthors]{Definitions/mdpi}
   ```
4. Replace the template body with content from **`mdpi_body.tex`** in this folder (section text + tables + bibliography stubs). Keep MDPI front-matter commands (`\Title`, `\Author`, `\address`, `\abstract`, `\keyword`, etc.).
5. Fill author affiliations, ORCID, funding, and SuSy metadata before submit.
6. Optional: Overleaf → **Submit** → **Submit to an MDPI journal**.

## Files

| File | Role |
|------|------|
| `main.tex` | Standalone draft — upload to Overleaf now |
| `mdpi_body.tex` | Section body to paste into official MDPI template |
| `references.bib` | BibTeX starters (complete before submission) |
| `README.md` | This file |

## Source of truth

Prose is reconciled from `docs/paper/*.md` (2026-08). Locked numeric headlines:

- Voisard BiLSTM-AE 2-method ensemble ROC-AUC **0.7545**, PR-AUC **0.8669**
- Rejected 3-method (AE included) **0.6238**
- DAPHNET AE recon **0.7046**; ensemble **0.5314**; IF **0.5884**
- Tabular supervised numbers = **provisional** (DAPHNET bleed)

After changing paper Markdown, re-sync this Overleaf folder before submission.

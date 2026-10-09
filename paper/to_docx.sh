#!/bin/bash
# Produce the MS Word file the award requires, from main.tex, with pandoc.
# Pandoc converts text, headings, tables, equations, citations and \includegraphics figures.
# It cannot convert TikZ drawings: export each TikZ figure to PDF/PNG first and include it
# with \includegraphics, or paste the image into Word after conversion.
# Always open main.docx and check the title page, the Arabic text direction and the tables.
set -u
cd "$(dirname "$0")"
CSL=ieee.csl
if [ ! -f "$CSL" ]; then
  curl -sL -o "$CSL" https://raw.githubusercontent.com/citation-style-language/styles/master/ieee.csl || true
fi
# Pandoc cannot read tabularx or custom column types: convert a simplified copy.
python3 - <<'PY'
import re
s = open("main.tex", encoding="utf-8").read()
s = s.replace(r"\begin{tabularx}{\textwidth}{@{}L{2.7cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}", r"\begin{tabular}{lll}")
s = re.sub(r"\\begin\{tabularx\}\{\\textwidth\}\{[^\n]*\}", r"\\begin{tabular}{ll}", s)
s = s.replace(r"\end{tabularx}", r"\end{tabular}")
s = re.sub(r"\\multicolumn\{2\}\{@\{\}l\}\{(.*?)\}\\\\", r"\1 & \\\\", s)
s = re.sub(r"\\orcidlink\{[^}]*\}", "", s)
s = s.replace(r"\showguidancetrue", r"\showguidancefalse")
open("_docx_src.tex", "w", encoding="utf-8").write(s)
PY
pandoc _docx_src.tex -o main.docx \
  --from=latex --to=docx \
  --citeproc --bibliography=references.bib --csl="$CSL" \
  --resource-path=.:figures \
  --metadata title="GDRFA Dubai Scientific Research Award submission" \
  && echo "WROTE main.docx (check it in Word before uploading)"
rm -f _docx_src.tex

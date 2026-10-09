#!/bin/bash
# Full build; fails on undefined references or citations. Mirrors 04-Writing/Thesis/build.sh.
set -u
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null || { echo "pdflatex pass 1 failed"; grep -A3 "^!" main.log | head -40; exit 1; }
bibtex main > /dev/null || { echo "bibtex failed"; grep -i "warning\|error" main.blg | head -20; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null || { echo "pdflatex pass 2 failed"; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null || { echo "pdflatex pass 3 failed"; exit 1; }
undef_ref=$(grep -c "Reference .* undefined" main.log)
undef_cit=$(grep -c "Citation .* undefined" main.log)
echo "undefined references: $undef_ref, undefined citations: $undef_cit"
if [ "$undef_ref" -ne 0 ] || [ "$undef_cit" -ne 0 ]; then
  grep -E "Reference .* undefined|Citation .* undefined" main.log | sort -u
  exit 1
fi
echo "BUILD OK: main.pdf"

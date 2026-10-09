#!/bin/bash
# Build the GDRFA submission PDF: LuaLaTeX (babel Arabic) + BibTeX. Fails on undefined references or citations.
set -u
cd "$(dirname "$0")"
lualatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null || { echo "lualatex pass 1 failed"; grep -A3 "^!" main.log | head -40; exit 1; }
if grep -q '\\citation' main.aux; then
  bibtex main > /dev/null || { echo "bibtex failed"; grep -i "error" main.blg | head -20; exit 1; }
else
  echo "no \\cite yet: bibliography skipped"
fi
lualatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null || { echo "lualatex pass 2 failed"; exit 1; }
lualatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null || { echo "lualatex pass 3 failed"; exit 1; }
undef_ref=$(grep -c "Reference .* undefined" main.log)
undef_cit=$(grep -c "Citation .* undefined" main.log)
echo "undefined references: $undef_ref, undefined citations: $undef_cit"
if [ "$undef_ref" -ne 0 ] || [ "$undef_cit" -ne 0 ]; then
  grep -E "Reference .* undefined|Citation .* undefined" main.log | sort -u
  exit 1
fi
echo "BUILD OK: main.pdf"

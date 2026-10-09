#!/bin/bash
# Word counts against the award rules:
#   total 4,000 to 8,000 words including body, tables, figures, appendices and references
#   Arabic abstract <= 250 words, English abstract <= 250 words
# Counts the built PDF (so tables, captions and the bibliography are included) and excludes the
# title page. Guidance placeholders are counted too: hide them (\showguidancefalse) for a final count.
set -u
cd "$(dirname "$0")"
[ -f main.pdf ] || { echo "main.pdf missing; run build.sh first"; exit 1; }
python3 - <<'EOF'
import re, sys
try:
    import fitz
except ImportError:
    sys.exit("PyMuPDF (fitz) is required: pip install pymupdf")
doc = fitz.open("main.pdf")
pages = [p.get_text() for p in doc]
words = lambda t: len(re.findall(r"[^\W\d_][\w'\-]*|\d[\d.,%]*", t))
body = "\n".join(pages[1:])                      # exclude the title page
# strip running header and page numbers
body = re.sub(r"GDRFA Dubai Scientific Research Award, second cycle|Digital Innovation Award", " ", body)
total = words(body)
# abstracts: text between headings
ar = re.search(r"^(.*?)\nAbstract\n", body, re.S)   # everything before the English abstract heading
en = re.search(r"\nAbstract\n(.*?)\n1\.? *\n?Introduction", body, re.S)
arabic = lambda t: len(re.findall(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]+", t))
ar_n = arabic(ar.group(1)) if ar else -1
en_n = words(en.group(1)) if en else -1
ok = lambda c: "OK" if c else "OUT OF RANGE"
print(f"Total (excluding title page): {total:6d}   {ok(4000 <= total <= 8000)}  (rule: 4,000 to 8,000)")
print(f"Arabic abstract:              {ar_n:6d}   {ok(0 <= ar_n <= 250)}  (rule: <= 250)")
print(f"English abstract:             {en_n:6d}   {ok(0 <= en_n <= 250)}  (rule: <= 250)")
EOF

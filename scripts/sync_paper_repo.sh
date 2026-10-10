#!/bin/bash
# Push paper/ (the master copy of the manuscript) to the paper-only repository that Overleaf syncs with.
# Usage: bash scripts/sync_paper_repo.sh "commit message"
# Afterwards, in Overleaf: Menu > GitHub > Pull changes from GitHub.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/../paper" && pwd)"
DST="${PAPER_REPO:-$HOME/Documents/GitHub/GDRFA-Award-Paper}"
MSG="${1:-Sync from refer-or-clear/paper}"
[ -d "$DST/.git" ] || { echo "paper repo not found at $DST (clone 7abushahla/GDRFA-Award-Paper)"; exit 1; }
rsync -a --delete --exclude '.git' --exclude '.gitignore' --exclude '*.aux' --exclude '*.log' --exclude '*.out' \
  --exclude '*.bbl' --exclude '*.blg' --exclude '*.fls' --exclude '*.fdb_latexmk' --exclude '*.synctex.gz' \
  --exclude 'main.pdf' --exclude 'main.docx' --exclude '_docx_src.tex' "$SRC/" "$DST/"
cd "$DST" && git add -A && { git diff --cached --quiet && echo "nothing to sync" || { git commit -qm "$MSG

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git push -q && echo "pushed: $(git log --oneline -1)"; }; }

#!/bin/bash
# Download the public datasets into data/raw/. Sizes and sources: data/README.md.
# Run on the cluster login node or a machine with about 15 GB free. Re-running skips existing files.
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW=data/raw

echo "TODO: verify each URL on day 1 (public pages checked 8 Oct 2026)."

# MIDV-2020 (anonymous FTP, no form)
mkdir -p $RAW/midv2020 && cd $RAW/midv2020
for f in templates.tar photo.tar scan_upright.tar md5.txt; do
  [ -f "$f" ] || wget -q --show-progress "ftp://smartengines.com/midv-2020/dataset/$f"
done
cd - >/dev/null

# SIDTD (direct HTTP)
mkdir -p $RAW/sidtd && cd $RAW/sidtd
for f in templates.zip split_shot.zip split_kfold.zip split_normal.zip; do
  [ -f "$f" ] || wget -q --show-progress "http://datasets.cvc.uab.es/SIDTD/$f"
done
cd - >/dev/null

# FantasyID (Zenodo)
mkdir -p $RAW/fantasyid && cd $RAW/fantasyid
[ -f FantasyID.tgz ] || wget -q --show-progress -O FantasyID.tgz "https://zenodo.org/records/17063366/files/FantasyID.tgz?download=1"
cd - >/dev/null

echo "Done. Next: python scripts/prepare_data.py"

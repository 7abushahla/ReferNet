#!/bin/bash
# Download the public datasets into data/raw/. Sizes and sources: data/README.md.
# Run on the cluster login node or a machine with about 15 GB free. Re-running skips existing files.
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW=data/raw

# All URLs checked on 10 Oct 2026.

# SIDTD (direct HTTP; https://github.com/Oriolrt/SIDTD_Dataset, repository licence CC BY 4.0).
# templates.zip holds the 1,000 MIDV-2020 genuine templates (10 designs x 100) and the
# 1,222 SIDTD forgeries (Crop_and_Replace, Inpaint_and_Rewrite) with annotations.
# Split files are the names used by the official loader (Datasets.py). The official
# downloader is not used: it joins URLs with os.path.join and mirrors the whole
# folder, including 54 GB of video.
mkdir -p $RAW/sidtd && cd $RAW/sidtd
for f in templates.zip few_shot_split_templates.zip kfold_split_templates.zip; do
  [ -f "$f" ] || wget -q --show-progress "http://datasets.cvc.uab.es/SIDTD/$f"
done
cd - >/dev/null

# FantasyID (Zenodo)
mkdir -p $RAW/fantasyid && cd $RAW/fantasyid
[ -f FantasyID.tgz ] || wget -q --show-progress -O FantasyID.tgz "https://zenodo.org/records/17063366/files/FantasyID.tgz?download=1"
cd - >/dev/null

# MIDV-2020 photos and scans (optional: real captures of the same 10 designs).
# Enable with WITH_MIDV=1. Not needed for Protocols A and B.
if [ "${WITH_MIDV:-0}" = "1" ]; then
  mkdir -p $RAW/midv2020 && cd $RAW/midv2020
  for f in photo.tar scan_upright.tar; do
    [ -f "$f" ] || wget -q --show-progress "ftp://smartengines.com/midv-2020/dataset/$f"
  done
  cd - >/dev/null
fi

echo "Done. Next: python scripts/prepare_data.py"

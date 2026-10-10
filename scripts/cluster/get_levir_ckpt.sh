#!/bin/bash
# Download the LEVIR-CD ChangerEx r18 checkpoint (Open-CD model zoo, Apache-2.0) into
# checkpoints/pretrained/. Google Drive is the only host (evidence/code-audit-opencd-2026-10-10.md, A.3);
# the usercontent URL with confirm=t skips the virus-scan interstitial. Run on the login node (network).
# Re-running skips a file that already passes the size and md5 check.
#   bash scripts/cluster/get_levir_ckpt.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

NAME=ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth
URL='https://drive.usercontent.google.com/download?id=1SZ8DBgBryJ63X9AU0AQvaJ2ZH3oQHQHe&export=download&confirm=t'
EXPECT_BYTES=136880383
EXPECT_MD5=7cd73048926efa8f0535317f5dde87e8
DEST=checkpoints/pretrained/$NAME

md5_of() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
size_of() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }
check() { [ -f "$1" ] && [ "$(size_of "$1")" = "$EXPECT_BYTES" ] && [ "$(md5_of "$1")" = "$EXPECT_MD5" ]; }

mkdir -p "$(dirname "$DEST")"
if check "$DEST"; then echo "[get_levir_ckpt] $DEST present and verified"; exit 0; fi
curl -L --fail --retry 3 -o "$DEST.part" "$URL"
if ! check "$DEST.part"; then
  echo "[get_levir_ckpt] FATAL: got $(size_of "$DEST.part") bytes, md5 $(md5_of "$DEST.part");" \
       "expected $EXPECT_BYTES / $EXPECT_MD5 (Google Drive may have returned an HTML page)" >&2
  head -c 300 "$DEST.part" >&2; echo >&2
  exit 1
fi
mv "$DEST.part" "$DEST"
echo "[get_levir_ckpt] $DEST ok ($EXPECT_BYTES bytes, md5 $EXPECT_MD5)"

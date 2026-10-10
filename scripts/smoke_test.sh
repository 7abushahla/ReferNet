#!/bin/bash
# Smoke test of the training stack (implementation plan Section 6). Run from the repo root:
#
#   bash scripts/smoke_test.sh                                     # CPU: 16 pairs, 1 epoch, batch 2, crop 256 x 400
#   DEVICE=cuda LIMIT=64 EPOCHS=2 BATCH=8 CROP="448 704" bash scripts/smoke_test.sh   # GPU variant (3080 / cluster)
#
# Asserts that train.py ran to completion, that results/runs/smoke/metrics.json exists, and prints it.
# PYTHON overrides the interpreter (default: python on PATH).
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f pyproject.toml && -d src/refer_or_clear ]] || { echo "[FATAL] not at the repo root: $PWD"; exit 1; }

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cpu}"
LIMIT="${LIMIT:-16}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-2}"
CROP="${CROP:-256 400}"
WORKERS="${WORKERS:-0}"
DATA_ROOT="${ROC_DATA:-data/processed}"
RUN_DIR="${RUN_DIR:-results/runs/smoke}"
AMP=""
[[ "$DEVICE" == "cuda" ]] && AMP="--amp"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

rm -f "$RUN_DIR/metrics.json"
echo "[smoke] device=$DEVICE limit=$LIMIT epochs=$EPOCHS batch=$BATCH crop=($CROP) run_dir=$RUN_DIR"
# shellcheck disable=SC2086
"$PYTHON" -m refer_or_clear.train --config configs/protocol_a.yaml --arch siamese_r18 --fusion difference \
    --reference median --init scratch --seed 42 --fold 0 --data-root "$DATA_ROOT" --run-dir "$RUN_DIR" \
    --epochs "$EPOCHS" --limit "$LIMIT" --batch-size "$BATCH" --crop $CROP --device "$DEVICE" \
    --workers "$WORKERS" $AMP

[[ -f "$RUN_DIR/metrics.json" ]] || { echo "[FAIL] $RUN_DIR/metrics.json was not written"; exit 1; }
echo "[smoke] log.csv:"; cat "$RUN_DIR/log.csv"
echo "[smoke] metrics.json:"; cat "$RUN_DIR/metrics.json"
"$PYTHON" - "$RUN_DIR/metrics.json" <<'PY'
import json, math, sys
m = json.load(open(sys.argv[1]))
assert m["test"], "no test block"
for variant, b in m["test"].items():
    assert b["n_genuine"] + b["n_altered"] > 0, variant
    for k, v in b.items():
        assert v is None or isinstance(v, int) or math.isfinite(v), (variant, k, v)
print(f"[smoke] ok: tag={m['tag']} variants={list(m['test'])} subsets={list(m['subsets'])}")
PY
echo "[smoke] passed"

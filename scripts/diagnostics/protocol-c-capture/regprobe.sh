#!/bin/bash
set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/b00090279/conda/envs/refer-or-clear
SRC=/shared/b00090279/refer-or-clear/results/source/study-v2-fdb7351
export PYTHONPATH="$SRC/src" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
R=/shared/b00090279/refer-or-clear
python /shared/b00090279/diag-align/regprobe.py --raw-root $R/data/raw \
  --c-support $R/results/runs/protocol-c-current/protocol-c-support-38182/preparation/enrollment/groupC \
  --c-registration $R/results/runs/protocol-c-current/fold0-v2noaux-fdb7351-scored25847b/evaluation/registration.json \
  --c-labels /shared/b00090279/diag-align/agg-final/c/off_test_predictions.csv \
  --sidtd-manifest $R/results/diagnostics/anchor-pilots-v1/anchor-enrollment-fold0-37973/enrollment/fold0/sidtd/alb_id/enrollment/manifest.json \
  --a-eval-registration $R/results/runs/protocol-a-current/fold0-v2noaux-fdb7351/evaluation/registration.json \
  --out /shared/b00090279/diag-align/regprobe "$@"

#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=roc-verify
#SBATCH --output=logs/slurm/%x-%j.out
#
# GPU-node check with the same SBATCH resources as a training job. Submit from the repo root:
#
#   sbatch scripts/cluster/verify_gpu.sh
#   cat logs/slurm/roc-verify-<jobid>.out

set -uo pipefail

ENV="${ENV:-refer-or-clear}"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$ENV"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f pyproject.toml && -d src/refer_or_clear ]] || { echo "[FATAL] submit from the repo root, not $PWD"; exit 1; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="/scratch/$USER/tmp"; mkdir -p "$TMPDIR"

echo "node      : $(hostname)   job ${SLURM_JOB_ID:-none}"
echo "gpu       : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo none)"
echo "cpus      : $(nproc) visible / ${SLURM_CPUS_PER_TASK:-?} allocated"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python scripts/check_imports.py || exit 1
python - <<'PY' || exit 1
import torch, torch.nn as nn
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(2, 3, 512, 800, device="cuda")
nn.Conv2d(3, 16, 3, padding=1).cuda()(x); torch.cuda.synchronize()
print(f"[ok] cuDNN conv at 512 x 800 | {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda}")
# TODO (day 3): one forward and backward pass of the reference-conditioned ChangerEx on one real batch,
# with peak GPU memory printed, so the batch size is confirmed before the sweep is queued.
PY
echo "[ok] verify_gpu passed"

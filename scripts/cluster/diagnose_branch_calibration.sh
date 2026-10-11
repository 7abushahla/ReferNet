#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=branch-bn-diag
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate refer-or-clear
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export TMPDIR="/scratch/$USER/branch-bn-$SLURM_JOB_ID/tmp"
mkdir -p "$TMPDIR"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python tests/test_branch_calibration.py
DATA_ROOT="/scratch/$USER/branch-bn-$SLURM_JOB_ID/data"
mkdir -p "$DATA_ROOT"
rsync -a --exclude='._*' data/processed/ "$DATA_ROOT/"
python scripts/diagnose_branch_calibration.py --data-root "$DATA_ROOT" \
    --out "results/diagnostics/branch-bn-$SLURM_JOB_ID"

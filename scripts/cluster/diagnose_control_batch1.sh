#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=control-bn-batch1
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate refer-or-clear
[[ "$(basename "$CONDA_PREFIX")" == refer-or-clear ]]
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
[[ -f pyproject.toml && -d src/refer_or_clear ]]
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export TMPDIR="/scratch/$USER/control-batch1-$SLURM_JOB_ID/tmp"
mkdir -p "$TMPDIR"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python tests/test_diagnose_control.py
DATA_ROOT="/scratch/$USER/control-batch1-$SLURM_JOB_ID/data"
mkdir -p "$DATA_ROOT"
rsync -a --exclude='._*' data/processed/ "$DATA_ROOT/"
CONTROL=A-cex-single_noex-median-levir-f0-s42-e100
REFERENCE=A-cex-difference-median-levir-f0-s42-e100
python scripts/diagnose_control.py \
    --checkpoints "checkpoints/$CONTROL/last.pt" "checkpoints/$REFERENCE/best.pt" \
    --data-root "$DATA_ROOT" --out "results/diagnostics/control-batch1-$SLURM_JOB_ID" \
    --batch-size 1 --roles val --modes eval backbone_batch all_batch

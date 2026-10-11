#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=5:00:00
#SBATCH --job-name=control-bn-pilot
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
export TMPDIR="/scratch/$USER/control-bn-pilot-$SLURM_JOB_ID/tmp"
mkdir -p "$TMPDIR"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python tests/test_bn_policy.py
python tests/test_train_config.py
python tests/test_control_noex.py
DATA_ROOT="/scratch/$USER/control-bn-pilot-$SLURM_JOB_ID/data"
mkdir -p "$DATA_ROOT"
rsync -a --exclude='._*' data/processed/ "$DATA_ROOT/"
python -m refer_or_clear.train \
    --config configs/protocol_a.yaml --arch changerex_r18 --fusion single_noex \
    --reference median --init levir --seed 42 --fold 0 --epochs 1 --limit 32 \
    --bn-policy batch_stats --data-root "$DATA_ROOT" --workers 4 --device cuda \
    --run-dir "results/runs/A-cex-single_noex-median-levir-f0-s42-bnbatch-smoke-$SLURM_JOB_ID"
python -m refer_or_clear.train \
    --config configs/protocol_a.yaml --arch changerex_r18 --fusion single_noex \
    --reference median --init levir --seed 42 --fold 0 --epochs 100 \
    --bn-policy batch_stats --data-root "$DATA_ROOT" --workers 4 --device cuda \
    --run-dir results/runs/A-cex-single_noex-median-levir-f0-s42-bnbatch-e100

#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --job-name=anchor-wiring
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate refer-or-clear
repository_root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
code_root="${1:?Specify the immutable source directory}"
cd "$repository_root"
export PYTHONPATH="$code_root/src"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
python "$code_root/scripts/audit_anchor_wiring.py" \
    --checkpoint "$repository_root/checkpoints/pretrained/ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth" \
    --out "results/diagnostics/anchor-wiring-$SLURM_JOB_ID/summary.json"

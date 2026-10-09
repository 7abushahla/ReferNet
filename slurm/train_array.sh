#!/bin/bash
# Slurm array over the configuration grid (backbone x fusion x seed) for one protocol.
# Pattern copied from the Thesis repository's QCFS scripts. Submit from the repository root:
#   sbatch --array=0-47 slurm/train_array.sh configs/protocol_a.yaml
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out

set -e
CONFIG=${1:?usage: sbatch --array=0-47 slurm/train_array.sh <config.yaml>}

ENV_PATH="/shared/b00090279/shared/b00090279/refer-or-clear-env"   # TODO: create this environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$ENV_PATH"
[[ "$CONDA_PREFIX" == "$ENV_PATH" ]] || { echo "conda env did not activate"; exit 1; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$SLURM_SUBMIT_DIR"

BACKBONES=(resnet18 resnet50 resnest50 mit_b1)
FUSIONS=(single stacked difference attention)
SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
B=${BACKBONES[$(( i / 12 ))]}
F=${FUSIONS[$(( (i / 3) % 4 ))]}
S=${SEEDS[$(( i % 3 ))]}
echo "task $i: backbone=$B fusion=$F seed=$S config=$CONFIG"

python -m refer_or_clear.train --config "$CONFIG" --backbone "$B" --fusion "$F" --seed "$S"

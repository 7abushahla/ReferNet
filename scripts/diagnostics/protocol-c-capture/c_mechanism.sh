#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH --output=/shared/b00090279/diag-align/logs/%x-%j.out
set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate /shared/b00090279/conda/envs/refer-or-clear
SRC=/shared/b00090279/refer-or-clear/results/source/study-v2-fdb7351
export PYTHONPATH="$SRC/src:$SRC/scripts:/shared/b00090279/diag-align" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
R=/shared/b00090279/refer-or-clear
IN=$R/results/diagnostics/anchor-pilots-v1/anchor-enrollment-fold0-37973/inputs
A=$R/results/runs/protocol-a-current/fold0-v2noaux-fdb7351
python /shared/b00090279/diag-align/c_mechanism.py --records $IN/records.csv --raw-root $R/data/raw --a-eval-registration $A/evaluation/registration.json \
  --c-support $R/results/runs/protocol-c-current/protocol-c-support-38182/preparation/enrollment/groupC \
  --weights $R/results/source/study-shared-assets/weights.pth --weights-sha256 da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618 \
  --ckpt-on $A/training/on/best.pt --ckpt-off $A/training/off/best.pt --out /shared/b00090279/diag-align/c-mechanism --workers 4 "$@"

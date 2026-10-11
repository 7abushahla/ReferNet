#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=sidtd-ckpt-diag
#SBATCH --output=logs/slurm/%x-%j.out
#
# Diagnostic (spec Section 6, item 2): the fair-control unit test, then the paper's released template
# checkpoints scored on the paper's folds with the paper's preprocessing. Header copied from the
# make_sweep.py template. Submit from the repo root after scripts/cluster/get_sidtd_ckpts.sh has run:
#   sbatch scripts/cluster/score_sidtd_ckpts.sh
set -u
ENV="refer-or-clear"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate "$ENV"
[[ "$CONDA_PREFIX" == "$ENV" || "$(basename "$CONDA_PREFIX")" == "$ENV" ]] || { echo "[NO] conda env not active"; exit 1; }
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR"; fi
[[ -f pyproject.toml && -d src/refer_or_clear ]] || { echo "[FATAL] not at the repo root: $PWD"; exit 1; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/$USER/tmp"; mkdir -p "$TMPDIR"
python -c "import torch; assert torch.cuda.is_available(); print('[ok]', torch.cuda.get_device_name(0))" || exit 1

echo "== fair-control unit test"
python tests/test_control_noex.py || { echo "[FAIL] test_control_noex"; exit 1; }

echo "== released checkpoints"
MODELS="${MODELS:-resnet50,efficientnet-b3}"
python scripts/score_sidtd_checkpoints.py --models "$MODELS" --raw data/raw/sidtd \
    --ckpt-dir data/raw/sidtd/paper_checkpoints --out results/tables/sidtd_checkpoints.md || { echo "[FAIL] scoring"; exit 1; }
echo "Done."

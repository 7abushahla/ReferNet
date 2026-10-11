#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --job-name=input-ablation-smoke
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
source /opt/miniconda/etc/profile.d/conda.sh
conda activate refer-or-clear
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export TMPDIR="/scratch/$USER/input-smoke-$SLURM_JOB_ID/tmp"
mkdir -p "$TMPDIR"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python tests/test_input_preparation.py
python tests/test_train_config.py
python tests/test_augmentation_alignment.py
python tests/test_control_noex.py
python scripts/prepare_data.py --data-root data --datasets sidtd --workers 4 \
    --input-mode direct --no-register --anchor-fold 0 --out data/processed_direct_ablation
python scripts/check_input_ablation.py --out "results/diagnostics/input-smoke-$SLURM_JOB_ID/inputs.json"
RUNS=()
for condition in direct letterbox registered; do
    case "$condition" in
        direct) source_data=data/processed_direct_ablation;;
        letterbox) source_data=data/processed_noreg;;
        registered) source_data=data/processed;;
    esac
    data_root="/scratch/$USER/input-smoke-$SLURM_JOB_ID/$condition"
    mkdir -p "$data_root"
    rsync -a --exclude='._*' "$source_data/" "$data_root/"
    run="results/runs/A-input-$condition-f0-s42-smoke-$SLURM_JOB_ID"
    RUNS+=("$run")
    python -m refer_or_clear.train --config configs/protocol_a.yaml --arch changerex_r18 \
        --fusion single_noex --reference median --init levir --seed 42 --fold 0 \
        --epochs 1 --limit 32 --bn-policy batch_stats --validation-only --strict-fp32 \
        --data-root "$data_root" --workers 4 --device cuda --run-dir "$run"
    python -m refer_or_clear.evaluate --checkpoint "checkpoints/$(basename "$run")/best.pt" \
        --config configs/protocol_a.yaml --split val --batch-size 1 --limit 32 \
        --data-root "$data_root" --workers 4 --device cuda --run-dir "$run"
done
python scripts/check_input_ablation.py --smokes "${RUNS[@]}" \
    --out "results/diagnostics/input-smoke-$SLURM_JOB_ID/smokes.json"

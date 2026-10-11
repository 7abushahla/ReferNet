#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=classification-anchor
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
export ROC_LEVIR_CKPT="$repository_root/checkpoints/pretrained/ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth"
export TMPDIR="/scratch/$USER/classification-anchor-$SLURM_JOB_ID/tmp"
mkdir -p "$TMPDIR"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python "$code_root/tests/test_classification_loss.py"
python "$code_root/tests/test_train_config.py"
python "$code_root/tests/test_classification_anchor_gate.py"
data_root="/scratch/$USER/classification-anchor-$SLURM_JOB_ID/data"
mkdir -p "$data_root"
python - <<'PY'
import hashlib
import json
from pathlib import Path

gate = json.loads(Path("results/diagnostics/input-smoke-37920/inputs.json").read_text())["conditions"]["registered"]
for filename, field in (("records.csv", "records_sha256"), ("transforms.json", "transforms_sha256"),
                        ("splits/protocol_a_it0.csv", "split_sha256")):
    assert hashlib.sha256((Path("data/processed") / filename).read_bytes()).hexdigest() == gate[field]
print("registered input manifests verified")
PY
rsync -a --exclude='._*' data/processed/ "$data_root/"
baseline="results/runs/A-input-registered-f0-s42-fp32-jfix-e100/train.json"
common=(--config "$code_root/configs/protocol_a.yaml" --arch changerex_r18 --fusion single_noex
        --reference median --init levir --seed 42 --fold 0 --bn-policy batch_stats
        --validation-only --strict-fp32 --mask-loss-weight 0 --data-root "$data_root" --workers 4 --device cuda)
smoke="results/runs/A-classification-only-f0-s42-smoke-$SLURM_JOB_ID"
[[ ! -e "$smoke" && ! -e "checkpoints/$(basename "$smoke")" ]]
python -m refer_or_clear.train "${common[@]}" --epochs 1 --limit 32 --run-dir "$smoke"
python -m refer_or_clear.evaluate --checkpoint "checkpoints/$(basename "$smoke")/best.pt" \
    --config "$code_root/configs/protocol_a.yaml" --split val --batch-size 1 --limit 32 \
    --data-root "$data_root" --workers 4 --device cuda --run-dir "$smoke"
diagnostics="results/diagnostics/classification-anchor-$SLURM_JOB_ID"
python "$code_root/scripts/check_classification_anchor.py" --run "$smoke" --baseline "$baseline" \
    --smoke --out "$diagnostics/smoke.json"
run="results/runs/A-classification-only-registered-f0-s42-fp32-jfix-e100"
[[ ! -e "$run" && ! -e "checkpoints/$(basename "$run")" ]]
python -m refer_or_clear.train "${common[@]}" --epochs 100 --run-dir "$run"
python -m refer_or_clear.evaluate --checkpoint "checkpoints/$(basename "$run")/best.pt" \
    --config "$code_root/configs/protocol_a.yaml" --split val --batch-size 1 \
    --data-root "$data_root" --workers 4 --device cuda --run-dir "$run"
python "$code_root/scripts/check_classification_anchor.py" --run "$run" --baseline "$baseline" \
    --out "$diagnostics/completed.json"

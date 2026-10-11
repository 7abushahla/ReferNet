#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=input-ablation
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
condition="${1:?Specify direct, letterbox or registered}"
smoke_job="${2:?Specify the successful smoke job ID}"
source /opt/miniconda/etc/profile.d/conda.sh
conda activate refer-or-clear
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export TMPDIR="/scratch/$USER/input-$SLURM_JOB_ID/tmp"
mkdir -p "$TMPDIR"
case "$condition" in
    direct) source_data=data/processed_direct_ablation;;
    letterbox) source_data=data/processed_noreg;;
    registered) source_data=data/processed;;
    *) echo "unknown condition $condition"; exit 2;;
esac
python - "$smoke_job" "$condition" "$source_data" <<'PY'
import json
from pathlib import Path
import sys
from scripts.diagnose_control import sha256

smoke_job, condition, directory = sys.argv[1:]
gate = Path("results/diagnostics") / f"input-smoke-{smoke_job}"
assert json.loads((gate / "smokes.json").read_text())["passed"]
inputs = json.loads((gate / "inputs.json").read_text())["conditions"][condition]
assert inputs["data_root"] == directory
root = Path(directory)
for filename, field in (("splits/protocol_a_it0.csv", "split_sha256"),
                        ("records.csv", "records_sha256"), ("transforms.json", "transforms_sha256")):
    assert sha256(root / filename) == inputs[field], f"input manifest changed: {filename}"
print("smoke gate and input manifests verified")
PY
data_root="/scratch/$USER/input-$SLURM_JOB_ID/data"
mkdir -p "$data_root"
rsync -a --exclude='._*' "$source_data/" "$data_root/"
run="results/runs/A-input-$condition-f0-s42-fp32-jfix-e100"
[[ ! -e "$run" && ! -e "checkpoints/$(basename "$run")" ]]
python -m refer_or_clear.train --config configs/protocol_a.yaml --arch changerex_r18 \
    --fusion single_noex --reference median --init levir --seed 42 --fold 0 \
    --epochs 100 --bn-policy batch_stats --validation-only --strict-fp32 \
    --data-root "$data_root" --workers 4 --device cuda --run-dir "$run"
python -m refer_or_clear.evaluate --checkpoint "checkpoints/$(basename "$run")/best.pt" \
    --config configs/protocol_a.yaml --split val --batch-size 1 \
    --data-root "$data_root" --workers 4 --device cuda --run-dir "$run"

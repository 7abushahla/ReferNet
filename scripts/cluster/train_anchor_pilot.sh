#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=anchor-pilot
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -lt 10 || $# -gt 11 ]]; then
    echo "Usage: $0 PHASE SOURCE_DIR CONFIG_PATH RAW_ROOT LEGACY_ROOT RECORDS SPLIT WEIGHTS WEIGHTS_SHA256 OUTPUT_ROOT [REVIEW_JSON]" >&2
    exit 2
fi
phase="$1"
case "$phase" in
    diagnostic) [[ $# -eq 10 ]] || { echo 'Diagnostics do not consume review records' >&2; exit 2; } ;;
    train) [[ $# -eq 11 ]] || { echo 'Training requires an explicit review record' >&2; exit 2; } ;;
    *) echo 'PHASE must be diagnostic or train' >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'Compute work is forbidden on login nodes' >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == gpu ]] || { echo 'Allocated GPU partition required' >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
source /opt/miniconda/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
python_bin="$environment_path/bin/python"
cd "$SLURM_SUBMIT_DIR"
source_dir="$(cd "$2" && pwd -P)"
scratch="${ROC_SCRATCH_ROOT:-/scratch/$USER}/anchor-pilot-$SLURM_JOB_ID"
[[ "$scratch" == /* ]] || scratch="$PWD/$scratch"
run_dir="${10}/anchor-pilot-$phase-$SLURM_JOB_ID"
[[ "$run_dir" == /* ]] || run_dir="$PWD/$run_dir"
export PYTHONPATH="$source_dir/src"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TMPDIR="$scratch/tmp"
if [[ "$phase" == diagnostic ]]; then
    export ROC_ANCHOR_GPU_INTEGRATION=1
    export ROC_LEVIR_CKPT="$scratch/inputs/weights.pth"
    export ROC_LEVIR_SHA256="$9"
fi
review_args=()
if [[ "$phase" == train ]]; then review_args=(--review "${11}"); fi
"$python_bin" "$source_dir/scripts/anchor_job_manifest.py" prepare --phase "$phase" \
    --source-dir "$source_dir" --config "$3" --raw-root "$4" --legacy-root "$5" \
    --records "$6" --split "$7" --weights "$8" --weights-sha256 "$9" \
    --stage-dir "$scratch" --run-dir "$run_dir" "${review_args[@]}"
mkdir "$TMPDIR"
code_root="$scratch/code"
pilot_dir="$run_dir/pilot"
export PYTHONPATH="$code_root/src"
cd "$code_root"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"$python_bin" -m pytest --version | tee "$run_dir/pytest-version.txt"
common=(--config "$scratch/inputs/config.yaml" --raw-root "$scratch/raw" --legacy-root "$scratch/legacy"
        --records "$scratch/inputs/records.csv" --split "$scratch/inputs/split.csv"
        --weights "$scratch/inputs/weights.pth" --weights-sha256 "$9" --run-dir "$pilot_dir" --device cuda)
if [[ "$phase" == diagnostic ]]; then
    track="$("$python_bin" -c 'import json, sys; from pathlib import Path; print(json.loads(Path(sys.argv[1]).read_text())["track"])' "$run_dir/input_manifest.json")"
    if [[ "$track" == D || "$track" == D_running || "$track" == D_running_pilot || "$track" == N ]]; then
        "$python_bin" -m pytest "$code_root/tests/test_anchor_changer.py::test_actual_opencd_gpu_integration" \
            -q --maxfail=1 -s --junitxml "$run_dir/actual-stack-test.xml" 2>&1 | tee "$run_dir/actual-stack-test.log"
        "$python_bin" "$code_root/scripts/anchor_job_manifest.py" check-stack --report "$run_dir/actual-stack-test.xml"
    fi
    "$python_bin" "$code_root/scripts/check_anchor_pilots.py" "${common[@]}" --phase pretrain
    "$python_bin" "$code_root/scripts/anchor_job_manifest.py" check-gate --run-dir "$run_dir"
    exit 0
fi
"$python_bin" -m refer_or_clear.pilots "${common[@]}" --phase train --pretrain-gate "$scratch/inputs/pretrain_gate.json"
"$python_bin" "$code_root/scripts/check_anchor_pilots.py" "${common[@]}" --phase selected

#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=encoding-headers
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 5 ]]; then
    echo "Usage: $0 SOURCE_DIR RECORDS SPLIT RAW_ROOT OUTPUT_ROOT" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'Header audit is forbidden on login nodes' >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == gpu ]] || { echo 'Allocated compute partition required' >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
source /opt/miniconda/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
python_bin="$environment_path/bin/python"
cd "$SLURM_SUBMIT_DIR"
source_dir="$(cd "$1" && pwd -P)"
raw_root="$(cd "$4" && pwd -P)"
scratch="${ROC_SCRATCH_ROOT:-/scratch/$USER}/encoding-headers-$SLURM_JOB_ID"
[[ "$scratch" == /* ]] || scratch="$PWD/$scratch"
run_dir="$5/encoding-headers-$SLURM_JOB_ID"
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
"$python_bin" "$source_dir/scripts/anchor_job_manifest.py" prepare --phase audit \
    --source-dir "$source_dir" --records "$2" --split "$3" --raw-root "$raw_root" \
    --stage-dir "$scratch" --run-dir "$run_dir"
mkdir "$TMPDIR"
code_root="$scratch/code"
export PYTHONPATH="$code_root/src"
cd "$code_root"
"$python_bin" "$code_root/scripts/audit_encoding_headers.py" \
    --records "$scratch/inputs/records.csv" --split "$scratch/inputs/split.csv" \
    --raw-root "$raw_root" --out "$run_dir/audit" --inventory-test-directory
"$python_bin" "$code_root/scripts/anchor_job_manifest.py" check-audit --run-dir "$run_dir"

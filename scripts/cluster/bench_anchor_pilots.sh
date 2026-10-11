#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=anchor-cpu-v1
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 7 ]]; then
    echo "Usage: $0 SOURCE_DIR R_RUN D_RUN N_RUN RAW_ROOT LEGACY_ROOT OUTPUT_ROOT" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SLURM_CPUS_PER_TASK:?Allocated CPU count required}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'Compute work is forbidden on login nodes' >&2; exit 2 ;;
esac
[[ "$SLURM_CPUS_PER_TASK" =~ ^[0-9]+$ && "$SLURM_CPUS_PER_TASK" -ge 4 ]] || { echo 'Four allocated CPUs required' >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == cpu ]] || { echo 'Allocated CPU partition required' >&2; exit 2; }
source /opt/miniconda3/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
python_bin="$environment_path/bin/python"
cd "$SLURM_SUBMIT_DIR"
source_dir="$(cd "$1" && pwd -P)"
output_root="$7/anchor-cpu-v1-$SLURM_JOB_ID"
[[ ! -e "$output_root" ]] || { echo 'Output must be new' >&2; exit 2; }
mkdir -p "$output_root"
scratch="${ROC_SCRATCH_ROOT:-/tmp/$USER}/anchor-cpu-v1-$SLURM_JOB_ID"
[[ ! -e "$scratch" ]] || { echo 'Scratch must be new' >&2; exit 2; }
mkdir -p "$scratch/code" "$scratch/tmp"
rsync -a --exclude='._*' --exclude='.git' --exclude='results' --exclude='data' --exclude='checkpoints' "$source_dir/" "$scratch/code/"
code_root="$scratch/code"
export PYTHONPATH="$code_root/src"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TMPDIR="$scratch/tmp"
"$python_bin" -m pip freeze > "$output_root/environment.txt"
printf '%s\n' "contract=anchor_cpu_v1" "environment=$environment_path" "python=$python_bin" \
    'resource_contract=1 node,4 CPUs,16GiB,2h,no GPU,no concurrent model workload' > "$output_root/allocation.txt"
"$python_bin" "$code_root/scripts/bench_anchor_pilots.py" --runs "$2" "$3" "$4" \
    --runtime-manifest "$output_root/runtime-manifest.json" --raw-root "$5" --legacy-root "$6" \
    --out "$output_root/runtime.json"

#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=protocol-b-enrollment
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 7 ]]; then
    echo "Usage: $0 SOURCE_DIR RECORDS RECORDS_SHA SPLIT RAW_ROOT OUTPUT_PARENT GROUP" >&2
    exit 2
fi
group="$7"
[[ "$group" =~ ^[0-9]$ ]] || { echo 'Group must be an integer in 0..9' >&2; exit 2; }
[[ "$3" =~ ^[0-9a-f]{64}$ ]] || { echo 'Records SHA256 must be lowercase hexadecimal' >&2; exit 2; }
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SLURM_CPUS_PER_TASK:?At least four allocated CPUs required}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'B enrollment is forbidden on login nodes' >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == cpu ]] || { echo 'Allocated CPU partition required' >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
[[ "$SLURM_CPUS_PER_TASK" =~ ^[0-9]+$ ]] && (( 10#$SLURM_CPUS_PER_TASK >= 4 )) || {
    echo 'At least four allocated CPUs required' >&2; exit 2;
}
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" && "${SLURM_GPUS:-0}" == 0 ]] || {
    echo 'B enrollment requires a CPU-only allocation' >&2; exit 2;
}
source /opt/miniconda3/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
[[ "${CONDA_PREFIX:-}" == "$environment_path" ]] || { echo 'Incorrect project Conda environment' >&2; exit 2; }
python_bin="$environment_path/bin/python"
cd "$SLURM_SUBMIT_DIR"
source_dir="$(cd "$1" && pwd -P)"
raw_root="$(cd "$5" && pwd -P)"
[[ -f "$source_dir/scripts/prepare_protocol_b.py" && -f "$source_dir/scripts/anchor_job_manifest.py" && -d "$source_dir/src" ]] || {
    echo 'Complete scientific source snapshot required' >&2; exit 2;
}
scratch="${ROC_SCRATCH_ROOT:-/tmp/$USER}/protocol-b-group$group-$SLURM_JOB_ID"
[[ "$scratch" == /* ]] || scratch="$PWD/$scratch"
run_dir="$6/protocol-b-group$group-$SLURM_JOB_ID"
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
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="$scratch/tmp"
"$python_bin" -c 'import hashlib, pathlib, sys; actual = hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest();
if actual != sys.argv[2]: raise ValueError("Records SHA256 mismatch before staging")' "$2" "$3"
"$python_bin" "$source_dir/scripts/anchor_job_manifest.py" prepare --phase audit \
    --source-dir "$source_dir" --records "$2" --split "$4" --raw-root "$raw_root" \
    --stage-dir "$scratch" --run-dir "$run_dir"
mkdir "$TMPDIR"
code_root="$scratch/code"
export PYTHONPATH="$code_root/src"
cd "$code_root"
"$python_bin" "$code_root/scripts/prepare_protocol_b.py" \
    --records "$scratch/inputs/records.csv" --records-sha256 "$3" --split "$scratch/inputs/split.csv" \
    --group "$group" --raw-root "$raw_root" --output-root "$run_dir/preparation" --build

#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1:00:00
#SBATCH --exclude=gpu-dy-g5-0-16,gpu-dy-g5-0-3
#SBATCH --job-name=protocol-c-v2-score
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'Usage: SOURCE_DIR REQUEST REQUEST_SHA' >&2
    exit 2
fi
[[ "$3" =~ ^[0-9a-f]{64}$ ]] || { echo 'Lowercase request SHA256 required' >&2; exit 2; }
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SLURM_CPUS_PER_TASK:?Allocated CPU count required}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'Compute work is forbidden on login nodes' >&2; exit 2 ;;
esac
case "$SLURMD_NODENAME" in
    gpu-dy-g5-0-16|gpu-dy-g5-0-3) echo 'Excluded scoring node' >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == gpu ]] || { echo 'Allocated GPU partition required' >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_JOB_ID" =~ [1-9] ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
[[ "$SLURM_CPUS_PER_TASK" =~ ^[0-9]+$ && "$SLURM_CPUS_PER_TASK" -ge 4 ]] || { echo 'Four allocated CPUs required' >&2; exit 2; }
gpu_allocated=false
for count in "${SLURM_GPUS:-0}" "${SLURM_GPUS_ON_NODE:-0}"; do
    if [[ "$count" =~ ^[0-9]+$ && "$count" -gt 0 ]]; then gpu_allocated=true; fi
done
for devices in "${SLURM_JOB_GPUS:-}" "${SLURM_STEP_GPUS:-}"; do
    if [[ -n "$devices" && "$devices" != -1 && "$devices" != NoDevFiles ]]; then gpu_allocated=true; fi
done
[[ "$gpu_allocated" == true ]] || { echo 'Allocated GPU resource required' >&2; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != -1 && "$CUDA_VISIBLE_DEVICES" != NoDevFiles ]] || {
    echo 'Allocated GPU visibility required' >&2; exit 2;
}
unset PYTHONPATH PYTHONHOME ROC_ANCHOR_GPU_INTEGRATION ROC_DIRECT_REFERENCE_GPU_INTEGRATION ROC_LEVIR_CKPT ROC_LEVIR_SHA256
source /opt/miniconda/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
[[ "${CONDA_PREFIX:-}" == "$environment_path" ]] || { echo 'Exact environment prefix required' >&2; exit 2; }
python_bin="$environment_path/bin/python"
cd "$SLURM_SUBMIT_DIR"
source_dir="$(cd "$1" && pwd -P)"
"$python_bin" -c 'import sys
from pathlib import Path
if sys.prefix != "/shared/b00090279/conda/envs/refer-or-clear" or str(Path(sys.executable).parent) != sys.prefix + "/bin":
    raise SystemExit("interpreter prefix mismatch")
root = Path(sys.argv[1])
if not root.is_dir() or not (root / "src").is_dir() or not (root / "scripts/score_protocol_c_v2.py").is_file():
    raise SystemExit("complete frozen scientific source required")
for path in (root, *root.rglob("*")):
    if path.is_symlink() or path.stat().st_mode & 0o222:
        raise SystemExit("readonly non-symlink scientific source required")' "$source_dir"
export PYTHONPATH="$source_dir/src:$source_dir/scripts"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
"$python_bin" "$source_dir/scripts/score_protocol_c_v2.py" --request "$2" --request-sha256 "$3"

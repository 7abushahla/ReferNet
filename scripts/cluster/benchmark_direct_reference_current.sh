#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=direct-reference-current-cpu
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
[[ $# -eq 4 ]] || { echo 'Usage: SOURCE_DIR REQUEST_JSON REQUEST_SHA256 NEW_OUTPUT_DIR' >&2; exit 2; }
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit directory required}"
: "${SLURM_CPUS_PER_TASK:?Allocated CPUs required}"
[[ "${USER:-}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_JOB_ID" =~ [1-9] ]] || exit 2
[[ "${SLURM_JOB_PARTITION:-}" == cpu && "$SLURM_CPUS_PER_TASK" -ge 4 ]] || exit 2
case "${SLURMD_NODENAME,,}" in *login*|*head*|*frontend*|*lgn*) exit 2 ;; esac
[[ "${SLURM_GPUS:-0}" == 0 && "${SLURM_GPUS_ON_NODE:-0}" == 0 && -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ]] || exit 2
source /opt/miniconda3/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
python_bin="$environment_path/bin/python"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
unset PYTHONPATH PYTHONHOME
cd "$SLURM_SUBMIT_DIR"
"$python_bin" - "$1" "$2" "$3" "$4" <<'PY'
import hashlib
import os
from pathlib import Path
import sys

prefix = Path('/shared/b00090279/conda/envs/refer-or-clear').resolve()
if Path(sys.prefix).resolve() != prefix or Path(os.environ.get('CONDA_PREFIX', '')).resolve() != prefix:
    raise ValueError('exact resolved interpreter/environment prefix required')
source, request, checksum, output = sys.argv[1:]
for value in (source, request, output, '/tmp/' + os.environ['USER']):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError('absolute non-symlink path required')
if hashlib.sha256(Path(request).read_bytes()).hexdigest() != checksum:
    raise ValueError('request SHA256 mismatch')
if Path(output).exists():
    raise ValueError('new output directory required')
if not (Path(source) / 'scripts/benchmark_direct_reference_current.py').is_file():
    raise ValueError('isolated measurement source missing')
PY
scratch_root="/tmp/$USER"
mkdir -p "$scratch_root"
scratch="$(mktemp -d "$scratch_root/direct-reference-current-cpu-$SLURM_JOB_ID.XXXXXX")"
cd "$scratch"
exec "$python_bin" -B "$1/scripts/benchmark_direct_reference_current.py" --request "$2" --request-sha256 "$3" --out "$4"

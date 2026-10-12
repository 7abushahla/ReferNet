#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=direct-reference-diagnostic
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 5 ]]; then
    echo "Usage: $0 SOURCE_DIR SOURCE_SHA256 REQUEST_JSON REQUEST_SHA256 OUTPUT_ROOT" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SLURM_CPUS_PER_TASK:?Allocated CPU count required}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'Compute work is forbidden on login nodes' >&2; exit 2 ;;
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
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != -1 && "$CUDA_VISIBLE_DEVICES" != NoDevFiles ]] || { echo 'Allocated GPU visibility required' >&2; exit 2; }
unset PYTHONPATH PYTHONHOME ROC_ANCHOR_GPU_INTEGRATION ROC_DIRECT_REFERENCE_GPU_INTEGRATION
source /opt/miniconda/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
[[ "${CONDA_PREFIX:-}" == "$environment_path" ]] || { echo 'Exact environment prefix required' >&2; exit 2; }
python_bin="$environment_path/bin/python"
"$python_bin" -c 'import sys
if sys.prefix != "/shared/b00090279/conda/envs/refer-or-clear":
    raise SystemExit("interpreter prefix mismatch")'
cd "$SLURM_SUBMIT_DIR"
source_dir="$1"
[[ "$source_dir" == /* ]] || source_dir="$PWD/$source_dir"
request="$3"
[[ "$request" == /* ]] || request="$PWD/$request"
scratch="${ROC_SCRATCH_ROOT:-/scratch/$USER}/direct-reference-diagnostic-$SLURM_JOB_ID"
[[ "$scratch" == /* ]] || scratch="$PWD/$scratch"
run_dir="$5/direct-reference-diagnostic-$SLURM_JOB_ID"
[[ "$run_dir" == /* ]] || run_dir="$PWD/$run_dir"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
"$python_bin" "$source_dir/scripts/direct_reference_job.py" prepare \
    --source-dir "$source_dir" --source-sha256 "$2" --request "$request" --request-sha256 "$4" \
    --stage-dir "$scratch" --run-dir "$run_dir"
weights_sha256="$("$python_bin" - "$scratch/inputs/request.json" "$4" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys


def unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate frozen request key")
        result[name] = value
    return result


content = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(content).hexdigest() != sys.argv[2]:
    raise ValueError("frozen request SHA256 mismatch")
request = json.loads(content, object_pairs_hook=unique_object)
digest = request["weights"]["sha256"]
if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise ValueError("frozen request requires a lowercase weight SHA256")
print(digest)
PY
)"
export TMPDIR="$scratch/tmp"
[[ -d "$TMPDIR" && ! -L "$TMPDIR" ]] || { echo 'Staged temporary directory required' >&2; exit 2; }
code_root="$scratch/code"
export PYTHONPATH="$code_root/src"
export ROC_LEVIR_CKPT="$scratch/inputs/weights.pth"
export ROC_LEVIR_SHA256="$weights_sha256"
cd "$code_root"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"$python_bin" -m pytest --version | tee "$run_dir/pytest-version.txt"
ROC_ANCHOR_GPU_INTEGRATION=1 "$python_bin" -m pytest \
    "$code_root/tests/test_direct_reference_model.py::test_actual_direct_reference_gpu_integration" \
    -q --maxfail=1 -s --junitxml "$run_dir/actual-stack-test.xml" 2>&1 | tee "$run_dir/actual-stack-test.log"
"$python_bin" "$code_root/scripts/direct_reference_job.py" check-stack --report "$run_dir/actual-stack-test.xml"
"$python_bin" "$code_root/scripts/check_direct_reference.py" \
    --request "$scratch/inputs/request.json" --request-sha256 "$4" --run-dir "$run_dir" --device cuda

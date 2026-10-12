#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=direct-reference-control
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 7 ]]; then
    echo "Usage: $0 SOURCE_DIR SOURCE_SHA256 DIAGNOSTIC_REQUEST DIAGNOSTIC_SHA256 CONTROL_REQUEST CONTROL_SHA256 OUTPUT_ROOT" >&2
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
control_request="$5"
[[ "$control_request" == /* ]] || control_request="$PWD/$control_request"
source_sha256="$2"
request_sha256="$4"
control_sha256="$6"
scratch="${ROC_SCRATCH_ROOT:-/scratch/$USER}/direct-reference-control-$SLURM_JOB_ID"
[[ "$scratch" == /* ]] || scratch="$PWD/$scratch"
run_dir="$7/direct-reference-control-$SLURM_JOB_ID"
[[ "$run_dir" == /* ]] || run_dir="$PWD/$run_dir"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
control_preflight() {
    "$python_bin" - "$source_sha256" "$1" "$request_sha256" "$control_request" "$control_sha256" "$scratch" "$run_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "lowercase SHA256 required")
    return value


def absolute_path(value):
    require(isinstance(value, str), "path string required")
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts, "absolute path without traversal required")
    require(not any(parent.is_symlink() for parent in (path, *path.parents)), "symlink path forbidden")
    return path


def authenticate(path, expected):
    content = absolute_path(path).read_bytes()
    require(hashlib.sha256(content).hexdigest() == digest(expected), "authority SHA256 mismatch")
    return content


def pairs(entries):
    result = {}
    for name, value in entries:
        require(name not in result, "duplicate authority JSON key")
        result[name] = value
    return result


def invalid_constant(value):
    raise ValueError("nonfinite authority JSON")


def read_json(content):
    value = json.loads(content, object_pairs_hook=pairs, parse_constant=invalid_constant)
    require(isinstance(value, dict), "authority JSON object required")
    return value


source_hash, request_path, request_hash, control_path, control_hash, scratch, run = sys.argv[1:]
digest(source_hash)
request = read_json(authenticate(request_path, request_hash))
require(request.get("schema_version") == "direct_reference_diagnostic_request_v1"
        and request.get("source_sha256") == source_hash, "diagnostic staging source mismatch")
control = read_json(authenticate(control_path, control_hash))
require(set(control) == {"schema_version", "source_sha256", "config_sha256", "diagnostic_request_sha256",
                         "arm", "epochs", "fold", "seed", "diagnostic_gate"}, "fixed control authority fields required")
require(control["schema_version"] == "direct_reference_control_request_v1", "control authority schema mismatch")
require(control["source_sha256"] == source_hash and control["diagnostic_request_sha256"] == request_hash,
        "control source/diagnostic request mismatch")
require(control["arm"] == "off", "only reference-off control authorized")
for name, expected in (("epochs", 100), ("fold", 0), ("seed", 42)):
    require(type(control[name]) is int and control[name] == expected, "fixed control recipe mismatch: " + name)
config = request["config"]
require(isinstance(config, dict) and set(config) == {"path", "sha256"}, "config path/hash authority required")
authenticate(config["path"], config["sha256"])
require(control["config_sha256"] == config["sha256"], "original configuration bytes mismatch")
gate_binding = control["diagnostic_gate"]
require(isinstance(gate_binding, dict) and set(gate_binding) == {"path", "sha256"}, "diagnostic gate path/hash required")
gate = read_json(authenticate(gate_binding["path"], gate_binding["sha256"]))
require(gate.get("schema_version") == "direct_reference_diagnostic_v1" and gate.get("diagnostic_only") is True
        and gate.get("passed") is True and gate.get("no_test_open") is True, "successful diagnostic gate required")
require(isinstance(gate.get("identities"), dict) and gate["identities"].get("source_sha256") == source_hash,
        "same-source diagnostic gate required")
protected = (absolute_path(control_path).parent, absolute_path(gate_binding["path"]).parent)
for destination in (absolute_path(scratch), absolute_path(run)):
    require(all(not destination.is_relative_to(root) and not root.is_relative_to(destination) for root in protected),
            "output must be isolated from control input trees")
digest(request["weights"]["sha256"])
print(request["weights"]["sha256"])
PY
}
weights_sha256="$(control_preflight "$request")"
"$python_bin" "$source_dir/scripts/direct_reference_job.py" prepare \
    --source-dir "$source_dir" --source-sha256 "$source_sha256" --request "$request" --request-sha256 "$request_sha256" \
    --stage-dir "$scratch" --run-dir "$run_dir"
weights_sha256="$(control_preflight "$scratch/inputs/request.json")"
export TMPDIR="$scratch/tmp"
[[ -d "$TMPDIR" && ! -L "$TMPDIR" ]] || { echo 'Staged temporary directory required' >&2; exit 2; }
code_root="$scratch/code"
export PYTHONPATH="$code_root/src"
export ROC_LEVIR_CKPT="$scratch/inputs/weights.pth"
export ROC_LEVIR_SHA256="$weights_sha256"
cd "$code_root"
"$python_bin" -c 'import torch
if not torch.cuda.is_available():
    raise SystemExit("allocated CUDA required")
print(torch.cuda.get_device_name(0))'
"$python_bin" -m pytest --version | tee "$run_dir/pytest-version.txt"
ROC_ANCHOR_GPU_INTEGRATION=1 "$python_bin" -m pytest \
    "$code_root/tests/test_direct_reference_model.py::test_actual_direct_reference_gpu_integration" \
    -p no:cacheprovider -q --maxfail=1 -s --junitxml "$run_dir/actual-stack-test.xml" 2>&1 | tee "$run_dir/actual-stack-test.log"
"$python_bin" "$code_root/scripts/direct_reference_job.py" check-stack --report "$run_dir/actual-stack-test.xml"
weights_sha256="$(control_preflight "$scratch/inputs/request.json")"
"$python_bin" "$code_root/scripts/train_direct_reference_control.py" \
    --request "$scratch/inputs/request.json" --request-sha256 "$request_sha256" \
    --control-request "$control_request" --control-request-sha256 "$control_sha256" --run-dir "$run_dir" --device cuda

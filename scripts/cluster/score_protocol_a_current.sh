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
#SBATCH --job-name=protocol-a-current-score
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 6 ]]; then
    echo 'Usage: SOURCE_DIR REQUEST REQUEST_SHA INPUTS_DIR INPUTS_MANIFEST INPUTS_SHA' >&2
    exit 2
fi
[[ "$3" =~ ^[0-9a-f]{64}$ && "$6" =~ ^[0-9a-f]{64}$ ]] || { echo 'Lowercase external SHA256 bindings required' >&2; exit 2; }
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SLURM_CPUS_PER_TASK:?Four allocated CPUs required}"
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
[[ "${CONDA_PREFIX:-}" == "$environment_path" ]] || { echo 'Exact project environment required' >&2; exit 2; }
python_bin="$environment_path/bin/python"
cd "$SLURM_SUBMIT_DIR"
source_dir="$1"
[[ "$source_dir" == /* ]] || source_dir="$PWD/$source_dir"
restored_inputs="/scratch/b00090279/direct-reference-control-38041/inputs"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
"$python_bin" - "$source_dir" "$2" "$3" "$4" "$5" "$6" "$restored_inputs" <<'PY'
import hashlib
from pathlib import Path
import re
import shutil
import stat
import sys

if sys.prefix != '/shared/b00090279/conda/envs/refer-or-clear' or str(Path(sys.executable).parent) != sys.prefix + '/bin':
    raise ValueError('interpreter prefix mismatch')


def path(value):
    result = Path(value).absolute()
    if '..' in result.parts or any(parent.is_symlink() for parent in (result, *result.parents)):
        raise ValueError('non-symlink paths without traversal required')
    return result


def digest(value):
    metadata = value.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError('regular non-hardlinked input required')
    checksum = hashlib.sha256()
    with value.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def authenticate(value, expected):
    if digest(value) != expected:
        raise ValueError('external artifact SHA256 mismatch')


source, request, request_hash, inputs, manifest, manifest_hash, restored = sys.argv[1:]
source, request, inputs, manifest, restored = map(path, (source, request, inputs, manifest, restored))
if not source.is_dir() or not (source / 'src').is_dir() or not (source / 'scripts/score_protocol_a_current.py').is_file():
    raise ValueError('complete frozen EEC overlay required')
for entry in (source, *source.rglob('*')):
    if entry.is_symlink() or entry.stat().st_mode & 0o222:
        raise ValueError('readonly non-symlink source required')
authenticate(request, request_hash)
authenticate(manifest, manifest_hash)
payload = manifest.read_bytes()
if hashlib.sha256(payload).hexdigest() != manifest_hash:
    raise ValueError('inventory changed during read')
names = {'enrollment.json', 'weights.pth', 'split.csv', 'product-root.json', 'request.json', 'config.yaml', 'records.csv'}
inventory = {}
for line in payload.decode('utf-8').splitlines():
    match = re.fullmatch(r'([0-9a-f]{64})  ([A-Za-z0-9_.-]+)', line)
    if match is None or match[2] not in names or match[2] in inventory:
        raise ValueError('unique approved basename inventory required')
    inventory[match[2]] = match[1]
if set(inventory) != names or not inputs.is_dir() or {entry.name for entry in inputs.iterdir()} != names:
    raise ValueError('exact seven original inputs required')
if any(restored.is_relative_to(item) or item.is_relative_to(restored) for item in (source, inputs, request, manifest)):
    raise ValueError('restoration overlaps immutable authority')
for name, expected in inventory.items():
    authenticate(path(inputs / name), expected)
if restored.exists():
    if not restored.is_dir() or not {entry.name for entry in restored.iterdir()} <= names:
        raise ValueError('unexpected existing staged input')
    for entry in restored.iterdir():
        authenticate(path(entry), inventory[entry.name])
restored.mkdir(parents=True, exist_ok=True)
for name, expected in inventory.items():
    destination = restored / name
    if destination.exists():
        continue
    with path(inputs / name).open('rb') as source_stream, destination.open('xb') as output:
        shutil.copyfileobj(source_stream, output, length=1048576)
    authenticate(destination, expected)
    destination.chmod(0o444)
for name, expected in inventory.items():
    authenticate(path(restored / name), expected)
authenticate(request, request_hash)
PY
export PYTHONPATH="$source_dir/src:$source_dir/scripts"
"$python_bin" "$source_dir/scripts/score_protocol_a_current.py" --request "$2" --request-sha256 "$3"

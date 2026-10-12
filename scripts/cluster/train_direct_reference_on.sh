#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --job-name=direct-reference-on
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 8 ]]; then
    echo 'Usage: ARCHIVE TAR_SHA256 FILES_MANIFEST MANIFEST_SHA256 LAUNCHER ON_REQUEST ON_SHA256 RUN_DIR' >&2
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
unset PYTHONPATH PYTHONHOME ROC_ANCHOR_GPU_INTEGRATION ROC_DIRECT_REFERENCE_GPU_INTEGRATION ROC_LEVIR_CKPT ROC_LEVIR_SHA256
source /opt/miniconda/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
[[ "${CONDA_PREFIX:-}" == "$environment_path" ]] || { echo 'Exact environment prefix required' >&2; exit 2; }
python_bin="$environment_path/bin/python"
"$python_bin" -c 'import sys
from pathlib import Path
if sys.prefix != "/shared/b00090279/conda/envs/refer-or-clear" or str(Path(sys.executable).parent) != sys.prefix + "/bin":
    raise SystemExit("interpreter prefix mismatch")'
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
restored_root="/scratch/b00090279/direct-reference-control-38041"
request_sha256="$("$python_bin" - "$@" "$restored_root" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts
            and not any(parent.is_symlink() for parent in (path, *path.parents)), 'absolute non-symlink path required')
    return path


def digest(stream):
    checksum = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1048576), b''):
        checksum.update(chunk)
    return checksum.hexdigest()


def authenticate(path, expected):
    path = absolute(path)
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'lowercase SHA256 required')
    with path.open('rb') as stream:
        require(digest(stream) == expected, 'external SHA256 mismatch')
    return path


def pairs(entries):
    result = {}
    for name, value in entries:
        require(name not in result, 'duplicate JSON key')
        result[name] = value
    return result


def read_json(path):
    value = json.loads(absolute(path).read_bytes(), object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    require(isinstance(value, dict), 'JSON object required')
    return value


def document(binding, expected_path=None):
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, 'external path/hash required')
    if expected_path is not None:
        require(binding['path'] == str(expected_path), 'canonical artifact path required')
    return read_json(authenticate(binding['path'], binding['sha256']))


def relative(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and not name.startswith('/') and '..' not in path.parts
            and '\\' not in name and str(path) == name.rstrip('/') and path.parts[0] in roots,
            'unsafe or unexpected snapshot path')
    return path


archive_path, archive_hash, inventory_path, inventory_hash, launcher_path, authority_path, authority_hash, run_path, stage_path = sys.argv[1:]
source_hash = 'a5ff28ae2a332b977551241f2d07641d51df78d54a4cc1b58169c7338ce11292'
launcher_hash = 'f2697cbf85067baeba354077dec2bc0427e39fcf71b0005396f0f265a054c012'
roots = {'code', 'inputs', 'raw', 'enrollment'}
stage, run, launcher = absolute(stage_path), absolute(run_path), absolute(launcher_path)
require(not stage.exists(), 'original scratch root must be absent; never overwrite')
require(run.is_dir() and not (run / 'training/on').exists(), 'original run and absent ON output required')
require(not run.is_relative_to(stage) and not stage.is_relative_to(run), 'shared output must not be extracted')
require(not launcher.is_relative_to(stage) and not launcher.is_relative_to(run), 'external launcher required')
authority = read_json(authenticate(authority_path, authority_hash))
fixed = dict(schema_version='direct_reference_on_request_v1', arm='on', epochs=100, fold=0, seed=42,
             source_sha256=source_hash, launcher_sha256=launcher_hash)
require(set(authority) == set(fixed) | {'config_sha256', 'diagnostic_request_sha256', 'off_completion', 'registration', 'phase_authority'},
        'fixed ON authority fields required')
require(all(type(authority.get(name)) is type(value) and authority[name] == value for name, value in fixed.items()),
        'fixed ON authority mismatch')
authenticate(launcher, launcher_hash)
completion = document(authority['off_completion'], run / 'training/off/control.json')
require(completion.get('schema_version') == 'direct_reference_control_result_v1' and completion.get('passed') is True
        and completion.get('arm') == 'off' and completion.get('phase') == 'train'
        and type(completion.get('epochs_completed')) is int and completion['epochs_completed'] == 100, 'completed passing OFF100 required')
control = document(completion['control_request'])
require(control.get('schema_version') == 'direct_reference_control_request_v1' and control.get('arm') == 'off'
        and all(control.get(name) == authority[name] for name in ('source_sha256', 'config_sha256', 'diagnostic_request_sha256')),
        'original OFF source/config/request required')
document(completion['train_result'], run / 'training/off/train.json')
replay = document(completion['fresh_replay'], run / 'training/off/fresh-replay.json')
require(replay.get('schema_version') == 'direct_reference_control_replay_v1' and replay.get('passed') is True
        and replay.get('fresh_process') is True and replay.get('arm') == 'off' and replay.get('phase') == 'train'
        and replay.get('train_result') == completion['train_result'] and replay.get('control_request') == completion['control_request'],
        'bound successful OFF selected replay required')
document(authority['registration'], run / 'training/shared/registration.json')
document(authority['phase_authority'], run / 'training/shared/phase-authority.json')
manifest = read_json(run / 'input_manifest.json')
expected_request = dict(path=str(stage / 'inputs/request.json'), sha256=authority['diagnostic_request_sha256'])
expected_paths = dict(stage_dir=str(stage), source_dir=str(stage / 'code'), raw_root=str(stage / 'raw'),
                      product_root=str(stage / 'enrollment'), run_dir=str(run))
require(manifest.get('schema_version') == 'direct_reference_job_v1' and manifest.get('diagnostic_only') is True
        and manifest.get('source_sha256') == source_hash and manifest.get('request') == expected_request
        and all(manifest.get('paths', {}).get(name) == value for name, value in expected_paths.items()),
        'original exact staged paths/context required')
require(manifest.get('authorities', {}).get('config') == dict(path=str(stage / 'inputs/config.yaml'), sha256=authority['config_sha256']),
        'original config bytes/location required')
inventory = {}
for line in authenticate(inventory_path, inventory_hash).read_text().splitlines():
    require(re.fullmatch('[0-9a-f]{64}  .+', line), 'sha256sum relative file manifest required')
    checksum, name = line.split('  ', 1)
    relative(name)
    require(name not in inventory, 'duplicate file manifest path')
    inventory[name] = checksum
require({PurePosixPath(name).parts[0] for name in inventory} == roots, 'all four snapshot trees required')
require(inventory.get('inputs/request.json') == expected_request['sha256']
        and inventory.get('inputs/config.yaml') == authority['config_sha256'], 'snapshot request/config hash mismatch')
with authenticate(archive_path, archive_hash).open('rb') as archive_stream:
    with tarfile.open(fileobj=archive_stream, mode='r:') as archive:
        members, regular = {}, {}
        for member in archive.getmembers():
            path = relative(member.name)
            name = str(path)
            require(name not in members and (member.isdir() or member.isfile()), 'duplicate, link, or special tar entry forbidden')
            require(len(path.parts) > 1 or member.isdir(), 'snapshot root must be a directory')
            members[name] = member
            if member.isfile():
                require(name in inventory, 'unexpected archive file')
                with archive.extractfile(member) as stream:
                    require(digest(stream) == inventory[name], 'snapshot member SHA256 mismatch')
                regular[name] = member
        require(set(regular) == set(inventory), 'archive and file inventory must match exactly')
        for name in members:
            require(all(str(parent) not in regular for parent in PurePosixPath(name).parents), 'file used as parent directory')
        stage.parent.mkdir(parents=True, exist_ok=True)
        absolute(stage)
        stage.mkdir(mode=0o700, exist_ok=False)
        for root in sorted(roots):
            (stage / root).mkdir()
        for name, member in regular.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream, target.open('xb') as output:
                shutil.copyfileobj(stream, output, length=1048576)
            authenticate(target, inventory[name])
            target.chmod(0o444)
        for name, member in members.items():
            if member.isdir():
                (stage / name).mkdir(parents=True, exist_ok=True)
        for directory, folders, files in os.walk(stage, topdown=False):
            if Path(directory) != stage:
                Path(directory).chmod(0o555)
        (stage / 'tmp').mkdir(mode=0o700)
print(expected_request['sha256'])
PY
)"
export TMPDIR="$restored_root/tmp"
cd "$restored_root"
"$python_bin" - "$5" "$6" "$7" <<'PY'
import hashlib
from pathlib import Path
import sys

for value, expected in ((sys.argv[1], 'f2697cbf85067baeba354077dec2bc0427e39fcf71b0005396f0f265a054c012'),
                        (sys.argv[2], sys.argv[3])):
    path = Path(value)
    if any(parent.is_symlink() for parent in (path, *path.parents)) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('launch boundary authority drift')
PY
"$python_bin" "$5" --source-dir "$restored_root/code" \
    --request "$restored_root/inputs/request.json" --request-sha256 "$request_sha256" \
    --on-request "$6" --on-request-sha256 "$7" --run-dir "$8" --device cuda

#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpu-dy-g5-0-16
#SBATCH --job-name=direct-reference-global
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 9 ]]; then
    echo 'Usage: ARCHIVE TAR_SHA256 FILES_MANIFEST MANIFEST_SHA256 GLOBAL_LAUNCHER HELPER_ON_LAUNCHER REQUEST_JSON REQUEST_SHA256 PHASE' >&2
    exit 2
fi
case "$9" in
    smoke|train|replay) ;;
    *) echo 'Phase must be smoke, train, or replay' >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SLURM_CPUS_PER_TASK:?Allocated CPU count required}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'Compute work is forbidden on login nodes' >&2; exit 2 ;;
esac
[[ "$SLURMD_NODENAME" != gpu-dy-g5-0-16 ]] || { echo 'Excluded global node' >&2; exit 2; }
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
"$python_bin" - "$@" "$restored_root" <<'PY'
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
    require(path.is_file(), 'regular input file required')
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


def relative(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and not name.startswith('/') and '..' not in path.parts
            and '\\' not in name and str(path) == name.rstrip('/') and path.parts[0] in roots,
            'unsafe or unexpected snapshot path')
    return path


archive_path, archive_hash, inventory_path, inventory_hash, launcher_path, helper_path, authority_path, authority_hash, phase, stage_path = sys.argv[1:]
source_hash = 'a5ff28ae2a332b977551241f2d07641d51df78d54a4cc1b58169c7338ce11292'
helper_hash = 'f2697cbf85067baeba354077dec2bc0427e39fcf71b0005396f0f265a054c012'
request_hash = '2f6d479835003242790f6d10f44249d73aa9a43f6a465132e618bc99c24c87b9'
manifest_hash = 'dbe33a68c945cb5f6b5c4140fb6000e293342150537bef4d09ff2304d46d613a'
roots = {'code', 'inputs', 'raw', 'enrollment'}
stage, launcher, helper = absolute(stage_path), absolute(launcher_path), absolute(helper_path)
authority = read_json(authenticate(authority_path, authority_hash))
run = absolute(authority['original_run_dir'])
output = absolute('/shared/b00090279/refer-or-clear/results/diagnostics/direct-reference-v1') / ('direct-reference-global-' + os.environ['SLURM_JOB_ID'])
absolute(output)
require(run.is_dir() and not output.exists(), 'original run and new global output required')
require(not run.is_relative_to(stage) and not stage.is_relative_to(run), 'shared output must not be extracted')
protected = [stage, run, absolute(archive_path), absolute(inventory_path), launcher, helper, absolute(authority_path)]
require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in protected),
        'global output must not overlap inputs or original run')
require(all(not path.is_relative_to(stage) and not path.is_relative_to(run) for path in (launcher, helper, absolute(authority_path))),
        'external launchers and authority required')
authenticate(launcher, authority['launcher_sha256'])
authenticate(helper, helper_hash)
manifest = read_json(authenticate(run / 'input_manifest.json', manifest_hash))
expected_request = dict(path=str(stage / 'inputs/request.json'), sha256=request_hash)
expected_paths = dict(stage_dir=str(stage), source_dir=str(stage / 'code'), raw_root=str(stage / 'raw'),
                      product_root=str(stage / 'enrollment'), run_dir=str(run))
require(manifest.get('schema_version') == 'direct_reference_job_v1' and manifest.get('diagnostic_only') is True
        and manifest.get('source_sha256') == source_hash and manifest.get('request') == expected_request
        and all(manifest.get('paths', {}).get(name) == value for name, value in expected_paths.items()),
        'original exact staged paths/context required')
config = manifest.get('authorities', {}).get('config')
require(isinstance(config, dict) and set(config) == {'path', 'sha256'}
        and config['path'] == str(stage / 'inputs/config.yaml'), 'original config binding required')
inventory = {}
for line in authenticate(inventory_path, inventory_hash).read_text().splitlines():
    require(re.fullmatch('[0-9a-f]{64}  .+', line), 'sha256sum relative file manifest required')
    checksum, name = line.split('  ', 1)
    relative(name)
    require(name not in inventory, 'duplicate file manifest path')
    inventory[name] = checksum
require({PurePosixPath(name).parts[0] for name in inventory} == roots, 'all four snapshot trees required')
require(inventory.get('inputs/request.json') == expected_request['sha256']
        and inventory.get('inputs/config.yaml') == config['sha256'], 'snapshot request/config hash mismatch')
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
        if stage.exists():
            require(stage.is_dir() and {path.name for path in stage.iterdir()} == roots | {'tmp'},
                    'existing scratch must be an exact reviewed restoration')
            require((stage / 'tmp').is_dir() and not (stage / 'tmp').is_symlink(), 'safe scratch tmp directory required')
            expected_dirs = set(roots)
            for name, member in members.items():
                expected_dirs.update(str(parent) for parent in PurePosixPath(name).parents if str(parent) != '.')
                if member.isdir():
                    expected_dirs.add(name)
            found_files, found_dirs = set(), set()
            for root in roots:
                tree = stage / root
                require(tree.is_dir() and not tree.is_symlink() and tree.stat().st_mode & 0o222 == 0,
                        'read-only restored tree required')
                found_dirs.add(root)
                for path in tree.rglob('*'):
                    absolute(path)
                    require(not path.is_symlink(), 'restored links forbidden')
                    name = path.relative_to(stage).as_posix()
                    require(path.stat().st_mode & 0o222 == 0, 'read-only restoration required')
                    if path.is_dir():
                        found_dirs.add(name)
                    else:
                        require(path.is_file() and name in inventory, 'unexpected restored file')
                        authenticate(path, inventory[name])
                        found_files.add(name)
            require(found_files == set(inventory) and found_dirs == expected_dirs, 'exact restored inventory required')
        else:
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
PY
export TMPDIR="$restored_root/tmp"
cd "$restored_root"
"$python_bin" - "$5" "$6" "$7" "$8" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def authenticate(value, expected):
    path = Path(value)
    if not path.is_file() or any(parent.is_symlink() for parent in (path, *path.parents)) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('launch boundary authority drift')
    return path

authority = json.loads(authenticate(sys.argv[3], sys.argv[4]).read_bytes())
authenticate(sys.argv[1], authority['launcher_sha256'])
authenticate(sys.argv[2], 'f2697cbf85067baeba354077dec2bc0427e39fcf71b0005396f0f265a054c012')
PY
"$python_bin" "$5" --authority "$7" --authority-sha256 "$8" \
    --source-dir "$restored_root/code" --on-helper "$6" \
    --output-root "/shared/b00090279/refer-or-clear/results/diagnostics/direct-reference-v1/direct-reference-global-$SLURM_JOB_ID" --phase "$9"

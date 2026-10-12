#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=protocol-c-support
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 7 ]]; then
    echo 'Usage: SOURCE_DIR RECORDS RECORDS_SHA SOURCE_SPLIT TARGET_SPLIT RAW_ROOT OUTPUT_PARENT' >&2
    exit 2
fi
[[ "$3" =~ ^[0-9a-f]{64}$ ]] || { echo 'Lowercase records SHA256 required' >&2; exit 2; }
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SLURM_CPUS_PER_TASK:?Four allocated CPUs required}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'C support preparation is forbidden on login nodes' >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == cpu ]] || { echo 'Allocated CPU partition required' >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_JOB_ID" =~ [1-9] ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
[[ "$SLURM_CPUS_PER_TASK" =~ ^[0-9]+$ ]] && (( 10#$SLURM_CPUS_PER_TASK >= 4 )) || {
    echo 'Four allocated CPUs required' >&2; exit 2;
}
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_STEP_GPUS:-}" && "${SLURM_GPUS:-0}" == 0 && "${SLURM_GPUS_ON_NODE:-0}" == 0 ]] || {
    echo 'C support preparation requires a CPU-only allocation' >&2; exit 2;
}
unset PYTHONPATH PYTHONHOME ROC_ANCHOR_GPU_INTEGRATION ROC_DIRECT_REFERENCE_GPU_INTEGRATION ROC_LEVIR_CKPT ROC_LEVIR_SHA256
source /opt/miniconda3/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
[[ "${CONDA_PREFIX:-}" == "$environment_path" ]] || { echo 'Exact project environment required' >&2; exit 2; }
python_bin="$environment_path/bin/python"
cd "$SLURM_SUBMIT_DIR"
source_dir="$1"
[[ "$source_dir" == /* ]] || source_dir="$PWD/$source_dir"
raw_root="$6"
[[ "$raw_root" == /* ]] || raw_root="$PWD/$raw_root"
scratch="${ROC_SCRATCH_ROOT:-/tmp/$USER}/protocol-c-support-$SLURM_JOB_ID"
[[ "$scratch" == /* ]] || scratch="$PWD/$scratch"
run_dir="$7/protocol-c-support-$SLURM_JOB_ID"
[[ "$run_dir" == /* ]] || run_dir="$PWD/$run_dir"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
export TMPDIR="$scratch/tmp"
"$python_bin" - "$source_dir" "$2" "$3" "$4" "$5" "$raw_root" "$run_dir" "$scratch" <<'PY'
import hashlib
from pathlib import Path
import shutil
import sys

if sys.prefix != '/shared/b00090279/conda/envs/refer-or-clear' or str(Path(sys.executable).parent) != sys.prefix + '/bin':
    raise ValueError('interpreter prefix mismatch')


def path(value):
    result = Path(value).absolute()
    if '..' in result.parts or any(parent.is_symlink() for parent in (result, *result.parents)):
        raise ValueError('non-symlink paths without traversal required')
    return result


def digest(value):
    checksum = hashlib.sha256()
    with value.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


source, records, expected, source_split, target_split, raw, run, scratch = sys.argv[1:]
source, records, source_split, target_split, raw, run, scratch = map(path, (source, records, source_split, target_split, raw, run, scratch))
if not source.is_dir() or not (source / 'src').is_dir() or not (source / 'scripts/prepare_protocol_c.py').is_file() or not raw.is_dir():
    raise ValueError('complete source and raw root required')
for entry in (source, *source.rglob('*')):
    if entry.is_symlink() or entry.stat().st_mode & 0o222:
        raise ValueError('readonly non-symlink source required')
if run.exists() or scratch.exists():
    raise ValueError('new scratch and output directories required')
protected = (source, raw, records, source_split, target_split)
for destination in (run, scratch):
    if any(destination.is_relative_to(item) or item.is_relative_to(destination) for item in protected):
        raise ValueError('output overlaps input authority')
if run.is_relative_to(scratch) or scratch.is_relative_to(run):
    raise ValueError('durable output and scratch must be separate')
inputs = {'records.csv': records, 'source-split.csv': source_split, 'target-split.csv': target_split}
hashes = {name: digest(original) for name, original in inputs.items()}
if hashes['records.csv'] != expected:
    raise ValueError('Records SHA256 mismatch before staging')
scratch.mkdir(parents=True, exist_ok=False)
(scratch / 'tmp').mkdir()
run.mkdir(parents=True, exist_ok=False)
durable = run / 'inputs'
durable.mkdir()
for name, original in inputs.items():
    copied = durable / name
    with original.open('rb') as source_stream, copied.open('xb') as output:
        shutil.copyfileobj(source_stream, output, length=1048576)
    if digest(copied) != hashes[name]:
        raise ValueError('metadata hash changed during staging')
    copied.chmod(0o444)
durable.chmod(0o555)
PY
export PYTHONPATH="$source_dir/src"
cd "$source_dir"
"$python_bin" "$source_dir/scripts/prepare_protocol_c.py" \
    --records "$run_dir/inputs/records.csv" --records-sha256 "$3" \
    --source-split "$run_dir/inputs/source-split.csv" --target-split "$run_dir/inputs/target-split.csv" \
    --raw-root "$raw_root" --output-root "$run_dir/preparation" --build

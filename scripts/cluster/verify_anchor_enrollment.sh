#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=anchor-enrollment-integrity
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 4 && $# -ne 5 ]]; then
    echo "Usage: $0 SOURCE_DIR MANIFEST MANIFEST_SHA256 OUTPUT_ROOT [EXPECTED_FOLD (0..9, default 0)]" >&2
    exit 2
fi
expected_fold="${5-0}"
[[ "$expected_fold" =~ ^[0-9]$ ]] || { echo 'Expected fold must be an integer in 0..9' >&2; exit 2; }
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SLURM_CPUS_PER_TASK:?Four allocated CPUs required}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'Integrity verification is forbidden on login nodes' >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == cpu ]] || { echo 'Allocated CPU partition required' >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
[[ "$SLURM_CPUS_PER_TASK" == 4 ]] || { echo 'Four allocated CPUs required' >&2; exit 2; }
source /opt/miniconda3/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
[[ "${CONDA_PREFIX:-}" == "$environment_path" ]] || { echo 'Wrong Conda environment' >&2; exit 2; }
python_bin="$environment_path/bin/python"
launcher="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
cd "$SLURM_SUBMIT_DIR"
scratch="${ROC_SCRATCH_ROOT:-/tmp/$USER}/anchor-enrollment-integrity-$SLURM_JOB_ID"
[[ "$scratch" == /* ]] || scratch="$PWD/$scratch"
run_dir="$4/anchor-enrollment-integrity-$SLURM_JOB_ID"
[[ "$run_dir" == /* ]] || run_dir="$PWD/$run_dir"
export PYTHONPATH=""
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TMPDIR="$scratch/tmp"
"$python_bin" - "$1" "$2" "$3" "$4" "$scratch" "$run_dir" "$environment_path" "$launcher" "$expected_fold" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

source, authority, expected, output = sys.argv[1:5]
scratch, run, environment, launcher = map(Path, sys.argv[5:9])
expected_fold = int(sys.argv[9])
if (Path(sys.prefix).resolve() != environment.resolve()
        or Path(sys.executable).resolve() != (environment / "bin/python").resolve()):
    raise ValueError("absolute project Python environment required")
if any(path.exists() or path.is_symlink() for path in (scratch, run)):
    raise FileExistsError("new scratch and output directories required")
source, authority = (Path(value).resolve(strict=True) for value in (source, authority))
scratch, run = scratch.resolve(), run.resolve()
script = source / "scripts/verify_anchor_enrollment_products.py"

def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()

script_sha = digest(script)
if digest(authority) != expected:
    raise ValueError("authority manifest hash mismatch")
manifest = json.loads(authority.read_text())
if type(manifest.get("fold")) is not int or manifest["fold"] != expected_fold:
    raise ValueError("authority manifest differs from explicitly expected fold")
protected = [source, authority.parent]
for key in ("output_root", "raw_root"):
    if manifest.get(key):
        protected.append(Path(manifest[key]).resolve())
for banks in manifest.get("banks", {}).values():
    for bank in banks.values():
        for product in bank.get("products", {}).values():
            protected.append(Path(product["path"]).resolve().parent)
if (scratch.is_relative_to(run) or run.is_relative_to(scratch)
        or any(destination.is_relative_to(root) or root.is_relative_to(destination)
               for destination in (scratch, run) for root in protected)):
    raise ValueError("scratch and output must be isolated from source and enrollment products")
scratch.mkdir(parents=True, exist_ok=False)
(scratch / "tmp").mkdir()
run.mkdir(parents=True, exist_ok=False)

def freeze(original, destination, checksum):
    destination.parent.mkdir()
    shutil.copyfile(original, destination)
    if digest(original) != checksum or digest(destination) != checksum:
        raise ValueError(f"hash changed during freezing: {original}")
    destination.chmod(0o444)
    destination.parent.chmod(0o555)

frozen_script = scratch / "code/verify_anchor_enrollment_products.py"
frozen_authority = scratch / "inputs/manifest.json"
freeze(script, frozen_script, script_sha)
freeze(authority, frozen_authority, expected)
ledger = dict(schema_version="anchor_enrollment_integrity_launcher_v1", verification_only=True,
              expected_fold=expected_fold,
              script=dict(path=str(script), frozen_path=str(frozen_script), sha256=script_sha),
              authority=dict(path=str(authority), frozen_path=str(frozen_authority), sha256=expected),
              launcher_sha256=digest(launcher),
              paths=dict(scratch_dir=str(scratch), run_dir=str(run), report=str(run / "report.json")),
              resource_contract=dict(partition="cpu", cpus=4, memory="8G", time="1:00:00"),
              allocation={key: os.environ.get(value) for key, value in
                          dict(job_id="SLURM_JOB_ID", node="SLURMD_NODENAME", partition="SLURM_JOB_PARTITION",
                               cpus="SLURM_CPUS_PER_TASK").items()},
              environment=dict(conda_prefix=os.environ.get("CONDA_PREFIX"), python_prefix=sys.prefix,
                               python_executable=sys.executable, python_version=platform.python_version(),
                               hostname=platform.node(),
                               flags={key: os.environ.get(key) for key in
                                      ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TMPDIR")}))
ledger_path = run / "launcher_manifest.json"
ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
ledger_path.chmod(0o444)
PY
status=0
"$python_bin" "$scratch/code/verify_anchor_enrollment_products.py" \
    --manifest "$scratch/inputs/manifest.json" --manifest-sha256 "$3" \
    --report "$run_dir/report.json" --expected-fold "$expected_fold" || status=$?
"$python_bin" - "$run_dir" "$status" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
ledger = json.loads((run / "launcher_manifest.json").read_text())
hashes = {}
for name in ("script", "authority"):
    checksum = hashlib.sha256(Path(ledger[name]["frozen_path"]).read_bytes()).hexdigest()
    if checksum != ledger[name]["sha256"]:
        raise ValueError(f"{name} snapshot hash changed after invocation")
    hashes[name] = checksum
completion = dict(returncode=int(sys.argv[2]), script_sha256=hashes["script"], manifest_sha256=hashes["authority"])
with (run / "completion.json").open("x") as stream:
    stream.write(json.dumps(completion, indent=2, sort_keys=True) + "\n")
(run / "completion.json").chmod(0o444)
PY
exit "$status"

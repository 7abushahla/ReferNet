#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=anchor-candidate-cpu-v1
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo "Usage: $0 MEASUREMENT_SOURCE_DIR CANDIDATES_JSON CANDIDATES_SHA256 OUTPUT_ROOT" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit directory required}"
: "${SLURM_CPUS_PER_TASK:?Allocated CPU count required}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'Compute work is forbidden on login nodes' >&2; exit 2 ;;
esac
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && "$SLURM_JOB_ID" =~ [1-9] ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
[[ "$SLURM_CPUS_PER_TASK" =~ ^[0-9]+$ && "$SLURM_CPUS_PER_TASK" -ge 4 ]] || { echo 'Four allocated CPUs required' >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == cpu ]] || { echo 'Allocated CPU partition required' >&2; exit 2; }
[[ "${SLURM_GPUS:-0}" == 0 && "${SLURM_GPUS_ON_NODE:-0}" == 0 && -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ]] || { echo 'GPU allocation is forbidden' >&2; exit 2; }
source /opt/miniconda3/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
[[ "${CONDA_PREFIX:-}" == "$environment_path" ]] || { echo 'Exact environment prefix required' >&2; exit 2; }
python_bin="$environment_path/bin/python"
wrapper_path="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"
cd "$SLURM_SUBMIT_DIR"
scratch="${ROC_SCRATCH_ROOT:-/tmp/$USER}/anchor-candidate-cpu-v1-$SLURM_JOB_ID"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
unset PYTHONPATH PYTHONHOME
staged_paths="$("$python_bin" - "$1" "$2" "$3" "$4" "$scratch" "$wrapper_path" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(value):
    require(isinstance(value, str) and bool(value.strip()) and "\n" not in value and "\r" not in value,
            "nonempty single-line path required")
    requested = Path(value).absolute()
    require(".." not in requested.parts, f"path traversal forbidden: {requested}")
    for component in (requested, *requested.parents):
        require(not component.is_symlink(), f"symlink path forbidden: {component}")
    return requested.resolve()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def decode_json(content):
    return json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_constant)


def authenticated(path, expected):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "lowercase SHA256 required")
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == expected, f"SHA256 mismatch: {path}")
    return content


try:
    prefix = Path("/shared/b00090279/conda/envs/refer-or-clear").resolve()
    require(Path(os.environ["CONDA_PREFIX"]).resolve() == prefix, "resolved environment prefix mismatch")
    require(Path(sys.prefix).resolve() == prefix, "resolved interpreter prefix mismatch")
    source, catalogue_path = map(safe_path, sys.argv[1:3])
    require(source.is_dir(), "measurement source directory missing")
    expected_catalogue = sys.argv[3]
    catalogue_bytes = authenticated(catalogue_path, expected_catalogue)
    catalogue = decode_json(catalogue_bytes)
    tools = catalogue["measurement_tools"]
    directory = safe_path(tools["directory"])
    require(directory.is_relative_to(source), "measurement tools outside supplied source")
    files = tools["files"]
    require(isinstance(files, dict) and {
        "bench_anchor_candidates.py", "bench_anchor_pilots.py", "report_anchor_pilots.py"
    } <= set(files), "authenticated coordinator, benchmark and inspector required")
    staged_files = {}
    for name, expected in files.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts and relative != Path("."),
                "tool path traversal forbidden")
        require(not any(part.startswith("._") for part in relative.parts), "AppleDouble tool forbidden")
        require(relative.suffix in (".py", ".sh", ".json", ".csv", ".md", ".txt"),
                "only metadata and measurement scripts may be staged")
        staged_files[name] = authenticated(safe_path(str(directory / relative)), expected)
    protected = {source, directory, catalogue_path.parent, safe_path(sys.argv[6]).parent}

    def protect_paths(value):
        if isinstance(value, list):
            for item in value:
                protect_paths(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in ("path", "from", "to", "directory", "run_dir", "raw_root", "legacy_root"):
                    if item is not None:
                        require(isinstance(item, str) and Path(item).is_absolute(), "bound input path must be absolute")
                        path = safe_path(item)
                        protected.add(path.parent if key in ("path", "from", "to") else path)
                else:
                    protect_paths(item)

    protect_paths(catalogue.get("candidates", []))
    for candidate in catalogue.get("candidates", []):
        for key in ("parent_manifest", "train_json"):
            binding = candidate.get(key)
            if binding is not None:
                path = safe_path(binding["path"])
                require(path.suffix == ".json", "candidate ledger must be JSON metadata")
                metadata = decode_json(authenticated(path, binding["sha256"]))
                for name, value in metadata.get("paths", {}).items():
                    if value is not None:
                        require(isinstance(value, str) and Path(value).is_absolute(), "ledger path must be absolute")
                        original = safe_path(value)
                        protected.add(original.parent if name in ("records", "split", "config", "weights") else original)
    output = safe_path(str(Path(sys.argv[4]) / f"anchor-candidate-cpu-v1-{os.environ['SLURM_JOB_ID']}"))
    scratch = safe_path(sys.argv[5])
    for destination in (output, scratch):
        require(not destination.exists(), f"destination must be new: {destination}")
        for original in protected:
            require(not destination.is_relative_to(original) and not original.is_relative_to(destination),
                    f"destination overlaps protected input: {original}")
    require(not output.is_relative_to(scratch) and not scratch.is_relative_to(output), "output/scratch overlap")
    if hasattr(os, "sched_getaffinity"):
        require(len(os.sched_getaffinity(0)) >= 4, "four affinity entries required")
    output.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    (scratch / "tmp").mkdir()
    for name, content in staged_files.items():
        destination = scratch / "tools" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)
        destination.chmod(0o444)
    snapshot = {
        "contract": "anchor_candidate_cpu_v1", "phase": "prepare", "execution_authorized": False,
        "candidate_order": ["R", "D_running_pilot", "N"], "candidates": catalogue.get("candidates", []),
        "catalogue": str(catalogue_path), "catalogue_sha256": expected_catalogue,
        "measurement_source": str(source), "measurement_tools": tools,
        "wrapper_sha256": hashlib.sha256(Path(sys.argv[6]).read_bytes()).hexdigest(),
        "environment_prefix": str(prefix), "python": "/shared/b00090279/conda/envs/refer-or-clear/bin/python",
        "interpreter_prefix": str(Path(sys.prefix).resolve()),
        "allocation": {name: os.environ[name] for name in (
            "SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK")},
        "resource_contract": "1 node,4 CPUs,16GiB,2h,no GPU,no concurrent wrapper workers",
        "environment": {name: os.environ.get(name) for name in (
            "CONDA_PREFIX", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
            "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE", "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")},
    }
    with (output / "allocation.json").open("x") as stream:
        json.dump(snapshot, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)
    print(scratch)
    print(catalogue_path)
except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
    print(str(error), file=sys.stderr)
    sys.exit(1)
PY
)"
mapfile -t paths <<< "$staged_paths"
output_root="${paths[0]}"
scratch="${paths[1]}"
candidates="${paths[2]}"
export TMPDIR="$scratch/tmp"
"$python_bin" -m pip freeze > "$output_root/environment.txt"
"$python_bin" "$scratch/tools/bench_anchor_candidates.py" --phase prepare \
    --candidates "$candidates" --candidates-sha256 "$3" --run-dir "$output_root/preparation"

#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=anchor-d-context
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
if [[ $# -ne 15 ]]; then
    echo "Usage: $0 TOOLING_DIR PARENT_MANIFEST PARENT_SHA256 TRAIN_JSON TRAIN_SHA256 CONFIG RECORDS SPLIT WEIGHTS CHECKPOINT PREDICTIONS PREDICTIONS_SHA256 RAW_ROOT LEGACY_ROOT OUTPUT_ROOT" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${SLURM_CPUS_PER_TASK:?Four allocated CPUs required}"
case "${SLURMD_NODENAME,,}" in
    *login*|*head*|*frontend*|*lgn*) echo 'Compute work is forbidden on login nodes' >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == gpu ]] || { echo 'Allocated GPU partition required' >&2; exit 2; }
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm job ID' >&2; exit 2; }
[[ "$SLURM_CPUS_PER_TASK" == 4 ]] || { echo 'Four allocated CPUs required' >&2; exit 2; }
source /opt/miniconda/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
[[ "${CONDA_PREFIX:-}" == "$environment_path" ]] || { echo 'Wrong Conda environment' >&2; exit 2; }
python_bin="$environment_path/bin/python"
cd "$SLURM_SUBMIT_DIR"
scientific_dir=/shared/b00090279/refer-or-clear/results/source/anchor-0959335bd9a6a22510f6706c49c62d84560cd524/code
weights_path="$(cd "$(dirname "$9")" && pwd -P)/$(basename "$9")"
checkpoint_path="$(cd "$(dirname "${10}")" && pwd -P)/$(basename "${10}")"
raw_root="$(cd "${13}" && pwd -P)"
legacy_root="$(cd "${14}" && pwd -P)"
scratch="${ROC_SCRATCH_ROOT:-/scratch/$USER}/anchor-d-context-$SLURM_JOB_ID"
[[ "$scratch" == /* ]] || scratch="$PWD/$scratch"
run_dir="${15}/anchor-d-context-$SLURM_JOB_ID"
[[ "$run_dir" == /* ]] || run_dir="$PWD/$run_dir"
export PYTHONPATH="$scientific_dir/src"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TMPDIR="$scratch/tmp"
"$python_bin" - "$@" "$scratch" "$run_dir" "$scientific_dir" "${BASH_SOURCE[0]}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

environment = Path("/shared/b00090279/conda/envs/refer-or-clear")
if Path(sys.prefix).resolve() != environment.resolve() or Path(sys.executable).resolve() != (environment / "bin/python").resolve():
    raise ValueError("absolute project Python environment required")
(tooling, parent_path, parent_hash, train_path, train_hash, config, records, split, weights,
 checkpoint, predictions, predictions_hash, raw, legacy, output) = sys.argv[1:16]
scratch, run, scientific, launcher = map(Path, sys.argv[16:])
scratch, run, scientific = scratch.resolve(), run.resolve(), scientific.resolve()
tooling, raw, legacy = (Path(value).resolve(strict=True) for value in (tooling, raw, legacy))
if not all(path.is_dir() for path in (tooling, raw, legacy)):
    raise ValueError("tooling, raw and legacy directories required")
if any(path.exists() or path.is_symlink() for path in (scratch, run)):
    raise FileExistsError("new scratch and output directories required")

def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()

script_sha = "4253fc893e41d8a31cc95c35e5b170f1553b102eb53f2a7134f3faa8da7114b4"
source_sha = "2f8a833e5cb3450fdccd8c58dba8f07f58af330747a69cca7363d1a21528da9a"
checkpoint_sha = "c78d0b4a06b5dbde434d77f34ffa14ea437dbde4e1d3eafbd2d8aadfff72c47f"
tool_files = {name: tooling / name for name in ("scripts/check_anchor_d_context.py", "tests/test_anchor_d_context.py")}
tool_hashes = {name: digest(path) for name, path in tool_files.items()}
if tool_hashes["scripts/check_anchor_d_context.py"] != script_sha:
    raise ValueError("reviewed diagnostic script hash mismatch")
paths = {name: Path(value).resolve(strict=True) for name, value in
         dict(parent_manifest=parent_path, train_json=train_path, config=config, records=records,
              split=split, weights=weights, checkpoint=checkpoint, predictions=predictions).items()}
hashes = {name: digest(path) for name, path in paths.items()}
if any(hashes[name] != expected for name, expected in
       (("parent_manifest", parent_hash), ("train_json", train_hash), ("predictions", predictions_hash))):
    raise ValueError("input authority hash mismatch")
parent = json.loads(paths["parent_manifest"].read_text())
train = json.loads(paths["train_json"].read_text())
if (parent.get("schema_version") != "anchor_job_manifest_v1" or parent.get("phase") != "diagnostic"
        or parent.get("track") != "D" or parent.get("allocation", {}).get("job_id") != "37963"
        or parent.get("data_scope") != "train_val_only"
        or parent.get("paths", {}).get("source_dir") != str(scientific)
        or parent.get("runtime_identities", {}).get("source_sha256") != source_sha):
    raise ValueError("original frozen 095 D37963 diagnostic authority required")
if (train.get("schema_version") != "anchor_pilot_v1" or train.get("phase") != "overfit"
        or train.get("track") != "D" or train.get("opened_roles") != ["train"]
        or train.get("no_test_open") is not True or train.get("n_images") != {"train": 16}):
    raise ValueError("training-only D overfit authority required")
if hashes["checkpoint"] != checkpoint_sha or train.get("selected_checkpoint_sha256") != checkpoint_sha:
    raise ValueError("original D37963 checkpoint hash required")
if any(parent["provenance"].get(f"{name}_sha256") != hashes[name] for name in ("config", "records", "split", "weights")):
    raise ValueError("config/records/split/weights hash mismatch")
inventory_sha = hashlib.sha256(json.dumps(parent["source_files"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
if inventory_sha != parent["provenance"]["source_sha256"]:
    raise ValueError("scientific source inventory hash mismatch")
protected = [scientific, tooling, raw, legacy, *(path.parent for path in paths.values())]
if (scratch.is_relative_to(run) or run.is_relative_to(scratch)
        or any(destination.is_relative_to(root) or root.is_relative_to(destination)
               for destination in (scratch, run) for root in protected)):
    raise ValueError("scratch and output must be isolated from historical inputs and tooling")
scratch.mkdir(parents=True, exist_ok=False)
(scratch / "tmp").mkdir()
run.mkdir(parents=True, exist_ok=False)

def freeze(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(source) != expected or digest(destination) != expected:
        raise ValueError(f"input hash changed during staging: {source}")
    destination.chmod(0o444)

for name, source in tool_files.items():
    freeze(source, run / "tooling" / name, tool_hashes[name])
names = dict(parent_manifest="parent_manifest.json", train_json="train.json", config="config.yaml",
             records="records.csv", split="split.csv", predictions="predictions.csv")
for name, filename in names.items():
    freeze(paths[name], run / "inputs" / filename, hashes[name])
for root in (run / "tooling", run / "inputs"):
    for directory, _, _ in os.walk(root, topdown=False):
        Path(directory).chmod(0o555)
versions = {}
for package in ("torch", "torchvision", "pandas", "numpy", "Pillow", "mmcv", "mmengine", "opencd"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
ledger = dict(schema_version="anchor_d_context_launcher_v1", diagnostic_only=True, scientific_gate_passed=None,
              scientific_source=dict(path=str(scientific), runtime_sha256=source_sha, inventory_sha256=inventory_sha),
              diagnostic_script_sha256=script_sha, launcher_sha256=digest(launcher.resolve(strict=True)),
              tooling={name: dict(path=str(path), sha256=tool_hashes[name]) for name, path in tool_files.items()},
              inputs={name: dict(path=str(path), sha256=hashes[name]) for name, path in paths.items()},
              paths=dict(run_dir=str(run), diagnostic_dir=str(run / "context"), scratch_dir=str(scratch),
                         raw_root=str(raw), legacy_root=str(legacy)),
              resource_contract=dict(partition="gpu", gpus=1, cpus=4, memory="8G", time="1:00:00"),
              allocation={key: os.environ.get(value) for key, value in
                          dict(job_id="SLURM_JOB_ID", node="SLURMD_NODENAME", partition="SLURM_JOB_PARTITION",
                               cpus="SLURM_CPUS_PER_TASK").items()},
              environment=dict(conda_prefix=os.environ.get("CONDA_PREFIX"), python_prefix=sys.prefix,
                               python_executable=sys.executable, python_version=platform.python_version(),
                               hostname=platform.node(), packages=versions,
                               flags={key: os.environ.get(key) for key in
                                      ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                       "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")}),
              execution=dict(device="cuda", workers=0, precision="float32", amp=False))
manifest = run / "launcher_manifest.json"
manifest.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
manifest.chmod(0o444)
PY
"$python_bin" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' > "$run_dir/gpu-device.txt"
export PYTHONPATH="$scientific_dir/src"
cd "$run_dir/tooling"
"$python_bin" "$run_dir/tooling/scripts/check_anchor_d_context.py" \
    --source-dir "$scientific_dir" \
    --parent-manifest "$run_dir/inputs/parent_manifest.json" --parent-manifest-sha256 "$3" \
    --train-json "$run_dir/inputs/train.json" --train-json-sha256 "$5" \
    --config "$run_dir/inputs/config.yaml" --records "$run_dir/inputs/records.csv" --split "$run_dir/inputs/split.csv" \
    --weights "$weights_path" --checkpoint "$checkpoint_path" \
    --predictions "$run_dir/inputs/predictions.csv" --predictions-sha256 "${12}" \
    --raw-root "$raw_root" --legacy-root "$legacy_root" --run-dir "$run_dir/context" --device cuda --workers 0

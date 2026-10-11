#!/bin/bash
#SBATCH --account=acc-mialhajri
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --job-name=anchor-full-tests
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail
[[ $# -eq 4 ]] || { echo 'Usage: SOURCE_DIR SOURCE_REVISION SOURCE_ARCHIVE_SHA256 OUTPUT_ROOT' >&2; exit 2; }
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${SLURMD_NODENAME:?Allocated compute node required}"
[[ "${SLURM_JOB_PARTITION:-}" == gpu ]] || exit 2
case "${SLURMD_NODENAME,,}" in *login*|*head*|*frontend*|*lgn*) exit 2;; esac
source /opt/miniconda/etc/profile.d/conda.sh
environment_path=/shared/b00090279/conda/envs/refer-or-clear
conda activate "$environment_path"
python_bin="$environment_path/bin/python"
repository=/shared/b00090279/refer-or-clear
source_dir="$(cd "$1" && pwd -P)"
scratch="${ROC_SCRATCH_ROOT:-/scratch/$USER}/anchor-full-tests-$SLURM_JOB_ID"
out="$4/anchor-full-tests-$SLURM_JOB_ID"
test ! -e "$scratch" && test ! -e "$out"
mkdir -p "$scratch/tmp" "$out"
export TMPDIR="$scratch/tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export ROC_ANCHOR_GPU_INTEGRATION=1 ROC_LEVIR_SHA256=da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618
cp "$repository/checkpoints/pretrained/ChangerEx_r18-512x512_40k_levircd_20221223_120511.pth" "$scratch/weights.pth"
export ROC_LEVIR_CKPT="$scratch/weights.pth"
echo "$ROC_LEVIR_SHA256  $ROC_LEVIR_CKPT" | sha256sum -c -
"$python_bin" - "$2" "$3" <<'PY' | tee "$out/environment.json"
import json, os, sys, torch
assert sys.prefix == '/shared/b00090279/conda/envs/refer-or-clear'
assert os.environ['CONDA_PREFIX'] == sys.prefix
assert sys.executable == sys.prefix + '/bin/python'
assert torch.cuda.is_available()
print(json.dumps(dict(schema_version='anchor_full_suite_v1',source_revision=sys.argv[1],source_archive_sha256=sys.argv[2],python=sys.executable,prefix=sys.prefix,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),job_id=os.environ['SLURM_JOB_ID'],node=os.environ['SLURMD_NODENAME'],scope='Synthetic tests, metadata and image-existence checks, actual Open-CD/LEVIR GPU integration; no corpus pixel decoding or scientific training')))
PY
cp -a "$source_dir" "$scratch/code"
chmod -R u+w "$scratch/code"
code_root="$scratch/code"
"$python_bin" "$code_root/scripts/stage_full_suite_metadata.py" --repository "$repository" --code "$code_root" --manifest "$out/metadata_manifest.json"
echo "afd661819fe9be97c83d2ee5123a7ce60869437c61d1e9029c4b8a96175aa276  $code_root/data/processed/records.csv" | sha256sum -c -
echo "985d10b47461e0fa41d5502fbb51cd6fc03f8c7e8ec8cd1040b21be318efc1d9  $code_root/data/processed/splits/protocol_a_it0.csv" | sha256sum -c -
export PYTHONPATH="$code_root/src"
cd "$code_root"
"$python_bin" -m pytest --version | tee "$out/pytest-version.txt"
"$python_bin" -m pytest tests/ -q -p no:cacheprovider --basetemp "$scratch/pytest-tmp" --junitxml "$out/test-results.xml" 2>&1 | tee "$out/tests.log"
